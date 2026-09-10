"""What a contract is written on, expressed as a small closed algebra.

A contract's underlying is not a price: it is a recipe for computing one number
from measured facts about the external world. Three node types cover every
instrument family in the current roadmap:

    Single      one measured metric             -> performance futures
    Difference  left minus right                -> relative-value spreads
    Basket      pinned weighted combination     -> class and meta indices

Keeping this closed (rather than, say, evaluating an expression string pulled
from a YAML file) is deliberate. Settlement has to be auditable long after the
fact, and an algebra with three constructors can be reasoned about
exhaustively.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["ALL", "MetricRef", "Underlying", "Single", "Difference", "Basket"]

# "ALL" means "do not filter on this dimension". It is spelled out in the
# contract rather than left implicit as an empty tuple, so that a truncated or
# half-written spec file can never silently widen a contract's universe.
ALL = "ALL"


@dataclass(frozen=True, slots=True, order=True)
class MetricRef:
    """A fully qualified, measurable quantity.

    Every field narrows the population the metric is computed over. Two
    contracts that differ in any field are written on different things, so a
    MetricRef is also the key the oracle resolves and caches against.
    """

    metric: str
    subject: str
    modes: tuple[str, ...] = (ALL,)
    maps: tuple[str, ...] = (ALL,)
    trophy_buckets: tuple[str, ...] = (ALL,)
    # The range the metric can take, declared by the contract rather than
    # inferred. Two jobs: it propagates through the algebra to give an exact
    # settlement range, which makes collateral computable rather than estimated;
    # and settlement verifies the resolved value falls inside it, which catches
    # an oracle returning something the contract never contemplated.
    #
    # A rate is the default because every metric in the first world is one.
    bounds: tuple[float, float] = (0.0, 1.0)
    # Whether this measures a proportion or an amount delivered.
    #
    # It is not decoration: it is the difference between a contract on how often
    # something happens and a contract on how much of it there was, and those
    # are different asset classes with different economics. A rate has no term
    # structure worth speaking of; a quantity does, because the amount delivered
    # in March is a different thing from the amount delivered in April.
    #
    # Declared here rather than inferred from the metric name, so the layer that
    # classifies instruments never has to know what a competitor is.
    kind: str = "rate"

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("metric name is required")
        if not self.subject:
            raise ValueError("subject is required")
        if self.bounds[0] > self.bounds[1]:
            raise ValueError(f"bounds are inverted: {self.bounds}")
        if self.kind not in ("rate", "quantity", "dispersion"):
            raise ValueError(
                f"metric kind {self.kind!r} must be 'rate', 'quantity' or "
                "'dispersion'; the distinction decides how the instrument is "
                "classified"
            )
        for field_name in ("modes", "maps", "trophy_buckets"):
            values = getattr(self, field_name)
            if not values:
                raise ValueError(
                    f"{field_name} must be non-empty; use ('ALL',) to include everything"
                )
            if ALL in values and len(values) > 1:
                raise ValueError(f"{field_name} mixes 'ALL' with explicit values")
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} contains duplicates")
            if tuple(sorted(values)) != values:
                raise ValueError(
                    f"{field_name} must be sorted so that equal filters compare equal"
                )

    @property
    def key(self) -> str:
        """Short stable label, used to order terms and tag diagnostics."""
        parts = (
            self.metric,
            self.subject,
            "|".join(self.modes),
            "|".join(self.maps),
            "|".join(self.trophy_buckets),
        )
        return ":".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "subject": self.subject,
            "modes": list(self.modes),
            "maps": list(self.maps),
            "trophy_buckets": list(self.trophy_buckets),
            "bounds": list(self.bounds),
            "kind": self.kind,
        }


class Underlying(ABC):
    """A deterministic function from resolved metrics to a single number."""

    @abstractmethod
    def atoms(self) -> tuple[MetricRef, ...]:
        """Every metric that must be resolved before this can be evaluated."""

    @abstractmethod
    def evaluate(self, values: Mapping[MetricRef, float]) -> float:
        """Combine already-resolved metric values into the underlying level."""

    @abstractmethod
    def bounds(self) -> tuple[float, float]:
        """The range this underlying's level can take.

        Interval arithmetic over the algebra. Exact rather than estimated,
        which is what lets a position's worst case be computed instead of
        modelled: unusual, and a direct consequence of every underlying here
        being a bounded statistic rather than an unbounded price.
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Canonical form, which feeds the contract spec digest."""


@dataclass(frozen=True, slots=True)
class Single(Underlying):
    ref: MetricRef

    def atoms(self) -> tuple[MetricRef, ...]:
        return (self.ref,)

    def evaluate(self, values: Mapping[MetricRef, float]) -> float:
        return values[self.ref]

    def bounds(self) -> tuple[float, float]:
        return self.ref.bounds

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "single", "ref": self.ref.to_dict()}


@dataclass(frozen=True, slots=True)
class Difference(Underlying):
    """left minus right: the relative-value primitive.

    A spread hedges out whatever moves both legs together, which is what makes
    cross-sectional strategies expressible at all.
    """

    left: Underlying
    right: Underlying

    def atoms(self) -> tuple[MetricRef, ...]:
        return _dedupe(self.left.atoms() + self.right.atoms())

    def evaluate(self, values: Mapping[MetricRef, float]) -> float:
        return self.left.evaluate(values) - self.right.evaluate(values)

    def bounds(self) -> tuple[float, float]:
        # Subtraction inverts the right interval: the widest the difference can
        # be is (left's best minus right's worst), and vice versa. Using
        # (lo - lo, hi - hi) is the classic interval-arithmetic mistake and
        # would understate a spread's range by half.
        left_lo, left_hi = self.left.bounds()
        right_lo, right_hi = self.right.bounds()
        return (left_lo - right_hi, left_hi - right_lo)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "difference",
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class Basket(Underlying):
    """A weighted combination whose weights are pinned in the contract spec.

    Because the weights live in the spec they are covered by its digest. An
    index whose weights could be recomputed from data inside its own
    observation window would leak the future into its own settlement, so the
    rule that weights are fixed before the window opens is structural here
    rather than a convention someone has to remember.
    """

    legs: tuple[tuple[Underlying, float], ...]

    def __post_init__(self) -> None:
        if not self.legs:
            raise ValueError("a basket needs at least one leg")

    def atoms(self) -> tuple[MetricRef, ...]:
        collected: tuple[MetricRef, ...] = ()
        for leg, _weight in self.legs:
            collected += leg.atoms()
        return _dedupe(collected)

    def evaluate(self, values: Mapping[MetricRef, float]) -> float:
        # Terms accumulate in canonical-shape order rather than declaration
        # order, so reordering legs in the YAML cannot move the last bit of a
        # settlement value. Float addition is not associative.
        contributions = sorted(
            (
                (_shape_key(leg), weight * leg.evaluate(values))
                for leg, weight in self.legs
            ),
            key=lambda item: item[0],
        )
        total = 0.0
        for _shape, contribution in contributions:
            total += contribution
        return total

    def bounds(self) -> tuple[float, float]:
        # A negative weight flips its leg's interval, which is how a long/short
        # index expresses itself. Ignoring the sign would report a range that
        # excludes values the basket can actually settle at.
        lower = 0.0
        upper = 0.0
        for leg, weight in self.legs:
            leg_lo, leg_hi = leg.bounds()
            scaled = (weight * leg_lo, weight * leg_hi)
            lower += min(scaled)
            upper += max(scaled)
        return (lower, upper)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "basket",
            "legs": [
                {"leg": leg.to_dict(), "weight": weight} for leg, weight in self.legs
            ],
        }


def _shape_key(leg: Underlying) -> str:
    from arena.determinism import canonical_json

    return canonical_json(leg.to_dict())


def _dedupe(refs: tuple[MetricRef, ...]) -> tuple[MetricRef, ...]:
    """Unique refs in sorted order, so each metric is resolved exactly once."""
    return tuple(sorted(set(refs)))
