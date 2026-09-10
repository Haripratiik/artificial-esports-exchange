"""How an underlying level becomes a settlement value.

Four shapes, and between them every instrument the project trades:

    Linear   scale * level + offset      futures, spreads, indices
    Binary   payout, or nothing          event and prediction contracts
    Call     max(scale * level - K, 0)   European call, cash settled
    Put      max(K - scale * level, 0)   European put, cash settled

Options are payoffs on the underlying rather than derivatives of a traded
future, and the equivalence is exact rather than a shortcut. A future here
settles at ``scale * level``; a European call on that future settles at
``max(F_T - K, 0)``; both settle from the same metric at the same instant, so
substituting one into the other gives precisely the expression above. That has
three consequences worth stating: options need no new machinery, they settle
even if their underlying future never traded, and put-call parity holds as an
exact identity rather than as an approximation.

What none of these provide is a *price*. Valuing an option needs a volatility
model, and these contracts have an unusual one (variance shrinks
deterministically as the observation window fills with evidence), so that
belongs with the agents that trade options, not with the contract that defines
them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

__all__ = ["Payoff", "Linear", "Binary", "Call", "Put"]

_COMPARISONS = {
    ">": lambda level, threshold: level > threshold,
    ">=": lambda level, threshold: level >= threshold,
    "<": lambda level, threshold: level < threshold,
    "<=": lambda level, threshold: level <= threshold,
}


class Payoff(ABC):
    @abstractmethod
    def apply(self, level: float) -> float:
        """Map an underlying level to a settlement value, before tick rounding."""

    @abstractmethod
    def bounds(self, level_bounds: tuple[float, float]) -> tuple[float, float]:
        """The range of settlement values, given the underlying's range.

        This is what makes collateral exact. Every instrument here settles
        inside a known interval, so a position's worst case is arithmetic
        rather than a value-at-risk estimate, which is not true of an ordinary
        future on an unbounded price.
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Canonical form, which feeds the contract spec digest."""


@dataclass(frozen=True, slots=True)
class Linear(Payoff):
    """``scale * level + offset``.

    The scale exists to move a rate in the 0-1 range onto a price grid with
    useful tick resolution. A win rate of 0.5537 at scale 10000 settles at
    5537, so one tick is a quotable amount rather than a rounding artifact.
    """

    scale: float
    offset: float = 0.0

    def apply(self, level: float) -> float:
        return self.scale * level + self.offset

    def bounds(self, level_bounds: tuple[float, float]) -> tuple[float, float]:
        # A negative scale is a legitimate inverse contract, and it flips the
        # interval. Assuming the low bound maps to the low settlement would
        # silently invert the collateral requirement on such a contract.
        ends = (self.apply(level_bounds[0]), self.apply(level_bounds[1]))
        return (min(ends), max(ends))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "linear", "scale": self.scale, "offset": self.offset}


@dataclass(frozen=True, slots=True)
class Binary(Payoff):
    """``payout`` if the comparison holds at expiry, otherwise zero.

    The price of such a contract is often read as a probability. Whether that
    reading survives contact with data is one of the questions this project
    exists to test, so nothing here enforces a 0-1 price range or treats the
    price as a probability. That interpretation has to be earned empirically.
    """

    comparison: str
    threshold: float
    payout: float = 1.0

    def __post_init__(self) -> None:
        if self.comparison not in _COMPARISONS:
            raise ValueError(
                f"comparison must be one of {sorted(_COMPARISONS)}, got {self.comparison!r}"
            )

    def apply(self, level: float) -> float:
        holds = _COMPARISONS[self.comparison](level, self.threshold)
        return self.payout if holds else 0.0

    def bounds(self, level_bounds: tuple[float, float]) -> tuple[float, float]:
        """Always zero to payout, regardless of the underlying's range.

        A binary discards everything about the level except which side of the
        threshold it fell on, so its settlement range does not depend on how
        wide the underlying's range is. Even a threshold outside the
        underlying's range keeps this interval: the contract is then certain to
        settle at one end, but it is still *defined* over both.
        """
        return (min(0.0, self.payout), max(0.0, self.payout))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "binary",
            "comparison": self.comparison,
            "threshold": self.threshold,
            "payout": self.payout,
        }


@dataclass(frozen=True, slots=True)
class Call(Payoff):
    """``max(scale * level - strike, 0)``. A European call, cash settled.

    Expressed as a payoff on the *underlying* rather than as a derivative of a
    traded future, and the equivalence is exact rather than a convenience: a
    future on this underlying settles at ``scale * level``, and a European call
    on that future settles at ``max(F_T - K, 0)``. Both settle from the same
    metric at the same expiry, so substituting gives precisely this expression.
    No new machinery, no dependency on the future being liquid, and the option
    settles even if its underlying future never traded.

    What this deliberately does *not* provide is a price. Valuing an option
    needs a volatility model, and these contracts have an unusual one: variance
    shrinks deterministically as the observation window fills with battles, so
    implied vol should follow a predictable decay whose violations are
    informative. That belongs with the agents that trade options, not with the
    contract that defines them.
    """

    strike: float
    scale: float = 1.0

    def apply(self, level: float) -> float:
        return max(self.scale * level - self.strike, 0.0)

    def bounds(self, level_bounds: tuple[float, float]) -> tuple[float, float]:
        """Floored at zero, capped by the best the underlying can do.

        An option's downside is bounded by its own structure rather than by the
        underlying's range, which is the whole point of buying one, and it
        makes the collateral for a long position exactly the premium paid.
        """
        ends = (
            self.scale * level_bounds[0] - self.strike,
            self.scale * level_bounds[1] - self.strike,
        )
        return (0.0, max(0.0, max(ends)))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "call", "strike": self.strike, "scale": self.scale}


@dataclass(frozen=True, slots=True)
class Put(Payoff):
    """``max(strike - scale * level, 0)``. A European put, cash settled.

    The mirror of :class:`Call`, and together they satisfy put-call parity
    exactly at settlement:

        max(F - K, 0) - max(K - F, 0) = F - K

    That identity is not an approximation here (both legs settle from the same
    metric at the same instant), so it is an exact invariant the test suite can
    assert rather than a relationship that holds to within a discount factor.
    """

    strike: float
    scale: float = 1.0

    def apply(self, level: float) -> float:
        return max(self.strike - self.scale * level, 0.0)

    def bounds(self, level_bounds: tuple[float, float]) -> tuple[float, float]:
        ends = (
            self.strike - self.scale * level_bounds[0],
            self.strike - self.scale * level_bounds[1],
        )
        return (0.0, max(0.0, max(ends)))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "put", "strike": self.strike, "scale": self.scale}
