"""The immutable contract specification.

A contract spec is the constitution of a market. Once published it cannot be
amended, because every price ever printed against it was an opinion about the
settlement rule as written at the time. The spec is therefore content-addressed:
its digest covers the underlying, the payoff, the window, the data policy, and
the identity of the pinned reference snapshot. Change any of them and you have
a different contract, with a different digest, which the settlement record will
show.

Two invariants are enforced at construction rather than left to reviewer
discipline, because both are lookahead bugs that would be invisible in results:

    published_at <= window.start        the market cannot be written after the
                                        outcome has begun to be determined
    reference.as_of <= window.start     index weights and priors cannot be
                                        fitted on the window they settle over
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from arena.contracts.payoff import Payoff
from arena.contracts.underlying import MetricRef, Underlying
from arena.determinism import digest, quantize_to_tick

__all__ = [
    "ObservationWindow",
    "DataPolicy",
    "MissingDataPolicy",
    "DistributionSchedule",
    "ContractSpec",
]


class MissingDataPolicy:
    """What to do when the oracle cannot produce a required metric.

    VOID is the conservative default and the only one enabled at first. A
    contract that silently settles on a guess is worse than one that does not
    settle at all, because the guess becomes indistinguishable from a measurement
    the moment it is written to the tape.
    """

    VOID = "VOID"

    ALL = (VOID,)


@dataclass(frozen=True, slots=True)
class ObservationWindow:
    """Half-open interval ``[start, end)`` in UTC.

    Half-open so that consecutive windows tile the timeline without a battle
    ever landing in two settlement periods at once.
    """

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        for label, value in (("start", self.start), ("end", self.end)):
            if value.tzinfo is None:
                raise ValueError(f"window {label} must be timezone-aware")
            if value.utcoffset() != timezone.utc.utcoffset(None):
                raise ValueError(f"window {label} must be UTC, got {value.tzinfo}")
        if self.start >= self.end:
            raise ValueError("window start must precede window end")

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment < self.end

    def to_dict(self) -> dict[str, Any]:
        return {"start": _iso(self.start), "end": _iso(self.end)}


@dataclass(frozen=True, slots=True)
class DataPolicy:
    """The evidential bar a settlement has to clear.

    ``min_sample_size`` guards the metric as a whole. ``min_stratum_battles``
    and ``min_strata_coverage`` guard its *composition*: a standardized rate
    computed from three well-sampled strata out of forty is not the quantity the
    contract named, even if the total battle count looks respectable.
    """

    min_sample_size: int
    min_stratum_battles: int = 0
    min_strata_coverage: float = 0.0
    missing_data_policy: str = MissingDataPolicy.VOID
    # How the metric treats a stratum the snapshot declares but the data does
    # not cover. Part of the contract, and therefore of its digest, because it
    # materially changes what the settlement number means.
    missing_strata_policy: str = "IMPUTE_FROM_PRIOR"

    def __post_init__(self) -> None:
        if self.min_sample_size < 0:
            raise ValueError("min_sample_size cannot be negative")
        if self.min_stratum_battles < 0:
            raise ValueError("min_stratum_battles cannot be negative")
        if not 0.0 <= self.min_strata_coverage <= 1.0:
            raise ValueError("min_strata_coverage must lie in [0, 1]")
        if self.missing_data_policy not in MissingDataPolicy.ALL:
            raise ValueError(
                f"missing_data_policy must be one of {MissingDataPolicy.ALL}, "
                f"got {self.missing_data_policy!r}"
            )
        if self.missing_strata_policy not in ("IMPUTE_FROM_PRIOR", "DROP_AND_RENORMALIZE"):
            raise ValueError(
                f"unknown missing_strata_policy {self.missing_strata_policy!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_sample_size": self.min_sample_size,
            "min_stratum_battles": self.min_stratum_battles,
            "min_strata_coverage": self.min_strata_coverage,
            "missing_data_policy": self.missing_data_policy,
            "missing_strata_policy": self.missing_strata_policy,
        }


@dataclass(frozen=True, slots=True)
class DistributionSchedule:
    """Cash paid to holders while the contract is still alive.

    This is the whole difference between a share and a future. A future pays
    once, at the end; a share pays as it goes and is worth the stream of what
    is left. Everything else about the two is the same machinery, which is why
    this is a field on the spec rather than a new kind of contract.

    Each window is measured separately, so the amount paid in one period is a
    different number from the amount paid in the next: the point of a dividend,
    and the reason a share's price moves on news about one quarter rather than
    only on news about the end of its life.

    What this deliberately is *not* is a perpetual claim. Every contract here
    settles inside a known interval, which is what makes collateral arithmetic
    rather than a value-at-risk estimate, and a claim that never settles has no
    such interval. The honest finite version is this: a stream with a last
    payment, after which the contract is worth whatever its terminal payoff
    says: zero, for a pure strip. Perpetuity would need funding rates and
    margin calls, which is a different risk model from the one this venue is
    built on, and adopting it silently would weaken the guarantee everything
    else here depends on.
    """

    windows: tuple[ObservationWindow, ...]
    # Applied to the metric level measured over each window, exactly as the
    # contract's own payoff is applied to the level over the whole window.
    payoff: Payoff

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError(
                "a distribution schedule with no windows is a future; leave "
                "distribution unset instead"
            )
        for earlier, later in zip(self.windows, self.windows[1:]):
            if earlier.end > later.start:
                raise ValueError(
                    f"distribution windows overlap: {_iso(earlier.end)} runs past "
                    f"{_iso(later.start)}. Overlapping periods would pay for the "
                    "same battles twice."
                )

    def bounds(self, level_bounds: tuple[float, float]) -> tuple[float, float]:
        """The range the whole stream can pay, across every window."""
        low, high = self.payoff.bounds(level_bounds)
        return (low * len(self.windows), high * len(self.windows))

    def to_dict(self) -> dict[str, Any]:
        return {
            "windows": [window.to_dict() for window in self.windows],
            "payoff": self.payoff.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ContractSpec:
    """Everything needed to settle a contract, and nothing that could change."""

    contract_id: str
    underlying: Underlying
    payoff: Payoff
    window: ObservationWindow
    policy: DataPolicy
    # Identity of the frozen reference snapshot supplying standardization
    # weights and shrinkage priors. Named rather than embedded so that many
    # contracts can share one audited snapshot, but pinned by id so that
    # swapping it changes this spec's digest.
    reference_id: str
    published_at: datetime
    tick_size: str = "0.01"
    # Coarser increments that apply above given prices, as ``(from, step)``
    # pairs in ascending order. Empty means one increment everywhere.
    #
    # Real venues do this because a tick has two jobs that pull against each
    # other. Too fine and the queue means nothing: anyone can step in front of
    # a resting order for a hundredth of a penny, so priority is worthless and
    # nobody posts size. Too coarse and the spread cannot narrow to what the
    # market actually knows. The resolution that gets that balance right at 4
    # is the wrong one at 4,000, so the increment scales with the price.
    #
    # ``tick_size`` remains the finest increment and the unit everything is
    # represented in: the engine matches on integer ticks and a variable unit
    # would make a tick index mean different prices at different levels. The
    # table is a rule about which prices may be *quoted*, enforced by the
    # venue, and every increment in it has to be a whole multiple of the base
    # or the rule would forbid prices the representation can express.
    tick_table: tuple[tuple[str, str], ...] = field(default=())
    lot_size: int = 1
    metadata: tuple[tuple[str, str], ...] = field(default=())
    # Interim payments, if this contract makes any. Unset for everything
    # that pays once at the end, which is every contract but a share.
    distribution: DistributionSchedule | None = None

    def __post_init__(self) -> None:
        if not self.contract_id:
            raise ValueError("contract_id is required")
        if self.published_at.tzinfo is None:
            raise ValueError("published_at must be timezone-aware")
        if self.published_at > self.window.start:
            raise ValueError(
                f"{self.contract_id}: published_at {_iso(self.published_at)} is after "
                f"window start {_iso(self.window.start)}; a contract written after its "
                "observation window opens is lookahead by construction"
            )
        if not self.reference_id:
            raise ValueError("reference_id is required; standardization must be pinned")
        if float(self.tick_size) <= 0:
            raise ValueError("tick_size must be positive")

        base = Decimal(self.tick_size)
        previous: Decimal | None = None
        for threshold, increment in self.tick_table:
            at = Decimal(threshold)
            step = Decimal(increment)
            if step <= 0:
                raise ValueError(f"tick increment {increment} must be positive")
            if step % base != 0:
                raise ValueError(
                    f"tick increment {increment} is not a multiple of the base tick "
                    f"{self.tick_size}; the table would forbid prices the engine can "
                    "represent, which is a rule nobody could follow"
                )
            if previous is not None and at <= previous:
                raise ValueError(
                    f"tick table thresholds must ascend; {threshold} follows {previous}"
                )
            previous = at
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if self.distribution is not None:
            first = self.distribution.windows[0]
            last = self.distribution.windows[-1]
            if first.start < self.window.start or last.end > self.window.end:
                raise ValueError(
                    f"{self.contract_id}: distributions run "
                    f"{_iso(first.start)}..{_iso(last.end)}, outside the "
                    f"observation window {_iso(self.window.start)}.."
                    f"{_iso(self.window.end)}. A payment measured outside "
                    "the window the contract observes is measured on "
                    "evidence the contract never claimed to be about."
                )

    def atoms(self) -> tuple[MetricRef, ...]:
        """Every metric the oracle must resolve to settle this contract."""
        return self.underlying.atoms()

    @property
    def settlement_bounds(self) -> tuple[Decimal, Decimal]:
        """The interval this contract can settle in, on the tick grid.

        Every instrument in this market settles inside a known range, because
        every underlying is a bounded statistic rather than an unbounded price.
        That makes a position's worst case exact arithmetic instead of a
        value-at-risk estimate, and therefore makes full collateralisation
        computable: a short at 5100 on a contract bounded by 10000 can lose at
        most 4900 per lot, and nothing needs to model volatility to know it.
        """
        low, high = self.payoff.bounds(self.underlying.bounds())
        return (
            quantize_to_tick(low, self.tick_size),
            quantize_to_tick(high, self.tick_size),
        )

    @property
    def value_bounds(self) -> tuple[Decimal, Decimal]:
        """The range the whole claim can be worth: settlement and every payment.

        The same thing as :attr:`settlement_bounds` for every contract that
        pays once. For a contract that pays as it goes, a short can be asked
        for the stream as well as for the settlement, so this is what
        collateral has to cover, and it is still arithmetic, because a bounded
        metric paid a fixed number of times is bounded too.
        """
        low, high = self.settlement_bounds
        if self.distribution is None:
            return (low, high)
        stream_low, stream_high = self.distribution.bounds(self.underlying.bounds())
        return (
            low + quantize_to_tick(stream_low, self.tick_size),
            high + quantize_to_tick(stream_high, self.tick_size),
        )

    def claim_value(self, level: float) -> float:
        """What the whole claim pays, at one level of the underlying.

        Every payment priced off the same level, which is what someone holding
        a single view of the underlying can do, and a single view is what an
        agent here has. Settlement resolves each window separately, so the
        realised stream will not be flat; this is the expectation, not the
        path.
        """
        total = self.payoff.apply(level)
        if self.distribution is not None:
            total += self.distribution.payoff.apply(level) * len(
                self.distribution.windows
            )
        return total

    def collateral_for(self, quantity: int, price: Decimal) -> Decimal:
        """Worst-case loss on ``quantity`` lots opened at ``price``.

        Positive quantity is long, negative is short. This is the amount that
        must be posted for the position to be fully collateralised in the sense
        Kalshi uses: the holder can lose it all, and can never owe more.

        Floored at zero. A position opened outside the range the claim can
        settle in cannot lose anything at all (a short above the top of the
        range makes money whatever happens), and the arithmetic ran negative
        there rather than stopping at nothing. Measured: a short of one lot at
        12,000 on a contract bounded by [0, 10000] returned -2,000. That figure
        is summed with the rest of a portfolio's requirement, so a negative one
        does not merely misreport, it *funds* another position. It also let the
        netted requirement exceed the gross, which is arithmetically impossible
        for correct inputs and was the reason the netting test carried a
        tolerance.
        """
        low, high = self.value_bounds
        if quantity >= 0:
            return max(Decimal(0), Decimal(quantity) * (price - low))
        return max(Decimal(0), Decimal(-quantity) * (high - price))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "underlying": self.underlying.to_dict(),
            "payoff": self.payoff.to_dict(),
            "window": self.window.to_dict(),
            "policy": self.policy.to_dict(),
            "reference_id": self.reference_id,
            "published_at": _iso(self.published_at),
            "tick_size": self.tick_size,
            "tick_table": [list(row) for row in self.tick_table],
            "lot_size": self.lot_size,
            "metadata": [list(pair) for pair in self.metadata],
            "distribution": (
                None if self.distribution is None else self.distribution.to_dict()
            ),
        }

    @property
    def spec_digest(self) -> str:
        """Content address of this specification."""
        return digest(self.to_dict())


def _iso(moment: datetime) -> str:
    """RFC 3339 in UTC with a trailing Z, so digests do not depend on offset spelling."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
