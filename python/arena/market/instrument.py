"""What is actually tradeable, and what it settles into.

This module is the join that was missing. Until now the project had two halves
that never met: a contract layer that could define and settle a future, and an
exchange that could match orders in an abstract instrument. Nothing said *this
symbol on this venue settles according to that contract*, so a trade could never
turn into a settlement and a settlement could never turn into PnL.

An :class:`Instrument` is that statement. It carries:

    symbol        what agents quote and trade
    spec          the ContractSpec it settles by, digest and all
    tick / lot    the grid the exchange enforces
    expiry        when trading stops and settlement is attempted

The exchange still knows nothing about any of it. The venue translates between
integer ticks (the only thing the matching engine understands) and the Decimal
prices a contract settles in, at the boundary and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from arena.contracts.spec import ContractSpec
from arena.determinism import digest
from arena.exchange.types import Price
from arena.portfolio.money import Money, tick_in_minor, to_money

__all__ = ["Instrument", "InstrumentClass"]


class InstrumentClass:
    """What kind of thing this is, for reporting and for grouping experiments.

    Derived from the contract rather than declared, so it cannot disagree with
    what the instrument actually pays.
    """

    FUTURE = "future"
    EVENT = "event"
    SPREAD = "spread"
    INDEX = "index"
    CALL = "call"
    PUT = "put"
    # A linear claim on an amount delivered over a window, rather than on a
    # proportion. Different economics: a delivery window is a real part of
    # the contract, so these come in term structures and their prices carry
    # information about carry rather than only about the level.
    COMMODITY = "commodity"
    # A claim that pays while it is alive and settles at the end, rather
    # than paying once. What a share is, minus the perpetuity that this
    # venue's collateral model cannot express.
    EQUITY = "equity"
    # A claim on how far apart something's outcomes are, rather than on where
    # they sit. Two subjects with the same average and different spreads are
    # not the same thing to own, and this is the contract that says so.
    VOLATILITY = "volatility"


@dataclass(frozen=True, slots=True)
class Instrument:
    """A tradeable symbol bound to the contract that settles it."""

    symbol: str
    spec: ContractSpec
    lot_size: int = 1

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol is required")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")

    # -- the tick grid -----------------------------------------------------

    @property
    def tick_size(self) -> Decimal:
        return Decimal(self.spec.tick_size)

    def to_ticks(self, price: Decimal | float | int) -> Price:
        """Convert a contract price to the integer ticks the engine matches on.

        Exact division is required, not rounded: an agent quoting off the grid
        is a bug in the agent, and silently rounding it would put the order at
        a price the agent did not choose, then fill it there.
        """
        amount = price if isinstance(price, Decimal) else Decimal(str(price))
        ticks = amount / self.tick_size
        if ticks != ticks.to_integral_value():
            raise ValueError(
                f"{price} is not a multiple of {self.symbol}'s tick size "
                f"{self.tick_size}"
            )
        return Price(int(ticks))

    def from_ticks(self, ticks: Price | int) -> Decimal:
        return Decimal(int(ticks)) * self.tick_size

    def increment_at(self, price: Decimal) -> Decimal:
        """The smallest step a quote at this price may move in.

        The base tick below the table's first threshold, and the coarsest row
        whose threshold the price has reached above it. A contract with no
        table has one increment everywhere, which is what every contract here
        had until there was a reason not to.
        """
        step = self.tick_size
        for threshold, increment in self.spec.tick_table:
            if abs(price) >= Decimal(threshold):
                step = Decimal(increment)
        return step

    def on_grid(self, price: Decimal) -> bool:
        """Whether a quote at this price sits on the increment it must."""
        return price % self.increment_at(price) == 0

    # -- bounds and collateral ---------------------------------------------

    @property
    def settlement_bounds(self) -> tuple[Decimal, Decimal]:
        """What the contract can settle at, at the end."""
        return self.spec.settlement_bounds

    @property
    def value_bounds(self) -> tuple[Decimal, Decimal]:
        """What the whole claim can be worth, payments included.

        Identical to :attr:`settlement_bounds` for anything that pays once,
        which is everything but a share. Collateral, price bands and the
        opening anchor all work from this rather than from the settlement
        range, because a short in something that pays as it goes can be asked
        for the stream too.
        """
        return self.spec.value_bounds

    @property
    def tick_bounds(self) -> tuple[Price, Price]:
        low, high = self.value_bounds
        return (self.to_ticks(low), self.to_ticks(high))

    @property
    def tick_in_minor(self) -> int:
        """One tick expressed in the ledger's minor money units."""
        return tick_in_minor(self.tick_size)

    @property
    def bounds_in_minor(self) -> tuple[Money, Money]:
        low, high = self.value_bounds
        return (to_money(low), to_money(high))

    def price_in_minor(self, ticks: Price | int) -> Money:
        """Convert a tick price to minor units, exactly."""
        return Money(int(ticks) * self.tick_in_minor)

    def collateral_for(self, quantity: int, price: Decimal) -> Decimal:
        return self.spec.collateral_for(quantity, price)

    # -- lifecycle ---------------------------------------------------------

    @property
    def expiry(self) -> datetime:
        """Trading stops when the observation window closes.

        Not later: once the window has closed the outcome is determined, and
        anyone who learns it before the market does would be trading on a known
        answer. The *settlement* may well happen later, once collection catches
        up, but trading must not continue into that gap.
        """
        return self.spec.window.end

    @property
    def instrument_class(self) -> str:
        from arena.contracts.payoff import Binary, Call, Put
        from arena.contracts.underlying import Basket, Difference

        # The payoff decides first: an option on a spread is an option, not a
        # spread, because what it pays is shaped by the strike rather than by
        # the shape of what it is written on.
        if isinstance(self.spec.payoff, Call):
            return InstrumentClass.CALL
        if isinstance(self.spec.payoff, Put):
            return InstrumentClass.PUT
        if isinstance(self.spec.payoff, Binary):
            return InstrumentClass.EVENT
        # Paying before it settles is what makes a share a share, so it decides
        # ahead of what the payments are written on.
        if self.spec.distribution is not None:
            return InstrumentClass.EQUITY
        if isinstance(self.spec.underlying, Difference):
            return InstrumentClass.SPREAD
        if isinstance(self.spec.underlying, Basket):
            return InstrumentClass.INDEX
        # A linear claim on a quantity is a commodity; on a rate, a future. The
        # metric declares which it is, so this layer never has to know what the
        # subject means.
        reference = getattr(self.spec.underlying, "ref", None)
        kind = "rate" if reference is None else getattr(reference, "kind", "rate")
        if kind == "quantity":
            return InstrumentClass.COMMODITY
        if kind == "dispersion":
            return InstrumentClass.VOLATILITY
        return InstrumentClass.FUTURE

    def to_dict(self) -> dict[str, Any]:
        low, high = self.settlement_bounds
        return {
            "symbol": self.symbol,
            "class": self.instrument_class,
            "contract_id": self.spec.contract_id,
            "spec_digest": self.spec.spec_digest,
            "tick_size": str(self.tick_size),
            "lot_size": self.lot_size,
            "settlement_bounds": [str(low), str(high)],
            "expiry": self.expiry.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    @property
    def instrument_digest(self) -> str:
        return digest(self.to_dict())
