"""The interface a trading strategy is written against.

The point of this package is that somebody can write a market-making or
buy-side strategy, run it against a market that behaves like a market, and get
back a number they are entitled to believe: without real money, without a data
vendor, and without waiting for a session that only happens once a day. That
only works if the boundary between "the strategy" and "the exchange" is drawn
honestly, so this module draws it.

**A strategy sees a view and returns intents. It never touches the venue.**
That is not tidiness, it is the whole validity argument. An object holding a
reference to the venue can read the other participants' positions, the true
settlement level, or the book of a symbol it was never told about, and a
backtest of such a thing measures nothing. :class:`MarketView` is assembled
from exactly what the agent has been *sent* (its own fills, its own
acknowledgements, and the market data it subscribes to), so a strategy is
structurally unable to see anything a real desk could not.

**Staleness is preserved rather than smoothed away.** The view is built from
the agent's local books, which lag the venue by that agent's latency. A
strategy that reads a mid is reading the mid it would actually have had. This
is the single most common way a backtest lies, and the cheapest place to
refuse to.

**Prices leave as Decimal on the grid.** A strategy may model in floats (the
literature's formulas are floating-point and pretending otherwise would be
theatre), but the value it emits is quantised before it becomes an order, so
nothing floating-point ever reaches the ledger. :func:`snap` is where that
happens and strategies are expected to use it.

The two protocols are deliberately separate. A maker's job is to have a price
in the market at all times and be compensated for it; a taker's job is to
decide whether the price on the screen is wrong. They fail differently, they
are measured differently, one by realized spread, the other by hit rate and
edge, and a single `act()` method for both is what produced the makers in this
repository that take more often than they make.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from arena.agents.base import _on_grid
from arena.exchange.types import Side
from arena.market.instrument import Instrument

__all__ = [
    "Quote",
    "TwoSided",
    "Take",
    "SymbolView",
    "MarketView",
    "MakerStrategy",
    "TakerStrategy",
    "snap",
]


def snap(instrument: Instrument, side: Side, price: Decimal | float) -> Decimal:
    """Move a modelled price onto the grid, in the direction that is safe.

    Rounded *away* from the touch for a resting quote: a bid rounds down and an
    ask rounds up. Rounding a bid up would post it a tick better than the
    strategy asked for, which is a real order at a real price that nobody
    chose, and on a thin book that tick is most of the edge.
    """
    amount = price if isinstance(price, Decimal) else Decimal(str(price))
    tick = instrument.tick_size
    steps = amount / tick
    floor = int(steps)
    if steps != floor and steps < 0:
        floor -= 1
    ticks = floor if side is Side.BUY else (floor if steps == floor else floor + 1)
    low, high = instrument.tick_bounds
    ticks = max(int(low), min(int(high), ticks))
    # Then onto the increment the contract's band requires, which is not the
    # tick everywhere: VANTA_OBJECTIVE_WR steps by 1.00 above 4,000, so
    # rounding to the tick alone returns 5232.25, a price the venue refuses.
    # Delegated rather than repeated, because the subtle part is that one pass
    # can round *into* a coarser band and land off its grid, and `_on_grid`
    # already handles that and says why.
    #
    # `TradingAgent.quote` applies the same function before anything is sent,
    # so no bad order ever reached the venue. What was wrong is that the price
    # a strategy computed was not the price it would get, which is exactly the
    # comparison it makes against `working_bid` to decide whether to requote.
    return instrument.from_ticks(_on_grid(instrument, side, ticks))


@dataclass(frozen=True)
class Quote:
    """One side of an intended two-sided market."""

    price: Decimal
    size: int
    # Refuse this order outright rather than let it cross. The engine has
    # always supported it and the venue has always had a rejection reason for
    # it; the strategy layer simply had no way to ask, so every quote went out
    # good-till-cancelled and a maker whose fair value was through the touch
    # took liquidity instead of providing it.
    #
    # Off by default because it is not free. Measured on a 180s run: the
    # aggressive share of a fixed-spread maker's fills falls from 30.7% to
    # 1.4%, but 31,804 orders are refused, fills drop from 26,131 to 4,872,
    # and its P&L gets worse. A maker that cannot cross also cannot exit, so
    # this is a choice a strategy makes per quote and not a policy imposed on
    # every strategy.
    post_only: bool = False

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"a quote for {self.size} lots is not a quote")


@dataclass(frozen=True)
class TwoSided:
    """What a maker wants resting in one symbol.

    ``None`` on a side means *do not quote it*, which is a legitimate and
    frequently correct answer: pulling the side somebody keeps picking off is
    the cheapest defence there is, and a strategy that can only widen cannot
    express it. It is distinct from a zero size, which is not representable at
    all, because "an order for no lots" is not a thing to have an opinion
    about.
    """

    bid: Quote | None = None
    ask: Quote | None = None

    @property
    def is_empty(self) -> bool:
        return self.bid is None and self.ask is None


@dataclass(frozen=True)
class Take:
    """An intent to cross the spread.

    ``limit`` of ``None`` means marketable, and the venue's price band still
    applies. A taker cannot escape the listing rules by declining to name a
    price. Naming one is strictly safer and strategies are encouraged to.
    """

    symbol: str
    side: Side
    size: int
    limit: Decimal | None = None

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"a take of {self.size} lots is not a take")


@dataclass(frozen=True)
class SymbolView:
    """Everything a strategy may know about one contract, right now.

    Every field is either public market data or a fact about the strategy's
    own account. There is deliberately no settlement level, no other
    participant's position, and no future.
    """

    symbol: str
    instrument: Instrument
    best_bid: Decimal | None
    best_ask: Decimal | None
    last: Decimal | None
    position: int
    working_bid: Decimal | None
    working_ask: Decimal | None
    # Seconds of simulated time until this contract stops trading, or ``None``
    # where the calendar has not been wired. Not interchangeable with a real
    # clock and named in the units the strategy actually gets.
    seconds_to_expiry: float | None
    # The realized adverse selection on this strategy's own fills, per side, in
    # contract price units per lot: how far the mid moved against it after it
    # traded. GLFT's `xi`, measured rather than assumed. ``None`` until enough
    # of its own fills have matured, which is the honest answer early on.
    markout: Mapping[Side, float | None] = field(default_factory=dict)

    @property
    def mid(self) -> Decimal | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> Decimal | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @property
    def bounds(self) -> tuple[Decimal, Decimal]:
        """The range this contract can settle in.

        Public, not privileged: it is written in the contract and is the same
        thing the collateral engine charges against. A strategy that ignores it
        will quote prices the contract cannot pay.
        """
        return self.instrument.spec.value_bounds

    @property
    def reference(self) -> Decimal | None:
        """Mid where there is one, last print otherwise, midpoint of the range
        if the book has never traded. Somewhere to start, not a fair value.

        Tested with `is None` rather than for truth, because a price of zero is
        a real price here and `or` discards it. Latent when found (zero of
        1,200 sampled mids across 47 books were exactly zero) but one cancelled
        offer away in seven option books, where a genuinely worthless put would
        have reported the midpoint of its settlement range instead, which on
        the call that was measured is 2,350 against a fair value of nothing.
        """
        mid = self.mid
        if mid is not None:
            return mid
        if self.last is not None:
            return self.last
        low, high = self.bounds
        return (low + high) / 2


@dataclass(frozen=True)
class MarketView:
    """The whole of what a strategy is allowed to see, at one instant."""

    now: float
    symbols: tuple[str, ...]
    cash: Decimal
    free_cash: Decimal
    posted_collateral: Decimal
    equity: Decimal
    _by_symbol: Mapping[str, SymbolView] = field(default_factory=dict)
    # A deterministic source, seeded per agent so a strategy's draws do not
    # depend on how many other agents happen to be in the market. Anything
    # random a strategy does must come from here or the run stops being
    # reproducible, which is most of what this exchange is for.
    rng: Any = None

    def __getitem__(self, symbol: str) -> SymbolView:
        return self._by_symbol[symbol]

    def get(self, symbol: str) -> SymbolView | None:
        return self._by_symbol.get(symbol)

    def __iter__(self):
        return iter(self._by_symbol.values())

    def __contains__(self, symbol: object) -> bool:
        return symbol in self._by_symbol

    def positions(self) -> dict[str, int]:
        return {v.symbol: v.position for v in self._by_symbol.values() if v.position}


@runtime_checkable
class MakerStrategy(Protocol):
    """Posts two-sided prices and is paid for the risk of doing so."""

    def quote(self, view: MarketView, symbol: str) -> TwoSided:
        """What this strategy wants resting in one symbol, right now.

        Called on the agent's own schedule and again whenever it is filled,
        because in Glosten-Milgrom the arrival of an order *is* the news: a
        quote that does not move when it is lifted can be lifted again at the
        same price, which is measurably what happens here when it does not.

        Returning the same prices as last time costs nothing: the adapter
        compares against what is already working and sends only differences, so
        an unchanged quote keeps its place in the queue instead of going to the
        back of it.
        """
        ...

    def symbols(self, view: MarketView) -> Sequence[str]:
        """Which contracts to quote. Default is everything the agent lists."""
        ...


@runtime_checkable
class TakerStrategy(Protocol):
    """Decides the price on the screen is wrong and pays the spread to say so."""

    def orders(self, view: MarketView) -> Sequence[Take]:
        """Everything this strategy wants to execute now, in priority order.

        Priority matters because collateral is finite and the adapter stops at
        the first intent the account cannot fund. A strategy that wants a
        package rather than a list should say so by ordering its legs, and
        should expect to be left half-legged sometimes, which is a real
        execution risk and not an artefact.
        """
        ...
