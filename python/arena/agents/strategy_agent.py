"""Runs a strategy as a participant, without letting it become one.

:mod:`arena.strategies.base` says what a strategy is: something that reads a
view and returns intents. This is the only thing that turns those intents into
orders, and it exists so that a strategy author never writes agent plumbing,
never sees the venue, and cannot accidentally cheat.

Three jobs, and each of them is somewhere a naive version loses money for
reasons that have nothing to do with the strategy:

**It sends differences, not quotes.** A maker that cancels and reposts a price
that has not moved gives up its place in the queue for nothing. That was
measured here once already: removing the unconditional repost took the event
rate from 1.6M per simulated minute to 317K. So the adapter compares what the
strategy wants against what is already working and leaves an unchanged side
exactly where it is.

**It requotes on fill.** Glosten-Milgrom's ask *is* ``E[V | the next order is a
buy]``, so being lifted is news and the quote has to move. On a schedule
alone it does not: measured on this market, 17% of a maker's passive fills were
a second fill at the same price within 500ms. The strategy is asked again the
moment it trades, and what it does with that is its business.

**It measures its own markout and hands it back.** GLFT's adverse-selection
term ``xi`` is not a parameter to be guessed, it is the drift of the mid after
your own fills, and every desk computes it from its own trades. The adapter
does that here (from this agent's fills and this agent's view of the mid, so it
stays inside what the strategy is allowed to know) and puts it in the view. A
strategy can then defend itself, or ignore it, and the difference is
measurable.

The shadow ledger is the same idea. An agent is never told its cash; it knows
its fills, so it adds them up, exactly as a desk reconciles its own book
against the clearer's. Collateral is computed with the same public arithmetic
the venue charges with, per contract and ungrossed, because that is what the
venue actually charges today.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from arena.agents.base import TradingAgent
from arena.exchange.events import Filled
from arena.portfolio.account import Account
from arena.portfolio.money import MONEY_SCALE, from_money
from arena.exchange.types import AgentId, Side, TimeInForce
from arena.market.instrument import Instrument
from arena.sim.kernel import SimulationContext
from arena.sim.time import Duration, millis, seconds
from arena.strategies.base import (
    MakerStrategy,
    MarketView,
    SymbolView,
    TakerStrategy,
    TwoSided,
    snap,
)

__all__ = ["StrategyAgent", "MARKOUT_HORIZON"]

# How long after its own fill the adapter looks, to decide whether the fill was
# any good. One second, because that is where this market's damage actually
# arrives: measured over 600s on seed 7, the makers' adverse selection is
# *positive* through the first half second and only turns over between one and
# five. A horizon inside the first hundred milliseconds would measure latency,
# which is not the problem here, and would report the opposite sign.
MARKOUT_HORIZON: Duration = seconds(1)

# How much of the past the markout estimate keeps. A plain EWMA, because the
# quantity is a running average of a noisy per-fill number and anything more
# elaborate would be fitting.
MARKOUT_GAIN = 0.15


@dataclass
class _OpenMarkout:
    at: int
    symbol: str
    side: Side
    signed: int
    mid_before: float


class StrategyAgent(TradingAgent):
    """One strategy, wearing an agent.

    A maker, a taker, or both: a firm that quotes one asset class and takes in
    another is an ordinary shape and refusing it would be an artificial limit.
    What is not allowed is a single method that does both, which is how the
    makers in this repository ended up aggressive on 61% of their fills.
    """

    def __init__(
        self,
        agent_id: AgentId,
        venue_id: AgentId,
        instruments: dict[str, Instrument],
        wake_interval: Duration = millis(300),
        *,
        maker: MakerStrategy | None = None,
        taker: TakerStrategy | None = None,
        starting_cash: Decimal = Decimal(0),
        requote_on_fill: bool = True,
    ) -> None:
        if maker is None and taker is None:
            raise ValueError("a strategy agent needs a maker, a taker, or both")
        super().__init__(agent_id, venue_id, instruments, wake_interval)
        self.maker = maker
        self.taker = taker
        self.requote_on_fill = requote_on_fill
        self.starting_cash = starting_cash
        # The strategy's own book, kept with the venue's own arithmetic.
        #
        # An agent is never told its cash, so it adds its fills up, exactly as a
        # desk reconciles its own book against the clearer's. Using `Account`
        # rather than a hand-rolled tally is not a shortcut: it is public
        # arithmetic on this agent's own fills, the same way `worst_case` is,
        # and it is the only way the two agree by construction.
        #
        # The first version of this debited notional on every fill, which is
        # spot accounting on a venue that does not do spot. `Account.apply_fill`
        # says why in as many words: a futures position is a collateralised
        # commitment rather than a purchase, so only realised P&L and fees move
        # cash before settlement. Measured against the venue after 120 simulated
        # seconds, the hand-rolled version was 125,564 light on cash and, worse,
        # charged 242,354 of collateral where the venue charged 144,392, because
        # it valued the requirement at the current mid instead of the basis the
        # position actually holds. A strategy sizing off that under-trades by
        # two thirds.
        self._book = Account(f"shadow-{agent_id}", self._minor(starting_cash))
        # Fills of this agent's own, waiting for the markout horizon to pass.
        self._open_markouts: deque[_OpenMarkout] = deque()
        self._markout: dict[tuple[str, Side], float] = {}
        self._now = 0

    # -- what the strategy is allowed to see --------------------------------

    def _symbol_view(self, symbol: str) -> SymbolView:
        instrument = self.instruments[symbol]
        book = self.books[symbol]
        working = {}
        for side in (Side.BUY, Side.SELL):
            intent = self._intent.get((symbol, side))
            working[side] = (
                instrument.from_ticks(intent[0]) if intent is not None else None
            )
        return SymbolView(
            symbol=symbol,
            instrument=instrument,
            best_bid=None if book.bid is None else instrument.from_ticks(book.bid),
            best_ask=None if book.ask is None else instrument.from_ticks(book.ask),
            last=None if book.last is None else instrument.from_ticks(book.last),
            position=self.position.get(symbol, 0),
            working_bid=working[Side.BUY],
            working_ask=working[Side.SELL],
            seconds_to_expiry=None,
            markout={
                Side.BUY: self._markout.get((symbol, Side.BUY)),
                Side.SELL: self._markout.get((symbol, Side.SELL)),
            },
        )

    def view(self, ctx: SimulationContext) -> MarketView:
        """Assemble the strategy's whole picture of the world.

        Built fresh each wake rather than mutated, so a strategy that keeps a
        reference to last tick's view is holding last tick's numbers instead of
        silently seeing the present through a stale object.
        """
        by_symbol = {symbol: self._symbol_view(symbol) for symbol in self.instruments}
        collateral = self.posted_collateral()
        return MarketView(
            now=int(ctx.now) / 1e9,
            symbols=tuple(self.instruments),
            cash=self.cash,
            free_cash=self.cash - collateral,
            posted_collateral=collateral,
            equity=self.cash + self.unrealized(),
            _by_symbol=by_symbol,
            rng=getattr(ctx, "rng", None),
        )

    @staticmethod
    def _minor(amount: Decimal) -> int:
        return int(amount * MONEY_SCALE)

    @property
    def cash(self) -> Decimal:
        """Realised P&L against the starting balance, not a purchase ledger."""
        return from_money(self._book.cash)

    def posted_collateral(self) -> Decimal:
        """What this agent's open positions cost to hold.

        Charged against each position's own basis, which is what it would
        actually lose, and per contract rather than netted, because that is
        what the venue charges today. Computed here rather than asked for: the
        arithmetic is public, and an agent that had to ask would be an agent
        with a channel to privileged state.
        """
        return from_money(sum(self._book.collateral.values(), start=0))

    def unrealized(self) -> Decimal:
        marks = {}
        for symbol, instrument in self.instruments.items():
            book = self.books[symbol]
            reference = book.mid if book.mid is not None else book.last
            if reference is None:
                continue
            marks[symbol] = instrument.price_in_minor(int(reference))
        return from_money(self._book.equity(marks)) - self.cash

    # -- measuring its own fills -------------------------------------------

    def _on_private(self, ctx: SimulationContext, event: Any, symbol: str) -> None:
        """Track cash, open a markout, and ask the strategy again.

        Overriding the private handler rather than the public `on_private`
        hook, because a `Filled` does not carry its symbol; the venue sends one
        private channel per book and the symbol arrives with the envelope, not
        with the event. Read from the event, `symbol` is always `None`, and an
        override that trusts it silently books nothing: measured, 386 fills
        moved this ledger by exactly zero.
        """
        super()._on_private(ctx, event, symbol)
        if not isinstance(event, Filled):
            return
        quantity, price, side = event.quantity, event.price, event.side
        if price is None or symbol not in self.instruments:
            return
        instrument = self.instruments[symbol]
        signed = int(quantity) * (1 if side is Side.BUY else -1)
        # Fee zero, because a `Filled` does not carry one and the venue never
        # tells an agent what it was charged. That is realistic rather than a
        # gap (a desk reconciles fees afterwards), but it means this book is
        # pre-fee and will sit slightly above the venue's. Measured on the
        # incumbent makers, fees were 0.3% of P&L.
        self._book.apply_fill(
            symbol,
            signed,
            instrument.price_in_minor(int(price)),
            instrument.bounds_in_minor,
        )

        book = self.books[symbol]
        mid = book.mid if book.mid is not None else float(int(price))
        self._open_markouts.append(
            _OpenMarkout(int(ctx.now), symbol, side, signed, float(mid))
        )
        if self.requote_on_fill and self.maker is not None:
            # Now, not at the next wake. The whole point is that being filled
            # is news, and news that waits 300ms for a timer is a quote that
            # can be lifted again at the same price in the meantime.
            self._apply_quote(ctx, symbol, self.maker.quote(self.view(ctx), symbol))

    def _mature_markouts(self, ctx: SimulationContext) -> None:
        """Close out fills old enough to have an answer.

        Signed so that a positive number always means the same thing: the mid
        moved the strategy's way after it traded. Negative is being picked off,
        in either direction, which is the only convention under which a maker
        can add its bid and ask markouts together and get something meaningful.
        """
        now = int(ctx.now)
        while self._open_markouts and now >= self._open_markouts[0].at + MARKOUT_HORIZON:
            fill = self._open_markouts.popleft()
            book = self.books[fill.symbol]
            if book.mid is None:
                continue
            instrument = self.instruments[fill.symbol]
            drift_ticks = (book.mid - fill.mid_before) * (1 if fill.signed > 0 else -1)
            per_lot = float(instrument.tick_size) * drift_ticks
            key = (fill.symbol, fill.side)
            previous = self._markout.get(key)
            self._markout[key] = (
                per_lot
                if previous is None
                else previous + MARKOUT_GAIN * (per_lot - previous)
            )

    # -- driving the strategy ----------------------------------------------

    def act(self, ctx: SimulationContext) -> None:
        self._mature_markouts(ctx)
        view = self.view(ctx)
        if self.maker is not None:
            wanted = self._maker_symbols(view)
            for symbol in wanted:
                self._apply_quote(ctx, symbol, self.maker.quote(view, symbol))
        if self.taker is not None:
            for intent in self.taker.orders(view):
                self._apply_take(ctx, intent)

    def _maker_symbols(self, view: MarketView) -> list[str]:
        chooser = getattr(self.maker, "symbols", None)
        if chooser is None:
            return list(self.instruments)
        return [s for s in chooser(view) if s in self.instruments]

    def _apply_quote(
        self, ctx: SimulationContext, symbol: str, wanted: TwoSided
    ) -> None:
        instrument = self.instruments[symbol]
        for side, quote in ((Side.BUY, wanted.bid), (Side.SELL, wanted.ask)):
            if quote is None:
                self.withdraw(ctx, symbol, side)
                continue
            # `post` is what decides whether this is a change worth sending,
            # and it decides on the price that will actually rest, after the
            # range clamp and the grid snap. Repeating that test here against a
            # differently-rounded number would report a move that is not one.
            price = snap(instrument, side, quote.price)
            self.post(
                ctx,
                symbol,
                side,
                instrument.to_ticks(price),
                int(quote.size),
                TimeInForce.POST_ONLY if quote.post_only else TimeInForce.GTC,
            )

    def _apply_take(self, ctx: SimulationContext, intent: Any) -> None:
        symbol = intent.symbol
        if symbol not in self.instruments:
            return
        instrument = self.instruments[symbol]
        if intent.limit is None:
            self.take(ctx, symbol, intent.side, int(intent.size))
            return
        price = snap(instrument, intent.side, intent.limit)
        self.quote(
            ctx,
            symbol,
            intent.side,
            instrument.to_ticks(price),
            int(intent.size),
            TimeInForce.IOC,
        )
