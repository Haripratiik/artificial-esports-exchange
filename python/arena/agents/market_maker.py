"""An inventory-skewing market maker.

Quotes both sides around a reference price and shades those quotes against its
own position, so the trade that flattens it is the more attractive one. That is
the essential mechanic of market making: earn the spread, but do not accumulate
a directional bet you were never paid to hold.

    reservation = reference - inventory * skew
    bid         = reservation - half_spread
    ask         = reservation + half_spread

This is deliberately *not* Avellaneda-Stoikov. The proper treatment for these
contracts is more interesting than the classical one and belongs in its own
phase: because every instrument here settles inside a known interval, price
volatility is state-dependent and vanishes at the boundaries, and inventory held
to expiry faces settlement risk rather than liquidation risk. The published
adaptation adds a terminal penalty proportional to ``q^2 * p * (1-p)`` for
exactly that reason. Building that on top of a maker whose basic inventory
control has not been observed working would be premature.

What this maker does have, and what a naive one usually lacks:

* **A position limit it actually respects.** Without one, an inventory-skewing
  maker in a trending market accumulates without bound and dies.
* **Widening under uncertainty.** When its view of the book is stale or the
  market has moved against its inventory, it quotes wider. A maker that quotes
  the same spread regardless of what it knows is being adversely selected on
  purpose.
"""

from __future__ import annotations

from arena.agents.base import TradingAgent
from arena.exchange.types import AgentId, Price, Side, TimeInForce
from arena.market.instrument import Instrument
from arena.sim.kernel import SimulationContext
from arena.sim.messages import TradePrint
from arena.sim.time import Duration, millis

__all__ = ["MarketMaker"]


class MarketMaker(TradingAgent):
    """Two-sided quoting with inventory skew and a hard position limit."""

    def __init__(
        self,
        agent_id: AgentId,
        venue_id: AgentId,
        instruments: dict[str, Instrument],
        wake_interval: Duration = millis(250),
        half_spread: int = 4,
        quote_size: int = 25,
        max_skew_fraction: float = 0.12,
        position_limit: int = 400,
        reference: dict[str, float] | None = None,
        trade_weight: float = 0.15,
        quote_without_reference: bool = True,
    ) -> None:
        super().__init__(agent_id, venue_id, instruments, wake_interval)
        self.half_spread = half_spread
        self.quote_size = quote_size
        # How far a *full* inventory shifts the reservation price, as a fraction
        # of the contract's whole settlement range.
        #
        # Expressed as a fraction rather than as ticks-per-lot on purpose. An
        # absolute skew means wildly different things across instruments here: a
        # win-rate future spans 40,000 ticks and a binary spans 100, so a
        # constant that gives sensible inventory control on one leaves the other
        # either immovable or violently unstable. Every contract in this market
        # has a known range, so parameters can be written as fractions of it and
        # mean the same thing everywhere.
        self.max_skew_fraction = max_skew_fraction
        self.position_limit = position_limit
        # An opening anchor per symbol, used only until trades start printing.
        # Someone has to name an opening price, and on a venue with no prior
        # session that someone is the market maker.
        self.reference = reference or {}
        self.trade_weight = trade_weight
        # Whether to quote a book that has neither traded nor been given an
        # opening price, by falling back to the middle of its settlement range.
        #
        # False is the right answer on a venue that opens with a call auction,
        # and it took running one to see why. The auction is the discovery
        # mechanism; if the maker turns up to it with a mid-range guess, the
        # auction clears at the guess, that guess becomes the official opening
        # price, and the market's subsequent walk to fair value is a 6% move
        # that trips the circuit breaker. Measured: every one of 26 symbols
        # paused inside the first minute. A real maker does not name the
        # opening price: the interest in the auction does, and the maker quotes
        # around the print.
        self.quote_without_reference = quote_without_reference
        # A slow average of where trades actually print. This, not the book's
        # mid, is what the maker anchors on, and the distinction is the whole
        # reason the market can discover a price at all.
        self._anchor: dict[str, float] = {}

    def on_print(self, ctx: SimulationContext, print_: TradePrint) -> None:
        """Drag the anchor toward wherever trades are happening.

        Anchoring on the book's mid instead is the obvious thing to do and it
        is badly wrong: the maker *is* both sides of that mid, so it would be
        quoting around its own quotes and the price could never move however
        one-sided the flow got. Anchoring on executions makes the maker follow
        the flow that is actually lifting or hitting it, which is precisely the
        adverse-selection channel these experiments exist to measure, and it
        should be visible rather than assumed away.

        The gain is fixed, and one adaptive alternative was measured and
        rejected rather than merely considered. Splitting the residual into a
        drift and a magnitude, both smoothed at ``trade_weight``, and using
        Trigg and Leach's tracking signal ``|drift| / magnitude`` to raise the
        gain toward one on a run of same-signed residuals is the textbook
        answer, and here it made the maker 2.2 times worse: on seed 7 over 180s
        the three makers lost 65.7M against 29.3M, while the share of passive
        fills carrying a negative effective half spread went the wrong way,
        from 10.5% to 12.3%. The reason is specific to this market and worth
        recording. These makers are about 98% of the book, so almost every
        print lands on one of their own quotes and the residual is ``-half`` or
        ``+half`` by construction rather than because the price moved. A
        tracking signal reads that as a trend, the gain saturates at one, the
        anchor steps onto the last print, and the informed flow then walks the
        maker's own price wherever it likes. It is the same failure as widening
        against the markout, arriving through the estimator instead of through
        the spread: on a market this thin the maker moves the thing it is
        measuring itself against.
        """
        current = self._anchor.get(print_.symbol)
        price = float(int(print_.price))
        if current is None:
            self._anchor[print_.symbol] = price
        else:
            self._anchor[print_.symbol] = (
                current + self.trade_weight * (price - current)
            )

    def act(self, ctx: SimulationContext) -> None:
        for symbol in sorted(self.instruments):
            self._requote(ctx, symbol)

    def _requote(self, ctx: SimulationContext, symbol: str) -> None:
        instrument = self.instruments[symbol]
        inventory = self.position.get(symbol, 0)

        anchor = self._anchor.get(symbol)
        if anchor is None:
            anchor = self.reference.get(symbol)
        if anchor is None:
            if not self.quote_without_reference:
                return
            low, high = instrument.tick_bounds
            anchor = (int(low) + int(high)) / 2.0

        low, high = instrument.tick_bounds
        span = max(1.0, float(int(high) - int(low)))
        skew_per_lot = (span * self.max_skew_fraction) / max(1, self.position_limit)
        reservation = anchor - inventory * skew_per_lot

        # Widen with inventory. A maker deep in a position is more likely to be
        # on the wrong side of whatever is driving the market, so the extra
        # width is compensation for that risk rather than greed.
        pressure = abs(inventory) / max(1, self.position_limit)
        half = self.half_spread * (1.0 + 2.0 * pressure)

        bid = Price(int(reservation - half))
        ask = Price(int(reservation + half) + 1)

        # Respect the limit by simply not adding to the side that would breach
        # it. Quoting and relying on the venue to reject is worse: it burns
        # order ids and hides the constraint from the agent's own logic.
        if inventory < self.position_limit:
            self.post(ctx, symbol, Side.BUY, bid,
                       min(self.quote_size, self.position_limit - inventory))
        else:
            self.withdraw(ctx, symbol, Side.BUY)
        if -inventory < self.position_limit:
            self.post(ctx, symbol, Side.SELL, ask,
                       min(self.quote_size, self.position_limit + inventory))
        else:
            self.withdraw(ctx, symbol, Side.SELL)

    def _post(
        self, ctx: SimulationContext, symbol: str, side: Side, price: Price, size: int
    ) -> None:
        """Work a quote, replacing the existing one only if it has moved."""
        wanted = (int(price), int(size))
        working = any(
            order_symbol == symbol and self.order_side.get(key) is side
            for key, order_symbol in self.live_orders.items()
        )
        if working and self._intent.get((symbol, side)) == wanted:
            return
        self.cancel_side(ctx, symbol, side)
        self._intent[(symbol, side)] = wanted
        self.quote(ctx, symbol, side, price, size, TimeInForce.GTC)

    def _withdraw(self, ctx: SimulationContext, symbol: str, side: Side) -> None:
        if self.cancel_side(ctx, symbol, side):
            self._intent.pop((symbol, side), None)
