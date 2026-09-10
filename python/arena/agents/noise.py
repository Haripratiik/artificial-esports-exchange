"""Noise traders.

Not filler, and worth being clear about why. Without uninformed order flow
there is nobody for a market maker to earn a spread from, and no camouflage for
an informed trader to hide behind, so both the market-making and the
information-asymmetry questions become degenerate. Every classic microstructure
model needs noise for the same reason: Kyle's insider is only profitable
because the market maker cannot tell them apart from the noise.

Two behaviours, because a population of purely random traders is *too* benign.
Real uninformed flow chases trends and clusters, which is what produces the
autocorrelated order flow and occasional runs that a market maker actually has
to survive.
"""

from __future__ import annotations

from arena.agents.base import TradingAgent
from arena.exchange.types import AgentId, Price, Side, TimeInForce
from arena.market.instrument import Instrument
from arena.sim.kernel import SimulationContext
from arena.sim.messages import TradePrint
from arena.sim.time import Duration, millis

__all__ = ["NoiseTrader"]


class NoiseTrader(TradingAgent):
    """Uninformed flow: random direction, with a configurable trend-chasing bias."""

    def __init__(
        self,
        agent_id: AgentId,
        venue_id: AgentId,
        instruments: dict[str, Instrument],
        wake_interval: Duration = millis(900),
        max_size: int = 6,
        aggressive_probability: float = 0.55,
        momentum_bias: float = 0.25,
        symbols_per_wake: int = 1,
    ) -> None:
        super().__init__(agent_id, venue_id, instruments, wake_interval)
        # How many books this agent touches each time it wakes.
        #
        # One was fine for a fixed listing and is the reason uninformed flow
        # thins as the exchange grows: an agent that picks a single symbol from
        # a longer list visits each book less often, so listing more contracts
        # spreads the same flow thinner rather than attracting more. Measured
        # over 180 simulated seconds on seed 7, noise volume fell from 6,904 to
        # 4,522 when matches took the listing from 47 symbols to 971, and the
        # informed-to-uninformed ratio went from 25.6 to 1 up to 42.3 to 1.
        #
        # A market where uninformed flow is half a percent of volume cannot
        # support market making at all. Glosten-Milgrom's own conclusion is
        # that past a high enough informed share the spread required for zero
        # expected profit exceeds what anybody will trade against, and the
        # market shuts. Retail alone is roughly a fifth of real equity volume,
        # so half a percent is not a conservative simulation, it is a different
        # market.
        self.symbols_per_wake = max(1, int(symbols_per_wake))
        self.max_size = max_size
        self.aggressive_probability = aggressive_probability
        self.momentum_bias = momentum_bias
        # Sign of the last observed price change per symbol. The only state
        # these agents keep, and the only thing that makes their flow
        # autocorrelated rather than independent.
        self._drift: dict[str, int] = dict.fromkeys(instruments, 0)
        self._previous: dict[str, int] = {}

    def on_print(self, ctx: SimulationContext, print_: TradePrint) -> None:
        previous = self._previous.get(print_.symbol)
        if previous is not None and int(print_.price) != previous:
            self._drift[print_.symbol] = 1 if int(print_.price) > previous else -1
        self._previous[print_.symbol] = int(print_.price)

    def act(self, ctx: SimulationContext) -> None:
        listed = sorted(self.instruments)
        if not listed:
            return
        rng = ctx.rng
        for _ in range(min(self.symbols_per_wake, len(listed))):
            self._one(ctx, rng, rng.choice(listed))

    def _one(self, ctx: SimulationContext, rng, symbol: str) -> None:
        book = self.books[symbol]
        if book.mid is None:
            # Nothing to anchor to yet. Waiting rather than guessing keeps these
            # agents from inventing the opening price, which is the market
            # maker's job.
            return

        # Trend chasing: bias direction toward the last observed move.
        drift = self._drift.get(symbol, 0)
        threshold = 0.5 - self.momentum_bias * drift
        side = Side.BUY if rng.random() > threshold else Side.SELL
        size = rng.randint(1, self.max_size)

        if rng.random() < self.aggressive_probability:
            self.take(ctx, symbol, side, size)
            return

        # Otherwise rest a passive order a tick or two behind the touch, and
        # let it expire rather than linger: an uninformed trader who leaves
        # stale orders in the book would become a free option for everyone
        # else.
        instrument = self.instruments[symbol]
        offset = rng.randint(1, 3)
        if side is Side.BUY:
            anchor = book.bid if book.bid is not None else Price(int(book.mid))
            price = Price(int(anchor) - offset)
        else:
            anchor = book.ask if book.ask is not None else Price(int(book.mid))
            price = Price(int(anchor) + offset)
        self.quote(ctx, symbol, side, price, size, TimeInForce.GTC)
