"""Order flow with the shape real order flow has, and a warning about it.

Real limit order books do not look like a stream of uniform orders arriving on
a Poisson clock. Four regularities are robust across markets and decades:

* **Order sizes are power-law distributed.** A few enormous orders among very
  many small ones. Gopikrishnan et al. (2000) and Maslov & Mills (2001) put the
  tail exponent for market-order volume around 2.3-2.5.
* **Limit orders are placed a power-law distance from the touch.** Zovko &
  Farmer (2002) find the relative-limit-price distribution decaying with an
  exponent near 1.5: most orders near the touch, a long tail placed far away.
* **Books are dominated by cancellation.** Well above 90% of orders in a modern
  equity book are cancelled rather than filled, and queue dynamics depend
  entirely on that churn. **This market reaches about 60%, not 90%**, measured
  across every agent, and adding these traders barely moves it (58.8% without,
  60.4% with). The missing 30 points are not this agent's to supply: real
  cancellation is dominated by market makers requoting on every tick at
  microsecond scale, and nothing here requotes faster than 300ms. Recorded as a
  gap rather than tuned away, because a cancel rate hit by inflating one
  agent's churn would be the number without the mechanism behind it.
* **Arrivals cluster.** Order flow comes in bursts. Self-exciting (Hawkes)
  models fit it well: each event raises the intensity of the next. Fitted
  branching ratios for equity order flow sit high, with kernels decaying over
  sub-second to second timescales: Filimonov & Sornette (2012) and Hardiman,
  Bercot & Bouchaud (2013) report values near 0.8-0.95, the latter arguing
  order flow is close to critical.

The warning
-----------

**This agent assumes what the rest of the project measures**, so any statistic
gathered with it running has to be read differently from one gathered without.
The market already reproduces fat-tailed returns, volatility clustering and
long-memory order flow *emergently*, from agents with none of these
distributions built in, and that is the stronger result.

The warning originally written here went further: that switching this population
on would inflate the measured tails, because a power-law order-size distribution
had been put in by hand. **Measured over three paired seeds, that was wrong, and
in the opposite direction:**

    statistic                     without flow    with flow
    Hill tail index                      1.86         2.06     lighter, not heavier
    excess kurtosis of returns          152.9        131.4     lower
    volatility clustering |r|            0.16         0.14     slightly lower
    bid-ask bounce (lag 1)              +0.13        -0.05     correct sign at last

The mechanism appears to be that the extra population deepens the book faster
than heavy order sizes can move it, and dilution wins. So the emergence result
survives this population rather than being undermined by it. And the bid-ask
bounce, which had the wrong sign entirely without these agents, becomes
negative as a real tape's does, because they both post and take.

Three seeds is not many, and the honest reading is "did not inflate the tails"
rather than "provably reduced them". The reason to keep this **off by default**
is therefore weaker than first thought, but still real: a number gathered with
an assumed size distribution present is a claim about a market that was told
what its order sizes are.

What it is genuinely for: queue dynamics under heavy cancellation, whether the
matching engine holds up under bursty load, and how a market maker fares against
order flow with a realistic size distribution.

Parameters are the literature's, cited above, not fitted to make this market's
output look like anything.
"""

from __future__ import annotations

import math

from arena.agents.base import TradingAgent
from arena.exchange.types import AgentId, OrderId, Price, Side, TimeInForce
from arena.market.instrument import Instrument
from arena.market.venue import SymbolCommand
from arena.exchange.events import Cancel
from arena.sim.kernel import SimulationContext
from arena.sim.time import Duration, millis

__all__ = ["FlowTrader", "power_law_size", "placement_offset"]


def power_law_size(rng, exponent: float, minimum: int, maximum: int) -> int:
    """A draw from a discrete Pareto tail, truncated at ``maximum``.

    ``exponent`` is the **survival** exponent: P(X > x) decays as x**-exponent.
    That is the convention the microstructure literature quotes its numbers in,
    so a value read from a paper can be passed here unchanged. Getting this
    wrong is silent (the draws still look heavy-tailed, just with the wrong
    tail), so the property test measures the realised exponent back out.

    Inverse-transform on the continuous Pareto, then floored. Truncation is not
    cosmetic: an untruncated power law with an exponent near 2 has infinite
    variance, so one draw could exceed every position limit in the market and
    the run would be about that order rather than about the market.
    """
    if exponent <= 0.0:
        raise ValueError("a power-law tail needs a positive exponent")
    u = rng.random()
    # Guard the singularity at u = 1 rather than letting it produce infinity.
    u = min(u, 1.0 - 1e-12)
    value = minimum * (1.0 - u) ** (-1.0 / exponent)
    return int(max(minimum, min(maximum, math.floor(value))))


def placement_offset(rng, exponent: float, maximum: int) -> int:
    """Ticks from the touch at which a limit order is placed.

    Same construction as the size draw, with the exponent Zovko & Farmer report.
    Most orders land at or near the touch; a long tail sits far behind it, and
    that tail is what gives a real book its shape away from the best price.
    """
    return power_law_size(rng, exponent, 1, maximum) - 1


class FlowTrader(TradingAgent):
    """Bursty, cancel-heavy, power-law-sized order flow.

    Carries no view on value at all. That is deliberate: this agent is a model
    of *flow*, and mixing a forecast into it would make it impossible to tell
    which of its effects came from the shape of the flow and which from the
    information in it.
    """

    def __init__(
        self,
        agent_id: AgentId,
        venue_id: AgentId,
        instruments: dict[str, Instrument],
        wake_interval: Duration = millis(500),
        size_exponent: float = 2.4,
        placement_exponent: float = 1.5,
        max_size: int = 400,
        max_offset: int = 60,
        market_order_fraction: float = 0.15,
        cancel_rate: float = 0.92,
        branching_ratio: float = 0.85,
        decay: Duration = millis(300),
        position_limit: int = 1_500,
    ) -> None:
        super().__init__(agent_id, venue_id, instruments, wake_interval)
        self.size_exponent = size_exponent
        self.placement_exponent = placement_exponent
        self.max_size = max_size
        self.max_offset = max_offset
        self.market_order_fraction = market_order_fraction
        # The share of resting orders this agent pulls per wakeup. Above 90% of
        # orders in a real book are cancelled rather than filled, and queue
        # behaviour depends on that churn far more than on the fills.
        self.cancel_rate = cancel_rate
        # Hawkes self-excitation. This is the *branching ratio*: the expected
        # number of further orders each order triggers. It must stay below one,
        # and that is not a matter of taste: at one or above the process is
        # explosive. A first attempt here added excitation in the wrong units
        # with no stability condition, and one agent reached 26,000 times its
        # baseline intensity and emitted 6,000 orders a second forever.
        if not 0.0 <= branching_ratio < 1.0:
            raise ValueError(
                f"branching ratio {branching_ratio} must be in [0, 1); at one or "
                "above the arrival process explodes rather than clusters"
            )
        self.branching_ratio = branching_ratio
        # How long a burst stays hot: the excitation decays by 1/e over this
        # interval. Measured against the clock rather than per wakeup, because
        # tying decay to wakeups would make burst length depend on how often
        # the agent happened to be scheduled, which is circular, and was
        # exactly how the explosive version failed to decay at all.
        self.decay = max(1, int(decay))
        self.position_limit = position_limit
        self._intensity = 0.0
        self._last_wake = 0
        self.submitted = 0
        self.cancelled = 0

    # -- clustered arrivals -------------------------------------------------

    def schedule_next(self, ctx: SimulationContext) -> None:
        """Wake sooner when recently active, and drift back when not.

        A proper Hawkes intensity, in events per nanosecond:

            lambda(t) = mu + sum_i alpha * exp(-beta * (t - t_i))

        with ``beta = 1/decay`` and ``alpha = branching_ratio * beta``, so the
        expected offspring per event is exactly the branching ratio. Below one
        this clusters; at one or above it explodes, which is why the constructor
        refuses to build it.

        The long-run mean rate is ``mu / (1 - branching_ratio)``: bursty, and
        bounded, which is the combination a Poisson clock cannot give.
        """
        elapsed = max(0, int(ctx.now) - self._last_wake)
        self._last_wake = int(ctx.now)
        self._intensity *= math.exp(-elapsed / self.decay)
        baseline = 1.0 / int(self.wake_interval)
        delay = ctx.rng.expovariate(1.0) / (baseline + self._intensity)
        ctx.request_wakeup(Duration(max(1, int(delay))))

    # -- behaviour ----------------------------------------------------------

    def act(self, ctx: SimulationContext) -> None:
        # Manage the whole book, not just the symbol about to be traded. A
        # real algorithm re-evaluates everything it has resting on every
        # pass; sweeping only one symbol left most orders unmanaged, which
        # showed up as a cancel rate half what a real book carries.
        self._churn(ctx)
        symbol = ctx.rng.choice(sorted(self.instruments))

        book = self.books[symbol]
        if book.bid is None or book.ask is None:
            return

        side = Side.BUY if ctx.rng.random() < 0.5 else Side.SELL
        inventory = self.position.get(symbol, 0)
        if (side is Side.BUY and inventory >= self.position_limit) or (
            side is Side.SELL and inventory <= -self.position_limit
        ):
            return

        size = power_law_size(ctx.rng, self.size_exponent, 1, self.max_size)
        self.submitted += 1
        # alpha = branching_ratio * beta, with beta = 1/decay.
        self._intensity += self.branching_ratio / self.decay

        if ctx.rng.random() < self.market_order_fraction:
            self.take(ctx, symbol, side, size)
            return

        # Placed behind the touch by a power-law distance, so the book acquires
        # depth away from the best price rather than only at it.
        offset = placement_offset(ctx.rng, self.placement_exponent, self.max_offset)
        anchor = int(book.bid) if side is Side.BUY else int(book.ask)
        price = anchor - offset if side is Side.BUY else anchor + offset
        self.quote(ctx, symbol, side, Price(price), size, TimeInForce.GTC)

    def _churn(self, ctx: SimulationContext) -> None:
        """Pull most of what is resting. This is what a real book mostly does."""
        for key, order_symbol in list(self.live_orders.items()):
            if ctx.rng.random() >= self.cancel_rate:
                continue
            ctx.send(
                self.venue_id,
                SymbolCommand(order_symbol, Cancel(self.agent_id, OrderId(key[1]))),
            )
            self.live_orders.pop(key, None)
            self.cancelled += 1
