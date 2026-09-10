"""A fundamental trader: forms a view of settlement, trades the gap.

This is the agent that makes prices mean something. A market of makers and
noise traders has liquidity and volatility but no anchor: its price is whatever
the last flow happened to push it to. A fundamental agent supplies the force
that drags price toward the value the contract will actually settle at, which
is what turns "does the market aggregate information?" into a question with an
answer.

**Its information is deliberately imperfect and deliberately parameterised.**
The agent is told the true settlement value, then given a *noisy* view of it:

    estimate = truth + noise,   noise scaled by `precision`

That is the knob every information experiment turns. A population of agents
with different precisions is a population with heterogeneous information, and
the market's job is to aggregate them. Later phases replace the noise term with
something better motivated (an agent that has observed ``n`` battles has a
posterior whose width follows from ``n`` rather than from a free parameter),
but the interface is the same, and the trading logic does not change.

It trades on **edge relative to its own uncertainty**, not on raw distance from
price. An agent with a vague view should not bet heavily on a small gap, and one
with a sharp view should. Sizing by conviction is what stops a noisy agent from
dominating the book simply by being wrong loudly.
"""

from __future__ import annotations

from arena.agents.base import TradingAgent
from arena.exchange.types import AgentId, Price, Side, TimeInForce
from arena.market.instrument import Instrument
from arena.sim.kernel import SimulationContext
from arena.sim.time import Duration, millis

__all__ = ["FundamentalTrader"]


def underlying_key(instrument) -> str:
    """What a contract is written on, ignoring how it pays.

    Every contract on the same thing has to share one belief about that thing.
    Keying a view by symbol instead gives an agent three different opinions
    about one competitor's win rate at once, one per strike, and the
    surface it then quotes is not the surface of any distribution at all. It
    was measured: `fund-vague` valued one call at 119 while valuing the strike
    fifty ticks below it, a strictly more valuable contract, at 37. That is not
    a small error, and it is not an error of judgement; it is three independent
    draws being treated as one opinion.
    """
    from arena.determinism import canonical_json

    return canonical_json(instrument.spec.underlying.to_dict())


class FundamentalTrader(TradingAgent):
    """Trades the gap between price and its own estimate of settlement."""

    def __init__(
        self,
        agent_id: AgentId,
        venue_id: AgentId,
        instruments: dict[str, Instrument],
        truth_level: dict[str, float],
        wake_interval: Duration = millis(1_500),
        precision: float = 1.0,
        metric_sigma: float = 0.02,
        draws: int = 128,
        max_position: int = 150,
        base_size: int = 8,
        patience: float = 0.5,
        open_interest: bool = False,
        reveal_over: Duration | None = None,
        prior_level: dict[str, float] | None = None,
    ) -> None:
        super().__init__(agent_id, venue_id, instruments, wake_interval)
        self.truth_level = truth_level
        self.precision = max(0.01, precision)
        self.metric_sigma = metric_sigma
        self.draws = max(8, draws)
        self.max_position = max_position
        self.base_size = base_size
        self.patience = patience
        self._estimate: dict[str, float] = {}
        self._noise_scale: dict[str, float] = {}
        # One view per underlying, shared by every contract written on it.
        self._views: dict[str, tuple[float, list[float]]] = {}
        # Over what span this agent's evidence arrives. ``None`` means all of
        # it at once, which is what every published experiment here was run
        # with and therefore stays the default.
        #
        # The difference is not a detail of pacing. With everything known at
        # t=0 the market has an information *stock*: the price converges in
        # seconds and then nothing can move it, because there is nothing left
        # to arrive. Measured, the realised dispersion of a win rate future
        # over a ten-minute session was 14.6 on a price near 4,670, so options
        # carried almost no time value, binaries were foregone conclusions
        # within a minute, and a market whose whole subject is disagreement had
        # nothing to disagree about. A flow of evidence is what makes the
        # underlying diffuse.
        self.reveal_over = reveal_over
        # What the metric did over the period *before* the contract's window,
        # per symbol. This is where an agent's belief starts.
        #
        # Starting from the truth and adding noise, which is what this did,
        # makes every agent unbiased from the first instant, so the market
        # opens at the answer with a wide spread and merely tightens. Real
        # markets open at what history said and then discover that history was
        # wrong, which on a real fixture is a real event. Measured on the
        # retired statistics world, before the circuit: one subject ran at
        # 0.4839 for the twelve weeks before the window and settled at 0.4669.
        # An agent that begins at 0.4839 and is pulled toward 0.4669 by
        # evidence is doing what an analyst does; one that begins at 0.4669 has
        # nothing to learn.
        #
        # It is lookahead-free by construction: the prior window ends where the
        # contract's window begins, so it is data that exists on the day the
        # contract is published.
        self.prior_level = prior_level or {}
        # Accumulated Brownian information per underlying: (w, tau, normals).
        self._learning: dict[str, tuple[float, float, list[float]]] = {}
        # Whether to post interest into a book that has no price yet.
        self.open_interest = open_interest

    def _draws(self, ctx, instrument, level: float, sigma: float):
        """This agent's view of one underlying: a centre, and a sample around it.

        Cached by what the contract is written on, so every contract on the
        same thing is valued from the same numbers.
        """
        key = underlying_key(instrument)
        if self.reveal_over is None:
            cached = self._views.get(key)
            if cached is None:
                centre = level + ctx.rng.gauss(0.0, sigma)
                cached = (
                    centre,
                    [centre + ctx.rng.gauss(0.0, sigma) for _ in range(self.draws)],
                )
                self._views[key] = cached
            return cached
        # Falls back to the truth when no prior is supplied, which keeps the
        # agent usable in a world that has no history to look back at.
        prior = self.prior_level.get(instrument.symbol, level)
        return self._learned(ctx, key, level, sigma, prior)

    def _learned(self, ctx, key: str, level: float, sigma: float, prior: float):
        """A belief that starts at the prior and is pulled toward the truth.

        Ordinary Gaussian updating, written in precision because that is the
        unit evidence actually arrives in. With a prior ``N(mu, 1/t0)`` and
        evidence of accumulated precision ``te`` centred on the truth, the
        posterior mean is

            m = (t0 * mu + te * truth + W(te)) / (t0 + te)

        for a standard Brownian motion ``W`` in accumulated precision, and the
        posterior variance is exactly ``1 / (t0 + te)``. At ``te = 0`` that is
        the prior, unchanged and unperturbed; as evidence accumulates it slides
        to the truth.

        The Brownian term is what makes it a *martingale*: the agent's view at
        any moment is its own best guess at its later view, so it never drifts
        predictably and cannot be anticipated by anything except better
        information. Redrawing the whole view each wakeup would have been far
        simpler and would have produced a noise trader wearing a posterior: its
        opinion would average away and exert no directional pull on price. The
        increments are independent; the *belief* accumulates.
        """
        prior_precision = 1.0 / max(1e-12, self.metric_sigma * self.metric_sigma)
        evidence_precision = 1.0 / max(1e-12, sigma * sigma)
        fraction = min(1.0, max(0.0, int(ctx.now) / max(1, int(self.reveal_over))))
        target = evidence_precision * fraction

        walk, seen, normals = self._learning.get(key, (0.0, 0.0, []))
        if not normals:
            normals = [ctx.rng.gauss(0.0, 1.0) for _ in range(self.draws)]
        if target > seen:
            walk += ctx.rng.gauss(0.0, (target - seen) ** 0.5)
            seen = target
        self._learning[key] = (walk, seen, normals)

        total = prior_precision + seen
        centre = (prior_precision * prior + seen * level + walk) / total
        spread = (1.0 / total) ** 0.5
        return centre, [centre + z * spread for z in normals]

    def dispersion_for(self, ctx, instrument) -> float:
        """This agent's current uncertainty about one underlying, in metric units."""
        sigma = self.metric_sigma / self.precision
        if self.reveal_over is None:
            return sigma
        _walk, seen, _normals = self._learning.get(
            underlying_key(instrument), (0.0, 0.0, [])
        )
        prior_precision = 1.0 / max(1e-12, self.metric_sigma * self.metric_sigma)
        return (1.0 / (prior_precision + seen)) ** 0.5

    def _view(self, ctx: SimulationContext, symbol: str) -> tuple[float, float]:
        """This agent's estimate of settlement, and its own uncertainty.

        **The noise is on the metric, not on the settlement value.** A
        fundamental analyst forms a view on a competitor's win rate; what that
        implies for a future, an option, or an event contract then follows from
        the contract's own terms. Perturbing the settlement value directly
        would be modelling an analyst who somehow has an opinion about an
        option premium without having one about the underlying.

        The difference is not cosmetic. Scaling uncertainty to a *contract's*
        range gives an option (whose value is a small fraction of its range) a
        noise term larger than the entire quantity being estimated, so the
        agent's view is dominated by noise and the option collapses to its
        floor. Perturbing the metric instead makes uncertainty propagate
        through the payoff, which also gives the agent the right sensitivity
        for free: it reacts to a rate change in proportion to the contract's
        delta.

        Drawn once per *underlying* and then held, and both halves of that
        matter. Once, because an agent redrawing its view every wakeup would be
        a noise trader with extra steps. Its "information" would average to
        nothing and exert no directional pull on price. Per underlying, because
        an agent with a different view of a competitor for every contract
        written on that competitor does not have a view of the competitor.
        """
        if self.reveal_over is not None:
            # The estimate is a function of evidence that is still arriving, so
            # the cached value is only good for as long as nothing has arrived.
            # Leaving it in place was the first version and made the whole
            # mechanism inert: the agent learned continuously and quoted its
            # first opinion forever.
            self._estimate.pop(symbol, None)
        if symbol not in self._estimate:
            instrument = self.instruments[symbol]
            level = self.truth_level.get(symbol)
            if level is None:
                self._estimate[symbol] = float("nan")
                self._noise_scale[symbol] = 1.0
                return self._estimate[symbol], self._noise_scale[symbol]

            # Uncertainty in metric units, percentage points of win rate. A
            # sharper agent has a tighter posterior about the same quantity.
            sigma = self.metric_sigma / self.precision
            payoff = instrument.spec.payoff
            centre, draws = self._draws(ctx, instrument, level, sigma)

            # **E[payoff(level)], not payoff(E[level]).** For a linear future
            # the two coincide, so the distinction is invisible until an option
            # appears, and then it is the whole of the option's time value. A
            # put struck just above where the metric will land is worth
            # something precisely because the metric might land lower; a point
            # estimate says it is worth its intrinsic value and nothing more,
            # which prices every out-of-the-money option at zero.
            #
            # Averaged over draws rather than integrated, because that works
            # for a kinked payoff, a step payoff and a linear one without any
            # of them being special-cased, and because a closed form would need
            # a volatility model this agent has no business owning.
            #
            # The *same* draws for every contract on this underlying. Fresh
            # draws per contract is the ordinary way to write this and it is
            # wrong here: the Monte Carlo error is then independent across
            # strikes, so the agent's own option ladder is neither monotone nor
            # convex and it trades on the difference. Common random numbers
            # make every valuation a function of one sample path, and
            # monotonicity in strike then holds draw by draw rather than on
            # average. The whole claim, not only the settlement: a contract
            # that pays as it goes is worth the stream as well as the end, and
            # an agent valuing only the end would price a share at whatever is
            # left after the last payment. Nothing, for a pure strip.
            values = [instrument.spec.claim_value(d) for d in draws]
            value = sum(values) / len(values)

            # Dispersion of the payoff under the agent's own uncertainty. This
            # is what conviction is measured against, so a contract whose value
            # is insensitive to the metric produces small trades even when the
            # price looks far away.
            variance = sum((v - value) ** 2 for v in values) / len(values)
            tick = float(instrument.tick_size)
            self._estimate[symbol] = value / tick
            # Never zero: a binary far from its threshold has no dispersion at
            # all, and an agent with zero uncertainty would trade infinitely
            # hard on any deviation.
            self._noise_scale[symbol] = max(1.0, (variance**0.5) / tick)
        return self._estimate[symbol], self._noise_scale[symbol]

    def act(self, ctx: SimulationContext) -> None:
        for symbol in sorted(self.instruments):
            self._trade(ctx, symbol)

    def _open_interest(
        self, ctx: SimulationContext, symbol: str, estimate: float, uncertainty: float
    ) -> None:
        """Post two-sided interest at this agent's own valuation.

        Only reached when the book has no mid, which on a venue with an opening
        call is the whole of the pre-open. Every agent here reacts to a price,
        so with nobody posting first the auction cleared nothing, the market
        opened empty, and it stayed empty for ten minutes; a market where every
        participant is waiting for a price is not a market.

        A trader with a view and no price to react to posts the view, which is
        what an opening auction is for: it collects the prices people are
        willing to deal at instead of asking who arrives first. The width is
        the agent's own uncertainty, so a vague agent brings wide interest and
        a sharp one brings tight interest, and the auction weighs them by size
        exactly as it should.
        """
        width = max(1.0, uncertainty * self.patience)
        instrument = self.instruments[symbol]
        low, high = instrument.tick_bounds
        bid = Price(int(max(int(low), min(int(high), estimate - width))))
        ask = Price(int(max(int(low), min(int(high), estimate + width))))
        if int(ask) <= int(bid):
            return
        inventory = self.position.get(symbol, 0)
        size = max(1, self.base_size // 2)
        if inventory < self.max_position:
            self.post(ctx, symbol, Side.BUY, bid, size)
        else:
            self.withdraw(ctx, symbol, Side.BUY)
        if -inventory < self.max_position:
            self.post(ctx, symbol, Side.SELL, ask, size)
        else:
            self.withdraw(ctx, symbol, Side.SELL)

    def _trade(self, ctx: SimulationContext, symbol: str) -> None:
        book = self.books[symbol]
        estimate, uncertainty = self._view(ctx, symbol)
        if estimate != estimate:  # NaN: no view on this symbol
            return
        if book.mid is None:
            if self.open_interest:
                self._open_interest(ctx, symbol, estimate, uncertainty)
            return

        edge = estimate - book.mid
        # Trade only when the gap is large relative to what this agent actually
        # knows. The threshold scales with uncertainty, so a vague agent needs a
        # bigger discrepancy before it acts.
        if abs(edge) < uncertainty * self.patience:
            self.cancel_all(ctx, symbol)
            return

        side = Side.BUY if edge > 0 else Side.SELL
        inventory = self.position.get(symbol, 0)
        if (side is Side.BUY and inventory >= self.max_position) or (
            side is Side.SELL and inventory <= -self.max_position
        ):
            return

        # Size by conviction: how many uncertainties away the price is.
        conviction = min(4.0, abs(edge) / max(1e-9, uncertainty))
        size = max(1, int(self.base_size * conviction))
        headroom = self.max_position - abs(inventory)
        size = max(1, min(size, headroom))

        # Whatever it does next, it is no longer interested in the other side.
        self.withdraw(ctx, symbol, side.opposite)

        # Cross when the touch is already inside the estimate; otherwise post at
        # the touch and wait. Always crossing would pay the spread away on every
        # trade and turn a correct view into a losing strategy.
        if side is Side.BUY and book.ask is not None and int(book.ask) < estimate:
            self.withdraw(ctx, symbol, side)
            self.take(ctx, symbol, side, size)
        elif side is Side.SELL and book.bid is not None and int(book.bid) > estimate:
            self.withdraw(ctx, symbol, side)
            self.take(ctx, symbol, side, size)
        else:
            anchor = book.bid if side is Side.BUY else book.ask
            if anchor is None:
                return
            self.post(ctx, symbol, side, Price(int(anchor)), size)
