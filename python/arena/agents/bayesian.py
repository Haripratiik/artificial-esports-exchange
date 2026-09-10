"""A fundamental trader whose information is measured in battles observed.

This replaces a free noise parameter with something that has a unit, and that
substitution is the load-bearing idea of the whole research programme.

The old model gave an agent ``precision=3.0`` and perturbed its view by a
Gaussian scaled to that. It worked, but "precision 3.0" means nothing: it
cannot be compared across contracts, it has no natural prior, and the question
"how much is better information worth?" has no denominator.

Here an agent is instead given **n_j battles**, drawn from the same
data-generating process the contract settles on:

    k_j ~ Binomial(n_j, p*)                 what this agent happened to see
    posterior = Beta(a0 + k_j, b0 + n_j - k_j)

with the prior (a0, b0) coming from the reference snapshot's own mode prior and
shrinkage strength: the same kappa and m already estimated from data in
:mod:`arena.worlds.circuit.metrics`, rather than a second set of constants
invented here.

Four things follow, none of which the noise knob could give:

  * Information has a unit. "One more battle observed" is a quantity, so
    "what is faster or better information worth?" gets an answer in dollars
    per battle.
  * The posterior is exact rather than assumed. No distributional hand-waving:
    Beta is conjugate to Binomial, so the agent's belief is the correct one
    given what it saw.
  * Error must fall as 1/sqrt(n). That is a hard, checkable property, and the
    test suite checks it. A noise knob has no such constraint, so nothing
    about it could ever be wrong.
  * It maps onto how evidence actually reaches a participant here. An agent's
    n_j is literally "how many of the season's matches has this fund seen",
    and the season is drawn from the world seed rather than collected, so the
    quantity is re-derivable by anyone holding that seed. There is no crawl
    waiting to replace it and none is owed.

Valuation is **E[payoff(theta)]** under the posterior, never
``payoff(E[theta])``. For a linear future the two coincide; for an option the
difference is the entire time value, and for a binary it is the difference
between a probability and a point prediction. Binaries and linear payoffs are
computed analytically; anything else falls back to sampling the posterior.
"""

from __future__ import annotations

import math

from arena.agents.base import TradingAgent
from arena.agents.fundamental import underlying_key
from arena.contracts.payoff import Binary, Linear
from arena.exchange.types import AgentId, Price, Side, TimeInForce
from arena.market.instrument import Instrument
from arena.sim.kernel import SimulationContext
from arena.sim.time import Duration, millis

__all__ = [
    "BayesianFundamental",
    "posterior_for",
    "binary_probability",
    "predictive_probability",
    "settlement_probability",
]


def posterior_for(
    level: float, battles: int, prior_mean: float, prior_strength: float, rng
) -> tuple[float, float]:
    """Draw one agent's evidence and return its Beta posterior ``(a, b)``.

    The draw happens once per agent per contract and is then held. An agent
    that re-drew its evidence each wakeup would be a noise trader wearing a
    posterior: its "information" would average away and it would exert no
    directional pull on price at all.
    """
    if battles < 0:
        raise ValueError("battles observed cannot be negative")
    a0 = max(1e-9, prior_strength * prior_mean)
    b0 = max(1e-9, prior_strength * (1.0 - prior_mean))
    if battles == 0:
        return a0, b0
    wins = rng.binomialvariate(battles, min(max(level, 0.0), 1.0))
    return a0 + wins, b0 + (battles - wins)


def binary_probability(a: float, b: float, threshold: float, comparison: str) -> float:
    """P(theta > threshold) and friends under Beta(a, b), analytically.

    Exact rather than sampled, because a binary's whole value *is* this tail
    probability. Estimating it by Monte Carlo would put sampling noise directly
    into the quantity the experiment measures.
    """
    from scipy.stats import beta as beta_dist

    if threshold <= 0.0:
        upper = 1.0
    elif threshold >= 1.0:
        upper = 0.0
    else:
        upper = float(beta_dist.sf(threshold, a, b))
    # The distribution is continuous, so strict and non-strict comparisons
    # coincide; the distinction survives only in the contract's wording.
    return upper if comparison in (">", ">=") else 1.0 - upper


def settlement_probability(
    level: float, window_battles: int, threshold: float
) -> float:
    """P(measured rate > threshold) when the window holds ``window_battles``.

    The ground truth for a binary contract, and the reason there *is* a truth
    to be right about. These contracts do not settle on a platonic win rate;
    they settle on a rate measured over an observation window of finite size,
    so even an agent that knew the true rate exactly would still face genuine
    uncertainty about where the measurement lands.

    That is what makes a probability the correct answer rather than a hedge,
    and it is the quantity every forecast in the experiment is scored against.
    """
    from scipy.stats import binom

    if window_battles <= 0:
        raise ValueError("the observation window must contain battles")
    return float(binom.sf(math.floor(threshold * window_battles), window_battles, level))


def predictive_probability(
    a: float, b: float, window_battles: int, threshold: float
) -> float:
    """The same probability, under a Beta(a, b) posterior over the true rate.

    Integrating the Binomial tail over the posterior gives a Beta-Binomial
    tail, which is the *Bayes-optimal* forecast for an agent holding this
    posterior, not an approximation of it.

    This matters for the experiment's central claim. If agents forecast the
    posterior tail over the true rate instead, they are systematically
    overconfident, and a market that beat the aggregate of them would only have
    beaten a population of sloppy forecasters. Scoring the market against
    individually optimal agents is what makes the comparison worth making.
    """
    from scipy.stats import betabinom

    if window_battles <= 0:
        raise ValueError("the observation window must contain battles")
    return float(
        betabinom.sf(math.floor(threshold * window_battles), window_battles, a, b)
    )


class BayesianFundamental(TradingAgent):
    """Trades the gap between price and an exact posterior over settlement."""

    def __init__(
        self,
        agent_id: AgentId,
        venue_id: AgentId,
        instruments: dict[str, Instrument],
        truth_level: dict[str, float],
        battles: int,
        prior_mean: float = 0.5,
        prior_strength: float = 50.0,
        window_battles: int | None = None,
        wake_interval: Duration = millis(900),
        max_position: int = 400,
        base_size: int = 10,
        patience: float = 0.5,
        draws: int = 256,
    ) -> None:
        super().__init__(agent_id, venue_id, instruments, wake_interval)
        self.truth_level = truth_level
        self.battles = battles
        self.prior_mean = prior_mean
        self.prior_strength = prior_strength
        # How many battles the settlement window itself will contain. Set, the
        # agent knows the contract settles on a *measurement* and prices the
        # Beta-Binomial tail, which is Bayes-optimal. Left None, it prices the
        # posterior tail over the true rate: the infinite-window limit.
        self.window_battles = window_battles
        self.max_position = max_position
        self.base_size = base_size
        self.patience = patience
        self.draws = max(16, draws)
        # Keyed by what the contract is written on, not by symbol. An agent
        # holding a different posterior per contract on one competitor does
        # not have a posterior for that competitor, and the ladder it quotes
        # off those is not the ladder of any distribution.
        self._posterior: dict[str, tuple[float, float]] = {}
        self._levels: dict[str, list[float]] = {}
        self._value: dict[str, float] = {}
        self._dispersion: dict[str, float] = {}

    def on_start(self, ctx: SimulationContext) -> None:
        """Draw the evidence before the market opens, not when someone asks.

        The posterior was drawn lazily, on first use. That is fine while an
        agent always trades, and wrong the moment one does not: the draw then
        happens whenever the harness reads the forecast instead, at a different
        point in the agent's random stream. Two runs that differ only in the
        venue (which is exactly Experiment 2's control) came back with
        different *beliefs*, and the comparison stopped being a comparison.

        Evidence is something an agent has, not something it produces on
        demand.
        """
        super().on_start(ctx)
        for symbol in sorted(self.instruments):
            self.posterior(ctx, symbol)

    # -- belief ------------------------------------------------------------

    def posterior(self, ctx: SimulationContext, symbol: str) -> tuple[float, float] | None:
        """This agent's Beta posterior over the metric, drawn once and held.

        Once per *underlying*: the battles it observed are matches involving a
        competitor, not matches involving a contract, so every contract written
        on that competitor is priced off the same sample.
        """
        key = underlying_key(self.instruments[symbol])
        if key not in self._posterior:
            level = self.truth_level.get(symbol)
            if level is None:
                return None
            self._posterior[key] = posterior_for(
                level, self.battles, self.prior_mean, self.prior_strength, ctx.rng
            )
        return self._posterior[key]

    def levels(self, ctx: SimulationContext, symbol: str, a: float, b: float) -> list[float]:
        """A sample from the posterior, shared across every contract on it.

        Common random numbers, and the reason is not efficiency. Drawing fresh
        each time gives each strike its own Monte Carlo error, so the agent's
        own option ladder is neither monotone nor convex, and then it trades on
        the difference. With one sample path per underlying, ``max(F - K, 0)``
        is decreasing in ``K`` draw by draw, so the average is too, and the
        agent's surface is a real surface.
        """
        key = underlying_key(self.instruments[symbol])
        drawn = self._levels.get(key)
        if drawn is None:
            drawn = [ctx.rng.betavariate(a, b) for _ in range(self.draws)]
            self._levels[key] = drawn
        return drawn

    def posterior_mean(self, ctx: SimulationContext, symbol: str) -> float | None:
        posterior = self.posterior(ctx, symbol)
        if posterior is None:
            return None
        a, b = posterior
        return a / (a + b)

    def forecast(self, ctx: SimulationContext, symbol: str) -> float | None:
        """The agent's probability, for a binary contract. Its stated belief.

        This is what the experiment harness collects as the agent's individual
        forecast, so the market's aggregate can be scored against it.
        """
        posterior = self.posterior(ctx, symbol)
        if posterior is None:
            return None
        payoff = self.instruments[symbol].spec.payoff
        if not isinstance(payoff, Binary):
            return None
        return self._binary_view(*posterior, payoff)

    def _binary_view(self, a: float, b: float, payoff: Binary) -> float:
        """P(the contract settles in the money) under this agent's posterior."""
        if self.window_battles is None:
            return binary_probability(a, b, payoff.threshold, payoff.comparison)
        above = predictive_probability(a, b, self.window_battles, payoff.threshold)
        return above if payoff.comparison in (">", ">=") else 1.0 - above

    def _view(self, ctx: SimulationContext, symbol: str) -> tuple[float, float] | None:
        """Value and dispersion in ticks: E[payoff] and sd(payoff)."""
        if symbol in self._value:
            return self._value[symbol], self._dispersion[symbol]

        posterior = self.posterior(ctx, symbol)
        if posterior is None:
            return None
        a, b = posterior
        instrument = self.instruments[symbol]
        payoff = instrument.spec.payoff
        tick = float(instrument.tick_size)

        pays_as_it_goes = instrument.spec.distribution is not None

        if isinstance(payoff, Binary) and not pays_as_it_goes:
            probability = self._binary_view(a, b, payoff)
            value = probability * payoff.payout
            # A Bernoulli payout: variance is p(1-p) times the payout squared.
            spread = math.sqrt(max(0.0, probability * (1.0 - probability))) * abs(payoff.payout)
        elif isinstance(payoff, Linear) and not pays_as_it_goes:
            mean = a / (a + b)
            variance = a * b / ((a + b) ** 2 * (a + b + 1.0))
            value = payoff.apply(mean)
            spread = abs(payoff.scale) * math.sqrt(variance)
        else:
            # Kinked, otherwise non-linear, or paying as it goes: sample the
            # posterior. E[payoff] and payoff(E) differ for a kinked payoff,
            # and the difference is the option's time value. A share takes this
            # path because what it is worth is the stream plus the end, which
            # is what claim_value adds up; valuing only the payoff would price
            # a pure strip at nothing.
            claim = instrument.spec.claim_value
            samples = [claim(level) for level in self.levels(ctx, symbol, a, b)]
            value = sum(samples) / len(samples)
            spread = math.sqrt(
                sum((s - value) ** 2 for s in samples) / len(samples)
            )

        self._value[symbol] = value / tick
        self._dispersion[symbol] = max(1.0, spread / tick)
        return self._value[symbol], self._dispersion[symbol]

    # -- trading -----------------------------------------------------------

    def act(self, ctx: SimulationContext) -> None:
        for symbol in sorted(self.instruments):
            self._trade(ctx, symbol)

    def _trade(self, ctx: SimulationContext, symbol: str) -> None:
        book = self.books[symbol]
        if book.mid is None:
            return
        view = self._view(ctx, symbol)
        if view is None:
            return
        estimate, uncertainty = view

        edge = estimate - book.mid
        # The threshold scales with the agent's own uncertainty, so a vague
        # agent needs a bigger discrepancy before it acts. This is what stops a
        # poorly-informed agent from dominating the book by being wrong loudly.
        if abs(edge) < uncertainty * self.patience:
            self.cancel_all(ctx, symbol)
            return

        side = Side.BUY if edge > 0 else Side.SELL
        inventory = self.position.get(symbol, 0)
        if (side is Side.BUY and inventory >= self.max_position) or (
            side is Side.SELL and inventory <= -self.max_position
        ):
            return

        conviction = min(4.0, abs(edge) / max(1e-9, uncertainty))
        size = max(1, int(self.base_size * conviction))
        size = max(1, min(size, self.max_position - abs(inventory)))

        self.cancel_all(ctx, symbol)

        # Cross only when the touch is already through the estimate; otherwise
        # post and wait. Always crossing would pay the spread away on every
        # trade and turn a correct view into a losing strategy.
        if side is Side.BUY and book.ask is not None and int(book.ask) < estimate:
            self.take(ctx, symbol, side, size)
        elif side is Side.SELL and book.bid is not None and int(book.bid) > estimate:
            self.take(ctx, symbol, side, size)
        else:
            anchor = book.bid if side is Side.BUY else book.ask
            if anchor is not None:
                self.quote(ctx, symbol, side, Price(int(anchor)), size, TimeInForce.GTC)
