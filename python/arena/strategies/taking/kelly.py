"""Sizing a belief when the stake and the collateral are the same number.

Kelly (1956) is about a gambler who stakes a fraction ``f`` of a bankroll on a
bet that either pays or does not. It reaches portfolio choice by analogy, and
the analogy is where it usually goes wrong: a position is not a stake, leverage
exists, and ``f`` has to be reinterpreted as an exposure whose loss is only
probable. None of that reinterpretation is needed here.
``Instrument.collateral_for(q, price)`` is the exact worst case of a position,
it is posted in full, there is no borrowing and no margin call, so the
collateral a trade requires *is* the amount that can be lost on it. Kelly's
``f`` and the exchange's collateral requirement are literally the same number,
which makes this the original gambling setup rather than an analogy, and is not
true on any venue that lets a position lose more than it posted.

So for a binary at price ``p`` under a belief ``pi`` the closed form applies
directly: stake ``(pi - p) / (1 - p)`` of the bankroll to buy, ``(p - pi) / p``
to sell, both in [0, 1], and both meaning "this fraction of what I have is at
risk" in the same sense the exchange means it.

**Collateral equals plausible risk only for binaries, and the difference is what
this strategy is really about.** Measured against the informed agents' own
posterior uncertainty, the collateral charge over-states the loss that is
actually in play by roughly 47x on a future, 130x on a short option, 180x on a
short commodity and 70x on the spread contract. The consequence is not a
detail: full Kelly on a future asks for about 1,290% of bankroll, so on this
venue the capital constraint binds roughly 13 times for every once the sizing
preference does. A strategy that computed ``f`` and stopped would be computing a
number the exchange refuses about thirteen trades in fourteen.

The honest object is therefore the *constrained* log-optimal program,

    maximise  E[ log( W + sum_i q_i (V_i - p_i) ) ]
    subject to  sum_i q_i c_i  <=  W

with ``c_i`` the per-lot collateral, and its Lagrange multiplier is the piece
worth exposing. At the optimum every traded contract satisfies
``E[(V_i - p_i) / W_after] = lambda c_i``, so ``lambda`` is the marginal
expected log wealth bought by one more unit of collateral, and a trade is
refused for capital exactly when its own density falls below it.
:attr:`KellyBayesian.shadow_price` is that number and
:attr:`KellyBayesian.sizing` is the per-trade record behind it, so a caller can
separate the trades this strategy declined on conviction from the ones the
balance sheet declined for it.

**Fractional Kelly is deliberately not implemented, and the reason is that the
argument for it does not hold here.** Halving ``f`` is a tail-risk and
drawdown control: ``worst_case()`` on this venue is the exact minimum of a
piecewise-linear function rather than a quantile, so there is no tail beyond it
to be shrunk away from; there is no leverage and no margin call, so no forced
liquidation; and Thorp's drawdown result is about continuous rebalancing while
positions here are held to settlement. Choosing a fraction would be choosing a
number.

What does transfer is shrinkage for *estimation* error, which is a different
problem with a principled answer. The belief is a Beta posterior from ``n``
observed battles, ``Beta(a, b)`` has exact variance ``ab / ((a+b)^2 (a+b+1))``,
so this strategy values the contract at a credible bound on the underlying rate
rather than at its posterior mean: the level distribution is displaced by
``z * sd`` in whichever direction is adverse to the position, and the contract
is revalued there. For a linear payoff that is exactly ``mean - z*sd`` when
buying. For a binary it is the tail past a threshold moved ``z*sd`` against the
position, which is the same statement about the same parameter. The displacement
falls as ``1 / sqrt(n)``, so this reduces to full Kelly as evidence accumulates,
which is the property that makes it a posterior rather than a chosen fraction.

Valuation is ``E[payoff(theta)]`` and never ``payoff(E[theta])``, reusing
:mod:`arena.agents.bayesian` rather than restating it: binaries analytically
through ``binary_probability`` or the Beta-Binomial ``predictive_probability``,
linear claims through their exact mean, and everything kinked through a
quadrature on the posterior held per underlying, so the strategy's own option
ladder is monotone in strike exactly rather than in expectation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from arena.agents.bayesian import (
    binary_probability,
    posterior_for,
    predictive_probability,
)
from arena.agents.fundamental import underlying_key
from arena.contracts.payoff import Binary, Linear
from arena.exchange.types import Side
from arena.strategies.base import MarketView, SymbolView, Take, snap

__all__ = [
    "KellyBayesian",
    "Sized",
    "binary_kelly",
    "log_optimal_units",
    "log_wealth_slope",
    "tradeable_touch",
]


def tradeable_touch(view_of: SymbolView) -> tuple[Decimal, Decimal] | None:
    """The two-sided touch, or ``None`` where there is not one to trade against.

    Stricter than "both sides are present", and the test is the contract's own
    range, which is public and exact. A bid below the least the claim can pay,
    an offer above the most it can pay, or a book quoted the wrong way round are
    each things no quote of this contract can be.

    It was written against a real leak and the leak has since been closed
    upstream, which is worth saying plainly rather than leaving the comment to
    imply otherwise. During a call phase a market order rests at a sentinel so
    that it crosses every candidate the auction considers, and the top-of-book
    feed was publishing it: 78,742 touches outside the settlement range across
    all 47 symbols in twenty seconds on seed 7, each exactly
    4,611,686,018,427,387,904 ticks. Read as a touch it gives a spread of minus
    2.3e18, so every cost hurdle computed from it is negative and every contract
    looks like a free lunch; before this guard the sibling arbitrage strategy
    fired ten packages in the first two minutes against prices no contract could
    have quoted. ``top_of_book`` now filters the sentinel the way the depth feed
    always did.

    The guard stays because it is cheap and because the failure it prevents is
    silent and expensive in one direction only. A crossed book is legitimate on
    this venue for reasons that have nothing to do with sentinels, and a hurdle
    built from a negative spread does not make a strategy cautious, it makes it
    reckless.
    """
    bid, ask = view_of.best_bid, view_of.best_ask
    if bid is None or ask is None or bid >= ask:
        return None
    low, high = view_of.bounds
    if bid < low or ask > high:
        return None
    return bid, ask


def binary_kelly(price: float, belief: float) -> float:
    """The Kelly stake on a unit binary, signed: positive buys, negative sells.

    The magnitude is the fraction of bankroll staked, which on this venue is the
    collateral the trade posts, so it is directly the ``f`` of the 1956 paper
    and not a rescaling of it. Buying at ``p`` risks ``p`` to win ``1 - p``,
    which is odds ``b = (1 - p) / p`` at probability ``pi``, and Kelly's
    ``(pi b - (1 - pi)) / b`` collapses to ``(pi - p) / (1 - p)``. Selling is
    the same bet read the other way round and gives ``(p - pi) / p``.

    Zero at ``pi == p`` and at a price the contract cannot move away from: a
    binary offered at the bottom of its range costs no collateral, so the
    fraction of bankroll it stakes is not defined and its size is a question
    about depth rather than about belief.
    """
    if not 0.0 < price < 1.0:
        return 0.0
    if belief > price:
        return (belief - price) / (1.0 - price)
    if belief < price:
        return -(price - belief) / price
    return 0.0


def log_wealth_slope(
    wealth: float, outcomes: Sequence[tuple[float, float]], units: float
) -> float:
    """``d/dq E[log(W + q x)]`` at ``q = units``, for profits ``x`` per lot.

    ``outcomes`` is ``(profit per lot, weight)`` already signed for the side
    being taken, so a sell is not a separate expression. Strictly decreasing in
    ``units`` wherever the wealth inside the logarithm stays positive, which is
    what makes the objective concave and a bisection on this the whole solver.
    """
    total = 0.0
    for profit, weight in outcomes:
        after = wealth + units * profit
        if after <= 0.0:
            return -math.inf
        total += weight * profit / after
    return total


def log_optimal_units(
    wealth: float,
    outcomes: Sequence[tuple[float, float]],
    collateral_per_lot: float,
    cap_units: float,
) -> float:
    """Lots that maximise expected log wealth, capped by what capital allows.

    Bisection rather than a formula because the objective is a formula only for
    a two-outcome bet. Everything else here settles on a continuum: a future
    pays ``10,000 theta``, an option pays a kinked function of it, and the
    stationary condition ``E[(V - p) / (W + q(V - p))] = 0`` has no closed form
    in ``q``. It has something better, which is that the left side is monotone
    decreasing, so the root is bracketed by construction and eighty halvings
    reach it to a part in ``2^80`` of the bracket.

    The upper bracket is the ruin boundary ``W / c`` and not the caller's cap,
    stepped one part in a billion inside it, because the logarithm diverges
    exactly there. Kelly's own answer never reaches it, so the step is a guard
    on the arithmetic rather than a constraint on the result.
    """
    if wealth <= 0.0 or collateral_per_lot <= 0.0:
        return 0.0
    ceiling = min(float(cap_units), wealth / collateral_per_lot)
    high = ceiling * (1.0 - 1e-9)
    if high <= 0.0:
        return 0.0
    if log_wealth_slope(wealth, outcomes, 0.0) <= 0.0:
        return 0.0
    if log_wealth_slope(wealth, outcomes, high) >= 0.0:
        return high
    low = 0.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if log_wealth_slope(wealth, outcomes, middle) > 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


@dataclass(frozen=True, slots=True)
class Sized:
    """What one contract asked for and what the balance sheet let it have.

    Kept per wakeup rather than aggregated, because the question this strategy
    exists to answer is which trades were refused for capital and which for
    conviction, and an aggregate cannot tell those apart.
    """

    symbol: str
    side: Side
    # Expected profit per lot against the touch, before the round-trip hurdle.
    edge: float
    collateral_per_lot: float
    # The unconstrained log-optimal size, and what was actually asked for.
    wanted: int
    funded: int
    # Marginal expected log wealth per unit of collateral, at zero. The
    # ordering key, and the quantity the shadow price is measured in.
    density: float

    @property
    def capital_bound(self) -> bool:
        """True when the balance sheet, not the belief, decided the size."""
        return self.funded < self.wanted


class KellyBayesian:
    """A Beta posterior, valued through each contract's payoff, sized by Kelly.

    Evidence is measured in battles observed, exactly as
    :class:`~arena.agents.bayesian.BayesianFundamental` measures it, and drawn
    through the same :func:`~arena.agents.bayesian.posterior_for` so that "how
    much is one more battle worth" has the same denominator on both sides of
    the comparison. A posterior may also be handed in directly, which is what a
    test with a hand-computed Beta wants and what a desk with real counts would
    do.
    """

    def __init__(
        self,
        levels: Mapping[str, float],
        battles: int,
        *,
        prior_mean: float = 0.5,
        prior_strength: float = 50.0,
        window_battles: int | None = None,
        posterior: Mapping[str, tuple[float, float]] | None = None,
        credible_z: float = 1.0,
        cost_multiple: float = 1.0,
        fee_bps: float = 0.0,
        capital_fraction: float = 1.0,
        max_position: int | None = None,
        draws: int = 256,
    ) -> None:
        self.levels = dict(levels)
        self.battles = battles
        self.prior_mean = prior_mean
        self.prior_strength = prior_strength
        # Set, the contract is known to settle on a *measurement* over a window
        # of this many battles and the belief is the Beta-Binomial predictive
        # tail, which is Bayes-optimal for it. Left None, it is the posterior
        # tail over the true rate, which is the infinite-window limit.
        self.window_battles = window_battles
        # How many posterior standard deviations of the underlying rate to give
        # up before sizing. One by default, which is a 68% credible bound, and
        # it is material rather than decorative: measured on a Beta(150, 150)
        # belief about a rung struck at 0.44 quoted 0.69 at 0.70, the stake goes
        # from 93.8% of bankroll at zero, to 81.0% at half a deviation, to 53.3%
        # at one. It vanishes as 1/sqrt(n), so this is a statement about how much
        # evidence there is and not a fraction of Kelly.
        self.credible_z = credible_z
        # The edge is measured from the touch, so the half-spread paid to get in
        # is already inside it. What remains of a round trip is the half-spread
        # to get out plus the taker fee on both legs, and this multiplies that.
        # One is therefore the whole round trip rather than a fraction of it.
        self.cost_multiple = cost_multiple
        self.fee_bps = fee_bps
        self.capital_fraction = capital_fraction
        # None by default and deliberately. A position limit here would compete
        # with the collateral constraint for the right to be the binding one,
        # and which of the two binds is the measurement this strategy exists to
        # report.
        self.max_position = max_position
        # Strata in the posterior quadrature, which only a kinked payoff uses:
        # a binary is analytic and a linear claim is its own mean. Measured on a
        # Beta(2350, 2650) belief, five thousand battles at a rate of 0.47, and a
        # call struck at 4,650 on scale 10,000 worth 59.932: 16 strata price it
        # 0.53 low, 64 price it 0.12 low, and 256 come within 0.027, which is a
        # ninth of that contract's 0.25 tick. Quadrupling again buys another
        # factor of two and costs four times the inverse CDF calls.
        self.draws = max(16, draws)
        # Keyed by what the contract is written on, never by symbol. An agent
        # holding a different posterior per contract on one competitor does not
        # have a posterior for that competitor, and the ladder it prices off
        # those is not the ladder of any distribution.
        self._posterior: dict[str, tuple[float, float]] = dict(posterior or {})
        self._given = posterior is not None
        self._sample: dict[str, list[float]] = {}
        self._drawn = False
        # The Lagrange multiplier on collateral, in expected log wealth per unit
        # of collateral. Zero means capital was free this wakeup.
        self.shadow_price = 0.0
        self.sizing: tuple[Sized, ...] = ()
        # How many candidate trades each constraint decided, over the run.
        self.capital_bound = 0
        self.preference_bound = 0

    # -- belief ------------------------------------------------------------

    def posterior_for_symbol(self, view: MarketView, symbol: str) -> tuple[float, float] | None:
        """This strategy's Beta over the metric a contract is written on.

        ``None`` for anything whose underlying is not a proportion. A Beta is a
        distribution on [0, 1] and the evidence behind it is a count of battles
        won out of battles seen, so a claim on a battle *count* or on a
        difference of two rates has no such posterior. Saying so is cheaper
        than clamping a level into [0, 1] and pretending the answer means
        something: ``HALCYON_FORMAT_SPD`` is bounded by [-1, 1] and the clamp
        would put its entire negative half at zero.
        """
        view_of = view.get(symbol)
        if view_of is None:
            return None
        low, high = view_of.instrument.spec.underlying.bounds()
        if low < 0.0 or high > 1.0:
            return None
        return self._posterior.get(underlying_key(view_of.instrument))

    def _draw(self, view: MarketView) -> None:
        """Draw the evidence once, in symbol order, before anything is priced.

        Order matters and sorted order is the only one that is stable. Drawn
        lazily per symbol, the draws would come off the agent's random stream in
        whatever sequence the books happened to become two-sided in, so two runs
        differing only in the venue would differ in *beliefs*, and the
        comparison would stop being a comparison. That has already happened once
        in this repository; see ``BayesianFundamental.on_start``.
        """
        if self._drawn:
            return
        if self._given:
            self._drawn = True
            return
        if view.rng is None:
            # No stream to draw from yet, so nothing is drawn and nothing is
            # marked done. A guard whose input can be None is a guard that can
            # silently not exist, and this one would have left the strategy with
            # no belief at all for the rest of the session.
            return
        self._drawn = True
        for symbol in sorted(view.symbols):
            view_of = view.get(symbol)
            level = self.levels.get(symbol)
            if view_of is None or level is None:
                continue
            low, high = view_of.instrument.spec.underlying.bounds()
            if low < 0.0 or high > 1.0:
                continue
            key = underlying_key(view_of.instrument)
            if key in self._posterior:
                continue
            self._posterior[key] = posterior_for(
                level, self.battles, self.prior_mean, self.prior_strength, view.rng
            )

    @staticmethod
    def dispersion(a: float, b: float) -> float:
        """The exact posterior sd of the rate, ``sqrt(ab / ((a+b)^2 (a+b+1)))``.

        Exact rather than asymptotic, which matters at the front of a session
        where an agent has seen few battles and the normal approximation to a
        Beta is at its worst.
        """
        total = a + b
        return math.sqrt(a * b / (total * total * (total + 1.0)))

    def _levels(self, key: str, a: float, b: float) -> list[float]:
        """The posterior as equal-probability strata, shared across a subject.

        Quadrature rather than sampling: the ``k``-th of ``n`` levels is the
        posterior quantile at ``(k + 0.5) / n``, so the list *is* the
        distribution rather than a draw from it. Three things follow, and the
        third is why it is worth the inverse CDF.

        There is no Monte Carlo error, so this strategy's own option ladder is
        monotone in strike exactly rather than in expectation. Drawn fresh per
        contract instead, each strike carries its own error, ``max(F - K, 0)``
        stops being decreasing in ``K``, and the strategy then trades the
        difference between two of its own rounding errors.

        And nothing is consumed from the agent's random stream to value a
        contract, so what this strategy believes cannot depend on how many
        other contracts it happened to price first. That failure has already
        happened once here: ``BayesianFundamental`` drew its evidence lazily and
        two runs differing only in the venue came back with different beliefs.

        Held per underlying rather than per symbol, because every contract
        written on a subject has to be priced off one distribution or the
        surface it implies is not the surface of any distribution.
        """
        drawn = self._sample.get(key)
        if drawn is None:
            from scipy.stats import beta as beta_dist

            count = self.draws
            grid = [(k + 0.5) / count for k in range(count)]
            drawn = [float(x) for x in beta_dist.ppf(grid, a, b)]
            self._sample[key] = drawn
        return drawn

    def value(
        self, view_of: SymbolView, a: float, b: float, shift: float = 0.0
    ) -> float:
        """``E[claim_value(theta + shift)]`` in price units, under Beta(a, b).

        ``shift`` displaces the level distribution, which is how the credible
        bound is applied: it is one subtraction for a linear claim and a moved
        threshold for a binary, and in both cases it is the same statement about
        the same parameter rather than two different shrinkages.
        """
        spec = view_of.instrument.spec
        payoff = spec.payoff
        low, high = spec.underlying.bounds()

        if isinstance(payoff, Binary) and spec.distribution is None:
            # Displacing the level by `shift` is exactly moving the threshold by
            # `-shift`, so the analytic tail survives the shrinkage instead of
            # being replaced by a sampled one. A binary's whole value *is* this
            # probability, and sampling it would put Monte Carlo noise straight
            # into the quantity being traded.
            threshold = payoff.threshold - shift
            if self.window_battles is None:
                probability = binary_probability(a, b, threshold, payoff.comparison)
            else:
                above = predictive_probability(a, b, self.window_battles, threshold)
                probability = above if payoff.comparison in (">", ">=") else 1.0 - above
            return probability * payoff.payout

        linear_stream = spec.distribution is None or isinstance(
            spec.distribution.payoff, Linear
        )
        if isinstance(payoff, Linear) and linear_stream:
            # Linear in the level, stream included, so the expectation is the
            # payoff at the mean and no sampling error enters at all.
            mean = min(high, max(low, a / (a + b) + shift))
            return spec.claim_value(mean)

        key = underlying_key(view_of.instrument)
        samples = self._levels(key, a, b)
        total = 0.0
        for level in samples:
            total += spec.claim_value(min(high, max(low, level + shift)))
        return total / len(samples)

    def _outcomes(
        self, view_of: SymbolView, a: float, b: float, shift: float
    ) -> list[tuple[float, float]]:
        """The settlement distribution this trade is sized against, as atoms."""
        spec = view_of.instrument.spec
        payoff = spec.payoff
        low, high = spec.underlying.bounds()
        if isinstance(payoff, Binary) and spec.distribution is None:
            probability = self.value(view_of, a, b, shift) / payoff.payout
            return [(payoff.payout, probability), (0.0, 1.0 - probability)]
        key = underlying_key(view_of.instrument)
        samples = self._levels(key, a, b)
        weight = 1.0 / len(samples)
        return [
            (spec.claim_value(min(high, max(low, level + shift))), weight)
            for level in samples
        ]

    # -- sizing ------------------------------------------------------------

    def _candidate(
        self, view: MarketView, view_of: SymbolView, wealth: float
    ) -> tuple[Side, float, float, float, float, float] | None:
        """One contract's side, price, edge, collateral and unconstrained size.

        ``None`` where there is nothing to trade: a one-sided book, no posterior,
        an edge inside the round-trip band, or a price at the edge of the
        contract's range where the trade posts no collateral at all. That last
        one is left alone on purpose. A position that cannot lose anything has
        no Kelly fraction, because the fraction of bankroll it stakes is zero,
        and how much of it to do is a question about the depth of the book,
        which this strategy structurally cannot see.
        """
        posterior = self.posterior_for_symbol(view, view_of.symbol)
        if posterior is None:
            return None
        quoted = tradeable_touch(view_of)
        if quoted is None:
            return None
        bid, ask = quoted
        a, b = posterior
        shift = self.credible_z * self.dispersion(a, b)
        # Whichever displacement is adverse, found by evaluating both rather
        # than by assuming the value rises with the level. A Linear with a
        # negative scale is a legitimate inverse contract and would have the
        # sign backwards.
        down = self.value(view_of, a, b, -shift)
        up = self.value(view_of, a, b, +shift)
        conservative_low, conservative_high = min(down, up), max(down, up)

        half_spread = float(ask - bid) / 2.0
        fee = self.fee_bps / 10_000.0
        buy_hurdle = self.cost_multiple * (half_spread + 2.0 * fee * float(ask))
        sell_hurdle = self.cost_multiple * (half_spread + 2.0 * fee * float(bid))

        low, high = (float(x) for x in view_of.bounds)
        if conservative_low - float(ask) > buy_hurdle:
            side, price, belief, hurdle = Side.BUY, float(ask), conservative_low, buy_hurdle
        elif float(bid) - conservative_high > sell_hurdle:
            side, price, belief, hurdle = Side.SELL, float(bid), conservative_high, sell_hurdle
        else:
            return None

        # The price at which the edge is exactly used up, which is where the
        # limit goes. Naming the touch instead would reach only the top of the
        # book; naming this reaches whatever depth lies between the touch and
        # the point where the trade stops being worth doing, and never pays past
        # it.
        limit = belief - hurdle if side is Side.BUY else belief + hurdle

        # Collateral is charged against the *limit* while the edge is measured
        # against the touch, and the asymmetry is deliberate. The order can fill
        # anywhere between the two, so the touch is what it expects to pay and
        # the limit is the most it can be asked to post. Sizing on the touch
        # would let a sweep that walked up to the limit post more collateral
        # than the program allotted, which is the constraint the whole thing is
        # solved subject to.
        collateral = (limit - low) if side is Side.BUY else (high - limit)
        if collateral <= 0.0:
            return None

        direction = 1.0 if side is Side.BUY else -1.0
        outcomes = [
            (direction * (value - price), weight)
            for value, weight in self._outcomes(
                view_of, a, b, -shift if side is Side.BUY else shift
            )
        ]
        payoff = view_of.instrument.spec.payoff
        binary = (
            isinstance(payoff, Binary)
            and view_of.instrument.spec.distribution is None
            and payoff.payout > 0.0
        )
        if binary:
            # The closed form, on the contract normalised to a unit binary. A
            # payout of one is not assumed: dividing price and payout through by
            # it is the same bet, and the fraction it returns is a fraction of
            # bankroll either way.
            probability = belief / payoff.payout
            fraction = abs(binary_kelly(price / payoff.payout, probability))
            units = fraction * wealth / collateral
        else:
            units = log_optimal_units(wealth, outcomes, collateral, math.inf)
        edge = sum(profit * weight for profit, weight in outcomes)
        return side, price, edge, collateral, units, limit

    def orders(self, view: MarketView) -> Sequence[Take]:
        """Every trade this strategy wants, richest use of capital first.

        The ordering is the constrained program's own answer rather than a
        heuristic. With a budget on total collateral, the multiplier ``lambda``
        prices a unit of it, and each contract's first unit buys
        ``E[V - p] / (W c)`` of expected log wealth; funding them in that order
        and stopping when the budget runs out is the greedy solution, and the
        density at which it stops is ``lambda``. Since the adapter stops at the
        first intent the account cannot fund, that is also exactly the order the
        intents have to arrive in.
        """
        self._draw(view)
        wealth = float(view.equity)
        if wealth <= 0.0:
            self.sizing, self.shadow_price = (), 0.0
            return ()

        candidates = []
        for view_of in view:
            found = self._candidate(view, view_of, wealth)
            if found is None:
                continue
            side, price, edge, collateral, units, limit = found
            # The ranking key is the marginal expected log wealth the first unit
            # of collateral buys, `E[V - p] / (W c)`, which is the constrained
            # program's own price of capital and not a heuristic score.
            candidates.append(
                (
                    edge / (wealth * collateral),
                    view_of,
                    side,
                    price,
                    edge,
                    collateral,
                    units,
                    limit,
                )
            )
        # Ties broken by symbol so the priority order is a function of the market
        # rather than of dictionary order, which is what makes a run replayable.
        candidates.sort(key=lambda row: (-row[0], row[1].symbol))

        budget = wealth * self.capital_fraction - float(view.posted_collateral)
        shadow = 0.0
        sized: list[Sized] = []
        intents: list[Take] = []
        for density, view_of, side, price, edge, collateral, units, limit in candidates:
            held = view_of.position * (1 if side is Side.BUY else -1)
            wanted = max(0, int(units) - held)
            if self.max_position is not None:
                wanted = min(wanted, self.max_position - held)
            affordable = int(max(0.0, budget) / collateral)
            funded = max(0, min(wanted, affordable))
            if funded < wanted:
                self.capital_bound += 1
                if shadow == 0.0:
                    # The multiplier is the density of the last unit the budget
                    # could not buy, which is this trade's marginal utility at
                    # the size it was cut to. Read off the first cut candidate
                    # because the list is sorted by density.
                    shadow = self._marginal(
                        view, view_of, wealth, side, price, funded, collateral
                    )
            elif wanted > 0:
                self.preference_bound += 1
            sized.append(
                Sized(view_of.symbol, side, edge, collateral, wanted, funded, density)
            )
            if funded <= 0:
                continue
            budget -= funded * collateral
            intents.append(
                Take(
                    view_of.symbol,
                    side,
                    funded,
                    snap(view_of.instrument, side, limit),
                )
            )
        self.sizing = tuple(sized)
        self.shadow_price = shadow
        return intents

    def _marginal(
        self,
        view: MarketView,
        view_of: SymbolView,
        wealth: float,
        side: Side,
        price: float,
        units: int,
        collateral: float,
    ) -> float:
        """``lambda`` implied by cutting this trade to ``units`` lots."""
        posterior = self.posterior_for_symbol(view, view_of.symbol)
        if posterior is None:
            return 0.0
        a, b = posterior
        shift = self.credible_z * self.dispersion(a, b)
        direction = 1.0 if side is Side.BUY else -1.0
        outcomes = [
            (direction * (value - price), weight)
            for value, weight in self._outcomes(
                view_of, a, b, -shift if side is Side.BUY else shift
            )
        ]
        slope = log_wealth_slope(wealth, outcomes, float(units))
        return 0.0 if slope == -math.inf else slope / collateral

    # -- reporting ---------------------------------------------------------

    @property
    def refused_for_capital(self) -> tuple[Sized, ...]:
        """The trades the balance sheet cut, from the most recent wakeup."""
        return tuple(row for row in self.sizing if row.capital_bound)

    def marks(self, view: MarketView) -> dict[str, Decimal]:
        """What this strategy thinks each contract is worth, for reporting.

        Posterior mean rather than credible bound: the bound is a sizing
        device and quoting it as a valuation would confuse caution with belief.
        """
        values: dict[str, Decimal] = {}
        self._draw(view)
        for view_of in view:
            posterior = self.posterior_for_symbol(view, view_of.symbol)
            if posterior is None:
                continue
            values[view_of.symbol] = Decimal(str(self.value(view_of, *posterior)))
        return values
