"""The two optimal-control makers, against the papers and against this venue.

Both models are closed form, so most of what is worth testing is arithmetic and
can be checked exactly rather than approximately. The hand computations here
are written out in plain ``math`` from the published equations, so a test
failing means the implementation drifted from the paper rather than that a
tolerance was too tight.

Three things beyond the arithmetic are worth pinning.

**Avellaneda-Stoikov's spread does not depend on inventory.** The paper says so
of its equation 25 and it follows from exponentially distributed arrival
intensities: the two first-order conditions separate, so inventory moves the
centre of the market and never its width. The incumbent maker in this
repository does the opposite, computing ``half = half_spread * (1 + 2 *
pressure)``, and a test that did not pin this would let the model quietly
become the heuristic it is supposed to be compared against.

**Volatility vanishes at the bounds.** Every contract here settles inside
``spec.value_bounds``, so a constant volatility lets the modelled price diffuse
out of the interval it must settle in. The construction is a diffusion scaled
by ``(p - low) * (high - p)`` over the range, and the tests check both the
shape and that it survives into the emitted quotes.

**Inventory skew has one sign.** Long inventory lowers both quotes, in both
models, because the maker wants to be flattened. Getting this backwards
produces a maker that is superficially reasonable and accumulates until it
dies, which is exactly the failure the sign is there to prevent.

**What the two of them actually did.** Measured at 300s on seeds 7 and 11, one
challenger added to the standard three-maker market on a 20,000,000 seat, the
decomposition taken at the five second horizon, totals in contract currency
with the per-lot figure beside them:

    GueantLehalleFT   -2,191,340 and -3,167,307 on 47,377 and 51,512 lots
                      which is -46.3 and -61.5 a lot, 80% and 82% passive
    AvellanedaStoikov -5,036,716 and -2,682,044 on 305,989 and 325,048 lots
                      which is -16.5 and -8.3 a lot, 76% passive on both
    the incumbents    -26,583,122 and -32,614,574 alongside GLFT, and
                      -4,775,400 and -3,195,354 alongside A-S, on roughly
                      1.2M lots either way

Both models lose money, and both lose more a lot than the three makers they are
measured against. What they reproduce exactly is the *shape* of the incumbents'
loss, which is the finding: spread capture is positive for both, and adverse
selection is 124% and 140% of GLFT's loss and 142% and 141% of A-S's, against
the 117% the incumbents run at. The quoting rule is not what is wrong here.

The one large lever is GLFT's ``xi`` width term, section 5.2's own
prescription, which CONTRIBUTING.md records as 2.2x *worse* when it was applied to
the incumbents and which is 5.7x and 4.1x **better** here. Per lot it is
slightly worse, -43.3 and -46.5 becoming -46.3 and -61.5; the whole of the gain
is that it trades a sixth as many lots. Width is a brake rather than a better
quote, and the incumbents could not use it because they are 98% of the book, so
widening them widens the mid the informed price off and those traders size by
|edge|/uncertainty.

None of those numbers is asserted anywhere below. They move with the listing,
with the seed, and with anything else added to the market, and CONTRIBUTING.md is
explicit that a number like that gets re-measured rather than loosened.
"""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from arena.agents.strategy_agent import StrategyAgent
from arena.exchange.types import Side
from arena.market.live import VENUE_ID
from arena.research.attribution import TradeAttribution
from arena.sim.time import millis, seconds
from arena.strategies.base import MarketView, SymbolView, TwoSided
from arena.strategies.making.avellaneda import (
    AvellanedaStoikov,
    BoundedVolatility,
    bounded_shape,
    reference_price,
)
from arena.strategies.making.gueant import GueantLehalleFT

from dashboard.build_market import build, instruments as build_instruments

CHALLENGER = "opt-1"
SEAT = 20_000_000

# One contract from each shape the listing has, chosen so the arithmetic is
# exercised across the whole 800x range of tick spans rather than on whichever
# symbol happened to be first. `EMBER_OBJECTIVE_WR` spans 40,000 ticks and
# `EMBER_OBJECTIVE_GT500` spans 100, and a parameterisation that means the same
# thing on both is the entire claim the normalised units make.
FUTURE = "EMBER_OBJECTIVE_WR"
BINARY = "EMBER_OBJECTIVE_GT500"
CALL = "EMBER_OBJECTIVE_C4800"


@pytest.fixture(scope="module")
def listed():
    return {i.symbol: i for i in build_instruments()}


class FixedVolatility:
    """A volatility that is whatever the test says it is.

    The estimator has its own tests. Everywhere the *model* is being checked
    against the paper, sigma has to be an input rather than something inferred
    from a synthetic price path, or the hand computation is checking the
    estimator twice and the model not at all.
    """

    def __init__(self, value: float, *, shaped: bool = False) -> None:
        self.value = value
        self.shaped = shaped
        self.observed: list[tuple[str, float, float]] = []

    def observe(self, symbol: str, now: float, level: float) -> None:
        self.observed.append((symbol, now, level))

    def sigma(self, symbol: str, level: float) -> float:
        return self.value * (bounded_shape(level) if self.shaped else 1.0)


def symbol_view(
    instrument,
    level: float,
    position: int = 0,
    *,
    half_spread_ticks: int = 3,
    markout: dict | None = None,
    priced: bool = True,
) -> SymbolView:
    """One contract's view, sitting at ``level`` of its own settlement range."""
    low, high = instrument.spec.value_bounds
    mid = low + Decimal(str(level)) * (high - low)
    half = instrument.tick_size * half_spread_ticks
    return SymbolView(
        symbol=instrument.symbol,
        instrument=instrument,
        best_bid=(mid - half) if priced else None,
        best_ask=(mid + half) if priced else None,
        last=mid if priced else None,
        position=position,
        working_bid=None,
        working_ask=None,
        seconds_to_expiry=None,
        markout=markout or {Side.BUY: None, Side.SELL: None},
    )


def market_view(view: SymbolView, now: float = 100.0) -> MarketView:
    return MarketView(
        now=now,
        symbols=(view.symbol,),
        cash=Decimal(SEAT),
        free_cash=Decimal(SEAT),
        posted_collateral=Decimal(0),
        equity=Decimal(SEAT),
        _by_symbol={view.symbol: view},
    )


def span_of(instrument) -> tuple[float, float]:
    """The contract's range in price units and the same range in ticks."""
    low, high = instrument.spec.value_bounds
    span = float(high - low)
    return span, span / float(instrument.tick_size)


# --------------------------------------------------------------------------
# Avellaneda-Stoikov: the two published equations
# --------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", [FUTURE, BINARY, CALL])
@pytest.mark.parametrize("inventory", [-40, 0, 25])
def test_reservation_price_is_the_papers_equation(listed, symbol, inventory):
    """``r = s - q * gamma * sigma^2 * (T - t)``, with the horizon substituted.

    Computed here from the published form with nothing borrowed from the
    implementation but the parameters, on a binary spanning 100 ticks and a
    future spanning 40,000 so that the normalisation is exercised rather than
    assumed.
    """
    instrument = listed[symbol]
    span, _ = span_of(instrument)
    sigma, horizon, gamma = 0.002, 3.6, 40.0
    strategy = AvellanedaStoikov(
        gamma=gamma,
        horizon_seconds=horizon,
        volatility=FixedVolatility(sigma),
    )
    level = 0.4
    solved = strategy.solve(
        market_view(symbol_view(instrument, level, inventory)), symbol
    )

    variance = min(sigma * sigma * horizon, level * (1.0 - level))
    assert solved.variance == pytest.approx(variance)
    assert solved.reservation == pytest.approx(
        level - inventory * gamma * variance, rel=1e-12
    )
    # And the same statement in the contract's own price units, because a model
    # that is right in normalised space and wrong at the boundary is the first
    # bug class in CONTRIBUTING.md.
    low, _ = instrument.spec.value_bounds
    assert float(low) + solved.reservation * span == pytest.approx(
        float(low) + level * span - inventory * gamma * variance * span
    )


@pytest.mark.parametrize("symbol", [FUTURE, BINARY, CALL])
def test_optimal_spread_is_the_papers_equation(listed, symbol):
    """``d_a + d_b = gamma * sigma^2 (T - t) + (2/gamma) ln(1 + gamma/k)``.

    ``k`` is given in ticks and converted per contract as ``k_ticks *
    span_in_ticks``, because the measured half-spread of this book is a tick
    quantity. Re-measured on the circuit listing, taking the median half spread
    of each contract while it is two sided: 2.5 to 10.5 ticks across all 50
    contracts on seed 7 over 600s, a factor of 4.2, against 0.000072 to 0.040
    of the range for the same spreads, a factor of 556. The tick reading is the
    stable one, which is what makes it the unit to parameterise in.
    """
    instrument = listed[symbol]
    _, span_ticks = span_of(instrument)
    sigma, horizon, gamma, k_ticks = 0.002, 3.6, 40.0, 0.2
    strategy = AvellanedaStoikov(
        gamma=gamma,
        k_ticks=k_ticks,
        horizon_seconds=horizon,
        max_half_spread_ticks=None,
        volatility=FixedVolatility(sigma),
    )
    level = 0.4
    solved = strategy.solve(market_view(symbol_view(instrument, level)), symbol)

    k = k_ticks * span_ticks
    variance = min(sigma * sigma * horizon, level * (1.0 - level))
    expected = gamma * variance + (2.0 / gamma) * math.log(1.0 + gamma / k)
    assert 2.0 * solved.half_spread == pytest.approx(expected, rel=1e-12)


def test_the_spread_does_not_depend_on_inventory_but_the_centre_does(listed):
    """Avellaneda and Stoikov's equation 25 has no ``q`` in it.

    This is the property that separates the model from the maker already in
    this repository, whose ``_requote`` computes ``half = half_spread * (1 + 2
    * pressure)`` with pressure the fraction of its position limit used. Both
    are defensible; only one of them is this paper, and a comparison between
    them is worthless if the model has quietly become the heuristic.
    """
    instrument = listed[FUTURE]
    strategy = AvellanedaStoikov(volatility=FixedVolatility(0.002))
    solved = [
        strategy.solve(market_view(symbol_view(instrument, 0.5, q)), FUTURE)
        for q in (-90, -30, 0, 30, 90)
    ]
    widths = {round(s.half_spread, 15) for s in solved}
    assert len(widths) == 1, "the spread moved with inventory"
    centres = [s.reservation for s in solved]
    assert centres == sorted(centres, reverse=True), "the centre did not move"
    assert centres[0] > centres[-1]


@pytest.mark.parametrize(
    "strategy_class", [AvellanedaStoikov, GueantLehalleFT], ids=["AS", "GLFT"]
)
def test_long_inventory_lowers_both_quotes(listed, strategy_class):
    """The sign of the skew, which is the whole of a maker's inventory control.

    Long means both quotes come down: the bid because buying more is the thing
    to discourage, the ask because selling is the thing to reward. Backwards,
    the maker adds to whichever position it already has and runs to its limit,
    which is the failure the incumbent's own position limit exists to survive.
    """
    instrument = listed[FUTURE]
    strategy = strategy_class(volatility=FixedVolatility(0.002), cross_touch=True)
    flat = strategy.quote(market_view(symbol_view(instrument, 0.5, 0)), FUTURE)
    long = strategy.quote(market_view(symbol_view(instrument, 0.5, 40)), FUTURE)
    short = strategy.quote(market_view(symbol_view(instrument, 0.5, -40)), FUTURE)

    assert long.bid.price < flat.bid.price < short.bid.price
    assert long.ask.price < flat.ask.price < short.ask.price


def test_the_settlement_horizon_replaces_the_liquidation_horizon(listed):
    """``horizon_seconds=None`` is the hold-to-settlement variance, exactly.

    The paper's ``(T - t)`` is time to liquidation at the mid, and nothing here
    is liquidated at the mid: every contract settles inside its own bounds. The
    largest variance a martingale on ``[0, 1]`` sitting at ``u`` can have is
    ``u (1 - u)``, attained by the two-point distribution on the bounds, so
    that is what the term becomes when the horizon is dropped. The adapter
    passes ``seconds_to_expiry=None`` unconditionally, so the paper's own term
    is not merely inconvenient here, it is unavailable.
    """
    instrument = listed[FUTURE]
    strategy = AvellanedaStoikov(
        horizon_seconds=None, volatility=FixedVolatility(0.5)
    )
    for level in (0.1, 0.35, 0.5, 0.8):
        solved = strategy.solve(
            market_view(symbol_view(instrument, level)), FUTURE
        )
        assert solved.variance == pytest.approx(level * (1.0 - level))

    # And with a horizon, the cap is what stops a large volatility asking for
    # more inventory risk than the contract can deliver.
    capped = AvellanedaStoikov(horizon_seconds=1e6, volatility=FixedVolatility(0.5))
    solved = capped.solve(market_view(symbol_view(instrument, 0.3)), FUTURE)
    assert solved.variance == pytest.approx(0.3 * 0.7)


# --------------------------------------------------------------------------
# Gueant, Lehalle and Fernandez-Tapia: the closed form and section 5.2
# --------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", [FUTURE, BINARY, CALL])
@pytest.mark.parametrize("inventory", [-30, 0, 17])
def test_closed_form_distances_match_the_paper(listed, symbol, inventory):
    """Both quoted distances, from arXiv:1105.3115 written out longhand."""
    instrument = listed[symbol]
    _, span_ticks = span_of(instrument)
    sigma, gamma, k_ticks, arrival = 0.002, 40.0, 0.2, 40.0
    strategy = GueantLehalleFT(
        gamma=gamma,
        k_ticks=k_ticks,
        arrival_rate=arrival,
        xi_width=0.0,
        xi_skew=0.0,
        volatility=FixedVolatility(sigma),
    )
    solved = strategy.solve(
        market_view(symbol_view(instrument, 0.45, inventory)), symbol
    )

    k = k_ticks * span_ticks
    base = (1.0 / gamma) * math.log(1.0 + gamma / k)
    inner = (
        sigma * sigma * gamma / (2.0 * k * arrival)
        * (1.0 + gamma / k) ** (1.0 + k / gamma)
    )
    root = math.sqrt(inner)
    assert solved.base == pytest.approx(base, rel=1e-12)
    assert solved.skew_per_lot == pytest.approx(root, rel=1e-9)
    assert solved.bid == pytest.approx(
        base + ((2 * inventory + 1) / 2.0) * root, rel=1e-9
    )
    assert solved.ask == pytest.approx(
        base - ((2 * inventory - 1) / 2.0) * root, rel=1e-9
    )


def test_at_flat_inventory_the_two_distances_are_symmetric(listed):
    """``q = 0`` leaves ``+c/2`` on each side, which is the paper's own check."""
    instrument = listed[CALL]
    strategy = GueantLehalleFT(
        xi_width=0.0, xi_skew=0.0, volatility=FixedVolatility(0.002)
    )
    solved = strategy.solve(market_view(symbol_view(instrument, 0.5, 0)), CALL)
    assert solved.bid == pytest.approx(solved.ask, rel=1e-12)
    assert solved.bid == pytest.approx(
        solved.base + solved.skew_per_lot / 2.0, rel=1e-12
    )


def test_the_closed_form_width_is_also_free_of_inventory(listed):
    """``d_a + d_b = 2 base + c`` whatever ``q`` is, which is not obvious.

    The two skew terms are ``(2q + 1)/2`` and ``-(2q - 1)/2``, and they sum to
    one, so the inventory cancels out of the width exactly as it does in
    Avellaneda-Stoikov's equation 25 and for the same underlying reason. Both
    papers move the centre and neither widens, which is worth pinning because
    the maker already in this repository widens with inventory and it would be
    easy to import that habit while porting a formula.
    """
    instrument = listed[CALL]
    strategy = GueantLehalleFT(
        xi_width=0.0, xi_skew=0.0, volatility=FixedVolatility(0.002)
    )
    widths = set()
    for q in (-80, -20, 0, 20, 80):
        solved = strategy.solve(market_view(symbol_view(instrument, 0.5, q)), CALL)
        widths.add(round(solved.bid + solved.ask, 15))
        assert solved.bid + solved.ask == pytest.approx(
            2 * solved.base + solved.skew_per_lot, rel=1e-12
        )
    assert len(widths) == 1


def test_xi_is_the_measured_markout_and_only_its_adverse_half(listed):
    """Section 5.2's ``xi``, taken from the adapter rather than invented.

    ``StrategyAgent`` signs its markout so that positive means the mid moved
    the strategy's way. GLFT's ``xi`` is the move *against* the maker, so a
    rewarded side contributes nothing rather than a subsidy, and a side with no
    matured fills yet contributes nothing rather than a guess.
    """
    instrument = listed[FUTURE]
    span, _ = span_of(instrument)
    damage = 12.5  # contract price units per lot
    strategy = GueantLehalleFT(
        xi_width=1.0, xi_skew=1.0, xi_size=False, volatility=FixedVolatility(0.002)
    )
    plain = strategy.solve(market_view(symbol_view(instrument, 0.5, 0)), FUTURE)

    hurt = strategy.solve(
        market_view(
            symbol_view(
                instrument,
                0.5,
                0,
                markout={Side.BUY: -damage, Side.SELL: -damage},
            )
        ),
        FUTURE,
    )
    assert hurt.xi_bid == pytest.approx(damage / span)
    assert hurt.xi_ask == pytest.approx(damage / span)
    # The paper adds xi/2 to each side.
    assert hurt.bid == pytest.approx(plain.bid + damage / span / 2.0)
    assert hurt.ask == pytest.approx(plain.ask + damage / span / 2.0)

    rewarded = strategy.solve(
        market_view(
            symbol_view(
                instrument, 0.5, 0, markout={Side.BUY: damage, Side.SELL: damage}
            )
        ),
        FUTURE,
    )
    assert rewarded.xi_bid == 0.0
    assert rewarded.bid == pytest.approx(plain.bid)

    absent = strategy.solve(
        market_view(
            symbol_view(instrument, 0.5, 0, markout={Side.BUY: None, Side.SELL: None})
        ),
        FUTURE,
    )
    assert absent.xi_bid == 0.0 and absent.xi_ask == 0.0


def test_xi_splits_into_a_width_half_and_a_skew_half(listed):
    """Turning the width off leaves the centre free to move, and only that.

    CONTRIBUTING.md records widening each side by half the measured markout, which is
    literally this term, costing 2.2x on this venue. The symmetric part of
    ``xi`` is the whole of that width and the antisymmetric part is skew, so
    they are separable and the evidence says to spend one and not the other.
    """
    instrument = listed[FUTURE]
    span, _ = span_of(instrument)
    kwargs = dict(xi_size=False, volatility=FixedVolatility(0.002))
    lopsided = {Side.BUY: -20.0, Side.SELL: -4.0}
    view = market_view(symbol_view(instrument, 0.5, 0, markout=lopsided))

    paper = GueantLehalleFT(xi_width=1.0, xi_skew=1.0, **kwargs).solve(view, FUTURE)
    skew_only = GueantLehalleFT(xi_width=0.0, xi_skew=1.0, **kwargs).solve(view, FUTURE)
    none = GueantLehalleFT(xi_width=0.0, xi_skew=0.0, **kwargs).solve(view, FUTURE)

    def width(d):
        return (d.bid + d.ask) / 2.0

    def centre(d):
        return (d.ask - d.bid) / 2.0

    assert width(skew_only) == pytest.approx(width(none))
    assert width(paper) > width(none)
    # Buying is being punished four times harder, so the centre comes down.
    assert centre(skew_only) < centre(none)
    assert centre(skew_only) == pytest.approx(centre(paper))
    assert width(paper) - width(none) == pytest.approx((20.0 + 4.0) / span / 4.0)


def test_a_side_with_no_edge_left_is_not_quoted(listed):
    """Size, which is the channel the measurement here says to prefer.

    A fill at distance ``d`` on a side whose measured markout is ``xi`` earns
    ``d - xi``. When that is negative the side is not worth showing, and
    ``TwoSided`` is built to say so: its own docstring calls a ``None`` side "a
    legitimate and frequently correct answer", distinct from a size of zero,
    which is not representable.
    """
    instrument = listed[FUTURE]
    span, _ = span_of(instrument)
    strategy = GueantLehalleFT(volatility=FixedVolatility(0.002))
    flat = strategy.solve(market_view(symbol_view(instrument, 0.5, 0)), FUTURE)
    ruinous = (flat.bid + 1.0) * span

    quoted = strategy.quote(
        market_view(
            symbol_view(
                instrument, 0.5, 0, markout={Side.BUY: -ruinous, Side.SELL: None}
            )
        ),
        FUTURE,
    )
    assert quoted.bid is None
    assert quoted.ask is not None

    # A partial loss of edge shows less rather than nothing.
    half_gone = strategy.quote(
        market_view(
            symbol_view(
                instrument,
                0.5,
                0,
                markout={Side.BUY: -flat.bid * span / 2.0, Side.SELL: None},
            )
        ),
        FUTURE,
    )
    full = strategy.quote(market_view(symbol_view(instrument, 0.5, 0)), FUTURE)
    assert 0 < half_gone.bid.size < full.bid.size


# --------------------------------------------------------------------------
# The bounded adaptation
# --------------------------------------------------------------------------


def test_the_volatility_shape_vanishes_at_both_bounds():
    """``4 u (1 - u)``: zero at each bound, one in the middle, never negative.

    This is the whole of why a bounded claim can be modelled as a diffusion at
    all. Without it the modelled price walks out of the interval it is
    contractually required to settle inside.
    """
    assert bounded_shape(0.0) == 0.0
    assert bounded_shape(1.0) == 0.0
    assert bounded_shape(0.5) == pytest.approx(1.0)
    assert bounded_shape(-0.2) == 0.0 and bounded_shape(1.3) == 0.0
    rising = [bounded_shape(u) for u in (0.01, 0.1, 0.25, 0.4, 0.5)]
    assert rising == sorted(rising)


def test_estimated_volatility_goes_to_zero_at_the_bounds():
    """The estimate is transported by the shape, so the bounds still hold it.

    The estimator is fed a walk in the middle of the range and then asked what
    the volatility is near each bound. It has to fall away, because a contract
    trading at its floor cannot move much further down and the model has to
    know that before it prices inventory risk against it.
    """
    volatility = BoundedVolatility(gain=0.4, warmup=4, sample_interval=0.25)
    now = 0.0
    for n in range(40):
        now += 0.3
        volatility.observe("X", now, 0.5 + (0.01 if n % 2 else -0.01))

    middle = volatility.sigma("X", 0.5)
    assert middle > 0
    for level in (0.2, 0.05, 0.01, 0.001):
        assert volatility.sigma("X", level) < middle
    assert volatility.sigma("X", 0.0) == 0.0
    assert volatility.sigma("X", 1.0) == 0.0
    assert volatility.sigma("X", 0.001) == pytest.approx(0.0, abs=middle / 100)


def test_the_estimator_recovers_a_known_volatility():
    """A walk of known size comes back as a volatility of the right size.

    The estimator is a mean absolute deviation scaled by ``sqrt(pi/2)`` rather
    than a mean square, because the mid series here jumps. Re-measured over
    600s on seed 7 across the circuit listing, sampling every 250ms: on
    ``QUILL_SOLO_WR`` the RMS normalised volatility is 0.0729 per root second
    against a median-based 0.0019, a factor of 39. On twenty of the fifty
    contracts the median move is exactly zero while the RMS runs 0.027 to
    0.342, which says the same thing more sharply: the ordinary interval on
    this market has no move in it at all, and the whole of a squared estimate
    is a handful of excursions. A square hands those the estimate.
    """
    step, interval = 0.004, 0.25
    volatility = BoundedVolatility(gain=0.5, warmup=4, sample_interval=interval)
    now = 0.0
    # Symmetric about the midpoint, so every increment is sampled at a shape of
    # exactly one and nothing is transported. Walking from 0.5 upward instead
    # leaves the estimate 0.0016% high, which is the transport doing its job on
    # a walk whose average level is not where the answer is asked for.
    for n in range(80):
        now += interval
        volatility.observe("X", now, 0.5 + (step / 2 if n % 2 else -step / 2))
    # Every increment is exactly `step` in magnitude, so the mean absolute
    # deviation is `step / sqrt(interval)` and the estimate is that times
    # `sqrt(pi/2)`.
    expected = step / math.sqrt(interval) * math.sqrt(math.pi / 2.0)
    assert volatility.sigma("X", 0.5) == pytest.approx(expected, rel=1e-9)


def test_the_estimator_ignores_samples_that_are_too_close_together():
    """A strategy is asked to requote the instant it fills, which is not a sample.

    ``StrategyAgent`` calls ``quote`` again on every fill because being lifted
    is news, so without a floor on the sampling interval the estimator divides
    a zero increment by a microsecond and reports either nothing or everything.
    """
    volatility = BoundedVolatility(sample_interval=0.25)
    volatility.observe("X", 0.0, 0.5)
    for n in range(50):
        volatility.observe("X", 0.001 * n, 0.5 + 0.01 * n)
    assert volatility._by_symbol["X"].samples == 0


@pytest.mark.parametrize(
    "strategy_class", [AvellanedaStoikov, GueantLehalleFT], ids=["AS", "GLFT"]
)
def test_every_quote_is_a_decimal_on_the_grid_inside_the_bounds(
    listed, strategy_class
):
    """Nothing floating-point reaches an order, and nothing leaves the range.

    Swept across every listed contract, the whole normalised range and
    inventories out past the bound, because the failure this catches is a
    strategy that is correct in the middle and quotes a price the contract
    cannot pay at the edge. The range clamp is applied to each side separately:
    clamping the *centre* is in the CONTRIBUTING.md table of things that look like
    improvements, since it makes the mid a function of the half-spread.
    """
    for instrument in listed.values():
        low, high = instrument.spec.value_bounds
        strategy = strategy_class(volatility=FixedVolatility(0.02))
        for level in (0.0, 0.001, 0.25, 0.5, 0.75, 0.999, 1.0):
            for position in (-200, -30, 0, 30, 200):
                view = market_view(symbol_view(instrument, level, position))
                quoted = strategy.quote(view, instrument.symbol)
                for quote in (quoted.bid, quoted.ask):
                    if quote is None:
                        continue
                    assert isinstance(quote.price, Decimal)
                    assert low <= quote.price <= high, (instrument.symbol, level)
                    assert instrument.on_grid(quote.price)
                    assert quote.size > 0


@pytest.mark.parametrize(
    "strategy_class", [AvellanedaStoikov, GueantLehalleFT], ids=["AS", "GLFT"]
)
def test_the_inventory_bound_is_hard(listed, strategy_class):
    """At the bound the side that would breach it is withdrawn, not shrunk.

    Quoting anyway and letting the venue refuse burns order ids and hides the
    constraint from the strategy's own logic, which is the reason the incumbent
    maker in this repository gives for doing the same thing.
    """
    instrument = listed[FUTURE]
    bound = 40
    strategy = strategy_class(
        inventory_bound=bound, quote_size=25, volatility=FixedVolatility(0.002)
    )
    at_ceiling = strategy.quote(
        market_view(symbol_view(instrument, 0.5, bound)), FUTURE
    )
    assert at_ceiling.bid is None and at_ceiling.ask is not None

    at_floor = strategy.quote(
        market_view(symbol_view(instrument, 0.5, -bound)), FUTURE
    )
    assert at_floor.ask is None and at_floor.bid is not None

    nearly = strategy.quote(
        market_view(symbol_view(instrument, 0.5, bound - 6)), FUTURE
    )
    assert nearly.bid.size == 6


@pytest.mark.parametrize(
    "strategy_class", [AvellanedaStoikov, GueantLehalleFT], ids=["AS", "GLFT"]
)
def test_a_book_that_has_never_priced_is_not_quoted(listed, strategy_class):
    """No mid, no last, no quote, unless the caller asks for a guess.

    ``build_market`` records what happens when a maker names the opening price
    on a venue that opens with a call auction: the auction clears at the guess,
    and the market's walk to fair value trips the breaker on every symbol
    inside the first minute.
    """
    instrument = listed[FUTURE]
    view = market_view(symbol_view(instrument, 0.5, 0, priced=False))
    assert strategy_class(volatility=FixedVolatility(0.002)).quote(
        view, FUTURE
    ) == TwoSided()
    guessing = strategy_class(
        quote_without_reference=True, volatility=FixedVolatility(0.002)
    )
    assert not guessing.quote(view, FUTURE).is_empty


@pytest.mark.parametrize(
    "strategy_class", [AvellanedaStoikov, GueantLehalleFT], ids=["AS", "GLFT"]
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": 0.0},
        {"k_ticks": 0.0},
        {"quote_size": 0},
        {"inventory_bound": 0},
    ],
)
def test_a_parameter_that_would_divide_by_zero_is_refused(strategy_class, kwargs):
    """Refused at construction, not several frames into the closed form.

    Both papers need a positive risk aversion and a positive arrival decay:
    ``gamma`` is a constant absolute risk aversion coefficient and ``A exp(-k
    d)`` is not an intensity otherwise. A zero surfaces as a ZeroDivisionError
    inside ``solve`` on whichever contract happened to be quoted first, which
    names the wrong thing.
    """
    with pytest.raises(ValueError):
        strategy_class(**kwargs)


def test_reference_price_reads_a_mid_of_exactly_zero(listed):
    """A contract worth nothing is worth nothing, not the middle of its range.

    ``SymbolView.reference`` is written ``self.mid or self.last or midpoint``,
    and ``Decimal(0)`` is falsy, so a book whose mid is exactly zero reads as
    having no price at all and falls through to the middle of the range.

    Measured, that state does not occur today: over 300s on seed 7 not one of
    the 47 books had a mid of exactly zero at any of 1,200 samples, because a
    bid at the floor against an offer a tick above gives half a tick rather
    than nothing. It is one cancelled offer away, though, and seven of the
    option books sit at a median normalised level of 0.000, where the value it
    would fall through to is 2,350 on a put worth nothing.
    """
    instrument = listed[CALL]
    zero = symbol_view(instrument, 0.0, 0, half_spread_ticks=0)
    assert zero.mid == 0
    assert reference_price(zero) == Decimal(0)

    empty = symbol_view(instrument, 0.5, 0, priced=False)
    assert reference_price(empty) is None


def test_not_crossing_the_touch_holds_each_side_independently(listed):
    """The centre never moves to keep a side inside the book.

    Clamping the centre of a quote is in the CONTRIBUTING.md table of things that
    look like improvements and are not: it makes the mid a function of the
    half-spread, so two calls both worth nothing marked 40 points apart. Each
    side is held on its own here, and a side with no room left is dropped
    rather than pinned on top of the touch.
    """
    instrument = listed[FUTURE]
    tick = instrument.tick_size
    # Enough inventory that the model wants to quote through the market.
    view = symbol_view(instrument, 0.5, 60, half_spread_ticks=3)
    strategy = GueantLehalleFT(
        cross_touch=False, xi_size=False, volatility=FixedVolatility(0.004)
    )
    held = strategy.quote(market_view(view), FUTURE)
    crossing = GueantLehalleFT(
        cross_touch=True, xi_size=False, volatility=FixedVolatility(0.004)
    ).quote(market_view(view), FUTURE)

    assert crossing.ask.price <= view.best_bid, "the test needs a crossing quote"
    assert held.ask.price == view.best_bid + tick
    assert held.bid.price <= view.best_ask - tick


# --------------------------------------------------------------------------
# In the live market
# --------------------------------------------------------------------------


def run_challenger(strategy, seed: int = 7, until: float = 45.0, step: float = 0.25):
    """One strategy, on a 20,000,000 seat, in the standard three-maker market."""
    market = build(seed=seed)
    market.venue.open_account(CHALLENGER, Decimal(SEAT))
    by_symbol = {
        symbol: market.venue.registry.require(symbol)
        for symbol in market.venue.registry.symbols
    }
    agent = StrategyAgent(
        CHALLENGER,
        VENUE_ID,
        by_symbol,
        millis(320),
        maker=strategy,
        starting_cash=Decimal(SEAT),
    )
    market.kernel.add(agent)
    attribution = TradeAttribution(market.venue, horizons=(seconds(1), seconds(5)))
    attribution.attach()
    market.kernel.start()
    now = 0
    while now < seconds(until):
        now += seconds(step)
        market.kernel.advance(until=now)
        attribution.sample(now)
    # Drain the wire before reading anything. A fill printed at the last
    # instant is still in flight to the agent that owns it, so stopping the
    # clock on the sample and then comparing the agent's own book against the
    # venue's compares two moments and calls the difference a defect.
    #
    # Draining is not a passive act, though, which is why a fixed window is not
    # enough. Advancing the clock lets the operator run its scheduled auctions,
    # so the drain can print the very fill it exists to deliver. Measured on
    # the Avellaneda-Stoikov run: an uncross at t=46.000s, exactly the old
    # one-second deadline, filled a sell of twelve lots that could not reach a
    # 320ms agent before the test read its position, and the seat came out at
    # +6 against the venue's -6. One dropped fill, and the sign of the whole
    # position with it.
    #
    # Waiting for the market to go quiet does not work either, because it never
    # does: the seat keeps quoting, so there is always another fill on the wire
    # and every snapshot is racing one. The seat has to be taken out of the
    # market first, and the venue already has the tool for it. `kill` pulls the
    # participant's working orders and refuses it any more, so once the cancels
    # land its position cannot change again and the drain has a fixed point to
    # reach.
    #
    # It is the assertion that stays honest here: nothing is being waited for
    # until it agrees. The seat is stopped, the wire is drained for three times
    # its own latency, and the agent is then asked what it holds. A fill this
    # agent genuinely failed to book would still be missing.
    market.venue.kill(CHALLENGER, reason="end of session")
    now += seconds(1)
    market.kernel.advance(until=now)
    attribution.sample(now)
    attribution.detach()
    return market, attribution, agent


@pytest.mark.parametrize(
    "strategy_class", [AvellanedaStoikov, GueantLehalleFT], ids=["AS", "GLFT"]
)
def test_the_strategy_trades_and_conservation_stays_zero(strategy_class):
    """It gets filled, its own ledger agrees with the venue, and money is closed.

    Conservation is checked as an exact integer, because it is one:
    ``venue.conservation_check()`` returns ``int`` and returns ``0``, and
    CONTRIBUTING.md is explicit that a tolerance here would be hiding a bug
    elsewhere.

    What it traded is in the module docstring. Nothing about the size of the
    P&L is asserted here, only that there was one: the third rule of this
    repository is that nothing may be tuned to make a number appear, and a
    threshold on a strategy's profit is exactly the kind of assertion that
    later gets loosened instead of re-measured.
    """
    market, attribution, agent = run_challenger(strategy_class())
    report = attribution.report(seconds(1))
    row = report.get(CHALLENGER)

    assert row is not None, "the strategy never traded"
    assert row.lots > 0
    assert market.venue.conservation_check() == 0

    # The agent's own shadow ledger against the venue's book for it, which is
    # the check that the fills it thinks it had are the fills it had.
    account = market.venue.account(CHALLENGER)
    for symbol, quantity in agent.position.items():
        held = account.positions.get(symbol)
        assert int(held.quantity if held is not None else 0) == quantity


def test_the_decomposition_reconciles_for_the_challenger():
    """Spread plus adverse selection plus residual is the whole trading P&L.

    The identity is arithmetic rather than a model, ``M`` and ``M_h`` each
    added once and subtracted once, so a failure here would be an accounting
    bug rather than a strategy that behaved unexpectedly. Checked on the
    challenger because a decomposition that only reconciles for agents that
    were in the market at build time is not a decomposition.
    """
    _, attribution, _ = run_challenger(GueantLehalleFT())
    row = attribution.report(seconds(1))[CHALLENGER]
    assert row.total == pytest.approx(
        row.spread_captured + row.adverse_selection + row.residual, rel=1e-9
    )
    assert row.realized_spread == pytest.approx(
        row.spread_captured + row.adverse_selection, rel=1e-9
    )
    assert row.passive_lots + row.aggressive_lots == row.lots
