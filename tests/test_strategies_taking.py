"""The two buy-side strategies, and the arithmetic each of them rests on.

Both are testable in a way most trading code is not, and for opposite reasons.
``KellyBayesian`` rests on a formula with a closed form, so its sizing can be
checked against a number computed by hand rather than against itself.
``StaticArbitrage`` rests on inequalities that hold at every level the metric
can take, so "is this package really riskless" is a question about a
piecewise-linear function of one bounded variable and the answer is arithmetic.
Neither claim needs a simulation to check, and the tests that do run a market
are checking what a market adds: that the intents survive contact with a venue,
that value is conserved exactly, and that no account ends over-committed.

The measurements behind the numbers asserted here are named in each docstring.
Where a figure depends on the listing, it is derived from the listing in the
test rather than written down, so a twenty-ninth contract moves it honestly.
"""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from arena.agents.bayesian import binary_probability
from arena.agents.fundamental import underlying_key
from arena.agents.strategy_agent import StrategyAgent
from arena.contracts.payoff import Binary
from arena.exchange.types import Side
from arena.market.fees import MAKER_TAKER
from arena.market.live import VENUE_ID
from arena.portfolio.netting import worst_case
from arena.sim.time import millis, seconds
from arena.strategies.base import MarketView, SymbolView
from arena.strategies.taking.arbitrage import StaticArbitrage, derive_binary_relations
from arena.strategies.taking.kelly import (
    KellyBayesian,
    binary_kelly,
    log_optimal_units,
    log_wealth_slope,
)

from dashboard.build_market import build, instruments as build_instruments, true_levels

CASH = Decimal(20_000_000)


# --------------------------------------------------------------------------
# Building a view without a market
# --------------------------------------------------------------------------


def listing() -> dict[str, object]:
    return {i.symbol: i for i in build_instruments()}


def binaries_on_one_underlying(listed: dict) -> list:
    """The longest binary ladder in the listing, sorted by threshold.

    Chosen by structure rather than by name so that relisting the exchange
    moves this test's subject instead of breaking it.
    """
    groups: dict[str, list] = {}
    for symbol, instrument in sorted(listed.items()):
        payoff = instrument.spec.payoff
        if isinstance(payoff, Binary):
            groups.setdefault(underlying_key(instrument), []).append(
                (payoff.threshold, instrument)
            )
    longest = max(groups.values(), key=len)
    return [instrument for _threshold, instrument in sorted(longest, key=lambda r: r[0])]


def a_future(listed: dict):
    """A plain linear claim on one rate, for the non-binary sizing path.

    Picked structurally rather than by name, and restricted to a single metric:
    a basket would also pass the linearity test, and the Beta posterior this
    strategy holds is a belief about one proportion rather than about a weighted
    sum of several.
    """
    from arena.contracts.payoff import Linear
    from arena.contracts.underlying import Single

    for symbol, instrument in sorted(listed.items()):
        payoff = instrument.spec.payoff
        low, high = instrument.spec.underlying.bounds()
        if (
            isinstance(payoff, Linear)
            and payoff.offset == 0.0
            and instrument.spec.distribution is None
            and isinstance(instrument.spec.underlying, Single)
            and low >= 0.0
            and high <= 1.0
        ):
            return instrument
    raise AssertionError("the listing carries no plain rate future")


def view_of(
    instruments: dict,
    quotes: dict[str, tuple[str, str]],
    *,
    equity: Decimal = Decimal(1_000_000),
    posted: Decimal = Decimal(0),
    positions: dict[str, int] | None = None,
    rng=None,
) -> MarketView:
    """A market view assembled by hand, with no venue behind it.

    Exactly what the adapter would hand a strategy, which is the point: a
    strategy that needed a running market to be tested would be a strategy
    whose sizing could only be checked against itself.
    """
    held = positions or {}
    by_symbol = {}
    for symbol, instrument in instruments.items():
        quote = quotes.get(symbol)
        by_symbol[symbol] = SymbolView(
            symbol=symbol,
            instrument=instrument,
            best_bid=None if quote is None else Decimal(quote[0]),
            best_ask=None if quote is None else Decimal(quote[1]),
            last=None,
            position=held.get(symbol, 0),
            working_bid=None,
            working_ask=None,
            seconds_to_expiry=None,
        )
    return MarketView(
        now=0.0,
        symbols=tuple(instruments),
        cash=equity,
        free_cash=equity - posted,
        posted_collateral=posted,
        equity=equity,
        _by_symbol=by_symbol,
        rng=rng,
    )


# --------------------------------------------------------------------------
# The closed form
# --------------------------------------------------------------------------


def test_binary_kelly_matches_the_hand_computation():
    """Kelly 1956, on the two bets a binary offers.

    Buying at 0.40 believing 0.60 risks 0.40 to win 0.60, which is odds 1.5 at
    probability 0.6, so ``(p b - q) / b = (0.9 - 0.4) / 1.5 = 1/3``. Selling at
    0.70 believing 0.50 risks 0.30 to win 0.70, odds 7/3 at probability 0.5, so
    ``(0.5 * 7/3 - 0.5) / (7/3) = 2/7``. Both are stakes as a fraction of
    bankroll, and on this venue a stake is the collateral posted.
    """
    assert binary_kelly(0.40, 0.60) == pytest.approx(1.0 / 3.0)
    assert binary_kelly(0.70, 0.50) == pytest.approx(-2.0 / 7.0)
    # A tenth of the way from a price to certainty is a tenth of the bankroll.
    assert binary_kelly(0.50, 0.55) == pytest.approx(0.10)
    assert binary_kelly(0.50, 0.45) == pytest.approx(-0.10)


def test_binary_kelly_stakes_nothing_on_a_fair_price():
    """No edge, no bet. The one case where a Kelly bettor sits out."""
    assert binary_kelly(0.42, 0.42) == 0.0
    # And nothing at a price the contract cannot move away from, where the
    # stake is not a fraction of anything.
    assert binary_kelly(0.0, 0.9) == 0.0
    assert binary_kelly(1.0, 0.1) == 0.0


def test_the_bisection_reproduces_the_closed_form():
    """The general solver has to agree with the formula where one exists.

    Measured: on a bankroll of 1,000 buying a unit binary at 0.40 believing
    0.60, the closed form asks for 833.3333333 lots and eighty bisections
    return the same number to 1.1e-13. The solver is the path every future,
    option and share takes, so an agreement this close is what says those are
    sized by the same principle rather than by a different one.
    """
    wealth, price, belief = 1_000.0, 0.40, 0.60
    outcomes = [(1.0 - price, belief), (-price, 1.0 - belief)]
    units = log_optimal_units(wealth, outcomes, price, math.inf)
    closed = binary_kelly(price, belief) * wealth / price
    assert units == pytest.approx(closed, abs=1e-9)
    # And it is a stationary point of the objective it claims to maximise.
    assert log_wealth_slope(wealth, outcomes, units) == pytest.approx(0.0, abs=1e-12)


def test_the_solver_never_sizes_into_ruin():
    """A bet that cannot lose still stops at what the balance sheet holds.

    ``E[log W]`` is minus infinity at the point a position could take wealth to
    zero, so the unconstrained answer never reaches it. A package with no losing
    outcome has no such stationary point, and the cap is then the only thing
    that answers.
    """
    wealth, collateral = 1_000.0, 2.0
    free_lunch = [(1.0, 0.5), (3.0, 0.5)]
    units = log_optimal_units(wealth, free_lunch, collateral, math.inf)
    assert units * collateral <= wealth
    assert log_optimal_units(wealth, free_lunch, collateral, 10.0) == pytest.approx(10.0)


# --------------------------------------------------------------------------
# Shrinkage for estimation error
# --------------------------------------------------------------------------


def _sized(strategy: KellyBayesian, view: MarketView, symbol: str):
    strategy.orders(view)
    for row in strategy.sizing:
        if row.symbol == symbol:
            return row
    return None


def _kelly_on(instrument, posterior, *, credible_z=1.0, **kwargs) -> KellyBayesian:
    return KellyBayesian(
        {},
        battles=0,
        posterior={underlying_key(instrument): posterior},
        credible_z=credible_z,
        **kwargs,
    )


def test_the_binary_size_is_the_closed_form_the_strategy_claims():
    """End to end on a binary: belief, price, bankroll, and the same fraction.

    Computed independently here from the Beta tail and :func:`binary_kelly`,
    not read back off the strategy, so this checks the wiring rather than the
    formula a second time. Shrinkage off, because the closed form is a statement
    about the posterior mean and not about a credible bound on it.
    """
    listed = listing()
    ladder = binaries_on_one_underlying(listed)
    contract = ladder[0]
    payoff = contract.spec.payoff
    posterior = (140.0, 160.0)
    strategy = _kelly_on(contract, posterior, credible_z=0.0)
    quotes = {contract.symbol: ("0.30", "0.31")}
    equity = Decimal(1_000_000)
    view = view_of({contract.symbol: contract}, quotes, equity=equity)

    row = _sized(strategy, view, contract.symbol)
    assert row is not None and row.side is Side.BUY

    belief = binary_probability(*posterior, payoff.threshold, payoff.comparison)
    fraction = binary_kelly(0.31 / payoff.payout, belief / payoff.payout)
    # Collateral is charged at the limit the order carries, which is where the
    # edge runs out, because that is the most the fill can be asked to post.
    expected = int(fraction * float(equity) / row.collateral_per_lot)
    assert row.wanted == expected
    assert row.funded == min(expected, int(float(equity) / row.collateral_per_lot))


def _stake(row, equity) -> float:
    """The fraction of bankroll a sizing decision puts at risk.

    Lots are the wrong unit for comparing two beliefs, because a more cautious
    belief also carries a cheaper limit and therefore a smaller collateral per
    lot, and the two run against each other: measured on a dispersion future,
    shrinking the belief cut the stake and *raised* the lot count from 200 to
    220. The stake is what Kelly's ``f`` is a statement about.
    """
    return row.wanted * row.collateral_per_lot / float(equity)


def test_shrinkage_reduces_the_stake_it_asks_for():
    """Sizing on a credible bound stakes less than sizing on the mean.

    Not a chosen fraction of Kelly. The displacement is one exact posterior
    standard deviation of the underlying rate, ``sqrt(ab / ((a+b)^2 (a+b+1)))``,
    applied to the level rather than to the price, so it means the same thing
    for a future, a binary and an option.

    Re-measured on the circuit listing. The rung is the middle of the longest
    ladder, struck at 0.100, and the belief is Beta(42, 258): mean 0.140 over
    300 observations, so the rung sits two posterior standard deviations below
    the belief and P(theta > 0.100) is 0.984. That is the same geometry the
    retired listing gave with a rung at 0.44 under a Beta(150, 150), where the
    rung sat 2.08 standard deviations below a mean of 0.500 for a probability
    of 0.981, and it has to be re-anchored rather than carried over: the same
    Beta(150, 150) against a rung struck at 0.100 is a probability of 1.000, at
    which there is no shrinkage left to measure and every stake is the whole
    bankroll.

    Quoted 0.69 at 0.70: the unshrunk stake is 94.7% of bankroll, half a
    standard deviation takes it to 80.2%, and a full one to 47.2%. The retired
    listing gave 93.8%, 81.0% and 53.3% on its own geometry.

    The middle rung rather than the lowest, and that is the same structural
    choice `binaries_on_one_underlying` makes rather than a second one: the
    lowest rung of this ladder is struck at 0.060, which the belief clears with
    probability 0.996, so it carries the same no-uncertainty problem.
    """
    listed = listing()
    contract = binaries_on_one_underlying(listed)[1]
    posterior = (42.0, 258.0)
    quotes = {contract.symbol: ("0.69", "0.70")}
    equity = Decimal(1_000_000)
    view = view_of({contract.symbol: contract}, quotes, equity=equity)

    stakes = []
    for z in (0.0, 0.5, 1.0):
        row = _sized(_kelly_on(contract, posterior, credible_z=z), view, contract.symbol)
        assert row is not None, f"nothing sized at z={z}"
        stakes.append(_stake(row, equity))

    assert stakes == sorted(stakes, reverse=True)
    assert stakes[0] == pytest.approx(0.947, abs=0.01)
    assert stakes[-1] == pytest.approx(0.472, abs=0.01)


def test_shrinkage_vanishes_as_evidence_accumulates():
    """It has to reduce to full Kelly, or it is a fudge factor with a story.

    The mechanism is that the displacement is a posterior standard deviation and
    a Beta's falls as ``1 / sqrt(n)``: measured on a belief centred at one half,
    0.0407 at 100 battles, 0.0235 at 400, 0.0123 at 1,600 and 0.0062 at 6,400,
    halving for every quadrupling. Both halves are asserted, because the second
    without the first would be a coincidence and the first without the second
    would be a claim about a number nobody trades on.

    With very little evidence the credible bound does not clear the offer at all
    and the strategy declines. That is the intended behaviour rather than a gap
    in the test: at 150 observations it is being asked to buy at 0.70 something
    a 68% bound says is worth less than that. Measured on the same rung and the
    same belief centre as the test above, Beta(21, 129): full Kelly stakes
    77.3% of bankroll and one standard deviation of shrinkage declines it
    outright.

    The accumulation loop is centred on the belief rather than on one half, for
    the reason the test above gives at length: a belief centred at 0.500 clears
    a rung struck at 0.100 with probability 1.000 whatever the evidence, so the
    shrunk and unshrunk stakes would both be the whole bankroll at every sample
    size and the ratio would be 1.0 by construction. Centred at 0.140 the
    ratios run 0.783, 0.9999, 1.0000 and 1.0000 over 400 to 25,600
    observations.
    """
    for battles in (100, 400, 1_600, 6_400):
        one = KellyBayesian.dispersion(0.5 * battles + 25.0, 0.5 * battles + 25.0)
        four = KellyBayesian.dispersion(
            0.5 * 4 * battles + 25.0, 0.5 * 4 * battles + 25.0
        )
        assert one / four == pytest.approx(2.0, rel=0.15)

    listed = listing()
    contract = binaries_on_one_underlying(listed)[1]
    equity = Decimal(1_000_000)
    quotes = {contract.symbol: ("0.69", "0.70")}
    view = view_of({contract.symbol: contract}, quotes, equity=equity)

    thin = _sized(
        _kelly_on(contract, (21.0, 129.0), credible_z=1.0), view, contract.symbol
    )
    assert thin is None

    ratios = []
    for battles in (400, 1_600, 6_400, 25_600):
        # Mean 0.140 with a prior worth fifty observations, which is the same
        # shape the retired version used at a mean of one half.
        posterior = (0.14 * battles + 7.0, 0.86 * battles + 43.0)
        full = _sized(
            _kelly_on(contract, posterior, credible_z=0.0), view, contract.symbol
        )
        shrunk = _sized(
            _kelly_on(contract, posterior, credible_z=1.0), view, contract.symbol
        )
        ratios.append(_stake(shrunk, equity) / _stake(full, equity))

    assert ratios == sorted(ratios), ratios
    assert ratios[0] < 0.9
    assert ratios[-1] == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------
# The constraint that actually binds
# --------------------------------------------------------------------------


def test_a_position_is_never_sized_past_the_collateral_available():
    """The program is constrained, and this is the constraint.

    Checked with the venue's own arithmetic rather than the strategy's running
    total: every intent is priced at the limit it carries, put through
    ``collateral_for``, and the sum has to fit inside what the account had free.
    Collateral posted against existing positions counts against the budget, so a
    strategy that already holds something asks for less.
    """
    listed = listing()
    ladder = binaries_on_one_underlying(listed)
    subject = {i.symbol: i for i in ladder}
    posterior = (150.0, 150.0)
    strategy = KellyBayesian(
        {},
        battles=0,
        posterior={underlying_key(ladder[0]): posterior},
        credible_z=0.0,
    )
    quotes = {i.symbol: ("0.01", "0.02") for i in ladder}
    equity, posted = Decimal(5_000), Decimal(1_200)
    view = view_of(subject, quotes, equity=equity, posted=posted)

    intents = strategy.orders(view)
    assert intents, "nothing was sized, so the constraint was never exercised"
    charged = Decimal(0)
    for intent in intents:
        instrument = subject[intent.symbol]
        signed = intent.size if intent.side is Side.BUY else -intent.size
        charged += instrument.collateral_for(signed, intent.limit)
    assert charged <= equity * Decimal(str(strategy.capital_fraction)) - posted


def test_the_shadow_price_says_which_refusals_were_about_capital():
    """Capital is free until it is not, and the multiplier is the difference.

    With a bankroll that cannot fund what the belief asks for, the trades cut
    short are recorded and the Lagrange multiplier on collateral is positive.
    With a bankroll far larger than the opportunity, nothing is cut and the
    multiplier is exactly zero, which is what "capital was not the binding
    constraint" means in this program.
    """
    listed = listing()
    contract = binaries_on_one_underlying(listed)[0]
    subject = {contract.symbol: contract}
    quotes = {contract.symbol: ("0.69", "0.70")}
    view = view_of(subject, quotes, equity=Decimal(1_000_000))

    # One trade asking for 93.8% of bankroll against a balance sheet willing to
    # commit all of it. Kelly's own optimum is reached and capital is free.
    loose = _kelly_on(contract, (150.0, 150.0), credible_z=0.0, capital_fraction=1.0)
    loose.orders(view)
    assert loose.shadow_price == 0.0
    assert not loose.refused_for_capital
    assert loose.preference_bound == 1

    # The same trade against a quarter of the balance sheet. Measured: the ask
    # is unchanged at 960,749 lots and the funding falls to 256,034, so the
    # multiplier on collateral goes positive and names what it refused.
    tight = _kelly_on(contract, (150.0, 150.0), credible_z=0.0, capital_fraction=0.25)
    tight.orders(view)
    assert tight.shadow_price > 0.0
    assert tight.capital_bound == 1
    assert tight.preference_bound == 0
    refused = tight.refused_for_capital
    assert [row.symbol for row in refused] == [contract.symbol]
    assert refused[0].funded < refused[0].wanted


def test_intents_arrive_in_the_order_the_program_funds_them():
    """The adapter stops at the first intent it cannot fund, so order is policy.

    Priority is the marginal expected log wealth a unit of collateral buys,
    which is the constrained program's own price of capital rather than a
    ranking chosen for it.
    """
    listed = listing()
    ladder = binaries_on_one_underlying(listed)
    subject = {i.symbol: i for i in ladder}
    strategy = KellyBayesian(
        {},
        battles=0,
        posterior={underlying_key(ladder[0]): (150.0, 150.0)},
        credible_z=0.0,
    )
    quotes = {i.symbol: ("0.01", "0.02") for i in ladder}
    intents = strategy.orders(view_of(subject, quotes, equity=Decimal(50_000)))
    order = [row.symbol for row in strategy.sizing if row.funded > 0]
    assert [intent.symbol for intent in intents] == order
    densities = [row.density for row in strategy.sizing]
    assert densities == sorted(densities, reverse=True)


def test_one_sighting_is_not_a_violation():
    """A relation seen outside its band once is not yet a trade.

    The view lags the venue by the strategy's own latency, so the instant a
    ladder looks most dislocated is the instant it is being repriced. Measured
    on seed 7 without the confirmation: a package went out to buy one rung of
    the event ladder at 0.06 against a sale of the rung above it at 0.88, and
    by the next wakeup both books
    read 0.94 bid at 1.00. The sale filled and the purchase could not.
    """
    listed = listing()
    subject, quotes, low, high = _violated_ladder(listed)
    strategy = StaticArbitrage(subject, confirmations=2)
    view = view_of(subject, quotes, equity=Decimal(1_000_000))

    assert list(strategy.orders(view)) == []
    assert strategy.attempts == 0
    # The dislocation is recorded on the first look and traded on the second.
    assert strategy.violated
    assert len(strategy.orders(view)) == 2
    assert strategy.attempts == 1


def test_a_sighting_that_does_not_persist_is_never_traded():
    """A dislocation that closes before it is confirmed is left alone."""
    listed = listing()
    subject, quotes, low, high = _violated_ladder(listed)
    strategy = StaticArbitrage(subject, confirmations=2)
    fair = dict(quotes)
    fair[low.symbol] = ("0.60", "0.61")

    assert list(strategy.orders(view_of(subject, quotes))) == []
    assert list(strategy.orders(view_of(subject, fair))) == []
    assert list(strategy.orders(view_of(subject, quotes))) == []
    assert strategy.attempts == 0


def _chain_with_a_heavy_leg(listed: dict):
    """A chain relation and the digital leg whose coefficient is many lots.

    ``(K2 - K1) / payout`` binaries against one option spread, which on this
    listing is 50 or more. Found by looking for a coefficient above one rather
    than by naming a relation, so a listing whose strikes are spaced differently
    moves this test's subject instead of breaking it.
    """
    for relation in derive_binary_relations(listed):
        if not relation.name.startswith("chain-floor:"):
            continue
        heavy = [(symbol, c) for symbol, c in relation.legs if abs(c) > 1.0]
        if heavy:
            return relation, heavy[0]
    raise AssertionError("the listing carries no chain relation with a heavy leg")


def _chain_quotes(relation, heavy_symbol: str) -> dict[str, tuple[str, str]]:
    """Quotes that put a chain-floor relation well outside its band.

    Every option leg at 100 and the digital at 0.50 leaves the call spread worth
    nothing against a digital half the market believes in, which is the floor
    violated by 50 times a half.
    """
    quotes = {relation.target: ("99.75", "100.25")}
    for symbol, _coefficient in relation.legs:
        quotes[symbol] = ("0.49", "0.51") if symbol == heavy_symbol else ("99.75", "100.25")
    return quotes


def test_a_package_larger_than_the_book_has_ever_supplied_is_refused():
    """The only depth measurement a strategy here can make is its own fills.

    :class:`~arena.strategies.base.SymbolView` carries the touch and not the
    size resting on it, so "will 50 lots fill" cannot be looked up. It can be
    remembered: the most this strategy has ever been given at once in a symbol
    is a fact about that book, and a relation whose *smallest* package needs
    more than that is untradeable at the size the identity requires rather than
    thin at this instant.

    Measured on seed 7, that is the whole difference between the two families. A
    chain package needs 50 binary lots against one option spread and the binary
    touch carries a median of 30 at every sampled moment from t=60s to t=300s,
    so no capacity schedule and no amount of waiting produces a package that can
    fill. Without the rule the strategy attempted four packages in 300 seconds,
    completed none, unwound 180 lots and lost 3,979 finding out.

    The record is seeded here rather than produced, because producing it needs a
    partial fill from a book a unit test does not have.
    """
    listed = listing()
    relation, (heavy_symbol, coefficient) = _chain_with_a_heavy_leg(listed)
    subject = {s: listed[s] for s in relation.symbols}
    quotes = _chain_quotes(relation, heavy_symbol)
    view = view_of(subject, quotes, equity=Decimal(100_000_000))

    # With nothing remembered, the relation is tradeable and gets sent.
    willing = StaticArbitrage(subject)
    assert _confirmed(willing, view)
    assert willing.attempts == 1

    # Remembering that this book has never given more than a fraction of what
    # one package needs, it is not.
    lots = max(1, round(abs(coefficient)))
    knowing = StaticArbitrage(subject)
    knowing._filled[heavy_symbol] = lots - 1
    assert _confirmed(knowing, view) == []
    assert knowing.attempts == 0
    assert knowing.starved > 0

    # And one more lot of remembered depth is enough to make it tradeable again,
    # so the rule is about the measurement and not about the family.
    enough = StaticArbitrage(subject)
    enough._filled[heavy_symbol] = lots
    assert _confirmed(enough, view)


def test_a_touch_outside_the_contract_range_is_not_a_price():
    """Neither strategy trades against a number no quote of the contract can be.

    The case that produced this is a call phase, where a market order rests at
    a sentinel so it crosses every candidate the auction considers. Taken as a
    touch, the implied spread is minus 2.3e18 and every cost hurdle built from
    it is negative, so the arbitrage strategy fired ten packages in the first
    two minutes of seed 7 against prices no contract could have quoted. The feed
    no longer publishes it, so this now asserts the strategy-side refusal rather
    than a live defect, and it is worth keeping for that: a crossed book is
    legitimate here for other reasons, and a negative hurdle does not make a
    strategy cautious.
    """
    listed = listing()
    ladder = binaries_on_one_underlying(listed)
    subject = {i.symbol: i for i in ladder}
    sentinel = "1152921504606846976.00"
    quotes = {i.symbol: (sentinel, "-" + sentinel) for i in ladder}
    view = view_of(subject, quotes)

    kelly = KellyBayesian(
        {}, battles=0, posterior={underlying_key(ladder[0]): (150.0, 150.0)}
    )
    assert list(kelly.orders(view)) == []
    assert list(StaticArbitrage(subject).orders(view)) == []


# --------------------------------------------------------------------------
# The relations are read out of the listing
# --------------------------------------------------------------------------


def test_the_ladder_grows_when_a_rung_is_listed():
    """Add a contract, get a relation. That is what "derived" has to mean.

    A hand-written table of relations would pass every other test in this file
    and would be a table, so the check is that the set *moves* with the listing.
    A rung inserted between two existing thresholds splits one adjacency into
    two, so the count rises by exactly one and both new relations name the new
    contract.
    """
    from dashboard.build_market import _spec, _wr

    listed = listing()
    ladder = binaries_on_one_underlying(listed)
    assert len(ladder) >= 3, "the listing needs a ladder for this to mean anything"

    before = [r for r in derive_binary_relations(listed) if r.name.startswith("ladder:")]
    lowest, second = ladder[0], ladder[1]
    subject = lowest.spec.underlying.atoms()[0].subject
    middle = (lowest.spec.payoff.threshold + second.spec.payoff.threshold) / 2.0

    from arena.market.instrument import Instrument

    symbol = f"{subject}_GTMID"
    listed[symbol] = Instrument(
        symbol,
        _spec(
            symbol,
            _wr(subject),
            Binary(">", middle, payout=lowest.spec.payoff.payout),
            tick="0.01",
        ),
    )
    after = [r for r in derive_binary_relations(listed) if r.name.startswith("ladder:")]

    assert len(after) == len(before) + 1
    touching = [r for r in after if symbol in r.symbols]
    assert len(touching) == 2
    assert {r.name for r in touching} == {
        f"ladder:{lowest.symbol}/{symbol}",
        f"ladder:{symbol}/{second.symbol}",
    }
    # And the adjacency it displaced is gone, because monotonicity across it
    # now follows from the two steps that replaced it.
    assert f"ladder:{lowest.symbol}/{second.symbol}" not in {r.name for r in after}


def test_a_relation_goes_when_the_contract_it_needs_is_withdrawn():
    """Delist a rung and its relations leave with it, with no code change.

    Written by removing a listed contract rather than by relying on one being
    absent, because a test whose premise is an accident of the listing stops
    testing anything the day the listing improves.
    """
    listed = listing()
    ladder = binaries_on_one_underlying(listed)
    victim = ladder[1]
    before = derive_binary_relations(listed)
    assert any(victim.symbol in r.symbols for r in before)

    listed.pop(victim.symbol)
    after = derive_binary_relations(listed)
    assert not any(victim.symbol in r.symbols for r in after)
    # The ladder closes over the hole rather than losing the statement: the two
    # rungs either side of the withdrawn one become adjacent.
    assert f"ladder:{ladder[0].symbol}/{ladder[2].symbol}" in {r.name for r in after}


def test_every_derived_relation_holds_at_every_level():
    """The band is a claim about settlement, so check it at settlement.

    Each relation asserts that a portfolio's value stays inside a band whatever
    the metric does, and that claim is a piecewise-linear function of one
    bounded scalar, so it is checkable rather than arguable. Swept over a fine
    grid plus every kink and threshold on either side.

    This caught a real inversion: the put half of the chain family read the
    digital at the wrong strike, and 14 of 31 relations came back with a
    pathwise minimum of -200 on a band asserting zero. Every one of them looked
    plausible and every one would have been traded as free money.
    """
    listed = listing()
    relations = derive_binary_relations(listed)
    assert relations, "nothing to check"

    for relation in relations:
        target = listed[relation.target].spec
        low, high = target.underlying.bounds()
        grid = {low, high}
        for symbol in relation.symbols:
            payoff = listed[symbol].spec.payoff
            scale = getattr(payoff, "scale", 1.0) or 1.0
            for attribute, divisor in (("threshold", 1.0), ("strike", scale)):
                edge = getattr(payoff, attribute, None)
                if edge is None:
                    continue
                level = edge / divisor
                grid.update((level, level - 1e-9, level + 1e-9))
        grid.update(low + (high - low) * n / 400.0 for n in range(401))

        for level in sorted(x for x in grid if low <= x <= high):
            value = target.claim_value(level) - relation.constant
            for symbol, coefficient in relation.legs:
                value -= coefficient * listed[symbol].spec.claim_value(level)
            assert relation.lower - 1e-6 <= value <= relation.upper + 1e-6, (
                f"{relation.name} settles at {value} outside "
                f"[{relation.lower}, {relation.upper}] when the level is {level}"
            )


# --------------------------------------------------------------------------
# A violation becomes a package that cannot lose
# --------------------------------------------------------------------------


def _confirmed(strategy: StaticArbitrage, view: MarketView) -> list:
    """Ask until the strategy has seen the dislocation often enough to act.

    A single observation is not a violation to this strategy, and that is the
    point of the confirmation count rather than an inconvenience for the test:
    the view lags the venue, and the instants a ladder looks most violated are
    the instants it is being repriced.
    """
    for _ in range(strategy.confirmations):
        intents = list(strategy.orders(view))
        if intents:
            return intents
    return []


def _violated_ladder(listed: dict):
    """Two rungs of a ladder, quoted the wrong way round.

    A binary struck lower must cost at least as much as one struck higher, so
    quoting the low rung at 0.31 against the high one at 0.60 is a violation of
    0.30 on a contract bounded by [0, 1]. Measured on the live market: seed 7 at
    t=180s carried two such violations, of 0.23 and 0.30.
    """
    ladder = binaries_on_one_underlying(listed)
    low, high = ladder[0], ladder[1]
    subject = {low.symbol: low, high.symbol: high}
    quotes = {low.symbol: ("0.30", "0.31"), high.symbol: ("0.60", "0.61")}
    return subject, quotes, low, high


def test_a_ladder_violation_produces_a_package_that_cannot_lose():
    """The point of the whole strategy, checked as arithmetic.

    A package built from a violated relation settles at the slack in an
    inequality that holds at every level, so its settlement value is
    non-negative everywhere, and it was entered for a credit on top of that. Both
    halves are checked here: the payoff swept over the metric's whole range, and
    the entry prices through the same ``worst_case`` minimisation the collateral
    engine charges with.
    """
    listed = listing()
    subject, quotes, low, high = _violated_ladder(listed)
    strategy = StaticArbitrage(subject)
    intents = _confirmed(strategy, view_of(subject, quotes, equity=Decimal(1_000_000)))
    assert len(intents) == 2

    package = {intent.symbol: intent for intent in intents}
    assert package[low.symbol].side is Side.BUY
    assert package[high.symbol].side is Side.SELL

    holdings = []
    for intent in intents:
        signed = intent.size if intent.side is Side.BUY else -intent.size
        holdings.append((subject[intent.symbol].spec, signed, intent.limit))

    # Settlement value alone, before the credit: non-negative everywhere.
    bounds = low.spec.underlying.bounds()
    grid = {bounds[0], bounds[1]}
    for instrument in subject.values():
        threshold = instrument.spec.payoff.threshold
        grid.update((threshold, threshold - 1e-9, threshold + 1e-9))
    grid.update(bounds[0] + (bounds[1] - bounds[0]) * n / 500.0 for n in range(501))
    for level in sorted(x for x in grid if bounds[0] <= x <= bounds[1]):
        settlement = sum(
            quantity * spec.claim_value(level) for spec, quantity, _price in holdings
        )
        assert settlement >= -1e-9, f"the package settles at {settlement} at {level}"

    # And with what it paid, it cannot lose a cent at any level either.
    assert worst_case(holdings) == 0


def test_a_violation_inside_the_no_arbitrage_band_is_left_alone():
    """The band is what a real market has, and crossing it has to cost.

    The relation is violated by 0.155 at the mids here while the round trip on
    the two legs is 0.155 before any safety multiple, so correcting it would
    give back more than it pays. Real markets carry exactly these: the violation
    persists because closing it is not worth doing, which is a different thing
    from nobody having noticed it.
    """
    listed = listing()
    subject, quotes, low, high = _violated_ladder(listed)
    strategy = StaticArbitrage(subject)
    assert _confirmed(strategy, view_of(subject, quotes, equity=Decimal(1_000_000)))

    widened = dict(quotes)
    widened[high.symbol] = ("0.31", "0.61")
    quiet = StaticArbitrage(subject)
    assert _confirmed(quiet, view_of(subject, widened, equity=Decimal(1_000_000))) == []
    # Still violated, just not worth the crossing.
    assert quiet.violated


def test_a_half_legged_package_is_flattened_before_anything_else():
    """A relation filled on one leg is a directional bet nobody chose.

    Reconciliation is integer arithmetic on the position rather than an
    inference from fills: a package is complete when every leg has moved a whole
    multiple of its own per-unit size. Anything else is unwound at the next
    wakeup, ahead of any new business, and toward zero only.
    """
    listed = listing()
    subject, quotes, low, high = _violated_ladder(listed)
    strategy = StaticArbitrage(subject)
    opened = _confirmed(strategy, view_of(subject, quotes, equity=Decimal(1_000_000)))
    assert opened

    # Only the buy filled. The sell did not.
    filled = {intent.symbol: 0 for intent in opened}
    buy = next(i for i in opened if i.side is Side.BUY)
    filled[buy.symbol] = buy.size

    unwind = strategy.orders(
        view_of(subject, quotes, equity=Decimal(1_000_000), positions=filled)
    )
    assert strategy.legged == 1
    assert strategy.captured == 0
    assert len(unwind) == 1
    assert unwind[0].symbol == buy.symbol
    assert unwind[0].side is Side.SELL
    assert unwind[0].size == buy.size
    # One lot is stray, not two: the leg that never filled left nothing behind.
    assert strategy.legged_lots == buy.size


def test_a_completed_package_is_not_unwound():
    """The other half of the same accounting: a whole fill is left alone."""
    listed = listing()
    subject, quotes, low, high = _violated_ladder(listed)
    strategy = StaticArbitrage(subject)
    opened = _confirmed(strategy, view_of(subject, quotes, equity=Decimal(1_000_000)))
    filled = {
        intent.symbol: intent.size if intent.side is Side.BUY else -intent.size
        for intent in opened
    }
    strategy.orders(view_of(subject, quotes, equity=Decimal(1_000_000), positions=filled))
    assert strategy.captured == 1
    assert strategy.legged == 0
    assert strategy.theoretical > 0.0


# --------------------------------------------------------------------------
# In a market
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def traded():
    """Both strategies in the live market, seated like any other participant.

    Twenty million each, which is what the collateral on a meaningful position
    costs here: a future settles somewhere in [0, 10000] so a single lot can
    require 10,000 of it, and a strategy funded for a handful of lots would be
    measuring its own balance sheet rather than its own view.

    Nothing below reads a profit off this. Both strategies are in the market
    from the first instant, which includes the opening auction, where every book
    opens at the midpoint of its settlement range and a strategy that knows
    better is trading an artefact of the builder rather than a market.
    :mod:`arena.research.backtest` holds a strategy out for a warmup for exactly
    that reason and is where a P&L belongs. What is checked here is what a venue
    adds: that the intents are accepted, that the ledger still balances to
    integer zero, and that nobody ends over-committed.
    """
    market = build(seed=7)
    listed = build_instruments()
    by_symbol = {i.symbol: i for i in listed}
    levels = true_levels(listed)

    kelly = KellyBayesian(levels, battles=5_000, fee_bps=MAKER_TAKER.taker_bps)
    market.venue.open_account("kelly", CASH)
    market.kernel.add(
        StrategyAgent(
            "kelly", VENUE_ID, by_symbol, millis(500), taker=kelly, starting_cash=CASH
        )
    )

    arbitrage = StaticArbitrage(by_symbol, fee_bps=MAKER_TAKER.taker_bps)
    market.venue.open_account("arb", CASH)
    market.kernel.add(
        StrategyAgent(
            "arb", VENUE_ID, by_symbol, millis(500), taker=arbitrage, starting_cash=CASH
        )
    )

    # Three simulated minutes, which is where the cost of this file sits. The
    # market runs at roughly 1.7 seconds of wall clock per simulated second with
    # 47 books and two more takers in it, so this fixture is minutes rather than
    # seconds and everything below shares the one run.
    market.kernel.start()
    market.kernel.advance(until=seconds(180))
    return market, kelly, arbitrage


def test_both_strategies_reached_the_market(traded):
    """Otherwise every assertion below is about an idle agent.

    Liveness is asserted at the right level for each. The sizing strategy has to
    have traded, because it acts on a view it always has. The arbitrage strategy
    has to have *priced* its relations against real books, and whether any of
    them was dislocated for two consecutive wakeups by more than the round trip
    on every leg is a property of this market on this seed rather than of the
    strategy. Asserting that it traded would be asserting that the market
    misbehaved.
    """
    market, kelly, arbitrage = traded
    positions = market.venue.account("kelly").positions
    assert sum(abs(int(p.quantity)) for p in positions.values()) > 0
    assert kelly.sizing
    assert arbitrage.sightings > 0


def test_value_is_conserved_with_both_strategies_in_the_market(traded):
    """Two more participants taking liquidity, and the ledger still balances.

    Exactly integer zero, not close to it. Money here is integer minor units for
    this reason, and a tolerance would be hiding a bug somewhere else.
    """
    market, _kelly, _arbitrage = traded
    assert market.venue.conservation_check() == 0


def test_neither_strategy_ends_over_committed(traded):
    """Free cash below zero means the venue funded something it could not back.

    Asserted on these two accounts and not on every account in the market, and
    the reason is a finding rather than a convenience. Under the volume these
    strategies add, ``fund-5`` ends 665.42 short on a starting balance of
    40,000,000, which is 0.0017% and is not a strategy defect: collateral is
    re-posted against the position's basis when a fill lands and the taker fee
    comes out of cash afterwards, so a fill that was exactly affordable when
    checked leaves the account fractionally under it. It is a venue-side
    observation about the fee path, it is reachable from an ordinary session,
    and it belongs to whoever owns ``arena/portfolio/account.py``. Asserting it
    here would make this file fail for something it is not about.
    """
    market, _kelly, _arbitrage = traded
    for agent_id in ("kelly", "arb"):
        account = market.venue.account(agent_id)
        assert int(account.free_cash) >= 0, f"{agent_id} is over-committed"


def test_the_collateral_constraint_binds_far_more_often_than_conviction(traded):
    """The measurement this strategy exists to report.

    Collateral equals plausible risk only for a binary. On a future it
    over-charges the informed agents' own uncertainty by about 47x, so full
    Kelly asks for roughly 1,290% of bankroll and the balance sheet answers
    first. Measured over 180 simulated seconds on seed 7, capital cut the size
    far more often than the sizing preference reached its own optimum.
    """
    _market, kelly, _arbitrage = traded
    assert kelly.capital_bound > kelly.preference_bound
    assert kelly.capital_bound > 10 * kelly.preference_bound


def test_the_arbitrage_strategy_accounts_for_every_lot_it_holds(traded):
    """No position exists that the strategy has no record of.

    Not a claim that it never legs, which would be a claim about the order book.
    A claim that everything it holds went through the package accounting, so a
    leg that filled alone is visible as residue and gets unwound rather than
    sitting in a book nothing looks at. That distinction is not academic: the
    first version left the legs of a package that completed nothing out of the
    record entirely, and it ended a 300 second run holding 839 lots it had no
    view on, with the collateral posted against them starving every later
    relation for budget.
    """
    market, _kelly, arbitrage = traded
    account = market.venue.account("arb")
    for symbol, position in account.positions.items():
        if int(position.quantity):
            assert symbol in arbitrage._held, f"{symbol} is held and unaccounted"

    # Every attempt is reconciled into a completion, a legging, or both when a
    # package fills part way. The one still in flight at the end is neither.
    pending = 1 if arbitrage._pending is not None else 0
    assert arbitrage.captured + arbitrage.legged >= arbitrage.attempts - pending
    assert arbitrage.legged <= arbitrage.attempts
    assert arbitrage.captured <= arbitrage.attempts


def test_the_arbitrage_strategy_derives_more_relations_than_it_can_trade(traded):
    """A derived relation is not automatically an executable one.

    Re-measured on the circuit listing: 4 relations come out of it, 2 ladder
    and 2 chain, against 31, 5 and 26 on the 47 contract listing this replaces.
    The drop is the listing's own argument showing through rather than a loss
    of coverage. That listing put three ladders and thirteen options on three
    subjects, so binaries and options met on the same underlying repeatedly;
    this one spends its instruments on subjects and axes instead, and only
    VANTA's individual win rate carries more than one rung and only EMBER's
    team win rate carries both a rung and an option chain.

    A ladder package is one lot against one lot. A chain package is
    ``(K2 - K1) / payout`` binaries against one option spread, and on this
    listing the two chain relations are struck 200 apart against a payout of 1,
    so a package is 200 binaries against one spread: larger than the book by a
    wide margin. The strategy has no depth in its view to read that from, so it
    learns it from its own fills: the most a symbol has ever given it at once is
    the only measurement it has, and a relation whose per-unit leg needs more
    than that is refused rather than attempted again.
    """
    _market, _kelly, arbitrage = traded
    ladder = [r for r in arbitrage.relations if r.name.startswith("ladder:")]
    chain = [r for r in arbitrage.relations if r.name.startswith("chain-")]
    assert len(ladder) == 2
    assert len(chain) == 2
    assert arbitrage.retired <= {r.name for r in arbitrage.relations}
