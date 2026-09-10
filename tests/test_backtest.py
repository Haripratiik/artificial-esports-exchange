"""What the backtest harness is allowed to claim, and what it must refuse to.

Two halves, for two different failure modes. The statistics are pure functions
of a return series, so they are tested against series whose answer is known by
arithmetic rather than by running anything: if ``SE(SR)`` is wrong it is wrong
on four numbers and there is no market to blame. The rest runs real markets,
because the things that go wrong there go wrong in the wiring: a warmup that
does not exclude what it claims to, a pairing that is not paired, a digest that
is stable because it is stable against nothing.

Every market test here is short on purpose. The opening call clears at ten
simulated seconds, so a warmup of fourteen is the least that can exclude it, and
the runs are sized just above that. Their P&L figures are therefore not evidence
about any strategy and nothing here asserts on one.
"""

from __future__ import annotations

import math
import statistics
from decimal import Decimal

import pytest

from arena.exchange.types import Side
from arena.research.attribution import Decomposition
from arena.research.backtest import (
    BacktestConfig,
    Evaluation,
    RunResult,
    backtest,
    compare,
    compare_many,
    deflated_sharpe,
    expected_max_drawdown,
    implied_hit_rate,
    manifest,
    min_track_record_length,
    probabilistic_sharpe,
    sharpe_ratio,
    sharpe_standard_error,
    sortino_ratio,
    summarise,
    verify_reproducible,
)
from arena.strategies.base import Quote, Take, TwoSided

NORMAL = statistics.NormalDist()

# The opening call clears at ten simulated seconds, so this is the shortest
# configuration that can honestly say the auction is behind it. Two-second
# sampling because the attribution sampler reads every one of the market's 47
# books on every call, measured at 6.9ms a book, and at a half-second grid the
# watching costs more wall time than the market being watched.
FAST = BacktestConfig(
    until=26.0,
    warmup=14.0,
    return_interval=2.0,
    sample_interval=2.0,
    symbols=("EMBER_OBJECTIVE_WR",),
)


# --------------------------------------------------------------------------
# Strategies written here rather than imported
#
# arena.strategies.making and arena.strategies.taking are being written
# concurrently and may not import cleanly. Nothing in this file needs them: what
# is under test is the harness, and the cheapest strategy that exercises it is
# the one whose behaviour is obvious by inspection.
# --------------------------------------------------------------------------


class Flat:
    """A constant half-spread around the reference, skewed by inventory.

    The skew is a ``Decimal`` because the reference is one, and a float times a
    ``Decimal`` is a ``TypeError`` rather than a coercion. Modelling in floats is
    allowed and is what the literature's formulas are written in, but the
    arithmetic has to be done on one side of the boundary or the other.
    """

    def __init__(self, half: int = 8, size: int = 4, skew: str = "0.25") -> None:
        self.half, self.size, self.skew = half, size, Decimal(skew)

    def symbols(self, view):
        return view.symbols

    def quote(self, view, symbol):
        row = view[symbol]
        reference = row.reference
        if reference is None:
            return TwoSided()
        tick = row.instrument.tick_size
        centre = reference - row.position * self.skew * tick
        return TwoSided(
            bid=Quote(centre - self.half * tick, self.size),
            ask=Quote(centre + self.half * tick, self.size),
        )


class Idle:
    """Quotes nothing, ever. The zero of the strategy space."""

    def symbols(self, view):
        return ()

    def quote(self, view, symbol):
        return TwoSided()


class Nibbler:
    """Lifts one lot every wake, so that a taker path is exercised at all."""

    def orders(self, view):
        for row in view:
            if row.best_ask is not None:
                return (Take(row.symbol, Side.BUY, 1, row.best_ask),)
        return ()


# --------------------------------------------------------------------------
# The statistics, on series whose answers are arithmetic
# --------------------------------------------------------------------------


def two_point(n: int, mean: float, half: float) -> list[float]:
    """``n`` returns, half at ``mean+half`` and half at ``mean-half``.

    Every moment is exact: the population variance is ``half**2``, the sample
    variance is ``half**2 * n/(n-1)``, the skew is zero and the raw kurtosis is
    one. That makes it the right series for testing estimators, because a
    disagreement cannot be a floating-point accident in the fixture.
    """
    assert n % 2 == 0
    return [mean + half] * (n // 2) + [mean - half] * (n // 2)


def test_the_sharpe_is_the_sample_ratio_and_nothing_is_annualised():
    """No sqrt(252) anywhere: a period here is simulated seconds, not a day."""
    series = two_point(40, mean=0.01, half=0.02)
    expected = 0.01 / (0.02 * math.sqrt(40 / 39))
    assert sharpe_ratio(series) == pytest.approx(expected, rel=1e-12)


def test_the_standard_error_of_a_sharpe_is_los_expression():
    """``sqrt((1 + SR^2/2)/n)``, and the general form must reduce to it.

    The two are the same expression with skew 0 and raw kurtosis 3 substituted
    in, so a mismatch here means one of them has drifted from the other, which
    is how a normal-case shortcut stops agreeing with the case it is a shortcut
    for.
    """
    for sharpe, n in ((0.0, 50), (0.5, 50), (1.0, 50), (2.0, 250)):
        gaussian = math.sqrt((1.0 + sharpe * sharpe / 2.0) / n)
        assert sharpe_standard_error(sharpe, n) == pytest.approx(gaussian, rel=1e-12)
        assert sharpe_standard_error(sharpe, n, 0.0, 3.0) == pytest.approx(
            gaussian, rel=1e-12
        )


def test_negative_skew_and_fat_tails_widen_the_error_rather_than_narrowing_it():
    """The direction is the whole point of carrying the moments at all.

    A left-skewed, fat-tailed series is the shape a short-volatility strategy
    produces, and it is exactly the shape whose Sharpe is least trustworthy. At
    SR 1.0 over 100 periods, skew -3 and raw kurtosis 10 take the standard error
    from 0.1225 to 0.2500, a factor of 2.04, so the same ratio is twice as
    uncertain and nothing about the point estimate says so.
    """
    plain = sharpe_standard_error(1.0, 100, 0.0, 3.0)
    nasty = sharpe_standard_error(1.0, 100, -3.0, 10.0)
    assert plain == pytest.approx(math.sqrt(1.5 / 100), rel=1e-12)
    assert nasty == pytest.approx(math.sqrt(6.25 / 100), rel=1e-12)
    assert nasty / plain == pytest.approx(2.0412414523193148, rel=1e-9)


def test_the_relative_error_of_a_sharpe_floors_at_one_over_root_two_n():
    """However good the strategy is, the measurement cannot beat this.

    ``SE/SR = sqrt(1/(n SR^2) + 1/(2n))``, which decreases in SR and tends to
    ``1/sqrt(2n)`` from above. At the 48 periods a short session yields the
    floor is 10.2%, so nothing measured over one of these sessions can be known
    to better than a tenth of itself.
    """
    n = 48
    floor = 1.0 / math.sqrt(2 * n)
    assert floor == pytest.approx(0.10206207261596575, rel=1e-12)
    previous = math.inf
    for sharpe in (0.25, 0.5, 1.0, 2.0, 4.0, 16.0, 256.0):
        relative = sharpe_standard_error(sharpe, n) / sharpe
        assert relative > floor
        assert relative < previous
        previous = relative
    assert previous == pytest.approx(floor, rel=1e-4)


def test_the_probabilistic_sharpe_is_the_normal_cdf_of_the_standardised_ratio():
    """Hand-computed: SR 0.1 over 101 periods is Phi(1/sqrt(1.005))."""
    got = probabilistic_sharpe(0.1, 101, 0.0, 3.0)
    assert got == pytest.approx(NORMAL.cdf(1.0 / math.sqrt(1.005)), rel=1e-12)
    assert got == pytest.approx(0.8407413278013518, rel=1e-12)


def test_the_probabilistic_sharpe_rises_with_the_sample_at_a_fixed_ratio():
    """Same ratio, longer record, more belief. The point of the statistic."""
    values = [probabilistic_sharpe(0.2, n, 0.0, 3.0) for n in (10, 50, 200, 1000)]
    assert values == sorted(values)
    assert values[0] < 0.75 < values[-1]


def test_the_deflated_sharpe_falls_as_more_variants_are_tried():
    """The reason a testbed has to be told how many things its user tried.

    With the dispersion of the trials supplied, a ratio of 0.30 over 250 periods
    at a trial variance of 0.006 scores 0.974 after 46 trials and 0.946 after
    100. That is the shape of Bailey and Lopez de Prado's worked example: one
    track record, unchanged, passing at 46 and failing at 100 on nothing but how
    many other things were tried beside it.
    """
    variance = 0.006
    values = [
        deflated_sharpe(0.30, 250, trials, 0.0, 3.0, trial_variance=variance)
        for trials in (1, 2, 10, 46, 100, 1000)
    ]
    assert values == sorted(values, reverse=True)
    at_46, at_100 = values[3], values[4]
    assert at_46 == pytest.approx(0.9743, abs=5e-4)
    assert at_100 == pytest.approx(0.9458, abs=5e-4)
    assert at_46 > 0.95 > at_100


def test_the_deflated_sharpe_at_one_trial_is_the_undeflated_one():
    """One variant has no maximum to take, so there is nothing to deflate.

    The order-statistic expression diverges at N=1 because ``Z^-1(0)`` is minus
    infinity; the expected maximum of a single zero-mean draw is zero, and that
    is the value used rather than the limit of the formula.
    """
    assert deflated_sharpe(0.5, 100, 1, 0.0, 3.0) == pytest.approx(
        probabilistic_sharpe(0.5, 100, 0.0, 3.0), rel=1e-12
    )


def test_the_minimum_track_record_length_is_the_published_expression():
    """``1 + [1 - g3*SR + (g4-1)/4*SR^2] * (Z_a/(SR - SR*))^2``, by hand.

    At SR 0.2 with normal moments and a 95% level this is 70.0 periods, so a
    session that produced 48 of them has not established it whatever the point
    estimate says.
    """
    z = NORMAL.inv_cdf(0.95)
    expected = 1.0 + 1.02 * (z / 0.2) ** 2
    assert min_track_record_length(0.2, 0.0, 3.0) == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(69.99135807943297, rel=1e-12)


def test_a_sharpe_that_does_not_beat_the_benchmark_has_no_track_record_length():
    """Infinite rather than large: no amount of sample establishes a loss."""
    assert min_track_record_length(-0.4, 0.0, 3.0) == math.inf
    assert min_track_record_length(0.3, 0.0, 3.0, benchmark=0.3) == math.inf


def test_sortino_divides_by_every_period_not_only_the_losing_ones():
    """The common implementation is wrong and is wrong in the flattering direction.

    Ninety-nine periods at +0.01 and one at -0.10: the downside deviation over
    all hundred is 0.010, and over the single losing period it is 0.100, which
    reports a Sortino ten times larger for a series that took exactly the same
    risk.
    """
    series = [0.01] * 99 + [-0.10]
    mean = sum(series) / 100
    over_all = math.sqrt(0.10 ** 2 / 100)
    over_losers = math.sqrt(0.10 ** 2 / 1)
    assert sortino_ratio(series) == pytest.approx(mean / over_all, rel=1e-12)
    assert over_losers == pytest.approx(10 * over_all, rel=1e-12)
    assert sortino_ratio(series) < mean / over_all * 1.0000001


def test_a_hit_rate_means_opposite_things_at_different_sample_sizes():
    """``p = 0.5*(1 + sqrt(theta^2/(n + theta^2)))``.

    A per-period Sharpe of 2 needs 53.2% of periods to win over a thousand of
    them and 63.6% over fifty. The same 55% is therefore excellent in one column
    and poor in the other, which is why nothing here reports a hit rate without
    its ``n`` attached.
    """
    assert implied_hit_rate(2.0, 1000) == pytest.approx(0.5315597201548902, rel=1e-12)
    assert implied_hit_rate(2.0, 50) == pytest.approx(0.6360827634879543, rel=1e-12)


def test_the_expected_drawdown_of_no_edge_grows_with_the_horizon():
    """Which is why drawdown is never pooled across runs of different lengths.

    ``E[MDD] = 1.2533*sigma*sqrt(T)``. Doubling the horizon multiplies the
    expected drawdown of a strategy with no edge at all by 1.414, so ranking two
    strategies on realised drawdown ranks them on how long they were run.
    """
    short = expected_max_drawdown(1.0, 100)
    long = expected_max_drawdown(1.0, 200)
    assert short == pytest.approx(1.2533141373155003 * 10.0, rel=1e-9)
    assert long / short == pytest.approx(math.sqrt(2), rel=1e-12)


def test_a_series_with_no_dispersion_reports_no_sharpe_rather_than_a_huge_one():
    """Every period identical is a division by zero, not an infinite ratio."""
    stats = summarise([0.001] * 20)
    assert math.isnan(stats.sharpe)
    assert not stats.supported
    assert any("division by zero" in note for note in stats.notes)
    assert "undefined" in stats.describe()


def test_a_sample_too_short_for_its_own_sharpe_says_so_instead_of_boasting():
    """The honesty requirement, stated as a test.

    Twenty periods alternating +0.024 and -0.016 give a Sharpe of 0.195 with a
    standard error of 0.224 and a minimum track record of 72 periods. The point
    estimate is positive, its own error bar covers zero three times over, and
    the result object has to be the thing that says so rather than printing
    0.195 and leaving it there.
    """
    stats = summarise(two_point(20, mean=0.004, half=0.02))
    assert stats.sharpe == pytest.approx(0.1949358868961793, rel=1e-9)
    assert stats.sharpe_error == pytest.approx(0.22360679774997896, rel=1e-9)
    assert stats.min_track_record == pytest.approx(72.19851194987922, rel=1e-9)
    assert stats.n < stats.min_track_record
    assert not stats.supported
    assert any("minimum track record" in note for note in stats.notes)
    assert "NOT supported" in stats.describe()


# --------------------------------------------------------------------------
# Reporting rules that need no market
# --------------------------------------------------------------------------


def _run(seed: int, pnl: int, periods: int) -> RunResult:
    equity = tuple(20_000_000_000_000 + pnl * i // periods for i in range(periods + 1))
    returns = tuple(
        (equity[i] - equity[i - 1]) / equity[0] for i in range(1, len(equity))
    )
    return RunResult(
        seed=seed,
        agent_id="strategy",
        pnl=pnl,
        opening_equity=equity[0],
        closing_equity=equity[-1],
        warmup_pnl=0,
        returns=returns,
        equity=equity,
        decomposition=Decomposition("strategy", 1),
        adverse_curve={},
        informed_share=0.0,
        flow_imbalance={},
        lots=0,
        fills=0,
        aggressive_lots=0,
        notional=0,
        conservation=0,
        symbols_traded=(),
    )


def _evaluation(name: str, runs: tuple[RunResult, ...]) -> Evaluation:
    pooled: list[float] = []
    for run in runs:
        pooled.extend(run.returns)
    return Evaluation(name, FAST, tuple(r.seed for r in runs), runs, summarise(pooled))


def test_drawdown_is_withheld_when_the_runs_do_not_share_a_horizon():
    """Because the figure is a fact about the horizon, not about the strategy."""
    same = _evaluation("same", (_run(0, 1_000, 10), _run(1, -2_000, 10)))
    mixed = _evaluation("mixed", (_run(0, 1_000, 10), _run(1, -2_000, 20)))
    assert same.drawdown is not None
    assert mixed.drawdown is None
    assert "do not share a horizon" in str(mixed)


def test_a_paired_comparison_will_not_pair_arms_that_are_not_paired():
    """Different seeds, or different agent ids, and the pairing is a fiction.

    The agent id matters as much as the seed here: the kernel derives an
    agent's random stream and its latency jitter from the seed and the id
    together, so two arms under two ids share no draws at all and their
    covariance would be measuring nothing.
    """
    from arena.research.backtest import pair

    a = _evaluation("a", (_run(0, 10, 4), _run(1, 20, 4)))
    b = _evaluation("b", (_run(0, 10, 4), _run(2, 20, 4)))
    with pytest.raises(ValueError, match="same seeds"):
        pair(a, b)

    c = Evaluation("c", FAST.evolve(agent_id="other"), a.seeds, a.runs, a.stats)
    with pytest.raises(ValueError, match="different agent ids"):
        pair(a, c)


def test_a_warmup_longer_than_the_run_is_refused_at_configuration_time():
    """A guard that fires before a forty minute run rather than after it."""
    with pytest.raises(ValueError, match="leaves nothing"):
        BacktestConfig(until=30.0, warmup=30.0)
    with pytest.raises(ValueError, match="whole number"):
        BacktestConfig(return_interval=3.0, sample_interval=2.0)


def test_a_config_survives_the_round_trip_through_its_manifest():
    """`replay` rebuilds the config from JSON, so a field that does not survive
    the round trip is a field that silently reverts to its default on the replay
    and makes the digest disagree for a reason that is not the simulation.
    """
    for config in (
        BacktestConfig(),
        BacktestConfig(
            until=30.0,
            warmup=14.0,
            return_interval=2.0,
            sample_interval=2.0,
            symbols=("EMBER_OBJECTIVE_WR", "BASTION_SOLO_EQ"),
            trade_during_warmup=True,
            market=(("makers", 1), ("surface", False)),
        ),
    ):
        assert BacktestConfig.from_dict(config.to_dict()) == config


def test_a_horizon_shorter_than_the_sampling_grid_is_dropped_and_said_so():
    """It would be measuring the grid. The alternative is reporting it anyway."""
    config = BacktestConfig(sample_interval=1.0)
    collected = config.collected_horizons
    assert all(h >= 1_000_000_000 for h in collected)
    assert 100_000_000 not in collected


# --------------------------------------------------------------------------
# The market
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_a_strategy_that_does_nothing_produces_a_zero_result_not_an_exception():
    """The zero of the space has to be representable.

    Every ratio here divides by something a silent strategy makes zero, so the
    honest output is a P&L of exactly zero and a Sharpe that says it does not
    exist. A harness that raises on this cannot be used to find out that a
    strategy stopped quoting, which is the most common way one fails.
    """
    result = backtest(Idle(), [3], config=FAST, name="idle")
    run = result.runs[0]
    assert run.pnl == 0
    assert run.warmup_pnl == 0
    assert run.lots == 0
    assert run.conservation == 0
    assert run.decomposition.fills == 0
    assert math.isnan(result.stats.sharpe)
    assert not result.stats.supported
    assert any("never traded" in note for note in result.notes)
    assert "n/a" in str(result)


@pytest.mark.slow
def test_the_warmup_holds_the_strategy_out_of_the_opening_auction():
    """The measurement that decides whether any other number here means anything.

    Every book opens at the midpoint of its settlement range, so the opening
    call is a dislocation of the builder's making rather than a market.
    Re-measured on the circuit listing, on seed 3, quoting all 50 books from
    t=0: the fourteen warmup seconds book -1,356,601, which is 6.8% of the
    strategy's 20,000,000 of capital and none of it earned. Carrying that
    inventory into the measured window takes the post-warmup P&L from
    -2,861,035 to -2,563,057, a difference of 297,978 or 10.4% of the measured
    number, so excluding the warmup from the statistics without also holding
    the strategy out of the auction would leave the artefact inside the
    measurement.

    **Both halves of that moved, and the direction is the interesting one.** On
    the 47 book listing this replaces, the warmup booked a *profit* of 2,739,688
    against the same capital, 13.7%, and carrying it took the measured P&L from
    -2,044,297 to -9,028,498, more than trebling it. The artefact is now
    smaller and its sign is the other way round, which says the size of the
    opening dislocation is a property of what is listed rather than of the
    auction: this listing opens 50 books at their range midpoints and the team
    win rates sit near theirs, where the previous one opened 47 books whose
    midpoints were mostly nowhere near.

    So the assertion is on the share of the measured P&L the artefact moves,
    against a measured 10.4%, rather than on it exceeding the measurement
    outright as it did when the artefact was three times the size of what it
    contaminated.

    Two runs, one seed, identical windows, differing only in whether the
    strategy was allowed to quote before the window opened.
    """
    everything = FAST.evolve(symbols=None)
    held = backtest(Flat(), [3], config=everything, name="held-out").runs[0]
    control = backtest(
        Flat(),
        [3],
        config=everything.evolve(trade_during_warmup=True),
        name="through-the-open",
    ).runs[0]

    capital = held.opening_equity
    assert held.warmup_pnl == 0, "the quarantine let something through"
    assert held.lots > 0, "the strategy did not trade in the measured window either"
    assert abs(control.warmup_pnl) > 0.02 * capital, (control.warmup_pnl, capital)
    assert abs(control.pnl - held.pnl) > 0.05 * abs(held.pnl), (
        control.pnl,
        held.pnl,
    )


@pytest.mark.slow
def test_identical_strategies_pair_perfectly_and_report_no_difference():
    """The strongest available check that the pairing is real.

    The simulation is bit-identical for a seed and the kernel seeds each agent's
    random stream from the seed and the agent's own id, so two arms running the
    same strategy under the same id must produce the same trace. Then every
    paired difference is exactly zero, ``Cov(A,B)`` equals ``Var(A)``, and
    ``Var(A-B) = Var(A) + Var(B) - 2Cov(A,B)`` collapses to zero. A covariance
    below that is the signal that the two arms were not sharing their draws.
    """
    result = compare(Flat(), Flat(), seeds=[3, 4, 5], config=FAST, names=("a", "b"))

    assert result.values_a == result.values_b
    assert result.mean_difference == 0.0
    assert result.variance_paired == 0.0
    assert result.variance_unpaired > 0.0, "the seeds produced no dispersion to pair"
    assert result.covariance == pytest.approx(result.variance_unpaired / 2)
    assert result.correlation == pytest.approx(1.0)
    assert result.variance_reduction == pytest.approx(1.0)
    assert result.pairing_helped
    assert not result.significant
    assert result.verdict.startswith("identical")


@pytest.mark.slow
def test_the_manifest_digest_is_stable_across_runs_and_moves_with_the_config():
    """Determinism is a claim until something re-runs it and compares.

    Replayed from the manifest alone, the run must land on a byte-identical
    digest. `verify_reproducible` is the check and it is deliberately expensive:
    it rebuilds the config out of the manifest's own JSON and runs the whole
    thing again rather than comparing a stored hash to itself. One field of the
    config changed, or one seed, and the digest has to move, because a manifest
    that did not would let a run be reproduced into the wrong answer without
    anything noticing.
    """
    first = manifest(backtest(Flat(), [3], config=FAST, name="fixed"))
    second = verify_reproducible(Flat(), first)
    assert first.digest == second.digest
    assert first.results_digest == second.results_digest
    assert first.json() == second.json()

    moved = manifest(
        backtest(Flat(), [3], config=FAST.evolve(warmup=16.0), name="fixed")
    )
    assert moved.digest != first.digest

    other_seed = manifest(backtest(Flat(), [4], config=FAST, name="fixed"))
    assert other_seed.digest != first.digest


@pytest.mark.slow
def test_several_comparisons_are_corrected_across_the_set():
    """Two candidates against one baseline, with the false discovery rate held.

    Benjamini-Hochberg comes from ``aggregation.py`` rather than being written
    again here, so a claim made by this harness is corrected exactly the way a
    claim made by the information-aggregation experiment is. The candidate that
    is the baseline in disguise must come back with a difference of exactly zero
    whatever the correction does.
    """
    results = compare_many(
        {"wider": Flat(half=20), "same": Flat()},
        Flat(),
        seeds=[3, 4, 5],
        config=FAST,
        baseline_name="fixed",
    )
    assert len(results) == 2
    by_name = {r.name_a: r for r in results}
    assert by_name["same"].mean_difference == 0.0
    assert by_name["same"].verdict.startswith("identical")
    for row in results:
        assert row.p_adjusted >= row.p_value or math.isnan(row.p_value)
        assert 0.0 <= row.p_adjusted <= 1.0


@pytest.mark.slow
def test_a_taker_is_measured_as_a_taker():
    """The harness must accept a strategy that only crosses the spread.

    A strategy object is routed by which method it has, not by an isinstance
    against the maker protocol, because that protocol also requires ``symbols``
    and the adapter treats it as optional. A taker has neither and would be
    refused by a stricter test than the runtime applies.
    """
    result = backtest(Nibbler(), [3], config=FAST, name="nibbler")
    run = result.runs[0]
    assert run.conservation == 0
    if run.lots:
        assert run.aggressor_fraction == pytest.approx(1.0)
