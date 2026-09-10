"""Tests for the statistics that judge the market.

A diagnostic nobody has checked is worse than no diagnostic: it produces
numbers that look like evidence. So each estimator is fed series whose answer
is known analytically (a Gaussian random walk, a trending series, a
mean-reverting one, a deliberately fat-tailed one) and has to get them right
before it is allowed to have an opinion about the market.
"""

from __future__ import annotations

import numpy as np
import pytest

from arena.research.stylized import (
    acf_decay_exponent,
    analyse,
    autocorrelation,
    bid_ask_bounce,
    excess_kurtosis,
    hill_tail_index,
    hurst_exponent,
    ljung_box,
    order_flow_autocorrelation,
    returns,
    variance_ratio,
)

RNG = np.random.default_rng(20260824)


def random_walk(n: int = 4000, sigma: float = 1.0) -> np.ndarray:
    return np.cumsum(RNG.normal(0.0, sigma, n)) + 100.0


def trending(n: int = 4000) -> np.ndarray:
    """A walk with persistent drift in its increments."""
    steps = RNG.normal(0.0, 1.0, n)
    for i in range(1, n):
        steps[i] += 0.6 * steps[i - 1]
    return np.cumsum(steps) + 100.0


def mean_reverting(n: int = 4000) -> np.ndarray:
    series = np.zeros(n)
    for i in range(1, n):
        series[i] = 0.75 * series[i - 1] + RNG.normal(0.0, 1.0)
    return series + 100.0


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def test_returns_handles_negative_prices():
    """A spread trades below zero, so log returns are not an option.

    Defaulting to log returns would make the whole diagnostic inapplicable to
    a large part of this market's instrument universe.
    """
    prices = np.array([-100.0, -95.0, -110.0])
    assert returns(prices).tolist() == pytest.approx([5.0, -15.0])
    with pytest.raises(ValueError, match="strictly positive"):
        returns(prices, log=True)


def test_excess_kurtosis_is_zero_for_a_gaussian():
    assert excess_kurtosis(RNG.normal(0, 1, 200_000)) == pytest.approx(0.0, abs=0.1)


def test_excess_kurtosis_detects_fat_tails():
    """Student-t with 3 degrees of freedom is heavily fat-tailed."""
    assert excess_kurtosis(RNG.standard_t(3, 100_000)) > 3.0


def test_hill_index_recovers_a_known_tail_exponent():
    """A Pareto(alpha) sample must estimate back to alpha."""
    for alpha in (2.0, 3.0, 4.0):
        sample = (1.0 - RNG.random(60_000)) ** (-1.0 / alpha)
        assert hill_tail_index(sample) == pytest.approx(alpha, rel=0.15)


def test_hill_index_reports_thin_tails_as_large():
    """A Gaussian has no finite tail index; the estimate should be big."""
    assert hill_tail_index(RNG.normal(0, 1, 60_000)) > 5.0


def test_autocorrelation_of_white_noise_is_near_zero():
    assert abs(autocorrelation(RNG.normal(0, 1, 100_000), 1)) < 0.02


def test_autocorrelation_recovers_a_known_ar1():
    series = np.zeros(100_000)
    for i in range(1, series.size):
        series[i] = 0.5 * series[i - 1] + RNG.normal(0, 1)
    assert autocorrelation(series, 1) == pytest.approx(0.5, abs=0.03)


def test_ljung_box_accepts_noise_and_rejects_structure():
    _q, p_noise = ljung_box(RNG.normal(0, 1, 5_000), lags=10)
    assert p_noise > 0.05

    structured = np.zeros(5_000)
    for i in range(1, structured.size):
        structured[i] = 0.4 * structured[i - 1] + RNG.normal(0, 1)
    _q, p_structured = ljung_box(structured, lags=10)
    assert p_structured < 0.01


def test_hurst_separates_walk_trend_and_reversion():
    """0.5 is a random walk; above trends, below mean-reverts."""
    assert hurst_exponent(random_walk()) == pytest.approx(0.5, abs=0.1)
    assert hurst_exponent(trending()) > 0.6
    assert hurst_exponent(mean_reverting()) < 0.4


def test_variance_ratio_separates_walk_trend_and_reversion():
    assert variance_ratio(random_walk(), 4) == pytest.approx(1.0, abs=0.25)
    assert variance_ratio(trending(), 4) > 1.4
    assert variance_ratio(mean_reverting(), 4) < 0.8


def test_bid_ask_bounce_is_negative_when_trades_alternate():
    """The signature of trades crossing a spread in both directions."""
    value, spread = 100.0, 2.0
    signs = RNG.choice([-1, 1], 5_000)
    prices = value + signs * spread / 2
    assert bid_ask_bounce(prices) < -0.3

    # A one-sided tape has no bounce, which is what a wash-traded market shows.
    assert abs(bid_ask_bounce(np.full(5_000, value))) < 0.05 or np.isnan(
        bid_ask_bounce(np.full(5_000, value))
    )


def test_order_flow_autocorrelation_detects_persistence():
    """Persistent flow is what metaorder splitting produces."""
    independent = RNG.choice([-1.0, 1.0], 20_000)
    assert abs(order_flow_autocorrelation(independent)) < 0.03

    persistent = np.empty(20_000)
    persistent[0] = 1.0
    for i in range(1, persistent.size):
        persistent[i] = persistent[i - 1] if RNG.random() < 0.8 else -persistent[i - 1]
    assert order_flow_autocorrelation(persistent) > 0.4


def test_acf_decay_exponent_declines_to_measure_pure_noise():
    """No autocorrelation means no decay to fit, and it must say so.

    Fitting a power law to a series with no memory produced -0.03, a confident
    number manufactured from noise, which reads as "slower decay than any real
    market" rather than "nothing here". NaN is the honest answer.
    """
    assert np.isnan(acf_decay_exponent(RNG.normal(0, 1, 20_000)))


def test_acf_decay_exponent_recovers_real_memory():
    """A series with genuine persistence in |value| must produce a finite slope."""
    n = 40_000
    volatility = np.zeros(n)
    for i in range(1, n):
        volatility[i] = 0.97 * volatility[i - 1] + RNG.normal(0, 0.2)
    series = RNG.normal(0, 1, n) * np.exp(volatility)
    value = acf_decay_exponent(series)
    assert np.isfinite(value) and 0.0 < value < 2.0


# --------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn", [excess_kurtosis, hill_tail_index, acf_decay_exponent, hurst_exponent]
)
def test_estimators_return_nan_rather_than_raising_on_short_input(fn):
    """A thin market must not crash the diagnostic; it must say 'not enough data'."""
    assert np.isnan(fn(np.array([1.0, 2.0, 3.0])))


def test_a_flat_series_does_not_produce_a_spurious_number():
    flat = np.full(1_000, 42.0)
    assert np.isnan(autocorrelation(returns(flat), 1))
    assert np.isnan(excess_kurtosis(returns(flat)))


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def test_report_marks_expectations_rather_than_only_reporting_values():
    """Every number carries the prior it is judged against.

    A statistic with no expectation attached invites reading whatever one
    hoped for into it, which is precisely how a simulator gets tuned until it
    looks real.
    """
    prices = random_walk(3_000)
    signs = RNG.choice([-1.0, 1.0], 3_000)
    trades = prices + signs

    report = analyse("TEST", prices, trades, signs)
    assert report.observations == 3_000
    assert len(report.verdicts) >= 8
    assert all(v.expected for v in report.verdicts)
    assert all(v.verdict in ("as expected", "unexpected", "n/a") for v in report.verdicts)

    named = {v.name: v for v in report.verdicts}
    # A random walk must read as efficient and as normal-tailed.
    assert named["return autocorrelation (lag 1)"].verdict == "as expected"
    assert named["excess kurtosis of returns"].verdict == "unexpected"
