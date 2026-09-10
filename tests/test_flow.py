"""Realistic order flow: power-law sizes, clustered arrivals, heavy cancellation.

The parameters here are quoted from the literature, so the tests that matter are
the ones checking the code actually *delivers* the distribution it was asked
for. A sampler that produced a heavy tail with the wrong exponent would look
entirely plausible in every plot and would quietly misstate the one number these
agents exist to control.

So the exponent is measured back out with a Hill estimator rather than trusted.
It comes back slightly high (flooring a continuous draw moves mass down, which
shortens the log-distances the estimator sums) and the tolerance is set to
admit that known bias and nothing larger.
"""

from __future__ import annotations

import random
import statistics

import numpy as np
import pytest

from arena.agents.flow import FlowTrader, placement_offset, power_law_size


def _draws(exponent: float, count: int = 400_000, cap: int = 10**7, seed: int = 11):
    rng = random.Random(seed)
    return np.array(
        [power_law_size(rng, exponent, 1, cap) for _ in range(count)], dtype=float
    )


def _hill(sample: np.ndarray, threshold: float) -> float:
    tail = sample[sample >= threshold]
    return len(tail) / float(np.sum(np.log(tail / threshold)))


# --------------------------------------------------------------------------
# The samplers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("exponent", [1.5, 2.0, 2.4])
def test_the_sampler_delivers_the_exponent_it_was_given(exponent):
    """Measured back out, not assumed. The convention is the survival exponent.

    Had the parameter meant something else (the density exponent, say, which
    differs by one), every draw would still look heavy-tailed and the number
    would be wrong by 40%.
    """
    estimate = _hill(_draws(exponent), threshold=10)
    assert estimate == pytest.approx(exponent, rel=0.15), (
        f"asked for {exponent}, got {estimate:.2f}"
    )


def test_a_larger_exponent_means_a_lighter_tail():
    light, heavy = _draws(3.0), _draws(1.5)
    assert np.percentile(heavy, 99) > np.percentile(light, 99)
    assert heavy.mean() > light.mean()


@pytest.mark.parametrize("exponent", [1.5, 2.4])
def test_volume_concentration_matches_the_pareto_prediction(exponent):
    """The point of a power law, checked against theory rather than a guess.

    For a Pareto with tail index a, the share of total mass held by the top
    fraction q is exactly ``q ** (1 - 1/a)``, about 21% for a = 1.5 and 6.8%
    for a = 2.4 at q = 1%. Flooring the draws lifts that slightly, because
    rounding costs a small order proportionally far more than a large one, and
    the tolerance admits that and nothing bigger.
    """
    sample = np.sort(_draws(exponent, count=200_000, cap=10_000))
    observed = sample[-2_000:].sum() / sample.sum()
    predicted = 0.01 ** (1.0 - 1.0 / exponent)
    assert observed == pytest.approx(predicted, rel=0.35), (
        f"top 1% held {observed:.1%}, theory says {predicted:.1%}"
    )
    assert observed >= predicted, "flooring should lift concentration, not lower it"


def test_draws_respect_their_bounds():
    rng = random.Random(2)
    values = [power_law_size(rng, 2.0, 3, 40) for _ in range(20_000)]
    assert min(values) >= 3
    assert max(values) <= 40


def test_the_cap_actually_binds_on_a_heavy_tail():
    """Truncation is load-bearing: an exponent below 2 has infinite variance,
    and one unbounded draw would make the run about that order."""
    rng = random.Random(5)
    values = [power_law_size(rng, 1.2, 1, 500) for _ in range(20_000)]
    assert max(values) == 500


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_exponent_is_refused(bad):
    with pytest.raises(ValueError):
        power_law_size(random.Random(0), bad, 1, 10)


@pytest.mark.parametrize("bad", [1.0, 1.5, -0.1])
def test_an_unstable_branching_ratio_is_refused(bad):
    """The regression test for a real explosion.

    A first version added excitation in the wrong units with no stability
    condition. One agent reached 26,000 times its baseline intensity, its
    inter-arrival time collapsed to microseconds, and it emitted roughly 6,000
    orders a second forever: the test suite simply stopped returning. At a
    branching ratio of one or above the process is explosive by construction,
    so the constructor refuses rather than leaving it to be discovered.
    """
    from arena.exchange.types import AgentId

    with pytest.raises(ValueError, match="branching ratio"):
        FlowTrader(AgentId("f"), AgentId("v"), {}, branching_ratio=bad)


def test_the_arrival_rate_matches_hawkes_theory():
    """A stable Hawkes process runs at mu / (1 - n). Explosion shows up as a
    realised rate far above it, which is what the bug looked like."""
    import math

    from arena.sim.time import millis

    base, decay, ratio = int(millis(500)), int(millis(1_200)), 0.55
    rng = random.Random(1)
    now, intensity, last = 0, 0.0, 0
    events = 60_000
    for _ in range(events):
        elapsed = now - last
        last = now
        intensity *= math.exp(-elapsed / decay)
        now += max(1, int(rng.expovariate(1.0) / (1.0 / base + intensity)))
        intensity += ratio / decay

    realised = events / (now / 1e9)
    predicted = (1e9 / base) / (1.0 - ratio)
    assert realised == pytest.approx(predicted, rel=0.10)


def test_most_limit_orders_land_at_or_near_the_touch():
    rng = random.Random(7)
    offsets = np.array([placement_offset(rng, 1.5, 60) for _ in range(50_000)])
    assert offsets.min() >= 0
    assert (offsets == 0).mean() > 0.4, "not concentrated at the touch"
    assert (offsets > 10).mean() > 0.01, "no tail placed away from the touch"


# --------------------------------------------------------------------------
# The agent in a market
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def flow_market():
    from arena.sim.time import seconds
    from dashboard.build_market import build

    market = build(seed=19, flow_traders=6)
    market.kernel.start()
    market.kernel.advance(until=seconds(240))
    return market


def _flow_agents(market):
    return [a for a in market.agents if isinstance(a, FlowTrader)]


def test_the_flow_traders_are_active(flow_market):
    submitted = sum(a.submitted for a in _flow_agents(flow_market))
    assert submitted > 200, "the population barely traded, so nothing below means much"


def test_the_book_is_cancel_dominated(flow_market):
    """More orders pulled than filled, which is the direction, not the level.

    Real equity books cancel well above 90% of orders. This market reaches
    about 60%, and adding these agents barely moves it, because real
    cancellation is dominated by makers requoting at microsecond scale and
    nothing here requotes faster than 300ms. The assertion is therefore what
    the model actually delivers; the gap is documented rather than tuned away.
    """
    agents = _flow_agents(flow_market)
    cancelled = sum(a.cancelled for a in agents)
    filled = sum(a.fills for a in agents)
    assert cancelled > filled, f"{cancelled} cancels against {filled} fills"
    assert cancelled / max(1, sum(a.submitted for a in agents)) > 0.4


def test_arrivals_are_clustered_rather_than_poisson(flow_market):
    """A Poisson clock has a coefficient of variation of exactly one.

    Self-excitation makes the inter-arrival distribution over-dispersed, so a CV
    meaningfully above one is the signature. Without it these agents would be
    the existing noise traders with a different size distribution.
    """
    from arena.sim.time import Timestamp, seconds
    from dashboard.build_market import build

    market = build(seed=23, flow_traders=1)
    market.kernel.start()
    agent = _flow_agents(market)[0]

    stamps: list[int] = []
    original = FlowTrader.act

    def spy(self, ctx):
        stamps.append(int(ctx.now))
        return original(self, ctx)

    FlowTrader.act = spy
    try:
        market.kernel.advance(until=seconds(600))
    finally:
        FlowTrader.act = original

    gaps = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
    assert len(gaps) > 100
    cv = statistics.stdev(gaps) / statistics.mean(gaps)
    assert cv > 1.05, f"inter-arrival CV {cv:.2f} is indistinguishable from Poisson"


def test_value_is_conserved_with_realistic_flow(flow_market):
    """Bursty, cancel-heavy flow is where an accounting leak would surface."""
    assert flow_market.venue.conservation_check() == 0


def test_the_book_survives_the_churn(flow_market):
    """Heavy cancellation is exactly what broke depth accounting once before.

    Depth is checked on every book; the crossing check only on books that are
    trading. A call phase accumulates orders without matching them, so a
    crossed book there is the mechanism working rather than failing.
    """
    from arena.exchange.session import SessionState
    from arena.exchange.types import Side

    for symbol in flow_market.venue.registry.symbols:
        book = flow_market.venue.engine(symbol).book
        snapshot = book.snapshot(levels=10)
        for side, ladder in ((Side.BUY, snapshot.bids), (Side.SELL, snapshot.asks)):
            for price, quantity in ladder:
                assert int(quantity) > 0
                assert int(book.depth_at(side, price)) == int(quantity)
        if flow_market.venue.session(symbol) is not SessionState.CONTINUOUS:
            continue
        if snapshot.best_bid is not None and snapshot.best_ask is not None:
            assert int(snapshot.best_bid) < int(snapshot.best_ask)


def test_flow_traders_are_absent_by_default():
    """Emergence claims require them off, so off is the default."""
    from dashboard.build_market import build

    assert _flow_agents(build(seed=3)) == []
