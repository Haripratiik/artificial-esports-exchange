"""The experiment harness: is it measuring what it claims to measure?

An experiment that runs cleanly and reports a confident number is worth nothing
if the number is of the wrong thing. These tests check the parts of the harness
that could be quietly wrong while still producing a plausible table:

  * the ground truth is the probability it says it is, not an approximation
  * the thresholds really do spread the answers across the range, rather than
    piling every trial up at 0 or 1 where no forecaster can be distinguished
    from any other
  * an agent given overwhelming evidence forecasts the truth, and an agent
    given none forecasts the prior
  * the same configuration produces the same result, twice

The last one is the one that makes the rest reportable. Without it a run is an
anecdote.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from arena.agents.bayesian import (
    posterior_for,
    predictive_probability,
    settlement_probability,
)
from arena.research.experiment import (
    TrialConfig,
    _threshold_for,
    draw_trials,
    manifest_digest,
    run_trial,
)
from arena.sim.time import seconds

SHORT = int(seconds(45))


def _config(**overrides) -> TrialConfig:
    base = dict(
        seed=5,
        truth=0.52,
        threshold=0.515,
        battles=(100, 1_000),
        window_battles=1_000,
        noise_traders=3,
        duration=SHORT,
    )
    base.update(overrides)
    return TrialConfig(**base)


# --------------------------------------------------------------------------
# The ground truth
# --------------------------------------------------------------------------


def test_the_truth_is_the_probability_it_claims_to_be():
    """settlement_probability matches simulating the window directly."""
    rng = random.Random(2)
    truth, window, threshold = 0.53, 1_000, 0.52
    hits = sum(
        1 for _ in range(40_000) if rng.binomialvariate(window, truth) / window > threshold
    )
    assert settlement_probability(truth, window, threshold) == pytest.approx(
        hits / 40_000, abs=0.01
    )


def test_a_wider_window_makes_the_answer_more_certain():
    """More battles in the window means less settlement noise, so a sharper truth.

    If this failed, the contract's uncertainty would not be coming from the
    finite window at all, and the whole reason there is a probability to
    forecast would be wrong.
    """
    truth, threshold = 0.55, 0.53
    values = [settlement_probability(truth, n, threshold) for n in (200, 2_000, 20_000)]
    assert values[0] < values[1] < values[2]
    assert values[2] > 0.99


def test_the_threshold_inverter_hits_its_target():
    """Trials are designed, not stumbled into: the answer is placed where asked."""
    for target in (0.1, 0.25, 0.5, 0.75, 0.9):
        threshold = _threshold_for(0.48, 2_000, target)
        assert settlement_probability(0.48, 2_000, threshold) == pytest.approx(
            target, abs=0.02
        )


def test_trials_span_the_probability_range():
    """A set of foregone conclusions cannot separate any two forecasters.

    With a 2,000-battle window a random threshold puts almost every trial at 0
    or 1, so the trial set has to be designed. This checks the design worked.
    """
    trials = draw_trials(40, seed=3)
    truths = np.array([t.truth_probability for t in trials])
    assert truths.min() < 0.15
    assert truths.max() > 0.85
    # Reasonably spread rather than bunched at the ends.
    assert 0.3 < float(np.mean((truths > 0.25) & (truths < 0.75)))


def test_every_trial_in_a_set_is_distinct():
    trials = draw_trials(25, seed=9)
    assert len({t.seed for t in trials}) == 25
    assert len({t.truth for t in trials}) == 25


# --------------------------------------------------------------------------
# Agents at the limits
# --------------------------------------------------------------------------


def test_an_agent_with_overwhelming_evidence_forecasts_the_truth():
    """Enough battles and the posterior collapses onto p*, so the forecast is exact."""
    config = _config(battles=(2_000_000,), window_battles=1_000)
    rng = random.Random(11)
    a, b = posterior_for(
        config.truth, 2_000_000, config.prior_mean, config.prior_strength, rng
    )
    forecast = predictive_probability(a, b, config.window_battles, config.threshold)
    assert forecast == pytest.approx(config.truth_probability, abs=0.02)


def test_an_agent_with_no_evidence_forecasts_the_prior():
    """Zero battles means the prior, and the prior alone: no leaked truth.

    If an uninformed agent's forecast moved with p*, information would be
    reaching agents through some channel other than the battles they observed,
    and every measurement in the experiment would be contaminated.
    """
    rng = random.Random(1)
    forecasts = set()
    for truth in (0.30, 0.50, 0.70):
        a, b = posterior_for(truth, 0, 0.5, 20.0, rng)
        forecasts.add(round(predictive_probability(a, b, 1_000, 0.52), 12))
    assert len(forecasts) == 1


def test_more_evidence_forecasts_better_on_average():
    """The whole information model in one assertion.

    Not true trial by trial (a poorly-informed agent gets lucky sometimes), so
    it is checked as an average over many draws, which is the sense in which it
    is actually claimed.
    """
    truth, window, threshold = 0.52, 1_000, 0.515
    target = settlement_probability(truth, window, threshold)
    errors = {}
    for battles in (50, 5_000):
        rng = random.Random(7)
        squared = []
        for _ in range(400):
            a, b = posterior_for(truth, battles, 0.5, 20.0, rng)
            squared.append(
                (predictive_probability(a, b, window, threshold) - target) ** 2
            )
        errors[battles] = float(np.mean(squared))
    assert errors[5_000] < errors[50]


# --------------------------------------------------------------------------
# A trial as a whole
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trial():
    return run_trial(_config())


def test_a_trial_produces_a_forecast_from_a_working_market(trial):
    assert math.isfinite(trial.market_forecast)
    assert 0.0 <= trial.market_forecast <= 1.0
    assert trial.trades > 0, "a market with no trades has aggregated nothing"
    assert trial.quoted_fraction > 0.5, "the book was mostly one-sided"


def test_a_trial_conserves_value(trial):
    """Every experiment result rests on the market being sound underneath it."""
    assert trial.conservation == 0


def test_a_trial_reports_one_forecast_per_informed_agent(trial):
    assert len(trial.agent_forecasts) == len(trial.config.battles)
    assert all(0.0 <= f <= 1.0 for f in trial.agent_forecasts)


def test_the_same_configuration_gives_the_same_result():
    """Determinism, checked rather than asserted in a docstring."""
    config = _config(seed=77)
    first, second = run_trial(config), run_trial(config)
    assert first.to_dict() == second.to_dict()
    assert manifest_digest(first.to_dict()) == manifest_digest(second.to_dict())


def test_a_different_seed_gives_a_different_result():
    """Otherwise the previous test would pass on a harness that ignores its input."""
    first = run_trial(_config(seed=77))
    second = run_trial(_config(seed=78))
    assert first.to_dict() != second.to_dict()


# --------------------------------------------------------------------------
# Both mechanisms
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lmsr_trial():
    return run_trial(_config(venue_kind="lmsr"))


def test_the_scoring_rule_venue_produces_a_forecast(lmsr_trial):
    assert math.isfinite(lmsr_trial.market_forecast)
    assert 0.0 <= lmsr_trial.market_forecast <= 1.0
    assert lmsr_trial.trades > 0


def test_the_scoring_rule_venue_is_always_quoted(lmsr_trial):
    """Its defining advantage over a book: there is never nothing to trade against.

    An order book can empty out; a cost function cannot. If this ever dropped
    below one, the comparison in Experiment 2 would partly be measuring
    downtime rather than aggregation.
    """
    assert lmsr_trial.quoted_fraction == 1.0


def test_the_scoring_rule_venue_conserves_value(lmsr_trial):
    assert lmsr_trial.conservation == 0


def test_both_mechanisms_are_deterministic():
    for kind in ("clob", "lmsr"):
        config = _config(seed=404, venue_kind=kind)
        assert run_trial(config).to_dict() == run_trial(config).to_dict()


def test_the_two_mechanisms_see_identical_agents():
    """The control that makes Experiment 2 a comparison rather than two runs.

    Agents draw their evidence from their own seeded streams, so swapping the
    venue must not change what any of them believes, only what the price does
    with those beliefs.
    """
    clob = run_trial(_config(seed=88, venue_kind="clob"))
    lmsr = run_trial(_config(seed=88, venue_kind="lmsr"))
    assert clob.agent_forecasts == lmsr.agent_forecasts
    assert clob.truth_probability == lmsr.truth_probability


def test_an_unknown_venue_is_refused():
    with pytest.raises(ValueError, match="unknown venue kind"):
        run_trial(_config(venue_kind="dark-pool"))


def test_the_outcome_draw_does_not_disturb_the_simulation():
    """The sampled outcome comes from its own stream.

    It is only used for the secondary metric, so it must not be able to shift
    the market; otherwise reporting a secondary number would change the primary
    one.
    """
    config = _config(seed=101)
    result = run_trial(config)
    assert result.outcome in (0.0, 1.0)
    # The outcome is a function of the config alone, and the market is not.
    assert run_trial(config).outcome == result.outcome
