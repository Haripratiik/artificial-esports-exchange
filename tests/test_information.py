"""Information measured in battles, and the aggregation ladder that scores it.

The point of :mod:`arena.agents.bayesian` is to replace a free noise knob with a
quantity that has a unit, and a quantity with a unit can be *wrong*. These are
the tests that could catch it being wrong.

The sharpest of them is not the 1/sqrt(n) slope everyone quotes but
``test_error_matches_the_closed_form_bayes_risk``: when the true level is drawn
from the same prior the agent holds, the expected squared error of the posterior
mean has an exact closed form,

    E[(E[p|k] - p)^2] = p_bar (1 - p_bar) * kappa / ((kappa + 1)(kappa + n))

by the law of total variance applied to the Beta-Binomial marginal. Matching a
constant is a much stronger check than matching an exponent: a conjugate update
with the prior applied twice, or with the win count off by one, still produces
a -0.5 slope but lands on the wrong constant.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from arena.agents.bayesian import binary_probability, posterior_for
from arena.research.aggregation import (
    Comparison,
    benjamini_hochberg,
    expit,
    extremized_mean,
    fit_extremization,
    logit,
    murphy_decomposition,
    precision_weighted_mean,
    simple_mean,
)

SAMPLE_SIZES = (25, 100, 400, 1600, 6400)


def posterior_mean_errors(
    battles: int,
    prior_mean: float,
    prior_strength: float,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Errors of the posterior mean when truth is drawn from the agent's prior.

    Drawing the truth from the prior is what makes the Bayes risk exact. It is
    also the honest setup for this world: the experiment draws each contract's
    true level from the reference snapshot's prior, and gives the agents that
    same prior.
    """
    rng = random.Random(seed)
    a0 = prior_strength * prior_mean
    b0 = prior_strength * (1.0 - prior_mean)
    errors = np.empty(replicates)
    for i in range(replicates):
        truth = rng.betavariate(a0, b0)
        a, b = posterior_for(truth, battles, prior_mean, prior_strength, rng)
        errors[i] = a / (a + b) - truth
    return errors


def bayes_risk(battles: int, prior_mean: float, prior_strength: float) -> float:
    """Closed-form RMSE of the posterior mean under the prior."""
    variance = (
        prior_mean
        * (1.0 - prior_mean)
        * prior_strength
        / ((prior_strength + 1.0) * (prior_strength + battles))
    )
    return math.sqrt(variance)


# --------------------------------------------------------------------------
# The information model
# --------------------------------------------------------------------------


def test_error_falls_as_one_over_root_n():
    """The exit criterion: log-log slope of RMSE against n is -1/2.

    A weak prior is used deliberately. With a strong one the slope is *not*
    -0.5 and should not be; see the next test, which pins the shrinkage
    exactly. Quoting -0.5 while holding a strong prior would be quoting the
    asymptotic law outside the regime where it applies.
    """
    prior_mean, prior_strength = 0.5, 2.0
    rmse = [
        float(
            np.sqrt(
                np.mean(
                    posterior_mean_errors(n, prior_mean, prior_strength, 20_000, 900 + n)
                    ** 2
                )
            )
        )
        for n in SAMPLE_SIZES
    ]
    slope = float(np.polyfit(np.log(SAMPLE_SIZES), np.log(rmse), 1)[0])
    assert slope == pytest.approx(-0.5, abs=0.05), f"slope {slope:.4f}, rmse {rmse}"


@pytest.mark.parametrize("prior_strength", [2.0, 50.0])
@pytest.mark.parametrize("battles", SAMPLE_SIZES)
def test_error_matches_the_closed_form_bayes_risk(battles, prior_strength):
    """Simulated RMSE equals the analytic Bayes risk, constant included."""
    prior_mean = 0.55
    errors = posterior_mean_errors(
        battles, prior_mean, prior_strength, 20_000, 31 * battles + int(prior_strength)
    )
    observed = float(np.sqrt(np.mean(errors**2)))
    expected = bayes_risk(battles, prior_mean, prior_strength)
    # An RMSE from R replicates carries relative error of order 1/sqrt(2R);
    # 5% is a comfortable multiple of that at R = 20,000.
    assert observed == pytest.approx(expected, rel=0.05)


def test_a_strong_prior_makes_the_slope_shallower():
    """Shrinkage is real, and the model shows it rather than hiding it.

    With prior strength kappa the slope is -0.5 * n/(kappa + n), so a strong
    prior flattens the curve at small n. If this test failed (if the slope were
    -0.5 regardless of kappa), the prior would not be doing anything and the
    "empirical Bayes" framing would be decoration.
    """
    small = SAMPLE_SIZES[:3]

    def slope_for(prior_strength: float) -> float:
        rmse = [
            float(
                np.sqrt(
                    np.mean(
                        posterior_mean_errors(n, 0.5, prior_strength, 20_000, 7 + n) ** 2
                    )
                )
            )
            for n in small
        ]
        return float(np.polyfit(np.log(small), np.log(rmse), 1)[0])

    weak, strong = slope_for(2.0), slope_for(200.0)
    assert weak < strong, f"weak {weak:.3f} should be steeper than strong {strong:.3f}"
    assert strong > -0.25, f"strong-prior slope {strong:.3f} should be visibly flattened"


def test_no_evidence_leaves_the_prior_untouched():
    """An agent that saw nothing holds exactly the prior: no accidental drift."""
    a, b = posterior_for(0.9, 0, prior_mean=0.4, prior_strength=30.0, rng=random.Random(1))
    assert a == pytest.approx(12.0)
    assert b == pytest.approx(18.0)
    assert a / (a + b) == pytest.approx(0.4)


def test_evidence_accumulates_as_counts():
    """Total posterior weight is prior strength plus battles, exactly."""
    for battles in (1, 10, 1000):
        a, b = posterior_for(0.6, battles, 0.5, 20.0, random.Random(battles))
        assert a + b == pytest.approx(20.0 + battles)


def test_negative_evidence_is_rejected():
    with pytest.raises(ValueError):
        posterior_for(0.5, -1, 0.5, 10.0, random.Random(0))


def test_binary_probability_is_the_exact_tail():
    """The analytic tail matches Monte Carlo, and complements sum to one."""
    rng = random.Random(4)
    a, b = 14.0, 9.0
    threshold = 0.6
    draws = [rng.betavariate(a, b) for _ in range(200_000)]
    empirical = sum(1 for d in draws if d > threshold) / len(draws)

    above = binary_probability(a, b, threshold, ">")
    below = binary_probability(a, b, threshold, "<")
    assert above == pytest.approx(empirical, abs=0.005)
    assert above + below == pytest.approx(1.0)


def test_binary_probability_is_monotone_in_the_threshold():
    a, b = 8.0, 12.0
    values = [binary_probability(a, b, t / 20, ">") for t in range(21)]
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.0)


def test_more_evidence_concentrates_the_belief():
    """Two agents on the same truth: the better-informed one is less unsure."""
    rng = random.Random(11)
    truth = 0.62
    spreads = []
    for battles in (50, 5000):
        a, b = posterior_for(truth, battles, 0.5, 20.0, rng)
        spreads.append(math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))
    assert spreads[1] < spreads[0] / 5


# --------------------------------------------------------------------------
# The aggregation ladder
# --------------------------------------------------------------------------


def test_extremization_at_one_is_the_log_odds_mean():
    forecasts = [0.3, 0.5, 0.8]
    expected = expit(sum(logit(p) for p in forecasts) / 3)
    assert extremized_mean(forecasts, 1.0) == pytest.approx(expected)


def test_extremization_pushes_away_from_one_half():
    """Above one half it raises, below it lowers, and one half is a fixed point."""
    assert extremized_mean([0.7, 0.7], 2.0) > 0.7
    assert extremized_mean([0.3, 0.3], 2.0) < 0.3
    assert extremized_mean([0.5, 0.5], 3.0) == pytest.approx(0.5)


def test_extremization_recovers_the_factor_it_was_generated_with():
    """fit_extremization is an estimator, so it must recover a known answer."""
    rng = random.Random(5)
    true_d = 2.4
    forecast_sets, truths = [], []
    for _ in range(300):
        base = rng.uniform(-2.0, 2.0)
        forecasts = [expit(base + rng.gauss(0, 0.3)) for _ in range(7)]
        forecast_sets.append(forecasts)
        truths.append(extremized_mean(forecasts, true_d))
    assert fit_extremization(forecast_sets, truths) == pytest.approx(true_d, abs=0.1)


def test_precision_weighting_follows_the_weights():
    """All weight on one agent reproduces that agent; equal weights average."""
    forecasts = [0.2, 0.9]
    assert precision_weighted_mean(forecasts, [1.0, 0.0]) == pytest.approx(0.2, abs=1e-5)
    assert precision_weighted_mean(forecasts, [0.0, 1.0]) == pytest.approx(0.9, abs=1e-5)
    assert precision_weighted_mean(forecasts, [1.0, 1.0]) == pytest.approx(
        simple_mean(forecasts)
    )


def test_murphy_decomposition_adds_up():
    """Brier = reliability - resolution + uncertainty, as an identity."""
    rng = random.Random(9)
    forecasts, outcomes = [], []
    for _ in range(4000):
        p = rng.uniform(0.05, 0.95)
        forecasts.append(p)
        outcomes.append(1.0 if rng.random() < p else 0.0)
    d = murphy_decomposition(forecasts, outcomes, bins=10)
    assert d["brier"] == pytest.approx(
        d["reliability"] - d["resolution"] + d["uncertainty"], abs=2e-3
    )


def test_a_calibrated_forecaster_has_near_zero_reliability_penalty():
    rng = random.Random(3)
    forecasts, outcomes = [], []
    for _ in range(20_000):
        p = rng.choice([0.1, 0.3, 0.5, 0.7, 0.9])
        forecasts.append(p)
        outcomes.append(1.0 if rng.random() < p else 0.0)
    d = murphy_decomposition(forecasts, outcomes, bins=10)
    assert d["reliability"] < 0.002


def _comparison(p_value: float) -> Comparison:
    return Comparison("x", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_value, 10)


def test_benjamini_hochberg_matches_a_hand_computed_case():
    """Four p-values, known answer, including the monotonicity step."""
    raw = [0.001, 0.008, 0.039, 0.041]
    adjusted = [c.p_adjusted for c in benjamini_hochberg([_comparison(p) for p in raw])]
    # 0.001*4/1=0.004, 0.008*4/2=0.016, 0.039*4/3=0.052, 0.041*4/4=0.041;
    # the third is then pulled down to the fourth by the monotonicity rule.
    assert adjusted == pytest.approx([0.004, 0.016, 0.041, 0.041])


def test_a_comparison_that_could_not_be_computed_does_not_correct_the_others():
    """A NaN is not a test, and counting it as one weakened every real result.

    `paired_comparison` returns all-NaN below three trials rather than raising,
    and scipy's paired t-test on two identical arms is 0/0, so being handed one
    is ordinary. Measured before the fix, on 0.001, 0.04, 0.30 and 0.90, which
    correct to 0.004, 0.080, 0.400 and 0.900: adding one NaN row moved them to
    0.005, 0.100, 0.500 and 1.000, because it counted toward the total and
    multiplied every real p-value by 5/4. One of them crossed 0.05 the wrong
    way purely because a fifth comparison had failed to compute.

    The NaN row itself came back as 1.0 rather than NaN, since
    `min(previous, nan)` returns `previous`, so a comparison nobody could
    compute reported "not significant" as though it had been tested.
    """
    raw = [0.001, 0.04, 0.30, 0.90]
    expected = [c.p_adjusted for c in benjamini_hochberg([_comparison(p) for p in raw])]

    with_nan = benjamini_hochberg(
        [_comparison(p) for p in raw] + [_comparison(float("nan"))]
    )
    assert [c.p_adjusted for c in with_nan[:-1]] == pytest.approx(expected)
    assert math.isnan(with_nan[-1].p_adjusted)


def test_every_comparison_failing_to_compute_corrects_nothing():
    """No tests were run, so no correction is owed and none is invented."""
    adjusted = benjamini_hochberg([_comparison(float("nan")) for _ in range(3)])
    assert all(math.isnan(c.p_adjusted) for c in adjusted)


def test_benjamini_hochberg_is_monotone_and_never_lowers_a_p_value():
    raw = [0.5, 0.01, 0.2, 0.049]
    results = benjamini_hochberg([_comparison(p) for p in raw])
    for comparison in results:
        assert comparison.p_adjusted >= comparison.p_value - 1e-12
        assert comparison.p_adjusted <= 1.0
    ordered = sorted(results, key=lambda c: c.p_value)
    values = [c.p_adjusted for c in ordered]
    assert values == sorted(values)
