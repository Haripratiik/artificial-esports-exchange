"""The baseline ladder, and the statistics for comparing against it.

The question is whether a market aggregates dispersed information better than
the agents inside it. That question is only interesting if the thing it is
compared against is genuinely good, and this is where the experiment is most
easily rigged: comparing a market to the *worst* agent, or to a naive average,
produces a win that the literature established decades ago and that says
nothing about the market.

So the market is scored against a ladder, weakest rung first:

  1. best single agent, chosen after the fact: an upper bound nobody could have
     picked in advance, included precisely because it is unfair
  2. simple mean of the agents' probabilities
  3. precision-weighted mean, weighting by how much evidence each agent saw.
     In this world those weights are *known exactly*, which no real study can
     say, so this rung is stronger here than it could be in the field
  4. extremized log-odds mean, the rung that matters

Why extremizing is the rung that matters
----------------------------------------

When information is dispersed, each forecaster sees only part of it, so each
one's probability is pulled toward the prior. Averaging them keeps that
timidity: the average is systematically under-confident, not because anyone
erred but because nobody saw everything. The fix is to push the average away
from one half in log-odds space:

    logit(p_ext) = d * mean(logit(p_j)),  d > 1

Satopaa et al. find d in roughly [1.16, 3.92] optimal on geopolitical
forecasting tournaments. A market that beats rungs 1 and 2 has reproduced a
known result. A market that beats rung 4 has done something worth reporting.

**d is fitted out of sample.** Fitting it on the same trials it is scored on
would let the baseline see the answers and would understate the market's
performance is unfair in the other direction, so the trials are split, d is
fitted on one half and evaluated on the other, and both halves take both roles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

__all__ = [
    "logit",
    "expit",
    "simple_mean",
    "precision_weighted_mean",
    "extremized_mean",
    "fit_extremization",
    "brier",
    "murphy_decomposition",
    "paired_comparison",
    "benjamini_hochberg",
    "Comparison",
]

EPSILON = 1e-6


def _clip(p: float) -> float:
    """Keep probabilities off the boundary so log-odds stay finite."""
    return min(1.0 - EPSILON, max(EPSILON, p))


def logit(p: float) -> float:
    q = _clip(p)
    return math.log(q / (1.0 - q))


def expit(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


def simple_mean(forecasts: Sequence[float]) -> float:
    """Unweighted average. The rung the literature already beat."""
    if not forecasts:
        return float("nan")
    return float(np.mean([_clip(p) for p in forecasts]))


def precision_weighted_mean(
    forecasts: Sequence[float], weights: Sequence[float]
) -> float:
    """Average weighted by how much evidence each agent saw.

    The weights are the agents' battle counts, which in this world are known
    exactly rather than estimated. That makes this a stronger baseline than any
    field study could construct, and it is included for exactly that reason.
    """
    if not forecasts or len(forecasts) != len(weights):
        return float("nan")
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        return simple_mean(forecasts)
    p = np.asarray([_clip(x) for x in forecasts], dtype=float)
    return float((p * w).sum() / w.sum())


def extremized_mean(forecasts: Sequence[float], d: float) -> float:
    """Log-odds mean pushed away from one half by factor ``d``.

    ``d = 1`` is the plain log-odds (geometric-odds) mean, which is already a
    slightly different aggregate from the arithmetic one; ``d > 1`` sharpens it.
    """
    if not forecasts:
        return float("nan")
    mean_logit = float(np.mean([logit(p) for p in forecasts]))
    return expit(d * mean_logit)


def fit_extremization(
    forecast_sets: Sequence[Sequence[float]],
    truths: Sequence[float],
    grid: Sequence[float] | None = None,
) -> float:
    """Choose the extremization factor that minimises squared error to truth.

    Searched over a grid rather than solved analytically because the objective
    is cheap and a grid cannot diverge. The range covers and extends past the
    [1.16, 3.92] band the forecasting literature reports.

    It also extends *below* one, which Satopaa et al.'s formulation does not.
    That is deliberate. Extremizing assumes the pooled forecast is
    under-confident, which is what happens when forecasters see disjoint pieces
    of the evidence. If a population is instead over-confident, the best factor
    is below one and clamping the grid at 1.0 would hide it, reporting a
    baseline pinned at its own boundary as though it were an interior optimum.
    Letting the factor go below one can only make this baseline stronger, which
    is the conservative direction for anything the market is claimed to beat.
    """
    if grid is None:
        grid = np.arange(0.50, 4.01, 0.05)
    best_d, best_loss = 1.0, float("inf")
    for d in grid:
        loss = 0.0
        for forecasts, truth in zip(forecast_sets, truths):
            loss += (extremized_mean(forecasts, float(d)) - truth) ** 2
        if loss < best_loss:
            best_d, best_loss = float(d), loss
    return best_d


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def brier(forecast: float, outcome: float) -> float:
    """Squared error. Against a realised 0/1 outcome, or against a known truth.

    This market can do something field studies cannot: the true probability is
    known, so error can be measured against *it* rather than against a single
    sampled outcome. That removes the Bernoulli noise that dominates
    outcome-based scoring and is why the primary metric here is error-to-truth.
    """
    return (forecast - outcome) ** 2


def murphy_decomposition(
    forecasts: Sequence[float], outcomes: Sequence[float], bins: int = 10
) -> dict[str, float]:
    """Split the Brier score into reliability, resolution and uncertainty.

        Brier = reliability - resolution + uncertainty

    Reliability is miscalibration (lower is better). Resolution is the ability
    to separate events that happen from those that do not (higher is better).
    Uncertainty is a property of the events themselves and no forecaster can
    change it, so a Brier difference driven entirely by uncertainty is not a
    skill difference at all.
    """
    f = np.asarray(forecasts, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    n = f.size
    if n == 0:
        return {"brier": float("nan"), "reliability": float("nan"),
                "resolution": float("nan"), "uncertainty": float("nan")}

    base = float(o.mean())
    uncertainty = base * (1.0 - base)

    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(f, edges[1:-1]), 0, bins - 1)

    reliability = resolution = 0.0
    for b in range(bins):
        mask = index == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_forecast = float(f[mask].mean())
        mean_outcome = float(o[mask].mean())
        reliability += count * (mean_forecast - mean_outcome) ** 2
        resolution += count * (mean_outcome - base) ** 2

    return {
        "brier": float(np.mean((f - o) ** 2)),
        "reliability": reliability / n,
        "resolution": resolution / n,
        "uncertainty": uncertainty,
    }


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Comparison:
    """The market against one baseline, as a paired test."""

    baseline: str
    market_score: float
    baseline_score: float
    mean_difference: float
    ci_low: float
    ci_high: float
    t_statistic: float
    p_value: float
    n: int
    p_adjusted: float = float("nan")

    @property
    def market_wins(self) -> bool:
        """Lower score is better, so the market wins on a negative difference."""
        return self.mean_difference < 0

    @property
    def significant(self) -> bool:
        return math.isfinite(self.p_adjusted) and self.p_adjusted < 0.05

    def __str__(self) -> str:
        verdict = (
            ("market better" if self.market_wins else "baseline better")
            if self.significant
            else "no difference"
        )
        return (
            f"  vs {self.baseline:<26} market {self.market_score:.5f}  "
            f"baseline {self.baseline_score:.5f}  "
            f"diff {self.mean_difference:+.5f} "
            f"[{self.ci_low:+.5f}, {self.ci_high:+.5f}]  "
            f"p={self.p_adjusted:.4f}  {verdict}"
        )


def paired_comparison(
    baseline_name: str,
    market_scores: Sequence[float],
    baseline_scores: Sequence[float],
    bootstrap: int = 10_000,
    seed: int = 0,
) -> Comparison:
    """Paired test on per-trial score differences.

    Paired because both forecasts are scored on the *same* trial, so trial
    difficulty cancels. That is the whole point: an unpaired comparison would
    be swamped by how hard the questions happened to be, and would need
    enormously more trials to see the same effect.

    Reports both a t-statistic and a bootstrap interval. The t-test assumes the
    mean difference is approximately normal, which holds by the central limit
    theorem at these sample sizes; the bootstrap does not assume it, and if the
    two disagree the bootstrap is the one to believe.
    """
    from scipy import stats

    m = np.asarray(market_scores, dtype=float)
    b = np.asarray(baseline_scores, dtype=float)
    mask = np.isfinite(m) & np.isfinite(b)
    m, b = m[mask], b[mask]
    n = m.size
    if n < 3:
        nan = float("nan")
        return Comparison(baseline_name, nan, nan, nan, nan, nan, nan, nan, n)

    differences = m - b
    mean_difference = float(differences.mean())
    t_statistic, p_value = stats.ttest_rel(m, b)

    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(bootstrap, n), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(draws, [2.5, 97.5])

    return Comparison(
        baseline=baseline_name,
        market_score=float(m.mean()),
        baseline_score=float(b.mean()),
        mean_difference=mean_difference,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        t_statistic=float(t_statistic),
        p_value=float(p_value),
        n=n,
    )


def benjamini_hochberg(comparisons: Sequence[Comparison]) -> list[Comparison]:
    """Control the false discovery rate across the ladder.

    Four comparisons are made against the same market, so at the usual 5% level
    one spurious "win" per twenty runs is expected by chance alone.
    Benjamini-Hochberg is the standard correction and is less brutal than
    Bonferroni, which matters when the comparisons are correlated. And these
    are, since every rung is computed from the same agents.
    """
    # A comparison whose p-value could not be computed is not a test, and
    # counting it as one corrupts every comparison that *was* computed.
    #
    # `paired_comparison` returns an all-NaN result below three trials rather
    # than raising, and scipy's paired t-test on two identical arms is 0/0, so
    # a NaN here is an ordinary thing to be handed rather than a pathology.
    #
    # Measured on p-values 0.001, 0.04, 0.30 and 0.90, which adjust to 0.004,
    # 0.080, 0.400 and 0.900. Adding a single NaN row moved them to 0.005,
    # 0.100, 0.500 and 1.000: it inflated `total` from four to five, so every
    # real p-value was multiplied by 5/4 and one of them crossed 0.05 in the
    # wrong direction. The NaN row itself came out as **1.0** rather than NaN,
    # because `min(previous, nan)` returns `previous`, so a comparison nobody
    # could compute reported "not significant" as though it had been tested.
    # Sorting was unreliable for the same reason: every comparison against NaN
    # is False, so the NaN sat wherever the sort happened to leave it.
    real = [
        index
        for index in range(len(comparisons))
        if math.isfinite(comparisons[index].p_value)
    ]
    ordered = sorted(real, key=lambda i: comparisons[i].p_value)
    total = len(real)
    adjusted = [float("nan")] * len(comparisons)
    previous = 1.0
    for rank, index in enumerate(reversed(ordered), start=1):
        position = total - rank + 1
        value = comparisons[index].p_value * total / position
        previous = min(previous, value)
        adjusted[index] = min(1.0, previous)
    return [
        replace(comparison, p_adjusted=adjusted[index])
        for index, comparison in enumerate(comparisons)
    ]
