"""Running a strategy, and being entitled to believe the number that comes back.

Anyone can run a strategy in a simulator and print a Sharpe ratio. The work is
in the four places that number is usually a lie, and this module exists to make
each of them impossible to skip rather than merely documented somewhere.

**The open is not a market.** The makers here anchor every book at the midpoint
of its settlement range, because a maker that opened on the answer would be the
thing that already knew. Measured on seed 3, a call struck just inside the
money opened at 2725 against a fair value of 119, and the win rate future it
was written on opened near 5000 against 4669.
A strategy that sells that open books a profit that is a property of the
builder, not of the strategy. So every evaluation here holds the strategy out of
the market for a configurable warmup and starts every statistic at the end of
it, and the paired control is a supported configuration rather than an argument:
run the same seeds with ``trade_during_warmup=True`` and read the difference.

**A comparison is only as good as its pairing.** This simulation is
bit-identical for a seed, and the kernel gives every agent its own random stream
derived from the seed and the agent's own id, so adding a strategy cannot shift
another agent's draws. That is unusual and it is the biggest asset here.
``Var(A-B) = Var(A) + Var(B) - 2Cov(A,B)``, so pairing pays exactly to the
extent that ``Cov > 0``, and it can only be positive if the seed drives the same
draws in both arms. Both arms therefore run under one agent id, one latency and
one seed list, and :class:`Paired` reports the realised covariance so the reader
can see whether the pairing actually bought anything instead of taking it on
faith.

**A Sharpe ratio without its sampling error is a number, not evidence.** Lo
(2002) gives ``SE(SR) = sqrt((1 + SR^2/2)/n)`` under normal iid returns and the
non-normal form ``sqrt([1 - skew*SR + ((kurt-1)/4)*SR^2]/n)`` otherwise, and the
relative error floors at ``1/sqrt(2n)`` however good the strategy is: at the 108
five-second periods a 600 second session yields, no strategy can be measured to
better than 6.8% of its own Sharpe. So the standard error is reported beside
every Sharpe, along with the probabilistic Sharpe ratio, the deflated Sharpe
against however many variants were tried, and the minimum track record length.
When the sample does not support a claim the result says so in words instead of
printing a number that invites one.

**Some statistics are not comparable across horizons at all.** With no edge the
expected maximum drawdown is ``1.2533 * sigma * sqrt(T)``, so it grows with the
length of the run and reverses rankings between a 300 second session and a 600
second one. Drawdown is therefore reported only when every run in the pool
shares a horizon, always beside the no-edge expectation, and Calmar is not
computed at all.

The rest is bookkeeping that has to be exact: P&L comes from the ledger rather
than from a reconstruction, the three-way decomposition comes from
:class:`~arena.research.attribution.TradeAttribution` and reconciles to it, and
a run writes a manifest carrying its config, its seeds, the commit it ran on and
a digest of its results, so that "reproducible" is a thing that can be checked
rather than a thing that is said.

Floats here, like in the attribution module and for the same reason. The ledger
stays integer minor units and every P&L figure is read out of it as an integer;
the ratios computed on top are differences of prices that nobody is ever paid.
"""

from __future__ import annotations

import math
import statistics
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from arena.agents.bayesian import BayesianFundamental
from arena.agents.fundamental import FundamentalTrader
from arena.agents.strategy_agent import StrategyAgent
from arena.determinism import canonical_json, digest
from arena.exchange.events import Filled
from arena.market.live import VENUE_ID
from arena.portfolio.money import MONEY_SCALE
from arena.research.aggregation import (
    Comparison,
    benjamini_hochberg,
    paired_comparison,
)
from arena.research.attribution import DEFAULT_HORIZONS, Decomposition, TradeAttribution
from arena.research.experiment import manifest_digest
from arena.sim.time import millis, seconds
from arena.strategies.base import TwoSided

__all__ = [
    "BacktestConfig",
    "Evaluation",
    "Manifest",
    "Paired",
    "RunResult",
    "SampleStats",
    "backtest",
    "compare",
    "compare_many",
    "deflated_sharpe",
    "expected_max_drawdown",
    "hit_rate",
    "implied_hit_rate",
    "manifest",
    "max_drawdown",
    "min_track_record_length",
    "pair",
    "probabilistic_sharpe",
    "replay",
    "run_once",
    "sharpe_ratio",
    "sharpe_standard_error",
    "sortino_ratio",
    "summarise",
    "verify_reproducible",
]


# Euler-Mascheroni, which is where it enters: the expected maximum of ``N``
# draws from a standard normal is asymptotically ``(1-g)*Z^-1[1-1/N] +
# g*Z^-1[1-1/(N e)]``, and the deflated Sharpe ratio is the probabilistic
# Sharpe ratio measured against that expected maximum rather than against zero.
EULER_MASCHERONI = 0.5772156649015329

# The constant in E[MDD] for a driftless random walk, 1.2533 = sqrt(pi/2). It is
# here so that a drawdown figure always arrives next to what a strategy with no
# edge at all would have produced over the same horizon.
MDD_CONSTANT = math.sqrt(math.pi / 2)

# Which participants in the built market were told what a contract settles at.
# Identified by type rather than by name, because a name is a fact about one
# builder and the question "how much of my passive volume came from somebody who
# knew more than me" is Glosten-Milgrom's mu, which is a fact about the
# population. This is the one measurement here that a real venue cannot make.
INFORMED_TYPES: tuple[type, ...] = (FundamentalTrader, BayesianFundamental)

_NORMAL = statistics.NormalDist()

_REPO = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------
# Statistics on a return series
#
# Every function here is pure and takes the series it is told about, so the
# tests can hand them a hand-built series whose answer is known by arithmetic
# rather than by running a market.
# --------------------------------------------------------------------------


def _moments(returns: Sequence[float]) -> tuple[float, float, float, float]:
    """Mean, sample standard deviation, skew and raw kurtosis of a series.

    The standard deviation carries ``ddof=1`` because that is the Sharpe ratio
    everybody means, and the third and fourth moments are the population
    versions, which is what the probabilistic Sharpe ratio's derivation uses.
    The mismatch is deliberate and named rather than silently averaged over:
    matching them by using ddof=1 throughout would change PSR by O(1/n) and
    would no longer be the published estimator.
    """
    n = len(returns)
    if n < 2:
        return (float(returns[0]) if n else float("nan"), float("nan"),
                float("nan"), float("nan"))
    mean = sum(returns) / n
    centred = [r - mean for r in returns]
    m2 = sum(c * c for c in centred) / n
    if m2 <= 0.0:
        return mean, 0.0, float("nan"), float("nan")
    sample_sd = math.sqrt(sum(c * c for c in centred) / (n - 1))
    m3 = sum(c ** 3 for c in centred) / n
    m4 = sum(c ** 4 for c in centred) / n
    return mean, sample_sd, m3 / m2 ** 1.5, m4 / (m2 * m2)


def sharpe_ratio(returns: Sequence[float]) -> float:
    """Mean over standard deviation, per period, and not annualised.

    Not annualised because there is nothing here to annualise against. A period
    is a slice of simulated seconds chosen by the caller, and multiplying by
    ``sqrt(252)`` would be inventing a calendar. Every derived quantity below is
    in the same per-period units, which is also the unit Lo's standard error and
    Bailey and Lopez de Prado's track record length are written in.
    """
    mean, sd, _, _ = _moments(returns)
    if not math.isfinite(sd) or sd <= 0.0:
        return float("nan")
    return mean / sd


def sharpe_standard_error(
    sharpe: float,
    n: int,
    skew: float | None = None,
    kurtosis: float | None = None,
) -> float:
    """Lo (2002), in the general form when the moments are known.

    ``SE = sqrt([1 - skew*SR + ((kurt-1)/4)*SR^2] / n)`` with ``kurt`` the raw
    fourth moment ratio, 3 for a Gaussian. Substituting skew 0 and kurtosis 3
    recovers ``sqrt((1 + SR^2/2)/n)``, which is what the two ``None`` defaults
    give, so the iid-normal case is a special case of one expression rather than
    a second one that can drift away from it.
    """
    if n < 2 or not math.isfinite(sharpe):
        return float("nan")
    g3 = 0.0 if skew is None or not math.isfinite(skew) else skew
    g4 = 3.0 if kurtosis is None or not math.isfinite(kurtosis) else kurtosis
    variance = 1.0 - g3 * sharpe + 0.25 * (g4 - 1.0) * sharpe * sharpe
    if variance <= 0.0:
        return float("nan")
    return math.sqrt(variance / n)


def probabilistic_sharpe(
    sharpe: float,
    n: int,
    skew: float | None = None,
    kurtosis: float | None = None,
    benchmark: float = 0.0,
) -> float:
    """The probability that the true Sharpe exceeds ``benchmark``.

    Bailey and Lopez de Prado: the observed ratio divided by its own standard
    error, read through the normal CDF. It answers the question a Sharpe ratio
    is usually asked to answer and almost never does, which is whether the
    sample is long enough for the number to mean anything.
    """
    if n < 2 or not math.isfinite(sharpe):
        return float("nan")
    g3 = 0.0 if skew is None or not math.isfinite(skew) else skew
    g4 = 3.0 if kurtosis is None or not math.isfinite(kurtosis) else kurtosis
    variance = 1.0 - g3 * sharpe + 0.25 * (g4 - 1.0) * sharpe * sharpe
    if variance <= 0.0:
        return float("nan")
    return _NORMAL.cdf((sharpe - benchmark) * math.sqrt(n - 1) / math.sqrt(variance))


def expected_maximum_sharpe(trials: int, trial_variance: float) -> float:
    """What the best of ``trials`` variants scores when none of them has an edge.

    The order statistic, not a correction factor: draw ``N`` Sharpe ratios from
    a zero-mean distribution and the largest is around
    ``sqrt(V) * [(1-g) Z^-1(1 - 1/N) + g Z^-1(1 - 1/(N e))]``. At one trial
    there is no maximum to take and the expectation of a single zero-mean draw
    is zero, which is the value returned rather than the divergent limit of the
    formula.
    """
    if trials <= 1 or not math.isfinite(trial_variance) or trial_variance <= 0.0:
        return 0.0
    n = float(trials)
    upper = _NORMAL.inv_cdf(1.0 - 1.0 / n)
    lower = _NORMAL.inv_cdf(1.0 - 1.0 / (n * math.e))
    return math.sqrt(trial_variance) * ((1.0 - EULER_MASCHERONI) * upper
                                        + EULER_MASCHERONI * lower)


def deflated_sharpe(
    sharpe: float,
    n: int,
    trials: int,
    skew: float | None = None,
    kurtosis: float | None = None,
    trial_variance: float | None = None,
) -> float:
    """The probabilistic Sharpe measured against the best of ``trials`` nulls.

    The point of the deflation is that somebody using a testbed will try many
    things, and the best of many worthless things looks good. Bailey and Lopez
    de Prado's worked example is the whole argument: the same track record
    passes at 46 trials and fails at 100, so a result reported without the
    number of variants behind it is not a result.

    ``trial_variance`` is the dispersion of the Sharpe ratios actually tried,
    which is the right input and which the caller usually has. When it is not
    supplied this falls back to the variance of the estimator itself,
    ``(1 - skew*SR + ((kurt-1)/4)*SR^2)/n``, which deflates only for estimation
    noise. That fallback understates the deflation whenever the variants tried
    were genuinely different from one another, and callers are told so in the
    notes rather than left to infer it.
    """
    if n < 2 or not math.isfinite(sharpe):
        return float("nan")
    if trial_variance is None:
        error = sharpe_standard_error(sharpe, n, skew, kurtosis)
        trial_variance = float("nan") if not math.isfinite(error) else error * error
    floor = expected_maximum_sharpe(trials, trial_variance)
    return probabilistic_sharpe(sharpe, n, skew, kurtosis, benchmark=floor)


def min_track_record_length(
    sharpe: float,
    skew: float | None = None,
    kurtosis: float | None = None,
    benchmark: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """How many periods it would take to believe this Sharpe at ``confidence``.

    ``1 + [1 - skew*SR + ((kurt-1)/4)*SR^2] * (Z_alpha / (SR - SR*))^2``. It is
    the honest counterpart to the Sharpe itself: a 600 second session is short,
    and whether it is long enough to support the claim being made is a live
    question rather than a formality.
    """
    if not math.isfinite(sharpe) or sharpe <= benchmark:
        return float("inf")
    g3 = 0.0 if skew is None or not math.isfinite(skew) else skew
    g4 = 3.0 if kurtosis is None or not math.isfinite(kurtosis) else kurtosis
    variance = 1.0 - g3 * sharpe + 0.25 * (g4 - 1.0) * sharpe * sharpe
    if variance <= 0.0:
        return float("inf")
    z = _NORMAL.inv_cdf(confidence)
    return 1.0 + variance * (z / (sharpe - benchmark)) ** 2


def sortino_ratio(returns: Sequence[float], target: float = 0.0) -> float:
    """Mean excess over downside deviation, divided by ``N``.

    By ``N``, the number of periods, and not by the number of losing ones. The
    second is the common implementation and it is wrong: it rewards a strategy
    for having few losses twice, once in the numerator and again by shrinking
    its own denominator, so a series with one bad period and ninety-nine flat
    ones scores as though it never took risk at all.

    Not annualised, and specifically not by ``sqrt(12)``. Downside deviation is
    not the standard deviation of anything and does not scale that way.
    """
    n = len(returns)
    if n < 2:
        return float("nan")
    mean = sum(returns) / n - target
    downside = math.sqrt(sum(min(r - target, 0.0) ** 2 for r in returns) / n)
    if downside <= 0.0:
        return float("nan")
    return mean / downside


def hit_rate(returns: Sequence[float]) -> tuple[float, int]:
    """The fraction of periods that made money, and how many periods that was.

    Returned as a pair so that the count cannot be dropped on the way to a
    report. A hit rate on its own is uninformative in the strict sense that the
    same value is evidence of opposite things at different ``n``.
    """
    n = len(returns)
    if not n:
        return float("nan"), 0
    return sum(1 for r in returns if r > 0.0) / n, n


def implied_hit_rate(sharpe: float, n: int) -> float:
    """The hit rate a symmetric strategy would need for this Sharpe over ``n``.

    ``p = 0.5*(1 + sqrt(theta^2/(n + theta^2)))``. This is the number that makes
    a hit rate readable: a per-period Sharpe of 2 needs 0.5316 over a thousand
    periods and 0.6361 over fifty, so 55% is excellent in one column and
    unremarkable in the other, and the column is the sample size.
    """
    if n <= 0 or not math.isfinite(sharpe):
        return float("nan")
    return 0.5 * (1.0 + math.sqrt(sharpe * sharpe / (n + sharpe * sharpe)))


def expected_max_drawdown(sigma: float, periods: int) -> float:
    """``1.2533 * sigma * sqrt(T)``, the drawdown of having no edge at all.

    Reported next to any realised drawdown because the realised figure grows
    with the horizon whether or not the strategy is any good, which is why
    drawdown and Calmar cannot be compared across runs of different lengths and
    why nothing here ranks on them.
    """
    if periods <= 0 or not math.isfinite(sigma):
        return float("nan")
    return MDD_CONSTANT * sigma * math.sqrt(periods)


def max_drawdown(equity: Sequence[float]) -> float:
    """Largest peak-to-trough fall in an equity path, in the path's own units."""
    peak = -math.inf
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        worst = min(worst, value - peak)
    return 0.0 if worst == 0.0 else -worst


# --------------------------------------------------------------------------
# The summary of one return series, with what it does and does not support
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleStats:
    """Everything a return series supports, and a plain statement of what it does not.

    ``notes`` is the load-bearing field. A result object that only carried
    numbers would let a reader take a Sharpe of 3.1 from twelve periods at face
    value, and twelve periods cannot distinguish that from zero.
    """

    n: int
    mean: float
    volatility: float
    skew: float
    kurtosis: float
    sharpe: float
    sharpe_error: float
    relative_error: float
    relative_error_floor: float
    psr: float
    dsr: float
    trials: int
    min_track_record: float
    sortino: float
    hit_rate: float
    required_hit_rate: float
    confidence: float
    notes: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        """Whether the sample is long enough to make the claim it looks like.

        Both conditions, because they fail separately. The track record length
        asks whether ``n`` periods could ever establish this Sharpe at this
        confidence; the deflated ratio asks whether it survives the number of
        variants tried. A strategy can pass either alone and be nothing.
        """
        return (
            math.isfinite(self.sharpe)
            and self.n >= self.min_track_record
            and math.isfinite(self.dsr)
            and self.dsr >= self.confidence
        )

    def describe(self) -> str:
        """One line, which refuses to print a verdict the sample cannot carry."""
        if not math.isfinite(self.sharpe):
            return f"Sharpe undefined over n={self.n} ({self.notes[0] if self.notes else ''})"
        need = ("never" if not math.isfinite(self.min_track_record)
                else f"{self.min_track_record:.0f}")
        verdict = "supported" if self.supported else "NOT supported by this sample"
        return (
            f"Sharpe {self.sharpe:+.3f} +/- {self.sharpe_error:.3f} per period "
            f"(n={self.n}, floor {self.relative_error_floor:.1%}), "
            f"PSR {self.psr:.3f}, DSR {self.dsr:.3f} over {self.trials} trial(s), "
            f"MinTRL {need}: {verdict}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "volatility": self.volatility,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "sharpe": self.sharpe,
            "sharpe_error": self.sharpe_error,
            "relative_error": self.relative_error,
            "relative_error_floor": self.relative_error_floor,
            "psr": self.psr,
            "dsr": self.dsr,
            "trials": self.trials,
            "min_track_record": self.min_track_record,
            "sortino": self.sortino,
            "hit_rate": self.hit_rate,
            "required_hit_rate": self.required_hit_rate,
            "confidence": self.confidence,
            "supported": self.supported,
            "notes": list(self.notes),
        }


def summarise(
    returns: Sequence[float],
    trials: int = 1,
    trial_variance: float | None = None,
    benchmark: float = 0.0,
    confidence: float = 0.95,
) -> SampleStats:
    """Turn a per-period return series into what it does and does not support."""
    n = len(returns)
    mean, sd, skew, kurt = _moments(returns)
    sr = sharpe_ratio(returns)
    error = sharpe_standard_error(sr, n, skew, kurt)
    relative = abs(error / sr) if math.isfinite(sr) and sr else float("nan")
    floor = 1.0 / math.sqrt(2 * n) if n > 0 else float("nan")
    trl = min_track_record_length(sr, skew, kurt, benchmark, confidence)
    rate, count = hit_rate(returns)

    notes: list[str] = []
    if n < 2:
        notes.append(f"n={n}: a return series this short has no dispersion to measure")
    elif not math.isfinite(sd) or sd == 0.0:
        notes.append(
            "every period returned exactly the same amount, so the Sharpe ratio "
            "is a division by zero rather than a large number"
        )
    if math.isfinite(sr) and math.isfinite(trl) and n < trl:
        notes.append(
            f"n={n} periods against a minimum track record of {trl:.0f}: this "
            f"sample cannot establish a Sharpe of {sr:.3f} at {confidence:.0%}"
        )
    if math.isfinite(sr) and not math.isfinite(trl):
        notes.append(
            f"a Sharpe of {sr:.3f} does not beat the benchmark of {benchmark:.3f}, "
            "so no track record length would establish it"
        )
    if trial_variance is None and trials > 1:
        notes.append(
            f"the deflation over {trials} trials used the variance of the Sharpe "
            "estimator, not the dispersion of the variants actually tried, which "
            "understates it when the variants differed"
        )
    if math.isfinite(rate) and count:
        notes.append(
            f"hit rate {rate:.1%} over n={count}; "
            f"{implied_hit_rate(sr, count):.1%} is what a Sharpe of magnitude "
            f"{abs(sr):.3f} takes at this n"
            if math.isfinite(sr)
            else f"hit rate {rate:.1%} over n={count}"
        )
    return SampleStats(
        n=n,
        mean=mean,
        volatility=sd,
        skew=skew,
        kurtosis=kurt,
        sharpe=sr,
        sharpe_error=error,
        relative_error=relative,
        relative_error_floor=floor,
        psr=probabilistic_sharpe(sr, n, skew, kurt, benchmark),
        dsr=deflated_sharpe(sr, n, trials, skew, kurt, trial_variance),
        trials=trials,
        min_track_record=trl,
        sortino=sortino_ratio(returns),
        hit_rate=rate,
        required_hit_rate=implied_hit_rate(sr, count),
        confidence=confidence,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestConfig:
    """Everything that determines a run, and nothing that does not.

    Frozen and canonically serialisable, so the manifest is a complete
    description: if a field is not here it cannot influence the result, which is
    what makes the digest check mean something.
    """

    # Simulated seconds of session, and how many of them are thrown away. The
    # default warmup is a tenth of the default session, which on this market is
    # enough for the opening dislocation to be arbitraged away: measured on seed
    # 3, the call this module's docstring opens on left its 2725 within the
    # first 20 seconds.
    until: float = 600.0
    warmup: float = 60.0
    # The grid the return series is sampled on. Five seconds gives 108 periods
    # over a 600 second session after warmup, which floors the Sharpe's relative
    # error at 1/sqrt(2*108) = 6.8%. Shorter periods buy a smaller floor and a
    # noisier per-period series; this is the trade and it is the caller's.
    return_interval: float = 5.0
    # The grid the attribution's mid series is sampled on, which is also the
    # staleness of the mid each fill is attributed to. Half a second because the
    # sampler reads every book on every call and measured at 6.9ms per book
    # snapshot across 47 symbols, a quarter-second grid costs more wall time
    # than the market it is watching.
    sample_interval: float = 0.5
    # Which contracts the strategy may see. None is every listed contract.
    symbols: tuple[str, ...] | None = None
    # How often the strategy is asked, and how far from the exchange it sits.
    # The built market's makers are at 0.15 to 0.23ms and its informed traders
    # at 3 to 13ms, so a strategy at 5ms is a serious desk that is not
    # colocated, and it will be picked off by the makers accordingly. The number
    # is here rather than inherited from the wire default so that it is a choice
    # somebody made.
    wake_ms: float = 320.0
    latency_ms: float = 5.0
    starting_cash: int = 20_000_000
    agent_id: str = "strategy"
    # Both arms of a comparison must use one id, because the kernel seeds an
    # agent's random stream and its latency jitter from that id. Two ids would
    # give the two arms different draws and destroy the pairing this whole
    # module is built on.
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    attribution_horizon: int = int(seconds(1))
    # The paired control for the opening auction. Off by default: a strategy
    # that trades the open carries the position it took there into the measured
    # window, so excluding the warmup from the statistics alone does not exclude
    # the artefact.
    trade_during_warmup: bool = False
    confidence: float = 0.95
    benchmark_sharpe: float = 0.0
    # Keyword arguments forwarded to dashboard.build_market.build, as pairs so
    # the config stays hashable and canonically serialisable.
    market: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.until <= self.warmup:
            raise ValueError(
                f"a warmup of {self.warmup}s leaves nothing of a {self.until}s run"
            )
        if self.sample_interval <= 0 or self.return_interval <= 0:
            raise ValueError("sampling intervals must be positive")
        ratio = self.return_interval / self.sample_interval
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(
                f"return_interval {self.return_interval}s is not a whole number of "
                f"{self.sample_interval}s sampling steps"
            )

    @property
    def market_kwargs(self) -> dict[str, Any]:
        return dict(self.market)

    @property
    def report_horizon(self) -> int:
        """The horizon the headline decomposition is read at.

        Raised to the sampling grid when the caller asked for something finer,
        because the alternative is a report that asks the attribution for a
        horizon it was never able to collect and dies forty minutes into a run.
        """
        return max(int(seconds(self.sample_interval)), int(self.attribution_horizon))

    @property
    def collected_horizons(self) -> tuple[int, ...]:
        """Horizons the sampler can actually resolve, shortest first.

        A horizon below the sampling interval would be measuring the grid rather
        than the market, so it is dropped and said so in the notes instead of
        being collected and quietly meaning something else. The reported horizon
        is always in the set, so the ladder and the headline number can never
        disagree about what was measured.
        """
        floor = int(seconds(self.sample_interval))
        kept = {h for h in self.horizons if h >= floor}
        kept.add(self.report_horizon)
        return tuple(sorted(kept))

    def evolve(self, **changes: Any) -> BacktestConfig:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "until": self.until,
            "warmup": self.warmup,
            "return_interval": self.return_interval,
            "sample_interval": self.sample_interval,
            "symbols": None if self.symbols is None else list(self.symbols),
            "wake_ms": self.wake_ms,
            "latency_ms": self.latency_ms,
            "starting_cash": self.starting_cash,
            "agent_id": self.agent_id,
            "horizons": [int(h) for h in self.horizons],
            "attribution_horizon": int(self.attribution_horizon),
            "trade_during_warmup": self.trade_during_warmup,
            "confidence": self.confidence,
            "benchmark_sharpe": self.benchmark_sharpe,
            "market": [[k, v] for k, v in self.market],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BacktestConfig:
        symbols = payload.get("symbols")
        return cls(
            until=float(payload["until"]),
            warmup=float(payload["warmup"]),
            return_interval=float(payload["return_interval"]),
            sample_interval=float(payload["sample_interval"]),
            symbols=None if symbols is None else tuple(symbols),
            wake_ms=float(payload["wake_ms"]),
            latency_ms=float(payload["latency_ms"]),
            starting_cash=int(payload["starting_cash"]),
            agent_id=str(payload["agent_id"]),
            horizons=tuple(int(h) for h in payload["horizons"]),
            attribution_horizon=int(payload["attribution_horizon"]),
            trade_during_warmup=bool(payload["trade_during_warmup"]),
            confidence=float(payload["confidence"]),
            benchmark_sharpe=float(payload["benchmark_sharpe"]),
            market=tuple((str(k), v) for k, v in payload.get("market", ())),
        )


# --------------------------------------------------------------------------
# Wearing the strategy
# --------------------------------------------------------------------------


class _Quarantine:
    """Holds a strategy out of the market until the warmup has elapsed.

    Written against the strategy protocols rather than against the agent,
    because that is the only boundary that is guaranteed to catch everything a
    strategy can do. Gating the agent instead would leave a strategy that
    quotes on a fill still quoting during the auction.

    ``symbols`` returning nothing is what actually keeps it out: the adapter
    never asks for a quote on a contract the strategy did not list, so no order
    is composed, snapped or sent, and the warmup costs nothing.
    """

    def __init__(self, inner: Any, start_at: float) -> None:
        self.inner = inner
        self.start_at = start_at

    def _open(self, view: Any) -> bool:
        return view.now >= self.start_at

    def quote(self, view: Any, symbol: str) -> TwoSided:
        if not self._open(view):
            return TwoSided()
        return self.inner.quote(view, symbol)

    def symbols(self, view: Any) -> Sequence[str]:
        if not self._open(view):
            return ()
        chooser = getattr(self.inner, "symbols", None)
        return view.symbols if chooser is None else chooser(view)

    def orders(self, view: Any) -> Sequence[Any]:
        if not self._open(view):
            return ()
        return self.inner.orders(view)


@dataclass(frozen=True)
class _Fill:
    at: int
    symbol: str
    signed: int
    price_minor: int
    aggressor: bool


class _RecordingAgent(StrategyAgent):
    """A strategy agent that also keeps its own blotter.

    The venue keeps only the last sixty fills per participant and the
    attribution's record is restricted to fills that reached a horizon, so
    neither can answer "how many lots did this trade over the measured window".
    A desk answers that from its own blotter and so does this: the fills are the
    ones the agent was told about, at the prices it was told, which keeps the
    figure inside what the strategy itself is allowed to know.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.blotter: list[_Fill] = []

    def _on_private(self, ctx: Any, event: Any, symbol: str) -> None:
        super()._on_private(ctx, event, symbol)
        if not isinstance(event, Filled) or event.price is None:
            return
        if symbol not in self.instruments:
            return
        instrument = self.instruments[symbol]
        signed = int(event.quantity) * (1 if event.side.value == "buy" else -1)
        self.blotter.append(
            _Fill(
                at=int(ctx.now),
                symbol=symbol,
                signed=signed,
                price_minor=int(event.price) * int(instrument.tick_in_minor),
                aggressor=bool(event.aggressor),
            )
        )


def _split(strategy: Any) -> tuple[Any | None, Any | None]:
    """Decide whether a strategy is a maker, a taker, or both.

    By the presence of the methods rather than by ``isinstance`` against the
    protocols. ``MakerStrategy`` requires ``symbols`` as well as ``quote``, but
    the adapter treats ``symbols`` as optional and falls back to everything the
    agent lists, so an isinstance test here would refuse a maker the runtime
    accepts and the harness would be stricter than the thing it measures.
    """
    maker = strategy if callable(getattr(strategy, "quote", None)) else None
    taker = strategy if callable(getattr(strategy, "orders", None)) else None
    if maker is None and taker is None:
        raise TypeError(
            f"{type(strategy).__name__} is neither a maker nor a taker: it needs "
            "a quote() for the maker protocol or an orders() for the taker one"
        )
    return maker, taker


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    """One strategy, one seed, everything measured after the warmup."""

    seed: int
    agent_id: str
    # Post-warmup, in ledger minor units, read out of the account rather than
    # reconstructed. The starting point is the equity at the end of the warmup,
    # so nothing the opening auction did is in here.
    pnl: int
    opening_equity: int
    closing_equity: int
    warmup_pnl: int
    returns: tuple[float, ...]
    equity: tuple[int, ...]
    decomposition: Decomposition
    adverse_curve: dict[int, float]
    informed_share: float
    flow_imbalance: dict[str, float]
    lots: int
    fills: int
    aggressive_lots: int
    notional: int
    conservation: int
    symbols_traded: tuple[str, ...]

    @property
    def aggressor_fraction(self) -> float:
        return self.aggressive_lots / self.lots if self.lots else float("nan")

    @property
    def turnover(self) -> float:
        """Traded notional as a multiple of the equity it started the window with."""
        if not self.opening_equity:
            return float("nan")
        return self.notional / self.opening_equity

    def summary(self, trials: int = 1, confidence: float = 0.95) -> SampleStats:
        """This one seed's statistics, on demand rather than stored.

        Computed here rather than held on the object because one seed is almost
        never the sample anybody should be quoting: at the default five second
        period a single 600 second session gives 108 returns, and the pooled
        figure over eight seeds gives 864. Both are available and the per-seed
        one is what tells you whether a pooled result is one lucky seed.
        """
        return summarise(self.returns, trials=trials, confidence=confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "agent_id": self.agent_id,
            "pnl": self.pnl,
            "opening_equity": self.opening_equity,
            "closing_equity": self.closing_equity,
            "warmup_pnl": self.warmup_pnl,
            "periods": len(self.returns),
            "decomposition": self.decomposition.to_dict(),
            "adverse_curve": {str(k): v for k, v in sorted(self.adverse_curve.items())},
            "informed_share": self.informed_share,
            "flow_imbalance": dict(sorted(self.flow_imbalance.items())),
            "lots": self.lots,
            "fills": self.fills,
            "aggressive_lots": self.aggressive_lots,
            "notional": self.notional,
            "conservation": self.conservation,
            "symbols_traded": list(self.symbols_traded),
        }


def _build_market(config: BacktestConfig, seed: int) -> Any:
    # Imported here rather than at module scope. The builder pulls in the whole
    # dashboard package and the fixture dataset, and a research module that
    # cannot be imported without them is a research module that cannot be
    # imported from a notebook that only wanted the statistics above.
    from dashboard.build_market import build

    return build(seed=seed, **config.market_kwargs)


def run_once(strategy: Any, seed: int, config: BacktestConfig) -> RunResult:
    """Run one strategy on one seed and measure everything after the warmup."""
    market = _build_market(config, seed)
    registry = market.venue.registry
    listed = tuple(registry.symbols)
    wanted = listed if config.symbols is None else tuple(
        s for s in listed if s in set(config.symbols)
    )
    if not wanted:
        raise ValueError(
            f"none of {config.symbols} is listed; this market lists {listed[:5]}..."
        )
    by_symbol = {symbol: registry.require(symbol) for symbol in wanted}

    agent_id = config.agent_id
    market.venue.open_account(agent_id, Decimal(config.starting_cash))
    if market.latency is not None:
        # Named rather than inherited. Without this the strategy silently takes
        # the wire default, which on this market is 4ms, and the run would be
        # reporting a latency nobody chose.
        market.latency.per_agent[agent_id] = millis(config.latency_ms)

    start_at = 0.0 if config.trade_during_warmup else config.warmup
    maker, taker = _split(strategy)
    gate = _Quarantine(strategy, start_at)
    agent = _RecordingAgent(
        agent_id,
        VENUE_ID,
        by_symbol,
        millis(config.wake_ms),
        maker=gate if maker is not None else None,
        taker=gate if taker is not None else None,
        starting_cash=Decimal(config.starting_cash),
    )
    market.kernel.add(agent)

    informed = frozenset(
        str(a.agent_id) for a in market.agents if isinstance(a, INFORMED_TYPES)
    )
    attribution = TradeAttribution(
        market.venue,
        horizons=config.collected_horizons,
        # Restricted to this strategy. The counterparty breakdown still adds up,
        # because it is computed per fill from both sides of the print, and the
        # open-fill list stays short enough that the sampler is linear in the
        # window rather than in the whole market's volume.
        agents=frozenset({agent_id}),
    )

    step = int(seconds(config.sample_interval))
    total_steps = max(1, round(config.until / config.sample_interval))
    warmup_steps = round(config.warmup / config.sample_interval)
    per_return = round(config.return_interval / config.sample_interval)
    # One sample before the window opens, so the first fill inside it is
    # attributed to a mid that was actually standing rather than to its own
    # print price. Sampling the whole warmup would cost a book snapshot per
    # symbol per step and buy nothing.
    prime_step = max(0, warmup_steps - 1)

    equity_path: list[int] = []
    marks_for = tuple(by_symbol)
    account = market.venue.account(agent_id)

    def equity_now() -> int:
        marks = {symbol: market.venue.mark(symbol) for symbol in marks_for}
        return int(account.equity(marks))

    market.kernel.start()
    attached = False
    opening_equity = 0
    if warmup_steps == 0:
        # The window opens before the first slice of time is run, not after it.
        # Reading the opening equity one step in instead would put the opening
        # auction inside the warmup even when the caller asked for none, which
        # would silently disable the very control this flag exists to provide.
        attribution.attach()
        attribution.sample(0)
        attached = True
        opening_equity = equity_now()
        equity_path.append(opening_equity)
    for index in range(1, total_steps + 1):
        now = index * step
        market.kernel.advance(until=now)
        if index >= warmup_steps and not attached:
            attribution.attach()
            attached = True
            opening_equity = equity_now()
            equity_path.append(opening_equity)
        if index >= prime_step:
            attribution.sample(now)
        if attached and index > warmup_steps and (index - warmup_steps) % per_return == 0:
            equity_path.append(equity_now())
    market.kernel.finish()
    attribution.detach()

    closing_equity = equity_path[-1] if equity_path else equity_now()
    base = opening_equity or 1
    returns = tuple(
        (equity_path[i] - equity_path[i - 1]) / base for i in range(1, len(equity_path))
    )

    row = attribution.report(config.report_horizon).get(agent_id)
    if row is None:
        row = Decomposition(agent_id, config.report_horizon)

    window_start = warmup_steps * step
    mine = [f for f in agent.blotter if f.at >= window_start]
    lots = sum(abs(f.signed) for f in mine)
    notional = sum(abs(f.signed) * f.price_minor for f in mine)
    aggressive = sum(abs(f.signed) for f in mine if f.aggressor)

    return RunResult(
        seed=seed,
        agent_id=agent_id,
        pnl=closing_equity - opening_equity,
        opening_equity=opening_equity,
        closing_equity=closing_equity,
        warmup_pnl=opening_equity - int(account.starting_cash),
        returns=returns,
        equity=tuple(equity_path),
        decomposition=row,
        adverse_curve=attribution.curve(agent_id),
        informed_share=attribution.informed_share(agent_id, informed),
        flow_imbalance=attribution.flow_imbalance(agent_id),
        lots=lots,
        fills=len(mine),
        aggressive_lots=aggressive,
        notional=notional,
        conservation=int(market.venue.conservation_check()),
        symbols_traded=tuple(sorted({f.symbol for f in mine})),
    )


# --------------------------------------------------------------------------
# Many seeds
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Evaluation:
    """One strategy over a set of seeds, pooled and honest about the pooling."""

    name: str
    config: BacktestConfig
    seeds: tuple[int, ...]
    runs: tuple[RunResult, ...]
    stats: SampleStats
    notes: tuple[str, ...] = ()

    @property
    def pnl(self) -> int:
        """Total post-warmup P&L across every seed, in ledger minor units."""
        return sum(r.pnl for r in self.runs)

    @property
    def pnl_per_seed(self) -> tuple[int, ...]:
        return tuple(r.pnl for r in self.runs)

    @property
    def mean_pnl(self) -> float:
        return self.pnl / len(self.runs) if self.runs else float("nan")

    @property
    def attribution(self) -> dict[str, float]:
        """The decomposition summed over seeds, in ledger minor units.

        Summed rather than averaged because the three terms are money and money
        adds. Read them against each other and not against zero: the diagnosis
        is which term is negative, since the identity guarantees they sum to the
        trading P&L on the fills that matured.
        """
        keys = ("spread_captured", "adverse_selection", "realized_spread",
                "residual", "total")
        pooled = {k: sum(getattr(r.decomposition, k) for r in self.runs) for k in keys}
        pooled["fills"] = float(sum(r.decomposition.fills for r in self.runs))
        pooled["lots"] = float(sum(r.decomposition.lots for r in self.runs))
        return pooled

    @property
    def lots(self) -> int:
        return sum(r.lots for r in self.runs)

    @property
    def turnover(self) -> float:
        opening = sum(r.opening_equity for r in self.runs)
        return sum(r.notional for r in self.runs) / opening if opening else float("nan")

    @property
    def aggressor_fraction(self) -> float:
        lots = self.lots
        if not lots:
            return float("nan")
        return sum(r.aggressive_lots for r in self.runs) / lots

    @property
    def informed_share(self) -> float:
        """Lot-weighted across seeds, so a quiet seed does not count as loudly."""
        weights = [r.decomposition.passive_lots for r in self.runs]
        total = sum(weights)
        if not total:
            return float("nan")
        pairs = zip(self.runs, weights, strict=True)
        return sum(r.informed_share * w for r, w in pairs) / total

    @property
    def drawdown(self) -> float | None:
        """The worst peak-to-trough fall on any one seed, or ``None``.

        ``None`` whenever the runs do not share a horizon. With no edge the
        expected maximum drawdown is ``1.2533*sigma*sqrt(T)``, so it grows with
        the length of a run and ranking on it across different lengths reverses
        the ranking. Returning a number here would invite exactly that.
        """
        lengths = {len(r.equity) for r in self.runs}
        if len(lengths) != 1 or not self.runs:
            return None
        return max(max_drawdown([float(v) for v in r.equity]) for r in self.runs)

    @property
    def drawdown_with_no_edge(self) -> float | None:
        """What a driftless walk of the same volatility would have drawn down."""
        if self.drawdown is None:
            return None
        base = self.runs[0].opening_equity or 1
        sigma = self.stats.volatility * base
        return expected_max_drawdown(sigma, len(self.runs[0].equity) - 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seeds": list(self.seeds),
            "pnl": self.pnl,
            "pnl_per_seed": list(self.pnl_per_seed),
            "runs": [r.to_dict() for r in self.runs],
            "stats": self.stats.to_dict(),
            "attribution": self.attribution,
            "lots": self.lots,
            "turnover": self.turnover,
            "aggressor_fraction": self.aggressor_fraction,
            "informed_share": self.informed_share,
            "notes": list(self.notes),
        }

    def __str__(self) -> str:
        money = self.pnl / MONEY_SCALE

        def pct(value: float) -> str:
            # A strategy that never traded has no aggressor fraction and no
            # informed share, and "nan%" reads as a broken calculation rather
            # than as an absent one.
            return "n/a" if not math.isfinite(value) else f"{value:.1%}"

        lines = [
            f"{self.name}: {money:+,.2f} over {len(self.runs)} seed(s), "
            f"{self.lots:,} lots, turnover {self.turnover:.2f}x, "
            f"aggressor {pct(self.aggressor_fraction)}, "
            f"informed share of passive volume {pct(self.informed_share)}",
            "  " + self.stats.describe(),
        ]
        decomposition = self.attribution
        lines.append(
            "  spread {spread:+,.2f}  adverse {adverse:+,.2f}  "
            "realized {realized:+,.2f}  residual {residual:+,.2f}".format(
                spread=decomposition["spread_captured"] / MONEY_SCALE,
                adverse=decomposition["adverse_selection"] / MONEY_SCALE,
                realized=decomposition["realized_spread"] / MONEY_SCALE,
                residual=decomposition["residual"] / MONEY_SCALE,
            )
        )
        drawdown = self.drawdown
        expected = self.drawdown_with_no_edge
        if drawdown is None:
            lines.append("  drawdown withheld: the runs do not share a horizon")
        elif expected is None or not math.isfinite(expected):
            lines.append(f"  drawdown {drawdown / MONEY_SCALE:,.2f}, no-edge figure "
                         "unavailable without a volatility to compare it against")
        else:
            lines.append(
                f"  drawdown {drawdown / MONEY_SCALE:,.2f} against "
                f"{expected / MONEY_SCALE:,.2f} for no edge over the same horizon"
            )
        for note in self.notes:
            lines.append(f"  note: {note}")
        for note in self.stats.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def _pool(runs: Sequence[RunResult]) -> list[float]:
    """Concatenate per-run return series without diffing across the joins.

    Each run's returns are already differences taken inside that run, so
    concatenating them is concatenating independent samples of the same
    per-period distribution. Concatenating the equity paths and differencing
    afterwards would manufacture one spurious return per seed boundary, which at
    eight seeds and a hundred periods is a percent of the sample and all of it
    in the tails.
    """
    pooled: list[float] = []
    for run in runs:
        pooled.extend(run.returns)
    return pooled


def backtest(
    strategy: Any,
    seeds: Sequence[int] | range = (0,),
    *,
    name: str | None = None,
    config: BacktestConfig | None = None,
    trials: int = 1,
    trial_variance: float | None = None,
    **overrides: Any,
) -> Evaluation:
    """Run one strategy across seeds and report what the sample supports.

    ``trials`` is how many variants the caller has tried in total, not how many
    seeds this run used. It is the input to the deflated Sharpe ratio and there
    is no way to infer it from inside a single call, so it defaults to one and
    is wrong the moment somebody is on their fortieth idea. The result says so.
    """
    resolved = (config or BacktestConfig()).evolve(**overrides) if overrides else (
        config or BacktestConfig()
    )
    seed_list = tuple(int(s) for s in seeds)
    if not seed_list:
        raise ValueError("a backtest needs at least one seed")

    runs = tuple(run_once(strategy, seed, resolved) for seed in seed_list)
    stats = summarise(
        _pool(runs),
        trials=trials,
        trial_variance=trial_variance,
        benchmark=resolved.benchmark_sharpe,
        confidence=resolved.confidence,
    )

    notes: list[str] = []
    dropped = tuple(h for h in resolved.horizons if h not in resolved.collected_horizons)
    if dropped:
        notes.append(
            f"horizons {[h / 1e9 for h in dropped]}s were not collected: they are "
            f"shorter than the {resolved.sample_interval}s sampling grid, so they "
            "would measure the grid rather than the market"
        )
    if resolved.report_horizon != int(resolved.attribution_horizon):
        notes.append(
            f"the decomposition is reported at {resolved.report_horizon / 1e9:g}s and "
            f"not the {int(resolved.attribution_horizon) / 1e9:g}s asked for, because "
            f"the mid series is sampled every {resolved.sample_interval}s"
        )
    if resolved.trade_during_warmup:
        notes.append(
            "the strategy traded through the opening auction, where every book "
            "opens at the midpoint of its settlement range; this is the paired "
            "control, not a measurement of the strategy"
        )
    if len(seed_list) < 3:
        notes.append(
            f"{len(seed_list)} seed(s): per-seed dispersion cannot be estimated, so "
            "nothing here separates the strategy from this particular market"
        )
    bad = [r.seed for r in runs if r.conservation != 0]
    if bad:
        notes.append(f"conservation was not zero on seeds {bad}: the ledger leaked")
    if not any(r.lots for r in runs):
        notes.append("the strategy never traded, so every figure below is about nothing")

    return Evaluation(
        name=name or type(strategy).__name__,
        config=resolved,
        seeds=seed_list,
        runs=runs,
        stats=stats,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# Paired comparison on common random numbers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Paired:
    """Two strategies on the same seeds, and whether the pairing helped.

    ``mean_difference`` is in the natural direction, A minus B, positive when A
    made more money. The wrapped :class:`~arena.research.aggregation.Comparison`
    is in the opposite one, because that function scores losses and treats lower
    as better, so this class holds both rather than leaving the reader to guess
    which convention a sign is in.
    """

    name_a: str
    name_b: str
    a: Evaluation
    b: Evaluation
    statistic: str
    values_a: tuple[float, ...]
    values_b: tuple[float, ...]
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    p_adjusted: float
    # What pairing bought. Var(A-B) = Var(A) + Var(B) - 2Cov(A,B), so the
    # reduction is real if and only if the covariance is positive, and reporting
    # it is the only way a reader can tell whether common random numbers did
    # anything here or whether the two arms simply diverged.
    covariance: float
    correlation: float
    variance_paired: float
    variance_unpaired: float
    n: int
    notes: tuple[str, ...] = ()

    @property
    def variance_reduction(self) -> float:
        """Fraction of the unpaired variance that pairing removed."""
        if not math.isfinite(self.variance_unpaired) or self.variance_unpaired <= 0:
            return float("nan")
        return 1.0 - self.variance_paired / self.variance_unpaired

    @property
    def pairing_helped(self) -> bool:
        return math.isfinite(self.covariance) and self.covariance > 0.0

    @property
    def effective_seeds(self) -> float:
        """Seeds an unpaired design would need for this paired precision."""
        reduction = self.variance_reduction
        if not math.isfinite(reduction) or reduction >= 1.0:
            return float("inf")
        return self.n / (1.0 - reduction)

    @property
    def significant(self) -> bool:
        return math.isfinite(self.p_adjusted) and self.p_adjusted < 0.05

    @property
    def verdict(self) -> str:
        if self.n < 3:
            return f"undecidable: {self.n} paired seeds cannot support a test"
        if self.variance_paired == 0.0 and self.mean_difference == 0.0:
            return "identical: every paired difference is exactly zero"
        if not self.significant:
            return "no difference this sample can see"
        return f"{self.name_a} better" if self.mean_difference > 0 else \
            f"{self.name_b} better"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name_a": self.name_a,
            "name_b": self.name_b,
            "statistic": self.statistic,
            "values_a": list(self.values_a),
            "values_b": list(self.values_b),
            "mean_difference": self.mean_difference,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
            "covariance": self.covariance,
            "correlation": self.correlation,
            "variance_paired": self.variance_paired,
            "variance_unpaired": self.variance_unpaired,
            "variance_reduction": self.variance_reduction,
            "pairing_helped": self.pairing_helped,
            "n": self.n,
            "verdict": self.verdict,
            "notes": list(self.notes),
        }

    def __str__(self) -> str:
        scale = MONEY_SCALE if self.statistic == "pnl" else 1.0
        pairing = (
            # The covariance is in the square of the statistic's units, so it is
            # shown as its square root as well: a number the reader can compare
            # against the arms' own spread rather than one they have to unsquare
            # in their head.
            f"cov {self.covariance / scale / scale:+,.4g} "
            f"(sqrt {_signed_root(self.covariance) / scale:+,.2f}), "
            f"rho {self.correlation:+.3f}, "
            f"Var(A-B)/[Var(A)+Var(B)] = "
            f"{self.variance_paired / self.variance_unpaired:.4f}"
            if math.isfinite(self.variance_unpaired) and self.variance_unpaired > 0
            else "pairing: no dispersion to reduce"
        )
        lines = [
            f"{self.name_a} vs {self.name_b} on {self.n} paired seed(s), "
            f"statistic {self.statistic}",
            f"  difference {self.mean_difference / scale:+,.2f} "
            f"[{self.ci_low / scale:+,.2f}, {self.ci_high / scale:+,.2f}], "
            f"p={self.p_adjusted:.4f}: {self.verdict}",
            f"  {pairing}",
        ]
        if math.isfinite(self.variance_reduction):
            lines.append(
                f"  pairing removed {self.variance_reduction:.1%} of the variance, "
                f"worth {self.effective_seeds:.1f} unpaired seeds"
            )
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def _signed_root(value: float) -> float:
    """The square root of a covariance, carrying its sign.

    Only for display. A covariance is in the square of its statistic's units, so
    a reader comparing it against the arms' own spread has to unsquare it in
    their head, and this does it for them without pretending the sign went away.
    """
    if not math.isfinite(value):
        return value
    return math.copysign(math.sqrt(abs(value)), value)


def _pnl(run: RunResult) -> float:
    return float(run.pnl)


def _covariance(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    pairs = zip(xs, ys, strict=True)
    return sum((x - mx) * (y - my) for x, y in pairs) / (n - 1)


def _variance(xs: Sequence[float]) -> float:
    return _covariance(xs, xs)


def compare(
    a: Any,
    b: Any,
    seeds: Sequence[int] | range = (0,),
    *,
    names: tuple[str, str] | None = None,
    config: BacktestConfig | None = None,
    statistic: Callable[[RunResult], float] = _pnl,
    statistic_name: str = "pnl",
    trials: int | None = None,
    **overrides: Any,
) -> Paired:
    """Run two strategies as a paired trial on common random numbers.

    Both arms take the same seeds, the same agent id and the same latency, which
    is what makes them paired: the kernel derives an agent's random stream and
    its latency jitter from the seed and the id, so everything except the
    strategy's own orders is held fixed between the two runs. Then
    ``Var(A-B) = Var(A) + Var(B) - 2Cov(A,B)`` and the covariance decides
    whether that bought anything, so the covariance is reported rather than
    assumed.
    """
    resolved = (config or BacktestConfig()).evolve(**overrides) if overrides else (
        config or BacktestConfig()
    )
    seed_list = tuple(int(s) for s in seeds)
    label_a, label_b = names or (type(a).__name__, type(b).__name__)
    count = 2 if trials is None else trials

    eval_a = backtest(a, seed_list, name=label_a, config=resolved, trials=count)
    eval_b = backtest(b, seed_list, name=label_b, config=resolved, trials=count)
    return pair(eval_a, eval_b, statistic=statistic, statistic_name=statistic_name)


def pair(
    a: Evaluation,
    b: Evaluation,
    *,
    statistic: Callable[[RunResult], float] = _pnl,
    statistic_name: str = "pnl",
) -> Paired:
    """Turn two evaluations run on the same seeds into a paired comparison."""
    if a.seeds != b.seeds:
        raise ValueError(
            f"{a.name} ran on {a.seeds} and {b.name} on {b.seeds}; a paired trial "
            "needs the same seeds in both arms or the pairing is a fiction"
        )
    if a.config.agent_id != b.config.agent_id:
        raise ValueError(
            "the two arms used different agent ids, so the kernel gave them "
            "different random streams and they are not on common random numbers"
        )
    values_a = tuple(statistic(r) for r in a.runs)
    values_b = tuple(statistic(r) for r in b.runs)
    n = len(values_a)

    covariance = _covariance(values_a, values_b)
    var_a, var_b = _variance(values_a), _variance(values_b)
    differences = [x - y for x, y in zip(values_a, values_b, strict=True)]
    variance_paired = _variance(differences)
    variance_unpaired = var_a + var_b
    denominator = math.sqrt(var_a * var_b) if var_a > 0 and var_b > 0 else 0.0
    correlation = covariance / denominator if denominator else float("nan")

    # `paired_comparison` scores losses, so it is fed the negated statistic and
    # its sign is flipped back on the way out. Feeding it the raw statistic
    # would make `market_wins` mean "A lost", which is exactly the kind of two
    # true things sharing one field that this repository has been bitten by.
    comparison = paired_comparison(
        b.name, [-v for v in values_a], [-v for v in values_b], seed=a.seeds[0]
    )

    notes: list[str] = []
    if n < 3:
        notes.append(
            f"{n} paired seed(s): the t-test needs three and returns nothing below "
            "it, so no p-value is reported"
        )
    if variance_unpaired == 0.0:
        notes.append(
            "neither arm varied across seeds, so there was no variance for pairing "
            "to reduce"
        )
    elif covariance <= 0.0:
        notes.append(
            f"covariance {covariance:+.4g} is not positive, so pairing did not help "
            "here: the two arms drove the market apart rather than sharing it"
        )
    if all(d == 0.0 for d in differences):
        notes.append(
            "every paired difference is exactly zero, which is what identical "
            "strategies on a bit-identical simulation should produce"
        )

    p_value = comparison.p_value
    if not math.isfinite(p_value) and all(d == 0.0 for d in differences) and n >= 3:
        # The paired t-test on an all-zero difference series is 0/0, and scipy
        # returns NaN. The honest reading is not "unknown" but "no difference at
        # all", so it is reported as p=1 with the reason recorded above.
        p_value = 1.0

    return Paired(
        name_a=a.name,
        name_b=b.name,
        a=a,
        b=b,
        statistic=statistic_name,
        values_a=values_a,
        values_b=values_b,
        mean_difference=-comparison.mean_difference if math.isfinite(
            comparison.mean_difference
        ) else (sum(differences) / n if n else float("nan")),
        ci_low=-comparison.ci_high,
        ci_high=-comparison.ci_low,
        p_value=p_value,
        p_adjusted=p_value,
        covariance=covariance,
        correlation=correlation,
        variance_paired=variance_paired,
        variance_unpaired=variance_unpaired,
        n=n,
        notes=tuple(notes),
    )


def compare_many(
    candidates: Mapping[str, Any],
    baseline: Any,
    seeds: Sequence[int] | range = (0,),
    *,
    baseline_name: str | None = None,
    config: BacktestConfig | None = None,
    statistic: Callable[[RunResult], float] = _pnl,
    statistic_name: str = "pnl",
    **overrides: Any,
) -> list[Paired]:
    """Compare several strategies against one baseline, controlling the FDR.

    Every candidate is tested against the same baseline on the same seeds, so at
    the usual 5% level one spurious win per twenty candidates is expected by
    chance alone. Benjamini-Hochberg is applied across the set by
    :func:`~arena.research.aggregation.benjamini_hochberg`, which is the same
    correction the information-aggregation experiment uses, so a claim made here
    and a claim made there are corrected the same way.

    The baseline arm is run once and reused for every comparison. That is
    legitimate and it is also the point: the candidates are then compared
    against exactly the same paired sample rather than against re-runs that
    happen to differ.
    """
    resolved = (config or BacktestConfig()).evolve(**overrides) if overrides else (
        config or BacktestConfig()
    )
    seed_list = tuple(int(s) for s in seeds)
    trials = len(candidates) + 1
    base_label = baseline_name or type(baseline).__name__
    base = backtest(baseline, seed_list, name=base_label, config=resolved, trials=trials)

    results: list[Paired] = []
    for label, candidate in candidates.items():
        arm = backtest(candidate, seed_list, name=label, config=resolved, trials=trials)
        results.append(
            pair(arm, base, statistic=statistic, statistic_name=statistic_name)
        )

    # Only the ones with a finite p-value go into the correction. Benjamini-
    # Hochberg sorts on the p-value, and a NaN neither sorts nor survives the
    # running minimum, so an undecidable comparison does not fail loudly: it
    # counts toward the total and shifts everybody else. Measured on p-values
    # of 0.001, 0.04, 0.30 and 0.90, the adjusted values are 0.004, 0.080,
    # 0.400 and 0.900; adding a single NaN row moves them to 0.005, 0.100,
    # 0.500 and 1.000 without any of them being wrong about anything.
    testable = [i for i, r in enumerate(results) if math.isfinite(r.p_value)]
    if not testable:
        return results
    stubs = [
        Comparison(
            baseline=results[i].name_a,
            market_score=float("nan"),
            baseline_score=float("nan"),
            mean_difference=results[i].mean_difference,
            ci_low=results[i].ci_low,
            ci_high=results[i].ci_high,
            t_statistic=float("nan"),
            p_value=results[i].p_value,
            n=results[i].n,
        )
        for i in testable
    ]
    for position, adjusted in zip(testable, benjamini_hochberg(stubs), strict=True):
        results[position] = replace(results[position], p_adjusted=adjusted.p_adjusted)
    return results


# --------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        finished = subprocess.run(
            ["git", *args],
            cwd=_REPO,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if finished.returncode != 0:
        return None
    return finished.stdout.strip()


def _round(value: Any, places: int = 12) -> Any:
    """Sanitise a payload for canonical JSON, and round the floats.

    Two things at once because they have the same cause. ``canonical_json``
    refuses NaN and infinity, which are ordinary answers here for a Sharpe that
    does not exist, so they become strings that say which one they were. And a
    float is rounded to twelve significant figures because its last bit is a
    property of the machine rather than of the run, while everything that comes
    out of the ledger is an integer and is left exactly alone. The integers are
    what the reproducibility check is really about.
    """
    if isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if value == 0.0:
            return 0.0
        magnitude = math.floor(math.log10(abs(value)))
        return round(value, places - 1 - magnitude)
    if isinstance(value, dict):
        return {str(k): _round(v, places) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_round(v, places) for v in value]
    return value


@dataclass(frozen=True)
class Manifest:
    """What a run was, what it produced, and a digest tying the two together.

    Enough to rebuild the run and check that it lands in the same place. The
    commit is recorded alongside whether the working tree was clean, because a
    manifest that names a commit while the tree was dirty is describing code
    that exists nowhere.
    """

    kind: str
    name: str
    config: dict[str, Any]
    seeds: list[int]
    commit: str | None
    dirty: bool
    results: dict[str, Any]
    results_digest: str
    digest: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "config": self.config,
            "seeds": list(self.seeds),
            "commit": self.commit,
            "dirty": self.dirty,
            "results": self.results,
            "results_digest": self.results_digest,
            "digest": self.digest,
            "notes": list(self.notes),
        }

    def json(self) -> str:
        return canonical_json(self.to_dict())

    def config_object(self) -> BacktestConfig:
        return BacktestConfig.from_dict(self.config)


def _digestible(evaluation: Evaluation) -> dict[str, Any]:
    """The part of a result a digest should cover.

    The per-seed ledger integers, the pooled statistics and the attribution.
    Deliberately not the equity path or the return series: they are derived from
    the same integers, they would make the payload thousands of entries long,
    and a digest nobody can diff by eye is a digest nobody checks.
    """
    return {
        "pnl_per_seed": [r.pnl for r in evaluation.runs],
        "lots_per_seed": [r.lots for r in evaluation.runs],
        "fills_per_seed": [r.fills for r in evaluation.runs],
        "notional_per_seed": [r.notional for r in evaluation.runs],
        "conservation_per_seed": [r.conservation for r in evaluation.runs],
        "warmup_pnl_per_seed": [r.warmup_pnl for r in evaluation.runs],
        "periods_per_seed": [len(r.returns) for r in evaluation.runs],
        "attribution": evaluation.attribution,
        "stats": evaluation.stats.to_dict(),
    }


def manifest(evaluation: Evaluation, kind: str = "backtest") -> Manifest:
    """Write the manifest for a finished evaluation."""
    config = evaluation.config.to_dict()
    results = _round(_digestible(evaluation))
    results_digest = digest(results)
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    payload = {
        "kind": kind,
        "name": evaluation.name,
        "config": config,
        "seeds": list(evaluation.seeds),
        "results_digest": results_digest,
    }
    return Manifest(
        kind=kind,
        name=evaluation.name,
        config=config,
        seeds=list(evaluation.seeds),
        commit=commit,
        dirty=bool(status),
        results=results,
        results_digest=results_digest,
        digest=manifest_digest(payload),
        notes=evaluation.notes,
    )


def replay(strategy: Any, source: Manifest) -> Evaluation:
    """Re-run exactly what a manifest describes.

    The strategy has to be supplied because a manifest records a configuration
    and not a closure. Everything else, the seeds and every field of the config,
    comes from the manifest, so a digest that disagrees is a disagreement about
    the simulation rather than about how it was set up.
    """
    return backtest(
        strategy,
        source.seeds,
        name=source.name,
        config=source.config_object(),
        trials=int(source.results["stats"]["trials"]),
    )


def verify_reproducible(strategy: Any, source: Manifest) -> Manifest:
    """Replay a manifest and refuse to return unless the digest is identical.

    Determinism is a claim until something checks it. This is the check, and it
    is deliberately expensive: it runs the whole thing again rather than
    comparing a stored hash to itself.
    """
    again = manifest(replay(strategy, source), kind=source.kind)
    if again.digest != source.digest:
        raise AssertionError(
            f"replaying {source.name} gave digest {again.digest} against "
            f"{source.digest}; the run is not reproducible from its manifest"
        )
    return again
