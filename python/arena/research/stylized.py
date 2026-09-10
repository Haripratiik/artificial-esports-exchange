"""Measuring whether the market behaves like a market.

A matching engine can be provably correct and still produce prices no real
market would produce. Correctness says the mechanism does what it claims;
these statistics say whether what emerges from it resembles a traded asset.
That is a different question and it needs different evidence.

**This module measures. It does not tune.** Every function here reads a price
or order-flow series and returns a number. Nothing feeds back into the agents,
because a market adjusted until it reproduces a target statistic has been made
to *look* real rather than shown to *be* plausible, and the statistic then
measures the adjustment rather than the market.

Which facts to expect, and which not to
---------------------------------------

The canonical stylized facts were established on equities and FX, where prices
are unbounded and information arrives continuously. This market is neither:
every contract is a bounded statistic that settles at a value fixed for the
whole session. So some regularities should appear, some should not, and saying
which in advance is what keeps this honest rather than a fishing expedition.

**Expected to hold**

  no return autocorrelation   an efficient price is unforecastable from its own
                              past, whatever it is written on
  bid-ask bounce              trade prices alternate between bid and offer, so
                              trade-to-trade returns carry negative lag-1
                              autocorrelation. A pure microstructure signature,
                              and one this market should show if the tape is real
  convergence to settlement   the truth is fixed and known to informed agents,
                              so price should approach it and stay
  variance decays             uncertainty about a settling contract falls as its
                              window fills. Equities have no analogue

**Not expected, and it would be suspicious if strong**

  volatility clustering       driven by clustered *information arrival*. The truth
                              here is constant for the session, so there is no
                              mechanism to cluster
  long-memory order flow      caused by large traders splitting metaorders across
                              hours. No agent here splits anything
  fat tails                   an open question. Discrete flow and inventory limits
                              can produce them, but the literature is clear that
                              *stabilising* agents dampen them, and this market
                              is anchored by fundamental traders

A result of "absent" against the second group is not a failure. It is the model
telling the truth about a market that has no mechanism to produce it, and
manufacturing one would be the actual failure.

What was actually found
-----------------------

Those predictions were run and **three of them were wrong**. All three "not
expected" facts appeared, unprompted, and the mechanisms are worth recording
because they were not designed in:

**Fat tails did emerge.** Hill tail index 2.06 to 3.86 across instruments,
straddling the ~3 of the empirical inverse-cubic law. The mechanism is the
market maker's position limit: once it stops quoting a side, the next
aggressive order jumps several price levels instead of one, and those jumps are
the tail. Nothing in the agent is tuned for this: it falls out of a risk limit
interacting with a discrete book.

**Volatility clustering did emerge**, at 0.17 to 0.30 lag-1 autocorrelation of
|returns|, with a decay exponent of 0.36 on one instrument, inside the
empirical 0.2 to 0.4 band. The reasoning that predicted its absence was wrong:
clustering does not require clustered *information*, only clustered *impact*.
When the maker is run over it widens and skews, which makes the next order move
price further, which runs it over again. That feedback is endogenous
volatility, and the literature argues it dominates news-driven volatility in
real markets too.

**Order-flow autocorrelation did emerge**, at 0.34 to 0.48. The prediction
assumed no agent splits a metaorder. In fact the fundamental agents hold one
fixed view for the whole session and accumulate toward a position limit over
many wakeups, which *is* metaorder splitting, arrived at by accident rather
than by design.

The lesson is about method rather than about any one statistic: a mechanism
absent from the design can still be present in the behaviour, so the honest
move is to measure first and explain second.

One prediction that looked wrong was a measurement error instead. Lag-1 return
autocorrelation came out at -0.22, suggesting a mean-reverting and inefficient
price. Sampling the same market at different rates gave -0.20 at 100ms, +0.10
at 500ms and +0.38 at one second. So the number was microstructure noise, not a
property of the price, and any single reading of it would have supported
whichever conclusion was reached first. This is why ``variance_signature``
exists and why a single-frequency autocorrelation should not be trusted here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "Verdict",
    "returns",
    "excess_kurtosis",
    "hill_tail_index",
    "autocorrelation",
    "ljung_box",
    "acf_decay_exponent",
    "hurst_exponent",
    "bid_ask_bounce",
    "order_flow_autocorrelation",
    "variance_ratio",
    "variance_signature",
    "StylizedReport",
    "analyse",
]


@dataclass(frozen=True, slots=True)
class Verdict:
    """One measured statistic, with what was expected and whether it matched.

    The expectation is recorded alongside the value on purpose. A number with
    no prior attached invites reading whatever one hoped for into it.
    """

    name: str
    value: float
    expected: str
    verdict: str
    detail: str = ""

    def __str__(self) -> str:
        mark = {"as expected": "ok", "unexpected": "??", "n/a": "--"}.get(self.verdict, "  ")
        return f"  [{mark}] {self.name:<34} {self.value:>10.4f}   {self.detail}"


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def returns(prices: np.ndarray, log: bool = False) -> np.ndarray:
    """Successive price changes.

    Arithmetic by default rather than log. A contract here can legitimately
    settle at or below zero (a spread trades negative most of the time), so log
    returns are undefined for a large part of the instrument universe.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.size < 2:
        return np.array([])
    if log:
        if np.any(prices <= 0):
            raise ValueError("log returns require strictly positive prices")
        return np.diff(np.log(prices))
    return np.diff(prices)


def excess_kurtosis(series: np.ndarray) -> float:
    """Kurtosis above the Gaussian value of 3. Zero means normal-tailed."""
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    if series.size < 4:
        return float("nan")
    centred = series - series.mean()
    variance = float((centred**2).mean())
    if variance <= 0:
        return float("nan")
    return float((centred**4).mean() / variance**2 - 3.0)


def hill_tail_index(series: np.ndarray, tail_fraction: float = 0.05) -> float:
    """Hill estimator of the tail exponent of |series|.

    Lower means heavier. Empirical equity returns sit near 3, the "inverse cubic
    law"; a Gaussian has no finite tail index and the estimator drifts upward
    with sample size instead of settling.

    Reported on the upper ``tail_fraction`` of absolute values, which is the
    usual compromise: too few points and the estimate is noise, too many and it
    is measuring the body of the distribution rather than its tail.
    """
    series = np.abs(np.asarray(series, dtype=float))
    series = series[np.isfinite(series) & (series > 0)]
    if series.size < 40:
        return float("nan")
    ordered = np.sort(series)[::-1]
    k = max(5, int(len(ordered) * tail_fraction))
    k = min(k, len(ordered) - 1)
    threshold = ordered[k]
    if threshold <= 0:
        return float("nan")
    return float(1.0 / np.mean(np.log(ordered[:k] / threshold)))


def autocorrelation(series: np.ndarray, lag: int) -> float:
    """Sample autocorrelation at one lag."""
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    if series.size <= lag + 2:
        return float("nan")
    centred = series - series.mean()
    denominator = float((centred**2).sum())
    if denominator <= 0:
        return float("nan")
    return float((centred[:-lag] * centred[lag:]).sum() / denominator)


def ljung_box(series: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """Ljung-Box Q statistic and its p-value under a chi-squared null.

    Tests joint significance of autocorrelation across lags rather than one at
    a time, which matters because a single lag will clear any threshold roughly
    one time in twenty by chance.
    """
    from scipy import stats

    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    n = series.size
    if n <= lags + 2:
        return float("nan"), float("nan")

    q = 0.0
    for lag in range(1, lags + 1):
        rho = autocorrelation(series, lag)
        if not math.isfinite(rho):
            continue
        q += rho**2 / (n - lag)
    q *= n * (n + 2)
    return float(q), float(stats.chi2.sf(q, lags))


def acf_decay_exponent(series: np.ndarray, max_lag: int = 40) -> float:
    """Power-law decay exponent of the autocorrelation of |series|.

    Empirical equity data gives roughly 0.2 to 0.4: slow decay, the signature
    of long-memory volatility. A fast decay produces a large exponent and means
    volatility has no memory.
    """
    series = np.abs(np.asarray(series, dtype=float))
    series = series[np.isfinite(series)]
    if series.size < 100:
        return float("nan")

    # A decay exponent only means something if there is autocorrelation to
    # decay. Fitting a power law to a series with none produces a confident
    # number from pure noise; measured on white noise this returned -0.03,
    # which reads as "slower decay than any real market" rather than as "no
    # memory at all". So the fit is attempted only once lag-1 clears the
    # standard 2/sqrt(n) significance band; below that the honest answer is
    # that there is nothing here to measure.
    threshold = 2.0 / math.sqrt(series.size)
    first = autocorrelation(series, 1)
    if not math.isfinite(first) or first < threshold:
        return float("nan")

    lags, values = [], []
    for lag in range(1, max_lag + 1):
        rho = autocorrelation(series, lag)
        if math.isfinite(rho) and rho > threshold:
            lags.append(lag)
            values.append(rho)
    if len(lags) < 6:
        return float("nan")
    slope, _intercept = np.polyfit(np.log(lags), np.log(values), 1)
    return float(-slope)


def hurst_exponent(series: np.ndarray, max_lag: int = 40) -> float:
    """Hurst exponent by the variance-of-differences method.

    0.5 is a random walk. Above is trending, below is mean-reverting. A price
    that is being discovered should start above 0.5 while it converges and fall
    toward 0.5 once it arrives.
    """
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    if series.size < max_lag * 3:
        return float("nan")

    lags = range(2, max_lag)
    deviations = []
    kept = []
    for lag in lags:
        diff = series[lag:] - series[:-lag]
        sd = float(np.std(diff))
        if sd > 0:
            deviations.append(sd)
            kept.append(lag)
    if len(kept) < 6:
        return float("nan")
    slope, _ = np.polyfit(np.log(kept), np.log(deviations), 1)
    return float(slope)


def bid_ask_bounce(trade_prices: np.ndarray) -> float:
    """Lag-1 autocorrelation of trade-to-trade returns.

    Should be *negative*. Trades alternate between hitting the bid and lifting
    the offer, so consecutive trade prices bounce across the spread even when
    the underlying value has not moved. Roll's classic result: the more negative
    it is, the wider the effective spread.

    Its absence in a simulated market usually means trades are not really being
    initiated from both sides, which is what a wash-trading engine looks like.
    """
    return autocorrelation(returns(trade_prices), 1)


def order_flow_autocorrelation(signs: np.ndarray, lag: int = 1) -> float:
    """Autocorrelation of trade-sign order flow.

    Strongly positive and long-memoried in real markets, because institutions
    split large parent orders into many child orders that all lean the same way.
    A market with no order splitting has no mechanism to produce it.
    """
    return autocorrelation(np.asarray(signs, dtype=float), lag)


def variance_ratio(prices: np.ndarray, q: int = 4) -> float:
    """Lo-MacKinlay variance ratio: Var(q-period) / (q * Var(1-period)).

    One means a random walk. Above one is trending, below is mean-reverting.
    A cleaner efficiency test than a single autocorrelation because it
    aggregates evidence across horizons.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.size < q * 8:
        return float("nan")
    single = np.diff(prices)
    multi = prices[q:] - prices[:-q]
    var_single = float(np.var(single, ddof=1))
    if var_single <= 0:
        return float("nan")
    return float(np.var(multi, ddof=1) / (q * var_single))


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def variance_signature(prices: np.ndarray, horizons: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)) -> list[tuple[int, float]]:
    """Variance ratio across sampling horizons: the signature plot.

    The single most useful diagnostic here, and the one that stops a wrong
    conclusion being drawn from a single number.

    An observed price is the efficient price plus microstructure noise: the
    quote flickering as a maker requotes, the tick grid, the bid-ask bounce.
    That noise is transient and reverses, so it inflates variance at short
    horizons and washes out at long ones. Measuring the variance ratio at one
    sampling frequency therefore says nothing on its own; it is a mixture of
    the two, in unknown proportion.

    The signature separates them. Sub-one at short horizons rising toward one
    at long horizons is the classic shape of microstructure noise sitting on an
    efficient price, and it is what real high-frequency equity data looks like.
    A ratio that stays well below one at every horizon is a genuinely
    mean-reverting market; one that stays above is genuinely trending.

    Measured on this market before it was understood, the lag-1 return
    autocorrelation ran from -0.20 at 100ms to +0.38 at one second. Read at any
    single frequency it would have supported whichever conclusion one reached
    for first.
    """
    prices = np.asarray(prices, dtype=float)
    out: list[tuple[int, float]] = []
    for q in horizons:
        if prices.size < q * 12:
            continue
        ratio = variance_ratio(prices, q) if q > 1 else 1.0
        if math.isfinite(ratio):
            out.append((q, ratio))
    return out


@dataclass
class StylizedReport:
    symbol: str
    observations: int
    verdicts: list[Verdict] = field(default_factory=list)

    def add(
        self, name: str, value: float, expected: str, matches, detail: str = ""
    ) -> None:
        if not math.isfinite(value):
            verdict = "n/a"
        else:
            verdict = "as expected" if matches(value) else "unexpected"
        self.verdicts.append(Verdict(name, value, expected, verdict, detail))

    def __str__(self) -> str:
        lines = [f"{self.symbol}  ({self.observations} observations)"]
        lines += [str(v) for v in self.verdicts]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "observations": self.observations,
            "verdicts": [
                {
                    "name": v.name,
                    "value": None if not math.isfinite(v.value) else v.value,
                    "expected": v.expected,
                    "verdict": v.verdict,
                }
                for v in self.verdicts
            ],
        }


def analyse(
    symbol: str,
    mid_prices: np.ndarray,
    trade_prices: np.ndarray,
    trade_signs: np.ndarray,
) -> StylizedReport:
    """Measure a symbol's price series against the expectations set out above.

    Takes mid prices *and* trade prices because they answer different
    questions. Mid is the market's view and is where efficiency should show;
    trade prices carry the bid-ask bounce, which is a property of execution
    rather than of value.
    """
    mid = np.asarray(mid_prices, dtype=float)
    report = StylizedReport(symbol, int(mid.size))
    r = returns(mid)

    report.add(
        "return autocorrelation (lag 1)",
        autocorrelation(r, 1),
        "|rho| < 0.1, unforecastable",
        lambda v: abs(v) < 0.1,
        "an efficient price cannot be predicted from its own past",
    )
    report.add(
        "variance ratio (q=4)",
        variance_ratio(mid, 4),
        "near 1, a random walk",
        lambda v: 0.6 < v < 1.6,
        "above 1 trends, below 1 mean-reverts",
    )
    report.add(
        "bid-ask bounce (trades, lag 1)",
        bid_ask_bounce(trade_prices),
        "negative",
        lambda v: v < -0.02,
        "trades alternate across the spread; absence suggests one-sided flow",
    )
    report.add(
        "excess kurtosis of returns",
        excess_kurtosis(r),
        "> 1 would be fat-tailed",
        lambda v: v > 1.0,
        "NOT expected here: stabilising agents dampen tails",
    )
    report.add(
        "Hill tail index",
        hill_tail_index(r),
        "~3 in equities; higher is thinner",
        lambda v: 2.0 < v < 5.0,
        "lower means heavier tails",
    )
    report.add(
        "volatility clustering |r| (lag 1)",
        autocorrelation(np.abs(r), 1),
        "> 0.1 would be clustering",
        lambda v: v > 0.1,
        "NOT expected here: the truth is fixed, so information cannot cluster",
    )
    report.add(
        "|r| ACF decay exponent",
        acf_decay_exponent(r),
        "0.2-0.4 in equities",
        lambda v: 0.15 < v < 0.5,
        "slow decay is the long-memory signature",
    )
    report.add(
        "order-flow autocorrelation (lag 1)",
        order_flow_autocorrelation(trade_signs, 1),
        "> 0.2 would be long-memory flow",
        lambda v: v > 0.2,
        "NOT expected here: no agent splits a metaorder",
    )
    report.add(
        "Hurst exponent",
        hurst_exponent(mid),
        "near 0.5 once converged",
        lambda v: 0.35 < v < 0.65,
        "above 0.5 trends, below mean-reverts",
    )
    return report
