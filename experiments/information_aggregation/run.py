"""Experiment 1: does the market aggregate better than the agents inside it?

Run it::

    python experiments/information_aggregation/run.py --quick     # 30 trials
    python experiments/information_aggregation/run.py --full      # 200 trials

Every run writes a manifest (configuration, seeds, code digests, results digest)
next to a per-trial CSV, so a result can be reproduced from the manifest alone
and two runs of the same manifest can be compared byte for byte.

Reading the output
------------------

The market is scored against four baselines built from the *same* agents, so
the comparison is paired trial by trial and question difficulty cancels. Lower
score is better, so a negative difference means the market won.

Three of the four rungs are deliberately stacked against the market:

  * best single agent is chosen *after* seeing the answers. Nobody could pick
    it in advance, and no real forecasting system gets to
  * precision-weighted uses the exact battle counts as weights, which a field
    study could only estimate
  * extremized has its factor fitted on held-out trials, the treatment the
    forecasting literature says is the strong one

If the market loses to the extremized aggregate, that is the outcome the
literature predicts and it gets reported as such. The experiment is worth
running either way; it is only worth trusting if the answer was allowed to come
out badly.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "python")]

import numpy as np

from arena.research.aggregation import (
    Comparison,
    benjamini_hochberg,
    extremized_mean,
    fit_extremization,
    murphy_decomposition,
    paired_comparison,
    precision_weighted_mean,
    simple_mean,
)
from arena.research.experiment import (
    TrialConfig,
    TrialResult,
    draw_trials,
    manifest_digest,
    run_trial,
)

RESULTS = Path(__file__).resolve().parent / "results"
LADDER = ("best single agent", "simple mean", "precision weighted", "extremized")


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def best_single_agent(result: TrialResult) -> float:
    """The agent closest to the truth, chosen with hindsight.

    An upper bound on what picking one forecaster could ever achieve, not a
    strategy. It is on the ladder to keep the market honest: a market that
    cannot beat the simple mean is uninteresting, but a market that beats even
    this is doing something no selection rule could.
    """
    forecasts = [f for f in result.agent_forecasts if np.isfinite(f)]
    if not forecasts:
        return float("nan")
    return min(forecasts, key=lambda f: abs(f - result.truth_probability))


def held_out_extremized(results: list[TrialResult]) -> tuple[list[float], float, float]:
    """Extremized aggregate with the factor fitted out of sample.

    The trials are split in half; the factor is fitted on one half and applied
    to the other, then the roles swap. Fitting and evaluating on the same trials
    would let the baseline see the answers, and since the baseline is the one
    the market has to beat, that would quietly bias the experiment toward the
    market. The two fitted factors are returned so a large gap between them can
    be spotted as instability rather than hidden in an average.
    """
    forecast_sets = [list(r.agent_forecasts) for r in results]
    truths = [r.truth_probability for r in results]
    half = len(results) // 2
    folds = ((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half)))

    predictions = [float("nan")] * len(results)
    factors = []
    for fit_slice, apply_slice in folds:
        d = fit_extremization(forecast_sets[fit_slice], truths[fit_slice])
        factors.append(d)
        for index in range(*apply_slice.indices(len(results))):
            predictions[index] = extremized_mean(forecast_sets[index], d)
    return predictions, factors[0], factors[1]


def baseline_table(results: list[TrialResult]) -> dict[str, list[float]]:
    extremized, _d_a, _d_b = held_out_extremized(results)
    return {
        "best single agent": [best_single_agent(r) for r in results],
        "simple mean": [simple_mean(r.agent_forecasts) for r in results],
        "precision weighted": [
            precision_weighted_mean(r.agent_forecasts, r.agent_battles) for r in results
        ],
        "extremized": extremized,
    }


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def _run_one(config: TrialConfig) -> TrialResult:
    return run_trial(config)


def run_all(configs: list[TrialConfig], workers: int) -> list[TrialResult]:
    """Trials are independent and each is deterministic from its own config.

    So parallelism cannot change a result, only how long it takes to get it.
    The order is restored by the executor's map, which keeps the output stable
    regardless of the worker count.
    """
    if workers <= 1:
        return [run_trial(c) for c in configs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_one, configs))


def code_digests() -> dict[str, str]:
    """Digests of the modules that determine a result.

    A manifest that records the configuration but not the code is only half a
    record: the same seeds through changed logic produce different numbers, and
    without this the difference would be invisible.
    """
    import hashlib

    files = [
        "python/arena/research/experiment.py",
        "python/arena/research/aggregation.py",
        "python/arena/agents/bayesian.py",
        "python/arena/agents/market_maker.py",
        "python/arena/agents/noise.py",
        "python/arena/exchange/engine.py",
        "python/arena/market/venue.py",
        "python/arena/sim/kernel.py",
    ]
    digests = {}
    for name in files:
        path = REPO / name
        if path.exists():
            digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return digests


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Report:
    comparisons: list[Comparison]
    market_error: float
    outcome_brier: dict[str, float]
    extremization: tuple[float, float]
    diagnostics: dict[str, float]


def analyse(results: list[TrialResult], bootstrap: int, seed: int) -> Report:
    truth = np.array([r.truth_probability for r in results])
    market = np.array([r.market_forecast for r in results])
    outcomes = np.array([r.outcome for r in results])

    market_scores = (market - truth) ** 2
    baselines = baseline_table(results)

    comparisons = [
        paired_comparison(
            name,
            market_scores,
            (np.asarray(baselines[name]) - truth) ** 2,
            bootstrap=bootstrap,
            seed=seed,
        )
        for name in LADDER
    ]

    _predictions, d_a, d_b = held_out_extremized(results)
    return Report(
        comparisons=benjamini_hochberg(comparisons),
        market_error=float(np.nanmean(market_scores)),
        outcome_brier={
            "market": float(np.nanmean((market - outcomes) ** 2)),
            **{
                name: float(np.nanmean((np.asarray(values) - outcomes) ** 2))
                for name, values in baselines.items()
            },
        },
        extremization=(d_a, d_b),
        diagnostics={
            **murphy_decomposition(market, outcomes),
            "trades": float(np.mean([r.trades for r in results])),
            "quoted_fraction": float(np.mean([r.quoted_fraction for r in results])),
            "conservation_failures": float(
                sum(1 for r in results if r.conservation != 0)
            ),
            "truth_spread": float(np.std(truth)),
        },
    )


def render(report: Report, results: list[TrialResult]) -> str:
    lines = [
        "",
        "=" * 78,
        f"  Experiment 1: information aggregation      {len(results)} trials",
        "=" * 78,
        "",
        "  Primary metric: squared error to the KNOWN true probability.",
        "  Negative difference = the market is closer to the truth.",
        "",
        f"  market mean squared error   {report.market_error:.5f}",
        "",
    ]
    lines += [str(c) for c in report.comparisons]
    lines += [
        "",
        f"  extremization factor d, fitted per fold: "
        f"{report.extremization[0]:.2f} / {report.extremization[1]:.2f}",
        "",
        "  Secondary metric: Brier against one sampled outcome (noisier by design).",
    ]
    for name, value in report.outcome_brier.items():
        lines.append(f"    {name:<28}{value:.5f}")

    diagnostics = report.diagnostics
    lines += [
        "",
        "  Market health (a broken market would win or lose for the wrong reason):",
        f"    mean trades per trial       {diagnostics['trades']:.0f}",
        f"    fraction of time two-sided  {diagnostics['quoted_fraction']:.1%}",
        f"    conservation failures       {diagnostics['conservation_failures']:.0f}",
        f"    spread of true probability  {diagnostics['truth_spread']:.3f}",
        f"    market reliability          {diagnostics['reliability']:.5f}",
        f"    market resolution           {diagnostics['resolution']:.5f}",
        "",
        "=" * 78,
        "",
    ]
    return "\n".join(lines)


def write_outputs(
    tag: str,
    configs: list[TrialConfig],
    results: list[TrialResult],
    report: Report,
    args: argparse.Namespace,
) -> tuple[Path, str]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    baselines = baseline_table(results)

    csv_path = RESULTS / f"{tag}_trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["seed", "truth", "threshold", "truth_probability", "market", "closing",
             "outcome", "trades", *LADDER,
             *[f"agent_{i}" for i in range(len(configs[0].battles))]]
        )
        for index, result in enumerate(results):
            writer.writerow(
                [result.config.seed, result.config.truth, result.config.threshold,
                 result.truth_probability, result.market_forecast,
                 result.closing_forecast, result.outcome, result.trades,
                 *[baselines[name][index] for name in LADDER],
                 *result.agent_forecasts]
            )

    payload = {
        "trials": [r.to_dict() for r in results],
        "comparisons": [
            {
                "baseline": c.baseline,
                "market_score": c.market_score,
                "baseline_score": c.baseline_score,
                "mean_difference": c.mean_difference,
                "ci": [c.ci_low, c.ci_high],
                "p_value": c.p_value,
                "p_adjusted": c.p_adjusted,
                "n": c.n,
            }
            for c in report.comparisons
        ],
    }
    digest = manifest_digest(payload)

    manifest = {
        "experiment": "information_aggregation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "code_digests": code_digests(),
        "arguments": vars(args),
        "trial_count": len(configs),
        "results_digest": digest,
        "summary": {
            "market_error": report.market_error,
            "extremization": list(report.extremization),
            "outcome_brier": report.outcome_brier,
            "diagnostics": report.diagnostics,
        },
        "comparisons": payload["comparisons"],
    }
    manifest_path = RESULTS / f"{tag}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, digest


def compare_venues(configs: list[TrialConfig], args: argparse.Namespace) -> int:
    """Experiment 2: the same trials through both mechanisms.

    Identical seeds, truths, thresholds, agents and information: the only
    thing that varies is where the liquidity comes from. That makes the
    difference between the two a property of the mechanism, and lets it be
    tested as a paired difference rather than as two separate averages that
    happen to differ.
    """
    print(f"running {len(configs)} trials on both mechanisms "
          f"({args.workers} worker(s))...")

    outcomes: dict[str, list[TrialResult]] = {}
    for kind in ("clob", "lmsr"):
        variant = [replace(c, venue_kind=kind) for c in configs]
        outcomes[kind] = run_all(variant, args.workers)
        report = analyse(outcomes[kind], bootstrap=args.bootstrap, seed=args.seed)
        print(render(report, outcomes[kind]).replace(
            "Experiment 1: information aggregation", f"{kind.upper():<5} against the ladder"
        ))
        write_outputs(f"venue_{kind}", variant, outcomes[kind], report, args)

    truth = np.array([r.truth_probability for r in outcomes["clob"]])
    clob = (np.array([r.market_forecast for r in outcomes["clob"]]) - truth) ** 2
    lmsr = (np.array([r.market_forecast for r in outcomes["lmsr"]]) - truth) ** 2
    head_to_head = paired_comparison(
        "order book", lmsr, clob, bootstrap=args.bootstrap, seed=args.seed
    )

    print("=" * 78)
    print("  Experiment 2: scoring rule against order book, paired by trial")
    print("=" * 78)
    print()
    print(f"  order book     {clob.mean():.5f}")
    print(f"  scoring rule   {lmsr.mean():.5f}")
    print()
    print(f"  difference {head_to_head.mean_difference:+.5f} "
          f"[{head_to_head.ci_low:+.5f}, {head_to_head.ci_high:+.5f}]  "
          f"p={head_to_head.p_value:.5f}")
    verdict = (
        "the scoring rule aggregates better"
        if head_to_head.mean_difference < 0
        else "the order book aggregates better"
    )
    print(f"  {verdict if head_to_head.p_value < 0.05 else 'no detectable difference'}")
    print()
    for kind in ("clob", "lmsr"):
        trades = np.mean([r.trades for r in outcomes[kind]])
        print(f"  {kind:<6} mean trades per trial {trades:8.0f}")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="30 trials, for CI")
    parser.add_argument("--full", action="store_true", help="200 trials")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--duration", type=int, default=300, help="simulated seconds")
    parser.add_argument("--window-battles", type=int, default=2_000)
    parser.add_argument(
        "--homogeneous-latency",
        action="store_true",
        help="ablation: every agent equally fast",
    )
    parser.add_argument(
        "--concentrated",
        action="store_true",
        help="ablation: one agent holds all the information instead of many",
    )
    parser.add_argument(
        "--noise-traders",
        type=int,
        default=8,
        help="ablation: set to 0 to remove uninformed order flow entirely",
    )
    parser.add_argument(
        "--position-limit",
        type=int,
        default=None,
        help="ablation: cap on each informed agent's position, the thing that "
        "decides how much of its view reaches the price",
    )
    parser.add_argument(
        "--venue",
        choices=("clob", "lmsr"),
        default="clob",
        help="market mechanism: a limit order book, or a logarithmic scoring rule",
    )
    parser.add_argument(
        "--subsidy",
        type=float,
        default=None,
        help="what the scoring-rule venue will lose making the market; "
        "defaults to the depth-matched value",
    )
    parser.add_argument(
        "--compare-venues",
        action="store_true",
        help="Experiment 2: run both mechanisms on the same trials and report "
        "the paired difference between them",
    )
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    count = args.trials or (200 if args.full else 30)
    from arena.sim.time import seconds as sim_seconds

    overrides = {
        "duration": int(sim_seconds(args.duration)),
        "heterogeneous_latency": not args.homogeneous_latency,
        "noise_traders": args.noise_traders,
        "venue_kind": args.venue,
    }
    if args.position_limit is not None:
        overrides["position_limit"] = args.position_limit
    if args.subsidy is not None:
        overrides["subsidy"] = args.subsidy
    configs = draw_trials(
        count,
        seed=args.seed,
        agents=args.agents,
        window_battles=args.window_battles,
        **overrides,
    )

    if args.concentrated:
        # Same total information, held by one agent instead of spread across
        # the population. This is the Kyle question: whether a market with one
        # informed trader prices as well as a market with many small ones.
        # Everything else (seeds, truths, thresholds, the maker, the noise)
        # is held fixed, so the only thing that moved is where the information
        # sits.
        configs = [
            replace(c, battles=(sum(c.battles),) + (0,) * (len(c.battles) - 1))
            for c in configs
        ]

    tag = args.tag or ("full" if args.full else "quick")
    if args.concentrated:
        tag += "_concentrated"
    if args.homogeneous_latency:
        tag += "_flatlatency"
    if args.noise_traders != 8:
        tag += f"_noise{args.noise_traders}"
    if args.position_limit is not None:
        tag += f"_limit{args.position_limit}"

    if args.compare_venues:
        return compare_venues(configs, args)

    print(f"running {count} trials (seed {args.seed}, {args.workers} worker(s))...")
    results = run_all(configs, args.workers)
    report = analyse(results, bootstrap=args.bootstrap, seed=args.seed)
    print(render(report, results))

    manifest_path, digest = write_outputs(tag, configs, results, report, args)
    print(f"  manifest  {manifest_path}")
    print(f"  digest    {digest}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
