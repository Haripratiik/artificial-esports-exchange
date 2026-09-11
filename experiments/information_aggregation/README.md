# Does a market aggregate information better than the agents inside it?

**Experiment 1: no.** It reproduces the simple average of their beliefs, and
loses decisively to weighting those same beliefs by how much evidence each
rests on.

**Experiment 2: and the mechanism is not why.** A logarithmic scoring rule,
which has none of an order book's structural limitations, always quotes, and
needs no counterparty, lands in the same place (p = 0.68).

What is left, measured rather than assumed: agents stop trading once the price
is inside their own uncertainty band, using under half their capacity. Nothing
in either venue makes a well-informed agent bet bigger than a poorly informed
one.

---

# Experiment 1: The market against its own agents

```bash
python experiments/information_aggregation/run.py --full --workers 8
```

## What was measured

200 trials. Each is a self-contained market with one binary contract: *will the
measured win rate over the observation window exceed θ?* The window holds 2,000
battles, so the settled rate is a draw and

    truth = P(Binomial(2000, p*) / 2000 > θ)

is a real probability, known exactly. Thresholds are chosen by bisection so the
answers spread across [0.05, 0.95], with a 2,000-battle window a *random*
threshold makes nearly every trial a foregone conclusion, and a set of foregone
conclusions cannot separate any two forecasters.

Eight informed agents see log-spaced battle counts from 50 to 5,000 (10,319
battles in total), form Beta posteriors, and forecast the Beta-Binomial tail,
**exactly Bayes-optimal** for what each has seen. They trade against one market
maker and eight noise traders on the real matching engine, with heterogeneous
latency. The market's forecast is the time-weighted mid over the final 10%.

Scoring against a known truth rather than a sampled outcome is the one thing
this setup can do that no field study can: it removes the Bernoulli noise that
otherwise dominates, which is why 200 trials suffice where a real forecasting
tournament would need thousands.

## Result

Primary metric, squared error to the true probability. Lower is better,
negative difference means the market won. Paired by trial; Benjamini–Hochberg
across the four comparisons.

| baseline | baseline error | market − baseline | 95% CI | p (adj) | verdict |
|---|---|---|---|---|---|
| best single agent (hindsight) | 0.00272 | **+0.02865** | [+0.023, +0.035] | <0.0001 | market loses |
| simple mean | 0.02960 | +0.00177 | [−0.002, +0.006] | 0.54 | **no difference** |
| precision-weighted | 0.01436 | **+0.01701** | [+0.012, +0.023] | <0.0001 | market loses |
| extremized log-odds | 0.02980 | +0.00157 | [−0.003, +0.007] | 0.54 | **no difference** |

Market error: **0.03137**. Extremization factor, fitted out of sample:
**0.95 / 1.10** across the two folds.

The market is statistically indistinguishable from an unweighted average of its
agents. It is 11.5× worse than the best agent picked with hindsight, and **2.2×
worse than the same agents weighted by evidence**.

## Why: three ablations, each ruling something out

| condition | market error | vs default |
|---|---|---|
| default (8 dispersed agents, 8 noise, limit 800) | 0.03137 | n/a |
| noise traders removed entirely | 0.02796 | −11%, comparisons unchanged |
| position limits raised 5× (800 → 4000) | 0.03159 | no change |
| all 10,319 battles held by **one** agent | 0.08554 | **2.7× worse** |

Plus a convergence check: market error by session length was 0.0144 (60s),
0.0095 (300s), 0.0113 (600s), 0.0113 (1800s), **flat from 300s onward**. The
market has converged; it is not short of time.

So it is not noise traders, not position limits, and not insufficient time.

The conclusion drawn here was that what remained was the mechanism, that a
limit order book aggregates by capital rather than by evidence, because it needs
a counterparty for every trade and its market maker anchors price to its own
reference. **Experiment 2 tested that and refuted it.** The paragraph is left
standing rather than quietly rewritten, because it was the obvious inference
from these ablations and it was wrong; what replaced it is in Experiment 2's
final section.

What survives from this experiment is the finding itself: every agent here
carries the same base size, the same position limit and the same cash, so the
50-battle agent pushes the price about as hard as the 5,000-battle agent, and
the price settles at an unweighted consensus. The precision-weighted baseline
knows each agent's battle count exactly and uses it; the market has no channel
through which that information reaches the price.

The concentration ablation is the same point from the other side. Holding total
information fixed and moving it all into one trader makes the market **2.7×
worse**, because one agent has one balance sheet. Dispersed information reaches
the price better than concentrated information, the opposite of the intuition
that an informed monopolist prices most efficiently.

## What the extremization factor says

Fitted out of sample at 0.95 and 1.10, straddling 1.0, so log-odds
extremization neither helps nor hurts here. The forecasting literature finds
d ∈ [1.16, 3.92] optimal on geopolitical tournaments, where pooled forecasts are
systematically under-confident. That does not reproduce here, and the reason is
visible: those tournaments pool *human* forecasters who are individually
miscalibrated, while these agents are individually Bayes-optimal by
construction. Pooling optimal forecasters produces a well-calibrated aggregate
with nothing left to sharpen.

The grid was widened below 1.0 for this run. In the concentrated ablation the
fitted factor pins at **0.50**, the grid floor, seven agents at the prior and
one sharp agent produce a wildly over-confident log-odds pool that wants heavy
shrinking. Had the grid been clamped at 1.0 as the original formulation
specifies, a boundary value would have been reported as an interior optimum.

## Honesty notes

- **This is the outcome the literature predicts,** and it was pre-committed to
  in the plan before the run. Markets beating simple averages is a known result;
  beating a well-specified statistical aggregate is not, and did not happen.
- **The baselines are deliberately unfair to the market.** "Best single agent"
  is chosen after seeing the answers. "Precision-weighted" uses exact battle
  counts, which no real study can observe. Losing to these is not embarrassing;
  it is the point of putting them on the ladder.
- **Secondary (outcome-based) Brier** tells the same story more noisily: market
  0.198, simple mean 0.201, precision-weighted 0.182, best agent 0.176. The
  ordering is preserved, and the noise is exactly why the primary metric scores
  against the known truth instead.
- **The market is healthy**, so it is not winning or losing for the wrong
  reason: 2,370 trades per trial, two-sided 99.8% of the time, conservation
  exact in every trial, reliability 0.0076 (well calibrated).

---

# Experiment 2: Is it the mechanism?

**Answer: no. Swapping the limit order book for a logarithmic scoring rule
makes no detectable difference.**

```bash
python experiments/information_aggregation/run.py --compare-venues --full --workers 8
```

Experiment 1 left one obvious suspect. A book needs a counterparty for every
trade, so an informed agent can only move price as fast as someone takes the
other side; and its market maker quotes around its own reference, which acts as
an anchor. A scoring rule has neither problem, it always quotes, at any size,
and its price is a function of cumulative net flow with nothing to revert to.

The same 200 trials, the same agents, the same seeds, the same information. The
only thing that varies is where liquidity comes from.

| mechanism | error to truth | trades/trial |
|---|---|---|
| limit order book | 0.03137 | 2,370 |
| logarithmic scoring rule | 0.03063 | 1,782 |

Paired difference **−0.00074**, 95% CI **[−0.0045, +0.0024]**, p = 0.68.

This is a tight null, not an underpowered one: the interval runs from −0.0045 to
+0.0024, against a 0.0170 gap to precision-weighting. The mechanism accounts for
**none** of it. The scoring rule lands on the simple mean too, at p = 0.58.

**The sign of that difference flipped between runs, and it is published
flipped.** The table here before the open-order reserve had the order book
ahead by +0.00007; on the current code the scoring rule is ahead by 0.00074.
Both sit inside the same null, which is what the reversal is evidence for: an
estimate that changes sign under a collateral change the experiment was not
testing had nothing in it to begin with. The conclusion does not move, because
it never rested on the sign.

## Depth doesn't rescue it either

Depth is the scoring rule's one real parameter, set here by the subsidy the
venue is willing to lose. Both venues were calibrated to the same depth at the
touch (40 lots a tick) so the comparison above isolates mechanism. Varying it,
at full 200-trial power:

| shares/tick | subsidy | error to truth |
|---|---|---|
| 5 | 87 | 0.03255 |
| 12 | 208 | 0.03225 |
| **40** | **693** | **0.03079** |
| 115 | 1,993 | 0.03546 |
| 346 | 5,997 | 0.04646 |

A shallow U with its minimum at the depth-matched point, degrading in both
directions, too deep and informed trading cannot move price, too thin and noise
traders push it around. Nothing anywhere near precision-weighting's 0.01436.

> An 8-trial pilot of this sweep showed a clean monotonic improvement toward
> shallow markets (0.0184 at 12 shares/tick) and would have supported a tidy
> story about depth being the binding constraint. At 200 trials that number is
> 0.0322. The pilot was noise. It is recorded here because it is exactly the
> result that would have been reported if the sweep had stopped where it looked
> most interesting.

## What is actually binding: measured, not inferred

If agents were constrained, they would end sessions pinned at their limits. They
do not:

| venue | mean \|position\|/limit | at limit | \|price − own estimate\| | own uncertainty |
|---|---|---|---|---|
| order book | 0.47 | 38% | 19.1 ticks | 40.8 ticks |
| scoring rule | 0.44 | 38% | 18.9 ticks | 40.8 ticks |

> These four columns were measured on the build before the open-order reserve
> and have not been re-measured. `TrialResult` carries forecasts, not
> end-of-session positions, so neither command on this page reproduces them and
> the run that produced them was instrumented separately. Every other figure on
> this page is from the current code. Since the reserve charges collateral for
> a working order from the moment it is acknowledged, capacity is exactly the
> axis it could have moved, so treat this table as the weakest evidence here
> until it is run again.

Agents use under half their capacity, and the price ends up **well inside** each
one's uncertainty band. They stop because they are satisfied, not because they
have run out of room, which is exactly what their rule says to do: stop once
the price is within `patience × uncertainty` of your estimate.

The two venues produce these numbers to two decimal places of each other. That
is the whole of Experiment 2 in one table: the binding constraint lives in how
an agent converts belief into pressure, and both mechanisms are downstream of it.

So the price rests where pressure balances. A sharp agent has a narrow band and
keeps pushing until price is close to its view; a vague agent has a wide band
and gives up early. That *is* a form of information weighting, it is simply far
too weak to beat weighting by evidence directly, because a vague agent's early
position moves price just as far as a sharp one's.

## What this sets up next

Three things are now ruled out: the mechanism, liquidity depth, and trading
capacity. What remains is the hypothesis Experiment 1 named and this one
sharpens, **nothing makes a well-informed agent bet bigger than a poorly
informed one.** Two ways to test it, and they are complementary:

1. **Kelly-proportional sizing.** Let an agent's stake scale with its edge over
   its uncertainty without the conviction cap. If the market then approaches
   precision-weighting, the gap was a sizing rule all along.
2. **Wealth that tracks skill.** Carry P&L across repeated trials with the same
   agents and let capital accumulate. If the market migrates from the simple
   mean toward precision-weighting over time, markets aggregate by evidence only
   once they have had time to reallocate capital toward the people who were
   right, which would be a far more interesting claim than either experiment
   here has established.

## Reproducing

Every run writes `results/<tag>_manifest.json` (config, seeds, git commit,
per-module code digests, results digest) and `results/<tag>_trials.csv`. Trials
are independent and deterministic from their own config, so `--workers` changes
runtime only. `tests/test_experiment.py` asserts that the same config produces
byte-identical results and that a different seed does not.
