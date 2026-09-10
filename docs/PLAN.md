# Artificial Esports Exchange: Attack Plan

**Status:** planning document, kept as a record. Parts I and IV describe decisions that have
since been overtaken, and where they have been the entry says so rather than being deleted.
**Stack:** Python throughout. See finding 3 below for what happened to the C++ half.

---

## Part I: What the research changed

The kickoff spec is sound. These findings changed *how* it was sequenced, and two of them were
later overturned by measurement, which is recorded here rather than tidied away.

### 1. ~~The data layer is the critical path.~~ Overturned: the world is generated.

The plan opened with a data-collection track, on the reasoning that the underlying had to be
crawled from somebody else's service, that nothing else in the project accrued value with
wall-clock time, and that the crawl therefore had to start on day one and never be blocked.

That whole track is gone. The exchange now settles on a synthetic esport this repository
generates: `worlds/circuit/` draws a season from a seed, plays each match as a pure function of
that seed and its own number, and the oracle replays the matches a window names. The decision
was an intellectual-property one in the first instance, since the previous underlying was
another party's published statistics under names that are their trademarks, and formats and
mechanics are ideas that anyone may model where names are not. But it paid for itself twice
over on the research grounds this document already cared about:

- **The critical path disappeared.** There is nothing to wait for and nothing to leave running.
- **A crawl can never be representative, and a generated season does not have to be.** A
  statistic built from a crawl moves when the crawler's reach moves, which is why the old metric
  needed standardization onto pinned weights to stop contracts pricing the crawler. A generated
  season is a census of itself.
- **Settlement is re-derivable from a seed** rather than from a corpus somebody has to hold and
  keep, which is a much stronger form of the reproducibility this project already claimed.

What is lost is external validity, and that is a real cost rather than a technicality. See the
note on Phase 8.

### 2. ~~A third-party statistics site is a bootstrap reference.~~ No longer applicable.

The plan used an outside site's published aggregates to sanity-check our own and to shape
priors. With the world generated, there is nothing outside to check against and no prior to
borrow: the metrics are defined in `worlds/circuit/metrics.py` and computed from match records
this repository produced.

The one point worth keeping from that finding outlives its source. A rate is only well defined
*relative to a stated population*, and the settlement schema still forces a contract to name
one. The circuit world enforces the same thing in a sharper form: a per-match average has to
name its format, because a win rate sits at a neutral 0.100 in the ten-way elimination and 0.500
in the three-a-side, and the oracle refuses to answer a contract that pools them.

### 3. ~~Match-outcome binaries are the highest-risk family. Defer them indefinitely.~~ Inverted.

This entry argued that match-outcome binaries carried the most external risk and the least data,
and should be deferred indefinitely in favour of statistical contracts.

Both halves have reversed. The matches are ours, so there is no external risk in listing them,
and they are generated, so they are the one family that is never short of data. They are now the
larger half of the exchange: a ten-entrant solo match lists 270 contracts and a three-a-side
lists 38, and 316 of the board's 355 markets are prediction markets where 8 were before. They
also fixed the problem the statistical contracts could not, which is that nothing resolved
inside a session, so a trader had no terminal event to be scored against.

Nothing about this makes the exchange a betting product. The capital is imaginary, there is no
cash-out, every counterparty is simulated, and the outcomes are drawn from a seed rather than
observed anywhere.

### 4. The market-making adaptation the spec flagged as an open problem now has a published answer.

The spec notes Avellaneda–Stoikov "will need adaptation because this exchange's contracts have
bounded / settling payoffs." *Optimal Market Making in Prediction Markets* (arXiv 2607.17991,
July 2026) does precisely that adaptation:

- model a latent belief process `L_t` and set `p_t = f(L_t)` with logistic `f`, so price stays
  in (0,1) by construction rather than by clamping;
- price evolves as `dp_t = ς(t, p_t) dW_t` with **state-dependent volatility that vanishes at
  the boundaries**, a 0.02 contract simply cannot move like a 0.50 contract;
- add a **terminal settlement penalty** `Φ(p_T, q_T) = −γ_T · q_T² · p_T(1−p_T)`, the variance
  of the settlement value of inventory still held at resolution. This term has no analogue in
  Avellaneda–Stoikov and is the whole difference;
- no closed-form reservation price; optimal quotes come from the value function numerically.

We get a principled, citable market maker instead of an improvised one.

### 5. The obvious version of Experiment 1 has a baseline that is too weak to be interesting.

"Does the market beat its constituent agents?" is nearly free to win against the *worst* agent
or a naive average. The literature is sharper than that: Atanasov et al. (*Management Science*)
find prediction markets beat the **simple mean** of forecasters, but **lose** to prediction polls
once those are aggregated properly, temporal decay, performance weighting, and recalibration.

So the experiment must be run against a **ladder** of baselines:

| Baseline | Why it's there |
|---|---|
| Best single agent (ex post) | Upper bound on any individual |
| Simple mean of agent beliefs | The weak baseline the literature already beat |
| Precision-weighted mean | Uses each agent's own uncertainty |
| Recalibrated / extremized aggregate | The one that actually beats markets in the literature |

If the market only beats rows 1–2, that is a known result. If it beats row 4, that is a finding.
Designing this in from the start is the difference between a project and a *result*.

---

## Part II: The design idea this points to

The single most useful consequence of the above:

> **Model information heterogeneity as different sample sizes drawn from the same
> data-generating process.**

Instead of hand-tuning "agent A gets Gaussian noise with σ=0.03", give agent *j* a sample of
`n_j` matches from the world's own generating process. Then:

- its posterior over the win rate is an exact Beta/Normal update, no arbitrary noise model;
- information quality has a **unit**: one match observed;
- "how much is faster/better information worth?" becomes a measurable dollar-per-match number;
- ground truth `p*` is known exactly, so Brier score, calibration, and *resolution* can be
  decomposed properly rather than estimated from one realization;
- it maps directly onto Kyle-style informed-vs-noise structure, and an agent's `n_j` is
  literally how much of the season that trader has watched.

This is why the synthetic world is not a shortcut. For the core information-aggregation
question it is **strictly better instrumentation** than real data, because real data gives you
one realization and no `p*`. The world is now synthetic all the way down, which sharpens the
instrument and gives up the external-validity comparison entirely; see Phase 8.

A second structural property worth exploiting, specific to these contracts:

> **Volatility has a known term structure.** A win-rate future settling over window
> `[T₀, T₁]` decomposes at time *t* into an already-realized part and a still-unknown part:
> `W_T = (n_seen/n_total)·W_seen + (n_left/n_total)·W_future`.
> The conditional variance shrinks *deterministically* as the window fills.

A live match is the same property at a much shorter horizon and with the resolution visible.
Each elimination removes an outcome from the set, so the still-unknown part shrinks in steps
anyone can watch, and the contract on the competitor who just went out does not decay toward
zero, it settles at zero.

Equities do not do this. It gives the market maker an analytic σ(t), gives options a predictable
IV decay whose violations are informative, and gives a clean convergence trade near expiry. It
is a genuinely distinctive feature of the asset class this project invents.

---

## Part III: Two tracks

```
TRACK A  (data)     retired. See Part I, finding 1.

TRACK B  (engine)   day 1 ──> contracts ──> exchange ──> agents ──> experiments
                    all effort, and now the only track
```

The seam the two tracks were to meet at is still there and still earns its place, because it is
what lets a second world be added without touching the exchange: a `World` protocol yielding
(a) timestamped observations and (b) settlement truth, with the exchange, the agents and the
experiment harness never knowing which one is underneath. One world implements it today. The
engine never imports from `worlds/`, and that is enforced by the import graph rather than by
convention, so the seam is a fact about the code rather than an intention.

---

## Part IV: Phases

Each phase has an exit test. Do not start the next phase until it passes.

### Phase 0: ~~Start the clock~~ **RETIRED**
Track A was a crawler, a frontier and an append-only store, built to accrue a corpus while the
engine was written. It is gone with the world it fed, and its exit test with it. What replaced
it is not a faster crawler: `worlds/circuit/` generates the season, so the phase that had to
start on day one and run forever now takes a seed.

### Phase 1: The economy (Python): **DONE**
Contracts and settlement before any exchange, per the spec's Milestone 0.
Implemented in detail in [ECONOMY.md](ECONOMY.md), including the open judgment calls.
- Canonical metric definitions, written down in code rather than in prose: six metrics over a
  window, each naming its format, each bounded by construction, each a pure function of the
  match record.
- Contract spec model, content-addressed by digest; underlying algebra (single / difference /
  basket) so futures, spreads, and indices come from one mechanism.
- Deterministic settlement engine: min sample size, missing-data policy, provenance, tick
  quantization.
- **Exit:** same contract + same dataset ⇒ byte-identical settlement digest, and a mutated
  reference-weight file changes that digest.

### Phase 2: Exchange kernel, Python reference: **DONE**
Deliberately in Python first.
- Price-time priority book, limit/market/cancel/replace, partial fills, deterministic sequence
  numbers, trade tape, L2 snapshots.
- Discrete-event kernel: priority queue, agent wakeups, pairwise latency matrix + latency noise
  (ABIDES' design, which is itself modeled on NASDAQ ITCH/OUCH messaging).
- Property tests: no crossed book, conservation of quantity, replay determinism under seed.
- **Exit:** adversarial matching-engine suite green; seeded run reproduces bit-for-bit.

### Phase 3: Minimal artificial market
- ~hundreds of noise agents, one fundamental agent, one inventory-skew market maker.
- One instrument: a linear competitor-performance future.
- **Exit:** a full session produces plausible order flow, trades, inventory, and PnL that
  survives eyeballing.

### Phase 4: ~~C++ kernel~~ **NOT SCHEDULED, and the profile is why**
This was to be where the C++ half landed, with the Python engine as the **correctness oracle**
so the port would be validated rather than merely written. The ordering was right and the
premise was not. Profiled over two simulated minutes of the live market, the time is in
`kernel.send`, `latency.delay` and the book snapshot the venue broadcasts; matching does not
appear in the fourteen most expensive functions by self time. Porting the matcher would move a
small share of a total spent on message plumbing. The differential harness was built anyway and
is worth keeping green, so if a profile ever asks for the port the specification is already
there. What follows is what the port would have been.
- C++20 core, `scikit-build-core` + CMake, **nanobind** bindings (vs pybind11: ~4× faster
  compiles, ~5× smaller binaries, ~10× lower call overhead, and one `ndarray` type that works
  across NumPy/JAX/PyTorch, which matters given the JAX ambitions later).
- Toolchain: MSVC Build Tools 14.50 / VS 2026, defaults to C++20 and ships CMake 4.1.1, so
  it is one install. *(Neither a compiler nor CMake is currently on this machine.)*
- Differential test: identical order streams through both engines must produce identical tapes.
- **Exit:** engines agree on randomized order streams; C++ is meaningfully faster on a
  throughput benchmark.

### Phase 5: Synthetic world + heterogeneous information: **DONE**
- `World` protocol, with the circuit world behind it and the engine importing nothing from
  `worlds/`.
- Agent information = `n_j` sampled matches. No-lookahead information interface.
- **Exit:** an agent's forecast error scales as `1/√n_j` as theory demands.

### Phase 6: Experiment 1, done properly
- Market vs the **full baseline ladder** (best agent / simple mean / precision-weighted /
  recalibrated-extremized).
- Brier decomposition into calibration + resolution + uncertainty against known `p*`.
- **Exit:** a defensible result with error bars, including the case where the market *loses*.

### Phase 7: Multi-asset and microstructure: **mostly done**
Spreads, class indices, cross-market arbitrage, stat-arb; then latency tiers, maker/taker fees,
queue position, adverse-selection diagnostics.

Eight instrument classes at that point, all derived from the contract rather than declared:
`future`, `event`, `call`, `put`, `spread`, `index`, `commodity`, `equity`. The last two
are the ones that needed new machinery rather than a new combination.

A **commodity** is written on an amount delivered rather than on a proportion, which
makes its delivery window part of the contract and gives consecutive weeks a term
structure instead of four copies of one thing. A **share** pays before it settles, which
is the whole difference between a share and a future, and it needed a distribution that
moves cash between holders while narrowing the range collateral is computed from, so
that meeting an obligation cannot be what makes an account insolvent.

Deliberately not built: a perpetual. Every contract here settles inside a known interval,
which is what makes collateral arithmetic rather than a value-at-risk estimate, and a
claim that never settles has none. `docs/GAPS.md` carries the reasoning.

Both of the things left open here are now closed. The option surface was internally
inconsistent for two reasons, and the larger one was not the market maker: every agent
held a separate view of the same competitor for every contract written on it, so its own
option ladder was neither monotone nor convex and it traded on the difference. The maker
now quotes each chain off one distribution, the agents hold one view per underlying, and
the arbitrageur enforces vertical and butterfly bounds as well as identities. A share is
related to the four weekly futures it pays, exactly, because the legs are listed.

The exchange also runs its own machinery now: maker-taker fees, an opening call auction,
a limit-state circuit breaker, three market makers, and the scoring-rule venue reachable
from the Lab. Turning it on found four bugs in tested code and cost 4.00 percentage points
of pricing error on six paired seeds. Both are recorded in `docs/GAPS.md`.

The cancel-rate gap turned out to be a measurement of a bug: agents keyed their working
orders by an id that is only unique within one book, so with twenty-six contracts they lost
track of most of them and most cancels were never sent. Fixing that, and the auction that
filled orders without telling their owners, took position disagreement between agents and
the ledger from 362 of 494 pairs to 2.

Evidence now arrives over the session rather than all at `t=0`, anchored on the pre-window
level, so the underlying genuinely diffuses. Late-session dispersion of the future goes
from 11 to 279 across six paired seeds, at an accuracy cost of +2.20% of range, 95%
interval [-0.21%, +4.61%].

Stop, stop-limit and iceberg orders are in. Stops are held off the book, because publishing
one says where the market must go to set off a cascade, and a cascade is measured rather
than prevented, with a bound only so that a chain cannot run forever. Icebergs refresh to
the back of their level, which is the priority they pay for hiding.

Nine instrument classes now. The newest, `volatility`, is the first claim here on a
*second* moment: how unevenly a competitor performs rather than how well. On the current
listing QUILL and BASTION sit 60% apart on their solo win rates and 8.9% apart on dispersion,
which is a different ordering rather than a rescaling of the same one, and no other contract on
the exchange could express it. It is bounded, so collateral stays arithmetic.
It also earns its place for a reason that only became visible later: within one format every
skill statistic ranks the field almost identically, Spearman +1.000, so a second moment is
one of the few places a genuinely different number lives.

Tiered tick tables, message throttling and a kill switch are in. Writing the tests for the
last two found that the throttle disabled the kill switch: a runaway is at its message cap
by definition at the moment someone reaches for the switch.

Pegged and minimum-quantity orders are in, and so is the half of a clearing house that
matters here: collateral nets across contracts on one underlying, exactly. A conversion and
a strip are riskless by identity and are now charged nothing for it. Novation is already
true by construction: collateral is exact, so nobody can default and there is nothing to
be protected from.

**Matches, which were not in this plan at all.** The statistical contracts settle once, at
the end of a four-week window, so nothing resolved inside a session and the exchange drained
as contracts expired. A match fixes both: it opens, runs for minutes, and ends with an
outcome that is a fact, and the operator opens another whenever the board has room. Every
contract on one match is priced off a single ensemble of drawn outcomes, so its 395 relations
hold as arithmetic rather than being policed afterwards, measured at zero violations across
22,650 relation checks in exact rational arithmetic. This is the largest thing the exchange
gained after Phase 7 was declared done, and it is recorded here because the plan should not
read as though it predicted it.

Phase 7 is done. What remains of it is deliberate: margin and liquidation stay out, because
leverage is the decision to hold less collateral than the worst case, which replaces an
exact subtraction with an estimate; and simultaneous venues with order routing, which is a
different experiment rather than a missing feature.

The next thing worth building is not on the exchange at all. Options here carry little time
value because the underlying barely moves once its evidence has arrived, and the market's
whole information budget is spent in the first minutes of a session. Experiment 3, wealth
dynamics, carrying P&L across trials to see whether the market migrates from a simple mean
toward precision-weighting as capital tracks skill, is the question the apparatus was built
to answer. Matches give it a second setting to run in, one where the terminal event arrives
every few minutes rather than once.

### Phase 8: ~~Real historical replay~~ **DROPPED**
This was the external-validity result: the same harness over a replay of real history, with one
real patch as the shock. It depended on Track A, and it goes with it.

Worth stating plainly rather than quietly, because it is the one thing the new world costs. No
result in this repository is evidence that a real venue behaves this way. Every claim here is a
claim about a mechanism under conditions this project controls, which is what makes the
measurements sharp and is also exactly what limits them. A future world implementing the same
protocol could restore this without touching the exchange, and that is what the seam is for.

### Phase 9+: Prediction-market venue (LMSR vs CLOB), options and event vol, margin and
liquidation cascades, dashboard. In that order.

---

## Part V: Standing decisions

| Decision | Choice | Reason |
|---|---|---|
| The world | A synthetic esport, generated from a seed | Ours to name; a census of itself; a settlement re-derivable from the seed |
| First instrument | Linear competitor win-rate future | Continuous, and the metric every other statistical contract is built from |
| Match binaries | Listed, and now the larger half of the board | The matches are ours, they never run short, and they resolve inside a session |
| Pooling formats | Refused, not standardized | A win rate is neutral at 0.100 in the elimination and 0.500 in the objective, so a pooled average prices the schedule |
| Match pricing | One ensemble per match, priced by the settlement rule | Every identity that holds outcome by outcome holds in the prices as arithmetic |
| Engine order | Python reference; a port only if a profile asks | The profile puts the cost in message plumbing, not in matching |
| Repo identity | `artificial-esports-exchange` | Names the engine and the world it now carries |
| Disclaimer | Simulated capital, no cash-out, no real venue, no endorsement by anybody | True, and the only claim that needs making now that nothing external is referenced |

Decisions retired with the data track, kept so the table does not read as though they were
never taken: the API proxy that worked around an IP-locked key, the always-on collector host,
the third-party statistics export used to shape priors, and the verbatim third-party notice
that a borrowed underlying required. None of them has anything to attach to now.

The C++ toolchain decisions, nanobind for bindings and scikit-build-core with CMake for the
build, stand as written and are unscheduled rather than reversed. If Phase 4 is ever revived by
a profile, that is what it would use.

---

## References

- Byrd, Hybinette & Balch, *ABIDES*, https://arxiv.org/abs/1904.12066
- *Optimal Market Making in Prediction Markets* (2026), https://arxiv.org/html/2607.17991v1
- Avellaneda & Stoikov (2008), https://doi.org/10.1080/14697680701381228
- Kyle (1985), https://www.jstor.org/stable/1913210
- Atanasov et al., *Distilling the Wisdom of Crowds*, https://pubsonline.informs.org/doi/10.1287/mnsc.2015.2374
- Frey et al., *JAX-LOB* (2023), https://arxiv.org/abs/2308.13289
- nanobind benchmarks, https://nanobind.readthedocs.io/en/latest/benchmark.html
- Plackett (1975) and Luce (1959), the ranking model the match ensembles are compared against
