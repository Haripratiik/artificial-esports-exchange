# The Economy: the world, the contracts, and how they settle

A precise description of what these contracts are written on and how a settlement value is
produced, written to be argued with. The last section lists every judgment call I made that
could reasonably have gone the other way.

The world is a **synthetic esport**: twelve competitors of this project's invention, two
formats, and a season generated from a seed. It is faithful to the shape of a genre because that
shape is what makes the markets interesting, and it borrows nothing from anybody. Formats and
mechanics are ideas and are free to model; names are not, so every competitor, mode, metric and
ticker here is our own.

---

## 1. Layering

```
  arena.contracts       what a contract IS        generic, no world knowledge
  arena.settlement      how a contract SETTLES    generic, talks to an Oracle protocol
  arena.worlds.circuit  what the numbers MEAN     all domain knowledge lives here
```

The engine never imports from `worlds/`. `settle()` takes a `ContractSpec` and anything
satisfying the `Oracle` protocol. That is the seam a second data-generating world plugs into,
and it is enforced by the import graph rather than by convention. It has been used once
already, for exactly the reason a seam exists: the world this exchange settles on was replaced
wholesale, and nothing in `contracts/`, `settlement/`, `exchange/` or `market/` had to know.

---

## 2. Data model

### The world is generated, not collected

There is no corpus. `arena.worlds.circuit` produces the season the contracts settle on:

```
roster.py    twelve competitors, six archetypes, one latent strength per format
modes.py     what a format is, and what a result has to satisfy to be a result
match.py     one match, a pure function of (seed, match_id, format)
metrics.py   six quantities measured on a run of match records
oracle.py    the calendar that turns a date range into a run of match ids
```

A match is drawn from the world seed and its own number and reads no clock, so the same seed
always produces the same season and a replay reproduces every result exactly. That is a
stronger reproducibility claim than a frozen dataset can make, because there is no file anyone
has to still have.

### `MatchResult`: the whole of what settlement may see

```
match_id
format_name
field           who was in it
placements      1 is the winner; in a team format every member of the winning side is 1
eliminations    credits, which must conserve
scores          only where the format keeps score, empty otherwise rather than zero-filled
```

**Latent strength is not in it, and that is structural rather than polite.** `metrics.py` does
not import `roster.draw_strengths`, every metric has the signature `(results, subject)` so
there is no argument a strength could arrive through, and `oracle.py` does not import `roster`
at all. The oracle reports what matches did, not what the roster is. A market that could read
the parameter generating the outcomes would be pricing a lookup, and every forecasting result
measured on it would be a result about arithmetic rather than about aggregation.

`scores` being empty rather than zero-filled is the same discipline one level down. A zero
meaning "this format keeps no score" and a zero meaning "this side scored nothing" are
different facts, and a column that cannot tell them apart will eventually settle one as the
other.

### `MatchCalendar`: the only bridge between dates and matches

A contract names a window of dates. A window has no answer until something decides which
matches those dates name, and the calendar is the whole of that decision: two constants, an
epoch and a period, because every additional degree of freedom is a way for two participants to
disagree about which matches a contract meant.

Match `n` occupies the slot starting at `epoch + n * period`, and a window `[start, end)` holds
exactly the matches whose slot starts inside it. Both endpoints round the same way, up, so
consecutive windows tile the season: no match lands in two settlements and none falls between
them. The default is one match every five minutes from the epoch, which is 288 a day per
format.

A window is capped at `MAX_WINDOW_MATCHES`, 8,640, thirty days at that schedule. The cap is
what makes the ceiling `match_volume` declares a fact rather than a hope: a subject cannot
appear in more matches than the window holds, so no schedule, seed or roster can exceed the
bound collateral is sized against.

---

## 3. The metrics

Six live in a registry; a contract naming anything else fails at resolution rather than
silently measuring something adjacent. Every one is a pure function of the match records.

    win_rate                how often the subject's side won, per appearance
    eliminations_per_match  credited eliminations per appearance
    match_volume            appearances in the window
    placement_dispersion    how widely the subject finishes around its own mean placement
    format_dispersion       how differently the subject performs across the formats it played
    score_margin            average normalized points margin

**They are deliberately not six views of one number, and the measurement says how hard that was
to arrange.** The world carries one latent scalar per competitor per format, so within a single
format every skill statistic ranks the field almost identically: over 4,000 solo matches on
seed 7, eliminations per match ranks the field exactly as the win rate does, Spearman
**+1.000**. That is a fact about the world rather than a defect in the metrics, and it is why
the set is separated along the axes that genuinely do carry different information:

- **the two formats**, whose win rates rank the field at Spearman **-0.357**;
- **scheduling volume**, which cannot depend on strength because the field is drawn without
  reading it;
- **second moments**, which are not monotone in the scalar the first moments are monotone in.

It is also why depth on this exchange comes from formats and matches rather than from listing a
seventh statistic on the same mean.

### On pooling formats: refuse, do not standardize

A win rate in a ten-way elimination sits against a neutral point of 0.100 and a win rate in a
three-a-side objective sits against 0.500, so an average over a window holding both is a
function of how many of each the schedule happened to run. A contract on it would be pricing
the schedule as much as the competitor.

The previous world met the same problem and answered it by standardizing onto weights pinned
before the window opened. Here the answer is cleaner: **refuse**. Every per-match average
checks that the matches it was handed all came from one format and raises otherwise, so a
contract on a win rate has to name its format. `match_volume` is the exception and says so: a
count of appearances means the same thing whichever format produced it, so it pools freely.

Refusing beats standardizing wherever refusing is available, because a standardized number is
still a number and still settles. A refusal cannot be traded on by mistake.

### On absence: raise, do not return zero

A subject that never appeared raises rather than returning 0. Zero is a real value here: a
competitor that entered forty matches and won none has a win rate of 0.0, and a competitor that
entered none has no win rate. Collapsing the two settles a contract on evidence that does not
exist, which is the exact shape of failure the venue refuses everywhere else.

The same applies one step further in. A dispersion over a single appearance is exactly 0.0 and
perfectly well formed, so the metric refuses it rather than handing the engine a number it
would happily settle. `InsufficientEvidence` is a normal outcome rather than a failure: the
settlement engine turns it into a void with a reason attached.

### Kinds, and why the distinction is load-bearing

`METRIC_KINDS` marks each metric `rate`, `quantity` or `dispersion`. The test that matters is
scale invariance in the window length, not whether the number sits in [0, 1].

`match_volume` is the one **quantity**: doubling the window doubles it, so a contract on one
week and a contract on the next are different contracts. Everything else is scale invariant.
`eliminations_per_match` is a rate despite not being a proportion, and `score_margin` is a rate
despite being able to go negative.

The distinction has already cost real money in the market. A build step re-dated every contract
onto a four-week prior window without rescaling a quantity, so a one-week volume contract
settling at 71.09 was handed to informed traders with a prior of 274.92: a ratio of 3.87x
against a window ratio of 3.87. The rate contracts on the same run came back at 1.00 to 1.04x,
which is what identified the error as a unit rather than a fixture, because a rate is scale
invariant in window length and a count is not. The type system had already said so and was
ignored downstream.

---

## 3b. What the format's rules pin exactly

Most of a world is empirical. A few things are not, and separating them out matters more here
than anywhere else in this layer, because these are the facts the market's coherence is built
on rather than estimated from.

| Format | Entrants | Sides | Winners | Neutral win rate | Eliminations credited |
|---|---:|---:|---:|---:|---:|
| Solo elimination | 10 | 10 | 1 | **0.100** | **exactly 9** |
| Three-a-side objective | 6 | 2 | 3 | **0.500** | one per point, up to the target |

Pooled over all competitors these are arithmetic, not estimates. Ten entrants and one survivor
is a win rate of one in ten by counting; two sides and one winner is one in two.

**The elimination law is enforced, not assumed.** `fmt.check` is asked on every match the world
plays and on every hypothetical outcome the pricing ensemble draws, and a result that violates
it raises rather than settling. Measured: eliminations sum to exactly nine in every one of
4,000 solo matches. This is the world's version of `conservation_check` and it is held to in
the same spirit. A match whose arithmetic does not close is not a match with a small error in
it, it is a match that did not happen.

The team check needed a second condition that counting alone missed. Requiring three winners
accepted a winning trio drawn from both sides, which is not a match anybody played, and the
damage was specific and silent: both "this side wins" contracts settle to zero, so a set that
must sum to one sums to zero, with nothing raising. The winners now have to *be* one of the
sides that played.

### Two consequences

**Neutral points are taken from the rules, not fitted.** An exact constant beats any estimate of
the same quantity, and it means a thin or skewed window cannot drag the reference level around.

**The pinned numbers are a free, exact correctness check.** Pooled across all competitors the
win rate in a format must come out at its neutral point, and eliminations must sum to
`entrants - 1` every time. A material gap means competitors are being dropped from results,
matches double-counted, or credits misattributed. Nothing else in this project catches those
failure modes.

---

## 4. The reference: identity without a snapshot

A collected world pins a snapshot: frozen weights, frozen priors, an `as_of` date and a file
digest, so a settlement can be checked years later against the exact evidence it used. A
generated world has no crawl to pin and no snapshot to freeze, and it still needs an identity to
check a contract against and a date to check for lookahead.

Both fall out of the calendar.

**The identity** is `"circuit-"` plus a digest over the seed, the calendar, and the declared
shape of every registered format. So a contract pinned to one season can never be settled by an
oracle configured for another, and registering a format that changes what a match is moves the
identity rather than quietly changing the answers. The roster is deliberately not in it:
reading the roster would mean importing the module that draws strengths, and the identity is
not worth that.

**The date** is the epoch, because the generator was fixed when the circuit opened and every
window this oracle will answer begins at or after that moment. The engine's lookahead check and
the oracle's own refusal of a pre-epoch window are two statements of one fact, and neither can
fire without the other being wrong.

What this removes is worth naming, because it was most of the previous version of this
document. There are no standardization weights to estimate, no shrinkage strength to fit, no
per-stratum priors, no coverage gate, no missing-strata policy, no design effect, and no
question about whether a sample is representative. Those were all answers to one problem: the
underlying was somebody else's sample and the sample moved. A generated season is a census of
itself.

What it costs is external validity, and that is a genuine loss rather than a technicality. No
measurement here is evidence about how a real venue on a real underlying behaves.
`docs/PLAN.md` records the trade under Phase 8.

---

## 5. The contract

### `MetricRef`: a fully qualified measurable quantity

```python
MetricRef(metric="win_rate", subject="VANTA",
          modes=("solo",), maps=("ALL",), trophy_buckets=("ALL",))
```

Filters must be sorted, duplicate-free, and never mix `"ALL"` with explicit values, so two
contracts naming the same universe compare and hash equal regardless of how they were written.
`"ALL"` is spelled out rather than represented by an empty tuple, so a truncated spec file cannot
silently widen a contract's universe.

`modes` carries format names in this world, and a per-match average has to name one, for the
reason argued in §3. Prefer `metric_ref("win_rate", "VANTA", modes=("solo",))` to building one
by hand: it attaches the metric's declared bounds and kind, both of which are easy to get wrong
and load-bearing for collateral.

**Two of these field names are older than the world they describe.** `maps` and
`trophy_buckets` were filters on the previous underlying. `maps` now carries a `match-N` tag on
a contract written on a live match and `ALL` otherwise, and `trophy_buckets` is always `ALL`.
They are part of the spec digest and appear in the API payload, so renaming them is a breaking
change rather than a tidy-up, and they are recorded here rather than quietly left to be
misread.

### `Underlying`: a closed three-node algebra

| Node | Meaning | Instrument family |
|---|---|---|
| `Single(ref)` | one metric | performance future |
| `Difference(a, b)` | a − b | relative-value spread |
| `Basket(((leg, w), ...))` | Σ wᵢ · legᵢ | class / meta index |

Closed on purpose. Settlement must be auditable years later, and an algebra with three
constructors can be reasoned about exhaustively, unlike an expression string evaluated from
YAML, which is both a correctness and a security problem.

`atoms()` returns deduplicated, sorted `MetricRef`s, so each metric is resolved exactly once and
resolution order does not depend on how the tree was nested. `Basket.evaluate` accumulates in
canonical-shape order, because float addition is not associative and reordering legs in a config
file must not move the last bit of a settlement value.

### `Payoff`

| | |
|---|---|
| `Linear(scale, offset)` | `scale·level + offset` |
| `Binary(comparison, threshold, payout)` | `payout` if the comparison holds, else 0 |

`Linear(scale=10_000)` moves a rate in [0,1] onto a price grid with useful tick resolution: a
0.5537 win rate settles at 5537, so one tick is a quotable amount rather than a rounding artifact.

Nothing enforces a 0–1 price range on binaries. Whether their price is readable as a probability
is a question this project exists to *test*, not an identity to assume.

Options are absent by design, an option is a function of a *traded future*, not of the raw
metric, so it belongs to a later phase and a different module.

### `ContractSpec`: immutable, content-addressed

```python
ContractSpec(contract_id, underlying, payoff, window, policy,
             reference_id, published_at, tick_size="0.01", lot_size=1, metadata=())
```

Two invariants enforced at construction, both lookahead bugs that would otherwise be invisible:

1. **`published_at <= window.start`**: a market written after its outcome has begun forming is
   lookahead by construction.
2. **`reference_id` is required**: the season must be pinned; it cannot be left implicit. A
   contract that did not name one could be settled by an oracle running a different seed, which
   is the same match number played by a different field.

`spec_digest` is `sha256` over canonical JSON of the whole spec.

### `DataPolicy`: the evidential bar

| Field | Guards |
|---|---|
| `min_sample_size` | the metric as a whole |
| `missing_data_policy` | `VOID` only, currently |
| `min_stratum_battles` | nothing here. Vestigial |
| `min_strata_coverage` | nothing here. Vestigial |
| `missing_strata_policy` | nothing here. Vestigial |

`min_sample_size` is the bar the engine applies to the finished number, and it is the one bar
that genuinely is a post-hoc check. The metric applies its own bar first, refusing evidence it
cannot support at all rather than returning a well-formed number for the engine to wave
through.

The last three fields are composition guards belonging to a standardized metric, and this world
has no strata to be composed of. They default to inert, they remain part of the spec digest, and
they are named here rather than silently ignored: a policy field that does nothing is worse
than an absent one, because a contract can be written with it set and read as though it took
effect.

---

## 6. Settlement

`settle(spec, oracle)`, one function, deliberately small enough to verify by reading.

```
1. reject if oracle.reference_id != spec.reference_id      -> raises ReferenceMismatch
2. for each atom in spec.atoms()  (canonical order):
       resolve via oracle                                  -> MetricUnavailable ⇒ VOID
       if sample_size < policy.min_sample_size              -> VOID
3. level = spec.underlying.evaluate(resolved values)
4. raw   = spec.payoff.apply(level)
5. value = quantize_to_tick(raw, spec.tick_size)            -> ROUND_HALF_EVEN
```

Steps 2 and 5 are the ones people skip. Skipping 2 settles on evidence the contract declared
insufficient; skipping 5 prints a closing value the exchange cannot represent.

**Errors vs. voids.** A VOID says *the world did not supply enough evidence*, a normal outcome,
recorded with a reason. A raise says *the experiment is wired up wrong*. `ReferenceMismatch` is a
raise, because silently voiding a config error would hide it behind a plausible market outcome.

**Void records keep partial evidence.** A spread whose second leg fails still records the first
leg's resolution. "Voided because Quill's sample was thin" is far more useful than "voided".

### `SettlementResult`

Carries `contract_id`, `spec_digest`, status, settlement value (`Decimal`, or `None` if void),
underlying level, every `MetricResolution` with source digests and diagnostics, and a void reason.
`result_digest` covers all of it.

**No wall-clock timestamp.** Determinism means identical inputs produce byte-identical output; a
`computed_at` field would break that for nothing. *When* a settlement ran belongs in the run
manifest, that is about the experiment. The record is about the world.

---

## 7. Determinism

Four mechanisms in `arena/determinism.py`:

| | |
|---|---|
| `canonical_json` | sorted keys, no whitespace, rejects NaN and unserializable types |
| `digest` | `sha256:` over canonical JSON |
| `stable_sum` | fixed accumulation order, smallest magnitude first |
| `quantize_to_tick` | Decimal at 60-digit precision, `ROUND_HALF_EVEN` |

Half-even rather than half-up because repeated settlements under half-up drift systematically
upward. `quantize_to_tick(5537.125, "0.25") == 5537`, not 5537.5.

Verified: same spec + same data ⇒ identical `result_digest`; changing the payoff, tick size,
policy, contract id, reference id, or dataset each changes a digest.

---

## 8. Lookahead prevention

A lookahead bug does not crash and does not produce implausible numbers. It produces *better*
results. That is the worst possible failure mode, because nothing about the output invites
suspicion, so every channel is closed structurally rather than by a check somebody has to
remember.

| Channel | Mechanism |
|---|---|
| Contract written after the fact | `published_at <= window.start`, enforced in `ContractSpec.__post_init__` |
| Contract settled by the wrong season | `oracle.reference_id != spec.reference_id` raises `ReferenceMismatch` |
| Season fixed after the window opened | `reference_as_of <= window.start`, enforced in `settle()`, raises `ReferenceLookahead` |
| A window before the circuit existed | the oracle refuses it, which is the same fact as the row above seen from the other side |
| Settlement reading the parameter it should be measuring | `metrics.py` does not import `roster`, no metric takes a seed, and `MatchResult` carries no strength field |
| An agent reading the parameter | the same three facts, since an agent resolves through the oracle like anything else |

The fifth row is the one this world had to invent, and it replaces a whole family of dataset
visibility rules that a collected world needs. When the underlying is generated, the dangerous
leak is not future data, it is the *generating parameter*: an agent that could see a
competitor's latent strength would forecast perfectly, the price would be exact, and the
information-aggregation result would be a statement about arithmetic. Closing it by convention
would be hopeless. It is closed by the import graph and by function signatures, so there is no
argument through which a strength could arrive even by accident.

The research harness is the deliberate exception. It evaluates the market against outcomes the
market could not have known, which is the entire point of measuring forecast error.

---

## 9. Judgment calls

### Resolved

| # | Was | Now |
|---|---|---|
| 1 | An underlying that had to be crawled, and a metric standardized onto pinned weights so contracts did not price the crawler | A generated season. No weights, no shrinkage strength, no coverage gate, no representativeness question. See §4 |
| 2 | A pooled rate across formats, with the neutral point moving as the schedule moved | Refused. A per-match average names its format or the oracle raises. `match_volume` pools and says why |
| 3 | A match result trusted because the generator produced it | Checked against the format every time, and every drawn outcome in the pricing ensemble is checked too |
| 4 | A team result validated by counting winners | Validated against the sides that actually played, after counting alone accepted a trio drawn from both |
| 5 | Absence returning zero | Raises. `InsufficientEvidence` becomes a void with a reason, and zero stays available as a real value |
| 6 | Metric bounds written at the call site | Attached by `metric_ref` from `METRIC_BOUNDS`, because a wrong bound sizes collateral against the wrong worst case |

### Still open

**1. Bounds are honest but loose in two places.** `eliminations_per_match` is bounded at 9,
which the elimination law makes reachable in principle by a single survivor, but over 4,000
matches per format the largest single-match credit anyone took was 8 in the solo mode and 3 in
the objective, and a window average of 9 would need every credit in every match to go to one
competitor. `match_volume` is bounded at the 8,640-match window ceiling against a realistic
978 to 1,710 appearances a week. Both overestimates cost collateral, where an underestimate
would make settlement raise, so the direction is deliberate. Tightening them means proving a
bound rather than observing one.

**2. `DataPolicy` carries three fields that do nothing here.** `min_stratum_battles`,
`min_strata_coverage` and `missing_strata_policy` guard the composition of a standardized
metric, and this world has no strata. They are still part of the spec digest, so removing them
is a breaking change. A policy field that silently does nothing is worse than an absent one.

**3. `MetricRef.maps` and `MetricRef.trophy_buckets` are named for a world that is gone.**
`maps` now carries a `match-N` tag and `trophy_buckets` is always `ALL`. They are in the public
API payload as well as the digest, so this is a rename with consequences rather than a tidy-up.

**4. `min_sample_size` applies per-atom, uniformly.** For a spread, both legs face the same bar
regardless of how sensitive the payoff is to each. Arguably the bar should reflect the
contract's sensitivity.

**5. `VOID` is the only missing-data policy.** No fallback to a wider universe, no partial
settlement. Conservative, and I think correct to start, but real exchanges do have fallbacks.

**6. Basket weights need not sum to 1.** Deliberate, since an index can be a sum rather than a
mean, but it means payoff scale and weight normalization interact in a way that is easy to get
wrong.

**7. No YAML round-trip.** Specs are only constructible in Python, so they cannot be authored
outside code or diffed readably in git. Cheap to add; left out to keep the settlement core
dependency-free.

**8. Mixed float64 and Decimal.** The metric is float64 throughout, converted to Decimal only at
tick quantization. IEEE-754 `+`, `*`, `/` are deterministic across platforms in a fixed order,
so I believe this is sound, but it is an assumption worth stating rather than burying. The match
pricing path is the exception and goes further: prices come back as `Fraction` with denominator
equal to the ensemble size, so the relations between contracts on one match are exact rather
than exact to within a rounding.

---

## 10. Test coverage

Counted this session with `pytest --collect-only`. These numbers move as tests are added, and
the point of listing them is which files carry the weight rather than the totals.

| File | Tests | Covers |
|---|---:|---|
| `test_circuit_world.py` | 33 | the roster's silence about strength, format checks, match determinism, the six metrics, format pooling refused, absence raising, the calendar tiling windows |
| `test_settlement.py` | 25 | determinism, tick grid, provenance, digest sensitivity, every void path, reference mismatch and lookahead, leg-order invariance |
| `test_contracts.py` | 25 | lookahead invariants, visibility semantics, spec validation, determinism primitives |
| `test_match_markets.py` | 25 | the 395 relations at exact zero excess, the ensemble mirroring `match.play`, boundary prices, quoting a zero fair value |
| `test_match_operator.py` | 13 | opening on a schedule, revealing eliminations, settling only what is certain, one clock |

Four are load-bearing:

- `test_the_oracle_cannot_reach_latent_strength`, with
  `test_a_metric_is_a_function_of_the_record_and_nothing_else` and
  `test_a_resolution_carries_no_trace_of_a_strength` behind it. If these fail the exchange is
  pricing a lookup and every result in the repository is about arithmetic.
- `test_no_settleable_outcome_can_violate_a_relation`, with
  `test_one_distribution_violates_no_relation`. If they fail there is a riskless trade sitting
  in a match book. `test_pricing_each_contract_on_its_own_law_is_where_the_arbitrage_comes_from`
  is the control that shows the alternative is not free.
- `test_a_competitor_who_never_appeared_is_refused_rather_than_zeroed`, paired with
  `test_a_competitor_who_appeared_and_won_nothing_settles_at_zero`. The pair is the point: one
  zero is a measurement and the other is an absence.
- `test_the_reference_can_never_postdate_a_window_the_oracle_answers`, the lookahead channel
  that makes results *better*, and therefore the one nobody would catch by inspection.
