# Artificial Esports Exchange

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-1473-2ea44f?style=flat)
![Asset classes](https://img.shields.io/badge/asset%20classes-9-0A0A0A?style=flat)
![API](https://img.shields.io/badge/API-REST%20%2B%20WebSocket-0ea5e9?style=flat)
![Collateral](https://img.shields.io/badge/collateral-exact%2C%20not%20modelled-8b5cf6?style=flat)
![Determinism](https://img.shields.io/badge/replay-bit%20identical-0ea5e9?style=flat)

> An electronic exchange built from scratch, in which **a synthetic esport is the economic fundamental**. Matches open, run for a few minutes and settle while they are still being played; statistical contracts settle on the same world's own measured metrics over a window. Nine asset classes clear on one matching engine, every position is collateralised by exact arithmetic instead of a risk model, and money conservation is integer zero rather than approximately zero.

It was built to answer one question with evidence rather than opinion: **does a market aggregate dispersed information better than the agents trading inside it?** Answering that needs an instrument precise enough to trust, so the exchange is complete and runs end to end. A deterministic discrete-event kernel with per-agent latency. A price-time book carrying the order types real venues actually list. Opening and closing auctions, circuit breakers and a participant kill switch. A clearing house that nets portfolios exactly. Live settlement against a world this repository generates and can replay. A population of heterogeneous trading agents. A research harness that runs paired ablations from a manifest. And a browser front end that anyone can log into and trade on.

```
41,000 lines of exchange, agents and research     1,473 tests across 45 files
30,000 lines of tests                             358 markets, 318 of them binaries
9 asset classes on one matching engine            conservation: integer zero
bit-identical replay from a seed                  collateral: exact, not modelled
```

**No real money, no real securities, and no real sport.** The world is generated here: the competitors, the formats and every statistic written on them are this project's own invention, drawn from a seed rather than collected from anywhere. Nothing on this exchange refers to a real game, league, team or player, nobody has endorsed it, every participant is simulated, and no position here connects to a venue where money is real.

---

## What the contracts are written on

The world lives in [worlds/circuit/](python/arena/worlds/circuit/). Twelve competitors of our own invention, six archetypes between them, and two formats: a ten-entrant last-one-standing mode, and a three-a-side mode where two sides race to a target score. A match is a pure function of the season seed and its own number, so the same seed always produces the same season and any settlement can be re-derived by anyone holding the seed.

Latent skill is drawn once per season and **never published**. The oracle reports what matches did, not what the roster is, and the module that computes metrics cannot reach the module that draws strengths. Without that separation a market would be pricing a lookup, and every forecasting result measured here would be a result about arithmetic.

Two families of contract come out of that world, and they are genuinely different instruments rather than two views of one number.

**Live match markets.** A match opens, runs for minutes, and settles in play. A ten-entrant solo match lists **270 contracts** in four families, winner, placement, eliminations and head to head, and yields **395 exact relations** between them; a three-a-side objective match lists **38** and yields **58**. The live part is not decoration, it is settlement arriving in pieces: a competitor who has been eliminated cannot win, so that contract is worth zero and is settled the moment it happens while everyone still in keeps trading.

**Statistical contracts.** Six metrics over an observation window: win rate, eliminations per match, match volume, placement dispersion, format dispersion and score margin. Each names a format, because a win rate sits at a neutral 0.100 in the ten-way elimination and 0.500 in the three-a-side, so a pooled average would move with whatever the schedule ran rather than with the competitor. The oracle refuses the pooled version rather than returning a number that reads as fine.

Why two families rather than a longer list of contracts on one number: **within a single format every skill statistic ranks the field almost identically, Spearman +1.000.** That is a fact about the world, not a defect in the metrics, and it means depth cannot come from writing more contracts on one statistic. It comes from the formats, which rank the field at **Spearman -0.357**, and from matches, which resolve in minutes rather than in a four-week window.

---

## Highlights

- **Match markets are arbitrage free by construction, not by policing.** Every contract on a match is priced as the mean of *the same settlement rule that will settle it* over one finite ensemble of drawn outcomes, so any identity that holds outcome by outcome holds in the prices as arithmetic. The winner set sums to one because each drawn match has exactly one winner, not because anything was normalised afterwards. Measured in exact rational arithmetic: **zero violations across 22,650 relation checks**. Pricing the same contracts off eight ensembles drawn from the same belief, which is what a book of independently quoted contracts amounts to, breaches **53 of the 395 relations by up to 0.1163**. See [market/match_book.py](python/arena/market/match_book.py).
- **The world conserves too.** Eliminations sum to exactly nine in every one of **4,000 solo matches**, which is the match's version of what the ledger holds itself to. A match whose eliminations do not add up is refused rather than settled.
- **Exact collateral.** Every instrument settles as a known function of one bounded scalar, so a portfolio's worst case is the minimum of a piecewise-linear function of a single bounded variable. That minimum sits at an endpoint or a kink, every kink is known in advance, and there are a handful of them, so the engine evaluates all of them and takes the smallest. Measured on real packages: **100% of collateral released on a conversion, 99% on a vertical spread, 100% on a share against its weekly strip.** See [netting.py](python/arena/portfolio/netting.py).
- **Integer money.** All cash is held in integer minor units at a scale of 1,000,000. `conservation_check()` returns integer zero after every trade, fee, auction and settlement, in every test and every experiment trial.
- **The market ties the simple mean of its agents and loses to precision weighting.** 200 paired trials scored against a *known* true probability, which no field study can do. Squared error to truth: market **0.03108**, simple mean 0.02957 (p = 0.59, indistinguishable), precision-weighted 0.01436 (**market 2.2x worse**, p < 0.0001), best agent chosen with hindsight 0.00272. See [Findings](#findings).
- **Experiment 2 rules out the matching rule.** The natural reading of Experiment 1 is that a limit order book aggregates by capital, since it needs a counterparty for every trade. So Experiment 2 built an LMSR venue, which always quotes and needs no counterparty. It lands in the same place, p = 0.97, which leaves the agents' balance sheets as the binding constraint.
- **Concentrating information makes the price worse.** Holding total evidence fixed at 10,319 matches and moving all of it into a single trader raises market error **2.7x**, because one agent has one balance sheet. The usual intuition about an informed monopolist runs the other way.
- **A programmatic API.** 20 REST routes plus a streaming socket, covering market data, candles, accounts, positions, fills and the whole order lifecycle across every asset class through one code path. The stream resumes after a disconnect and replays what you missed instead of restarting its sequence, so an algorithm that drops a connection keeps its state. HMAC-SHA256 signing covers the timestamp, method, path and body together, so a captured signature cannot be moved onto a different order. An API order is enqueued onto the same agent, crosses the same latency link and meets the same collateral check as one clicked in the browser. See [docs/API.md](docs/API.md).
- **A strategy testbed.** Write a market maker or a buy-side strategy against one interface, run it inside the live market, and get back a number you are entitled to believe. A strategy reads a view and returns intents, and never touches the venue, so it cannot reach the settlement level, another participant's position, or a book it was never told about. The book it does read is honestly stale by its own latency, which is the phenomenon rather than an inconvenience. Six reference strategies ship against it: a fixed-spread control, Glosten-Milgrom, Avellaneda-Stoikov, Gueant-Lehalle-Fernandez-Tapia, a Kelly-sized Bayesian and a static arbitrageur, plus a firm that budgets collateral per netting group. Comparisons run as paired trials on common random numbers, and the harness reports whether pairing actually helped instead of assuming it: measured, it removed **93% of the variance** in one configuration and **made things 3% worse** in another. See [docs/STRATEGIES.md](docs/STRATEGIES.md).
- **Nine asset classes on one matching engine.** Futures, binaries, calls, puts, calendar spreads, an index, commodities, equities and a volatility contract. They share one collateral rule because they all reduce to a bounded payoff function.

---

## Findings

### Experiment 1: the market against its own agents

Eight informed agents see log-spaced match counts from 50 to 5,000, form Beta posteriors, and forecast the Beta-Binomial tail, which is exactly Bayes-optimal for what each has seen. They trade on the real matching engine against a market maker and eight noise traders, with heterogeneous latency. Paired by trial, Benjamini-Hochberg corrected across the four comparisons.

| Baseline | Baseline error | Market minus baseline | 95% CI | p (adj) | Result |
|---|---|---|---|---|---|
| Best single agent (hindsight) | 0.00272 | **+0.02836** | [+0.023, +0.034] | <0.0001 | baseline wins |
| Simple mean | 0.02957 | +0.00151 | [-0.002, +0.006] | 0.59 | **indistinguishable** |
| Precision-weighted | 0.01436 | **+0.01672** | [+0.012, +0.022] | <0.0001 | baseline wins |
| Extremized log-odds | 0.02980 | +0.00128 | [-0.003, +0.006] | 0.59 | **indistinguishable** |

Three ablations rule out the easy explanations. Removing noise traders entirely changes nothing, 0.02818, with every comparison unchanged. Raising position limits five-fold changes nothing, 0.03152. Error by session length is flat from 300 seconds onward, so the market has converged and is not short of time.

The mechanism that survives: every agent carries the same base size, the same position limit and the same cash, so the 50-match agent pushes the price about as hard as the 5,000-match agent. **The market has no channel through which evidence quality reaches the price.** The precision-weighted baseline has exactly that channel by construction, and wins with it.

The market is healthy while this happens, which is what makes it a result about markets rather than about a broken simulation: 2,360 trades per trial, two-sided 99.9% of the time, reliability 0.0098, conservation exact in every trial.

### Experiment 2: is it the mechanism?

An LMSR venue carries none of the constraints an order book imposes. It always quotes, it needs no counterparty, and it has no market maker anchoring price to its own reference. It reaches the same answer, p = 0.97, which isolates the balance sheet rather than the matching rule as the binding constraint. Full write-up in [experiments/information_aggregation/](experiments/information_aggregation/README.md).

---

## How it is verified

Correctness in a simulated market is checked mechanically. This is where most of the work went, and it is what makes the numbers above worth reading.

- **Determinism.** The same seed and the same manifest produce a bit-identical result digest, down to the byte. A paired ablation is then a comparison between two runs that differ in exactly one thing.
- **Conservation.** `conservation_check()` runs after every trade, fee, auction and settlement, in every test and every trial, and returns integer zero. Cash is in integer minor units, so there is no tolerance to tune.
- **The world's own conservation.** A match result is checked against its format before it can be settled against. Eliminations must sum to one fewer than the field, placements must be a permutation, and the winning side of a team match must be a side that actually played rather than merely the right number of people.
- **Differential testing across instruments.** A whole class of pricing and quoting defects is invisible on any single instrument and appears only when two that must agree by arithmetic are compared: a call chain that is not monotone in strike, a share that disagrees with its own weekly strip, an iceberg advertising depth it has already sold. The harness compares them and fails when they disagree.
- **The settlement boundary.** Settlement closes cash and positions both out to zero, and the test asserts that identity exactly.
- **The front end.** Two browsers are two traders, and the dashboard drives the same order path the agents use. Several unit-mismatch bugs were caught this way, after passing the unit tests.
- **Pre-committed hypotheses.** The information-aggregation result was written into the plan before the run, so the reported comparison is the specified one.

Market data is conflated instead of replayed event by event, which keeps a suite that simulates full trading sessions fast enough to run on every change.

---

## Architecture

Python throughout: the exchange, the agents, the research harness and the front end. The Python exchange is the reference implementation and the specification any future port has to match, and the kernel is written in integer time and integer money so those semantics survive being ported.

There is no C++ in the repository, and the plan that once listed a C++ kernel as half the stack was making a promise the profile does not support. Profiled over two simulated minutes of the live market, the time sits in `kernel.send`, `latency.delay` and the book snapshot the venue broadcasts, which is message plumbing; matching does not appear in the fourteen most expensive functions by self time at all. So a port is an evidence-led performance decision to be taken when a profile asks for one, not an unmet commitment. `tests/test_differential.py` already exists as its acceptance test either way, hardened over roughly 1.2 million fuzzed commands.

| Layer | Module | What it holds |
|---|---|---|
| Time | [sim/](python/arena/sim/) | Discrete-event kernel in integer nanoseconds, per-agent latency, FIFO per ordered pair. Same seed, same bytes. |
| Matching | [exchange/](python/arena/exchange/) | Price-time book. Limit, market, IOC, FOK, stop, stop-limit, iceberg, pegged, minimum-quantity, post-only. Stop cascades are iterative and depth-bounded. |
| Contracts | [contracts/](python/arena/contracts/) | Payoff algebra over a bounded underlying: linear, call, put, binary. The bounded range is what makes collateral exact. |
| Venue | [market/](python/arena/market/) | Listing rules, tiered tick tables, maker-taker fees, opening and closing auctions, LULD circuit breakers, a participant kill switch, message throttles. |
| Matches | [market/match_book.py](python/arena/market/match_book.py) | Every contract on one match, listed from the format, priced off one ensemble so its 395 relations cannot be broken. Opened, revealed and settled in play by `match_operator.py`. |
| Clearing | [portfolio/](python/arena/portfolio/) | Integer minor units, per-account cash and positions, exact portfolio netting evaluated at kinks. |
| Agents | [agents/](python/arena/agents/) | Market makers, an option surface maker, a match maker, Bayesian informed traders, noise, flow, and an arbitrageur that enforces parity and index identities. |
| Settlement | [settlement/](python/arena/settlement/) | The oracle boundary, deterministic settlement, and the proof that cash and positions both close out to zero. |
| World | [worlds/circuit/](python/arena/worlds/circuit/) | The synthetic esport: roster, formats, match simulation, six metrics and the calendar that turns a date range into a run of matches. Every metric is bounded by construction. |
| Research | [research/](python/arena/research/) | Trial harness, manifests carrying seeds and code digests, aggregation baselines, stylized-fact diagnostics. |
| Front end | [dashboard/](dashboard/) | The exchange as a website. |

---

## The exchange as a website

`python -m dashboard.server` serves the venue to a browser: per-user accounts with signed session cookies, order entry across all nine asset classes, live depth and tape, positions and P&L, settlement, a research tab, and a simulation-speed control. Two browsers are two traders on the same book. The conservation indicator sits in the header, so the invariant is visible while you trade against it.

Matches change what browsing has to do. The board presents **355 markets as 9 rows**, of which **316 are prediction markets against 8 before matches existed**, because a match is one row rather than 270. Listing them one card per instrument would render a page nobody could read.

They also cost time. The market runs at **1.85x real time without matches and 0.75x with them**, which is the price of pricing a 270-contract book off an ensemble. Uninformed flow is the other dial: it was 0.5% of volume and is now 3.0%, while a realistic 16% costs ten times real time, so the composition is set where the market still moves faster than the wall clock.

Vanilla ES modules. No framework, no bundler, no build step.

---

## Running it

```bash
pip install -e .
```

```bash
python -m pytest -q
```

The exchange as a website, and as an API on the same process:

```bash
python -m dashboard.server
```

Trading it from a program, the way you would trade Kalshi or Alpaca:

```python
from arena_client import ArenaClient

arena = ArenaClient("http://localhost:8000", key_id=KEY, secret=SECRET)
book = arena.book("VANTA_OBJECTIVE_WR", depth=5)
arena.place_order(symbol="VANTA_OBJECTIVE_WR", side="buy", quantity=10,
                  price=book["bids"][0]["price"])
```

The experiment, which writes a manifest with seeds and code digests so a run reproduces byte for byte:

```bash
python experiments/information_aggregation/run.py --quick
```

---

## Documentation

- [AGENTS.md](AGENTS.md), for an autonomous agent trading here over the API: what this venue guarantees that a real one does not, what will refuse your orders and why, and the things that have actually broken clients
- [docs/STRATEGIES.md](docs/STRATEGIES.md), to write a market-making or buy-side strategy that runs inside the simulation, with a backtest harness that reports whether its own paired trials helped
- [docs/PLAN.md](docs/PLAN.md), the phase plan and what each phase had to prove before it closed
- [docs/API.md](docs/API.md), the programmatic interface, with worked signing vectors so a client in another language can check itself
- [docs/ECONOMY.md](docs/ECONOMY.md), the world the contracts are written on, where value comes from and why every contract is bounded
- [docs/GAPS.md](docs/GAPS.md), this exchange measured against real venue mechanics
- [experiments/information_aggregation/](experiments/information_aggregation/README.md), the full experimental write-up
- [CONTRIBUTING.md](CONTRIBUTING.md), for changing this repository rather than trading on it, which is a different job with different traps
