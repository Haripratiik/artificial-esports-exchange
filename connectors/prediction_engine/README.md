# This venue, for a prediction-market engine

A connector that presents this exchange in the shape a Kalshi-shaped trading
engine already speaks, plus the one piece of research that shape unlocks.

It exists because two codebases arrived at the same architecture
independently. A prediction-market engine built against Kalshi keeps its
strategies pure and its venue behind three files:

> `C4.2b` Sleeves never call a VenueClient. Only the executor does.
> `MarketSnapshot` is everything a sleeve is allowed to see. Time is IN here,
> never read live.

This exchange, written separately, says:

> **A strategy sees a view and returns intents. It never touches the venue.**
> `MarketView` is the whole of what a strategy is allowed to see, at one
> instant.

Both put time inside the snapshot, both have the strategy declare desired
state, both diff that against reality in an executor. Where two designs agree
without consulting each other, the seam between them is small.

## What is here

| file | what it is |
|---|---|
| `models.py` | `Series` / `Event` / `Market`, the engine's shapes, filled in from this venue |
| `auth.py` | request signing, the same interface as the Kalshi signer over a different primitive |
| `client.py` | the venue over HTTP, method for method against the Kalshi client |
| `jointarb.py` | the engine's joint no-arbitrage LP, generalised off its score grid |
| `audit.py` | that LP pointed at the live match board |

Nothing here imports the exchange. Every call is HTTP against the published
API exactly as an outside client makes it, which is the property that makes
the audit worth running at all: a test that shared the venue's own model of a
coherent price could not falsify that model.

## The two venues, where they genuinely differ

**Signing.** Kalshi signs with RSA-PSS and excludes the query string; this
venue signs with HMAC-SHA256 and includes the query string and the body. The
field order is `timestamp, method, path, body` and it is not guessable: written
the obvious way round, method first, the signer produces signatures that verify
against nothing and a client that reads every public endpoint and authenticates
on none of them. `test_connector.py` pins it against the venue's own
`canonical_request` rather than against a reading of it.

**Money.** The engine holds prices as integer cents because a Kalshi contract
is one. This venue holds integer minor units at a scale of 1,000,000 over five
settlement ranges. The conversion is exact for the class that matters, since an
event contract here is bounded [0, 1] on a 0.01 tick, so 0.47 is 47 cents with
nothing lost. Every other class raises `PriceOutOfModel` rather than being
approximated: a future bounded [0, 10,000] is not a probability, and rounding
one into cents puts a number downstream that is the right type and a lie.

**Scope.** 318 of this venue's 358 markets are binaries and fit the engine's
model exactly. The other 40 are futures, options, spreads, an index,
commodities and a volatility class, and they are dropped rather than mangled.
The engine is a prediction-market engine; this venue is mostly, not entirely, a
prediction-market venue.

**Exhaustiveness.** Kalshi flags a mutually exclusive event and the engine
verifies it. Here the venue publishes the partition itself through
`ExclusiveSet`, so the flag is a fact to read rather than a claim to check.

## The audit, and why it is not redundant

This exchange prices every contract on a match as an expectation under one
distribution. Its *model* prices therefore cannot disagree with each other, and
that guarantee is real. It is not the one the LP tests.

Quotes are not model prices. A maker widens for inventory, skews for adverse
selection, and reprices off a book it last saw 160ms ago. Whether the prices
actually standing on the screen admit *any* consistent distribution is a
different question, and nothing in this repository asked it before. If the
answer is no, a portfolio exists with non-negative payoff in every state and
negative cost, which is riskless and needs no forecast.

Three boards are tested per match, each with a state space small enough to
enumerate exactly: the winner partition, the eliminations ladder per
competitor, and the top-N ladder per competitor. One LP over every family at
once is deliberately not attempted, because the joint state of a ten-entrant
match is a finishing order and there are 3,628,800 of them. Three exact answers
beat one sampled answer that cannot tell a violation from its own sampling.

## What it found

**The model is coherent and the quotes are not.** Asked of `coherent_prices()`
directly, before any maker has touched the result: **0 of 16** elimination
ladders are non-monotone, 10 on a solo match and 6 on a team match, at 4,000
draws on seed 7. The guarantee this venue makes is exactly true.

Asked of the live book, it fails. On `OBJECTIVE3`, CINDER's ladder stood at

    P(X > 0)   0.53 / 0.62
    P(X > 1)   0.40 / 0.49
    P(X > 2)   0.60 / 0.70

and `X > 2` implies `X > 1`, so the third line cannot exceed the second. Buying
GT1 at 0.49 and selling GT2 at 0.60 collects **0.11** up front for a payoff that
is non-negative in every state: the legs cancel if X > 2, the long pays if
X is exactly 2, and both expire worthless below. Four distinct ladders showed it
across eight snapshots, worst slack 0.065.

**The arbitrageur halves it and does not clear it.** Paired on seed 17, matches
listed, sampled at four moments to 300 simulated seconds, 48 tradeable ladders
each way:

| | ladders admitting no distribution | worst slack |
|---|---|---|
| arbitrageur off | 31 of 48 (64.6%) | 0.0550 |
| arbitrageur on | 14 of 48 (29.2%) | 0.0350 |

So the agent whose job this is removes most of them and cannot keep up with its
own market. That is a finding about liquidity rather than about pricing, and it
is the kind this repository could not previously see, because every check it
had shared the pricing model's assumptions.

## Read the filter before you read a result

Only books that are matching *right now* carry a price anybody could trade on,
and the first run of this audit ignored that. It reported 18 of 27 boards
admitting no consistent distribution, with a winner partition 0.3533 outside
its spread, which is far too large to be real and was not.

This venue opens every contract at the midpoint of its own settlement range
rather than at an estimate, on purpose, so that convergence has to be produced
by agents trading against the maker. Ten seconds after listing, all ten legs of
a winner set are quoted near 0.50 and sum to 4.54 against a partition that must
sum to 1. That is not an arbitrage, it is a market that has not opened yet
being measured as though it had. Contracts on a settled match have the matching
failure: a closed book keeps its final touch forever.

Both are now filtered, and the lesson is the engine's own: a violation too
large to be real is a bug in the measurement, and it is worth checking before
it is worth reporting.

## Running it

```bash
python -m dashboard.server
python -m connectors.prediction_engine.audit
```

`--tol` widens every spread by that many probability units before testing, so a
one-tick numerical wobble is not reported as an opportunity. Exit status is
zero when every board admits a consistent distribution.
