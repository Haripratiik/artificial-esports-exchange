# Working in this repository

For **changing this codebase**. If you are an agent that wants to *trade* on
the exchange over its API, `AGENTS.md` is the file you want, and `docs/API.md`
is the reference behind it.

This is not a summary of the docs. `README.md` says what the exchange is,
`docs/API.md` documents the interface, `docs/GAPS.md` audits it against a real
venue. Read those for facts about the system.

This file is about working *here*: the traps that have bitten more than once,
the things that look like improvements and are not, and how to find out whether
you are right. It exists because most of what went wrong in this project was
not hard to fix once seen, and was invisible until something forced it into
view.

---

## The one habit that matters

**Measure. Do not reason about it.**

Almost every defect in this repository's history was found by running code and
reading a number, and almost every one of them had survived a careful reading
first. A representative sample, all real:

- A mark price that reported the price *falling* when a buy swept every offer.
- A settlement value rendered as 18,677 against a real 4,663, found by opening
  the page, not by any of the 700 tests then passing.
- A tick flash that was lit on 32.5% of rows continuously and disagreed with
  its own row's direction 45.1% of the time.
- An option chain where two calls both worth nothing marked at 1.63 and 68.38,
  purely because one book carried more inventory.

None of those are subtle in hindsight. All of them read as fine.

So: when you have a hypothesis, write a throwaway script that prints the
number, and put the number in the commit message and the code comment. The
comment style in this codebase is not decoration. Nearly every non-obvious
line names the concrete measurement that motivated it, so the next person can
tell a deliberate choice from an accident.

**Corollary: be willing to be wrong out loud.** Several times in this project a
confident diagnosis was wrong and the measurement said so. "I assumed 47
instruments across three makers had thinned the books; measured, 94% two-sided,
so that was false" is a better contribution than a plausible fix.

---

## Three invariants. Do not negotiate with these.

**1. Conservation is exactly integer zero.**
`venue.conservation_check()` returns an `int`, and it returns `0`. Not "close
to", not "within tolerance". Money is integer minor units at a scale of
1,000,000 for exactly this reason. If you find yourself adding a tolerance, you
have introduced a bug somewhere else and are hiding it.

The world holds itself to the same rule, and for the same reason. A
ten-entrant elimination credits exactly nine eliminations, placements are a
permutation, and the winning side of a team match has to be a side that
actually played rather than merely the right number of people. `fmt.check` is
asked on every drawn match, not in a test, and a match that fails it raises
rather than settling. Measured: eliminations sum to exactly nine in every one
of 4,000 solo matches. A contract settled against a match whose arithmetic does
not close is paying out on something that did not happen, which is the ledger's
own failure wearing different clothes.

**2. Collateral is arithmetic, never a model.**
Every instrument settles as a known function of one bounded scalar, so a
portfolio's worst case is the minimum of a piecewise-linear function evaluated
at its endpoints and kinks. That is not an approximation of the answer, it is
the answer. No VaR, no correlations, no volatility estimates in a collateral
path. Positions on different underlyings **do not net**, because netting them
would require a correlation and a correlation is an estimate. This is the line
between arithmetic and modelling and it is the project's whole claim.

**3. No hardcoding.**
Nothing may nudge a price, a spread or a distribution toward what a real market
would show. Every observed regularity has to fall out of the mechanism or not
appear at all. Concretely: never special-case a symbol, a strike, a seed or an
agent to make a test pass. A market tuned to look real has stopped being
evidence about anything, which defeats the point of the whole exercise.

A softer fourth: **no floats in a money or price path.** Prices cross the wire
as strings and parse to `Decimal`. A client that parses a price into a float
silently reintroduces the error the venue spent its life avoiding.

And a fifth, which is newer and easier to break by accident: **every name in
this repository is our own.** The world used to be a real game's published
statistics under that game's character names, and it is now a synthetic esport
in `worlds/circuit/`: twelve competitors we invented, two formats we defined,
six metrics measured on matches this repository generates. Formats and
mechanics are ideas and are free to model. Names are trademarks and are not, so
do not name a competitor, a mode, a metric, a ticker, a fixture or a test after
anything real, and do not swap one borrowed name for another. If you need a new
competitor, add one to `roster.py`.

---

## Six bug classes that have each bitten more than once

If you are about to touch one of these areas, read the relevant entry first.
Each of these recurred *after* being fixed once, which is why they are here
rather than in a changelog.

### 1. Units crossing a boundary in the wrong unit

**Five instances.** A settlement value stored in ticks and rendered as a price
(18,677 against a real 4,663). Account equity published in minor units and
rendered by the money formatter (`"143745.00M"` next to a header reading
`143.7k`). Halt records published in ticks under columns headed "Price".
`fees_collected` published in minor units with the browser silently correcting
it with `/ 1e6`. And the one that was not a display bug at all:
`build_market.prior_levels` re-dated every contract onto a four-week prior
window without rescaling a **quantity**, so `RIFT_SOLO_VOL_W1` observes one week,
settles at 71.09, and handed all six informed traders a prior of 274.92. The
ratio was 3.87x against a window ratio of 3.87. The four win-rate futures came
back at 1.00 to 1.04x on the same run, which is what said the error was the
unit and not the fixture, because a rate is scale-invariant in window length
and a count is not.

That last one is worth dwelling on, because the type system had already said
so. `MetricRef.kind` distinguishes `rate` from `quantity` and its comment reads
"the amount delivered in March is a different thing from the amount delivered
in April". The distinction was declared upstream and ignored downstream, and
the market wore it: at t=180s the contract traded 171.93 against a fair 71.10,
so any strategy measured on that book was harvesting the artifact.

The pattern is always the same: a number crosses a boundary in a unit its label
does not claim, **one consumer compensates**, and every other consumer is
wrong. It stopped being cosmetic the moment this exchange grew an API, because
a client calling `GET /v1/exchange` has no `/ 1e6` to apply.

There is now a general guard: `test_no_client_compensates_for_minor_units`
fails if any browser code divides a payload field by the money scale. Note it
requires a *property access*, since dividing by a million is legitimate as a
magnitude abbreviation, and the first version of that test flagged the
formatters and was wrong.

**Rule: convert at the serialiser, never at the consumer.**

### 2. An identity captured once, then reused after it stopped meaning anything

**Four instances, and this one is genuinely dangerous** because the failure is
silent and looks like normal operation.

`runner.reconfigure()` discards the entire market and every account in it. And
`LiveMarket.trader()` answers an **unknown** agent id with the *shared*
account. So anything that captures an `agent_id` and reuses it after a rebuild
does not fail. It quietly starts trading a communal seat.

It has happened to: browser sessions (every signed-in visitor collapsed onto
one account), a returning visitor matching a *newcomer's* freshly issued
`you-1`, API keys, and the streaming socket's private channels.

**Rule: bind to a stable seat *token* (a name, a session id), never to an
account id, and re-resolve through `runner.market.seat(name)` on every request
or every tick.** `dashboard/server.py::_seat_now` is the reference
implementation. If you add a new surface that trades on someone's behalf, write
the rebuild test *first*: issue a credential, rebuild the market, assert the
credential still reaches its own separate account and not the shared one.

### 3. Two things that are correct about different questions, sharing one surface

The tick flash followed the *last tick* while the row's percentage followed the
*whole session*. Both were right. Rendered on the same row they disagreed 45%
of the time, and the stylesheet's own first rule is that green and red always
mean the same thing.

Same shape: a percentage computed from a base that has no resolution
(`+308,825%` on an option that opened at one tick), and a percentage from a
*negative* base, which inverts the sign, so a spread falling from -5 to -10
reported `+100%`.

**Rule: when two true things want the same pixel or the same field, one of them
has to move.**

### 4. A guard that is disabled by another guard

The kill switch was rate-limited, and a participant worth killing is at its
message cap by definition. It also reported success while doing nothing. A
rate-limited participant could not cancel its own orders, which traps it in
exposure nobody is allowed to manage. `Replace` bypassed the tick-grid check
because the guard read `price` and `Replace` carries `new_price`.

**Rule: for every control, ask what state the thing you are controlling is in
when you need the control. Then test it in that state, not in a calm one.**

### 5. Dropping instead of delaying

Conflation that dropped updates rather than holding and flushing them
deadlocked the market: a trial that traded 2,039 times traded 0. The streaming
API has the same shape of trap, and its resume path has a subtler one: restoring
subscriptions while reopening the tape cursor at *now* drops every event from
the disconnected window **silently**, because the sequence runs gapless
straight across the hole.

**Rule: never silently discard something a consumer has not seen. Hold it,
flush it, or tell them explicitly that they lost it.**

### 6. A check that never fires because its input was never wired

The venue has always had the right rule: `_enforce_lifecycle` closes a symbol
once `self._clock() >= instrument.expiry`, so nobody trades against an outcome
that is already determined. It was correct, it was tested, and in the live
market it never ran once, because `_clock` was `None` and the question was
therefore never asked.

Measured: after a simulated hour, all 47 listed contracts were still
`continuous` and the settled set was empty. Positions were marked forever and
realised never. The settlement machinery was complete the whole time;
`build_market.prior_levels` had been calling `settle(spec, oracle)`
successfully on every listing since long before anyone noticed the live path
never did.

The root cause is worth naming because it generalises: **there were two clocks
and nothing connected them.** The kernel counts simulated nanoseconds from
zero; a contract expires on a calendar date. Neither is wrong and they could
never meet.

The same shape, arriving by a different door, in `StrategyAgent`: a `Filled`
does not carry its symbol. The venue runs one private channel per book and the
symbol arrives with the envelope, so a handler written as
`getattr(event, "symbol", None)` reads `None` on every fill, returns early, and
books nothing. Measured before it was found: 386 fills moved the agent's ledger
by exactly zero, no exception, no warning, and the positions still looked right
because the base class tracked those separately. `getattr` with a default is
the same hazard as an unwired clock, because both turn a missing input into a
silent no-op rather than a failure.

**Rule: a guard whose input can be `None` is a guard that can silently not
exist, and `getattr(x, "field", None)` manufactures exactly that. When you find
one, check what supplies it in production, not in the test that covers it.**
The test suite had settlement covered thoroughly, and every one of those tests
supplied its own clock.

---

## How to verify things here

### Running tests

```bash
python -m pytest -q -p no:warnings
```

1,550 tests across 49 files. **The full suite takes roughly 20 to 40 minutes**,
because many tests run real simulated markets. Run the file you touched first;
run the whole thing before you commit, in the background, and wait for it.

Some tests carry **measured numbers that depend on market composition**:
`test_netting.py`, `test_surface.py`, `test_stylized.py`, the experiment
ablations. Listing a new contract legitimately moves those numbers. When one
fails after such a change, re-measure and update the recorded number *with the
new measurement in the docstring*; do not loosen the assertion to make it pass.

A test that fails only in the full suite and passes alone is almost always
**cross-file state**, not a real defect. The known instance: `test_api.py`
sorts before `test_dashboard.py` and permanently replaces `rest.configure`'s
module globals with its own seat hook.

### Seeing the interface

The Browser pane cannot screenshot in this environment. It only composites
while displayed, so `visibilityState` is `hidden`, screenshots time out, **and
`requestAnimationFrame` never fires**, which makes the app render nothing at
all. That is not a bug in the app. I lost real time to it.

Use Playwright with the system Chrome instead, then read the PNG:

```bash
python -c "
from playwright.sync_api import sync_playwright
with sync_playwright() as pw:
    b = pw.chromium.launch(channel='chrome', headless=True)
    page = b.new_page(viewport={'width':1440,'height':900}, device_scale_factor=2)
    page.goto('http://localhost:8000', wait_until='networkidle')
    page.wait_for_timeout(5000)
    page.screenshot(path='shot.png')
    b.close()"
```

Layout still computes while hidden, so `getBoundingClientRect` and computed
styles are trustworthy even when pixels are not.

### Environment traps

- **The Bash heredoc mangles quotes and escapes.** `cat <<'EOF'` has repeatedly
  produced broken files here: a `\n` inside a JS string became a real newline
  and broke the file. Use the Write and Edit tools for anything with quoting.
- The server holds one market in one process. Restarting it discards
  everything; a server already running will not pick up your changes to
  `build_market.py`.
- `dashboard/server.py` defines `async def stream(...)` for the browser socket,
  which shadows a bare `from arena.api import stream`. Import it as
  `api_stream`.

---

## Things that look like improvements and are not

Each of these was tried and measured. The number is why it was reverted.

| Change | What happened |
| --- | --- |
| Collar limit orders like market orders | 2,492 limit states in five minutes; a future quoting 9,267 against a settlement of 4,669. A collar protects an order that did not name a price; a limit order named one. |
| More market makers for deeper books | Five makers gave *worse* two-sided coverage than three, 85% against 94%. |
| Clamp the centre of a quote into the settlement range | Makes the mid a function of the half-spread, which widens with inventory, so two worthless calls priced 40 points apart. Clamp each side independently. |
| Drop updates when a consumer is slow | Deadlocked the market: 2,039 trades became 0. |
| Cancel and repost a quote whose price has not moved | This is what a real maker never does. Removing it took 1.6M events/simulated minute to 317K and the suite from 38.8s to 11.0s. |

---

## What "done" means here

A change is finished when:

1. The suite is green, and you ran it, not just the file you touched.
2. `conservation_check()` is exactly `0`.
3. Any new behaviour has a test whose docstring names the measurement that
   motivated it, so the next person can tell why the number is what it is.
4. Nothing was hardcoded to make a test pass.
5. If it is visible in the browser, you *looked at it*. The unit bug that read
   18,677 against 4,663 was found by opening the page while 700 tests passed.

Commit messages here are prose, not bullet lists, and they lead with the
finding rather than the change. `git log` is a genuinely useful record of what
was wrong and why. Keep it that way.

---

## Two gaps that were named for years, and what they are now

Both were listed as the project's outstanding debts. Neither survives contact
with a measurement, and the entries are kept here rather than deleted because
the correction is the useful part.

- **There is no C++ kernel, and none is owed.** The plan listed Python for
  research and C++ for the exchange kernel, so half the stack read as missing.
  Profiled over two simulated minutes of the live market, the time is in
  `kernel.send`, `latency.delay` and the book snapshot the venue broadcasts, and
  matching does not appear in the fourteen most expensive functions by self
  time. Porting the matcher would move a small share of a total that is spent
  somewhere else. So a port is a performance decision to take when a profile
  asks for one, not a promise outstanding. `tests/test_differential.py` remains
  its acceptance test, hardened over roughly 1.2 million fuzzed commands, and it
  is worth keeping green whether or not the port ever happens.
- **There is no data to collect.** This used to read "no real data has been
  collected", with an empty `data/raw` and a crawler waiting on an API key. The
  world is now generated: `worlds/circuit/` draws a season from a seed, and a
  settlement is re-derivable by anyone holding that seed rather than by anyone
  holding a crawl. That closes the gap by removing it, and it also removes the
  ways the old answer could go wrong, since a statistic built from a crawl moves
  when the crawler's reach moves.

The work this frees up is the listing. Matches relist continuously, so the
market no longer drains as contracts expire, and the interesting questions are
now about depth and composition rather than about supply.
