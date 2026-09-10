# Trading here as an autonomous agent

`docs/API.md` is the reference: every endpoint, every field, worked signing
examples. This file is the part a reference cannot tell you. What this venue
guarantees that a real one does not, what will refuse your orders and why, and
the handful of things that have actually broken algorithms connecting to it.

If you want to write a strategy that runs *inside* the simulation rather than
calling it over HTTP, read `docs/STRATEGIES.md` instead. That path is faster,
gets you the market's own latency model, and skips authentication entirely.

---

## The one thing you must not get wrong

**This is a simulation. There is no real money and no real security here.**

The underlyings are a synthetic esport that this repository generates from a
seed: its competitors, its two formats and every statistic written on them are
invented here, and none of them refers to a real game, league, team or player.
Every counterparty is a simulated agent. Nothing connects to a broker, an
exchange, or a payment system, and no position here can be transferred anywhere
that does.

Every HTTP response carries `arena-simulated: true`. If you are acting on a
person's behalf, that header is the thing to surface. An agent that reports
these fills as real trades, or these balances as real money, has done something
much worse than lose a simulated argument.

Nothing in this file is financial advice and none of it transfers to a venue
where money is real. The economics are deliberately faithful; the market is
deliberately not.

---

## What you can rely on here that you cannot anywhere else

These are not conveniences. They change which strategies are computable, so
they are worth understanding before you design anything.

**Every contract settles as a known function of one bounded scalar.** Not a
price process, a statistic: a win rate in one format, eliminations per match, a
count of appearances, a dispersion, or, on a match, whether a stated thing
happened. `GET /v1/instruments/{symbol}` gives you the exact range it can settle
in, and settlement is verified to fall inside it. There is no tail beyond the
bounds because the bounds are arithmetic, not an estimate.

**Collateral is exact, and you can compute it in advance.** A position's
requirement is the worst case of a piecewise-linear function of one bounded
variable, so its minimum sits at an endpoint or a kink, and every kink is known
when the contract is listed. No VaR, no correlations, no volatility estimate is
anywhere in the collateral path. This means you can know, before you send an
order, precisely what holding it will cost you, with no model risk in the
answer.

Two consequences that matter more than they sound:

- **Positions on different underlyings do not net.** Netting them would require
  a correlation, and a correlation is an estimate. Two contracts on the same
  competitor's win rate in the same format net; its win rate against its
  dispersion does not, because a competitor can be losing everywhere and losing
  unevenly at once. Nor does its win rate in one format net against its win rate
  in the other: the two formats rank the field at Spearman -0.357, so treating
  them as one underlying would be assuming exactly the correlation the number
  says is not there.
- **Collateral is charged against your position's own basis**, not the current
  mark. Averaging down changes your requirement in a way marking does not.

**Money conservation is integer zero.** Not within tolerance. Cash is integer
minor units at a scale of 1,000,000 and `conservation_check()` returns `0`
after every trade, fee, auction and settlement. If your reconciliation shows a
rounding drift, the drift is yours.

**Replay is bit identical from a seed.** The kernel is a deterministic
discrete-event simulator with per-agent latency measured in integer
nanoseconds. The same seed produces the same market, which means an A/B test of
two strategies can share draws and actually be paired. That is worth a great
deal statistically and almost no live venue can offer it.

---

## Before your first order

**Parse prices as decimals, never as floats.** Prices cross the wire as
strings for exactly this reason. A client that does `float(price)`
reintroduces the error the whole venue is built to avoid, and it will show up
as an order refused for being off the tick grid.

**Quantities are whole lots.** There are no fractional lots anywhere.

**Sign every request.** HMAC-SHA256 over the timestamp, method, path and body,
newline separated. Newline separated rather than concatenated because
concatenation collides: path `/v1/orders` with body `x` and path `/v1/order`
with body `sx` produce identical bytes when joined. The path is taken verbatim,
query string included, so a signature cannot be lifted from one filter onto
another.

**A signature is good once.** The timestamp gives you a thirty second skew
window, and within that window a captured request used to be replayable as
itself. It is not any more. Do not retry a failed request by resending the
exact bytes; re-sign it with a fresh timestamp. Two genuinely identical orders
are fine, because they differ in their timestamp, which the signature covers.

**One error for every authentication failure.** Unknown key, bad signature,
stale timestamp, revoked key and replayed signature all return the same code
and the same message. This is deliberate: distinguishing them tells a caller
holding no valid key which key ids exist. Do not try to branch on the reason.

---

## What will refuse your orders, and what to do about it

The venue rejects rather than silently adjusting, so every one of these is
recoverable if you read the reason.

**`invalid_price` means you are off the grid, and the grid is not uniform.**
Most contracts have one tick everywhere. At least one does not: `VANTA_OBJECTIVE_WR`
steps by 1.00 above 4,000 while its base tick is 0.25. Rounding a modelled
price to the base tick returns 5232.25 there, which is refused. Round to the
increment in force *at that price*, and check again afterwards, because one
pass can round you up into a coarser band and land off its grid. Round a bid
down and an ask up, so snapping never makes your quote more aggressive than you
intended.

**`outside_price_band` is a fraction of the settlement range, not of the
price.** Five percent of what the contract can be worth. On a future ranging
zero to 10,000 that is generous. On a binary ranging zero to one it is five
ticks of a hundred, and any ordinary change of opinion crosses it. This is the
single most common reason a market order does not fill on an event contract.
Measured on this venue: **97.9% of market orders reaching an event contract
were refused** while the market maker was quoting wider than the band, against
6.1% on the futures. If you are trading binaries, price your orders inside the
band deliberately rather than sending marketable orders and hoping.

**`insufficient_collateral` usually means the capital constraint bound, not
that you were wrong.** Measured against the informed agents' own uncertainty,
collateral over-charges plausible loss by roughly 47x on a future, 130x on a
short option and 180x on a short commodity. Only binaries are charged anything
close to their real risk. A Kelly-sized bet on a future here asks for about
1,290% of bankroll, so on everything except binaries your capital binds an
order of magnitude before your conviction does. Size against free capital
first and against your edge second.

**`post_only_would_cross`** means your quote would have taken rather than
made. That is the flag doing its job. Note it refuses the order outright rather
than repricing it, so a maker that cannot cross also cannot exit a position.

**`rate_limited` has an exemption worth knowing.** A rolling one second
allowance, so a burst cannot be split across a boundary and counted as two
quiet windows. But a command that only reduces risk, a cancel, is counted and
never refused. A participant that cannot withdraw is holding exposure nobody is
allowed to manage, so you can always pull your orders even at your cap. Cancel
first, then re-add.

**`not_accepted_in_auction`** means the symbol is in a call phase. Orders
accumulate there without matching, so a crossed book during an auction is
correct rather than broken. Do not treat a bid above its own ask as bad data.

---

## Four things that have actually broken algorithms here

**Bind to a stable name, never to an account id.** The market can be rebuilt,
and a rebuild discards every account. The dangerous part is that an unknown
agent id does not error; it silently resolves to a shared account. So a client
that caches an id and reuses it after a rebuild does not fail, it quietly
starts trading a communal seat alongside everyone else who did the same. This
has bitten browser sessions, API keys and the streaming socket's private
channels. Re-resolve your seat from your key on every request.

**A restart loses the market.** The whole exchange lives in one process's
memory. There is a journal in the repository that would survive a restart, and
it is deliberately not wired in yet, so today a restart means a fresh market
with fresh accounts. Hold no state across a disconnect that you cannot
rediscover from `GET /v1/account`.

**Resume the stream by sequence, not by reconnecting.** The socket resumes from
`(session, next_expected_seq)`. If you reconnect and reopen the cursor at
"now", every event from the disconnected window is dropped and you will not be
told, because the sequence runs gapless straight across the hole. Track the last
sequence you processed and ask for it.

**Contracts expire and settle.** Positions realise, cash moves, and the symbol
stops trading. This is not an error condition and your position keeper has to
expect it. Since matches were listed it is no longer only an end-of-window
event: a match contract can settle mid-session, seconds after you traded it,
because the outcome it named became certain. Code that treats "the symbol is
gone" as a transport failure will thrash against this several times a minute.

---

## The market you are trading against

Nine asset classes on one matching engine: futures, binaries, calls, puts,
calendar spreads, an index, commodities, equities and a volatility contract.
The listing runs to **355 markets, 316 of them prediction markets**, and it is
mostly matches: a ten-entrant solo match lists 270 contracts on its own and a
three-a-side objective match lists 38.

Collateral is released at the **netting group**, and nowhere else. Netting is
currently off, so the venue charges gross per contract, which is worth knowing
before you size a package that looks self-hedging.

The population is three market makers, a maker that quotes whole matches,
informed traders holding Bayesian posteriors over the underlying statistic,
noise traders, a flow trader and an arbitrageur. They are not obstacles placed
to be beaten; the informed agents genuinely know things, and the price genuinely
aggregates what they know.

Uninformed flow is 3.0% of volume, up from 0.5%. That is a dial rather than a
fact about the world, and it is set low deliberately: a realistic 16% costs ten
times real time to simulate. Do not calibrate an adverse-selection model here
against a real venue's toxicity and expect the numbers to transfer.

### Trading a live match

A match is where this venue stops resembling a statistics book, and there are
four things about it that will surprise a client written against the rest of
the exchange.

**Symbols name the match, not the world.** `SOLO7_WIN_CINDER`,
`SOLO7_TOP3_QUILL`, `SOLO7_ELIM_CINDER_GT2`, `SOLO7_H2H_VANTA_OVER_QUILL`, and
on a team format `OBJECTIVE12_WIN_KESTREL-QUILL-VANTA`, where the side is named
by its members. The field is drawn per match, so the same competitor is not in
every match and a symbol from one match does not exist in the next.

Do not parse a ticker to work out what a contract is. `GET
/v1/instruments/{symbol}` returns the contract, and its underlying's metric
reference is where the answer actually lives: `metric` is `match_win`,
`match_place` or `match_eliminations`, `subject` is the competitor or the side,
and the match tag rides in `maps` as `match-7`. A relation recovered by string
surgery on a ticker breaks the first time anything is renamed, and it breaks
silently, because the wrong string still parses.

**The prices are coherent, and they are coherent by construction.** Every
contract on one match is priced as the mean of the settlement rule over one
ensemble of drawn outcomes, so the winner set sums to one, the placement ladders
are monotone and a competitor who wins is above every other competitor, exactly
rather than approximately. Measured: zero violations across 22,650 relation
checks in exact rational arithmetic. **There is no free arbitrage inside a
match**, and a strategy built to harvest one will find nothing. The same
contracts quoted off eight independently drawn ensembles breach 53 of the 395
relations by up to 0.1163, which is what you would be looking for, and it is not
there.

**Contracts settle while the match is still running.** A competitor who has been
eliminated cannot win, so that contract is worth zero and is settled the moment
the elimination is revealed, while everyone still in keeps trading. Your
position keeper has to expect settlement inside a session, not only at an
expiry. Only certainties settle early: an elimination says nothing final about
where the survivors will place, so their contracts stay open.

**A price of exactly zero is a real price, and a zero offer is not.** The
pricing ensemble is finite, so an outcome it never drew prices at exactly zero.
Measured at 4,000 draws over 20 beliefs, as many as 34 of a solo match's 270
contracts priced at exactly 0 or 1 on a single belief, and 611 of the 612 such
prices across all twenty were rungs of an eliminations ladder, which is where
the tail is. Those prices are coherent, they are simply at the boundary. A maker
quotes a zero fair value as 0 bid against a one-tick offer, because the offer is
a ceiling and never a floor.

Two measured facts to design around:

**Do not trade the opening auction unless that is the strategy.** Every book
opens at the midpoint of its settlement range, which for a deep out of the
money call is 2,725 against a fair value of 119. A strategy that sold the open
would post a spectacular and meaningless result. It also matters after the
open: in a paired test, auction inventory ate one arm's collateral so
thoroughly that it traded 1,803 lots against the other's 23,383.

**Adverse selection dominates market making here.** Decomposed into spread
captured, the drift of the mid after each fill, and residual inventory, the
makers' entire loss is the middle term, while spread capture is positive.
Widening in response is usually wrong: the makers are most of the book, so a
wider quote widens the mid everyone prices off, and the informed traders size
by edge over uncertainty and simply trade bigger. Size and skew are the levers
that work.

---

## Where to go next

- `docs/API.md` for every endpoint, error code and a worked signing example.
- `docs/STRATEGIES.md` to write a strategy that runs inside the simulation,
  with a backtest harness that reports whether its own paired trials helped.
- `docs/ECONOMY.md` for what the contracts are written on and how they settle.
- `docs/GAPS.md` for an honest audit of this venue against a real one.
- `CONTRIBUTING.md` if you are changing this repository rather than trading on
  it. It is a different job with different traps.
