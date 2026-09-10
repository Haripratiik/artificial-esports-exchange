# Writing a strategy

This is the document that takes you from nothing to a strategy running on this
exchange, with a number at the end you are entitled to believe. It assumes you
know what a limit order book is and what adverse selection means, and assumes
nothing at all about this repository.

**No real money and no real securities are involved.** The underlyings are a
synthetic esport this repository generates from a seed, its matches and the
statistics measured on them. Every counterparty is a simulated agent and the
capital is imaginary. What is real is the microstructure: a matching
engine with price-time priority, per-agent latency, an opening auction, a
circuit breaker, integer collateral, and a ledger whose conservation check is
exactly zero. A strategy that loses money here loses it for reasons that would
also lose money elsewhere.

- [What a strategy is](#what-a-strategy-is)
- [A minimal maker](#a-minimal-maker)
- [A minimal taker](#a-minimal-taker)
- [What the view gives you](#what-the-view-gives-you)
- [What the view refuses to give you](#what-the-view-refuses-to-give-you)
- [Running one](#running-one)
- [Comparing two](#comparing-two)
- [Reading the attribution](#reading-the-attribution)
- [The contracts](#the-contracts)
- [Traps, with the numbers](#traps-with-the-numbers)
- [The baseline library](#the-baseline-library)

---

## What a strategy is

A strategy reads a `MarketView` and returns intents. It never touches the venue.
That is not tidiness, it is the whole validity argument: an object holding a
reference to the venue could read another participant's position, the true
settlement level, or the book of a contract it was never told about, and a
backtest of such a thing measures nothing. Everything in `MarketView` is
assembled from what the agent was *sent*, so a strategy is structurally unable
to see what a real desk could not.

There are two protocols, in `arena/strategies/base.py`, and they are separate on
purpose.

```python
class MakerStrategy(Protocol):
    def quote(self, view: MarketView, symbol: str) -> TwoSided: ...
    def symbols(self, view: MarketView) -> Sequence[str]: ...   # optional

class TakerStrategy(Protocol):
    def orders(self, view: MarketView) -> Sequence[Take]: ...
```

A maker's job is to have a price in the market at all times and be paid for the
risk of doing so. A taker's job is to decide the price on the screen is wrong
and pay the spread to say so. They fail differently and they are measured
differently, one by realized spread and one by hit rate and edge. A single
`act()` method for both is what produced the makers in this repository that were
aggressive on 61% of their fills, which is why there are two.

One object may implement both. A firm that quotes one asset class and takes in
another is an ordinary shape and refusing it would be an artificial limit.
`arena.strategies.firm.Firm` composes several strategies under one budget.

The three intent types are small:

```python
Quote(price: Decimal, size: int)              # one side of a market
TwoSided(bid: Quote | None, ask: Quote | None)  # None means do not quote that side
Take(symbol: str, side: Side, size: int, limit: Decimal | None = None)
```

`None` on a side of a `TwoSided` means *do not quote it*, which is a legitimate
and frequently correct answer. Pulling the side somebody keeps picking off is
the cheapest defence there is. It is distinct from a size of zero, which is not
representable, because an order for no lots is not a thing to have an opinion
about.

`StrategyAgent` is the only thing that turns intents into orders. It does three
things you would otherwise have to write yourself and get wrong:

It **sends differences, not quotes.** A maker that cancels and reposts a price
that has not moved gives up its place in the queue for nothing. Returning the
same prices as last time therefore costs nothing and keeps your queue position.
Removing the unconditional repost from this repository's own makers took the
event rate from 1.6M per simulated minute to 317K.

It **asks you again the moment you are filled.** In Glosten-Milgrom the arrival
of an order *is* the news, and a quote that does not move when it is lifted can
be lifted again at the same price. Measured here, 17% of a maker's passive fills
were a second fill at the same price within 500ms before this existed.

It **measures your markout and hands it back.** GLFT's adverse selection term
`xi` is not a parameter to guess, it is the drift of the mid after your own
fills, and it arrives in `view[symbol].markout` per side, signed so that
positive always means the mid moved your way.

---

## A minimal maker

This runs. It is not pasted from a sketch; it is the code that produced the
output in [Running one](#running-one).

```python
from decimal import Decimal

from arena.strategies.base import Quote, TwoSided


class ShadeTheMid:
    """Rest a fixed half-spread either side of the mid, leaning against inventory.

    Everything it uses is on the view: the local best bid and offer, its own
    position, and the range the contract can settle in. It never sees the
    settlement level, so it cannot be right for the wrong reason.
    """

    def __init__(self, half_spread=8, size=4, lean="0.25", limit=400):
        self.half_spread = half_spread
        self.size = size
        # Decimal, because the reference is a Decimal and multiplying one by a
        # float raises rather than coercing.
        self.lean = Decimal(lean)
        self.limit = limit

    def symbols(self, view):
        return view.symbols

    def quote(self, view, symbol):
        row = view[symbol]
        reference = row.reference
        if reference is None:
            return TwoSided()

        tick = row.instrument.tick_size
        # Long inventory pushes the whole quote down, so the offer is the
        # attractive side and the position comes back on its own.
        centre = reference - row.position * self.lean * tick
        low, high = row.bounds

        bid = None
        if row.position < self.limit:
            bid = Quote(max(low, centre - self.half_spread * tick), self.size)
        ask = None
        if row.position > -self.limit:
            ask = Quote(min(high, centre + self.half_spread * tick), self.size)
        return TwoSided(bid=bid, ask=ask)
```

Three things in there are load-bearing.

**Clamp each side independently.** `max(low, ...)` and `min(high, ...)` hold the
quote inside the range the contract can actually settle in. Clamping the
*centre* instead was tried in this repository and was worse: it makes the mid a
function of the half-spread, which widens with inventory, and it priced two
worthless calls forty points apart.

**Prices leave as `Decimal` on the tick grid.** You may model in floats, and the
literature's formulas are floating point so pretending otherwise would be
theatre. But `row.position * self.lean * tick` is Decimal arithmetic and a float
in the middle of it raises `TypeError` rather than coercing. `snap()` in
`arena.strategies.base` quantises whatever you emit before it becomes an order,
rounding a bid down and an offer up so that snapping never makes your quote more
aggressive than you asked for.

**A position limit expressed by pulling a side.** Widening is the reflex and it
is usually the wrong tool. If you are long your limit, the bid is the problem,
and `bid=None` says exactly that.

---

## A minimal taker

```python
from arena.exchange.types import Side
from arena.strategies.base import Take


class FadeTheMove:
    """Buy what has fallen away from its own recent average, and sell the mirror.

    The view carries no history, so the strategy keeps its own. That is the
    honest arrangement: everything it remembers is something it was sent.
    """

    def __init__(self, size=3, bands=2.5, gain=0.05, limit=120):
        self.size = size
        self.bands = bands
        self.gain = gain
        self.limit = limit
        self.level: dict[str, float] = {}
        self.width: dict[str, float] = {}

    def orders(self, view):
        wanted = []
        for row in view:
            if row.mid is None or row.best_bid is None or row.best_ask is None:
                continue
            mid = float(row.mid)
            anchor = self.level.get(row.symbol)
            if anchor is None:
                # The first sighting is the anchor. Trading on it would be
                # trading on one observation.
                self.level[row.symbol] = mid
                self.width[row.symbol] = 0.0
                continue

            gap = mid - anchor
            width = self.width[row.symbol]
            self.level[row.symbol] = anchor + self.gain * gap
            self.width[row.symbol] = width + self.gain * (abs(gap) - width)
            if width <= 0.0:
                continue

            if gap < -self.bands * width and row.position < self.limit:
                wanted.append(Take(row.symbol, Side.BUY, self.size, row.best_ask))
            elif gap > self.bands * width and row.position > -self.limit:
                wanted.append(Take(row.symbol, Side.SELL, self.size, row.best_bid))
        return wanted
```

Two notes.

**Order matters.** The adapter funds intents in the order you return them and
stops at the first one the account cannot collateralise. A strategy that wants a
package rather than a list should say so by ordering its legs, and should expect
to be left half-legged sometimes. That is a real execution risk, not an
artefact.

**Name your price.** `limit=None` means marketable, and the venue's price band
still applies, so declining to name a price does not escape the listing rules.
Naming one is strictly safer. A limit order carries the price you chose;
a market order carries nothing, which is why the venue cancels an unfilled
market-on-open rather than letting it rest.

---

## What the view gives you

`MarketView` is a mapping from symbol to `SymbolView`, plus your account.

| On the view | What it is |
| --- | --- |
| `view.now` | simulated seconds since the session began, as a float |
| `view.symbols` | every contract this agent lists |
| `view.cash`, `view.free_cash` | the strategy's own shadow ledger, and cash not backing a position |
| `view.posted_collateral`, `view.equity` | computed with the same public arithmetic the venue charges with |
| `view.rng` | a deterministic stream seeded per agent; anything random must come from here |
| `view.positions()` | non-zero positions, by symbol |

| On a `SymbolView` | What it is |
| --- | --- |
| `best_bid`, `best_ask`, `last` | the agent's local book, as of the last message it received |
| `mid`, `spread` | derived, `None` when either side is missing |
| `reference` | mid, else last print, else the midpoint of the settlement range |
| `position` | lots, signed |
| `working_bid`, `working_ask` | what this strategy already has resting |
| `bounds` | the range this contract can settle in, from the contract itself |
| `instrument` | tick size, tick bounds, `to_ticks`, `from_ticks`, `collateral_for` |
| `markout` | realized adverse selection on your own fills, per side, or `None` |
| `seconds_to_expiry` | seconds until this contract stops trading, or `None` |

`reference` is somewhere to start, not a fair value. On a contract that has
never traded it is the midpoint of the settlement range, which for a
deep out-of-the-money call is 2650 against a fair value of zero. Read
[Traps](#traps-with-the-numbers) before you anchor anything to it.

`markout` is `None` until enough of your own fills have matured, which is the
honest answer early on rather than a zero that reads as "no adverse selection".
The horizon is one second, chosen by measurement: on this market a maker's
adverse selection is *positive* through the first half second and only turns
over between one and five seconds, so a hundred-millisecond horizon would
measure latency and report the opposite sign.

---

## What the view refuses to give you

Four things, and each one is a place a backtest usually lies.

**No settlement level.** Nothing on the view says what a contract will be worth.
The informed agents in this market are told, and you are not, which is the
asymmetry the whole exchange exists to study. `bounds` is not a loophole: it is
written in the contract, it is the same thing the collateral engine charges
against, and it tells you what the claim *can* be worth, not what it *will* be.

**No other participant's position or identity.** You see prints and top of book.
You do not see who is behind them, what they are carrying, or what they are
about to do. The counterparty breakdown in the attribution report exists for the
researcher reading the result afterwards, not for the strategy while it runs.

**No future.** There is no lookahead of any kind, and there is no way to ask for
one, because the view is rebuilt fresh on every wake from messages that have
already arrived rather than mutated in place. A strategy that keeps a reference
to last tick's view is holding last tick's numbers, which is correct, rather
than silently seeing the present through a stale object.

**The local book is honestly stale.** `best_bid` is the best bid as of the last
`TopOfBook` message that reached this agent, which lags the venue by this
agent's latency. A market maker quoting off a view that is 100ms old *is* the
adverse selection problem, and smoothing it away would delete the phenomenon.
The backtest harness puts a strategy 5ms from the exchange by default. The
built-in makers are at 0.15 to 0.23ms and the informed traders at 3 to 13ms, so
by default you are slower than the makers and about as fast as the sharpest
informed agent. `BacktestConfig(latency_ms=...)` moves you, and it is a first
class knob rather than a constant, because "how much is speed worth" is one of
the questions this simulator was built to answer.

---

## Running one

```python
from arena.research.backtest import BacktestConfig, backtest, compare, manifest

result = backtest(ShadeTheMid(), seeds=range(8), until=600)
print(result)
```

`backtest` builds the full 47-contract market once per seed, adds your strategy
as a `StrategyAgent`, runs it, and pools the seeds. Everything it reports is
measured after a warmup that your strategy did not trade through. The defaults
are a 600 second session, a 60 second warmup, a five second return period and a
half-second attribution grid; all of them are fields on `BacktestConfig`, and
passing keyword arguments to `backtest` overrides them.

Run on one seed over a deliberately short session so the block below also shows
what a refusal looks like:

```python
CONFIG = BacktestConfig(until=26.0, warmup=14.0,
                        return_interval=2.0, sample_interval=2.0)
print(backtest(ShadeTheMid(), seeds=[3], config=CONFIG, name="shade"))
```

```
shade: -1,419,390.97 over 1 seed(s), 21,563 lots, turnover 2.82x, aggressor 51.9%, informed share of passive volume 38.6%
  Sharpe -2.454 +/- 0.688 per period (n=6, floor 28.9%), PSR 0.001, DSR 0.001 over 1 trial(s), MinTRL never: NOT supported by this sample
  spread +693,390.27  adverse -3,066,646.85  realized -2,373,256.58  residual +957,485.17
  drawdown 1,419,390.97 against 295,937.79 for no edge over the same horizon
  note: horizons [0.1, 0.5, 1.0]s were not collected: they are shorter than the 2.0s sampling grid, so they would measure the grid rather than the market
  note: the decomposition is reported at 2s and not the 1s asked for, because the mid series is sampled every 2.0s
  note: 1 seed(s): per-seed dispersion cannot be estimated, so nothing here separates the strategy from this particular market
  note: a Sharpe of -2.454 does not beat the benchmark of 0.000, so no track record length would establish it
  note: hit rate 0.0% over n=6; 85.4% is what a Sharpe of magnitude 2.454 takes at this n
```

Six periods on one seed is not a result and the object says so twice, in the
verdict and in the notes. Run the same strategy properly, five seeds over a 75
second session with a 15 second warmup across all 47 contracts, and the picture
is the same shape at a sample worth reading: 134,581 lots, +14,259,122 of spread
captured against -17,924,856 of adverse selection and -15,604,898 of inventory,
a per-period Sharpe of -0.113 +/- 0.082 over 150 periods, and 37.7% of its
passive volume taken by an agent who knew the settlement level.


What is in that block, and why each line is there:

**Sharpe with its standard error, always together.** Lo (2002) gives
`SE(SR) = sqrt((1 + SR^2/2)/n)` for normal iid returns, and the general form
`sqrt([1 - skew*SR + ((kurt-1)/4)*SR^2]/n)` when the moments are known, with
`kurt` the raw fourth-moment ratio, 3 for a Gaussian. Skew and fat tails widen
it: at a Sharpe of 1.0 over 100 periods, skew -3 and kurtosis 10 take the
standard error from 0.122 to 0.250, so the same ratio is twice as uncertain and
nothing about the point estimate says so. Nothing here is annualised, because a
period is a slice of simulated seconds and multiplying by `sqrt(252)` would be
inventing a calendar.

**The relative error floor.** `SE/SR = sqrt(1/(n*SR^2) + 1/(2n))`, which
decreases in `SR` and tends to `1/sqrt(2n)` from above. At the 108 five-second
periods a 600 second session yields, no strategy however good can be measured to
better than 6.8% of its own Sharpe. That floor is printed next to every ratio.

**PSR, DSR and MinTRL.** The probabilistic Sharpe ratio is the probability that
the true ratio beats the benchmark, which is the question a Sharpe is usually
asked and never answers. The deflated Sharpe ratio measures it against the
expected best of `N` variants rather than against zero, because the best of many
worthless things looks good; Bailey and Lopez de Prado's worked example is the
whole argument, since the same track record passes at 46 trials and fails at
100. `backtest(..., trials=N)` is how you tell it how many things you have
tried, and it defaults to 1 and is therefore wrong the moment you are on your
fortieth idea. The minimum track record length is how many periods it would take
to establish the observed ratio at 95%, and a 600 second session is short enough
that this is a live question rather than a formality.

**A verdict that can be "not supported".** When `n` is below the minimum track
record length, or the deflated ratio does not clear the confidence level, the
result says `NOT supported by this sample` and explains why in the notes. This
is the point of the object. A harness that prints a Sharpe of 3.1 from twelve
periods has told you nothing and made you feel it told you something.

**Turnover, aggressor fraction and informed share.** Turnover is traded notional
over the equity the window started with. The aggressor fraction is the share of
your lots that crossed the spread, and for a maker it is a defect detector: a
maker that is aggressive on most of its volume is a taker with extra steps, and
this repository's own makers were at 61% before anyone measured it. The informed
share is Glosten-Milgrom's `mu`, the fraction of your passive volume that came
from an agent who was told the settlement level. It is directly observable here
and unobservable on a real venue, which is most of the reason to run a market
you built.

**No max drawdown across horizons, and no Calmar at all.** With no edge,
`E[MDD] = 1.2533*sigma*sqrt(T)`, so drawdown grows with the length of a run
whether or not the strategy is any good, and ranking two strategies on it ranks
them on how long they were run. The harness prints a drawdown only when every
run in the pool shares a horizon, always beside what a driftless walk of the
same volatility would have produced, and refuses with `drawdown withheld: the
runs do not share a horizon` otherwise.

### The manifest

```python
record = manifest(result)
record.digest          # sha256 over config, seeds and a results digest
record.commit          # the commit it ran on
record.dirty           # whether the working tree was clean when it ran
```

The digest covers the configuration, the seed list and a digest of the results.
The results side is deliberately split: the per-seed ledger integers go in
exactly, and the derived statistics are rounded to twelve significant figures,
because the last bit of a float is a property of the machine and the integers
are what the reproducibility claim is really about.
`verify_reproducible(strategy, record)` replays the whole run from the manifest
and raises unless the digest is identical. It is deliberately expensive: it runs
the thing again rather than comparing a stored hash to itself.

---

## Comparing two

```python
comparison = compare(AvellanedaStoikov(), FixedSpread(), seeds=range(8))
print(comparison)
```

Both arms run on the same seeds, under the same agent id, at the same latency.
That is what makes them paired, and it matters more here than in most simulation
studies. The kernel derives every agent's random stream from the seed and the
agent's own id, and the latency model draws its jitter per ordered pair, so
adding your strategy to the market cannot shift any other agent's draws. Two
arms under one id therefore share everything except the consequences of their
own orders.

`Var(A-B) = Var(A) + Var(B) - 2*Cov(A,B)`, so pairing pays exactly to the extent
that the covariance is positive, and it can only be positive if the seed drives
the same draws in both arms. Most simulation studies cannot say that. Because it
is a claim rather than a guarantee, the result reports the realised covariance,
the correlation, the ratio `Var(A-B) / [Var(A) + Var(B)]`, and how many unpaired
seeds the pairing was worth.

```python
duel = compare(ShadeTheMid(), ShadeTheMid(half_spread=20), seeds=[3, 4, 5],
               config=CONFIG, names=("shade-8", "shade-20"))
print(duel)
```

```
shade-8 vs shade-20 on 3 paired seed(s), statistic pnl
  difference +461,626.08 [-81,116.98, +978,856.79], p=0.2707: no difference this sample can see
  cov +1.86e+12 (sqrt +1,363,650.19), rho +0.953, Var(A-B)/[Var(A)+Var(B)] = 0.0703
  pairing removed 93.0% of the variance, worth 42.7 unpaired seeds
```

That third line is the one to read first. A correlation of +0.953 between the two
arms' per-seed P&L means the seed really is driving the same market in both, and
`Var(A-B)` came out at 7.03% of `Var(A) + Var(B)`, so three paired seeds bought
the precision of about 43 unpaired ones. The verdict is still "no difference this
sample can see", which is the correct answer at three seeds and is what the
object says rather than reporting the +461,626 as a finding.

**Pairing is not free and it is not guaranteed.** The same comparison over a
longer session, five seeds of 120 simulated seconds with half-spreads of 8 and
24 ticks, came back the other way:

```
fixed-8 vs fixed-24 on 5 paired seed(s), statistic pnl
  difference -544,743.94 [-3,416,658.74, +2,077,142.48], p=0.7444: no difference this sample can see
  cov -1.751e+11 (sqrt -418,425.67), rho -0.033, Var(A-B)/[Var(A)+Var(B)] = 1.0297
  pairing removed -3.0% of the variance, worth 4.9 unpaired seeds
  note: covariance -1.751e+23 is not positive, so pairing did not help here: the two arms drove the market apart rather than sharing it
```

Correlation -0.033 and a variance ratio slightly *above* one: pairing made this
comparison marginally worse than an unpaired design would have been. The reason
is mechanical. Common random numbers hold the other agents' draws fixed, but a
maker that quotes 8 ticks wide and one that quotes 24 trade different volume
into the same books, so past a certain horizon each arm has moved the market
somewhere the other one never went and the shared seed is no longer shared
experience. Two arms that behave identically stay at rho exactly +1.000 with
`Var(A-B)` exactly zero; two arms that behave very differently over a long
window decorrelate. This is why the covariance is a reported number and not an
assumption: it tells you which of those two regimes you are actually in.


The p-value comes from `aggregation.paired_comparison`, which is the same paired
t-test and bootstrap interval the information-aggregation experiment uses. One
wart worth knowing: that function scores *losses*, so lower is better, and
`compare` feeds it the negated P&L and flips the sign back. The `Paired` object
you get holds `mean_difference` in the natural direction, A minus B, positive
when A made more money.

For several candidates against one baseline:

```python
rows = compare_many({"as": AvellanedaStoikov(), "glft": GueantLehalleFT()},
                    FixedSpread(), seeds=range(8))
```

Benjamini-Hochberg is applied across the set by
`aggregation.benjamini_hochberg`, reused rather than rewritten, so a claim made
here is corrected the same way a claim made by the experiments is. Comparisons
whose p-value is undefined are held out of the correction rather than passed in,
because the procedure sorts on the p-value and a `NaN` sorts unpredictably.

---

## Reading the attribution

Total trading P&L telescopes exactly, for any fill sequence and any horizon `h`:

```
  sum q_i (M_T - P_i)  =  sum q_i (M_i     - P_i)      spread captured
                        + sum q_i (M_{i+h} - M_i)      adverse selection over h
                        + sum q_i (M_T     - M_{i+h})  inventory and residual
```

with `q_i` the signed lots, `P_i` the fill price, `M_i` the mid prevailing before
the fill and `M_T` the closing mark. Nothing is assumed and nothing is
estimated, because the two middle terms are added and subtracted. The first two
are the Huang-Stoll effective and realized spread, and their difference is what
the literature means by adverse selection.

The three terms fail in different directions and want opposite fixes.

**Negative spread capture** means you were behind the market and are being run
over: your quote was on the wrong side of the mid at the moment it traded. Widen
and you will simply be run over less often at a worse price. The fix is a faster
anchor. Use `row.mid` rather than `row.last`, requote on fill instead of only on
your timer, or shorten `wake_ms`.

**Negative adverse selection** means you were picked off by somebody who knew
more. The mid moved against you after you traded. The fix is size, skew, or not
quoting that side at all. Widening helps here and only here, and on a market
this thin not even always, because a maker that is most of the book widens the
mid it is measured against. Check `informed_share` next to it: if most of your
passive volume came from the informed agents, that is `mu` and no spread may be
profitable.

**Negative residual** is inventory. You were right about the trades and wrong
about what you were left holding. The fix is a limit or a hedge, not a wider
quote. `RunResult.flow_imbalance` is the cheapest detector available: it reports
net passive lots over gross per symbol, and zero is a healthy quote. Inventory that
swings around zero is risk, but a maker persistently taken on the same side of
the same contract is not unlucky, it is priced wrong.

**Read the ladder, not a number.** `RunResult.adverse_curve` gives adverse
selection at every collected horizon. A jump inside the first hundred
milliseconds that is flat afterwards is a stale quote being arbitraged, which is
a latency problem. A drift that keeps going for seconds is information, which is
not. The two look identical in a single-horizon summary and want completely
different work. The harness collects horizons only at or above its sampling
interval and says so in the notes, because a horizon shorter than the grid is
measuring the grid.

---

## The contracts

Fifty statistical contracts in nine classes, all settling from one bounded
scalar per underlying, which is what makes collateral arithmetic rather than a
model. Positions on different underlyings do not net, because netting them would
need a correlation and a correlation is an estimate. Nor does one competitor
across the two formats: their win rates rank the field at Spearman -0.357, so
treating them as one underlying would be assuming a relationship the number says
is not there.

Whatever matches are open sit on top of that. A ten-entrant solo match adds 270
event contracts and a three-a-side objective match 38, so the board runs to 355
markets of which 316 are prediction markets. The table below is the statistical
listing, because that is what a strategy written against a level or a spread is
trading; matches have their own section after it.

A ticker names its subject, its format and what it claims: `VANTA_SOLO_WR` is
VANTA's win rate in the ten-way elimination, `VANTA_OBJECTIVE_WR` the same
competitor in the three-a-side. They are different underlyings and they do not
net, which the naming is there to keep visible.

| Class | Count | Tick | What is different about it |
| --- | --- | --- | --- |
| `future` | 19 | 0.25 | A linear claim on a rate. The plainest thing here, and what everything else is priced off. Twelve are win rates, two are eliminations per match, one is a score margin, and four are the weekly legs of the share. |
| `event` | 10 | 0.01 | Binaries, bounded in [0, 1], so the price *is* a probability and a tick is a percentage point. The only class where a wrong quote is bounded by one unit of loss. |
| `call` | 5 | 0.25 | Convex in the same underlying as the futures, so its fair value is a function of the *distribution* rather than the level. Two of the five settle at exactly zero. |
| `put` | 5 | 0.25 | The mirror. One settles at exactly zero. Together with the calls this is a surface, and the maker that quotes it prices one distribution rather than each book on its own. |
| `commodity` | 5 | 0.25 | A claim on an amount delivered over a window rather than a proportion, so these come in a term structure and carry information about carry as well as level. The one metric here that is a quantity. |
| `volatility` | 3 | 0.25 | A claim on dispersion. Not a level, so a strategy that treats it as one will be systematically wrong in the same direction. |
| `spread` | 1 | 0.25 | `HALCYON_FORMAT_SPD`, one competitor's solo win rate against its objective one, bounded in [-10000, 10000]. **The only contract that can be worth a negative number.** Any strategy that assumes prices are positive breaks here and nowhere else. |
| `index` | 1 | 0.25 | `CIRCUIT_SOLO_IDX`, an equal-weight basket of six solo win rates. Its identity against its constituents is exactly the kind of relation `StaticArbitrage` trades. |
| `equity` | 1 | 0.25 | Pays as it goes and settles at the end, so a short can be asked for the stream as well as the settlement. `BASTION_SOLO_EQ` has `settlement_bounds` of (0, 0): all of its value is the stream, and collateral has to cover the stream. |

Two things generalise from that table.

**Bounds are the contract, not a convention.** A call bounded in [0, 5200] and a
binary bounded in [0, 1] are not the same instrument scaled. Collateral is the
worst case of a piecewise-linear payoff evaluated at its endpoints and kinks, so
a short in the call ties up capital in proportion to 5200 and a short in the
binary ties up capital in proportion to 1. `row.instrument.collateral_for()`
computes it, and it is the same arithmetic the venue charges with.

**Tick sizes differ, and one contract has a tiered grid.** `VANTA_OBJECTIVE_WR`
moves in 0.25 low down and 1.00 above 4,000, and every other contract has one
increment everywhere. Snapping a price is therefore not a single division. Use `snap()`,
and note that the agent layer repeats the snap because a single pass can round
*into* a coarser band and land off its grid: with steps of 1.00 from 100 and
5.00 from 203, an offer at 202.75 snaps up to 203.00, which is not a multiple of
five, and comes back `INVALID_PRICE`.

### Matches, which behave differently enough to say so

A match contract is an `event` like any other, and three things about it will
surprise a strategy written against the statistical listing.

**They settle mid-session.** A competitor who has been eliminated cannot win, so
that contract is worth zero and settles the moment the elimination is revealed,
while the rest of the match keeps trading. Your position keeper cannot assume
settlement only happens at an expiry, and a strategy that treats a vanished
symbol as a data error will thrash several times a minute.

**There is no arbitrage inside a match, and that is by construction.** Every
contract on one match is priced as the mean of the settlement rule over a single
ensemble of drawn outcomes, so the winner set sums to one and both ladders are
monotone as arithmetic rather than as policy. Measured: zero violations across
22,650 relation checks in exact rational arithmetic. `StaticArbitrage` will find
nothing here, and that is the mechanism working. The control is worth knowing
because it says what the alternative costs: the same contracts priced off eight
ensembles drawn from the same belief breach 53 of the 395 relations by up to
0.1163.

**A price of exactly zero is real, and a zero offer is not.** The ensemble is
finite, so an outcome it never drew prices at zero. Measured at 4,000 draws over
20 beliefs, as many as 34 of a solo match's 270 contracts priced at exactly 0 or
1 on one belief, and 611 of the 612 such prices across all twenty were rungs of
an eliminations ladder. A maker quotes a zero fair value as 0 bid against a
one-tick offer, because the offer is a ceiling and never a floor.

Symbols look like `SOLO7_WIN_CINDER`, `SOLO7_TOP3_QUILL`, `SOLO7_ELIM_CINDER_GT2`
and `SOLO7_H2H_VANTA_OVER_QUILL`, with a team side named by its members as in
`OBJECTIVE12_WIN_KESTREL-QUILL-VANTA`. Read the contract rather than the ticker:
the view carries the underlying, whose metric is `match_win`, `match_place` or
`match_eliminations` and whose `maps` filter carries the `match-N` tag.

---

## Traps, with the numbers

**The opening auction is not a market.** Every book opens at the midpoint of its
settlement range, because a maker that opened on the answer would be the thing
that already knew. The dislocation is enormous and it is not uniform:

| Contract | Opens at | Fair value | Ratio |
| --- | --- | --- | --- |
| `VANTA_OBJECTIVE_WR` | 5000.00 | 4893.50 | 1.02 |
| `VANTA_SOLO_WR` | 5000.00 | 1397.50 | 3.58 |
| `EMBER_OBJECTIVE_C4800` | 2600.00 | 238.25 | 10.9 |
| `QUILL_SOLO_C600` | 4700.00 | 164.75 | 28.5 |
| `EMBER_OBJECTIVE_C5000` | 2500.00 | 38.25 | 65.4 |
| `QUILL_SOLO_C800` | 4600.00 | 0.00 | infinite |
| `VANTA_SOLO_GT140` | 0.50 | 0.00 | infinite |

Seven of the fifty statistical contracts settle at exactly zero, two calls, one
put and four binaries, and every one of them opens somewhere between 0.50 and
4,600. A strategy that sells the open would post a spectacular and
meaningless Sharpe ratio, and it would be measuring the builder rather than
itself. The call clears at ten simulated seconds, so the harness holds your
strategy out of the market until the warmup has elapsed and starts every
statistic at the end of it. Both halves matter: excluding the warmup from the
statistics without also holding the strategy out would leave the inventory it
took in the auction inside the measured window. Measured on seed 3 with the
minimal maker quoting the statistical books, letting it trade the open books 2,739,688 in
fourteen seconds, which is 13.7% of its capital, and then takes the post-warmup
P&L from -2,044,297 to -9,028,498. Run that control yourself with
`BacktestConfig(trade_during_warmup=True)`.

**Do not report max drawdown or Calmar across runs of different length.** With
no edge, `E[MDD] = 1.2533*sigma*sqrt(T)`. Doubling the horizon multiplies the
expected drawdown of a strategy with no edge at all by 1.414. Either normalise
to a fixed horizon or omit it; the harness omits it, and prints the no-edge
expectation next to the figure when it does report one.

**A Sharpe ratio needs its sampling error printed beside it.** `SE(SR) =
sqrt((1 + SR^2/2)/n)` under normal iid returns, and `sqrt([1 - skew*SR +
((kurt-1)/4)*SR^2]/n)` in general, with `kurt` raw. The relative error floors at
`1/sqrt(2n)`, which is 10.2% at 48 periods and 6.8% at 108, however good the
strategy is.

**Report the deflated Sharpe ratio against the number of variants you tried.**
Anyone using a testbed tries many things and the best of many worthless things
looks good. At a per-period ratio of 0.30 over 250 periods with a trial variance
of 0.006, the deflated ratio is 0.974 after 46 trials and 0.946 after 100: one
unchanged track record, passing and failing on nothing but how many other things
were tried beside it. Report `MinTRL` too, since a 600 second session is short
and whether the sample supports the claim at all is a live question.

**Sortino divides by `N`, not by the count of losing periods.** The common
implementation divides by the loss count and it is wrong in the flattering
direction. Ninety-nine periods at +0.01 and one at -0.10: the downside deviation
over all hundred is 0.010 and over the single loser it is 0.100, so the wrong
version reports a Sortino ten times larger for a series that took exactly the
same risk. Downside deviation is also not the standard deviation of anything and
does not annualise by `sqrt(12)`.

**A hit rate alone is uninformative.** `p = 0.5*(1 + sqrt(theta^2/(n +
theta^2)))`. A per-period Sharpe of 2 needs 53.2% of periods to win over a
thousand of them and 63.6% over fifty. The same 55% is therefore excellent in
one column and poor in the other, and the column is the sample size. Never
report a hit rate without its `n`; the harness returns them as a pair so the
count cannot be dropped on the way to a slide.

**The book you read is stale and the fill you get is not the price you saw.**
`snap()` rounds a bid down and an offer up, which gives up a fraction of a tick
rather than crossing something you did not mean to cross. On a thin book that
tick is most of the edge. If your aggressor fraction is high for a maker, you
are quoting through a stale mid: your reference has moved and the price you post
crosses what is actually there.

**Nothing may be hardcoded to a symbol, a seed, a strike or an agent.** That is
a house rule for this repository and it is also self-interested advice. A
strategy tuned to `QUILL_SOLO_C600` on seed 7 has stopped being evidence about
anything, and the harness makes that visible rather than comfortable: run it on
`seeds=range(8)` and the per-seed dispersion is right there.

---

## The baseline library

The library in `arena/strategies/` is the baseline set, not a recommendation.
Each one is a named model from the literature so that a strategy you write has
something honest to be compared against.

| Strategy | Where it lives | What it assumes |
| --- | --- | --- |
| `FixedSpread` | `making` | A constant half-spread skewed by inventory. The baseline, and roughly what this repository's own makers do. |
| `AvellanedaStoikov` | `making` | Inventory-optimal quoting. The spread is independent of inventory and the *centre* moves instead. |
| `GueantLehalleFT` | `making` | The closed-form solution with a hard inventory bound and an additive term for measured adverse selection. |
| `GlostenMilgrom` | `making` | Quotes conditioned on the order that just arrived. The only one of the four in which being filled is news. |
| `KellyBayesian` | `taking` | Sizes a belief against a price. On this venue the stake and the collateral are the same number, which is Kelly's own setup rather than an analogy. |
| `StaticArbitrage` | `taking` | Trades relations that must hold at settlement whatever the outcome, so a loss is an execution failure and not a wrong view. |
| `Firm` | `firm` | Composes several strategies under one budget and attributes the result back to each. |

The comparison that means something is against `FixedSpread` on the same seeds,
paired, with the covariance printed and Benjamini-Hochberg applied if you are
testing more than one idea at once. Everything else is a story.
