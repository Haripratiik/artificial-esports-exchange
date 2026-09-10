"""Two exact binary families that nothing in this market currently enforces.

`derive_relations` in :mod:`arena.agents.arbitrageur` reads the pricing algebra
out of the listed contracts and trades it, and for options it is complete: the
vertical and butterfly bands together are the whole of static-arbitrage freedom
for a call chain at one maturity, and measured at t=540s they hold, 0 of 8
violations on seed 3 and 0 of 6 on seed 11. So this module does not touch them.

What it never looks at is a :class:`~arena.contracts.payoff.Binary`. The chains
it builds are keyed on ``Call`` and ``Put`` payoffs and a binary matches
neither, so the event ladder on this exchange is priced by whoever happens to be
in each of its books and is tied to nothing. That is not a theoretical gap.
Binary ladder monotonicity is an exact static-arbitrage condition, and it was
violated in 3 of 3 observed runs: seed 7 at t=180s had 2 of its 5 ladder
relations violated, by 0.23 and 0.31 on a contract bounded by [0, 1]; seed 3 at
t=540s, 1 of 5 by 0.16; seed 11, 1 of 5 by 0.045. A quarter of a dollar on a
dollar contract is not a no-arbitrage band, it is a free lunch nobody was
listed to eat.

Two families, both exact at settlement rather than statistical:

    ladder      P(theta > t1) >= P(theta > t2)          for t1 < t2
    chain       (K2-K1) D(K2/S) <= C(K1) - C(K2) <= (K2-K1) D(K1/S)

The first is monotonicity of a survival function, written as a portfolio: buy
the low rung, sell the high one, and the position settles at 1 when the level
lands between the two thresholds and at 0 everywhere else. The second is the
call spread sandwiched between the digitals at its own strikes, which holds
pathwise because ``max(F-K1,0) - max(F-K2,0)`` is ``0`` below ``K1``, ``F-K1``
between them and ``K2-K1`` above, and both bounds are that step function
evaluated at one end of the ramp. The put chain carries the mirror of it, with
the down-digital in place of the up. Every one of them is an inequality that
holds at every level the metric can take, so a violated one is a package that
cannot lose, and a loss on it is an execution failure rather than a wrong view.

Relations are derived from the listing, never written down: the ladder is formed
from whatever binaries share an underlying and a window, and a chain relation
only when a binary is listed at exactly the strike the bound needs, so a
contract listed tomorrow is arbitraged tomorrow and a contract withdrawn takes
its relations with it.

Three things are kept honest rather than assumed away:

**A rounded package is checked, not trusted.** The coefficient on a digital leg
is ``(K2 - K1) / payout``, which is 50 or 100 lots of a binary per call spread,
and integer lots cannot always reproduce a coefficient exactly. So the proposed
package is put through :func:`~arena.portfolio.netting.worst_case` with the
prices it would actually pay, and traded only when that comes back at exactly
zero. That is the same piecewise-linear minimisation the collateral engine
charges with, not an approximation of it, and it refuses a package whose hedge
ratio rounding broke.

**Quantization residue is charged for.** Legs of different scale do not cancel
exactly once each settlement is snapped to its own tick: netting.py measures one
four week win rate future against ten of its first weekly leg as riskless
before quantization and losing 1.25 after it, bounded above by
``sum |q| tick / 2``. Binaries tick at 0.01 and
calls at 0.25, so that bound is added to the cost hurdle rather than left to be
discovered.

**Half a package is a directional bet.** Legs go out together and fills are not
atomic. Anything the market does not fill in package proportion is unwound at
the next wakeup ahead of any new business, because the alternative is holding a
position nobody chose, and chasing the missing leg is a second attempt at the
execution risk that just failed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from arena.agents.arbitrageur import Relation, _underlying_key
from arena.contracts.payoff import Binary, Call, Put
from arena.exchange.types import Side
from arena.market.instrument import Instrument
from arena.portfolio.netting import worst_case
from arena.strategies.base import MarketView, SymbolView, Take, snap
from arena.strategies.taking.kelly import tradeable_touch

__all__ = ["Package", "StaticArbitrage", "derive_binary_relations"]

# How close a binary's threshold has to sit to a call's strike before the two
# are the same event. Exact in this listing, since 4600 / 10000.0 and the
# literal 0.46 round to the same double, but a strike that arrives as an
# arithmetic result rather than a literal need not, and a relation formed
# between a digital at 0.4600001 and a call at 4600 would be a bet rather than
# an arbitrage. Tight enough that only representation error passes.
STRIKE_TOLERANCE = 1e-12


def _up_indicator(payoff: Binary) -> tuple[float, float] | None:
    """``(alpha, beta)`` with ``alpha * price + beta == P(theta > threshold)``.

    A binary's price is its own probability only when it is written the way up
    and pays one. Reading the coefficients off the comparison instead means the
    families below are stated once, in probabilities, and a contract written
    ``<`` or paying 5 joins them without a second code path.
    """
    if payoff.payout == 0.0:
        return None
    if payoff.comparison in (">", ">="):
        return (1.0 / payoff.payout, 0.0)
    return (-1.0 / payoff.payout, 1.0)


def _band(
    name: str,
    target: str,
    terms: dict[str, float],
    constant: float,
) -> Relation | None:
    """``sum(w price) + constant >= 0``, as a :class:`Relation` on ``target``.

    Every inequality here arrives as a weighted sum over prices that must stay
    non-negative, and `Relation` wants one named target with an implicit
    coefficient of one. Dividing through by the target's own weight is that
    change of variable, and dividing by a negative weight reverses the
    inequality, which is why the band comes out one-sided in either direction
    rather than always upward.
    """
    weight = terms.get(target, 0.0)
    if weight == 0.0:
        return None
    legs = tuple(
        (symbol, -other / weight)
        for symbol, other in sorted(terms.items())
        if symbol != target
    )
    return Relation(
        name=name,
        target=target,
        legs=legs,
        constant=-constant / weight,
        lower=0.0 if weight > 0 else float("-inf"),
        upper=float("inf") if weight > 0 else 0.0,
    )


def derive_binary_relations(instruments: dict[str, Instrument]) -> list[Relation]:
    """Read the binary families out of the listed contracts.

    Only families the existing arbitrageur does not already enforce. Its
    vertical and butterfly bands are the option chain's own consistency
    conditions and they are met, so duplicating them would be two agents
    fighting over the same trade.
    """
    relations: list[Relation] = []

    # Grouped by underlying *and window*, which is what `_underlying_key`
    # carries. Two contracts that differ only in the week they measure are
    # different events, and a ladder built across them would be a term-structure
    # bet wearing an identity.
    ladders: dict[str, list[tuple[float, str, Binary]]] = {}
    for symbol, instrument in sorted(instruments.items()):
        payoff = instrument.spec.payoff
        if not isinstance(payoff, Binary) or instrument.spec.distribution is not None:
            continue
        ladders.setdefault(_underlying_key(instrument), []).append(
            (payoff.threshold, symbol, payoff)
        )

    # -- the ladder: a survival function is decreasing in its threshold -------
    #
    # Adjacent rungs only, exactly as the vertical band is formed adjacently:
    # monotonicity across a gap follows from monotonicity across each step in
    # it, so the wider pairs are implied and enforcing them separately would be
    # the same trade counted twice.
    for rungs in ladders.values():
        rungs.sort()
        for (low, low_symbol, low_payoff), (high, high_symbol, high_payoff) in zip(
            rungs, rungs[1:]
        ):
            if low == high:
                continue
            first = _up_indicator(low_payoff)
            second = _up_indicator(high_payoff)
            if first is None or second is None:
                continue
            relation = _band(
                f"ladder:{low_symbol}/{high_symbol}",
                low_symbol,
                {low_symbol: first[0], high_symbol: -second[0]},
                first[1] - second[1],
            )
            if relation is not None:
                relations.append(relation)

    # -- the chain: a vertical spread sandwiched by digitals at its strikes ---
    #
    # Formed for every pair of strikes rather than adjacent ones. Across a full
    # ladder the wide pairs would be implied, and the digitals here are far too
    # sparse for that: `EMBER_OBJECTIVE_WR` is the only underlying carrying both
    # families, and it lists calls at 4,800 and 5,000, puts at 5,000 and 5,200,
    # and one digital struck at 5,000. A pair is bounded only where a digital
    # happens to sit at the strike the bound reads, so narrowing the family to
    # adjacent pairs would give up bounds nothing else on the board states.
    chains: dict[tuple[str, float, str], list[tuple[float, str]]] = {}
    digitals: dict[str, list[tuple[float, str, Binary]]] = {}
    for symbol, instrument in sorted(instruments.items()):
        payoff = instrument.spec.payoff
        if instrument.spec.distribution is not None:
            continue
        key = _underlying_key(instrument)
        if isinstance(payoff, Call):
            chains.setdefault((key, payoff.scale, "call"), []).append(
                (payoff.strike, symbol)
            )
        elif isinstance(payoff, Put):
            chains.setdefault((key, payoff.scale, "put"), []).append(
                (payoff.strike, symbol)
            )
        elif isinstance(payoff, Binary):
            digitals.setdefault(key, []).append((payoff.threshold, symbol, payoff))

    for (key, scale, kind), rungs in sorted(chains.items()):
        if not scale:
            continue
        rungs.sort()
        for index, (low_strike, low_symbol) in enumerate(rungs):
            for high_strike, high_symbol in rungs[index + 1 :]:
                width = high_strike - low_strike
                if width <= 0:
                    continue
                # A call spread is worth more the further the level runs past
                # the low strike, and it is bounded below by the digital at the
                # high strike and above by the digital at the low one. A put
                # spread is the same statement about the level running the other
                # way, so it reads the down-digital and the two strikes swap
                # roles: its floor is the digital at the *low* strike. Getting
                # that the wrong way round is not a near miss, it inverts the
                # relation, and it did: the put half of this family came back
                # with a pathwise minimum of -200 on a band asserting zero,
                # 14 of 31 relations wrong, before the grid check caught it.
                picks = (
                    (("floor", high_strike), ("cap", low_strike))
                    if kind == "call"
                    else (("floor", low_strike), ("cap", high_strike))
                )
                for edge, strike in picks:
                    found = _digital_at(digitals.get(key, ()), strike / scale)
                    if found is None:
                        continue
                    digital_symbol, digital_payoff = found
                    indicator = _up_indicator(digital_payoff)
                    if indicator is None:
                        continue
                    alpha, beta = indicator
                    if kind == "put":
                        # P(theta <= t) = 1 - P(theta > t). The threshold is
                        # crossed with probability zero under any continuous
                        # posterior, so the two comparisons price the same.
                        alpha, beta = -alpha, 1.0 - beta
                    # A put spread is dear where a call spread is cheap, so its
                    # own high strike takes the role the low strike plays for a
                    # call.
                    dear = low_symbol if kind == "call" else high_symbol
                    cheap = high_symbol if kind == "call" else low_symbol
                    if edge == "floor":
                        terms = {dear: 1.0, cheap: -1.0, digital_symbol: -width * alpha}
                        constant = -width * beta
                    else:
                        terms = {dear: -1.0, cheap: 1.0, digital_symbol: width * alpha}
                        constant = width * beta
                    if digital_symbol in (dear, cheap):
                        continue
                    relation = _band(
                        f"chain-{edge}:{dear}/{cheap}/{digital_symbol}",
                        dear,
                        terms,
                        constant,
                    )
                    if relation is not None:
                        relations.append(relation)

    return relations


def _digital_at(
    digitals: Sequence[tuple[float, str, Binary]], threshold: float
) -> tuple[str, Binary] | None:
    """The binary that fires at exactly this level, if one is listed."""
    for candidate, symbol, payoff in digitals:
        if math.isclose(candidate, threshold, rel_tol=STRIKE_TOLERANCE, abs_tol=1e-15):
            return symbol, payoff
    return None


@dataclass(frozen=True, slots=True)
class Package:
    """One unit of a relation's replicating trade, in whole lots.

    Signed lots per unit, so ``units`` scales it without ever producing a
    fractional leg. That is what makes the fill accounting exact: a package of
    ``units`` is complete when every leg has moved a whole multiple of its own
    per-unit size, and whatever is left over is residue to be unwound.
    """

    relation: str
    legs: tuple[tuple[str, int], ...]
    # The limit each leg carries, already on its own grid. The touch plus
    # whatever share of the mispricing that leg was allowed to pay away, and the
    # prices the riskless check was run at, so what was verified and what is
    # sent are the same numbers.
    limits: tuple[tuple[str, Decimal], ...]
    units: int
    # The mispricing at mid, per unit, and the round trip paid to reach it.
    gap: float
    cost: float


class StaticArbitrage:
    """Trades the binary relations that must hold whatever the outcome.

    Sizing is by capital and not by Kelly, because the log-optimal stake on a
    bet that cannot lose is everything there is, and what remains to decide is
    only how much of the balance sheet one package may consume.

    Capacity is neither capital nor conviction. It is how many lots the market
    will give up at once, and this strategy cannot see depth:
    :class:`~arena.strategies.base.SymbolView` carries the touch and not the
    size resting on it, deliberately. So capacity is learned from its own fills
    the way a desk learns it, and the two things it learns answer different
    questions. How much to work at once starts at one unit and doubles after a
    package completes whole. Whether a relation is executable at all is the most
    lots a symbol has ever supplied at once, against the smallest package the
    algebra admits.

    Measured on seed 7, the second is what separates the two families. A ladder
    package is one lot against one lot and is executable. A chain package needs
    50 binary lots against one option spread, and the binary touch carries a
    median of 30 at every sampled moment from t=60s to t=300s, so no capacity
    schedule reaches it. Sized straight to the capital cap instead, with nothing
    learned from its own fills, the strategy attempted 250 packages over 300
    seconds, completed 0 of them, and unwound 10,333 lots to get back to flat.

    Staleness is the other half, and it is not a cost problem. The instants a
    ladder looks most dislocated are the instants it is being repriced, and a
    view that lags the venue reports the dislocation after the market has
    already answered it. So a relation has to be outside its band on
    ``confirmations`` consecutive wakeups, in the same direction, before
    anything is sent.
    """

    # A relation whose smallest package has failed this many times is not thin
    # at this instant, and retrying it forever is a standing order to pay
    # spreads for nothing. Three attempts, with two and then four wakeups of
    # standdown between them, spans three seconds at a 500ms cadence, which is
    # long enough for a book that was momentarily thin to refill.
    #
    # This is the answer to a book that emptied, not to one that is too small
    # for the identity. `_filled` answers that in one attempt instead of three,
    # because it is a measurement rather than a schedule.
    MAX_FAILURES = 3

    def __init__(
        self,
        instruments: dict[str, Instrument],
        *,
        edge_multiple: float = 1.5,
        capital_fraction: float = 0.25,
        max_units: int = 12,
        position_limit: int = 600,
        confirmations: int = 2,
        fee_bps: float = 0.0,
    ) -> None:
        self.relations = derive_binary_relations(instruments)
        # The mispricing must clear the round trip on every leg by this multiple
        # before anything is sent. Below it lies the no-arbitrage band, where a
        # violation persists because closing it costs more than it pays, which
        # is what a real market has and what a market without one looks like:
        # the existing arbitrageur firing on 93% of its wakeups and taking a
        # book from 207 resting lots to 26.
        self.edge_multiple = edge_multiple
        # The share of free capital one package may consume, so that correcting
        # one violation does not leave the strategy unable to correct the next.
        # A quarter leaves room for four at once, against a measured maximum of
        # two of the five ladder relations violated at the same instant, seed 7
        # at t=180s.
        self.capital_fraction = capital_fraction
        # The ceiling learned capacity may reach. Twelve lots on a leg is 40% of
        # the 30 that sits at a binary's touch here, and a package that is most
        # of the book is one whose own unwind moves the price against it.
        self.max_units = max_units
        # Not a risk limit, since a package carries no risk at settlement. A
        # concentration limit, and set to one full-size package on the heaviest
        # leg any relation in this listing asks for: twelve units of a chain
        # relation whose digital coefficient is 50 is 600 lots. So it bites on
        # accumulating packages rather than on entering one.
        self.position_limit = position_limit
        # How many consecutive wakeups a relation must be outside its band, in
        # the same direction, before a package is sent.
        #
        # This is the answer to staleness rather than to cost. The view lags
        # the venue by this strategy's latency, and the moments a ladder looks
        # most violated are exactly the moments it is being repriced, so a
        # single observation is as likely to be a stale picture as an
        # arbitrage. Measured on seed 7 without it: the strategy sent a package
        # to buy one rung of a binary ladder at 0.06 against a sale of the rung
        # above it at 0.88, and by the next wakeup both books were 0.94 bid at
        # 1.00. The sale filled, the purchase could not, and the position it
        # was left holding was the one the market had just moved against. Two
        # observations is the smallest number that can tell a level from a
        # transient and costs half a wakeup.
        self.confirmations = max(1, confirmations)
        self.fee_bps = fee_bps

        # What this strategy believes it holds in complete packages. Every
        # difference between this and the real position is unintended exposure,
        # which is the whole of the legging check and needs no separate record
        # of what was sent when.
        self._held: dict[str, int] = {}
        self._pending: Package | None = None
        self._capacity: dict[str, int] = {}
        self._standdown: dict[str, int] = {}
        self._failures: dict[str, int] = {}
        # The most lots this strategy has ever been given at once in a symbol.
        # The only evidence it has about depth, because the view carries the
        # touch and not the size resting on it, and enough to answer the one
        # question that matters: whether the smallest package a relation admits
        # is larger than the book has ever supplied.
        self._filled: dict[str, int] = {}
        # Consecutive wakeups each relation has been outside its band, and which
        # way. A change of direction restarts the count, because a relation that
        # is rich now and cheap a moment later is not one dislocation seen twice.
        self._streak: dict[str, tuple[int, int]] = {}

        self.attempts = 0
        self.captured = 0
        # Completed packages per relation, so a report can say which of the two
        # families the strategy actually corrected rather than only how many
        # packages it finished.
        self.captured_by: dict[str, int] = {}
        self.starved = 0
        self.refused = 0
        self.legged = 0
        self.legged_lots = 0
        self.retired: set[str] = set()
        # Relations seen outside their band at least once, and the number of
        # relation-wakeup pairs that were. The first answers "how many
        # violations existed", the second says how persistent they were.
        self.violated: set[str] = set()
        self.sightings = 0
        # Per relation, so a report can say which ones were dislocated and how
        # persistently rather than only how many wakeups saw something.
        self.sightings_by: dict[str, int] = {}
        # Mispricing captured at mid on packages that completed, and the round
        # trip paid to capture it.
        self.theoretical = 0.0
        self.paid = 0.0

    # -- what the book says ------------------------------------------------

    @staticmethod
    def _quote(view_of: SymbolView) -> tuple[float, float, Decimal, Decimal] | None:
        """Mid, half-spread and both sides of a touch worth pricing against.

        Read through :func:`tradeable_touch` rather than off
        ``SymbolView.mid``, because during a call phase the top of book is a
        pair of sentinels and the spread it implies is minus 2.3e18. Every cost
        hurdle here is built from a half-spread, so a negative one does not make
        this strategy cautious, it makes every relation look like an arbitrage
        worth 10^18: measured on seed 7, ten packages fired inside two minutes
        against prices no contract could have quoted.
        """
        quoted = tradeable_touch(view_of)
        if quoted is None:
            return None
        bid, ask = quoted
        return float(bid + ask) / 2.0, float(ask - bid) / 2.0, bid, ask

    # -- the loop ----------------------------------------------------------

    def orders(self, view: MarketView) -> Sequence[Take]:
        """Unwind anything half-legged, then open at most one new package.

        One package in flight at a time, and that is what makes the fill
        accounting exact rather than inferred. With two packages sharing a leg
        there is no way to tell from a position which of them filled, so "am I
        half-legged" stops having an answer and the strategy would be deciding
        whether it holds a directional bet by guesswork.
        """
        for name, remaining in list(self._standdown.items()):
            if remaining <= 1:
                self._standdown.pop(name, None)
            else:
                self._standdown[name] = remaining - 1
        if self._pending is not None:
            self._reconcile(view)
        flatten = self._flatten(view)
        if flatten:
            return flatten
        package = self._propose(view)
        if package is None:
            return ()
        return self._send(view, package)

    def _reconcile(self, view: MarketView) -> None:
        """Book whole packages that filled, stand down the ones that did not.

        Completeness is integer arithmetic on the position rather than an
        inference from fills: a package of ``units`` is done when every leg has
        moved a whole multiple of its own per-unit size, and the number of
        complete units is the smallest of those multiples.
        """
        pending = self._pending
        self._pending = None
        if pending is None:
            return
        done = pending.units
        moved: dict[str, int] = {}
        for symbol, per_unit in pending.legs:
            view_of = view.get(symbol)
            if view_of is None or per_unit == 0:
                # Nothing readable about this leg, so no package is complete.
                # Carried on through the rest rather than broken out of: leaving
                # the loop here would skip the record-keeping below for every
                # leg after it, which is the same hole this comment describes.
                done = 0
                continue
            # Every leg enters the record whether or not it filled, and that is
            # not bookkeeping tidiness. `_flatten` reads the record, so a leg
            # left out of it is a position nothing ever looks at. Recorded only
            # on completion, a package that completed nothing put none of its
            # legs in, and the fills it did get were invisible to the unwind:
            # measured, the strategy ended a 300 second run holding 839 lots it
            # had no view on, with the collateral posted against them starving
            # every later relation for budget.
            self._held.setdefault(symbol, 0)
            moved[symbol] = view_of.position - self._held[symbol]
            done = min(done, max(0, moved[symbol] // per_unit))
            if moved[symbol]:
                self._filled[symbol] = max(
                    self._filled.get(symbol, 0), abs(moved[symbol])
                )
        if done:
            self.captured += 1
            self.captured_by[pending.relation] = (
                self.captured_by.get(pending.relation, 0) + done
            )
            self.theoretical += done * abs(pending.gap)
            self.paid += done * pending.cost
            for symbol, per_unit in pending.legs:
                self._held[symbol] = self._held.get(symbol, 0) + done * per_unit
        # Counted once here, against what the package asked for, rather than in
        # `_flatten`: an unwind that does not fill is retried, and counting it
        # there charges the same stub again on every wakeup it survives.
        for symbol, per_unit in pending.legs:
            self.legged_lots += abs(moved.get(symbol, 0) - done * per_unit)
        name = pending.relation
        if done == pending.units:
            self._failures.pop(name, None)
            self._capacity[name] = min(self.max_units, max(1, done) * 2)
            return
        self.legged += 1
        self._capacity[name] = 1
        failures = self._failures.get(name, 0) + 1
        self._failures[name] = failures
        if failures >= self.MAX_FAILURES:
            self.retired.add(name)
        else:
            # Doubling, because what is being waited on is not a queue that
            # clears on a timer. It is whether the book has grown to the size
            # the identity needs, and asking again immediately is asking the
            # same question at the same price.
            self._standdown[name] = 2**failures

    def _flatten(self, view: MarketView) -> list[Take]:
        """Reverse every lot not accounted for by a complete package.

        Immediately, and ahead of any new business. A relation entered on three
        legs and filled on two is a directional position in the two, chosen by
        the order book rather than by this strategy, and the cheapest moment to
        be out of it is the first one.

        Only ever toward zero. A flatten that would *open* a position is not a
        flatten, and the case that produces one is real rather than theoretical:
        a contract that settles takes the position to zero at the venue while
        this record still says a package is held, and a strategy that trusted
        its own record over the venue's would answer settlement by
        re-establishing the trade.
        """
        out: list[Take] = []
        for symbol in sorted(self._held):
            view_of = view.get(symbol)
            if view_of is None:
                continue
            position = view_of.position
            residue = position - self._held[symbol]
            if residue == 0:
                continue
            if position == 0 or (residue > 0) != (position > 0):
                # Nothing left to reduce, or the record disagrees with reality
                # about the sign. Reality is the authority on what is held.
                self._held[symbol] = position
                continue
            lots = min(abs(residue), abs(position))
            side = Side.SELL if residue > 0 else Side.BUY
            # Marketable, with no limit. Being out of a position nobody chose is
            # worth more than the tick a limit might save, and a limit at a touch
            # that has already moved is an order that does not fill and a stub
            # carried into the next wakeup.
            out.append(Take(symbol, side, lots, None))
        return out

    def _propose(self, view: MarketView) -> Package | None:
        """The best return on collateral among the relations worth correcting.

        Ranked by what a package pays per unit of balance sheet it ties up, not
        by how far outside its band the relation sits. The two disagree, and
        badly: a chain package on one competitor's win rate is worth about 90
        in a mispricing and costs 10,170 in collateral to hold, while a ladder
        package on the same subject is worth 0.2 and costs 0.71, so absolute
        mispricing ranks the
        chain first at 0.9% of capital against the ladder's 28%. This strategy
        can carry one package at a time, so the ordering decides what it spends
        the session doing.
        """
        best: Package | None = None
        best_return = 0.0
        for relation in self.relations:
            if relation.name in self.retired or relation.name in self._standdown:
                continue
            found = self._evaluate(view, relation)
            if found is None:
                continue
            package, on_capital = found
            if on_capital > best_return:
                best, best_return = package, on_capital
        return best

    def _evaluate(
        self, view: MarketView, relation: Relation
    ) -> tuple[Package, float] | None:
        """Price one relation, and build the package that would correct it."""
        target = view.get(relation.target)
        if target is None:
            return None
        quoted = self._quote(target)
        if quoted is None:
            return None
        target_mid, target_half, _bid, _ask = quoted

        legs: list[tuple[str, int]] = []
        prices: dict[str, float] = {relation.target: target_mid}
        touches: dict[str, tuple[Decimal, Decimal]] = {
            relation.target: (quoted[2], quoted[3])
        }
        cost = target_half
        quantization = float(target.instrument.tick_size) / 2.0
        for symbol, coefficient in relation.legs:
            view_of = view.get(symbol)
            if view_of is None:
                return None
            leg_quote = self._quote(view_of)
            if leg_quote is None:
                return None
            mid, half, leg_bid, leg_ask = leg_quote
            prices[symbol] = mid
            touches[symbol] = (leg_bid, leg_ask)
            lots = max(1, round(abs(coefficient)))
            legs.append((symbol, lots if coefficient > 0 else -lots))
            cost += lots * half
            quantization += lots * float(view_of.instrument.tick_size) / 2.0

        gap = relation.excess(target_mid, prices)
        if gap == 0.0:
            self._streak.pop(relation.name, None)
            return None
        self.sightings += 1
        self.sightings_by[relation.name] = self.sightings_by.get(relation.name, 0) + 1
        self.violated.add(relation.name)

        fee = self.fee_bps / 10_000.0
        # Charged on notional going in and again coming out, on every leg, so
        # fees are part of the round trip rather than a rounding of it.
        fees = 2.0 * fee * (
            abs(target_mid) + sum(abs(lots) * abs(prices[s]) for s, lots in legs)
        )
        hurdle = self.edge_multiple * cost + quantization + fees
        slack = abs(gap) - hurdle
        if slack <= 0.0:
            self._streak.pop(relation.name, None)
            return None

        # Long the target when it is too cheap, and a negative coefficient flips
        # its leg again. The same convention the existing arbitrageur uses, so a
        # package here reads the same way a package there does.
        units_sign = -1 if gap > 0 else 1
        direction, seen = self._streak.get(relation.name, (units_sign, 0))
        seen = seen + 1 if direction == units_sign else 1
        self._streak[relation.name] = (units_sign, seen)
        if seen < self.confirmations:
            return None
        signed: list[tuple[str, int]] = [(relation.target, units_sign)]
        for symbol, lots in legs:
            signed.append((symbol, -units_sign * lots))

        # Some of the slack is spent on the legs rather than kept.
        #
        # Limits at the touch fill only what is resting there right now, against
        # a view that is already stale by this strategy's latency, and in a book
        # this thin that is mostly a miss on one leg and a fill on the other.
        # Measured on seed 7 with limits at the touch: 7 packages attempted over
        # 300 seconds, 0 completed, every one of them unwound. So each leg is
        # given part of the mispricing to pay away, split evenly per lot, and
        # what makes that safe rather than merely optimistic is that the check
        # below is run at the padded prices. A package that stops being free
        # once the padding is added is not sent.
        total_lots = sum(abs(quantity) for _symbol, quantity in signed)
        entry: dict[str, Decimal] = {}
        holdings: list = []
        for share in (1.0, 0.5, 0.25, 0.0):
            concession = Decimal(str(slack * share / max(1, total_lots)))
            entry, holdings = {}, []
            for symbol, quantity in signed:
                view_of = view[symbol]
                bid, ask = touches[symbol]
                side = Side.BUY if quantity > 0 else Side.SELL
                touch = ask + concession if quantity > 0 else bid - concession
                # Snapped here rather than at the order, because the price this
                # is checked at has to be the price that is sent. `snap` rounds
                # a bid down and an offer up, so it can only ever give back part
                # of the concession, never add to it.
                limit = snap(view_of.instrument, side, touch)
                entry[symbol] = limit
                holdings.append((view_of.instrument.spec, quantity, limit))
            # The package as it would actually be traded, at the prices it would
            # actually pay, put through the same exact minimisation the
            # collateral engine charges with. Zero means it cannot lose a cent at
            # any level the metric can take. Anything else means either the
            # padding went too far or integer rounding of a coefficient left a
            # directional stub in it, and neither is the trade this strategy
            # claims to be doing.
            if worst_case(holdings) == 0:
                break
        else:
            self.refused += 1
            return None

        per_unit_collateral = 0.0
        for symbol, quantity in signed:
            per_unit_collateral += float(
                view[symbol].instrument.collateral_for(quantity, entry[symbol])
            )
        if per_unit_collateral <= 0.0:
            return None

        # A relation whose *smallest* package needs more lots of a symbol than
        # that symbol has ever supplied at once is not thin today, it is
        # untradeable at the size the identity requires, and no capacity or
        # backoff can change that. Measured on seed 7: a chain relation asks for
        # 50 binary lots against one option spread and the binary touch carries
        # a median of 30, so the first attempt buys the measurement and every
        # attempt after it would only buy the same measurement again.
        #
        # Tested on the per-unit size rather than on the package, because
        # ``units`` is what capacity scales and ``per_unit`` is what the algebra
        # fixes. A symbol with no record yet is left alone: a leg that filled
        # nothing says the price moved, not that the book was empty.
        for symbol, quantity in signed:
            seen = self._filled.get(symbol)
            if seen is not None and abs(quantity) > seen:
                self.starved += 1
                self._standdown[relation.name] = 2**self.MAX_FAILURES
                return None

        budget = (
            float(view.equity) * self.capital_fraction - float(view.posted_collateral)
        )
        units = int(max(0.0, budget) / per_unit_collateral)
        units = min(units, self._capacity.get(relation.name, 1))
        for symbol, quantity in signed:
            if quantity == 0:
                continue
            room = self.position_limit - abs(view[symbol].position)
            units = min(units, room // abs(quantity))
        if units <= 0:
            self.starved += 1
            return None

        package = Package(
            relation=relation.name,
            legs=tuple(signed),
            limits=tuple(sorted(entry.items())),
            units=units,
            gap=gap,
            cost=cost,
        )
        return package, slack / per_unit_collateral

    def _send(self, view: MarketView, package: Package) -> list[Take]:
        """Emit the legs, dearest collateral first.

        The order matters because the adapter stops at the first intent the
        account cannot fund, and stopping before anything has gone out is the
        only stopping point that does not leave a half-entered relation. So the
        leg that costs the most to carry is tested first: if the balance sheet
        refuses it, nothing behind it is sent either.
        """
        self.attempts += 1
        self._pending = package
        limits = dict(package.limits)
        rows = []
        for symbol, per_unit in package.legs:
            view_of = view[symbol]
            quantity = per_unit * package.units
            side = Side.BUY if quantity > 0 else Side.SELL
            collateral = float(
                view_of.instrument.collateral_for(quantity, limits[symbol])
            )
            rows.append((collateral, symbol, side, abs(quantity), limits[symbol]))
        rows.sort(key=lambda row: (-row[0], row[1]))
        return [
            Take(symbol, side, quantity, snap(view[symbol].instrument, side, price))
            for _collateral, symbol, side, quantity, price in rows
        ]

    # -- reporting ---------------------------------------------------------

    @property
    def cost_ratio(self) -> float:
        """Round trip paid against mispricing captured, on filled packages.

        ``1 - cost_ratio`` is the share of a violation that survives the cost of
        correcting it, before any legging. The realised figure is the account's
        to report; this is what the strategy paid to reach the gap it saw.
        """
        return 0.0 if self.theoretical == 0.0 else self.paid / self.theoretical
