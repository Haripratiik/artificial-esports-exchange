"""Every contract on one match, listed from the format and priced off one law.

The exchange's existing prediction markets carry a riskless trade that has been
sitting in the book the whole time. Eight standalone ``Binary(> threshold)``
contracts are written on one win rate, and the ladder identity
``P(>0.44) >= P(>0.46) >= P(>0.47) >= P(>0.48)`` is enforced by nothing:
measured, 3 of 3 observed runs violated it, by 0.23 and 0.31 on a claim bounded
in [0, 1]. The option chain on the same underlying broke its own vertical bound
0 times out of 8 in those same runs, and the reason is not that options are
better policed. It is that the surface maker prices every strike as an
expectation under a single law, and two expectations under one measure cannot
disagree any more than ``E[max(F-4600,0)]`` and ``E[max(F-4700,0)]`` can
disagree under a single ``F``.

A match carries many more identities than a strike ladder does, so listing one
the standalone way would be listing many more free trades. A ten entrant solo
match here lists 50 contracts across four families and 47 exact relations
between them, and every one of those relations is a statement that holds
outcome by outcome rather than on average: exactly one competitor wins, exactly
five finish in the top five, a competitor credited with more than one
elimination was credited with more than none, and a competitor who wins is
above every other competitor. Policing 47 relations after the fact is not a
plan. Making them unbreakable is, and that is the whole of what this module
does.

**What a match lists, and what it deliberately does not.** The first version of
this listing wrote down every rung and every ordered pair its formats admitted,
which came to 270 contracts on a ten entrant solo and 38 on the 3v3. Measured
on the default board at 120 simulated seconds, that put 626 of the exchange's
666 listed contracts in the event class, 616 of them written on live matches,
and left the other eight asset classes holding 40 between them: a venue built
to demonstrate nine asset classes was 94% one of them, and the match board
alone was 92% of it. A real prediction venue lists a handful of markets on one
game. So the listing is now 10 + 10 + 20 + 10 = 50 on the ten entrant solo and
2 + 0 + 12 + 6 = 20 on the 3v3, and the three rules that pick them are
``_placement_rungs``, ``_elimination_rungs`` and ``_featured_pairs``, each
derived from the format so a format registered later is listed rather than
special cased. What went, and what it cost, is set out there and in
``list_match``. The short version is that the placement ladder keeps the median
cut alone, the eliminations ladder keeps the two rungs the count is actually
likely to cross, the head to heads keep a bracket of featured matchups instead
of the complete tournament of them, and **the eliminations conservation set is
gone**: a two rung ladder cannot add up to the field's nine credits, so the
board no longer quotes the conservation law, though the format still enforces
it on every result.

The same board, measured the same way afterwards: 190 contracts, 150 of them
events, 140 written on live matches. So the match board fell from 92% of the
exchange to 74%, the event class from 94% to 79%, and the other eight asset
classes hold the same 40 contracts they always did while being a quarter of the
board instead of a sixteenth. That is the number the trim was aimed at. It is
still the largest family on the exchange by a distance, which is honest: a match
is the only thing here that opens and settles while somebody is watching.

**The mechanism is one bag of outcomes.** The one distribution is a finite
ensemble of hypothetical ``MatchResult`` objects drawn under the caller's
belief about the field, and the price of a contract is the mean of *the same
settlement rule that will settle it* over that ensemble. Nothing else prices
anything. Any identity that holds on every outcome therefore holds in the
prices as arithmetic: the winner set sums to one because each drawn match has
exactly one winner, not because anything was normalised afterwards. Prices come
back as ``Fraction`` with denominator equal to the draw count, so the identities
are exact rather than exact to within a rounding. Both are measured over 50
random beliefs on each format: every one of the 47 solo relations and 22 team
relations sits at excess exactly 0 in rational arithmetic, and after converting
the prices to float the worst excess anywhere is 1.3e-15, on the cross
sectional set that adds ten contracts up to five. The two ladders and the head
to head floor, which are most of the list, are still exactly 0 in float,
because those compare two prices and a float comparison cannot invert what the
exact one said.

**Where this approximates, and why it is the honest place to do it.** The
ensemble is Monte Carlo, so a price is an exact expectation under an
approximate measure rather than an approximate expectation under an exact one.
That is the right way round for this problem, because coherence is what is
being bought and coherence is a property of the first. Closed forms do exist
for two of the families under a Plackett-Luce belief, ``w_i / W`` for the winner
set and ``w_A / (w_A + w_B)`` for a head to head, and they are deliberately not
used: mixing a closed form for one family with an ensemble for another is two
measures, which is exactly the defect above wearing better clothes. What that
costs is sampling error, measured on the default solo field: the worst winner
price sits 0.0297 from ``w_i / W`` at 1,000 draws, 0.0100 at the default 4,000
and 0.0042 at 16,000, against a quoting increment of 0.01, while the whole 50
contract book takes 0.06s, 0.24s and 0.96s to price. The sampling error is
untouched by the trim and the wall clock fell with it, which is the expected
shape: the draws are shared across the book and the settlement pass is what
scales with the listing. The eliminations family
has no closed form to be tempted by at all, because a credit goes to somebody
still alive at the moment it happens, so the count depends on the whole
finishing order rather than on the weights alone. Exact enumeration is not an
option either: the placement marginals need a walk over the 2^n prefixes of the
finishing order, and the elimination counts a further n states per competitor
on top of that, which is 100k states at ten entrants and 16M at sixteen.

One consequence of a finite ensemble is worth naming rather than leaving to be
discovered. Its support is smaller than the world's, so an outcome it never
drew prices at exactly zero. On the untrimmed listing that was most of the
eliminations ladder: at 4,000 draws over 20 beliefs, as many as 34 of the 270
solo contracts priced at exactly 0 or 1 on a single belief, and 611 of the 612
such prices across all twenty were elimination rungs, which is where the tail
is. Dropping the rungs above ``> 1`` dropped almost all of that with them. Same
sweep on the 50 contract listing: 0 prices at a boundary, on any of the twenty
beliefs, at 800 draws or at 4,000.

That is a real improvement and it is not a fix, so the guard stays. Widen the
beliefs from a lognormal sigma of 0.7 to 1.8, roughly a factor of a thousand
across the field, and as many as 5 of the 50 still pin on a single belief at
4,000 draws and 11 at 800. Across all twenty of those beliefs that is 29 prices
at 4,000, 25 of them elimination rungs and 4 placement, and 57 at 800, which
reach the winner and head to head families too. Those prices are coherent, they
are simply at the boundary, and a maker that published a zero as its offer
would be selling a lottery ticket for nothing. That is the other thing
``two_sided_quote`` is for: a zero fair value with a one tick half spread quotes
0 bid against a 0.01 offer, because the offer is a ceiling and never a floor.
Wanting a finer answer on a rare rung is a reason to raise the draw count, not
to reach for a second measure.

**The model of the world is the format's own rules.** The sampler mirrors
``match.play``: a weighted sample without replacement for an elimination
format, a race to the target with each point credited inside the scoring side
for a team format. It is not a fitted approximation of the world, it is the
world's mechanics driven by the caller's weights instead of by the latent
strengths nobody outside ``roster.py`` may read, which is precisely what a
market maker has. Every drawn outcome is put through ``fmt.check`` exactly as a
real match is, so a model that drifted from the format would raise rather than
quietly price a match that cannot happen. And the mirror is tested as an
identity rather than as a resemblance: handed ``exp(strength)`` as its belief
and ``match.play``'s own random stream, ``draw_outcome`` returns the match the
world actually played, placement for placement and credit for credit, on 400
real matches in each of the two formats and 0 mismatches.

Nothing here carries a list of formats or a list of contracts. The families,
the ladder rungs, the sides, the exclusive sets and every relation are read off
``modes.FORMATS`` and the field, so registering a format lists its market with
no edit to this file. What is dispatched on is the mechanic, ``team_size == 1``
against a team race, and a format with a genuinely new mechanic raises here
rather than being priced by whichever branch it fell into.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from fractions import Fraction
from random import Random

from arena.agents.arbitrageur import Relation
from arena.contracts.payoff import Binary
from arena.contracts.spec import ContractSpec, DataPolicy, ObservationWindow
from arena.contracts.underlying import Difference, MetricRef, Single, Underlying
from arena.determinism import digest
from arena.market.instrument import Instrument
from arena.worlds.circuit.match import MatchResult
from arena.worlds.circuit.modes import FORMATS, MatchFormat
from arena.worlds.circuit.roster import roster_for

__all__ = [
    "METRIC_WIN",
    "METRIC_PLACE",
    "METRIC_ELIMINATIONS",
    "WINNER",
    "PLACEMENT",
    "ELIMINATIONS",
    "HEAD_TO_HEAD",
    "FAMILIES",
    "SIDE_SEPARATOR",
    "MatchContract",
    "ExclusiveSet",
    "MatchBook",
    "match_window",
    "list_match",
    "metric_levels",
    "settlement",
    "derive_match_relations",
    "draw_outcome",
    "outcome_ensemble",
    "coherent_prices",
    "excess_exact",
    "two_sided_quote",
]

UTC = timezone.utc

# The three quantities an oracle has to be able to read off a finished match for
# any of this to settle. Named here rather than in the world, because they are
# what the *contracts* are written on: a world that wants to list match markets
# resolves these three and nothing else. Every contract below is a Binary on one
# of them, or on a Difference of two of them.
METRIC_WIN = "match_win"
METRIC_PLACE = "match_place"
METRIC_ELIMINATIONS = "match_eliminations"

# A side is named by its members, so a name has to be splittable back into
# them. Nothing in the roster carries this character today and the listing
# refuses a field that does, rather than letting a subject silently round trip
# into two competitors that do not exist.
SIDE_SEPARATOR = "-"

WINNER = "winner"
PLACEMENT = "placement"
ELIMINATIONS = "eliminations"
HEAD_TO_HEAD = "head_to_head"
FAMILIES = (WINNER, PLACEMENT, ELIMINATIONS, HEAD_TO_HEAD)

# A match is minutes long, so its contracts observe minutes rather than the four
# week window a rate contract needs. The schedule also does identity work: the
# oracle resolves a MetricRef over a window, so two matches on the same
# competitor in the same format are only different questions if their windows
# differ. They are pinned in the ref as well, and both are checked at
# settlement, because a contract that can be settled against the wrong match is
# the "identity reused after it stopped meaning anything" failure with money
# attached.
EPOCH = datetime(2026, 9, 1, tzinfo=UTC)
DURATION = timedelta(minutes=5)

# Payout is 1.0 on every contract here, so a price is a probability on a grid of
# hundredths. Same increment the existing event contracts use.
TICK = "0.01"

# 4,000 draws. Chosen by measurement rather than by taste: the largest gap
# between an ensemble winner price and the closed form w_i/W was 0.0297 at 1,000
# draws, 0.0100 at 4,000 and 0.0042 at 16,000, against a quoting increment of
# 0.01, while the 50 contract solo book takes 0.06s, 0.24s and 0.96s to price.
# So 4,000 is the first count at which the sampling noise is down to the tick a
# price is quoted on, and the next step buys 2.4 times the accuracy for 4 times
# the wall clock. The error falls as 1/sqrt(draws) and the cost rises linearly,
# which is the whole trade, and a caller who needs a finer fair value passes a
# larger number rather than being given one nobody priced the cost of. The trim
# from 270 contracts to 50 moved none of the accuracy figures, because the draws
# are the same draws; it moved only the three wall clocks, and by less than five
# times, since drawing the ensemble is not part of what got shorter.
DEFAULT_DRAWS = 4_000


@dataclass(frozen=True, slots=True)
class MatchContract:
    """One tradeable claim on a match, with the tags its family is derived from.

    The instrument is the real thing, spec and digest and all, so these list on
    the venue and settle through ``settlement.engine`` like anything else. The
    tags are carried alongside rather than parsed back out of the symbol,
    because a symbol is a label and a relation derived by string surgery on
    labels is a relation that breaks the day somebody renames a contract.
    """

    instrument: Instrument
    family: str
    subject: str
    # The other competitor in a head to head, empty everywhere else.
    versus: str = ""
    # N for a top N contract, k for an "eliminations > k" rung, 0 for a winner.
    threshold: int = 0

    @property
    def symbol(self) -> str:
        return self.instrument.symbol


@dataclass(frozen=True, slots=True)
class ExclusiveSet:
    """Contracts whose settlement values add to a known constant on every match.

    The mutually exclusive and exhaustive winner set is the ``total == 1`` case
    and the one worth naming, because a set of claims that sum to one is what
    everybody means by a probability and what nothing in the old prediction
    listing enforced. The same object says one other true thing about a match
    without changing shape: exactly ``N * team_size`` competitors finish in the
    top N, so a top N rung read across the whole field sums to that.

    **Exhaustive is not decoration, it is the precondition.** Both cases above
    are sums over the *entire* field at one rung, and the moment a member is
    missing the total is a number the remaining contracts cannot produce and the
    relation handed to the arbitrageur becomes a false statement it will trade
    on. So a rung is listed for every competitor or not at all, and
    ``_placement_rungs`` chooses which rungs exist rather than which competitors
    get them.

    There used to be a third case and it is worth saying plainly why it is gone
    rather than leaving a reader to notice. A format that pins its eliminations
    pins the sum of the *whole* eliminations ladder across the field, since
    ``E[X] = sum_k P(X > k)`` for a count, but only when the rungs are every
    integer threshold the count can cross. ``_elimination_rungs`` now lists two
    of them, so that sum is no longer the field's nine credits and no
    ``ExclusiveSet`` claims it is. The law itself did not change: ``modes.py``
    still refuses a result whose eliminations do not close, and
    ``MatchBook.elimination_total`` still records the constant. The board simply
    has no package that trades it.

    Kept as an object rather than as a comment next to a loop for one reason:
    the thing that went wrong before was that the identity existed only in
    somebody's head. Here it is a value that can be listed, asked whether it
    holds, and handed to the arbitrageur as a Relation.
    """

    name: str
    symbols: tuple[str, ...]
    total: float = 1.0

    def relation(self) -> Relation:
        """The identity in the vocabulary the arbitrageur already trades.

        One member is the target and the rest are legs at -1, so the relation's
        excess is ``sum(prices) - total`` with a zero width band. That is the
        same shape ``derive_relations`` gives an index, which matters: it means
        this needs no new execution path, only a longer list.
        """
        head, *rest = self.symbols
        return Relation(
            name=f"exclusive:{self.name}",
            target=head,
            legs=tuple((symbol, -1.0) for symbol in rest),
            constant=self.total,
        )

    def shortfall(self, values: Mapping[str, float]) -> float:
        """How far a set of settlement values or prices is from the total."""
        return math.fsum(values[symbol] for symbol in self.symbols) - self.total


@dataclass(frozen=True, slots=True)
class MatchBook:
    """The whole listing for one match, and the facts it was derived from."""

    seed: int
    match_id: int
    format_name: str
    # Draw order, which is also the grouping order: a team format takes its
    # sides from consecutive runs of this, exactly as ``match.play`` does.
    field: tuple[str, ...]
    sides: tuple[tuple[str, ...], ...]
    window: ObservationWindow
    contracts: tuple[MatchContract, ...]
    exclusive_sets: tuple[ExclusiveSet, ...]
    # Every metric the book needs resolved, precomputed because pricing walks it
    # once per drawn outcome and rebuilding it there cost more than the payoffs.
    atoms: tuple[MetricRef, ...]
    # The distinct placement values the format can produce, 10 for a ten entrant
    # elimination and 2 for a team race, and the most eliminations one
    # competitor can be credited.
    places: int
    elimination_cap: int
    # What the field's eliminations must add to, where the format fixes it.
    # None for a race, whose length is not fixed: a race to 3 ends after
    # anything from 3 to 5 points, so there is no constant to write down and no
    # relation is formed rather than one being approximated.
    #
    # A fact about the format, not a claim about the listing. Since the
    # eliminations ladder was cut to two rungs no ``ExclusiveSet`` is built from
    # this, and it is kept because callers that reason about the world want it:
    # the tests draw arbitrary settleable results from it, and it is what
    # ``modes.py`` checks every result against.
    elimination_total: int | None

    @property
    def format(self) -> MatchFormat:
        return FORMATS[self.format_name]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(contract.symbol for contract in self.contracts)

    def by_symbol(self) -> dict[str, MatchContract]:
        return {contract.symbol: contract for contract in self.contracts}

    def instruments(self) -> dict[str, Instrument]:
        """What the venue lists, keyed the way ``Venue`` and the agents want it."""
        return {contract.symbol: contract.instrument for contract in self.contracts}

    def of_family(self, family: str) -> tuple[MatchContract, ...]:
        if family not in FAMILIES:
            raise KeyError(f"{family!r} is not one of {FAMILIES}")
        return tuple(c for c in self.contracts if c.family == family)

    def side_of(self, competitor: str) -> tuple[str, ...]:
        for side in self.sides:
            if competitor in side:
                return side
        raise KeyError(f"{competitor} is not in match {self.match_id}")


# -- reading the shape of a format ----------------------------------------


@dataclass(frozen=True, slots=True)
class _Mechanic:
    """How a format turns weights into a result, and what that bounds.

    Derived from the format's own declared numbers, never from its name. The
    kind decides which sampler runs and how many placement values exist, and it
    is read from ``team_size`` because that is the only structural difference
    between the two mechanics the world has: everybody for themselves produces a
    full ranking, sides produce a winner and a loser.
    """

    kind: str
    places: int
    cap: int
    total: int | None
    target: int
    # What one competitor is credited on average, over the whole field. Read
    # off the format's own arithmetic rather than off a belief, because the
    # rungs the book lists are chosen from it and a listing that read a
    # strength would be writing the answer into the contract. It is the figure
    # ``_elimination_rungs`` picks its two thresholds around.
    credit_mean: float


def _mechanic(fmt: MatchFormat) -> _Mechanic:
    if fmt.entrants <= 1:
        raise ValueError(f"{fmt.name}: a match needs at least two entrants")
    if fmt.team_size == 1:
        # One elimination per round until one competitor is left, so the field
        # is credited exactly ``entrants - 1`` of them and a single competitor
        # can in principle collect every one: they need only survive, which the
        # winner does by definition.
        return _Mechanic(
            kind="elimination",
            places=fmt.entrants,
            cap=fmt.entrants - 1,
            total=fmt.entrants - 1,
            target=0,
            # Exactly ``entrants - 1`` credits are handed out to ``entrants``
            # competitors, so the mean is below one for every field size there
            # can ever be, and the two rungs that straddle it are 0 and 1.
            credit_mean=(fmt.entrants - 1) / fmt.entrants,
        )

    if fmt.entrants % fmt.team_size:
        raise ValueError(
            f"{fmt.name}: {fmt.entrants} entrants do not divide into sides of "
            f"{fmt.team_size}"
        )
    if fmt.teams != 2:
        # Not a limitation of the pricing, a limitation of the world: a race
        # puts everyone who did not win on the same second place, so a third
        # side would leave two sides' worth of competitors on place 2 and the
        # format's own check refuses that result. Listing a market that can
        # never settle is worse than listing none.
        raise ValueError(
            f"{fmt.name}: {fmt.teams} sides, and a race produces only a winner "
            "and a loser, so a result with three sides cannot pass check()"
        )
    target = getattr(fmt, "target", None)
    if not isinstance(target, int) or target < 1:
        # ``match.play`` reads the same attribute and falls back to 3. Adopting
        # that default here would be a pricing model quietly inventing a race
        # length the format never declared, and the two would disagree in
        # silence, which is the shape of every bug in CONTRIBUTING's list.
        raise ValueError(
            f"{fmt.name}: a team format must declare an integer target for its "
            "race; nothing may guess it on the format's behalf"
        )
    # A point is credited inside the side that scored it, so one competitor can
    # collect at most its own side's points, and the most any side scores is the
    # target that ends the race. The field's total is not fixed, because the
    # loser can end anywhere from 0 to target - 1.
    return _Mechanic(
        kind="race",
        places=2,
        cap=target,
        total=None,
        target=target,
        # A race ends the moment one side reaches the target, so the field is
        # credited at least ``target`` points and at most ``2 * target - 1``.
        # Nothing here may narrow that further without a belief, so the mean is
        # taken at the midpoint of the range the format's own arithmetic pins.
        # On the 3v3 that is 8/12 = 0.667 against 0.6813 measured over 2,000
        # played matches, which moves no rung: both floor to the same pair.
        credit_mean=(3 * target - 1) / (2 * fmt.entrants),
    )


# -- listing ---------------------------------------------------------------


def _side_subject(members: Sequence[str]) -> str:
    """A side's name is its members, sorted, so it does not depend on seating.

    For an elimination format a side is one competitor and this is that
    competitor's key, which is why the winner family needs no special case for
    the two mechanics.
    """
    return SIDE_SEPARATOR.join(sorted(members))


# -- what a match is worth listing, and what it is not --------------------
#
# The three rules below are the whole of the trim, and they exist because the
# first version listed every rung and every ordered pair a format admitted. On
# the ten entrant solo that was 270 contracts, against 50 for everything else on
# the same board put together, so the exchange was a match board with an option
# chain attached. What follows picks the rungs by what they are
# worth rather than by what is available, and each rule is derived from the
# format so a format registered later gets a listing rather than a special case.


def _placement_rungs(mechanic: _Mechanic) -> tuple[int, ...]:
    """The one placement cutoff a match lists: the median finish.

    One rung rather than eight, and this one rather than another, by
    measurement. Priced over 24 lognormal beliefs on the default solo field at
    2,000 draws, a rung's mean distance from the nearer of 0 and 1 runs 0.195 at
    the top 2, 0.271 at the top 3, 0.322 at the top 4, **0.334 at the top 5**,
    0.312, 0.259, 0.187, and 0.099 at the top 9, where a quarter of all prices
    sit within a tick of a boundary. So the ladder's information is a hump with
    its peak at the median cut, and the rungs at the ends are contracts that
    price at a near certainty most of the time.

    ``places // 2`` and not the number 5, so a four place format lists its top 2
    and a two place format lists nothing at all: there the rung at 1 is the
    winner contract under another name and the rung at 2 is a certainty, which
    is why a race listed no placement ladder before this trim either.

    Keeping exactly one rung is what keeps the cross-sectional ``ExclusiveSet``
    honest. Exactly ``N * team_size`` competitors finish in the top N, and that
    is a statement about the whole field at one rung, so a rung is listed for
    every competitor or not at all. Thinning a rung across competitors would
    leave the set claiming a sum the remaining contracts cannot produce.
    """
    cut = max(2, mechanic.places // 2)
    return (cut,) if cut <= mechanic.places - 1 else ()


def _elimination_rungs(mechanic: _Mechanic) -> tuple[int, ...]:
    """The two eliminations thresholds the count is actually likely to cross.

    Same measurement, same field, and a much steeper hump. Mean distance from
    the nearer boundary by rung: 0.284 at ``> 0``, 0.186 at ``> 1``, 0.106 at
    ``> 2``, then 0.055, 0.024, 0.009, 0.003, 0.0006 and 0.0000 at ``> 8``,
    where the share of prices sitting within a tick of a boundary runs 0.8%,
    6.7%, 27.1%, 54.2%, 79.2%, 90.0%, 96.7%, 99.6% and 100%. The top rung of a
    ten entrant ladder is a contract that has never once priced away from zero
    in this sweep. The 3v3 is the same shape over three rungs: 0.333, 0.153,
    0.034, the last of them pinned on 63.9% of prices.

    So the rungs are chosen around ``credit_mean``, which is what the format
    pins the average credit at, and two of them are kept rather than one so the
    ladder still has an adjacent pair for ``derive_match_relations`` to write a
    monotonicity relation between. One rung would list a family with no internal
    identity at all, which is the thing this module exists not to do.
    """
    # floor(mean) and one above it, pulled back far enough that both rungs exist
    # on a short ladder. A format crediting at most one elimination gets the
    # single rung at 0 and no relation, because there is no second threshold to
    # write one against.
    low = min(math.floor(mechanic.credit_mean), max(mechanic.cap - 2, 0))
    return tuple(rung for rung in (low, low + 1) if 0 <= rung < mechanic.cap)


def _featured_pairs(
    sides: Sequence[Sequence[str]],
) -> tuple[tuple[str, str], ...]:
    """The head to heads a match features: the draw's own bracket.

    Every ordered pair was 90 contracts on a ten entrant solo, a third of the
    old listing and the least defensible third of it. A real venue lists a
    handful of featured matchups, not the complete tournament of them, and the
    honest question is which handful.

    Nothing here may answer "the close ones". A listing runs before any belief
    exists and must never read a latent strength, for the reason
    ``build_market`` gives about every threshold it writes: a rung chosen from a
    settlement value is the answer written into the contract. Closeness is a
    property of a belief, so it is not available and pretending otherwise would
    be worse than an arbitrary bracket.

    What is available is the draw. Sides are taken two at a time in the order
    the roster dealt them, and inside a pairing competitors are matched slot for
    slot, so a ten entrant solo features 5 matchups and the 3v3 features 3, one
    per lane. That rule has one property worth having: with an even number of
    sides every competitor appears in exactly one head to head, so the family
    covers the field evenly instead of concentrating on whoever a listing
    thought was interesting. An odd side out wraps onto the first side, which
    gives that side a second matchup and is the price of nobody being left
    without one. Neither current format pays it: both seat an even number.

    Both directions of each matchup are listed, which is what a two way market
    is, and it is what makes the complement identity ("exactly one of them
    finishes above the other") a thing the board states rather than implies.
    """
    pairs: list[tuple[str, str]] = []
    count = len(sides)
    for index in range(0, count - 1, 2):
        left, right = sides[index], sides[index + 1]
        pairs.extend(zip(left, right))
    if count % 2 and count > 2:
        pairs.extend(zip(sides[count - 1], sides[0]))
    return tuple(pairs)


def match_window(
    match_id: int, epoch: datetime = EPOCH, duration: timedelta = DURATION
) -> ObservationWindow:
    """The minutes this match occupies. Matches run back to back from the epoch.

    This is the forward half of a mapping whose backward half lives in the
    circuit oracle's calendar, which turns a window into the match ids inside
    it. Two clocks that never meet is a failure this repository has already had,
    so the pair share ``EPOCH`` and ``DURATION`` rather than each carrying a
    cadence, and ``test_circuit_world`` asserts they still agree. Moving either
    one alone would leave a contract listed on match 400 settling against
    whatever match 400 meant to the other side.
    """
    start = epoch + duration * match_id
    return ObservationWindow(start, start + duration)


def _ref(
    metric: str,
    subject: str,
    format_name: str,
    match_id: int,
    bounds: tuple[float, float],
    kind: str,
) -> MetricRef:
    """One metric, narrowed to one match.

    The match id goes in the ref rather than being left to the window alone.
    The window would in fact separate two matches on this schedule, but that is
    a property of how the schedule happens to be laid out, and a ref that
    identifies its match only by when it was played is one calendar change away
    from resolving somebody else's result.
    """
    return MetricRef(
        metric=metric,
        subject=subject,
        modes=(format_name,),
        maps=(f"match-{match_id}",),
        bounds=bounds,
        kind=kind,
    )


def _spec(
    contract_id: str,
    underlying: Underlying,
    payoff: Binary,
    window: ObservationWindow,
    reference_id: str,
    tick_size: str,
    metadata: tuple[tuple[str, str], ...],
) -> ContractSpec:
    return ContractSpec(
        contract_id=contract_id,
        underlying=underlying,
        payoff=payoff,
        window=window,
        # A match is its own evidence. There is no averaging window to be thin,
        # so the bar is that the match happened and its result satisfied the
        # format, which ``metric_levels`` checks before it reads anything.
        policy=DataPolicy(min_sample_size=1),
        # No standardization snapshot exists to pin, because nothing here is
        # standardized. What the id pins instead is the rulebook a result is
        # checked against, so changing a format's size or shape changes the
        # digest of every contract ever written under it.
        reference_id=reference_id,
        published_at=window.start - (window.end - window.start),
        tick_size=tick_size,
        metadata=metadata,
    )


def list_match(
    seed: int,
    match_id: int,
    format_name: str = "solo",
    *,
    epoch: datetime = EPOCH,
    duration: timedelta = DURATION,
    tick_size: str = TICK,
    prefix: str | None = None,
) -> MatchBook:
    """Every contract one match carries, derived from its format and its field.

    Four families, each of which exists because it has exact identities against
    the others rather than because it sounded like a market.

    * **Winner**, one per side, mutually exclusive and exhaustive. Never
      trimmed, because this is the partition that makes the board a probability
      space and everything else is anchored to it.
    * **Top N**, one rung across the whole field, at the median finish. See
      ``_placement_rungs``.
    * **Eliminations**, two over/under rungs per competitor, at the thresholds
      the credit count is actually likely to cross. See ``_elimination_rungs``.
    * **Head to head**, both directions, on a bracket of featured matchups drawn
      from the seating rather than on every ordered pair. See
      ``_featured_pairs``.

    A ten entrant solo match lists 10 + 10 + 20 + 10 = 50 contracts and the 3v3
    lists 2 + 0 + 12 + 6 = 20. The empty family is not an oversight: with only
    two places a top 1 rung is the winner contract listed a second time under a
    different name and a top 2 rung is a certainty, so a race lists no placement
    ladder at all.

    **What this listing used to be, and what the difference costs.** It was
    10 + 80 + 90 + 90 = 270 and 2 + 0 + 18 + 18 = 38, every rung and every
    ordered pair the format admitted, and on the default board that made 92% of
    the exchange's 666 contracts match markets. Three things went:

    1. Seven of the eight placement rungs. The board can no longer be asked
       whether a competitor makes the podium, or avoids finishing last, only
       whether they finish in the top half. The rungs that went were the weak
       ones by measurement (``_placement_rungs`` has the numbers) but
       "measurably weak" is not "worthless": a podium market is a market people
       want and this board does not have one.
    2. Seven of the nine elimination rungs, and with them the ``ExclusiveSet``
       that added the field's whole ladder up to the nine credits a solo match
       conserves. That was the board's only cross-sectional statement about
       eliminations and it is genuinely lost: the conservation law still holds
       and is still checked at settlement, but no package of listed contracts
       trades it any more.
    3. Eighty of the ninety head to heads. Every pair of competitors used to be
       quotable against every other; now only the five featured matchups are,
       and a trader with a view on two competitors who did not draw into the
       same matchup has no contract to express it with.

    What survives is every *kind* of identity the module ever derived: the
    winner partition, a cross-sectional top N count, a monotone placement chain
    from the winner contract upward, a monotone elimination pair, and the head
    to head complement with its two bounds against the winner set. Measured,
    47 relations on the solo book against 395, and 22 on the 3v3 against 58.
    """
    if format_name not in FORMATS:
        raise KeyError(
            f"no format named {format_name!r} is registered; a match outside "
            f"{sorted(FORMATS)} cannot be listed, let alone settled"
        )
    fmt = FORMATS[format_name]
    mechanic = _mechanic(fmt)

    field = tuple(roster_for(format_name, fmt.entrants, seed, match_id))
    for key in field:
        if SIDE_SEPARATOR in key:
            raise ValueError(
                f"competitor {key!r} contains {SIDE_SEPARATOR!r}, which names a "
                "side's members and would split back into competitors nobody "
                "entered"
            )
    sides = tuple(
        tuple(field[i : i + fmt.team_size])
        for i in range(0, fmt.entrants, fmt.team_size)
    )
    window = match_window(match_id, epoch, duration)
    # The symbol names the match, not the world, the way a ticker does. Two
    # worlds run from different seeds would therefore hand a venue the same
    # symbol for two different fields, which is why ``prefix`` exists and why
    # the world seed is part of the reference id below: the label can repeat
    # across worlds, the content address must not.
    tag = prefix or f"{format_name.upper()}{match_id}"
    reference_id = f"circuit-{fmt.name}-{fmt.entrants}x{fmt.team_size}-seed{seed}"

    def build(
        contract_id: str,
        underlying: Underlying,
        payoff: Binary,
        family: str,
        subject: str,
        versus: str = "",
        threshold: int = 0,
    ) -> MatchContract:
        metadata = (
            ("family", family),
            ("match", str(match_id)),
            ("subject", subject),
            ("threshold", str(threshold)),
            ("versus", versus),
        )
        spec = _spec(
            contract_id,
            underlying,
            payoff,
            window,
            reference_id,
            tick_size,
            metadata,
        )
        return MatchContract(
            instrument=Instrument(contract_id, spec),
            family=family,
            subject=subject,
            versus=versus,
            threshold=threshold,
        )

    place_ref = {
        key: _ref(
            METRIC_PLACE,
            key,
            format_name,
            match_id,
            (1.0, float(mechanic.places)),
            "quantity",
        )
        for key in field
    }
    elim_ref = {
        key: _ref(
            METRIC_ELIMINATIONS,
            key,
            format_name,
            match_id,
            (0.0, float(mechanic.cap)),
            "quantity",
        )
        for key in field
    }

    contracts: list[MatchContract] = []
    exclusive: list[ExclusiveSet] = []

    # -- winner: one per side, and the set that sums to one -----------------
    winner_symbols: list[str] = []
    for side in sides:
        subject = _side_subject(side)
        symbol = f"{tag}_WIN_{subject}"
        contracts.append(
            build(
                symbol,
                Single(
                    _ref(
                        METRIC_WIN,
                        subject,
                        format_name,
                        match_id,
                        (0.0, 1.0),
                        "rate",
                    )
                ),
                Binary(">", 0.5),
                WINNER,
                subject,
            )
        )
        winner_symbols.append(symbol)
    exclusive.append(
        ExclusiveSet(name=f"{tag}_WIN", symbols=tuple(winner_symbols), total=1.0)
    )

    # -- top N: the placement cutoff, and its cross-sectional count ---------
    #
    # The rung at N is a Binary("<=", N) on the placement itself rather than a
    # new metric per rung, so a ladder is one metric read at several thresholds
    # and the monotonicity below is a property of the payoff, not a convention.
    # One rung, across the whole field, for the reasons in ``_placement_rungs``.
    for rung in _placement_rungs(mechanic):
        rung_symbols: list[str] = []
        for key in field:
            symbol = f"{tag}_TOP{rung}_{key}"
            contracts.append(
                build(
                    symbol,
                    Single(place_ref[key]),
                    Binary("<=", float(rung)),
                    PLACEMENT,
                    key,
                    threshold=rung,
                )
            )
            rung_symbols.append(symbol)
        exclusive.append(
            ExclusiveSet(
                name=f"{tag}_TOP{rung}",
                symbols=tuple(rung_symbols),
                total=float(rung * fmt.team_size),
            )
        )

    # -- eliminations: the two rungs the count is likely to cross ------------
    #
    # And no ``ExclusiveSet`` over them, which is the one guarantee this trim
    # spends. ``E[X] = sum over k >= 0 of P(X > k)`` needs *every* threshold from
    # 0 to cap - 1 on every competitor before the ladder read across the field
    # adds up to what the format conserves, so a two rung ladder cannot make
    # that claim and does not. The conservation law itself is untouched: the
    # format still checks it on every result and ``elimination_total`` still
    # records it. What the board no longer does is quote it.
    elim_rungs = _elimination_rungs(mechanic)
    for key in field:
        for threshold in elim_rungs:
            contracts.append(
                build(
                    f"{tag}_ELIM_{key}_GT{threshold}",
                    Single(elim_ref[key]),
                    Binary(">", float(threshold)),
                    ELIMINATIONS,
                    key,
                    threshold=threshold,
                )
            )

    # -- head to head: both directions, on the featured matchups only --------
    #
    # Written as a Binary on the Difference of two placements, which is the
    # contract algebra already in the box: no new metric, and the settlement
    # range falls out of interval arithmetic. Same side pairs cannot appear,
    # because the bracket only ever pairs one side against another, and that is
    # load bearing rather than tidy. In a team format two team mates always
    # place equally, so "A above B" and "B above A" would both settle at zero
    # and the pair would not complement. Listing them with an identity that is
    # false a third of the time is how a relation stops meaning anything.
    side_of = {key: side for side in sides for key in side}
    for a, b in _featured_pairs(sides):
        if a == b or b in side_of[a]:  # pragma: no cover - the bracket cannot
            raise ValueError(
                f"match {match_id}: {a} and {b} share a side, so they finish "
                "level and their head to heads would not complement"
            )
        for subject, versus in ((a, b), (b, a)):
            contracts.append(
                build(
                    f"{tag}_H2H_{subject}_OVER_{versus}",
                    Difference(Single(place_ref[subject]), Single(place_ref[versus])),
                    Binary("<", 0.0),
                    HEAD_TO_HEAD,
                    subject,
                    versus=versus,
                )
            )

    atoms: list[MetricRef] = []
    seen: set[MetricRef] = set()
    for contract in contracts:
        for atom in contract.instrument.spec.atoms():
            if atom not in seen:
                seen.add(atom)
                atoms.append(atom)

    return MatchBook(
        seed=seed,
        match_id=match_id,
        format_name=format_name,
        field=field,
        sides=sides,
        window=window,
        contracts=tuple(contracts),
        exclusive_sets=tuple(exclusive),
        atoms=tuple(atoms),
        places=mechanic.places,
        elimination_cap=mechanic.cap,
        elimination_total=mechanic.total,
    )


# -- settlement ------------------------------------------------------------


def metric_levels(book: MatchBook, result: MatchResult) -> dict[MetricRef, float]:
    """What an oracle has to return for this match, read off the result.

    This is the whole of the boundary between the world and these contracts. An
    oracle for the circuit world resolves exactly these refs and nothing else,
    and everything downstream, settlement and pricing alike, goes through the
    contract's own underlying algebra and payoff from here.

    Four guards run first, in this order, and none of them is defensive
    decoration. The result must be *this* match, because a MetricRef is
    resolved by metric and subject and a competitor appears in many matches, so
    a mismatched result would settle silently and plausibly. The format must
    accept the result, because the identities these contracts are priced under
    are the format's conservation laws and a result that breaks them breaks
    them. Every side must have finished together, for a reason found by trying
    it: ``TeamObjective.check`` counts three competitors at each place and is
    handed placements and eliminations without the field, so it cannot see the
    sides and it accepts a winning trio made of one competitor from one side and
    two from the other. Settled, that result pays both winner contracts zero and
    the set that must sum to one sums to nothing. The book knows the grouping and
    therefore checks the thing the format is not in a position to. And every
    level must land inside the bounds its own ref declared, because those bounds
    are what collateral was computed from.
    """
    if result.format_name != book.format_name or result.match_id != book.match_id:
        raise ValueError(
            f"match {result.match_id} in {result.format_name} cannot settle the "
            f"book for match {book.match_id} in {book.format_name}"
        )
    if tuple(sorted(result.field)) != tuple(sorted(book.field)):
        raise ValueError(
            f"match {book.match_id} was listed on {sorted(book.field)} and the "
            f"result is on {sorted(result.field)}"
        )
    fmt = FORMATS[book.format_name]
    fmt.check(result.placements, result.eliminations, book.sides)
    # Only a team format can fail this, and the first version skipped it for a
    # mechanic with one competitor to a side on the grounds that settlement runs
    # once per drawn outcome and the loop is vacuous there. Measured back to
    # back on the same machine, the vacuous version of the loop moved an 0.8s
    # pricing pass by -0.04s and +0.08s on two trials, which is noise. The
    # slowdown that prompted the special case was another process on the box. So
    # there is no special case: the guard runs for every mechanic.
    for side in book.sides:
        if len({result.placements[member] for member in side}) != 1:
            raise ValueError(
                f"match {book.match_id}: side {list(side)} did not finish "
                "together, so no side won it and the winner contracts would "
                "settle to nothing at all"
            )

    placements = result.placements
    eliminations = result.eliminations
    levels: dict[MetricRef, float] = {}
    for ref in book.atoms:
        if ref.metric == METRIC_WIN:
            members = ref.subject.split(SIDE_SEPARATOR)
            value = 1.0 if all(placements[m] == 1 for m in members) else 0.0
        elif ref.metric == METRIC_PLACE:
            value = float(placements[ref.subject])
        elif ref.metric == METRIC_ELIMINATIONS:
            value = float(eliminations[ref.subject])
        else:  # pragma: no cover - unreachable while the book lists four families
            raise KeyError(f"{ref.metric} is not a match metric")
        low, high = ref.bounds
        if not low <= value <= high:
            raise ValueError(
                f"{ref.key} resolved to {value}, outside the [{low}, {high}] the "
                "contract declared and sized its collateral against"
            )
        levels[ref] = value
    return levels


def settlement(book: MatchBook, result: MatchResult) -> dict[str, float]:
    """What every contract in the book pays, given what actually happened.

    One rule, used twice. This is the settlement path and it is also the only
    thing ``coherent_prices`` averages, so a price is by construction the mean
    of the number the contract will actually pay. Going through the contract's
    own underlying and payoff rather than reading the placement dict directly is
    what costs the pricing loop its time: of the 0.25s a 4,000 draw solo book
    takes, 0.07s is drawing the outcomes and 0.15s is this. Drawing is what the
    trim could not touch, which is why the book got five times shorter and the
    pricing pass only twice as fast. A second, faster
    settlement rule written for the pricing loop would buy that back and would
    be two things that are correct about different questions sharing one
    surface, which is a bug class in CONTRIBUTING with four instances already.
    """
    levels = metric_levels(book, result)
    return {
        contract.symbol: contract.instrument.spec.payoff.apply(
            contract.instrument.spec.underlying.evaluate(levels)
        )
        for contract in book.contracts
    }


# -- relations -------------------------------------------------------------


def derive_match_relations(book: MatchBook) -> list[Relation]:
    """Read the identities out of the listed contracts, the way the arbitrageur does.

    Same vocabulary and same conventions as ``arbitrageur.derive_relations``: a
    zero width band is an identity, a one sided band is a bound, the target is
    the leg that is too dear when the excess is positive. So an Arbitrageur
    already built on these instruments enforces them by extending its list,
    with no new execution path.

    It has to be extended, though, and that is worth stating rather than
    assuming. Measured: ``derive_relations`` handed a whole match book returns
    0 relations, because it looks for linear futures, options, spreads and
    indices and a match book lists none of those. An agent built on these
    instruments and left alone is inert on them.

    Every relation here is a statement about a single match outcome, not a
    statistical regularity, which is what makes trading them riskless rather
    than a bet. A ten entrant solo book yields 47 of them: 2 exclusive sets, 10
    placement rungs, 10 eliminations rungs, 5 head to head complements and 20
    head to head bounds against the winner set. The 3v3 yields 22 from a 20
    contract book.

    Relations per contract fell with the trim, from 1.46 on the untrimmed solo
    book to 0.94, and that is the expected direction rather than a loss of
    grip. A long ladder is cheap in relations: each extra rung brings one more
    contract and one more comparison against the rung beside it, so it raises
    the ratio without the board saying anything it could not say before. What
    the ratio now counts is mostly relations that reach across families, which
    are the ones a single mispriced contract cannot hide inside.
    """
    relations = [exclusive.relation() for exclusive in book.exclusive_sets]
    contracts = book.contracts

    # -- the placement ladder is monotone in N ------------------------------
    #
    # Finishing in the top N implies finishing in the top M for any wider M, so
    # the wider rung is worth at least the narrower one, and that holds between
    # any two listed rungs rather than only between adjacent integers, which is
    # what lets the ladder be thinned without the chain breaking. The rung below
    # the first listed one is the winner contract, because placing first and
    # one's side winning are the same event in both mechanics, which is what
    # ties the ladder to the exclusive set and gives "a competitor's win price
    # cannot exceed its top 5 price" without anybody writing that down. With one
    # listed rung that is now the whole chain.
    winner_of: dict[str, str] = {}
    for contract in contracts:
        if contract.family == WINNER:
            for member in contract.subject.split(SIDE_SEPARATOR):
                winner_of[member] = contract.symbol
    rungs: dict[str, list[tuple[int, str]]] = {}
    for contract in contracts:
        if contract.family == PLACEMENT:
            rungs.setdefault(contract.subject, []).append(
                (contract.threshold, contract.symbol)
            )
    for subject, ladder in sorted(rungs.items()):
        ladder.sort()
        ladder = [(1, winner_of[subject])] + ladder
        for (_narrow, narrower), (_wide, wider) in zip(ladder, ladder[1:]):
            relations.append(
                Relation(
                    name=f"placement:{wider}/{narrower}",
                    target=wider,
                    legs=((narrower, 1.0),),
                    lower=0.0,
                    # No upper bound worth writing: the gap between two rungs is
                    # the chance of finishing exactly at the wider one, which
                    # nothing else here pins. The venue's own settlement range
                    # already caps it at the payout.
                    upper=float("inf"),
                )
            )

    # -- the eliminations ladder is monotone in the threshold ---------------
    ladders: dict[str, list[tuple[int, str]]] = {}
    for contract in contracts:
        if contract.family == ELIMINATIONS:
            ladders.setdefault(contract.subject, []).append(
                (contract.threshold, contract.symbol)
            )
    for _subject, ladder in sorted(ladders.items()):
        ladder.sort()
        for (_low, dear), (_high, cheap) in zip(ladder, ladder[1:]):
            # More than a low threshold is implied by more than a high one, so
            # the lower threshold is the dearer contract. True of any two
            # thresholds and not only of adjacent integers, which is what lets
            # ``_elimination_rungs`` list two of nine and still write a relation
            # between them. Same shape as a vertical on a call ladder, for the
            # same reason: one payoff dominates the other everywhere.
            relations.append(
                Relation(
                    name=f"elimination:{dear}/{cheap}",
                    target=dear,
                    legs=((cheap, 1.0),),
                    lower=0.0,
                    upper=float("inf"),
                )
            )

    # -- head to head: complements, and the sandwich against the winner set --
    h2h = {
        (contract.subject, contract.versus): contract.symbol
        for contract in contracts
        if contract.family == HEAD_TO_HEAD
    }
    two_sided = book.places == 2
    for (a, b), symbol in sorted(h2h.items()):
        mirror = h2h.get((b, a))
        if mirror is not None and a < b:
            # Exactly one of them finishes above the other, because the pairs
            # that could finish level were never listed.
            relations.append(
                Relation(
                    name=f"complement:{symbol}/{mirror}",
                    target=symbol,
                    legs=((mirror, -1.0),),
                    constant=1.0,
                )
            )
        # Winning implies being above everybody, so a head to head is worth at
        # least its subject's winner contract. Where a format has only two
        # places the implication runs both ways, since being above the other
        # side is being on the side that won, and the bound closes into an
        # identity.
        relations.append(
            Relation(
                name=f"h2h-floor:{symbol}/{winner_of[a]}",
                target=symbol,
                legs=((winner_of[a], 1.0),),
                lower=0.0,
                upper=0.0 if two_sided else float("inf"),
            )
        )
        # And the other side of the sandwich: if B wins, A is not above B, so
        # the two cannot both be paid. Exact for the same reason, and it is what
        # stops a head to head book drifting up together with the winner book.
        relations.append(
            Relation(
                name=f"h2h-cap:{symbol}/{winner_of[b]}",
                target=symbol,
                legs=((winner_of[b], -1.0),),
                constant=1.0,
                lower=float("-inf"),
                upper=0.0,
            )
        )
    return relations


def excess_exact(relation: Relation, prices: Mapping[str, Fraction]) -> Fraction:
    """``Relation.excess`` in rational arithmetic, for prices that are exact.

    The agent's own version takes floats because it works from mids off a book,
    where a hundredth of a tick of drift is beneath notice. The claim this
    module makes is stronger than that, so it is checked in arithmetic that can
    carry it: every coefficient a relation here carries is +1 or -1 and every
    constant is a whole number, so all of this is exact and the only thing that
    can make an excess nonzero is a genuine violation. Measured over 50 random
    belief vectors on each format, every relation came back at exactly 0.
    """
    theoretical = Fraction(relation.constant)
    for symbol, coefficient in relation.legs:
        theoretical += Fraction(coefficient) * prices[symbol]
    raw = prices[relation.target] - theoretical
    if raw > relation.upper:
        return raw - Fraction(relation.upper)
    if raw < relation.lower:
        return raw - Fraction(relation.lower)
    return Fraction(0)


# -- the one distribution --------------------------------------------------


def _check_belief(book: MatchBook, belief: Mapping[str, float]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for key in book.field:
        if key not in belief:
            raise KeyError(f"{key} is in match {book.match_id} and has no weight")
        weight = float(belief[key])
        if not math.isfinite(weight) or weight <= 0.0:
            # A zero weight makes that competitor's whole family a constant, and
            # a market that lists a constant is spending a book on a countdown.
            # It also divides by zero in the race sampler when a side is all
            # zeros, which would be a crash a long way from its cause.
            raise ValueError(
                f"{key} was given weight {belief[key]!r}; weights must be finite "
                "and strictly positive"
            )
        weights[key] = weight
    return weights


def _ensemble_rng(
    book: MatchBook, weights: Mapping[str, float], draws: int, seed: int | str | None
) -> Random:
    """Deterministic in the belief, so the same view quotes the same prices.

    A maker whose fair values jittered every time it recomputed them would be
    quoting noise on top of its own model, and the difference between the two
    would be invisible in the tape.
    """
    if seed is not None:
        return Random(f"{seed}:{book.format_name}:{book.match_id}:{draws}")
    return Random(
        digest(
            {
                "match": [book.seed, book.match_id, book.format_name],
                "field": list(book.field),
                "belief": [[k, repr(weights[k])] for k in sorted(weights)],
                "draws": draws,
            }
        )
    )


def _draw_elimination(
    book: MatchBook, weights: Mapping[str, float], rng: Random
) -> MatchResult:
    """One hypothetical match under an elimination format.

    Mirrors ``match.play``'s solo branch exactly, with the caller's weights
    where the world uses ``exp(strength)``: a weighted sample without
    replacement for the finishing order, then one credit per elimination handed
    to somebody who finished ahead of the competitor going out, weighted the
    same way. It is the world's mechanics driven by a belief, which is what a
    maker's model is supposed to be.

    The running total is a plain ``sum`` and not ``math.fsum`` on purpose,
    which is the one place in this module where the better conditioned
    arithmetic is the wrong arithmetic. Handed ``exp(strength)`` and the same
    random stream, this reproduces ``match.play``'s own result bit for bit, and
    the test asserts that on 400 real matches rather than settling for a
    distribution that agrees to within a sampling band. A differently
    associated sum would make the two processes differ in the last bit and that
    claim would quietly weaken to nearly.
    """
    remaining = list(book.field)
    order: list[str] = []
    while remaining:
        total = sum(weights[k] for k in remaining)
        cut = rng.random() * total
        running = 0.0
        for key in remaining:
            running += weights[key]
            if running >= cut:
                order.append(key)
                remaining.remove(key)
                break
        else:  # pragma: no cover - float drift only
            order.append(remaining.pop())

    placements = {key: place for place, key in enumerate(order, start=1)}
    eliminations = {key: 0 for key in book.field}
    for position in range(len(order) - 1, 0, -1):
        alive = order[:position]
        killer = rng.choices(alive, weights=[weights[k] for k in alive], k=1)[0]
        eliminations[killer] += 1
    return MatchResult(
        book.match_id, book.format_name, book.field, placements, eliminations
    )


def _draw_race(
    book: MatchBook, weights: Mapping[str, float], rng: Random, target: int
) -> MatchResult:
    """One hypothetical match under a team format. Mirrors ``match.play``'s race.

    Same arithmetic, in the same order, for the same reason as the elimination
    sampler: driven by ``exp(strength)`` and the world's own random stream this
    reproduces the played match exactly, and that is checked rather than
    claimed.
    """
    sides = book.sides
    power = [sum(weights[k] for k in side) for side in sides]
    scores = [0] * len(sides)
    eliminations = {key: 0 for key in book.field}
    while max(scores) < target:
        total = sum(power)
        cut = rng.random() * total
        running = 0.0
        # match.play has no fallback here and would spin forever if float drift
        # ever left the running total short of the cut. Crediting the last side
        # is the same resolution its order sampler already uses.
        index = len(sides) - 1
        for candidate, weight in enumerate(power):
            running += weight
            if running >= cut:
                index = candidate
                break
        scores[index] += 1
        side = sides[index]
        scorer = rng.choices(side, weights=[weights[k] for k in side], k=1)[0]
        eliminations[scorer] += 1

    won = scores.index(max(scores))
    placements = {
        key: (1 if index == won else 2)
        for index, side in enumerate(sides)
        for key in side
    }
    return MatchResult(
        book.match_id,
        book.format_name,
        book.field,
        placements,
        eliminations,
        {key: scores[i] for i, side in enumerate(sides) for key in side},
    )


def draw_outcome(
    book: MatchBook, belief: Mapping[str, float], rng: Random
) -> MatchResult:
    """One hypothetical match, drawn from the caller's own random stream.

    Public because the claim that this model is the world's mechanic rather
    than a likeness of it is only testable if both can be driven from the same
    stream. Given ``exp(strength)`` as the belief and ``match.play``'s own
    generator, this returns the match the world actually played, placement for
    placement and credit for credit.
    """
    fmt = FORMATS[book.format_name]
    return _draw(book, _check_belief(book, belief), _mechanic(fmt), fmt, rng)


def _draw(
    book: MatchBook,
    weights: Mapping[str, float],
    mechanic: _Mechanic,
    fmt: MatchFormat,
    rng: Random,
) -> MatchResult:
    """Dispatch to the mechanic, then let the format refuse what came back.

    One place rather than two. A single draw and a whole ensemble taking
    different routes to the same outcome is how the two quietly stop agreeing.
    """
    if mechanic.kind == "elimination":
        outcome = _draw_elimination(book, weights, rng)
    else:
        outcome = _draw_race(book, weights, rng, mechanic.target)
    fmt.check(outcome.placements, outcome.eliminations, book.sides)
    return outcome


def outcome_ensemble(
    book: MatchBook,
    belief: Mapping[str, float],
    draws: int = DEFAULT_DRAWS,
    seed: int | str | None = None,
) -> tuple[MatchResult, ...]:
    """The one distribution, as the bag of matches it is an average over.

    Returned rather than hidden inside the pricer because it is the object the
    whole guarantee rests on. Anything that can be settled against a
    ``MatchResult`` can be priced by averaging it over this, coherently, with no
    pricing code of its own: that is the entire trick, and it is checkable
    because the bag is right here.

    Every draw is put through the format's ``check`` exactly as ``match.play``
    does, so an outcome this model can produce and the world cannot raises here
    instead of quietly becoming a price.
    """
    if draws < 1:
        raise ValueError("an ensemble needs at least one outcome")
    weights = _check_belief(book, belief)
    fmt = FORMATS[book.format_name]
    mechanic = _mechanic(fmt)
    rng = _ensemble_rng(book, weights, draws, seed)

    return tuple(_draw(book, weights, mechanic, fmt, rng) for _ in range(draws))


def coherent_prices(
    book: MatchBook,
    belief: Mapping[str, float],
    draws: int = DEFAULT_DRAWS,
    seed: int | str | None = None,
    ensemble: Sequence[MatchResult] | None = None,
) -> dict[str, Fraction]:
    """A price for every contract in the match, as one expectation under one law.

    This is the mechanism, and it is deliberately dull: settle the whole book
    against every outcome in the ensemble and divide by the count. A maker
    quoting from this cannot quote an arbitrage between two of these contracts,
    for the same reason the option surface cannot quote one between two strikes.
    Any package of them that pays a fixed amount on every outcome is priced at
    that amount here, because the price *is* the average of what it pays.

    Exact by construction rather than by cleanup. Every payoff in the book is a
    binary paying 0 or 1, so the running total is an integer that float
    addition carries without loss up to 2^53 draws, and the price is that
    integer over the draw count as a Fraction. The winner set therefore sums to
    exactly 1 because every drawn match has exactly one winner. Nothing is
    normalised, and there is no residual hidden in the last contract.

    Passing an ``ensemble`` prices against that bag and never looks at the
    belief, which is not an oversight: the bag is the distribution, the belief
    is only the recipe for drawing one, and a caller who has already drawn is
    entitled to reuse it across several books rather than pay for it twice.
    """
    outcomes = (
        outcome_ensemble(book, belief, draws, seed) if ensemble is None else ensemble
    )
    count = len(outcomes)
    if count == 0:
        raise ValueError("cannot price against an empty ensemble")

    totals = {contract.symbol: 0.0 for contract in book.contracts}
    for outcome in outcomes:
        for symbol, value in settlement(book, outcome).items():
            totals[symbol] += value

    prices: dict[str, Fraction] = {}
    for symbol, total in totals.items():
        if total != int(total):
            # Guard rather than decoration: the exactness claim above rests on
            # the totals being integers, and a family whose payout was not 0 or
            # 1 would break it quietly. This says so instead.
            raise ValueError(
                f"{symbol} accumulated {total}, which is not an integer count; a "
                "fractional payout has to be summed in Fraction, not in float"
            )
        prices[symbol] = Fraction(int(total), count)
    return prices


def two_sided_quote(
    price: Fraction, half_spread: Fraction | float, tick: str = TICK, payout: float = 1.0
) -> tuple[Decimal, Decimal]:
    """A coherent fair value turned into a quote that is still coherent.

    The grid is where a family like this quietly stops being arbitrage free.
    Rounding each of ten exact winner prices to the nearest hundredth breaks the
    set that must sum to one: measured over 300 random beliefs at 4,000 draws,
    the nearest-tick sum was wrong on 58.7% of them and ran from 0.98 to 1.03,
    and the 1.03 end is a book where buying every winner costs three cents more
    than the certificate they add up to is worth.

    Flooring the bid and ceiling the ask cannot do that, and not by luck. Both
    are monotone, so every ladder inequality survives them, and each moves its
    own side away from the fair value, so a sum of bids can only fall short of a
    total it should not exceed and a sum of asks can only overshoot one it
    should not fall below. Measured at the hardest setting, a maker quoting with
    no spread at all, that held on 300 of 300 beliefs. That is the entire reason
    this is arithmetic here rather than a rounding convention left to each maker.
    """
    step = Decimal(tick)
    if step <= 0:
        raise ValueError("tick must be positive")
    half = Fraction(half_spread)
    if half < 0:
        raise ValueError("half spread cannot be negative")

    # Both edges are computed from the exact rational fair value and rounded
    # once, at the end. Converting to Decimal first and subtracting there would
    # round twice, and the second rounding is the one that could move a bid the
    # wrong way.
    def to_grid(value: Fraction, rounding: str) -> Decimal:
        ticks = value / Fraction(step)
        return (
            Decimal(ticks.numerator) / Decimal(ticks.denominator)
        ).to_integral_value(rounding=rounding) * step

    bid = to_grid(Fraction(price) - half, ROUND_FLOOR)
    ask = to_grid(Fraction(price) + half, ROUND_CEILING)
    # Quantized after the clamp so both sides carry the tick's own exponent. A
    # price crosses the wire as a string here, and an offer that reads "1.0"
    # where its bid reads "1.00" is the same number wearing two spellings.
    return (
        max(Decimal(0), bid).quantize(step),
        min(Decimal(str(payout)), ask).quantize(step),
    )
