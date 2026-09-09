"""Match markets, and the property that they cannot disagree with themselves.

The claim under test is not that these prices are good. It is that they are
coherent: that no package of contracts on one match can be assembled into a
riskless profit, because every price in the book is an expectation of the same
settlement rule under one distribution. That is a property, so it is tested as
one, over thousands of played matches, a hundred random beliefs and a
thousand outcomes drawn from the settleable space directly, rather than on an
example.

Two tests carry the weight. ``test_one_distribution_violates_no_relation``
prices a whole book off the single ensemble and finds every one of the 395
derived relations at excess exactly zero, in rational arithmetic, on 50 random
beliefs for each format. ``test_pricing_each_contract_on_its_own_law_is_where
_the_arbitrage_comes_from`` prices the same contracts off eight unrelated
ensembles instead, which is what the old prediction listing did by accident,
and 53 of those 395 relations are then breached by up to 0.1163 even though
all eight ensembles were drawn from one belief and disagree only by sampling
noise. The second number is what the first one is worth: without it, zero
violations could just mean the relations have no teeth.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction

import pytest

from arena.market.instrument import InstrumentClass
from arena.market.match_book import (
    ELIMINATIONS,
    HEAD_TO_HEAD,
    PLACEMENT,
    SIDE_SEPARATOR,
    WINNER,
    coherent_prices,
    derive_match_relations,
    draw_outcome,
    excess_exact,
    list_match,
    metric_levels,
    outcome_ensemble,
    settlement,
    two_sided_quote,
)
from arena.worlds.circuit.match import MatchResult, play
from arena.worlds.circuit.modes import FORMATS, TeamObjective, register_format
from arena.worlds.circuit.roster import draw_strengths

SEED = 7
FORMAT_NAMES = ("solo", "objective")

# Enough matches that a settlement identity holding is evidence rather than
# coincidence, and enough beliefs that a relation holding is the same. Both
# loops are deterministic, so these are a fixed cost rather than a flake risk:
# settling 2,000 matches of each format takes 5.5s and pricing the hundred
# beliefs takes 9.9s, which is most of the file's 35s.
MATCHES = 2_000
BELIEFS = 50
# Below the module's default of 4,000. The identities are exact at any draw
# count, since they hold outcome by outcome, so the draws here buy resolution
# for the convergence test and nothing else.
DRAWS = 800


@pytest.fixture(scope="module")
def priced():
    """Every format's book, its relations, and prices under BELIEFS beliefs.

    Priced once and shared, because the ensemble is the expensive part and
    eight tests want to ask different questions of the same prices.
    """
    out = {}
    for name in FORMAT_NAMES:
        book = list_match(SEED, 3, name)
        relations = derive_match_relations(book)
        rng = random.Random(42)
        rows = []
        for _ in range(BELIEFS):
            # Lognormal weights spanning roughly a factor of twenty across the
            # field, so the beliefs are lopsided enough to put real mass on the
            # ends of every ladder rather than sitting near a uniform field
            # where most contracts price near their own boundary.
            belief = {k: math.exp(rng.gauss(0.0, 0.7)) for k in book.field}
            rows.append((belief, coherent_prices(book, belief, draws=DRAWS)))
        out[name] = (book, relations, rows)
    return out


# --------------------------------------------------------------------------
# The contract set is read off the format
# --------------------------------------------------------------------------


def test_each_format_lists_the_families_its_shape_supports():
    """270 contracts for the solo, 38 for the 3v3, and the difference is real.

    The 3v3 lists no placement ladder and that is not an omission. With two
    places, "top 1" is the winner contract already listed and "top 2" is a
    certainty, so a placement ladder there would be six contracts nobody can
    lose money on. The eliminations ladder is shorter for the same kind of
    reason: a point is credited inside the side that scored it, so a competitor
    can be credited at most the three that end the race, against nine in a ten
    entrant elimination.
    """
    solo = list_match(SEED, 0, "solo")
    assert len(solo.contracts) == 270
    assert len(solo.of_family(WINNER)) == 10
    assert len(solo.of_family(PLACEMENT)) == 80
    assert len(solo.of_family(ELIMINATIONS)) == 90
    assert len(solo.of_family(HEAD_TO_HEAD)) == 90
    assert solo.places == 10 and solo.elimination_cap == 9
    assert solo.elimination_total == 9

    objective = list_match(SEED, 0, "objective")
    assert len(objective.contracts) == 38
    assert len(objective.of_family(WINNER)) == 2
    assert len(objective.of_family(PLACEMENT)) == 0
    assert len(objective.of_family(ELIMINATIONS)) == 18
    assert len(objective.of_family(HEAD_TO_HEAD)) == 18
    assert objective.places == 2 and objective.elimination_cap == 3
    # A race ends after anything from three to five points, so there is no
    # constant for the field's eliminations to conserve and no relation claiming
    # otherwise is formed.
    assert objective.elimination_total is None


def test_the_listing_is_something_the_venue_can_actually_carry():
    """These are instruments, not a private structure with a price attached.

    Every contract is a Binary on a bounded metric, so the venue classifies all
    270 as events, every one settles inside [0, 1], and a short's worst case is
    arithmetic: one lot sold at 0.40 can lose 0.60 and never more. The digests
    are distinct too, which is the check that no two derived contracts are
    secretly the same claim listed twice under different names.
    """
    book = list_match(SEED, 0, "solo")
    assert len({c.symbol for c in book.contracts}) == len(book.contracts)
    assert len({c.instrument.spec.spec_digest for c in book.contracts}) == len(
        book.contracts
    )
    # A symbol names the match the way a ticker does, so a competitor drawn into
    # match 0 of two different worlds carries the same one twice. The digest may
    # not repeat: those are two different fields behind one label, and a
    # settlement record that could not tell them apart would be a record of
    # nothing. The world seed is in the reference id for exactly this.
    other_world = list_match(SEED + 1, 0, "solo")
    assert set(other_world.symbols) & set(book.symbols)
    assert {c.instrument.spec.spec_digest for c in other_world.contracts}.isdisjoint(
        {c.instrument.spec.spec_digest for c in book.contracts}
    )
    for contract in book.contracts:
        instrument = contract.instrument
        assert instrument.instrument_class == InstrumentClass.EVENT
        assert instrument.settlement_bounds == (Decimal("0"), Decimal("1"))
        assert instrument.collateral_for(-1, Decimal("0.40")) == Decimal("0.60")
        assert instrument.expiry == book.window.end


def test_head_to_head_is_listed_only_where_the_pair_can_be_separated():
    """Team mates always finish level, so their pair would not complement.

    Listed anyway, "A above B" and "B above A" would both settle at zero every
    time the two are on the same side, and the complement identity that the
    whole family is anchored on would be false on every one of those pairs. The
    3v3 lists 18 head to heads, which is the 30 ordered pairs of six
    competitors less the 12 that share a side.
    """
    book = list_match(SEED, 0, "objective")
    pairs = {(c.subject, c.versus) for c in book.of_family(HEAD_TO_HEAD)}
    assert len(pairs) == 18
    for a, b in pairs:
        assert book.side_of(a) != book.side_of(b)


@dataclass(frozen=True)
class FourWay:
    """A format nobody has seen before, registered by a test rather than a file.

    Its name is its own rather than something obvious like "duel", because
    ``test_match_engine`` registers a throwaway of that name too and two tests
    that leak the same key would fail each other rather than themselves.
    """

    name: str = "fourway"
    entrants: int = 4
    team_size: int = 1

    @property
    def teams(self) -> int:
        return self.entrants

    def check(self, placements, eliminations, sides=None) -> None:
        if sorted(placements.values()) != list(range(1, self.entrants + 1)):
            raise ValueError(f"{self.name}: placements are not a permutation")
        if sum(eliminations.values()) != self.entrants - 1:
            raise ValueError(f"{self.name}: eliminations do not close")


@pytest.fixture
def throwaway_formats():
    """Register two formats for one test and take them out again.

    FORMATS is process global and ``register_format`` refuses to replace a name,
    so a test that leaked one would fail the next run of the same session rather
    than the one that leaked it.
    """
    added = (
        FourWay(),
        replace(TeamObjective(), name="pairs", entrants=4, team_size=2, target=2),
    )
    try:
        # Registration is inside the try. Outside it, a second format that
        # failed to register would leave the first one in the registry for the
        # rest of the session, and the test that tripped over it would not be
        # this one.
        for fmt in added:
            register_format(fmt)
        yield added
    finally:
        for fmt in added:
            FORMATS.pop(fmt.name, None)


def test_a_newly_registered_format_lists_and_prices_with_no_edit_here(
    throwaway_formats,
):
    """Adding a format is a registry entry, and its market follows from it.

    This is the test that says the derivation is real rather than a list that
    happens to match. Neither format below exists in the source: a four entrant
    elimination lists 4 + 8 + 12 + 12 = 36 contracts and a 2v2 race to two lists
    2 + 0 + 8 + 8 = 18, both counts falling out of the entrants, the team size
    and the target alone. Both then price off the same ensemble with no
    violations, which matters more than the counts: a contract set derived for a
    format the pricer cannot draw would be a listing nobody can quote.
    """
    fourway, pairs = throwaway_formats

    book = list_match(SEED, 1, fourway.name)
    assert len(book.contracts) == 36
    families = (WINNER, PLACEMENT, ELIMINATIONS, HEAD_TO_HEAD)
    assert [len(book.of_family(f)) for f in families] == [4, 8, 12, 12]
    assert book.places == 4 and book.elimination_cap == 3 and book.elimination_total == 3

    team = list_match(SEED, 1, pairs.name)
    assert len(team.contracts) == 18
    assert [len(team.of_family(f)) for f in families] == [2, 0, 8, 8]
    assert team.places == 2 and team.elimination_cap == 2

    for listing in (book, team):
        rng = random.Random(5)
        belief = {k: math.exp(rng.gauss(0.0, 0.7)) for k in listing.field}
        prices = coherent_prices(listing, belief, draws=400)
        assert len(prices) == len(listing.contracts)
        worst = max(
            abs(excess_exact(relation, prices))
            for relation in derive_match_relations(listing)
        )
        assert worst == 0


def test_a_format_whose_mechanic_is_not_modelled_refuses_to_list():
    """A pricer that guessed would be worse than one that stopped.

    ``match.play`` reads a team format's target through a getattr with a default
    of 3. A pricing model that adopted the same default would be quoting a race
    length the format never declared, and the two would disagree in silence,
    which is how a market ends up trading an artifact. So a team format with no
    target refuses to list at all.
    """

    @dataclass(frozen=True)
    class Untargeted:
        name: str = "untargeted"
        entrants: int = 4
        team_size: int = 2

        @property
        def teams(self) -> int:
            return self.entrants // self.team_size

        def check(self, placements, eliminations, sides=None) -> None:
            return None

    register_format(Untargeted())
    try:
        with pytest.raises(ValueError, match="must declare an integer target"):
            list_match(SEED, 0, "untargeted")
    finally:
        FORMATS.pop("untargeted", None)


# --------------------------------------------------------------------------
# Settlement against what actually happened
# --------------------------------------------------------------------------


def test_the_winner_set_settles_to_exactly_one_on_every_match():
    """Exactly one winner, on 2,000 matches of each format, summed exactly.

    Float equality on purpose. Every contract in the set settles at 0.0 or 1.0,
    so the sum is exact arithmetic and a tolerance here would be hiding
    something rather than allowing for something. This is the fact the whole
    module is built to keep true in prices as well as in settlements: if it can
    fail here it was never going to hold there.
    """
    for name in FORMAT_NAMES:
        for match_id in range(MATCHES):
            book = list_match(SEED, match_id, name)
            values = settlement(book, play(SEED, match_id, name))
            winners = [c.symbol for c in book.of_family(WINNER)]
            assert sum(values[symbol] for symbol in winners) == 1.0
            assert sum(1 for symbol in winners if values[symbol] == 1.0) == 1


def test_every_exclusive_set_settles_to_its_declared_total():
    """The other two things an exclusive set says, checked the same way.

    Exactly N times the team size finish in the top N, and the field's
    eliminations add to the number the format conserves. Both are the same
    object as the winner set with a different total, and both have to hold on a
    real result or the relations derived from them are trading on a fiction.
    Five hundred matches per format rather than the two thousand the winner set
    gets, which is the honest tradeoff: this sweep settles ten exclusive sets
    per match instead of one and buys the same evidence at a quarter of the
    matches, and the wall clock is the reason.
    """
    for name in FORMAT_NAMES:
        for match_id in range(500):
            book = list_match(SEED, match_id, name)
            values = settlement(book, play(SEED, match_id, name))
            for exclusive in book.exclusive_sets:
                assert exclusive.shortfall(values) == 0.0


def test_settlement_agrees_with_the_match_result_contract_by_contract():
    """Every contract, recomputed from the result independently, on 2,000 matches.

    The expected value here is worked out from ``placements`` and
    ``eliminations`` directly rather than by asking the module a second time, so
    this is a check on the settlement rule and not a check that a function
    equals itself. It covers all four families on every match, which is 540,000
    settled claims on the solo book and 76,000 on the 3v3.
    """
    for name in FORMAT_NAMES:
        for match_id in range(MATCHES):
            book = list_match(SEED, match_id, name)
            result = play(SEED, match_id, name)
            values = settlement(book, result)
            places = result.placements
            credits = result.eliminations
            for contract in book.contracts:
                if contract.family == WINNER:
                    members = contract.subject.split("-")
                    expected = 1.0 if all(places[m] == 1 for m in members) else 0.0
                elif contract.family == PLACEMENT:
                    expected = 1.0 if places[contract.subject] <= contract.threshold else 0.0
                elif contract.family == ELIMINATIONS:
                    expected = (
                        1.0 if credits[contract.subject] > contract.threshold else 0.0
                    )
                else:
                    expected = (
                        1.0
                        if places[contract.subject] < places[contract.versus]
                        else 0.0
                    )
                assert values[contract.symbol] == expected


def _arbitrary_settleable_result(book, rng):
    """A result the settlement path would accept, drawn with no regard for the world.

    Deliberately not from ``match.play``. The question a relation has to answer
    is whether it can be violated by any outcome the exchange would settle, not
    by any outcome the simulator tends to produce, and those are different sets:
    this generator happily returns a match where the winner took no eliminations
    and the competitor knocked out first took five.
    """
    fmt = FORMATS[book.format_name]
    field = list(book.field)
    if fmt.team_size == 1:
        order = field[:]
        rng.shuffle(order)
        placements = {key: place for place, key in enumerate(order, start=1)}
        credits = book.elimination_total
    else:
        sides = list(book.sides)
        winner = rng.randrange(len(sides))
        placements = {
            key: (1 if index == winner else 2)
            for index, side in enumerate(sides)
            for key in side
        }
        credits = rng.randint(book.elimination_cap, 2 * book.elimination_cap - 1)

    eliminations = {key: 0 for key in field}
    for _ in range(credits):
        room = [k for k in field if eliminations[k] < book.elimination_cap]
        eliminations[rng.choice(room)] += 1
    return MatchResult(
        book.match_id, book.format_name, book.field, placements, eliminations
    )


def test_no_settleable_outcome_can_violate_a_relation():
    """The claim behind every relation, tested against the outcome space itself.

    A relation is only riskless if it holds on every result the venue would
    settle, and the venue settles whatever passes the checks in
    ``metric_levels``. So this samples that space directly, 500 arbitrary
    results per format that owe nothing to the simulator, settles the book
    against each and evaluates every relation on the settlement values. All of
    them hold, which is the property, and it is not an accident: each relation
    is a consequence of the permutation and the sum that the format's own check
    enforces, plus the side cohesion the book checks because the format cannot.

    This is also why the module derives no relation from the fact that a
    competitor can only be credited with eliminating people who finish behind
    them. That is true of ``match.play`` and it is not enforced anywhere at
    settlement, so a contract pair resting on it would be a bet on the world's
    implementation rather than an identity.
    """
    for name in FORMAT_NAMES:
        book = list_match(SEED, 9, name)
        relations = derive_match_relations(book)
        rng = random.Random(2024)
        for _ in range(500):
            values = settlement(book, _arbitrary_settleable_result(book, rng))
            exact = {symbol: Fraction(value) for symbol, value in values.items()}
            for relation in relations:
                assert excess_exact(relation, exact) == 0, relation.name


def test_a_side_that_did_not_finish_together_is_refused():
    """Found by trying it: the format's own check cannot catch this one.

    ``TeamObjective.check`` counted three competitors at each place and
    accepted a winning trio drawn from both sides. Settled, that pays both
    winner contracts zero and the set that must sum to one sums to zero, which
    is a certificate the exchange sold for nothing.

    The format now refuses it too, once it is handed the grouping it was
    previously never given. Both layers are asserted here rather than one:
    the book knows the sides and checks them, and the format checks them when
    a caller passes them, and neither is load-bearing alone.
    """
    book = list_match(SEED, 0, "objective")
    first, second = book.sides
    mixed = {first[0], first[1], second[0]}
    placements = {key: (1 if key in mixed else 2) for key in book.field}
    with pytest.raises(ValueError, match="not one of the sides"):
        FORMATS["objective"].check(
            placements, {key: 0 for key in book.field}, book.sides
        )
    spliced = MatchResult(
        0, "objective", book.field, placements, {key: 0 for key in book.field}
    )
    # Settlement refuses it too. The format now speaks first, since it is asked
    # before the book's own grouping check, so the message is the format's and
    # the book's check is the one that would still catch this if a caller
    # arrived without sides. Matched on either rather than on the current
    # ordering, because which layer speaks is an implementation detail and the
    # refusal is not.
    with pytest.raises(
        ValueError, match="did not finish together|not one of the sides"
    ):
        settlement(book, spliced)


def test_a_book_refuses_to_settle_against_another_match():
    """The bug class where an identity is reused after it stopped meaning anything.

    A MetricRef is resolved by metric and subject, and a competitor appears in
    many matches, so a result from the wrong match would settle silently and
    plausibly. Both halves of the identity are checked, the match id and the
    field, because two matches in a season can share neither, either or both.
    """
    book = list_match(SEED, 4, "solo")
    with pytest.raises(ValueError, match="cannot settle the book"):
        settlement(book, play(SEED, 5, "solo"))
    with pytest.raises(ValueError, match="cannot settle the book"):
        settlement(book, play(SEED, 4, "objective"))


# --------------------------------------------------------------------------
# One distribution, and what it buys
# --------------------------------------------------------------------------


def test_one_distribution_violates_no_relation(priced):
    """Every derived relation, at excess exactly zero, on 50 beliefs per format.

    Exact means exact. The prices are Fractions over the draw count and every
    relation coefficient is +1 or -1 against a whole number constant, so this
    arithmetic has no rounding in it to hide behind. 395 relations on the solo
    book and 58 on the 3v3, so the sweep is 22,650 checks and the worst excess
    across all of them is 0.
    """
    for name in FORMAT_NAMES:
        _book, relations, rows = priced[name]
        assert relations, f"{name} derived no relations to check"
        for _belief, prices in rows:
            for relation in relations:
                assert excess_exact(relation, prices) == 0, relation.name


def test_float_prices_survive_the_conversion(priced):
    """What a consumer that works in floats actually sees. Measured, not assumed.

    The exact prices are Fractions, and everything downstream of a quote works
    in floats, so the honest question is what the conversion costs. Measured
    over 100 priced beliefs: 3.9e-15 at worst, on the conservation set that adds
    90 contracts up to nine, and a half ulp on the two families that subtract
    from a constant. The families that do no arithmetic at all, the two ladders
    and the head to head floor, come back at exactly 0.0, because those compare
    two prices and a float comparison cannot invert what the exact one said. So
    a consumer working in floats needs a tolerance of a few ulps and nothing
    more, which is a different kind of requirement from the 0.23 the old
    listing would have needed.
    """
    worst = 0.0
    for name in FORMAT_NAMES:
        _book, relations, rows = priced[name]
        for _belief, prices in rows:
            floats = {symbol: float(price) for symbol, price in prices.items()}
            for relation in relations:
                excess = abs(relation.excess(floats[relation.target], floats))
                worst = max(worst, excess)
                # A relation with one leg at +1 and no constant is a straight
                # comparison of two prices, so nothing rounds and the float
                # answer is the exact one. The rest sum or subtract and are
                # allowed their ulp.
                pass_through = (
                    len(relation.legs) == 1
                    and relation.legs[0][1] == 1.0
                    and relation.constant == 0.0
                )
                if pass_through:
                    assert excess == 0.0, relation.name
    assert worst < 1e-13


def test_pricing_each_contract_on_its_own_law_is_where_the_arbitrage_comes_from(
    priced,
):
    """The control. Without it, zero violations could mean the relations are toothless.

    Eight independent ensembles, each a perfectly reasonable distribution over
    this match, with every contract priced off whichever one it was assigned.
    That is what the old prediction listing did without meaning to: standalone
    binaries whose prices came from unrelated books. This is the mildest
    incoherence available, because all eight ensembles are drawn from the *same*
    belief and differ only by sampling noise, so nothing here is biased or
    broken and every individual price is a sound estimate of the same number.
    Measured anyway: 53 of the 395 solo relations breached, 13.4%, by up to
    0.1163 on the set that conserves the field's nine eliminations, and 31 of
    58 on the 3v3. The breaches land on the sums first, because the noise at 800
    draws is smaller than the gap between two adjacent ladder rungs and larger
    than nothing at all, which is the whole point: an identity is violated by
    any disagreement whatever, and disagreement is the default when two prices
    come from two laws.
    """
    worst = 0.0
    for name in FORMAT_NAMES:
        book, relations, rows = priced[name]
        belief, _prices = rows[0]
        laws = [
            coherent_prices(book, belief, draws=DRAWS, seed=f"crowd-{n}")
            for n in range(8)
        ]
        scattered = {
            contract.symbol: laws[index % len(laws)][contract.symbol]
            for index, contract in enumerate(book.contracts)
        }
        excesses = [
            abs(excess_exact(relation, scattered)) for relation in relations
        ]
        breached = [excess for excess in excesses if excess != 0]
        assert breached, f"{name}: eight unrelated laws broke nothing here"
        assert len(breached) / len(relations) > 0.10
        worst = max(worst, float(max(excesses)))
    assert worst > 0.05


def test_the_eliminations_ladder_is_monotone_in_the_threshold(priced):
    """More than k+1 implies more than k, so the rungs cannot cross.

    Exact in the prices because it is exact in every outcome: the ensemble
    counts the same drawn match into both rungs or into neither, and an integer
    count over a common denominator cannot invert. Checked on both formats over
    50 beliefs each, so 9 rungs on ten solo competitors and 3 on six team
    competitors.
    """
    for name in FORMAT_NAMES:
        book, _relations, rows = priced[name]
        ladders: dict[str, list[tuple[int, str]]] = {}
        for contract in book.of_family(ELIMINATIONS):
            ladders.setdefault(contract.subject, []).append(
                (contract.threshold, contract.symbol)
            )
        for _belief, prices in rows:
            for subject, ladder in ladders.items():
                rungs = [prices[symbol] for _t, symbol in sorted(ladder)]
                assert rungs == sorted(rungs, reverse=True), subject
                assert all(0 <= rung <= 1 for rung in rungs)


def test_a_win_never_prices_above_its_own_top_three(priced):
    """Winning is finishing in the top three, so it cannot be worth more.

    The relation that enforces it is derived as a chain of adjacent rungs
    starting at the winner contract, so this is the transitive statement rather
    than the derived one, which is exactly why it is worth asserting separately:
    it is the reading a trader would do, and it holds without anybody deriving
    it directly.
    """
    book, _relations, rows = priced["solo"]
    top3 = {c.subject: c.symbol for c in book.of_family(PLACEMENT) if c.threshold == 3}
    winner = {c.subject: c.symbol for c in book.of_family(WINNER)}
    assert len(top3) == 10 and len(winner) == 10
    for _belief, prices in rows:
        for competitor in book.field:
            assert prices[winner[competitor]] <= prices[top3[competitor]]


def test_the_placement_ladder_is_monotone_and_reaches_the_winner(priced):
    """The whole placement chain, not only its first and third rungs."""
    book, _relations, rows = priced["solo"]
    ladders: dict[str, list[tuple[int, str]]] = {}
    for contract in book.of_family(WINNER):
        ladders.setdefault(contract.subject, []).append((1, contract.symbol))
    for contract in book.of_family(PLACEMENT):
        ladders.setdefault(contract.subject, []).append(
            (contract.threshold, contract.symbol)
        )
    for _belief, prices in rows:
        for subject, ladder in ladders.items():
            rungs = [prices[symbol] for _t, symbol in sorted(ladder)]
            assert rungs == sorted(rungs), subject


def test_head_to_head_is_sandwiched_by_the_winner_set(priced):
    """Winning implies being above everyone, and being above B implies B did not win.

    Two exact bounds rather than one, and together they pin a head to head into
    an interval that closes completely in a two place format: measured on the
    3v3, the width of that interval is exactly zero on every pair and every
    belief, because being above the other side is being on the side that won.
    """
    for name in FORMAT_NAMES:
        book, _relations, rows = priced[name]
        winner = {}
        for contract in book.of_family(WINNER):
            for member in contract.subject.split(SIDE_SEPARATOR):
                winner[member] = contract.symbol
        for _belief, prices in rows:
            for contract in book.of_family(HEAD_TO_HEAD):
                a, b = contract.subject, contract.versus
                assert prices[winner[a]] <= prices[contract.symbol]
                assert prices[contract.symbol] + prices[winner[b]] <= 1
                if book.places == 2:
                    assert prices[contract.symbol] == prices[winner[a]]


def test_complementary_head_to_heads_sum_to_exactly_one(priced):
    """Exactly one of two competitors who cannot tie finishes above the other.

    Exact rather than nearly, on both formats and every belief, because the
    ensemble counts each drawn match into one of the pair and never into both
    or neither.
    """
    for name in FORMAT_NAMES:
        book, _relations, rows = priced[name]
        listed = {(c.subject, c.versus): c.symbol for c in book.of_family(HEAD_TO_HEAD)}
        for _belief, prices in rows:
            for (a, b), symbol in listed.items():
                assert prices[symbol] + prices[listed[(b, a)]] == 1


# --------------------------------------------------------------------------
# The distribution is the world's own process
# --------------------------------------------------------------------------


def test_the_sampler_reproduces_matches_the_world_actually_played():
    """The model is the format's mechanic, proven by driving both from one stream.

    A distribution that merely agrees to within a sampling band would leave
    room for the model to have drifted from ``match.play`` in a way no
    tolerance would catch. So this reaches into the world's own seeding rule on
    purpose and asks for the identity instead: handed ``exp(strength)`` as its
    belief and the generator ``play`` builds for that match, ``draw_outcome``
    returns the match that was played, placement for placement and credit for
    credit. Measured: 400 matches per format, 0 mismatches. It is why the
    sampler adds its weights with a plain sum rather than a better conditioned
    one, and if the world's seeding rule ever changes, this failing is the
    correct outcome rather than a nuisance.
    """
    for name in FORMAT_NAMES:
        strengths = draw_strengths(SEED, name)
        for match_id in range(400):
            real = play(SEED, match_id, name)
            book = list_match(SEED, match_id, name)
            belief = {key: math.exp(strengths[key]) for key in book.field}
            rng = random.Random(f"{SEED}:match:{name}:{match_id}")
            modelled = draw_outcome(book, belief, rng)
            assert modelled.placements == real.placements
            assert modelled.eliminations == real.eliminations


def test_the_ensemble_converges_on_the_closed_forms_it_declines_to_use():
    """The Plackett-Luce identities, as a check on the sampler's law.

    A weighted sample without replacement puts competitor i first with
    probability w_i / W exactly, and puts A ahead of B with probability
    w_A / (w_A + w_B) exactly, whatever the rest of the field does. Neither is
    used to price anything, because a closed form for one family and an ensemble
    for another is two measures and two measures is the defect this module
    exists to remove. They are worth this test anyway: they say the ensemble is
    sampling the law it claims to sample. Measured on the default field, the
    worst gap falls 0.0297, 0.0100, 0.0042 as the draws go 1,000, 4,000, 16,000,
    which is the 1/sqrt(draws) it should be.
    """
    book = list_match(SEED, 0, "solo")
    rng = random.Random(11)
    belief = {key: math.exp(rng.gauss(0.0, 0.35)) for key in book.field}
    total = math.fsum(belief.values())
    prices = coherent_prices(book, belief, draws=4_000)

    winner = {c.subject: c.symbol for c in book.of_family(WINNER)}
    assert (
        max(
            abs(float(prices[winner[key]]) - belief[key] / total) for key in book.field
        )
        < 0.02
    )
    assert (
        max(
            abs(
                float(prices[c.symbol])
                - belief[c.subject] / (belief[c.subject] + belief[c.versus])
            )
            for c in book.of_family(HEAD_TO_HEAD)
        )
        < 0.03
    )


def test_prices_are_deterministic_in_the_belief():
    """A maker whose fair value jittered would be quoting noise over its model.

    The ensemble is seeded from the belief itself, so recomputing a price
    without new information cannot move it, and two callers holding the same
    view get the same number. A belief that differs in the last bit is a
    different belief and is allowed to price differently.
    """
    book = list_match(SEED, 2, "solo")
    rng = random.Random(3)
    belief = {key: math.exp(rng.gauss(0.0, 0.5)) for key in book.field}
    first = coherent_prices(book, belief, draws=400)
    second = coherent_prices(book, dict(reversed(list(belief.items()))), draws=400)
    assert first == second


def test_every_drawn_outcome_would_be_accepted_by_the_format():
    """A model that can produce a match the world cannot is priced on a fiction.

    ``outcome_ensemble`` runs the format's own check on every draw, so this
    asserts the guard is live rather than that the outcomes look plausible: it
    settles the whole book against every drawn outcome, which is the same path
    a real settlement takes and therefore the same checks.
    """
    for name in FORMAT_NAMES:
        book = list_match(SEED, 6, name)
        rng = random.Random(8)
        belief = {key: math.exp(rng.gauss(0.0, 0.9)) for key in book.field}
        for outcome in outcome_ensemble(book, belief, draws=200):
            levels = metric_levels(book, outcome)
            assert len(levels) == len(book.atoms)


def test_a_belief_that_rules_a_competitor_out_is_refused():
    """A zero weight lists a constant, and a book on a constant is a countdown."""
    book = list_match(SEED, 0, "solo")
    belief = {key: 1.0 for key in book.field}
    belief[book.field[0]] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        coherent_prices(book, belief, draws=10)


# --------------------------------------------------------------------------
# Getting the price onto the tick grid without giving the property back
# --------------------------------------------------------------------------


def test_nearest_tick_rounding_breaks_the_set_and_flooring_the_bid_does_not(priced):
    """The grid is where a coherent family quietly stops being coherent.

    Rounding each of the ten exact winner prices to the nearest hundredth breaks
    the sum that must be one on 58.7% of beliefs, running from 0.98 to 1.03, and
    the 1.03 end is a book where buying the whole winner set costs three cents
    more than it can ever pay. Flooring the bid and ceiling the ask cannot do
    that: both are monotone and each moves its own side away from fair value.
    Checked at the hardest setting, a maker quoting with no spread at all.
    """
    book, _relations, rows = priced["solo"]
    winners = [c.symbol for c in book.of_family(WINNER)]
    # A finite ensemble prices an outcome it never drew at exactly zero, and a
    # maker that published that as an offer would be selling a lottery ticket
    # for nothing. The ceiling on the ask is what stops it, so it is asserted
    # here rather than left to the prose.
    assert two_sided_quote(Fraction(0), Fraction(1, 100)) == (
        Decimal("0"),
        Decimal("0.01"),
    )
    off_by_rounding = 0
    for _belief, prices in rows:
        nearest = sum(
            (Decimal(prices[s].numerator) / Decimal(prices[s].denominator)).quantize(
                Decimal("0.01")
            )
            for s in winners
        )
        if nearest != 1:
            off_by_rounding += 1
        quotes = [two_sided_quote(prices[s], 0) for s in winners]
        assert sum(bid for bid, _ask in quotes) <= 1
        assert sum(ask for _bid, ask in quotes) >= 1
        for (bid, ask), symbol in zip(quotes, winners):
            assert bid <= Fraction(prices[symbol]) <= ask
    assert off_by_rounding > len(rows) // 2
