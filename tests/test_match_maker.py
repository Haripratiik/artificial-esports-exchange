"""The match maker, and the property it exists to hold.

The exchange's standalone prediction markets carry real arbitrage: a violation
in 3 of 3 runs, by 0.23 and 0.31 on a contract bounded in [0, 1]. The option
chain measures 0 of 8 vertical violations over the same runs, and the reason is
structural rather than lucky, because `SurfaceMarketMaker` prices every strike
off one distribution and no two quotes are then able to disagree.

`MatchMaker` does the same for a match family, and these tests are written to
*prove* the arbitrage-freeness rather than to observe it holding. A solo match
has ten outcomes, which is small enough to enumerate exactly, so the question
"is there any package a client could build out of this maker's own quotes that
wins whatever happens" is answered by a linear program over the whole outcome
set rather than by sampling a few packages and hoping.

The program is the standard one. With ``q_i`` bought at the maker's offer and
``r_i`` sold at its bid, the client's profit at outcome ``w`` is
``sum_i q_i (v_i(w) - ask_i) - sum_i r_i (v_i(w) - bid_i)``, and the LP
maximises the worst outcome's profit subject to the package having unit size,
which is without loss because a package that wins everywhere still wins
everywhere after scaling. The empty package is always feasible at zero, so the
optimum is at least zero and an arbitrage is an optimum strictly above it. The
number to read is therefore the maximum over every trial, and it should be
exactly zero.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import pytest
from scipy.optimize import linprog

from arena.agents.match_maker import (
    MatchLeg,
    MatchListing,
    MatchMaker,
    field_leg,
    solo_outcomes,
    team_outcomes,
    winner_legs,
)
from arena.contracts.payoff import Binary, Linear
from arena.contracts.spec import ContractSpec, DataPolicy, ObservationWindow
from arena.contracts.underlying import MetricRef, Single
from arena.exchange.types import AgentId, Price, Side
from arena.market.instrument import Instrument
from arena.market.venue import SymbolCommand
from arena.worlds.circuit.match import play
from arena.worlds.circuit.modes import FORMATS
from arena.worlds.circuit.roster import ROSTER

UTC = timezone.utc
WINDOW = ObservationWindow(
    start=datetime(2026, 8, 3, tzinfo=UTC), end=datetime(2026, 8, 31, tzinfo=UTC)
)
VENUE = AgentId("venue")


# --------------------------------------------------------------------------
# Fixtures. Deliberately this module's own, not `match_book`'s.
# --------------------------------------------------------------------------


def instrument(symbol: str, payout: float = 1.0, tick: str = "0.01") -> Instrument:
    """A contract paying ``payout`` on an event, or scaling a rate onto [0, p].

    Built here rather than imported so these tests pin the maker's behaviour and
    not a listing module's choices. What the maker actually reads off an
    instrument is its tick size and its value bounds, so anything with the right
    range exercises the same paths.
    """
    ref = MetricRef(metric="outcome", subject=symbol, bounds=(0.0, 1.0))
    payoff = Binary(">=", 0.5, payout) if payout == 1.0 else Linear(scale=payout)
    return Instrument(
        symbol=symbol,
        spec=ContractSpec(
            contract_id=symbol,
            underlying=Single(ref),
            payoff=payoff,
            window=WINDOW,
            policy=DataPolicy(min_sample_size=1),
            reference_id="ref-test",
            published_at=WINDOW.start - timedelta(days=1),
            tick_size=tick,
        ),
    )


def solo_family(field: tuple[str, ...], key: str = "m1"):
    """A ten-way winner set, a field claim over three of them, and a count.

    The field claim is what makes a package interesting: it overlaps the winner
    claims, so a client can trade one against the others, and a maker that
    priced them independently would leave a gap between them. The count leg is
    bounded [0, 9] rather than [0, 1], which is what exercises the span in the
    learning rule and the bounds in the clamp.
    """
    outcomes = solo_outcomes(field)
    legs = list(winner_legs(outcomes, lambda o: f"WIN_{o[0]}"))
    trio = frozenset(field[:3])
    legs.append(field_leg("FIELD_TRIO", outcomes, trio))
    # A leg that is not a function of the outcome, stated as its expected
    # settlement given each outcome. The winner of a ten-way elimination is
    # credited more eliminations than a competitor who went out early, so the
    # conditional expectation rises with the outcome being this competitor.
    counts = tuple(4.0 if o[0] == field[0] else 0.7 for o in outcomes)
    legs.append(MatchLeg("ELIMS_0", counts, 0.0, 9.0))
    listing = MatchListing(key, outcomes, tuple(legs))
    listed = {leg.symbol: instrument(leg.symbol, leg.high) for leg in legs}
    return listing, listed


def maker(listing, listed, **kwargs) -> MatchMaker:
    return MatchMaker(
        AgentId("mm-match"), VENUE, listed, (listing,), **kwargs
    )


class Recorder:
    """Enough of a `SimulationContext` to see what the maker sends."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, object]] = []
        self.rng = random.Random(0)

    @property
    def now(self) -> int:
        return 0

    def send(self, recipient, message) -> None:
        self.sent.append((recipient, message))

    def request_wakeup(self, delay) -> None:
        pass


def submitted(recorder: Recorder) -> list[tuple[str, Side, int]]:
    out = []
    for _to, message in recorder.sent:
        if isinstance(message, SymbolCommand) and hasattr(message.command, "side"):
            command = message.command
            if command.price is not None:
                out.append((message.symbol, command.side, int(command.price)))
    return out


def prices(agent: MatchMaker, symbol: str) -> tuple[float, float]:
    """The maker's quote in settlement units."""
    bid, ask = agent.quote_ticks(symbol)
    tick = float(agent.instruments[symbol].tick_size)
    return bid * tick, ask * tick


def worst_case_package(agent: MatchMaker, listing: MatchListing) -> float:
    """The best a client can do against these quotes at its worst outcome.

    Zero when the maker cannot be arbitraged, because the empty package is
    always available and nothing beats it. Strictly positive is a free lunch and
    the value is what it is worth per unit of package.
    """
    symbols = sorted(listing.legs)
    payoffs = [listing.legs[s].payoff for s in symbols]
    quotes = [prices(agent, s) for s in symbols]
    n = len(symbols)
    outcomes = len(listing.outcomes)

    # x = [buy_0..buy_n-1, sell_0..sell_n-1, t]; minimise -t.
    cost = [0.0] * (2 * n) + [-1.0]
    rows, rhs = [], []
    for w in range(outcomes):
        row = [-(payoffs[i][w] - quotes[i][1]) for i in range(n)]
        row += [payoffs[i][w] - quotes[i][0] for i in range(n)]
        row += [1.0]
        rows.append(row)
        rhs.append(0.0)
    rows.append([1.0] * (2 * n) + [0.0])
    rhs.append(1.0)
    bounds = [(0.0, None)] * (2 * n) + [(None, None)]
    answer = linprog(cost, A_ub=rows, b_ub=rhs, bounds=bounds, method="highs")
    assert answer.status == 0, f"the arbitrage program did not solve: {answer.message}"
    return float(-answer.fun)


# --------------------------------------------------------------------------
# The property, enumerated
# --------------------------------------------------------------------------


def test_no_package_of_the_makers_own_quotes_wins_at_every_outcome():
    """Exhaustive over all ten solo outcomes, over 400 beliefs and books.

    This is the claim the class exists to support and it is checked by solving
    for the best package rather than by trying a few. Every leg is an
    expectation ``q . payoff`` under one strictly positive measure and every
    quote brackets its own fair value, so a package that paid at all ten
    outcomes would have to have positive expectation under ``q`` while costing
    at least its expectation, which cannot happen.

    Measured over 400 random trials spanning beliefs three orders of magnitude
    wide, inventories at twice the position limit, half-spreads from one tick to
    two hundred, and skew strengths to four: **maximum arbitrage exactly 0.0**,
    and every one of the 400 optima is exactly 0.0 rather than within a
    tolerance of it. The comparison worth having is what the same program finds
    against a maker whose inventory rule works per contract, which is the next
    test.
    """
    rng = random.Random(20260909)
    field = tuple(c.key for c in ROSTER)[:10]
    listing, listed = solo_family(field)
    worst = 0.0
    for _trial in range(400):
        agent = maker(
            listing,
            listed,
            half_spread=rng.choice([1, 4, 12, 40, 200]),
            position_limit=rng.choice([50, 400, 1200]),
            skew_strength=rng.choice([0.0, 0.5, 1.0, 4.0]),
            resolve_gain=rng.choice([0.0, 1.0, 2.0]),
        )
        # Beliefs spanning three orders of magnitude, so the test covers a match
        # most of the way through an elimination as well as an even field.
        for competitor in field:
            agent.weights[competitor] = 10 ** rng.uniform(-3, 0)
        limit = agent.position_limit
        for symbol in listing.legs:
            agent.position[symbol] = rng.randint(-2 * limit, 2 * limit)
        worst = max(worst, worst_case_package(agent, listing))
    assert worst == pytest.approx(0.0, abs=1e-9), (
        f"a package worth {worst:.6f} at its worst outcome exists in the maker's "
        "own quotes"
    )


def test_skewing_each_contract_on_its_own_is_what_a_free_lunch_looks_like():
    """The same program against the same maker, with the inventory rule swapped.

    Without a counterexample the zero above is not evidence, because a program
    that returns zero against everything would also return zero here. So this
    runs it against `MarketMaker`'s own inventory rule, which is the sensible
    thing to do on one book and the wrong thing to do on a family: a reservation
    price of ``anchor - inventory * skew_per_lot`` with the skew a fixed
    fraction of *that contract's* settlement range.

    Short the whole family at a 400 lot limit, which is where a maker that has
    been sold every claim in a match ends up. The leg that breaks is the count,
    bounded [0, 9] by its terms while this field can only produce 0.70 or 4.00,
    so 12% of its range is 108 ticks against a fair value of 352. The per
    contract rule bids **4.48 for a claim that cannot pay more than 4.00**, and
    the LP takes it: sell one unit, worst outcome **0.480**, riskless.

    `SurfaceMarketMaker` records the same shape on the option chain, where a
    range-sized skew moved a strike by 540 against a fair value near 70. It
    cannot happen here, and not because the number is smaller: the match level
    bid is at most the fair value, the fair value is a convex combination of the
    payoffs, so no bid can exceed the most the contract can pay at any outcome.
    The same short book under the tilt offers 0.000, because a uniform short
    across a partition is a constant exposure and a constant factors out.
    """
    field = tuple(c.key for c in ROSTER)[:10]
    listing, listed = solo_family(field)
    agent = maker(listing, listed, half_spread=4, position_limit=400)
    for symbol in listing.legs:
        agent.position[symbol] = -400

    assert worst_case_package(agent, listing) == pytest.approx(0.0, abs=1e-9)

    def per_contract_skew(symbol: str, fair: float, half: float) -> tuple[int, int]:
        low, high = agent.instruments[symbol].tick_bounds
        tick = float(agent.instruments[symbol].tick_size)
        span = float(int(high) - int(low))
        per_lot = span * agent.max_skew_fraction / agent.position_limit
        centre = fair / tick - agent.position.get(symbol, 0) * per_lot
        return (
            max(int(low), math.floor(centre - half)),
            min(int(high), math.ceil(centre + half)),
        )

    # The bid the tilt would never make: above every payoff the leg can produce.
    counts = listing.legs["ELIMS_0"].payoff
    good_bid, _good_ask = prices(agent, "ELIMS_0")
    assert good_bid <= max(counts)
    agent._bracket = per_contract_skew
    bad_bid, _bad_ask = prices(agent, "ELIMS_0")
    assert bad_bid > max(counts), (
        f"the per contract skew bid {bad_bid} against a top payoff of {max(counts)}"
    )

    broken = worst_case_package(agent, listing)
    assert broken > 0.4, (
        f"the per contract skew only leaked {broken:.3f}; if this ever reaches "
        "zero the comparison above has stopped being a comparison"
    )
    assert broken == pytest.approx(0.480, abs=0.02)


def test_the_centre_clamp_detaches_the_mid_from_the_fair_value():
    """Two claims worth different amounts marking at the same price.

    `CONTRIBUTING.md` lists clamping the centre of a quote into the settlement
    range among the changes that look like improvements and are not, because the
    mid becomes a function of the half-spread. It does not create a package
    arbitrage the way the per contract skew above does, and saying so is the
    point: it puts the bid at the floor, which is still under the fair value, so
    the bracket survives and the LP finds nothing. What it wrecks is the mark,
    and on a partition the mark is what says the set sums to one.

    Measured with six claims worth 0.0049 and four worth 0.2427, at a half
    spread of 8.66 ticks: under the centre clamp all six cheap claims mark at
    0.0900 whatever they are worth, so the marks sum to 1.5000 on a set certain
    to pay 1.0000. Under the per side clamp the same six mark at 0.0500, the
    sum is 1.2600, and every claim's mid still moves when its fair value does.
    """
    field = tuple(c.key for c in ROSTER)[:10]
    listing, listed = solo_family(field)
    agent = maker(listing, listed, half_spread=4, skew_strength=0.0)
    for index, competitor in enumerate(field):
        agent.weights[competitor] = 0.02 if index < 6 else 1.0
    winners = [f"WIN_{k}" for k in field]
    tick = float(listed[winners[0]].tick_size)
    half = agent.half_spread * agent.resolution("m1")

    def centre_clamped(symbol: str, fair: float, unused: float) -> tuple[int, int]:
        low, high = agent.instruments[symbol].tick_bounds
        centre = min(
            max(fair / tick, float(int(low)) + half), float(int(high)) - half
        )
        return math.floor(centre - half), math.ceil(centre + half)

    per_side = [prices(agent, s) for s in winners]
    agent._bracket = centre_clamped
    centred = [prices(agent, s) for s in winners]

    cheap = {round((b + a) / 2.0, 6) for b, a in centred[:6]}
    assert len(cheap) == 1, "the cheap claims did not collapse onto one mark"
    assert cheap.pop() == pytest.approx(half * tick, abs=tick)
    assert sum((b + a) / 2.0 for b, a in centred) > 1.4, (
        "the centre clamp did not inflate the marks of a partition"
    )
    # Under the per side clamp the six cheap claims still mark above their fair
    # value, because their bids are truncated at the floor, but by half the
    # truncation rather than by the whole half-spread.
    assert sum((b + a) / 2.0 for b, a in per_side) < sum(
        (b + a) / 2.0 for b, a in centred
    )


def test_every_quote_brackets_its_own_fair_value():
    """The one inequality the whole argument rests on, checked directly.

    ``bid <= fair <= ask`` after the outward rounding and after both clamps.
    Worth asserting separately from the linear program because it is the reason
    the program returns zero, and because a future change that breaks it would
    show up here as one line rather than as a number in an LP.
    """
    rng = random.Random(4)
    field = tuple(c.key for c in ROSTER)[:10]
    listing, listed = solo_family(field)
    for _trial in range(200):
        agent = maker(
            listing, listed, half_spread=rng.choice([1, 4, 60, 400]),
            skew_strength=rng.choice([0.0, 1.0, 3.0]),
        )
        for competitor in field:
            agent.weights[competitor] = 10 ** rng.uniform(-4, 0)
        for symbol in listing.legs:
            agent.position[symbol] = rng.randint(-900, 900)
        for symbol in sorted(listing.legs):
            bid, ask = prices(agent, symbol)
            fair = agent.fair(symbol)
            assert bid <= fair + 1e-12, f"{symbol} bids {bid} above its fair {fair}"
            assert ask >= fair - 1e-12, f"{symbol} offers {ask} below its fair {fair}"


# --------------------------------------------------------------------------
# Coherence of the mutually exclusive exhaustive set
# --------------------------------------------------------------------------


def test_the_winner_quotes_sum_to_one_and_the_residual_is_the_truncated_bid():
    """Ten claims, one survivor, so the fair values sum to exactly one.

    The *mids* do not, and the residual is neither floating point nor a
    mystery. It has exactly two sources and both have closed forms. A claim
    worth less than the half-spread has its bid truncated at the settlement
    floor while its offer is untouched, which lifts its mid by
    ``(half - fair) / 2``; a claim within a half-spread of the ceiling has its
    offer truncated, which lowers its mid by the mirror of that. And a bid
    rounds down onto the tick grid while an offer rounds up, which moves each
    mid by under half a tick either way. On a partition the floor truncation
    dominates, because a set of ten claims summing to one has most of its
    members cheap, so the residual is positive.

    Measured with a four tick half-spread. On an even field no clamp binds and
    the residual is **0.000000**, the mids summing to exactly 1.000000. With six
    of the ten claims below the half-spread, which is what a match looks like
    once it has started resolving, the residual is **0.260000** and every claim's
    mid lands within half a tick of the closed form below.

    None of that is tradeable and that is the point of asserting the two sums
    separately. A client cannot deal at a mid. Buying the whole exhaustive set
    costs the offers, which sum above the one unit it is certain to pay, and
    selling it collects the bids, which sum below.
    """
    field = tuple(c.key for c in ROSTER)[:10]
    listing, listed = solo_family(field)
    winners = [f"WIN_{k}" for k in field]
    tick = float(listed[winners[0]].tick_size)
    measured = {}

    for label, weights in (
        ("even field", {k: 1.0 for k in field}),
        (
            "part resolved",
            {k: (0.02 if i < 6 else 1.0) for i, k in enumerate(field)},
        ),
    ):
        agent = maker(listing, listed, half_spread=4, skew_strength=0.0)
        agent.weights.update(weights)

        fairs = [agent.fair(s) for s in winners]
        assert sum(fairs) == pytest.approx(1.0, abs=1e-12), (
            f"{label}: the fair values of a partition sum to {sum(fairs)}"
        )

        quotes = [prices(agent, s) for s in winners]
        mids = [(b + a) / 2.0 for b, a in quotes]
        residual = sum(mids) - 1.0
        measured[label] = residual

        half = agent.half_spread * agent.resolution("m1") * tick
        for symbol, fair, mid in zip(winners, fairs, mids):
            low, high = (float(b) for b in listed[symbol].value_bounds)
            predicted = (
                fair
                + max(0.0, low + half - fair) / 2.0
                - max(0.0, fair + half - high) / 2.0
            )
            assert mid == pytest.approx(predicted, abs=tick / 2.0), (
                f"{label}: {symbol} mids at {mid} against a truncation model "
                f"predicting {predicted}"
            )

        assert sum(b for b, _a in quotes) <= 1.0 + 1e-12, (
            f"{label}: the set could be sold to the maker for more than it pays"
        )
        assert sum(a for _b, a in quotes) >= 1.0 - 1e-12, (
            f"{label}: the set could be bought from the maker for less than it pays"
        )

    assert measured["even field"] == pytest.approx(0.0, abs=1e-12)
    assert measured["part resolved"] == pytest.approx(0.26, abs=0.01)


def test_a_field_claim_never_disagrees_with_the_names_inside_it():
    """One belief, so the subset and its members cannot come apart.

    The relation is the same shape as put-call parity on the option chain and it
    holds for the same reason: the field claim is not priced, it is the sum of
    the three winner claims under the identical measure, so it agrees to the
    last bit rather than to within a tolerance.
    """
    field = tuple(c.key for c in ROSTER)[:10]
    listing, listed = solo_family(field)
    agent = maker(listing, listed, skew_strength=0.0)
    for index, competitor in enumerate(field):
        agent.weights[competitor] = 0.3 + 0.4 * index

    parts = sum(agent.fair(f"WIN_{k}") for k in field[:3])
    assert agent.fair("FIELD_TRIO") == pytest.approx(parts, abs=1e-15)


# --------------------------------------------------------------------------
# Inventory, at the level of the match
# --------------------------------------------------------------------------


def test_long_the_whole_set_skews_by_exactly_nothing():
    """A flat book across a partition is riskless, so it must not move a price.

    This is the difference a maker skewing each contract by its own position
    cannot see. Long 300 lots of all ten winner claims pays 300 whatever
    happens, and the correct response is to keep quoting where it was. Skewing
    per contract instead would shade all ten down, which invents a directional
    view out of a position that has none, and would then push the ten quotes to
    sum to less than the one unit the set is certain to pay.

    Exactly nothing rather than nearly nothing, because a constant exposure
    factors out of the tilt's normalisation as an algebraic identity.
    """
    field = tuple(c.key for c in ROSTER)[:10]
    listing, listed = solo_family(field)
    agent = maker(listing, listed, skew_strength=1.0, position_limit=400)
    for index, competitor in enumerate(field):
        agent.weights[competitor] = 0.5 + 0.2 * index
    flat = [agent.fair(f"WIN_{k}") for k in field]

    for competitor in field:
        agent.position[f"WIN_{competitor}"] = 300
    held = [agent.fair(f"WIN_{k}") for k in field]
    assert held == pytest.approx(flat, abs=1e-15)
    assert agent.pressure("m1") == 0.0


def test_inventory_skew_has_the_right_sign_and_scales_with_the_bet():
    """Long lowers the price it is long, short raises it, and one name moves most.

    The comparison the task turns on: with the same 300 lots, spread across the
    whole partition it moves nothing, and concentrated in one name it moves that
    name. Measured on an even ten-way field at a 400 lot limit and unit skew
    strength: long the whole set 0.000000, long one name 0.050132, which is
    ``0.1`` falling to ``0.1 e^(-0.75) / (0.1 e^(-0.75) + 0.9)``.
    """
    field = tuple(c.key for c in ROSTER)[:10]
    listing, listed = solo_family(field)
    target = f"WIN_{field[0]}"

    agent = maker(listing, listed, skew_strength=1.0, position_limit=400)
    base = agent.fair(target)

    agent.position[target] = 300
    long_one = agent.fair(target)
    assert long_one < base, "a long position did not cheapen its own claim"

    agent.position[target] = -300
    short_one = agent.fair(target)
    assert short_one > base, "a short position did not lift its own claim"

    flat = maker(listing, listed, skew_strength=1.0, position_limit=400)
    for competitor in field:
        flat.position[f"WIN_{competitor}"] = 300
    whole_set = abs(flat.fair(target) - base)
    one_name = abs(long_one - base)
    assert whole_set < one_name, (
        f"long the whole set skewed {whole_set:.6f} and long one name "
        f"{one_name:.6f}; a maker that cannot tell those apart is pricing a "
        "riskless book as a directional one"
    )
    assert whole_set == pytest.approx(0.0, abs=1e-15)
    assert one_name == pytest.approx(0.050132, abs=5e-6)


def test_long_all_but_one_is_a_bet_against_that_one():
    """Because it is. The skew has to notice, and it has to point the right way.

    Nine tenths of a partition pays one unit unless the tenth wins, so holding it
    is short the tenth, and the tenth's quote has to rise. A per contract skew
    reads this position as nine independent longs and lowers nine prices while
    leaving the tenth alone, which is the opposite of what the book is.
    """
    field = tuple(c.key for c in ROSTER)[:10]
    listing, listed = solo_family(field)
    agent = maker(listing, listed, skew_strength=1.0, position_limit=400)
    odd = f"WIN_{field[-1]}"
    base = agent.fair(odd)
    for competitor in field[:-1]:
        agent.position[f"WIN_{competitor}"] = 300
    assert agent.fair(odd) > base, "long nine of ten did not lift the tenth"


# --------------------------------------------------------------------------
# The bounded payoff
# --------------------------------------------------------------------------


def test_no_quote_ever_leaves_the_contracts_settlement_range():
    """Checked on what is sent, not on what is computed.

    A quote outside the range is an order the contract cannot pay, and the venue
    would refuse to collateralise it. Driven at half-spreads far wider than the
    whole instrument and at inventories past the position limit, because a bound
    that only holds in a calm state is the bug class `CONTRIBUTING.md` calls a
    guard tested in the wrong state.
    """
    rng = random.Random(11)
    field = tuple(c.key for c in ROSTER)[:10]
    listing, listed = solo_family(field)
    checked = 0
    for _trial in range(60):
        agent = maker(
            listing,
            listed,
            half_spread=rng.choice([1, 4, 500, 5000]),
            skew_strength=rng.choice([0.0, 1.0, 8.0]),
            position_limit=rng.choice([25, 400]),
        )
        for competitor in field:
            agent.weights[competitor] = 10 ** rng.uniform(-6, 0)
        for symbol in listing.legs:
            agent.position[symbol] = rng.randint(-3000, 3000)
        recorder = Recorder()
        agent.act(recorder)
        for symbol, side, price in submitted(recorder):
            low, high = agent.instruments[symbol].tick_bounds
            assert int(low) <= price <= int(high), (
                f"{symbol} quoted {price} outside {int(low)}..{int(high)}"
            )
            # And it is the same price the linear program was run against.
            # `TradingAgent.quote` clamps and snaps to the tick grid on the way
            # out, so a proof about `quote_ticks` is a proof about the orders
            # only while nothing in between moves them.
            bid, ask = agent.quote_ticks(symbol)
            assert price == (bid if side is Side.BUY else ask), (
                f"{symbol} sent {price} where it computed {(bid, ask)}"
            )
            checked += 1
        for symbol in listing.legs:
            bid, ask = agent.quote_ticks(symbol)
            low, high = agent.instruments[symbol].tick_bounds
            assert int(low) <= bid <= int(high)
            assert int(low) <= ask <= int(high)
    assert checked > 500, f"only {checked} orders were seen; the test proves little"


def test_a_wider_span_is_not_a_wider_belief():
    """The [0, 9] leg prices off the same measure as the [0, 1] ones.

    The count contract is the one whose settlement range is not the unit
    interval, so it is where a maker that carried a range into its belief would
    show it. Its fair value is the dot product and nothing else.
    """
    field = tuple(c.key for c in ROSTER)[:10]
    listing, listed = solo_family(field)
    agent = maker(listing, listed, skew_strength=0.0)
    for index, competitor in enumerate(field):
        agent.weights[competitor] = 1.0 + index
    measure = agent.measure("m1")
    expected = sum(q * v for q, v in zip(measure, listing.legs["ELIMS_0"].payoff))
    assert agent.fair("ELIMS_0") == pytest.approx(expected, abs=1e-15)
    low, high = listed["ELIMS_0"].value_bounds
    assert float(low) == 0.0 and float(high) == 9.0


# --------------------------------------------------------------------------
# Widening as the match resolves
# --------------------------------------------------------------------------


def test_the_quote_widens_as_the_outcome_set_collapses():
    """Because the repricing per elimination grows 7.07x, and it was measured.

    Measured on seed 7 over 2,000 solo matches, the mean L1 revision of the
    winner probabilities at each elimination runs 0.1401, 0.1787, 0.2154,
    0.2568, 0.3090, 0.3818, 0.4789, 0.6578, 0.9900 from the first elimination to
    the last. A maker holding one width across that is quoting the first
    elimination's adverse selection into the last one's, and the last one is
    where the repricing is seven times larger.

    The statistic it widens on is the effective number of outcomes still live,
    ``exp(H(p))``, which correlates 0.8605 with that realised revision sample by
    sample and rises 4.79x over the same span. Twice the Simpson concentration
    correlates 0.8537 and the exact reverse-Luce expectation 0.8567, so the
    entropy is not being preferred on elegance. It under-responds by a third at
    the final elimination and is left at a gain of one anyway, because a gain
    chosen to recover 7.07 would be a gain fitted to seed 7.

    On an even belief over ``m`` live outcomes the multiplier is exactly
    ``10 / m``, which is what this asserts.
    """
    field = tuple(c.key for c in ROSTER)[:10]
    listing, listed = solo_family(field)
    agent = maker(listing, listed, half_spread=4, skew_strength=0.0)

    widths, multipliers = [], []
    for live in range(10, 1, -1):
        for index, competitor in enumerate(field):
            agent.weights[competitor] = 1.0 if index < live else 1e-9
        multipliers.append(agent.resolution("m1"))
        bid, ask = agent.quote_ticks(f"WIN_{field[0]}")
        widths.append(ask - bid)

    assert multipliers == sorted(multipliers), "the width did not follow the collapse"
    assert widths == sorted(widths)
    assert multipliers[0] == pytest.approx(1.0, abs=1e-6)
    assert multipliers[-1] == pytest.approx(5.0, abs=1e-3)
    for live, got in zip(range(10, 1, -1), multipliers):
        assert got == pytest.approx(10.0 / live, rel=1e-3)


def test_widening_near_settlement_does_not_break_the_bracket():
    """Because a wide quote is still clamped per side and still brackets.

    Worth its own test: widening is exactly the moment a clamp starts binding,
    since a claim near one and a half-spread of five ticks puts the offer over
    the ceiling. That is where a centre clamp would have been reintroduced.
    """
    field = tuple(c.key for c in ROSTER)[:10]
    listing, listed = solo_family(field)
    agent = maker(listing, listed, half_spread=20, skew_strength=0.0)
    for index, competitor in enumerate(field):
        agent.weights[competitor] = 1.0 if index == 0 else 1e-6
    for symbol in sorted(listing.legs):
        bid, ask = prices(agent, symbol)
        fair = agent.fair(symbol)
        low, high = agent.instruments[symbol].value_bounds
        assert bid <= fair <= ask
        assert float(low) <= bid <= float(high)
        assert float(low) <= ask <= float(high)


# --------------------------------------------------------------------------
# Learning from the tape
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed, expected", [(7, 0.0029), (3, 0.0033)])
def test_the_belief_converges_to_the_realised_win_rates(seed, expected):
    """4,000 solo matches, from an even field, on settlement prints alone.

    The baseline this is scored against is the simulation's own: over 4,000 solo
    matches on seed 7 the win rate per appearance ranges 0.036 to 0.178 against
    an even-field 0.100, so a maker that learned nothing would be wrong by up to
    0.078 on a contract bounded in [0, 1].

    The update is exponentiated gradient on the squared pricing error and it
    starts from an even field, because the strengths that decide the answer are
    latent by design and nothing here may read them. Measured: **maximum
    absolute error 0.0029 on seed 7 and 0.0033 on seed 3**, mean 0.0010 and
    0.0014. Seeds 11 and 41 give 0.0030 and 0.0021 and are left out only because
    two seeds already say the figure is not one seed's luck.

    The ranking is asserted pairwise and not as a total order, and the reason is
    a measurement rather than a concession. On seed 7 the maker's order is the
    realised order exactly. On seed 3 it swaps one pair, HALCYON and WISP, whose
    realised rates differ by **0.0003 against a standard error of 0.0063** on
    roughly 3,330 appearances each. Those two are twenty times closer together
    than the noise in the frequency they are being ranked against, so the
    realised order is not evidence about them and a test that demanded the maker
    reproduce it would be asserting that the maker match one sample's noise.
    Seed 11 has one such pair at a gap of 0.0004 and seed 41 has none. Every
    pair separated by more than one standard error is ordered correctly on all
    four seeds, and that is what this checks.

    The smallest normalised weight the rule ever produced over a run was 0.0185,
    0.0241, 0.0334 and 0.0335 on the four seeds. That is asserted too, and not
    as tidiness: the no-arbitrage argument needs the belief to have full
    support, and this is the measurement that says the multiplicative rule
    delivers it without a floor to clamp it at.
    """
    matches = 4000
    keys = [c.key for c in ROSTER]
    listed = {f"WIN_{k}": instrument(f"WIN_{k}") for k in keys}
    agent = MatchMaker(AgentId("mm-learn"), VENUE, listed)

    implied = {k: 0.0 for k in keys}
    appear = {k: 0 for k in keys}
    wins = {k: 0 for k in keys}
    smallest = 1.0

    for number in range(matches):
        result = play(seed, number, "solo")
        outcomes = solo_outcomes(result.field)
        listing = MatchListing(f"solo:{number}", outcomes)
        for leg in winner_legs(outcomes, lambda o: f"WIN_{o[0]}"):
            listing.add(leg)
        agent.list_match(listing)

        belief = agent.belief(listing.key)
        smallest = min(smallest, min(belief))
        for competitor, probability in zip(result.field, belief):
            implied[competitor] += probability
            appear[competitor] += 1
        wins[result.winners[0]] += 1

        # One settlement print per claim: the winner's printed at one and every
        # other at zero, which is what the closing print on a settled binary is.
        for competitor in result.field:
            agent.learn(
                f"WIN_{competitor}", 1.0 if competitor == result.winners[0] else 0.0
            )
        agent.close_match(listing.key)

    errors = {k: implied[k] / appear[k] - wins[k] / appear[k] for k in keys}
    worst = max(abs(e) for e in errors.values())
    mean = sum(abs(e) for e in errors.values()) / len(errors)
    assert worst < 0.006, f"implied rates miss the realised ones by {worst:.4f}"
    assert mean < 0.002, f"mean miss {mean:.4f}"
    assert worst == pytest.approx(expected, abs=0.0012)

    rate = {k: wins[k] / appear[k] for k in keys}
    guess = {k: implied[k] / appear[k] for k in keys}
    separated = 0
    for index, first in enumerate(keys):
        for second in keys[index + 1 :]:
            error = math.sqrt(
                rate[first] * (1 - rate[first]) / appear[first]
                + rate[second] * (1 - rate[second]) / appear[second]
            )
            if abs(rate[first] - rate[second]) <= error:
                continue
            separated += 1
            assert (guess[first] - guess[second]) * (
                rate[first] - rate[second]
            ) > 0, (
                f"{first} and {second} finished {rate[first]:.4f} and "
                f"{rate[second]:.4f}, a gap of "
                f"{abs(rate[first]-rate[second]):.4f} against a standard error "
                f"of {error:.4f}, and the maker ranks them the other way"
            )
    assert separated > 50, f"only {separated} pairs were separable; too weak a claim"

    # A maker that had not learned would be flat at 0.100 and would be wrong by
    # more than 0.06 on the strongest name, so the assertions above have to be
    # measured against something a do-nothing maker fails.
    assert max(abs(0.1 - rate[k]) for k in keys) > 0.06

    assert smallest > 0.015, (
        f"the belief fell to {smallest:.4f}; full support is what the "
        "no-arbitrage argument rests on and the update is supposed to keep it "
        "without a clamp"
    )


def test_a_print_moves_the_belief_in_the_direction_of_the_print():
    """And moves the rest of the set the other way, because it has to.

    The property that makes this an update to a belief rather than to a price. A
    print above the maker's fair on one claim in a partition is evidence that
    the other nine are worth less between them, and a maker that raised one
    without lowering the others would have a set summing above one, which is the
    arbitrage this class exists to prevent arriving through the learning rule.
    """
    field = tuple(c.key for c in ROSTER)[:10]
    listing, listed = solo_family(field)
    agent = maker(listing, listed)
    before = agent.belief("m1")

    agent.learn(f"WIN_{field[0]}", 0.9)
    after = agent.belief("m1")
    assert after[0] > before[0]
    assert all(a < b for a, b in zip(after[1:], before[1:]))
    assert sum(after) == pytest.approx(1.0, abs=1e-12)

    agent.learn(f"WIN_{field[0]}", 0.0)
    lower = agent.belief("m1")
    assert lower[0] < after[0]


def test_a_match_teaches_nothing_about_the_competitors_who_sat_it_out():
    """The rescaling earns its place here.

    A weight per competitor is what lets the maker carry what it learned into a
    match with a different field, which is the whole reason it is a weight
    rather than a probability. That only works if a match leaves the absentees
    where they were. Without the rescale, the field's weights drift against the
    rest of the roster on every print and a competitor's standing changes
    because of a match it was not in.
    """
    keys = [c.key for c in ROSTER]
    field = tuple(keys[:10])
    absent = keys[10:]
    listing, listed = solo_family(field)
    agent = maker(listing, listed)
    for index, competitor in enumerate(keys):
        agent.weights[competitor] = 1.0 + 0.1 * index

    before = {k: agent.weights[k] for k in absent}
    field_total = sum(agent.weights[k] for k in field)
    for _print in range(50):
        agent.learn(f"WIN_{field[0]}", 0.9)
        agent.learn(f"WIN_{field[4]}", 0.05)
    assert {k: agent.weights[k] for k in absent} == before
    assert sum(agent.weights[k] for k in field) == pytest.approx(field_total, rel=1e-9)


def test_a_print_arriving_through_the_kernel_path_reaches_the_belief():
    """`on_print` is the only wire from the tape to the belief, so it is tested.

    `CONTRIBUTING.md` records two agents whose handlers ran and booked nothing
    because an input was never supplied, and counts the shape as its own bug
    class. The counter and this test are what make the wire visible: a maker
    whose belief never moves quotes its prior all session and looks fine doing
    it.
    """
    from arena.exchange.types import Quantity, SequenceNumber
    from arena.sim.messages import TradePrint
    from arena.sim.time import Timestamp

    field = tuple(c.key for c in ROSTER)[:10]
    listing, listed = solo_family(field)
    agent = maker(listing, listed)
    before = agent.belief("m1")

    symbol = f"WIN_{field[0]}"
    # 0.90 in a contract ticking at 0.01, which is the conversion the handler
    # has to do and the one place a unit crosses this boundary.
    agent.on_print(
        Recorder(),
        TradePrint(
            symbol, Timestamp(0), SequenceNumber(1), Price(90), Quantity(5), Side.BUY
        ),
    )
    assert agent.learned[listing.key] == 1
    assert agent.belief("m1")[0] > before[0]


# --------------------------------------------------------------------------
# Everything else the maker is
# --------------------------------------------------------------------------


def test_a_symbol_with_no_listing_falls_through_to_the_plain_maker():
    """A match contract with no belief behind it must not be invented one.

    The plain maker anchors an unpriced book at the middle of its settlement
    range, which on a claim paying zero or one is a standing 0.50 for the
    session. That is wrong, and it is still better than pricing it off a belief
    that does not cover it, so the fallback is explicit rather than accidental.
    """
    field = tuple(c.key for c in ROSTER)[:10]
    listing, listed = solo_family(field)
    listed["UNRELATED"] = instrument("UNRELATED")
    agent = maker(listing, listed)
    assert agent.fair("UNRELATED") is None
    assert agent.quote_ticks("UNRELATED") is None

    recorder = Recorder()
    agent.act(recorder)
    quoted = {symbol for symbol, _side, _price in submitted(recorder)}
    assert "UNRELATED" in quoted, "the fallback never quoted the unlisted symbol"
    assert f"WIN_{field[0]}" in quoted


def test_a_team_format_is_the_same_maker_with_a_coarser_outcome_set():
    """Two sides of three, so the weights stay per competitor and the belief is 2-way.

    The generalisation is not decoration: it is what lets one learned weight
    price a name in a solo match and inside a team, and it is exactly how
    `match.play` composes a side's strength, by summing over the members. The
    partition property is unchanged, so the no-arbitrage argument is unchanged
    with it, which is what this checks.
    """
    fmt = FORMATS["objective"]
    result = play(7, 0, "objective")
    outcomes = team_outcomes(result.field, fmt.team_size)
    assert len(outcomes) == 2 and all(len(o) == 3 for o in outcomes)

    legs = winner_legs(outcomes, lambda o: f"SIDE_{o[0]}")
    listing = MatchListing("obj:0", outcomes, legs)
    listed = {leg.symbol: instrument(leg.symbol) for leg in legs}
    agent = MatchMaker(AgentId("mm-obj"), VENUE, listed, (listing,))

    for index, competitor in enumerate(result.field):
        agent.weights[competitor] = 1.0 + index
    first = sum(agent.weights[k] for k in outcomes[0])
    total = sum(agent.weights[k] for k in result.field)
    assert agent.belief("obj:0")[0] == pytest.approx(first / total, abs=1e-15)
    assert sum(agent.fair(leg.symbol) for leg in legs) == pytest.approx(1.0, abs=1e-12)
    assert worst_case_package(agent, listing) == pytest.approx(0.0, abs=1e-9)


def test_a_listing_refuses_an_outcome_set_that_is_not_a_partition():
    """Because a belief over an overlapping set is not a belief about anything.

    `modes.py` makes the same argument about ties: a market on an ambiguous
    question cannot be settled by arithmetic. A competitor in two outcomes would
    let the prices of a supposedly exhaustive set sum to something other than
    one while every individual quote looked right.
    """
    with pytest.raises(ValueError, match="mutually exclusive"):
        MatchListing("bad", (("A", "B"), ("B", "C")))
    with pytest.raises(ValueError, match="fewer than two"):
        MatchListing("bad", (("A",),))
    with pytest.raises(ValueError, match="outside its declared range"):
        MatchLeg("X", (0.0, 2.0), 0.0, 1.0)
