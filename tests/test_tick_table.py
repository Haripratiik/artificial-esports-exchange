"""One contract, several increments: the tiered tick table.

A tick has two jobs and they pull against each other. Too fine and queue
priority is worthless -- anyone can step in front of a resting order for a
hundredth of a penny, so nobody posts size and the depth that makes a market
usable never appears. Too coarse and the spread cannot narrow to what the
market actually knows, so the last part of the information in the price has
nowhere to go. The resolution that is right at a price of 4 is wrong at 4,000,
which is why a venue lets the increment grow with the price rather than
picking one number and living with it at both ends of the range.

What the table is *not* is a change of unit. ``tick_size`` stays the finest
increment and the unit every price is represented in, because the matching
engine counts in integer ticks: a variable unit would make tick index 16,001
mean one price near the bottom of a contract's range and a different price
near the top, and order ids, price bands, marks and settlement figures are all
denominated in those ticks. So the table is a rule about which prices may be
*quoted*, enforced by the venue at the door, and every increment in it has to
be a whole multiple of the base -- otherwise the rule would forbid prices the
representation can express, which is a rule nobody could follow.

The feature arrived with no tests. ``VANTA_OBJECTIVE_WR`` is the one listed
contract that carries a table, a quarter of a point up to 4,000 and a whole
point above it, so the last tests here run the live market and check the rule
against real order flow rather than against hand-written orders. It is a team
win rate rather than an individual one for a reason the listing gives: a team
rate settles near 5,000 and an individual one near 1,100, so a table
thresholded at 4,000 would never fire on the individual side.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest

from arena.agents.base import _on_grid
from arena.contracts.payoff import Linear
from arena.contracts.spec import ContractSpec, DataPolicy, ObservationWindow
from arena.contracts.underlying import Difference, MetricRef, Single
from arena.exchange.events import Acknowledged, Replace, Submit
from arena.exchange.types import (
    AgentId,
    OrderType,
    Price,
    Quantity,
    RejectReason,
    Side,
    TimeInForce,
)
from arena.market.instrument import Instrument
from arena.portfolio.money import from_money
from arena.market.venue import Venue
from arena.sim.time import seconds

UTC = timezone.utc
WINDOW = ObservationWindow(
    datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 9, 30, tzinfo=UTC)
)
MAKER, TAKER = AgentId("maker"), AgentId("taker")
SYM = "SYM"

# The shape the one listed contract uses: a quarter of a point up to 4,000 and
# a whole point above it.
LISTED = (("4000.00", "1.00"),)
# Three bands, so "the coarsest row the price has reached" is a real choice
# rather than a two-way branch that a wrong loop would still pass.
LADDER = (("10.00", "0.50"), ("100.00", "1.00"), ("1000.00", "5.00"))


def _wr(subject: str) -> Single:
    return Single(MetricRef("adjusted_win_rate", subject, bounds=(0.0, 1.0)))


def spec(
    *,
    tick: str = "0.25",
    table: tuple[tuple[str, str], ...] = (),
    underlying=None,
    contract_id: str = SYM,
) -> ContractSpec:
    return ContractSpec(
        contract_id=contract_id,
        underlying=_wr("A") if underlying is None else underlying,
        payoff=Linear(10_000.0),
        window=WINDOW,
        policy=DataPolicy(min_sample_size=1),
        reference_id="ref-1",
        published_at=WINDOW.start - timedelta(days=1),
        tick_size=tick,
        tick_table=table,
    )


def future(**kwargs) -> Instrument:
    return Instrument(SYM, spec(**kwargs))


def venue_with(instrument: Instrument) -> Venue:
    venue = Venue(starting_cash=100_000_000)
    venue.list_instrument(instrument)
    return venue


def limit(agent: AgentId, side: Side, price: Price, quantity: int = 1) -> Submit:
    return Submit(
        agent, side, Quantity(quantity), price, OrderType.LIMIT, TimeInForce.GTC
    )


def market_order(agent: AgentId, side: Side, quantity: int = 1) -> Submit:
    return Submit(
        agent, side, Quantity(quantity), None, OrderType.MARKET, TimeInForce.IOC
    )


def reasons(events) -> list[RejectReason]:
    return [event.reason for event in events if hasattr(event, "reason")]


def resting(book, instrument: Instrument) -> list[D]:
    """Every priced level on both sides, as contract prices.

    ``priced_bids`` and ``priced_asks`` rather than the raw levels: a
    market-on-open order rests at a sentinel price during a call phase so that
    it crosses every candidate the auction considers, and that sentinel is not
    a price anyone quoted.
    """
    levels = book.priced_bids + book.priced_asks
    return [instrument.from_ticks(price) for price, _size in levels]


# --------------------------------------------------------------------------
# Which increment applies where
# --------------------------------------------------------------------------


def test_below_the_first_threshold_the_increment_is_the_base_tick():
    """The table adds bands above a price; it does not replace the base one.

    An implementation that read the first row as the increment everywhere, or
    that started its search from the coarsest row, would coarsen the whole
    contract -- and the bottom of the range is exactly where a fine tick is
    doing its work.
    """
    tiered = future(table=LISTED)
    assert tiered.increment_at(D("0.25")) == D("0.25")
    assert tiered.increment_at(D("1250.50")) == D("0.25")
    assert tiered.increment_at(D("3999.75")) == D("0.25")


def test_the_threshold_price_itself_belongs_to_the_coarser_band():
    """The boundary is inclusive, and which side it falls on has to be pinned.

    A strict ``>`` here would leave exactly one quotable price -- the threshold
    -- governed by the fine tick while its neighbours a quarter away are
    governed by the coarse one. Nobody would notice until an order landed
    there, and then the answer would depend on which of two functions was
    asked.
    """
    tiered = future(table=LISTED)
    assert tiered.increment_at(D("3999.75")) == D("0.25")
    assert tiered.increment_at(D("4000.00")) == D("1.00")
    assert tiered.increment_at(D("4000.25")) == D("1.00")


@pytest.mark.parametrize(
    ("price", "increment"),
    [
        ("9.75", "0.25"),
        ("10.00", "0.50"),
        ("99.50", "0.50"),
        ("100.00", "1.00"),
        ("999.00", "1.00"),
        ("1000.00", "5.00"),
        ("8000.00", "5.00"),
    ],
)
def test_a_price_takes_the_coarsest_band_it_has_reached(price, increment):
    """With three bands, a loop that stops at the first match is wrong.

    Two bands cannot tell "the last row whose threshold the price has passed"
    apart from "the first row whose threshold the price has passed", so a
    contract with a middle tier is the only thing that catches a search which
    breaks early.
    """
    assert future(table=LADDER).increment_at(D(price)) == D(increment)


def test_a_contract_with_no_table_has_one_increment_at_every_price():
    """Every contract but one is in this case, so it is the case to protect.

    A table-aware lookup that mishandled the empty table -- indexing row zero,
    or defaulting to something other than the base tick -- would change the
    grid of twenty-odd listed contracts that never asked for a tiered tick.
    """
    plain = future()
    for price in ("0.25", "4000.00", "9999.75"):
        assert plain.increment_at(D(price)) == D("0.25")
        assert plain.on_grid(D(price))


def test_the_band_a_price_falls_in_follows_its_size_not_its_sign():
    """A spread trades on both sides of zero and its bands have to be symmetric.

    Comparing a signed price against the thresholds would leave every negative
    price in the finest band, so a spread at -4,000 would be quotable in
    quarters while the same distance from zero on the other side was quotable
    in points. The reason to coarsen is the magnitude of the number, and
    magnitude has no sign.
    """
    spread = Instrument(
        SYM, spec(table=LISTED, underlying=Difference(_wr("A"), _wr("B")))
    )
    assert spread.increment_at(D("-3999.75")) == D("0.25")
    assert spread.increment_at(D("-4000.00")) == D("1.00")
    assert spread.on_grid(D("-4004.00"))
    assert not spread.on_grid(D("-4000.25"))


def test_the_same_fraction_is_quotable_below_the_threshold_and_not_above_it():
    """The whole point of the feature, stated as one comparison.

    If both of these answered the same way the table is doing nothing, whether
    because it was dropped on the way to the instrument or because the lookup
    ignores the price it was handed.
    """
    tiered = future(table=LISTED)
    assert tiered.on_grid(D("3990.25"))
    assert not tiered.on_grid(D("4000.25"))


def test_a_price_off_its_band_grid_is_still_a_whole_number_of_base_ticks():
    """The table constrains quoting, not representation, and the two differ.

    4,000.25 is a perfectly good tick index -- 16,001 -- and the engine can
    match, mark and settle it. What it cannot be is *quoted* on this contract.
    Conflating the two would mean either an engine that cannot represent half
    its own price range, or a listing rule that nothing enforces.
    """
    tiered = future(table=LISTED)
    assert int(tiered.to_ticks(D("4000.25"))) == 16_001
    assert not tiered.on_grid(D("4000.25"))


# --------------------------------------------------------------------------
# The venue enforces it
# --------------------------------------------------------------------------


def test_the_venue_accepts_a_limit_price_on_its_bands_grid():
    """The permissive half of the rule, which a too-eager check would break.

    A grid test that compared against the base tick, or that used the
    increment at the wrong price, would refuse legitimate orders -- and an
    order refused for a price the contract does allow is worse than one
    allowed off the grid, because the agent has no way to comply.
    """
    tiered = future(table=LISTED)
    venue = venue_with(tiered)
    events = venue.submit(
        MAKER, SYM, limit(MAKER, Side.BUY, tiered.to_ticks(D("4000.00")))
    )
    assert RejectReason.INVALID_PRICE not in reasons(events)
    assert resting(venue.engine(SYM).book.snapshot(), tiered) == [D("4000.00")]


def test_the_venue_refuses_a_limit_price_off_its_bands_grid():
    """A rule the venue does not enforce is documentation, not a listing rule.

    ``increment_at`` and ``on_grid`` can both be right while nothing ever
    calls them, in which case the table exists only in the spec and agents
    quote through it freely.
    """
    tiered = future(table=LISTED)
    venue = venue_with(tiered)
    events = venue.submit(
        MAKER, SYM, limit(MAKER, Side.BUY, tiered.to_ticks(D("4000.25")))
    )
    assert reasons(events) == [RejectReason.INVALID_PRICE]


def test_an_order_refused_for_its_price_does_not_rest():
    """A rejection that still reaches the book is the worst of both answers.

    The agent is told no and the venue quotes the price anyway, so the tape
    and the event stream disagree about what happened. Checking the book as
    well as the event is what separates a refusal from a warning.
    """
    tiered = future(table=LISTED)
    venue = venue_with(tiered)
    venue.submit(MAKER, SYM, limit(MAKER, Side.BUY, tiered.to_ticks(D("4000.25"))))
    snapshot = venue.engine(SYM).book.snapshot()
    assert snapshot.priced_bids == ()
    assert snapshot.priced_asks == ()


def test_a_market_order_names_no_price_and_so_cannot_be_off_grid():
    """The check has to read the price before deciding, not assume there is one.

    A market order carries ``None``, so a grid test written as though every
    order had a price either crashes on the conversion or refuses the order
    for a price it never named. Either way the venue stops accepting the one
    order type that exists precisely to leave the price to the book.
    """
    tiered = future(table=LISTED)
    venue = venue_with(tiered)
    venue.submit(MAKER, SYM, limit(MAKER, Side.BUY, tiered.to_ticks(D("4000.00"))))
    events = venue.submit(TAKER, SYM, market_order(TAKER, Side.SELL))
    assert reasons(events) == []
    assert venue.account(TAKER).position(SYM).quantity == -1


# --------------------------------------------------------------------------
# The table is part of the contract
# --------------------------------------------------------------------------


def test_two_specs_differing_only_in_their_tick_table_are_different_contracts():
    """The grid is a term, so it has to move the digest like every other term.

    Every price ever printed against a contract was an opinion about the terms
    as written, and the set of prices that could be printed at all is one of
    those terms. Leaving the table out of ``to_dict`` would let a venue
    re-grid a live contract while the settlement record went on claiming it
    was the same one.
    """
    plain = spec()
    tiered = spec(table=LISTED)
    assert plain.spec_digest != tiered.spec_digest
    assert spec(table=LISTED).spec_digest == tiered.spec_digest


# --------------------------------------------------------------------------
# What the table may not say
# --------------------------------------------------------------------------


def test_an_increment_that_is_not_a_whole_multiple_of_the_base_tick_is_refused():
    """A rule nobody could follow is worse than no rule.

    With a base tick of 0.25 and a band increment of 0.30, the quotable prices
    the band describes -- 4,000.30, 4,000.60 -- are not prices the engine can
    represent, because it counts in quarter-point ticks. Every order in that
    band would be refused whatever the agent did. Caught at construction
    because the alternative is discovering it from a live book that cannot be
    quoted.
    """
    with pytest.raises(ValueError, match="not a multiple of the base tick"):
        spec(table=(("4000.00", "0.30"),))


@pytest.mark.parametrize("increment", ["0.00", "-1.00"])
def test_a_non_positive_increment_is_refused(increment):
    """Zero is a modulus that raises; negative is one that silently misjudges.

    ``on_grid`` divides by the increment, so a zero step turns the venue's
    price check into an exception on the order path, and a negative one makes
    "on the grid" mean something nobody intended. Neither is a listing rule.
    """
    with pytest.raises(ValueError, match="must be positive"):
        spec(table=(("4000.00", increment),))


@pytest.mark.parametrize(
    "table",
    [
        (("4000.00", "1.00"), ("4000.00", "5.00")),
        (("4000.00", "1.00"), ("2000.00", "5.00")),
    ],
    ids=["equal thresholds", "descending thresholds"],
)
def test_thresholds_that_do_not_ascend_are_refused(table):
    """The lookup takes the last matching row, so order is meaning, not style.

    An out-of-order table does not fail loudly -- it quietly resolves to
    whichever row happens to be last, so a contract would get a grid nobody
    wrote down. Since the reading depends on the order, the order has to be a
    checked property of the table rather than a convention.
    """
    with pytest.raises(ValueError, match="must ascend"):
        spec(table=table)


# --------------------------------------------------------------------------
# Snapping a quote onto the grid
# --------------------------------------------------------------------------


def test_a_bid_snaps_down_and_an_offer_snaps_up():
    """Snapping must cost the agent a fraction of a tick, never buy it a trade.

    Rounding to nearest -- the obvious implementation -- moves half of all
    quotes toward the touch, so an agent that meant to bid 4,000.25 crosses at
    4,001 instead and pays for a fill it did not ask for. The direction of the
    rounding is the entire safety property.
    """
    tiered = future(table=LISTED)
    off = tiered.to_ticks(D("4000.75"))
    assert tiered.from_ticks(_on_grid(tiered, Side.BUY, off)) == D("4000.00")
    assert tiered.from_ticks(_on_grid(tiered, Side.SELL, off)) == D("4001.00")


def test_a_price_already_on_the_grid_is_returned_unchanged():
    """Snapping is a correction, and a correction applied to nothing is a bug.

    An unconditional round-up on the offer side would push every valid ask a
    whole point away from the market, which widens the spread for no reason
    and would look like the market maker declining to compete.
    """
    tiered = future(table=LISTED)
    for price in ("3999.75", "4000.00", "4001.00"):
        ticks = tiered.to_ticks(D(price))
        assert _on_grid(tiered, Side.BUY, ticks) == ticks
        assert _on_grid(tiered, Side.SELL, ticks) == ticks


def test_snapping_never_makes_a_quote_more_aggressive():
    """Swept across the boundary, where an off-by-one band lookup would show.

    The two spot checks above pass on a snap that reads the increment at the
    wrong price -- at the original rather than the snapped one, say. Walking
    every tick from below 4,000 to well above it also pins the third property
    the spot checks leave out: the result is always on the grid, and always
    less than one increment from where the agent aimed.
    """
    tiered = future(table=LISTED)
    for tick in range(15_980, 16_041):
        wanted = tiered.from_ticks(Price(tick))
        step = tiered.increment_at(wanted)

        bid = tiered.from_ticks(_on_grid(tiered, Side.BUY, Price(tick)))
        assert bid <= wanted, f"a bid at {wanted} snapped up to {bid}"
        assert tiered.on_grid(bid), f"a bid at {wanted} snapped to {bid}, off grid"
        assert wanted - bid < step

        ask = tiered.from_ticks(_on_grid(tiered, Side.SELL, Price(tick)))
        assert ask >= wanted, f"an offer at {wanted} snapped down to {ask}"
        assert tiered.on_grid(ask), f"an offer at {wanted} snapped to {ask}, off grid"
        assert ask - wanted < step


def test_snapping_leaves_a_uniform_tick_contract_alone():
    """Every listed contract but one goes through this path on every quote.

    The helper runs on the hot path for the whole population, so it takes a
    fast exit when the increment is the base tick. That exit is only safe
    because there is nothing to correct there, and this is the assertion that
    says so.
    """
    plain = future()
    for tick in (1, 16_001, 39_999):
        for side in (Side.BUY, Side.SELL):
            assert _on_grid(plain, side, Price(tick)) == Price(tick)


# --------------------------------------------------------------------------
# The rule against real order flow
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_market():
    """The listed market, run for four minutes of simulated time.

    Long enough for the opening auction to clear and for the makers, informed
    traders and noise to work the books, so the grid is being tested against
    orders nobody wrote for this file.
    """
    from dashboard.build_market import build

    market = build(seed=7)
    market.kernel.start()
    market.kernel.advance(until=seconds(240))
    return market


def prices_on(market, symbol: str) -> list[D]:
    instrument = market.venue.registry.require(symbol)
    book = market.venue.engine(symbol).book.snapshot(levels=1 << 20)
    return resting(book, instrument)


def volume_on(market, symbol: str) -> int:
    return sum(
        account.positions[symbol].volume
        for account in market.venue.accounts.values()
        if symbol in account.positions
    )


def test_the_tiered_contract_trades_and_leaves_a_book_behind(live_market):
    """The premise of the two tests below, asserted rather than assumed.

    A contract that never traded and never rested an order would pass a
    check for off-grid prices trivially, and the pass would mean the tiered
    tick had made the symbol unquotable rather than that it worked.
    """
    assert volume_on(live_market, "VANTA_OBJECTIVE_WR") > 0
    assert prices_on(live_market, "VANTA_OBJECTIVE_WR")


def test_nothing_rests_off_the_grid_on_the_tiered_contract(live_market):
    """The end-to-end claim: no path into the book skips the listing rule.

    Agents reach the venue through several routes -- the opening auction,
    quotes, replaces, the arbitrageur -- and each one is a chance for a price
    to arrive without having been snapped or checked. This is the only test
    here that would notice.
    """
    tiered = live_market.venue.registry.require("VANTA_OBJECTIVE_WR")
    off_grid = [
        str(price)
        for price in prices_on(live_market, "VANTA_OBJECTIVE_WR")
        if not tiered.on_grid(price)
    ]
    assert off_grid == []


def test_the_tiered_contract_quotes_whole_points_above_its_threshold(live_market):
    """Where it actually trades is above 4,000, so the coarse band is the live one.

    Without this the test above could pass on a session that spent all its
    time below the threshold, where the coarse band is never consulted and
    every quarter-point price is on the grid anyway.
    """
    above = [p for p in prices_on(live_market, "VANTA_OBJECTIVE_WR") if abs(p) >= D("4000")]
    assert above, "nothing rested in the coarse band, so it was never exercised"
    assert all(price % 1 == 0 for price in above)


def test_the_uniform_contract_still_quotes_quarters(live_market):
    """A listing rule for one contract must not become a rule for the venue.

    QUILL_OBJECTIVE_WR is a team win rate too, so it trades in the same range
    as the tiered contract, and it has no table. If the coarse band had leaked,
    through a shared default or a grid check that read some other instrument,
    its quarter-point quotes would be the first casualty, and the symptom would
    be a market that had quietly got worse rather than an error.
    """
    prices = prices_on(live_market, "QUILL_OBJECTIVE_WR")
    assert volume_on(live_market, "QUILL_OBJECTIVE_WR") > 0
    assert any(price % 1 != 0 for price in prices), (
        "no fractional price rested, so the uniform tick was not exercised"
    )


# --------------------------------------------------------------------------
# The other half of the same listing rule: the lot
# --------------------------------------------------------------------------


def test_the_venue_refuses_a_quantity_off_the_contracts_lot():
    """An instrument declares "tick / lot -- the grid the exchange enforces",
    and only the tick was ever enforced.

    A quantity that cannot exist is exactly as unquotable as a price that
    cannot exist, and the argument the price case already makes applies word
    for word: a rule the venue does not enforce is documentation rather than a
    listing rule. Measured on a contract listed in lots of ten, an order for
    **seven** was acknowledged, rested, and would have traded.
    """
    tiered = Instrument(SYM, spec(), lot_size=10)
    venue = venue_with(tiered)
    events = venue.submit(
        MAKER, SYM, limit(MAKER, Side.BUY, tiered.to_ticks(D("4000.00")), quantity=7)
    )
    assert reasons(events) == [RejectReason.INVALID_QUANTITY]
    assert venue.engine(SYM).book.total_resting_quantity == 0


def test_the_venue_accepts_a_whole_number_of_lots():
    """The permissive half. A check that refused every size, or that compared
    against the wrong number, would take the contract off the market entirely
    -- and an agent told no for a size the contract does allow has no way to
    comply.
    """
    tiered = Instrument(SYM, spec(), lot_size=10)
    venue = venue_with(tiered)
    events = venue.submit(
        MAKER, SYM, limit(MAKER, Side.BUY, tiered.to_ticks(D("4000.00")), quantity=30)
    )
    assert reasons(events) == []
    assert venue.engine(SYM).book.total_resting_quantity == 30


def test_an_amendment_cannot_resize_an_order_off_the_lot():
    """The same hole the price check already closed on the same command.

    Guarding only new orders leaves the way in wide open: an accepted 30-lot
    order is amended to 7 and rests at a size the contract does not list, with
    nothing rejected. A modification is a request for a size exactly as it is a
    request for a price.
    """
    tiered = Instrument(SYM, spec(), lot_size=10)
    venue = venue_with(tiered)
    accepted = venue.submit(
        MAKER, SYM, limit(MAKER, Side.BUY, tiered.to_ticks(D("4000.00")), quantity=30)
    )
    order_id = next(e.order_id for e in accepted if isinstance(e, Acknowledged))

    events = venue.submit(
        MAKER, SYM, Replace(MAKER, order_id, Quantity(7), tiered.to_ticks(D("3999.00")))
    )
    assert reasons(events) == [RejectReason.INVALID_QUANTITY]
    assert venue.engine(SYM).book.total_resting_quantity == 30


def test_a_contract_listed_in_single_lots_is_unaffected():
    """Every contract on this venue but a deliberately-listed one has a lot
    size of one, where every quantity is a whole number of lots. The rule has
    to be silent there or it would be a change to the whole market rather than
    a fact about one listing.
    """
    plain = future()
    venue = venue_with(plain)
    for quantity in (1, 7, 33):
        events = venue.submit(
            MAKER,
            SYM,
            limit(MAKER, Side.BUY, plain.to_ticks(D("4000.00")), quantity=quantity),
        )
        assert reasons(events) == []
    assert venue.engine(SYM).book.total_resting_quantity == 41


# --------------------------------------------------------------------------
# And inside the range the contract can settle in
# --------------------------------------------------------------------------


def _at(venue, instrument, price: str, side=Side.BUY, quantity: int = 5, who=MAKER):
    return venue.submit(
        who, SYM, limit(who, side, instrument.to_ticks(D(price)), quantity)
    )


def test_the_venue_refuses_a_price_the_contract_can_never_settle_at():
    """A listing rule of the same kind as the grid, and the one nobody wrote.

    Measured on a contract bounded by [0, 10,000]: a bid at -100 was
    acknowledged and rested, and a bid at 10,100 was acknowledged and *traded*
    -- a print above everything the claim can ever be worth, which dragged the
    mark of every position in the symbol up with it.

    Collateral structurally cannot catch this, which is exactly why it needs a
    rule of its own. The requirement is the worst case over the settlement
    range, so a bid *below* the floor scores as the safest order on the book:
    it can only gain. The venue's central safety mechanism rates the impossible
    order as the safe one.
    """
    tiered = future()
    venue = venue_with(tiered)
    low, high = tiered.value_bounds

    below = _at(venue, tiered, str(low - D("100")))
    above = _at(venue, tiered, str(high + D("100")))
    assert reasons(below) == [RejectReason.INVALID_PRICE]
    assert reasons(above) == [RejectReason.INVALID_PRICE]
    assert venue.engine(SYM).book.total_resting_quantity == 0
    assert venue.engine(SYM).tape == ()
    assert venue.conservation_check() == 0


def test_a_price_on_either_bound_is_still_a_price_the_contract_can_pay():
    """The permissive half, and the bounds are inclusive because a settlement
    value can legitimately land on one. A check written with strict
    inequalities would forbid the two prices a binary spends its whole life
    converging on.
    """
    tiered = future()
    venue = venue_with(tiered)
    low, high = tiered.value_bounds

    assert reasons(_at(venue, tiered, str(low))) == []
    assert reasons(_at(venue, tiered, str(high), side=Side.SELL, who=TAKER)) == []
    assert resting(venue.engine(SYM).book.snapshot(), tiered) == [low, high]
    assert venue.conservation_check() == 0


def test_an_amendment_cannot_move_an_order_outside_the_range():
    """Both paths, because the grid check was once bypassed on exactly this one:
    it read ``price`` where a replace carries ``new_price``, so an accepted bid
    could be amended onto a price the listing forbids and rest there with
    nothing rejected.
    """
    tiered = future()
    venue = venue_with(tiered)
    _low, high = tiered.value_bounds
    accepted = _at(venue, tiered, "4000.00")
    order_id = next(e.order_id for e in accepted if isinstance(e, Acknowledged))

    outside = venue.submit(
        MAKER,
        SYM,
        Replace(MAKER, order_id, Quantity(5), tiered.to_ticks(high + D("100"))),
    )
    assert reasons(outside) == [RejectReason.INVALID_PRICE]
    assert resting(venue.engine(SYM).book.snapshot(), tiered) == [D("4000.00")]

    inside = venue.submit(
        MAKER, SYM, Replace(MAKER, order_id, Quantity(5), tiered.to_ticks(high))
    )
    assert reasons(inside) == []
    assert venue.conservation_check() == 0


def test_a_payout_narrows_the_rule_without_touching_what_is_resting():
    """A share is worth less after it pays by exactly what it paid, and the
    range the venue polices follows it down -- because that is the same range
    the collateral is computed from, and quoting a wider one would let an order
    be entered above what the claim can still deliver.

    What does *not* move is anything already standing. The venue does not
    reprice an order whose owner named a price, and pulling one would close a
    position its owner never asked to close. Measured: a bid resting at the
    pre-payout ceiling of 11,000 is untouched by a payout of 500, a fresh order
    at that same price is refused, and one at the surviving ceiling of 10,500
    is accepted.
    """
    from arena.contracts.spec import DistributionSchedule
    from arena.portfolio.money import to_money

    week = ObservationWindow(WINDOW.start, WINDOW.start + timedelta(days=7))
    share = Instrument(
        SYM,
        ContractSpec(
            contract_id=SYM,
            underlying=_wr("A"),
            payoff=Linear(10_000.0),
            window=WINDOW,
            policy=DataPolicy(min_sample_size=1),
            reference_id="ref-1",
            published_at=WINDOW.start - timedelta(days=1),
            tick_size="0.25",
            distribution=DistributionSchedule(windows=(week,), payoff=Linear(1_000.0)),
        ),
    )
    venue = Venue(starting_cash=800_000_000)
    venue.list_instrument(share)
    ceiling = share.value_bounds[1]

    # Somebody long and somebody short, so the payout has both sides to move.
    _at(venue, share, "5000.00", quantity=4)
    _at(venue, share, "5000.00", side=Side.SELL, quantity=4, who=TAKER)
    stranded = _at(venue, share, str(ceiling), quantity=2)
    assert reasons(stranded) == []

    venue.distribute(SYM, to_money(D("500")))
    surviving = venue.bounds_in_minor(share)[1]

    assert resting(venue.engine(SYM).book.snapshot(), share) == [ceiling], (
        "a payout disturbed an order that was already working"
    )
    assert reasons(_at(venue, share, str(ceiling), quantity=1)) == [
        RejectReason.INVALID_PRICE
    ]
    assert reasons(
        _at(venue, share, str(from_money(surviving)), quantity=1)
    ) == []
    assert venue.conservation_check() == 0
