"""Message rate limits and the participant kill switch.

Both controls answer the same question -- what does a venue do when one
participant stops behaving like a participant -- and they answer it in different
registers.

A rate limit is the routine one, and it is not politeness. An algorithm that
malfunctions emits orders faster than anything downstream can process them, and
the venue with no limit is the one that goes down with it. It is also the only
defence against a participant that discovers it can profit by simply sending
more messages than everyone else. The window is a rolling second rather than a
fixed one, and that is the whole difficulty of the feature: a fixed window that
resets on a boundary lets a participant send its entire allowance twice in a few
milliseconds by straddling one.

A kill switch is the other one, reached for when a participant is doing
something nobody wants to reason about at the time. It is deliberately blunt,
because the point of a kill switch is that it is the one control that always
works: everything working is pulled and everything new is refused. Cancels are
the single exception, and that is not a softening of the rule. Refusing those
too would leave the participant trapped in the orders it already has -- unable
to place, unable to withdraw, holding exposure nobody is permitted to manage.

Neither control may create or destroy a penny, so every case here ends at the
ledger.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from arena.contracts.payoff import Linear
from arena.contracts.spec import ContractSpec, DataPolicy, ObservationWindow
from arena.contracts.underlying import Single
from arena.exchange.events import (
    Acknowledged,
    Cancel,
    Cancelled,
    Rejected,
    Replace,
    Submit,
)
from arena.exchange.types import (
    AgentId,
    OrderType,
    PegReference,
    Price,
    Quantity,
    RejectReason,
    Side,
    TimeInForce,
)
from arena.market.fees import MAKER_TAKER
from arena.market.instrument import Instrument
from arena.market.venue import Venue
from arena.worlds.circuit.metrics import metric_ref

UTC = timezone.utc
START = datetime(2026, 8, 31, tzinfo=UTC)

B, S = Side.BUY, Side.SELL

ONE_SECOND_NS = 1_000_000_000


def _instrument(symbol: str = "F") -> Instrument:
    return Instrument(
        symbol,
        ContractSpec(
            contract_id=symbol,
            underlying=Single(metric_ref("win_rate", "SUBJECT")),
            payoff=Linear(10_000.0),
            window=ObservationWindow(START, START + timedelta(days=28)),
            policy=DataPolicy(
                min_sample_size=1_000, min_stratum_battles=200, min_strata_coverage=0.80
            ),
            reference_id="ref-2026S09-v1",
            published_at=START - timedelta(days=1),
            tick_size="0.25",
        ),
    )


def _venue(**kwargs) -> Venue:
    venue = Venue("arena", starting_cash=80_000_000, **kwargs)
    venue.list_instrument(_instrument())
    return venue


def _send(venue, who, side, price, quantity, tif=TimeInForce.GTC, symbol="F"):
    return venue.submit(
        AgentId(who),
        symbol,
        Submit(
            AgentId(who),
            side,
            Quantity(quantity),
            None if price is None else Price(price),
            OrderType.MARKET if price is None else OrderType.LIMIT,
            tif,
        ),
    )


def _reasons(events) -> list[RejectReason]:
    return [e.reason for e in events if isinstance(e, Rejected)]


def _reason(events) -> RejectReason:
    """The one reason a command was refused, so a test fails loudly if it was not."""
    refusals = _reasons(events)
    assert len(refusals) == 1, f"expected exactly one refusal, got {refusals}"
    return refusals[0]


def _order_id(events):
    return next(e.order_id for e in events if isinstance(e, Acknowledged))


def _resting(venue, symbol, who) -> list[int]:
    """What this agent has left standing in a book, in arrival order."""
    return [
        int(order.remaining)
        for order in venue.engine(symbol).book.resting_orders
        if order.agent_id == AgentId(who)
    ]


def _limited_venue(rate: int, clock: dict) -> Venue:
    """A venue whose clock the test moves by hand.

    ``sim_clock`` defaults to None, which means simulated time never advances --
    right for a venue being poked rather than run, and useless for a rolling
    window, which measures nothing at all if the clock is stuck.
    """
    venue = _venue(message_rate=rate)
    venue.sim_clock = lambda: clock["now"]
    return venue


# --------------------------------------------------------------------------
# The message rate limit
# --------------------------------------------------------------------------


def test_a_participant_is_refused_only_once_it_is_over_its_message_cap():
    """A cap that bites early throttles traffic the venue agreed to take; one
    that never bites is not a cap. The refused order also has to stay out of the
    book -- a refusal that quietly rested anyway would be worse than no limit,
    because the participant would be told no and get the order regardless.
    """
    clock = {"now": 0}
    venue = _limited_venue(5, clock)

    for i in range(5):
        assert _reasons(_send(venue, "a", B, 18_600 - i, 1)) == []

    assert _reason(_send(venue, "a", B, 18_700, 1)) is RejectReason.RATE_LIMITED
    assert venue.engine("F").book.total_resting_quantity == 5


def test_the_message_window_rolls_rather_than_resetting_on_a_boundary():
    """A fixed window is the obvious implementation and it gives away double the
    allowance.

    Spend it all just before the reset and all of it again just after, and the
    venue has taken twice the traffic it agreed to, delivered inside the few
    hundred milliseconds either side of one boundary -- which is precisely the
    burst the limit exists to stop. A rolling window still sees the first half,
    so the second one is refused.
    """
    clock = {"now": 0}
    venue = _limited_venue(4, clock)

    clock["now"] = 900_000_000
    for i in range(4):
        assert _reasons(_send(venue, "a", B, 18_600 - i, 1)) == []

    clock["now"] = 1_100_000_000
    for i in range(4):
        assert _reason(_send(venue, "a", B, 18_500 - i, 1)) is RejectReason.RATE_LIMITED


def test_a_quiet_second_restores_the_whole_allowance():
    """A limit that never let go would turn one momentary burst into permanent
    exclusion from the market, which is a far heavier penalty than the venue ever
    agreed to impose.
    """
    clock = {"now": 0}
    venue = _limited_venue(2, clock)
    for i in range(2):
        _send(venue, "a", B, 18_600 - i, 1)

    clock["now"] = ONE_SECOND_NS - 1_000_000
    assert _reason(_send(venue, "a", B, 18_550, 1)) is RejectReason.RATE_LIMITED

    clock["now"] = ONE_SECOND_NS + 1
    for i in range(2):
        assert _reasons(_send(venue, "a", B, 18_500 - i, 1)) == []


def test_being_refused_does_not_lengthen_a_lockout():
    """A refused message is refused, not remembered.

    Counting refusals against the allowance would mean a client that retries --
    which is what an automated client does when it is told no -- keeps its own
    lockout alive by trying to get out of it. How long a burst costs would then
    depend on how quickly the offender gave up, which is the wrong thing for it
    to depend on.
    """
    clock = {"now": 0}
    venue = _limited_venue(2, clock)
    for i in range(2):
        _send(venue, "a", B, 18_600 - i, 1)

    clock["now"] = 500_000_000
    for i in range(20):
        events = _send(venue, "a", B, 18_550 - i, 1)
        assert _reason(events) is RejectReason.RATE_LIMITED

    clock["now"] = ONE_SECOND_NS + 1
    assert _reasons(_send(venue, "a", B, 18_400, 1)) == []


def test_a_venue_with_no_message_rate_never_limits_anything():
    """The default, so every measurement taken before the limit existed still
    means what it meant.
    """
    venue = _venue()
    assert venue.message_rate is None
    for i in range(200):
        assert _reasons(_send(venue, "a", B, 18_600 - (i % 40), 1)) == []


def test_one_participant_exhausting_its_allowance_does_not_touch_another():
    """The limit is a fact about a participant, not about the venue's total
    traffic. A shared budget would let a single runaway algorithm lock every
    other participant out of the market -- the outage the limit exists to
    prevent, arriving by a different route.
    """
    clock = {"now": 0}
    venue = _limited_venue(3, clock)
    for i in range(6):
        _send(venue, "a", B, 18_600 - i, 1)
    assert _reason(_send(venue, "a", B, 18_580, 1)) is RejectReason.RATE_LIMITED

    for i in range(3):
        assert _reasons(_send(venue, "c", S, 18_900 + i, 1)) == []


# --------------------------------------------------------------------------
# The kill switch
# --------------------------------------------------------------------------


def test_killing_a_participant_pulls_its_orders_and_names_the_symbols():
    """An operator throwing the switch needs to know what was standing in the
    participant's name, not only that the switch was thrown: liquidity has just
    left several books at once, and somebody has to account for it now rather
    than discover it later.
    """
    venue = _venue()
    venue.list_instrument(_instrument("G"))
    _send(venue, "a", B, 18_600, 30)
    _send(venue, "a", S, 18_900, 20, symbol="G")
    _send(venue, "c", B, 18_500, 10)

    assert venue.kill(AgentId("a"), reason="runaway") == ["F", "G"]
    assert venue.halted_participants[AgentId("a")] == "runaway"
    assert _resting(venue, "F", "a") == []
    assert _resting(venue, "G", "a") == []
    assert _resting(venue, "F", "c") == [10], "somebody else's order was pulled"
    assert venue.conservation_check() == 0


def test_a_killed_participant_cannot_place_or_amend_an_order():
    """An amendment is a request for risk exactly as an order is.

    Guarding only new orders leaves the way back in wide open: the participant
    reworks the orders it already had, at whatever size and price it likes, and
    is trading again without ever having been let back in.
    """
    venue = _venue()
    order_id = _order_id(_send(venue, "a", B, 18_600, 30))
    venue.kill(AgentId("a"))

    fresh = _send(venue, "a", B, 18_550, 10)
    assert _reason(fresh) is RejectReason.PARTICIPANT_HALTED

    amend = venue.submit(
        AgentId("a"),
        "F",
        Replace(AgentId("a"), order_id, Quantity(500), Price(18_800)),
    )
    assert _reason(amend) is RejectReason.PARTICIPANT_HALTED
    assert _resting(venue, "F", "a") == []


def test_a_killed_participant_can_still_cancel():
    """Refusing cancels too would trap the participant in the orders it already
    has, which is the opposite of what stopping it is for.

    The order named here has already gone, because the kill pulled it. What is
    being pinned is where the answer comes from: the book, rather than the
    venue's door.
    """
    venue = _venue()
    order_id = _order_id(_send(venue, "a", B, 18_600, 30))
    venue.kill(AgentId("a"))

    events = venue.submit(AgentId("a"), "F", Cancel(AgentId("a"), order_id))
    assert RejectReason.PARTICIPANT_HALTED not in _reasons(events)


def test_a_stopped_participant_can_pull_an_order_that_is_still_working():
    """The same rule where it has teeth: a live order and an owner who is not
    allowed to trade. The cancel has to reach the book and take the order out of
    it, or the exposure stays in the market with nobody permitted to manage it.
    """
    venue = _venue()
    order_id = _order_id(_send(venue, "a", B, 18_600, 25))
    # Stopped without its orders being pulled first, which is what a participant
    # that stops itself looks like from the venue's side.
    venue.halted_participants[AgentId("a")] = "self"

    events = venue.submit(AgentId("a"), "F", Cancel(AgentId("a"), order_id))
    assert any(isinstance(e, Cancelled) for e in events)
    assert _resting(venue, "F", "a") == []
    assert venue.conservation_check() == 0


def test_reviving_a_participant_lets_it_back_in_with_nothing_working():
    """Coming back to find its old orders restored would hand the participant
    exposure it never re-entered, at prices struck before whatever got it
    stopped.
    """
    venue = _venue()
    _send(venue, "a", B, 18_600, 30)
    _send(venue, "a", S, 18_950, 20)
    venue.kill(AgentId("a"))

    venue.revive(AgentId("a"))
    assert AgentId("a") not in venue.halted_participants
    assert not venue._working.get((AgentId("a"), "F"))
    assert _resting(venue, "F", "a") == []

    assert _reasons(_send(venue, "a", B, 18_450, 5)) == []
    assert _resting(venue, "F", "a") == [5]
    assert venue.conservation_check() == 0


def test_killing_one_participant_leaves_the_rest_of_the_market_trading():
    """Stopping the market is a halt, and a halt is a different decision with
    different costs -- everyone's, rather than one participant's. Confusing the
    two turns a narrow tool into an outage.
    """
    venue = _venue()
    _send(venue, "a", B, 18_600, 20)
    venue.kill(AgentId("a"))

    _send(venue, "c", B, 18_500, 20)
    _send(venue, "d", S, 18_500, 20)
    assert venue.engine("F").tape, "the book stopped trading for everybody"
    assert venue.account(AgentId("c")).positions["F"].quantity == 20
    assert venue.account(AgentId("a")).positions.get("F") is None
    assert venue.conservation_check() == 0


# --------------------------------------------------------------------------
# The ledger, through all of it
# --------------------------------------------------------------------------


def test_the_ledger_stays_exactly_balanced_through_limits_kills_and_revivals():
    """Neither control may cost or create a penny.

    Both refuse commands, and one of them cancels orders on the participant's
    behalf -- exactly the shape of thing that leaves collateral reserved against
    an order that no longer exists. A leak like that shows up nowhere else: the
    books look right, the positions look right, and the only symptom is a number
    that should be zero and is not.
    """
    clock = {"now": 0}
    venue = _limited_venue(3, clock)
    venue.fees = MAKER_TAKER
    rng = random.Random(7)

    refusals = 0
    for step in range(80):
        clock["now"] = step * 200_000_000
        for who, side in (("a", B), ("c", S)):
            low, high = (18_400, 18_700) if side is B else (18_500, 18_800)
            events = _send(venue, who, side, rng.randint(low, high), rng.randint(1, 20))
            refusals += _reasons(events).count(RejectReason.RATE_LIMITED)
        if step == 30:
            venue.kill(AgentId("a"), reason="runaway")
        if step == 50:
            venue.revive(AgentId("a"))
        assert venue.conservation_check() == 0

    assert venue.engine("F").tape, "nothing traded, so the run proved nothing"
    assert refusals > 0, "nothing was rate limited, so the run proved nothing"
    assert venue.conservation_check() == 0


# --------------------------------------------------------------------------
# Neither control may trap a participant in its own orders
# --------------------------------------------------------------------------


def test_a_participant_at_its_cap_can_still_withdraw_an_order():
    """The rule the kill switch already keeps, kept by the other control too.

    A participant at its message cap could not cancel: measured at a cap of
    five, it sent five orders and then every attempt to pull one came back
    RATE_LIMITED, through five retries, with fifty lots still standing in the
    book. That is the same failure refusing a stopped participant's cancels
    would be -- unable to place, unable to withdraw, holding exposure nobody is
    permitted to manage -- arriving by the other door.

    A cancel is also the one command that only ever makes things smaller: less
    risk for the participant and less book for the venue. Refusing it is the
    one refusal that makes both sides worse off.
    """
    clock = {"now": 0}
    venue = _limited_venue(5, clock)

    ids = []
    for i in range(5):
        ids.append(_order_id(_send(venue, "a", B, 18_600 - i, 10)))
    assert _resting(venue, "F", "a") == [10] * 5
    assert _reason(_send(venue, "a", B, 18_500, 10)) is RejectReason.RATE_LIMITED

    for order_id in ids:
        events = venue.submit(AgentId("a"), "F", Cancel(AgentId("a"), order_id))
        assert RejectReason.RATE_LIMITED not in _reasons(events)
    assert _resting(venue, "F", "a") == []
    assert venue.conservation_check() == 0


def test_a_cancel_is_still_counted_against_the_allowance():
    """Never refused is not the same as free.

    A burst of cancels is still traffic the venue has to take, so it still
    costs the sender its ability to add anything. Exempting them from the count
    as well would hand any participant an unmetered channel, which is the thing
    the limit exists to deny.
    """
    clock = {"now": 0}
    venue = _limited_venue(4, clock)
    ids = [_order_id(_send(venue, "a", B, 18_600 - i, 10)) for i in range(2)]

    for order_id in ids:
        venue.submit(AgentId("a"), "F", Cancel(AgentId("a"), order_id))

    # Two orders and two cancels is the whole allowance for this second.
    assert _reason(_send(venue, "a", B, 18_400, 10)) is RejectReason.RATE_LIMITED


# --------------------------------------------------------------------------
# The kill switch believes the book
# --------------------------------------------------------------------------


def test_the_kill_switch_pulls_a_market_on_open_order():
    """A kill switch that walks past an order is not a kill switch.

    A market-on-open order names no price, so nothing was ever written into the
    venue's record of what the participant is working -- and the record was the
    only place the kill switch looked. Measured: ``kill`` reported the symbol
    as pulled while a 40-lot market-on-open buy stayed standing, and the
    stopped participant then took 40 lots in the very auction it had been
    stopped before.
    """
    venue = _venue()
    venue.begin_session("F")
    _send(venue, "a", B, 18_600, 30)
    _send(venue, "a", B, None, 40, tif=TimeInForce.IOC)
    assert len(_resting(venue, "F", "a")) == 2

    # The record, not only the outcome. An order that named no price is still
    # an order this participant is working, and it is reserved against the far
    # end of the contract's range because that is the worst price an order
    # that named none could get.
    working = venue._working[(AgentId("a"), "F")]
    assert len(working) == 2
    _low, high = venue.registry.require("F").bounds_in_minor
    assert sorted(price for _s, _q, price in working.values())[-1] == int(high)

    assert venue.kill(AgentId("a"), reason="runaway") == ["F"]
    assert _resting(venue, "F", "a") == []

    _send(venue, "c", S, 18_000, 100)
    venue.uncross("F")
    assert venue.account(AgentId("a")).positions.get("F") is None
    assert venue.conservation_check() == 0


def test_a_refused_amendment_does_not_hide_an_order_from_the_kill_switch():
    """A refusal is not a removal, and treating it as one lost the order.

    The engine refuses an amendment it dislikes and leaves the original exactly
    where it was resting. The venue dropped its own record anyway, so
    ``Replace(order, quantity=0)`` came back INVALID_QUANTITY, the order stayed
    in the book for thirty lots, and ``kill`` then reported no symbols at all
    and left it standing. One refused message was enough to make a participant
    unstoppable.
    """
    venue = _venue()
    order_id = _order_id(_send(venue, "a", B, 18_600, 30))
    refused = venue.submit(
        AgentId("a"), "F", Replace(AgentId("a"), order_id, Quantity(0), Price(18_650))
    )
    assert _reason(refused) is RejectReason.INVALID_QUANTITY
    assert _resting(venue, "F", "a") == [30], "the refusal moved the order"
    # The record has to survive the refusal, because collateral is reserved
    # against it: forgetting a live order under-reserves the account as surely
    # as it hides the order from the kill switch.
    assert order_id in venue._working[(AgentId("a"), "F")]

    assert venue.kill(AgentId("a")) == ["F"]
    assert _resting(venue, "F", "a") == []
    assert venue.conservation_check() == 0


def test_the_kill_switch_pulls_an_order_the_venue_has_no_record_of():
    """What the book is asked for, and the reason it is asked at all.

    The venue's record and the engine's book are two accounts of the same
    thing, and a kill switch that consults only the first is only as good as
    the bookkeeping. It is the one control that has to work when something has
    already gone wrong -- which is the situation in which the bookkeeping is
    least trustworthy -- so it takes the union of both and pulls that.

    The record is emptied here by hand rather than through a bug, because the
    point is the property and not the route to it.
    """
    venue = _venue()
    _send(venue, "a", B, 18_600, 30)
    venue._working[(AgentId("a"), "F")].clear()

    assert venue.kill(AgentId("a")) == ["F"]
    assert _resting(venue, "F", "a") == []
    assert venue.conservation_check() == 0


def test_a_rejection_that_did_terminate_an_order_still_clears_the_reservation():
    """The other direction, because believing the engine has to mean believing
    it both ways.

    A post-only order is acknowledged and then refused for crossing, so it is
    tracked and then must be untracked -- otherwise the reservation outlives an
    order that never existed, which is the same phantom by the opposite route.
    """
    venue = _venue()
    _send(venue, "c", S, 18_600, 50)
    refused = venue.submit(
        AgentId("a"),
        "F",
        Submit(
            AgentId("a"), B, Quantity(40), Price(18_700), OrderType.LIMIT,
            TimeInForce.POST_ONLY,
        ),
    )
    assert _reason(refused) is RejectReason.POST_ONLY_WOULD_CROSS
    assert not venue._working.get((AgentId("a"), "F"))
    assert venue.kill(AgentId("a")) == []
    assert venue.conservation_check() == 0


# --------------------------------------------------------------------------
# What the venue thinks a participant is working
# --------------------------------------------------------------------------


def test_a_quote_that_gets_filled_stops_being_reserved_against():
    """Collateral is reserved against working orders, so the venue's idea of
    what is working has to survive somebody else's command.

    A match produces a fill for the incoming order and one for the resting
    order, and those belong to two different participants. The whole batch was
    booked under whoever sent the command, so the passive side's fill was
    looked up in the wrong agent's record, found nothing, and did nothing.

    Measured on a maker quoting two lots a round and being lifted every round:
    after 120 rounds the venue believed it was working **120 orders for 240
    lots** while the engine's book held none. With a million in capital the
    maker was refused for insufficient collateral at round 47, holding 497,100
    of free cash and nothing at all in the book -- an account charged twice for
    a risk it holds once.
    """
    venue = Venue("arena", starting_cash=1_000_000)
    venue.list_instrument(_instrument())
    maker = AgentId("mm")

    for _ in range(60):
        events = _send(venue, "mm", S, 18_600, 2)
        assert RejectReason.INSUFFICIENT_COLLATERAL not in _reasons(events)
        _send(venue, "taker", B, None, 2, tif=TimeInForce.IOC)

    assert _resting(venue, "F", "mm") == []
    assert venue._working[(maker, "F")] == {}, "reserved against orders that are gone"
    assert venue.account(maker).positions["F"].quantity == -120
    assert venue.conservation_check() == 0


def test_a_partly_filled_quote_is_still_reserved_against_what_is_left():
    """The passive side's record is updated, not simply discarded: an order
    half taken is still an order, and the collateral behind the remainder has
    to stay posted.
    """
    venue = _venue()
    maker = AgentId("mm")
    order_id = _order_id(_send(venue, "mm", S, 18_600, 50))
    _send(venue, "taker", B, None, 20, tif=TimeInForce.IOC)

    working = venue._working[(maker, "F")]
    assert order_id in working
    assert working[order_id][1] == 30
    assert _resting(venue, "F", "mm") == [30]
    assert venue.conservation_check() == 0


def test_one_agents_command_cannot_book_another_agents_order_against_it():
    """The same fix from the other side.

    A command's event batch can carry events for people who never sent it: a
    peg repricing is a ``Replaced`` for the peg's owner, produced inside
    somebody else's order. Booking the whole batch under the sender wrote a
    stranger's order into the sender's working book -- measured, a five-lot
    order left its sender reserving collateral against **thirty** lots, its own
    five and twenty-five of a peg it had never seen, while the peg's owner kept
    a record at the price the peg had already left.
    """
    venue = _venue()
    _send(venue, "seed", B, 18_000, 10)
    _send(venue, "seed", S, 18_400, 10)
    venue.submit(
        AgentId("pegger"),
        "F",
        Submit(
            AgentId("pegger"), B, Quantity(25), None, OrderType.PEGGED,
            TimeInForce.GTC, peg_to=PegReference.BID, peg_offset=0,
        ),
    )

    # A better bid from somebody else, which drags the peg with it.
    _send(venue, "stranger", B, 18_100, 5)

    stranger = venue._working[(AgentId("stranger"), "F")]
    pegger = venue._working[(AgentId("pegger"), "F")]
    assert sum(q for _s, q, _p in stranger.values()) == 5
    assert sum(q for _s, q, _p in pegger.values()) == 25
    assert not set(stranger) & set(pegger), "one order in two working books"
    # And the peg's owner holds the price its order actually rests at.
    priced = venue.registry.require("F").price_in_minor(Price(18_100))
    assert [p for _s, _q, p in pegger.values()] == [int(priced)]
    assert venue.conservation_check() == 0


# --------------------------------------------------------------------------
# What the order path charges collateral against
# --------------------------------------------------------------------------


def _priced(venue, price: str) -> int:
    """A contract price as the integer ticks an order carries."""
    from decimal import Decimal

    return int(venue.registry.require("F").to_ticks(Decimal(price)))


def test_an_order_is_charged_against_the_position_it_would_create():
    """The scenario check has to price the position the fills would leave
    behind, and that position's exposure comes from its basis rather than from
    the price of the trade that finished it.

    ``resulting * incoming_price`` simply omits what is already held, and it
    omits it in the dangerous direction. Measured through this order path on an
    account holding 50,500 against a contract bounded by [0, 10,000]: ten lots
    long at 5,000, then ten more at 100. The check evaluated ``20 * 100 =
    2,000``, accepted the order, and the position it produced carried a basis
    of 51,000 -- collateral of 51,000,000,000 minor units against cash of
    50,500,000,000, which is **free cash of -500,000,000**. An account owing
    money it does not have is the one outcome full collateralisation exists to
    make impossible.
    """
    venue = Venue(
        "arena",
        starting_cash=10_000_000_000,
        balances={AgentId("buyer"): 50_500},
    )
    venue.list_instrument(_instrument())
    buyer = venue.account(AgentId("buyer"))

    _send(venue, "seller", S, _priced(venue, "5000"), 10)
    _send(venue, "buyer", B, _priced(venue, "5000"), 10)
    assert buyer.positions["F"].quantity == 10
    assert int(buyer.free_cash) >= 0

    _send(venue, "seller", S, _priced(venue, "100"), 10)
    events = _send(venue, "buyer", B, _priced(venue, "100"), 10)

    assert _reason(events) is RejectReason.INSUFFICIENT_COLLATERAL
    assert buyer.positions["F"].quantity == 10, "the add went through anyway"
    assert int(buyer.free_cash) >= 0, "the venue allowed a position it cannot cover"
    assert venue.conservation_check() == 0


def test_an_add_the_account_can_actually_cover_still_goes_through():
    """The permissive half, and it matters as much as the refusal.

    Charging against the basis is a correction, not a tightening: an account
    that can carry the position the fills would produce must still be able to
    open it, or the fix would simply be a smaller market wearing a safety
    argument.
    """
    venue = Venue(
        "arena",
        starting_cash=10_000_000_000,
        balances={AgentId("buyer"): 60_000},
    )
    venue.list_instrument(_instrument())
    buyer = venue.account(AgentId("buyer"))

    _send(venue, "seller", S, _priced(venue, "5000"), 10)
    _send(venue, "buyer", B, _priced(venue, "5000"), 10)
    _send(venue, "seller", S, _priced(venue, "100"), 10)
    assert _reasons(_send(venue, "buyer", B, _priced(venue, "100"), 10)) == []

    assert buyer.positions["F"].quantity == 20
    assert int(buyer.positions["F"].cost_basis) == 51_000_000_000
    assert int(buyer.free_cash) >= 0
    assert venue.conservation_check() == 0


def test_closing_a_position_stays_admissible_with_no_free_cash():
    """A trade that reduces exposure must never be refused for collateral.

    The scenario check prices the *resulting* position, so a sale that takes a
    long back to flat is charged against nothing -- which is the property that
    keeps an agent able to get out of a losing position at exactly the moment
    it has no room left to get into anything.
    """
    venue = Venue(
        "arena",
        starting_cash=10_000_000_000,
        balances={AgentId("buyer"): 50_000},
    )
    venue.list_instrument(_instrument())
    buyer = venue.account(AgentId("buyer"))

    _send(venue, "seller", S, _priced(venue, "5000"), 10)
    _send(venue, "buyer", B, _priced(venue, "5000"), 10)
    assert int(buyer.free_cash) == 0, "the account was not fully committed"

    _send(venue, "bidder", B, _priced(venue, "4900"), 10)
    assert _reasons(_send(venue, "buyer", S, _priced(venue, "4900"), 10)) == []
    assert buyer.positions["F"].quantity == 0
    assert venue.conservation_check() == 0


def test_an_order_is_charged_against_the_cash_the_fill_would_leave():
    """The scenario books a realised loss as well as posting collateral, and
    the loss comes out of cash the instant the fill prints.

    Comparing the requirement against the cash the account holds *now* compares
    it with money the trade is about to take away. Measured through this order
    path: short four lots at an average of 50, then buy eleven at 9,500. The
    flip realises **-37,800,000,000** minor units, which the check never saw,
    so 66,500,000,000 of collateral was approved against 100,000,000,000 of
    cash that became 62,200,000,000 the moment it filled -- free cash of
    **-4,300,000,000**.

    Swept over four hundred fills across the whole settlement range, nine of
    thirty random runs finished with some account underwater this way.
    """
    venue = Venue(
        "arena",
        starting_cash=10_000_000_000,
        balances={AgentId("flipper"): 100_000},
    )
    venue.list_instrument(_instrument())
    flipper = venue.account(AgentId("flipper"))

    _send(venue, "deep", B, _priced(venue, "50"), 4)
    _send(venue, "flipper", S, _priced(venue, "50"), 4)
    assert flipper.positions["F"].quantity == -4

    _send(venue, "deep", S, _priced(venue, "9500"), 11)
    events = _send(venue, "flipper", B, _priced(venue, "9500"), 11)

    assert _reason(events) is RejectReason.INSUFFICIENT_COLLATERAL
    assert flipper.positions["F"].quantity == -4, "the flip went through anyway"
    assert int(flipper.free_cash) >= 0
    assert venue.conservation_check() == 0


def test_a_flip_the_account_can_pay_for_still_goes_through():
    """The permissive half again. An account with the cash to absorb the loss
    and collateralise what it is left holding must still be able to turn its
    position around -- a check that refused every flip would be a market where
    nobody can change their mind.
    """
    venue = Venue(
        "arena",
        starting_cash=10_000_000_000,
        balances={AgentId("flipper"): 200_000},
    )
    venue.list_instrument(_instrument())
    flipper = venue.account(AgentId("flipper"))

    _send(venue, "deep", B, _priced(venue, "50"), 4)
    _send(venue, "flipper", S, _priced(venue, "50"), 4)
    _send(venue, "deep", S, _priced(venue, "9500"), 11)
    assert _reasons(_send(venue, "flipper", B, _priced(venue, "9500"), 11)) == []

    assert flipper.positions["F"].quantity == 7
    assert int(flipper.positions["F"].realized_pnl) == -37_800_000_000
    assert int(flipper.free_cash) >= 0
    assert venue.conservation_check() == 0

