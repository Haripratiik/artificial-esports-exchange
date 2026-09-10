"""Matching engine tests.

This engine's job is to be the oracle the C++ port is validated against, so a
subtle bug here would be transcribed into the port and validated into looking
correct. The suite is therefore built around invariants that must hold for *any*
command stream, not only around hand-picked scenarios:

    * quantity is conserved: matching neither creates nor destroys it
    * the book never crosses: best bid is always below best ask
    * price-time priority is respected
    * identical command streams produce identical event streams
"""

from __future__ import annotations

import random

import pytest

from arena.exchange.engine import MatchingEngine
from arena.exchange.session import SessionState
from arena.exchange.events import (
    Acknowledged,
    Cancel,
    Cancelled,
    Filled,
    Rejected,
    Replace,
    Replaced,
    Submit,
    Traded,
)
from arena.exchange.types import (
    AgentId,
    OrderId,
    OrderStatus,
    OrderType,
    PegReference,
    Price,
    Quantity,
    RejectReason,
    SelfTradePrevention,
    Side,
    TimeInForce,
)

A = AgentId("alice")
B = AgentId("bob")
C = AgentId("carol")


def limit(agent, side, price, qty, tif=TimeInForce.GTC) -> Submit:
    return Submit(agent, side, Quantity(qty), Price(price), OrderType.LIMIT, tif)


def market(agent, side, qty, tif=TimeInForce.IOC) -> Submit:
    return Submit(agent, side, Quantity(qty), None, OrderType.MARKET, tif)


def trades(events) -> list[Traded]:
    return [e for e in events if isinstance(e, Traded)]


def acked(events) -> Acknowledged:
    return next(e for e in events if isinstance(e, Acknowledged))


@pytest.fixture
def engine() -> MatchingEngine:
    return MatchingEngine("EMBER_OBJECTIVE_WR")


# --------------------------------------------------------------------------
# Resting and crossing
# --------------------------------------------------------------------------


def test_non_crossing_orders_rest(engine):
    engine.apply(limit(A, Side.BUY, 100, 10))
    engine.apply(limit(B, Side.SELL, 102, 10))
    book = engine.book.snapshot()

    assert book.best_bid == 100
    assert book.best_ask == 102
    assert book.spread == 2
    assert book.mid == 101.0
    assert engine.tape == ()


def test_crossing_order_trades(engine):
    engine.apply(limit(A, Side.SELL, 100, 10))
    events = engine.apply(limit(B, Side.BUY, 100, 10))

    (trade,) = trades(events)
    assert trade.price == 100
    assert trade.quantity == 10
    assert trade.aggressor_side is Side.BUY
    assert engine.book.snapshot().best_bid is None


def test_trade_prints_at_the_resting_price_not_the_aggressors(engine):
    """Price improvement accrues to the taker, and the passive side set the terms.

    A buyer willing to pay 105 that hits a 100 offer pays 100. Printing the
    aggressor's limit instead would inflate every effective-spread measurement.
    """
    engine.apply(limit(A, Side.SELL, 100, 10))
    events = engine.apply(limit(B, Side.BUY, 105, 10))
    assert trades(events)[0].price == 100


def test_partial_fill_leaves_the_remainder_resting(engine):
    engine.apply(limit(A, Side.SELL, 100, 4))
    events = engine.apply(limit(B, Side.BUY, 100, 10))

    assert trades(events)[0].quantity == 4
    book = engine.book.snapshot()
    assert book.best_bid == 100
    assert book.bids[0][1] == 6


def test_an_order_walks_multiple_levels(engine):
    engine.apply(limit(A, Side.SELL, 100, 5))
    engine.apply(limit(B, Side.SELL, 101, 5))
    engine.apply(limit(C, Side.SELL, 102, 5))

    # A fourth agent, because A already has resting offers here and self-match
    # prevention would cancel them rather than trade against them.
    events = engine.apply(limit(AgentId("dave"), Side.BUY, 102, 12))
    prints = trades(events)

    assert [(int(t.price), int(t.quantity)) for t in prints] == [(100, 5), (101, 5), (102, 2)]
    assert engine.book.snapshot().best_ask == 102


# --------------------------------------------------------------------------
# Priority
# --------------------------------------------------------------------------


def test_price_priority_beats_time(engine):
    """A better price trades first even if it arrived later."""
    engine.apply(limit(A, Side.SELL, 101, 10))
    later = acked(engine.apply(limit(B, Side.SELL, 100, 10)))

    events = engine.apply(limit(C, Side.BUY, 101, 10))
    (trade,) = trades(events)

    assert trade.price == 100
    assert trade.sell_order_id == later.order_id


def test_time_priority_within_a_price_level(engine):
    """Equal prices trade in arrival order, first in first out."""
    first = acked(engine.apply(limit(A, Side.SELL, 100, 10)))
    second = acked(engine.apply(limit(B, Side.SELL, 100, 10)))

    events = engine.apply(limit(C, Side.BUY, 100, 15))
    prints = trades(events)

    assert [t.sell_order_id for t in prints] == [first.order_id, second.order_id]
    assert [int(t.quantity) for t in prints] == [10, 5]


def test_cancelling_the_front_order_promotes_the_next(engine):
    first = acked(engine.apply(limit(A, Side.SELL, 100, 10)))
    second = acked(engine.apply(limit(B, Side.SELL, 100, 10)))
    engine.apply(Cancel(A, first.order_id))

    events = engine.apply(limit(C, Side.BUY, 100, 10))
    assert trades(events)[0].sell_order_id == second.order_id


# --------------------------------------------------------------------------
# Order types and time in force
# --------------------------------------------------------------------------


def test_market_order_sweeps_the_book(engine):
    engine.apply(limit(A, Side.SELL, 100, 5))
    engine.apply(limit(B, Side.SELL, 200, 5))

    events = engine.apply(market(C, Side.BUY, 8))
    assert [int(t.price) for t in trades(events)] == [100, 200]


def test_market_order_with_no_liquidity_cancels(engine):
    events = engine.apply(market(A, Side.BUY, 10))
    assert trades(events) == []
    assert any(isinstance(e, Cancelled) for e in events)
    assert engine.book.snapshot().best_bid is None


def test_market_order_may_not_rest(engine):
    """An unpriced resting order would match anything, forever."""
    events = engine.apply(
        Submit(A, Side.BUY, Quantity(10), None, OrderType.MARKET, TimeInForce.GTC)
    )
    (rejection,) = [e for e in events if isinstance(e, Rejected)]
    assert rejection.reason is RejectReason.MARKET_ORDER_MUST_BE_IOC


def test_ioc_takes_what_it_can_and_cancels_the_rest(engine):
    engine.apply(limit(A, Side.SELL, 100, 4))
    events = engine.apply(limit(B, Side.BUY, 100, 10, TimeInForce.IOC))

    assert trades(events)[0].quantity == 4
    cancel = next(e for e in events if isinstance(e, Cancelled))
    assert cancel.remaining == 6
    assert engine.book.snapshot().best_bid is None


def test_fok_fills_completely_or_not_at_all(engine):
    engine.apply(limit(A, Side.SELL, 100, 4))

    rejected = engine.apply(limit(B, Side.BUY, 100, 10, TimeInForce.FOK))
    assert trades(rejected) == []
    assert any(
        isinstance(e, Rejected) and e.reason is RejectReason.FOK_NOT_FILLABLE
        for e in rejected
    )
    # The resting sell is untouched: nothing was consumed and unwound.
    assert engine.book.snapshot().best_ask == 100

    filled = engine.apply(limit(C, Side.BUY, 100, 4, TimeInForce.FOK))
    assert trades(filled)[0].quantity == 4


def test_fok_spanning_several_levels_is_allowed(engine):
    engine.apply(limit(A, Side.SELL, 100, 4))
    engine.apply(limit(B, Side.SELL, 101, 6))
    events = engine.apply(limit(C, Side.BUY, 101, 10, TimeInForce.FOK))
    assert sum(int(t.quantity) for t in trades(events)) == 10


def test_cancelled_orders_do_not_inflate_reported_depth(engine):
    """A tombstone mid-queue must not be counted as available liquidity.

    Cancellation tombstones rather than splices, so the level's queue still
    holds the dead order until something prunes past it. Summing the queue
    counted it; the level's maintained total does not.
    """
    first = acked(engine.apply(limit(A, Side.SELL, 100, 10)))
    engine.apply(limit(B, Side.SELL, 100, 10))
    third = acked(engine.apply(limit(C, Side.SELL, 100, 10)))

    # Cancel the middle one by cancelling first then third would prune; cancel
    # a strictly interior order instead.
    engine.apply(Cancel(A, first.order_id))       # front: pruned on next touch
    engine.apply(Cancel(C, third.order_id))       # interior relative to nothing

    assert int(engine.book.depth_at(Side.SELL, Price(100))) == 10
    assert engine.book.snapshot().asks == ((100, 10),)


def test_fill_or_kill_is_not_fooled_by_dead_liquidity(engine):
    """The consequence of over-reported depth, and why it mattered.

    ``_fillable`` reads the level totals. If a cancelled order still counted, a
    FOK order would be accepted as satisfiable and then partially fill, exactly
    what fill-or-kill exists to prevent.
    """
    engine.apply(limit(A, Side.SELL, 100, 5))
    doomed = acked(engine.apply(limit(B, Side.SELL, 100, 20)))
    engine.apply(Cancel(B, doomed.order_id))

    events = engine.apply(limit(C, Side.BUY, 100, 15, TimeInForce.FOK))

    assert trades(events) == [], "FOK filled against liquidity that was cancelled"
    assert any(
        isinstance(e, Rejected) and e.reason is RejectReason.FOK_NOT_FILLABLE
        for e in events
    )
    # And the genuine five lots are untouched.
    assert int(engine.book.depth_at(Side.SELL, Price(100))) == 5


def test_fill_or_kill_is_not_admitted_by_the_taker_s_own_liquidity(engine):
    """Quantity you own is not quantity you can take.

    Self-trade prevention cancels the taker's own resting lots rather than
    printing against them, so counting them when deciding whether the whole
    order can fill admits an order that then partially fills. Measured before
    the fix: against its own five lots at 100 and somebody else's five at 101,
    a fill-or-kill buy for ten printed **five** at 101 while prevention
    cancelled the rest of what the check had counted.

    Both the engine and ``tests/reference_matcher.py`` counted it, so both had
    to move in the same step. Correcting one alone made twelve differential
    tests disagree, and a harness that disagrees is a worse failure than the
    one it reports, while a harness that agrees on the wrong answer reports
    nothing at all.
    """
    engine.apply(limit(A, Side.SELL, 100, 5))
    engine.apply(limit(B, Side.SELL, 101, 5))

    events = engine.apply(limit(A, Side.BUY, 101, 10, TimeInForce.FOK))

    assert trades(events) == [], "FOK filled against quantity prevention would cancel"
    assert any(
        isinstance(e, Rejected) and e.reason is RejectReason.FOK_NOT_FILLABLE
        for e in events
    )
    # Rejected before any walk, so nothing was disturbed on the way out: the
    # taker keeps its own resting lots and the genuine liquidity is untouched.
    assert int(engine.book.depth_at(Side.SELL, Price(100))) == 5
    assert int(engine.book.depth_at(Side.SELL, Price(101))) == 5


@pytest.mark.parametrize("quantity", [0, -5])
def test_non_positive_quantity_is_rejected(engine, quantity):
    events = engine.apply(Submit(A, Side.BUY, Quantity(quantity), Price(100)))
    assert events[0].reason is RejectReason.INVALID_QUANTITY


def test_limit_order_without_a_price_is_rejected(engine):
    events = engine.apply(Submit(A, Side.BUY, Quantity(10), None, OrderType.LIMIT))
    assert events[0].reason is RejectReason.LIMIT_ORDER_REQUIRES_PRICE


# --------------------------------------------------------------------------
# Cancel and replace
# --------------------------------------------------------------------------


def test_cancel_removes_resting_quantity(engine):
    ack = acked(engine.apply(limit(A, Side.BUY, 100, 10)))
    events = engine.apply(Cancel(A, ack.order_id))

    assert isinstance(events[0], Cancelled)
    assert events[0].remaining == 10
    assert engine.book.snapshot().best_bid is None


def test_cancelling_twice_is_rejected(engine):
    ack = acked(engine.apply(limit(A, Side.BUY, 100, 10)))
    engine.apply(Cancel(A, ack.order_id))
    events = engine.apply(Cancel(A, ack.order_id))
    assert events[0].reason is RejectReason.ALREADY_TERMINAL


def test_cancelling_someone_elses_order_is_rejected(engine):
    ack = acked(engine.apply(limit(A, Side.BUY, 100, 10)))
    events = engine.apply(Cancel(B, ack.order_id))

    assert events[0].reason is RejectReason.NOT_ORDER_OWNER
    assert engine.book.snapshot().best_bid == 100


def test_cancelling_an_unknown_order_is_rejected(engine):
    events = engine.apply(Cancel(A, OrderId(999)))
    assert events[0].reason is RejectReason.UNKNOWN_ORDER


def test_shrinking_at_the_same_price_keeps_queue_position(engine):
    """The rule that decides whether a market maker is honest or magic.

    Reducing size asks for nothing the order was not already entitled to, so it
    keeps its place. Alice shrinks and must still trade before Bob.
    """
    alice = acked(engine.apply(limit(A, Side.BUY, 100, 10)))
    engine.apply(limit(B, Side.BUY, 100, 10))

    replaced = engine.apply(Replace(A, alice.order_id, Quantity(5)))
    assert isinstance(replaced[0], Replaced)
    assert replaced[0].kept_priority is True

    events = engine.apply(limit(C, Side.SELL, 100, 5))
    assert trades(events)[0].buy_order_id == alice.order_id


def test_increasing_size_loses_queue_position(engine):
    """A bigger claim goes to the back. Otherwise size buys priority."""
    alice = acked(engine.apply(limit(A, Side.BUY, 100, 10)))
    bob = acked(engine.apply(limit(B, Side.BUY, 100, 10)))

    replaced = engine.apply(Replace(A, alice.order_id, Quantity(20)))
    assert replaced[0].kept_priority is False

    events = engine.apply(limit(C, Side.SELL, 100, 5))
    assert trades(events)[0].buy_order_id == bob.order_id


def test_changing_price_loses_queue_position(engine):
    alice = acked(engine.apply(limit(A, Side.BUY, 100, 10)))
    engine.apply(limit(B, Side.BUY, 101, 10))

    replaced = engine.apply(Replace(A, alice.order_id, Quantity(10), Price(101)))
    assert replaced[0].kept_priority is False

    events = engine.apply(limit(C, Side.SELL, 101, 10))
    assert trades(events)[0].buy_order_id != alice.order_id


def test_replacing_into_a_crossing_price_trades_immediately(engine):
    engine.apply(limit(A, Side.SELL, 100, 10))
    bid = acked(engine.apply(limit(B, Side.BUY, 98, 10)))

    events = engine.apply(Replace(B, bid.order_id, Quantity(10), Price(100)))
    assert trades(events)[0].price == 100


def test_shrinking_is_not_a_fill(engine):
    """Nothing traded, so nothing may claim to have traded.

    The shrink used to be routed through the same book operation as a fill,
    which reduces ``remaining`` and leaves ``quantity`` alone. On an order for
    a hundred shrunk to sixty that read as forty lots filled and a status of
    partially-filled, against an empty tape. And every reconciliation
    downstream computes what an order traded from exactly those two numbers.
    """
    ack = acked(engine.apply(limit(A, Side.SELL, 50, 100)))
    engine.apply(Replace(A, ack.order_id, Quantity(60)))

    order = engine.book.get(ack.order_id)
    assert engine.tape == ()
    assert int(order.filled) == 0
    assert order.status is OrderStatus.NEW
    assert (int(order.quantity), int(order.remaining)) == (60, 60)
    assert int(engine.book.depth_at(Side.SELL, Price(50))) == 60


def test_a_shrink_after_a_partial_fill_keeps_what_really_traded(engine):
    """The same arithmetic, on an order that has genuinely traded some of itself.

    Shrinking must take the same amount off the quantity as off the remainder,
    or the fills already printed are counted twice.
    """
    ack = acked(engine.apply(limit(A, Side.SELL, 50, 30)))
    engine.apply(limit(B, Side.BUY, 50, 20))
    order = engine.book.get(ack.order_id)
    assert (int(order.filled), int(order.remaining)) == (20, 10)

    engine.apply(Replace(A, ack.order_id, Quantity(4)))
    assert int(order.filled) == 20, "the shrink was counted as a fill"
    assert int(order.remaining) == 4
    assert order.status is OrderStatus.PARTIALLY_FILLED


def test_a_post_only_order_stays_post_only_through_a_replace(engine):
    """A replace is the same order at a new price, and the promise travels with it.

    Post-only exists so that an order can never pay the taker fee, and the
    promise lived only in the time-in-force of the command that created it,
    which has been processed and thrown away by the time a replace arrives. A
    post-only offer resting at 105 over a bid of 100, replaced to 100, printed
    ten lots as the aggressor: exactly the thing the order type forbids.
    """
    engine.apply(limit(A, Side.BUY, 100, 10))
    ask = acked(engine.apply(limit(B, Side.SELL, 105, 10, TimeInForce.POST_ONLY)))

    events = engine.apply(Replace(B, ask.order_id, Quantity(10), Price(100)))

    assert trades(events) == []
    assert events[0].reason is RejectReason.POST_ONLY_WOULD_CROSS
    assert engine.book.snapshot().best_ask == 105, "the refused replace moved it anyway"
    assert engine.book.snapshot().best_bid == 100


def test_a_post_only_order_may_still_be_replaced_where_it_cannot_cross(engine):
    """The refusal is about crossing, not about post-only orders being frozen."""
    engine.apply(limit(A, Side.BUY, 100, 10))
    ask = acked(engine.apply(limit(B, Side.SELL, 105, 10, TimeInForce.POST_ONLY)))

    events = engine.apply(Replace(B, ask.order_id, Quantity(10), Price(101)))
    assert isinstance(events[0], Replaced)
    assert engine.book.snapshot().best_ask == 101


# --------------------------------------------------------------------------
# Invariants under random flow
# --------------------------------------------------------------------------


def random_commands(seed: int, count: int = 400) -> list:
    rng = random.Random(seed)
    agents = [A, B, C]
    commands = []
    for _ in range(count):
        agent = rng.choice(agents)
        side = rng.choice([Side.BUY, Side.SELL])
        roll = rng.random()
        if roll < 0.72:
            commands.append(
                limit(agent, side, rng.randint(95, 105), rng.randint(1, 20))
            )
        elif roll < 0.85:
            commands.append(market(agent, side, rng.randint(1, 10)))
        else:
            commands.append(
                limit(agent, side, rng.randint(95, 105), rng.randint(1, 20), TimeInForce.IOC)
            )
    return commands


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 11, 42])
def test_book_never_crosses(seed):
    """Best bid must always sit strictly below best ask.

    A crossed book means a trade that should have happened did not, and it is
    the single most damaging silent failure a matching engine can have.
    """
    engine = MatchingEngine()
    for command in random_commands(seed):
        engine.apply(command)
        book = engine.book.snapshot()
        if book.best_bid is not None and book.best_ask is not None:
            assert book.best_bid < book.best_ask, f"crossed book at seed {seed}"


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 11, 42])
def test_quantity_is_conserved(seed):
    """Matching moves quantity; it never creates or destroys it.

    For every order: submitted = filled + remaining + cancelled. Summed over the
    book, traded volume must reconcile against fills exactly.
    """
    engine = MatchingEngine()
    events = engine.apply_all(random_commands(seed))

    filled_by_side: dict[Side, int] = {Side.BUY: 0, Side.SELL: 0}
    for event in events:
        if isinstance(event, Filled):
            filled_by_side[event.side] += int(event.quantity)

    traded = sum(int(t.quantity) for t in engine.tape)
    # Every trade has exactly one buyer and one seller, so each side's filled
    # volume must equal total traded volume.
    assert filled_by_side[Side.BUY] == traded
    assert filled_by_side[Side.SELL] == traded


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 11, 42])
def test_depth_matches_the_sum_of_resting_orders(seed):
    """Aggregated L2 depth must equal what is actually in the queues."""
    engine = MatchingEngine()
    engine.apply_all(random_commands(seed))

    book = engine.book.snapshot(levels=1 << 20)
    for side, levels in ((Side.BUY, book.bids), (Side.SELL, book.asks)):
        for price, quantity in levels:
            actual = sum(
                o.remaining
                for o in engine.book.resting_orders
                if o.side is side and o.price == price
            )
            assert int(quantity) == actual


def rich_commands(seed: int, count: int = 250) -> list:
    """Random flow that uses the whole order-type surface rather than a corner.

    ``random_commands`` above sends limits, markets and immediate-or-cancels,
    which is the matching core and none of the machinery layered on top of it.
    Every defect this file's newest tests were written for was found by adding
    the rest (icebergs, stops, pegs, minimum quantities, post-only, and
    cancels and replaces against whatever happens to be resting) and then
    asserting the same structural invariants after every single command. A
    generator that only sends the easy order types is a generator that only
    finds bugs in the easy paths.
    """
    rng = random.Random(seed)
    agents = [A, B, C]
    commands: list = []
    resting: list = []
    for _ in range(count):
        agent = rng.choice(agents)
        side = rng.choice([Side.BUY, Side.SELL])
        quantity = Quantity(rng.randint(1, 20))
        price = Price(rng.randint(95, 105))
        roll = rng.random()

        if roll < 0.08 and resting:
            order_id, owner = rng.choice(resting)
            commands.append(Cancel(owner, order_id))
        elif roll < 0.18 and resting:
            order_id, owner = rng.choice(resting)
            commands.append(
                Replace(
                    owner,
                    order_id,
                    Quantity(rng.randint(1, 25)),
                    price if rng.random() < 0.5 else None,
                )
            )
        elif roll < 0.26:
            commands.append(market(agent, side, rng.randint(1, 10)))
        elif roll < 0.36:
            trigger = Price(rng.randint(95, 105))
            commands.append(
                Submit(
                    agent,
                    side,
                    quantity,
                    price if rng.random() < 0.5 else None,
                    OrderType.STOP_LIMIT if rng.random() < 0.5 else OrderType.STOP,
                    TimeInForce.GTC,
                    stop_price=trigger,
                )
            )
        elif roll < 0.46:
            commands.append(
                Submit(
                    agent,
                    side,
                    quantity,
                    None,
                    OrderType.PEGGED,
                    rng.choice([TimeInForce.GTC, TimeInForce.POST_ONLY]),
                    peg_to=rng.choice(list(PegReference)),
                    peg_offset=rng.randint(-2, 2),
                    display_size=rng.choice([0, 0, 3]),
                )
            )
        else:
            display = rng.choice([0, 0, 0, 3, 5])
            minimum = rng.choice([0, 0, 0, 0, min(3, quantity), min(5, quantity)])
            if display and minimum > display:
                minimum = 0
            commands.append(
                Submit(
                    agent,
                    side,
                    quantity,
                    price,
                    OrderType.LIMIT,
                    rng.choice(
                        [TimeInForce.GTC] * 3
                        + [TimeInForce.IOC, TimeInForce.FOK, TimeInForce.POST_ONLY]
                    ),
                    display_size=display,
                    min_quantity=minimum,
                )
            )
        # A plausible id to aim a later cancel or replace at. Ids are assigned
        # in order, so this is the id the command just built will be given if
        # the engine accepts it.
        resting.append((OrderId(len(commands)), agent))
    return commands


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 11, 35, 42, 68])
def test_every_level_reports_exactly_what_is_resting_on_it(seed):
    """A level's running total must equal the slices really queued at it.

    Maintained incrementally, so any operation that changes a queue without
    changing the total by the same amount corrupts it silently and permanently.
    A shrinking replace routed through the fill path did exactly that: it
    reduced a level by lots taken out of an iceberg's hidden reserve, which the
    level had never counted, and drove one to **-3** while five lots rested on
    it and the published depth read **0**.
    """
    engine = MatchingEngine()
    for command in rich_commands(seed):
        engine.apply(command)
        for side in (Side.BUY, Side.SELL):
            for price, level in engine.book._levels[side].items():
                live = [
                    o for o in level.orders if not o.status.terminal and o.remaining > 0
                ]
                assert int(level.total) == sum(int(o.shown) for o in live), (
                    f"level {side.value}@{int(price)} desynchronised"
                )
                for order in live:
                    assert 0 < int(order.shown) <= int(order.remaining)
                    assert int(order.price) == int(price)


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 11, 23, 29, 38, 42, 55, 66])
def test_nothing_rests_crossed_without_a_minimum_to_explain_it(seed):
    """A cross is allowed only where the liquidity is conditional.

    Minimum-quantity orders can sit on top of each other while neither is
    allowed to trade, which is measured and deliberate. A cross between two
    orders that carry no minimum has no such excuse, and one was reachable: a
    stop released before its aggressor's own remainder had reached the book
    matched a book that was missing it, and rested where it was about to rest
    through. Bid 101 against ask 99, with nothing to clear it.
    """
    engine = MatchingEngine()
    for command in rich_commands(seed):
        engine.apply(command)
        live = [
            o
            for o in engine.book.resting_orders
            if not o.min_quantity and abs(int(o.price)) < (1 << 61)
        ]
        best_bid = max(
            (int(o.price) for o in live if o.side is Side.BUY), default=None
        )
        best_ask = min(
            (int(o.price) for o in live if o.side is Side.SELL), default=None
        )
        if best_bid is not None and best_ask is not None:
            assert best_bid < best_ask, f"crossed at {best_bid}/{best_ask}"


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 11, 42])
def test_no_order_leaves_the_engine_without_saying_so(seed):
    """Every order is working, or something in the stream says it is not.

    Two ways out were found by asking. A stop is taken off the parked list the
    moment a cascade claims it, so when the cascade hit its depth bound the
    stops it had claimed and not yet run were simply dropped: no order, no
    cancellation, and a later cancel answered ``unknown_order``. And a stop
    that *did* run was minted a fresh order id on the way through, so the id it
    was acknowledged under, the one its owner holds and the venue reserves
    against, referred to nothing: acknowledged as **4**, traded as **6**, and a
    cancel of 4 rejected as unknown.

    Working here means one of the four places an order can legitimately be: in
    the book, parked as a stop, waiting as a peg with no price yet, or finished
    with an event that says so.
    """
    engine = MatchingEngine()
    events = engine.apply_all(rich_commands(seed))

    ended = {
        e.order_id
        for e in events
        if isinstance(e, (Cancelled, Rejected)) and e.order_id is not None
    }
    working = {o.order_id for o in engine.book.resting_orders}
    parked = {stop.order_id for stop in engine._stops}
    # A peg whose reference has gone is live and has no price, so it is in no
    # level and no book query can find it. Not lost: waiting.
    waiting = {peg.order.order_id for peg in engine._pegs}
    for event in events:
        if not isinstance(event, Acknowledged):
            continue
        order = engine.book.get(event.order_id)
        filled = order is not None and order.status is OrderStatus.FILLED
        assert (
            event.order_id in working
            or event.order_id in parked
            or event.order_id in waiting
            or event.order_id in ended
            or filled
        ), f"order {int(event.order_id)} vanished"


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 9, 11, 12, 19, 42])
def test_the_conditional_order_types_keep_their_conditions(seed):
    """Post-only never takes; an aggressor's minimum is never undercut.

    Both are checked per command, because both are statements about one pass of
    the matcher: post-only about every execution in it, a minimum about the
    total of them. A minimum for five that came away with four is not a small
    error, it is the field not working.

    This once carried an exception. A fill-or-kill order could be admitted on
    the strength of quantity its own owner was resting and then prevented from
    taking it, so the assertion allowed a partial fill whenever prevention had
    cancelled something in the same pass. That over-count is fixed on both the
    engine and the reference matcher, which had to move together, so the
    exception is gone and the term now means what it says: nothing, or all of
    it.
    """
    engine = MatchingEngine()
    for command in rich_commands(seed):
        events = engine.apply(command)
        ack = next((e for e in events if isinstance(e, Acknowledged)), None)
        if not isinstance(command, Submit) or ack is None:
            continue
        mine = [e for e in events if isinstance(e, Filled) and e.order_id == ack.order_id]
        took = sum(int(e.quantity) for e in mine if e.aggressor)

        if command.time_in_force is TimeInForce.POST_ONLY:
            assert took == 0, "a post-only order took"
        if command.min_quantity:
            assert took == 0 or took >= command.min_quantity
        if command.time_in_force is TimeInForce.FOK:
            assert took in (0, int(command.quantity)), (
                f"a fill-or-kill order for {int(command.quantity)} took {took}"
            )


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 11, 42])
def test_identical_command_streams_produce_identical_events(seed):
    """The acceptance test for the C++ port, run against the reference itself."""
    commands = random_commands(seed)

    first = [e.to_dict() for e in MatchingEngine().apply_all(commands)]
    second = [e.to_dict() for e in MatchingEngine().apply_all(commands)]

    assert first == second
    assert [e["sequence"] for e in first] == list(range(1, len(first) + 1))


def test_sequence_numbers_are_gapless_and_ordered(engine):
    events = engine.apply_all(random_commands(5, count=120))
    sequences = [int(e.sequence) for e in events]
    assert sequences == sorted(sequences)
    assert sequences == list(range(1, len(sequences) + 1))


# --------------------------------------------------------------------------
# Self-match prevention
# --------------------------------------------------------------------------


def test_an_agent_cannot_trade_with_itself(engine):
    """Wash trades net to zero, which is exactly why they are dangerous.

    A market maker quoting both sides crosses with itself every time it
    requotes; its cancels are still in flight when the new quote arrives. The
    position nets, the PnL nets, and nothing looks wrong. What it destroys is
    the tape: before this was fixed, 90% of the live market's volume was one
    agent trading with itself, so every price, every volume figure and every
    impact measurement derived from them was fiction.
    """
    engine.apply(limit(A, Side.SELL, 100, 10))
    events = engine.apply(limit(A, Side.BUY, 100, 10))

    assert trades(events) == []
    assert any(isinstance(e, Cancelled) for e in events)


def test_the_stale_quote_goes_and_the_order_trades_with_everyone_else(engine):
    """Cancel-oldest is right for the case that actually occurs.

    The resting order is a stale quote its owner has already tried to cancel,
    so removing it and continuing against other participants is what the agent
    meant to happen.
    """
    engine.apply(limit(A, Side.SELL, 100, 10))   # A's stale offer
    engine.apply(limit(B, Side.SELL, 100, 10))   # B's genuine offer

    events = engine.apply(limit(A, Side.BUY, 100, 15))
    prints = trades(events)

    assert len(prints) == 1
    assert int(prints[0].quantity) == 10          # traded with B only
    assert engine.book.snapshot().best_bid == 100  # A's remainder rests


@pytest.mark.parametrize(
    "policy,expect_resting_bid",
    [
        (SelfTradePrevention.CANCEL_OLDEST, 100),
        (SelfTradePrevention.CANCEL_NEWEST, None),
        (SelfTradePrevention.CANCEL_BOTH, None),
    ],
)
def test_prevention_policies(policy, expect_resting_bid):
    engine = MatchingEngine("X", self_trade_prevention=policy)
    engine.apply(limit(A, Side.SELL, 100, 10))
    events = engine.apply(limit(A, Side.BUY, 100, 10))

    assert trades(events) == []
    assert engine.book.snapshot().best_bid == expect_resting_bid


def test_prevention_can_be_disabled_for_study(engine):
    """The raw behaviour stays reachable, so its effect can be measured."""
    permissive = MatchingEngine("X", self_trade_prevention=SelfTradePrevention.ALLOW)
    permissive.apply(limit(A, Side.SELL, 100, 10))
    assert len(trades(permissive.apply(limit(A, Side.BUY, 100, 10)))) == 1


@pytest.mark.parametrize(
    "policy",
    [SelfTradePrevention.CANCEL_NEWEST, SelfTradePrevention.CANCEL_BOTH],
)
def test_a_prevented_order_is_cancelled_rather_than_filled(policy):
    """An order that traded nothing must not report that it traded everything.

    Prevention finishes the incoming order mid-walk, and the arithmetic that
    settles an order afterwards read "nothing left" as "completely filled". The
    tape was empty, a ``Cancelled`` event had gone out for all ten lots, and
    the order's own record said status **filled** with ``filled`` of **10**,
    the two numbers every position reconciliation is built on, both describing
    a trade that never printed.
    """
    engine = MatchingEngine("X", self_trade_prevention=policy)
    engine.apply(limit(A, Side.SELL, 100, 10))
    ack = acked(engine.apply(limit(A, Side.BUY, 100, 10)))

    order = engine.book.get(ack.order_id)
    assert engine.tape == ()
    assert order.status is OrderStatus.CANCELLED
    assert int(order.filled) == 0
    assert int(order.remaining) == 10


def test_prevention_still_runs_when_the_taker_carries_a_minimum(engine):
    """The minimum decides whether to print, not whether to walk.

    Checking it before the walk started meant the walk never happened, and the
    walk is where an agent meets its own resting orders. A maker that bid
    fourteen at 99 and then offered seven at 98 with a minimum of three kept
    both: the same agent on both sides of a book crossed 99 against 98, which
    neither of them could ever clear. Without the minimum, on the identical
    book, the stale bid is withdrawn and the offer rests alone.
    """
    engine.apply(limit(A, Side.BUY, 99, 14))
    events = engine.apply(
        Submit(A, Side.SELL, Quantity(7), Price(98), min_quantity=3)
    )

    assert trades(events) == []
    assert any(isinstance(e, Cancelled) for e in events), "the stale bid survived"
    book = engine.book.snapshot()
    assert book.best_bid is None
    assert book.best_ask == 98


# --------------------------------------------------------------------------
# The auction
# --------------------------------------------------------------------------


def test_a_wash_dropped_by_the_auction_is_really_gone(engine):
    """Reported cancelled, and cancelled.

    The uncross drops the older side of an agent's own cross, and it took the
    order's quantity out of its level without marking the order terminal. It
    stayed live in the queue: the book published **no bid at all** while a
    continuous sell arriving afterwards filled **ten lots** against an order the
    tape had already cancelled. Invisible and tradeable is the worst of both.
    """
    engine.phase = SessionState.PRE_OPEN
    bid = acked(engine.apply(limit(A, Side.BUY, 100, 10)))
    engine.apply(limit(A, Side.SELL, 100, 10))
    engine.apply(limit(B, Side.SELL, 100, 4))

    events = engine.uncross(Price(100))
    assert [e.order_id for e in events if isinstance(e, Cancelled)] == [bid.order_id]

    engine.phase = SessionState.CONTINUOUS
    assert engine.book.get(bid.order_id).status is OrderStatus.CANCELLED
    assert bid.order_id not in [o.order_id for o in engine.book.resting_orders]
    assert trades(engine.apply(limit(C, Side.SELL, 100, 10))) == []
