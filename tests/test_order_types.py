"""Orders that hide, and orders that wait.

The exchange offered limit and market, with GTC, IOC, FOK and post-only. A
production venue offers twenty or more, and two of the missing ones are missing
in a way that changes what the market *is* rather than what it can express:

**Iceberg orders** trade visibility for queue priority. Size is information --
an order for ten thousand lots announces what you are doing before you have
done any of it -- so it is worked in slices, and each refreshed slice goes to
the back of its level behind everything that arrived while the last one worked.
docs/GAPS.md recorded them as "absent, and they change queue dynamics
materially", which is exactly right and is why they are worth having.

**Stop orders** wait for a price before they exist. They are the classic risk
tool and the classic accelerant: a stop sells into a fall, which pushes the
price down, which triggers more stops. Nothing here prevents that cascade and
nothing should -- being able to *measure* one is most of the reason to model
stops at all.
"""

from __future__ import annotations

import pytest

from arena.exchange.engine import MatchingEngine
from arena.exchange.events import Submit, Traded
from arena.exchange.types import (
    AgentId,
    OrderType,
    Price,
    Quantity,
    RejectReason,
    Side,
    TimeInForce,
)


def _limit(engine, who, side, quantity, price, **kwargs):
    return engine.apply(
        Submit(AgentId(who), side, Quantity(quantity), Price(price), **kwargs)
    )


def _market(engine, who, side, quantity):
    return engine.apply(
        Submit(
            AgentId(who),
            side,
            Quantity(quantity),
            None,
            OrderType.MARKET,
            TimeInForce.IOC,
        )
    )


def _stop(engine, who, side, quantity, trigger, limit=None):
    return engine.apply(
        Submit(
            AgentId(who),
            side,
            Quantity(quantity),
            None if limit is None else Price(limit),
            OrderType.STOP if limit is None else OrderType.STOP_LIMIT,
            TimeInForce.GTC,
            stop_price=Price(trigger),
        )
    )


def _prints(events):
    return [(int(e.quantity), int(e.price)) for e in events if isinstance(e, Traded)]


# --------------------------------------------------------------------------
# Iceberg
# --------------------------------------------------------------------------


def test_an_iceberg_shows_only_its_slice():
    engine = MatchingEngine()
    _limit(engine, "berg", Side.SELL, 100, 50, display_size=10)
    assert engine.book.snapshot().asks == ((Price(50), Quantity(10)),)


def test_a_refreshed_slice_goes_to_the_back_of_the_queue():
    """The price of hiding, and the whole reason it is not simply better.

    A venue that refreshed in place would let one participant hold the front of
    a queue indefinitely while showing a single lot.
    """
    engine = MatchingEngine()
    _limit(engine, "berg", Side.SELL, 100, 50, display_size=10)
    _limit(engine, "plain", Side.SELL, 20, 50)

    events = _market(engine, "buyer", Side.BUY, 25)
    assert _prints(events) == [(10, 50), (15, 50)]

    resting = {o.agent_id: (int(o.remaining), int(o.shown)) for o in engine.book.resting_orders}
    assert resting["berg"] == (90, 10), "the iceberg refreshed"
    assert resting["plain"] == (5, 5), "the order behind it got its turn"


def test_an_iceberg_hides_its_reserve_from_the_depth():
    engine = MatchingEngine()
    _limit(engine, "berg", Side.SELL, 100, 50, display_size=10)
    _market(engine, "buyer", Side.BUY, 10)
    # Ninety still to sell, ten of it visible. Both numbers are true and they
    # are different questions: the depth is what the market can see, and the
    # resting quantity is what is really there.
    assert engine.book.snapshot().asks == ((Price(50), Quantity(10)),)
    assert engine.book.total_resting_quantity == 90


def test_an_iceberg_fills_completely_if_you_keep_taking():
    engine = MatchingEngine()
    _limit(engine, "berg", Side.SELL, 100, 50, display_size=10)
    taken = 0
    for _ in range(10):
        taken += sum(q for q, _p in _prints(_market(engine, "buyer", Side.BUY, 10)))
    assert taken == 100
    assert not engine.book.resting_orders


def test_a_shrinking_replace_leaves_an_iceberg_exactly_where_it_was():
    """The one modification that is promised to cost nothing, costing everything.

    Reducing size at an unchanged price keeps queue position, and the shrink was
    carried out by the same book operation that carries out a fill. On an
    iceberg that is wrong twice. The lots come out of the reserve, which was
    never in the level's published total, so the total was reduced by lots it
    never held: an iceberg for twelve showing three, with four lots behind it,
    shrunk to six left the level reporting **4** against **7** really resting,
    and shrunk to one left it reporting **-3** with a published depth of **0**
    over five live lots. And a spent slice refreshes to the back of its queue,
    so the iceberg was moved behind the order that arrived after it -- while the
    event it produced said ``kept_priority=True``.
    """
    engine = MatchingEngine()
    _limit(engine, "berg", Side.SELL, 12, 96, display_size=3)
    _limit(engine, "behind", Side.SELL, 4, 96)
    assert int(engine.book.depth_at(Side.SELL, Price(96))) == 7

    events = engine.apply(Replace(AgentId("berg"), _order_id(engine, "berg"), Quantity(6)))
    replaced = next(e for e in events if isinstance(e, Replaced))
    assert replaced.kept_priority is True

    level = engine.book._levels[Side.SELL][Price(96)]
    live = [o for o in level.orders if not o.status.terminal and o.remaining > 0]
    assert [o.agent_id for o in live] == ["berg", "behind"], "the shrink cost it its place"
    assert int(level.total) == sum(int(o.shown) for o in live) == 7
    assert int(engine.book.depth_at(Side.SELL, Price(96))) == 7

    # The promise, tested the only way it can be: the next taker meets it first.
    assert _passive_fills(_market(engine, "taker", Side.BUY, 3)) == [("berg", 3)]


def test_shrinking_an_iceberg_below_its_slice_keeps_the_level_honest():
    """The same arithmetic where it went furthest negative.

    Twelve showing three, shrunk to one: the level's running total was reduced
    by eleven when only three of them were ever counted in it.
    """
    engine = MatchingEngine()
    _limit(engine, "berg", Side.SELL, 12, 96, display_size=3)
    _limit(engine, "behind", Side.SELL, 4, 96)
    engine.apply(Replace(AgentId("berg"), _order_id(engine, "berg"), Quantity(1)))

    level = engine.book._levels[Side.SELL][Price(96)]
    live = [o for o in level.orders if not o.status.terminal and o.remaining > 0]
    assert [(o.agent_id, int(o.remaining), int(o.shown)) for o in live] == [
        ("berg", 1, 1),
        ("behind", 4, 4),
    ]
    assert int(level.total) == 5
    assert int(engine.book.depth_at(Side.SELL, Price(96))) == 5


def test_a_fill_or_kill_reaches_an_iceberg_s_reserve():
    """Published depth under-counts an iceberg, and fill-or-kill believed it.

    A hundred-lot iceberg showing ten was published as ten, so a fill-or-kill
    for a hundred was refused as unfillable -- while the identical order sent
    good-till-cancelled filled all hundred against the same book, because each
    exhausted slice refreshes and the walk arrives back at it.
    """
    engine = MatchingEngine()
    _limit(engine, "berg", Side.SELL, 100, 50, display_size=10)
    assert int(engine.book.depth_at(Side.SELL, Price(50))) == 10

    events = _limit(engine, "f", Side.BUY, 100, 50, time_in_force=TimeInForce.FOK)
    assert _reasons(events) == []
    assert sum(q for q, _p in _prints(events)) == 100


def test_an_order_with_no_price_cannot_hide():
    """It never rests, so there is no queue for a reserve to wait in."""
    engine = MatchingEngine()
    events = engine.apply(
        Submit(
            AgentId("a"),
            Side.SELL,
            Quantity(50),
            None,
            OrderType.MARKET,
            TimeInForce.IOC,
            display_size=10,
        )
    )
    assert any(
        getattr(e, "reason", None) is RejectReason.INVALID_QUANTITY for e in events
    )


# --------------------------------------------------------------------------
# Stops
# --------------------------------------------------------------------------


def test_a_stop_is_not_liquidity_and_does_not_appear_as_any():
    """Publishing one would say exactly where the market must go to set off a
    cascade, which is the thing its owner most wants kept quiet."""
    engine = MatchingEngine()
    _limit(engine, "mm", Side.BUY, 20, 100)
    _stop(engine, "a", Side.SELL, 30, trigger=99)

    book = engine.book.snapshot()
    assert book.asks == ()
    assert book.bids == ((Price(100), Quantity(20)),)


def test_a_stop_stays_asleep_until_the_market_reaches_it():
    engine = MatchingEngine()
    _limit(engine, "mm", Side.BUY, 20, 100)
    _stop(engine, "a", Side.SELL, 30, trigger=90)

    _market(engine, "seller", Side.SELL, 5)
    assert len(engine._stops) == 1, "a print at 100 woke a stop set at 90"


def test_a_triggered_stop_trades():
    engine = MatchingEngine()
    for price, quantity in ((100, 20), (99, 20), (98, 20)):
        _limit(engine, "mm", Side.BUY, quantity, price)
    _stop(engine, "a", Side.SELL, 30, trigger=99)

    events = _market(engine, "seller", Side.SELL, 25)
    prints = _prints(events)
    assert prints[:2] == [(20, 100), (5, 99)], "the order that triggered it"
    assert sum(q for q, _p in prints[2:]) == 30, "the stop's own thirty lots"
    assert not engine._stops


def test_one_stop_sets_off_another():
    """A cascade, which is a real thing markets do and this one does not prevent."""
    engine = MatchingEngine()
    for price, quantity in ((100, 20), (99, 20), (98, 20), (97, 20), (96, 60)):
        _limit(engine, "mm", Side.BUY, quantity, price)
    _stop(engine, "a", Side.SELL, 30, trigger=99)
    _stop(engine, "b", Side.SELL, 30, trigger=98)

    events = _market(engine, "seller", Side.SELL, 25)
    assert not engine._stops, "both stops should have gone off"
    assert engine.cascade_depth, "nothing recorded a cascade"
    assert engine.cascade_depth[-1] >= 2, (
        f"the second stop was not set off by the first: {engine.cascade_depth}"
    )
    # Everything that was sold: the original order plus both stops.
    assert sum(q for q, _p in _prints(events)) == 25 + 30 + 30


def test_a_cascade_cannot_run_forever():
    """A chain that never ends is a bug in the model, not an event in a market."""
    engine = MatchingEngine()
    assert engine._max_cascade > 0
    for price in range(200, 100, -1):
        _limit(engine, "mm", Side.BUY, 5, price)
        _stop(engine, f"s{price}", Side.SELL, 5, trigger=price)
    _market(engine, "seller", Side.SELL, 5)
    assert engine.cascade_depth[-1] <= engine._max_cascade


def test_the_cascade_bound_stops_the_chain_without_deleting_orders():
    """The bound limits the chain. It must not be a way for orders to vanish.

    A stop is taken out of the parked list at the moment it is handed to the
    cascade, so anything still waiting when the bound bites had left the engine
    entirely -- no order, no acknowledgement, no cancellation, and a later
    cancel answered ``unknown_order``. Measured on a forty-deep ladder with the
    bound at twenty-four: twenty-four released, fifteen still parked, and
    **one** that simply ceased to exist with nothing in the stream to say so.
    """
    engine = MatchingEngine()
    depth = engine._max_cascade + 16
    for i in range(depth + 1):
        _limit(engine, "mm", Side.BUY, 1, 1000 - i)
    for i in range(depth):
        _stop(engine, "s", Side.SELL, 1, trigger=1000 - i)
    assert len(engine._stops) == depth

    events = _limit(engine, "kick", Side.SELL, 1, 1000)
    released = len([e for e in events if isinstance(e, Acknowledged)]) - 1

    assert engine.cascade_depth[-1] == engine._max_cascade, "the bound did not bite"
    assert released + len(engine._stops) == depth, "a triggered stop disappeared"
    assert not [e for e in events if isinstance(e, Cancelled) and e.remaining == 1]

    # Still live, so the next print that reaches one sets it off.
    remaining = len(engine._stops)
    _limit(engine, "mm", Side.BUY, 1, 1000 - depth)
    _limit(engine, "again", Side.SELL, 1, 1000 - depth)
    assert len(engine._stops) < remaining


def test_a_triggered_stop_keeps_the_id_it_was_acknowledged_under():
    """The agent's handle on its order, and the venue's, must survive the trigger.

    A parked stop is acknowledged under an id: that id is what its owner cancels
    with, and what the venue reserves collateral against. The order it became
    was minted a fresh one, with nothing in the stream linking the two.
    Measured: acknowledged as order **4**, traded as order **6**, and a cancel
    of 4 rejected as ``unknown_order`` with the order neither parked nor in the
    book. The reservation against 4 had nothing left to release it and 6 traded
    under an id nobody had reserved for.
    """
    engine = MatchingEngine()
    for price, quantity in ((100, 20), (99, 20), (98, 40)):
        _limit(engine, "mm", Side.BUY, quantity, price)
    parked = next(
        e.order_id for e in _stop(engine, "a", Side.SELL, 10, trigger=99)
        if isinstance(e, Acknowledged)
    )

    events = _market(engine, "seller", Side.SELL, 30)
    released = [e.order_id for e in events if isinstance(e, Acknowledged)]

    assert parked in released, "the released stop traded under a different id"
    assert any(
        isinstance(e, Filled) and e.order_id == parked and e.aggressor for e in events
    )
    assert engine.book.get(parked) is not None, "the id resolves to nothing"


def test_a_stop_hears_every_print_of_a_walk_not_only_the_last():
    """A walk that sweeps several levels prints at each of them.

    Only the final price was offered to the parked stops, so any trigger the
    walk passed through on the way was dropped. Offers of five at 100 and five
    at 110, a sell stop parked at 100, and a buy for ten: the tape read
    ``[(5, 100), (5, 110)]``, the market had plainly traded at 100, and the stop
    was still parked afterwards. The rest of the cascade already checked every
    print of every released order, so the first round was the odd one out.
    """
    engine = MatchingEngine()
    _limit(engine, "mm", Side.SELL, 5, 100)
    _limit(engine, "mm", Side.SELL, 5, 110)
    _limit(engine, "bid", Side.BUY, 5, 95)
    _stop(engine, "s", Side.SELL, 5, trigger=100)

    events = _limit(engine, "t", Side.BUY, 10, 110)
    assert _prints(events)[:2] == [(5, 100), (5, 110)]
    assert not engine._stops, "the print at 100 was never offered to the stop"
    assert (5, 95) in _prints(events), "the released stop did not trade"


def test_a_stop_is_released_only_once_the_order_that_set_it_off_has_rested():
    """Sequencing, and it decides whether the book ends crossed.

    A triggered stop was released while the aggressor's own unfilled remainder
    was still in flight, so it matched against a book missing liquidity that
    logically preceded it and rested where that remainder was about to rest
    through. Measured: a stop-limit sell for nineteen at 99 set off by a print
    at 97, released before a taker's seven unfilled lots reached the book, left
    **bid 101 against ask 99** -- a spread of minus two, crossed and stuck.
    """
    engine = MatchingEngine()
    _stop(engine, "stopper", Side.SELL, 19, trigger=103, limit=99)
    _limit(engine, "maker", Side.SELL, 5, 97)

    events = _limit(engine, "taker", Side.BUY, 12, 101)

    book = engine.book.snapshot()
    if book.best_bid is not None and book.best_ask is not None:
        assert int(book.best_bid) < int(book.best_ask), "the book ended crossed"
    # The taker's remainder was on the book, so the stop traded against it
    # rather than resting through it.
    assert (7, 101) in _prints(events)


def test_a_replace_that_crosses_sets_off_a_stop():
    """A print is a print, whichever message produced it.

    Stops were released after a submission and not after a modification, so a
    resting offer moved down onto a bid traded ten lots at 100 while a stop
    parked at 100 sat untouched -- and the identical print delivered as a new
    order set the same stop off immediately. Whether a stop fires cannot depend
    on which message the tape came from.
    """
    engine = MatchingEngine()
    _limit(engine, "maker", Side.BUY, 10, 100)
    _limit(engine, "maker", Side.BUY, 10, 95)
    _limit(engine, "mover", Side.SELL, 10, 105)
    _stop(engine, "stopper", Side.SELL, 5, trigger=100)

    events = engine.apply(
        Replace(AgentId("mover"), _order_id(engine, "mover"), Quantity(10), Price(100))
    )

    assert (10, 100) in _prints(events)
    assert not engine._stops, "the replace's print did not reach the stop"
    assert (5, 95) in _prints(events), "the released stop did not trade"


def test_a_stop_limit_will_not_fill_below_its_limit():
    """Which is the trade every stop user actually faces: protection against a
    bad fill, paid for with the risk of no fill at all."""
    engine = MatchingEngine()
    _limit(engine, "mm", Side.BUY, 20, 100)
    _limit(engine, "mm", Side.BUY, 20, 90)
    _stop(engine, "a", Side.SELL, 30, trigger=99, limit=95)

    _market(engine, "seller", Side.SELL, 25)
    prints = _prints(engine.apply(Submit(AgentId("noop"), Side.BUY, Quantity(1), Price(1))))
    resting = [o for o in engine.book.resting_orders if o.agent_id == "a"]
    assert resting, "the stop-limit did not rest"
    assert all(int(o.price) == 95 for o in resting)
    assert engine.book.snapshot().bids[0][0] == Price(90), (
        "it should not have sold into the 90 bid"
    )


def test_a_stop_needs_a_trigger_and_nothing_else_may_have_one():
    engine = MatchingEngine()
    missing = engine.apply(
        Submit(AgentId("a"), Side.SELL, Quantity(5), None, OrderType.STOP, TimeInForce.GTC)
    )
    assert any(
        getattr(e, "reason", None) is RejectReason.INVALID_STOP_PRICE for e in missing
    )

    spurious = engine.apply(
        Submit(
            AgentId("a"),
            Side.SELL,
            Quantity(5),
            Price(100),
            OrderType.LIMIT,
            TimeInForce.GTC,
            stop_price=Price(99),
        )
    )
    assert any(
        getattr(e, "reason", None) is RejectReason.INVALID_STOP_PRICE for e in spurious
    )


def test_a_stop_cannot_be_immediate():
    """"Do this now" and "do this later" are contradictory instructions."""
    engine = MatchingEngine()
    events = engine.apply(
        Submit(
            AgentId("a"),
            Side.SELL,
            Quantity(5),
            None,
            OrderType.STOP,
            TimeInForce.IOC,
            stop_price=Price(99),
        )
    )
    assert any(getattr(e, "reason", None) is not None for e in events)


def test_the_venue_reserves_for_a_stop_the_moment_it_is_parked():
    """The engine releases a triggered stop inside its own matching, which never
    passes back through the collateral check. An unreserved stop would create a
    position the account was never asked to cover."""
    from arena.exchange.session import SessionState
    from arena.market.live import HUMAN_ID
    from arena.market.venue import SymbolCommand
    from arena.sim.time import Timestamp, seconds
    from dashboard.build_market import build

    symbol = "EMBER_OBJECTIVE_WR"
    market = build(seed=7, human_cash=4_000_000)
    market.kernel.start()
    market.kernel.advance(until=seconds(180))
    for moment in range(185, 400, 5):
        market.kernel.advance(until=seconds(moment))
        book = market.venue.engine(symbol).book.snapshot()
        if (
            market.venue.session(symbol) is SessionState.CONTINUOUS
            and book.best_bid is not None
        ):
            break

    instrument = market.venue.registry.require(symbol)
    trigger = float(instrument.from_ticks(book.best_bid)) - 400.0
    market.submit(symbol, "sell", 10, None, stop=f"{trigger:.2f}", trader=None)
    market.kernel.advance(until=Timestamp(int(market.kernel.now) + int(seconds(1))))

    working = market.venue._working.get((HUMAN_ID, symbol), {})
    assert working, "a parked stop reserved nothing"
    assert int(market.venue.conservation_check()) == 0


# --------------------------------------------------------------------------
# Pegged and minimum-quantity
#
# **Pegged orders** quote a position rather than a number. A maker that wants
# to be at the touch and writes it as "bid 4,700" is wrong the moment the touch
# is 4,703, and has to cancel and replace to fix it; a peg says "at the touch"
# and cannot be wrong. What it pays for that is queue priority, every time it
# moves, and the tests below insist on that rather than excusing it.
#
# **Minimum-quantity orders** refuse to be picked off in pieces. Distinct from
# fill-or-kill, which is a statement about the whole order: an MPL order is
# content to fill part of itself and only insists that no part be tiny. It
# spends the same currency an iceberg does -- an aggressor too small to satisfy
# it passes it by, and the order behind it gets the fill.
# --------------------------------------------------------------------------

from arena.exchange.events import (  # noqa: E402
    Acknowledged,
    Cancel,
    Cancelled,
    Filled,
    Rejected,
    Replace,
    Replaced,
)
from arena.exchange.session import SessionState  # noqa: E402
from arena.exchange.types import PegReference  # noqa: E402


def _peg(engine, who, side, quantity, reference, offset=0, tif=TimeInForce.GTC, **kwargs):
    return engine.apply(
        Submit(
            AgentId(who),
            side,
            Quantity(quantity),
            None,
            OrderType.PEGGED,
            tif,
            peg_to=reference,
            peg_offset=offset,
            **kwargs,
        )
    )


def _reasons(events):
    return [e.reason for e in events if isinstance(e, Rejected)]


def _passive_fills(events):
    """Who was filled while resting, and for how much."""
    return [
        (str(e.agent_id), int(e.quantity))
        for e in events
        if isinstance(e, Filled) and not e.aggressor
    ]


def _order_id(engine, who):
    return next(o.order_id for o in engine.book.resting_orders if o.agent_id == who)


# --------------------------------------------------------------------------
# Pegs
# --------------------------------------------------------------------------


def test_a_peg_rests_at_the_price_it_tracks_rather_than_one_it_chose():
    engine = MatchingEngine()
    _limit(engine, "mm", Side.BUY, 10, 100)
    _limit(engine, "mm", Side.SELL, 10, 102)
    _peg(engine, "p", Side.BUY, 5, PegReference.BID)

    assert engine.book.snapshot().bids == ((Price(100), Quantity(15)),)


def test_a_peg_acknowledges_at_the_price_it_will_rest_at():
    """The venue reserves collateral from the acknowledged price.

    A peg that acknowledged its offset would have one tick's worth of cash
    reserved against an order that will rest at a hundred, which is why the
    offset is a field of its own rather than a reuse of ``price``.
    """
    engine = MatchingEngine()
    _limit(engine, "mm", Side.BUY, 10, 100)
    events = _peg(engine, "p", Side.BUY, 5, PegReference.BID, offset=-2)

    acks = [e for e in events if isinstance(e, Acknowledged)]
    assert [e.price for e in acks] == [Price(98)]


def test_a_peg_with_no_price_yet_acknowledges_without_one():
    """Rather than inventing a number. There is nothing dishonest to report."""
    engine = MatchingEngine()
    events = _peg(engine, "p", Side.BUY, 5, PegReference.BID)
    assert [e.price for e in events if isinstance(e, Acknowledged)] == [None]


def test_a_peg_follows_the_touch_when_it_moves():
    engine = MatchingEngine()
    _limit(engine, "mm", Side.BUY, 10, 100)
    _peg(engine, "p", Side.BUY, 5, PegReference.BID)
    _limit(engine, "better", Side.BUY, 8, 103)

    assert engine.book.snapshot().bids == (
        (Price(103), Quantity(13)),
        (Price(100), Quantity(10)),
    ), "the peg did not follow the bid up"


def test_a_peg_does_not_count_itself_in_the_reference_it_tracks():
    """Otherwise it can follow the market up and can never follow it back down.

    A peg that counted its own quantity would be pegged to itself the moment it
    became the touch: every step down is blocked by the price it is already
    quoting. It would ratchet, which looks exactly like a peg that works right
    up until the market falls.
    """
    engine = MatchingEngine()
    _limit(engine, "low", Side.BUY, 10, 98)
    _limit(engine, "high", Side.BUY, 10, 100)
    _peg(engine, "p", Side.BUY, 5, PegReference.BID)
    assert engine.book.snapshot().bids[0] == (Price(100), Quantity(15))

    engine.apply(Cancel(AgentId("high"), _order_id(engine, "high")))
    assert engine.book.snapshot().bids == ((Price(98), Quantity(15)),), (
        "the peg stayed at 100, a price only it was quoting"
    )


def test_repricing_a_peg_costs_it_its_place_in_the_queue():
    """The honest behaviour, and not an implementation limitation.

    A new price is a new claim on a queue other orders were already waiting in.
    A peg that kept its position through a reprice would have a standing
    advantage nobody else can buy: sit at the front of one level, follow the
    touch to another, and still be at the front there. This is the rule a
    replace already obeys, for the same reason.
    """
    engine = MatchingEngine()
    _limit(engine, "ahead", Side.BUY, 10, 100)
    _peg(engine, "p", Side.BUY, 5, PegReference.BID)
    _limit(engine, "behind", Side.BUY, 5, 100)

    # Up to 101 and back again. The peg leaves the queue at 100 from the middle
    # of it and rejoins at the end, behind an order that had been behind it.
    _limit(engine, "better", Side.BUY, 8, 101)
    engine.apply(Cancel(AgentId("better"), _order_id(engine, "better")))

    events = _limit(engine, "seller", Side.SELL, 15, 100)
    assert _passive_fills(events) == [("ahead", 10), ("behind", 5)], (
        "the peg kept its place through a reprice"
    )


def test_a_peg_with_nothing_to_track_rests_inert_rather_than_being_rejected():
    """"There is no best bid yet" is a fact about the market, not an error in
    the order. Rejecting would make a peg unusable at the one moment it is most
    useful -- the open, when nobody has quoted yet."""
    engine = MatchingEngine()
    events = _peg(engine, "p", Side.BUY, 5, PegReference.BID)

    assert not _reasons(events)
    assert engine.book.snapshot().bids == ()
    assert len(engine._pegs) == 1 and not engine._pegs[0].on_book


def test_an_inert_peg_takes_a_price_as_soon_as_there_is_one():
    engine = MatchingEngine()
    _peg(engine, "p", Side.BUY, 5, PegReference.BID)
    _limit(engine, "mm", Side.BUY, 10, 100)

    assert engine.book.snapshot().bids == ((Price(100), Quantity(15)),)


def test_a_peg_that_loses_its_reference_comes_off_the_book():
    """One rule for both directions rather than two.

    Leaving it at the last price it happened to track would turn a peg into a
    stale limit order at exactly the moment the market it was following stopped
    existing -- quoting a number nobody else is quoting, chosen by a reference
    that no longer has a value.
    """
    engine = MatchingEngine()
    _limit(engine, "mm", Side.BUY, 10, 100)
    _peg(engine, "p", Side.BUY, 5, PegReference.BID)
    engine.apply(Cancel(AgentId("mm"), _order_id(engine, "mm")))

    assert engine.book.snapshot().bids == ()
    assert len(engine._pegs) == 1, "the order is waiting, not cancelled"
    assert not engine._pegs[0].on_book


def test_a_peg_offset_moves_it_that_many_ticks_from_the_reference():
    engine = MatchingEngine()
    _limit(engine, "mm", Side.BUY, 10, 100)
    _limit(engine, "mm", Side.SELL, 10, 110)
    _peg(engine, "behind", Side.BUY, 5, PegReference.BID, offset=-3)
    _peg(engine, "inside", Side.SELL, 5, PegReference.ASK, offset=-4)

    book = engine.book.snapshot()
    assert book.bids == ((Price(100), Quantity(10)), (Price(97), Quantity(5)))
    assert book.asks == ((Price(106), Quantity(5)), (Price(110), Quantity(10)))


def test_a_mid_peg_rounds_away_from_the_side_it_would_cross():
    """A one-tick spread has no integer midpoint, so the half tick has to go
    somewhere. Rounding it the other way would make a midpoint order cross the
    spread on every odd-width market, which is the one thing it is not for."""
    engine = MatchingEngine()
    _limit(engine, "mm", Side.BUY, 10, 100)
    _limit(engine, "mm", Side.SELL, 10, 101)
    _peg(engine, "pb", Side.BUY, 5, PegReference.MID)
    _peg(engine, "ps", Side.SELL, 5, PegReference.MID)

    book = engine.book.snapshot()
    assert book.bids == ((Price(100), Quantity(15)),)
    assert book.asks == ((Price(101), Quantity(15)),)
    assert engine.tape == (), "the two midpoint orders crossed each other"


def test_a_mid_peg_needs_both_sides_to_exist():
    engine = MatchingEngine()
    _limit(engine, "mm", Side.BUY, 10, 100)
    _peg(engine, "p", Side.BUY, 5, PegReference.MID)

    assert engine.book.snapshot().bids == ((Price(100), Quantity(10)),)
    assert not engine._pegs[0].on_book


def test_a_peg_to_the_far_side_of_the_book_is_an_order_that_takes():
    """"Pay the offer, whatever the offer is" is a real instruction, and this is
    how it is written. It walks the book, because the offer it consumes is
    replaced by the next one and the peg follows that too."""
    engine = MatchingEngine()
    _limit(engine, "mm", Side.SELL, 4, 102)
    _limit(engine, "mm", Side.SELL, 4, 103)
    events = _peg(engine, "p", Side.BUY, 6, PegReference.ASK)

    assert _prints(events) == [(4, 102), (2, 103)]


def test_a_post_only_peg_declines_to_follow_a_reference_into_a_cross():
    """Post-only promises the order never takes, and a peg does not choose its
    own price. The promise is kept by not following, because there is nothing
    left to reject -- the order was accepted before the touch moved."""
    engine = MatchingEngine()
    _limit(engine, "mm", Side.BUY, 10, 100)
    _limit(engine, "mm", Side.SELL, 10, 102)
    _peg(engine, "p", Side.SELL, 4, PegReference.BID, tif=TimeInForce.POST_ONLY)

    assert engine.tape == ()
    assert engine.book.snapshot().bids == ((Price(100), Quantity(10)),)
    assert not engine._pegs[0].on_book


def test_a_post_only_peg_keeps_its_promise_after_it_has_repriced():
    """Repricing rebuilds the order, and a rebuild that drops something is a hole.

    A peg is taken off the book and put back as a new order every time its
    reference moves, and the new order was built without the post-only flag. So
    the promise survived exactly as long as the peg never moved: after one
    reprice, a replace no longer knew to refuse a crossing price, and a
    post-only peg replaced to 102 **took a lot at 98**. The same lesson as the
    display size a replace used to strip, in a third place.
    """
    engine = MatchingEngine()
    _limit(engine, "mm", Side.SELL, 5, 98)
    _limit(engine, "mm", Side.BUY, 5, 90)
    _peg(engine, "p", Side.BUY, 2, PegReference.BID, tif=TimeInForce.POST_ONLY)

    # Move the reference, so the peg is rebuilt at least once.
    _limit(engine, "other", Side.BUY, 5, 92)
    peg_order = engine._pegs[0].order
    assert int(peg_order.price) == 92, "the peg did not follow the touch"

    events = engine.apply(Replace(AgentId("p"), peg_order.order_id, Quantity(2), Price(98)))
    assert _prints(events) == []
    assert _reasons(events) == [RejectReason.POST_ONLY_WOULD_CROSS]


def test_a_peg_takes_no_price_from_a_book_that_is_only_accumulating():
    """During a call phase the top of the book can be a market-on-open order
    resting at a sentinel so it crosses every candidate. That is not a price,
    and a peg that read it would have joined the auction quoting 2^62."""
    engine = MatchingEngine()
    engine.phase = SessionState.PRE_OPEN
    _limit(engine, "mm", Side.BUY, 10, 100)
    _peg(engine, "p", Side.BUY, 5, PegReference.BID)
    assert engine.book.snapshot().bids == ((Price(100), Quantity(10)),)

    engine.phase = SessionState.CONTINUOUS
    _limit(engine, "far", Side.SELL, 1, 500)
    assert engine.book.snapshot().bids == ((Price(100), Quantity(15)),)


def test_pegs_that_chase_each_other_settle_instead_of_running_forever():
    """Two orders pegged to opposite sides have no fixed point: each one moving
    moves the reference the other reads. The bound is what makes that a bad
    idea rather than a hang."""
    engine = MatchingEngine()
    assert engine._max_peg_passes > 0
    _limit(engine, "mm", Side.BUY, 10, 100)
    _limit(engine, "mm", Side.SELL, 10, 102)
    _peg(engine, "pb", Side.BUY, 4, PegReference.ASK)
    _peg(engine, "ps", Side.SELL, 4, PegReference.BID)

    # Both peg to the far side, so both are aggressive and both trade.
    assert [(int(t.quantity), int(t.price)) for t in engine.tape] == [
        (4, 102),
        (4, 100),
    ]
    assert engine._pegs == []


def test_an_inert_peg_can_still_be_cancelled():
    """It is in no price level and its id resolves to the tombstone of the last
    order it was, so the book cannot answer for it. An order nobody can withdraw
    is worse than one that never rested."""
    engine = MatchingEngine()
    _limit(engine, "mm", Side.BUY, 10, 100)
    _peg(engine, "p", Side.BUY, 5, PegReference.BID)
    engine.apply(Cancel(AgentId("mm"), _order_id(engine, "mm")))
    assert not engine._pegs[0].on_book

    order_id = engine._pegs[0].order.order_id
    events = engine.apply(Cancel(AgentId("p"), order_id))
    assert not _reasons(events)
    assert engine._pegs == []

    # And it stays gone: a reference reappearing must not revive it.
    _limit(engine, "mm2", Side.BUY, 10, 100)
    assert engine.book.snapshot().bids == ((Price(100), Quantity(10)),)


def test_only_the_owner_can_cancel_an_inert_peg():
    engine = MatchingEngine()
    _peg(engine, "p", Side.BUY, 5, PegReference.BID)
    order_id = engine._pegs[0].order.order_id

    events = engine.apply(Cancel(AgentId("thief"), order_id))
    assert _reasons(events) == [RejectReason.NOT_ORDER_OWNER]
    assert len(engine._pegs) == 1


def test_replacing_a_peg_that_has_no_price_is_refused_as_a_peg_problem():
    """Reported honestly rather than as an unknown or a terminal order. Both of
    those would be untrue and would send whoever is debugging it looking for an
    order that is sitting right there."""
    engine = MatchingEngine()
    _peg(engine, "p", Side.BUY, 5, PegReference.BID)
    order_id = engine._pegs[0].order.order_id

    events = engine.apply(Replace(AgentId("p"), order_id, Quantity(3), Price(99)))
    assert _reasons(events) == [RejectReason.INVALID_PEG]


def test_naming_a_price_stops_an_order_being_a_peg():
    """A replace names a price, which is the one thing a peg does not have. The
    order stays and keeps its id; what it loses is the tracking."""
    engine = MatchingEngine()
    _limit(engine, "mm", Side.BUY, 10, 100)
    _peg(engine, "p", Side.BUY, 5, PegReference.BID)
    engine.apply(Replace(AgentId("p"), _order_id(engine, "p"), Quantity(5), Price(95)))
    assert engine._pegs == []

    _limit(engine, "better", Side.BUY, 8, 103)
    assert engine.book.snapshot().bids == (
        (Price(103), Quantity(8)),
        (Price(100), Quantity(10)),
        (Price(95), Quantity(5)),
    ), "the replaced order followed the touch, so it was still pegged"


def test_a_peg_needs_something_to_track_and_nothing_else_may_track():
    engine = MatchingEngine()
    missing = engine.apply(
        Submit(AgentId("a"), Side.BUY, Quantity(5), None, OrderType.PEGGED, TimeInForce.GTC)
    )
    assert _reasons(missing) == [RejectReason.INVALID_PEG]

    spurious = engine.apply(
        Submit(
            AgentId("a"),
            Side.BUY,
            Quantity(5),
            Price(100),
            OrderType.LIMIT,
            TimeInForce.GTC,
            peg_to=PegReference.BID,
        )
    )
    assert _reasons(spurious) == [RejectReason.INVALID_PEG]

    offset_alone = engine.apply(
        Submit(
            AgentId("a"),
            Side.BUY,
            Quantity(5),
            Price(100),
            OrderType.LIMIT,
            TimeInForce.GTC,
            peg_offset=2,
        )
    )
    assert _reasons(offset_alone) == [RejectReason.INVALID_PEG]


def test_a_peg_cannot_also_name_a_price():
    """Two prices for one order, and no rule saying which of them wins."""
    engine = MatchingEngine()
    events = engine.apply(
        Submit(
            AgentId("a"),
            Side.BUY,
            Quantity(5),
            Price(100),
            OrderType.PEGGED,
            TimeInForce.GTC,
            peg_to=PegReference.BID,
        )
    )
    assert _reasons(events) == [RejectReason.INVALID_PRICE]


def test_a_peg_cannot_be_immediate():
    """A peg is an instruction to keep tracking; immediate-or-cancel is an
    instruction not to rest. Only one of them can be obeyed."""
    engine = MatchingEngine()
    for tif in (TimeInForce.IOC, TimeInForce.FOK):
        events = engine.apply(
            Submit(
                AgentId("a"),
                Side.BUY,
                Quantity(5),
                None,
                OrderType.PEGGED,
                tif,
                peg_to=PegReference.BID,
            )
        )
        assert _reasons(events) == [RejectReason.INVALID_PEG], tif


def test_a_peg_may_hide_a_reserve_because_it_does_rest():
    """The rule that stops a market order hiding is that it never rests, and a
    peg does. Having no price of its own is a different thing from having no
    price."""
    engine = MatchingEngine()
    _limit(engine, "mm", Side.SELL, 10, 100)
    _peg(engine, "berg", Side.SELL, 60, PegReference.ASK, display_size=10)

    assert engine.book.snapshot().asks == ((Price(100), Quantity(20)),)
    assert engine.book.total_resting_quantity == 70


def test_a_peg_that_takes_sets_off_a_stop_like_any_other_print():
    """A print is a print whoever made it.

    Without this a stop could be set off by a peg that repriced into a trade but
    not by one that arrived already crossing -- a distinction the tape cannot
    see and nobody could have justified.
    """
    engine = MatchingEngine()
    for price, quantity in ((100, 20), (99, 20), (98, 40)):
        _limit(engine, "mm", Side.BUY, quantity, price)
    _stop(engine, "s", Side.SELL, 25, trigger=99)

    # Pegged to the bid, so it crosses on arrival and prints straight through
    # the trigger.
    events = _peg(engine, "p", Side.SELL, 30, PegReference.BID)
    assert engine._stops == [], "the peg's own print did not release the stop"
    assert sum(q for q, _p in _prints(events)) == 30 + 25

# --------------------------------------------------------------------------
# Minimum quantity
# --------------------------------------------------------------------------


def test_a_minimum_quantity_order_does_not_trade_when_less_is_available():
    engine = MatchingEngine()
    _limit(engine, "mm", Side.SELL, 3, 100)
    events = _limit(engine, "a", Side.BUY, 10, 100, min_quantity=5)

    assert _prints(events) == []
    assert engine.book.snapshot().asks == ((Price(100), Quantity(3)),), "the offer traded"


def test_a_minimum_quantity_order_is_not_fill_or_kill():
    """The distinction the field exists for.

    Fill-or-kill asks whether the *whole* order can be done. A minimum asks only
    whether it is worth starting, so an order for ten with a minimum of five
    takes the six that are there and rests the other four -- where fill-or-kill
    on the same book trades nothing.
    """
    engine = MatchingEngine()
    _limit(engine, "mm", Side.SELL, 6, 100)
    events = _limit(engine, "a", Side.BUY, 10, 100, min_quantity=5)

    assert _prints(events) == [(6, 100)]
    assert engine.book.snapshot().bids == ((Price(100), Quantity(4)),)

    _limit(engine, "mm", Side.SELL, 6, 100)
    killed = _limit(engine, "b", Side.BUY, 10, 100, time_in_force=TimeInForce.FOK)
    assert _reasons(killed) == [RejectReason.FOK_NOT_FILLABLE]


def test_a_minimum_counts_what_could_really_trade_rather_than_the_depth():
    """Aggregate depth over-counts, and over-counting is the one error that
    matters here: it would admit the order and then fill it for less than its
    minimum, which is the outcome the field exists to prevent. Eight of the
    eleven lots offered are the buyer's own, and it cannot trade with itself."""
    engine = MatchingEngine()
    _limit(engine, "a", Side.SELL, 8, 100)
    _limit(engine, "mm", Side.SELL, 3, 100)
    assert engine.book.snapshot().asks == ((Price(100), Quantity(11)),)

    events = _limit(engine, "a", Side.BUY, 10, 100, min_quantity=5)
    assert _prints(events) == []


def test_a_resting_minimum_is_passed_over_by_an_aggressor_too_small_for_it():
    """The order behind it gets the fill, at the same price. That is the bargain
    a minimum strikes -- conditional execution instead of an unconditional place
    in the queue -- and it is the same currency an iceberg spends."""
    engine = MatchingEngine()
    _limit(engine, "big", Side.SELL, 20, 100, min_quantity=10)
    _limit(engine, "small", Side.SELL, 8, 100)

    events = _limit(engine, "a", Side.BUY, 3, 100)
    assert _passive_fills(events) == [("small", 3)]
    assert {o.agent_id: int(o.remaining) for o in engine.book.resting_orders} == {
        "big": 20,
        "small": 5,
    }


def test_a_resting_minimum_trades_normally_with_an_aggressor_large_enough():
    engine = MatchingEngine()
    _limit(engine, "big", Side.SELL, 20, 100, min_quantity=10)
    _limit(engine, "small", Side.SELL, 8, 100)

    events = _limit(engine, "a", Side.BUY, 12, 100)
    assert _passive_fills(events) == [("big", 12)], "time priority stopped applying"


def test_an_aggressor_walks_past_a_price_it_cannot_satisfy():
    """It pays more than the best offer, and that is deliberate.

    An order carrying a minimum offers conditional liquidity, and the condition
    is not met, so it is not a quote this order has to respect. Stopping instead
    would let one large minimum make its whole price level untradeable by
    everyone smaller than it, which is a far worse market than a worse fill.
    """
    engine = MatchingEngine()
    _limit(engine, "big", Side.SELL, 20, 100, min_quantity=10)
    _limit(engine, "other", Side.SELL, 5, 101)

    events = _limit(engine, "a", Side.BUY, 3, 101)
    assert _prints(events) == [(3, 101)]
    assert engine.book.snapshot().asks == (
        (Price(100), Quantity(20)),
        (Price(101), Quantity(2)),
    )


def test_a_book_of_minimums_can_show_a_cross_it_will_not_execute():
    """A measured consequence rather than a bug, and worth writing down because
    it looks exactly like one.

    Minimum-quantity liquidity is conditional, so the displayed best bid and
    best offer can sit on top of each other while neither side is allowed to
    trade. It clears the moment either side grows enough to satisfy the other,
    which is the difference between this and the locked book a collared limit
    order used to produce -- that one had no way out at all.
    """
    engine = MatchingEngine()
    _limit(engine, "mm", Side.SELL, 3, 100)
    _limit(engine, "a", Side.BUY, 10, 100, min_quantity=5)

    book = engine.book.snapshot()
    assert book.best_bid == Price(100) and book.best_ask == Price(100)
    assert engine.tape == ()

    events = _limit(engine, "mm", Side.SELL, 5, 100)
    assert _prints(events) == [(5, 100)], "the cross did not clear once size arrived"


def test_fill_or_kill_is_not_admitted_by_liquidity_a_minimum_protects():
    """Conditional liquidity is not liquidity, and the check was counting it.

    Fill-or-kill asked published depth, which does not know that an offer
    refusing anything under twenty is unavailable to a buyer of ten. Twenty-five
    lots were on offer at 100 and a fill-or-kill for ten was admitted -- and
    then **printed five**, which is the one outcome fill-or-kill exists to make
    impossible. What the walk could really take was five.
    """
    engine = MatchingEngine()
    _limit(engine, "big", Side.SELL, 20, 100, min_quantity=20)
    _limit(engine, "small", Side.SELL, 5, 100)
    assert int(engine.book.depth_at(Side.SELL, Price(100))) == 25

    events = _limit(engine, "f", Side.BUY, 10, 100, time_in_force=TimeInForce.FOK)
    assert _prints(events) == []
    assert _reasons(events) == [RejectReason.FOK_NOT_FILLABLE]
    assert int(engine.book.depth_at(Side.SELL, Price(100))) == 25, "it traded anyway"


def test_an_aggressor_s_minimum_counts_slices_rather_than_totals():
    """How much a level yields depends on the order the executions happen in.

    An iceberg's spent slice goes to the *back* of its queue, so an aggressor
    meets whatever was behind it before it can reach the reserve -- and by then
    it may be too small for a minimum that was satisfiable a moment earlier.
    Counting the iceberg's remaining quantity as one lump said six lots were
    reachable here. The walk took three from the slice, one from the lot behind
    it, and found its last two below the iceberg's own minimum of three: an
    order with a minimum of five **executed for four**.
    """
    engine = MatchingEngine()
    _limit(engine, "berg", Side.BUY, 8, 96, display_size=3, min_quantity=3)
    _limit(engine, "behind", Side.BUY, 1, 96)

    events = _limit(engine, "a", Side.SELL, 6, 96, min_quantity=5)
    assert _prints(events) == []
    assert int(engine.book.total_resting_quantity) == 9 + 6, "it traded anyway"


def test_an_aggressor_s_minimum_is_met_when_the_slices_really_reach_it():
    """The same book, one lot larger behind the iceberg, and the trade happens.

    The count has to be exact in both directions: refusing what a walk could
    have done is as wrong as admitting what it could not.
    """
    engine = MatchingEngine()
    _limit(engine, "berg", Side.BUY, 8, 96, display_size=3, min_quantity=3)
    _limit(engine, "behind", Side.BUY, 2, 96)

    events = _limit(engine, "a", Side.SELL, 5, 96, min_quantity=5)
    assert sum(q for q, _p in _prints(events)) == 5


def test_a_minimum_larger_than_the_order_is_refused():
    """It is not a strict order, it is an order that can never execute, and the
    difference matters -- the first one rests quietly forever and looks like bad
    luck."""
    engine = MatchingEngine()
    events = _limit(engine, "a", Side.BUY, 5, 100, min_quantity=6)
    assert _reasons(events) == [RejectReason.INVALID_QUANTITY]


def test_a_minimum_larger_than_the_displayed_slice_is_refused():
    """Shows ten at a time and refuses to trade fewer than fifty: every
    execution it could offer is one it would then decline."""
    engine = MatchingEngine()
    events = _limit(engine, "a", Side.SELL, 100, 100, display_size=10, min_quantity=50)
    assert _reasons(events) == [RejectReason.INVALID_QUANTITY]


def test_a_negative_minimum_is_refused():
    engine = MatchingEngine()
    events = _limit(engine, "a", Side.BUY, 5, 100, min_quantity=-1)
    assert _reasons(events) == [RejectReason.INVALID_QUANTITY]


def test_a_minimum_survives_a_replace():
    """A replace changes price and size. It does not change what the order is
    willing to be executed for, and one that quietly lost its minimum on being
    resized would start filling in exactly the pieces it was written to refuse.
    """
    engine = MatchingEngine()
    _limit(engine, "big", Side.SELL, 20, 100, min_quantity=10)
    engine.apply(
        Replace(AgentId("big"), _order_id(engine, "big"), Quantity(20), Price(99))
    )

    events = _limit(engine, "a", Side.BUY, 3, 99)
    assert _prints(events) == [], "the minimum was dropped by the replace"


def test_a_minimum_rides_through_a_stop_being_triggered():
    """A stop becomes an order, and it has to become the order that was parked.

    Losing the minimum on the way through would matter most exactly where it
    matters: a stop fires into a falling market, which is the thinnest the book
    ever is and the easiest place to be filled in pieces.
    """
    engine = MatchingEngine()
    _limit(engine, "mm", Side.BUY, 20, 100)
    _limit(engine, "mm", Side.BUY, 2, 99)
    engine.apply(
        Submit(
            AgentId("a"),
            Side.SELL,
            Quantity(10),
            None,
            OrderType.STOP,
            TimeInForce.GTC,
            stop_price=Price(99),
            min_quantity=5,
        )
    )

    _market(engine, "seller", Side.SELL, 20)
    # The print at 100 took the whole 20-lot bid, so all the released stop can
    # reach is two lots at 99 -- fewer than its minimum, so it takes none.
    assert engine.book.snapshot().bids == ((Price(99), Quantity(2)),)


def test_an_iceberg_behind_a_skipped_minimum_refreshes_without_evicting_it():
    """Found by writing the two features down together, and worth keeping.

    An iceberg's slice refresh took the front of the queue by position, which
    was the same order it had just filled -- until a minimum-quantity order
    ahead of it could be passed over. Then the refresh deleted *that* order from
    its level instead. It stayed live everywhere else, so the depth still
    counted it and a cancel would still find it, while the matcher could no
    longer reach it, and the iceberg appeared in the queue twice.
    """
    engine = MatchingEngine()
    _limit(engine, "mpl", Side.SELL, 20, 100, min_quantity=10)
    _limit(engine, "berg", Side.SELL, 30, 100, display_size=5)

    # Too small for the minimum, so it takes the iceberg's slice instead and
    # sends the iceberg to the back of a queue the minimum is still in.
    assert _prints(_limit(engine, "small", Side.BUY, 5, 100)) == [(5, 100)]

    level = engine.book._levels[Side.SELL][Price(100)]
    assert [o.agent_id for o in level.orders] == ["mpl", "berg"]
    assert int(level.total) == 25, "displayed depth: twenty plus one five-lot slice"

    events = _limit(engine, "big", Side.BUY, 20, 100)
    assert _passive_fills(events) == [("mpl", 20)], "the minimum became unreachable"


def test_a_slice_too_small_for_its_own_minimum_refreshes_rather_than_sticking():
    """Spent is not the same as empty.

    An iceberg showing three with a minimum of five sat on the book forever:
    never refreshing, because its slice was not empty, and never trading,
    because every execution it could offer was one it would then refuse. The
    remainder of the slice goes back into the reserve and a whole one is posted.
    """
    engine = MatchingEngine()
    _limit(engine, "berg", Side.SELL, 40, 100, display_size=10, min_quantity=5)
    assert _prints(_limit(engine, "a", Side.BUY, 7, 100)) == [(7, 100)]

    order = next(o for o in engine.book.resting_orders if o.agent_id == "berg")
    assert (int(order.remaining), int(order.shown)) == (33, 10)
    assert engine.book.snapshot().asks == ((Price(100), Quantity(10)),), (
        "the leftover three were counted as well as the fresh slice"
    )
    assert _prints(_limit(engine, "b", Side.BUY, 6, 100)) == [(6, 100)]

# --------------------------------------------------------------------------
# Cancelling a parked stop
# --------------------------------------------------------------------------


def test_a_parked_stop_can_be_cancelled():
    """It is held off the book on purpose, so the book could not answer for it
    and the cancel came back as an unknown order.

    That was not merely unhelpful. The venue drops its working-order entry on a
    rejection, so a cancel the agent believed had failed released the collateral
    reserved against the stop while leaving the stop parked and still able to
    trigger -- one stop still armed, nothing reserved against it. The same gap
    meant the kill switch could report a participant as flat while its stops
    were live.
    """
    engine = MatchingEngine()
    _limit(engine, "mm", Side.BUY, 20, 100)
    acks = _stop(engine, "a", Side.SELL, 30, trigger=99)
    order_id = next(e.order_id for e in acks if isinstance(e, Acknowledged))
    assert len(engine._stops) == 1

    events = engine.apply(Cancel(AgentId("a"), order_id))
    assert not _reasons(events)
    cancelled = [e for e in events if isinstance(e, Cancelled)]
    assert [(e.order_id, int(e.remaining)) for e in cancelled] == [(order_id, 30)], (
        "a stop is not an order yet, so none of it can have traded"
    )
    assert engine._stops == []


def test_a_cancelled_stop_does_not_fire_when_the_market_reaches_it():
    """The point of the cancel, and the half that the reservation accounting
    cannot check for you."""
    engine = MatchingEngine()
    for price, quantity in ((100, 20), (99, 20), (98, 20)):
        _limit(engine, "mm", Side.BUY, quantity, price)
    acks = _stop(engine, "a", Side.SELL, 30, trigger=99)
    order_id = next(e.order_id for e in acks if isinstance(e, Acknowledged))
    engine.apply(Cancel(AgentId("a"), order_id))

    events = _market(engine, "seller", Side.SELL, 25)
    assert _prints(events) == [(20, 100), (5, 99)], "the cancelled stop sold anyway"
    assert engine._stops == []


def test_a_stop_belonging_to_somebody_else_cannot_be_cancelled():
    """Reported as not-owner rather than unknown, matching the book path. Ids
    are engine-assigned and sequential, so an agent could enumerate them anyway,
    and the honest error is far easier to debug than a misleading one."""
    engine = MatchingEngine()
    acks = _stop(engine, "a", Side.SELL, 30, trigger=99)
    order_id = next(e.order_id for e in acks if isinstance(e, Acknowledged))

    events = engine.apply(Cancel(AgentId("thief"), order_id))
    assert _reasons(events) == [RejectReason.NOT_ORDER_OWNER]
    assert len(engine._stops) == 1, "somebody else's cancel removed it"
