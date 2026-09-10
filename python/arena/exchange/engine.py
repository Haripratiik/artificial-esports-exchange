"""The matching engine.

Deterministic by construction: no clock, no randomness, no I/O. Feed it the same
commands in the same order and it emits byte-identical events, every time, on
any machine. That property is not a nicety: it is what makes a differential
test statable at all, since a second implementation has to produce an identical
event stream from an identical command stream, and it is what makes a seeded
experiment reproducible months later.

No such second implementation is owed. A C++ kernel was once listed as
outstanding work; profiled over two simulated minutes of the live market, the
time is in ``kernel.send``, ``latency.delay`` and the book snapshot the venue
broadcasts, and matching does not appear in the fourteen most expensive
functions by self time. So a port is a performance decision to take when a
profile asks for one, and ``tests/test_differential.py`` is worth keeping green
either way.

Matching rules, in the order they are applied:

  1. **Price priority.** The best price trades first.
  2. **Time priority.** Within a price, the order that arrived first trades
     first, by arrival sequence rather than by timestamp.
  3. **Trades print at the resting order's price.** The passive side set the
     terms; the aggressor accepted them. This is what makes price improvement
     accrue to the taker and what makes effective spread measurable.
  4. **Partial fills are normal.** An order walks as many levels as it needs.

Two order types bend the second rule, and both bend it by consent rather than
by exception. An **iceberg** goes to the back of its level each time it
refreshes. A **minimum-quantity** order is passed over by any aggressor too
small to satisfy it, and the fill goes to whoever is behind it. Neither is
losing something it was promised: each is spending queue position on something
it wanted more.

A **pegged** order does not bend it at all, and that is worth saying because it
looks like it should. Its price is a reference plus an offset rather than a
number it chose, and every time the reference moves it is taken off the book and
put back at the new price with a new arrival number, exactly as a replace
would be, and for exactly the same reason. A new price is a new claim on a queue
other orders were already waiting in.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from arena.exchange.book import Order, OrderBook, PriceLevel
from arena.exchange.session import SENTINEL, SessionState, indicative_auction
from arena.exchange.events import (
    Acknowledged,
    Cancel,
    Cancelled,
    Command,
    Event,
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
    SequenceNumber,
    Side,
    TimeInForce,
)

__all__ = ["MatchingEngine"]


@dataclass(slots=True)
class _PendingStop:
    """A stop order waiting for its trigger.

    Not an :class:`Order`, because it is not one yet: it has no place in a
    queue, no price priority, and nothing anyone can trade against. It becomes
    an order when the market reaches it.
    """

    order_id: OrderId
    agent_id: AgentId
    side: Side
    quantity: Quantity
    stop_price: Price
    limit_price: Price | None
    time_in_force: TimeInForce
    display_size: int
    min_quantity: int
    arrival: int

    def as_submit(self) -> Submit:
        """The order this becomes once it is triggered.

        A plain stop becomes a *market* order, and a market order is
        immediate-or-cancel by construction here: an unpriced order that rested
        would match anything forever. Carrying the stop's own time-in-force
        through would hand the engine a GTC market order, which it refuses, and
        the stop would vanish on being triggered: parked, released, rejected,
        gone, with nothing in the tape to say so.
        """
        if self.limit_price is None:
            return Submit(
                self.agent_id,
                self.side,
                self.quantity,
                None,
                OrderType.MARKET,
                TimeInForce.IOC,
                min_quantity=self.min_quantity,
            )
        return Submit(
            self.agent_id,
            self.side,
            self.quantity,
            self.limit_price,
            OrderType.LIMIT,
            self.time_in_force,
            self.display_size,
            min_quantity=self.min_quantity,
        )


@dataclass(slots=True, eq=False)
class _DryOrder:
    """One resting order as a dry run of the walk sees it.

    Compared by identity rather than by value, because the dry run takes entries
    out of its queue by name and two copies holding the same numbers are not the
    same order.

    A copy of the three numbers a walk changes, so that asking "how much would
    this trade" can consume slices, refresh icebergs and drop filled orders
    exactly as the real walk does without any of it reaching the book. Anything
    else about the order is read through ``order``, which is not modified.
    """

    order: Order
    shown: int
    remaining: int

    @property
    def available(self) -> int:
        """What one execution against this order could take.

        An iceberg offers its slice and no more; everything else offers all of
        itself. The same rule the matching loop applies, for the same reason.
        """
        return self.shown if self.order.is_iceberg else self.remaining


@dataclass(slots=True)
class _Peg:
    """A pegged order and whatever order it currently is.

    Two pieces of state that cannot be collapsed into one. ``order`` is the only
    record of quantity, remaining and status, so there is no second copy to fall
    out of step. ``on_book`` is separate because a peg is not always on the book:
    with nothing to track it has no price, and an order with no price cannot sit
    in a price-ordered book. ``order.price`` means nothing while ``on_book`` is
    false.

    The order is replaced outright every time the peg moves, rather than being
    edited in place. It has to be: ``OrderBook.remove`` tombstones rather than
    splicing, so an order whose price were mutated would sit live in the old
    level's queue at a price that level is not, and the matcher would happily
    trade it there.
    """

    reference: PegReference
    offset: int
    time_in_force: TimeInForce
    order: Order
    on_book: bool = False


class MatchingEngine:
    """A single-instrument exchange."""

    def __init__(
        self,
        instrument: str = "DEFAULT",
        self_trade_prevention: SelfTradePrevention = SelfTradePrevention.CANCEL_OLDEST,
    ) -> None:
        self.instrument = instrument
        self.self_trade_prevention = self_trade_prevention
        self.book = OrderBook()
        # Continuous by default, so an engine used on its own behaves exactly
        # as it always has and every existing test keeps its meaning.
        self.phase = SessionState.CONTINUOUS
        # Prices a trade may print at, or ``None`` for no limit.
        #
        # The rule this models does not only pause a runaway after the fact: it
        # *prevents trades outside the bands*, and that is the half that
        # protects anyone. Without it a market order with no price protection
        # walks a thin book to the floor: measured here, a resting bid at
        # **0.25** was filled on a contract worth 4,700, and the breaker then
        # dutifully halted a symbol whose damage was already done.
        #
        # Set by the venue before each command, because the band moves with the
        # reference price and only the venue tracks that.
        self.execution_band: tuple[int, int] | None = None
        # Stop orders waiting for their trigger. Off the book on purpose.
        self._stops: list[_PendingStop] = []
        # Live pegged orders, whether or not they currently have a price.
        self._pegs: list[_Peg] = []
        # How many times a single command may set the pegs moving. A peg that
        # reprices changes the touch, which can move another peg, which can move
        # the first one back: two orders pegged to each other's side of the
        # book have no fixed point and would otherwise chase each other forever.
        # The bound is what makes that a bad idea rather than a hang.
        self._max_peg_passes = 8
        # How many rounds each cascade of stops ran for, oldest first. A
        # measurement rather than a control: a stop that fills moves the price,
        # which triggers more stops, and how often that chains is exactly the
        # thing worth knowing.
        self.cascade_depth: list[int] = []
        # A cascade that never ends is a bug in the model, not an event in the
        # market. High enough that a real chain is never cut short.
        self._max_cascade = 24
        # True while a cascade is being worked through, so a triggered stop's
        # own trades do not start a nested release inside the loop that is
        # already handling them. Without it the chain recurses instead of
        # iterating: each level records a depth of one, the measurement says
        # nothing, and the bound above guards a loop the cascade is not using.
        self._releasing = False
        self._sequence = 0
        self._next_order_id = 0
        self._arrival = 0
        self._tape: list[Traded] = []

    # -- identity ----------------------------------------------------------

    def _seq(self) -> SequenceNumber:
        self._sequence += 1
        return SequenceNumber(self._sequence)

    def _order_id(self) -> OrderId:
        self._next_order_id += 1
        return OrderId(self._next_order_id)

    @property
    def tape(self) -> tuple[Traded, ...]:
        """Every trade printed, in order. The public record."""
        return tuple(self._tape)

    # -- dispatch ----------------------------------------------------------

    def apply(self, command: Command) -> list[Event]:
        """Process one command, returning the events it caused."""
        if isinstance(command, Submit):
            events = self._submit(command)
        elif isinstance(command, Cancel):
            events = self._cancel(command)
        elif isinstance(command, Replace):
            events = self._replace(command)
        else:
            raise TypeError(f"unknown command type {type(command).__name__}")

        # Any command can move the touch, and a peg tracks the touch, so the
        # pegs are settled here rather than in each handler. Guarded on there
        # being any: an engine nobody has pegged an order on must run exactly
        # the code it always ran, down to the sequence numbers it hands out,
        # which is what the differential harness checks.
        #
        # Here rather than inside `_submit`, so a stop cascade repricing pegs
        # between its own half-finished orders is not a thing that can happen.
        if self._pegs:
            events.extend(self._reprice_pegs())
        return events

    def apply_all(self, commands: Iterable[Command]) -> list[Event]:
        events: list[Event] = []
        for command in commands:
            events.extend(self.apply(command))
        return events

    # -- submit ------------------------------------------------------------

    def _submit(self, command: Submit, order_id: OrderId | None = None) -> list[Event]:
        """Place an order. ``order_id`` reuses an id already acknowledged.

        Only a released stop passes one, and it has to. A stop is acknowledged
        under an id while it is parked (that id is the agent's handle on it,
        and it is what the venue reserves collateral against), and the order
        it becomes was being minted a *fresh* id, with nothing in the stream
        linking the two. Measured: a stop acknowledged as order **4** traded as
        order **6**, and a cancel of 4 came back ``unknown_order`` with the
        order neither parked nor in the book. The reservation against 4 had
        nothing left to release it, and 6 traded under an id nobody had reserved
        for. An order that changes identity mid-life is an order nobody can
        follow.
        """
        reason = _validate(command)
        if reason is not None:
            return [Rejected(self._seq(), command.agent_id, reason)]

        if order_id is None:
            order_id = self._order_id()
        self._arrival += 1
        # A market order is priced at the extreme so it crosses everything; the
        # limit-price comparison then needs no special case for it.
        limit = command.price if command.price is not None else _unbounded(command.side)
        order = Order(
            order_id=order_id,
            agent_id=command.agent_id,
            side=command.side,
            price=limit,
            quantity=command.quantity,
            remaining=command.quantity,
            priority=self._arrival,
            display_size=command.display_size,
            min_quantity=command.min_quantity,
            # Recorded on the order, not left behind in the command. A replace
            # builds a new order out of the resting one, and a promise that
            # lives only in a processed command is a promise the replace cannot
            # keep.
            post_only=command.time_in_force is TimeInForce.POST_ONLY,
        )

        # A stop acknowledges at the price it is contingent on: its limit if it
        # has one, otherwise its trigger. The venue reserves collateral from
        # this, and a stop that acknowledged no price at all would be reserved
        # against by nothing: an agent could park a hundred of them, each
        # individually affordable and collectively not.
        acknowledged_at = command.price
        if command.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            acknowledged_at = command.price or command.stop_price
        peg: _Peg | None = None
        if command.order_type is OrderType.PEGGED:
            # A peg acknowledges at the price it will actually rest at, which is
            # the only price it has. `None` when there is nothing to track: the
            # order is real and accepted, and it genuinely has no price yet.
            peg = _Peg(
                reference=command.peg_to,
                offset=command.peg_offset,
                time_in_force=command.time_in_force,
                order=order,
            )
            acknowledged_at = self._peg_target(peg)

        events: list[Event] = [
            Acknowledged(
                self._seq(),
                command.agent_id,
                order_id,
                command.side,
                command.quantity,
                acknowledged_at,
            )
        ]

        if command.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            return events + self._park_stop(command, order_id)

        if peg is not None:
            # Before the call-phase branch, and deliberately. `_accumulate`
            # rests an order at `limit`, which for a peg is the sentinel a
            # priceless order gets, so an auction would have counted it as
            # crossing every candidate at 2^62 and cleared the book against it.
            self._pegs.append(peg)
            if acknowledged_at is None:
                return events
            placed = self._place_peg(peg, acknowledged_at)
            events.extend(placed)
            # A peg tracking the far side of the book takes, and a print is a
            # print whoever made it. Without this a stop could be set off by a
            # peg that repriced into a trade but not by one that arrived already
            # crossing, which is a distinction the tape cannot see and nobody
            # could have justified.
            events.extend(self._release_after(placed))
            return events

        if not self.phase.matches_continuously:
            return events + self._accumulate(command, order)

        # Post-only is decided before anything trades, for the same reason a
        # maker uses it: the whole point is that no part of it may ever take.
        if command.time_in_force is TimeInForce.POST_ONLY and self._crossable_levels(
            command.side, limit
        ):
            order.status = OrderStatus.REJECTED
            self.book.track(order)
            events.append(
                Rejected(
                    self._seq(),
                    command.agent_id,
                    RejectReason.POST_ONLY_WOULD_CROSS,
                    order_id,
                )
            )
            return events

        # Fill-or-kill is decided before anything trades, so a partial walk is
        # never left half-done and then unwound. Checking first is simpler and
        # leaves no intermediate state an observer could see.
        if command.time_in_force is TimeInForce.FOK and not self._fillable(
            order,
            self.execution_band if command.order_type is OrderType.MARKET else None,
        ):
            order.status = OrderStatus.CANCELLED
            self.book.track(order)
            events.append(
                Rejected(
                    self._seq(), command.agent_id, RejectReason.FOK_NOT_FILLABLE, order_id
                )
            )
            return events

        events.extend(
            self._match(
                order,
                self.execution_band
                if command.order_type is OrderType.MARKET
                else None,
            )
        )
        events.extend(self._settle(order, command.time_in_force))
        # After the order has been settled, never before it. Releasing first
        # ran a triggered stop against a book that was still missing the
        # aggressor's own remainder, so the stop rested where the remainder was
        # about to rest through. Measured: a stop-limit sell for nineteen at 99
        # triggered by a print at 97, released while the taker's seven unfilled
        # lots were still in flight, left the book bid **101** against ask
        # **99**: a spread of minus two, crossed and stuck. The peg path
        # already rested before releasing; this one was the odd case out.
        events.extend(self._release_after(events))

        return events

    def _settle(self, order: Order, time_in_force: TimeInForce) -> list[Event]:
        """Rest, cancel or retire an order once its matching pass is over.

        The terminal check comes first, and it is not defensive. Self-trade
        prevention can finish an order mid-walk (CANCEL_NEWEST and CANCEL_BOTH
        both do), and the arithmetic test that follows would then read "nothing
        left" as "completely filled". Measured: an agent whose own resting
        offer met its own incoming bid got a ``Cancelled`` event, an empty
        tape, and an order whose status said **filled** and whose ``filled``
        property said **10**. Every reconciliation downstream reads those two,
        and both of them were describing a trade that never printed.
        """
        if order.status.terminal:
            self.book.track(order)
            return []
        if order.remaining <= 0:
            order.status = OrderStatus.FILLED
            self.book.track(order)
            return []
        if time_in_force in (TimeInForce.GTC, TimeInForce.POST_ONLY):
            order.status = (
                OrderStatus.PARTIALLY_FILLED
                if order.remaining < order.quantity
                else OrderStatus.NEW
            )
            self.book.add(order)
            return []
        order.status = OrderStatus.CANCELLED
        self.book.track(order)
        return [
            Cancelled(self._seq(), order.agent_id, order.order_id, order.remaining)
        ]

    def _accumulate(self, command: Submit, order: Order) -> list[Event]:
        """Rest an order during a call phase, where nothing matches yet.

        A *limit* order marked immediate-or-cancel or fill-or-kill is refused
        rather than silently rested: both are instructions about what to do
        *right now*, and during a call phase there is no right now. Accepting
        them would quietly convert a "do not leave this working" order into one
        that works until the uncross.

        A **market** order is the exception, and not an inconsistency. It is
        required to be IOC in continuous trading only because an unpriced
        resting order would match anything forever: during a call phase nothing
        matches until the uncross, so the danger does not exist. What it becomes
        is a market-on-open order: willing to trade at whatever price the
        auction clears at, which is exactly what such an order means.
        """
        if command.order_type is not OrderType.MARKET and command.time_in_force in (
            TimeInForce.IOC,
            TimeInForce.FOK,
        ):
            order.status = OrderStatus.CANCELLED
            self.book.track(order)
            return [
                Rejected(
                    self._seq(),
                    command.agent_id,
                    RejectReason.NOT_ACCEPTED_IN_AUCTION,
                    order.order_id,
                )
            ]
        order.status = OrderStatus.NEW
        self.book.add(order)
        return []

    def uncross(self, reference: Price | None = None) -> list[Event]:
        """Clear the accumulated book at a single price.

        Everything trades at the auction price, including orders that were
        willing to pay more: the price improvement is the reward for having
        been in the auction, and it is why the clearing price is trustworthy in
        a way a first-arrival price is not.

        Nobody is the aggressor here, so every fill is booked as passive. Under
        a maker-taker schedule that means auction fills earn the maker rate on
        both sides, which is what venues that run auctions actually charge.
        """
        result = indicative_auction(self.book, reference)
        if result is None or result.volume <= 0:
            # Even a call that clears nothing has to take the market orders
            # back out. An auction with no crossing interest is the *most*
            # likely place to leave one behind, and one left behind is a
            # sentinel-priced order sitting at the touch of a continuous book.
            return self._cancel_unfilled_market_orders()

        limit = int(result.price)
        buys = sorted(
            (o for o in self.book.resting_orders
             if o.side is Side.BUY and int(o.price) >= limit),
            key=lambda o: (-int(o.price), o.priority),
        )
        sells = sorted(
            (o for o in self.book.resting_orders
             if o.side is Side.SELL and int(o.price) <= limit),
            key=lambda o: (int(o.price), o.priority),
        )

        events: list[Event] = []
        i = j = 0
        while i < len(buys) and j < len(sells):
            buy, sell = buys[i], sells[j]
            if buy.remaining <= 0:
                i += 1
                continue
            if sell.remaining <= 0:
                j += 1
                continue
            if buy.agent_id == sell.agent_id:
                # A wash print is worse in an auction than in continuous
                # trading: it would be struck at the official price and could
                # move a settlement. Drop the older side and carry on, which is
                # the CANCEL_OLDEST policy applied to a two-sided book.
                stale = buy if buy.priority <= sell.priority else sell
                events.append(
                    Cancelled(self._seq(), stale.agent_id, stale.order_id, stale.remaining)
                )
                self.book.remove(stale)
                i, j = (i + 1, j) if stale is buy else (i, j + 1)
                continue

            quantity = Quantity(min(int(buy.remaining), int(sell.remaining)))
            price = Price(limit)
            for order in (buy, sell):
                self.book.consume(order, quantity)
                events.append(
                    Filled(
                        self._seq(),
                        order.agent_id,
                        order.order_id,
                        order.side,
                        quantity,
                        price,
                        False,
                        order.remaining,
                    )
                )
            trade = Traded(
                self._seq(),
                quantity,
                price,
                # An auction has no aggressor. The surplus side is the closest
                # honest analogue, so order-flow statistics stay meaningful.
                result.surplus_side or Side.BUY,
                buy.order_id,
                sell.order_id,
            )
            self._tape.append(trade)
            events.append(trade)

        events.extend(self._cancel_unfilled_market_orders())
        return events

    def _cancel_unfilled_market_orders(self) -> list[Event]:
        """Take market-on-open orders that did not trade back out of the book.

        They rest at the sentinel price so that they cross every candidate in
        the auction, which is the whole point of them, and it is why leaving
        one behind is catastrophic rather than untidy. An unfilled market sell
        sits at minus 2^61, which is the best offer in the book by a margin of
        2^61, so the first continuous buy order matches it *at that price*. Run
        for the first time, this printed trades at -4,611,686,018,427,387,904,
        marked the book at zero and billed 4.8e22 in fees before anything else
        went wrong.

        A market order is an instruction about the auction it was entered for.
        Once that auction has cleared there is no price it was willing to pay,
        because it never named one, so cancelling is the only honest outcome,
        and it is what venues do with unexecuted market-on-open interest.
        """
        events: list[Event] = []
        for order in list(self.book.resting_orders):
            if abs(int(order.price)) < SENTINEL:
                continue
            events.append(
                Cancelled(self._seq(), order.agent_id, order.order_id, order.remaining)
            )
            self.book.remove(order)
            # The status, and not only the level bookkeeping. `Book.remove`
            # reduces the level's total and leaves the order in the queue as a
            # tombstone; what makes the matcher skip it on the way past is its
            # status being terminal. Removing without marking left an order the
            # depth no longer counted but the matcher would still fill, so it
            # was invisible to every diagnostic that reads resting orders while
            # remaining perfectly tradeable, which is why the sentinel prints
            # survived two attempts at fixing them.
            order.status = OrderStatus.CANCELLED
        return events

    def _fillable(self, order: Order, collar: tuple[int, int] | None) -> bool:
        """Whether all of ``order`` could be filled immediately.

        Asks ``_executable`` (what the walk would really take) rather than
        summing published depth, and the difference is the whole contract of
        fill-or-kill. Aggregate depth counts liquidity the walk cannot have:
        an order whose own minimum quantity the taker is too small to satisfy,
        and an order belonging to the taker itself. Both were counted, both
        admitted the order, and the walk then filled it for less than all of it.

        Measured before the change. Against a twenty-lot offer refusing
        anything under twenty with five plain lots behind it, a fill-or-kill
        buy for **ten** printed **five**. Against the taker's own five lots at
        100 and somebody else's five at 101, a fill-or-kill buy for **ten**
        printed **five** at 101 while self-trade prevention quietly cancelled
        the rest of what the check had counted. Partial execution is the one
        outcome fill-or-kill exists to make impossible.

        It differs the other way too, and that is the same correction rather
        than a second one. Published depth *under*-counts an iceberg, which
        shows a slice and holds a reserve. A fill-or-kill for a hundred against
        a hundred-lot iceberg showing ten was rejected as unfillable, while the
        identical order sent good-till-cancelled filled all hundred: the walk
        reaches the reserve because each exhausted slice refreshes behind it.

        Quantity the taker itself owns is not counted either, and that is the
        third case of the same error rather than a separate rule. Self-trade
        prevention cancels those lots instead of printing them, so they are
        liquidity the walk can never have. Measured before the change: against
        the taker's own five lots at 100 and somebody else's five at 101, a
        fill-or-kill buy for **ten** printed **five** at 101 while prevention
        quietly cancelled the rest of what the check had counted.

        Correcting this required moving `tests/reference_matcher.py` in the
        same step. The reference matcher counted that quantity too, so fixing
        one side alone made twelve differential tests disagree, and a harness
        that disagrees is a worse failure than the one it is reporting, while a
        harness that agrees on the wrong answer reports nothing at all. Both
        sides now skip the taker's own resting orders in the same place their
        walks already skip them.
        """
        return (
            self._executable(order, collar, count_self_matched=False)
            >= int(order.remaining)
        )

    def _crossable_levels(
        self, side: Side, limit: Price
    ) -> list[tuple[Price, int]]:
        """Opposite-side levels this order could reach, best first."""
        opposite = side.opposite
        snapshot = self.book.snapshot(levels=1 << 20)
        levels = snapshot.asks if opposite is Side.SELL else snapshot.bids
        return [(p, int(q)) for p, q in levels if side.crosses(p, limit)]

    def _prevent_self_trade(
        self, incoming: Order, resting: Order, level
    ) -> list[Event]:
        """Resolve a would-be wash trade according to the configured policy.

        Removing the resting order also pops it off the level, so the matching
        loop advances rather than meeting the same order forever, which would
        be an infinite loop rather than a wrong price. Popped only when it is
        the front of the queue, which it is unless a minimum-quantity order
        ahead of it was passed over; anywhere else the terminal status is what
        makes the loop skip it, and popping would remove somebody else's order.
        """
        events: list[Event] = []
        policy = self.self_trade_prevention

        if policy in (SelfTradePrevention.CANCEL_OLDEST, SelfTradePrevention.CANCEL_BOTH):
            remaining = resting.remaining
            self.book.remove(resting)
            resting.status = OrderStatus.CANCELLED
            if level.orders and level.orders[0] is resting:
                level.popleft()
            events.append(
                Cancelled(self._seq(), resting.agent_id, resting.order_id, remaining)
            )

        if policy in (SelfTradePrevention.CANCEL_NEWEST, SelfTradePrevention.CANCEL_BOTH):
            events.append(
                Cancelled(
                    self._seq(), incoming.agent_id, incoming.order_id, incoming.remaining
                )
            )
            # The status, and not the quantity. Zeroing ``remaining`` to stop
            # the walk was reading as "completely filled" everywhere afterwards:
            # ``filled`` is ``quantity - remaining``, so an order that traded
            # nothing reported its whole size as traded, against an empty tape.
            # The matching loop already breaks on a terminal status, so the
            # quantity never had to carry that signal.
            incoming.status = OrderStatus.CANCELLED

        return events

    def _tradeable(self, price: Price) -> bool:
        """Whether a trade may print at this price."""
        if self.execution_band is None:
            return True
        low, high = self.execution_band
        return low <= int(price) <= high


    # -- stops -------------------------------------------------------------

    def _park_stop(self, command: Submit, order_id: OrderId) -> list[Event]:
        """Hold a stop off the book until its trigger is reached.

        Off the book, not on it: a resting stop is not liquidity and must not
        appear as any. Publishing one would tell everybody exactly where the
        market has to go to set off a cascade, which is the single piece of
        information a stop order's owner most wants kept quiet.
        """
        self._stops.append(
            _PendingStop(
                order_id=order_id,
                agent_id=command.agent_id,
                side=command.side,
                quantity=command.quantity,
                stop_price=command.stop_price,
                limit_price=command.price,
                time_in_force=command.time_in_force,
                display_size=command.display_size,
                min_quantity=command.min_quantity,
                arrival=self._arrival,
            )
        )
        return []

    def _release_after(self, events: list[Event]) -> list[Event]:
        """Fire whatever stops the prints in ``events`` triggered.

        A print can bring stop orders to life, and one of those can print again.
        Released after the order that caused them has finished matching rather
        than inside the loop, so a cascade is a sequence of complete orders
        rather than an interleaving of half-filled ones.

        **Every** print, not the last one. A walk that sweeps several levels
        prints at each of them, and offering only the final price to the stops
        drops any trigger the walk passed through on the way. Measured: offers
        of five at 100 and five at 110, a sell stop parked at 100, and a buy for
        ten: the tape read ``[(5, 100), (5, 110)]``, the market had traded at
        100, and the stop was still parked afterwards. The rest of the cascade
        already worked this way, checking each print of every released order, so
        the first round was the one that disagreed.
        """
        if not self._stops or self._releasing:
            return []
        prices = [e.price for e in events if isinstance(e, Traded)]
        if not prices:
            return []
        return self._release_stops(prices)

    def _triggered_by(self, price: Price) -> list["_PendingStop"]:
        """Stops this print sets off, in a deterministic order.

        A buy stop triggers when the market trades at or above its price, a
        sell stop at or below. Ordered by how far through the trigger the print
        went and then by arrival, so a single print that sets off several stops
        releases them in the order the market reached them rather than in the
        order they happened to be entered.
        """
        hit = [
            stop
            for stop in self._stops
            if (stop.side is Side.BUY and int(price) >= int(stop.stop_price))
            or (stop.side is Side.SELL and int(price) <= int(stop.stop_price))
        ]
        if not hit:
            return []
        self._stops = [stop for stop in self._stops if stop not in hit]
        hit.sort(
            key=lambda s: (
                -int(s.stop_price) if s.side is Side.BUY else int(s.stop_price),
                s.arrival,
            )
        )
        return hit

    def _release_stops(self, prices: list[Price]) -> list[Event]:
        """Fire everything these prints triggered, and everything that triggers.

        Iterative rather than recursive, and bounded. A stop that fills moves
        the price, which can trigger more stops: that is a cascade, it is
        real, and this does not prevent it. What it does prevent is a cascade
        that never terminates, which would be a bug in the model rather than an
        event in the market. `cascade_depth` records how far each one went.
        """
        events: list[Event] = []
        pending: list[_PendingStop] = []
        for price in prices:
            pending.extend(self._triggered_by(price))
        if not pending:
            return events
        depth = 0
        self._releasing = True
        try:
            depth = self._work_cascade(pending, events)
        finally:
            self._releasing = False
        if depth:
            self.cascade_depth.append(depth)
        return events

    def _work_cascade(self, pending: list["_PendingStop"], events: list[Event]) -> int:
        depth = 0
        while pending and depth < self._max_cascade:
            depth += 1
            following: list[_PendingStop] = []
            for stop in pending:
                # Under the id it was acknowledged with, so the order an agent
                # parked and the order it becomes are the same order.
                released = self._submit(stop.as_submit(), stop.order_id)
                events.extend(released)
                for event in released:
                    if isinstance(event, Traded):
                        following.extend(self._triggered_by(event.price))
            pending = following
        # The bound stops the *chain*, and it must not delete the orders the
        # chain had already reached. `_triggered_by` takes a stop out of the
        # parked list to hand it over, so anything still pending when the bound
        # bites has left the engine entirely: no order, no acknowledgement, no
        # cancellation, and a later cancel answered `unknown_order`. Measured on
        # a forty-deep ladder with the bound at 24: twenty-four stops released,
        # fifteen still parked, and **one** that simply ceased to exist with
        # nothing in the stream to say so. Parked again, it stays a live order
        # and the next print that reaches it sets it off.
        if pending:
            self._stops.extend(pending)
        return depth

    # -- pegs ---------------------------------------------------------------

    def _peg_target(self, peg: _Peg) -> Price | None:
        """Where this peg may rest right now, or ``None`` if nowhere.

        Three reasons a peg has no price. There is no reference: an empty
        book, or a mid peg with only one side quoted. The session is not trading
        continuously, where the only touch on offer is the sentinel a
        market-on-open order rests at, which is not a price. Or the peg is
        post-only and the reference has moved somewhere it would cross.

        None of those is an error, and none of them cancels the order. A peg
        with no price waits, which is the honest reading of "quote at the touch"
        when there is no touch.
        """
        if not self.phase.matches_continuously:
            return None

        order = peg.order
        # The peg's own resting quantity is excluded from what it reads. A peg
        # that counted itself would be pegged to itself the moment it became the
        # touch: it could follow the market up, because a better price is
        # somebody else's, and could never follow it back down, because every
        # step down is blocked by the price it is already quoting.
        ignore = order.order_id
        if peg.reference is PegReference.BID:
            base = self.book.best_priced(Side.BUY, ignore)
        elif peg.reference is PegReference.ASK:
            base = self.book.best_priced(Side.SELL, ignore)
        else:
            bid = self.book.best_priced(Side.BUY, ignore)
            ask = self.book.best_priced(Side.SELL, ignore)
            if bid is None or ask is None:
                return None
            # Prices are integer ticks and a one-tick spread has no midpoint, so
            # the half tick has to go somewhere. It goes to the passive side: a
            # buy rounds down, a sell rounds up. Rounding the other way would
            # make a mid peg cross the spread on every odd-width market, which
            # is the one thing a midpoint order is not for.
            total = int(bid) + int(ask)
            base = Price(total // 2 if order.side is Side.BUY else -((-total) // 2))
        if base is None:
            return None

        target = Price(int(base) + peg.offset)
        if peg.time_in_force is TimeInForce.POST_ONLY and self._crossable_levels(
            order.side, target
        ):
            # Post-only promises the order never takes. A peg does not choose
            # its own price, so the promise is kept by declining to follow the
            # reference rather than by rejecting the order: there is nothing
            # left to reject, the order was accepted before the touch moved.
            return None
        return target

    def _place_peg(self, peg: _Peg, target: Price) -> list[Event]:
        """Rest a peg at ``target``, matching first if it crosses.

        The order must be off the book when this is called, because it sets the
        price directly. A peg tracking the opposite side of the book is an
        aggressive order and is meant to trade: "pay the offer, whatever the
        offer is" is a real instruction, and it is the one a market peg writes.
        """
        order = peg.order
        order.price = target
        events = self._match(order)
        # Through the same settlement as any other order, so a peg that self-
        # trade prevention cancelled mid-walk is not then recorded as filled.
        # A peg is validated as neither immediate-or-cancel nor fill-or-kill, so
        # its time-in-force always rests.
        events.extend(self._settle(order, peg.time_in_force))
        peg.on_book = order.is_resting
        return events

    def _lift_peg(self, peg: _Peg) -> Order:
        """Take a peg off the book and give it a fresh order to become.

        The old order is tombstoned rather than spliced out, which is how every
        removal works here, so it must be marked terminal or the matcher will
        still fill it on the way past. The replacement carries the same id (an
        agent's handle on its order does not change because the touch moved)
        and a new arrival number, which is the whole cost of repricing.

        The book is touched only if the order was in it. ``OrderBook.remove``
        finds its level by the order's price, and a peg that is off the book
        still carries the last price it held, so removing one twice would
        subtract its quantity from whatever level some other order has since
        opened there.
        """
        old = peg.order
        if peg.on_book:
            self.book.remove(old)
            old.status = OrderStatus.CANCELLED
        peg.on_book = False

        # Repricing loses queue priority, and that is the honest behaviour
        # rather than an implementation limitation. A new price is a new claim
        # on a queue that other orders were already waiting in, and letting a
        # peg keep its position through a reprice would hand it a standing
        # advantage no other order can buy: it could sit at the front of one
        # level, follow the touch to another, and still be at the front there.
        self._arrival += 1
        fresh = Order(
            order_id=old.order_id,
            agent_id=old.agent_id,
            side=old.side,
            price=old.price,
            quantity=old.quantity,
            remaining=old.remaining,
            priority=self._arrival,
            display_size=old.display_size,
            min_quantity=old.min_quantity,
            # Every rebuild of an order has to carry everything the order was,
            # and this is the third place that lesson has been learnt. A
            # post-only peg that had repriced even once came back without the
            # flag, so a replace no longer knew to refuse a crossing price:
            # measured, a post-only peg replaced to 102 took a lot at 98.
            post_only=old.post_only,
        )
        peg.order = fresh
        return fresh

    def _reprice_pegs(self) -> list[Event]:
        """Move every peg to where its reference now says it belongs.

        Iterative and bounded for the same reason a stop cascade is: a peg that
        moves changes the book, which can move another peg. Unlike a cascade
        this one usually settles in a single pass, because a peg that is already
        at its target is left alone and most commands move at most one touch.
        """
        events: list[Event] = []
        for _ in range(self._max_peg_passes):
            moved: list[Event] = []
            if not self._move_pegs(moved):
                break
            events.extend(moved)
            # A peg can print, and a print can set off a stop. Released between
            # passes rather than inside one, so the pegs a stop's fills move are
            # moved by the pass after it rather than mid-flight.
            events.extend(self._release_after(moved))
        return events

    def _move_pegs(self, events: list[Event]) -> bool:
        """One pass over the pegs. True if any of them moved."""
        moved = False
        for peg in list(self._pegs):
            order = peg.order
            if order.status.terminal or order.remaining <= 0:
                # Filled, cancelled, or replaced into a plain limit order by an
                # agent that named a price. Whatever it is now, it is not a peg.
                self._pegs.remove(peg)
                continue

            target = self._peg_target(peg)
            if target is None:
                if peg.on_book:
                    # Its reference has gone, so it has no price. Leaving it at
                    # the last price it happened to track would turn it into a
                    # stale limit order at exactly the moment the market it was
                    # following stopped existing.
                    #
                    # No event: the order is not cancelled, it is waiting, and
                    # there is nothing in this vocabulary that says so. The
                    # silence errs toward a venue holding collateral for an
                    # order that is still live, which is the safe direction.
                    self._lift_peg(peg)
                    moved = True
                continue

            if peg.on_book and int(order.price) == int(target):
                continue

            moved = True
            fresh = self._lift_peg(peg)
            events.append(
                Replaced(
                    self._seq(),
                    fresh.agent_id,
                    fresh.order_id,
                    fresh.remaining,
                    target,
                    kept_priority=False,
                )
            )
            events.extend(self._place_peg(peg, target))
        return moved

    def _inert_peg(self, order_id: OrderId) -> _Peg | None:
        """A live peg that currently has no price, and so is not in the book.

        The book cannot answer for these: an order with no price is in no level,
        and ``OrderBook.remove`` would reduce whatever level happened to exist at
        the last price it held. So cancellation asks here first.
        """
        return next(
            (
                peg
                for peg in self._pegs
                if not peg.on_book
                and peg.order.order_id == order_id
                and not peg.order.status.terminal
            ),
            None,
        )

    def _match(
        self, incoming: Order, collar: tuple[int, int] | None = None
    ) -> list[Event]:
        """Walk the opposite side until filled, out of crossable price, or out
        of collar.

        The collar applies to **market orders only**, and that distinction is
        the whole of it. A market order names no price, so it needs protecting
        from the book: without a collar one walked a thin book to the floor and
        filled a resting bid at **0.25** on a contract worth 4,700. A limit
        order names a price and is entitled to it; collaring one too was tried
        and was much worse than the disease. Orders slid to a band edge, the
        band later moved away from them, and the book locked: bid above offer,
        neither allowed to trade, nothing in continuous trading able to clear
        it. Measured on that version: 2,492 limit states in five minutes and a
        future marking at 9,267 against a settlement of 4,669.
        """
        events: list[Event] = []

        # A minimum quantity is checked once, against everything this pass could
        # reach, and not against each print. That is what makes it different
        # from fill-or-kill: fill-or-kill asks whether the *whole* order can be
        # done, MPL asks only whether it is worth starting. An order for a
        # thousand with a minimum of a hundred will happily take two hundred and
        # rest the remaining eight; what it will not do is take three.
        #
        # A decision about *printing*, and not a reason to skip the walk. The
        # walk is also where an agent meets its own resting orders, and
        # returning here before it started meant self-trade prevention never
        # ran. Measured: a maker bid fourteen at 99 and then offered seven at 98
        # with a minimum of three. Without the minimum its stale bid was
        # cancelled and the offer rested alone; with it, both orders stayed:
        # the same agent on both sides of a book crossed 99 against 98, which
        # only a third party could ever clear.
        may_print = incoming.min_quantity <= 0 or (
            self._executable(incoming, collar) >= incoming.min_quantity
        )

        # Prices this pass has exhausted of liquidity it is allowed to take.
        # Only ever non-empty when a resting order refuses executions this
        # small, so an order using none of this costs one `not blocked` test per
        # level and takes exactly the path it always took.
        blocked: set[int] = set()

        while incoming.remaining > 0:
            level = self._next_level(incoming.side.opposite, blocked)
            if level is None:
                break
            resting_price = level.price
            if not incoming.side.crosses(resting_price, incoming.price):
                break
            if collar is not None:
                low, high = collar
                if not low <= int(resting_price) <= high:
                    # Past the edge of the collar. The order stops here rather
                    # than printing beyond it, and whatever is left is
                    # cancelled: a market order was never willing to rest.
                    break

            level.prune()
            if level.empty:
                # Level emptied by pruning; loop and let best_level move on.
                continue

            resting = self._first_tradeable(level, incoming)
            if resting is None:
                # Everything at this price refuses an execution this small. The
                # order moves on rather than waiting, and the next price it
                # reaches may be a worse one. That is not a trade-through of a
                # protected quote: an order carrying a minimum is offering
                # conditional liquidity, and the condition is not met. The
                # alternative (stopping here) would let one order with a
                # large minimum make its whole price level untradeable by
                # everyone smaller than it, which is a far worse market than a
                # slightly worse fill.
                blocked.add(int(resting_price))
                continue

            if (
                resting.agent_id == incoming.agent_id
                and self.self_trade_prevention is not SelfTradePrevention.ALLOW
            ):
                events.extend(self._prevent_self_trade(incoming, resting, level))
                if incoming.status.terminal:
                    break
                continue

            # The first order this pass could really trade with, and the
            # minimum said it is not worth trading. Here rather than before the
            # walk, so the agent's own resting orders in front of it have
            # already been dealt with by the policy that governs them.
            if not may_print:
                break

            # At most the slice an iceberg is showing. Taking its reserve
            # in one go would make the reserve pointless: the aggressor would
            # get the whole order at one price and nobody else at that level
            # would ever get a turn, which is precisely what a hidden order is
            # not entitled to.
            available = resting.shown if resting.is_iceberg else resting.remaining
            traded = Quantity(min(int(incoming.remaining), int(available)))
            if traded <= 0:
                # Defensive; `consume` refreshes an exhausted iceberg, so an
                # order showing nothing while holding something should not
                # exist. Pruning alone would only clear it from the front of the
                # queue, and this order need not be at the front, so the price
                # is set aside as well rather than risking a loop that cannot
                # make progress.
                level.prune()
                blocked.add(int(resting_price))
                continue

            self.book.consume(resting, traded)
            incoming.remaining = Quantity(incoming.remaining - traded)

            buy_id, sell_id = (
                (incoming.order_id, resting.order_id)
                if incoming.side is Side.BUY
                else (resting.order_id, incoming.order_id)
            )

            # Trades print at the RESTING price: the passive side set the terms.
            events.append(
                Filled(
                    self._seq(),
                    resting.agent_id,
                    resting.order_id,
                    resting.side,
                    traded,
                    resting_price,
                    aggressor=False,
                    remaining=resting.remaining,
                )
            )
            events.append(
                Filled(
                    self._seq(),
                    incoming.agent_id,
                    incoming.order_id,
                    incoming.side,
                    traded,
                    resting_price,
                    aggressor=True,
                    remaining=incoming.remaining,
                )
            )
            trade = Traded(
                self._seq(),
                traded,
                resting_price,
                aggressor_side=incoming.side,
                buy_order_id=buy_id,
                sell_order_id=sell_id,
            )
            events.append(trade)
            self._tape.append(trade)

            if resting.remaining <= 0 and level.orders and level.orders[0] is resting:
                level.popleft()

        return events

    def _next_level(self, side: Side, blocked: set[int]) -> PriceLevel | None:
        """The best level on ``side`` that this pass has not already set aside."""
        if not blocked:
            return self.book.best_level(side)
        for level in self.book.live_levels(side):
            if int(level.price) not in blocked:
                return level
        return None

    def _first_tradeable(self, level: PriceLevel, incoming: Order) -> Order | None:
        """The first order at this price the incoming order may trade with.

        Almost always the front of the queue, which is what time priority means.
        It is not the front only when the order there insists on a minimum this
        order is too small to meet, and then the fill goes to whoever is behind
        it at the same price. Nobody loses anything they were promised: the
        order with the minimum chose conditional execution over an unconditional
        place in the queue, which is the same bargain an iceberg strikes when it
        trades queue position for concealment.
        """
        for resting in level.orders:
            if resting.status.terminal or resting.remaining <= 0:
                continue
            if resting.min_quantity <= 0:
                return resting
            available = int(resting.shown) if resting.is_iceberg else int(resting.remaining)
            if min(int(incoming.remaining), available) >= resting.min_quantity:
                return resting
        return None

    def _executable(
        self,
        incoming: Order,
        collar: tuple[int, int] | None,
        *,
        count_self_matched: bool = False,
    ) -> int:
        """How much of ``incoming`` could really trade against the book now.

        Counts what the walk would actually take rather than aggregate depth,
        because the two differ: a resting order with a minimum of its own may be
        passed over, and an order that would only meet itself contributes
        nothing. Used for the minimum-quantity test, where an over-count is the
        one error that matters: it would admit an order and then fill it for
        less than its minimum, which is the outcome the field exists to prevent.

        A **dry run of the walk**, one execution at a time, rather than a sum
        over resting orders. Summing was tried twice and was wrong both times,
        because how much a level yields depends on the order the executions
        happen in. An iceberg's exhausted slice goes to the *back* of its
        queue, so the aggressor meets whatever was behind it before it can
        reach the reserve, and by then it may be too small for a minimum that
        was satisfiable a moment earlier. Measured: an iceberg for eight at 96
        showing three and refusing anything under three, with a single plain lot
        behind it, against a sell for six carrying a minimum of five. The sum
        said six were reachable, the order was admitted, and the walk took
        three from the iceberg, one from the lot behind it, and then found its
        remaining two below the iceberg's minimum: **an order with a minimum
        of five executed for four**.

        So the queue is copied and consumed exactly as ``_match`` consumes it:
        the same choice of counterparty, the same slice sizes, the same refresh
        to the back. Copied, because a count must not move anything.

        ``count_self_matched`` treats the taker's own resting orders as if they
        belonged to somebody else. It exists for one caller and is wrong on its
        own terms: see ``_fillable``, which explains why it is there and what
        has to change before it can go.
        """
        wanted = int(incoming.remaining)
        policy = self.self_trade_prevention
        prevents = policy is not SelfTradePrevention.ALLOW and not count_self_matched
        total = 0
        for level in self.book.live_levels(incoming.side.opposite):
            if not incoming.side.crosses(level.price, incoming.price):
                break
            if collar is not None:
                low, high = collar
                if not low <= int(level.price) <= high:
                    break

            queue = [
                _DryOrder(resting, int(resting.shown), int(resting.remaining))
                for resting in level.orders
                if not resting.status.terminal and resting.remaining > 0
            ]
            while total < wanted:
                entry = _first_dry_tradeable(queue, wanted - total)
                if entry is None:
                    # Nothing here will trade with an order this size. The walk
                    # sets the price aside and moves on to a worse one.
                    break
                resting = entry.order
                if prevents and resting.agent_id == incoming.agent_id:
                    if policy is SelfTradePrevention.CANCEL_OLDEST:
                        # The resting order is removed and the walk carries on.
                        queue.remove(entry)
                        continue
                    # The incoming order's remainder is cancelled where it
                    # stands, so nothing past this point is reachable.
                    return total

                take = min(wanted - total, entry.available)
                total += take
                entry.remaining -= take
                entry.shown -= take
                if entry.remaining <= 0:
                    queue.remove(entry)
                elif resting.is_iceberg and entry.shown < max(1, resting.min_quantity):
                    # Spent, by the same rule ``OrderBook.consume`` uses: empty,
                    # or too small for the minimum this order itself insists on.
                    # A fresh slice, at the back of the queue.
                    entry.shown = min(resting.display_size, entry.remaining)
                    queue.remove(entry)
                    queue.append(entry)
            if total >= wanted:
                return total
        return total

    # -- cancel ------------------------------------------------------------

    def _cancel(self, command: Cancel) -> list[Event]:
        parked = next(
            (stop for stop in self._stops if stop.order_id == command.order_id), None
        )
        if parked is not None:
            # A parked stop is off the book on purpose, so the book cannot
            # answer for it and `UNKNOWN_ORDER` was the answer it gave. That was
            # not merely unhelpful. The venue drops its working-order entry on a
            # rejection, so a cancel an agent believed had failed released the
            # collateral reserved against the stop while leaving the stop parked
            # and able to trigger: measured as one stop still parked with the
            # reservation gone. It also meant the kill switch could report a
            # participant as flat while its stops were still armed.
            if parked.agent_id != command.agent_id:
                return [
                    Rejected(
                        self._seq(),
                        command.agent_id,
                        RejectReason.NOT_ORDER_OWNER,
                        command.order_id,
                    )
                ]
            self._stops.remove(parked)
            # Its full quantity: a stop is not an order yet, so none of it can
            # have traded.
            return [
                Cancelled(
                    self._seq(), command.agent_id, command.order_id, parked.quantity
                )
            ]

        inert = self._inert_peg(command.order_id) if self._pegs else None
        if inert is not None:
            # Asked before the book, because a peg with no price is in no level
            # and the id may still resolve to the tombstone of the last order it
            # was. Answering from the book would report a live order as already
            # terminal, and leave an order nobody can withdraw.
            if inert.order.agent_id != command.agent_id:
                return [
                    Rejected(
                        self._seq(),
                        command.agent_id,
                        RejectReason.NOT_ORDER_OWNER,
                        command.order_id,
                    )
                ]
            remaining = inert.order.remaining
            inert.order.status = OrderStatus.CANCELLED
            self._pegs.remove(inert)
            return [
                Cancelled(self._seq(), command.agent_id, command.order_id, remaining)
            ]

        order = self.book.get(command.order_id)
        if order is None:
            return [
                Rejected(
                    self._seq(),
                    command.agent_id,
                    RejectReason.UNKNOWN_ORDER,
                    command.order_id,
                )
            ]
        if order.agent_id != command.agent_id:
            # Reported as not-owner rather than unknown. Leaking "this id exists"
            # is harmless here (ids are engine-assigned and sequential, so an
            # agent could enumerate them anyway), and the honest error is far
            # easier to debug than a misleading one.
            return [
                Rejected(
                    self._seq(),
                    command.agent_id,
                    RejectReason.NOT_ORDER_OWNER,
                    command.order_id,
                )
            ]
        if order.status.terminal:
            return [
                Rejected(
                    self._seq(),
                    command.agent_id,
                    RejectReason.ALREADY_TERMINAL,
                    command.order_id,
                )
            ]

        remaining = order.remaining
        self.book.remove(order)
        order.status = OrderStatus.CANCELLED
        return [Cancelled(self._seq(), command.agent_id, command.order_id, remaining)]

    # -- replace -----------------------------------------------------------

    def _replace(self, command: Replace) -> list[Event]:
        if self._pegs and self._inert_peg(command.order_id) is not None:
            # A peg that currently has no price cannot be repriced. Reported as
            # a peg problem rather than as an unknown or terminal order, both of
            # which would be untrue and would send whoever is debugging it
            # looking for an order that is sitting right there.
            return [
                Rejected(
                    self._seq(),
                    command.agent_id,
                    RejectReason.INVALID_PEG,
                    command.order_id,
                )
            ]

        order = self.book.get(command.order_id)
        if order is None:
            return [
                Rejected(
                    self._seq(),
                    command.agent_id,
                    RejectReason.UNKNOWN_ORDER,
                    command.order_id,
                )
            ]
        if order.agent_id != command.agent_id:
            return [
                Rejected(
                    self._seq(),
                    command.agent_id,
                    RejectReason.NOT_ORDER_OWNER,
                    command.order_id,
                )
            ]
        if order.status.terminal:
            return [
                Rejected(
                    self._seq(),
                    command.agent_id,
                    RejectReason.ALREADY_TERMINAL,
                    command.order_id,
                )
            ]
        if command.new_quantity <= 0:
            return [
                Rejected(
                    self._seq(),
                    command.agent_id,
                    RejectReason.INVALID_QUANTITY,
                    command.order_id,
                )
            ]

        new_price = command.new_price if command.new_price is not None else order.price
        # Priority survives only a strict reduction at an unchanged price. Any
        # price change or size increase is a new claim on the queue.
        keeps_priority = new_price == order.price and command.new_quantity < order.remaining

        if (
            order.post_only
            and not keeps_priority
            and self._crossable_levels(order.side, new_price)
        ):
            # Post-only promised the order would never take, and a replace does
            # not retract the promise: it is the same order at a new price.
            # Before the flag was carried on the order, a post-only sell resting
            # at 105 over a bid of 100 and then replaced to 100 printed **ten
            # lots as the aggressor**, which is the single thing that order type
            # exists to make impossible. Refused rather than repriced, so the
            # order stays exactly where it was resting.
            #
            # Only on the branch that re-runs the match. A shrink at an
            # unchanged price never touches the other side of the book, so there
            # is nothing there for it to promise about.
            return [
                Rejected(
                    self._seq(),
                    command.agent_id,
                    RejectReason.POST_ONLY_WOULD_CROSS,
                    command.order_id,
                )
            ]

        if keeps_priority:
            # `OrderBook.shrink`, not `consume`. Routing a shrink through the
            # fill path took the lots out of the level's total even when they
            # came out of an iceberg's reserve, which was never counted in it,
            # and refreshed the exhausted slice to the back of the queue: the
            # exact priority loss this branch had just promised did not happen.
            # Measured on an iceberg for twelve showing three with four lots
            # behind it: shrinking to six left the level at 4 against 7 really
            # resting, and put the iceberg behind the order that arrived after
            # it while the event said `kept_priority=True`.
            self.book.shrink(order, Quantity(order.remaining - command.new_quantity))
            # A shrink is not a fill, so the order is only partially filled if
            # something really traded. `shrink` takes the same amount off
            # `quantity` as off `remaining`, so `filled` answers that honestly.
            order.status = (
                OrderStatus.PARTIALLY_FILLED
                if order.filled > 0
                else OrderStatus.NEW
            )
            return [
                Replaced(
                    self._seq(),
                    command.agent_id,
                    command.order_id,
                    command.new_quantity,
                    new_price,
                    kept_priority=True,
                )
            ]

        # Otherwise: pull the old order and resubmit at the back of the queue,
        # re-running the match in case the new price now crosses.
        self.book.remove(order)
        order.status = OrderStatus.CANCELLED

        self._arrival += 1
        replacement = Order(
            order_id=command.order_id,
            agent_id=command.agent_id,
            side=order.side,
            price=new_price,
            quantity=command.new_quantity,
            remaining=command.new_quantity,
            priority=self._arrival,
            min_quantity=order.min_quantity,
            # Carried, or a replace quietly strips an iceberg of the only
            # property that made it one: the order comes back fully displayed
            # and publishes the size its owner was working in slices precisely
            # so that nobody could see it.
            display_size=order.display_size,
            # For the same reason, and it is the one the display size argument
            # missed. Post-only is a property of the order, not of the command
            # that created it.
            post_only=order.post_only,
        )
        events: list[Event] = [
            Replaced(
                self._seq(),
                command.agent_id,
                command.order_id,
                command.new_quantity,
                new_price,
                kept_priority=False,
            )
        ]
        events.extend(self._match(replacement))
        events.extend(
            self._settle(
                replacement,
                TimeInForce.POST_ONLY if replacement.post_only else TimeInForce.GTC,
            )
        )
        # A replace that crosses is a print like any other, and a stop cannot
        # tell how the print was caused. Without this, a resting offer moved
        # down onto a bid traded ten lots at 100 while a stop parked at 100 sat
        # untouched: the identical print delivered as a new order set the same
        # stop off immediately. Whether a stop fires must not depend on which
        # message the tape came from.
        events.extend(self._release_after(events))
        return events


def _validate(command: Submit) -> RejectReason | None:
    if command.quantity <= 0:
        return RejectReason.INVALID_QUANTITY
    if command.display_size < 0:
        return RejectReason.INVALID_QUANTITY
    if command.min_quantity < 0:
        return RejectReason.INVALID_QUANTITY
    if command.min_quantity > command.quantity:
        # An order refusing to trade for less than more than all of itself. It
        # is not a strict order, it is an order that can never execute, and the
        # difference matters because the first one rests quietly forever and
        # looks like bad luck.
        return RejectReason.INVALID_QUANTITY
    if command.display_size and command.min_quantity > command.display_size:
        # Shows ten at a time and refuses to trade fewer than fifty. Every
        # execution it could offer is one it would then decline, so the two
        # instructions cancel out to "never trade".
        return RejectReason.INVALID_QUANTITY

    pegging = command.order_type is OrderType.PEGGED
    if pegging and command.peg_to is None:
        return RejectReason.INVALID_PEG
    if not pegging and (command.peg_to is not None or command.peg_offset):
        return RejectReason.INVALID_PEG
    if pegging and command.price is not None:
        # Its price is the reference plus its offset. A named price as well
        # would be two prices for one order and no rule saying which wins.
        return RejectReason.INVALID_PRICE
    if pegging and command.time_in_force in (TimeInForce.IOC, TimeInForce.FOK):
        # A peg is an instruction to keep tracking, and both of these are
        # instructions not to rest. Only one of them can be obeyed.
        return RejectReason.INVALID_PEG

    stopping = command.order_type in (OrderType.STOP, OrderType.STOP_LIMIT)
    if stopping and command.stop_price is None:
        return RejectReason.INVALID_STOP_PRICE
    if not stopping and command.stop_price is not None:
        return RejectReason.INVALID_STOP_PRICE
    if command.order_type is OrderType.STOP and command.price is not None:
        return RejectReason.INVALID_PRICE
    if command.order_type is OrderType.STOP_LIMIT and command.price is None:
        return RejectReason.LIMIT_ORDER_REQUIRES_PRICE
    if stopping and command.time_in_force in (TimeInForce.IOC, TimeInForce.FOK):
        # "Do this now" and "do this later" are contradictory instructions.
        return RejectReason.MARKET_ORDER_MUST_BE_IOC
    if stopping:
        # Everything below is about an order that exists now. A stop does not:
        # its price rules were checked above, against what it will become.
        return None
    if command.display_size and command.order_type not in (
        OrderType.LIMIT,
        OrderType.STOP_LIMIT,
        OrderType.PEGGED,
    ):
        # An order with no price cannot hide anything: it never rests, so
        # there is no queue for a reserve to wait in. A peg is on the list
        # because it does rest: it has no price of its own, which is a
        # different thing from having no price.
        return RejectReason.INVALID_QUANTITY
    if pegging:
        # Everything below concerns an order that names a price. A peg does not,
        # and its own rules were checked above.
        return None
    if command.order_type is OrderType.MARKET:
        if command.price is not None:
            return RejectReason.INVALID_PRICE
        if command.time_in_force in (TimeInForce.GTC, TimeInForce.POST_ONLY):
            # An unpriced resting order would match anything forever, and a
            # post-only market order is a contradiction in terms, since a market
            # order is defined by being willing to cross.
            return RejectReason.MARKET_ORDER_MUST_BE_IOC
        return None
    if command.price is None:
        return RejectReason.LIMIT_ORDER_REQUIRES_PRICE
    return None


def _first_dry_tradeable(queue: list["_DryOrder"], wanted: int) -> "_DryOrder | None":
    """``_first_tradeable``, asked of a copied queue instead of a live one.

    Deliberately the same rule, written twice rather than shared, because the
    two answer about different things: one picks a real order to trade with, the
    other picks a copy to subtract from. Any difference between them is a case
    where the count and the walk disagree, which is the whole failure this
    machinery exists to prevent, so they are kept side by side, a few lines
    apart, where a change to one that is not made to the other is visible.
    """
    for entry in queue:
        available = entry.available
        if available <= 0:
            continue
        if min(wanted, available) >= entry.order.min_quantity:
            return entry
    return None


def _unbounded(side: Side) -> Price:
    """The price a market order behaves as if it had."""
    return Price(1 << 62) if side is Side.BUY else Price(-(1 << 62))
