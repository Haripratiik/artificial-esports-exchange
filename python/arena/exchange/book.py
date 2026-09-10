"""The limit order book: price-time priority, one instrument.

The data structure is chosen for a reference implementation rather than for
throughput. Correctness has to be obvious by reading, because this engine's job
is to be the oracle a second implementation is validated against. If the
reference is subtly wrong, whatever is checked against it gets validated into
being identically wrong.

    price -> PriceLevel(FIFO deque of orders, running total)
    plus a heap of prices per side, for O(log n) best-price lookup

Levels are removed lazily: emptying a level leaves its price in the heap, and
the price is discarded when it surfaces and turns out to be stale. Eager removal
would need a heap supporting arbitrary deletion, which is more machinery for no
behavioural difference.

**Priority is (price, arrival), and arrival is a monotonically increasing
sequence, never a timestamp.** Two orders arriving in the same microsecond must
still have a defined order, and a wall clock cannot promise that. The sequence
can, and it is also what makes a seeded replay reproduce exactly.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field

from arena.exchange.session import SENTINEL as _SENTINEL
from arena.exchange.types import (
    AgentId,
    OrderId,
    OrderStatus,
    Price,
    Quantity,
    Side,
)

__all__ = ["Order", "PriceLevel", "OrderBook", "BookSnapshot"]


@dataclass(slots=True)
class Order:
    """A resting or in-flight order. Mutable: quantity falls as it fills."""

    order_id: OrderId
    agent_id: AgentId
    side: Side
    price: Price
    quantity: Quantity
    remaining: Quantity
    # Arrival rank, assigned by the engine. Defines time priority within a level.
    priority: int
    status: OrderStatus = OrderStatus.NEW
    # How much of the order is shown at a time. Zero means all of it.
    #
    # An iceberg exists because size is information: an order for ten thousand
    # lots tells everyone what you are doing before you have done any of it, so
    # it is worked in slices. The cost is queue priority (each refreshed slice
    # goes to the back of its level, behind everything that arrived while the
    # last one was working), and that trade, visibility against position, is
    # the whole design.
    display_size: int = 0
    # The slice currently on the book. The rest of ``remaining`` is reserve, and
    # nobody outside this order can see it.
    shown: Quantity = Quantity(0)
    # Refuse any single execution smaller than this. Zero means no minimum.
    #
    # It buys protection from being picked off in dribs and pays for it in
    # certainty, and the currency is the same one an iceberg spends: an order
    # an aggressor cannot satisfy is passed over, and the order behind it in the
    # queue gets the fill instead. Attaching a minimum is therefore giving up
    # unconditional time priority, not adding a guarantee on top of it.
    min_quantity: int = 0
    # Whether this order promised never to take.
    #
    # Carried on the order rather than left in the time-in-force of the command
    # that created it, for the same reason ``display_size`` is: a replace builds
    # a new order out of the old one, and anything the old one did not carry is
    # silently dropped. Measured before this field existed, a post-only sell
    # resting at 105 over a bid of 100, replaced to 100, printed ten lots as
    # the aggressor. The order had promised that could never happen, and the
    # promise lived only in a command that had already been processed.
    post_only: bool = False

    def __post_init__(self) -> None:
        if self.display_size < 0:
            raise ValueError("display size cannot be negative")
        if self.min_quantity < 0:
            raise ValueError("minimum quantity cannot be negative")
        if not self.shown:
            self.shown = self.visible_slice()

    def visible_slice(self) -> Quantity:
        """What the next slice on the book should be."""
        if self.display_size <= 0:
            return self.remaining
        return Quantity(min(int(self.remaining), self.display_size))

    @property
    def hidden(self) -> Quantity:
        return Quantity(max(0, int(self.remaining) - int(self.shown)))

    @property
    def is_iceberg(self) -> bool:
        return self.display_size > 0

    @property
    def filled(self) -> Quantity:
        return Quantity(self.quantity - self.remaining)

    @property
    def is_resting(self) -> bool:
        return not self.status.terminal and self.remaining > 0


@dataclass(slots=True)
class PriceLevel:
    """All orders at one price, in arrival order.

    ``total`` is maintained incrementally rather than summed on demand. Depth is
    queried far more often than it changes. Every market-data update reads it,
    and recomputing it would make snapshot cost proportional to queue length.
    """

    price: Price
    orders: deque[Order] = field(default_factory=deque)
    total: Quantity = Quantity(0)

    def append(self, order: Order) -> None:
        self.orders.append(order)
        # Only the visible slice counts toward depth, because only the visible
        # slice is what anyone can see. Adding the reserve here would publish
        # the very thing an iceberg exists not to publish.
        self.total = Quantity(self.total + order.shown)

    def reduce(self, amount: Quantity) -> None:
        self.total = Quantity(self.total - amount)

    def popleft(self) -> Order:
        return self.orders.popleft()

    def peek(self) -> Order:
        return self.orders[0]

    @property
    def empty(self) -> bool:
        return not self.orders

    def refresh(self, order: Order) -> None:
        """Put an exhausted iceberg's next slice at the back of the queue.

        Behind everything that arrived while the last slice was working, which
        is the price an iceberg pays and the reason it is not simply a better
        order. A venue that refreshed in place would let one participant hold
        the front of a queue indefinitely while showing a single lot.
        """
        order.shown = order.visible_slice()
        self.total = Quantity(self.total + order.shown)
        self.orders.append(order)

    def prune(self) -> None:
        """Drop terminal or exhausted orders from the front of the queue.

        Cancelled orders are not removed from their level when cancelled: that
        would be O(n) in the queue. They are tombstoned and skipped here, on
        the way past, which keeps cancellation O(1). Cancel-heavy flow is the
        norm in electronic markets, so this is the operation worth optimising.
        """
        while self.orders and (
            self.orders[0].status.terminal or self.orders[0].remaining <= 0
        ):
            self.orders.popleft()


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    """An L2 view: aggregated depth per price, best prices, and the spread."""

    bids: tuple[tuple[Price, Quantity], ...]
    asks: tuple[tuple[Price, Quantity], ...]

    @property
    def best_bid(self) -> Price | None:
        """The best *priced* bid, which is not always the first level.

        Market orders rest at a sentinel price during a call phase so they
        cross every candidate the auction considers. That makes them the top of
        the book by a margin of 2^61, and it makes them not prices: an order
        that names no price cannot be the best price. Reporting one as the
        touch marked books at zero, produced spreads of 2^62, and fed the
        market maker a mid that no instrument could quote around.

        The levels themselves keep them, because the auction has to count that
        interest to know what would trade.
        """
        return _first_priced(self.bids)

    @property
    def best_ask(self) -> Price | None:
        return _first_priced(self.asks)

    @property
    def priced_bids(self) -> tuple[tuple[Price, Quantity], ...]:
        """Bids that name a price, for anything a person will look at.

        The raw levels keep market-on-open interest at its sentinel price
        because the auction has to count it. A ladder on a screen must not
        show it: the API published a bid of 4,611,686,018,427,387,904 and the
        page dutifully rendered it.
        """
        return tuple((p, q) for p, q in self.bids if abs(int(p)) < _SENTINEL)

    @property
    def priced_asks(self) -> tuple[tuple[Price, Quantity], ...]:
        return tuple((p, q) for p, q in self.asks if abs(int(p)) < _SENTINEL)

    @property
    def spread(self) -> int | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return int(self.best_ask) - int(self.best_bid)

    @property
    def mid(self) -> float | None:
        """Midpoint in ticks. Fractional, because a one-tick spread has none."""
        if self.best_bid is None or self.best_ask is None:
            return None
        return (int(self.best_bid) + int(self.best_ask)) / 2.0


def _first_priced(levels: tuple[tuple[Price, Quantity], ...]) -> Price | None:
    """The first level that names a price rather than "any"."""
    for price, _quantity in levels:
        if abs(int(price)) < _SENTINEL:
            return price
    return None


class OrderBook:
    """Price-time priority book for a single instrument."""

    def __init__(self) -> None:
        self._levels: dict[Side, dict[Price, PriceLevel]] = {
            Side.BUY: {},
            Side.SELL: {},
        }
        # Bids are stored negated so both sides use a min-heap and "best" is
        # always heap[0]. Avoids two nearly-identical code paths.
        self._prices: dict[Side, list[int]] = {Side.BUY: [], Side.SELL: []}
        self._orders: dict[OrderId, Order] = {}

    # -- queries -----------------------------------------------------------

    def get(self, order_id: OrderId) -> Order | None:
        return self._orders.get(order_id)

    def best_price(self, side: Side) -> Price | None:
        """Best resting price on ``side``, discarding stale heap entries."""
        heap = self._prices[side]
        levels = self._levels[side]
        while heap:
            price = Price(-heap[0] if side is Side.BUY else heap[0])
            level = levels.get(price)
            if level is not None:
                level.prune()
                if not level.empty:
                    return price
                del levels[price]
            heapq.heappop(heap)
        return None

    def best_level(self, side: Side) -> PriceLevel | None:
        price = self.best_price(side)
        return None if price is None else self._levels[side][price]

    def live_levels(self, side: Side) -> list[PriceLevel]:
        """Every level with something live on it, best first.

        The whole ladder rather than the top of it, for the two questions that
        cannot be answered from the touch alone: how much an incoming order
        could really trade, and which level it moves to when the one in front of
        it holds nothing it is allowed to take. Built from the level map rather
        than the price heap, because the heap carries stale prices and draining
        them here would mutate the book on a read.
        """
        levels = self._levels[side]
        out: list[PriceLevel] = []
        for price in sorted(levels, reverse=side is Side.BUY):
            level = levels[price]
            level.prune()
            if not level.empty:
                out.append(level)
        return out

    def best_priced(self, side: Side, ignore: OrderId | None = None) -> Price | None:
        """The best price on one side, with two things left out of it.

        Market-on-open interest, which rests at a sentinel so it crosses every
        candidate in an auction and is therefore the top of the book by a margin
        of 2^61 while naming no price at all. And, optionally, one order of the
        caller's choosing, which is what a pegged order needs, because a peg
        that counts its own quantity in the reference it tracks is pegged to
        itself and can never step back down.

        Reads the queue rather than the level total, since the total cannot say
        which orders make it up and ``ignore`` is a question about one order.
        """
        levels = self._levels[side]
        for price in sorted(levels, reverse=side is Side.BUY):
            if abs(int(price)) >= _SENTINEL:
                continue
            for order in levels[price].orders:
                if order.status.terminal or order.remaining <= 0:
                    continue
                if ignore is not None and order.order_id == ignore:
                    continue
                return price
        return None

    def depth_at(self, side: Side, price: Price) -> Quantity:
        """Live resting quantity at a price.

        Returns the level's maintained total rather than summing the deque, and
        the distinction is a correctness one rather than a performance one.
        Cancellation tombstones an order instead of splicing it out, and
        ``prune`` only clears tombstones from the *front* of the queue, so a
        cancelled order sitting mid-queue is invisible to matching but was
        still being counted by a naive sum.

        That over-reported depth was not merely cosmetic. ``_fillable`` reads it
        to decide whether a fill-or-kill order can be satisfied, so an inflated
        figure let a FOK order be accepted and then partially fill, which is
        precisely what fill-or-kill exists to prevent.

        The total is exact because every path that removes quantity (a fill via
        ``consume``, a cancellation via ``remove``, a shrinking replace)
        reduces it at the same moment.
        """
        level = self._levels[side].get(price)
        if level is None:
            return Quantity(0)
        level.prune()
        return Quantity(max(0, int(level.total)))

    def snapshot(self, levels: int = 5) -> BookSnapshot:
        """Aggregated depth, best first, at most ``levels`` prices per side."""
        return BookSnapshot(
            bids=self._aggregate(Side.BUY, levels),
            asks=self._aggregate(Side.SELL, levels),
        )

    def _aggregate(self, side: Side, limit: int) -> tuple[tuple[Price, Quantity], ...]:
        # Sorted from the live level map rather than the heap, because the heap
        # may hold stale prices and popping them here would mutate on a read.
        prices = sorted(self._levels[side], reverse=side is Side.BUY)
        out: list[tuple[Price, Quantity]] = []
        for price in prices:
            quantity = self.depth_at(side, price)
            # A level can be present but empty until something prunes it, so
            # depth rather than presence decides whether it is shown.
            if quantity > 0:
                out.append((price, quantity))
                if len(out) >= limit:
                    break
        return tuple(out)

    @property
    def resting_orders(self) -> tuple[Order, ...]:
        """Every live order, in deterministic (side, price, priority) order."""
        return tuple(
            sorted(
                (o for o in self._orders.values() if o.is_resting),
                key=lambda o: (o.side.value, int(o.price), o.priority),
            )
        )

    @property
    def total_resting_quantity(self) -> int:
        return sum(o.remaining for o in self._orders.values() if o.is_resting)

    # -- mutation ----------------------------------------------------------

    def add(self, order: Order) -> None:
        """Rest an order at its price, behind everything already there.

        The visible slice is computed *here*, as the order joins a level, and
        not once at construction. An order that partially filled on the way in
        and then rested still carried the slice it was born with, so the depth
        published its original size rather than what was left of it, caught by
        the differential harness as a book one lot deeper than the reference
        matcher's, which is exactly the kind of quiet arithmetic error that
        harness exists for.
        """
        order.shown = order.visible_slice()
        levels = self._levels[order.side]
        level = levels.get(order.price)
        if level is None:
            level = PriceLevel(price=order.price)
            levels[order.price] = level
            heapq.heappush(
                self._prices[order.side],
                -int(order.price) if order.side is Side.BUY else int(order.price),
            )
        level.append(order)
        self._orders[order.order_id] = order

    def track(self, order: Order) -> None:
        """Record an order without resting it, so it can still be looked up.

        Needed for orders that fully fill or expire on arrival: they never join
        a level, but their id must still resolve for acknowledgements and for a
        subsequent cancel to be rejected as terminal rather than unknown.
        """
        self._orders[order.order_id] = order

    def consume(self, order: Order, quantity: Quantity) -> None:
        """Remove ``quantity`` from a resting order and its level's total.

        An iceberg whose visible slice is spent is taken out of the queue and
        put back at the end with a fresh slice, which is what costs it its
        priority. *Spent* is not always *empty*: a slice smaller than the
        order's own minimum quantity is finished too, because there is nobody
        left who is allowed to take it. Without that, an iceberg showing three
        with a minimum of five sat on the book forever: never refreshing,
        because its slice was not empty, and never trading, because every
        execution it could offer was one it would refuse.
        """
        order.remaining = Quantity(order.remaining - quantity)
        order.shown = Quantity(max(0, int(order.shown) - int(quantity)))
        level = self._levels[order.side].get(order.price)
        if level is not None:
            level.reduce(quantity)
        if order.remaining <= 0:
            order.status = OrderStatus.FILLED
            return
        order.status = OrderStatus.PARTIALLY_FILLED
        spent = int(order.shown) <= 0 or (
            0 < int(order.shown) < order.min_quantity
        )
        if level is not None and order.is_iceberg and spent:
            # By name rather than by position. This is the front of the queue
            # unless a minimum-quantity order ahead of it was passed over, and
            # popping blindly in that case deleted *that* order from its level,
            # leaving it live and countable everywhere else, so the depth
            # over-reported it while the matcher could no longer reach it, and
            # the iceberg appeared in the queue twice.
            if level.orders and level.orders[0] is order:
                level.popleft()
            else:
                level.orders.remove(order)
            # Whatever is left of the spent slice goes back into the reserve, so
            # `refresh` adding a whole new slice does not count it twice. A no-op
            # on the ordinary path, where the slice is empty by definition.
            level.reduce(order.shown)
            level.refresh(order)

    def shrink(self, order: Order, amount: Quantity) -> None:
        """Take ``amount`` off a resting order without it having traded.

        Not ``consume``, and the difference is the whole reason this exists. A
        shrinking replace used to be routed through ``consume`` on the grounds
        that both make ``remaining`` smaller, and it was wrong twice over.

        ``consume`` reduces the level's total by the amount it removes, which is
        right for a fill: every lot a fill takes came off the visible slice.
        A shrink takes its lots out of the *reserve*, which was never in the
        total. Measured on an iceberg for twelve showing three, with four lots
        resting behind it: shrinking to one left the level reporting **-3** and
        the published depth **0**, while five lots sat there live.

        ``consume`` also refreshes an exhausted iceberg to the back of its
        queue, which is exactly the priority loss the replace had just promised
        did not happen. The same shrink moved the iceberg behind the order that
        arrived after it while the ``Replaced`` event said ``kept_priority``.

        So: reduce the visible slice only by what actually came off it, leave
        the order where it is in the queue, and take the same amount off
        ``quantity`` as off ``remaining``; otherwise ``filled`` reports lots
        that never traded, which on a plain order for a hundred shrunk to sixty
        read as **40 filled** against an empty tape.
        """
        removed = Quantity(min(int(amount), int(order.remaining)))
        order.remaining = Quantity(order.remaining - removed)
        order.quantity = Quantity(order.quantity - removed)
        slice_now = Quantity(min(int(order.shown), int(order.remaining)))
        level = self._levels[order.side].get(order.price)
        if level is not None:
            level.reduce(Quantity(order.shown - slice_now))
        order.shown = slice_now

    def remove(self, order: Order) -> None:
        """Tombstone an order. The level skips it on the way past.

        Deliberately not O(queue): cancellation is the most common operation in
        an electronic market, and scanning a deque to splice one out would make
        the common case the expensive one.

        Marking the order terminal is *part of* removing it rather than
        something each caller remembers to do afterwards. Every caller but one
        did remember; the one that did not was the auction's wash-trade
        handler, which printed a ``Cancelled`` event, took the order's quantity
        out of its level, and left the order live in the queue. Measured: a
        ten-lot bid reported cancelled, the book publishing no bid at all, and a
        continuous sell arriving afterwards filling all ten lots against it.
        Invisible and tradeable is the worst of both, and the only way to stop
        it recurring is for the two halves not to be separable.

        Idempotent for the same reason: a second removal would subtract the
        order's slice from its level again, and a peg that comes off the book
        twice would take somebody else's quantity with it.
        """
        if order.status.terminal:
            return
        level = self._levels[order.side].get(order.price)
        if level is not None and order.shown > 0:
            # By what it was showing, because that is what was added. Reducing
            # by the whole remaining would take an iceberg's hidden reserve out
            # of a total it was never in.
            level.reduce(order.shown)
        order.status = OrderStatus.CANCELLED
