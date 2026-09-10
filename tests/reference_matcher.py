"""A deliberately naive matching engine, for differential testing.

Written to be obviously correct rather than fast. No heaps, no price-level
buckets, no lazy deletion, no tombstoning, just a flat list of orders that is
sorted from scratch on every operation. It would be hopeless in production and
that is the point: it has nowhere to hide a bug.

The real engine earns its speed with three optimisations, and each is somewhere
a subtle error could live:

    a heap of prices per side, with stale entries discarded lazily
    price-level buckets holding a running total
    cancellation by tombstone rather than by splicing the queue

Differential testing is how those are shown to be safe. Feed both engines the
same command stream and demand the same trades, in the same order, at the same
prices. Any divergence is a bug in one of them, and the naive one is small
enough to check by reading.

This is also the exact technique the C++ port will be validated with, so the
harness exists before the port does rather than being improvised afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from arena.exchange.events import Cancel, Command, Replace, Submit
from arena.exchange.types import (
    AgentId,
    OrderId,
    OrderType,
    Price,
    Quantity,
    Side,
    TimeInForce,
)

__all__ = ["ReferenceMatcher", "NaiveTrade"]


@dataclass(frozen=True, slots=True)
class NaiveTrade:
    """One execution, in the smallest form worth comparing."""

    price: int
    quantity: int
    aggressor_side: str
    buy_order_id: int
    sell_order_id: int


@dataclass
class _Order:
    order_id: int
    agent_id: AgentId
    side: Side
    price: int
    remaining: int
    arrival: int
    dead: bool = False


class ReferenceMatcher:
    """Price-time priority, implemented in the least clever way possible."""

    def __init__(self) -> None:
        self.orders: list[_Order] = []
        self.trades: list[NaiveTrade] = []
        self._next_id = 0
        self._arrival = 0

    # -- helpers -----------------------------------------------------------

    def _live(self, side: Side) -> list[_Order]:
        """Resting orders on one side, best first.

        Sorted from scratch every time. A bid's priority rises with price, an
        offer's falls with it, and ties break on arrival, which is the whole of
        price-time priority, written out.
        """
        live = [o for o in self.orders if o.side is side and not o.dead and o.remaining > 0]
        return sorted(
            live, key=lambda o: (-o.price, o.arrival) if side is Side.BUY else (o.price, o.arrival)
        )

    def _find(self, order_id: int) -> _Order | None:
        for order in self.orders:
            if order.order_id == order_id:
                return order
        return None

    def _crosses(self, side: Side, resting: int, incoming: int) -> bool:
        return incoming >= resting if side is Side.BUY else incoming <= resting

    # -- commands ----------------------------------------------------------

    def apply(self, command: Command) -> None:
        if isinstance(command, Submit):
            self._submit(command)
        elif isinstance(command, Cancel):
            order = self._find(int(command.order_id))
            if order is not None and order.agent_id == command.agent_id:
                order.dead = True
        elif isinstance(command, Replace):
            self._replace(command)

    def apply_all(self, commands) -> None:
        for command in commands:
            self.apply(command)

    def _submit(self, command: Submit) -> None:
        if int(command.quantity) <= 0:
            return
        if command.order_type is OrderType.MARKET:
            if command.price is not None or command.time_in_force is TimeInForce.GTC:
                return
            limit = (1 << 62) if command.side is Side.BUY else -(1 << 62)
        else:
            if command.price is None:
                return
            limit = int(command.price)

        self._next_id += 1
        self._arrival += 1
        order = _Order(
            order_id=self._next_id,
            agent_id=command.agent_id,
            side=command.side,
            price=limit,
            remaining=int(command.quantity),
            arrival=self._arrival,
        )

        if command.time_in_force is TimeInForce.FOK and not self._fillable(order):
            order.dead = True
            self.orders.append(order)
            return

        self._match(order)

        if order.remaining > 0 and command.time_in_force is not TimeInForce.GTC:
            order.dead = True
        self.orders.append(order)

    def _fillable(self, order: _Order) -> bool:
        available = 0
        for resting in self._live(order.side.opposite):
            if not self._crosses(order.side, resting.price, order.price):
                break
            # The taker's own resting quantity is not liquidity it can
            # have. `_match` cancels it under self-match prevention rather
            # than printing against it, so counting it here admits a
            # fill-or-kill that then partially fills, the one outcome the
            # term exists to make impossible. Modelled here as well as in
            # the engine because a differential harness that agrees on the
            # wrong answer reports nothing.
            if resting.agent_id == order.agent_id:
                continue
            available += resting.remaining
            if available >= order.remaining:
                return True
        return False

    def _match(self, incoming: _Order) -> None:
        while incoming.remaining > 0:
            book = self._live(incoming.side.opposite)
            if not book:
                return
            resting = book[0]
            if not self._crosses(incoming.side, resting.price, incoming.price):
                return

            # Self-match prevention, cancel-oldest. Modelled here too so the
            # policy is differentially tested rather than trusted: a wash trade
            # nets to zero and so hides in every aggregate except the tape.
            if resting.agent_id == incoming.agent_id:
                resting.dead = True
                continue

            traded = min(incoming.remaining, resting.remaining)
            resting.remaining -= traded
            incoming.remaining -= traded

            buy_id, sell_id = (
                (incoming.order_id, resting.order_id)
                if incoming.side is Side.BUY
                else (resting.order_id, incoming.order_id)
            )
            self.trades.append(
                NaiveTrade(
                    price=resting.price,
                    quantity=traded,
                    aggressor_side=incoming.side.value,
                    buy_order_id=buy_id,
                    sell_order_id=sell_id,
                )
            )

    def _replace(self, command: Replace) -> None:
        order = self._find(int(command.order_id))
        if order is None or order.dead or order.remaining <= 0:
            return
        if order.agent_id != command.agent_id or int(command.new_quantity) <= 0:
            return

        new_price = (
            int(command.new_price) if command.new_price is not None else order.price
        )
        keeps = new_price == order.price and int(command.new_quantity) < order.remaining
        if keeps:
            order.remaining = int(command.new_quantity)
            return

        # Mutated in place rather than tombstoned-and-recreated. The engine
        # keeps the order id across a replace, so appending a second entry
        # under the same id leaves two orders answering to it, and the next
        # lookup finds the dead one and silently discards the command. Losing
        # priority is expressed by taking a fresh arrival number, which is what
        # priority actually is.
        self._arrival += 1
        order.price = new_price
        order.remaining = int(command.new_quantity)
        order.arrival = self._arrival
        self._match(order)

    # -- comparison --------------------------------------------------------

    @property
    def depth(self) -> dict[tuple[str, int], int]:
        """Aggregated resting quantity per (side, price)."""
        totals: dict[tuple[str, int], int] = {}
        for order in self.orders:
            if order.dead or order.remaining <= 0:
                continue
            key = (order.side.value, order.price)
            totals[key] = totals.get(key, 0) + order.remaining
        return totals
