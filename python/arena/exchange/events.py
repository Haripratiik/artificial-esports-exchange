"""Commands into the engine, and events out of it.

The engine is a pure function from a command and a book state to a list of
events. It performs no I/O, reads no clock, and holds no reference to an agent.
That is what makes a seeded run exactly reproducible, and what lets the same
engine sit behind a discrete-event simulator, a differential test against a
second implementation, or a live paper market without modification.

Every event carries a sequence number assigned by the engine in strict order.
The sequence is the canonical ordering of everything that ever happened, and two
engines fed identical commands must emit identical sequences, which is what
``tests/test_differential.py`` asserts, whether or not a second implementation
is ever written.

Message shapes follow the spirit of NASDAQ's OUCH (commands in) and ITCH (events
out): a command is acknowledged or rejected, fills are reported per-order, and a
trade is reported once for the market. ABIDES models its messaging on the same
protocols, so this also keeps us comparable to the reference simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arena.exchange.types import (
    AgentId,
    OrderId,
    OrderType,
    PegReference,
    Price,
    Quantity,
    RejectReason,
    SequenceNumber,
    Side,
    TimeInForce,
)

__all__ = [
    "Command",
    "Submit",
    "Cancel",
    "Replace",
    "Event",
    "Acknowledged",
    "Rejected",
    "Filled",
    "Traded",
    "Cancelled",
    "Replaced",
]


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Command:
    """Base for anything an agent can ask the engine to do."""

    agent_id: AgentId


@dataclass(frozen=True, slots=True)
class Submit(Command):
    """Place a new order.

    ``price`` is None for a market order and required for a limit order. The
    engine assigns the order id; agents identify their orders by the id returned
    on the acknowledgement, not by one they choose, so two agents can never
    collide.
    """

    side: Side
    quantity: Quantity
    price: Price | None = None
    order_type: OrderType = OrderType.LIMIT
    time_in_force: TimeInForce = TimeInForce.GTC
    # Show this much at a time; zero shows all of it. An iceberg trades
    # visibility for queue priority: each refreshed slice goes to the back of
    # its level, behind everything that arrived while the last one worked.
    display_size: int = 0
    # The price that brings a stop order to life. Required for a stop, refused
    # for anything else.
    stop_price: Price | None = None
    # What a pegged order tracks. Required for a peg, refused for anything else.
    # A peg carries no ``price`` of its own: its price is the reference plus
    # ``peg_offset``, recomputed whenever the reference moves.
    #
    # A separate field rather than a reuse of ``price``, and the reason is not
    # tidiness. Everything above the engine reads ``price`` as a price: the
    # venue reserves collateral from it, and an offset of one tick sitting in
    # that field would have it reserve one tick's worth of cash for an order
    # that will rest at four thousand.
    peg_to: PegReference | None = None
    # Ticks away from the reference, signed toward worse prices for the side
    # that is quoting: a buy with offset -1 bids one tick behind the reference,
    # a sell with +1 offers one tick behind it. Zero means at the reference.
    peg_offset: int = 0
    # Refuse to execute for less than this in any one execution. Zero means no
    # minimum.
    #
    # Not fill-or-kill, which is a statement about the whole order: an MPL order
    # is content to fill in pieces, and only insists that no piece be tiny. The
    # motive is cost rather than impatience; a thousand-lot order picked off
    # one lot at a time pays the spread a thousand times and tells the market
    # what it is doing while it does so.
    min_quantity: int = 0


@dataclass(frozen=True, slots=True)
class Cancel(Command):
    order_id: OrderId


@dataclass(frozen=True, slots=True)
class Replace(Command):
    """Modify a resting order's price or quantity.

    Priority follows the standard rule, which is not arbitrary: a strict
    reduction in quantity at an unchanged price **keeps** queue position,
    because the order is not asking for anything it was not already entitled to.
    Any price change, or any increase in quantity, **loses** position. It is a
    new claim on the queue and jumping ahead of orders that were already
    waiting would be unfair. Getting this backwards is a classic way to build a
    market maker that looks far better than it is.
    """

    order_id: OrderId
    new_quantity: Quantity
    new_price: Price | None = None


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Event:
    sequence: SequenceNumber

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Acknowledged(Event):
    """The engine accepted a command. Carries the assigned order id."""

    agent_id: AgentId
    order_id: OrderId
    side: Side
    quantity: Quantity
    price: Price | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ack",
            "sequence": int(self.sequence),
            "agent_id": str(self.agent_id),
            "order_id": int(self.order_id),
            "side": self.side.value,
            "quantity": int(self.quantity),
            "price": None if self.price is None else int(self.price),
        }


@dataclass(frozen=True, slots=True)
class Rejected(Event):
    agent_id: AgentId
    reason: RejectReason
    order_id: OrderId | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "reject",
            "sequence": int(self.sequence),
            "agent_id": str(self.agent_id),
            "reason": self.reason.value,
            "order_id": None if self.order_id is None else int(self.order_id),
        }


@dataclass(frozen=True, slots=True)
class Filled(Event):
    """One side's view of an execution. Emitted once per order per trade.

    ``aggressor`` distinguishes the order that crossed the spread from the one
    that was resting. Nearly every microstructure measurement the project cares
    about (effective spread, order-flow imbalance, adverse selection) needs
    that distinction, and it cannot be recovered from prices after the fact.
    """

    agent_id: AgentId
    order_id: OrderId
    side: Side
    quantity: Quantity
    price: Price
    aggressor: bool
    remaining: Quantity

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "fill",
            "sequence": int(self.sequence),
            "agent_id": str(self.agent_id),
            "order_id": int(self.order_id),
            "side": self.side.value,
            "quantity": int(self.quantity),
            "price": int(self.price),
            "aggressor": self.aggressor,
            "remaining": int(self.remaining),
        }


@dataclass(frozen=True, slots=True)
class Traded(Event):
    """The public print. One per execution, regardless of how many orders saw it.

    ``aggressor_side`` is the trade's sign for order-flow purposes: a buy-side
    aggressor is an uptick in demand. This is what a market-data subscriber sees;
    the two :class:`Filled` events are private to their owners.
    """

    quantity: Quantity
    price: Price
    aggressor_side: Side
    buy_order_id: OrderId
    sell_order_id: OrderId

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "trade",
            "sequence": int(self.sequence),
            "quantity": int(self.quantity),
            "price": int(self.price),
            "aggressor_side": self.aggressor_side.value,
            "buy_order_id": int(self.buy_order_id),
            "sell_order_id": int(self.sell_order_id),
        }


@dataclass(frozen=True, slots=True)
class Cancelled(Event):
    agent_id: AgentId
    order_id: OrderId
    remaining: Quantity

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "cancel",
            "sequence": int(self.sequence),
            "agent_id": str(self.agent_id),
            "order_id": int(self.order_id),
            "remaining": int(self.remaining),
        }


@dataclass(frozen=True, slots=True)
class Replaced(Event):
    """A modification succeeded. ``kept_priority`` records what it cost."""

    agent_id: AgentId
    order_id: OrderId
    quantity: Quantity
    price: Price
    kept_priority: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "replace",
            "sequence": int(self.sequence),
            "agent_id": str(self.agent_id),
            "order_id": int(self.order_id),
            "quantity": int(self.quantity),
            "price": int(self.price),
            "kept_priority": self.kept_priority,
        }
