"""Messages between agents and the exchange.

Split by who is allowed to see them, because that distinction is the substrate
for every information-asymmetry experiment in the project:

**Private**: acknowledgements, fills, cancels, rejects. Sent only to the agent
whose order it was, after that agent's own latency.

**Public**: trade prints and book updates. Broadcast to subscribers, each after
*their* latency, so two agents subscribed to the same feed genuinely see the same
event at different times. That is not a simulation artifact to be smoothed over;
it is the thing being studied.

A market-data subscription names a feed, so an agent can be given top-of-book
without depth, or trades without quotes. Restricting what an agent can see is how
a stat-arb fund is denied fundamentals, or a retail agent denied L2.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from arena.exchange.book import BookSnapshot
from arena.exchange.events import Event
from arena.exchange.types import AgentId, Price, Quantity, SequenceNumber, Side
from arena.sim.time import Timestamp

__all__ = [
    "Feed",
    "Subscribe",
    "Unsubscribe",
    "PrivateEvent",
    "TradePrint",
    "TopOfBook",
    "DepthUpdate",
    "MarketOpen",
    "MarketClose",
]


class Feed(Enum):
    """What an agent may subscribe to."""

    TRADES = "trades"
    TOP_OF_BOOK = "top_of_book"
    DEPTH = "depth"


@dataclass(frozen=True, slots=True)
class Subscribe:
    """Ask for a feed.

    ``throttle`` is a minimum interval between updates on this feed for this
    agent. It models a subscriber that cannot or does not consume every tick:
    a retail client on a slow connection, or an agent that pays for a slower
    data product. Zero means every update.
    """

    feed: Feed
    symbol: str | None = None
    throttle: int = 0


@dataclass(frozen=True, slots=True)
class Unsubscribe:
    feed: Feed
    symbol: str | None = None


@dataclass(frozen=True, slots=True)
class PrivateEvent:
    """An engine event addressed to the agent that caused it.

    Carries the symbol because the matching engine does not: each engine serves
    one book and so has no need of one. The venue knows which book produced the
    event, and attaching it here is what lets an agent attribute a fill to an
    instrument without having to remember what it last sent, a guess that
    breaks the moment it has orders working in two symbols at once.
    """

    event: Event
    symbol: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "private",
            "symbol": self.symbol,
            "event": self.event.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TradePrint:
    """A public execution. What the tape shows.

    ``sequence`` is the engine's own match number, carried through unchanged.
    Real feeds publish one for the same reasons we need one: it uniquely
    identifies a print, so a subscriber can detect a gap, deduplicate a replay,
    and align its view against another subscriber's. Prices repeat constantly in
    a narrow market: anything that tries to identify a trade by its price will
    quietly match the wrong one.
    """

    symbol: str
    timestamp: Timestamp
    sequence: SequenceNumber
    price: Price
    quantity: Quantity
    aggressor_side: Side

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "trade_print",
            "symbol": self.symbol,
            "timestamp": int(self.timestamp),
            "sequence": int(self.sequence),
            "price": int(self.price),
            "quantity": int(self.quantity),
            "aggressor_side": self.aggressor_side.value,
        }


@dataclass(frozen=True, slots=True)
class TopOfBook:
    """Best bid and ask with their sizes. The cheapest useful quote feed."""

    symbol: str
    timestamp: Timestamp
    bid: Price | None
    bid_size: Quantity
    ask: Price | None
    ask_size: Quantity

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return (int(self.bid) + int(self.ask)) / 2.0

    @property
    def spread(self) -> int | None:
        if self.bid is None or self.ask is None:
            return None
        return int(self.ask) - int(self.bid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "top_of_book",
            "symbol": self.symbol,
            "timestamp": int(self.timestamp),
            "bid": None if self.bid is None else int(self.bid),
            "bid_size": int(self.bid_size),
            "ask": None if self.ask is None else int(self.ask),
            "ask_size": int(self.ask_size),
        }


@dataclass(frozen=True, slots=True)
class DepthUpdate:
    """Aggregated L2 depth, best first."""

    symbol: str
    timestamp: Timestamp
    snapshot: BookSnapshot

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "depth",
            "symbol": self.symbol,
            "timestamp": int(self.timestamp),
            "bids": [[int(p), int(q)] for p, q in self.snapshot.bids],
            "asks": [[int(p), int(q)] for p, q in self.snapshot.asks],
        }


@dataclass(frozen=True, slots=True)
class MarketOpen:
    timestamp: Timestamp
    instrument: str


@dataclass(frozen=True, slots=True)
class MarketClose:
    """The session has ended. Settlement, if any, happens outside the kernel."""

    timestamp: Timestamp
    instrument: str
