"""Trading sessions, and the call auction that opens and reopens them.

A continuous book cannot start itself. At the open there is no price, no
reference and no liquidity, so the first order to arrive would trade against
whatever happened to be resting, which is how a market opens at a number that
means nothing. Real venues solve this by accumulating orders without matching
and then clearing them all at a single price, and so does this.

The same machinery does three jobs, which is why it is one module:

* **the open**: orders accumulate in ``PRE_OPEN``, then one uncross sets the
  first price of the day
* **a halt**: trading stops, orders keep arriving into ``AUCTION``, and the
  reopen is an uncross rather than a free-for-all. A halt that resumed straight
  into continuous trading would hand the first arrival the whole dislocation.
* **the close**: the last price of the day is a cleared price rather than
  whatever the final trade happened to be, which is why closing auctions exist
  at all: index funds and settlement prices need a price that size can actually
  transact at.

The clearing rule
-----------------

Standard, and standard for a reason. Each tie-break exists because the one
before it can leave more than one answer:

1. **maximum executable volume.** The auction's purpose is to trade as much as
   possible at one price.
2. **minimum surplus.** Among prices that trade the same volume, prefer the one
   leaving least unfilled.
3. **the surplus's own side.** If every remaining candidate leaves buyers
   unfilled, the price is too low, so take the highest; if sellers, the lowest.
4. **nearest the reference price.** Only reached when the book is symmetric
   about a range, where any price in it is equally defensible and the previous
   close is the least arbitrary choice.

Market orders take part at any price, which is what makes them market-on-open
orders. They are held at a sentinel price so they cross every candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from arena.exchange.types import Price, Side

__all__ = ["SessionState", "AuctionResult", "indicative_auction", "SENTINEL"]

# Market orders rest at a price that crosses everything. Anything at or beyond
# this magnitude is a market order rather than a genuine limit, and is excluded
# from the candidate prices: clearing "at" 2^62 would be nonsense.
SENTINEL = 1 << 61


class SessionState(Enum):
    """Where a symbol is in its trading day."""

    # Orders accumulate; nothing matches. The opening auction's call phase.
    PRE_OPEN = "pre_open"
    # Normal continuous trading.
    CONTINUOUS = "continuous"
    # Trading suspended, orders still accepted, reopening via an uncross. Both
    # a volatility halt and the closing call use this.
    AUCTION = "auction"
    # No new orders. The outcome is determined; only settlement remains.
    CLOSED = "closed"

    @property
    def accepts_orders(self) -> bool:
        return self is not SessionState.CLOSED

    @property
    def matches_continuously(self) -> bool:
        return self is SessionState.CONTINUOUS


@dataclass(frozen=True, slots=True)
class AuctionResult:
    """What an uncross would do, before it does it."""

    price: Price
    volume: int
    # Demand minus supply at the clearing price. Positive means buyers are left
    # unfilled, which is the signal an opening imbalance feed publishes.
    imbalance: int

    @property
    def surplus_side(self) -> Side | None:
        if self.imbalance > 0:
            return Side.BUY
        if self.imbalance < 0:
            return Side.SELL
        return None

    def to_dict(self) -> dict[str, int | str | None]:
        side = self.surplus_side
        return {
            "price": int(self.price),
            "volume": self.volume,
            "imbalance": self.imbalance,
            "surplus_side": None if side is None else side.value,
        }


def indicative_auction(book, reference: Price | None = None) -> AuctionResult | None:
    """The price an uncross would clear at, and how much would trade.

    Pure: reads the book and changes nothing, so it can be published as an
    indicative-price feed during the call phase as real venues do, and tested
    without an engine.

    Returns ``None`` when nothing would trade: an uncrossed book, or one side
    empty. That is not an error; most of a call phase looks like that.
    """
    snapshot = book.snapshot(levels=1 << 20)
    bids = [(int(p), int(q)) for p, q in snapshot.bids]
    asks = [(int(p), int(q)) for p, q in snapshot.asks]
    if not bids or not asks:
        return None

    candidates = sorted(
        {p for p, _ in bids if abs(p) < SENTINEL}
        | {p for p, _ in asks if abs(p) < SENTINEL}
    )
    if not candidates:
        # Only market orders on both sides. There is no price they imply, so the
        # reference is the only defensible answer, and without one, no auction.
        if reference is None:
            return None
        candidates = [int(reference)]

    best: list[tuple[int, int, int]] = []  # (price, volume, imbalance)
    for price in candidates:
        demand = sum(q for p, q in bids if p >= price)
        supply = sum(q for p, q in asks if p <= price)
        volume = min(demand, supply)
        if volume > 0:
            best.append((price, volume, demand - supply))
    if not best:
        return None

    # 1. maximum executable volume
    top = max(v for _p, v, _i in best)
    best = [row for row in best if row[1] == top]

    # 2. minimum surplus
    least = min(abs(i) for _p, _v, i in best)
    best = [row for row in best if abs(row[2]) == least]

    if len(best) > 1:
        signs = {(1 if i > 0 else -1 if i < 0 else 0) for _p, _v, i in best}
        if signs == {1}:
            # 3. buyers left unfilled everywhere: the price is too low.
            best = [max(best, key=lambda row: row[0])]
        elif signs == {-1}:
            best = [min(best, key=lambda row: row[0])]
        else:
            # 4. genuinely balanced across a range; the previous price is the
            # least arbitrary choice available, and the midpoint if there is none.
            anchor = (
                int(reference)
                if reference is not None
                else (best[0][0] + best[-1][0]) // 2
            )
            best = [min(best, key=lambda row: (abs(row[0] - anchor), row[0]))]

    price, volume, imbalance = best[0]
    return AuctionResult(Price(price), volume, imbalance)
