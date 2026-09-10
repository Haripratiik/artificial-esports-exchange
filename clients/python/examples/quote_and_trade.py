"""Read the book, rest an order behind the touch, look at it, pull it.

The shortest path that exercises everything a systematic trader needs on the
first day: public market data without a credential, a signed order, a signed
read of that order, and a signed cancel. Run it against a venue and it narrates
each step, so a failure lands on a line you can point at.

    export ARENA_URL=http://localhost:8000
    export ARENA_KEY_ID=ak_...
    export ARENA_SECRET=...
    python clients/python/examples/quote_and_trade.py VANTA_OBJECTIVE_WR

Without a key it still runs, and stops after the market data with a note saying
what it skipped. That is deliberate: the first thing to check when a client is
not working is whether it can read the book at all, and that needs no
credential.

The order it sends is a **post-only limit one tick behind the best bid**, so it
rests rather than trading. An example that fills leaves a position behind, on a
venue whose whole discipline is that positions are collateralised and accounted
exactly, and cleaning it up would be a longer example about something else.

Nothing here is real. The venue is a simulation, the underlyings are public
game statistics, and the capital is imaginary.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The client is a separate distribution and is not installed when this file is
# run straight out of the repository, so its root goes on the path first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arena_client import ArenaClient, ArenaError, AuthError, Rejected  # noqa: E402

DEFAULT_SYMBOL = "VANTA_OBJECTIVE_WR"


def rule(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def show_book(book: dict, depth: int = 5) -> None:
    """Draw the ladder the way a person reads one: asks above, bids below."""
    asks = list(book.get("asks", []))[:depth]
    bids = list(book.get("bids", []))[:depth]
    for level in reversed(asks):
        print(f"      ask  {level.price:>12}  x{level.quantity}")
    if asks and bids:
        print(f"      ---  spread {asks[0].price - bids[0].price}")
    for level in bids:
        print(f"      bid  {level.price:>12}  x{level.quantity}")
    if not asks and not bids:
        print("      (the book is empty)")


def order_id_of(payload: object) -> object:
    """Find the order's id in whatever the venue answered with.

    Tolerant on purpose. The exact response body for a placed order belongs to
    the venue, and an example that dies on a field name teaches nothing about
    the API. If none of the usual shapes is present the caller prints the whole
    payload, which is more useful than a KeyError anyway.
    """
    if isinstance(payload, dict):
        for key in ("order_id", "id"):
            if payload.get(key) is not None:
                return payload[key]
        for nested in ("order", "ack"):
            found = order_id_of(payload.get(nested))
            if found is not None:
                return found
    return None


def main() -> int:
    base_url = os.environ.get("ARENA_URL", "http://localhost:8000")
    key_id = os.environ.get("ARENA_KEY_ID")
    secret = os.environ.get("ARENA_SECRET")
    symbol = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SYMBOL

    print(f"venue   {base_url}")
    print(f"symbol  {symbol}")
    print(f"key     {key_id or '(none, market data only)'}")

    with ArenaClient(base_url, key_id=key_id, secret=secret) as client:
        # public: what is listed, and on what grid -----------------------
        rule("1. The instrument")
        instrument = client.instrument(symbol)
        tick = instrument["tick_size"]
        low, high = instrument["settlement_bounds"]
        print(f"  class            {instrument.get('class')}")
        print(f"  tick size        {tick}")
        # Formatted rather than printed raw: str(Decimal) renders a scaled
        # bound as "1E+4", which is exact and unreadable. Formatting is a
        # display concern and belongs here, not in the parsing.
        print(f"  settles between  {low:f} and {high:f}")
        print(f"  expires          {instrument.get('expiry')}")
        print(f"  contract         {instrument.get('spec_digest')}")

        # public: the book ----------------------------------------------
        rule("2. The book")
        book = client.book(symbol, depth=5)
        print(f"  session  {book.get('session')}")
        show_book(book)

        bids = book.get("bids", [])
        if not bids:
            print("\n  No bid to quote behind. Try again once the book has two sides.")
            return 0

        # Exact arithmetic, which is the entire point of parsing to Decimal.
        # A float subtraction here is very nearly right, and a price that is
        # very nearly on a quarter grid is off it, and is rejected.
        best_bid = bids[0].price
        price = best_bid - tick
        print(f"\n  best bid {best_bid}, quoting one tick behind at {price}")
        print(f"  on the grid: {price % tick == 0}")

        if not client.authenticated:
            rule("Stopping here")
            print("  Set ARENA_KEY_ID and ARENA_SECRET to place an order.")
            print("  Everything above needed no credential, which is how it should be.")
            return 0

        # signed: the account -------------------------------------------
        rule("3. The account")
        account = client.account()
        print(f"  seat        {account.get('agent_id')}")
        print(f"  cash        {account.get('cash')}")
        print(f"  free cash   {account.get('free_cash')}")
        print(f"  equity      {account.get('equity')}")

        # signed: place --------------------------------------------------
        rule("4. Placing a post-only order")
        try:
            placed = client.place_order(
                symbol,
                "buy",
                1,
                price=price,
                time_in_force="post_only",
                client_order_id="example-quote-and-trade",
            )
        except Rejected as err:
            # The venue understood and refused. Nothing is wrong with the
            # client; the market simply would not take this order now.
            print(f"  refused: {err.code}: {err.message}")
            if err.detail:
                print(f"  detail:  {err.detail}")
            return 1

        order_id = order_id_of(placed)
        print(f"  response  {placed}")
        if order_id is None:
            print("\n  No order id in that response, so there is nothing to cancel.")
            print("  The payload above is what the venue actually sent.")
            return 1
        print(f"  order id  {order_id}")

        # signed: inspect -------------------------------------------------
        rule("5. Looking at it")
        print(f"  {client.order(symbol, order_id)}")
        working = client.orders()
        print(f"  working orders now: {working}")

        # signed: cancel --------------------------------------------------
        rule("6. Cancelling it")
        print(f"  {client.cancel(symbol, order_id)}")
        print(f"  working orders after: {client.orders()}")

    rule("Done")
    print("  Read the book, rested an order, read it back, pulled it.")
    print("  No real money and no real securities were involved.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuthError as err:
        # Worth its own branch because retrying cannot help, and because the
        # two usual causes look identical from the outside.
        print(f"\nauthentication failed: {err.code}: {err.message}")
        print("Check the key is not revoked, and that this machine's clock is")
        print("within 30 seconds of the venue's. Both arrive as this one code,")
        print("because saying which would tell an unauthenticated caller which")
        print("key ids exist.")
        raise SystemExit(2) from None
    except ArenaError as err:
        print(f"\n{err.code}: {err.message}")
        raise SystemExit(1) from None
