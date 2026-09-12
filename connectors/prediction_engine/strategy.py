"""Trading the eliminations ladder, which is the one edge the audit proved.

`audit.py` asks whether the prices standing on the screen admit any consistent
distribution. This file acts on the simplest way they fail.

`ELIM_k_GT{t}` pays 1 if competitor k finishes with more than `t` eliminations.
`X > 2` implies `X > 1`, so P(X > 2) can never exceed P(X > 1) under any
distribution at all. When the book offers the lower rung more cheaply than it
bids for the higher rung, buying the lower and selling the higher collects the
difference for a payoff that is non-negative in every state: above the higher
threshold the two legs cancel, between the two thresholds the long pays and the
short does not, and below both they expire worthless together. No forecast is
involved and none is wanted.

Measured on this venue, from the run recorded in `README.md`: CINDER's ladder on
`OBJECTIVE3` stood at GT1 0.40 / 0.49 and GT2 0.60 / 0.70, so GT1 could be
bought at 0.49 while GT2 bid 0.60. That is 0.11 a contract.

WHAT THIS DELIBERATELY DOES NOT TRADE
=====================================
The top-N ladder is monotone the other way (`TOP2` implies `TOP3`, so there the
cheap leg is the higher rung) and the winner partition is a sum rather than a
pair. Both are left out rather than guessed at. Every number in this module is
measured on the eliminations ladder and none is measured on those.

HOW OFTEN IT FIRES
==================
One board on seed 7 with matches listed, read at eight moments over 24 seconds
of simulated time, counting a ladder as firing when a pair on it clears the fee:

    arbitrageur off   79 of 128 ladder readings, 61.7%, 120 pairs
    arbitrageur on    28 of 128 ladder readings, 21.9%,  32 pairs

The best credit either way was 0.14 a contract. So on the default board, where
the venue's arbitrageur agent is off, this has something to do most of the time.
Turn its competition on and it has something to do about one reading in five,
and in bursts rather than evenly: six of those eight samples carried one to
three inverted ladders and the other two carried seven and nine, which is the
arbitrageur briefly falling behind its own market.

That is the shape the README's LP measurement found, 64.6% against 29.2%, and
it sits slightly below it, which it should: a pairwise inversion at the touch is
a subset of the inconsistencies an LP over the whole ladder detects.

THE WAYS THIS STILL LOSES MONEY
===============================
**Legging risk, which is the real hazard.** These are two books and two orders
and there is no atomic pair on this venue. Both legs go out as
immediate-or-cancel at the touch, which is the honest minimum: neither rests
where the rest of the market can take it at leisure, and neither chases a price
that is no longer the one that was measured. What that cannot do is make the
pair atomic. A fill on one leg and a miss on the other leaves a naked position,
and this module reports that rather than hiding it: `execute` returns
`half_legged` when the venue refuses the second leg outright, and `reconcile`
reads the fills feed and reports how much of each leg actually traded. Nothing
here unwinds a naked leg, because unwinding into the same thin book is how a
small naked position becomes a larger one.

**Collateral is posted per symbol and does not net the hedge.**
`Account.collateral_for_basis` charges each position its own worst case: a long
at 0.49 posts 0.49, a short at 0.60 posts 1 - 0.60 = 0.40. The pair therefore
ties up 0.89 of buying power to collect 0.11 and holds it until the match
settles, even though the two together cannot lose. A run that fires often enough
runs out of collateral, and this venue refuses for collateral *after* accepting
the order, in the fills feed under `rejections`.

**Riskless at settlement is not riskless on a mark.** Both legs are marked until
the match resolves, and the inversion can widen. Long GT1 at 0.49 and short GT2
at 0.60, with the book later at 0.40 and 0.70, is 0.19 of unrealised loss on a
position that must be worth at least +0.11 at settlement. A leaderboard that
marks to market will show that.

**The counterparty is the venue's own arbitrageur.** This trades a stale quote
rather than a broken model: asked of `coherent_prices()` directly, 0 of 16
ladders are non-monotone. The agent whose job this is takes two thirds of these
first, per the measurement above, and what is left is what it cannot keep up
with. Anything that made that agent faster would take this strategy's income
away, and nothing here would notice except the fill rate.
"""

from __future__ import annotations

import argparse
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable, Sequence

from connectors.prediction_engine.client import ArenaClient, ArenaError, Listing
from connectors.prediction_engine.models import CENTS, Market

__all__ = [
    "Trade",
    "inversions",
    "execute",
    "reconcile",
    "scan",
    "fee_per_contract",
    "taker_bps_from",
    "TAKER_BPS",
]

# The same shape `audit.py` parses, restated rather than imported because that
# module's patterns are private to it and this one needs the market itself
# rather than the `Leg` it builds: a `Leg` carries the touch and drops the size,
# and the size is what decides how much of a pair can be lifted.
_ELIM = re.compile(r"^(?P<tag>[A-Z]+\d+)_ELIM_(?P<who>.+)_GT(?P<threshold>\d+)$")

# Only a book that is matching right now carries a price anybody could trade on,
# and `audit.py` records what leaving this out costs. This venue opens every
# contract at the midpoint of its own settlement range on purpose, so ten
# seconds after listing a whole board is quoted near 0.50; a settled book keeps
# its last touch forever. An unfiltered read of those two produced a 3.5 unit
# arbitrage that was a market which had not opened yet. Trading one is worse
# than measuring one: the order rests into an opening call and is never the
# trade that was priced.
_TRADEABLE = "continuous"

# This venue's taker rate in basis points, under the schedule the dashboard runs
# by default. `arena.market.fees.MAKER_TAKER` is `taker_bps=2.0, maker_bps=-1.0`
# and `dashboard/state.py` names it as the default of `MarketConfig.fees`.
#
# Not imported. Nothing in this package imports the exchange, which is the
# property that makes the audit worth running at all, so the rate is read off
# the wire instead: the venue publishes its live schedule at `GET /v1/exchange`
# and `taker_bps_from` asks for it. This constant is the documented default for
# a caller who has not asked, and it is checked against the live venue in
# `tests/test_engine_strategy.py`.
TAKER_BPS = 2.0

# What the venue charges, per fill, on notional, rounded toward itself. A fill
# of `q` lots at price `p` in minor units pays `taker_bps / 10,000 * p * q`
# rounded up, so per contract the charge is the rate against the price.
BASIS_POINTS = 10_000.0

# The settlement range a price has to lie in for a difference of two of them to
# be a credit in probability units. This venue lists over five ranges.
_PROBABILITY = (Decimal(0), Decimal(1))


@dataclass(frozen=True, slots=True)
class Trade:
    """Two legs that together cannot lose.

    `credit` is the gross credit at the touch, the `sell` leg's bid minus the
    `buy` leg's ask, in probability units per contract. That is the number the
    README's worked example calls 0.11. It is gross rather than net because the
    fee is charged per fill against each leg's own notional and is therefore a
    property of the execution rather than of the pair; `fee_per_contract` is
    where it is applied, and `inversions` will not return a pair whose credit
    does not clear it strictly.
    """

    buy: str            # the symbol to buy, the LOWER threshold
    sell: str           # the symbol to sell, the HIGHER threshold
    size: int
    credit: float       # per contract, in probability units
    reason: str


@dataclass(frozen=True, slots=True)
class _Pair:
    """One candidate inversion, before it is sized against the touch."""

    who: str
    low: int
    high: int
    buy: str
    sell: str
    buy_price: float
    sell_price: float
    credit: float
    fee: float


def fee_per_contract(buy_price: float, sell_price: float,
                     taker_bps: float = TAKER_BPS) -> float:
    """What the venue takes out of one contract of the pair, in probability units.

    Both legs cross the spread, so both pay the taker rate rather than earning
    the maker rebate: that is the whole reason `TimeInForce.POST_ONLY` exists on
    this venue, and it is not a thing this strategy can use. The charge is
    `taker_bps` basis points of notional per fill and notional is price times
    quantity, so per contract it is the rate against each leg's own price.

    The venue's own rounding is toward itself and per fill, which adds at most
    one minor unit (0.000001) to each fill. That is left out here because it
    depends on how the engine slices the order, and it is four orders of
    magnitude below the 0.01 tick that bounds the smallest credit worth having.
    """
    return (taker_bps / BASIS_POINTS) * (buy_price + sell_price)


def taker_bps_from(client: ArenaClient) -> float:
    """The live taker rate, from the venue's own exchange endpoint.

    Asked rather than assumed, because the schedule is configuration: this venue
    ships `free`, `maker-taker`, `taker-only` and `generous`, and under
    `taker-only` the rate is 3.0 rather than 2.0. A strategy holding a compiled
    constant would be sizing its break-even against a fee it is not paying.
    """
    fees = client.exchange_status().get("fees") or {}
    return float(fees.get("taker_bps", TAKER_BPS))


def _touch(market: Market) -> tuple[int, int, int, int] | None:
    """Bid and ask in cents with their sizes, or None if this book cannot be traded.

    Two refusals, and each is a way a price can exist without being a price
    anybody can trade at. A book that is not matching is `audit.py`'s filter and
    its reasoning is quoted at `_TRADEABLE`. A book quoted one side only cannot
    be both bought and sold, and completing the missing side with half a spread
    would manufacture the credit this is looking for.
    """
    if market.status != _TRADEABLE or not market.is_quoted_both_sides:
        return None
    return (
        int(market.yes_bid),
        int(market.yes_ask),
        int(market.yes_bid_size),
        int(market.yes_ask_size),
    )


def _ladders(markets: Iterable[Market], tag: str) -> dict[str, list[tuple[int, Market]]]:
    """This match's eliminations rungs, per competitor, in threshold order."""
    rungs: dict[str, list[tuple[int, Market]]] = defaultdict(list)
    for market in markets:
        found = _ELIM.match(market.ticker)
        if found is not None and found.group("tag") == tag:
            rungs[found.group("who")].append((int(found.group("threshold")), market))
    return {
        who: sorted(entries, key=lambda entry: entry[0])
        for who, entries in rungs.items()
    }


def _reason(pair: _Pair) -> str:
    return (
        f"{pair.who}: P(X>{pair.low}) offered at {pair.buy_price:.2f} against "
        f"P(X>{pair.high}) bid at {pair.sell_price:.2f}, and X>{pair.high} implies "
        f"X>{pair.low}, so the pair pays nothing worse than zero in every state. "
        f"{pair.credit:.2f} a contract, against {pair.fee:.4f} of taker fee"
    )


def inversions(markets: Sequence[Market], tag: str, *,
               taker_bps: float = TAKER_BPS,
               min_credit: float = 0.0) -> list[Trade]:
    """Every pair on this match's eliminations ladders worth lifting right now.

    A pair is returned only when `bid(higher) - ask(lower)` is strictly greater
    than the taker fee on both legs plus `min_credit`. The fee part is a
    measured break-even rather than a guessed floor: it is this venue's own
    published rate applied the way `FeeSchedule.charge` applies it. `min_credit`
    is the margin a caller wants on top of break-even, and it defaults to zero
    because the binding constraint here is the tick rather than the fee. An
    event contract ticks at 0.01, and at the default 2 basis points the fee on
    the README's pair of 0.49 against 0.60 is 0.000218, so the smallest
    inversion that can exist at all clears its costs about forty five times
    over.

    Every rung is paired against every higher rung rather than only against its
    neighbour, because monotonicity binds across the whole ladder: GT0 against
    GT3 is as real an inversion as GT0 against GT1. Overlapping pairs are then
    allocated greedily by credit, richest first, against a per symbol budget of
    what is actually on each touch. Without that budget a rung whose ask is
    inverted against two higher bids would be bought twice over and the second
    copy would be an open position rather than a hedge.
    """
    pairs: list[_Pair] = []
    ask_left: dict[str, int] = {}
    bid_left: dict[str, int] = {}

    for who, entries in _ladders(markets, tag).items():
        quoted: list[tuple[int, str, tuple[int, int, int, int]]] = []
        for threshold, market in entries:
            touch = _touch(market)
            if touch is not None:
                quoted.append((threshold, market.ticker, touch))
                ask_left[market.ticker] = touch[3]
                bid_left[market.ticker] = touch[2]

        for low, buy, low_touch in quoted:
            for high, sell, high_touch in quoted:
                if high <= low:
                    continue
                # The lower rung is lifted at its ask and the higher rung is
                # sold into its bid. Held in cents until the last moment: the
                # quotes are integers, and 0.60 - 0.49 in binary floating point
                # is not 0.11, which would put a rounding artefact in the number
                # a caller reads as the edge.
                buy_cents, sell_cents = low_touch[1], high_touch[0]
                credit = (sell_cents - buy_cents) / CENTS
                buy_price, sell_price = buy_cents / CENTS, sell_cents / CENTS
                fee = fee_per_contract(buy_price, sell_price, taker_bps)
                if credit <= fee + min_credit:
                    continue
                pairs.append(_Pair(
                    who=who, low=low, high=high, buy=buy, sell=sell,
                    buy_price=buy_price, sell_price=sell_price,
                    credit=credit, fee=fee,
                ))

    # Richest first, then by symbol, so that two pairs paying the same credit
    # are ordered by something stable rather than by dictionary order.
    pairs.sort(key=lambda pair: (-pair.credit, pair.buy, pair.sell))

    trades: list[Trade] = []
    for pair in pairs:
        size = min(ask_left.get(pair.buy, 0), bid_left.get(pair.sell, 0))
        if size <= 0:
            continue
        ask_left[pair.buy] -= size
        bid_left[pair.sell] -= size
        trades.append(Trade(
            buy=pair.buy, sell=pair.sell, size=size,
            credit=pair.credit, reason=_reason(pair),
        ))
    return trades


def scan(client: ArenaClient, *, taker_bps: float = TAKER_BPS,
         min_credit: float = 0.0) -> list[Trade]:
    """Every inversion on every live match, in one pass over the universe.

    One request rather than one per book. `fetch_universe` carries the touch on
    the listing itself, which is why this costs what a single read costs.
    """
    events, markets = client.fetch_universe()
    by_match: dict[str, list[Market]] = defaultdict(list)
    for market in markets:
        by_match[market.event_ticker].append(market)
    found: list[Trade] = []
    for event in events:
        found.extend(inversions(
            by_match[event.event_ticker], event.event_ticker,
            taker_bps=taker_bps, min_credit=min_credit,
        ))
    return found


def _send(client: ArenaClient, symbol: str, side: str, size: int,
          price: Decimal, client_order_id: str) -> dict[str, Any]:
    """One leg, immediate-or-cancel at the touch, and what the venue said.

    `price` is the venue's own Decimal off the listing rather than a float
    rebuilt from cents, so the order names the price that was measured, to the
    tick. A refusal the venue makes synchronously comes back as a result rather
    than as an exception, because an exception raised on the second leg would
    lose the report of the first leg that has already gone.
    """
    row: dict[str, Any] = {
        "symbol": symbol, "side": side, "size": size, "price": str(price),
        "client_order_id": client_order_id, "status": "", "detail": "",
    }
    try:
        reply = client.create_order(
            symbol, side, size, price,
            time_in_force="ioc", client_order_id=client_order_id,
        )
    except ArenaError as refusal:
        row["status"] = "refused"
        row["detail"] = f"{refusal.code}: {refusal.message}"
        return row
    row["status"] = str(reply.get("status", ""))
    row["reply"] = reply
    return row


def execute(client: ArenaClient, trades: Iterable[Trade], *,
            dry_run: bool = False, taker_bps: float = TAKER_BPS,
            min_credit: float = 0.0, prefix: str = "elim") -> list[dict[str, Any]]:
    """Send each pair as two immediate-or-cancel orders, and report what happened.

    Every pair is re-read before anything is sent. One `listings()` call covers
    the whole batch and carries three things a `Trade` from a minute ago cannot:
    the session, so a book that has left continuous trading since the scan is
    not traded into; the current touch, so a pair whose inversion has closed is
    dropped rather than lifted at a worse price; and the sizes, so the order is
    still bounded by what is actually there.

    The leg with the smaller naked exposure goes first. A synchronous refusal on
    the first leg means nothing traded at all, so the first leg should be the
    one this seat would rather be left holding: a naked short of the higher rung
    risks `1 - sell`, a naked long of the lower rung risks `buy`, and which of
    those is smaller depends on the prices. That ordering does nothing about the
    refusal that matters most, which this venue makes after accepting the order
    (`test_broker.py` pins that for collateral), and it is not claimed to.

    `dry_run` sends nothing at all. It still reads, and reports the size and the
    prices a live run would have used, because a dry run that reported nothing
    would be indistinguishable from a dry run that found nothing.
    """
    batch = list(trades)
    results: list[dict[str, Any]] = []
    if not batch:
        return results

    quotes: dict[str, Listing] = {row.symbol: row for row in client.listings()}
    ask_left: dict[str, int] = {}
    bid_left: dict[str, int] = {}

    for trade in batch:
        outcome: dict[str, Any] = {
            "buy": trade.buy, "sell": trade.sell,
            "size": 0, "credit": 0.0,
            "status": "stale", "detail": "",
            "orders": {"buy": None, "sell": None},
            "client_order_ids": {"buy": None, "sell": None},
        }
        results.append(outcome)

        buy_row, sell_row = quotes.get(trade.buy), quotes.get(trade.sell)
        if buy_row is None or sell_row is None:
            outcome["detail"] = "one of the two legs is no longer listed"
            continue
        if buy_row.session != _TRADEABLE or sell_row.session != _TRADEABLE:
            outcome["detail"] = (
                f"{trade.buy} is {buy_row.session} and {trade.sell} is "
                f"{sell_row.session}; only a matching book carries a price "
                "anybody can trade at"
            )
            continue
        if buy_row.ask is None or sell_row.bid is None:
            outcome["detail"] = "the touch is quoted one side only"
            continue
        if buy_row.bounds != _PROBABILITY or sell_row.bounds != _PROBABILITY:
            # The credit below is a difference of two prices, which only means
            # anything when both are probabilities. `models.to_cents` refuses
            # the same thing for the same reason: a future bounded [0, 10,000]
            # has a perfectly good price of 0.47 and it is not 47 cents.
            outcome["detail"] = (
                f"{trade.buy} is bounded {buy_row.bounds} and {trade.sell} is "
                f"bounded {sell_row.bounds}; a difference of those is not a credit"
            )
            continue

        # Subtracted as the venue's own decimals and made a float afterwards.
        # Taken the other way round, 0.60 - 0.49 in binary floating point is
        # 0.10999999999999999, which is the edge reported with a rounding
        # artefact in it.
        buy_price, sell_price = float(buy_row.ask), float(sell_row.bid)
        credit = float(sell_row.bid - buy_row.ask)
        outcome["credit"] = credit
        if credit <= fee_per_contract(buy_price, sell_price, taker_bps) + min_credit:
            outcome["detail"] = (
                f"the inversion has gone: {trade.sell} bids {sell_row.bid} "
                f"against {trade.buy} offered at {buy_row.ask}"
            )
            continue

        room_buy = ask_left.get(trade.buy, int(buy_row.ask_size))
        room_sell = bid_left.get(trade.sell, int(sell_row.bid_size))
        size = min(trade.size, room_buy, room_sell)
        if size <= 0:
            outcome["detail"] = "nothing left on one of the two touches"
            continue
        ask_left[trade.buy] = room_buy - size
        bid_left[trade.sell] = room_sell - size
        outcome["size"] = size

        if dry_run:
            outcome["status"] = "dry_run"
            outcome["detail"] = (
                f"would sell {size} {trade.sell} at {sell_row.bid} and buy "
                f"{size} {trade.buy} at {buy_row.ask}"
            )
            continue

        stamp = uuid.uuid4().hex[:12]
        buy_id, sell_id = f"{prefix}-{stamp}-b", f"{prefix}-{stamp}-s"
        outcome["client_order_ids"] = {"buy": buy_id, "sell": sell_id}

        legs = [
            ("sell", trade.sell, sell_row.bid, sell_id),
            ("buy", trade.buy, buy_row.ask, buy_id),
        ]
        if (1.0 - sell_price) > buy_price:
            legs.reverse()

        first, second = legs
        sent = _send(client, first[1], first[0], size, first[2], first[3])
        outcome["orders"][first[0]] = sent
        if sent["status"] == "refused":
            outcome["status"] = "refused"
            outcome["detail"] = (
                f"the {first[0]} leg was refused before anything traded, so the "
                f"{second[0]} leg was not sent: {sent['detail']}"
            )
            continue

        after = _send(client, second[1], second[0], size, second[2], second[3])
        outcome["orders"][second[0]] = after
        if after["status"] == "refused":
            outcome["status"] = "half_legged"
            outcome["detail"] = (
                f"the {first[0]} leg went and the {second[0]} leg did not, so "
                f"this seat may now hold up to {size} {first[1]} with nothing "
                f"against it: {after['detail']}"
            )
            continue

        outcome["status"] = "sent"
        outcome["detail"] = (
            "both legs accepted. This venue accepts asynchronously, so nothing "
            "here says either one traded; reconcile() reads the fills feed"
        )
    return results


def reconcile(client: ArenaClient, results: Iterable[dict[str, Any]], *,
              limit: int = 200) -> list[dict[str, Any]]:
    """How much of each leg actually traded, read back from the fills feed.

    This exists because `execute` cannot answer it. The POST returns before the
    order reaches a book, so an accepted order is not a filled order, and an
    order the account cannot fund is accepted and then refused at the book with
    the reason in this same feed under `rejections`. A loop that read only
    `execute`'s return value would believe every pair worked.

    Fills are tied back by `client_order_id`, which is the only identifier this
    strategy chose and the only one that survives the round trip. Rejections
    carry a symbol and a reason but no client id, so they are reported against
    the legs by symbol and can include refusals that predate this batch: the
    client's `fills()` takes a page size and not the cursor the endpoint offers,
    so there is nothing here to filter them by.

    `naked` is the imbalance in contracts, positive long the lower rung and
    negative short the higher one. Zero against a non-zero fill is the pair that
    was wanted, and it is the only case where `collected` is not zero.
    """
    feed = client.fills(limit)
    filled: dict[str, int] = defaultdict(int)
    for row in feed.get("fills") or []:
        key = row.get("client_order_id")
        if key:
            filled[str(key)] += int(row.get("quantity", 0))
    refusals = list(feed.get("rejections") or [])

    rows: list[dict[str, Any]] = []
    for outcome in results:
        ids = outcome.get("client_order_ids") or {}
        bought = filled.get(str(ids.get("buy")), 0)
        sold = filled.get(str(ids.get("sell")), 0)
        legs = (outcome.get("buy"), outcome.get("sell"))
        credit = float(outcome.get("credit", 0.0))
        rows.append({
            "buy": outcome.get("buy"),
            "sell": outcome.get("sell"),
            "status": outcome.get("status"),
            "intended": int(outcome.get("size", 0)),
            "bought": bought,
            "sold": sold,
            "matched": bought == sold,
            "naked": bought - sold,
            "credit": credit,
            "collected": (bought if bought == sold else 0) * credit,
            "refusals": [
                str(row.get("reason", "")) for row in refusals
                if row.get("symbol") in legs
            ],
        })
    return rows


def main() -> int:
    """Scan, and with `--live` trade. Dry by default, deliberately.

    A strategy that trades the first time somebody runs it is a strategy whose
    first run is an accident.
    """
    parser = argparse.ArgumentParser(description="trade the eliminations ladder")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--key-id", default="")
    parser.add_argument("--secret", default="")
    parser.add_argument("--live", action="store_true",
                        help="actually send orders, rather than reporting them")
    parser.add_argument("--min-credit", type=float, default=0.0,
                        help="probability units of margin demanded above break-even")
    args = parser.parse_args()

    signer = None
    if args.key_id and args.secret:
        from connectors.prediction_engine.auth import ArenaSigner
        signer = ArenaSigner(args.key_id, args.secret)
    if args.live and signer is None:
        # Said once here rather than as one refusal per leg. Reading the board
        # needs no account, which is the whole reason the audit can be run
        # against a venue nobody holds a seat on; trading it does.
        print("--live needs --key-id and --secret: an order has to be signed")
        return 2

    with ArenaClient(args.url, signer) as client:
        rate = taker_bps_from(client)
        trades = scan(client, taker_bps=rate, min_credit=args.min_credit)
        print(f"taker fee {rate} bps, {len(trades)} inversions on the live board")
        for trade in trades:
            print(f"  buy {trade.size:>4} {trade.buy:44s} sell {trade.sell}")
            print(f"       {trade.reason}")
        if not trades:
            return 0
        done = execute(client, trades, dry_run=not args.live, taker_bps=rate,
                       min_credit=args.min_credit)
        for outcome in done:
            print(f"  {outcome['status']:12s} {outcome['buy']} / {outcome['sell']}")
            print(f"       {outcome['detail']}")
        if args.live and signer is not None:
            for row in reconcile(client, done):
                print(f"  bought {row['bought']} sold {row['sold']} naked "
                      f"{row['naked']} on {row['buy']} / {row['sell']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
