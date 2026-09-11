"""This venue behind the broker interface a portfolio manager already calls.

A portfolio optimiser written for Alpaca keeps its broker behind one abstract
base class with nine methods and one property, and `AlpacaBroker` is merely the
first implementation of it. This is the second. Everything above it, the
allocator, the sizing, the pattern research and the bot loop, is written against
the base class and does not know or care which venue is underneath.

That makes this a small file with an outsized consequence: a system built to
trade equities through a real broker can trade a synthetic esports exchange
without a branch, and every order it sends is settled by a ledger whose
conservation is integer zero rather than by a paper-trading endpoint that
approximates one.

WHERE THE TWO BROKERS GENUINELY DIFFER
--------------------------------------
**An order is accepted before it exists.** Alpaca returns an order id from
`place_order`. This venue accepts asynchronously: the POST comes back before
the order has reached a book, so there is no venue id yet, and a record that
stays pending was refused at the book with the reason in the fills feed under
`rejections`. So `orderId` is the **client** order id throughout, minted here
when the caller brings none. That is the only handle that exists at the moment
`place_order` returns, and handing back anything else would give the caller an
identifier `cancel_order` could not take.

**An order id is not unique here either.** One matching engine per symbol means
an order is keyed by `(symbol, id)`, so cancelling needs to know which book.
`cancel_order` resolves both questions against the blotter, and reports False
rather than guessing: an order still in flight genuinely cannot be cancelled,
and cancelling the wrong symbol's order is a worse failure than not cancelling.

**There is no short side.** Alpaca sells short against a margin agreement. Here
a negative position is an ordinary collateralised short and the venue charges
its exact worst case up front, so `Position.side` reports `short` for a negative
quantity and nothing else changes.

**Prices are strings on the wire and are not floats here.** This venue holds
money in integer minor units so that conservation can be exactly zero, and the
broker interface above wants floats. The conversion happens in this file and
only in this file, which is the honest place for it: a float is what the
allocator's arithmetic needs, and it is not what the ledger is made of.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import uuid4

from connectors.prediction_engine.auth import ArenaSigner
from connectors.prediction_engine.client import ArenaClient, ArenaError

__all__ = [
    "Position",
    "OrderResult",
    "BrokerOrder",
    "AccountInfo",
    "ArenaBroker",
]


# The four shapes the portfolio manager passes around. Copied rather than
# imported for the same reason the prediction engine's models are: a connector
# that could only be built by installing the other project is a connector
# nobody runs. Field names and casing match it exactly, so an instance built
# here is accepted there unchanged.


@dataclass
class Position:
    symbol: str
    qty: float
    side: str                       # long or short
    avgEntryPrice: float
    currentPrice: float
    marketValue: float
    unrealizedPL: float
    unrealizedPLPct: float


@dataclass
class OrderResult:
    orderId: str
    symbol: str
    side: str                       # buy or sell
    qty: float
    orderType: str                  # market or limit
    limitPrice: float | None
    status: str
    filledPrice: float | None
    filledAt: str | None
    message: str = ""


@dataclass
class BrokerOrder:
    orderId: str
    symbol: str
    side: str
    qty: float
    orderType: str
    filledQty: float
    filledAvgPrice: float | None
    status: str
    createdAt: str | None = None
    limitPrice: float | None = None


@dataclass
class AccountInfo:
    equity: float
    cash: float
    buyingPower: float
    portfolioValue: float
    dayPL: float
    dayPLPct: float


def _number(raw: Any, default: float = 0.0) -> float:
    """A price off the wire, as the float the interface above wants."""
    if raw is None:
        return default
    try:
        return float(Decimal(str(raw)))
    except Exception:
        return default


class ArenaBroker:
    """The exchange, as a broker.

    Structurally a `BrokerClient`: every abstract method is implemented with the
    same name and signature. It does not inherit from one, because the base
    class lives in the other project and importing it would make this file
    unusable without that project checked out. Dropping this beside
    `BrokerClient.py` and adding the base to the class statement is a one line
    change for anyone who wants `isinstance` to answer yes.
    """

    def __init__(self, base_url: str = "http://localhost:8000",
                 key_id: str = "", secret: str = "", *,
                 dry_run: bool = False, timeout: float = 10.0,
                 http: Any = None) -> None:
        signer = ArenaSigner(key_id, secret) if key_id and secret else None
        self.client = ArenaClient(base_url, signer, timeout=timeout, http=http)
        self._dry_run = dry_run
        self._marks: dict[str, float] = {}

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> ArenaBroker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- the account ------------------------------------------------------

    def get_account(self) -> AccountInfo:
        """Equity, cash and what is left to trade with.

        `buyingPower` is free cash rather than cash, and the difference is the
        whole point of this venue: collateral already posted against open
        positions is not available to open more, and the venue computes that
        figure by exact arithmetic rather than by a margin model.
        """
        raw = self.client.balance()
        equity = _number(raw.get("equity"))
        start = _number(raw.get("starting_cash"))
        pnl = _number(raw.get("pnl"))
        return AccountInfo(
            equity=equity,
            cash=_number(raw.get("cash")),
            buyingPower=_number(raw.get("free_cash")),
            portfolioValue=equity,
            dayPL=pnl,
            dayPLPct=(100.0 * pnl / start) if start else 0.0,
        )

    def get_cash(self) -> float:
        return _number(self.client.balance().get("cash"))

    def marks(self, refresh: bool = False) -> dict[str, float]:
        """The venue's mark per symbol, cached.

        A position row carries what it cost and what it realised; it does not
        carry what it is worth now, because that is a fact about the market
        rather than about the position. One listing request answers it for
        every symbol at once.
        """
        if refresh or not self._marks:
            self._marks = {
                row.symbol: float(row.mark) for row in self.client.listings()
            }
        return self._marks

    def get_positions(self) -> list[Position]:
        payload = self.client.positions()
        marks = self.marks(refresh=True)
        out = []
        for row in payload.get("positions", []):
            quantity = float(row.get("quantity", 0) or 0)
            if quantity == 0:
                # A flat row is kept by the venue so a day's trading stays
                # auditable, and is not a position anything upstream should
                # size against.
                continue
            average = _number(row.get("average_price"))
            mark = marks.get(row["symbol"], average)
            value = quantity * mark
            cost = quantity * average
            unrealised = value - cost
            out.append(Position(
                symbol=row["symbol"],
                qty=quantity,
                side="long" if quantity > 0 else "short",
                avgEntryPrice=average,
                currentPrice=mark,
                marketValue=value,
                unrealizedPL=unrealised,
                unrealizedPLPct=(100.0 * unrealised / abs(cost)) if cost else 0.0,
            ))
        return out

    def get_position(self, symbol: str) -> Position | None:
        for position in self.get_positions():
            if position.symbol == symbol:
                return position
        return None

    # -- orders -----------------------------------------------------------

    def get_orders(self, status: str = "open") -> list[BrokerOrder]:
        """Working orders, identified the way the caller identified them.

        `orderId` is the **client** order id throughout, not the venue's. This
        venue accepts an order asynchronously: the POST returns before the
        order has reached a book, so there is no venue id yet, and the only
        handle that exists at the moment `place_order` returns is the one the
        caller sent. Reporting the venue id here would hand back a different
        identifier from the one that was issued, and `cancel_order` takes
        whatever `place_order` gave out.

        Orders still in flight are **not** included, and that is a judgement
        rather than an omission. The venue lists them separately because a
        pending record that stays pending was refused at the book, and a
        refusal carries the venue's order id, which an order that never rested
        never got. So a refused order cannot be matched back to the client id
        it was sent under, and anything reporting pending records as open would
        report a refused order as working forever. An allocator reweighting
        around an order that does not exist is a worse failure than one that
        cannot see an order for the few milliseconds it is on the wire.
        `pending()` answers the in-flight question and `rejections()` answers
        what became of them.

        Only `open` is answerable. This blotter holds what is live, and history
        lives in fills and positions, so `closed` returns nothing rather than
        raising: a loop polling for open orders should not die because it asked
        a question this venue answers somewhere else.
        """
        if status not in ("open", "all") or self.client.signer is None:
            return []
        payload = self.client.orders()
        out = []
        for row in payload.get("orders", []):
            quantity = float(row.get("quantity", 0) or 0)
            remaining = float(row.get("remaining", quantity) or 0)
            filled = float(row.get("filled", quantity - remaining) or 0)
            limit = row.get("price")
            out.append(BrokerOrder(
                orderId=str(row.get("client_order_id") or row.get("order_id")),
                symbol=row["symbol"],
                side=str(row.get("side", "")).lower(),
                qty=quantity,
                orderType="limit" if limit is not None else "market",
                filledQty=filled,
                filledAvgPrice=None,
                status="partially_filled" if filled else "new",
                limitPrice=_number(limit, None) if limit is not None else None,
            ))
        return out

    def pending(self) -> list[BrokerOrder]:
        """Orders this venue has taken but not yet put in a book.

        A record here is on the wire or was refused when it arrived, and the
        two are indistinguishable from outside until `rejections()` says which.
        Separate from `get_orders` for that reason: they are not working
        orders, and treating them as such is how a sizing loop ends up
        reweighting around exposure it does not have.
        """
        if self.client.signer is None:
            return []
        return [
            BrokerOrder(
                orderId=str(row.get("client_order_id")),
                symbol=row["symbol"],
                side=str(row.get("side", "")).lower(),
                qty=float(row.get("quantity", 0) or 0),
                orderType="limit",
                filledQty=0.0,
                filledAvgPrice=None,
                status="new",
            )
            for row in self.client.orders().get("pending", [])
        ]

    def _resting(self, client_order_id: str) -> tuple[str, str] | None:
        """Which book a client id is resting in, and the venue's id for it."""
        if self.client.signer is None:
            return None
        for row in self.client.orders().get("orders", []):
            if str(row.get("client_order_id")) == str(client_order_id):
                return row["symbol"], str(row.get("order_id"))
        return None

    def rejections(self, limit: int = 200) -> list[dict[str, Any]]:
        """Orders this venue took and then refused at the book.

        Not part of the broker interface, and it has to exist anyway. Alpaca
        refuses synchronously, so a caller there learns from the return value
        of `place_order`. Here an order is accepted in flight and can still be
        refused when it arrives, most often for collateral it turns out not to
        have, and that refusal is reported in the fills feed rather than in the
        response the caller already read.

        An allocator that only checks `OrderResult.status` will therefore
        believe every order worked. This is where it finds out otherwise, and
        the sizing loop should read it before it reweights on a position it
        does not hold.
        """
        if self.client.signer is None:
            return []
        return list(self.client.fills(limit).get("rejections", []))

    def place_order(self, symbol: str, side: str, qty: float,
                    order_type: str = "market", limit_price: float | None = None,
                    time_in_force: str = "day",
                    client_order_id: str | None = None) -> OrderResult:
        """Send one order.

        `time_in_force` is translated rather than passed through. Alpaca's
        `day` means until the session closes; this venue's sessions are
        simulated and a contract can outlive many of them, so `day` maps to
        good-till-cancelled, which is what a caller asking for `day` actually
        wants here. `ioc` and `fok` pass through unchanged because both venues
        mean the same thing by them.
        """
        quantity = int(qty)
        if quantity <= 0:
            return OrderResult(
                orderId="", symbol=symbol, side=side, qty=qty,
                orderType=order_type, limitPrice=limit_price,
                status="rejected", filledPrice=None, filledAt=None,
                message="an order for no lots is not an order",
            )
        if self._dry_run:
            return OrderResult(
                orderId="dry", symbol=symbol, side=side, qty=quantity,
                orderType=order_type, limitPrice=limit_price,
                status="dry_run", filledPrice=None, filledAt=None,
                message="dry run, nothing was sent",
            )

        tif = {"day": "gtc", "gtc": "gtc", "ioc": "ioc", "fok": "fok"}.get(
            str(time_in_force).lower(), "gtc"
        )
        price = None if order_type == "market" else limit_price
        # A client id, minted here when the caller did not bring one. This
        # venue accepts asynchronously and the POST returns before the order
        # has reached a book, so there is no venue id to hand back: the client
        # id is the only handle that exists at this moment, and it is the one
        # `cancel_order` and `get_orders` speak in.
        handle = client_order_id or f"pm-{uuid4().hex[:16]}"
        try:
            raw = self.client.create_order(
                symbol, str(side).lower(), quantity,
                None if price is None else str(price), time_in_force=tif,
                client_order_id=handle,
            )
        except ArenaError as refused:
            return OrderResult(
                orderId="", symbol=symbol, side=side, qty=quantity,
                orderType=order_type, limitPrice=limit_price,
                status="rejected", filledPrice=None, filledAt=None,
                message=f"{refused.code}: {refused.message}",
            )
        # `accepted` means in flight, not filled and not certain to rest. An
        # order this venue cannot collateralise is taken here and refused at
        # the book, and that refusal arrives in the fills feed under
        # `rejections` rather than in this response.
        return OrderResult(
            orderId=str(raw.get("client_order_id") or handle),
            symbol=symbol,
            side=str(side).lower(),
            qty=quantity,
            orderType=order_type,
            limitPrice=limit_price,
            status=str(raw.get("status", "submitted")),
            filledPrice=None,
            filledAt=None,
            message="",
        )

    def close_position(self, symbol: str) -> OrderResult:
        """Flatten one symbol with a marketable order the other way."""
        position = self.get_position(symbol)
        if position is None:
            return OrderResult(
                orderId="", symbol=symbol, side="", qty=0.0,
                orderType="market", limitPrice=None, status="rejected",
                filledPrice=None, filledAt=None, message="no position to close",
            )
        side = "sell" if position.qty > 0 else "buy"
        return self.place_order(symbol, side, abs(position.qty), order_type="market")

    def cancel_order(self, orderId: str) -> bool:
        """Cancel one order by the id `place_order` handed out.

        Two translations happen here, and both are this venue rather than this
        adapter. The id is a client id and the venue cancels by its own, and
        the id alone does not locate an order in any case: one matching engine
        per symbol means an order is keyed by `(symbol, id)`. The blotter
        answers both questions at once.

        A miss is False rather than a guess. An order still in flight has no
        venue id yet and genuinely cannot be cancelled, and cancelling the
        wrong symbol's order is a worse failure than not cancelling.
        """
        found = self._resting(str(orderId))
        if found is None:
            return False
        symbol, venue_id = found
        try:
            self.client.cancel(symbol, venue_id)
        except ArenaError:
            return False
        return True

    def cancel_all_orders(self) -> int:
        try:
            raw = self.client.cancel_all()
        except ArenaError:
            return 0
        return int(raw.get("count", raw.get("cancelled", 0)) or 0)

    # -- what this is -----------------------------------------------------

    @property
    def mode(self) -> str:
        """`paper`, because no position here settles into money that exists.

        Not `dry_run`, which the interface above reserves for a broker that
        sends nothing: orders sent here reach a real matching engine, take real
        queue priority, and settle against a real ledger. The money is the only
        part that is synthetic, which is exactly what paper trading means.
        """
        return "dry_run" if self._dry_run else "paper"
