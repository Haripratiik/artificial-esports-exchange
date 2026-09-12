"""This venue behind the interface a Kalshi-shaped engine calls.

The engine keeps one rule about venues: strategies never touch a client, only
the executor does. That makes this file the whole surface between an engine
and this exchange, and it is deliberately the same surface the Kalshi client
offers, method for method, so pointing an engine here is a matter of
constructing a different client.

What this does not do is import the exchange. Every call is HTTP against the
published API, exactly as an outside client makes it. That is not tidiness: an
audit built on a client that reached into the venue's own pricing would be
asking the venue whether it agrees with itself.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import httpx

from connectors.prediction_engine.auth import ArenaSigner, body_bytes
from connectors.prediction_engine.models import Event, Market, Series, to_cents

__all__ = ["ArenaClient", "ArenaError", "match_of", "family_of"]

# A match contract's symbol carries its match and its family: `SOLO0_WIN_VANTA`
# is the winner leg for VANTA on solo match 0, and `OBJECTIVE1_ELIM_MIRE_GT2`
# is an eliminations rung. The venue builds these in `match_book`, and the tag
# is everything before the first underscore.
_MATCH = re.compile(r"^(?P<tag>[A-Z]+\d+)_(?P<family>WIN|TOP\d+|ELIM|H2H)_(?P<rest>.+)$")


class ArenaError(RuntimeError):
    """A refusal from the venue, carrying the venue's own reason code."""

    def __init__(self, status: int, code: str, message: str, path: str) -> None:
        super().__init__(f"{code}: {message} ({path}, HTTP {status})")
        self.status, self.code, self.message, self.path = status, code, message, path


def match_of(symbol: str) -> str | None:
    """Which match a symbol belongs to, or None if it is a statistical contract.

    The statistical listing settles on a four week observation window and has
    no event to belong to in the engine's sense, so it is not given a fake one.
    """
    found = _MATCH.match(symbol)
    return found.group("tag") if found else None


def family_of(symbol: str) -> str | None:
    """WIN, TOPn, ELIM or H2H. The question a contract asks about its match."""
    found = _MATCH.match(symbol)
    return found.group("family") if found else None


@dataclass(frozen=True, slots=True)
class Listing:
    """One row of the instruments endpoint, before it becomes a Market."""

    symbol: str
    instrument_class: str
    bounds: tuple[Decimal, Decimal]
    session: str
    mark: Decimal
    trades: int
    expiry: str
    subjects: tuple[str, ...]
    # The touch, carried on the listing itself. This is why building the whole
    # universe costs one request rather than one per market: reading 308 books
    # to learn 308 touches took longer than this venue's own read timeout, and
    # the answer was already in the first response.
    bid: Decimal | None = None
    ask: Decimal | None = None
    bid_size: int = 0
    ask_size: int = 0


class ArenaClient:
    """The venue, over HTTP.

    Credentials are optional, and that is the point rather than a convenience.
    Every reading method below works without them, so the joint no-arbitrage
    audit, which needs the listing and the book and nothing else, can be run
    against a venue nobody holds an account on.
    """

    def __init__(self, base_url: str = "http://localhost:8000",
                 signer: ArenaSigner | None = None, *, timeout: float = 10.0,
                 prefix: str = "", http: httpx.Client | None = None) -> None:
        # `http` is the seam that makes this testable without a socket. A
        # FastAPI `TestClient` is itself an `httpx.Client`, so passing one runs
        # every method below against the real application in process, signatures
        # and refusals included. Without it the only way to check a connector is
        # to start a server, and a connector nobody can test is a connector
        # whose first bug is found in production.
        self.base_url = base_url.rstrip("/")
        self.signer = signer
        self._prefix = prefix.rstrip("/")
        self._owns_http = http is None
        self._http = http or httpx.Client(base_url=self.base_url, timeout=timeout)

    def __enter__(self) -> ArenaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        # Only what this client opened. A caller who handed us a session is
        # still using it, and closing somebody else's transport is the kind of
        # helpfulness that shows up three tests later as a connection error.
        if self._owns_http:
            self._http.close()

    # -- the wire ---------------------------------------------------------

    def _request(self, method: str, path: str, payload: Any = None) -> Any:
        full = f"{self._prefix}{path}"
        body = body_bytes(payload)
        headers = {"content-type": "application/json"} if payload is not None else {}
        if self.signer is not None:
            headers.update(self.signer.headers(method, full, body))
        response = self._http.request(
            method.upper(), full, content=body or None, headers=headers
        )
        if response.status_code >= 400:
            try:
                error = response.json()["error"]
                code, message = error.get("code", ""), error.get("message", "")
            except Exception:
                code, message = "unreadable", response.text[:200]
            raise ArenaError(response.status_code, code, message, full)
        return response.json()

    # -- reference data ---------------------------------------------------

    def exchange_status(self) -> dict[str, Any]:
        return self._request("GET", "/v1/exchange")

    def listings(self) -> list[Listing]:
        payload = self._request("GET", "/v1/instruments?limit=1000")
        rows = []
        for row in payload["instruments"]:
            low, high = (Decimal(bound) for bound in row["bounds"])
            rows.append(Listing(
                symbol=row["symbol"],
                instrument_class=row["class"],
                bounds=(low, high),
                session=row["session"],
                mark=Decimal(row["mark"]),
                trades=int(row.get("trades", 0)),
                expiry=row.get("expiry", ""),
                subjects=tuple(row.get("subjects", ())),
                bid=None if row.get("bid") is None else Decimal(row["bid"]),
                ask=None if row.get("ask") is None else Decimal(row["ask"]),
                bid_size=int(row.get("bid_size") or 0),
                ask_size=int(row.get("ask_size") or 0),
            ))
        return rows

    def list_series(self) -> list[Series]:
        """The contract families this venue lists, as the engine's series.

        Derived from what is listed rather than fetched, because this venue has
        no series endpoint: a family is a shape its symbols take, and the shapes
        are fixed by `match_book`.
        """
        titles = {
            "WIN": "Who wins the match",
            "ELIM": "Eliminations, over and under",
            "H2H": "One competitor against another",
        }
        families = {family_of(row.symbol) for row in self.listings()}
        return [
            Series(ticker=family, title=titles.get(family, "Top of the field"),
                   category="esports", fee_type="maker_taker")
            for family in sorted(f for f in families if f)
        ]

    def fetch_universe(self) -> tuple[list[Event], list[Market]]:
        """Every live match and the markets on it, in the engine's shapes.

        Only the binary classes come back. This venue also lists futures,
        options, spreads, an index, commodities and a volatility class, and
        none of those is a probability: handing them to an engine that reads
        every price as cents would put nonsense in front of every strategy.
        """
        by_match: dict[str, list[Listing]] = {}
        for row in self.listings():
            tag = match_of(row.symbol)
            if tag is not None and row.instrument_class == "event":
                by_match.setdefault(tag, []).append(row)

        events, markets = [], []
        for tag, rows in sorted(by_match.items()):
            winners = tuple(sorted(
                row.symbol for row in rows if family_of(row.symbol) == "WIN"
            ))
            events.append(Event(
                event_ticker=tag,
                series_ticker="WIN" if winners else "",
                category="esports",
                title=f"Match {tag}",
                mutually_exclusive=bool(winners),
                exhaustive_verified=bool(winners),
                partitions={f"{tag}_WIN": (winners, 1.0)} if winners else {},
            ))
            markets.extend(
                self.market_from(row, tag) for row in sorted(rows, key=lambda r: r.symbol)
            )
        return events, markets

    def market_from(self, row: Listing, tag: str) -> Market:
        """One listing as the engine's Market, using the touch it came with."""
        return Market(
            ticker=row.symbol,
            event_ticker=tag,
            series_ticker=family_of(row.symbol) or "",
            title=row.symbol,
            status=row.session,
            yes_bid=None if row.bid is None else to_cents(row.bid, row.bounds),
            yes_ask=None if row.ask is None else to_cents(row.ask, row.bounds),
            yes_bid_size=float(row.bid_size),
            yes_ask_size=float(row.ask_size),
            volume=float(row.trades),
        )

    def get_market(self, ticker: str) -> Market:
        for row in self.listings():
            if row.symbol == ticker:
                return self.market_from(row, match_of(ticker) or "")
        raise ArenaError(404, "not_found", f"{ticker} is not listed", "/v1/instruments")

    def get_orderbook(self, ticker: str, *, depth: int = 10) -> dict[str, Any]:
        """The touch and the ladder, in this venue's own price units.

        Left in contract units rather than converted, because this is also the
        method a caller uses on the classes that are not probabilities. The
        conversion happens where a Market is built, which is the only place the
        engine's units are owed.
        """
        payload = self._request("GET", f"/v1/instruments/{ticker}/book?depth={depth}")
        # Levels arrive as [price, quantity] pairs with the price as a string,
        # which is this venue keeping prices out of JSON numbers: a double
        # cannot hold a tick exactly and the whole ledger is integers so that
        # conservation can be integer zero.
        bids = [(Decimal(price), int(size)) for price, size in payload.get("bids") or []]
        asks = [(Decimal(price), int(size)) for price, size in payload.get("asks") or []]
        return {
            "ticker": ticker,
            "bid": bids[0][0] if bids else None,
            "ask": asks[0][0] if asks else None,
            "bid_size": bids[0][1] if bids else 0,
            "ask_size": asks[0][1] if asks else 0,
            "bids": bids,
            "asks": asks,
        }

    # -- the account ------------------------------------------------------

    def balance(self) -> dict[str, Any]:
        return self._request("GET", "/v1/account")

    def positions(self) -> dict[str, Any]:
        return self._request("GET", "/v1/account/positions")

    def fills(self, limit: int = 200) -> dict[str, Any]:
        return self._request("GET", f"/v1/account/fills?limit={limit}")

    def orders(self) -> dict[str, Any]:
        return self._request("GET", "/v1/orders")

    # -- trading ----------------------------------------------------------

    def create_order(self, ticker: str, side: str, quantity: int,
                     price: Decimal | str | None = None, *,
                     time_in_force: str = "gtc",
                     client_order_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": ticker,
            "side": side,
            "quantity": int(quantity),
            "time_in_force": time_in_force,
        }
        if price is not None:
            payload["price"] = str(price)
        if client_order_id is not None:
            payload["client_order_id"] = client_order_id
        return self._request("POST", "/v1/orders", payload)

    def cancel(self, ticker: str, order_id: int | str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/orders/{ticker}/{order_id}")

    def cancel_all(self) -> dict[str, Any]:
        return self._request("DELETE", "/v1/orders")

    # -- the rest of the engine's venue protocol ---------------------------
    #
    # `venues/base.py` in the engine names five slices a venue must offer
    # before it can be recorded, and this client covered one of them. The API
    # has since grown what the other four need, so they are implemented here
    # rather than left as a seam somebody finds when a recorder written against
    # the protocol comes back empty.
    #
    # `market_rows`, `trade_page` and `candles` return the venue's own JSON
    # rows on purpose. The protocol's own docstring says it standardises which
    # call a recorder makes and not the payload, because normalising the rows
    # means a per-venue row model, and that is a separate piece of work.

    def market_rows(self, tickers: Sequence[str]) -> list[dict[str, Any]]:
        """A bulk read of named markets, as the venue's rows.

        One request for a whole watchlist. Reading 300 symbols one at a time
        makes a recorder's poll interval a function of how many contracts it
        follows, and this venue lists contracts faster than a poll loop grows.

        A name the venue does not list comes back under `missing` rather than
        raising, so one dead ticker does not cost the caller every live one
        beside it.
        """
        wanted = ",".join(str(name) for name in tickers)
        if not wanted:
            return []
        payload = self._request("GET", f"/v1/instruments?symbols={wanted}")
        return list(payload.get("instruments", []))

    def trade_page(
        self, *, cursor: str | None = None, limit: int = 1000
    ) -> tuple[list[dict[str, Any]], str | None]:
        """One page of the venue-wide tape, newest first, and the next cursor.

        The cursor walks backwards, which is the right way round for a feed
        that is still growing: a cursor into history addresses rows new prints
        cannot disturb, so a client paging back sees every row exactly once and
        a client catching up starts at the head each poll and stops at the
        first `trade_id` it already holds.

        `None` means there is nothing older and is the only thing that means
        it. The venue publishes the cursor quoted for that reason: an ordinal
        of zero, tested for truthiness, would read the oldest page as the end
        of the feed.
        """
        query = f"/v1/trades?limit={int(limit)}"
        if cursor is not None:
            query += f"&cursor={cursor}"
        payload = self._request("GET", query)
        following = payload.get("cursor")
        rows = list(payload.get("trades", []))
        return rows, (None if following is None else str(following))

    def candles(
        self, ticker: str, series: str = "", *, start_ts: int, end_ts: int,
        period_minutes: int = 1,
    ) -> list[dict[str, Any]]:
        """OHLC bars over a closed window, in this venue's own units.

        `series` is accepted and ignored. It is Kalshi's grouping and this
        venue has no series registry, so a symbol alone addresses a book.
        Taking the argument anyway is what lets a caller written against the
        protocol pass its own arguments through unchanged.

        The stamps are simulated nanoseconds and the period is seconds, a
        factor of a billion apart and the easiest thing here to get wrong. The
        conversion happens on one line and nowhere else.
        """
        period = max(1, int(period_minutes)) * 60
        query = (
            f"/v1/instruments/{ticker}/candles?period={period}"
            f"&start={int(start_ts)}&end={int(end_ts)}"
        )
        return list(self._request("GET", query).get("candles", []))

    def max_window_s(self, period_minutes: int = 1) -> int | None:
        """How wide a candle window this venue will answer, in seconds.

        Read before a backfill rather than discovered by provoking a refusal. A
        venue that caps its window silently turns a long backfill into a short
        one with no error, which is why the protocol asks for the cap by name.
        """
        period = max(1, int(period_minutes)) * 60
        for row in self.exchange_status().get("candles", []) or []:
            if int(row.get("period", 0)) == period:
                return row.get("max_window_s")
        return None

    def market_result(self, ticker: str) -> tuple[str, int | None, bool]:
        """How one market resolved, as `(status, value in ticks, voided)`.

        Market data, so it needs no credential, and that is the property the
        engine depends on: this is the settlement call that works in shadow
        mode, where the account's own settled positions are empty by
        construction.

        The status vocabulary is the venue's own and is passed through rather
        than translated: `open` for a contract still trading, `settled`,
        `void`, and `unknown` where the venue cannot say. Guessing at it is a
        real mistake and was made here: this returned a default of `live`,
        which the venue never emits, so a caller branching on the status would
        have treated every open contract as a value it did not recognise.

        An open contract answers with no value. The venue refuses to price one
        even though its oracle could, which is the guard that stops a backtest
        reading the answer in advance.
        """
        for row in self.market_rows([ticker]):
            settled = row.get("settlement") or {}
            return (
                str(settled.get("status", "unknown")),
                settled.get("ticks"),
                bool(settled.get("voided", False)),
            )
        raise ArenaError(
            404, "not_found", f"{ticker} is not listed", "/v1/instruments"
        )

    def iter_settlements(self, **params: Any) -> Iterator[dict[str, Any]]:
        """This account's own settled positions.

        Distinct from `market_result`, deliberately: that one is about the
        market and answers without a credential, this one is about the holder
        and is empty in shadow mode because a shadow account never held
        anything.
        """
        if self.signer is None:
            return iter(())
        limit = int(params.get("limit", 500))
        rows = self.positions().get("positions", [])
        return iter([row for row in rows if row.get("settled")][:limit])

    def queue_position(
        self, ticker: str, order_id: int | str
    ) -> dict[str, Any] | None:
        """How much size rests ahead of one order, and how much behind it.

        The engine prices its shadow fills with this. Without it a backtest
        assumes it is at the front of every queue, which overstates fills in
        exactly the direction that flatters a strategy, and killing that class
        of optimism is what the engine's statistical protocol is for.

        Counts visible quantity only: an iceberg's reserve refreshes behind the
        order rather than ahead of it, so counting it would report a queue the
        order does not sit in.

        Takes the symbol as well as the id, because an order here is keyed by
        both. One matching engine per symbol means an id alone locates nothing.
        """
        row = self._request("GET", f"/v1/orders/{ticker}/{order_id}")
        return row.get("queue")
