"""Artificial Esports Exchange: the exchange, served to a browser.

    python -m dashboard.server
    # then open http://127.0.0.1:8000

The browser is a *viewer of the simulator*, never a second implementation of it.
Every order a person clicks is enqueued onto the same human agent an algorithm
would be, travels the same latency link, and is checked by the same collateral
code. A UI that applied orders directly would be showing you a market you are
not actually in.

Layout of the surface:

    GET  /api/instruments      what is listed, and the terms of each contract
    GET  /api/session          phases, halts, fees, the running configuration
    GET  /api/agents           who else is in the market
    GET  /api/history/{sym}    price path, for charting
    GET  /api/diagnostics/{s}  stylized-fact report on the live series
    GET  /api/book/{sym}       a deeper ladder than the socket carries
    POST /api/config           rebuild the market with a new configuration
    POST /api/session/{sym}/halt      suspend trading
    POST /api/session/{sym}/uncross   clear the call and resume

Those last three, plus kill and revive, are OPERATOR routes: they reach past
the caller and change the market for everyone in it. They need the token in
`dashboard.operator_auth` and answer 404 without it. `POST /api/config`
discards every account, position and working order for every connected user,
and `kill` takes an arbitrary agent id, so before they were gated, one visitor
could end another's session.
    WS   /ws                   live snapshot at the tick rate, and order entry

Those are the *page's* endpoints, shaped for one browser: they answer whatever
the screen needs and they trust a session cookie. A program wants neither. The
programmatic surface is separate and versioned, under ``/v1``, and is
documented in `docs/API.md`:

    GET  /v1/exchange          clock, generation, conservation, session summary
    GET  /v1/instruments       every listing, filterable by class or subject
    GET  /v1/instruments/{s}/book|trades|history
    GET  /v1/account           and /positions, /fills
    POST /v1/orders            place; GET, DELETE for the rest of the lifecycle
    POST /v1/keys              issue a credential for the caller's seat
    WS   /v1/stream            ticker, book and trades, plus a seat's own
                               orders and fills once the socket authenticates

Requests there are *signed* rather than labelled: HMAC-SHA256 over the
timestamp, method, path and body together, so a captured signature cannot be
moved onto a different order. See :mod:`arena.api.keys`.

Both surfaces reach one exchange, in one process. That is the point rather than
a simplification: an API served by a second process would be a second market,
and a client that could not see the book the page is showing would be a demo.
An order arriving over ``/v1/orders`` is enqueued onto the same agent, crosses
the same latency link and meets the same collateral check as one clicked in the
browser: there is no privileged lane for the machine.

Single-process and single-market by design. A second viewer sees the *same*
market, which is the useful behaviour when you want the book on one screen and
the blotter on another.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import secrets
import mimetypes
import threading
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from arena.api import rest
from arena.api import stream as api_stream
from arena.api.keys import KeyStore
from arena.exchange.types import AgentId
from arena.portfolio.money import from_money
from dashboard.identity import COOKIE, display_name, sign, verify
from dashboard.operator_auth import is_operator, operator_token, token_was_generated
from dashboard.state import FEE_SCHEDULES, MarketConfig, MarketRunner

# Starlette serves static files with whatever `mimetypes` reports, and on
# Windows `mimetypes` reads the registry, where `.js` is very often mapped to
# `text/plain`. Browsers enforce the MIME type of `<script type="module">`
# strictly and refuse a module served as anything but JavaScript, so the entire
# front end silently did not run: the page rendered its static HTML, no handler
# was bound, and nothing in the server logs said why, because every request had
# answered 200.
#
# Registered here rather than relied upon, so the answer does not depend on the
# machine the server happens to start on.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("image/svg+xml", ".svg")

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"

# How often the kernel is advanced and a snapshot goes out. 20 Hz is past what
# an eye resolves and cheap enough that the simulation never waits on a socket.
TICK_SECONDS = 0.05


@dataclass
class _Seat:
    """Which account a session id is sitting in, and what it calls itself.

    Stamped with the market that issued it, because an account id only means
    anything inside that market. Ids are handed out as `you-1`, `you-2`, ... and
    a rebuild starts counting again, so the *same* id names a different person
    in the next generation. Matching on the id alone, one visitor's remembered
    `you-1` found the `you-1` another visitor had just been given and settled
    into it: two people, one account, from the far side of the fix for two
    people sharing one account.

    It is the same argument the module docstring makes about a cookie from a
    previous run, one level down: a seat from a previous generation names
    something that is gone.
    """

    agent_id: AgentId
    name: str
    generation: int


# Sessions this process has seated, by cookie session id. Not a database: the
# accounts they name live in the running market, so both go away together. The
# chosen name is kept beside the account id because the id does not survive a
# rebuild and the name has to.
_SEATS: dict[str, _Seat] = {}

# Handing out a seat is a read-modify-write on the market's roster, and it is
# not atomic: `LiveMarket.seat` picks the next free id, opens an account under
# it and then registers the agent, and `Venue.open_account` checks for an
# existing account a few statements before it creates one. Two threads through
# that window both pick `you-1`, both pass the check, and the second overwrites
# the first, so two people end up sharing one account, which is the failure
# this whole module exists to prevent, arrived at from the other direction.
#
# It matters because a rebuild makes every open connection re-seat at the same
# instant. One event loop would serialise them (nothing here awaits), but the
# server is reachable from more than one, and "it happens to be single-threaded
# today" is not a property worth resting an account boundary on.
_SEAT_LOCK = threading.Lock()


app = FastAPI(title="Artificial Esports Exchange")
runner = MarketRunner()
_pump: asyncio.Task | None = None


async def _run_market() -> None:
    """Advance the kernel in step with the wall clock, forever."""
    runner.start()
    while True:
        try:
            runner.step()
        except Exception as failure:  # keep serving, and report it
            print(f"market step failed: {failure!r}")
        await asyncio.sleep(TICK_SECONDS)


@app.on_event("startup")
async def _startup() -> None:
    global _pump
    # Printed once, and only when nobody chose one. A default token is the
    # shape of every embarrassing breach, because the deployment that forgot to
    # override it looks exactly like the one that did.
    if token_was_generated():
        print(f"operator token: {operator_token()}")
        print("  set ARENA_OPERATOR_TOKEN to choose your own")
    _pump = asyncio.create_task(_run_market())


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _pump is not None:
        _pump.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _pump


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------


@app.get("/")
async def index(request: Request) -> FileResponse:
    """The page, and a session cookie for whoever asked for it.

    Issued on the first visit rather than behind a sign-in form, so someone who
    opens the exchange can trade immediately with their own account and rename
    themselves afterwards if they want to. The alternative (a wall between a
    visitor and the market) is the wrong default for a place whose capital is
    imaginary.
    """
    response = FileResponse(STATIC / "index.html")
    _ensure_session(request, response)
    return response


def _session_id(request_or_socket: Any) -> str:
    """The signed session id this connection carries, or ``""``."""
    payload = verify(request_or_socket.cookies.get(COOKIE))
    if payload is None:
        return ""
    return str(payload.get("sid", ""))


def _seat_now(sid: str) -> AgentId | None:
    """The account a session id holds *in the market that is running now*.

    Re-seated when the market has been rebuilt underneath it, which is the
    whole reason this is a function rather than a dictionary lookup.
    ``reconfigure`` discards the old :class:`LiveMarket` and every account in
    it, so a remembered id names a trader the new market has never heard of,
    and ``LiveMarket.trader`` answers an id it does not know with the *shared*
    account. Every signed-in visitor therefore collapsed onto one seat the
    moment anybody pressed Rebuild in the Lab: one balance, one blotter, and
    each of them able to cancel the others' orders. That is precisely the
    failure the session cookie exists to prevent, and a page reload did not
    clear it, because the stale entry was still in this table.

    Re-seating keeps the cookie and the chosen name, so from the visitor's side
    a rebuild looks like what it is: a new session in a new market.

    Returns ``None`` for a connection with no valid cookie, and the caller
    falls back to the shared account, which is what every test and every direct
    API user gets, unchanged.
    """
    with _SEAT_LOCK:
        seat = _SEATS.get(sid)
        if seat is None:
            return None
        if seat.generation != runner.generation:
            seat = _Seat(runner.market.seat(seat.name), seat.name, runner.generation)
            _SEATS[sid] = seat
        return seat.agent_id


def _seat_for(request_or_socket: Any) -> AgentId | None:
    """The account this connection is signed in as, if any."""
    return _seat_now(_session_id(request_or_socket))


def _ensure_session(request: Request, response: Response) -> AgentId:
    """Resolve the cookie to an account, seating a new one if needed.

    The cookie carries a session id and a name; the *account* is looked up from
    the session id in this process. So a cookie from a previous run names a
    session this market has never heard of, and the visitor is seated afresh,
    which is right, because the accounts that cookie referred to went away with
    the market that held them.
    """
    payload = verify(request.cookies.get(COOKIE)) or {}
    sid = str(payload.get("sid", ""))
    name = display_name(payload.get("name"))

    seat = _seat_now(sid)
    if seat is None:
        with _SEAT_LOCK:
            sid = secrets.token_urlsafe(12)
            seat = runner.market.seat(name)
            _SEATS[sid] = _Seat(seat, name, runner.generation)
        response.set_cookie(
            COOKIE,
            sign({"sid": sid, "name": name}),
            max_age=7 * 24 * 3600,
            httponly=False,
            samesite="lax",
        )
    return seat


@app.post("/api/me")
async def api_rename(request: Request, response: Response) -> dict[str, Any]:
    """Change the name other traders see. There is nothing else to change."""
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        # A rename with no JSON body is a client's mistake, and answering it
        # with a 500 and a traceback tells whoever wrote that client nothing.
        response.status_code = 400
        return {"ok": False, "error": 'expected a JSON body of {"name": ...}'}
    seat = _ensure_session(request, response)
    name = display_name(str(body.get("name", "")))
    agent = runner.market.traders.get(seat)
    if agent is not None:
        agent.display_name = name
    sid = _session_id(request) or next(
        (k for k, v in _SEATS.items() if v.agent_id == seat), ""
    )
    # Remembered here as well as in the cookie, so a rebuild re-seats this
    # person under the name they chose rather than a fresh random one.
    if sid in _SEATS:
        _SEATS[sid] = _Seat(seat, name, runner.generation)
    response.set_cookie(
        COOKIE,
        sign({"sid": sid, "name": name}),
        max_age=7 * 24 * 3600,
        httponly=False,
        samesite="lax",
    )
    return {"ok": True, "id": str(seat), "name": name}


# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------


@app.get("/api/instruments")
async def api_instruments() -> dict[str, Any]:
    return {"instruments": runner.instruments()}


@app.get("/api/session")
async def api_session() -> dict[str, Any]:
    payload = runner.session_state()
    payload["fee_schedules"] = {
        name: schedule.to_dict() for name, schedule in FEE_SCHEDULES.items()
    }
    return payload


@app.get("/api/agents")
async def api_agents() -> dict[str, Any]:
    return {"agents": runner.agents()}


@app.get("/api/history/{symbol}")
async def api_history(symbol: str) -> JSONResponse:
    series = runner.history.get(symbol)
    if series is None:
        return JSONResponse({"error": f"unknown symbol {symbol}"}, status_code=404)
    return JSONResponse({"symbol": symbol, **series.to_dict()})


@app.get("/api/diagnostics/{symbol}")
async def api_diagnostics(symbol: str) -> JSONResponse:
    report = runner.diagnostics(symbol)
    if "error" in report:
        return JSONResponse(report, status_code=404)
    return JSONResponse(report)


@app.get("/api/book/{symbol}")
async def api_book(symbol: str, levels: int = 20) -> JSONResponse:
    """A deeper ladder than the live socket carries.

    The socket sends eight levels because that is what fits a panel and it goes
    out twenty times a second. A ladder view wants more, and asks for it once.
    """
    venue = runner.market.venue
    instrument = venue.registry.get(symbol)
    if instrument is None:
        return JSONResponse({"error": f"unknown symbol {symbol}"}, status_code=404)
    snapshot = venue.engine(symbol).book.snapshot(max(1, min(60, levels)))
    return JSONResponse(
        {
            "symbol": symbol,
            "bids": [
                [str(instrument.from_ticks(p)), int(q)] for p, q in snapshot.priced_bids
            ],
            "asks": [
                [str(instrument.from_ticks(p)), int(q)] for p, q in snapshot.priced_asks
            ],
            "indicative": runner.indicative(symbol),
            "session": venue.session(symbol).value,
        }
    )


# --------------------------------------------------------------------------
# Control
# --------------------------------------------------------------------------


def _require_operator(request: Request) -> None:
    """Refuse anyone without the operator token.

    These five routes are the ones that reach past the caller and change the
    market for everybody in it, so they are the ones that need a credential a
    visitor does not have. `POST /api/config` in particular discards every
    account, position and working order for every connected user; before this
    guard, any visitor could send it. `kill` was quieter and no better: it
    takes an arbitrary agent id, so one visitor could reach across and disable
    another human's seat while they watched.

    Raised as 404 rather than 401. A 401 confirms the route exists and invites
    guessing at the token; a stranger who cannot operate this venue has no
    business learning its control surface. An operator who has the token sees
    no difference either way.
    """
    if not is_operator(request):
        raise HTTPException(status_code=404, detail="Not Found")


@app.post("/api/config")
async def api_config(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the market. Starts a fresh session, and says so."""
    _require_operator(request)
    return runner.reconfigure(MarketConfig.from_dict(payload or {}))


@app.post("/api/session/{symbol}/halt")
async def api_halt(request: Request, symbol: str) -> dict[str, Any]:
    _require_operator(request)
    return runner.halt(symbol)


@app.post("/api/participant/{agent_id}/kill")
async def api_kill(request: Request, agent_id: str) -> dict[str, Any]:
    """Stop a participant: pull everything it has working, refuse it more."""
    _require_operator(request)
    return runner.kill(agent_id)


@app.post("/api/participant/{agent_id}/revive")
async def api_revive(request: Request, agent_id: str) -> dict[str, Any]:
    _require_operator(request)
    return runner.revive(agent_id)


@app.post("/api/session/{symbol}/uncross")
async def api_uncross(request: Request, symbol: str) -> dict[str, Any]:
    _require_operator(request)
    return runner.uncross(symbol)


# --------------------------------------------------------------------------
# Live
# --------------------------------------------------------------------------


@app.websocket("/ws")
async def stream(socket: WebSocket) -> None:
    await socket.accept()
    # Read before accepting would be tidier, but the cookie is set by the page
    # load that preceded this, so by here it is there or the visitor never
    # loaded the page.
    #
    # The *session id* is what is held, not the account. A rebuild replaces
    # every account in the market, so a connection that captured an account id
    # once would spend the rest of its life reading someone else's: the shared
    # one. Resolving per tick means a socket open across a rebuild follows its
    # own person into the new market.
    sid = _session_id(socket)
    receiver = asyncio.create_task(_receive(socket, sid))
    try:
        while True:
            seat = _seat_now(sid)
            payload = runner.market.snapshot(seat)
            payload["generation"] = runner.generation
            payload["sessions"] = {
                symbol: runner.market.venue.session(symbol).value
                for symbol in runner.market.venue.registry.symbols
            }
            payload["speed"] = runner.market.speed
            await socket.send_json(payload)
            await asyncio.sleep(TICK_SECONDS)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receiver


class _BadOrder(ValueError):
    """Something a person typed, described back to them in their own terms."""


def _whole(value: Any, field: str) -> int:
    """A whole number out of a text box, or a sentence saying why not.

    ``int(...)`` on the raw field answered a blank size box with "int()
    argument must be a string, a bytes-like object or a real number, not
    'NoneType'" and a mistyped one with "invalid literal for int() with base
    10: 'many'". Both are the interpreter talking to itself, and both were
    shown verbatim in a toast on a screen people trade from.
    """
    if isinstance(value, bool) or value is None or value == "":
        raise _BadOrder(f"{field} is required: type a whole number")
    try:
        number = Decimal(str(value).strip())
    except (ArithmeticError, ValueError):
        raise _BadOrder(f"{value!r} is not a {field}: type a whole number") from None
    if number != number.to_integral_value():
        raise _BadOrder(f"{field} must be a whole number, not {value}")
    return int(number)


def _decimal(value: Any, field: str) -> Decimal:
    """A price out of a text box.

    The limit-price box is free text, so what arrives is whatever somebody
    typed, and a price copied off the ladder carries the thousands separator
    the ladder drew it with. ``Decimal("9,233.75")`` raises
    ``InvalidOperation``, whose ``str`` is "[<class
    'decimal.ConversionSyntax'>]": a Python class repr, offered to a stranger
    as advice.

    The comma is *not* stripped and retried. Quietly reinterpreting a number
    somebody typed is how an order ends up resting at a price they did not
    choose, which is the same argument ``Instrument.to_ticks`` makes about
    rounding onto the tick grid.
    """
    try:
        return Decimal(str(value).strip())
    except (ArithmeticError, ValueError):
        raise _BadOrder(
            f"{value!r} is not a {field}: digits and a decimal point only, "
            "with no commas or currency symbols"
        ) from None


def _in_range(price: Decimal, instrument: Any, field: str) -> Decimal:
    """Refuse a price the contract cannot possibly settle at.

    Nothing checked this, and the collateral model cannot: it sizes the worst
    case from the settlement range, so a bid *below* the floor looks safer than
    one inside it and passes every test the venue applies. A limit buy at -100
    on a contract bounded at 0 was accepted, rested, and was eventually filled,
    handing the account a hundred and thirty thousand of profit for having been
    paid to take delivery of something that cannot be worth less than nothing.
    """
    low, high = (
        from_money(bound)
        for bound in runner.market.venue.bounds_in_minor(instrument)
    )
    if not low <= price <= high:
        raise _BadOrder(
            f"{price} is outside {instrument.symbol}'s settlement range "
            f"{low} to {high}. It cannot settle there, so no {field} may rest there"
        )
    return price


def _order_from(message: dict[str, Any]) -> dict[str, Any]:
    """The arguments a submit needs, out of what a browser sent.

    Everything here came from a text box or a select, so it is checked the way
    :meth:`MarketConfig.from_dict` checks configuration: in the terms of the
    control it came from, refusing rather than guessing.
    """
    symbol = str(message.get("symbol") or "")
    instrument = runner.market.venue.registry.get(symbol)
    if instrument is None:
        raise _BadOrder(
            "choose a market before sending an order"
            if not symbol
            else f"unknown symbol {symbol}"
        )

    # Anything that was not exactly "buy" used to become a SELL, silently, so a
    # typo in a client sold instead of bought and nothing anywhere said so.
    side = str(message.get("side", "")).strip().lower()
    if side not in ("buy", "sell"):
        raise _BadOrder(f"side must be buy or sell, not {message.get('side')!r}")

    raw_price = message.get("price")
    price = (
        None
        if raw_price in (None, "", "market")
        else _in_range(_decimal(raw_price, "price"), instrument, "order")
    )
    raw_stop = message.get("stop")
    stop = (
        None
        if raw_stop in (None, "", "none")
        else _in_range(_decimal(raw_stop, "stop trigger"), instrument, "stop")
    )
    raw_display = message.get("display")
    return {
        "symbol": symbol,
        "side": side,
        "quantity": _whole(message.get("quantity"), "size"),
        "price": price,
        "tif": str(message.get("tif", "")),
        "stop": stop,
        "display": 0 if raw_display in (None, "", 0) else _whole(raw_display, "show size"),
    }


async def _receive(socket: WebSocket, sid: str = "") -> None:
    """Handle actions from the browser.

    Orders are queued onto the human agent rather than applied directly, so a
    person's click enters the same event queue as an algorithm's decision and is
    subject to the same latency.

    The seat is resolved per message rather than captured once, for the reason
    given in :func:`_seat_now`: a rebuild replaces every account, and a
    connection holding the old id would be trading the shared one.
    """
    while True:
        try:
            message = await socket.receive_json()
        except (WebSocketDisconnect, RuntimeError):
            return
        if not isinstance(message, dict):
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await socket.send_json(
                    {"ack": {"ok": False, "error": "expected a JSON object"}}
                )
            continue

        seat = _seat_now(sid)
        action = message.get("action")
        try:
            if action == "submit":
                result = runner.market.submit(trader=seat, **_order_from(message))
            elif action == "cancel":
                result = runner.market.cancel(
                    _whole(message.get("order_id"), "order number"),
                    trader=seat,
                    symbol=message.get("symbol"),
                )
            elif action == "flatten":
                result = runner.market.flatten(trader=seat)
            elif action == "cancel_all":
                result = runner.market.cancel_all(trader=seat)
            elif action == "speed":
                result = runner.set_speed(float(message["value"]))
            else:
                result = {"ok": False, "error": f"unknown action {action!r}"}
        except _BadOrder as bad:
            result = {"ok": False, "error": str(bad)}
        except (KeyError, ValueError, TypeError, InvalidOperation) as bad:
            result = {"ok": False, "error": str(bad)}

        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await socket.send_json({"ack": result})


class _FreshStatic(StaticFiles):
    """Static files that a reload actually re-fetches.

    Browsers cache ES modules hard, and the default headers here let them: a
    change to a view module simply did not appear until someone knew to force a
    reload, which is a miserable way to develop and an easy way to debug the
    wrong version of a file for twenty minutes.

    This is a single-process simulator served over localhost, so revalidating
    every asset costs nothing worth measuring.
    """

    def is_not_modified(self, response_headers, request_headers) -> bool:  # noqa: D102
        return False

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response


@app.middleware("http")
async def _stamp_simulated(request: Request, call_next):
    """Put `simulated: true` on every JSON response.

    Every sandbox surveyed separates itself from production by *hostname* and
    nothing else: `apisb.etrade.com`, `api-fxpractice.oanda.com`,
    `testnet.binance.vision`, `sandbox.tradier.com`. Kraken stated the design
    intent outright: "the only difference... is that the base URL is not
    futures.kraken.com but instead demo-futures.kraken.com."

    The failure mode that convention leaves open is the one worth guarding:
    a misconfigured base URL is **undetectable from inside the client**. OANDA
    ships an Account object with no environment field at all, so a captured
    payload is indistinguishable between practice and live. Interactive
    Brokers reduces the safeguard to a plea in its own documentation: "make
    sure your client application is connecting to the right TWS!"

    Deribit is the only venue in the survey that solves it properly, by
    stamping a `testnet` boolean on every JSON-RPC envelope, so a client can
    assert its environment from any response without trusting its own config.
    This is that, and it costs one header's worth of work.

    Also emitted as a response header, so a client can check it without
    parsing a body, including on an error, which is exactly when a confused
    client most needs to know which venue answered.
    """
    response = await call_next(request)
    response.headers["arena-simulated"] = "true"
    return response


if STATIC.is_dir():
    app.mount("/static", _FreshStatic(directory=STATIC), name="static")


# --------------------------------------------------------------------------
# The programmatic API
# --------------------------------------------------------------------------
#
# Mounted here rather than served by a second process, because a systematic
# trader and a person at the browser have to reach the *same* exchange. Two
# processes would be two markets, and a client that could not see the book the
# page is showing would be a demo rather than an API.
#
# The two facts `arena.api` cannot work out for itself are both about identity:
# how to recognise one of this application's browser sessions, and where that
# session is sitting in the market running right now. Both are answered from
# the same helpers the page uses, so a key and the cookie that minted it share
# one account, including across a rebuild, which discards every account and is
# the point at which a naive binding would silently fall back to the shared
# seat.
api_keys = KeyStore()

rest.configure(
    keys=api_keys,
    runner=runner,
    browser_seat=lambda request: (
        rest.Seat(sid, _SEATS[sid].name)
        if (sid := _session_id(request)) in _SEATS
        else None
    ),
    seat_now=_seat_now,
)
app.include_router(rest.router)

# The same key store and the same seat resolver, deliberately. Two stores would
# mean the streaming half refuses credentials the REST half issued; two
# resolvers would put one credential on two different accounts, so a client
# would place orders on one and watch the other's fills. That is
# indistinguishable from a broken feed and would be blamed on the feed.
api_stream.configure(keys=api_keys, runner=runner, seat_now=_seat_now)
app.add_api_websocket_route("/v1/stream", api_stream.stream_endpoint())


def main() -> None:
    """Serve the terminal.

    The port comes from ``PORT`` when it is set, because supervisors (the
    editor's preview runner among them) assign one and expect the process to
    take it. Hard-coding 8000 meant a second instance simply refused to start
    against whatever was already holding that port, with no way to redirect it
    short of editing the file.

    An explicit ``--port`` still wins over the environment, since someone who
    typed a port meant it.
    """
    import os

    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()

    print(f"Artificial Esports Exchange -> http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
