"""The exchange without a browser: a signed REST surface for programs.

The dashboard is one client of this venue. It is not a privileged one, and this
module exists so that nothing about it has to be. Everything a person can do by
clicking -- read the listings, read a book, place an order, cancel it, read
their own account -- a program can do here, over the same market, through the
same event queue, at the same latency. A systematic trader that had to drive a
WebSocket meant for a page would be trading a different exchange from the one
the page shows.

The shape follows Kalshi and Alpaca, because a client author should not have to
learn a third convention: ``/v1`` prefixed paths, plural collections, HMAC-signed
requests, machine codes on every refusal.

Six decisions are worth stating up front, because each of them was arrived at
by something going wrong rather than by preference.

**Market data is candles, and a candle has three OHLC blocks.**
``GET /v1/instruments/{symbol}/candles`` publishes open/high/low/close for the
trade price, for the bid and for the ask separately, which is Kalshi's
``price``/``yes_bid``/``yes_ask`` shape. On a thin book -- and this venue is
thin by construction -- the last print is a fact about whenever somebody last
crossed the spread, while the quotes are facts about the period; only the quote
candles let a backtester reconstruct what was actually transactable. The
sampled mid path at ``/history`` remains, and remains a chart's input rather
than a program's.

**A key is bound to a seat, never to an account id.** ``runner.reconfigure``
discards the whole market and every account in it, and ``LiveMarket.trader``
answers an id it has never heard of with the *shared* account. A credential that
captured ``you-3`` at issue time would therefore, the first time anybody pressed
Rebuild in the Lab, quietly start trading a communal seat alongside every other
stale credential -- one balance, one blotter, everyone able to cancel everyone
else's orders. That is the same failure the session cookie was fixed for in
``dashboard/server.py::_seat_now``, and it is fixed here the same way: the key
stores a stable seat token, and the account is re-resolved on every single
request. ``KeyStore.clear`` documents the opposite policy -- that a rebuild
should invalidate every key -- and it is right about a key bound to an account
and wrong about one bound to a seat, which is exactly why this module binds to a
seat.

**An order is accepted, not acknowledged.** ``POST /v1/orders`` answers 202,
because at the moment it answers the order is still crossing a 20ms latency link
and the matching engine has not seen it. Answering 200 with an order id would
mean either inventing one or waiting for the round trip -- and waiting would give
an API client a synchronous confirmation that no algorithm in this simulation
gets, which is precisely the exemption that makes a live view stop being a view
of the same system. The exchange's order id appears in ``GET /v1/orders`` when
the acknowledgement arrives, tied back to the caller's ``client_order_id``.

**Prices cross the wire as strings, in both directions.** A JSON number is a
double, and the ledger this venue keeps is exact integers precisely so that
``conservation_check`` can be *exactly* zero rather than nearly zero. A price
sent as ``4700.25`` is refused rather than accepted-and-rounded, for the same
reason ``Instrument.to_ticks`` refuses an off-grid price: silently reinterpreting
a number somebody sent is how an order rests at a price they did not choose.

**Every list has a cap, and says so.** Each list endpoint publishes ``limit``,
``cap`` and ``count`` in its own response, so a client can tell "that is all of
them" from "that is all you asked for" without reading documentation.

**A cancel is idempotent.** See :func:`cancel_order` for the argument.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Any, Callable

from fastapi import APIRouter, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from arena.api.errors import ERRORS, ApiError, error_body
from arena.exchange.session import SessionState
from arena.api.keys import (
    HEADER_KEY,
    HEADER_SIGNATURE,
    HEADER_TIMESTAMP,
    ApiKey,
    KeyStore,
    SignatureError,
)
from arena.exchange.types import OrderType, PegReference, TimeInForce
from arena.portfolio.money import from_money

__all__ = ["router", "configure", "Seat", "signed_path"]


# --------------------------------------------------------------------------
# Limits
#
# Named here rather than inlined at each endpoint, so that the number a client
# reads in a response and the number the server enforces cannot drift apart.
# Every one of them is a cap on a list, and every list endpoint publishes the
# cap it applied.
# --------------------------------------------------------------------------

# Book depth. The socket carries eight levels because that is what fits a
# panel; a ladder client wants more and asks once. Capped because a snapshot
# request for a million levels is a way to make the server do the work of a
# full book copy per call.
BOOK_DEPTH_DEFAULT = 10
BOOK_DEPTH_CAP = 100

# The tape. The engine keeps every print of the session, so an uncapped read of
# a busy symbol is unbounded by construction.
TRADES_DEFAULT = 50
TRADES_CAP = 500

# Instruments and positions are bounded by the listing itself -- twenty-eight
# symbols on the default configuration -- so these caps exist to keep the
# contract uniform rather than to protect anything.
INSTRUMENTS_DEFAULT = 200
INSTRUMENTS_CAP = 1_000
POSITIONS_DEFAULT = 200
POSITIONS_CAP = 1_000

# Working orders. One account can legitimately have hundreds resting; the cap
# is high enough that a normal client never meets it and finite so that a
# malfunctioning one cannot ask for an unbounded response.
ORDERS_DEFAULT = 200
ORDERS_CAP = 1_000

# The blotter. ``HumanAgent.log`` keeps the last 200 private events, so this is
# the whole of what exists rather than a policy of ours.
FILLS_DEFAULT = 50
FILLS_CAP = 200

# Price history. The runner keeps a bounded ring buffer per symbol; the real
# cap is read off that buffer at request time so the two can never disagree.
HISTORY_DEFAULT = 600

# Candles. Only the page size lives here: the retained depth, the period enum
# and the span they buy are all the runner's, read off its rings at request
# time for the same reason the history cap is -- a number restated in two files
# is a number that will disagree with itself.
#
# 240 as the default because it is a screen's worth and a warm-up's worth at
# once: 240 one-second bars is four minutes, which is longer than the entire
# 180-second window this API could answer for before candles existed.
CANDLES_DEFAULT = 240

# Credentials per seat. Unlike the other lists this one has no natural bound --
# a session can mint keys until it gets bored -- so it needs a stated one like
# everything else, and the list is ordered oldest first so a client that reaches
# the cap sees the keys it is most likely to have forgotten about.
KEYS_DEFAULT = 100
KEYS_CAP = 500

# How many client order ids one seat's reconciliation table remembers. Bounded
# because it is keyed by a string the client chooses, and an unbounded map keyed
# by client input is a way to spend the server's memory from outside it.
CLIENT_ORDER_MEMORY = 512

# The touch is read from a two-level snapshot rather than a one-level one.
# Market-on-open interest rests at a sentinel price so that it crosses every
# candidate an auction weighs, and a one-level snapshot of a book that holds any
# can therefore contain nothing but the sentinel -- which published a bid of
# 4,611,686,018,427,387,904 on the dashboard once already.
TOUCH_LEVELS = 2


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Seat:
    """Who is holding the browser, in terms this module can bind a key to.

    ``token`` is the part that matters: a stable identifier for a *person*
    that survives a market rebuild. The dashboard's signed session id is one;
    anything unique and durable would do. It is deliberately not the account
    id, which lasts only as long as the market that issued it, and deliberately
    not the display name, which is not an identity -- two visitors called
    "Ash" are two traders, and binding to the name would merge them into one
    account, which is the bug this whole module is arranged around.
    """

    token: str
    name: str = ""


@dataclass
class _ClientOrder:
    """One order this API accepted, from the client's point of view.

    Kept so that ``client_order_id`` -- the only identifier the client chose --
    can be tied back to the exchange's order id once the acknowledgement has
    travelled back over the latency link.
    """

    client_order_id: str
    symbol: str
    side: str
    quantity: int
    ticks: int | None
    submitted_at: int
    order_id: int | None = None


# The key store keys are issued from and verified against. A module-level
# default so the router works when it is mounted with no configuration at all;
# the application replaces it through :func:`configure`.
_KEYS: KeyStore = KeyStore()

# The running market. Injected, because this module must not decide which
# market it is serving: ``dashboard/server.py`` owns exactly one and mounts this
# router into the app that owns it.
_RUNNER: Any | None = None

# How the caller of a key-management request is identified. Those three
# endpoints are the bootstrap -- they are how a browser session mints the first
# credential -- so they cannot themselves be authenticated by a key.
_BROWSER_SEAT: Callable[[Any], Seat | None] | None = None

# Optionally, how the *application* resolves a seat token to the account it
# holds right now. When the app supplies this -- ``dashboard/server.py`` can,
# from the same table its cookies use -- a key and its owner's browser share one
# account across a rebuild rather than being re-seated into two. Without it this
# module re-seats on its own, which still gives the key its own account and
# still survives a rebuild; it just may not be the same account the cookie lands
# in.
_SEAT_NOW: Callable[[str], Any | None] | None = None

# Seat token -> display name. Never cleared: a name outlives the accounts it
# has held, which is the whole reason ``_seat_now`` re-seats under the
# remembered name rather than a fresh random one.
_SEAT_NAMES: dict[str, str] = {}

# Seat token -> (account id, the generation that issued it). Cleared whenever
# the market underneath changes, because an account id means nothing outside
# the market that minted it.
_SEAT_ACCOUNTS: dict[str, tuple[Any, int]] = {}

# Reconciliation and throttle state, keyed by *seat token* rather than by
# account id. Account ids are reused: a rebuild starts the counter again, so the
# next market's ``you-1`` is a different person from this one's, and a table
# keyed by the id would hand them the previous holder's pending orders and their
# spent message allowance. The token is unique to a person and survives the
# rebuild, which is the same argument ``dashboard/server.py::_Seat`` makes about
# stamping a seat with the generation that issued it.
_CLIENT_ORDERS: dict[str, "OrderedDict[str, _ClientOrder]"] = {}
_MESSAGES: dict[str, deque] = {}

# Seating is a read-modify-write on the market's roster and it is not atomic:
# ``LiveMarket.seat`` picks the next free id, opens an account under it and then
# registers the agent, and ``Venue.open_account`` checks for an existing account
# several statements before it creates one. Two threads through that window both
# pick ``you-1`` and the second overwrites the first, which is two people
# sharing one account -- the failure this module exists to prevent, arrived at
# from the other direction. ``dashboard/server.py`` takes the same lock for the
# same reason; a REST client is if anything more likely to race, because nothing
# about an HTTP request promises to arrive on one event loop.
_LOCK = threading.Lock()


def configure(
    *,
    keys: KeyStore | None = None,
    runner: Any | None = None,
    browser_seat: Callable[[Any], Seat | None] | None = None,
    seat_now: Callable[[str], Any | None] | None = None,
) -> None:
    """Point the router at a market, a key store and a way to know the caller.

    Called once by whichever application mounts :data:`router`. Every argument
    is optional so that a test can replace one piece without restating the
    others. From ``dashboard/server.py`` it reads::

        rest.configure(
            keys=KeyStore(),
            runner=runner,
            browser_seat=lambda request: (
                rest.Seat(sid, _SEATS[sid].name)
                if (sid := _session_id(request)) in _SEATS
                else None
            ),
            seat_now=_seat_now,
        )
        app.include_router(rest.router)

    -- which is the whole integration: the application already knows how to
    recognise one of its browser sessions and where that session is sitting
    right now, and those two facts are the only ones this module cannot work
    out for itself.

    Swapping the runner clears every seat binding, and it has to: a binding is
    an account id, an account id only means something inside the market that
    minted it, and carrying one across would hand the new market's ``you-1`` --
    somebody else entirely -- to the key that used to hold the old one. The
    names are kept, because a name is not an account.
    """
    global _KEYS, _RUNNER, _BROWSER_SEAT, _SEAT_NOW
    if keys is not None:
        _KEYS = keys
    if runner is not None:
        if _RUNNER is not None and runner is not _RUNNER:
            _SEAT_ACCOUNTS.clear()
            _CLIENT_ORDERS.clear()
            _MESSAGES.clear()
        _RUNNER = runner
    if browser_seat is not None:
        _BROWSER_SEAT = browser_seat
    if seat_now is not None:
        _SEAT_NOW = seat_now


def signed_path(path: str, query: str = "") -> str:
    """The path a signature covers, query string included.

    Published as a function so a client and this server cannot disagree about
    whether ``?limit=10`` is part of the signed material. It is: without it a
    captured signature could be moved from one filter onto another, which is
    the reason :func:`arena.api.keys.canonical_request` takes the path verbatim.
    """
    return f"{path}?{query}" if query else path


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


# One refusal this router needs that the catalogue did not have: a duplicate
# client order id, which is a conflict with a resource that already exists
# rather than a request that could not be read.
#
# Registered into ``ERRORS`` rather than raised as a locally minted
# ``ApiError``, and the distinction matters. ``_runner`` above argues the case
# out loud for the opposite decision -- it uses a wrong-shaped code rather than
# put one on the wire that ``errors.py`` does not list, because "a catalogue a
# client cannot look a code up in is the exact drift the catalogue exists to
# prevent". A code raised from here and never registered would be exactly that
# drift. So it is added to the one catalogue, once, with its status beside it,
# which keeps the property the catalogue is for: one code, one status,
# everywhere, discoverable by a client from one table.
#
# ``setdefault`` because this module is imported by tests more than once and a
# re-registration must not be able to change a status that a client has already
# branched on.
ERRORS.setdefault(
    "duplicate_client_order_id",
    ("this client_order_id has already been used by this seat", 409),
)


def _refuse(code: str, message: str = "", **detail: Any) -> ApiError:
    """One refusal, built from the catalogue so a code and its status agree.

    The sentence may be improved per call site -- ``errors.py`` says the codes
    are the contract and the sentences are free to get better -- but the code
    and the HTTP status always come from the catalogue, so a client branching on
    either can never be told two different things about the same failure.
    """
    catalogued, status = ERRORS[code]
    return ApiError(code, message or catalogued, status, detail or None)


class _Answers(APIRoute):
    """Turn every failure on this router into the one documented shape.

    A route class rather than an exception handler on the app, because the app
    is not ours: ``dashboard/server.py`` mounts this router into a FastAPI
    instance that already has its own handlers, and a bare ``HTTPException``
    escaping from here would answer ``{"detail": ...}`` -- the exact shape
    ``errors.py`` exists to abolish. Attaching it to the router means every
    route added to this file later is covered without anybody remembering to.
    """

    def get_route_handler(self) -> Callable:
        original = super().get_route_handler()

        async def answer(request: Request) -> Response:
            try:
                return await original(request)
            except ApiError as refusal:
                return JSONResponse(refusal.body(), status_code=refusal.status)
            except RequestValidationError as malformed:
                # FastAPI's own validation, reported in our vocabulary. A client
                # that gets a 422 full of pydantic internals from one endpoint
                # and a catalogued code from the next cannot write one error
                # path, which is the whole complaint errors.py opens with.
                return JSONResponse(
                    error_body(
                        "invalid_request",
                        ERRORS["invalid_request"][0],
                        fields=[
                            ".".join(str(part) for part in item.get("loc", ()))
                            for item in malformed.errors()
                        ],
                    ),
                    status_code=ERRORS["invalid_request"][1],
                )

        return answer


router = APIRouter(prefix="/v1", tags=["v1"], route_class=_Answers)


# --------------------------------------------------------------------------
# The market this router is serving
# --------------------------------------------------------------------------


def _runner() -> Any:
    """The market this router was configured against.

    Reaching this refusal means the application mounted the router and never
    called :func:`configure`, which is a bug in the server rather than in the
    request -- and ``invalid_request`` is admittedly the wrong shape for that.
    It is used anyway, because the alternative is to put a code on the wire that
    ``errors.py`` does not list, and a catalogue a client cannot look a code up
    in is the exact drift the catalogue exists to prevent. The sentence says
    what actually happened.
    """
    if _RUNNER is None:
        raise _refuse(
            "invalid_request",
            "this exchange is not serving API traffic: no market is configured",
        )
    return _RUNNER


def _market() -> Any:
    return _runner().market


def _venue() -> Any:
    return _runner().market.venue


def _instrument(symbol: str) -> Any:
    """The listing, or a refusal naming the symbol that is not one."""
    instrument = _venue().registry.get(symbol)
    if instrument is None:
        raise _refuse("invalid_symbol", f"no instrument listed as {symbol!r}")
    return instrument


def _clock_ns() -> int:
    """Elapsed simulated nanoseconds, the same way the venue reads them.

    The venue's throttle is a rolling *simulated* second -- see
    ``Venue._rate_limited`` -- so the surfacing of it here has to be measured on
    the same clock, or a market running at forty times speed would be refused by
    one and allowed by the other.
    """
    venue = _venue()
    if venue.sim_clock is not None:
        return int(venue.sim_clock())
    return time.monotonic_ns()


# --------------------------------------------------------------------------
# Reading what a client sent
# --------------------------------------------------------------------------


def _limit(raw: Any, default: int, cap: int, field_name: str) -> int:
    """One page size, bounded, or a refusal naming the parameter.

    Clamped at the top and refused at the bottom, deliberately. Asking for more
    than the cap is a client that does not know the cap and can be served the
    cap; asking for zero or minus one is a client with a bug, and answering it
    with an empty list would hide that bug behind a valid-looking response.
    """
    if raw is None or raw == "":
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        raise _refuse(
            "invalid_request",
            f"{field_name} must be a whole number, not {raw!r}",
        ) from None
    if value < 1:
        raise _refuse("invalid_request", f"{field_name} must be at least 1")
    return min(value, cap)


def _stamp(raw: Any, default: int, field_name: str) -> int:
    """One point on the *simulated* clock, in nanoseconds.

    The same clock ``GET /v1/exchange`` publishes as ``clock`` and
    ``GET /v1/instruments/{symbol}/history`` publishes in ``t``, which is the
    kernel's elapsed simulated nanoseconds and is deliberately not a wall clock.
    A market here runs at a speed the operator sets -- up to fifty times real
    time -- so an hour of this exchange is not an hour of anybody's afternoon,
    and stamping its data with a wall clock would make every candle's width a
    function of how fast the server happened to be turning.

    Negative is refused rather than clamped. Simulated time starts at zero, so a
    negative timestamp is a client that has computed one, and answering it with
    zero would hide the arithmetic that produced it.
    """
    if raw is None or raw == "":
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        raise _refuse(
            "invalid_request",
            f"{field_name} must be a whole number of simulated nanoseconds, "
            f"not {raw!r}",
        ) from None
    if value < 0:
        raise _refuse(
            "invalid_request",
            f"{field_name} must not be negative: this clock starts at zero",
        )
    return value


def _cursor(raw: Any, field_name: str) -> int | None:
    """One monotonic cursor value, or None when the caller sent none.

    Zero is a legitimate cursor -- "everything from the beginning" -- and is
    therefore distinguished from absent rather than folded into it. A client
    reconnecting with a cursor it has never advanced sends 0, and reading that
    as "no cursor" would hand it the newest page instead of the oldest.
    """
    if raw is None or raw == "":
        return None
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        raise _refuse(
            "invalid_request", f"{field_name} must be a whole number, not {raw!r}"
        ) from None
    if value < 0:
        raise _refuse("invalid_request", f"{field_name} must not be negative")
    return value


def _whole(raw: Any, field_name: str, code: str) -> int:
    """A whole number out of a JSON body.

    ``True`` is rejected explicitly. It is an ``int`` in Python, so a client
    sending ``"quantity": true`` would otherwise buy one lot.
    """
    if isinstance(raw, bool) or raw is None or raw == "":
        raise _refuse(code, f"{field_name} is required and must be a whole number")
    if isinstance(raw, float):
        raise _refuse(
            code,
            f"{field_name} must be a whole number; {raw!r} arrived as a JSON "
            "float, which cannot represent one exactly",
        )
    try:
        number = Decimal(str(raw).strip())
    except (ArithmeticError, ValueError):
        raise _refuse(code, f"{field_name} must be a whole number, not {raw!r}") from None
    # ``Decimal`` accepts "nan" and "infinity", and both survive the integrality
    # check below -- infinity is equal to its own integral value -- so without
    # this the next line is ``int(Decimal("Infinity"))``, which raises
    # ``OverflowError`` and answers a client's typo with a 500 and a traceback.
    if not number.is_finite():
        raise _refuse(code, f"{field_name} must be a whole number, not {raw!r}")
    if number != number.to_integral_value():
        raise _refuse(code, f"{field_name} must be a whole number, not {raw!r}")
    return int(number)


def _price(raw: Any, field_name: str) -> Decimal:
    """A price out of a JSON body, exactly as it was sent.

    A JSON number is refused when it carries a fraction, and the refusal is the
    point rather than an inconvenience. ``json.loads('{"price": 4700.10}')``
    yields a binary double that is not 4700.10, and this venue's whole
    accounting argument -- integers everywhere, so ``conservation_check`` is
    exactly zero and not nearly zero -- starts at the wire. Whole numbers are
    accepted because a JSON integer is exact, so a client quoting a round price
    does not have to quote it in quotes.
    """
    if isinstance(raw, bool):
        raise _refuse("invalid_price", f"{field_name} must be a price, not a boolean")
    if isinstance(raw, float):
        raise _refuse(
            "invalid_price",
            f"send {field_name} as a string: {raw!r} arrived as a JSON number, "
            "which is a binary double and cannot hold a decimal price exactly",
        )
    try:
        value = Decimal(str(raw).strip())
    except (ArithmeticError, ValueError):
        raise _refuse(
            "invalid_price",
            f"{raw!r} is not a {field_name} -- digits and a decimal point only, "
            "with no commas or currency symbols",
        ) from None
    # ``Decimal`` parses "nan" and "infinity" happily, and every arithmetic
    # check downstream then either propagates the NaN -- which compares false
    # against everything and so passes a range test by looking like neither too
    # high nor too low -- or raises out of the modulo in ``on_grid``. A price
    # that is not a number is refused as one rather than allowed to become a
    # 500 several frames later.
    if not value.is_finite():
        raise _refuse("invalid_price", f"{raw!r} is not a {field_name}")
    return value


def _quotable(instrument: Any, price: Decimal, field_name: str) -> Decimal:
    """Refuse a price this contract cannot rest at, before it costs a round trip.

    Both checks are the venue's own listing rules, applied here so the client
    learns which rule it broke instead of watching an order vanish. The venue
    refuses an off-grid or out-of-range price asynchronously, as a
    ``RejectReason`` in a blotter the client would have to poll for, and
    ``dashboard/server.py`` records what the range check is worth: a limit buy
    at -100 on a contract bounded at zero was accepted, rested and filled,
    handing the account a hundred and thirty thousand of profit for being paid
    to take delivery of something that cannot be worth less than nothing.
    """
    if not instrument.on_grid(price):
        raise _refuse(
            "invalid_price",
            f"{price} is not on {instrument.symbol}'s "
            f"{instrument.increment_at(price)} increment at that level",
        )
    low, high = (from_money(bound) for bound in _venue().bounds_in_minor(instrument))
    if not low <= price <= high:
        raise _refuse(
            "invalid_price",
            f"{price} is outside {instrument.symbol}'s settlement range {low} to "
            f"{high} -- it cannot settle there, so no {field_name} may rest there",
        )
    return price


async def _body(request: Request) -> tuple[bytes, dict[str, Any]]:
    """The raw bytes a signature covers, and the object they decode to.

    Both, because they are two different things and the signature is over the
    first. Re-serialising a parsed body to check a signature is how a client and
    a server end up disagreeing about a space after a colon.
    """
    raw = await request.body()
    if not raw:
        return raw, {}
    try:
        payload = json.loads(raw)
    except ValueError:
        raise _refuse("invalid_request", "the body is not valid JSON") from None
    if not isinstance(payload, dict):
        raise _refuse("invalid_request", "the body must be a JSON object")
    return raw, payload


# --------------------------------------------------------------------------
# Who is calling
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Caller:
    """An authenticated client, and the account it trades *right now*."""

    key: ApiKey | None
    token: str
    name: str
    account: Any


def _seat_name(token: str, fallback: str = "") -> str:
    remembered = _SEAT_NAMES.get(token)
    if remembered:
        return remembered
    name = fallback or token
    _SEAT_NAMES[token] = name
    return name


def _account_for(token: str, fallback_name: str = "") -> Any:
    """The account this seat holds in the market that is running now.

    The whole of the fix, and the reason it is a function rather than a
    dictionary lookup. Three conditions send it back to ``LiveMarket.seat``:

    * nothing is bound yet, which is a key's first request;
    * the generation moved, which is ``reconfigure`` having built a new market;
    * the bound id is not in the venue's account table, which catches a rebuild
      that arrived by any other route. That last check is the one that matters,
      because it tests the actual failure -- ``LiveMarket.trader`` answers an
      unknown id with the shared account -- rather than a bookkeeping proxy for
      it.

    Re-seating rather than refusing keeps the credential working across a
    rebuild, under the name its owner chose, which from the client's side looks
    like what it is: a new session in a new market.
    """
    runner = _runner()
    name = _seat_name(token, fallback_name)
    with _LOCK:
        venue = runner.market.venue
        generation = int(getattr(runner, "generation", 0))

        if _SEAT_NOW is not None:
            # The application knows where its own browser sessions are sitting.
            # Preferred when it does, so that a key and the cookie that minted
            # it stay in one account through a rebuild instead of being seated
            # separately into two.
            delegated = _SEAT_NOW(token)
            if delegated is not None and delegated in venue.accounts:
                _SEAT_ACCOUNTS[token] = (delegated, generation)
                return delegated

        bound = _SEAT_ACCOUNTS.get(token)
        if (
            bound is None
            or bound[1] != generation
            or bound[0] not in venue.accounts
        ):
            account = runner.market.seat(name)
            _SEAT_ACCOUNTS[token] = (account, generation)
            # Every order this seat had working went away with the market that
            # held it, so the records waiting to be reconciled against them are
            # void. Left in place they would attach a client order id from the
            # old market to whichever order in the new one happened to agree on
            # symbol, side and size.
            _CLIENT_ORDERS.pop(token, None)
            return account
        return bound[0]


def _browser(request: Request) -> Seat:
    """The browser session behind a key-management call.

    These three endpoints cannot be authenticated by a key, because they are
    where a key comes from. So they are authenticated the way the page is, and
    the application says how -- this module has no business reading a cookie
    whose signing secret belongs to somebody else.
    """
    if _BROWSER_SEAT is None:
        raise _refuse(
            "auth_required",
            "key management is not available: this server has not said how to "
            "recognise a browser session",
        )
    seat = _BROWSER_SEAT(request)
    if seat is None or not getattr(seat, "token", ""):
        raise _refuse("auth_required", "open the exchange in a browser to issue a key")
    return seat


def _authenticate(request: Request, raw: bytes) -> _Caller:
    """The account behind a signed request, or a refusal.

    The distinction between "you sent no credential" and "your credential did
    not verify" is kept, because they are different bugs in a client and only
    the first one is safe to be specific about. Everything past the presence
    check collapses to one code and one sentence, exactly as
    ``SignatureError`` does and for the same reason: telling a caller holding no
    valid key *which* part failed tells it which key ids exist.
    """
    key_id = request.headers.get(HEADER_KEY, "")
    timestamp = request.headers.get(HEADER_TIMESTAMP, "")
    signature = request.headers.get(HEADER_SIGNATURE, "")
    if not (key_id and timestamp and signature):
        raise _refuse(
            "auth_required",
            "sign this request",
            headers=[HEADER_KEY, HEADER_TIMESTAMP, HEADER_SIGNATURE],
        )
    try:
        key = _KEYS.verify(
            key_id,
            timestamp,
            signature,
            request.method,
            signed_path(request.url.path, request.url.query),
            raw,
        )
    except SignatureError as refused:
        raise _refuse("auth_invalid", str(refused)) from None

    token = key.agent_id
    return _Caller(
        key=key,
        token=token,
        name=_seat_name(token),
        account=_account_for(token),
    )


def _throttle(token: str, reducing: bool = False) -> None:
    """Surface the venue's per-participant message rate as a 429.

    The venue already throttles -- ``Venue.message_rate``, a rolling simulated
    second, ``RejectReason.RATE_LIMITED`` -- but it does it on the far side of a
    latency link, so a client that overruns it gets 202 for every order and then
    silence. The refusals land in a blotter it has to poll for and correlate,
    which is an obscure way to learn something a status code says in one number.

    Two properties are copied from the venue rather than invented, so the two
    cannot disagree about what is allowed:

    * when the venue has no rate configured, neither has this;
    * a cancel is counted but never refused. A participant that cannot withdraw
      is a participant holding exposure nobody is permitted to manage, which is
      the argument ``_rate_limited(reducing=True)`` makes.
    """
    rate = _venue().message_rate
    if rate is None:
        return
    now = _clock_ns()
    window = _MESSAGES.setdefault(token, deque())
    cutoff = now - 1_000_000_000
    while window and window[0] < cutoff:
        window.popleft()
    if len(window) >= rate and not reducing:
        # Not appended, for the venue's reason: a refusal that extended the
        # window would let a client keep its own lockout alive by retrying.
        raise _refuse(
            "rate_limited",
            f"this seat may send {rate} messages a second",
            limit=rate,
            per_seconds=1,
            retry_after_ns=max(0, window[0] + 1_000_000_000 - now),
        )
    window.append(now)


# --------------------------------------------------------------------------
# Reconciliation
# --------------------------------------------------------------------------


def _client_orders(token: str) -> "OrderedDict[str, _ClientOrder]":
    return _CLIENT_ORDERS.setdefault(token, OrderedDict())


def _remember(token: str, record: _ClientOrder) -> None:
    table = _client_orders(token)
    table[record.client_order_id] = record
    while len(table) > CLIENT_ORDER_MEMORY:
        table.popitem(last=False)


def _reconcile(token: str, acknowledged: list[dict[str, Any]]) -> None:
    """Tie each acknowledged order back to the ``client_order_id`` that placed it.

    The client chooses one identifier and the exchange chooses another, and
    nothing carries the first into the engine -- ``Submit`` has no client tag,
    by design, because the engine holds no reference to anything above it. So
    the join is made here, on the four facts both sides know: symbol, side,
    quantity and the price that was actually accepted.

    Matching on those is exact where it needs to be and harmless where it is
    not. Two orders that agree on all four are indistinguishable to the
    exchange and to the client, so attributing either to either is correct.
    Price is a preference rather than a requirement because an order beyond the
    price band slides to the band's edge and rests somewhere its sender did not
    name; falling back to the oldest unclaimed candidate keeps that order
    reconcilable instead of leaving it orphaned.

    A record that never finds an acknowledgement was refused by the venue or is
    still in flight, and those are distinguishable: a refusal shows up in
    ``GET /v1/account/fills`` under ``rejections``.
    """
    table = _client_orders(token)
    claimed = {record.order_id for record in table.values() if record.order_id is not None}
    pending = [record for record in table.values() if record.order_id is None]
    if not pending:
        return
    for order in acknowledged:
        if order["order_id"] in claimed:
            continue
        candidates = [
            record
            for record in pending
            if record.symbol == order["symbol"]
            and record.side == order["side"]
            and record.quantity == order["quantity"]
        ]
        if not candidates:
            continue
        exact = [record for record in candidates if record.ticks == order["ticks"]]
        chosen = (exact or candidates)[0]
        chosen.order_id = order["order_id"]
        claimed.add(order["order_id"])
        pending.remove(chosen)


def _client_id_for(token: str, symbol: str, order_id: int) -> str | None:
    for record in _client_orders(token).values():
        if record.order_id == order_id and record.symbol == symbol:
            return record.client_order_id
    return None


def _client_order_row(caller: "_Caller", record: _ClientOrder) -> dict[str, Any]:
    """What became of one order, addressed by the only id the client chose.

    Three states, and they are the three questions a client that has lost track
    of a POST actually has.

    ``pending`` -- this API accepted it and no acknowledgement has come back
    yet. Either it is still crossing the latency link, or the venue refused it
    on the far side; those are distinguishable, and the refusal is in
    ``GET /v1/account/fills`` under ``rejections``. No timeout is applied here
    for the reason ``working_orders`` gives: how long a round trip takes is a
    property of this seat's link and of how fast the market is being run, and a
    guessed timeout that fires early reports a failure that has not happened.

    ``working`` -- it rested, and ``order`` carries the live row.

    ``done`` -- the exchange assigned it an id and it is no longer in the book:
    filled outright, filled through, or cancelled. A market order is typically
    this from the first moment a client can ask.
    """
    listing = _venue().registry.get(record.symbol)
    status = "pending"
    order_row: dict[str, Any] | None = None
    if record.order_id is not None:
        status = "done"
        who = _market().trader(caller.account)
        if (record.symbol, record.order_id) in who.live_orders:
            order = _venue().engine(record.symbol).book.get(record.order_id)
            if order is not None:
                status = "working"
                order_row = _order_row(record.symbol, order)
    return {
        "client_order_id": record.client_order_id,
        "account_id": str(caller.account),
        "symbol": record.symbol,
        "side": record.side,
        "quantity": record.quantity,
        # The price as sent, in contract units, with the engine's own integer
        # beside it under a name that says which is which -- the same pairing
        # ``_order_row`` publishes and for the same reason.
        "price": (
            None
            if record.ticks is None or listing is None
            else str(listing.from_ticks(record.ticks))
        ),
        "ticks": record.ticks,
        "submitted_at": record.submitted_at,
        "status": status,
        "order_id": record.order_id,
        "order": order_row,
    }


def _acknowledged(caller: "_Caller") -> list[dict[str, Any]]:
    """Every order the exchange has acknowledged to this account, from its blotter.

    The blotter rather than the working orders, because an order that filled the
    instant it arrived was acknowledged and never rested -- a market order
    always, an immediate-or-cancel usually -- and matching only against what is
    resting leaves exactly those unreconcilable. They are the ones a client most
    wants named, since a fill is the event it has to book.

    Bounded by ``HumanAgent.log``'s own 200 entries: an account that generated
    two hundred private events between two polls loses the earliest
    acknowledgements from this join, and the orders they named stay pending
    until their records are evicted.
    """
    market = _market()
    venue = market.venue
    who = market.trader(caller.account)
    seen = []
    for entry in who.log:
        if entry.get("type") != "ack":
            continue
        symbol = str(entry.get("symbol", ""))
        instrument = venue.registry.get(symbol)
        if instrument is None:
            continue
        # The blotter converts an acknowledged price to contract units before
        # storing it, so it comes back a string and goes back to ticks exactly.
        price = entry.get("price")
        seen.append(
            {
                "order_id": int(entry.get("order_id", -1)),
                "symbol": symbol,
                "side": entry.get("side"),
                "quantity": int(entry.get("quantity", 0)),
                "ticks": None
                if price is None
                else int(instrument.to_ticks(Decimal(str(price)))),
            }
        )
    return seen


def _reconciled(caller: "_Caller") -> list[dict[str, Any]]:
    """This account's working orders, with the client's own ids bound first.

    Every authenticated read that can name an order goes through here, not just
    the list endpoint. Binding the ids only where the list is built made the
    answer depend on the order a client happened to call things in: a client
    that fetched one order by id, or read its fills, before ever listing its
    working orders was told ``client_order_id: null`` for orders this API could
    perfectly well identify.
    """
    _reconcile(caller.token, _acknowledged(caller))
    return _working_orders(caller)


# --------------------------------------------------------------------------
# Shapes
#
# One function per thing a client reads, so the same instrument means the same
# JSON on the list endpoint, the detail endpoint and nowhere a third way.
# --------------------------------------------------------------------------


def _subjects(instrument: Any) -> list[str]:
    """What this contract is written on, as plain names.

    Every class answers, because every contract is an underlying and a payoff
    and the underlying knows its atoms. A future on one competitor names one; a
    spread names the two it is a difference of; an index names its whole
    basket. Nothing here branches on the class, which is what makes the
    ``subject`` filter mean the same thing for all nine of them.
    """
    return sorted({atom.subject for atom in instrument.spec.underlying.atoms()})


def _touch(instrument: Any, book: Any) -> dict[str, Any]:
    snapshot = book.snapshot(TOUCH_LEVELS)
    bid = snapshot.priced_bids[0] if snapshot.priced_bids else None
    ask = snapshot.priced_asks[0] if snapshot.priced_asks else None
    return {
        "bid": None if bid is None else str(instrument.from_ticks(bid[0])),
        "bid_size": None if bid is None else int(bid[1]),
        "ask": None if ask is None else str(instrument.from_ticks(ask[0])),
        "ask_size": None if ask is None else int(ask[1]),
    }


def _instrument_row(symbol: str) -> dict[str, Any]:
    venue = _venue()
    instrument = venue.registry.require(symbol)
    engine = venue.engine(symbol)
    low, high = instrument.value_bounds
    settle_low, settle_high = instrument.settlement_bounds
    return {
        "symbol": symbol,
        "class": instrument.instrument_class,
        "subjects": _subjects(instrument),
        "tick": str(instrument.tick_size),
        "lot": instrument.lot_size,
        # The value bounds, not the settlement bounds. What an order may rest at
        # is what the claim can still be worth, payments included -- a share
        # that has paid part of itself out is worth less by exactly that, and
        # quoting the un-narrowed range would advertise prices the venue refuses.
        "bounds": [str(low), str(high)],
        "settlement_bounds": [str(settle_low), str(settle_high)],
        "expiry": instrument.expiry.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "session": venue.session(symbol).value,
        "mark": str(from_money(venue.mark(symbol))),
        "trades": len(engine.tape),
        "contract_id": instrument.spec.contract_id,
        "spec_digest": instrument.spec.spec_digest,
        **_touch(instrument, engine.book),
    }


def _candle_row(instrument: Any, candle: Any) -> dict[str, Any]:
    """One closed period, as three OHLC blocks and the volume behind them.

    The three blocks are the design, and they are copied from Kalshi rather than
    invented: their candlestick carries ``price``, ``yes_bid`` and ``yes_ask``
    as separate open/high/low/close structures, and the reason generalises past
    binary contracts. On a thin book the last trade is close to meaningless --
    it is a fact about whenever somebody last crossed the spread, which on nine
    of this venue's forty-seven contracts was more than a simulated second ago
    -- while the bid and the ask are facts about the period itself. A backtester
    that wants to know what it could actually have transacted at reads the quote
    candles; the trade candle only tells it what somebody else did.

    ``mean`` is Kalshi's fifth price field and is the volume-weighted average.
    It is the one figure on this payload that is not exact, because it is a
    quotient of two integers and a quotient of two integers is not always a
    decimal -- 100 lots at 3 and 200 at 4 average to 11/3. So ``notional``
    travels beside it, exact and integral, and a client that needs the exactness
    divides it itself and keeps the remainder. Everything else here is an
    integer or an exact decimal string, and none of it is a float.
    """
    tick = instrument.tick_size

    def price(ticks: int | None) -> str | None:
        return None if ticks is None else str(instrument.from_ticks(ticks))

    notional = Decimal(candle.notional) * tick
    if candle.volume > 0:
        # An explicit context, so the digits do not depend on what anything else
        # in this process has done to the global one. Twenty-eight significant
        # digits is Decimal's own default and far past any price this venue
        # lists; the point of pinning it is reproducibility, not range.
        with localcontext() as context:
            context.prec = 28
            mean = str(notional / Decimal(candle.volume))
    else:
        # No print in the period, so the mean of nothing is the last thing that
        # traded -- the same value the open, high, low and close all carry.
        # Consistent with the block rather than null, so a client averaging
        # across bars does not have to special-case the quiet ones.
        mean = price(candle.price_close)
    return {
        "end": int(candle.end),
        "volume": int(candle.volume),
        "trades": int(candle.trades),
        "notional": str(notional),
        "open_interest": int(candle.open_interest),
        "price": {
            "open": price(candle.price_open),
            "high": price(candle.price_high),
            "low": price(candle.price_low),
            "close": price(candle.price_close),
            "mean": mean,
        },
        "bid": {
            "open": price(candle.bid_open),
            "high": price(candle.bid_high),
            "low": price(candle.bid_low),
            "close": price(candle.bid_close),
        },
        "ask": {
            "open": price(candle.ask_open),
            "high": price(candle.ask_high),
            "low": price(candle.ask_low),
            "close": price(candle.ask_close),
        },
    }


def _order_row(symbol: str, order: Any) -> dict[str, Any]:
    """One resting order, in prices, with the engine's own integer beside it.

    ``ticks`` is published as well as ``price``, and the name is doing the work:
    everything that has gone wrong in this codebase around ticks went wrong
    because a raw internal unit appeared under a label that promised a price --
    a settlement of 18,677 on a contract marked at 4,663, a halt record reading
    1,989 for 497.25. A field called ``ticks`` promises a tick count, and a
    client reconciling against the exchange's own integers should not have to
    divide a decimal string by a tick size to get them back.
    """
    instrument = _venue().registry.require(symbol)
    return {
        "order_id": int(order.order_id),
        "symbol": symbol,
        "side": order.side.value,
        "price": str(instrument.from_ticks(order.price)),
        "ticks": int(order.price),
        "quantity": int(order.quantity),
        "remaining": int(order.remaining),
        "filled": int(order.filled),
        "display": int(order.display_size),
        "shown": int(order.shown),
        "post_only": bool(order.post_only),
        "status": order.status.value,
    }


def _working_orders(caller: _Caller) -> list[dict[str, Any]]:
    """This account's resting orders, as the account itself believes them.

    Read from the agent's own record rather than by scanning every book,
    because the agent's record is the set the account can actually act on:
    ``LiveMarket.cancel`` refuses an order that is not in it, so listing from
    the engine would publish orders whose acknowledgement is still crossing the
    return leg and which ``DELETE`` would answer "no such live order" for. A
    list that disagrees with the actions available on it is worse than a list
    that lags by one round trip.

    Each entry is then filled in from the engine, which is the authority on
    price, size and how much is left.
    """
    market = _market()
    venue = market.venue
    who = market.trader(caller.account)
    rows = []
    for symbol, order_id in sorted(who.live_orders):
        instrument = venue.registry.get(symbol)
        if instrument is None:
            continue
        order = venue.engine(symbol).book.get(order_id)
        if order is None:
            # Acknowledged to the agent and already gone from the book: the
            # cancellation or the last fill is on its way back. Reported rather
            # than dropped, so the client sees the same thing the account does.
            #
            # Given every key the ordinary row has, filled with nulls, because
            # a list whose members have different shapes makes every client
            # that reads it branch on which one it got -- and the one that
            # forgets to branch fails on the rare row rather than the common
            # one, which is the worst possible distribution of that bug.
            side = who.order_side.get((symbol, order_id))
            rows.append(
                {
                    "order_id": int(order_id),
                    "symbol": symbol,
                    "side": None if side is None else side.value,
                    "price": None,
                    "ticks": None,
                    "quantity": None,
                    "remaining": None,
                    "filled": None,
                    "display": None,
                    "shown": None,
                    "post_only": None,
                    "status": "unknown",
                }
            )
            continue
        rows.append(_order_row(symbol, order))
    return rows


def _position_rows(caller: _Caller) -> list[dict[str, Any]]:
    venue = _venue()
    marks = venue.marks()
    account = venue.account(caller.account)
    rows = []
    for symbol in venue.registry.symbols:
        position = account.positions.get(symbol)
        if position is None or (position.quantity == 0 and position.volume == 0):
            continue
        row = position.to_dict(marks.get(symbol))
        row["class"] = venue.registry.require(symbol).instrument_class
        rows.append(row)
    return rows


def _paged(rows: list, limit: int, cap: int, key: str) -> dict[str, Any]:
    """One list, with the numbers a client needs to know it has all of them.

    ``count`` against ``total`` is the difference between "that is everything"
    and "that is everything you asked for", and a client that cannot tell those
    apart either stops early or pages forever.
    """
    return {
        key: rows[:limit],
        "count": len(rows[:limit]),
        "total": len(rows),
        "limit": limit,
        "cap": cap,
    }


# --------------------------------------------------------------------------
# Public: the exchange itself
# --------------------------------------------------------------------------


@router.get("/exchange")
async def exchange() -> dict[str, Any]:
    """The clock, the generation, the invariant, and what is listed.

    ``generation`` is here because it is the one number a long-running client
    must watch: it changes when the market was rebuilt, which means every order
    id, every position and every mark it is holding describe a market that no
    longer exists. A client that ignores it will reconcile forever against
    nothing.

    ``conservation`` is published in the ledger's own minor units as an exact
    integer, not as a price, because the claim being made is that it is exactly
    zero and a rounded zero would be a different and much weaker claim.
    """
    runner = _runner()
    market = runner.market
    venue = market.venue
    classes: dict[str, int] = {}
    for symbol in venue.registry.symbols:
        name = venue.registry.require(symbol).instrument_class
        classes[name] = classes.get(name, 0) + 1
    sessions: dict[str, int] = {}
    for symbol in venue.registry.symbols:
        phase = venue.session(symbol).value
        sessions[phase] = sessions.get(phase, 0) + 1
    return {
        "venue": venue.name,
        "clock": int(market.kernel.now),
        "events": int(market.kernel.processed),
        "generation": int(getattr(runner, "generation", 0)),
        "speed": market.speed,
        "conservation": str(int(venue.conservation_check())),
        "counts": {
            "instruments": len(venue.registry.symbols),
            "classes": dict(sorted(classes.items())),
            "accounts": len(venue.accounts),
            "participants": len(market.agents),
            "seats": len(market.traders),
        },
        "session": {
            "phases": dict(sorted(sessions.items())),
            "fees": venue.fees.to_dict(),
            # In price units, like every other money figure here. The
            # dashboard's own ``/api/session`` publishes the raw minor units
            # and divides by a million in JavaScript, which is fine for one
            # page and is exactly the arrangement that put "113125513.21M" on
            # the participants table: a raw internal unit under a label that
            # promises money, with the conversion living somewhere else.
            "fees_collected": str(from_money(venue.fees_collected)),
            "price_band": venue.price_band,
            "message_rate": venue.message_rate,
            # Symbols not matching continuously, asked as a question about the
            # phase rather than about a name. A halt does not put a symbol into
            # a state called "halted" -- it puts it into an auction, which is
            # also where the opening call and the closing call live -- so a
            # client watching for the string would watch forever. What it
            # actually needs to know is which books will not trade its order
            # right now, and every one of those phases answers that the same way.
            "not_trading": sorted(
                symbol
                for symbol in venue.registry.symbols
                if not venue.session(symbol).matches_continuously
            ),
            "halted_participants": sorted(str(a) for a in venue.halted_participants),
            "halts": len(venue.halts),
        },
    }


@router.get("/instruments")
async def instruments(
    request: Request,
    limit: str | None = None,
) -> dict[str, Any]:
    """Everything listed, filtered by class or by subject.

    Both filters are derived rather than declared -- the class comes from the
    payoff and the underlying, the subjects from the underlying's atoms -- so
    they work identically for all nine classes the venue lists and there is
    nothing here that knows what a future or an option is.

    Filters arrive as query parameters and are matched case-insensitively;
    ``class`` is spelled out as a query parameter rather than an argument
    because it is a Python keyword.
    """
    params = request.query_params
    wanted_class = (params.get("class") or "").strip().lower()
    wanted_subject = (params.get("subject") or "").strip().lower()
    page = _limit(limit, INSTRUMENTS_DEFAULT, INSTRUMENTS_CAP, "limit")

    rows = []
    for symbol in _venue().registry.symbols:
        row = _instrument_row(symbol)
        if wanted_class and row["class"] != wanted_class:
            continue
        if wanted_subject and wanted_subject not in {
            subject.lower() for subject in row["subjects"]
        }:
            continue
        rows.append(row)
    payload = _paged(rows, page, INSTRUMENTS_CAP, "instruments")
    payload["filters"] = {"class": wanted_class or None, "subject": wanted_subject or None}
    return payload


@router.get("/instruments/{symbol}")
async def instrument(symbol: str) -> dict[str, Any]:
    """One listing, with the terms it settles by.

    The contract is published as well as the price, because a market where you
    cannot read the contract is a casino with extra steps.
    """
    listing = _instrument(symbol)
    venue = _venue()
    row = _instrument_row(symbol)
    schedule = listing.spec.distribution
    # The contract's own parameters -- a strike, a scale, a threshold -- are
    # published exactly as the contract layer holds them, which is as floats.
    # They are terms of the claim rather than money in the ledger, they feed
    # the spec digest in that form, and the WebSocket already publishes the
    # same objects. Stringifying them here would advertise an exactness the
    # contract does not have and would make the same contract read two
    # different ways on two surfaces of one exchange.
    row["contract"] = {
        "payoff": listing.spec.payoff.to_dict(),
        "underlying": listing.spec.underlying.to_dict(),
        "window": {
            "start": listing.spec.window.start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": listing.spec.window.end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "reference_id": listing.spec.reference_id,
        "tick_table": [[str(a), str(b)] for a, b in listing.spec.tick_table],
        # Only a share has one, and for a share it is the thing being bought:
        # what a holder is owed before settlement, and what a short can be
        # asked for.
        "distribution": None
        if schedule is None
        else {
            "periods": len(schedule.windows),
            "payoff": schedule.payoff.to_dict(),
            "first": schedule.windows[0].end.strftime("%Y-%m-%d"),
            "last": schedule.windows[-1].end.strftime("%Y-%m-%d"),
        },
    }
    # Only during a call phase. ``Venue.indicative`` computes what an auction
    # would clear at whenever it is asked, which is right for the venue and
    # wrong to publish: a continuously trading symbol has no auction, and a
    # client reading an "indicative price" beside a live book would take it for
    # a second opinion on where the market is rather than for the answer to a
    # question nobody asked.
    indicative = (
        None
        if venue.session(symbol) is SessionState.CONTINUOUS
        else venue.indicative(symbol)
    )
    row["indicative"] = (
        None
        if indicative is None or indicative.volume <= 0
        else {
            "price": str(listing.from_ticks(indicative.price)),
            "quantity": int(indicative.volume),
        }
    )
    return row


@router.get("/instruments/{symbol}/book")
async def book(symbol: str, depth: str | None = None) -> dict[str, Any]:
    """Aggregated depth, both sides, priced levels only.

    Market-on-open interest rests at a sentinel so that it crosses every
    candidate an auction weighs. It is real interest and it is not a price, so
    it is summarised as ``market_on_open`` rather than drawn as a level --
    publishing it as one put a bid of 4,611,686,018,427,387,904 on a screen.
    """
    listing = _instrument(symbol)
    venue = _venue()
    levels = _limit(depth, BOOK_DEPTH_DEFAULT, BOOK_DEPTH_CAP, "depth")
    snapshot = venue.engine(symbol).book.snapshot(levels)
    priced_bids = snapshot.priced_bids[:levels]
    priced_asks = snapshot.priced_asks[:levels]
    return {
        "symbol": symbol,
        "session": venue.session(symbol).value,
        "depth": levels,
        "cap": BOOK_DEPTH_CAP,
        "bids": [
            [str(listing.from_ticks(price)), int(quantity)]
            for price, quantity in priced_bids
        ],
        "asks": [
            [str(listing.from_ticks(price)), int(quantity)]
            for price, quantity in priced_asks
        ],
        "market_on_open": {
            "bids": sum(
                int(q) for p, q in snapshot.bids if (p, q) not in priced_bids
            ),
            "asks": sum(
                int(q) for p, q in snapshot.asks if (p, q) not in priced_asks
            ),
        },
        # A crossed book has no spread, and during a call phase it is meant to
        # be crossed: orders accumulate without matching until the uncross.
        # Reporting the arithmetic difference printed "-10,000.00" on a screen
        # once, which is not a spread and not a number anyone should interpret.
        "spread": (
            str(listing.from_ticks(snapshot.spread))
            if snapshot.spread is not None and snapshot.spread >= 0
            else None
        ),
        "mark": str(from_money(venue.mark(symbol))),
    }


@router.get("/instruments/{symbol}/trades")
async def trades(symbol: str, limit: str | None = None) -> dict[str, Any]:
    """The tape, most recent first.

    Ordered by the engine's sequence number rather than by a timestamp, because
    that is what the print actually carries: two trades in the same microsecond
    still have a defined order, and a wall clock cannot promise that. The
    sequence is per symbol, which is the only place it is comparable.
    """
    listing = _instrument(symbol)
    page = _limit(limit, TRADES_DEFAULT, TRADES_CAP, "limit")
    tape = _venue().engine(symbol).tape
    rows = [
        {
            "sequence": int(trade.sequence),
            "price": str(listing.from_ticks(trade.price)),
            "quantity": int(trade.quantity),
            "aggressor_side": trade.aggressor_side.value,
        }
        for trade in reversed(tape[-page:])
    ]
    payload = _paged(rows, page, TRADES_CAP, "trades")
    payload["symbol"] = symbol
    payload["total"] = len(tape)
    return payload


@router.get("/instruments/{symbol}/history")
async def history(symbol: str, limit: str | None = None) -> dict[str, Any]:
    """The sampled mid path, for anything that wants a series rather than a book.

    The runner keeps this as floats, because it exists to be charted and a
    chart is a float. It is published as strings anyway, and the conversion is
    exact rather than hopeful: every stored value is a whole number of ticks
    times the tick size, so it round-trips through ``Decimal(str(v))`` without
    inventing digits. A price leaves this API as a string even when the thing
    upstream of it was not one, because the alternative is a client that has to
    know which endpoints are exact.

    The cap is the sampling buffer's own length, read off the buffer, so raising
    ``MarketRunner.HISTORY`` cannot leave a stale number in this file.
    """
    _instrument(symbol)
    series = _runner().history.get(symbol)
    if series is None:
        raise _refuse("not_found", f"no history recorded for {symbol}")
    cap = series.stamps.maxlen or HISTORY_DEFAULT
    page = _limit(limit, min(HISTORY_DEFAULT, cap), cap, "limit")
    stamps = list(series.stamps)[-page:]
    mids = list(series.mids)[-page:]
    return {
        "symbol": symbol,
        "t": [int(stamp) for stamp in stamps],
        # NaN is the sample of a book with no two-sided quote. JSON has no NaN,
        # and emitting one anyway produces a document most parsers reject.
        "mid": [None if mid != mid else str(Decimal(str(mid))) for mid in mids],
        "count": len(stamps),
        "total": len(series.stamps),
        "limit": page,
        "cap": cap,
    }


def _candle_ring(symbol: str, raw_period: str) -> tuple[Any, list[int]]:
    """The ring for one symbol at one period, or a refusal naming the enum.

    The enum is read off the runner rather than declared here. This module is
    mounted by an application it must not import, and the periods are that
    application's -- so the values a client is refused against are the values
    that are actually being aggregated, and there is no second copy to go stale
    the way the sampling comment in ``dashboard/state.py`` did.
    """
    series = _runner().history.get(symbol)
    periods = list(getattr(series, "periods", ()) or ())
    if series is None or not periods:
        raise _refuse("not_found", f"no candles recorded for {symbol}")
    if not raw_period:
        raise _refuse(
            "invalid_request",
            "name the candle period, as ?period=<seconds>",
            supported=periods,
            unit="seconds",
        )
    try:
        period = int(raw_period)
    except (TypeError, ValueError):
        period = 0
    ring = series.ring(period) if period else None
    if ring is None:
        # A closed enum, refused rather than rounded to the nearest kept
        # period. Kalshi does the same and it is the right way round: a series
        # of bars that are not the width the client asked for is wrong in a way
        # nothing downstream can detect, because every bar still looks like a
        # bar.
        raise _refuse(
            "invalid_request",
            f"{raw_period!r} is not a candle period this venue keeps",
            supported=periods,
            unit="seconds",
        )
    return ring, periods


@router.get("/instruments/{symbol}/candles")
async def candles(request: Request, symbol: str) -> dict[str, Any]:
    """Closed candles: trade price, bid and ask, gap-free, in the venue's clock.

    ``?period=&start=&end=&limit=``. ``period`` is required and is a closed
    enum in seconds of simulated time; ``start`` and ``end`` are simulated
    nanoseconds, the same clock ``/v1/exchange`` publishes as ``clock``.

    Four decisions, each one measured against a venue that already ships it.

    **Three OHLC blocks, not one.** ``price``, ``bid`` and ``ask``, which is
    Kalshi's ``price``/``yes_bid``/``yes_ask`` shape. See :func:`_candle_row`
    for why it is the important one: this book is thin by construction and the
    last print alone cannot tell a client what was quotable.

    **Empty periods are emitted, not skipped.** A period nobody traded in comes
    back with zero volume and the previous close in all five price fields, and
    with its *real* bid and ask candles, because the sampler still saw the book
    ten times a second through it. Kalshi does this and the reason is
    mechanical: a gap-free series joins to a clock by arithmetic, and a sparse
    one has to be reindexed by every client that reads it.

    **The in-progress period is not published.** A partial bar has a high that
    is not the period's high and a close that is not a close. Nothing here
    reports it, so a bar that arrives is final.

    **An over-wide range is refused, not truncated.** This is the one worth
    arguing. Coinbase caps a candle request at 300 and refuses beyond it;
    Binance caps at 1,000 and silently hands back the first 1,000; Kalshi
    refuses with ``requested time range with candlesticks: 129600, max
    candlesticks: 5000``. Refusing is the honest one. A backtester that asked
    for a day and received an hour, with no field in the response saying which
    hour it lost, will compute a statistic over a window it does not have and
    will not find out. So a range wider than the ring retains is refused, and
    the refusal names both numbers.

    ``limit`` is *not* that check and is clamped like every other list here,
    because a limit is a page size rather than a claim about coverage -- the
    response says which one it applied, and ``start``/``end`` come back
    unchanged so a client can see exactly what it asked for.
    """
    listing = _instrument(symbol)
    params = request.query_params
    ring, periods = _candle_ring(symbol, (params.get("period") or "").strip())

    now = int(_market().kernel.now)
    end_ns = _stamp(params.get("end"), now, "end")
    page = _limit(params.get("limit"), min(CANDLES_DEFAULT, ring.depth), ring.depth, "limit")

    raw_start = params.get("start")
    if raw_start is None or raw_start == "":
        # No start given: one page back from the end, so the default request is
        # exactly the page the client asked for and never a refusal.
        start_ns = max(0, end_ns - page * ring.period_ns)
    else:
        start_ns = _stamp(raw_start, 0, "start")
        if start_ns > end_ns:
            raise _refuse(
                "invalid_request",
                f"start {start_ns} is after end {end_ns}",
                start=start_ns,
                end=end_ns,
            )
        # Ceiling, so a range covering exactly the retained depth is allowed and
        # one nanosecond past it is not.
        wanted = max(1, -(-(end_ns - start_ns) // ring.period_ns))
        if wanted > ring.depth:
            raise _refuse(
                "invalid_request",
                f"that range asks for {wanted} candlesticks of {ring.period}s and "
                f"this venue retains {ring.depth} per period -- narrow the range "
                f"or ask for a longer period",
                requested=wanted,
                cap=ring.depth,
                period=ring.period,
                retains_ns=ring.retains_ns,
            )

    rows = ring.window(start_ns, end_ns, page)
    held = ring.span()
    return {
        "symbol": symbol,
        "period": ring.period,
        "period_ns": ring.period_ns,
        # The whole enum, in the body, so a client learns the closed set from a
        # successful call rather than from a refusal or from documentation.
        "periods": periods,
        "start": start_ns,
        "end": end_ns,
        "clock": now,
        "candles": [_candle_row(listing, candle) for candle in rows],
        "count": len(rows),
        "total": len(ring),
        "limit": page,
        "cap": ring.depth,
        # What exists at this period at all, so a client can tell "your window
        # is empty" from "your window is older than anything kept".
        "retains_ns": ring.retains_ns,
        "oldest": None if held is None else held[0],
        "newest": None if held is None else held[1],
    }


# --------------------------------------------------------------------------
# Authenticated: the account
# --------------------------------------------------------------------------


@router.get("/account")
async def account(request: Request) -> dict[str, Any]:
    """Cash, collateral, equity and both halves of PnL, for the calling key.

    Every figure is a string in price units. The ledger underneath is integer
    minor units -- a millionth of a price unit -- and publishing those raw put a
    maker worth 113,125,513.21 on the dashboard as "113125513.21M" once
    already: a raw internal unit under a label that promises a price.
    """
    raw = await request.body()
    caller = _authenticate(request, raw)
    venue = _venue()
    marks = venue.marks()
    holding = venue.account(caller.account)
    return {
        "account_id": str(caller.account),
        "seat": caller.name,
        "generation": int(getattr(_runner(), "generation", 0)),
        "cash": str(from_money(holding.cash)),
        "free_cash": str(from_money(holding.free_cash)),
        "collateral": str(from_money(holding.posted_collateral)),
        "equity": str(from_money(holding.equity(marks))),
        "starting_cash": str(from_money(holding.starting_cash)),
        "realized_pnl": str(from_money(holding.realized_pnl)),
        "unrealized_pnl": str(from_money(holding.unrealized_pnl(marks))),
        "pnl": str(from_money(holding.equity(marks)) - from_money(holding.starting_cash)),
        "halted": caller.account in venue.halted_participants,
    }


@router.get("/account/positions")
async def positions(request: Request, limit: str | None = None) -> dict[str, Any]:
    """Every symbol this account has open risk or a realised result in.

    A closed position is kept rather than dropped: its realised PnL is the
    account's history, and an endpoint that forgot a position the moment it
    went flat would make a day's trading unauditable.
    """
    raw = await request.body()
    caller = _authenticate(request, raw)
    page = _limit(limit, POSITIONS_DEFAULT, POSITIONS_CAP, "limit")
    payload = _paged(_position_rows(caller), page, POSITIONS_CAP, "positions")
    payload["account_id"] = str(caller.account)
    return payload


def _numbered(entries: list[dict[str, Any]], total: int, key: str) -> dict[str, Any]:
    """Stamp a blotter's events with a monotonic id, and say what is missing.

    The id is not invented here and is not an index into anything. ``HumanAgent``
    appends exactly one log entry per private event and
    ``TradingAgent._on_private`` increments exactly one counter for the same
    event, in that order -- so the agent's own ``fills`` counter *is* the
    sequence number of its last fill, and the k entries the log still holds are
    the last k of them. Counting backwards from the counter gives every retained
    event a number that is stable, gap-free, and monotonic across symbols.

    Across symbols is the requirement, and it is why the engine's own
    ``sequence`` cannot be used: sequence numbers are minted per matching engine
    and there is one engine per book, so id 41 exists on every contract at once
    and ordering two fills in different symbols by it is meaningless.

    The counter also makes eviction visible, which is the part that matters
    after a disconnect. ``HumanAgent.log`` keeps the last 200 private events of
    every kind, so an account that generated three hundred between two polls has
    genuinely lost the earliest ones -- and a cursor that renumbered from
    whatever survived would hand a reconnecting client a contiguous-looking
    series with a hole in it. ``first_id`` against the client's own cursor is
    how it finds out instead.
    """
    first = total - len(entries) + 1
    for offset, entry in enumerate(entries):
        entry[key] = first + offset
    return {
        "total": total,
        "retained": len(entries),
        "first_id": first if entries else None,
        "last_id": total if entries else None,
    }


@router.get("/account/fills")
async def fills(request: Request, limit: str | None = None) -> dict[str, Any]:
    """This account's executions, most recent first -- and its refusals.

    Both, in one response, because a client that reads only its fills never
    learns that an order was refused. The venue rejects asynchronously: an
    off-lot size, a price outside the band, a post-only that would have crossed
    and a message over the rate all come back as private events long after the
    202 that accepted them. A blotter that showed only what traded would answer
    "nothing happened" to all four.

    ``amendments`` is the third list, and it is here rather than nowhere for
    the same argument. A replace's whole observable outcome is
    ``kept_priority``, which is a fact about what the engine did and cannot be
    inferred from the resulting order -- an order for six at 100 looks
    identical whether it kept its place in the queue or went to the back of it.
    Without this list a client could send an amendment through
    ``PATCH /v1/orders/{symbol}/{order_id}`` and never learn what it cost,
    which would make a REST client less capable than the page. It is also the
    only place a **pegged** order's tracking is visible: the engine emits one
    of these every time the reference moves.

    ``?after=<fill_id>`` returns only fills strictly after that id, the way
    Binance's ``myTrades?fromId=`` does, and ``?after_rejection=`` and
    ``?after_amendment=`` do the same for the other two. This is the endpoint a
    reconnecting algorithm resumes from: without a monotonic cursor it cannot
    tell a fill it has already booked from one it has not, and the only safe
    reading of an ambiguous blotter is to re-book everything or none of it.
    Three cursors rather than one because they number three different sequences
    -- fill 12, rejection 12 and amendment 12 are unrelated events -- and a
    single ``after`` applied to all of them would silently drop from two.

    Ordering stays newest-first even under a cursor, which is worth stating
    because Binance's ascending order is the more usual choice for one. It costs
    nothing here: the cap *is* ``HumanAgent.log``'s own bound of 200 events, so
    one request at the cap returns everything that exists and there is no second
    page to walk forward into. The ``cursor`` block publishes ``total``,
    ``retained`` and ``first_id`` for each sequence, and a client whose own
    cursor is below ``first_id - 1`` has lost events rather than caught up.
    """
    raw = await request.body()
    caller = _authenticate(request, raw)
    page = _limit(limit, FILLS_DEFAULT, FILLS_CAP, "limit")
    params = request.query_params
    after = _cursor(params.get("after"), "after")
    after_rejection = _cursor(params.get("after_rejection"), "after_rejection")
    after_amendment = _cursor(params.get("after_amendment"), "after_amendment")
    # Before the ids are looked up, so that a fill on an order this API has not
    # yet tied to its client id still comes back named.
    _reconciled(caller)
    who = _market().trader(caller.account)
    log = who.log
    # Copied, not referenced. These dictionaries are the agent's own blotter
    # entries and the WebSocket publishes the same objects, so annotating them
    # in place would have a read of this endpoint quietly change what a browser
    # watching the same account is sent. A read of a simulation must not alter
    # the simulation.
    executions = [dict(entry) for entry in log if entry.get("type") == "fill"]
    refusals = [dict(entry) for entry in log if entry.get("type") == "reject"]
    amendments = [dict(entry) for entry in log if entry.get("type") == "replace"]
    # Numbered before the cursor is applied and before the page is cut, because
    # an id has to mean the same thing whatever was asked for. Numbering the
    # page would make the id a property of the request.
    fill_cursor = _numbered(executions, int(getattr(who, "fills", len(executions))), "fill_id")
    reject_cursor = _numbered(
        refusals, int(getattr(who, "rejects", len(refusals))), "rejection_id"
    )
    amend_cursor = _numbered(
        amendments, int(getattr(who, "amendments", len(amendments))), "amendment_id"
    )
    if after is not None:
        executions = [entry for entry in executions if entry["fill_id"] > after]
    if after_rejection is not None:
        refusals = [entry for entry in refusals if entry["rejection_id"] > after_rejection]
    if after_amendment is not None:
        amendments = [
            entry for entry in amendments if entry["amendment_id"] > after_amendment
        ]
    executions = executions[-page:][::-1]
    refusals = refusals[-page:][::-1]
    amendments = amendments[-page:][::-1]
    for entry in executions + amendments:
        entry["client_order_id"] = _client_id_for(
            caller.token, entry.get("symbol", ""), entry.get("order_id", -1)
        )
    return {
        "account_id": str(caller.account),
        # A cursor only means something inside the market that issued it: a
        # rebuild seats this key behind a fresh agent whose counters start at
        # one, so a client holding fill_id 40 across a rebuild would discard the
        # new market's first forty fills as already seen. Published beside the
        # ids so that is checkable rather than a footnote.
        "generation": int(getattr(_runner(), "generation", 0)),
        "fills": executions,
        "rejections": refusals,
        "amendments": amendments,
        "cursor": {
            "after": after,
            "after_rejection": after_rejection,
            "after_amendment": after_amendment,
            "fills": fill_cursor,
            "rejections": reject_cursor,
            "amendments": amend_cursor,
        },
        "count": len(executions),
        "rejection_count": len(refusals),
        "amendment_count": len(amendments),
        "limit": page,
        "cap": FILLS_CAP,
    }


# --------------------------------------------------------------------------
# Authenticated: orders
# --------------------------------------------------------------------------


@router.get("/orders")
async def working_orders(request: Request, limit: str | None = None) -> dict[str, Any]:
    """What this account has resting, plus what it has in flight.

    ``pending`` is the half a synchronous API does not have to publish and this
    one does: an order accepted here is still crossing the wire, so between the
    202 and the acknowledgement it is in neither the book nor this list unless
    something says so. A client reconciling its own state needs to be able to
    tell "not placed" from "not there yet", and the difference between those two
    is a duplicate order.

    Immediate orders appear there too, briefly. A market order is acknowledged
    and filled in the same instant at the exchange, but that instant is still a
    round trip away from the client, and an order that has not been answered yet
    is in flight whether or not it will ever rest.

    Each pending row carries its age in simulated nanoseconds and no timeout.
    Deliberately: how long a round trip takes is a property of the seat's
    latency link and of how fast the market is being run, both of which this
    layer would have to guess at, and a guessed timeout that fires early tells a
    client an order failed while it is still on its way to the book. A record
    that stays pending was refused by the venue, and the refusal is in
    ``GET /v1/account/fills`` under ``rejections``.
    """
    raw = await request.body()
    caller = _authenticate(request, raw)
    page = _limit(limit, ORDERS_DEFAULT, ORDERS_CAP, "limit")

    rows = _reconciled(caller)
    for row in rows:
        row["client_order_id"] = _client_id_for(
            caller.token, row["symbol"], row["order_id"]
        )

    now = _clock_ns()
    payload = _paged(rows, page, ORDERS_CAP, "orders")
    payload["account_id"] = str(caller.account)
    payload["pending"] = [
        {
            "client_order_id": record.client_order_id,
            "symbol": record.symbol,
            "side": record.side,
            "quantity": record.quantity,
            "submitted_at": record.submitted_at,
            "age_ns": max(0, now - record.submitted_at),
        }
        for record in _client_orders(caller.token).values()
        if record.order_id is None
    ]
    payload["pending_cap"] = CLIENT_ORDER_MEMORY
    return payload


@router.get("/orders:by_client_order_id")
async def order_by_client_order_id(request: Request) -> dict[str, Any]:
    """Look one order up by the id the *client* chose. ``?id=...``

    The colon suffix rather than a fourth path segment, which is Alpaca's
    ``/v2/orders:by_client_order_id`` exactly. A segment would have had to be
    either ``/orders/{client_order_id}`` -- which collides with the exchange's
    own ids under ``/orders/{symbol}/{order_id}`` and would make the meaning of
    a path depend on how many segments follow it -- or a nested collection that
    does not exist. A colon is a legal path character, it sorts as a sibling of
    the collection rather than a member of it, and it reads as what it is: a
    lookup on the collection rather than an item in it.

    This is the half of the reconciliation story that was missing. ``POST
    /v1/orders`` refuses a ``client_order_id`` this seat has already used, and
    the argument for refusing is sound and is in :func:`place_order` -- but
    refusing without giving the client any way to *ask* what happened to the
    first attempt leaves a retried, timed-out POST exactly where it started. It
    knows its id was used. It still does not know whether it is long.

    An id this seat never sent answers 404, the same as one that never existed
    anywhere, because the table is per seat and a client cannot address another
    seat's ids at all.
    """
    raw = await request.body()
    caller = _authenticate(request, raw)
    wanted = (request.query_params.get("id") or "").strip()
    if not wanted:
        raise _refuse(
            "invalid_request",
            "name the client_order_id to look up, as ?id=<client_order_id>",
        )
    # First, so an order acknowledged since the last read is reported as
    # working rather than as still pending.
    _reconciled(caller)
    record = _client_orders(caller.token).get(wanted)
    if record is None:
        raise _refuse(
            "not_found",
            f"this seat has placed no order under the client_order_id {wanted!r}",
            client_order_id=wanted,
            remembered=CLIENT_ORDER_MEMORY,
        )
    return _client_order_row(caller, record)


def _order_request(payload: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Everything ``LiveMarket.submit`` needs, checked in the client's terms.

    Checked here rather than left to the venue because the venue's refusal
    arrives asynchronously as a ``RejectReason`` in a blotter, and "your
    quantity was not a multiple of the lot size" is a fact about the request
    that the request itself can be told. Anything the venue alone can know --
    collateral, the price band, an auction phase -- is still left to the venue.
    """
    symbol = str(payload.get("symbol") or "")
    if not symbol:
        raise _refuse("invalid_symbol", "an order needs a symbol")
    listing = _instrument(symbol)

    # Anything that was not exactly "buy" became a SELL once, silently, in the
    # browser's order path: a typo in a client sold instead of bought and
    # nothing anywhere said so.
    side = str(payload.get("side", "")).strip().lower()
    if side not in ("buy", "sell"):
        raise _refuse("invalid_side", f"side must be buy or sell, not {payload.get('side')!r}")

    quantity = _whole(payload.get("quantity"), "quantity", "invalid_quantity")
    if quantity <= 0:
        raise _refuse("invalid_quantity", "quantity must be a positive whole number")
    if quantity % listing.lot_size:
        raise _refuse(
            "invalid_quantity",
            f"{symbol} is listed in lots of {listing.lot_size}; {quantity} is not a "
            "whole number of them",
        )

    raw_price = payload.get("price")
    price = (
        None
        if raw_price in (None, "", "market")
        else _quotable(listing, _price(raw_price, "price"), "order")
    )
    raw_stop = payload.get("stop")
    stop = (
        None
        if raw_stop in (None, "", "none")
        else _quotable(listing, _price(raw_stop, "stop"), "stop")
    )

    tif = str(payload.get("time_in_force", "") or "").strip().lower()
    if tif and tif not in {choice.value for choice in TimeInForce}:
        raise _refuse(
            "invalid_time_in_force",
            f"{tif!r} is not a time in force",
            supported=sorted(choice.value for choice in TimeInForce),
        )

    display = payload.get("display")
    shown = 0 if display in (None, "", 0) else _whole(display, "display", "invalid_quantity")
    if shown < 0:
        raise _refuse("invalid_quantity", "display size cannot be negative")

    peg, offset = _peg_request(payload)

    # The order type is derived from what was sent -- a price makes it a limit,
    # a trigger makes it a stop, a reference makes it a peg -- exactly as
    # ``LiveMarket.submit`` derives it. A declared ``type`` is therefore checked
    # against that rather than obeyed, so a client whose fields and whose
    # declaration disagree is told which, instead of having one of them
    # silently win.
    #
    # A peg is tested before the other two for the reason ``LiveMarket.submit``
    # tests it first: from here a peg looks like a market order, because it
    # names no price of its own.
    if peg is not None:
        derived = OrderType.PEGGED.value
    elif stop is not None:
        derived = OrderType.STOP_LIMIT.value if price is not None else OrderType.STOP.value
    elif price is not None:
        derived = OrderType.LIMIT.value
    else:
        derived = OrderType.MARKET.value

    # Every combination below is a permanent fact about the request rather than
    # a fact about the market, which is why each one is a 400 here instead of
    # the 422 it would become if it were left to ``LiveMarket.submit``. The
    # catalogue's own grouping makes that distinction load-bearing:
    # ``rejected_by_venue`` means "the market may allow it later", and no
    # market will ever allow a pegged order that also names a price.
    if peg is not None:
        if price is not None:
            raise _refuse(
                "invalid_order_type",
                "a pegged order names no price of its own: its price is the "
                "reference plus the offset, recomputed as the reference moves",
                peg=peg,
            )
        if stop is not None:
            raise _refuse(
                "invalid_order_type", "a pegged order cannot also be a stop", peg=peg
            )
        if tif in (TimeInForce.IOC.value, TimeInForce.FOK.value):
            raise _refuse(
                "invalid_time_in_force",
                f"a pegged order cannot be {tif}: a peg is an instruction to keep "
                "tracking and that one says not to rest",
                peg=peg,
            )

    declared = str(payload.get("type", "") or "").strip().lower()
    if declared:
        supported = sorted(choice.value for choice in OrderType)
        if declared not in supported:
            raise _refuse(
                "invalid_order_type",
                f"{declared!r} is not an order type this API can send",
                supported=supported,
            )
        if declared != derived:
            raise _refuse(
                "invalid_order_type",
                f"you asked for a {declared} order but sent the fields of a "
                f"{derived} one",
                declared=declared,
                derived=derived,
            )

    return (
        {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "tif": tif,
            "stop": stop,
            "display": shown,
            "peg": peg,
            "peg_offset": offset,
        },
        listing,
    )


def _peg_request(payload: dict[str, Any]) -> tuple[str | None, int]:
    """The two fields that make an order a pegged one, or ``(None, 0)``.

    The vocabulary is read off :class:`PegReference` rather than restated, so a
    fourth reference added to the exchange is reachable from here without
    anybody editing this module -- the same argument the time-in-force check
    twenty lines up makes about ``TimeInForce``.

    Both failures answer ``invalid_order_type`` rather than earning a code of
    their own. That is deliberate: ``derived`` is ``pegged`` exactly when a
    reference is present, so these fields *are* the order type, and a catalogue
    entry for a field only one order type has would be a code almost no client
    would ever branch on.

    ``peg_offset`` is a signed whole number of **ticks**, not a price, for the
    reason ``Submit.peg_offset`` is one: on a contract carrying a tick table the
    same decimal is a different number of ticks at different levels, so an
    offset expressed as a price would silently change size as the reference
    moved -- which is precisely what a peg exists to stop happening.
    """
    raw_peg = payload.get("peg")
    peg: str | None = None
    if raw_peg not in (None, "", "none"):
        supported = sorted(choice.value for choice in PegReference)
        candidate = str(raw_peg).strip().lower()
        if candidate not in supported:
            raise _refuse(
                "invalid_order_type",
                f"{raw_peg!r} is not a reference this venue tracks",
                supported=supported,
            )
        peg = candidate

    raw_offset = payload.get("peg_offset")
    offset = (
        0
        if raw_offset in (None, "")
        else _whole(raw_offset, "peg_offset", "invalid_order_type")
    )
    if peg is None and offset:
        # The engine refuses this as INVALID_PEG and is right to: an offset from
        # nothing is not an instruction. Refused here so the client is told
        # which of the two fields it forgot.
        raise _refuse(
            "invalid_order_type",
            "a peg_offset is a number of ticks away from a reference; name the "
            "reference with peg",
            supported=sorted(choice.value for choice in PegReference),
        )
    return peg, offset


@router.post("/orders", status_code=202)
async def place_order(request: Request) -> JSONResponse:
    """Place an order. Answers 202, and the docstring above says why.

    Body::

        {"symbol": "SPIKE_WR_FUT", "side": "buy", "quantity": 5,
         "price": "4700.25", "type": "limit", "time_in_force": "gtc",
         "stop": null, "display": 0, "client_order_id": "abc-1"}

    ``symbol``, ``side`` and ``quantity`` are required; everything else has a
    default. No price means a market order, which is always immediate. A
    ``client_order_id`` is echoed back and is the identifier the order is
    reconcilable by until the exchange has assigned its own.

    A **pegged** order names ``peg`` -- one of ``bid``, ``ask``, ``mid`` -- and
    an optional signed ``peg_offset`` in ticks, and names no ``price``::

        {"symbol": "SPIKE_WR_FUT", "side": "buy", "quantity": 5,
         "peg": "bid", "peg_offset": -1}

    It exists because quoting a *number* and quoting a *position* are different
    intentions and only the second survives the market moving. What it costs is
    queue priority: each reprice is a new price and so a new place in the queue,
    and each one arrives in this account's blotter as a ``replace`` event
    carrying the same order id. The id does not change, which is why
    ``PATCH``/``DELETE /v1/orders/{symbol}/{order_id}`` keep working on a peg
    that has moved -- measured, and it did not before: a peg left this account's
    working-order list at its first reprice and could then not be cancelled at
    all.

    The acknowledgement echoes the fields as they were *sent*, not as the venue
    resolved them, so ``time_in_force`` comes back null when none was given
    rather than filled in with the default. Echoing cannot drift from what the
    venue does; restating its defaults here would be a second copy of a rule
    that lives in ``LiveMarket.submit`` -- where an unpriced order is immediate,
    a triggered one rests, and anything else rests unless told otherwise.

    A ``client_order_id`` this seat has already used is refused rather than
    replayed. Replaying it would mean answering "accepted" for an order this
    call did not place, and a client retrying a timed-out POST cannot tell that
    answer from the truth.

    That refusal is a **409 Conflict**, not a bad request, and it carries the
    existing order in its detail: its exchange id, its status, and its price and
    quantity as sent. The status code is doing real work here. 400 says "your
    request is malformed, fix it and resend" and this request is not malformed
    -- it is a perfectly good order that conflicts with one that already exists,
    which is what 409 means everywhere else on the web. And the detail is the
    answer to the question the retrying client is actually asking: it retried
    because it does not know whether the first attempt landed, and being told
    only "that id is taken" leaves it exactly as ignorant as it was. The same
    facts are readable at ``GET /v1/orders:by_client_order_id?id=...`` and are
    built by the same function, so the refusal and the lookup cannot drift.
    """
    raw, payload = await _body(request)
    caller = _authenticate(request, raw)
    order, listing = _order_request(payload)

    client_order_id = payload.get("client_order_id")
    if client_order_id is not None:
        client_order_id = str(client_order_id)
        if len(client_order_id) > 64:
            raise _refuse(
                "invalid_request", "client_order_id must be 64 characters or fewer"
            )
        existing = _client_orders(caller.token).get(client_order_id)
        if existing is not None:
            # Reconciled first, so the conflict reports "working, id 41" rather
            # than "pending" for an order the exchange acknowledged while the
            # client was deciding to retry -- which is precisely the window a
            # retry happens in.
            _reconciled(caller)
            raise _refuse(
                "duplicate_client_order_id",
                f"this seat has already used the client_order_id "
                f"{client_order_id!r}; that order is below",
                **_client_order_row(caller, existing),
            )

    _throttle(caller.token)

    result = _market().submit(trader=caller.account, **order)
    if not result.get("ok"):
        # The venue understood the order and refused it. Distinct from a
        # malformed request, and a client should treat it differently: the same
        # order may well be accepted a second later.
        raise _refuse("rejected_by_venue", str(result.get("error", "")))

    if client_order_id is not None:
        _remember(
            caller.token,
            _ClientOrder(
                client_order_id=client_order_id,
                symbol=order["symbol"],
                side=order["side"],
                quantity=order["quantity"],
                ticks=None
                if order["price"] is None
                else int(listing.to_ticks(order["price"])),
                submitted_at=_clock_ns(),
            ),
        )

    return JSONResponse(
        {
            "status": "accepted",
            "account_id": str(caller.account),
            "client_order_id": client_order_id,
            "symbol": order["symbol"],
            "side": order["side"],
            "quantity": order["quantity"],
            "price": None if order["price"] is None else str(order["price"]),
            "stop": None if order["stop"] is None else str(order["stop"]),
            "time_in_force": order["tif"] or None,
            "display": order["display"],
            "peg": order["peg"],
            # Echoed as a number rather than a string, because it is a count of
            # ticks and not a price. The rule that prices cross the wire as
            # strings is about decimals a binary double cannot hold exactly; a
            # signed integer of ticks is exact as JSON already, and quoting it
            # would tell a client it is the one thing it is not.
            "peg_offset": order["peg_offset"],
            "accepted_at": _clock_ns(),
            # Said out loud, because a client author reading a 202 for the first
            # time should not have to infer it from the status code.
            "note": (
                "queued to the exchange over this seat's latency link; the "
                "exchange's order id appears in GET /v1/orders once it rests, "
                "or in GET /v1/account/fills if it filled outright"
            ),
        },
        status_code=202,
    )


@router.get("/orders/{symbol}/{order_id}")
async def order_detail(request: Request, symbol: str, order_id: str) -> dict[str, Any]:
    """One working order, addressed the way the exchange addresses it.

    Both halves of the address are required. Order ids come from the matching
    engine and there is one engine per book, so id 5 exists on every contract at
    once -- addressing by id alone means addressing whichever of them was found
    first.

    An order belonging to somebody else answers exactly as one that never
    existed does. Confirming that an id exists but is not yours tells a stranger
    something about a stranger's account.
    """
    raw = await request.body()
    caller = _authenticate(request, raw)
    _instrument(symbol)
    identifier = _whole(order_id, "order_id", "invalid_request")

    who = _market().trader(caller.account)
    if (symbol, identifier) not in who.live_orders:
        raise _refuse("not_found", f"no working order {identifier} in {symbol}")
    order = _venue().engine(symbol).book.get(identifier)
    if order is None:
        raise _refuse("not_found", f"no working order {identifier} in {symbol}")
    _reconciled(caller)
    row = _order_row(symbol, order)
    row["client_order_id"] = _client_id_for(caller.token, symbol, identifier)
    return row


# What an amendment may change. Everything else on an order -- its side, its
# time in force, its display size, its trigger -- is a property of the order
# rather than of the claim it has on the queue, and ``Replace`` cannot carry
# any of them.
#
# A field outside this set is refused rather than ignored, and the reason is a
# bug this venue has already had once in the other direction: a replace used to
# drop ``display_size``, so an iceberg came back fully displayed and published
# the size its owner was working in slices precisely so that nobody could see
# it. The engine now carries it across. Silently accepting ``display`` here
# would rebuild exactly that failure from the client's side -- it would believe
# it had changed the visible size, and nothing would have.
_AMENDABLE = ("quantity", "price")


@router.patch("/orders/{symbol}/{order_id}", status_code=202)
async def amend_order(request: Request, symbol: str, order_id: str) -> JSONResponse:
    """Amend a working order's price, its size, or both, keeping its id.

    Body::

        {"quantity": 6}                     -- size only, price unchanged
        {"price": "4700.25"}                -- price only, size unchanged
        {"quantity": 6, "price": "4700.25"} -- both

    **Why PATCH and not POST.** POST on an item URL means "create a subordinate
    resource under this one", and nothing is created here: the exchange's order
    id survives the amendment, which is the entire reason to prefer this over
    cancel-and-resubmit. PATCH means "a set of changes to be applied to the
    resource", which is exactly what a ``Replace`` is.

    The partial semantics are not decoration either, and this is the argument
    that decided it. Measured on two bids for ten at 100, ours first, a sell of
    five sweeping the level:

        shrink 10 -> 6 at the same price     kept_priority=True    we fill
        grow   10 -> 14 at the same price    kept_priority=False   they fill
        10 -> 10 at the same price           kept_priority=False   they fill
        10 -> 6 at a different price         kept_priority=False   they fill

    Queue priority survives a strict reduction at an unchanged price and
    nothing else -- so an amendment that re-sends a field it is not changing
    goes to the back of the queue for asking for nothing. PATCH's contract is
    "send only what is changing", which makes the priority-preserving call the
    natural one to write. PUT's contract is "send the whole representation",
    which would have made the priority-destroying call the natural one, and a
    client would have paid for the method choice in fills it did not get.

    Answers **202** for the reason ``POST /v1/orders`` does: at the moment it
    answers, the amendment is still crossing this seat's latency link and the
    matching engine has not seen it. Whether priority survived is a fact about
    what the engine did, so it arrives one round trip later in
    ``GET /v1/account/fills`` under ``amendments``, carrying ``kept_priority``
    and its own cursor; the resulting order is at
    ``GET /v1/orders/{symbol}/{order_id}``. It cannot be inferred from that
    order -- one for six at 100 looks identical whether it kept its place in
    the queue or went to the back of it -- which is why the event is published
    rather than left to be worked out.

    Omitting ``quantity`` re-sends the order's current ``remaining``, which is
    what the engine counts an amendment against -- so amending a partly filled
    order to 4 leaves 4 lots working rather than 4 minus what already traded.
    Note the race that omission accepts: if a fill lands while the amendment is
    in flight, the quantity sent is above the new remaining and the engine reads
    it as an increase, which loses priority. A client that cares sends the
    number it wants.

    Omitting both is refused. An amendment that changes nothing still costs
    queue position, so answering it with success would charge a client for a
    call that did nothing it asked for.

    Not idempotent, and deliberately unlike ``DELETE``. A cancel for an order
    that is not resting is a correct outcome the client asked for; an amendment
    to an order that is not resting is not -- there is nothing to carry the new
    terms. So it answers 404, exactly as ``GET`` on the same address does, and
    for the same non-disclosure reason: an order belonging to somebody else
    answers exactly as one that never existed does.
    """
    raw, payload = await _body(request)
    caller = _authenticate(request, raw)
    listing = _instrument(symbol)
    identifier = _whole(order_id, "order_id", "invalid_request")

    unamendable = sorted(set(payload) - set(_AMENDABLE))
    if unamendable:
        raise _refuse(
            "invalid_request",
            f"an amendment can change {' and '.join(_AMENDABLE)}; "
            f"{', '.join(repr(field) for field in unamendable)} cannot be "
            "amended, so cancel and resubmit instead",
            amendable=list(_AMENDABLE),
            refused=unamendable,
        )

    # Addressed exactly as ``order_detail`` addresses it, including the case
    # where the account believes it has an order the book does not: a peg whose
    # reference has gone rests in no level and has no price, and the engine
    # refuses an amendment to one as INVALID_PEG. Refused here instead, so the
    # client learns it a round trip earlier and by the same 404 it would get for
    # any other order it cannot read.
    who = _market().trader(caller.account)
    order = None
    if (symbol, identifier) in who.live_orders:
        order = _venue().engine(symbol).book.get(identifier)
    if order is None:
        raise _refuse("not_found", f"no working order {identifier} in {symbol}")

    wants_quantity = payload.get("quantity") is not None
    wants_price = payload.get("price") not in (None, "")
    if not (wants_quantity or wants_price):
        raise _refuse(
            "invalid_request",
            "an amendment must change something: send quantity, price, or both",
            amendable=list(_AMENDABLE),
        )

    if wants_quantity:
        quantity = _whole(payload["quantity"], "quantity", "invalid_quantity")
        if quantity <= 0:
            # The engine refuses this as INVALID_QUANTITY and leaves the order
            # exactly where it was, which is right -- an amendment to zero is
            # not a cancel and must not be read as one. Said here so a client
            # that meant to cancel is told to use DELETE.
            raise _refuse(
                "invalid_quantity",
                "quantity must be a positive whole number; an amendment to zero "
                "is not a cancel -- DELETE the order instead",
            )
        # The listing rule the submit path applies, applied here too. It is the
        # half of the grid that nobody enforced for a long time: a contract
        # listed in lots of ten took an order for seven.
        if quantity % listing.lot_size:
            raise _refuse(
                "invalid_quantity",
                f"{symbol} is listed in lots of {listing.lot_size}; {quantity} is "
                "not a whole number of them",
            )
    else:
        quantity = int(order.remaining)

    # ``_quotable`` is the same call the submit path makes, and calling it is
    # the point rather than an implementation detail. The tick-grid guard was
    # once written against a field named ``price`` while ``Replace`` carries
    # ``new_price``, so an amendment walked straight past it and rested off the
    # grid; routing both paths through one function is what makes that class of
    # miss impossible rather than merely fixed.
    price = (
        _quotable(listing, _price(payload["price"], "price"), "order")
        if wants_price
        else None
    )

    # Counted and refusable, not counted and exempt. A cancel is exempt because
    # it can only ever reduce the venue's work and the participant's risk; an
    # amendment can raise both, which is why ``Venue._rate_limited`` exempts
    # only ``Cancel`` and why this must not pass ``reducing=True``.
    _throttle(caller.token)

    result = _market().replace(
        identifier, quantity, price, trader=caller.account, symbol=symbol
    )
    if not result.get("ok"):
        raise _refuse("rejected_by_venue", str(result.get("error", "")))

    # The client's own record follows the amendment, because that record holds
    # the order *as sent* and an amendment is also something this seat sent.
    # Leaving the original size there would have
    # ``GET /v1/orders:by_client_order_id`` report a quantity the client itself
    # has already superseded.
    #
    # Being straight about what this does not fix: if the venue then refuses
    # the amendment, the record carries a size that never took effect. That is
    # the same shape ``place_order`` already accepts -- it remembers an order
    # the venue may refuse a round trip later -- and it is survivable for the
    # same reason: the ``order`` block published beside it is read from the
    # engine and stays the authority on what is actually resting, and the
    # refusal itself is in ``GET /v1/account/fills`` under ``rejections``.
    #
    # ``_reconciled`` first, for the reason it is called from every other read
    # that can name an order: binding client ids only where the list is built
    # made the answer depend on the order a client happened to call things in.
    _reconciled(caller)
    tagged = _client_id_for(caller.token, symbol, identifier)
    if tagged is not None:
        record = _client_orders(caller.token)[tagged]
        record.quantity = quantity
        if price is not None:
            record.ticks = int(listing.to_ticks(price))

    return JSONResponse(
        {
            "status": "accepted",
            "account_id": str(caller.account),
            "symbol": symbol,
            "order_id": identifier,
            "client_order_id": tagged,
            "quantity": quantity,
            "price": None if price is None else str(price),
            # What is being replaced, so a client can see the delta it asked
            # for without having read the order first.
            "previous": {
                "quantity": int(order.quantity),
                "remaining": int(order.remaining),
                "price": str(listing.from_ticks(order.price)),
                "ticks": int(order.price),
            },
            "accepted_at": _clock_ns(),
            "note": (
                "queued to the exchange over this seat's latency link; queue "
                "priority survives only a strict reduction at an unchanged "
                "price, and the replace event in GET /v1/account/fills says "
                "which happened"
            ),
        },
        status_code=202,
    )


@router.delete("/orders/{symbol}/{order_id}")
async def cancel_order(request: Request, symbol: str, order_id: str) -> dict[str, Any]:
    """Pull one order. Idempotent: cancelling nothing succeeds.

    The decision, and it is a decision rather than an oversight. A cancel for an
    order that is not working answers 200 with ``already_done: true`` instead of
    404, for three reasons that all point the same way.

    The first is that the client is right. It wanted that order not to be
    resting, and it is not resting. Failing the call would make a correct
    outcome look like an error, and the standard response to a failed cancel --
    send it again -- cannot improve on it.

    The second is that a race is the normal case, not the exceptional one.
    Orders here are cancelled over a latency link, so a cancel and a fill cross
    routinely; a client that retried on 404 would spend the race retrying
    something that had already happened.

    The third is that it costs nothing to give away. A cancel for somebody
    else's order, a cancel for an id that never existed and a cancel for an
    order that filled a millisecond ago all answer identically, so this
    endpoint discloses nothing about orders that are not the caller's -- which
    a 404 for "no such order" and a 200 for "yours, now gone" would.

    An unknown *symbol* is still refused, because that is a typo rather than a
    race, and answering it with success would let a client believe it had
    cancelled something in a market that does not exist.
    """
    raw = await request.body()
    caller = _authenticate(request, raw)
    _instrument(symbol)
    identifier = _whole(order_id, "order_id", "invalid_request")

    # Counted against the rate and never refused by it, the way the venue
    # treats a reducing command: a participant that cannot withdraw is holding
    # exposure nobody is permitted to manage.
    _throttle(caller.token, reducing=True)

    result = _market().cancel(identifier, trader=caller.account, symbol=symbol)
    return {
        "status": "cancelled",
        "symbol": symbol,
        "order_id": identifier,
        "already_done": not result.get("ok"),
        "client_order_id": _client_id_for(caller.token, symbol, identifier),
    }


@router.delete("/orders")
async def cancel_all(request: Request) -> dict[str, Any]:
    """Pull everything this account has working, across every book.

    Distinct from flattening, which closes risk. This leaves every position
    exactly where it is and only stops the account adding to them, which is
    what a client reaching for a panic button at the order layer means.
    """
    raw = await request.body()
    caller = _authenticate(request, raw)
    _throttle(caller.token, reducing=True)
    working = _reconciled(caller)
    _market().cancel_all(trader=caller.account)
    return {
        "status": "cancelled",
        "account_id": str(caller.account),
        "orders": [
            {"symbol": row["symbol"], "order_id": row["order_id"]} for row in working
        ],
        "count": len(working),
    }


# --------------------------------------------------------------------------
# Keys: the bootstrap
#
# Authenticated by the browser session rather than by a key, because this is
# where a key comes from. Everything else on this router refuses an unsigned
# request; these three cannot, or there would be no way to get the first
# credential without one already.
# --------------------------------------------------------------------------


def _key_row(key: ApiKey) -> dict[str, Any]:
    """One credential, described without either secret it holds.

    ``ApiKey.public`` publishes ``agent_id``, and in this module that field
    holds the seat token -- the same value the owner's browser session is
    identified by. So the row is built here instead: what a key's owner needs to
    see is which seat it trades and which account that seat is in right now,
    and neither of those is the token.
    """
    return {
        "key_id": key.key_id,
        "label": key.label,
        "created_at": key.created_at,
        "revoked": key.revoked,
        "seat": _SEAT_NAMES.get(key.agent_id, ""),
        "account_id": str(_account_for(key.agent_id)),
    }


@router.post("/keys", status_code=201)
async def issue_key(request: Request) -> JSONResponse:
    """Mint a credential for the browser session that asked for it.

    The secret is in this response and nowhere else, ever. There is no endpoint
    that reads it back, which is the property that makes a leaked store the only
    way it escapes.

    The key is bound to the caller's *seat*, not to the account that seat is
    sitting in. The account will change -- every rebuild replaces it -- and a
    credential that had captured the old id would fall through
    ``LiveMarket.trader`` onto the shared account and trade a communal seat.
    """
    seat = _browser(request)
    _, payload = await _body(request)
    label = str(payload.get("label", "") or "")[:64]

    name = seat.name or _seat_name(seat.token)
    _SEAT_NAMES[seat.token] = name
    account = _account_for(seat.token, name)

    key = _KEYS.issue(seat.token, label)
    row = _key_row(key)
    row["secret"] = key.secret
    row["account_id"] = str(account)
    row["note"] = "this secret is shown once and is not stored anywhere readable"
    return JSONResponse(row, status_code=201)


@router.get("/keys")
async def list_keys(request: Request, limit: str | None = None) -> dict[str, Any]:
    """Every key this browser session has minted. Never a secret.

    Revoked keys are listed too, flagged rather than hidden. ``KeyStore``
    retires a key instead of forgetting it, so that a revoked id can never be
    reissued and inherit the first owner's history, and a list that dropped
    them would make that retirement invisible to the only person it protects.
    """
    seat = _browser(request)
    page = _limit(limit, KEYS_DEFAULT, KEYS_CAP, "limit")
    rows = [_key_row(key) for key in _KEYS.for_agent(seat.token)]
    rows.sort(key=lambda row: row["created_at"])
    payload = _paged(rows, page, KEYS_CAP, "keys")
    payload["seat"] = _SEAT_NAMES.get(seat.token, seat.name)
    return payload


@router.delete("/keys/{key_id}")
async def revoke_key(request: Request, key_id: str) -> dict[str, Any]:
    """Retire a key. Idempotent, on the same argument as cancelling an order.

    A key belonging to another session answers exactly as one that never
    existed does, so this endpoint cannot be used to discover which key ids are
    real.

    The key is retired rather than deleted, which is ``KeyStore.revoke``'s
    decision and a good one: a forgotten id could be reissued to somebody else
    and would inherit the first owner's history.
    """
    seat = _browser(request)
    existing = _KEYS.keys.get(key_id)
    if existing is None or existing.agent_id != seat.token:
        raise _refuse("not_found", "no such key")
    revoked = _KEYS.revoke(key_id)
    return {"status": "revoked", "key_id": key_id, "already_done": not revoked}
