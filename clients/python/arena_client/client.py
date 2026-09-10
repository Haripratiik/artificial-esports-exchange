"""A client for the exchange's trading API.

The venue already has a browser front end, and a browser is the wrong shape for
an algorithm: it holds a cookie scoped to one machine, it renders numbers for
eyes, and there is nothing in it to import. A systematic trader needs the
opposite: a credential it can carry, a function per endpoint, and values it
can compute with. This is that.

Three decisions are load-bearing, and each of them exists because of a specific
way clients go wrong.

**The signature covers the bytes that are actually sent.** The scheme signs the
timestamp, the method, the path *including its query string*, and the raw body
(see ``python/arena/api/keys.py``). That only works if the string this client
signs is byte-for-byte the string the server receives. An HTTP library is
entitled to re-encode a query it was handed as parameters, and to re-serialise a
body it was handed as an object, and the result is a request that is perfectly
well formed and fails to verify. So this client builds the path and the body
itself, hands httpx finished bytes, and then, before sending, compares what
httpx is about to put on the wire against what it signed, refusing loudly if
they differ. The alternative is a 401 with no way to tell whether the secret is
wrong or the encoding is.

**No price ever becomes a float.** This exchange's premise is exact arithmetic:
cash is integer minor units and conservation is integer zero, not a tolerance.
A client that parses a price with ``float()`` reintroduces exactly the error the
venue was built to avoid, silently, at the last possible moment. A real figure
from a running market makes the size of it concrete: this venue publishes an
average price of ``3479.328892044943820224719101``, twenty-eight significant
digits, because average cost is a ratio of two exact integers. Read that string
as a float and the shortest numeral that recovers it is
``3479.328892044944``: sixteen digits kept, twelve gone. So JSON numbers are parsed
with ``parse_float=Decimal``, money-bearing strings are converted to
``Decimal``, and anything this client does not recognise is handed back as the
exact string the venue sent, which is lossless and which you can always
convert yourself. There is no path through this module that produces a float,
and ``tests/test_api_client.py`` asserts that by walking parsed responses.

**Every failure is a typed exception carrying the venue's code.** The error
envelope is ``{"error": {"code": ..., "message": ...}}``, and the code is the
stable half; the sentence is free to improve. Branch on ``err.code``. The
subclasses are a convenience for the common groupings and nothing more, so a
code this client has never heard of still arrives as an :class:`ArenaError`
with its code intact rather than as a parse failure.

**This client deliberately does not import the exchange.** It depends on the
standard library and httpx, and on nothing in ``python/arena``. That is what
makes it distributable to someone who does not have the venue's source, and it
is why the header names and the signing scheme are transcribed here rather than
imported. Transcription can drift, so ``tests/test_api_client.py`` signs the
same inputs with both implementations and asserts the bytes are identical, so
the duplication is allowed to exist because it cannot silently disagree.

**Nothing here is real.** The venue is a simulation, the underlyings are public
game statistics, the capital is imaginary, and no order placed through this
client reaches any market.
"""

from __future__ import annotations

import hmac
import json
import time
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from typing import Any, NamedTuple
from urllib.parse import urlencode, urlsplit

import httpx

__all__ = [
    "ArenaClient",
    "ArenaError",
    "AuthError",
    "ClientError",
    "InvalidRequest",
    "Level",
    "NotFound",
    "RateLimited",
    "Rejected",
    "HEADER_KEY",
    "HEADER_SIGNATURE",
    "HEADER_TIMESTAMP",
    "MAX_SKEW_SECONDS",
    "amount",
    "body_bytes",
    "canonical_request",
    "sign",
]


# --------------------------------------------------------------------------
# The signing scheme
#
# Transcribed from python/arena/api/keys.py rather than imported, because a
# client that imports the server is not a client. tests/test_api_client.py
# cross-checks every function here against that module, so the copy is pinned.
# --------------------------------------------------------------------------

HEADER_KEY = "arena-key-id"
HEADER_TIMESTAMP = "arena-timestamp"
HEADER_SIGNATURE = "arena-signature"

# The venue refuses a request whose timestamp sits further than this from its
# own clock. Stated here so a caller with a drifting clock can find the number
# that is refusing them, rather than reading a 401 that says nothing.
MAX_SKEW_SECONDS = 30


def canonical_request(
    method: str, path: str, timestamp: str, body: bytes = b""
) -> bytes:
    """The exact bytes a signature covers.

    Newline-separated, in this order: timestamp, method, path, body. The
    separator matters: concatenation would let path ``/v1/orders`` with body
    ``x`` and path ``/v1/order`` with body ``sx`` produce identical bytes.

    ``path`` carries its query string, so a signature obtained for one filter
    cannot be lifted onto another.
    """
    return b"\n".join(
        [
            timestamp.encode("utf-8"),
            method.upper().encode("utf-8"),
            path.encode("utf-8"),
            body or b"",
        ]
    )


def sign(secret: str, method: str, path: str, timestamp: str, body: bytes = b"") -> str:
    """The hex HMAC-SHA256 signature for one request."""
    message = canonical_request(method, path, timestamp, body)
    return hmac.new(secret.encode("utf-8"), message, sha256).hexdigest()


def body_bytes(payload: Any) -> bytes:
    """Serialise a body the way both sides have to serialise it.

    Sorted keys and no incidental whitespace, so that the same object produces
    the same bytes here and at the venue regardless of dict ordering at either
    end. A signature covers bytes; if the two sides disagree about how an
    object becomes bytes, every signed request fails and nothing says why.
    """
    if payload is None:
        return b""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


# --------------------------------------------------------------------------
# Failures
# --------------------------------------------------------------------------


class ArenaError(Exception):
    """A refusal, carrying the venue's machine code.

    ``code`` is the contract and the thing to branch on. ``message`` is a
    sentence for a human and may change without notice. ``status`` is the HTTP
    status, and ``detail`` is whatever extra structure the venue attached.
    """

    def __init__(
        self,
        code: str,
        message: str,
        status: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.status = status
        self.detail = detail or {}


class AuthError(ArenaError):
    """``auth_*``: stop and fix the credential. Retrying cannot help."""


class InvalidRequest(ArenaError):
    """``invalid_*``: the request is malformed. Fix it and resend."""


class Rejected(ArenaError):
    """``rejected_*``: the venue understood and refused. Later it may not."""


class NotFound(ArenaError):
    """``not_found``: the thing addressed does not exist."""


class RateLimited(ArenaError):
    """``rate_limited``: back off and retry. Temporary by definition."""


class ClientError(ArenaError):
    """Raised by this client, never by the venue.

    Its codes all begin ``client_`` so that a log can tell "the venue refused
    me" apart from "I could not form or read the request", which are different
    problems with different fixes and were indistinguishable when both arrived
    as a bare exception.
    """


def _error_for(code: str, message: str, status: int, detail: Any) -> ArenaError:
    """Pick the class from the code's prefix.

    Prefix rather than an exhaustive table, so a code added to the venue after
    this client shipped still lands in the right group instead of falling
    through to the base class. A code matching nothing is still an
    :class:`ArenaError` with its code intact, since unrecognised is not the same as
    unusable.
    """
    kind: type[ArenaError] = ArenaError
    if code.startswith("auth_"):
        kind = AuthError
    elif code.startswith("invalid_"):
        kind = InvalidRequest
    elif code.startswith("rejected_"):
        kind = Rejected
    elif code == "not_found":
        kind = NotFound
    elif code == "rate_limited":
        kind = RateLimited
    elif code.startswith("client_"):
        kind = ClientError
    return kind(code, message, status, detail if isinstance(detail, dict) else None)


# --------------------------------------------------------------------------
# Decimal, everywhere a number means money
# --------------------------------------------------------------------------


class Level(NamedTuple):
    """One rung of the ladder.

    A tuple, so it still indexes like the ``[price, quantity]`` pair the venue
    sends and code written against the raw payload keeps working; named, so
    ``level.price`` reads as what it is.
    """

    price: Decimal
    quantity: int


# Keys whose string value is an amount of money or a price. Every name here is
# produced by a serialiser in this repository: `Account.to_dict` and
# `Position.to_dict` in python/arena/portfolio, `Instrument.to_dict` in
# python/arena/market/instrument.py, and the halt records in dashboard/state.py
# with the exception of `price` and `stop`, which the order contract names.
#
# A key that is not in this set is returned as the exact string the venue sent.
# That is deliberate: guessing that an unknown string is money and converting it
# would corrupt an identifier that happens to look numeric, while leaving it as
# a string loses nothing, because `Decimal(value)` is still available to the
# caller. The one thing never done is turning it into a float.
MONEY_FIELDS = frozenset(
    {
        "average_price",
        "cash",
        "cost_basis",
        "equity",
        "fees_paid",
        "free_cash",
        "mark",
        "posted_collateral",
        "price",
        "realized_pnl",
        "reference",
        "starting_cash",
        "stop",
        "tick_size",
        "unrealized_pnl",
    }
)

# Keys holding a sequence of amounts rather than one. `settlement_bounds` and
# `value_bounds` are pairs, and they are the reason this case exists at all:
# a future bounded at ten thousand serialises its upper bound as "1E+4", which
# `Decimal` reads exactly and `int()` refuses outright.
MONEY_SEQUENCE_FIELDS = frozenset({"settlement_bounds", "value_bounds"})

# Keys holding a ladder: a list of [price, quantity] pairs.
LADDER_FIELDS = frozenset({"bids", "asks"})


def _to_decimal(value: Any) -> Any:
    """An exact amount, or a loud failure. Never an approximation."""
    if value is None or isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        # Unreachable through this client's own parser, which never produces a
        # float. Raising rather than accepting keeps that guarantee true for
        # anyone calling this helper directly.
        raise ClientError(
            "client_float_amount",
            f"{value!r} arrived as a float, which has already lost precision; "
            "amounts must be parsed exactly",
        )
    if isinstance(value, (int, str)):
        try:
            return Decimal(value)
        except InvalidOperation:
            raise ClientError(
                "client_unreadable_response",
                f"{value!r} was published in a money field and is not a number",
            ) from None
    return value


def _level(entry: Any) -> Any:
    """One ladder rung, or the entry untouched if it is not a rung.

    Untouched rather than an error, because the shape of the ladder belongs to
    the venue. A client that crashes when a field is added to a book level is
    worse than one that hands the addition back unrecognised.
    """
    if isinstance(entry, (list, tuple)) and len(entry) == 2:
        price, quantity = entry
        try:
            return Level(_to_decimal(price), int(quantity))
        except (TypeError, ValueError):
            return entry
    return _normalise(entry)


def _normalise(value: Any, key: str | None = None) -> Any:
    """Walk a parsed payload, making money exact and changing nothing else."""
    if isinstance(value, dict):
        return {name: _normalise(item, name) for name, item in value.items()}
    if isinstance(value, list):
        if key in LADDER_FIELDS:
            return [_level(entry) for entry in value]
        if key in MONEY_SEQUENCE_FIELDS:
            return [_to_decimal(entry) for entry in value]
        return [_normalise(entry, key) for entry in value]
    if key in MONEY_FIELDS and isinstance(value, str):
        return _to_decimal(value)
    return value


def _reject_non_finite(token: str) -> Any:
    """JSON has no NaN or Infinity, and a price cannot be either.

    ``json.loads`` will happily hand back ``float('nan')`` for a bare ``NaN``
    token. Accepting it would put the one value in Python that is not equal to
    itself into a position size, so it is refused where it is cheap to see.
    """
    raise ClientError(
        "client_unreadable_response",
        f"the response contained the non-JSON token {token!r}",
    )


def _decode(text: str) -> Any:
    """Parse a response body without ever creating a float.

    ``parse_float=Decimal`` is the half of this that needs no knowledge of the
    schema: a bare JSON number such as ``4663.25`` becomes ``Decimal('4663.25')``
    from the literal digits, not from a binary approximation of them. Whether or
    not this client recognises the field it appeared in.
    """
    if not text.strip():
        return None
    try:
        return json.loads(
            text, parse_float=Decimal, parse_constant=_reject_non_finite
        )
    except json.JSONDecodeError as exc:
        raise ClientError(
            "client_unreadable_response",
            f"the venue returned something that is not JSON: {exc}",
        ) from None


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------


def amount(value: Decimal | int | str) -> str:
    """A price, on the wire, as the exact numeral the caller meant.

    Sent as a JSON string rather than a JSON number for the same reason it
    comes back as one: a number in JSON is a float to most readers, and this
    venue matches on a tick grid where being one part in 2**53 off the grid is
    a rejection rather than a rounding.

    A ``float`` argument is refused outright. ``0.1 + 0.2`` is not ``0.3``, and
    a price is the last place to discover that; ``Decimal('4663.25')`` or the
    string ``'4663.25'`` says exactly what was meant and costs nothing.
    """
    if isinstance(value, bool):
        raise TypeError(f"{value!r} is not a price")
    if isinstance(value, float):
        raise TypeError(
            f"{value!r} is a float, and a float price is how an order ends up on "
            "a price nobody chose. Pass Decimal('...') or the string '...' instead"
        )
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Decimal):
        return format(value, "f")
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise ValueError(
            f"{value!r} is not a price: digits and a decimal point only, with "
            "no commas or currency symbols"
        ) from None
    return format(parsed, "f")


def _whole(value: Any, field: str, minimum: int = 1) -> int:
    """A whole number no smaller than ``minimum``, or a sentence saying why not.

    ``bool`` is rejected explicitly because it is an ``int`` subclass in
    Python, and ``quantity=True`` would otherwise buy one lot.

    ``minimum`` is zero for a display size, where zero is a real answer meaning
    "no reserve" rather than a mistake, and one for a quantity, where it is not.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be a whole number, not {value!r}")
    if value < minimum:
        raise ValueError(f"{field} must be at least {minimum}, not {value}")
    return value


def _side(value: str) -> str:
    """One of exactly two words.

    Checked here because the failure is silent otherwise. The browser front end
    carries a note about the version of this that read anything not exactly
    ``buy`` as a SELL: a typo in a client sold instead of bought, and nothing
    anywhere said so.

    This is the only kind of validation the client does. It does not check the
    tick grid, the settlement range, or whether a symbol is listed, because
    those need the venue's state and a client that keeps its own copy of them
    will eventually disagree with the venue about what is legal, and the
    venue is right by definition.
    """
    text = str(value).strip().lower()
    if text not in ("buy", "sell"):
        raise ValueError(f"side must be buy or sell, not {value!r}")
    return text


def _query(params: dict[str, Any] | None) -> str:
    """A query string, or nothing.

    Parameters left as ``None`` are dropped rather than sent empty: ``?depth=``
    is a different path from no path parameter at all, and therefore a
    different signature over a different string.
    """
    if not params:
        return ""
    pairs = [(key, str(value)) for key, value in params.items() if value is not None]
    if not pairs:
        return ""
    return "?" + urlencode(pairs)


# --------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------


class ArenaClient:
    """A synchronous client for one exchange.

    Credentials are optional. Without them the public endpoints work and the
    signed ones raise before a request is sent, which is better than spending a
    round trip to be told the obvious.

        >>> client = ArenaClient("http://localhost:8000")            # doctest: +SKIP
        >>> book = client.book("VANTA_OBJECTIVE_WR", depth=5)              # doctest: +SKIP
        >>> book["bids"][0].price                                    # doctest: +SKIP
        Decimal('4689.00')

    ``transport`` exists so the suite can drive the client against a stub
    rather than a live venue: pass ``httpx.MockTransport(handler)`` and every
    request lands in ``handler`` with the exact bytes that would have gone on
    the wire, signature included.
    """

    def __init__(
        self,
        base_url: str,
        key_id: str | None = None,
        secret: str | None = None,
        *,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.key_id = key_id
        self.secret = secret
        # If the venue is served under a path prefix, that prefix is part of
        # the path the server receives, and therefore part of what it verifies
        # the signature over. Signing the bare "/v1/..." in that case produces
        # a request that is correct in every visible way and refused.
        self._prefix = urlsplit(self.base_url).path.rstrip("/")
        self._http = httpx.Client(timeout=timeout, transport=transport)

    # plumbing ------------------------------------------------------------

    def __enter__(self) -> ArenaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    @property
    def authenticated(self) -> bool:
        """Whether this client holds a credential it could sign with."""
        return bool(self.key_id and self.secret)

    def _timestamp(self) -> str:
        """Whole seconds since the epoch, as the string that gets signed.

        The header and the signature must carry the *same* string, not two
        renderings of the same instant, so it is produced once here and passed
        around rather than read from the clock twice.
        """
        return str(int(time.time()))

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: Any = None,
        signed: bool = False,
    ) -> Any:
        signed_path = f"{self._prefix}{path}{_query(params)}"
        raw = body_bytes(body) if body is not None else b""

        headers = {"accept": "application/json"}
        if raw:
            headers["content-type"] = "application/json"

        if signed:
            key_id, secret = self.key_id, self.secret
            if not (key_id and secret):
                raise ClientError(
                    "client_credentials_missing",
                    f"{method} {path} must be signed, and this client was built "
                    "without a key_id and secret",
                )
            timestamp = self._timestamp()
            headers[HEADER_KEY] = key_id
            headers[HEADER_TIMESTAMP] = timestamp
            headers[HEADER_SIGNATURE] = sign(
                secret, method, signed_path, timestamp, raw
            )

        request = self._http.build_request(
            method, f"{self.base_url}{path}{_query(params)}", content=raw, headers=headers
        )

        # The check that turns an unexplainable 401 into a sentence. httpx is
        # free to normalise a URL it was handed, a space becomes %20 for one,
        # and if it does, the bytes it sends are not the bytes we signed.
        sent_path = request.url.raw_path.decode("ascii")
        if sent_path != signed_path:
            raise ClientError(
                "client_path_mismatch",
                f"signed {signed_path!r} but the request would send {sent_path!r}; "
                "the signature covers the path, so this would be refused",
            )

        try:
            response = self._http.send(request)
        except httpx.HTTPError as exc:
            raise ClientError(
                "client_transport",
                f"{method} {signed_path} did not complete: {exc}",
            ) from None

        return self._payload(response, method, signed_path)

    def _payload(self, response: httpx.Response, method: str, path: str) -> Any:
        try:
            payload = _decode(response.text)
        except ClientError as exc:
            # A body that is not JSON is diagnosable only with the status
            # attached. "502" is the entire explanation for a gateway that
            # never reached the venue, and reporting it as "not JSON" with no
            # status leaves a caller with nothing to act on.
            raise ClientError(
                exc.code,
                f"{method} {path} returned {response.status_code}: {exc.message}",
                response.status_code,
            ) from None

        envelope = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(envelope, dict):
            # Raised even on a 2xx. A refusal delivered with a success status is
            # a venue bug, and the dangerous reading of it is the optimistic
            # one: a trading loop that treats it as success believes an order
            # is resting that is not.
            raise _error_for(
                str(envelope.get("code", "")),
                str(envelope.get("message", "")),
                response.status_code,
                envelope.get("detail"),
            )

        if response.status_code >= 400:
            # A failure that is not in the documented envelope. Anything can
            # produce this, a proxy or a crash before the handler ran, so it
            # is reported under a client_ code that says where it came from
            # rather than being dressed up as a venue error code.
            raise ClientError(
                "client_unreadable_response",
                f"{method} {path} returned {response.status_code} without an "
                f"error envelope: {response.text[:200]!r}",
                response.status_code,
            )

        return _normalise(payload)

    # public market data ------------------------------------------------

    def exchange(self) -> Any:
        """The venue's description of itself: fees, sessions, and its state."""
        return self._request("GET", "/v1/exchange")

    def instruments(
        self, instrument_class: str | None = None, subject: str | None = None
    ) -> Any:
        """Everything listed, optionally filtered.

        ``instrument_class`` rather than ``class`` because ``class`` is a
        keyword in Python; it is sent on the wire as ``class``.
        """
        return self._request(
            "GET",
            "/v1/instruments",
            params={"class": instrument_class, "subject": subject},
        )

    def instrument(self, symbol: str) -> Any:
        """One symbol: its contract, its tick grid, and what it settles into."""
        return self._request("GET", f"/v1/instruments/{symbol}")

    def book(self, symbol: str, depth: int | None = None) -> Any:
        """The ladder. ``bids`` and ``asks`` come back as :class:`Level` rungs."""
        return self._request(
            "GET", f"/v1/instruments/{symbol}/book", params={"depth": depth}
        )

    def trades(self, symbol: str, limit: int | None = None) -> Any:
        """The public tape: what printed, at what price, and who crossed."""
        return self._request(
            "GET", f"/v1/instruments/{symbol}/trades", params={"limit": limit}
        )

    def history(self, symbol: str) -> Any:
        """The recent price path, for charting or for a signal."""
        return self._request("GET", f"/v1/instruments/{symbol}/history")

    # the account -------------------------------------------------------

    def account(self) -> Any:
        """Cash, collateral, realised and unrealised PnL, and equity."""
        return self._request("GET", "/v1/account", signed=True)

    def positions(self) -> Any:
        """What is held, at what average price, marked to the market."""
        return self._request("GET", "/v1/account/positions", signed=True)

    def fills(self, limit: int | None = None) -> Any:
        """This account's executions, most recent first."""
        return self._request(
            "GET", "/v1/account/fills", params={"limit": limit}, signed=True
        )

    # orders ------------------------------------------------------------

    def orders(self) -> Any:
        """Working orders across every symbol."""
        return self._request("GET", "/v1/orders", signed=True)

    def order(self, symbol: str, order_id: int | str) -> Any:
        """One order by id.

        Addressed by symbol and id together because order ids are assigned per
        book, so the id alone does not identify an order.
        """
        return self._request("GET", f"/v1/orders/{symbol}/{order_id}", signed=True)

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        *,
        price: Decimal | int | str | None = None,
        order_type: str | None = None,
        time_in_force: str | None = None,
        stop: Decimal | int | str | None = None,
        display: int | None = None,
        client_order_id: str | None = None,
    ) -> Any:
        """Send an order.

        Only ``symbol``, ``side`` and ``quantity`` are required. Omitting
        ``price`` is how a market order is expressed, which is what the venue's
        own engine does: a market order there is a limit order with no price
        bound, not a separate matching path.

        ``client_order_id`` is worth using even though it is optional. It is
        the only thing that lets a retry after a timeout be told apart from a
        second order, and a timeout is the one failure a live trading loop is
        guaranteed to meet.
        """
        payload: dict[str, Any] = {
            "symbol": symbol,
            "side": _side(side),
            "quantity": _whole(quantity, "quantity"),
        }
        if price is not None:
            payload["price"] = amount(price)
        if order_type is not None:
            payload["type"] = order_type
        if time_in_force is not None:
            payload["time_in_force"] = time_in_force
        if stop is not None:
            payload["stop"] = amount(stop)
        if display is not None:
            payload["display"] = _whole(display, "display", minimum=0)
        if client_order_id is not None:
            payload["client_order_id"] = str(client_order_id)
        return self._request("POST", "/v1/orders", body=payload, signed=True)

    def cancel(self, symbol: str, order_id: int | str) -> Any:
        """Pull one order out of the book."""
        return self._request("DELETE", f"/v1/orders/{symbol}/{order_id}", signed=True)

    def cancel_all(self) -> Any:
        """Pull every working order this account has, on every symbol."""
        return self._request("DELETE", "/v1/orders", signed=True)

    # credentials -------------------------------------------------------

    def keys(self) -> Any:
        """The keys on this seat. Never the secrets: those are shown once."""
        return self._request("GET", "/v1/keys", signed=True)

    def create_key(self, label: str = "") -> Any:
        """Mint another key for this seat.

        Signed, so it cannot be the way the *first* key is obtained. Rotation
        is what it is for: mint the replacement, switch to it, then revoke the
        old one, with no window in which the account has no working key.
        """
        return self._request("POST", "/v1/keys", body={"label": label}, signed=True)

    def revoke_key(self, key_id: str) -> Any:
        """Retire a key. The venue keeps the id so it can never be reissued."""
        return self._request("DELETE", f"/v1/keys/{key_id}", signed=True)

    # streaming ---------------------------------------------------------

    def stream_url(self) -> str:
        """The websocket URL for this venue.

        This client does not open the socket. Adding a websocket dependency to
        get a loop the caller will want to own anyway is a poor trade, and
        every websocket library takes a URL. What is worth providing is the
        part that is easy to get wrong, which is the auth frame below.
        """
        if self.base_url.startswith("https://"):
            return "wss://" + self.base_url[len("https://") :] + "/v1/stream"
        if self.base_url.startswith("http://"):
            return "ws://" + self.base_url[len("http://") :] + "/v1/stream"
        return self.base_url + "/v1/stream"

    def stream_auth(self) -> dict[str, Any]:
        """The ``auth`` frame that upgrades a socket to the private channels.

        Signed over the same canonical string as a GET of the stream path with
        an empty body, so one implementation of :func:`sign` serves both
        transports.

        This is the one shape in this client that has not been checked against
        a running server, since the socket handler is being written in parallel with
        it. If the venue disagrees, the disagreement is in the field names of
        this frame, not in the signature, which is computed the same way here as
        for every other request.
        """
        key_id, secret = self.key_id, self.secret
        if not (key_id and secret):
            raise ClientError(
                "client_credentials_missing",
                "the private stream channels need a key_id and secret",
            )
        timestamp = self._timestamp()
        path = f"{self._prefix}/v1/stream"
        return {
            "op": "auth",
            "key_id": key_id,
            "timestamp": timestamp,
            "signature": sign(secret, "GET", path, timestamp, b""),
        }

    @staticmethod
    def subscribe(*channels: str) -> dict[str, Any]:
        """A ``subscribe`` frame. ``unsubscribe`` takes the same shape."""
        return {"op": "subscribe", "channels": list(channels)}

    @staticmethod
    def unsubscribe(*channels: str) -> dict[str, Any]:
        return {"op": "unsubscribe", "channels": list(channels)}
