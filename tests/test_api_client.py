"""The client, checked against the venue without a venue running.

Two things in ``clients/python/arena_client`` can be wrong in ways that no
amount of reading catches, and both are checked here.

**The signing scheme is transcribed, not imported.** A client that imports
``arena.api.keys`` is not distributable -- it would need the exchange's source
to sign a request -- so the header names, the canonical string and the body
serialisation exist twice. Duplication that can drift silently is a liability;
duplication that a test pins is just a copy. So this file signs the same inputs
with both implementations and demands identical bytes. If the venue changes the
scheme, this suite fails before any client does.

**Prices must not become floats.** The exchange counts money in integer minor
units so that conservation is integer zero rather than nearly zero. A client
that parses a price into a float undoes that at the last step, quietly. The
figures below are not invented: they were captured from a real run of
``dashboard.state.MarketRunner``, and one of them -- an average price of
``3479.328892044943820224719101``, which is a ratio of two exact integers --
loses twelve of its twenty-eight significant digits to ``float()``.

Nothing here talks to a server. Every request goes through
``httpx.MockTransport``, so the suite is fast, deterministic, and independent of
whether the REST and websocket handlers exist yet.
"""

from __future__ import annotations

import ast
import json
import sys
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from arena.api import errors, keys

# The client is a separate distribution living outside the ``arena`` package, on
# purpose: see the module docstring. That puts it outside pytest's importable
# paths, so the suite adds it explicitly rather than pretending the two ship
# together.
REPO = Path(__file__).resolve().parents[1]
CLIENT_ROOT = REPO / "clients" / "python"
if str(CLIENT_ROOT) not in sys.path:
    sys.path.insert(0, str(CLIENT_ROOT))

import arena_client  # noqa: E402
from arena_client import (  # noqa: E402
    ArenaClient,
    ArenaError,
    AuthError,
    ClientError,
    InvalidRequest,
    Level,
    NotFound,
    RateLimited,
    Rejected,
)

# --------------------------------------------------------------------------
# The published vector
#
# docs/API.md prints these exact strings so that somebody writing a client in
# another language can check themselves against a known answer. They are
# asserted here, against both implementations, so the documentation cannot
# drift away from the code that produces it.
# --------------------------------------------------------------------------

DOC_KEY_ID = "ak_0011223344556677"
DOC_SECRET = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0"
DOC_TIMESTAMP = "1787000000"
# Recomputed when the exchange moved to its own world and the old symbols
# went with it. The price moved as well as the symbol: the replacement
# contract carries the tiered tick grid, which steps to 1.00 above 4,000, so
# 4663.25 is a price this venue refuses and a worked example nobody could send.
DOC_BODY = {
    "symbol": "VANTA_OBJECTIVE_WR",
    "side": "buy",
    "quantity": 5,
    "price": "4663.00",
}
DOC_BODY_BYTES = b'{"price":"4663.00","quantity":5,"side":"buy","symbol":"VANTA_OBJECTIVE_WR"}'
DOC_CANONICAL = (
    b"1787000000\nPOST\n/v1/orders\n"
    b'{"price":"4663.00","quantity":5,"side":"buy","symbol":"VANTA_OBJECTIVE_WR"}'
)
DOC_SIGNATURE = "b6b5153733f44607c8793cd3f19ea62bf35a65b5f9651d478f0252d55948db88"

DOC_GET_PATH = "/v1/instruments/VANTA_OBJECTIVE_WR/book?depth=5"
DOC_GET_SIGNATURE = "6b3a1a689a12d10e9bae1f8f41db14fe1af53461be7c6159d872586fc4cfcae8"


# --------------------------------------------------------------------------
# Payloads captured from a real market
#
# Produced by building the live market from dashboard/build_market.py, running
# the kernel, and serialising with the venue's own `to_dict` methods. Nothing
# below was made up to make a test pass.
# --------------------------------------------------------------------------

REAL_BOOK = {
    "symbol": "VANTA_OBJECTIVE_WR",
    "bids": [["4689.00", 22], ["4688.75", 3], ["4684.50", 14], ["4671.75", 2]],
    "asks": [["4693.25", 6], ["4696.75", 30], ["4730.25", 4], ["4757.00", 6]],
    "session": "continuous",
}

REAL_INSTRUMENT = {
    "symbol": "VANTA_OBJECTIVE_WR",
    "class": "future",
    "contract_id": "VANTA_OBJECTIVE_WR",
    "spec_digest": (
        "sha256:5118899e743c009f2680e08970bf545d3970e1b00e960f4b128aa9ab6428c41b"
    ),
    "tick_size": "0.25",
    "lot_size": 1,
    "settlement_bounds": ["0", "1E+4"],
    "expiry": "2026-09-28T00:00:00Z",
}

REAL_ACCOUNT = {
    "agent_id": "mm-1",
    "cash": "128672864.139831",
    "starting_cash": "135306150",
    "posted_collateral": "12178767.015264",
    "free_cash": "116494097.124567",
    "realized_pnl": "-6633285.860169",
    "unrealized_pnl": "-4646834.405264",
    "equity": "124026029.734567",
    "positions": [
        {
            "symbol": "CROW_DISP",
            "quantity": 356,
            "cost_basis": "1238641.085568",
            "average_price": "3479.328892044943820224719101",
            "realized_pnl": "-290375.198532",
            "fees_paid": "183.2841",
            "volume": 3170,
            "mark": "4041.5",
            "unrealized_pnl": "200132.914432",
            "equity": "-90242.2841",
        }
    ],
}

REAL_TRADE = {
    "type": "trade",
    "sequence": 14911,
    "quantity": 14,
    "price": "4694.00",
    "aggressor_side": "sell",
    "buy_order_id": 5083,
    "sell_order_id": 5086,
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def stub(payload, status=200, **kwargs):
    """A client wired to a transport that answers with one fixed payload.

    Returns the client and a dict the handler records the request into, so a
    test can assert on the exact bytes that would have gone on the wire.
    """
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(
            status,
            content=json.dumps(payload) if payload is not None else b"",
            headers={"content-type": "application/json"},
        )

    client = ArenaClient(
        kwargs.pop("base_url", "http://venue.test"),
        transport=httpx.MockTransport(handler),
        **kwargs,
    )
    return client, seen


def signed_stub(payload, store, **kwargs):
    """A stub that verifies the signature with the venue's own key store.

    This is the strongest check available without a server: the request is
    built by the client, and accepted or refused by the same code path the
    exchange uses.
    """
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        seen["key"] = store.verify(
            request.headers.get(keys.HEADER_KEY, ""),
            request.headers.get(keys.HEADER_TIMESTAMP, ""),
            request.headers.get(keys.HEADER_SIGNATURE, ""),
            request.method,
            request.url.raw_path.decode("ascii"),
            request.content,
        )
        return httpx.Response(200, content=json.dumps(payload))

    client = ArenaClient(
        kwargs.pop("base_url", "http://venue.test"),
        transport=httpx.MockTransport(handler),
        **kwargs,
    )
    return client, seen


def floats_in(value, trail="") -> list[str]:
    """Every path in a parsed payload that holds a float. Should be empty."""
    found: list[str] = []
    if isinstance(value, float):
        found.append(trail or "<root>")
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(floats_in(item, f"{trail}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(floats_in(item, f"{trail}[{index}]"))
    return found


# --------------------------------------------------------------------------
# The signature is the venue's signature
# --------------------------------------------------------------------------

VECTORS = [
    ("GET", "/v1/account", "1787000000", b""),
    ("GET", "/v1/account/fills?limit=50", "1787000001", b""),
    ("POST", "/v1/orders", "1787000002", DOC_BODY_BYTES),
    ("DELETE", "/v1/orders/VANTA_OBJECTIVE_WR/5083", "1787000003", b""),
    ("DELETE", "/v1/orders", "0", b""),
    ("get", "/v1/instruments?class=future&subject=SPIKE", "1787000004", b""),
]


@pytest.mark.parametrize(("method", "path", "timestamp", "body"), VECTORS)
def test_the_canonical_string_is_the_one_the_venue_builds(method, path, timestamp, body):
    """The bytes under the HMAC, built twice, must be one string."""
    assert arena_client.canonical_request(method, path, timestamp, body) == (
        keys.canonical_request(method, path, timestamp, body)
    )


@pytest.mark.parametrize(("method", "path", "timestamp", "body"), VECTORS)
def test_the_signature_is_the_one_the_venue_computes(method, path, timestamp, body):
    """The whole point of the duplication test: same inputs, same hex."""
    assert arena_client.sign(DOC_SECRET, method, path, timestamp, body) == (
        keys.sign(DOC_SECRET, method, path, timestamp, body)
    )


def test_a_body_serialises_to_the_same_bytes_whatever_order_it_was_built_in():
    """A signature covers bytes, so dict ordering must not reach them."""
    one = {"symbol": "VANTA_OBJECTIVE_WR", "side": "buy", "quantity": 5, "price": "4663.00"}
    other = {"price": "4663.00", "quantity": 5, "side": "buy", "symbol": "VANTA_OBJECTIVE_WR"}
    assert arena_client.body_bytes(one) == arena_client.body_bytes(other)
    assert arena_client.body_bytes(one) == keys.body_bytes(one)
    assert arena_client.body_bytes(one) == DOC_BODY_BYTES
    assert arena_client.body_bytes(None) == keys.body_bytes(None) == b""


def test_the_header_names_and_the_skew_window_match_the_venue():
    """Three strings and a number that must never drift apart in spelling."""
    assert arena_client.HEADER_KEY == keys.HEADER_KEY
    assert arena_client.HEADER_TIMESTAMP == keys.HEADER_TIMESTAMP
    assert arena_client.HEADER_SIGNATURE == keys.HEADER_SIGNATURE
    assert arena_client.MAX_SKEW_SECONDS == keys.MAX_SKEW_SECONDS


def test_the_documented_signing_vector_is_the_real_answer():
    """docs/API.md publishes these strings. This is what keeps them true.

    Somebody implementing a client in another language checks their output
    against the document. A document that has drifted from the code sends them
    hunting for a bug in their own HMAC.
    """
    assert arena_client.body_bytes(DOC_BODY) == DOC_BODY_BYTES
    assert (
        arena_client.canonical_request("POST", "/v1/orders", DOC_TIMESTAMP, DOC_BODY_BYTES)
        == DOC_CANONICAL
    )
    assert (
        arena_client.sign(DOC_SECRET, "POST", "/v1/orders", DOC_TIMESTAMP, DOC_BODY_BYTES)
        == DOC_SIGNATURE
    )
    assert (
        keys.sign(DOC_SECRET, "POST", "/v1/orders", DOC_TIMESTAMP, DOC_BODY_BYTES)
        == DOC_SIGNATURE
    )
    assert (
        arena_client.sign(DOC_SECRET, "GET", DOC_GET_PATH, DOC_TIMESTAMP, b"")
        == DOC_GET_SIGNATURE
    )
    assert (
        keys.sign(DOC_SECRET, "GET", DOC_GET_PATH, DOC_TIMESTAMP, b"")
        == DOC_GET_SIGNATURE
    )


def test_the_query_string_is_inside_the_signature():
    """A signature lifted from one filter must not work on another.

    Without the query string in the canonical path, a captured signature for
    ``?limit=1`` would authorise ``?limit=100000``.
    """
    one = arena_client.sign(DOC_SECRET, "GET", "/v1/account/fills?limit=1", "1", b"")
    other = arena_client.sign(
        DOC_SECRET, "GET", "/v1/account/fills?limit=100000", "1", b""
    )
    assert one != other


# --------------------------------------------------------------------------
# What goes on the wire is what was signed
# --------------------------------------------------------------------------


def test_a_signed_request_verifies_against_the_venues_own_key_store():
    """End to end through the real verifier, with no server in the way.

    If the client and the exchange disagree anywhere -- header spelling,
    timestamp rendering, path assembly, body bytes -- ``KeyStore.verify``
    raises, because that is exactly what it is for.
    """
    store = keys.KeyStore()
    key = store.issue("mm-1", label="test")
    client, seen = signed_stub(
        REAL_ACCOUNT, store, key_id=key.key_id, secret=key.secret
    )
    client.account()
    assert seen["key"].key_id == key.key_id


def test_a_signed_post_verifies_with_its_body():
    """The body is part of the signature, so it has to survive the trip."""
    store = keys.KeyStore()
    key = store.issue("mm-1")
    client, seen = signed_stub(
        {"order_id": 5083}, store, key_id=key.key_id, secret=key.secret
    )
    client.place_order("VANTA_OBJECTIVE_WR", "buy", 5, price=Decimal("4663.00"))
    request = seen["request"]
    assert request.content == DOC_BODY_BYTES
    assert request.headers["content-type"] == "application/json"


def test_a_signed_request_with_a_query_string_verifies():
    """The path signed includes the query, so the one sent must too."""
    store = keys.KeyStore()
    key = store.issue("mm-1")
    client, seen = signed_stub([], store, key_id=key.key_id, secret=key.secret)
    client.fills(limit=50)
    assert seen["request"].url.raw_path == b"/v1/account/fills?limit=50"


def test_a_venue_served_under_a_path_prefix_signs_the_prefix():
    """The server verifies the path it receives, prefix and all.

    Signing the bare ``/v1/account`` behind a mount point produces a request
    that looks correct in every visible way and is refused, with a 401 that
    says nothing about which of the two possible causes it was.
    """
    store = keys.KeyStore()
    key = store.issue("mm-1")
    client, seen = signed_stub(
        REAL_ACCOUNT,
        store,
        base_url="http://venue.test/arena",
        key_id=key.key_id,
        secret=key.secret,
    )
    client.account()
    assert seen["request"].url.raw_path == b"/arena/v1/account"


def test_an_empty_parameter_is_dropped_rather_than_sent_empty():
    """``?depth=`` and no query at all are two different signed strings."""
    client, seen = stub(REAL_BOOK)
    client.book("VANTA_OBJECTIVE_WR")
    assert seen["request"].url.raw_path == b"/v1/instruments/VANTA_OBJECTIVE_WR/book"


def test_a_public_endpoint_needs_no_credential():
    """Market data is public, and asking for a key to read it would be theatre."""
    client, _ = stub(REAL_BOOK)
    assert client.authenticated is False
    assert client.book("VANTA_OBJECTIVE_WR", depth=5)["symbol"] == "VANTA_OBJECTIVE_WR"


def test_a_signed_endpoint_without_a_credential_fails_before_the_request():
    """Spending a round trip to be told the obvious is worse than not trying."""
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, content=b"{}")

    client = ArenaClient("http://venue.test", transport=httpx.MockTransport(handler))
    with pytest.raises(ClientError) as raised:
        client.account()
    assert raised.value.code == "client_credentials_missing"
    assert sent == []


def test_the_signed_headers_are_the_three_the_venue_reads():
    client, seen = stub(REAL_ACCOUNT, key_id=DOC_KEY_ID, secret=DOC_SECRET)
    client.account()
    headers = seen["request"].headers
    assert headers[keys.HEADER_KEY] == DOC_KEY_ID
    assert int(headers[keys.HEADER_TIMESTAMP]) > 0
    assert headers[keys.HEADER_SIGNATURE] == keys.sign(
        DOC_SECRET, "GET", "/v1/account", headers[keys.HEADER_TIMESTAMP], b""
    )


def test_the_stream_auth_frame_signs_the_stream_path():
    """One implementation of the signature serves both transports."""
    client = ArenaClient("http://venue.test", key_id=DOC_KEY_ID, secret=DOC_SECRET)
    frame = client.stream_auth()
    assert frame["op"] == "auth"
    assert frame["key_id"] == DOC_KEY_ID
    assert frame["signature"] == keys.sign(
        DOC_SECRET, "GET", "/v1/stream", frame["timestamp"], b""
    )
    assert client.stream_url() == "ws://venue.test/v1/stream"
    assert ArenaClient("https://x.test").stream_url() == "wss://x.test/v1/stream"


# --------------------------------------------------------------------------
# Money is Decimal, and nothing is a float
# --------------------------------------------------------------------------


def test_book_prices_come_back_as_decimal():
    """A ladder is prices, and a price compared as a float is not a price."""
    client, _ = stub(REAL_BOOK)
    book = client.book("VANTA_OBJECTIVE_WR", depth=5)
    best_bid = book["bids"][0]
    assert isinstance(best_bid, Level)
    assert best_bid.price == Decimal("4689.00")
    assert isinstance(best_bid.price, Decimal)
    assert best_bid.quantity == 22
    # Still a pair, so code written against the raw payload keeps working.
    assert tuple(best_bid) == (Decimal("4689.00"), 22)
    assert book["asks"][0].price == Decimal("4693.25")


def test_the_spread_is_exact():
    """The arithmetic a maker actually does, done without error.

    A float subtraction of two prices on a quarter grid is very nearly right,
    and "very nearly" is how a quote lands one tick off the grid and is
    rejected by a venue that will not round it back on.
    """
    client, _ = stub(REAL_BOOK)
    book = client.book("VANTA_OBJECTIVE_WR")
    spread = book["asks"][0].price - book["bids"][0].price
    assert spread == Decimal("4.25")
    assert spread % Decimal("0.25") == 0


def test_balances_come_back_as_decimal():
    client, _ = stub(REAL_ACCOUNT, key_id=DOC_KEY_ID, secret=DOC_SECRET)
    account = client.account()
    for field in (
        "cash",
        "starting_cash",
        "posted_collateral",
        "free_cash",
        "realized_pnl",
        "unrealized_pnl",
        "equity",
    ):
        assert isinstance(account[field], Decimal), field
    assert account["cash"] == Decimal("128672864.139831")
    assert account["realized_pnl"] == Decimal("-6633285.860169")


def test_an_average_price_keeps_every_digit_the_venue_published():
    """The concrete cost of a float, measured on a real figure.

    Average cost is a ratio of two exact integers, so it is as long as it needs
    to be. This one is twenty-eight significant digits; the shortest numeral
    that recovers the float is ``3479.328892044944``, sixteen of them. The
    twelve it drops are not noise -- they are what makes
    ``basis - closed_basis`` reconstruct the original exactly, which is the
    property the whole ledger rests on.
    """
    client, _ = stub(REAL_ACCOUNT, key_id=DOC_KEY_ID, secret=DOC_SECRET)
    position = client.account()["positions"][0]
    published = "3479.328892044943820224719101"
    assert position["average_price"] == Decimal(published)
    assert str(position["average_price"]) == published
    assert Decimal(repr(float(published))) != Decimal(published)


def test_no_float_appears_anywhere_in_a_parsed_response():
    """The invariant that does not depend on knowing the field names.

    A schema-aware conversion can only protect the fields it knows about. This
    holds for every field, including ones added to the venue after this client
    shipped, because the parser itself is never allowed to produce a float.
    """
    client, _ = stub(REAL_ACCOUNT, key_id=DOC_KEY_ID, secret=DOC_SECRET)
    assert floats_in(client.account()) == []

    client, _ = stub(REAL_BOOK)
    assert floats_in(client.book("VANTA_OBJECTIVE_WR")) == []

    client, _ = stub({"trades": [REAL_TRADE]})
    tape = client.trades("VANTA_OBJECTIVE_WR", limit=1)
    assert floats_in(tape) == []
    assert tape["trades"][0]["price"] == Decimal("4694.00")


def test_a_bare_json_number_becomes_a_decimal_not_a_float():
    """Even in a field this client has never heard of.

    ``4669.25`` written as a JSON number would be a float under the default
    parser. It is read from the literal digits instead, so the value is the one
    that was published rather than the nearest binary approximation of it.
    """
    client, _ = stub({"symbol": "VANTA_OBJECTIVE_WR", "settles_at": 4669.25, "band": 0.05})
    payload = client.instrument("VANTA_OBJECTIVE_WR")
    assert payload["settles_at"] == Decimal("4669.25")
    assert isinstance(payload["settles_at"], Decimal)
    assert payload["band"] == Decimal("0.05")
    assert Decimal("0.05") != Decimal(0.05)


def test_a_settlement_bound_in_exponent_notation_parses_exactly():
    """``Decimal`` reads "1E+4"; ``int()`` refuses it outright.

    This is not hypothetical formatting: a future bounded at ten thousand
    serialises its upper bound exactly that way, because that is what
    ``str(Decimal)`` does with a scaled value.
    """
    client, _ = stub(REAL_INSTRUMENT)
    instrument = client.instrument("VANTA_OBJECTIVE_WR")
    low, high = instrument["settlement_bounds"]
    assert (low, high) == (Decimal(0), Decimal(10_000))
    assert instrument["tick_size"] == Decimal("0.25")


def test_a_price_that_is_absent_stays_absent():
    """A market order has no price, and ``None`` is the honest answer.

    Coercing it to zero would put a fill at the bottom of the settlement range
    into a blotter, which is a real number and a false one.
    """
    client, _ = stub(
        {"type": "ack", "order_id": 5083, "side": "buy", "quantity": 5, "price": None},
        key_id=DOC_KEY_ID,
        secret=DOC_SECRET,
    )
    ack = client.place_order("VANTA_OBJECTIVE_WR", "buy", 5)
    assert ack["price"] is None


def test_a_field_this_client_does_not_recognise_is_handed_back_verbatim():
    """Lossless, not converted. Guessing that a string is money corrupts ids."""
    client, _ = stub(REAL_INSTRUMENT)
    instrument = client.instrument("VANTA_OBJECTIVE_WR")
    assert instrument["spec_digest"] == REAL_INSTRUMENT["spec_digest"]
    assert instrument["contract_id"] == "VANTA_OBJECTIVE_WR"
    assert instrument["lot_size"] == 1


def test_a_float_price_argument_is_refused():
    """The mirror of parsing exactly: sending exactly.

    Accepting ``4663.25`` as a float would send whatever the binary literal
    rounds to, which is the failure this client exists to prevent, arriving
    through the front door.
    """
    client, _ = stub({}, key_id=DOC_KEY_ID, secret=DOC_SECRET)
    with pytest.raises(TypeError, match="float"):
        client.place_order("VANTA_OBJECTIVE_WR", "buy", 1, price=4663.25)


def test_a_price_goes_on_the_wire_as_the_exact_numeral_it_was_given():
    client, seen = stub({}, key_id=DOC_KEY_ID, secret=DOC_SECRET)
    client.place_order("VANTA_OBJECTIVE_WR", "buy", 1, price=Decimal("4663.250"))
    body = json.loads(seen["request"].content)
    assert body["price"] == "4663.250"

    client, seen = stub({}, key_id=DOC_KEY_ID, secret=DOC_SECRET)
    client.place_order("CROW_GT47", "sell", 2, price="0.45")
    assert json.loads(seen["request"].content)["price"] == "0.45"


def test_a_mistyped_side_is_refused_at_the_call_site():
    """The failure this prevents is silent, which is why it is checked at all.

    The browser front end carries a note about the version that read anything
    not exactly ``buy`` as a sell: a typo sold instead of bought, and nothing
    said so.
    """
    client, _ = stub({}, key_id=DOC_KEY_ID, secret=DOC_SECRET)
    with pytest.raises(ValueError, match="buy or sell"):
        client.place_order("VANTA_OBJECTIVE_WR", "BID", 1, price="4663.00")
    with pytest.raises(TypeError):
        client.place_order("VANTA_OBJECTIVE_WR", "buy", True, price="4663.00")
    with pytest.raises(ValueError):
        client.place_order("VANTA_OBJECTIVE_WR", "buy", 0, price="4663.00")


def test_a_non_json_token_in_a_response_is_refused():
    """``NaN`` is not JSON, and a size that is not equal to itself is not a size."""
    client = ArenaClient(
        "http://venue.test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b'{"mid": NaN}')
        ),
    )
    with pytest.raises(ClientError) as raised:
        client.history("VANTA_OBJECTIVE_WR")
    assert raised.value.code == "client_unreadable_response"


# --------------------------------------------------------------------------
# Failures arrive as codes, not as prose
# --------------------------------------------------------------------------


@pytest.mark.parametrize("code", sorted(errors.ERRORS))
def test_every_documented_error_code_becomes_a_typed_exception(code):
    """Driven off the venue's own catalogue, so a new code is covered on sight."""
    message, status = errors.ERRORS[code]
    body = errors.ApiError(code, message, status).body()
    client, _ = stub(body, status=status, key_id=DOC_KEY_ID, secret=DOC_SECRET)
    with pytest.raises(ArenaError) as raised:
        client.orders()
    assert raised.value.code == code
    assert raised.value.message == message
    assert raised.value.status == status


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("auth_required", AuthError),
        ("auth_invalid", AuthError),
        ("invalid_price", InvalidRequest),
        ("invalid_quantity", InvalidRequest),
        ("rejected_by_venue", Rejected),
        ("not_found", NotFound),
        ("rate_limited", RateLimited),
    ],
)
def test_the_groupings_a_trading_loop_actually_branches_on(code, expected):
    """auth: stop. invalid: fix and resend. rejected: maybe later. limited: wait."""
    message, status = errors.ERRORS[code]
    client, _ = stub(
        errors.ApiError(code, message, status).body(),
        status=status,
        key_id=DOC_KEY_ID,
        secret=DOC_SECRET,
    )
    with pytest.raises(expected):
        client.orders()


def test_a_code_this_client_has_never_seen_still_arrives_with_its_code():
    """Unrecognised is not unusable. The code is the contract; classes are sugar."""
    client, _ = stub(
        {"error": {"code": "halted_symbol", "message": "trading is halted"}},
        status=409,
        key_id=DOC_KEY_ID,
        secret=DOC_SECRET,
    )
    with pytest.raises(ArenaError) as raised:
        client.orders()
    assert raised.value.code == "halted_symbol"
    assert type(raised.value) is ArenaError


def test_the_detail_a_refusal_carries_survives():
    """``invalid_price`` is far more useful with the grid it wanted attached."""
    body = errors.error_body(
        "invalid_price",
        "price must be a number on the instrument's tick grid",
        symbol="VANTA_OBJECTIVE_WR",
        tick_size="0.25",
    )
    client, _ = stub(body, status=400, key_id=DOC_KEY_ID, secret=DOC_SECRET)
    with pytest.raises(InvalidRequest) as raised:
        client.place_order("VANTA_OBJECTIVE_WR", "buy", 1, price="4663.30")
    assert raised.value.detail["tick_size"] == "0.25"
    assert raised.value.detail["symbol"] == "VANTA_OBJECTIVE_WR"


def test_a_failure_with_no_envelope_is_still_one_exception_type():
    """A proxy's HTML 502 is not a venue refusal, and must not look like one.

    It arrives as an ``ArenaError`` so one ``except`` still covers the loop,
    under a ``client_`` code so a log can tell the two apart.
    """
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(502, content=b"<html>bad gateway</html>")

    client = ArenaClient(
        "http://venue.test",
        key_id=DOC_KEY_ID,
        secret=DOC_SECRET,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ClientError) as raised:
        client.orders()
    assert raised.value.code == "client_unreadable_response"
    assert raised.value.status == 502


def test_an_error_envelope_delivered_with_a_success_status_still_raises():
    """The optimistic reading of a refusal is the dangerous one.

    A loop that treats it as success believes an order is resting that is not,
    and finds out when it tries to cancel it.
    """
    client, _ = stub(
        errors.ApiError("rejected_by_venue", "the venue refused this order").body(),
        status=200,
        key_id=DOC_KEY_ID,
        secret=DOC_SECRET,
    )
    with pytest.raises(Rejected):
        client.place_order("VANTA_OBJECTIVE_WR", "buy", 1, price="4663.00")


def test_a_transport_failure_is_reported_as_one():
    """"The venue is unreachable" and "the venue said no" are different problems."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = ArenaClient("http://venue.test", transport=httpx.MockTransport(handler))
    with pytest.raises(ClientError) as raised:
        client.exchange()
    assert raised.value.code == "client_transport"


# --------------------------------------------------------------------------
# The boundary
# --------------------------------------------------------------------------


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_the_client_imports_nothing_from_the_exchange():
    """What makes it a client rather than a second copy of the venue.

    Somebody installing this has httpx and the standard library, and does not
    have ``python/arena``. Importing the signing code from the exchange would
    make the cross-check above trivially true and the package undeliverable.
    """
    roots: set[str] = set()
    for path in sorted(CLIENT_ROOT.rglob("*.py")):
        roots |= _imported_roots(path)
    assert "arena" not in roots
    assert "dashboard" not in roots
    third_party = roots - set(sys.stdlib_module_names) - {"arena_client"}
    assert third_party == {"httpx"}, third_party
