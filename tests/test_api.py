"""The programmatic surface, exercised the way a trading client would drive it.

Everything here runs against a real market (a real kernel, real agents, real
latency) because the claims worth testing about this API are claims about a
market. A mocked venue would pass every one of them and prove nothing: that an
order reaches the book, that it appears in the account that sent it and in no
other, that value is still conserved afterwards, and that a credential issued
before a rebuild still trades its own seat after one.

The app is built here rather than imported from ``dashboard.server``. The
router is meant to be mountable by anything, and a test that could only reach it
through one particular application would not be testing that.

Time is driven by the test, not by a wall clock. ``LiveMarket.step`` advances
simulated time in proportion to elapsed real time, which makes a test's runtime
decide how much market it gets, so these drive ``Kernel.advance`` directly and
ask for an exact number of simulated milliseconds. Slower machines then run the
same market rather than a shorter one.
"""

from __future__ import annotations

import itertools
import json
import time
from decimal import Decimal

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from arena.api import rest
from arena.api.keys import (
    HEADER_KEY,
    HEADER_SIGNATURE,
    HEADER_TIMESTAMP,
    KeyStore,
    body_bytes,
    sign,
)
from arena.exchange.types import PegReference, TimeInForce
from arena.market.instrument import InstrumentClass
from arena.portfolio.money import from_money
from arena.sim.time import Timestamp
from dashboard.state import MarketConfig, MarketRunner

# The nine classes the venue lists. Read off ``InstrumentClass`` rather than
# typed out, so a tenth added later fails this file instead of quietly escaping
# it. That is the whole point of a test that says "every asset class".
ALL_CLASSES = sorted(
    value
    for name, value in vars(InstrumentClass).items()
    if not name.startswith("_") and isinstance(value, str)
)

# Distinct browser sessions per test. A seat token is supposed to be stable and
# unique per person, and the router remembers names against tokens for the
# lifetime of the process, so reusing one string across tests would have two
# tests sharing a seat, which is the exact confusion these tests exist to
# rule out.
_TOKENS = itertools.count()


def _token(prefix: str) -> str:
    return f"{prefix}-{next(_TOKENS)}"


# --------------------------------------------------------------------------
# The harness
# --------------------------------------------------------------------------


class Exchange:
    """A market, a router mounted over it, and a client that can sign.

    Bundled rather than passed around as five fixtures because they are one
    thing: the router is configured against this runner and this key store, and
    a test holding a client for one and a runner for another would be testing
    an arrangement that cannot exist.
    """

    def __init__(self, config: MarketConfig | None = None, **hooks) -> None:
        self.runner = MarketRunner(config or MarketConfig(opening_auction=False))
        self.runner.start()
        self.keys = KeyStore()
        self.seats: dict[str, rest.Seat] = {}

        def browser_seat(request) -> rest.Seat | None:
            """Who is at the browser, as this test app decides it.

            The dashboard reads a signed cookie. Nothing about the router
            depends on that, so the test app uses a header instead: the point
            of the hook is that the application, not this module, says how a
            browser session is recognised.
            """
            token = request.headers.get("x-test-session", "")
            return self.seats.get(token)

        app = FastAPI()
        app.include_router(rest.router)
        self._hooks = {
            "browser_seat": hooks.get("browser_seat", browser_seat),
            # Passed explicitly, and as a function rather than as ``None``,
            # because :func:`rest.configure` deliberately ignores a missing
            # argument: a test that left this set from the previous test would
            # be running against a delegation it never asked for.
            "seat_now": hooks.get("seat_now", lambda token: None),
        }
        self._client = TestClient(app)

    @property
    def client(self) -> TestClient:
        """The HTTP client, with the router pointed at *this* market first.

        The router's configuration is module state, because the application
        that mounts it has exactly one market. A test file has several, so
        every request re-asserts which one it belongs to rather than relying on
        whichever test ran last; otherwise the order the tests happen to run in
        decides which exchange a request reaches, which is a test suite that
        passes for the wrong reason.
        """
        rest.configure(keys=self.keys, runner=self.runner, **self._hooks)
        return self._client

    # -- time ------------------------------------------------------------

    def pump(self, milliseconds: int = 250, slices: int = 10) -> None:
        """Advance the market by an exact amount of simulated time.

        Sliced, and each slice followed by ``runner.step``, so the history
        buffer the chart endpoint reads gets samples at distinct timestamps
        rather than one repeated one. ``runner.step`` advances almost nothing
        on its own here: it scales elapsed wall clock, and no wall clock has
        elapsed.
        """
        step = int(milliseconds * 1_000_000 / slices)
        for _ in range(slices):
            target = Timestamp(int(self.runner.market.kernel.now) + step)
            self.runner.market.kernel.advance(until=target, max_events=500_000)
            self.runner.step()

    # -- identity --------------------------------------------------------

    def browser(self, name: str = "Ash") -> str:
        """Seat a browser session and return the header value that names it."""
        token = _token("sid")
        self.seats[token] = rest.Seat(token=token, name=name)
        return token

    def issue(self, token: str, label: str = "algo") -> dict:
        response = self.client.post(
            "/v1/keys", json={"label": label}, headers={"x-test-session": token}
        )
        assert response.status_code == 201, response.text
        return response.json()

    def trader(self, name: str = "Ash", label: str = "algo") -> "Client":
        """A browser session, a key minted from it, and a signing client."""
        token = self.browser(name)
        return Client(self, self.issue(token, label), token)

    # -- reading ---------------------------------------------------------

    @property
    def venue(self):
        return self.runner.market.venue

    def symbols(self) -> tuple[str, ...]:
        return self.venue.registry.symbols

    def symbol_of(self, instrument_class: str) -> str:
        for symbol in self.symbols():
            if self.venue.registry.require(symbol).instrument_class == instrument_class:
                return symbol
        raise AssertionError(f"nothing listed in class {instrument_class}")

    def restable(self) -> str:
        """A symbol whose book behaves predictably enough to test order handling on.

        Everything here trades, but not everything trades the same way. A
        contract bounded by [0, 1] travels its whole range within a session, so
        its touch and its depth move by amounts that are large next to the
        contract itself. A test that wants an order to sit still, or a market
        order to fill against resting size, is then measuring the weather. A
        win-rate future bounded by [0, 10,000] and quoted around 2,000 is not
        subject to that.

        The first symbol this venue lists is now an event binary, which is why
        tests that had been asking for `symbols()[0]` for years began failing
        on ordinary price moves and reading like API defects.

        So a test whose subject is order handling asks for this. One that only
        needs *a* symbol keeps asking for `symbols()[0]`.
        """
        return self.symbol_of("future")

    def close(self) -> None:
        self._client.close()


class Client:
    """A signing client, in the shape a client library would take.

    Written out here rather than imported so the test proves the documented
    scheme (timestamp, method, path with query, raw body) and not whatever some
    helper happens to do.
    """

    def __init__(self, exchange: Exchange, key: dict, token: str = "") -> None:
        self.exchange = exchange
        self.key_id = key["key_id"]
        self.secret = key["secret"]
        self.account_id = key["account_id"]
        self.token = token

    def headers(
        self,
        method: str,
        path: str,
        body: bytes = b"",
        *,
        timestamp: str | None = None,
        secret: str | None = None,
    ) -> dict[str, str]:
        stamp = timestamp if timestamp is not None else f"{time.time():.3f}"
        query = path.partition("?")[2]
        signed = rest.signed_path(path.partition("?")[0], query)
        return {
            HEADER_KEY: self.key_id,
            HEADER_TIMESTAMP: stamp,
            HEADER_SIGNATURE: sign(secret or self.secret, method, signed, stamp, body),
        }

    def request(self, method: str, path: str, payload=None, **overrides):
        body = body_bytes(payload) if payload is not None else b""
        headers = self.headers(method, path, body, **overrides)
        if payload is None:
            return self.exchange.client.request(method, path, headers=headers)
        headers["content-type"] = "application/json"
        return self.exchange.client.request(method, path, content=body, headers=headers)

    def get(self, path: str, **overrides):
        return self.request("GET", path, None, **overrides)

    def post(self, path: str, payload, **overrides):
        return self.request("POST", path, payload, **overrides)

    def delete(self, path: str, **overrides):
        return self.request("DELETE", path, None, **overrides)


@pytest.fixture(scope="module")
def exchange():
    """One market for the read-only tests, warmed until it has traded.

    Module-scoped because building and warming a market costs more than the
    tests that read from it, and none of them mutate it in a way another can
    see: each one signs in as its own seat.
    """
    venue = Exchange()
    venue.pump(600, slices=24)
    yield venue
    venue.close()


@pytest.fixture
def fresh():
    """A market of this test's own, for anything that writes to one."""
    made: list[Exchange] = []

    def build(config: MarketConfig | None = None, **hooks) -> Exchange:
        venue = Exchange(config, **hooks)
        made.append(venue)
        return venue

    yield build
    for venue in made:
        venue.close()


# How many one-second candles the candle tests want in front of them. Twelve is
# enough for a gap to be visible if there were one, for a quiet period to turn
# up somewhere on forty-seven thin books, and for the ten-second period to have
# closed at least once.
CANDLES_WANTED = 12


def closed_candles(venue: Exchange, period: int = 1) -> int:
    """How many candles of one period this market has finished, through the API."""
    symbol = venue.symbols()[0]
    response = venue.client.get(f"/v1/instruments/{symbol}/candles?period={period}")
    assert response.status_code == 200, response.text
    return response.json()["total"]


@pytest.fixture(scope="module")
def candled():
    """A market pumped until whole candle periods have closed.

    Separate from ``exchange``, which pumps 600 simulated milliseconds. A candle
    is published when its period *ends*, so a market that has run for less than
    one simulated second has no candles at all, and every assertion below would
    pass over an empty list.

    Pumped in a loop against the ring's own length rather than for a fixed
    number of milliseconds, for the reason ``test_dashboard``'s history test
    gives: ``MarketRunner.step`` also advances simulated time in proportion to
    elapsed *wall* clock, so how much market a fixed pump buys depends on how
    loaded the machine is. Asking for candles until there are candles is the
    same market everywhere; asking for milliseconds is not.
    """
    venue = Exchange()
    venue.pump(1_000, slices=20)
    for _ in range(12):
        if closed_candles(venue) >= CANDLES_WANTED:
            break
        venue.pump(2_000, slices=40)
    yield venue
    venue.close()


def busiest(venue: Exchange) -> str:
    return max(venue.symbols(), key=lambda s: len(venue.venue.engine(s).tape))


def quietest(venue: Exchange) -> str:
    return min(venue.symbols(), key=lambda s: len(venue.venue.engine(s).tape))


def candles(venue: Exchange, symbol: str, query: str = "period=1&limit=1000") -> dict:
    response = venue.client.get(f"/v1/instruments/{symbol}/candles?{query}")
    assert response.status_code == 200, response.text
    return response.json()


def floats_in(node, path: str = "") -> list[str]:
    """Every place a JSON float appears under this node, named by where it is.

    Named rather than counted so a failure says which field went wrong. ``bool``
    is checked first because it is an ``int`` in Python and would otherwise be
    reported as neither.
    """
    if isinstance(node, bool):
        return []
    if isinstance(node, float):
        return [path or "<root>"]
    if isinstance(node, dict):
        return [hit for key, value in node.items() for hit in floats_in(value, f"{path}.{key}")]
    if isinstance(node, list):
        return [hit for i, value in enumerate(node) for hit in floats_in(value, f"{path}[{i}]")]
    return []


def resting_price(
    exchange: Exchange, symbol: str, side: str = "buy", inside: int = 1
) -> Decimal:
    """A price that will rest rather than trade, on the instrument's own grid.

    ``inside`` increments in from the far end of what the contract can settle
    at. A bid there can only be filled by somebody selling a claim for close to
    the least it could possibly be worth, and the venue's own price band
    refuses to match that far from the reference in any case.

    It used to be halfway between the touch and that bound, which is a
    different rule that happens to agree on wide contracts and fails badly on
    narrow ones. Halfway is out of reach on a win-rate future bounded by
    [0, 10,000] and quoted at 2,003. It is nothing of the kind on a contract
    whose range *is* the distance its price travels: measured, a bid at 0.47 on
    an event binary bounded by [0, 1], parked against a touch of 0.95, was
    filled inside a single pump, and a call option's bid went the same way.
    Both are ordinary price moves rather than anything wrong with the venue,
    which is what made the failures read as API defects.

    ``inside=0`` is the bound itself, and it is the only price that rests on
    *every* class rather than on almost all of them. Measured on
    `EMBER_OBJECTIVE_P5000`, a put bounded by [0, 5,000] whose whole traded
    range sits at the bottom of that: its book was 0.00 bid at 1.50 offered, so
    one increment inside the floor is 0.25, which is inside the spread, and it
    filled within one pump. There is no room at all on that contract, because
    the floor is where the bid already is.

    The default is one increment in anyway, because the floor is also where a
    long stops costing anything: collateral is the worst case over the
    settlement range, so a bid exactly at the floor reserves zero, and the
    tests that measure what an account can fund would be dividing by it. A
    caller that needs the price to rest on every class asks for ``inside=0``
    and does not measure collateral against it.

    Derived per instrument rather than chosen, because a constant would be a
    price on one contract and off the grid of another. These are listed on
    three different increments over five different settlement ranges, one of
    them negative at the bottom, and one carries a tick *table* whose increment
    changes with the level, so the price is snapped by the same repeated walk
    ``TradingAgent`` uses, since a single pass can round into a coarser band
    and land off its grid.
    """
    instrument = exchange.venue.registry.require(symbol)
    low, high = instrument.value_bounds
    if side == "buy":
        target = low + inside * instrument.increment_at(low)
    else:
        target = high - inside * instrument.increment_at(high)

    # Rounded away from the market on both sides, so snapping never makes the
    # order more aggressive than the test intended.
    for _ in range(8):
        increment = instrument.increment_at(target)
        remainder = target % increment
        if remainder == 0:
            return target
        target = target - remainder if side == "buy" else target + increment - remainder
    raise AssertionError(f"no on-grid price found for {symbol}")


# --------------------------------------------------------------------------
# Public: the exchange describes itself
# --------------------------------------------------------------------------


def test_the_exchange_endpoint_answers_without_a_credential(exchange):
    """Reference data is public. A client should be able to see what is listed
    before it decides whether to ask for a key."""
    payload = exchange.client.get("/v1/exchange").json()
    assert payload["counts"]["instruments"] == len(exchange.symbols())
    assert payload["clock"] > 0
    assert payload["generation"] == 0
    assert set(payload["session"]) >= {"phases", "fees", "price_band", "message_rate"}
    assert sum(payload["session"]["phases"].values()) == len(exchange.symbols())


def test_the_exchange_publishes_conservation_as_an_exact_integer(exchange):
    """Not as a price. The claim is that it is *exactly* zero, and a rounded
    zero would be a different and much weaker claim."""
    conservation = exchange.client.get("/v1/exchange").json()["conservation"]
    assert int(conservation) == 0
    assert "." not in conservation


def test_every_asset_class_is_quotable_through_the_instruments_endpoint(exchange):
    """The venue is uniform over its classes and so is this API.

    Nothing in the router branches on a class, so this is really a test that
    nothing has quietly started to: each of the nine has to arrive with a tick,
    bounds, an expiry, a session and a mark, through the same code path.
    """
    rows = exchange.client.get("/v1/instruments?limit=1000").json()["instruments"]
    listed = {row["class"] for row in rows}
    assert listed == set(ALL_CLASSES), f"missing {set(ALL_CLASSES) - listed}"
    for row in rows:
        assert Decimal(row["tick"]) > 0
        low, high = (Decimal(bound) for bound in row["bounds"])
        assert low <= high
        assert row["expiry"].endswith("Z")
        assert row["session"]
        # Inside the contract's own range, which is the invariant that was
        # meant all along. This read `mark >= 0`, and that held only because no
        # listed contract had ever been worth less than nothing. A spread is
        # one leg minus another and is bounded below by a negative number, so
        # the moment the roster made one price under zero, a correct mark of
        # -1,693 failed a test that was really asserting the old listing.
        assert low <= Decimal(row["mark"]) <= high, (row["symbol"], row["mark"])
        assert row["subjects"]


def test_a_price_from_the_instruments_endpoint_is_never_a_json_number(exchange):
    """Prices cross the wire as strings, in both directions. A JSON number is a
    double, and this venue's accounting is exact integers precisely so that the
    conservation check can be exactly zero."""
    raw = json.loads(exchange.client.get("/v1/instruments?limit=1000").text)
    for row in raw["instruments"]:
        for money in ("tick", "mark", "bid", "ask"):
            assert row[money] is None or isinstance(row[money], str), (row["symbol"], money)


@pytest.mark.parametrize("instrument_class", ALL_CLASSES)
def test_the_class_filter_selects_exactly_that_class(exchange, instrument_class):
    payload = exchange.client.get(
        f"/v1/instruments?class={instrument_class}&limit=1000"
    ).json()
    assert payload["instruments"], f"nothing listed in {instrument_class}"
    assert {row["class"] for row in payload["instruments"]} == {instrument_class}
    assert payload["filters"]["class"] == instrument_class


def test_the_subject_filter_reaches_every_leg_of_a_multi_leg_contract(exchange):
    """A spread is written on two subjects and an index on a basket of them.

    Filtering by subject has to find a contract by any leg, or the filter means
    something different for a future than for the two classes that are not
    written on one thing, which is exactly the special-casing this venue is
    arranged to avoid.
    """
    rows = exchange.client.get("/v1/instruments?limit=1000").json()["instruments"]
    multi = [row for row in rows if len(row["subjects"]) > 1]
    assert multi, "no multi-leg contract listed"
    for row in multi:
        for subject in row["subjects"]:
            found = exchange.client.get(
                f"/v1/instruments?subject={subject}&limit=1000"
            ).json()["instruments"]
            assert row["symbol"] in {hit["symbol"] for hit in found}


def test_an_unknown_symbol_is_refused_with_a_catalogued_code(exchange):
    for path in (
        "/v1/instruments/NOPE",
        "/v1/instruments/NOPE/book",
        "/v1/instruments/NOPE/trades",
        "/v1/instruments/NOPE/history",
        "/v1/instruments/NOPE/candles?period=1",
        # An unknown symbol is refused before an unreadable period is, because
        # the symbol is the more fundamental mistake: telling a client its
        # period is wrong when the contract does not exist sends it to fix the
        # wrong thing.
        "/v1/instruments/NOPE/candles?period=nonsense",
    ):
        response = exchange.client.get(path)
        assert response.status_code == 400, path
        assert response.json()["error"]["code"] == "invalid_symbol"
        assert "detail" not in response.json(), "a bare FastAPI detail escaped"


def test_one_instrument_carries_the_contract_it_settles_by(exchange):
    """A market where you cannot read the contract is a casino with extra steps."""
    symbol = exchange.symbols()[0]
    payload = exchange.client.get(f"/v1/instruments/{symbol}").json()
    assert payload["symbol"] == symbol
    assert payload["contract"]["payoff"]
    assert payload["contract"]["underlying"]
    assert payload["contract"]["window"]["start"] < payload["contract"]["window"]["end"]


def test_an_indicative_price_is_published_during_a_call_and_not_otherwise(fresh):
    """``Venue.indicative`` answers whenever it is asked, which is right for the
    venue and wrong to publish: a continuously trading symbol has no auction,
    and an "indicative price" beside a live book reads as a second opinion on
    where the market is rather than as the answer to a question nobody asked.

    Built without matches, because a live match book does not open with the
    exchange. It opens when its match starts and closes when the match ends,
    which is the whole point of it, so it is trading continuously while the
    statistical listing is still in its opening call. With matches on, 308 of
    the 358 books this market lists are match books and every one of them
    fails an assertion that the exchange is uniformly in a call. That is the
    design working, not the venue misbehaving, so the market this test builds
    is the one whose opening the test is about.
    """
    calling = fresh(MarketConfig(opening_auction=True, matches=False))
    calling.pump(900, slices=18)
    quoted = [
        calling.client.get(f"/v1/instruments/{symbol}").json()
        for symbol in calling.symbols()
    ]
    assert all(row["session"] != "continuous" for row in quoted)
    assert any(row["indicative"] for row in quoted), "no auction ever formed"
    for row in quoted:
        if row["indicative"]:
            assert isinstance(row["indicative"]["price"], str)
            assert row["indicative"]["quantity"] > 0

    # And the exchange summary says the same thing about the same books. A halt
    # does not produce a phase called "halted": it produces an auction, which
    # is also where the opening call lives, so what a client is told is which
    # books will not trade its order right now.
    assert set(calling.client.get("/v1/exchange").json()["session"]["not_trading"]) == set(
        calling.symbols()
    )

    trading = fresh(MarketConfig(opening_auction=False, matches=False))
    trading.pump(400, slices=16)
    for symbol in trading.symbols():
        row = trading.client.get(f"/v1/instruments/{symbol}").json()
        assert row["session"] == "continuous"
        assert row["indicative"] is None, symbol
    assert trading.client.get("/v1/exchange").json()["session"]["not_trading"] == []


def test_fees_collected_is_a_price_and_not_the_ledgers_own_unit(exchange):
    """The dashboard publishes the raw minor units and divides by a million in
    JavaScript. That is fine for one page and is the arrangement that put
    "113125513.21M" on the participants table: a raw internal unit under a
    label that promises money, with the conversion living somewhere else."""
    published = exchange.client.get("/v1/exchange").json()["session"]["fees_collected"]
    assert Decimal(published) * 1_000_000 == int(exchange.venue.fees_collected)


def test_the_book_is_two_sided_and_bounded(exchange):
    symbol = exchange.symbols()[0]
    payload = exchange.client.get(f"/v1/instruments/{symbol}/book?depth=5").json()
    assert payload["depth"] == 5
    assert payload["cap"] == rest.BOOK_DEPTH_CAP
    assert len(payload["bids"]) <= 5 and len(payload["asks"]) <= 5
    for price, quantity in payload["bids"] + payload["asks"]:
        assert isinstance(price, str)
        assert quantity > 0
    # Descending on the bid, ascending on the ask: a ladder is only readable if
    # the best price is the first one.
    bids = [Decimal(price) for price, _ in payload["bids"]]
    asks = [Decimal(price) for price, _ in payload["asks"]]
    assert bids == sorted(bids, reverse=True)
    assert asks == sorted(asks)


def test_a_book_deeper_than_the_cap_is_clamped_not_refused(exchange):
    """A client that does not know the cap is served the cap. A client that asks
    for zero levels has a bug, and an empty list would hide it."""
    symbol = exchange.symbols()[0]
    clamped = exchange.client.get(f"/v1/instruments/{symbol}/book?depth=99999").json()
    assert clamped["depth"] == rest.BOOK_DEPTH_CAP

    refused = exchange.client.get(f"/v1/instruments/{symbol}/book?depth=0")
    assert refused.status_code == 400
    assert refused.json()["error"]["code"] == "invalid_request"

    nonsense = exchange.client.get(f"/v1/instruments/{symbol}/book?depth=deep")
    assert nonsense.status_code == 400
    assert nonsense.json()["error"]["code"] == "invalid_request"


def test_the_tape_is_capped_and_most_recent_first(exchange):
    symbol = max(
        exchange.symbols(), key=lambda s: len(exchange.venue.engine(s).tape)
    )
    payload = exchange.client.get(f"/v1/instruments/{symbol}/trades?limit=5").json()
    assert payload["limit"] == 5
    assert payload["cap"] == rest.TRADES_CAP
    assert payload["count"] <= 5
    if payload["count"] > 1:
        sequences = [trade["sequence"] for trade in payload["trades"]]
        assert sequences == sorted(sequences, reverse=True)
    for trade in payload["trades"]:
        assert isinstance(trade["price"], str)
        assert trade["aggressor_side"] in ("buy", "sell")


def test_history_publishes_the_sampled_path_as_strings(exchange):
    """The runner keeps this as floats because a chart is a float. It leaves
    here as a string anyway: a client should not have to know which endpoints
    are exact."""
    symbol = exchange.symbols()[0]
    payload = exchange.client.get(f"/v1/instruments/{symbol}/history?limit=50").json()
    assert payload["count"] == len(payload["t"]) == len(payload["mid"])
    assert payload["count"] > 0
    assert payload["cap"] >= payload["count"]
    for mid in payload["mid"]:
        assert mid is None or isinstance(mid, str)
        if mid is not None:
            Decimal(mid)  # exact, or this raises


# --------------------------------------------------------------------------
# Candles
#
# The market-data surface a systematic client actually reads. Everything here
# is measured against Kalshi's candlestick contract rather than invented: three
# OHLC blocks, gap-free periods, a closed period enum, and a refusal rather
# than a truncation when the range asked for is wider than the venue keeps.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("instrument_class", ALL_CLASSES)
def test_every_asset_class_candles_through_one_code_path(candled, instrument_class):
    """Nine classes, one aggregator, no branch anywhere on what kind of claim it is.

    The same test the instruments endpoint gets, for the same reason: the venue
    is uniform over its classes and nothing in the candle path knows what a
    future or an option or a spread is. A class that needed special handling
    here would mean the aggregator had started reading the contract instead of
    the book.
    """
    symbol = candled.symbol_of(instrument_class)
    payload = candles(candled, symbol)
    assert payload["symbol"] == symbol
    assert payload["period"] == 1
    assert payload["period_ns"] == 1_000_000_000
    assert payload["candles"], f"no closed candle for {instrument_class}"
    for candle in payload["candles"]:
        assert isinstance(candle["end"], int)
        assert isinstance(candle["volume"], int) and candle["volume"] >= 0
        assert isinstance(candle["open_interest"], int)
        # Notional is signed, and has to be: it is the sum of price times
        # quantity, so a VWAP derived from it would come out wrong for any
        # contract that trades below zero, and a spread does. Measured, a
        # candle on one came back at -20,300.00, which is the right number and
        # used to fail here. What is asserted instead is the property that
        # means something: no volume means no notional, and where there is
        # volume the implied VWAP sits inside the candle's own price range.
        if candle["volume"]:
            vwap = Decimal(candle["notional"]) / candle["volume"]
            assert Decimal(candle["price"]["low"]) <= vwap, (symbol, vwap)
            assert vwap <= Decimal(candle["price"]["high"]), (symbol, vwap)
        else:
            assert Decimal(candle["notional"]) == 0, symbol
        for block in ("price", "bid", "ask"):
            for edge in ("open", "high", "low", "close"):
                value = candle[block][edge]
                assert value is None or isinstance(value, str), (block, edge)


def test_a_candle_carries_three_separate_ohlc_blocks(candled):
    """Trade price, bid and ask, each candled on its own.

    This is the design decision the endpoint exists for, and it is Kalshi's
    ``price`` / ``yes_bid`` / ``yes_ask`` shape. On a book this thin the last
    print is a fact about whenever somebody last crossed the spread; the quotes
    are facts about the period. A backtester asking "what could I have
    transacted at during that second" is answered by the bid and ask candles
    and is not answered by the trade candle at all. That is why a bid block
    that merely echoed the trade block would be worthless, and why this asserts
    they come apart.
    """
    symbol = busiest(candled)
    payload = candles(candled, symbol)
    assert payload["candles"]

    for candle in payload["candles"]:
        assert set(candle["price"]) == {"open", "high", "low", "close", "mean"}
        assert set(candle["bid"]) == {"open", "high", "low", "close"}
        assert set(candle["ask"]) == {"open", "high", "low", "close"}
        for block in ("price", "bid", "ask"):
            edges = [candle[block][edge] for edge in ("open", "high", "low", "close")]
            if edges[0] is None:
                assert all(edge is None for edge in edges), (symbol, block)
                continue
            low = Decimal(candle[block]["low"])
            high = Decimal(candle[block]["high"])
            assert low <= high
            assert low <= Decimal(candle[block]["open"]) <= high
            assert low <= Decimal(candle[block]["close"]) <= high

    quoted = [
        candle
        for candle in payload["candles"]
        if candle["bid"]["close"] is not None and candle["ask"]["close"] is not None
    ]
    assert quoted, "no candle carried a two-sided quote"
    # A spread. If the bid block were the trade block under another name this
    # would be zero everywhere, and the endpoint would be publishing one series
    # three times.
    assert any(
        Decimal(candle["ask"]["close"]) > Decimal(candle["bid"]["close"])
        for candle in quoted
    )


def test_the_mean_is_the_volume_weighted_price_and_its_numerator_travels_with_it(
    candled,
):
    """``mean`` is a quotient, so the exact numerator is published beside it.

    A volume-weighted average of integers is not always a decimal (100 lots at
    3 and 200 at 4 average to 11/3), so this is the one figure on a candle that
    is rounded. It is published to the same precision the server computed it at
    and ``notional`` carries the exact sum it came from, so a client that needs
    the exactness divides for itself.
    """
    payload = candles(candled, busiest(candled))
    traded = [candle for candle in payload["candles"] if candle["volume"] > 0]
    assert traded, "nothing printed in any period"
    for candle in traded:
        notional = Decimal(candle["notional"])
        assert isinstance(candle["notional"], str)
        assert Decimal(candle["price"]["mean"]) == notional / Decimal(candle["volume"])
        assert (
            Decimal(candle["price"]["low"])
            <= Decimal(candle["price"]["mean"])
            <= Decimal(candle["price"]["high"])
        )

    quiet = [candle for candle in payload["candles"] if candle["volume"] == 0]
    for candle in quiet:
        assert Decimal(candle["notional"]) == 0
        # The mean of nothing is the last thing that traded, which is what the
        # other four price fields carry too. Consistent with its own block
        # rather than null, so a client averaging across bars has no quiet case.
        assert candle["price"]["mean"] == candle["price"]["close"]


def test_no_float_appears_anywhere_in_a_candle_payload(candled):
    """The premise of this whole venue is exact arithmetic, and it starts at the wire.

    A JSON number with a fraction is a binary double. A high and a low that a
    backtester will difference must not be one, or every statistic computed from
    these bars inherits an error the exchange's own ledger does not have.
    """
    symbol = busiest(candled)
    for period in candles(candled, symbol)["periods"]:
        response = candled.client.get(
            f"/v1/instruments/{symbol}/candles?period={period}&limit=1000"
        )
        assert response.status_code == 200
        found = floats_in(json.loads(response.text))
        assert not found, f"floats at period {period}: {found[:5]}"


def test_the_candle_period_is_a_closed_enum(candled):
    """Refused rather than rounded to the nearest period the venue does keep.

    Kalshi accepts 1, 60 and 1440 and refuses 5, 15 and 360. Rounding would be
    the friendlier-looking choice and is the wrong one: a series of bars that
    are not the width that was asked for is wrong in a way nothing downstream
    can detect, because every bar still looks like a bar.
    """
    symbol = candled.symbols()[0]
    supported = candles(candled, symbol)["periods"]
    assert supported == sorted(set(supported)) and supported

    for period in supported:
        accepted = candled.client.get(
            f"/v1/instruments/{symbol}/candles?period={period}"
        )
        assert accepted.status_code == 200, period
        assert accepted.json()["period"] == period

    # Including 5 and 15, which Kalshi refuses for the same reason, and 1440,
    # which is one of *their* periods and is not one of ours.
    for refused in ("5", "15", "0", "-1", "1440", "banana", "1.0", "1e0", " ", ""):
        response = candled.client.get(
            f"/v1/instruments/{symbol}/candles?period={refused}"
        )
        assert response.status_code == 400, refused
        error = response.json()["error"]
        assert error["code"] == "invalid_request", refused
        # The enum comes back in the refusal, so a client learns the closed set
        # from the failure rather than from documentation.
        assert error["detail"]["supported"] == supported, refused
        assert error["detail"]["unit"] == "seconds"

    missing = candled.client.get(f"/v1/instruments/{symbol}/candles")
    assert missing.status_code == 400
    assert missing.json()["error"]["detail"]["supported"] == supported


def test_a_range_wider_than_the_venue_keeps_is_refused_not_truncated(candled):
    """Binance truncates at 1,000 and Coinbase refuses at 300. Refusing is honest.

    A backtester that asks for a day, receives an hour, and is told nothing
    about which hour it lost will compute a statistic over a window it does not
    have. The refusal names the count that was asked for and the count that is
    kept, the way Kalshi's does (``requested time range with candlesticks:
    129600, max candlesticks: 5000``), so the client can narrow the range or
    ask for a longer period without guessing.
    """
    symbol = candled.symbols()[0]
    payload = candles(candled, symbol)
    cap = payload["cap"]
    second = payload["period_ns"]

    # One hundred thousand seconds. Far past the one-second ring, comfortably
    # inside the sixty-second one, which is the advice the refusal gives.
    span = 100_000 * second
    refused = candled.client.get(
        f"/v1/instruments/{symbol}/candles?period=1&start=0&end={span}"
    )
    assert refused.status_code == 400
    error = refused.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["detail"]["requested"] == 100_000
    assert error["detail"]["cap"] == cap
    assert "100000" in error["message"] and str(cap) in error["message"]

    coarser = candled.client.get(
        f"/v1/instruments/{symbol}/candles?period=60&start=0&end={span}"
    )
    assert coarser.status_code == 200, coarser.text

    # The boundary itself: exactly the retained depth is served, one nanosecond
    # more is refused. A cap a client cannot sit against is a cap it has to
    # binary-search for.
    exact = candled.client.get(
        f"/v1/instruments/{symbol}/candles?period=1&start=0&end={cap * second}"
    )
    assert exact.status_code == 200, exact.text
    over = candled.client.get(
        f"/v1/instruments/{symbol}/candles?period=1&start=0&end={cap * second + 1}"
    )
    assert over.status_code == 400
    assert over.json()["error"]["detail"]["requested"] == cap + 1

    backwards = candled.client.get(
        f"/v1/instruments/{symbol}/candles?period=1&start=900&end=100"
    )
    assert backwards.status_code == 400
    assert backwards.json()["error"]["code"] == "invalid_request"


def test_the_series_is_gap_free_and_a_quiet_period_carries_the_previous_close(candled):
    """Every period gets a bar, traded or not. Kalshi does this; the reason is joins.

    A gap-free series lines up with a clock by arithmetic. A sparse one has to
    be reindexed by every client that reads it, and each of those clients then
    writes its own interpolation policy in a hurry, which is how two backtests
    of the same strategy over the same data disagree.

    A quiet period is the case that proves the quote candles are worth having:
    zero volume, the previous close repeated across all five price fields, and a
    *real* bid and ask, because the sampler still saw the book ten times a
    second through a period nobody traded in.
    """
    step = candles(candled, candled.symbols()[0])["period_ns"]

    for symbol in (busiest(candled), quietest(candled)):
        ends = [candle["end"] for candle in candles(candled, symbol)["candles"]]
        assert len(ends) >= 2, symbol
        assert ends == sorted(ends)
        assert ends == list(range(ends[0], ends[0] + step * len(ends), step)), symbol
        # And every stamp is the end of a period rather than an arbitrary moment.
        assert all(end % step == 0 for end in ends), symbol

    # A quiet period that follows a traded one, anywhere on the venue: the
    # forty-seven books here are thin by construction and nine of them went a
    # whole simulated second without a print when this was measured.
    carried = None
    for symbol in candled.symbols():
        series = candles(candled, symbol)["candles"]
        for before, after in zip(series, series[1:]):
            if before["volume"] > 0 and after["volume"] == 0:
                carried = (symbol, before, after)
                break
        if carried:
            break
    assert carried, "no book was quiet for a whole period"

    symbol, before, after = carried
    assert Decimal(after["notional"]) == 0
    assert after["trades"] == 0
    for edge in ("open", "high", "low", "close", "mean"):
        assert after["price"][edge] == before["price"]["close"], (symbol, edge)
    # The point of the exercise: nothing traded, and the client is still told
    # where the market was.
    assert after["bid"]["close"] is not None or after["ask"]["close"] is not None


def test_the_candles_endpoint_publishes_what_it_holds_and_what_it_applied(candled):
    """A client has to be able to tell an empty window from a window that fell off.

    ``oldest`` and ``newest`` say what exists at this period at all, ``cap`` and
    ``retains_ns`` say how far back it can ever reach, and ``count`` against
    ``total`` says whether the page is the whole of it.
    """
    symbol = candled.symbols()[0]
    payload = candles(candled, symbol, "period=1&limit=3")
    assert payload["limit"] == 3
    assert payload["count"] <= 3
    assert payload["cap"] >= payload["limit"]
    assert payload["total"] >= payload["count"]
    assert payload["retains_ns"] == payload["cap"] * payload["period_ns"]
    assert payload["oldest"] <= payload["newest"]
    assert payload["clock"] >= payload["newest"]

    # A window from before this market existed is empty rather than a refusal:
    # nothing is wrong with the request, there is simply nothing there.
    empty = candles(candled, symbol, "period=1&start=0&end=1")
    assert empty["count"] == 0
    assert empty["oldest"] is not None, "the ring should still say what it holds"

    # And the page is the newest of the window, not the oldest, so a default
    # request is about the present.
    wide = candles(candled, symbol, "period=1&limit=1000")
    assert payload["candles"] == wide["candles"][-3:]


# --------------------------------------------------------------------------
# What a client outside this repository actually needs
#
# Two of them exist and neither imports the exchange: a Kalshi-shaped research
# engine that reads a universe, a tape, candles and settlements over HTTP with
# no account at all, and an Alpaca-shaped portfolio manager that trades through
# the nine methods of a broker interface. Everything in this section is a call
# one of them makes and a shape it parses, so a change that breaks one of them
# fails here rather than in somebody else's process.
# --------------------------------------------------------------------------


# How many prints the venue-wide tape tests want in front of them. Enough for
# more than one page at a quarter of the cap, so that paging is exercised
# rather than described.
TRADES_WANTED = 400


def tape_total(venue: Exchange) -> int:
    """How many prints this venue has published, through the API."""
    response = venue.client.get("/v1/trades?limit=1")
    assert response.status_code == 200, response.text
    return response.json()["total"]


@pytest.fixture(scope="module")
def taped():
    """A market pumped until its venue-wide tape is long enough to page.

    Pumped against the tape's own length rather than for a fixed number of
    milliseconds, for the reason the candle fixture gives and then some. How
    much market a fixed pump buys depends on the machine, and how much *tape*
    it buys depends on the listing as well: measured on one roster, a market
    pumped for 500 simulated milliseconds had printed nothing at all and the
    same market at 600 had printed seventy-two. A paging test written against
    a millisecond count passes over an empty feed the moment the contracts
    change, which is a test that reports on the fixture rather than on the API.
    """
    venue = Exchange()
    venue.pump(1_000, slices=20)
    for _ in range(12):
        if tape_total(venue) >= TRADES_WANTED:
            break
        venue.pump(2_000, slices=40)
    assert tape_total(venue) >= TRADES_WANTED, "this market never traded enough to page"
    yield venue
    venue.close()


def settle_now(venue: Exchange, symbol: str):
    """Settle one contract exactly the way the market settles it itself.

    ``LiveMarket.settle_due`` waits for the contract calendar to pass an
    expiry, which on the default season is weeks of simulated time away. This
    calls the same ``settlement_source`` with the same spec and applies the
    result through the same ``Venue.settle``, so what the API is then asked
    about is a genuine settlement and not a fixture pretending to be one.
    """
    instrument = venue.venue.registry.require(symbol)
    result = venue.runner.market.settlement_source(instrument.spec)
    venue.venue.settle(symbol, result)
    return result


def positions_of(trader: "Client") -> dict[str, dict]:
    """This account's position rows, by symbol."""
    payload = trader.get("/v1/account/positions").json()
    return {row["symbol"]: row for row in payload["positions"]}


def working_order(trader: "Client", client_order_id: str) -> dict:
    """One of this account's resting orders, by the id it was placed under."""
    working = trader.get("/v1/orders").json()
    rows = [
        row for row in working["orders"]
        if row["client_order_id"] == client_order_id
    ]
    assert rows, working
    return rows[0]


def test_a_named_bulk_read_answers_a_watchlist_in_one_request(exchange):
    """A client watching three hundred symbols must not make three hundred calls.

    The touch is already on every listing row, which is what makes one request
    enough; ``?symbols=`` is what lets a client ask for its own universe rather
    than downloading the whole listing to throw most of it away. Over a link
    with 20ms of latency the difference is one round trip against six seconds.
    """
    everything = exchange.client.get("/v1/instruments?limit=1000").json()
    rows = everything["instruments"]
    wanted = [rows[0]["symbol"], rows[3]["symbol"], rows[-1]["symbol"]]

    payload = exchange.client.get(
        "/v1/instruments?symbols=" + ",".join(wanted)
    ).json()
    assert {row["symbol"] for row in payload["instruments"]} == set(wanted)
    assert payload["total"] == len(wanted)
    assert payload["missing"] == []
    assert payload["filters"]["symbols"] == sorted(wanted)

    # The quote travels with the row, which is the reason one request is
    # enough. Compared against the unfiltered listing so the filter cannot be
    # serving a cheaper row than the dump does.
    full = {row["symbol"]: row for row in rows}
    for row in payload["instruments"]:
        assert row == full[row["symbol"]], row["symbol"]
        assert {"bid", "ask", "bid_size", "ask_size", "mark"} <= set(row)


def test_a_named_read_comes_back_in_listing_order_however_it_was_asked(exchange):
    """Rows are the venue's order, not the request's, or ``limit`` is unstable.

    A client that pages a filtered listing has to see the same rows in the same
    places every time. If the order came from the query string, a row would move
    between pages because the request happened to spell its name somewhere else,
    and the client would see it twice or not at all.
    """
    listed = list(exchange.symbols())
    wanted = [listed[4], listed[0], listed[2]]
    forwards = exchange.client.get(
        "/v1/instruments?symbols=" + ",".join(wanted)
    ).json()["instruments"]
    backwards = exchange.client.get(
        "/v1/instruments?symbols=" + ",".join(reversed(wanted))
    ).json()["instruments"]

    order = [row["symbol"] for row in forwards]
    assert order == [row["symbol"] for row in backwards]
    assert order == [symbol for symbol in listed if symbol in set(wanted)]


def test_a_name_that_is_not_listed_is_reported_rather_than_refused(exchange):
    """A universe outlives any one market here: contracts expire and a rebuild
    replaces the whole roster. A client that asked for three hundred names and
    got two hundred and ninety needs to know which ten it lost, which neither a
    400 for the whole request nor a silently short list tells it."""
    listed = exchange.symbols()[0]
    payload = exchange.client.get(
        f"/v1/instruments?symbols={listed},NOPE,GONE"
    ).json()
    assert [row["symbol"] for row in payload["instruments"]] == [listed]
    assert payload["missing"] == ["GONE", "NOPE"]

    # Case-insensitively, like the other two filters on this endpoint. A filter
    # that means something different depending on how it is spelled is a filter
    # that will be spelled wrong.
    lowered = exchange.client.get(
        f"/v1/instruments?symbols={listed.lower()}"
    ).json()
    assert [row["symbol"] for row in lowered["instruments"]] == [listed]


def test_a_symbol_list_longer_than_the_listing_is_refused(exchange):
    """``missing`` is built from what the client sent, so an unbounded list of
    names is a response whose size the request chooses, which is a way to spend
    the server's memory from outside it."""
    names = ",".join(f"S{n}" for n in range(rest.INSTRUMENTS_CAP + 1))
    response = exchange.client.get(f"/v1/instruments?symbols={names}")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["detail"]["cap"] == rest.INSTRUMENTS_CAP


def test_the_venue_wide_tape_needs_no_credential_and_names_every_print(taped):
    """The feed a recorder reads in shadow mode, with no account anywhere.

    Per symbol it would be one request per listed contract per poll, and a
    print identified by its price would be the wrong print: prices repeat
    constantly in a narrow market, so identity has to come from the engine's own
    match number and the book that minted it.
    """
    payload = taped.client.get("/v1/trades?limit=250").json()
    assert payload["count"] > 0
    assert payload["cap"] == rest.TAPE_CAP
    assert payload["generation"] == 0

    for row in payload["trades"]:
        assert row["trade_id"] == f"{row['symbol']}:{row['sequence']}"
        assert isinstance(row["price"], str)
        Decimal(row["price"])  # exact, or this raises
        assert isinstance(row["quantity"], int)
        assert row["aggressor_side"] in ("buy", "sell")
        assert row["t"] is None or isinstance(row["t"], int)

    # Venue-wide rather than one book's tape under a new name.
    assert len({row["symbol"] for row in payload["trades"]}) > 1


def test_the_tape_pages_by_cursor_without_repeating_or_skipping_a_print(taped):
    """The property a cursor exists for, asserted over the whole feed.

    Walked to the beginning and checked against the ordinals rather than
    against a count, because a count cannot tell "I saw two pages of fifty" from
    "I saw one page twice". Every ordinal from zero to the total has to turn up
    exactly once, and the walk has to be monotonically backwards.
    """
    head = taped.client.get("/v1/trades?limit=250").json()
    assert head["total"] > head["count"], "one page holds the whole tape"

    seen: list[int] = []
    cursor = head["cursor"]
    assert isinstance(cursor, str), "a cursor tested for truthiness must not be 0"
    seen.extend(row["ordinal"] for row in head["trades"])

    for _ in range(2_000):
        if cursor is None:
            break
        page = taped.client.get(f"/v1/trades?limit=250&cursor={cursor}").json()
        assert page["count"] <= 250
        seen.extend(row["ordinal"] for row in page["trades"])
        cursor = page["cursor"]
    else:
        raise AssertionError("the tape never reached its beginning")

    assert seen == sorted(seen, reverse=True), "a page went forwards"
    assert len(seen) == len(set(seen)), "a print was handed out twice"
    assert set(seen) == set(range(head["total"])), "a print was skipped"


def test_a_cursor_that_is_not_a_cursor_is_refused_in_the_catalogue_terms(taped):
    """A client that sends a cursor it built itself, or one from another market,
    should be told so rather than handed the head of the tape. Answering a
    nonsense cursor with the newest page is how a recorder resumes from the
    wrong place and never finds out."""
    for query in ("cursor=later", "cursor=-1", "limit=0"):
        response = taped.client.get(f"/v1/trades?{query}")
        assert response.status_code == 400, query
        assert response.json()["error"]["code"] == "invalid_request", query

    # Zero is a legitimate cursor and means "nothing older than the beginning",
    # which is an empty page rather than a refusal.
    empty = taped.client.get("/v1/trades?cursor=0").json()
    assert empty["trades"] == [] and empty["cursor"] is None

    # And one past the end is clamped rather than refused: a client whose
    # cursor is older than this market is behind, not wrong.
    head = taped.client.get("/v1/trades?limit=5").json()
    beyond = taped.client.get(
        f"/v1/trades?limit=5&cursor={head['total'] + 1_000}"
    ).json()
    assert beyond["trades"] == head["trades"]


def test_a_tape_cursor_is_not_disturbed_by_prints_that_arrive_after_it(fresh):
    """Newest first, with the cursor walking backwards, is the whole reason the
    paging is stable: new prints land above a cursor and cannot move the rows
    below it. The opposite arrangement was measured on the venue this shape is
    copied from, where a cursor that persisted across polls walked further into
    the past every time and the recorder never saw a new print at all."""
    venue = fresh()
    # Its own market, because this one has to keep trading after it has read a
    # cursor, and pumped against the tape rather than the clock for the reason
    # the ``taped`` fixture gives.
    venue.pump(1_000, slices=20)
    for _ in range(12):
        if tape_total(venue) > 20:
            break
        venue.pump(2_000, slices=40)

    head = venue.client.get("/v1/trades?limit=20").json()
    cursor = head["cursor"]
    assert cursor is not None, "one page holds the whole tape, so nothing is below it"
    before = venue.client.get(f"/v1/trades?limit=20&cursor={cursor}").json()
    assert before["trades"], "no rows below the cursor to be disturbed"

    venue.pump(2_000, slices=40)

    after = venue.client.get(f"/v1/trades?limit=20&cursor={cursor}").json()
    assert after["trades"] == before["trades"]
    assert venue.client.get("/v1/trades?limit=20").json()["total"] > head["total"]


def test_the_exchange_publishes_the_widest_candle_window_it_can_answer_for(candled):
    """A backfill chunks its window *before* it makes its first request.

    The candle endpoint refuses a range wider than its ring retains rather than
    truncating it, which is the right refusal and is worth nothing to a client
    that has to provoke it to learn the number. Published on the venue's own
    description, a client reads the cap once and chunks correctly; without it,
    it either guesses low and makes ten times the calls or guesses high and
    spends the backfill collecting refusals.
    """
    published = candled.client.get("/v1/exchange").json()["candles"]
    assert published, "no candle periods published"
    symbol = candled.symbols()[0]

    for entry in published:
        assert entry["max_window_ns"] == entry["period_ns"] * entry["depth"]
        assert entry["max_window_s"] * 1_000_000_000 == entry["max_window_ns"]
        # The same numbers the endpoint itself enforces against, or the cap a
        # client reads here is not the cap it will be refused by.
        page = candles(candled, symbol, f"period={entry['period']}&limit=1")
        assert page["cap"] == entry["depth"]
        assert page["retains_ns"] == entry["max_window_ns"]
        assert page["period_ns"] == entry["period_ns"]

    assert [entry["period"] for entry in published] == candles(
        candled, symbol, "period=1&limit=1"
    )["periods"]


def test_a_settled_contract_publishes_its_result_without_a_credential(fresh):
    """Settlement is market data, so it reads with no account.

    The half of the settlement story that works in shadow mode: how the
    contract resolved is a fact about the contract, and a research client
    scoring a strategy against outcomes it never held a position in needs it
    without holding an account at all.

    On a market of its own rather than the shared one, because settling is the
    one read-only-looking act in this file that is not: it pays out every
    account in the venue and closes the book for good.
    """
    exchange = fresh()
    exchange.pump(300, slices=12)
    symbol = exchange.symbols()[0]
    before = exchange.client.get(f"/v1/instruments/{symbol}").json()["settlement"]
    assert before == {
        "status": "open",
        "settled": False,
        "voided": False,
        "value": None,
        "ticks": None,
        "reason": None,
        "digest": None,
    }

    result = settle_now(exchange, symbol)
    after = exchange.client.get(f"/v1/instruments/{symbol}").json()["settlement"]
    assert after["settled"] is True
    assert after["status"] == ("void" if result.void_reason else "settled")
    assert after["voided"] is (result.settlement_value is None)
    if result.settlement_value is not None:
        assert after["value"] == str(result.settlement_value)
        assert isinstance(after["value"], str), "a settlement value is not a double"
        assert after["ticks"] == int(
            exchange.venue.registry.require(symbol).to_ticks(result.settlement_value)
        )
    # The content address of the whole record, inputs and provenance included,
    # so a client can check that two readings of this describe one settlement.
    assert after["digest"] == result.result_digest

    # And the same block on the bulk read, because a settlements recorder and a
    # quote recorder ask about the same three hundred rows.
    bulk = exchange.client.get(f"/v1/instruments?symbols={symbol}").json()
    assert bulk["instruments"][0]["settlement"] == after


def test_a_contract_that_is_still_live_is_never_given_a_settlement_value(exchange):
    """The look-ahead guard, and the only rule in this area that matters.

    The oracle will answer for a contract whose window has not closed:
    ``build_market`` settles every listed instrument at construction to price
    the agents' priors, so ``settlement_source`` returns a well formed number
    for a market that is still trading. Publishing it would hand a client the
    answer before the market has it, and a backtest scored against a price it
    could read in advance is not a backtest. So the value comes from the set
    the venue has actually paid out on and from nowhere else.
    """
    rows = exchange.client.get("/v1/instruments?limit=1000").json()["instruments"]
    settled = set(exchange.venue.settled_symbols)
    live = [row for row in rows if row["symbol"] not in settled]
    assert live, "every contract has settled; this test proves nothing"

    for row in live:
        assert row["settlement"]["settled"] is False, row["symbol"]
        assert row["settlement"]["value"] is None, row["symbol"]
        assert row["settlement"]["ticks"] is None, row["symbol"]

    # The guard is doing work rather than describing an oracle that could not
    # have answered anyway.
    spec = exchange.venue.registry.require(live[0]["symbol"]).spec
    assert exchange.runner.market.settlement_source(spec).settlement_value is not None


def test_a_settled_position_is_told_apart_from_one_that_was_traded_out(fresh):
    """The account's own half of settlement, which is a different question.

    Both leave a position flat with a realised result, and an algorithm can
    only be scored on one of them: closing a trade is a decision it made, and a
    settlement is the contract resolving. Without the flag on the row the two
    are indistinguishable, and a symbol that settled while this account held
    nothing is not this account's settlement at all.
    """
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Holder")
    held = venue.restable()
    untouched = next(symbol for symbol in venue.symbols() if symbol != held)

    placed = trader.post("/v1/orders", {"symbol": held, "side": "buy", "quantity": 2})
    assert placed.status_code == 202, placed.text
    venue.pump(300, slices=12)

    rows = positions_of(trader)
    assert held in rows, rows
    assert rows[held]["settled"] is False
    assert rows[held]["settlement"] is None

    settle_now(venue, held)
    settle_now(venue, untouched)

    rows = positions_of(trader)
    assert rows[held]["settled"] is True
    assert rows[held]["settlement"]["settled"] is True
    assert rows[held]["settlement"]["status"] in ("settled", "void")
    # The contract took the position away and left the result behind.
    assert rows[held]["quantity"] == 0
    assert isinstance(rows[held]["realized_pnl"], str)

    # Settled on the venue, and not this account's settlement: it never held it.
    assert untouched not in rows


def test_a_resting_order_reports_what_is_in_front_of_it_in_the_queue(fresh):
    """Without this an outside backtest overstates every fill it models.

    A shadow fill model that knows only "my order was at 4,700 and the book
    traded at 4,700" has to assume it was filled, which is an upper bound on its
    own performance rather than a measurement of it. The venue can answer
    exactly, because the engine's priority is (price, arrival) and arrival is a
    monotonic sequence rather than a clock.
    """
    venue = fresh()
    venue.pump(400, slices=16)
    symbol = venue.restable()
    price = resting_price(venue, symbol, "buy")
    front = venue.trader("Front")
    back = venue.trader("Back")

    for trader, size, handle in ((front, 4, "front"), (back, 6, "back")):
        placed = trader.post(
            "/v1/orders",
            {
                "symbol": symbol,
                "side": "buy",
                "quantity": size,
                "price": str(price),
                "type": "limit",
                "time_in_force": "gtc",
                "client_order_id": handle,
            },
        )
        assert placed.status_code == 202, placed.text
        # Between the two, so the arrival order the queue reports is the order
        # this test asked for rather than whichever crossed the link first.
        venue.pump(300, slices=12)

    leading = working_order(front, "front")["queue"]
    trailing = working_order(back, "back")["queue"]

    assert leading["own"] == 4 and trailing["own"] == 6
    assert leading["ahead"] == 0 and leading["orders_ahead"] == 0
    assert trailing["ahead"] == leading["ahead"] + leading["own"]
    assert trailing["orders_ahead"] == 1
    assert leading["behind"] >= trailing["own"]

    for queue in (leading, trailing):
        # The identity that says the walk skipped what the book skips: a cancel
        # is tombstoned in the queue rather than spliced out of it, and a naive
        # sum counts lots nothing can trade against.
        assert queue["ahead"] + queue["own"] + queue["behind"] == queue["level"]

    ladder = venue.client.get(f"/v1/instruments/{symbol}/book?depth=100").json()
    depth = {Decimal(level): size for level, size in ladder["bids"]}
    assert depth[price] == leading["level"]


def test_a_cancelled_order_stops_counting_against_the_queue_behind_it(fresh):
    """The tombstone, which is the subtle half of the arithmetic.

    ``OrderBook.remove`` marks a cancel terminal and leaves it in the deque,
    because splicing it out would make cancelling O(queue) in a market where
    cancelling is the common operation. A queue position that counted it would
    tell the order behind it that four lots it can never be beaten by are still
    in front of it, which is exactly the pessimism that makes a fill model wrong
    in the other direction.
    """
    venue = fresh()
    venue.pump(400, slices=16)
    symbol = venue.restable()
    price = resting_price(venue, symbol, "buy")
    front = venue.trader("Leaver")
    back = venue.trader("Stayer")

    for trader, handle in ((front, "leaver"), (back, "stayer")):
        placed = trader.post(
            "/v1/orders",
            {
                "symbol": symbol,
                "side": "buy",
                "quantity": 4,
                "price": str(price),
                "type": "limit",
                "time_in_force": "gtc",
                "client_order_id": handle,
            },
        )
        assert placed.status_code == 202, placed.text
        venue.pump(300, slices=12)

    leaving = working_order(front, "leaver")
    before = working_order(back, "stayer")["queue"]
    assert before["ahead"] >= 4

    pulled = front.delete(f"/v1/orders/{symbol}/{leaving['order_id']}")
    assert pulled.status_code == 200, pulled.text
    venue.pump(300, slices=12)

    after = working_order(back, "stayer")["queue"]
    assert after["ahead"] == before["ahead"] - leaving["quantity"]
    assert after["ahead"] + after["own"] + after["behind"] == after["level"]


def test_no_json_number_carries_money_on_the_feeds_an_outside_client_reads(taped):
    """The rule the whole ledger rests on, held at the two surfaces added for
    clients that parse them without a schema. A double cannot hold a tick, and
    a price that arrives as one has already lost the exactness that makes
    ``conservation_check`` exactly zero rather than nearly zero."""
    tape = json.loads(taped.client.get("/v1/trades?limit=100").text)
    assert floats_in(tape) == []
    assert tape["trades"], "no prints to check"

    listing = json.loads(taped.client.get("/v1/instruments?limit=1000").text)
    assert floats_in(listing) == []
    for row in listing["instruments"]:
        value = row["settlement"]["value"]
        assert value is None or isinstance(value, str), row["symbol"]


# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------


def test_a_key_is_issued_to_a_browser_session_and_shows_its_secret_once(fresh):
    venue = fresh()
    token = venue.browser("Ash")
    issued = venue.issue(token, label="my bot")

    assert issued["secret"]
    assert issued["label"] == "my bot"
    assert issued["account_id"].startswith("you-")

    listed = venue.client.get("/v1/keys", headers={"x-test-session": token}).json()
    assert [row["key_id"] for row in listed["keys"]] == [issued["key_id"]]
    assert all("secret" not in row for row in listed["keys"])
    # The seat token is the value the owner's own browser session is identified
    # by; there is no reason for it to come back out of the key list.
    assert all(token not in json.dumps(row) for row in listed["keys"])


def test_key_management_refuses_a_caller_with_no_browser_session(fresh):
    venue = fresh()
    for method, path in (("POST", "/v1/keys"), ("GET", "/v1/keys"), ("DELETE", "/v1/keys/x")):
        response = venue.client.request(method, path, json={})
        assert response.status_code == 401, path
        assert response.json()["error"]["code"] == "auth_required"


def test_a_key_belonging_to_another_session_cannot_be_revoked(fresh):
    """And answers exactly as a key that never existed does, so this endpoint
    cannot be used to discover which key ids are real."""
    venue = fresh()
    mine = venue.browser("Ash")
    theirs = venue.browser("Nita")
    victim = venue.issue(theirs)

    refused = venue.client.delete(
        f"/v1/keys/{victim['key_id']}", headers={"x-test-session": mine}
    )
    invented = venue.client.delete(
        "/v1/keys/ak_0000000000000000", headers={"x-test-session": mine}
    )
    assert refused.status_code == invented.status_code == 404
    assert refused.json() == invented.json()


def test_revoking_a_key_is_idempotent_and_stops_it_trading(fresh):
    venue = fresh()
    token = venue.browser("Ash")
    trader = Client(venue, venue.issue(token), token)

    assert trader.get("/v1/account").status_code == 200

    first = venue.client.delete(
        f"/v1/keys/{trader.key_id}", headers={"x-test-session": token}
    )
    second = venue.client.delete(
        f"/v1/keys/{trader.key_id}", headers={"x-test-session": token}
    )
    assert first.status_code == second.status_code == 200
    assert first.json()["already_done"] is False
    assert second.json()["already_done"] is True

    refused = trader.get("/v1/account")
    assert refused.status_code == 401
    assert refused.json()["error"]["code"] == "auth_invalid"


# --------------------------------------------------------------------------
# Signatures
# --------------------------------------------------------------------------


def test_a_signed_request_succeeds(exchange):
    trader = exchange.trader("Signed")
    response = trader.get("/v1/account")
    assert response.status_code == 200
    payload = response.json()
    assert payload["account_id"] == trader.account_id
    assert Decimal(payload["cash"]) > 0
    assert Decimal(payload["equity"]) == Decimal(payload["starting_cash"])


def test_an_unsigned_request_is_refused(exchange):
    for method, path in (
        ("GET", "/v1/account"),
        ("GET", "/v1/account/positions"),
        ("GET", "/v1/account/fills"),
        ("GET", "/v1/orders"),
        ("POST", "/v1/orders"),
        ("GET", "/v1/orders/X/1"),
        ("DELETE", "/v1/orders/X/1"),
        ("DELETE", "/v1/orders"),
    ):
        response = exchange.client.request(method, path, json={})
        assert response.status_code == 401, path
        body = response.json()
        assert body["error"]["code"] == "auth_required", path
        assert HEADER_SIGNATURE in body["error"]["detail"]["headers"]


def test_a_stale_signature_is_refused(exchange):
    """Thirty seconds is the window ``keys.py`` documents. A signature is
    replayable only inside it, which is the point of signing a timestamp."""
    trader = exchange.trader("Stale")
    old = f"{time.time() - 120:.3f}"
    response = trader.get("/v1/account", timestamp=old)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "auth_invalid"


def test_a_signature_cannot_be_moved_onto_another_request(exchange):
    """The signature covers the method, the path *with its query string*, and
    the body. Each of those is checked by lifting a valid signature onto a
    request that differs in exactly one of them."""
    trader = exchange.trader("Tamper")
    symbol = exchange.symbols()[0]
    body = body_bytes({"symbol": symbol, "side": "buy", "quantity": 1})

    honest = trader.headers("POST", "/v1/orders", body)

    # Same signature, different body: a captured order for one lot replayed as
    # an order for a thousand.
    swapped = body_bytes({"symbol": symbol, "side": "buy", "quantity": 1000})
    response = exchange.client.post("/v1/orders", content=swapped, headers=honest)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "auth_invalid"

    # Same signature, same path, different method: a signed order lifted onto
    # the endpoint that cancels every order the account has working.
    moved = exchange.client.request(
        "DELETE", "/v1/orders", content=body, headers=honest
    )
    assert moved.status_code == 401

    # Same signature, different query string on the same path.
    listed = trader.headers("GET", "/v1/orders?limit=1")
    shifted = exchange.client.get("/v1/orders?limit=1000", headers=listed)
    assert shifted.status_code == 401
    assert shifted.json()["error"]["code"] == "auth_invalid"


def test_a_signature_from_the_wrong_secret_is_refused(exchange):
    trader = exchange.trader("Wrong")
    response = trader.get("/v1/account", secret="not the secret")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "auth_invalid"


def test_an_unknown_key_and_a_bad_signature_are_indistinguishable(exchange):
    """One message for every authentication failure, exactly as
    ``SignatureError`` does it. Saying which one went wrong tells a caller
    holding no valid key which key ids exist."""
    trader = exchange.trader("Opaque")
    bad_secret = trader.get("/v1/account", secret="wrong")

    stamp = f"{time.time():.3f}"
    unknown = exchange.client.get(
        "/v1/account",
        headers={
            HEADER_KEY: "ak_0000000000000000",
            HEADER_TIMESTAMP: stamp,
            HEADER_SIGNATURE: sign("anything", "GET", "/v1/account", stamp),
        },
    )
    assert bad_secret.status_code == unknown.status_code == 401
    assert bad_secret.json() == unknown.json()


# --------------------------------------------------------------------------
# Orders
# --------------------------------------------------------------------------


def test_an_order_reaches_the_book_and_appears_in_positions(fresh):
    """The whole point of routing through ``LiveMarket``: the order enters the
    same event queue, at the same latency, and is settled by the same ledger."""
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Buyer")
    symbol = venue.restable()

    placed = trader.post(
        "/v1/orders",
        {"symbol": symbol, "side": "buy", "quantity": 2, "client_order_id": "cid-1"},
    )
    assert placed.status_code == 202, placed.text
    assert placed.json()["client_order_id"] == "cid-1"
    assert placed.json()["status"] == "accepted"

    venue.pump(300, slices=12)

    positions = trader.get("/v1/account/positions").json()["positions"]
    held = {row["symbol"]: row for row in positions}
    assert symbol in held, positions
    assert held[symbol]["quantity"] == 2

    fills = trader.get("/v1/account/fills").json()
    assert fills["fills"], "a market order that did not fill"
    assert fills["fills"][0]["symbol"] == symbol
    assert isinstance(fills["fills"][0]["price"], str)
    # Named by the id the client chose, and named here without any prior call
    # to /v1/orders. A market order is acknowledged and filled in the same
    # instant, so it never appears in a working-order list at all. And it is
    # the one a client most needs named, because a fill is the event it has to
    # book.
    assert fills["fills"][0]["client_order_id"] == "cid-1"

    account = trader.get("/v1/account").json()
    assert Decimal(account["collateral"]) > 0


def test_a_resting_order_is_reconcilable_by_its_client_order_id(fresh):
    """The client chooses one identifier and the exchange chooses another.
    Nothing carries the first into the engine, so the join is made in the API.
    If it were not, a client could not tell which of its orders is which."""
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Resting")
    symbol = venue.symbols()[0]
    price = resting_price(venue, symbol, "buy")

    placed = trader.post(
        "/v1/orders",
        {
            "symbol": symbol,
            "side": "buy",
            "quantity": 3,
            "price": str(price),
            "type": "limit",
            "time_in_force": "gtc",
            "client_order_id": "cid-rest",
        },
    )
    assert placed.status_code == 202, placed.text

    # Before the acknowledgement has crossed back, the order is in neither the
    # book nor the working list, and the client is told so rather than left to
    # guess, because "not placed" and "not there yet" differ by a duplicate.
    in_flight = trader.get("/v1/orders").json()
    assert "cid-rest" in {row["client_order_id"] for row in in_flight["pending"]}

    venue.pump(300, slices=12)

    working = trader.get("/v1/orders").json()
    mine = [row for row in working["orders"] if row["client_order_id"] == "cid-rest"]
    assert mine, working
    order = mine[0]
    assert order["symbol"] == symbol
    assert order["side"] == "buy"
    assert Decimal(order["price"]) == price
    assert order["remaining"] == 3
    assert working["cap"] == rest.ORDERS_CAP

    detail = trader.get(f"/v1/orders/{symbol}/{order['order_id']}").json()
    assert detail["order_id"] == order["order_id"]
    assert detail["client_order_id"] == "cid-rest"


def test_a_reused_client_order_id_is_a_conflict_carrying_the_existing_order(fresh):
    """Replaying it would mean answering "accepted" for an order this call did
    not place, and a client retrying a timed-out POST cannot tell that answer
    from the truth. So it is refused, but as a 409 rather than a 400, and with
    the first order attached.

    The status code is doing work. 400 says "your request is malformed, fix it
    and resend", and this request is not malformed: it is a perfectly good order
    that conflicts with one that already exists. And the detail is the answer to
    the question the retrying client is actually asking, which is not "is that
    id taken" but "am I long".
    """
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Duplicate")
    symbol = venue.symbols()[0]
    price = resting_price(venue, symbol, "buy")
    order = {
        "symbol": symbol,
        "side": "buy",
        "quantity": 1,
        "price": str(price),
        "time_in_force": "gtc",
        "client_order_id": "same",
    }

    assert trader.post("/v1/orders", order).status_code == 202

    # Retried before the acknowledgement has crossed back: the conflict is
    # reported and the order is honestly described as still in flight.
    immediate = trader.post("/v1/orders", order)
    assert immediate.status_code == 409
    error = immediate.json()["error"]
    assert error["code"] == "duplicate_client_order_id"
    assert error["detail"]["client_order_id"] == "same"
    assert error["detail"]["status"] == "pending"
    assert error["detail"]["order_id"] is None

    venue.pump(400, slices=16)

    # Retried after it rested: same conflict, and now it names the exchange's
    # own id and the live order behind it.
    settled = trader.post("/v1/orders", order)
    assert settled.status_code == 409
    detail = settled.json()["error"]["detail"]
    assert detail["status"] == "working"
    assert isinstance(detail["order_id"], int)
    assert detail["order"]["order_id"] == detail["order_id"]
    assert Decimal(detail["price"]) == price
    # And nothing was placed by either refusal: one order id, one working order.
    working = trader.get("/v1/orders").json()["orders"]
    assert [row["client_order_id"] for row in working].count("same") == 1


def test_an_order_is_findable_by_the_client_order_id_that_placed_it(fresh):
    """``GET /v1/orders:by_client_order_id?id=...``, which is Alpaca's path exactly.

    The colon suffix rather than a fourth segment, because
    ``/orders/{client_order_id}`` would collide with the exchange's own ids
    under ``/orders/{symbol}/{order_id}``.

    This is the half of reconciliation that the duplicate refusal above cannot
    supply on its own: a client that retried a timed-out POST learns that its id
    was used, and still needs somewhere to ask what became of it.
    """
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Finder")
    symbol = venue.symbols()[0]
    price = resting_price(venue, symbol, "buy")

    placed = trader.post(
        "/v1/orders",
        {
            "symbol": symbol,
            "side": "buy",
            "quantity": 3,
            "price": str(price),
            "time_in_force": "gtc",
            "client_order_id": "look-me-up",
        },
    )
    assert placed.status_code == 202, placed.text

    in_flight = trader.get("/v1/orders:by_client_order_id?id=look-me-up")
    assert in_flight.status_code == 200, in_flight.text
    assert in_flight.json()["status"] == "pending"
    assert in_flight.json()["order_id"] is None
    assert in_flight.json()["order"] is None
    assert Decimal(in_flight.json()["price"]) == price

    venue.pump(400, slices=16)

    resting = trader.get("/v1/orders:by_client_order_id?id=look-me-up").json()
    assert resting["status"] == "working"
    assert resting["symbol"] == symbol
    assert resting["side"] == "buy"
    assert resting["quantity"] == 3
    assert resting["order"]["remaining"] == 3
    assert resting["order"]["order_id"] == resting["order_id"]
    assert resting["account_id"] == trader.account_id

    # The same order, found the other way round, agrees with itself.
    detail = trader.get(f"/v1/orders/{symbol}/{resting['order_id']}").json()
    assert detail["client_order_id"] == "look-me-up"

    trader.delete(f"/v1/orders/{symbol}/{resting['order_id']}")
    venue.pump(400, slices=16)
    gone = trader.get("/v1/orders:by_client_order_id?id=look-me-up").json()
    assert gone["status"] == "done"
    assert gone["order_id"] == resting["order_id"]
    assert gone["order"] is None

    unknown = trader.get("/v1/orders:by_client_order_id?id=never-sent")
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "not_found"

    blank = trader.get("/v1/orders:by_client_order_id?id=")
    assert blank.status_code == 400
    assert blank.json()["error"]["code"] == "invalid_request"

    # And it is another seat's business, not this one's: an id one trader used
    # is not addressable by another, which is the same rule the order detail
    # endpoint keeps.
    stranger = venue.trader("Stranger")
    assert stranger.get("/v1/orders:by_client_order_id?id=look-me-up").status_code == 404


def test_the_fills_cursor_returns_strictly_after_and_never_repeats(fresh):
    """A reconnecting algorithm has to be able to say which fills it has booked.

    Without a monotonic id the only safe readings of a blotter after a
    disconnect are "re-book everything" and "book nothing", and both are wrong.
    ``?after=`` is Binance's ``myTrades?fromId=``, and the id is derived from
    the agent's own fill counter rather than from a position in the log. So it
    is stable across evictions, and it orders fills in different symbols, which
    the matching engine's per-book sequence number cannot.
    """
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Reconciler")

    for symbol in venue.symbols()[:4]:
        trader.post("/v1/orders", _order(symbol, quantity=2))
    venue.pump(500, slices=20)

    everything = trader.get("/v1/account/fills?limit=200").json()
    assert everything["fills"], "nothing filled"
    ids = [entry["fill_id"] for entry in everything["fills"]]
    assert len(set(ids)) == len(ids)
    # Newest first, which the response is documented to stay whatever cursor is
    # applied, because the cap is the blotter's own bound and one request at the
    # cap is the whole of it.
    assert ids == sorted(ids, reverse=True)

    cursor = everything["cursor"]["fills"]
    assert cursor["total"] == max(ids)
    assert cursor["last_id"] == max(ids)
    assert cursor["first_id"] == min(ids)
    assert cursor["retained"] == len(ids)

    # Strictly after: the id handed in never comes back.
    middle = sorted(ids)[len(ids) // 2]
    after = trader.get(f"/v1/account/fills?after={middle}&limit=200").json()
    assert after["cursor"]["after"] == middle
    seen = [entry["fill_id"] for entry in after["fills"]]
    assert seen == [i for i in ids if i > middle]
    assert middle not in seen

    # Caught up: nothing new, and no exception about it.
    caught_up = trader.get(f"/v1/account/fills?after={max(ids)}&limit=200").json()
    assert caught_up["fills"] == []
    assert caught_up["count"] == 0
    # The cursor block still describes what exists, so a client that has caught
    # up can still tell whether it lost anything on the way.
    assert caught_up["cursor"]["fills"]["total"] == max(ids)

    # Zero is a cursor, not an absence: a client that has never advanced its own
    # asks for everything and gets everything.
    from_zero = trader.get("/v1/account/fills?after=0&limit=200").json()["fills"]
    assert [entry["fill_id"] for entry in from_zero] == ids

    # Nothing is dropped and nothing is repeated across a resume: page once,
    # then resume from the last id seen, and the union is exactly the whole.
    first_half = trader.get("/v1/account/fills?after=0&limit=200").json()["fills"]
    resumed = trader.get(
        f"/v1/account/fills?after={min(ids)}&limit=200"
    ).json()["fills"]
    assert {e["fill_id"] for e in resumed} | {min(ids)} == {e["fill_id"] for e in first_half}

    for bad in ("-1", "many", "1.5"):
        refused = trader.get(f"/v1/account/fills?after={bad}")
        assert refused.status_code == 400, bad
        assert refused.json()["error"]["code"] == "invalid_request"

    # Rejections are a second sequence with a cursor of its own, because
    # fill 3 and rejection 3 are unrelated events and one ``after`` applied to
    # both would silently drop from one of them.
    refusals = trader.get("/v1/account/fills?after_rejection=0&limit=200").json()
    assert refusals["cursor"]["after_rejection"] == 0
    assert refusals["cursor"]["after"] is None
    assert set(refusals["cursor"]["rejections"]) == {
        "total",
        "retained",
        "first_id",
        "last_id",
    }
    for entry in refusals["rejections"]:
        assert isinstance(entry["rejection_id"], int)
    assert trader.get("/v1/account/fills?after_rejection=-1").status_code == 400

    # The generation travels with the cursor. A rebuild seats this key behind a
    # fresh agent whose counters start at one, so a client holding fill 40
    # across one would discard the new market's first forty fills as already
    # seen; the only thing that lets it notice is being told which market the
    # numbers belong to.
    assert everything["generation"] == 0


def test_cancelling_is_idempotent(fresh):
    """The decision, and the reason, are in ``cancel_order``'s docstring: the
    client wanted that order not to be resting and it is not resting, so a
    refusal would make a correct outcome look like an error."""
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Canceller")
    symbol = venue.symbols()[0]
    price = resting_price(venue, symbol, "buy")

    trader.post(
        "/v1/orders",
        {"symbol": symbol, "side": "buy", "quantity": 2, "price": str(price)},
    )
    venue.pump(300, slices=12)

    working = trader.get("/v1/orders").json()["orders"]
    assert working, "nothing resting to cancel"
    order_id = working[0]["order_id"]

    first = trader.delete(f"/v1/orders/{symbol}/{order_id}")
    assert first.status_code == 200
    assert first.json()["already_done"] is False

    venue.pump(200, slices=8)

    second = trader.delete(f"/v1/orders/{symbol}/{order_id}")
    assert second.status_code == 200
    assert second.json()["already_done"] is True

    # An id that never existed, and one belonging to nobody, answer the same
    # way. That is what makes this endpoint disclose nothing.
    never = trader.delete(f"/v1/orders/{symbol}/999999")
    assert never.status_code == 200
    assert never.json()["already_done"] is True

    # A typo in the *symbol* is still a refusal. That is not a race, and
    # answering it with success would let a client believe it had cancelled
    # something in a market that does not exist.
    mistyped = trader.delete("/v1/orders/NOT_A_SYMBOL/1")
    assert mistyped.status_code == 400
    assert mistyped.json()["error"]["code"] == "invalid_symbol"


def test_cancel_all_pulls_every_working_order_and_leaves_positions_alone(fresh):
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Flattener")
    symbols = venue.symbols()[:3]

    for symbol in symbols:
        trader.post(
            "/v1/orders",
            {
                "symbol": symbol,
                "side": "buy",
                "quantity": 1,
                "price": str(resting_price(venue, symbol, "buy")),
            },
        )
    venue.pump(400, slices=16)

    before = trader.get("/v1/orders").json()
    assert before["count"] >= 1

    cleared = trader.delete("/v1/orders")
    assert cleared.status_code == 200
    assert cleared.json()["count"] == before["count"]

    venue.pump(400, slices=16)
    after = trader.get("/v1/orders").json()
    assert after["count"] == 0, after


def test_one_account_never_sees_another_account(fresh):
    """Two keys are two traders. Sharing one blotter is the failure the seat
    binding exists to prevent, seen from the ordinary direction."""
    venue = fresh()
    venue.pump(400, slices=16)
    one = venue.trader("Ash")
    two = venue.trader("Nita")
    symbol = venue.symbols()[0]

    assert one.account_id != two.account_id

    one.post(
        "/v1/orders",
        {
            "symbol": symbol,
            "side": "buy",
            "quantity": 4,
            "price": str(resting_price(venue, symbol, "buy")),
        },
    )
    venue.pump(400, slices=16)

    assert one.get("/v1/orders").json()["count"] >= 1
    assert two.get("/v1/orders").json()["count"] == 0

    mine = one.get("/v1/orders").json()["orders"][0]
    # The other account cannot read it, and cannot cancel it either. Both
    # answer as though it did not exist.
    assert two.get(f"/v1/orders/{symbol}/{mine['order_id']}").status_code == 404
    assert two.delete(f"/v1/orders/{symbol}/{mine['order_id']}").json()["already_done"]

    venue.pump(300, slices=12)
    assert one.get("/v1/orders").json()["count"] >= 1


def test_every_list_endpoint_publishes_the_cap_it_applied(exchange):
    """A client has to be able to tell "that is all of them" from "that is all
    you asked for" without reading documentation, or it either stops early or
    pages forever."""
    trader = exchange.trader("Pager")
    symbol = exchange.symbols()[0]
    for path in (
        "/v1/instruments?limit=2",
        "/v1/trades?limit=2",
        f"/v1/instruments/{symbol}/trades?limit=2",
        f"/v1/instruments/{symbol}/history?limit=2",
        f"/v1/instruments/{symbol}/candles?period=1&limit=2",
    ):
        payload = exchange.client.get(path).json()
        assert payload["limit"] == 2, path
        assert payload["count"] <= 2, path
        assert payload["cap"] >= payload["limit"], path

    for path in (
        "/v1/account/positions?limit=2",
        "/v1/account/fills?limit=2",
        "/v1/orders?limit=2",
    ):
        payload = trader.get(path).json()
        assert payload["limit"] == 2, path
        assert payload["cap"] >= payload["limit"], path

    ladder = exchange.client.get(f"/v1/instruments/{symbol}/book?depth=2").json()
    assert ladder["depth"] == 2 and ladder["cap"] == rest.BOOK_DEPTH_CAP

    keys = exchange.client.get(
        "/v1/keys?limit=2", headers={"x-test-session": trader.token}
    ).json()
    assert keys["limit"] == 2 and keys["cap"] == rest.KEYS_CAP


@pytest.mark.parametrize("tif", sorted(choice.value for choice in TimeInForce))
def test_every_time_in_force_reaches_the_venue(fresh, tif):
    """The vocabulary comes from ``TimeInForce`` rather than a list in the API,
    so a fifth one added to the exchange is reachable here without anybody
    editing this file, and is refused by this test if it is not."""
    venue = fresh()
    venue.pump(300, slices=12)
    trader = venue.trader(f"Tif{tif}")
    symbol = venue.symbols()[0]
    response = trader.post(
        "/v1/orders",
        _order(
            symbol,
            quantity=1,
            price=str(resting_price(venue, symbol, "buy")),
            time_in_force=tif,
        ),
    )
    assert response.status_code == 202, response.text
    assert response.json()["time_in_force"] == tif


def test_a_stop_and_an_iceberg_reach_the_book_through_the_api(fresh):
    """Both are order types a person can reach from the browser ticket, so a
    program has to be able to reach them too: an API that could only send the
    two default order types would make the page the more capable client."""
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Exotic")
    symbol = venue.symbols()[0]
    price = resting_price(venue, symbol, "buy")

    iceberg = trader.post(
        "/v1/orders",
        _order(symbol, quantity=10, price=str(price), display=2, client_order_id="ice"),
    )
    assert iceberg.status_code == 202, iceberg.text

    stop = trader.post(
        "/v1/orders",
        _order(
            symbol,
            side="sell",
            quantity=1,
            stop=str(price),
            type="stop",
            client_order_id="stop",
        ),
    )
    assert stop.status_code == 202, stop.text
    assert stop.json()["stop"] == str(price)

    venue.pump(400, slices=16)
    working = {row["client_order_id"]: row for row in trader.get("/v1/orders").json()["orders"]}
    assert "ice" in working, working
    assert working["ice"]["display"] == 2
    # An iceberg shows a slice and holds the rest back. That is the whole trade
    # it makes: visibility for queue priority.
    assert working["ice"]["shown"] == 2
    assert working["ice"]["remaining"] == 10


# --------------------------------------------------------------------------
# Pegged orders: quoting a position rather than a number
# --------------------------------------------------------------------------


def _rest_a_peg(venue: Exchange, trader: "Client", symbol: str, **fields) -> dict:
    """A buy pegged well behind its reference, so it tracks rather than trades.

    Four hundred ticks behind, and the distance is doing work rather than being
    a round number: measured, a buy pegged to the bid at ``-2`` on
    ``ASSASSIN_IDX`` was lifted inside 650ms of simulated time and came back
    ``remaining=0``, which made every assertion about a *resting* peg a test of
    a filled one instead.
    """
    body = {
        "symbol": symbol,
        "side": "buy",
        "quantity": 2,
        "peg": "bid",
        "peg_offset": -400,
        **fields,
    }
    response = trader.post("/v1/orders", body)
    assert response.status_code == 202, response.text
    return response.json()


def test_a_pegged_order_reaches_the_book_through_the_api(fresh):
    """The engine has had pegs for as long as it has had stops, and until now
    this API refused them by name; ``LiveMarket.submit`` carried no peg
    reference, so ``type: "pegged"`` came back ``invalid_order_type``. That was
    a gap in the surface rather than in the exchange, and a person clicking the
    ticket could reach an order type a program could not."""
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Pegger")
    symbol = venue.symbols()[0]

    accepted = _rest_a_peg(venue, trader, symbol, client_order_id="peg-1")
    assert accepted["peg"] == "bid"
    # A count of ticks, not a price, so it is a JSON number rather than a
    # string: a signed integer is exact as JSON already, and quoting it would
    # tell a client it is the one thing it is not.
    assert accepted["peg_offset"] == -400
    assert isinstance(accepted["peg_offset"], int)
    # A peg names no price of its own. Echoing one would be inventing it.
    assert accepted["price"] is None

    venue.pump(400, slices=16)
    working = {row["client_order_id"]: row for row in trader.get("/v1/orders").json()["orders"]}
    assert "peg-1" in working, working
    row = working["peg-1"]
    assert row["remaining"] == 2

    # It rested at a price the client never sent (the reference plus the
    # offset), and the two renderings of that price agree. That it *tracks* the
    # reference is asserted by the test below; this one asserts only that a peg
    # reaches a book at all, which it could not do before.
    listing = venue.venue.registry.require(symbol)
    assert isinstance(row["ticks"], int)
    assert Decimal(row["price"]) == listing.from_ticks(row["ticks"])


def test_a_pegged_order_keeps_its_id_and_its_seat_across_every_reprice(fresh):
    """The defect this feature was built on top of, asserted rather than
    described.

    The engine emits a ``Replaced`` every time a peg's reference moves, and
    ``TradingAgent._on_private`` treated every event that was not an ack, a
    fill or a refusal as the end of the order. Measured before the fix, on a
    buy pegged to the bid at -400 on ``ASSASSIN_IDX``: at the **first** reprice
    the order was resting for two lots at 4,897.25, the venue was reserving
    collateral against it, and the account's working orders went empty.
    ``cancel`` then answered "no such live order", so the one order type whose
    defining behaviour is that it moves was the one order type that could not
    be pulled once it had.
    """
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Tracker")
    symbol = venue.symbols()[0]

    _rest_a_peg(venue, trader, symbol, client_order_id="peg-track")
    venue.pump(300, slices=12)
    resting = trader.get("/v1/orders").json()["orders"]
    assert resting, "the peg never rested"
    order_id = resting[0]["order_id"]

    # Run until the reference has moved under it at least once.
    moves: list[dict] = []
    for _ in range(20):
        venue.pump(250, slices=10)
        moves = trader.get("/v1/account/fills?limit=200").json()["amendments"]
        if moves:
            break
    assert moves, "the reference never moved, so nothing was measured"

    # Same id throughout. A reprice is an amendment, not a new order.
    assert {entry["order_id"] for entry in moves} == {order_id}
    assert all(entry["client_order_id"] == "peg-track" for entry in moves)
    # Prices cross the wire as strings here as everywhere else.
    assert all(isinstance(entry["price"], str) for entry in moves)
    # Repricing is a new price and so a new place in the queue. A peg that
    # tracks a jumpy touch is perpetually at the back of it, and saying so is
    # the only honest thing to publish.
    assert all(entry["kept_priority"] is False for entry in moves)

    still = trader.get("/v1/orders").json()["orders"]
    assert [row["order_id"] for row in still] == [order_id], still
    assert trader.get(f"/v1/orders/{symbol}/{order_id}").status_code == 200

    pulled = trader.delete(f"/v1/orders/{symbol}/{order_id}")
    assert pulled.status_code == 200
    assert pulled.json()["already_done"] is False, "the peg had left its own blotter"


def test_a_pegged_order_that_fills_conserves_value_exactly(fresh):
    """Not "close to" zero. Money is integer minor units at a scale of a
    million for exactly this reason, and a pegged order is the one that reaches
    the book at a price nothing above the engine chose, so if any path reserved
    against a price that was not the one it traded at, this is where it would
    show."""
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Crosser")
    symbol = venue.symbols()[0]

    # Pegged to the offer at zero offset: a buy quoting at the ask is a buy
    # willing to take, which types.py names as the usual way to write one.
    placed = trader.post(
        "/v1/orders",
        {"symbol": symbol, "side": "buy", "quantity": 2, "peg": "ask", "type": "pegged"},
    )
    assert placed.status_code == 202, placed.text
    venue.pump(600, slices=24)

    fills = trader.get("/v1/account/fills?limit=200").json()["fills"]
    assert fills, "the peg never crossed"
    assert all(isinstance(entry["price"], str) for entry in fills)

    conservation = venue.venue.conservation_check()
    assert isinstance(conservation, int)
    assert conservation == 0
    published = venue.client.get("/v1/exchange").json()["conservation"]
    assert int(published) == 0 and "." not in published


@pytest.mark.parametrize("reference", sorted(choice.value for choice in PegReference))
def test_every_peg_reference_reaches_the_venue(fresh, reference):
    """The vocabulary comes from ``PegReference`` rather than a list in the
    API, so a fourth reference added to the exchange is reachable here without
    anybody editing this file, and is refused by this test if it is not. The
    same argument the time-in-force test makes about ``TimeInForce``."""
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader(f"Peg{reference}")
    symbol = venue.symbols()[0]

    accepted = _rest_a_peg(venue, trader, symbol, peg=reference)
    assert accepted["peg"] == reference
    venue.pump(400, slices=16)

    # A peg whose reference does not exist yet is *accepted and waits* ("there
    # is no best bid" is a fact about the market and not an error in the
    # order), so what is asserted is that the venue did not refuse it, not that
    # it rested. ``mid`` needs both sides and this venue is thin by
    # construction.
    refusals = trader.get("/v1/account/fills?limit=200").json()["rejections"]
    assert not [row for row in refusals if row["reason"] == "invalid_peg"], refusals


@pytest.mark.parametrize(
    "fields, code",
    [
        # A peg names no price of its own: its price is the reference plus the
        # offset, recomputed as the reference moves.
        ({"peg": "bid", "price": "1"}, "invalid_order_type"),
        ({"peg": "bid", "stop": "1"}, "invalid_order_type"),
        ({"peg": "sideways"}, "invalid_order_type"),
        # An offset from nothing is not an instruction. The engine refuses it as
        # INVALID_PEG; refused here so the client is told which field it forgot.
        ({"peg_offset": -1}, "invalid_order_type"),
        ({"peg": "bid", "peg_offset": 1.5}, "invalid_order_type"),
        # A peg is an instruction to keep tracking and these two say not to rest.
        ({"peg": "bid", "time_in_force": "ioc"}, "invalid_time_in_force"),
        ({"peg": "bid", "time_in_force": "fok"}, "invalid_time_in_force"),
        # The declared type is checked against the fields rather than obeyed.
        ({"peg": "bid", "type": "limit"}, "invalid_order_type"),
        ({"type": "pegged"}, "invalid_order_type"),
    ],
)
def test_a_peg_that_cannot_mean_anything_is_refused_in_its_own_terms(
    exchange, fields, code
):
    """Each of these is a permanent fact about the request, so each is a 400
    here rather than the 422 it would become if it were left to the venue. The
    catalogue's grouping makes that load-bearing: ``rejected_by_venue`` means
    "the market may allow it later", and no market will ever allow a pegged
    order that also names a price."""
    trader = exchange.trader("BadPeg")
    symbol = exchange.symbols()[0]
    response = trader.post("/v1/orders", _order(symbol, **fields))
    assert response.status_code in (400, 422), response.text
    assert response.json()["error"]["code"] == code, response.text


# --------------------------------------------------------------------------
# Amending an order: PATCH, and the guards that have to come with it
# --------------------------------------------------------------------------


def _one_increment_below(venue: Exchange, symbol: str, price: Decimal) -> Decimal:
    """The next price down that this contract can actually rest at.

    Snapped by the same repeated walk ``resting_price`` uses rather than by one
    subtraction, because one of these contracts carries a tick *table* whose
    increment changes with the level, so a single pass can step into a coarser
    band and land off its grid.

    Its caller must not have asked ``resting_price`` for ``inside=0``: this
    steps below the price it is given, and nothing can rest below the
    settlement floor.
    """
    listing = venue.venue.registry.require(symbol)
    target = price - listing.increment_at(price)
    for _ in range(8):
        remainder = target % listing.increment_at(target)
        if remainder == 0:
            return target
        target -= remainder
    raise AssertionError(f"no on-grid price below {price} on {symbol}")


def _rest_one(
    venue: Exchange,
    trader: "Client",
    symbol: str,
    quantity: int = 10,
    inside: int = 1,
    **extra,
):
    """One order resting where nothing else quotes, and its live row."""
    price = resting_price(venue, symbol, "buy", inside=inside)
    placed = trader.post(
        "/v1/orders",
        {
            "symbol": symbol,
            "side": "buy",
            "quantity": quantity,
            "price": str(price),
            **extra,
        },
    )
    assert placed.status_code == 202, placed.text
    venue.pump(400, slices=16)
    working = trader.get("/v1/orders").json()["orders"]
    assert working, "nothing resting to amend"
    return price, working[0]


def test_an_amendment_keeps_the_order_id_and_the_account_can_still_manage_it(fresh):
    """The blotter defect, at the layer a client sees it.

    Measured before the fix, on a bid for ten amended to six at the same price:
    the engine held it resting for six, the venue held it and reserved
    collateral against it, and the *account's own* working orders went empty.
    So ``GET /v1/orders`` published nothing, ``DELETE`` answered
    ``already_done: true`` for an order standing in the book, and
    ``DELETE /v1/orders`` walked a list the order was no longer in and left it
    there. A successful amendment is the one event that keeps an order alive
    and it was being read as the end of one.
    """
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Amender")
    symbol = venue.restable()
    _, order = _rest_one(venue, trader, symbol, quantity=10, client_order_id="amend-1")
    order_id = order["order_id"]

    amended = trader.request("PATCH", f"/v1/orders/{symbol}/{order_id}", {"quantity": 6})
    assert amended.status_code == 202, amended.text
    body = amended.json()
    assert body["order_id"] == order_id
    assert body["client_order_id"] == "amend-1"
    assert body["quantity"] == 6
    assert body["price"] is None, "no price was sent, so none should be echoed"
    assert body["previous"]["remaining"] == 10

    venue.pump(400, slices=16)

    working = trader.get("/v1/orders").json()["orders"]
    assert [row["order_id"] for row in working] == [order_id], working
    assert working[0]["remaining"] == 6
    assert working[0]["client_order_id"] == "amend-1"

    # The identifier the client chose still reaches it, and reports the size it
    # itself amended to rather than the one it originally sent.
    looked_up = trader.get("/v1/orders:by_client_order_id?id=amend-1").json()
    assert looked_up["status"] == "working"
    assert looked_up["quantity"] == 6
    assert looked_up["order"]["remaining"] == 6

    assert trader.get(f"/v1/orders/{symbol}/{order_id}").status_code == 200
    pulled = trader.delete(f"/v1/orders/{symbol}/{order_id}")
    assert pulled.json()["already_done"] is False, "the order had left its own blotter"

    venue.pump(300, slices=12)
    conservation = venue.venue.conservation_check()
    assert isinstance(conservation, int)
    assert conservation == 0


def test_queue_priority_survives_a_strict_reduction_and_nothing_else(fresh):
    """The measurement that decided the method, asserted through the field that
    publishes it.

    Two bids for ten at 100, ours first, then a sell of five sweeping the
    level::

        shrink 10 -> 6 at the same price     kept_priority=True    we fill
        grow   10 -> 14 at the same price    kept_priority=False   they fill
        10 -> 10 at the same price           kept_priority=False   they fill
        10 -> 6 at a different price         kept_priority=False   they fill

    The usual summary ("raising size loses it, lowering it keeps it") is right
    about the two ends and silent about the middle, and the middle is the case
    a client hits by accident. That is why the route is a PATCH: its contract
    is "send only what is changing", so the priority-preserving call is the
    natural one to write, where PUT's "send the whole representation" would
    have made the priority-destroying one natural instead.
    """
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Queuer")
    symbol = venue.restable()
    price, order = _rest_one(venue, trader, symbol, quantity=10)
    order_id = order["order_id"]
    lower = _one_increment_below(venue, symbol, price)
    assert lower != price

    seen = 0

    def amend(body: dict) -> bool:
        nonlocal seen
        response = trader.request("PATCH", f"/v1/orders/{symbol}/{order_id}", body)
        assert response.status_code == 202, response.text
        venue.pump(400, slices=16)
        moves = trader.get("/v1/account/fills?limit=200").json()["amendments"]
        seen += 1
        # Counted rather than assumed. The list is newest first, so reading
        # ``moves[0]`` after an amendment that had not finished crossing the
        # latency link would quietly answer with the *previous* one's priority
        # and the test would pass on the wrong event.
        assert len(moves) == seen, f"expected {seen} amendments, saw {len(moves)}"
        return moves[0]["kept_priority"]

    assert amend({"quantity": 6}) is True, "a strict reduction should keep its place"
    assert amend({"quantity": 8}) is False, "an increase is a new claim on the queue"
    assert amend({"quantity": 8}) is False, "asking for nothing still costs the place"
    # A strict reduction *and* a price change. The reduction alone would have
    # kept the place, so this isolates the price as the thing that costs it.
    assert amend({"quantity": 4, "price": str(lower)}) is False

    # And every one of them is numbered in a sequence of its own: fill 3,
    # rejection 3 and amendment 3 are unrelated events, so one ``after``
    # applied to all three would silently drop from two of them.
    blotter = trader.get("/v1/account/fills?limit=200").json()
    ids = [entry["amendment_id"] for entry in blotter["amendments"]]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids, reverse=True)
    assert blotter["cursor"]["amendments"]["total"] == max(ids)
    resumed = trader.get(f"/v1/account/fills?after_amendment={max(ids)}&limit=200").json()
    assert resumed["amendments"] == []
    assert resumed["cursor"]["after_amendment"] == max(ids)
    assert trader.get("/v1/account/fills?after_amendment=-1").status_code == 400


def test_an_amendment_carries_an_icebergs_display_size_across(fresh):
    """A replace once stripped an iceberg of the only property that made it
    one: the order came back fully displayed and published the size its owner
    was working in slices precisely so that nobody could see it. The engine
    carries it now, and an amendment cannot change it, which is why ``display``
    in the body is refused rather than ignored, since silently accepting it
    would rebuild the same failure from the client's side."""
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Hidden")
    symbol = venue.restable()
    price, order = _rest_one(venue, trader, symbol, quantity=12, display=3)
    order_id = order["order_id"]
    assert order["display"] == 3 and order["shown"] == 3

    # A price change takes the branch that rebuilds the order, which is the one
    # the display size used to fall out of.
    moved = resting_price(venue, symbol, "buy")
    amended = trader.request(
        "PATCH", f"/v1/orders/{symbol}/{order_id}", {"quantity": 12, "price": str(moved)}
    )
    assert amended.status_code == 202, amended.text
    venue.pump(400, slices=16)

    row = trader.get(f"/v1/orders/{symbol}/{order_id}").json()
    assert row["display"] == 3, "the amendment stripped the iceberg"
    assert row["shown"] == 3
    assert row["remaining"] == 12

    refused = trader.request(
        "PATCH", f"/v1/orders/{symbol}/{order_id}", {"quantity": 6, "display": 1}
    )
    assert refused.status_code == 400
    assert refused.json()["error"]["code"] == "invalid_request"
    assert refused.json()["error"]["detail"]["refused"] == ["display"]


@pytest.mark.parametrize(
    "body, code",
    [
        # Every one of these is a guard the submit path applies. A modification
        # is a request for a price and for risk exactly as an order is, and
        # ``Replace`` has walked past one of them before: the tick-grid check
        # read ``price`` while ``Replace`` carries ``new_price``.
        ({"quantity": 0}, "invalid_quantity"),
        ({"quantity": -4}, "invalid_quantity"),
        ({"quantity": 1.5}, "invalid_quantity"),
        ({"quantity": True}, "invalid_quantity"),
        ({"price": 4700.25}, "invalid_price"),
        ({"price": "9,233.75"}, "invalid_price"),
        ({"price": "nan"}, "invalid_price"),
        ({"price": "Infinity"}, "invalid_price"),
        # Nothing to change. An amendment that changes nothing still costs
        # queue position, so answering it with success would charge a client
        # for a call that did nothing it asked for.
        ({}, "invalid_request"),
        # Not amendable. ``Replace`` cannot carry any of them.
        ({"quantity": 4, "side": "sell"}, "invalid_request"),
        ({"quantity": 4, "time_in_force": "ioc"}, "invalid_request"),
        ({"quantity": 4, "stop": "1"}, "invalid_request"),
        ({"quantity": 4, "type": "limit"}, "invalid_request"),
    ],
)
def test_an_amendment_is_refused_in_its_own_terms(fresh, body, code):
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("BadAmend")
    symbol = venue.restable()
    _, order = _rest_one(venue, trader, symbol)

    response = trader.request(
        "PATCH", f"/v1/orders/{symbol}/{order['order_id']}", body
    )
    assert response.status_code in (400, 422), response.text
    assert response.json()["error"]["code"] == code, response.text

    # And the order is exactly where it was. A refused amendment is not a
    # cancel: the engine leaves the original resting, and so must this.
    venue.pump(300, slices=12)
    still = trader.get("/v1/orders").json()["orders"]
    assert [row["order_id"] for row in still] == [order["order_id"]], still
    assert still[0]["remaining"] == 10


def test_an_amendment_onto_a_price_the_contract_cannot_rest_at_is_refused(fresh):
    """The tick grid and the settlement range, which are the venue's own
    listing rules and the two the replace path has skipped before. Both are
    applied by the same ``_quotable`` the submit path calls: routing both
    through one function is what makes the miss impossible rather than merely
    fixed."""
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("OffGrid")
    symbol = venue.symbols()[0]
    listing = venue.venue.registry.require(symbol)
    _, order = _rest_one(venue, trader, symbol)
    path = f"/v1/orders/{symbol}/{order['order_id']}"

    # Half an increment off the grid, derived from the instrument rather than
    # chosen: a constant would be on one contract's grid and off another's.
    price = resting_price(venue, symbol, "buy")
    off_grid = price + listing.increment_at(price) / 2
    refused = trader.request("PATCH", path, {"price": str(off_grid)})
    assert refused.status_code == 400, refused.text
    assert refused.json()["error"]["code"] == "invalid_price"

    # And outside what the claim can still settle at. Collateral structurally
    # cannot catch this: the requirement is the worst case over the settlement
    # range, so a bid *below* the floor scores as the safest order on the book.
    # The venue's central safety mechanism rates the impossible order as the
    # safe one, which is why the range needs a listing rule of its own.
    _, high = venue.venue.bounds_in_minor(listing)
    beyond = from_money(high) + listing.tick_size
    outside = trader.request("PATCH", path, {"price": str(beyond)})
    assert outside.status_code == 400, outside.text
    assert outside.json()["error"]["code"] == "invalid_price"

    # Off-lot size, the other half of the same listing rule. Only askable on a
    # contract listed in lots of more than one, and skipped rather than faked
    # elsewhere: on a lot size of one there is no such thing as an off-lot
    # quantity, and inventing one would be hardcoding a failure.
    if listing.lot_size > 1:
        off_lot = trader.request("PATCH", path, {"quantity": listing.lot_size + 1})
        assert off_lot.status_code == 400
        assert off_lot.json()["error"]["code"] == "invalid_quantity"

    venue.pump(300, slices=12)
    still = trader.get("/v1/orders").json()["orders"]
    assert [row["order_id"] for row in still] == [order["order_id"]]
    assert still[0]["remaining"] == 10


def test_an_amendment_is_refusable_by_the_rate_limit_and_a_cancel_is_not(fresh):
    """A cancel is exempt because it can only ever reduce the venue's work and
    the participant's risk. An amendment can raise both (it is how an account
    takes on exposure it could not otherwise fund), so it is counted *and*
    refusable, exactly as ``Venue._rate_limited`` exempts only ``Cancel``.

    Then the half that matters: a participant refused for the rate must still
    be able to pull the order it has, or the limiter has trapped it in exposure
    nobody is permitted to manage."""
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Chatty")
    symbol = venue.symbols()[0]
    _, order = _rest_one(venue, trader, symbol)
    path = f"/v1/orders/{symbol}/{order['order_id']}"

    venue.venue.message_rate = 2
    accepted = 0
    refusal = None
    for size in (9, 8, 7, 6, 5):
        response = trader.request("PATCH", path, {"quantity": size})
        if response.status_code == 202:
            accepted += 1
        else:
            refusal = response
            break

    assert accepted == 2, f"accepted {accepted} against a rate of 2"
    assert refusal is not None and refusal.status_code == 429
    assert refusal.json()["error"]["code"] == "rate_limited"

    assert trader.delete(path).status_code == 200
    assert trader.delete("/v1/orders").status_code == 200


def test_a_halted_participant_cannot_amend_and_keeps_what_it_has(fresh):
    """Tested in the state the control is for, not in a calm one.

    A stopped participant may still cancel (refusing that too would trap it in
    the orders it already has, which is the opposite of what stopping it is
    for), and ``Venue.submit`` refuses it a ``Replace`` for the same reason it
    refuses a ``Submit``: an amendment is a request for risk. What this asserts
    is that the refusal leaves the order resting *and* leaves it in the
    account's own blotter, which is the half that was broken: the venue's
    refusal used to delete the order from the agent's view, after which nobody
    could pull the exposure the halt had just frozen.
    """
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Stopped")
    symbol = venue.restable()
    _, order = _rest_one(venue, trader, symbol)
    path = f"/v1/orders/{symbol}/{order['order_id']}"

    # The state directly rather than through ``kill``, which pulls the working
    # orders on its way in and would leave nothing to amend. The account is
    # read back through the API rather than taken from the key, because a key
    # is bound to a *seat* and the account it holds is re-resolved per request.
    account_id = trader.get("/v1/account").json()["account_id"]
    venue.venue.halted_participants[account_id] = "test"
    assert trader.get("/v1/account").json()["halted"] is True

    amended = trader.request("PATCH", path, {"quantity": 4})
    assert amended.status_code == 202, amended.text
    venue.pump(400, slices=16)

    refusals = trader.get("/v1/account/fills?limit=200").json()["rejections"]
    assert "participant_halted" in {row["reason"] for row in refusals}, refusals

    still = trader.get("/v1/orders").json()["orders"]
    assert [row["order_id"] for row in still] == [order["order_id"]], still
    assert still[0]["remaining"] == 10, "the refusal moved the order it refused"
    assert trader.delete(path).json()["already_done"] is False


def test_an_amendment_during_a_call_phase_is_refused_and_the_order_is_untouched(fresh):
    """A call phase is defined by the fact that nothing matches in it, and a
    replace is the one command that breaks the definition: the engine's replace
    pulls the old order and re-runs the match unconditionally, never consulting
    the phase. Measured at the venue: a replace during a halt printed 20 lots
    at 17,000 against an order only resting there because the auction had not
    run yet, and against market-on-open interest (which rests at a sentinel so
    it crosses every candidate), it printed at -4,611,686,018,427,387,904."""
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Auctioned")
    symbol = venue.symbols()[0]
    _, order = _rest_one(venue, trader, symbol)
    path = f"/v1/orders/{symbol}/{order['order_id']}"

    venue.venue.halt(symbol, "test")
    assert venue.client.get(f"/v1/instruments/{symbol}").json()["session"] != "continuous"

    amended = trader.request("PATCH", path, {"quantity": 4})
    assert amended.status_code == 202, amended.text
    venue.pump(400, slices=16)

    refusals = trader.get("/v1/account/fills?limit=200").json()["rejections"]
    assert "not_accepted_in_auction" in {row["reason"] for row in refusals}, refusals

    still = trader.get("/v1/orders").json()["orders"]
    assert [row["order_id"] for row in still] == [order["order_id"]], still
    assert still[0]["remaining"] == 10
    # And it can still be withdrawn: cancels stay legal in a call phase so a
    # participant can tidy up.
    assert trader.delete(path).json()["already_done"] is False


def test_an_amendment_after_the_close_is_refused_and_the_order_is_untouched(fresh):
    """The other half of the session guard, and the one an expiry reaches.

    ``Venue.submit`` refuses a ``Submit`` and a ``Replace`` on the same line
    once a symbol stops accepting orders, because once the outcome is
    determined nobody may take new risk, and an amendment is new risk. What
    stays legal is the cancel, so an account can tidy up, and this asserts that
    the pair still works together: refused amendment, order untouched, order
    still pullable.
    """
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Closed")
    symbol = venue.symbols()[0]
    _, order = _rest_one(venue, trader, symbol)
    path = f"/v1/orders/{symbol}/{order['order_id']}"

    venue.venue.close(symbol)
    assert venue.client.get(f"/v1/instruments/{symbol}").json()["session"] == "closed"

    amended = trader.request("PATCH", path, {"quantity": 4})
    assert amended.status_code == 202, amended.text
    venue.pump(400, slices=16)

    refusals = trader.get("/v1/account/fills?limit=200").json()["rejections"]
    assert "already_terminal" in {row["reason"] for row in refusals}, refusals

    still = trader.get("/v1/orders").json()["orders"]
    assert [row["order_id"] for row in still] == [order["order_id"]], still
    assert still[0]["remaining"] == 10
    assert trader.delete(path).json()["already_done"] is False


def test_an_amendment_is_not_a_free_re_collateralisation(fresh):
    """An amendment is measured as the position it *results in*, not as
    exposure piled on top of the order it supersedes, and not as a fresh start
    either.

    Asserted against a control rather than against a number, because a number
    would be a hardcoded fact about one contract's price. Two identical
    accounts, one that works ten lots and amends to twenty and one that works
    twenty outright, must have reserved exactly the same collateral. Reserving
    on top of the superseded order would make the first larger; releasing the
    old reservation and forgetting the position would make it smaller.
    """
    venue = fresh()
    venue.pump(400, slices=16)
    amender = venue.trader("Grower")
    control = venue.trader("Control")
    symbol = venue.restable()

    price, order = _rest_one(venue, amender, symbol, quantity=10)
    placed = control.post(
        "/v1/orders",
        {"symbol": symbol, "side": "buy", "quantity": 20, "price": str(price)},
    )
    assert placed.status_code == 202, placed.text

    grown = amender.request(
        "PATCH", f"/v1/orders/{symbol}/{order['order_id']}", {"quantity": 20}
    )
    assert grown.status_code == 202, grown.text
    venue.pump(500, slices=20)

    assert amender.get("/v1/orders").json()["orders"][0]["remaining"] == 20
    assert control.get("/v1/orders").json()["orders"][0]["remaining"] == 20
    assert (
        amender.get("/v1/account").json()["collateral"]
        == control.get("/v1/account").json()["collateral"]
    )

    # And the ceiling is real. Beyond what the account can fund the venue
    # refuses, and (the half that was broken) leaves the order resting and in
    # its owner's blotter, so the exposure the refusal protected is still
    # something its owner can pull.
    #
    # Derived from the seat and from where the order actually rests, rather
    # than named. What an account cannot fund depends on the price, and a
    # constant cannot know it: this asked for 100,000 lots, which was far
    # beyond the seat when `resting_price` parked orders mid-range and is
    # 25,000 against a seat of 250,000 now that it parks them one increment
    # above the bottom of the range. The assertion still passed, on a venue
    # that had correctly refused nothing at all.
    listing = venue.venue.registry.require(symbol)
    funded = Decimal(amender.get("/v1/account").json()["cash"]) / (
        price * listing.lot_size
    )
    unaffordable = int(funded) * 2 * listing.lot_size
    over = amender.request(
        "PATCH", f"/v1/orders/{symbol}/{order['order_id']}", {"quantity": unaffordable}
    )
    assert over.status_code == 202, over.text
    venue.pump(400, slices=16)

    refusals = amender.get("/v1/account/fills?limit=200").json()["rejections"]
    assert "insufficient_collateral" in {row["reason"] for row in refusals}, refusals
    still = amender.get("/v1/orders").json()["orders"]
    assert [row["order_id"] for row in still] == [order["order_id"]], still
    assert still[0]["remaining"] == 20, "the refused amendment moved the order"
    assert amender.delete(f"/v1/orders/{symbol}/{order['order_id']}").json()[
        "already_done"
    ] is False

    venue.pump(300, slices=12)
    conservation = venue.venue.conservation_check()
    assert isinstance(conservation, int)
    assert conservation == 0


def test_amending_an_order_that_is_not_yours_answers_as_one_that_never_existed(fresh):
    """Confirming that an id exists but is not yours tells a stranger something
    about a stranger's account, which is the argument ``GET`` on the same
    address already makes. 404 rather than ``DELETE``'s 200, because an
    amendment to an order that is not resting is not a correct outcome the
    client asked for: there is nothing to carry the new terms."""
    venue = fresh()
    venue.pump(400, slices=16)
    owner = venue.trader("Owner")
    stranger = venue.trader("Stranger")
    symbol = venue.symbols()[0]
    _, order = _rest_one(venue, owner, symbol)
    order_id = order["order_id"]

    theirs = stranger.request(
        "PATCH", f"/v1/orders/{symbol}/{order_id}", {"quantity": 4}
    )
    assert theirs.status_code == 404, theirs.text
    assert theirs.json()["error"]["code"] == "not_found"

    # An id that never existed anywhere answers with the same code and the same
    # status. The sentences differ only by the id the caller itself sent back to
    # it, which discloses nothing: what a stranger must not be able to learn is
    # whether an id *exists*, and neither answer says.
    never = stranger.request("PATCH", f"/v1/orders/{symbol}/999999", {"quantity": 4})
    assert never.status_code == theirs.status_code == 404
    assert never.json()["error"]["code"] == theirs.json()["error"]["code"]

    mistyped = owner.request("PATCH", "/v1/orders/NOT_A_SYMBOL/1", {"quantity": 4})
    assert mistyped.status_code == 400
    assert mistyped.json()["error"]["code"] == "invalid_symbol"

    # Untouched by any of it.
    venue.pump(300, slices=12)
    assert owner.get("/v1/orders").json()["orders"][0]["remaining"] == 10


def test_no_float_appears_in_an_amendment_or_a_peg_payload(fresh):
    """Prices cross the wire as strings, in both directions. A JSON number is a
    double, and this venue's accounting is exact integers precisely so that the
    conservation check can be exactly zero rather than nearly zero."""
    venue = fresh()
    venue.pump(400, slices=16)
    # Two seats, because ``_rest_one`` reads the first working order of the
    # account it is given and a peg resting alongside would be that one.
    pegger = venue.trader("ExactPeg")
    trader = venue.trader("Exact")
    symbol = venue.symbols()[0]

    peg = _rest_a_peg(venue, pegger, symbol)
    assert floats_in(peg) == []

    _, order = _rest_one(venue, trader, symbol)
    amended = trader.request(
        "PATCH",
        f"/v1/orders/{symbol}/{order['order_id']}",
        {"quantity": 6, "price": str(resting_price(venue, symbol, "buy"))},
    )
    assert amended.status_code == 202, amended.text
    assert floats_in(json.loads(amended.text)) == []

    venue.pump(500, slices=20)
    blotter = json.loads(trader.get("/v1/account/fills?limit=200").text)
    assert floats_in(blotter) == [], blotter


# --------------------------------------------------------------------------
# Orders the venue should not have to see
# --------------------------------------------------------------------------


def _order(symbol: str, **fields) -> dict:
    return {"symbol": symbol, "side": "buy", "quantity": 1, **fields}


@pytest.mark.parametrize(
    "fields, code",
    [
        ({"side": "byu"}, "invalid_side"),
        ({"side": None}, "invalid_side"),
        ({"quantity": 0}, "invalid_quantity"),
        ({"quantity": -3}, "invalid_quantity"),
        ({"quantity": 1.5}, "invalid_quantity"),
        ({"quantity": "many"}, "invalid_quantity"),
        ({"quantity": True}, "invalid_quantity"),
        ({"quantity": None}, "invalid_quantity"),
        ({"price": "9,233.75"}, "invalid_price"),
        ({"price": "-100"}, "invalid_price"),
        ({"price": "1e9"}, "invalid_price"),
        # ``Decimal`` parses both of these happily. A NaN then compares false
        # against everything, so it passes a range check by looking like neither
        # too high nor too low, and an infinity survives the integrality check
        # and reaches ``int()``, which raises. Refused as prices rather than
        # allowed to become a 500 several frames later.
        ({"price": "nan"}, "invalid_price"),
        ({"price": "Infinity"}, "invalid_price"),
        ({"quantity": "Infinity"}, "invalid_quantity"),
        ({"stop": "nan"}, "invalid_price"),
        ({"time_in_force": "eventually"}, "invalid_time_in_force"),
        ({"type": "pegged", "price": "1"}, "invalid_order_type"),
        ({"type": "nonsense"}, "invalid_order_type"),
        ({"display": -1}, "invalid_quantity"),
    ],
)
def test_a_malformed_order_is_refused_in_its_own_terms(exchange, fields, code):
    """Every one of these is a fact about the request, so the request is told.
    What is left to the venue (collateral, the price band, an auction phase) is
    only knowable there."""
    trader = exchange.trader("Malformed")
    symbol = exchange.symbols()[0]
    response = trader.post("/v1/orders", _order(symbol, **fields))
    assert response.status_code in (400, 422), response.text
    assert response.json()["error"]["code"] == code, response.text


def test_a_price_sent_as_a_json_number_is_refused(exchange):
    """``json.loads('{"price": 4700.10}')`` yields a binary double that is not
    4700.10. Accepting it and rounding is how an order rests at a price nobody
    chose, which is the argument ``Instrument.to_ticks`` already makes."""
    trader = exchange.trader("Floaty")
    symbol = exchange.symbols()[0]
    response = trader.post("/v1/orders", _order(symbol, price=4700.10))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_price"
    assert "string" in response.json()["error"]["message"]


def test_a_price_off_the_instruments_grid_is_refused(exchange):
    """Per instrument, from the instrument's own increment. Nothing here knows
    what any contract's tick is."""
    trader = exchange.trader("Offgrid")
    for symbol in exchange.symbols():
        instrument = exchange.venue.registry.require(symbol)
        mark = exchange.venue.mark_price(symbol)
        off = mark + instrument.increment_at(mark) / 3
        if instrument.on_grid(off):
            continue
        response = trader.post("/v1/orders", _order(symbol, price=str(off)))
        assert response.status_code == 400, symbol
        assert response.json()["error"]["code"] == "invalid_price", symbol
        return
    pytest.skip("no instrument with a divisible increment")


def test_an_order_type_that_disagrees_with_its_fields_is_refused(exchange):
    """A declared type is checked against the fields rather than obeyed, so a
    client whose declaration and whose fields disagree is told which, instead
    of one of them silently winning."""
    trader = exchange.trader("Mismatch")
    symbol = exchange.symbols()[0]
    response = trader.post("/v1/orders", _order(symbol, type="limit"))
    assert response.status_code == 400
    body = response.json()["error"]
    assert body["code"] == "invalid_order_type"
    assert body["detail"] == {"declared": "limit", "derived": "market"}


def test_an_off_lot_quantity_is_refused_before_it_costs_a_round_trip(fresh):
    """The venue refuses it asynchronously, as a ``RejectReason`` in a blotter
    the client has to poll for. Measured on a contract listed in lots of ten,
    an order for seven was acknowledged and rested."""
    venue = fresh()
    trader = venue.trader("Lots")
    symbol = venue.symbols()[0]
    instrument = venue.venue.registry.require(symbol)
    # The listing is in lots of one here, so the rule is exercised by changing
    # the listing rather than by finding a contract that happens to break it:
    # there is no special symbol to reach for and there should not be.
    object.__setattr__(instrument, "lot_size", 10)
    try:
        response = trader.post("/v1/orders", _order(symbol, quantity=7))
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_quantity"
        assert trader.post("/v1/orders", _order(symbol, quantity=20)).status_code == 202
    finally:
        object.__setattr__(instrument, "lot_size", 1)


def test_a_body_that_is_not_json_is_refused_as_a_bad_request(exchange):
    trader = exchange.trader("Garbage")
    stamp = f"{time.time():.3f}"
    body = b"not json at all"
    response = exchange.client.post(
        "/v1/orders",
        content=body,
        headers={
            HEADER_KEY: trader.key_id,
            HEADER_TIMESTAMP: stamp,
            HEADER_SIGNATURE: sign(trader.secret, "POST", "/v1/orders", stamp, body),
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


# --------------------------------------------------------------------------
# The rate limit the venue already has, said out loud
# --------------------------------------------------------------------------


def test_the_venues_message_rate_surfaces_as_a_429(fresh):
    """The venue throttles on the far side of a latency link, so a client that
    overruns it gets 202 for every order and then silence: the refusals land in
    a blotter it has to poll for and correlate. The rate is the venue's own;
    nothing here invents a number."""
    venue = fresh()
    venue.pump(200, slices=8)
    venue.venue.message_rate = 5
    trader = venue.trader("Firehose")
    symbol = venue.symbols()[0]

    accepted = 0
    refusal = None
    for _ in range(12):
        response = trader.post("/v1/orders", _order(symbol, quantity=1))
        if response.status_code == 202:
            accepted += 1
        else:
            refusal = response
            break

    assert accepted == 5, f"accepted {accepted} against a rate of 5"
    assert refusal is not None and refusal.status_code == 429
    body = refusal.json()["error"]
    assert body["code"] == "rate_limited"
    assert body["detail"]["limit"] == 5

    # A cancel is counted and never refused, exactly as the venue treats a
    # reducing command: a participant that cannot withdraw is one holding
    # exposure nobody is permitted to manage.
    assert trader.delete(f"/v1/orders/{symbol}/1").status_code == 200
    assert trader.delete("/v1/orders").status_code == 200


def test_the_rate_limit_is_per_seat_not_per_venue(fresh):
    venue = fresh()
    venue.pump(200, slices=8)
    venue.venue.message_rate = 2
    loud = venue.trader("Loud")
    quiet = venue.trader("Quiet")
    symbol = venue.symbols()[0]

    for _ in range(3):
        loud.post("/v1/orders", _order(symbol))
    assert loud.post("/v1/orders", _order(symbol)).status_code == 429
    assert quiet.post("/v1/orders", _order(symbol)).status_code == 202


def test_no_configured_rate_means_no_throttle(fresh):
    """Copied from the venue rather than invented, so the two cannot disagree
    about what is allowed. The dashboard's venue configures none."""
    venue = fresh()
    venue.pump(200, slices=8)
    assert venue.venue.message_rate is None
    trader = venue.trader("Unlimited")
    symbol = venue.symbols()[0]
    for _ in range(30):
        assert trader.post("/v1/orders", _order(symbol)).status_code == 202


# --------------------------------------------------------------------------
# The trap: a key is bound to a seat, not to an account id
# --------------------------------------------------------------------------


def test_a_key_still_trades_its_own_account_after_a_rebuild(fresh):
    """The failure this module is arranged around.

    ``reconfigure`` discards the market and every account in it, and
    ``LiveMarket.trader`` answers an id it has never heard of with the *shared*
    account. A key that had captured ``you-1`` at issue time would therefore
    trade a communal seat after the first rebuild: one balance, one blotter,
    every stale credential able to cancel every other's orders. That is the
    same bug the browser cookie was fixed for, and this asserts the fix rather
    than the intention: the two keys must land in two different accounts, and
    neither of them in the shared one.
    """
    venue = fresh()
    venue.pump(200, slices=8)
    one = venue.trader("Ash")
    two = venue.trader("Nita")
    before = {one.account_id, two.account_id}
    assert len(before) == 2

    shared = str(venue.runner.market.human.agent_id)
    assert shared not in before

    venue.runner.reconfigure(MarketConfig(seed=11, opening_auction=False))
    venue.pump(300, slices=12)
    assert venue.runner.generation == 1

    after_one = one.get("/v1/account").json()
    after_two = two.get("/v1/account").json()
    assert after_one["generation"] == after_two["generation"] == 1
    assert after_one["account_id"] != after_two["account_id"]
    assert after_one["account_id"] != shared
    assert after_two["account_id"] != shared
    assert after_one["seat"] == "Ash" and after_two["seat"] == "Nita"

    # And they trade separately. One buys; the other's blotter stays empty.
    symbol = venue.symbols()[0]
    assert one.post("/v1/orders", _order(symbol, quantity=2)).status_code == 202
    venue.pump(400, slices=16)

    mine = one.get("/v1/account/positions").json()["positions"]
    theirs = two.get("/v1/account/positions").json()["positions"]
    assert any(row["quantity"] for row in mine), mine
    assert theirs == []
    assert Decimal(two.get("/v1/account").json()["cash"]) == Decimal(
        two.get("/v1/account").json()["starting_cash"]
    )


def test_a_rebuild_does_not_leak_a_new_account_per_request(fresh):
    """Re-seating on *every* request would be the same bug with the opposite
    sign: a key would get a fresh empty account each call and never find its
    own position again. The binding is re-resolved per request and re-seated
    only when the market underneath it has changed."""
    venue = fresh()
    trader = venue.trader("Steady")
    first = trader.get("/v1/account").json()["account_id"]
    seats = len(venue.runner.market.traders)
    for _ in range(5):
        assert trader.get("/v1/account").json()["account_id"] == first
    assert len(venue.runner.market.traders) == seats

    venue.runner.reconfigure(MarketConfig(seed=3, opening_auction=False))
    after = trader.get("/v1/account").json()["account_id"]
    seats_after = len(venue.runner.market.traders)
    for _ in range(5):
        assert trader.get("/v1/account").json()["account_id"] == after
    assert len(venue.runner.market.traders) == seats_after


def test_the_application_can_own_the_re_seating_instead(fresh):
    """``dashboard/server.py`` already resolves a session id to the account it
    holds now, from the same table its cookies use. When the app supplies that,
    a key and the browser session that minted it stay in one account across a
    rebuild rather than being seated separately into two."""
    routed: dict[str, object] = {}

    def seat_now(token: str):
        return routed.get(token)

    venue = fresh(seat_now=seat_now)
    token = venue.browser("Delegated")
    # The application seats this person itself, exactly as the dashboard does
    # when the page is first loaded.
    routed[token] = venue.runner.market.seat("Delegated")
    trader = Client(venue, venue.issue(token), token)

    assert trader.get("/v1/account").json()["account_id"] == str(routed[token])

    venue.runner.reconfigure(MarketConfig(seed=5, opening_auction=False))
    routed[token] = venue.runner.market.seat("Delegated")
    assert trader.get("/v1/account").json()["account_id"] == str(routed[token])


# --------------------------------------------------------------------------
# The invariant
# --------------------------------------------------------------------------


def test_value_is_conserved_exactly_through_the_api(fresh):
    """Zero, as an integer, after everything this API can do to a market.

    The sharpest check available on the whole portfolio layer, and the reason
    the ledger is kept in integers: trading moves value between participants,
    it does not create it. A near-zero would mean an accounting leak that the
    money layer's own docstring says compounds per fill.
    """
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Conserver")
    other = venue.trader("Counterparty")

    for symbol in venue.symbols()[:6]:
        trader.post("/v1/orders", _order(symbol, quantity=2))
        trader.post(
            "/v1/orders",
            _order(
                symbol,
                quantity=3,
                price=str(resting_price(venue, symbol, "buy")),
                time_in_force="gtc",
            ),
        )
        other.post(
            "/v1/orders",
            _order(
                symbol,
                side="sell",
                quantity=2,
                price=str(resting_price(venue, symbol, "sell")),
                time_in_force="gtc",
            ),
        )
        venue.pump(200, slices=8)
        assert int(venue.venue.conservation_check()) == 0

    trader.delete("/v1/orders")
    other.delete("/v1/orders")
    venue.pump(600, slices=24)

    assert int(venue.venue.conservation_check()) == 0
    assert int(venue.client.get("/v1/exchange").json()["conservation"]) == 0


def test_every_asset_class_takes_an_order_through_the_api(fresh):
    """The venue is uniform over its classes, so the API is too. Nine classes,
    one code path, no branch anywhere on what kind of claim it is.

    The only test that asks ``resting_price`` for the bound itself, because it
    is the only one that has to rest on *every* class at once. A deep put whose
    whole traded range sits at the bottom of its declared one has its bid on
    the floor already, so one increment in is inside its spread and fills;
    ``resting_price`` says what that measurement was.
    """
    venue = fresh()
    venue.pump(400, slices=16)
    trader = venue.trader("Everything")

    for instrument_class in ALL_CLASSES:
        symbol = venue.symbol_of(instrument_class)
        response = trader.post(
            "/v1/orders",
            _order(
                symbol,
                quantity=1,
                price=str(resting_price(venue, symbol, "buy", inside=0)),
                time_in_force="gtc",
                client_order_id=f"cid-{instrument_class}",
            ),
        )
        assert response.status_code == 202, (instrument_class, response.text)

    venue.pump(500, slices=20)
    working = trader.get("/v1/orders?limit=1000").json()
    classes = {
        venue.venue.registry.require(row["symbol"]).instrument_class
        for row in working["orders"]
    }
    assert classes == set(ALL_CLASSES), f"never rested: {set(ALL_CLASSES) - classes}"
    assert int(venue.venue.conservation_check()) == 0
