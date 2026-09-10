"""The exchange website: its API, and the views rendered against real data.

Two halves. The Python half exercises the HTTP and WebSocket surface against a
running market. The JavaScript half runs under node, the views are pure
functions of a store, so they can be rendered and inspected without a browser,
which catches the class of bug that looks fine until someone opens the page: a
server-side rename surfacing as ``undefined``, a decimal string rendering as
``NaN``, an object interpolated into text.

That does not replace looking at the page. It does mean renaming a field on the
server fails a test instead of silently blanking a panel.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arena.portfolio.account import Account
from arena.portfolio.money import from_money, to_money
from arena.sim.time import seconds

from dashboard.server import app, runner
from dashboard.state import FEE_SCHEDULES, MarketConfig, MarketRunner

REPO = Path(__file__).resolve().parents[1]
SYMBOL = "EMBER_OBJECTIVE_WR"


def as_operator() -> dict[str, str]:
    """The header that reaches the controls which change the market for all.

    Imported lazily inside the function so this module still imports cleanly if
    the auth module is ever moved: the tests that do not touch operator routes
    should not fail on an import they never use.
    """
    from dashboard.operator_auth import OPERATOR_HEADER, operator_token

    return {OPERATOR_HEADER: operator_token()}


@pytest.fixture(scope="module")
def client():
    """A live server with a market that has actually traded.

    Speed is raised so a few wall-clock seconds buy a few simulated minutes;
    the market is stepped by the server's own pump, never by the test, because
    two things advancing one event kernel corrupts it.
    """
    # Re-apply the server's OWN api configuration before yielding.
    #
    # `arena.api.rest` is configured through module globals, and
    # `dashboard/server.py` sets them at import with the cookie-reading seat
    # hook this file's tests depend on. But `tests/test_api.py` builds its own
    # app and calls `rest.configure(...)` with a header-based hook, which
    # replaces those globals process-wide, and pytest collects `test_api.py`
    # before `test_dashboard.py`, so by the time these tests run the cookie
    # path has been swapped out from under them. Every test here passed in
    # isolation and one failed in the suite, which is the signature of exactly
    # this and not of a bug in the thing under test.
    #
    # Restoring it here rather than in the other file, because a test file
    # should establish the state it needs instead of relying on nothing else
    # having disturbed it.
    import dashboard.server as server
    from arena.api import rest

    rest.configure(
        keys=server.api_keys,
        runner=runner,
        browser_seat=lambda request: (
            rest.Seat(sid, server._SEATS[sid].name)
            if (sid := server._session_id(request)) in server._SEATS
            else None
        ),
        seat_now=server._seat_now,
    )

    with TestClient(app) as test_client:
        runner.set_speed(40.0)
        time.sleep(8)
        runner.set_speed(1.0)
        yield test_client


# --------------------------------------------------------------------------
# Configuration comes from a browser, so it is never trusted
# --------------------------------------------------------------------------


def test_configuration_clamps_hostile_input():
    """Every field is bounded. An unbounded agent count or speed would let a
    page freeze the server that is serving it."""
    config = MarketConfig.from_dict(
        {
            "seed": -1,
            "speed": 10_000,
            "flow_traders": 999_999,
            "fees": "../../etc/passwd",
            "price_band": 900,
        }
    )
    assert 0 <= config.seed < 2**31
    assert config.speed <= 50.0
    assert config.flow_traders <= 24
    assert config.fees in FEE_SCHEDULES
    assert config.price_band <= 5.0


def test_configuration_survives_an_empty_payload():
    assert MarketConfig.from_dict({}).seed == 7


def test_a_blank_price_band_means_no_breaker():
    for blank in (None, "", "none"):
        assert MarketConfig.from_dict({"price_band": blank}).price_band is None


# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------


def test_instruments_carry_their_contract_terms(client):
    """A market where you cannot read the contract is a casino with extra steps."""
    payload = client.get("/api/instruments").json()["instruments"]
    assert len(payload) >= 5
    for instrument in payload:
        assert instrument["symbol"]
        assert instrument["spec_digest"]
        assert instrument["expiry"]
        assert instrument["session"] in {"continuous", "pre_open", "auction", "closed"}


def test_the_session_endpoint_describes_the_venue(client):
    payload = client.get("/api/session").json()
    assert set(payload["config"]) >= {"seed", "fees", "flow_traders", "arbitrageur"}
    assert "maker-taker" in payload["fee_schedules"]
    assert payload["sessions"][SYMBOL]


def test_the_roster_reports_equity_as_a_price_not_as_minor_units(client):
    """The same failure as the settlement bug, a million times over.

    ``Account.equity`` answers in the integer minor units the ledger is kept in
    (1e-6 of a price unit) and the roster published that straight into a column
    the page renders with the money formatter. A maker worth 113,125,513.21 was
    drawn as "113125513.21M", and a *person's own seat* appeared on the same
    table as "143745.00M" while the header two panels away read "143.7k".
    Nothing looked broken. It looked like a big number.
    """
    # Against a market nobody is pumping, so the marks cannot move between the
    # reading and the recomputation. A live one is checked below by magnitude,
    # which is all a factor of a million ever needed.
    still = MarketRunner()
    still.start()
    still.market.kernel.advance(until=seconds(90))
    venue = still.market.venue
    marks = venue.marks()

    checked = 0
    for entry in still.agents():
        account = venue.accounts.get(entry["id"])
        if entry["equity"] is None or account is None:
            continue
        assert Decimal(entry["equity"]) == from_money(account.equity(marks)), (
            f"{entry['id']} is published as {entry['equity']} against a real "
            "equity of "
            f"{from_money(account.equity(marks))}, which is what the ledger's "
            "own integer unit looks like when it is read as a price"
        )
        checked += 1
    assert checked > 5, "too few accounts had an equity to check"


def test_the_served_roster_reports_equity_at_a_believable_size(client):
    """The same check against the market that is actually running.

    Exactness is not available here: a mark that moved between the reading and
    the recomputation is the market working. Magnitude is, and magnitude is the
    whole of the bug: trading moves value between participants rather than
    creating it, so an account cannot drift a thousandfold from the capital it
    opened with, and the figure that was being published was off by a million.
    """
    roster = client.get("/api/agents").json()["agents"]
    checked = 0
    for entry in roster:
        account = runner.market.venue.accounts.get(entry["id"])
        if entry["equity"] is None or account is None:
            continue
        opened = from_money(account.starting_cash)
        assert abs(Decimal(entry["equity"])) <= opened * 1000, (
            f"{entry['id']} shows {entry['equity']} against opening capital of "
            f"{opened}"
        )
        checked += 1
    assert checked > 5, "too few accounts had an equity to check"


def test_a_halt_record_reports_prices_not_ticks():
    """The Lab draws `price` and `reference` into columns headed as such.

    The venue records both in the unit it matches in, which is ticks, and that
    is right for the venue. Published unconverted, a band break on a contract
    quoted on a 0.25 grid printed 1,989 in a price column against a real price
    of 497.25, on a claim whose entire settlement range is 0 to 1,000. A
    number four times outside the range the same page publishes reads as a
    number, not as a bug.

    Run with a band tight enough that the breaker trips within the first few
    simulated seconds, and from a fixed seed, so it is the conversion under
    test rather than the weather.
    """
    tight = MarketRunner(MarketConfig(seed=7, price_band=0.004))
    tight.start()
    for moment in range(5, 120, 5):
        tight.market.kernel.advance(until=seconds(moment))
        if any("price" in halt for halt in tight.market.venue.halts):
            break

    published = [h for h in tight.session_state()["halts"] if "price" in h]
    assert published, "the breaker never tripped, so nothing is being checked"

    for halt in published:
        instrument = tight.market.venue.registry.require(halt["symbol"])
        low, high = instrument.value_bounds
        for field in ("price", "reference"):
            value = Decimal(str(halt[field]))
            assert low <= value <= high, (
                f"{halt['symbol']} halted at {field}={value}, outside its own "
                f"settlement range {low}..{high}, which is what a tick count "
                "looks like when it is drawn under a price heading"
            )


def test_the_indicative_price_means_the_same_thing_on_both_endpoints():
    """One name, two units, four times apart.

    The socket converts the auction's clearing price and the ladder
    endpoint did not, so the same auction on the same contract at the same
    instant was published as "5003.00" in one place and 20012 in the other.
    Nothing draws the ladder's copy yet, which is the only reason it never
    reached a screen, and is exactly the position the settlement figure was
    in before someone drew it.
    """
    opening = MarketRunner()
    opening.start()
    opening.market.kernel.advance(until=seconds(3))

    checked = 0
    snapshot = opening.market.snapshot()
    for symbol, book in snapshot["books"].items():
        ladder = opening.indicative(symbol)
        if book["indicative"] is None or ladder is None:
            continue
        assert Decimal(str(ladder["price"])) == Decimal(book["indicative"]), (
            f"{symbol}: the ladder says {ladder['price']} and the socket says "
            f"{book['indicative']} about the same auction"
        )
        checked += 1
    assert checked, "no symbol was in a call phase, so nothing was compared"


def test_the_agent_roster_is_published(client):
    """Who else is in the market is part of the market's information."""
    agents = client.get("/api/agents").json()["agents"]
    # By role, not by class name. Asserting the class was wrong twice over: it
    # broke the moment a specialised maker was wired in, and it was checking
    # an implementation detail to answer a question about the market.
    roles = {a["role"] for a in agents}
    assert "market maker" in roles
    assert {"informed", "uninformed"} <= roles
    assert any(a["fills"] > 0 for a in agents), "nobody traded"


def test_the_book_endpoint_returns_a_two_sided_ladder(client):
    # Polled rather than sampled once. The market this serves has an opening
    # call and a circuit breaker, so there are real moments when a book has one
    # side or none, and a test that happens to land on one of them is reporting
    # the clock rather than the endpoint.
    for _ in range(40):
        book = client.get(f"/api/book/{SYMBOL}?levels=12").json()
        if book["bids"] and book["asks"]:
            break
        time.sleep(0.05)
    assert book["bids"] and book["asks"]
    assert book["session"], "a ladder without its phase cannot be read"
    if book["session"] == "continuous":
        best_bid = float(book["bids"][0][0])
        best_ask = float(book["asks"][0][0])
        assert best_bid < best_ask, "the published ladder is crossed"
    else:
        # A call phase accumulates orders without matching them, so a crossed
        # ladder is the mechanism rather than a fault, and the indicative price
        # is what a reader should be looking at instead.
        assert "indicative" in book


def test_the_book_endpoint_bounds_the_level_count(client):
    """A browser asking for a million levels must not get them."""
    book = client.get(f"/api/book/{SYMBOL}?levels=100000").json()
    assert len(book["bids"]) <= 60


def test_history_accumulates(client):
    """Asserts that the series *grows*, not that it grows at a given rate.

    Two earlier versions of this test asserted a sample count (first `> 5`,
    then `>= 12`) and both failed for the same reason: the recorder samples on
    the server's own tick, so how many points exist at any moment depends on
    how loaded the machine is and how many instruments are listed. Neither
    number was ever the point. What the chart needs is a series that
    accumulates and never doubles back, and that is what is checked.
    """
    first = client.get(f"/api/history/{SYMBOL}").json()
    deadline = time.monotonic() + 25
    later = first
    while len(later["t"]) <= len(first["t"]) and time.monotonic() < deadline:
        time.sleep(0.5)
        later = client.get(f"/api/history/{SYMBOL}").json()

    assert len(later["t"]) > len(first["t"]), "the series never grew"
    assert len(later["t"]) == len(later["mid"])
    # Simulated time only moves forward, so a chart drawn from this cannot
    # double back on itself.
    assert later["t"] == sorted(later["t"])


def test_diagnostics_use_the_research_estimators(client):
    """The same code the papers would quote, not a second implementation."""
    report = client.get(f"/api/diagnostics/{SYMBOL}").json()
    if report.get("pending"):
        pytest.skip("not enough observations yet in this run")
    names = {v["name"] for v in report["verdicts"]}
    assert any("Hill" in n for n in names)
    assert any("variance ratio" in n for n in names)


@pytest.mark.parametrize("path", ["/api/history/NOPE", "/api/diagnostics/NOPE", "/api/book/NOPE"])
def test_unknown_symbols_are_refused(client, path):
    assert client.get(path).status_code == 404


# --------------------------------------------------------------------------
# Control
# --------------------------------------------------------------------------


def test_halting_and_reopening_round_trips(client):
    """A halt accumulates orders; the reopen is an auction, not a free-for-all."""
    halted = client.post(f"/api/session/{SYMBOL}/halt", headers=as_operator()).json()
    assert halted["ok"] and halted["session"] == "auction"

    reopened = client.post(f"/api/session/{SYMBOL}/uncross", headers=as_operator()).json()
    assert reopened["ok"] and reopened["session"] == "continuous"


def test_uncrossing_a_continuous_symbol_is_refused(client):
    """There is no call phase to clear, and saying so beats pretending."""
    client.post(f"/api/session/{SYMBOL}/uncross", headers=as_operator())
    again = client.post(f"/api/session/{SYMBOL}/uncross", headers=as_operator()).json()
    assert again["ok"] is False


def test_halting_an_unknown_symbol_is_refused(client):
    assert client.post("/api/session/NOPE/halt", headers=as_operator()).json()["ok"] is False


# --------------------------------------------------------------------------
# The live socket
# --------------------------------------------------------------------------


def test_the_socket_carries_everything_a_screen_needs(client):
    with client.websocket_connect("/ws") as socket:
        payload = socket.receive_json()
    assert set(payload) >= {
        "clock", "events", "books", "tape", "account", "orders", "conservation",
        "sessions", "generation", "speed",
    }
    book = payload["books"][SYMBOL]
    assert set(book) >= {"bids", "asks", "mark", "class", "contract", "tick"}
    assert book["contract"]["payoff"]


def test_value_is_conserved_in_the_served_market(client):
    """The invariant the header light reports. It must be exactly zero."""
    with client.websocket_connect("/ws") as socket:
        payload = socket.receive_json()
    assert int(payload["conservation"]) == 0


@pytest.mark.parametrize("tif", ["gtc", "ioc", "fok", "post_only"])
def test_every_time_in_force_reaches_the_venue(client, tif):
    """The browser can reach post-only and fill-or-kill, not just the defaults."""
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        socket.send_json(
            {"action": "submit", "symbol": SYMBOL, "side": "buy",
             "quantity": 1, "price": "4600", "tif": tif}
        )
        for _ in range(40):
            message = socket.receive_json()
            if "ack" in message:
                assert message["ack"]["ok"] is True
                return
    pytest.fail("no acknowledgement returned")


def test_an_unknown_time_in_force_is_rejected(client):
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        socket.send_json(
            {"action": "submit", "symbol": SYMBOL, "side": "buy",
             "quantity": 1, "price": "4600", "tif": "whenever"}
        )
        for _ in range(40):
            message = socket.receive_json()
            if "ack" in message:
                assert message["ack"]["ok"] is False
                return
    pytest.fail("no acknowledgement returned")


# --------------------------------------------------------------------------
# Order entry, as a stranger gets it wrong
# --------------------------------------------------------------------------


def _ack(socket, message, tries=60):
    """Send one action and return the acknowledgement, ignoring snapshots."""
    socket.send_json(message)
    for _ in range(tries):
        payload = socket.receive_json()
        if "ack" in payload:
            return payload["ack"]
    raise AssertionError("no acknowledgement returned")


# Every one of these is reachable from the ticket: the size box is a number
# input that reports "" when what was typed is not one, and the limit price
# box is free text because a price is not an integer.
MISTAKES = [
    ("a blank size", {"quantity": None, "price": "4600"}),
    ("a size that is not a number", {"quantity": "many", "price": "4600"}),
    ("a fractional size", {"quantity": 2.5, "price": "4600"}),
    ("a price with the ladder's comma in it", {"quantity": 5, "price": "9,233.75"}),
    ("a price with a currency symbol", {"quantity": 5, "price": "$4600"}),
    ("a price that is a word", {"quantity": 5, "price": "cheap"}),
    ("a stop that is not a number", {"quantity": 5, "price": "4600", "stop": "soon"}),
]

# What "a clear message" excludes. Each of these was actually shown in a toast.
INTERNALS = ("int()", "<class", "decimal.", "Traceback", "NoneType", "literal for")


@pytest.mark.parametrize("what,fields", MISTAKES, ids=[m[0] for m in MISTAKES])
def test_a_mistyped_order_is_answered_in_the_terms_of_the_box(client, what, fields):
    """The toast used to carry the interpreter's opinion of the failure.

    A blank size came back as "int() argument must be a string, a bytes-like
    object or a real number, not 'NoneType'". A price copied off the ladder
    (with the thousands separator the ladder itself drew) came back as "[<class
    'decimal.ConversionSyntax'>]". Both are Python talking to itself, on a
    screen someone is trying to trade from.
    """
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        ack = _ack(socket, {"action": "submit", "symbol": SYMBOL, "side": "buy", **fields})

    assert ack["ok"] is False, f"{what} was accepted"
    message = ack["error"]
    assert message and message[0] != "[", message
    for internal in INTERNALS:
        assert internal not in message, f"{what} answered with {message!r}"


def test_a_price_the_contract_cannot_settle_at_is_refused(client):
    """Nothing checked this, and the collateral model cannot.

    Collateral is sized from the settlement range, so a bid *below* the floor
    scores as less risky than one inside it and passes every check the venue
    makes. A limit buy at -100 on a contract bounded at zero was accepted,
    rested, and was eventually filled, crediting the account for having been
    paid to take delivery of something that cannot be worth less than nothing.
    """
    with client.websocket_connect("/ws") as socket:
        first = socket.receive_json()
        low, high = (Decimal(b) for b in first["books"][SYMBOL]["bounds"])

        below = _ack(socket, {"action": "submit", "symbol": SYMBOL, "side": "buy",
                              "quantity": 5, "price": str(low - 100)})
        above = _ack(socket, {"action": "submit", "symbol": SYMBOL, "side": "sell",
                              "quantity": 5, "price": str(high * 4)})
        inside = _ack(socket, {"action": "submit", "symbol": SYMBOL, "side": "buy",
                               "quantity": 1, "price": str((low + high) / 4)})

    assert below["ok"] is False and "settlement range" in below["error"]
    assert above["ok"] is False and "settlement range" in above["error"]
    assert inside["ok"] is True, "a price inside the range must still be accepted"


def test_a_misspelled_side_is_refused_rather_than_sold(client):
    """Anything that was not exactly "buy" became a SELL, silently.

    One character wrong in a client and the order went the other way, with
    nothing anywhere saying so. And it filled.
    """
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        ack = _ack(socket, {"action": "submit", "symbol": SYMBOL, "side": "byu",
                            "quantity": 5, "price": "4600"})
    assert ack["ok"] is False
    assert "buy or sell" in ack["error"]


def test_an_order_with_no_market_says_which_field_is_missing(client):
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        ack = _ack(socket, {"action": "submit", "side": "buy", "quantity": 5, "price": "1"})
    assert ack["ok"] is False
    assert ack["error"] == "choose a market before sending an order"


def test_the_ticket_can_reach_a_stop_and_an_iceberg(client):
    """Both were supported end to end and unreachable from the page.

    Checked here at the socket because that is where the ticket's fields
    arrive; that the ticket now actually sends them is checked under node.
    """
    with client.websocket_connect("/ws") as socket:
        first = socket.receive_json()
        mark = Decimal(first["books"][SYMBOL]["mark"])
        tick = Decimal(first["books"][SYMBOL]["tick"])
        trigger = (mark / tick).to_integral_value() * tick
        ack = _ack(socket, {"action": "submit", "symbol": SYMBOL, "side": "buy",
                            "quantity": 5, "price": None, "stop": str(trigger),
                            "display": 2})
    assert ack["ok"] is True, ack


def test_renaming_with_no_body_is_a_bad_request_not_a_traceback(client):
    response = client.post("/api/me", content=b"", headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json()["ok"] is False


def test_a_nonsense_action_is_reported_not_ignored(client):
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        socket.send_json({"action": "self_destruct"})
        for _ in range(40):
            message = socket.receive_json()
            if "ack" in message:
                assert message["ack"]["ok"] is False
                return
    pytest.fail("no acknowledgement returned")


# --------------------------------------------------------------------------
# The entry point
# --------------------------------------------------------------------------


def test_the_server_starts_the_way_it_is_documented():
    """`python -m dashboard.server` must work outside pytest.

    It did not. The project keeps `arena` under `python/`, and the only thing
    putting that on the path was `pythonpath` in the pytest configuration. So
    every test passed while the documented command, the sole way anyone opens
    the UI, died on ModuleNotFoundError.

    This runs in a subprocess with a clean environment precisely so pytest's
    own path setup cannot hide the problem a second time.
    """
    environment = dict(os.environ)
    # Anything that would smuggle the package path in gets removed first.
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-c", "import dashboard.server; print(dashboard.server.app.title)"],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=environment,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Artificial Esports Exchange" in result.stdout


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/static/js/main.js", "javascript"),
        ("/static/js/views.js", "javascript"),
        ("/static/js/format.js", "javascript"),
        ("/static/css/terminal.css", "text/css"),
    ],
)
def test_assets_are_served_with_a_usable_content_type(client, path, expected):
    """A 200 is not enough: the browser checks the type before it runs anything.

    Starlette serves static files with whatever `mimetypes` reports, and on
    Windows that reads the registry, where `.js` is commonly mapped to
    `text/plain`. Browsers enforce the MIME type of `<script type="module">`
    strictly and refuse a module served as anything else, so the entire front
    end silently did not run. The page painted its static HTML, no handler was
    bound, no button worked, and the server logged nothing, because every
    request had answered 200.

    Checking status alone is exactly how that survived a green test suite.
    """
    response = client.get(path)
    assert response.status_code == 200
    content_type = response.headers["content-type"]
    assert expected in content_type, f"{path} served as {content_type}"


def test_every_static_asset_the_page_asks_for_exists():
    """A 404 on a module leaves a blank screen and one line in the console."""
    html = (REPO / "dashboard" / "static" / "index.html").read_text(encoding="utf-8")
    referenced = re.findall(r'(?:href|src)="(/static/[^"]+)"', html)
    assert referenced, "the page references no local assets at all"
    for reference in referenced:
        asset = REPO / "dashboard" / reference.lstrip("/")
        assert asset.is_file(), f"{reference} is referenced but missing"

    # And the modules the entry point pulls in behind it.
    entry = (REPO / "dashboard" / "static" / "js" / "main.js").read_text(encoding="utf-8")
    for module in re.findall(r"from '\./([\w.]+\.js)'", entry):
        assert (REPO / "dashboard" / "static" / "js" / module).is_file(), module


# --------------------------------------------------------------------------
# The views, rendered under node
# --------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_every_view_renders_against_a_real_snapshot(client, tmp_path):
    """Renders all five screens and fails on `undefined`, `NaN`, unbalanced tags.

    Uses a genuine snapshot rather than a hand-written one, so a field the
    server stops sending shows up here rather than as a blank panel later.
    """
    snapshot = runner.market.snapshot()
    snapshot["generation"] = runner.generation
    snapshot["sessions"] = {
        s: runner.market.venue.session(s).value
        for s in runner.market.venue.registry.symbols
    }
    snapshot["speed"] = runner.market.speed

    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "snapshot": snapshot,
                "instruments": client.get("/api/instruments").json()["instruments"],
                "session": client.get("/api/session").json(),
                "agents": client.get("/api/agents").json()["agents"],
                "depth": client.get(f"/api/book/{SYMBOL}?levels=18").json(),
                "diagnostics": client.get(f"/api/diagnostics/{SYMBOL}").json(),
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["node", str(REPO / "tests" / "frontend" / "render.mjs"), str(fixture)],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_controller_loads_and_throttles_its_rendering(client, tmp_path):
    """main.js under a stub DOM, driving frames by hand.

    Two things are being caught. The first is that the controller loads at all:
    it owns the socket, the render loop and every binding, and had no coverage.
    The second is that a burst of snapshots does not turn into a burst of
    subtree rebuilds: replacing the panel destroys focus, scroll and text
    selection, so rebuilding on every message at 20Hz made the whole screen
    impossible to use with a keyboard.
    """
    snapshot = runner.market.snapshot()
    snapshot["generation"] = runner.generation
    snapshot["sessions"] = {
        s: runner.market.venue.session(s).value
        for s in runner.market.venue.registry.symbols
    }
    snapshot["speed"] = runner.market.speed

    fixture = tmp_path / "controller.json"
    fixture.write_text(json.dumps({"snapshot": snapshot}), encoding="utf-8")

    result = subprocess.run(
        ["node", str(REPO / "tests" / "frontend" / "controller.mjs"), str(fixture)],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _snapshot_fixture() -> dict:
    """A real snapshot, dressed the way the socket dresses one."""
    snapshot = runner.market.snapshot()
    snapshot["generation"] = runner.generation
    snapshot["sessions"] = {
        s: runner.market.venue.session(s).value
        for s in runner.market.venue.registry.symbols
    }
    snapshot["speed"] = runner.market.speed
    return snapshot


# The handful of DOM APIs main.js touches, same shape as tests/frontend, with
# one addition: nodes here remember their listeners, so a click can be fired.
_STUB_DOM = """
function makeNode(id = '') {
  const listeners = {};
  return {
    id, dataset: {}, style: {},
    classList: { toggle() {}, add() {}, remove() {}, contains: () => false },
    children: [], childNodes: [], scrollTop: 0, textContent: '',
    get innerHTML() { return ''; }, set innerHTML(_v) {},
    setAttribute() {}, removeAttribute() {}, getAttribute: () => null,
    addEventListener(kind, fn) { (listeners[kind] ||= []).push(fn); },
    fire(kind) { (listeners[kind] || []).forEach((fn) => fn({ stopPropagation() {} })); },
    removeEventListener() {},
    append() {}, replaceChildren() {}, replaceWith() {}, remove() {},
    focus() {}, closest: () => null,
    querySelector: () => null, querySelectorAll: () => [],
    contains: () => false, value: '', checked: false,
  };
}
const nodes = new Map();
for (const id of ['main', 'nav', 'watchlist', 'clock', 'events', 'equity', 'pnl',
                  'health', 'health-text', 'speed', 'speed-label', 'toaster',
                  't-send', 't-qty', 't-px', 't-stop', 't-show', 't-tif', 't-preview']) {
  nodes.set(id, makeNode(id));
}
globalThis.document = {
  getElementById: (id) => nodes.get(id) ?? makeNode(id),
  querySelector: () => null, querySelectorAll: () => [],
  createElement: () => makeNode(), addEventListener() {},
  activeElement: null, body: makeNode('body'),
};
globalThis.CSS = { escape: (s) => String(s) };
globalThis.history = { replaceState() {} };
globalThis.confirm = () => true;
globalThis.fetch = async () => ({ ok: true, json: async () => ({ instruments: [], agents: [] }) });
const sockets = [];
let sent = [];
globalThis.WebSocket = class {
  static OPEN = 1;
  constructor() { this.readyState = 1; sockets.push(this); }
  send(text) { sent.push(JSON.parse(text)); }
  close() {}
};
let clockMs = 0;
const pending = [];
globalThis.requestAnimationFrame = (fn) => { pending.push(fn); return pending.length; };
const runFrame = (ms) => { clockMs += ms; pending.splice(0).forEach((fn) => fn(clockMs)); };
globalThis.setInterval = () => 0;
globalThis.setTimeout = (fn) => { void fn; return 0; };
globalThis.clearTimeout = () => {};
"""


def _run_node(script: str, tmp_path: Path) -> subprocess.CompletedProcess:
    harness = tmp_path / "harness.mjs"
    harness.write_text(script, encoding="utf-8")
    return subprocess.run(
        ["node", str(harness)],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_ticket_reserves_what_the_venue_reserves(tmp_path):
    """"Reserved now" was the notional, not the reservation.

    They agree only for a long on a claim whose floor is zero, which most of
    the board happens to be, so it looked right until you sold, or touched the
    spread. A sell of ten futures stopped at 4,600 announced 46,000 held
    against the 54,000 the venue actually took; a spread stopped at zero
    announced *nothing* held against 100,000. Understating every time, which is
    the direction that leaves a trader believing in free cash they do not have
    and their next order refused without explanation.

    The expected figures come from ``Account.collateral_required`` itself, so
    this is the page checked against the venue rather than against a second
    opinion about the venue.
    """
    still = MarketRunner()
    venue = still.market.venue
    cases = []
    for symbol in venue.registry.symbols:
        instrument = venue.registry.require(symbol)
        low, high = (from_money(b) for b in venue.bounds_in_minor(instrument))
        for buying in (True, False):
            for fraction in (Decimal("0.1"), Decimal("0.5"), Decimal("0.9")):
                at = (low + (high - low) * fraction).quantize(Decimal("0.01"))
                quantity = 10
                held = Account.collateral_required(
                    quantity if buying else -quantity,
                    to_money(at),
                    venue.bounds_in_minor(instrument),
                )
                cases.append(
                    {
                        "symbol": symbol,
                        "buying": buying,
                        "quantity": quantity,
                        "at": str(at),
                        "bounds": [str(low), str(high)],
                        "expected": str(from_money(held)),
                    }
                )
    assert len(cases) > 50

    fixture = tmp_path / "collateral.json"
    fixture.write_text(json.dumps(cases), encoding="utf-8")
    format_js = (REPO / "dashboard" / "static" / "js" / "format.js").as_uri()

    script = f"""
import {{ readFileSync }} from 'node:fs';
import {{ worstCase }} from {json.dumps(format_js)};
const cases = JSON.parse(readFileSync({json.dumps(str(fixture))}, 'utf8'));
const problems = [];
for (const c of cases) {{
  const shown = worstCase(c.quantity, c.at, c.bounds, c.buying);
  if (shown == null) {{ problems.push(`${{c.symbol}}: nothing computed`); continue; }}
  if (Math.abs(shown - Number(c.expected)) > 1e-6)
    problems.push(`${{c.symbol}} ${{c.buying ? 'buy' : 'sell'}} ${{c.quantity}} at `
      + `${{c.at}}: the ticket shows ${{shown}}, the venue holds ${{c.expected}}`);
}}
if (problems.length) {{
  problems.slice(0, 8).forEach((p) => console.error('  - ' + p));
  console.error(`(${{problems.length}} of ${{cases.length}} disagree)`);
  process.exit(1);
}}
console.log(`${{cases.length}} reservations match the venue`);
"""
    result = _run_node(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_ticket_sends_the_stop_and_the_iceberg_it_collected(client, tmp_path):
    """The Advanced panel collected two fields and threw them away.

    A trader typed a stop trigger and an iceberg size, watched the preview
    describe a stop ("Waits until 8,900.00", "Reserved now ..."), pressed Place
    Order, and got a plain limit that rested in full, visible, immediately.
    Both fields were supported by the server and by the venue the whole time;
    the submitter simply never read them.

    Driven through the real controller rather than asserted about the source,
    because what matters is the message that leaves the browser.
    """
    fixture = tmp_path / "snapshot.json"
    fixture.write_text(json.dumps(_snapshot_fixture()), encoding="utf-8")
    main_js = (REPO / "dashboard" / "static" / "js" / "main.js").as_uri()

    script = f"""
import {{ readFileSync }} from 'node:fs';
const snapshot = JSON.parse(readFileSync({json.dumps(str(fixture))}, 'utf8'));
const symbol = {json.dumps(SYMBOL)};
globalThis.location = {{ protocol: 'http:', host: 'localhost:8000',
  href: 'http://localhost:8000/?view=trade&symbol=' + symbol,
  search: '?view=trade&symbol=' + symbol }};
{_STUB_DOM}
await import({json.dumps(main_js)});
const socket = sockets[0];
socket.onmessage({{ data: JSON.stringify(snapshot) }});
runFrame(600);                       // past the panel interval, so bind() runs

nodes.get('t-qty').value = '10';
nodes.get('t-px').value = '4600';
nodes.get('t-stop').value = '4500';
nodes.get('t-show').value = '2';
nodes.get('t-tif').value = 'gtc';

sent = [];
nodes.get('t-send').fire('click');
const order = sent.find((m) => m.action === 'submit');
const problems = [];
if (!order) problems.push('the send button sent nothing at all');
else {{
  if (order.stop == null || String(order.stop) !== '4500')
    problems.push(`stop trigger 4500 was dropped: ${{JSON.stringify(order)}}`);
  if (order.display == null || String(order.display) !== '2')
    problems.push(`iceberg size 2 was dropped: ${{JSON.stringify(order)}}`);
  if (String(order.quantity) !== '10') problems.push('size did not survive');
  if (String(order.price) !== '4600') problems.push('limit price did not survive');
}}
if (problems.length) {{ problems.forEach((p) => console.error('  - ' + p)); process.exit(1); }}
console.log('the ticket sent ' + JSON.stringify(order));
"""
    result = _run_node(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_send_button_is_live_wherever_the_venue_takes_orders(client, tmp_path):
    """The first thing a stranger meets is a phase they could not trade in.

    The exchange opens with a call auction, so every contract is in `pre_open`
    on the first page load. Orders in a call phase are accepted, do rest, and
    are what sets the opening price, and the button was disabled, under a label
    reading "pre open, orders will rest". `closed` is the only phase that
    refuses orders, and there the same label was a lie the other way.
    """
    fixture = tmp_path / "snapshot.json"
    fixture.write_text(json.dumps(_snapshot_fixture()), encoding="utf-8")
    views_js = (REPO / "dashboard" / "static" / "js" / "views.js").as_uri()

    script = f"""
import {{ readFileSync }} from 'node:fs';
import {{ trade }} from {json.dumps(views_js)};
const snapshot = JSON.parse(readFileSync({json.dumps(str(fixture))}, 'utf8'));
const symbol = {json.dumps(SYMBOL)};

// Whether the venue takes orders in a phase is SessionState.accepts_orders:
// everything but `closed`.
const expected = {{ pre_open: true, auction: true, continuous: true, closed: false }};
const problems = [];
for (const [session, takesOrders] of Object.entries(expected)) {{
  const store = {{
    snapshot: {{ ...snapshot, sessions: {{ ...snapshot.sessions, [symbol]: session }} }},
    instruments: [], symbol, history: {{}}, depth: null, side: 'buy', reveal: false,
  }};
  const html = trade(store);
  const button = html.match(/<button[^>]*id="t-send"[^>]*>([\\s\\S]*?)<\\/button>/);
  if (!button) {{ problems.push(`${{session}}: no send button rendered`); continue; }}
  const disabled = /\\bdisabled\\b/.test(button[0]);
  if (disabled === takesOrders)
    problems.push(`${{session}}: button disabled=${{disabled}} but the venue `
      + `${{takesOrders ? 'accepts' : 'refuses'}} orders in it`);
  const label = button[1].trim();
  if (takesOrders && !/Place Order/.test(label))
    problems.push(`${{session}}: label ${{JSON.stringify(label)}} does not offer to place one`);
  if (!takesOrders && /will rest/.test(label))
    problems.push(`${{session}}: label ${{JSON.stringify(label)}} promises a rest that cannot happen`);
}}
if (problems.length) {{ problems.forEach((p) => console.error('  - ' + p)); process.exit(1); }}
console.log('the send button matches every phase');
"""
    result = _run_node(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_revealed_value_is_a_price_not_a_tick_count(client):
    """The one number whose job is to be compared against the market.

    It was reported in ticks while the mark beside it was in contract units,
    so a future marking at 4,663 revealed a settlement of 18,677 and the chart
    drew a target line four times off the top of its own series. Nothing looked
    broken; it just looked like a number.
    """
    listed = client.get("/api/instruments").json()["instruments"]
    snapshot = runner.market.snapshot()

    checked = 0
    for entry in listed:
        settles = entry.get("settles_at")
        if settles in (None, 0):
            continue
        low, high = (float(b) for b in snapshot["books"][entry["symbol"]]["bounds"])
        assert low <= settles <= high, (
            f"{entry['symbol']} reveals {settles}, outside the range {low}..{high} "
            "the same page prints, which is what a tick count looks like when "
            "it is read as a price"
        )
        checked += 1
    assert checked > 5, "too few contracts had a value to check"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_percentage_gives_way_to_points_when_its_base_has_no_resolution(tmp_path):
    """A ratio divided by almost nothing is correct and useless.

    An option can open a session worth one tick. One on this board did, reached
    96.375, and the market rail rendered the move as **+308,825.00%**, which is
    arithmetically right and says only that the contract went from the smallest
    price it can represent to a real one. The move itself says that better.
    Above 999% the figure is shown in points instead, which is a statement
    about the resolution of a ratio rather than about the market: nothing is
    clamped, hidden or invented.
    """
    probe = tmp_path / "probe.mjs"
    # An absolute file URL: the probe is written into pytest's tmp dir, so a
    # path relative to the repo resolves against the wrong root.
    module = (REPO / "dashboard" / "static" / "js" / "format.js").as_uri()
    probe.write_text(
        f"import {{ move }} from '{module}';\n"
        "const cases = [\n"
        "  [96.125, 0.25, 'pts'],\n"   # the measured case
        "  [50, 0, 'pts'],\n"          # no base at all
        "  [4.7, 470, '%'],\n"         # an ordinary move
        "  [-9.4, 470, '%'],\n"        # ordinary, negative
        "  [469, 47, '%'],\n"          # 998%, still inside the column
        "];\n"
        "for (const [change, base, want] of cases) {\n"
        "  const got = move(change, base).text;\n"
        "  const ok = want === 'pts' ? got.endsWith('pts') : got.endsWith('%');\n"
        "  if (!ok) { console.error(`move(${change}, ${base}) = ${got}, wanted ${want}`);"
        " process.exit(1); }\n"
        "}\n"
        "if (move(0.5, 100).text !== '+0.50%') { console.error('sign lost'); process.exit(1); }\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["node", str(probe)], capture_output=True, text=True, cwd=REPO, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_client_compensates_for_minor_units():
    """If the browser has to divide by a million, a serialiser is wrong.

    This is the third instance of one defect: a number crossing the wire in
    units its label does not claim, with one consumer quietly correcting it and
    every other consumer wrong. The first rendered a settlement value of 18,677
    against a real 4,663. The second rendered a seat's equity as "143745.00M"
    beside a header reading "143.7k". The third was `fees_collected`, published
    in raw minor units while every money field beside it went through
    `from_money`, with `views.js` compensating at the point of display.

    It stopped being a display bug the moment this exchange grew an API. A
    client calling `GET /v1/exchange` has no `/ 1e6` to apply and no reason to
    suspect it needs one, so the compensation has to live in the serialiser or
    nowhere. The old compensation was also `Number(...)`, which puts a float in
    a money path in a project whose entire claim is exact arithmetic.

    The pattern requires a *property access* in the numerator, and that is not
    incidental. Dividing by a million is perfectly legitimate as a magnitude
    abbreviation: `money()` renders 1,500,000 as "1.5M" and `count()` does the
    same for trade counts. The first version of this test flagged both and was
    wrong. What distinguishes the defect is that it divides a field which
    arrived from the server, and that always reads as `something.field`, where
    a formatter divides a local like `abs` or `v`.
    """
    root = REPO / "dashboard" / "static" / "js"
    # `x.field ... / 1e6`, tolerating a cast or close paren in between.
    compensation = re.compile(
        r"[\w$]+\.[\w$]+[^\n;]{0,40}?/\s*(?:1e6|1_?000_?000)\b", re.IGNORECASE
    )
    offenders: dict[str, list[str]] = {}
    for path in sorted(root.glob("*.js")):
        text = path.read_text(encoding="utf-8")
        hits = [m.group(0).strip() for m in compensation.finditer(text)]
        if hits:
            offenders[path.name] = hits
    assert not offenders, (
        f"the browser is correcting the server's units: {offenders}. "
        "Publish the field through `from_money` instead."
    )


def test_the_fee_ledger_is_published_in_contract_units():
    """The specific case, pinned, since a general rule catches only the shape."""
    from decimal import Decimal

    from arena.portfolio.money import from_money

    published = runner.session_state()["fees_collected"]
    expected = from_money(runner.market.venue.fees_collected)
    assert Decimal(published) == expected, (
        f"published {published}, venue holds {expected}"
    )


# --------------------------------------------------------------------------
# The controls that reach past the caller
# --------------------------------------------------------------------------


OPERATOR_ROUTES = [
    ("/api/config", {}),
    (f"/api/session/{SYMBOL}/halt", None),
    (f"/api/session/{SYMBOL}/uncross", None),
    ("/api/participant/mm-1/kill", None),
    ("/api/participant/mm-1/revive", None),
]


@pytest.mark.parametrize("path,body", OPERATOR_ROUTES)
def test_an_operator_route_refuses_a_visitor(client, path, body):
    """Five routes change the market for everybody, and had no guard at all.

    `POST /api/config` calls `MarketRunner.reconfigure`, whose own docstring
    says "The old one is discarded, not paused": every account, position,
    working order and price series, for every connected user, gone. Any visitor
    could send it. `kill` was quieter and no better: it takes an **arbitrary**
    agent id, so one visitor could reach across and disable another human's
    seat, pulling their working orders while they watched.

    That is survivable while this is one person's demo and fatal the moment two
    strangers trade against each other, which is what the exchange is for.

    404 rather than 401 is deliberate: a 401 confirms the route exists and
    invites guessing at the token, and a stranger who cannot operate this venue
    has no business learning its control surface.
    """
    from dashboard.operator_auth import OPERATOR_HEADER

    unguarded = client.post(path) if body is None else client.post(path, json=body)
    assert unguarded.status_code == 404, f"{path} answered a visitor"

    wrong = (
        client.post(path, headers={OPERATOR_HEADER: "not-the-token"})
        if body is None
        else client.post(path, json=body, headers={OPERATOR_HEADER: "not-the-token"})
    )
    assert wrong.status_code == 404, f"{path} accepted a wrong token"


def test_an_operator_route_answers_the_operator(client):
    """The gate has to let the right person through, or it is just a wall."""
    allowed = client.post(f"/api/session/{SYMBOL}/halt", headers=as_operator())
    assert allowed.status_code == 200
    client.post(f"/api/session/{SYMBOL}/uncross", headers=as_operator())


def test_the_operator_token_is_never_a_default():
    """A default token is the shape of every embarrassing breach.

    The deployment that forgot to override it looks exactly like the one that
    did, so there is no moment at which anybody notices. When the environment
    sets nothing, a fresh random token is minted and printed once at startup
    instead.
    """
    from dashboard import operator_auth

    token = operator_auth.operator_token()
    assert token, "there is no token at all"
    assert len(token) >= 16, f"token is only {len(token)} characters"
    assert token.lower() not in {"admin", "operator", "changeme", "token", "secret"}


def test_a_key_survives_a_rebuild_and_keeps_its_seat(client):
    """Invalidating credentials on reset breaks every algorithm running.

    The industry splits on this and one side is plainly right. Binance
    preserves API keys across its periodic testnet wipes and says so: "Starting
    from August 2020, API Keys are preserved during resets. Users no longer
    need to re-register new API Keys after a reset." tastytrade wipes state
    nightly and makes the same carve-out: "Users, customers, and accounts are
    untouched." Alpaca went the other way, replacing reset with
    create-and-delete, and warns "Don't forget to generate new API keys for any
    newly created account", which means every reset breaks every running bot.

    A rebuild here discards the whole market, so the risk is not just the key
    record surviving. It is that `LiveMarket.trader` answers an unknown id with
    the SHARED account, so a credential that captured an account id would come
    back pointing at a communal seat rather than failing loudly. This asserts
    the key still works AND still reaches its own seat.
    """
    from arena.api.keys import HEADER_KEY, HEADER_SIGNATURE, HEADER_TIMESTAMP, sign

    client.get("/")
    issued = client.post("/v1/keys", json={"label": "survives"})
    assert issued.status_code == 201, issued.text
    key = issued.json()

    def as_key(path: str):
        stamp = str(time.time())
        return client.get(
            path,
            headers={
                HEADER_KEY: key["key_id"],
                HEADER_TIMESTAMP: stamp,
                HEADER_SIGNATURE: sign(key["secret"], "GET", path, stamp, b""),
            },
        )

    before = as_key("/v1/account")
    assert before.status_code == 200
    seat_before = before.json()["account_id"]

    client.post("/api/config", json={"seed": 11}, headers=as_operator())
    time.sleep(2)

    after = as_key("/v1/account")
    assert after.status_code == 200, "the rebuild invalidated a live credential"
    assert after.json()["account_id"] == seat_before, (
        f"the key moved from {seat_before} to {after.json()['account_id']}, "
        "which is what landing on the shared account looks like"
    )


def test_every_response_says_it_is_simulated(client):
    """A misconfigured base URL is otherwise undetectable from inside a client.

    Every sandbox surveyed separates itself from production by hostname alone.
    Kraken stated the intent outright: "the only difference... is that the base
    URL is not futures.kraken.com but instead demo-futures.kraken.com." OANDA
    ships an Account object carrying no environment field, so a captured
    payload is indistinguishable between practice and live. Interactive Brokers
    reduces the safeguard to a plea in its own docs: "make sure your client
    application is connecting to the right TWS!"

    Deribit is the one venue that solves it, stamping a `testnet` boolean on
    every response envelope. This is that. Asserted on an error response too,
    because that is exactly when a confused client most needs to know which
    venue answered it.
    """
    for path in ("/v1/exchange", "/v1/instruments", "/api/session"):
        assert client.get(path).headers.get("arena-simulated") == "true", path

    refused = client.get("/v1/account")
    assert refused.status_code == 401
    assert refused.headers.get("arena-simulated") == "true", (
        "an error response does not say which venue produced it"
    )
