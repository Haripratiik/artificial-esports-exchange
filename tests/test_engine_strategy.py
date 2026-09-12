"""The eliminations strategy: what it trades, what it refuses, and what it sends.

The claim under test is narrow and it is the whole claim: `ELIM_k_GT{t+1}`
implies `ELIM_k_GT{t}`, so a book bidding more for the higher rung than it asks
for the lower one is offering a pair that cannot lose. Everything here is either
that arithmetic or one of the four ways acting on it goes wrong: a credit that
does not cover the fee, a size larger than the touch, a book that is not
matching, and two orders that are not one trade.

The prices in the inverted ladder are the ones the connector's README recorded
from a live board, so the pair this file asserts on is the pair that was
actually there rather than one chosen to make the arithmetic tidy.
"""

from __future__ import annotations

import re
from collections import defaultdict
from decimal import Decimal
from typing import Any

import pytest

from arena.market.fees import MAKER_TAKER
from arena.portfolio.money import MONEY_SCALE

from dashboard.state import MarketConfig

from connectors.prediction_engine.auth import ArenaSigner
from connectors.prediction_engine.client import ArenaClient, ArenaError, Listing
from connectors.prediction_engine.models import Market
from connectors.prediction_engine.strategy import (
    TAKER_BPS,
    Trade,
    execute,
    fee_per_contract,
    inversions,
    reconcile,
    scan,
    taker_bps_from,
)

from tests.test_api import Exchange

TAG = "SOLO0"
ELIM = re.compile(r"^(?P<tag>[A-Z]+\d+)_ELIM_(?P<who>.+)_GT(?P<threshold>\d+)$")


def _market(ticker: str, bid: int | None, ask: int | None, *,
            bid_size: int = 10, ask_size: int = 10,
            status: str = "continuous") -> Market:
    """One contract quoted in the engine's cents, with the size on each side."""
    return Market(
        ticker=ticker, event_ticker=TAG, series_ticker="ELIM", status=status,
        yes_bid=bid, yes_ask=ask,
        yes_bid_size=float(bid_size), yes_ask_size=float(ask_size),
    )


def _listing(symbol: str, bid: str | None, ask: str | None, *,
             bid_size: int = 10, ask_size: int = 10,
             session: str = "continuous") -> Listing:
    """The same book as the instruments endpoint publishes it, in venue units."""
    return Listing(
        symbol=symbol, instrument_class="event",
        bounds=(Decimal(0), Decimal(1)), session=session,
        mark=Decimal("0.5"), trades=0, expiry="", subjects=(),
        bid=None if bid is None else Decimal(bid),
        ask=None if ask is None else Decimal(ask),
        bid_size=bid_size, ask_size=ask_size,
    )


# CINDER's ladder on OBJECTIVE3, as the README recorded it: GT1 could be bought
# at 0.49 while GT2 bid 0.60, and X > 2 implies X > 1.
INVERTED = [
    _market(f"{TAG}_ELIM_CINDER_GT0", 53, 62),
    _market(f"{TAG}_ELIM_CINDER_GT1", 40, 49),
    _market(f"{TAG}_ELIM_CINDER_GT2", 60, 70),
]
INVERTED_BOOKS = [
    _listing(f"{TAG}_ELIM_CINDER_GT0", "0.53", "0.62"),
    _listing(f"{TAG}_ELIM_CINDER_GT1", "0.40", "0.49"),
    _listing(f"{TAG}_ELIM_CINDER_GT2", "0.60", "0.70"),
]


class Recorder:
    """A client that records orders instead of sending them.

    Only the three methods the strategy calls, because a double that implements
    more than that starts making claims about the venue that the venue has not
    agreed to. The end-to-end tests at the bottom of this file run the same code
    against the real application, which is where those claims belong.
    """

    def __init__(self, books: list[Listing], *, refuse: tuple[str, ...] = (),
                 feed: dict[str, Any] | None = None) -> None:
        self._books = list(books)
        self._refuse = set(refuse)
        self.feed = feed or {"fills": [], "rejections": []}
        self.sent: list[dict[str, Any]] = []

    def listings(self) -> list[Listing]:
        return list(self._books)

    def create_order(self, ticker: str, side: str, quantity: int,
                     price: Decimal | str | None = None, *,
                     time_in_force: str = "gtc",
                     client_order_id: str | None = None) -> dict[str, Any]:
        # Recorded before the refusal, so a test can tell an order that was
        # refused from one that was never sent.
        self.sent.append({
            "symbol": ticker, "side": side, "quantity": quantity,
            "price": None if price is None else str(price),
            "time_in_force": time_in_force, "client_order_id": client_order_id,
        })
        if ticker in self._refuse:
            raise ArenaError(422, "rejected_by_venue", "not here", "/v1/orders")
        return {"status": "accepted", "symbol": ticker, "side": side,
                "quantity": quantity, "client_order_id": client_order_id}

    def fills(self, limit: int = 200) -> dict[str, Any]:
        return self.feed


# -- what the pair is worth --------------------------------------------------


def test_the_break_even_is_the_venues_own_fee_arithmetic():
    """Not a guessed floor. The rate is this venue's and so is the rounding.

    `FeeSchedule.charge` takes basis points of notional in minor units and
    rounds toward the venue, and notional is price times quantity, so one
    contract of the pair pays the rate against each leg's own price. Held
    against `arena.market.fees` rather than against a reading of it, because a
    strategy whose break-even is a different number from the venue's is a
    strategy that trades pairs which lose money by a rounding.
    """
    charged = (
        int(MAKER_TAKER.charge(int(Decimal("0.49") * MONEY_SCALE), aggressor=True))
        + int(MAKER_TAKER.charge(int(Decimal("0.60") * MONEY_SCALE), aggressor=True))
    )
    assert MAKER_TAKER.taker_bps == TAKER_BPS
    assert fee_per_contract(0.49, 0.60) == pytest.approx(charged / MONEY_SCALE, abs=1e-9)
    # 0.000218 against a tick of 0.01: the smallest inversion that can exist on
    # this grid clears the fee forty five times over, which is why `min_credit`
    # defaults to zero rather than to a made-up margin.
    assert fee_per_contract(0.49, 0.60) * 45 < 0.01


# -- what it trades ----------------------------------------------------------


def test_a_ladder_that_falls_with_its_threshold_offers_nothing():
    """The shape a coherent book has. Nothing to collect, so nothing is returned."""
    monotone = [
        _market(f"{TAG}_ELIM_TALON_GT0", 60, 70),
        _market(f"{TAG}_ELIM_TALON_GT1", 40, 49),
        _market(f"{TAG}_ELIM_TALON_GT2", 10, 15),
    ]
    assert inversions(monotone, TAG) == []


def test_an_inverted_ladder_yields_the_pair_the_audit_found():
    """Buy the lower rung at its ask, sell the higher rung into its bid.

    The direction is the whole trade and getting it backwards is a position
    rather than a hedge, so it is asserted rather than implied: GT1 is bought
    because X > 2 implies X > 1, which makes the GT1 leg the one that can pay
    when the GT2 leg does not.
    """
    trades = inversions(INVERTED, TAG)
    assert len(trades) == 1
    only = trades[0]
    assert only.buy == f"{TAG}_ELIM_CINDER_GT1"
    assert only.sell == f"{TAG}_ELIM_CINDER_GT2"
    assert only.credit == pytest.approx(0.11, abs=1e-12)
    assert "X>2 implies X>1" in only.reason
    # The two rungs that are not inverted against each other are not traded:
    # GT0 is offered at 0.62 and nothing on the ladder bids more than that.
    assert f"{TAG}_ELIM_CINDER_GT0" not in (only.buy, only.sell)


def test_a_pair_that_collects_nothing_is_not_an_edge():
    """Ask 0.50 against bid 0.50 is a pair of trades, not an arbitrage."""
    flat = [
        _market(f"{TAG}_ELIM_MIRE_GT1", 40, 50),
        _market(f"{TAG}_ELIM_MIRE_GT2", 50, 60),
    ]
    assert inversions(flat, TAG) == []


def test_a_credit_under_the_floor_is_left_alone():
    """Two floors, and they are different claims.

    The fee is measured: at 200 basis points the taker takes 0.0202 out of a
    pair quoted around a half, which is more than the 0.01 on offer, so the
    trade is a loss and is not sent. `min_credit` is the margin a caller asks
    for on top of that, and it is not a fee and does not pretend to be one.
    """
    thin = [
        _market(f"{TAG}_ELIM_MIRE_GT1", 40, 50),
        _market(f"{TAG}_ELIM_MIRE_GT2", 51, 60),
    ]
    assert len(inversions(thin, TAG)) == 1
    assert inversions(thin, TAG)[0].credit == pytest.approx(0.01, abs=1e-12)

    assert inversions(thin, TAG, min_credit=0.02) == []
    assert inversions(thin, TAG, taker_bps=200.0) == []
    # And at a fee it does clear, it is traded again, so the refusal above is
    # the fee rather than the threshold being unreachable.
    assert len(inversions(thin, TAG, taker_bps=50.0)) == 1


def test_size_is_the_minimum_of_the_two_touches():
    """Lifting more than is there turns a riskless pair into an open position."""
    shallow_buy = [
        _market(f"{TAG}_ELIM_QUILL_GT1", 40, 49, ask_size=3),
        _market(f"{TAG}_ELIM_QUILL_GT2", 60, 70, bid_size=7),
    ]
    assert inversions(shallow_buy, TAG)[0].size == 3

    shallow_sell = [
        _market(f"{TAG}_ELIM_QUILL_GT1", 40, 49, ask_size=9),
        _market(f"{TAG}_ELIM_QUILL_GT2", 60, 70, bid_size=4),
    ]
    assert inversions(shallow_sell, TAG)[0].size == 4


def test_a_rung_inverted_against_two_others_is_not_sold_twice():
    """The budget that makes overlapping pairs add up to what is on the screen.

    GT0 is offered at 0.20 against bids of 0.30 and 0.40, so two pairs contain
    it and only five lots are on its ask. Richest first takes GT0 against GT2
    for all five, which leaves the GT0 against GT1 pair with nothing, and the
    remaining GT2 bid is still worth taking against GT1. Without the budget the
    ten lots bought on GT0 would be five lots of hedge and five lots of view.
    """
    overlapping = [
        _market(f"{TAG}_ELIM_WISP_GT0", 10, 20, ask_size=5, bid_size=5),
        _market(f"{TAG}_ELIM_WISP_GT1", 30, 35, ask_size=6, bid_size=4),
        _market(f"{TAG}_ELIM_WISP_GT2", 40, 45, ask_size=6, bid_size=9),
    ]
    trades = inversions(overlapping, TAG)
    assert [(t.buy[-3:], t.sell[-3:], t.size) for t in trades] == [
        ("GT0", "GT2", 5),
        ("GT1", "GT2", 4),
    ]
    bought: dict[str, int] = defaultdict(int)
    sold: dict[str, int] = defaultdict(int)
    for trade in trades:
        bought[trade.buy] += trade.size
        sold[trade.sell] += trade.size
    for market in overlapping:
        assert bought[market.ticker] <= int(market.yes_ask_size)
        assert sold[market.ticker] <= int(market.yes_bid_size)


def test_rungs_are_paired_across_the_ladder_rather_than_only_with_neighbours():
    """GT0 against GT2 is as real an inversion as GT0 against GT1.

    A wide rung in the middle is what makes this more than tidiness. GT1 here
    is quoted 0.30 against 0.50, so neither of its neighbours is inverted with
    it and a check that only compared adjacent rungs would report the ladder
    clean. The pair that exists skips over it: GT0 is offered at 0.35 and GT2
    bids 0.40.
    """
    wide_middle = [
        _market(f"{TAG}_ELIM_RIFT_GT0", 30, 35),
        _market(f"{TAG}_ELIM_RIFT_GT1", 30, 50),
        _market(f"{TAG}_ELIM_RIFT_GT2", 40, 45),
    ]
    trades = inversions(wide_middle, TAG)
    assert [(trade.buy[-3:], trade.sell[-3:]) for trade in trades] == [("GT0", "GT2")]
    assert trades[0].credit == pytest.approx(0.05, abs=1e-12)


def test_two_competitors_are_two_ladders():
    """P(TALON > 1) says nothing about P(MIRE > 2), so they are never paired."""
    mixed = [
        _market(f"{TAG}_ELIM_TALON_GT1", 10, 20),
        _market(f"{TAG}_ELIM_MIRE_GT2", 60, 70),
    ]
    assert inversions(mixed, TAG) == []


def test_another_matchs_ladder_is_not_this_matchs_ladder():
    """The tag is checked, because two matches share competitor names."""
    elsewhere = [
        _market("OBJECTIVE1_ELIM_CINDER_GT1", 40, 49),
        _market(f"{TAG}_ELIM_CINDER_GT2", 60, 70),
    ]
    assert inversions(elsewhere, TAG) == []
    assert inversions(elsewhere, "OBJECTIVE1") == []


# -- the books it refuses to touch -------------------------------------------


@pytest.mark.parametrize("session", ["pre_open", "auction", "closed"])
def test_a_book_that_is_not_matching_is_never_traded(session):
    """This is the filter that separates a finding from an artefact.

    `audit.py` records what leaving it out costs when only measuring: a board
    ten seconds after listing is quoted at the midpoint of its range on purpose
    and read as a 3.5 unit arbitrage. Acting on one is worse than measuring it,
    because an immediate-or-cancel into an opening call does not trade and a
    settled book cannot be traded at all.
    """
    one_side_shut = [
        _market(f"{TAG}_ELIM_CINDER_GT1", 40, 49, status=session),
        _market(f"{TAG}_ELIM_CINDER_GT2", 60, 70),
    ]
    assert inversions(one_side_shut, TAG) == []

    both_shut = [
        _market(f"{TAG}_ELIM_CINDER_GT1", 40, 49, status=session),
        _market(f"{TAG}_ELIM_CINDER_GT2", 60, 70, status=session),
    ]
    assert inversions(both_shut, TAG) == []


def test_a_book_quoted_one_side_only_is_left_out_rather_than_completed():
    """Inventing the missing side would manufacture the credit being looked for."""
    half = [
        _market(f"{TAG}_ELIM_CINDER_GT1", 40, None),
        _market(f"{TAG}_ELIM_CINDER_GT2", 60, 70),
    ]
    assert inversions(half, TAG) == []


def test_a_touch_with_no_size_on_it_is_not_a_touch():
    """A price nobody is showing any size at cannot be traded against."""
    empty = [
        _market(f"{TAG}_ELIM_CINDER_GT1", 40, 49, ask_size=0),
        _market(f"{TAG}_ELIM_CINDER_GT2", 60, 70),
    ]
    assert inversions(empty, TAG) == []


# -- sending it --------------------------------------------------------------


def test_a_dry_run_sends_nothing():
    """And still says what it would have done, which is the point of a dry run."""
    quiet = Recorder(INVERTED_BOOKS)
    done = execute(quiet, inversions(INVERTED, TAG), dry_run=True)

    assert quiet.sent == []
    assert len(done) == 1
    assert done[0]["status"] == "dry_run"
    assert done[0]["size"] == 10
    assert done[0]["credit"] == pytest.approx(0.11, abs=1e-9)
    assert "would sell 10" in done[0]["detail"]


def test_both_legs_go_out_as_immediate_or_cancel_at_the_touch():
    """Neither leg rests and neither chases.

    A resting order on the cheap leg is a free option for the rest of the
    market, and an order that walks the book is no longer trading the credit
    that was measured. The price is the venue's own decimal off the listing
    rather than a float rebuilt from cents, so the order names the tick.
    """
    recorder = Recorder(INVERTED_BOOKS)
    done = execute(recorder, inversions(INVERTED, TAG))

    assert done[0]["status"] == "sent"
    assert len(recorder.sent) == 2
    by_side = {row["side"]: row for row in recorder.sent}
    assert by_side["buy"]["symbol"] == f"{TAG}_ELIM_CINDER_GT1"
    assert by_side["buy"]["price"] == "0.49"
    assert by_side["sell"]["symbol"] == f"{TAG}_ELIM_CINDER_GT2"
    assert by_side["sell"]["price"] == "0.60"
    for row in recorder.sent:
        assert row["time_in_force"] == "ioc"
        assert row["quantity"] == 10
        assert row["client_order_id"]


def test_the_leg_that_goes_first_is_the_one_with_the_smaller_naked_exposure():
    """If only one leg lands, it should be the one this seat minds least.

    Short the higher rung at 0.60 risks 0.40; long the lower rung at 0.49 risks
    0.49, so the short goes first. Move the pair down to 0.05 against 0.20 and
    the comparison reverses: the short now risks 0.80 and the long risks 0.05.
    """
    rich = Recorder(INVERTED_BOOKS)
    execute(rich, inversions(INVERTED, TAG))
    assert rich.sent[0]["side"] == "sell"

    cheap_markets = [
        _market(f"{TAG}_ELIM_CINDER_GT1", 2, 5),
        _market(f"{TAG}_ELIM_CINDER_GT2", 20, 24),
    ]
    cheap_books = [
        _listing(f"{TAG}_ELIM_CINDER_GT1", "0.02", "0.05"),
        _listing(f"{TAG}_ELIM_CINDER_GT2", "0.20", "0.24"),
    ]
    cheap = Recorder(cheap_books)
    execute(cheap, inversions(cheap_markets, TAG))
    assert cheap.sent[0]["side"] == "buy"


def test_a_refusal_on_the_first_leg_stops_the_second():
    """Nothing traded, so nothing is at risk and the pair is simply not done."""
    recorder = Recorder(INVERTED_BOOKS, refuse=(f"{TAG}_ELIM_CINDER_GT2",))
    done = execute(recorder, inversions(INVERTED, TAG))

    assert done[0]["status"] == "refused"
    assert [row["symbol"] for row in recorder.sent] == [f"{TAG}_ELIM_CINDER_GT2"]
    assert "was not sent" in done[0]["detail"]


def test_a_refusal_on_the_second_leg_is_reported_as_half_legged():
    """The failure this strategy cannot design away, named rather than hidden.

    Two books and two orders are not one trade. When the short goes and the
    long does not, this seat is short the higher rung with nothing against it,
    and the result says so and says which symbol and how much.
    """
    recorder = Recorder(INVERTED_BOOKS, refuse=(f"{TAG}_ELIM_CINDER_GT1",))
    done = execute(recorder, inversions(INVERTED, TAG))

    assert done[0]["status"] == "half_legged"
    assert len(recorder.sent) == 2
    assert f"{TAG}_ELIM_CINDER_GT2" in done[0]["detail"]
    assert "with nothing" in done[0]["detail"]


def test_a_book_that_left_continuous_between_the_scan_and_the_send_is_dropped():
    """The trade is re-read before it is sent, and this is why."""
    shut = [
        _listing(f"{TAG}_ELIM_CINDER_GT1", "0.40", "0.49", session="auction"),
        _listing(f"{TAG}_ELIM_CINDER_GT2", "0.60", "0.70"),
    ]
    recorder = Recorder(shut)
    done = execute(recorder, inversions(INVERTED, TAG))

    assert recorder.sent == []
    assert done[0]["status"] == "stale"
    assert "auction" in done[0]["detail"]


def test_an_inversion_that_has_closed_since_the_scan_is_not_lifted_anyway():
    """A stale Trade is an instruction, not a permission."""
    moved = [
        _listing(f"{TAG}_ELIM_CINDER_GT1", "0.40", "0.49"),
        _listing(f"{TAG}_ELIM_CINDER_GT2", "0.45", "0.55"),
    ]
    recorder = Recorder(moved)
    done = execute(recorder, inversions(INVERTED, TAG))

    assert recorder.sent == []
    assert done[0]["status"] == "stale"
    assert "the inversion has gone" in done[0]["detail"]


def test_the_size_is_cut_to_what_is_left_on_the_touch_at_send_time():
    """The scan saw ten a side; by the time the order goes there are two."""
    thinner = [
        _listing(f"{TAG}_ELIM_CINDER_GT1", "0.40", "0.49", ask_size=2),
        _listing(f"{TAG}_ELIM_CINDER_GT2", "0.60", "0.70", bid_size=6),
    ]
    recorder = Recorder(thinner)
    done = execute(recorder, inversions(INVERTED, TAG))

    assert done[0]["size"] == 2
    assert {row["quantity"] for row in recorder.sent} == {2}


def test_a_leg_that_is_no_longer_listed_is_not_traded_around():
    """A `Trade` can outlive the board it was read from.

    Built by hand rather than by `inversions`, because that is the shape a
    caller holds who cached a scan and came back to it after the match settled.
    Half a pair is not a cheaper pair.
    """
    recorder = Recorder(INVERTED_BOOKS)
    settled = Trade(
        buy="OBJECTIVE9_ELIM_NOBODY_GT1", sell=f"{TAG}_ELIM_CINDER_GT2",
        size=4, credit=0.11, reason="read off a board that has since settled",
    )
    done = execute(recorder, [settled])

    assert recorder.sent == []
    assert done[0]["status"] == "stale"
    assert "no longer listed" in done[0]["detail"]


def test_a_contract_that_is_not_a_probability_is_not_half_of_a_credit():
    """A difference of two prices only means something when both are chances.

    This venue lists nine classes over five settlement ranges and 0.47 is a
    perfectly good price on a future bounded [0, 10,000]. `models.to_cents`
    refuses that reading and so does this, rather than reporting a credit of
    0.11 on two numbers that are not comparable.
    """
    wrong_class = _listing(f"{TAG}_ELIM_CINDER_GT1", "0.40", "0.49")
    wrong_class = Listing(
        symbol=wrong_class.symbol, instrument_class="future",
        bounds=(Decimal(0), Decimal("10000")), session="continuous",
        mark=Decimal("0.5"), trades=0, expiry="", subjects=(),
        bid=wrong_class.bid, ask=wrong_class.ask,
        bid_size=10, ask_size=10,
    )
    recorder = Recorder([wrong_class, INVERTED_BOOKS[2]])
    done = execute(recorder, inversions(INVERTED, TAG))

    assert recorder.sent == []
    assert done[0]["status"] == "stale"
    assert "is not a credit" in done[0]["detail"]


def test_nothing_to_do_is_no_requests_at_all():
    """A scan that finds nothing should not cost a read."""
    recorder = Recorder(INVERTED_BOOKS)
    assert execute(recorder, []) == []
    assert recorder.sent == []


# -- reading back what actually happened -------------------------------------


def _feed(rows: list[tuple[str, int]]) -> dict[str, Any]:
    return {
        "fills": [
            {"client_order_id": key, "quantity": quantity, "symbol": "", "price": 0}
            for key, quantity in rows
        ],
        "rejections": [],
    }


def test_reconcile_reports_a_matched_pair_as_matched():
    recorder = Recorder(INVERTED_BOOKS)
    done = execute(recorder, inversions(INVERTED, TAG))
    ids = done[0]["client_order_ids"]

    recorder.feed = _feed([(ids["buy"], 6), (ids["buy"], 4), (ids["sell"], 10)])
    row = reconcile(recorder, done)[0]

    assert (row["bought"], row["sold"]) == (10, 10)
    assert row["matched"] is True
    assert row["naked"] == 0
    assert row["collected"] == pytest.approx(10 * 0.11, abs=1e-9)


def test_reconcile_reports_an_uneven_fill_as_a_naked_position():
    """The outcome `execute` cannot see, because the venue accepts before it fills.

    Both legs were accepted and one of them traded less, which is the ordinary
    shape of legging risk rather than an error anybody reported. Nothing is
    collected: the credit on a pair that is not a pair is not income.
    """
    recorder = Recorder(INVERTED_BOOKS)
    done = execute(recorder, inversions(INVERTED, TAG))
    ids = done[0]["client_order_ids"]

    recorder.feed = _feed([(ids["buy"], 10), (ids["sell"], 4)])
    row = reconcile(recorder, done)[0]

    assert (row["bought"], row["sold"]) == (10, 4)
    assert row["matched"] is False
    assert row["naked"] == 6
    assert row["collected"] == 0


def test_a_refusal_at_the_book_is_reported_against_the_legs_it_names():
    """The refusal that arrives after the acceptance, which is most of them here.

    `test_broker.py` pins the shape: an order the account cannot fund comes back
    `accepted` and is refused at the book, with the reason in this feed. A loop
    reading only `execute`'s result would believe the pair worked.
    """
    recorder = Recorder(INVERTED_BOOKS)
    done = execute(recorder, inversions(INVERTED, TAG))
    recorder.feed = {
        "fills": [],
        "rejections": [
            {"symbol": f"{TAG}_ELIM_CINDER_GT2", "reason": "insufficient_collateral"},
            {"symbol": "SOMETHING_ELSE", "reason": "price_band"},
        ],
    }
    row = reconcile(recorder, done)[0]

    assert row["refusals"] == ["insufficient_collateral"]
    assert (row["bought"], row["sold"]) == (0, 0)


# -- against the real application --------------------------------------------


@pytest.fixture(scope="module")
def board():
    """A venue with live matches on it, warmed until its ladders are matching.

    Everything below runs against the real FastAPI application in process: real
    routes, real signature verification, real matching engine, real fee
    schedule. A strategy is the component least worth testing against a
    description of its venue, because the parts of it that are wrong are the
    parts that disagree with the venue.

    Module scoped because building a board of three hundred match books and
    running it to continuous trading costs more than the three tests that read
    from it, and none of them writes anything another can see: each takes its
    own seat, and the only one that sends is a dry run that sends nothing.
    """
    venue = Exchange(MarketConfig())
    # Until a match is listed and its books are actually matching, rather than
    # for a fixed spell and hoping. This venue holds an opening call, and a
    # board inside one is quoted at the midpoint of its range on purpose.
    for _ in range(40):
        if any(
            ELIM.match(symbol) and venue.venue.session(symbol).matches_continuously
            for symbol in venue.symbols()
        ):
            break
        venue.pump(400, slices=16)
    else:
        venue.close()
        pytest.fail("no eliminations book reached continuous trading")
    yield venue
    venue.close()


@pytest.fixture
def live(board):
    """The connector signing as a seat of this test's own.

    The key is minted through the venue's own endpoint rather than reached for
    in its internals, so these are credentials the venue issued.
    """
    token = board.browser("Arbitrageur")
    key = board.issue(token, label="eliminations-ladder")
    client = ArenaClient(
        "http://testserver",
        ArenaSigner(key["key_id"], key["secret"]),
        http=board.client,
    )
    return client, board


def test_the_fee_is_read_off_the_live_venue_rather_than_assumed(live):
    """The schedule is configuration, so it is asked for.

    This venue ships four of them and `taker-only` charges 3 basis points
    rather than 2. A strategy holding a compiled constant would be sizing its
    break-even against a fee it is not paying.
    """
    client, _ = live
    assert taker_bps_from(client) == pytest.approx(MAKER_TAKER.taker_bps)


def test_every_trade_the_live_board_offers_is_one_that_could_be_taken(live):
    """Whatever the scan finds, it has to satisfy its own rules against real data.

    This does not assert that the board is inverted right now. It is a market
    with a maker on it, so most of the time it is not, and a test that demanded
    an arbitrage would be a test that passed when the venue was broken. What is
    asserted is that every pair returned is tradeable: two rungs of one ladder,
    lower bought and higher sold, both books matching, credit above the fee, and
    size inside both touches.
    """
    client, _ = live
    trades = scan(client)
    _, markets = client.fetch_universe()
    quoted = {market.ticker: market for market in markets}

    for trade in trades:
        low, high = ELIM.match(trade.buy), ELIM.match(trade.sell)
        assert low is not None and high is not None
        assert (low["tag"], low["who"]) == (high["tag"], high["who"])
        assert int(low["threshold"]) < int(high["threshold"])

        buy, sell = quoted[trade.buy], quoted[trade.sell]
        assert buy.status == "continuous" and sell.status == "continuous"
        assert trade.credit > fee_per_contract(
            buy.yes_ask / 100, sell.yes_bid / 100
        )
        assert trade.credit == pytest.approx(
            (sell.yes_bid - buy.yes_ask) / 100, abs=1e-9
        )
        assert 1 <= trade.size <= int(buy.yes_ask_size)
        assert trade.size <= int(sell.yes_bid_size)


def test_a_dry_run_against_the_real_venue_leaves_the_blotter_empty(live):
    """The mode that has to be trustworthy, checked where it can actually fail.

    A double that records orders cannot prove this: it proves the strategy did
    not call a method. This asks the venue whether it has any orders from this
    seat, which is the question somebody switching the mode on actually has.
    """
    client, venue = live
    done = execute(client, scan(client), dry_run=True)
    venue.pump(400, slices=16)

    assert all(row["status"] in ("dry_run", "stale") for row in done), done
    blotter = client.orders()
    assert blotter.get("orders") == []
    assert blotter.get("pending") == []
    assert client.fills()["fills"] == []
