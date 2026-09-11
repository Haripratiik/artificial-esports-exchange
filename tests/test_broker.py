"""This venue as a broker, against the real application.

The point of a broker adapter is that something written for a different venue
calls it without knowing. That cannot be checked by reading it, so every test
here drives the actual FastAPI application in process: real routes, real
signature verification, real matching engine, real ledger.
"""

from __future__ import annotations

import pytest

from dashboard.state import MarketConfig

from connectors.portfolio_manager import ArenaBroker

from tests.test_api import Exchange, resting_price


@pytest.fixture
def broker():
    """One exchange, one seat, and a broker pointed at it.

    The seat's key is minted through the venue's own endpoint rather than
    reached for in its internals, so the credentials this broker signs with are
    credentials the venue issued.
    """
    # Without matches. A broker adapter is about orders, positions and cash,
    # and none of these tests looks at a match contract, but the default
    # listing carries 308 live match books against 50 statistical ones and
    # every test here builds its own venue. Paying seven times over for
    # instruments nothing asserts on is how a test file becomes the reason
    # nobody runs the suite.
    venue = Exchange(MarketConfig(matches=False))
    # Run until the book is actually matching, rather than for a fixed spell
    # and hoping. This venue holds an opening call, and a market order sent
    # into one does not trade: it rests as market-on-open and waits for the
    # uncross. Measured, a flat 400ms left the symbol in PRE_OPEN, and the test
    # that wanted a position read an empty list with no rejection and nothing
    # pending, which looks exactly like a broker that silently dropped an order.
    for _ in range(40):
        if venue.venue.session(venue.restable()).matches_continuously:
            break
        venue.pump(400, slices=16)
    token = venue.browser("Allocator")
    key = venue.issue(token, label="portfolio-manager")
    made = ArenaBroker(
        base_url="http://testserver",
        key_id=key["key_id"],
        secret=key["secret"],
        http=venue.client,
    )
    yield made, venue
    venue.close()


def test_the_account_reports_buying_power_as_free_cash(broker):
    """Not cash, and the difference is this venue's whole argument.

    Collateral posted against an open position is not available to open
    another one. A broker that reported cash as buying power would have every
    allocator above it sizing against money that is already spoken for, which
    on a venue with exact collateral is not a rounding error but a refusal
    waiting to happen.
    """
    made, _ = broker
    account = made.get_account()
    assert account.cash > 0
    assert account.equity > 0
    assert account.buyingPower <= account.cash
    assert account.portfolioValue == account.equity


def test_an_order_reaches_the_book_and_comes_back_as_a_broker_order(broker):
    """place_order then get_orders, through the same interface Alpaca offers."""
    made, venue = broker
    symbol = venue.restable()
    price = float(resting_price(venue, symbol, "buy"))

    placed = made.place_order(symbol, "buy", 5, order_type="limit", limit_price=price)
    assert placed.status != "rejected", placed.message
    assert placed.symbol == symbol
    assert placed.qty == 5

    venue.pump(400, slices=16)
    working = made.get_orders("open")
    assert [order.orderId for order in working] == [placed.orderId]
    resting = working[0]
    assert resting.symbol == symbol
    assert resting.side == "buy"
    assert resting.orderType == "limit"
    assert resting.qty == 5


def test_an_order_the_account_cannot_fund_is_refused_after_it_is_accepted(broker):
    """The difference between this venue and Alpaca that an allocator must know.

    Alpaca refuses synchronously, so a caller learns from the return value.
    Here the POST is accepted while the order is still in flight and the
    refusal happens at the book, which is why `status` comes back `accepted`
    on an order that will never rest. A loop that only reads `OrderResult`
    would believe this worked.

    So the refusal has to be findable, and `rejections()` is where: the venue
    reports it in the fills feed with its own reason code. Asserted here
    rather than described, because an allocator reweighting on a position it
    does not hold is the failure this prevents.
    """
    made, venue = broker
    symbol = venue.restable()
    price = float(resting_price(venue, symbol, "buy"))

    taken = made.place_order(
        symbol, "buy", 10_000_000, order_type="limit", limit_price=price
    )
    assert taken.status == "accepted", taken.message
    venue.pump(400, slices=16)

    reasons = {row.get("reason") for row in made.rejections()}
    assert "insufficient_collateral" in reasons, made.rejections()
    # And it never rested, so nothing above should think it has exposure.
    assert taken.orderId not in [order.orderId for order in made.get_orders("open")]


def test_a_transport_level_refusal_is_a_result_rather_than_an_exception(broker):
    """A malformed order must not take down the loop that sent it.

    The venue refuses this one outright, before the book, so the adapter has a
    refusal in hand and reports it where the caller already looks.
    """
    made, venue = broker
    refused = made.place_order(venue.restable(), "sideways", 5)
    assert refused.status == "rejected"
    assert refused.message


def test_an_order_for_no_lots_is_refused_without_asking_the_venue(broker):
    """Nothing is sent, because there is nothing to send."""
    made, venue = broker
    refused = made.place_order(venue.restable(), "buy", 0)
    assert refused.status == "rejected"
    assert refused.orderId == ""


def test_cancelling_by_id_finds_the_book_the_order_is_in(broker):
    """The interface passes an id; this venue needs a symbol as well.

    One matching engine per symbol means an order is keyed by both, so the
    blotter is walked to find which book holds the id. An id nobody is working
    is reported False rather than guessed at, because cancelling the wrong
    symbol's order is a worse failure than not cancelling.
    """
    made, venue = broker
    symbol = venue.restable()
    price = float(resting_price(venue, symbol, "buy"))
    placed = made.place_order(symbol, "buy", 3, order_type="limit", limit_price=price)
    venue.pump(400, slices=16)

    assert made.cancel_order(placed.orderId) is True
    venue.pump(400, slices=16)
    assert made.get_orders("open") == []

    assert made.cancel_order("999999") is False


def test_cancel_all_reports_how_many_it_pulled(broker):
    made, venue = broker
    symbol = venue.restable()
    price = float(resting_price(venue, symbol, "buy"))
    for _ in range(3):
        made.place_order(symbol, "buy", 2, order_type="limit", limit_price=price)
    venue.pump(400, slices=16)
    assert len(made.get_orders("open")) == 3

    assert made.cancel_all_orders() >= 1
    venue.pump(400, slices=16)
    assert made.get_orders("open") == []


def test_a_filled_order_becomes_a_position_the_allocator_can_read(broker):
    """The loop that matters: order, fill, position, close.

    Marked against the venue's own mark rather than against the entry price,
    because an allocator reweighting a book needs what a holding is worth now,
    and the position row deliberately does not carry that: what it is worth is
    a fact about the market, not about the position.
    """
    made, venue = broker
    symbol = venue.restable()

    bought = made.place_order(symbol, "buy", 2, order_type="market")
    assert bought.status != "rejected", bought.message
    venue.pump(400, slices=16)

    held = made.get_position(symbol)
    assert held is not None, made.get_positions()
    assert held.qty == 2
    assert held.side == "long"
    assert held.currentPrice > 0
    assert held.marketValue == pytest.approx(held.qty * held.currentPrice)

    closed = made.close_position(symbol)
    assert closed.side == "sell"
    assert closed.qty == 2


def test_closing_a_position_nobody_holds_is_refused_rather_than_sent(broker):
    made, venue = broker
    refused = made.close_position(venue.restable())
    assert refused.status == "rejected"
    assert refused.message == "no position to close"


def test_a_dry_run_broker_sends_nothing(broker):
    """The mode the interface reserves for a broker that does not trade."""
    _, venue = broker
    quiet = ArenaBroker(base_url="http://testserver", dry_run=True, http=venue.client)
    assert quiet.mode == "dry_run"
    result = quiet.place_order(venue.restable(), "buy", 5)
    assert result.status == "dry_run"
    assert quiet.get_orders("open") == []


def test_the_mode_is_paper_because_orders_really_match(broker):
    """Not dry_run: these orders take queue priority and settle on a real ledger.

    The money is the only synthetic part, which is what paper trading means.
    """
    made, _ = broker
    assert made.mode == "paper"


def test_closed_orders_are_answered_empty_rather_than_raising(broker):
    """This venue's blotter holds what is live, and history lives elsewhere.

    A caller polling on a loop should not die because it asked a question this
    venue answers somewhere else.
    """
    made, _ = broker
    assert made.get_orders("closed") == []
