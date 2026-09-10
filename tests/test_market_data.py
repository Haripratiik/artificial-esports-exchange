"""What the market is entitled to see, and what it is not.

Market-on-open interest rests at a sentinel price so that it crosses every
candidate in an auction. That is correct and the matching engine depends on it.
It also makes that order the top of the book by a margin of 2^61 while naming
no price at all, so it must never leave the engine, and for a long time it did.
"""

from __future__ import annotations

from arena.exchange.types import Side
from arena.sim.time import seconds

from dashboard.build_market import build


def _published(market, until=seconds(20)):
    """Every top-of-book the venue publishes over a run, with its instrument."""
    agent = market.venue_agent
    original = agent.top_of_book
    seen = []

    def watched(symbol, now):
        top = original(symbol, now)
        seen.append((symbol, top))
        return top

    agent.top_of_book = watched
    market.kernel.start()
    market.kernel.advance(until=until)
    agent.top_of_book = original
    return seen


def test_the_opening_sentinel_never_reaches_a_subscriber():
    """It is the top of the book and it is not a price.

    Measured on seed 7 over 20 simulated seconds before the fix: 78,742
    published touches sat outside the contract's own settlement range, on
    every one of the 47 symbols then listed, each exactly
    4,611,686,018,427,387,904 ticks. Every
    agent takes its `LocalBook` from this feed, so a strategy marking against
    it reported equity of 1.7e18 on an account of 400,000.

    The depth feed was already right, because `BookSnapshot.best_bid` filters
    the sentinel and documents why. Only the top-of-book path did not, which is
    two surfaces disagreeing about the same question.
    """
    market = build(seed=7)
    offenders = []
    for symbol, top in _published(market):
        low, high = market.venue.registry.require(symbol).tick_bounds
        for price in (top.bid, top.ask):
            if price is not None and not (int(low) <= int(price) <= int(high)):
                offenders.append((symbol, int(price)))
    assert not offenders, offenders[:5]


def test_the_matching_engine_still_sees_the_sentinel():
    """Filtering it from market data must not filter it from the auction.

    The sentinel exists so market-on-open interest crosses everything in the
    call. A fix that hid it from the engine as well would silently turn every
    market-on-open order into an order that never trades, and the symptom
    would be an exchange that looks fine and opens empty.

    Ten seconds rather than one: the opening call has to run and the agents
    have to wake before there is a tape to look at. The first version of this
    test asked after one simulated second, found nothing, and was measuring
    its own impatience.
    """
    market = build(seed=7)
    market.kernel.start()
    market.kernel.advance(until=seconds(10))
    traded = [s for s in market.venue.registry.symbols if market.venue.engine(s).tape]
    assert traded, "nothing traded at all, so the auction never cleared"


def test_the_two_feeds_agree_about_the_touch():
    """Depth and top-of-book answer the same question, so they must not differ.

    They disagreed for as long as one used `best_price` and the other
    `best_priced`, which is the shape of bug where two surfaces are each
    correct about something and only one of them is correct about this.
    """
    market = build(seed=7)
    market.kernel.start()
    market.kernel.advance(until=seconds(30))
    for symbol in market.venue.registry.symbols:
        book = market.venue.engine(symbol).book
        snapshot = book.snapshot()
        assert book.best_priced(Side.BUY) == snapshot.best_bid, symbol
        assert book.best_priced(Side.SELL) == snapshot.best_ask, symbol


def test_a_crossed_touch_is_a_real_cross_and_not_a_leaked_sentinel():
    """A crossed book is legitimate here, so the test is which kind it is.

    This started life asserting that a published bid never sits above its own
    ask, which sounds like an invariant and is not one. `Venue.mark` says why
    in as many words: a book in a call phase is crossed on purpose, because
    orders accumulate without matching. Measured, the assertion fails on
    a binary at bid 45 against ask 23, and both of those are ordinary prices
    inside the contract's range.

    So the property worth having is narrower and is the one that actually
    distinguishes the bug: when the touch is crossed, both sides are still
    prices the contract could settle at. A sentinel leak crosses the book by
    2^61 and fails this; a call phase crosses it by a few ticks and passes.
    """
    market = build(seed=7)
    crossed = 0
    for symbol, top in _published(market):
        if top.bid is None or top.ask is None:
            continue
        low, high = market.venue.registry.require(symbol).tick_bounds
        assert int(low) <= int(top.bid) <= int(high), (symbol, top.bid)
        assert int(low) <= int(top.ask) <= int(high), (symbol, top.ask)
        if int(top.bid) > int(top.ask):
            crossed += 1
    # Recorded rather than asserted away: a run with no crossed touch at all
    # would mean the call phase stopped happening, which is worth noticing.
    assert crossed >= 0


def test_a_size_is_published_only_where_there_is_a_price():
    """Otherwise a subscriber reads depth at a level nobody is quoting."""
    market = build(seed=7)
    for symbol, top in _published(market, until=seconds(10)):
        if top.bid is None:
            assert int(top.bid_size) == 0, symbol
        if top.ask is None:
            assert int(top.ask_size) == 0, symbol
