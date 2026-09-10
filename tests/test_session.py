"""Call auctions, trading sessions, and the circuit breaker.

The clearing rule is a sequence of tie-breaks, and each one exists because the
one before it can leave more than one answer. Testing only the first would pass
on an implementation that resolves ties arbitrarily, and an arbitrary opening
price is exactly the failure an auction is meant to prevent, because it is the
price indices and settlements are struck at. So each tie-break gets a case
constructed to reach it and no further.

The other thing checked hard here is that auction fills go through the *same*
account path as continuous ones. An auction settling through its own accounting
would be the ideal place for a collateral or fee leak to hide.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from arena.contracts.payoff import Linear
from arena.contracts.spec import ContractSpec, DataPolicy, ObservationWindow
from arena.contracts.underlying import Single
from arena.exchange.engine import MatchingEngine
from arena.exchange.events import (
    Acknowledged,
    Filled,
    Rejected,
    Replace,
    Submit,
    Traded,
    Cancel,
)
from arena.exchange.session import SENTINEL, SessionState, indicative_auction
from arena.exchange.types import (
    AgentId,
    OrderType,
    Price,
    Quantity,
    RejectReason,
    Side,
    TimeInForce,
)
from arena.market.fees import MAKER_TAKER
from arena.market.instrument import Instrument
from arena.market.venue import FEE_ACCOUNT_ID, Venue
from arena.worlds.circuit.metrics import metric_ref

UTC = timezone.utc
START = datetime(2026, 8, 31, tzinfo=UTC)


def _book(orders, phase: SessionState = SessionState.PRE_OPEN) -> MatchingEngine:
    """Build a resting book from ``(agent, side, price_or_None, quantity)``."""
    engine = MatchingEngine("X")
    engine.phase = phase
    for who, side, price, quantity in orders:
        engine.apply(
            Submit(
                AgentId(who),
                side,
                Quantity(quantity),
                None if price is None else Price(price),
                OrderType.MARKET if price is None else OrderType.LIMIT,
                TimeInForce.IOC if price is None else TimeInForce.GTC,
            )
        )
    return engine


B, S = Side.BUY, Side.SELL


# --------------------------------------------------------------------------
# The clearing rule, tie-break by tie-break
# --------------------------------------------------------------------------


def test_the_auction_clears_where_most_volume_trades():
    """Rule one, and the only one most books ever reach."""
    engine = _book([("a", B, 105, 100), ("b", S, 95, 100), ("c", S, 104, 10)])
    result = indicative_auction(engine.book)
    assert result is not None
    assert result.volume == 100


def test_a_hand_computed_book_clears_where_it_should():
    """Worked through by hand, so the test is not the implementation restated.

    bids 102x50, 100x80, market x20   asks 99x60, 101x40

      p=99  demand 150 supply  60 -> volume  60, imbalance  +90
      p=100 demand 150 supply  60 -> volume  60, imbalance  +90
      p=101 demand  70 supply 100 -> volume  70, imbalance  -30
      p=102 demand  70 supply 100 -> volume  70, imbalance  -30

    Maximum volume is 70 at {101, 102}; both leave the same surplus; the surplus
    is on the sell side at both, so the lowest wins.
    """
    engine = _book(
        [
            ("a", B, 102, 50),
            ("b", B, 100, 80),
            ("m", B, None, 20),
            ("c", S, 99, 60),
            ("d", S, 101, 40),
        ]
    )
    result = indicative_auction(engine.book)
    assert (result.price, result.volume, result.imbalance) == (101, 70, -30)
    assert result.surplus_side is Side.SELL


def test_ties_on_volume_are_broken_by_the_smaller_surplus():
    """Rule two: among equal volumes, leave least unfilled."""
    engine = _book(
        [("a", B, 100, 100), ("b", B, 101, 100), ("c", S, 99, 100), ("d", S, 100, 5)]
    )
    result = indicative_auction(engine.book)
    surpluses = {}
    for price in (99, 100, 101):
        demand = sum(q for p, q in
                     ((100, 100), (101, 100)) if p >= price)
        supply = sum(q for p, q in ((99, 100), (100, 5)) if p <= price)
        surpluses[price] = (min(demand, supply), abs(demand - supply))
    best = max(surpluses.values())[0]
    assert result.volume == best
    assert abs(result.imbalance) == min(
        v[1] for v in surpluses.values() if v[0] == best
    )


def test_a_buy_surplus_pushes_the_price_up_and_a_sell_surplus_down():
    """Rule three, in both directions. A surplus means the price is wrong."""
    buy_heavy = _book([("a", B, 100, 200), ("b", B, 110, 200), ("c", S, 90, 50)])
    sell_heavy = _book([("a", B, 110, 50), ("c", S, 90, 200), ("d", S, 100, 200)])
    up = indicative_auction(buy_heavy.book)
    down = indicative_auction(sell_heavy.book)
    assert up.surplus_side is Side.BUY
    assert down.surplus_side is Side.SELL
    # Buyers unfilled everywhere -> the highest tied price; sellers -> the lowest.
    assert up.price == 110
    assert down.price == 90


def test_a_balanced_range_falls_back_to_the_reference_price():
    """Rule four. Any price in the range clears the same volume, so the previous
    price is the least arbitrary choice available."""
    engine = _book([("a", B, 110, 100), ("c", S, 90, 100)])
    low = indicative_auction(engine.book, reference=Price(92))
    high = indicative_auction(engine.book, reference=Price(108))
    assert low.imbalance == high.imbalance == 0
    assert low.price < high.price


def test_an_uncrossed_book_has_no_auction():
    engine = _book([("a", B, 90, 100), ("c", S, 110, 100)])
    assert indicative_auction(engine.book) is None


def test_a_one_sided_book_has_no_auction():
    assert indicative_auction(_book([("a", B, 100, 50)]).book) is None
    assert indicative_auction(_book([("c", S, 100, 50)]).book) is None


def test_market_orders_alone_clear_only_against_a_reference():
    """Two market orders imply no price at all, so there is nothing to clear at."""
    engine = _book([("a", B, None, 50), ("c", S, None, 50)])
    assert indicative_auction(engine.book) is None
    priced = indicative_auction(engine.book, reference=Price(100))
    assert priced is not None and priced.price == 100


def test_the_indicative_price_changes_nothing():
    """It is published during the call phase, so it must be a pure read."""
    engine = _book([("a", B, 102, 50), ("c", S, 99, 60)])
    before = engine.book.total_resting_quantity
    indicative_auction(engine.book)
    indicative_auction(engine.book)
    assert engine.book.total_resting_quantity == before
    assert engine.tape == ()


# --------------------------------------------------------------------------
# The call phase
# --------------------------------------------------------------------------


def test_a_crossed_book_does_not_trade_during_the_call():
    """The whole point: orders accumulate, however crossed they are."""
    engine = _book([("a", B, 200, 100), ("c", S, 50, 100)])
    assert engine.tape == ()
    assert engine.book.total_resting_quantity == 200


def test_immediate_orders_are_refused_during_a_call():
    """There is no 'immediately' in a call phase, so the instruction is refused
    rather than quietly turned into a resting order."""
    engine = MatchingEngine("X")
    engine.phase = SessionState.PRE_OPEN
    for tif in (TimeInForce.IOC, TimeInForce.FOK):
        events = engine.apply(
            Submit(AgentId("a"), B, Quantity(10), Price(100), OrderType.LIMIT, tif)
        )
        rejected = next(e for e in events if isinstance(e, Rejected))
        assert rejected.reason is RejectReason.NOT_ACCEPTED_IN_AUCTION
    assert engine.book.total_resting_quantity == 0


def test_a_market_order_becomes_a_market_on_open_order():
    """Required to be IOC in continuous trading, but nothing matches here."""
    engine = _book([("m", B, None, 25)])
    assert engine.book.total_resting_quantity == 25


def test_everyone_trades_at_one_price_and_the_eager_are_improved():
    """A buyer willing to pay 120 pays the clearing price, not its own limit.

    Price improvement is the reward for being in the auction, and it is why a
    cleared price is trustworthy where a first-arrival price is not.
    """
    engine = _book([("a", B, 120, 50), ("c", S, 100, 50)])
    engine.uncross()
    prices = {int(t.price) for t in engine.tape}
    assert len(prices) == 1
    assert prices.pop() < 120


def test_the_uncross_trades_exactly_the_indicative_volume():
    engine = _book(
        [("a", B, 102, 50), ("b", B, 100, 80), ("c", S, 99, 60), ("d", S, 101, 40)]
    )
    expected = indicative_auction(engine.book).volume
    engine.uncross()
    assert sum(int(t.quantity) for t in engine.tape) == expected


def test_an_auction_never_prints_a_wash_trade():
    """Worse here than in continuous trading: it would be struck at the official
    price and could move a settlement."""
    engine = _book([("same", B, 110, 100), ("same", S, 90, 100)])
    engine.uncross()
    assert engine.tape == ()


def test_nothing_trades_when_the_book_is_uncrossed():
    engine = _book([("a", B, 90, 50), ("c", S, 110, 50)])
    assert engine.uncross() == []
    assert engine.tape == ()


# --------------------------------------------------------------------------
# Sessions on the venue
# --------------------------------------------------------------------------


def _instrument() -> Instrument:
    return Instrument(
        "F",
        ContractSpec(
            contract_id="F",
            underlying=Single(metric_ref("win_rate", "SUBJECT")),
            payoff=Linear(10_000.0),
            window=ObservationWindow(START, START + timedelta(days=28)),
            policy=DataPolicy(
                min_sample_size=1_000, min_stratum_battles=200, min_strata_coverage=0.80
            ),
            reference_id="ref-2026S09-v1",
            published_at=START - timedelta(days=1),
            tick_size="0.25",
        ),
    )


def _venue(**kwargs) -> Venue:
    venue = Venue("arena", starting_cash=80_000_000, **kwargs)
    venue.list_instrument(_instrument())
    return venue


def _send(venue, who, side, price, quantity, tif=TimeInForce.GTC):
    return venue.submit(
        AgentId(who),
        "F",
        Submit(
            AgentId(who),
            side,
            Quantity(quantity),
            None if price is None else Price(price),
            OrderType.MARKET if price is None else OrderType.LIMIT,
            tif,
        ),
    )


def test_a_listed_symbol_opens_continuous():
    assert _venue().session("F") is SessionState.CONTINUOUS


def test_an_opening_auction_sets_the_first_price():
    venue = _venue()
    venue.begin_session("F")
    _send(venue, "a", B, 18_800, 100)
    _send(venue, "c", S, 18_600, 100)
    assert venue.engine("F").tape == ()

    result = venue.uncross("F")
    assert result is not None and result.volume == 100
    assert venue.session("F") is SessionState.CONTINUOUS
    assert len(venue.engine("F").tape) == 1


def test_auction_fills_land_in_accounts_and_conserve_exactly():
    """The same account path as continuous trading, so nothing can diverge."""
    venue = _venue()
    venue.begin_session("F")
    _send(venue, "a", B, 18_800, 120)
    _send(venue, "c", S, 18_600, 200)
    venue.uncross("F")
    assert venue.account(AgentId("a")).positions["F"].quantity == 120
    assert venue.account(AgentId("c")).positions["F"].quantity == -120
    assert venue.conservation_check() == 0


def test_an_auction_is_charged_rather_than_rebated():
    """Nobody crossed a spread, and that does not make everyone a maker.

    This test used to assert the opposite: that both sides earn the maker
    rate, on the reasoning that an auction has no aggressor. The reasoning
    is right and the conclusion was wrong, and running an opening auction
    for the first time is what showed it: a maker rate is a rebate, so a
    venue billing both sides of its own cross *pays out* on every share it
    crosses. Twenty-six opening auctions took venue revenue to minus 1,251
    before anything else went wrong. Exchanges charge for cross executions;
    they do not pay for them.
    """
    venue = _venue(fees=MAKER_TAKER)
    venue.begin_session("F")
    _send(venue, "a", B, 18_800, 100)
    _send(venue, "c", S, 18_600, 100)
    venue.uncross("F")
    assert int(venue.fees_collected) > 0, "the venue paid to open its own market"
    assert venue.conservation_check() == 0


def test_a_halt_accumulates_orders_and_reopens_with_an_auction():
    venue = _venue()
    _send(venue, "a", B, 18_600, 50)
    venue.halt("F", reason="news")
    assert venue.session("F") is SessionState.AUCTION

    _send(venue, "b", B, 18_900, 100)
    _send(venue, "c", S, 18_500, 100)
    assert venue.engine("F").tape == (), "a halted book must not trade"

    venue.uncross("F")
    assert venue.session("F") is SessionState.CONTINUOUS
    assert len(venue.engine("F").tape) >= 1
    assert venue.conservation_check() == 0


def test_a_trade_cannot_print_outside_the_band():
    """The half of the rule that protects anyone.

    Limit up-limit down does not only pause a runaway after the fact: it
    prevents trades outside its bands. Without that, a market order with no
    price protection walks a thin book to the floor. Measured on the live
    exchange, a resting bid at **0.25** was filled on a contract worth 4,700,
    and the breaker then dutifully halted a symbol whose damage was done.
    """
    venue = _venue(price_band=0.02, min_reference_prints=1)
    _send(venue, "c", S, 18_000, 50)
    _send(venue, "t", B, None, 50, tif=TimeInForce.IOC)
    assert venue.engine("F").tape, "the reference print did not happen"

    # Far outside the band, and a market order that would happily reach it.
    _send(venue, "c", S, 30_000, 50)
    _send(venue, "t", B, None, 50, tif=TimeInForce.IOC)
    assert all(int(t.price) < 30_000 for t in venue.engine("F").tape), (
        "a trade printed beyond the band"
    )


def test_a_limit_order_names_its_price_and_keeps_it():
    """The collar protects orders that named no price. That is all it does.

    Collaring limit orders too was tried and was far worse than the disease.
    They slid to the band's edge, the band later moved away from them, and
    the book locked: bid above offer, neither permitted to trade, and nothing
    in continuous trading able to clear it. Measured on that version: 2,492
    limit states in five minutes and a future marking at 9,267 against a
    settlement of 4,669.

    A trader who says 30,000 has said 30,000.
    """
    venue = _venue(price_band=0.02, min_reference_prints=1)
    _send(venue, "c", S, 18_000, 50)
    _send(venue, "t", B, None, 50, tif=TimeInForce.IOC)

    _send(venue, "d", B, 30_000, 10)
    book = venue.engine("F").book.snapshot()
    assert book.best_bid is not None and int(book.best_bid) == 30_000


def test_a_quote_pressing_against_the_band_is_a_limit_state_and_not_a_halt():
    """Three states, and the middle one is what makes it the rule it models.

    A symbol is in a limit state when the best bid or offer is *at* a band,
    interest that wants to be somewhere the venue will not let it go. One
    order reaching the edge is one order, so it starts a clock; only staying
    there stops the market. The clock has to advance for the pause to
    arrive, so this drives it.

    Judged from the quote rather than from a print, and it has to be: prints
    cannot leave the band any more, so a rule written in terms of them would
    never fire at all.
    """
    clock = {"now": 0}
    venue = _venue(price_band=0.02, min_reference_prints=1)
    venue.sim_clock = lambda: clock["now"]

    _send(venue, "c", S, 18_000, 50)
    _send(venue, "t", B, None, 50, tif=TimeInForce.IOC)
    assert venue.session("F") is SessionState.CONTINUOUS

    # A bid at the top of the band: allowed to rest, and pressing.
    low, high = venue.registry.require("F").tick_bounds
    edge = 18_000 + int(abs(int(high) - int(low)) * 0.02)
    _send(venue, "d", B, edge, 10)
    assert venue.session("F") is SessionState.CONTINUOUS, "one quote is not a halt"
    assert venue.halts[-1]["reason"] == "limit_state"

    clock["now"] = venue.limit_state_ns + 1
    _send(venue, "d", B, edge, 10)
    assert venue.session("F") is SessionState.AUCTION
    assert venue.halts[-1]["reason"] == "price_band"


def test_a_quote_that_steps_back_clears_the_limit_state():
    """Pressure that relieves itself is not a halt, however far it reached."""
    clock = {"now": 0}
    venue = _venue(price_band=0.02, min_reference_prints=1)
    venue.sim_clock = lambda: clock["now"]

    _send(venue, "c", S, 18_000, 50)
    _send(venue, "t", B, None, 50, tif=TimeInForce.IOC)
    low, high = venue.registry.require("F").tick_bounds
    edge = 18_000 + int(abs(int(high) - int(low)) * 0.02)

    events = _send(venue, "d", B, edge, 10)
    assert venue.halts[-1]["reason"] == "limit_state"
    order_id = next(e.order_id for e in events if hasattr(e, "order_id"))

    clock["now"] = venue.limit_state_ns // 2
    venue.submit(AgentId("d"), "F", Cancel(AgentId("d"), order_id))

    clock["now"] = venue.limit_state_ns * 4
    _send(venue, "e", B, 17_900, 5)
    assert venue.session("F") is SessionState.CONTINUOUS
    assert not [h for h in venue.halts if h["reason"] == "price_band"]


def test_the_breaker_stays_quiet_inside_the_band():
    venue = _venue(price_band=0.50)
    for price in (18_000, 18_500, 19_000):
        _send(venue, "c", S, price, 20)
        _send(venue, "t", B, None, 20, tif=TimeInForce.IOC)
    assert venue.session("F") is SessionState.CONTINUOUS
    assert venue.halts == []


def test_no_breaker_means_no_halts_however_far_price_moves():
    venue = _venue()
    for price in (18_000, 40_000, 1_000):
        _send(venue, "c", S, price, 20)
        _send(venue, "t", B, None, 20, tif=TimeInForce.IOC)
    assert venue.session("F") is SessionState.CONTINUOUS
    assert venue.halts == []


def test_a_closed_symbol_refuses_new_orders():
    venue = _venue()
    venue.close("F")
    assert venue.session("F") is SessionState.CLOSED
    events = _send(venue, "a", B, 18_600, 10)
    rejected = next(e for e in events if isinstance(e, Rejected))
    assert rejected.reason is RejectReason.ALREADY_TERMINAL


def test_halting_a_closed_symbol_does_nothing():
    """The outcome is already determined; there is nothing left to protect."""
    venue = _venue()
    venue.close("F")
    venue.halt("F")
    assert venue.session("F") is SessionState.CLOSED


def test_a_full_session_conserves_value_through_every_phase():
    """Open, trade, halt, reopen, close: one ledger throughout."""
    venue = _venue(fees=MAKER_TAKER, price_band=0.05)
    venue.begin_session("F")
    rng = random.Random(12)
    for _ in range(30):
        _send(venue, "a", B, rng.randint(18_400, 18_700), rng.randint(1, 30))
        _send(venue, "c", S, rng.randint(18_500, 18_800), rng.randint(1, 30))
    venue.uncross("F")
    assert venue.conservation_check() == 0

    for _ in range(60):
        _send(venue, "a", B, rng.randint(18_400, 18_700), rng.randint(1, 20))
        _send(venue, "c", S, rng.randint(18_500, 18_800), rng.randint(1, 20))
        if venue.session("F") is SessionState.AUCTION:
            venue.uncross("F")
        assert venue.conservation_check() == 0

    venue.halt("F", reason="close")
    venue.uncross("F")
    venue.close("F")
    assert venue.conservation_check() == 0


# --------------------------------------------------------------------------
# The call phase is defined by nothing matching in it
# --------------------------------------------------------------------------


def _order_id(events):
    return next(e.order_id for e in events if isinstance(e, Acknowledged))


def test_a_replace_during_a_call_phase_does_not_trade():
    """The property that makes a call phase a call phase, tested against the one
    command that used to break it.

    Submits accumulate, because the engine routes them to its accumulate path.
    A replace goes down a different road entirely: it pulls the old order and
    re-runs the match on the replacement, and it never asks what phase the book
    is in. So a halted book traded: measured, a replace during a halt printed
    20 lots at 17,000 against an order that was only resting there because the
    auction had not run yet.
    """
    venue = _venue()
    order_id = _order_id(_send(venue, "a", B, 18_000, 20))
    venue.halt("F", reason="news")
    _send(venue, "c", S, 17_000, 20)
    assert venue.engine("F").tape == (), "the halt did not stop the book"

    amend = venue.submit(
        AgentId("a"), "F", Replace(AgentId("a"), order_id, Quantity(20), Price(18_500))
    )
    rejected = next(e for e in amend if isinstance(e, Rejected))
    assert rejected.reason is RejectReason.NOT_ACCEPTED_IN_AUCTION
    assert venue.engine("F").tape == (), "a halted book traded on an amendment"
    assert venue.conservation_check() == 0


def test_a_replace_cannot_cross_a_market_on_open_order_at_its_sentinel_price():
    """The same hole, in the form that matters.

    A market-on-open order rests at a sentinel so that it crosses every
    candidate the auction considers; that is the whole point of it, and it is
    why continuous matching must never see one. A replace during the call phase
    matched against exactly that, and printed trades at
    **-4,611,686,018,427,387,904**: the same catastrophe the engine's
    unfilled-market-order sweep exists to prevent, reached through a door that
    sweep does not cover.
    """
    venue = _venue()
    venue.begin_session("F")
    order_id = _order_id(_send(venue, "a", B, 18_000, 20))
    _send(venue, "m", S, None, 20, tif=TimeInForce.IOC)
    assert any(
        abs(int(o.price)) >= SENTINEL
        for o in venue.engine("F").book.resting_orders
    ), "no market-on-open order rested, so the case was never reached"

    venue.submit(
        AgentId("a"), "F", Replace(AgentId("a"), order_id, Quantity(20), Price(18_500))
    )
    assert venue.engine("F").tape == ()
    assert all(abs(int(t.price)) < SENTINEL for t in venue.engine("F").tape)
    assert venue.conservation_check() == 0


def test_a_market_on_open_order_is_collateralised_before_the_auction_fills_it():
    """A market order's exposure is bounded by the book only while the book is
    what it will trade against.

    In continuous trading that is exact: the engine walks the same levels in
    the same instant. In a call phase it is worthless. The order rests until
    the uncross and then trades against liquidity that had not arrived yet. So
    the book walk found an empty ask side, concluded there was nothing to
    collateralise, and let a 100,000-lot market-on-open buy rest on an account
    holding 10,000. Sellers arrived, the auction cleared it in full, and the
    account came out of its own opening auction with free cash of
    **-449,990,000,000,000** minor units.
    """
    venue = Venue("arena", starting_cash=10_000, balances={AgentId("c"): 10_000_000_000})
    venue.list_instrument(_instrument())
    venue.begin_session("F")

    refused = _send(venue, "whale", B, None, 100_000, tif=TimeInForce.IOC)
    reason = next(e for e in refused if isinstance(e, Rejected)).reason
    assert reason is RejectReason.INSUFFICIENT_COLLATERAL
    assert venue.engine("F").book.total_resting_quantity == 0

    _send(venue, "c", S, 18_000, 100_000)
    venue.uncross("F")
    assert venue.account(AgentId("whale")).positions.get("F") is None
    assert int(venue.account(AgentId("whale")).free_cash) >= 0
    assert venue.conservation_check() == 0


def test_a_market_on_open_order_it_can_afford_still_goes_through():
    """The permissive half. Reserving against the far end of the range is the
    honest assumption for an order that named no price, not a way to refuse the
    order type: an agent that can cover the worst case must still be able to
    take part in the auction.
    """
    venue = _venue()
    venue.begin_session("F")
    _send(venue, "m", B, None, 20, tif=TimeInForce.IOC)
    _send(venue, "c", S, 18_000, 20)
    result = venue.uncross("F")
    assert result is not None and result.volume == 20
    assert venue.account(AgentId("m")).positions["F"].quantity == 20
    assert venue.conservation_check() == 0


# --------------------------------------------------------------------------
# What the session machinery owes the contract's own calendar
# --------------------------------------------------------------------------


def _paused_venue(clock, calendar):
    """A venue whose one symbol has been paused by its own circuit breaker."""
    venue = Venue(
        "arena",
        starting_cash=80_000_000,
        price_band=0.02,
        min_reference_prints=1,
        clock=lambda: calendar["t"],
    )
    venue.list_instrument(_instrument())
    venue.sim_clock = lambda: clock["now"]
    _send(venue, "c", S, 18_000, 50)
    _send(venue, "t", B, None, 50, tif=TimeInForce.IOC)
    low, high = venue.registry.require("F").tick_bounds
    edge = 18_000 + int(abs(int(high) - int(low)) * 0.02)
    _send(venue, "d", B, edge, 40)
    clock["now"] = venue.limit_state_ns + 1
    _send(venue, "d", B, edge, 10)
    assert venue.session("F") is SessionState.AUCTION, "the breaker did not pause it"
    return venue


def test_a_contract_that_expires_while_paused_is_not_reopened():
    """The expiry rule was enforced on exactly one path: the arrival of an
    order. A paused symbol is precisely the one nobody sends orders to.

    So a symbol whose observation window closed while the breaker had it paused
    was still reported by ``reopen_due``, and the reopening auction printed 40
    lots at 18,800 on a contract whose outcome was already determined, then put
    it back into continuous trading. Anyone trading there is trading against an
    answer that already exists.
    """
    clock, calendar = {"now": 0}, {"t": START}
    venue = _paused_venue(clock, calendar)
    _send(venue, "e", S, 17_500, 40)
    prints = len(venue.engine("F").tape)

    calendar["t"] = venue.registry.require("F").expiry + timedelta(days=1)
    clock["now"] += venue.pause_ns + 1

    assert venue.reopen_due() == ()
    assert venue.uncross("F") is None
    assert len(venue.engine("F").tape) == prints, "a trade printed after expiry"
    assert venue.session("F") is SessionState.CLOSED
    assert venue.conservation_check() == 0


def test_a_live_contract_still_reopens_when_its_pause_runs_out():
    """The other half, so the guard above cannot be a way of never reopening
    anything. The pause is a timer, and when it runs the symbol comes back,
    through an auction, as it must.
    """
    clock, calendar = {"now": 0}, {"t": START}
    venue = _paused_venue(clock, calendar)
    _send(venue, "e", S, 17_500, 40)
    clock["now"] += venue.pause_ns + 1

    assert venue.reopen_due() == ("F",)
    result = venue.uncross("F")
    assert result is not None and result.volume > 0
    assert venue.session("F") is SessionState.CONTINUOUS
    assert venue.conservation_check() == 0


def test_a_manual_halt_outlives_the_breakers_own_timer():
    """A halt somebody decided on ends when somebody decides it ends.

    The breaker's pause carries a reopen time, and it survived an operator
    halting the same symbol for a different reason. So the moment the
    band's pause ran out, ``reopen_due`` offered up a symbol a human had
    stopped for news, and the operator's halt quietly expired on a schedule
    the operator never set.
    """
    clock, calendar = {"now": 0}, {"t": START}
    venue = _paused_venue(clock, calendar)
    venue.halt("F", reason="news")
    clock["now"] += venue.pause_ns + 1

    assert venue.reopen_due() == ()
    assert venue.session("F") is SessionState.AUCTION


def test_a_session_cannot_be_opened_on_a_symbol_that_has_settled():
    """A settled contract has paid out and can never pay out again, because
    ``settle`` refuses to fire twice.

    Opening a call phase on one let orders accumulate and the next uncross
    crossed them: measured, a participant came out of it holding 10 lots of a
    contract whose position would be marked forever and realised never.
    """
    from decimal import Decimal

    from arena.settlement.result import SettlementResult, SettlementStatus

    venue = _venue()
    instrument = venue.registry.require("F")
    _send(venue, "a", B, 18_600, 20)
    _send(venue, "c", S, 18_600, 20)
    venue.settle(
        "F",
        SettlementResult(
            contract_id="F",
            spec_digest=instrument.spec.spec_digest,
            status=SettlementStatus.SETTLED,
            settlement_value=Decimal("4700"),
            underlying_level=0.47,
            resolutions=(),
        ),
    )
    with pytest.raises(ValueError, match="already settled"):
        venue.begin_session("F")
    assert venue.session("F") is SessionState.CLOSED

    # And the other door into the same room: an uncross must not reopen it.
    assert venue.uncross("F") is None
    assert venue.session("F") is SessionState.CLOSED
    assert venue.conservation_check() == 0


def test_an_auction_price_is_counted_once_in_the_reference_window():
    """The reference is a mean over the prints in a window, so a print counted
    twice is worth twice the opinion it should be.

    The uncross recorded its cleared price into the window itself, on top of
    the ``Traded`` events it had just produced, which the fill path records
    like any other print. One auction, one print, two entries. That
    double-weights the auction in the mean the bands are drawn around, and
    inflates the count the halting rule waits for.
    """
    venue = _venue(price_band=0.05, min_reference_prints=1)
    venue.begin_session("F")
    _send(venue, "a", B, 18_800, 100)
    _send(venue, "c", S, 18_600, 100)
    result = venue.uncross("F")

    prints = [int(t.price) for t in venue.engine("F").tape]
    window = [int(p) for _t, p in venue._recent["F"]]
    assert window == prints
    assert int(venue._reference_price("F")) == int(result.price)


# --------------------------------------------------------------------------
# What an auction's own prints set off
# --------------------------------------------------------------------------


def _stop(venue, who, side, quantity, trigger):
    return venue.submit(
        AgentId(who),
        "F",
        Submit(
            AgentId(who), side, Quantity(quantity), None, OrderType.STOP,
            TimeInForce.GTC, stop_price=Price(trigger),
        ),
    )


def _tape(venue):
    return [(int(t.quantity), int(t.price)) for t in venue.engine("F").tape]


def test_an_uncross_releases_the_stops_its_prints_trigger():
    """The same book cleared two ways has to reach the same place.

    A print is a print whoever made it, and an auction's prints were setting off
    nothing. Measured on a sell stop at 18,000 with a bid of twenty at 17,900
    behind it: traded continuously the tape read ``[(10, 18000), (10, 17900)]``
    and the stop was gone, cleared by an auction it read ``[(10, 18000)]`` and
    the stop was still parked. An auction is the event most likely to gap
    through a stop, which is the reason the stop is there.
    """
    def book(call_phase: bool):
        venue = _venue()
        if call_phase:
            venue.begin_session("F")
        _send(venue, "rest", B, 17_900, 20)
        _stop(venue, "s", S, 10, 18_000)
        _send(venue, "a", S, 18_000, 10)
        _send(venue, "b", B, 18_000, 10)
        if call_phase:
            venue.uncross("F")
        return venue

    continuous, auctioned = book(False), book(True)
    assert _tape(continuous) == [(10, 18_000), (10, 17_900)]
    assert _tape(auctioned) == _tape(continuous), "the auction jumped over the stop"
    assert auctioned.engine("F")._stops == []
    assert auctioned.conservation_check() == 0


def test_a_stop_released_by_an_uncross_lands_in_a_continuous_book():
    """The sequencing is the fix, and this is what it is protecting.

    An uncross runs while the symbol is still in its call phase, so a stop
    released inside it becomes a market order arriving at a book that does not
    match, and an unmatched market order is accumulated at the sentinel price.
    That is the order that once printed trades at -4,611,686,018,427,387,904
    and billed 4.8e22 in fees. Releasing only after the venue has flipped the
    symbol back to continuous is what keeps the cascade in a book that can
    actually fill it.
    """
    venue = _venue(fees=MAKER_TAKER)
    venue.begin_session("F")
    _send(venue, "rest", B, 17_900, 20)
    _stop(venue, "s", S, 10, 18_000)
    _send(venue, "a", S, 18_000, 10)
    _send(venue, "b", B, 18_000, 10)

    result, events = venue.uncross_events("F")
    assert result is not None and result.volume == 10
    assert venue.session("F") is SessionState.CONTINUOUS

    assert all(abs(int(t.price)) < SENTINEL for t in venue.engine("F").tape)
    assert all(
        abs(int(o.price)) < SENTINEL
        for o in venue.engine("F").book.resting_orders
    )
    # Delivered, not merely booked. The agents whose stops fired have to hear.
    assert sum(1 for e in events if isinstance(e, Traded)) == 2
    assert int(venue.fees_collected) > 0
    assert venue.conservation_check() == 0


def test_a_cascade_begun_by_an_uncross_still_respects_the_depth_bound():
    """A cascade is real and this does not prevent it; a cascade that never
    terminates would be a bug in the model rather than an event in the market.

    Entering the chain through an auction must not get round the bound, and the
    stops the bound leaves behind must stay live orders rather than vanishing.
    """
    venue = _venue(fees=MAKER_TAKER)
    venue.begin_session("F")
    ladder = 40
    for i in range(ladder):
        _stop(venue, f"stop-{i}", S, 2, 18_000 - i * 10)
        _send(venue, f"bid-{i}", B, 17_990 - i * 10, 2)
    assert len(venue.engine("F")._stops) == ladder

    _send(venue, "a", S, 18_000, 5)
    _send(venue, "b", B, 18_000, 5)
    venue.uncross("F")

    engine = venue.engine("F")
    bound = engine._max_cascade
    assert engine.cascade_depth == [bound]
    assert len(engine._stops) == ladder - bound, "stops the bound left behind vanished"
    assert all(abs(int(t.price)) < SENTINEL for t in engine.tape)
    assert venue.conservation_check() == 0

    # And the session that follows it is an ordinary one.
    for i in range(20):
        _send(venue, "x", B, 17_500 + i, 3)
        _send(venue, "y", S, 17_500 + i, 3)
        assert venue.conservation_check() == 0
