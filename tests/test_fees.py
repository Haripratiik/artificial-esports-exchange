"""Maker-taker fees, and the order type that exists because of them.

Two things could go wrong here and only one of them is loud. The loud one is
charging the wrong amount. The quiet one is charging the right amount and
putting it nowhere: the ledger would still balance to within a rounding error,
every test that checks "roughly conserved" would pass, and the venue's central
invariant would have become an approximation without anyone noticing.

So the conservation tests here are exact-equality tests, and there is a separate
one asserting that the venue's own account holds precisely what the participants
paid.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from arena.contracts.payoff import Linear
from arena.contracts.spec import ContractSpec, DataPolicy, ObservationWindow
from arena.contracts.underlying import Single
from arena.exchange.events import Filled, Rejected, Submit
from arena.exchange.types import (
    AgentId,
    OrderType,
    Price,
    Quantity,
    RejectReason,
    Side,
    TimeInForce,
)
from arena.market.fees import FREE, MAKER_TAKER, FeeSchedule
from arena.market.instrument import Instrument
from arena.market.venue import FEE_ACCOUNT_ID, Venue
from arena.portfolio.money import Money
from arena.worlds.circuit.metrics import metric_ref

UTC = timezone.utc
START = datetime(2026, 8, 31, tzinfo=UTC)
MAKER = AgentId("maker")
TAKER = AgentId("taker")


def _instrument() -> Instrument:
    spec = ContractSpec(
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
    )
    return Instrument("F", spec)


def _venue(fees: FeeSchedule = MAKER_TAKER) -> Venue:
    venue = Venue("arena", starting_cash=50_000_000, fees=fees)
    venue.list_instrument(_instrument())
    return venue


def _rest(venue: Venue, side: Side, price: int, quantity: int, who: AgentId = MAKER):
    return venue.submit(
        who,
        "F",
        Submit(who, side, Quantity(quantity), Price(price), OrderType.LIMIT,
               TimeInForce.GTC),
    )


def _cross(venue: Venue, side: Side, quantity: int, who: AgentId = TAKER):
    return venue.submit(
        who,
        "F",
        Submit(who, side, Quantity(quantity), None, OrderType.MARKET, TimeInForce.IOC),
    )


# --------------------------------------------------------------------------
# The schedule itself
# --------------------------------------------------------------------------


def test_a_free_schedule_charges_nothing():
    assert FREE.free
    assert int(FREE.charge(10**9, aggressor=True)) == 0
    assert int(FREE.charge(10**9, aggressor=False)) == 0


def test_the_taker_pays_and_the_maker_is_paid():
    assert int(MAKER_TAKER.charge(1_000_000, aggressor=True)) > 0
    assert int(MAKER_TAKER.charge(1_000_000, aggressor=False)) < 0


@pytest.mark.parametrize("notional", [1, 7, 999, 333_333, 10**9])
def test_rounding_always_favours_the_venue(notional):
    """A charge rounds up and a rebate rounds down in magnitude.

    The other way round, a strategy of many tiny fills would extract a
    fraction of a unit per trade, precisely the leak the integer ledger
    exists to make impossible.
    """
    taker = int(MAKER_TAKER.charge(notional, aggressor=True))
    maker = int(MAKER_TAKER.charge(notional, aggressor=False))
    assert taker >= notional * MAKER_TAKER.taker_bps / 10_000
    assert maker >= notional * MAKER_TAKER.maker_bps / 10_000
    assert taker + maker >= 0, "the venue paid out more than it took in"


def test_a_schedule_that_pays_more_than_it_takes_is_representable():
    """Venues really do this, and really do lose money doing it.

    Refusing to model it would hide a genuine failure mode behind a validation
    error, so it is allowed and simply shows up as a negative venue balance.
    """
    generous = FeeSchedule(taker_bps=1.0, maker_bps=-3.0)
    total = int(generous.charge(1_000_000, True)) + int(
        generous.charge(1_000_000, False)
    )
    assert total < 0


# --------------------------------------------------------------------------
# Fees through the ledger
# --------------------------------------------------------------------------


def test_value_is_conserved_exactly_with_fees_on():
    """Zero, not nearly zero. A fee is a transfer, not an evaporation."""
    venue = _venue()
    rng = random.Random(4)
    for _ in range(120):
        _rest(venue, Side.BUY, rng.randint(18_000, 18_500), rng.randint(1, 40))
        _rest(venue, Side.SELL, rng.randint(18_600, 19_000), rng.randint(1, 40))
        _cross(venue, rng.choice([Side.BUY, Side.SELL]), rng.randint(1, 30))
        assert venue.conservation_check() == 0


def test_the_venue_account_holds_exactly_what_was_charged():
    """The quiet failure: right amount charged, wrong amount banked."""
    venue = _venue()
    _rest(venue, Side.SELL, 18_800, 500)
    _cross(venue, Side.BUY, 300)
    treasury = venue.account(FEE_ACCOUNT_ID)
    banked = int(treasury.cash) - int(treasury.starting_cash)
    assert banked == int(venue.fees_collected)
    assert banked > 0


def test_the_taker_is_charged_and_the_maker_credited_on_a_real_fill():
    venue = _venue()
    _rest(venue, Side.SELL, 18_800, 200)
    before_maker = int(venue.account(MAKER).cash)
    before_taker = int(venue.account(TAKER).cash)
    _cross(venue, Side.BUY, 100)
    assert int(venue.account(MAKER).cash) > before_maker
    assert int(venue.account(TAKER).cash) < before_taker


def test_fees_scale_with_the_notional_traded():
    small, large = _venue(), _venue()
    _rest(small, Side.SELL, 18_800, 500)
    _rest(large, Side.SELL, 18_800, 500)
    _cross(small, Side.BUY, 10)
    _cross(large, Side.BUY, 400)
    assert int(large.fees_collected) > int(small.fees_collected)


def test_a_free_venue_behaves_exactly_as_before():
    """Fees are off by default, so every existing measurement keeps its meaning."""
    venue = _venue(FREE)
    _rest(venue, Side.SELL, 18_800, 300)
    _cross(venue, Side.BUY, 200)
    assert int(venue.fees_collected) == 0
    assert FEE_ACCOUNT_ID not in venue.accounts
    assert venue.conservation_check() == 0


# --------------------------------------------------------------------------
# Post-only
# --------------------------------------------------------------------------


def _post_only(venue: Venue, side: Side, price: int, quantity: int = 50):
    return venue.submit(
        MAKER,
        "F",
        Submit(MAKER, side, Quantity(quantity), Price(price), OrderType.LIMIT,
               TimeInForce.POST_ONLY),
    )


def test_a_post_only_order_that_would_cross_is_rejected():
    venue = _venue()
    _rest(venue, Side.SELL, 18_800, 100, who=TAKER)
    events = _post_only(venue, Side.BUY, 18_900)
    rejected = next(e for e in events if isinstance(e, Rejected))
    assert rejected.reason is RejectReason.POST_ONLY_WOULD_CROSS
    assert not any(isinstance(e, Filled) for e in events)


def test_a_post_only_order_that_does_not_cross_rests_normally():
    venue = _venue()
    _rest(venue, Side.SELL, 18_800, 100, who=TAKER)
    _post_only(venue, Side.BUY, 18_700)
    book = venue.engine("F").book
    assert int(book.best_price(Side.BUY)) == 18_700


def test_a_post_only_market_order_is_a_contradiction_and_is_refused():
    """A market order is defined by being willing to cross."""
    venue = _venue()
    events = venue.submit(
        MAKER,
        "F",
        Submit(MAKER, Side.BUY, Quantity(10), None, OrderType.MARKET,
               TimeInForce.POST_ONLY),
    )
    rejected = next(e for e in events if isinstance(e, Rejected))
    assert rejected.reason is RejectReason.MARKET_ORDER_MUST_BE_IOC


def test_post_only_guarantees_the_rebate_rather_than_the_fee():
    """The reason the order type exists, measured end to end.

    A maker that crosses by accident pays the taker fee instead of earning the
    rebate, which can turn a profitable quote into a losing one. Post-only makes
    that outcome unreachable.
    """
    venue = _venue()
    _rest(venue, Side.SELL, 18_800, 300, who=TAKER)
    opening = int(venue.account(MAKER).cash)

    # Refused, so no fee either way.
    _post_only(venue, Side.BUY, 18_900)
    assert int(venue.account(MAKER).cash) == opening

    # Rests, gets hit, earns the rebate.
    _post_only(venue, Side.BUY, 18_700)
    venue.submit(
        TAKER,
        "F",
        Submit(TAKER, Side.SELL, Quantity(50), None, OrderType.MARKET, TimeInForce.IOC),
    )
    assert int(venue.account(MAKER).cash) > opening


# --------------------------------------------------------------------------
# The venue's own account, as a measurement
# --------------------------------------------------------------------------


def test_the_venue_starts_with_nothing_so_its_balance_is_its_revenue():
    """The venue is not a participant and nobody funded it.

    Its account fell through to the same opening balance as everybody else, so
    it began with capital it had never received, and that hid the single thing
    this account exists to show. Measured on a schedule that pays out more than
    it takes: two hundred fills took the venue **930,000,000** minor units into
    the red and its own account still read 39,999,070,000,000, which is
    comfortably solvent and a measurement of nothing.
    """
    generous = FeeSchedule(taker_bps=1.0, maker_bps=-3.0)
    venue = Venue("arena", starting_cash=40_000_000, fees=generous)
    venue.list_instrument(_instrument())
    for _ in range(50):
        _rest(venue, Side.SELL, 18_600, 5)
        _cross(venue, Side.BUY, 5)

    treasury = venue.account(FEE_ACCOUNT_ID)
    assert int(treasury.starting_cash) == 0
    assert int(treasury.cash) == int(venue.fees_collected)
    assert int(treasury.cash) < 0, "a venue paying rebates on both sides looked solvent"
    assert venue.conservation_check() == 0


def test_a_venue_that_takes_more_than_it_pays_banks_exactly_that():
    """The same account read the other way round, so the change above cannot be
    a sign error that happens to look right only when the venue is losing.
    """
    venue = _venue()
    _rest(venue, Side.SELL, 18_800, 500)
    _cross(venue, Side.BUY, 300)
    treasury = venue.account(FEE_ACCOUNT_ID)
    assert int(treasury.cash) == int(venue.fees_collected) > 0
    assert venue.conservation_check() == 0


def test_the_venues_own_capital_is_not_counted_as_capital_in_the_market():
    """Total equity is what the participants brought, wherever it has since
    moved to.

    An opening balance for an account nobody funded added itself to that figure
    the moment the first fee was charged, so the market appeared to gain a
    venue's worth of capital because somebody paid a fee.
    """
    venue = _venue()
    _rest(venue, Side.SELL, 18_800, 200)
    _cross(venue, Side.BUY, 100)
    assert FEE_ACCOUNT_ID in venue.accounts, "no fee was charged, so nothing was tested"

    brought = int(venue.account(MAKER).starting_cash) + int(
        venue.account(TAKER).starting_cash
    )
    assert Decimal(venue.summary()["total_equity"]) == Decimal(brought) / 1_000_000
    assert venue.conservation_check() == 0


# --------------------------------------------------------------------------
# Where the rounding rule stops being exact
# --------------------------------------------------------------------------


def test_the_rounding_rule_is_exact_at_every_notional_this_venue_can_reach():
    """The rule is stated absolutely (a charge rounds up, a rebate rounds down
    in magnitude) and it is computed in floating point, so it has a ceiling.

    Measured, that ceiling is around 8e16 minor units of notional: at
    79,310,569,539,990,007 the taker fee came out one unit *below* the exact
    answer, which is the venue short by one. Reaching it needs roughly ten
    million lots at the top of a contract's range, against accounts that open
    with 40,000,000, so it is four orders of magnitude out of reach, and it is
    not a leak in any case, because the treasury receives exactly what the
    participants were charged whatever the rounding does.

    This pins the reachable half of that statement. If capital here ever grows
    by four orders of magnitude, this is the assertion that says the fee
    arithmetic has to come off floats first.
    """
    reachable = 10**15
    for schedule in (MAKER_TAKER, FeeSchedule(taker_bps=2.5, maker_bps=-1.5)):
        for aggressor in (True, False):
            bps = Decimal(str(schedule.rate(aggressor)))
            for notional in (1, 7, 999, 333_333, 10**9, 10**12, reachable):
                got = Decimal(int(schedule.charge(notional, aggressor)))
                exact = Decimal(notional) * bps / Decimal(10_000)
                # Toward the venue, and never by a whole unit: a charge lands
                # at or above the exact figure, a rebate at or above it too,
                # which for a negative number means smaller in magnitude.
                assert got >= exact
                assert got - exact < 1
