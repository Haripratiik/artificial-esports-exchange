"""The scoring rule, and the venue built on it.

The properties checked here are the ones the mechanism is *chosen* for. If any
of them fails the comparison against the order book is meaningless, because the
thing being compared would not be an LMSR market.

The bounded-loss test is the important one. It is the reason a scoring-rule
market can be run at all. The venue is knowingly spending money to make a
market, and it can only do that if the amount is knowable in advance. The test
searches adversarially over trade paths rather than checking a formula against
itself.
"""

from __future__ import annotations

import math
import random
from decimal import Decimal

import pytest

from arena.market.lmsr import LN2, LmsrMarket, liquidity_for_subsidy


def _market(subsidy: float = 100.0, payout: float = 1.0) -> LmsrMarket:
    return LmsrMarket(liquidity_for_subsidy(subsidy, payout), payout=payout)


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------


def test_an_untraded_market_prices_the_outcome_at_even_odds():
    """No shares outstanding means no information, so the price is one half."""
    assert _market().price == pytest.approx(0.5)


def test_buying_raises_the_price_and_selling_lowers_it():
    market = _market()
    market.apply(500)
    raised = market.price
    assert raised > 0.5
    market.apply(-1_000)
    assert market.price < 0.5 < raised


def test_price_stays_inside_the_payout_range_under_extreme_flow():
    """A price outside [0, payout] would be an arbitrage against the venue."""
    market = _market()
    for quantity in (10**4, 10**6, -(10**7)):
        market.apply(quantity)
        assert 0.0 <= market.price <= market.payout


def test_cost_is_path_independent():
    """Splitting an order cannot change what it costs.

    If it could, an agent would manufacture profit by slicing, and every
    measurement of informed trading would be contaminated by execution games.
    """
    direct = _market()
    direct_cost = direct.apply(900)

    sliced = _market()
    rng = random.Random(3)
    remaining, sliced_cost = 900, 0.0
    while remaining > 0:
        lot = min(remaining, rng.randint(1, 120))
        sliced_cost += sliced.apply(lot)
        remaining -= lot

    assert sliced_cost == pytest.approx(direct_cost, rel=1e-12)
    assert sliced.net == direct.net


def test_a_round_trip_is_exactly_free():
    """Buy then sell the same size returns precisely what it cost.

    This surprises people who expect an AMM to charge a spread, but it is the
    direct consequence of path independence: cost depends only on where ``net``
    started and ended, and a round trip ends where it started. Raw LMSR has **no
    bid-ask spread**.

    It is also why real scoring-rule venues charge an explicit fee, and why the
    venue built on this one quantises prices to the tick grid in the maker's
    favour: that rounding is where this market's spread actually comes from,
    and it is a property of the venue rather than of the rule.
    """
    market = _market()
    paid = market.apply(400)
    received = -market.apply(-400)
    assert received == pytest.approx(paid, rel=1e-12)
    assert market.net == 0


def test_average_price_brackets_the_marginal_prices():
    """What a trade pays sits between the price before it and after it."""
    market = _market()
    before = market.price
    average = market.average_price(300)
    after = market.price_at(market.net + 300)
    assert before < average < after


def test_the_price_inverse_round_trips():
    market = _market()
    for target in (0.05, 0.25, 0.5, 0.75, 0.95):
        net = market.shares_for_price(target)
        assert market.price_at(net) == pytest.approx(target, abs=1e-9)


def test_a_bigger_subsidy_makes_the_market_deeper():
    """More money at risk buys less price impact per share. That is the trade."""
    impacts = []
    for subsidy in (50.0, 500.0):
        market = _market(subsidy)
        market.apply(200)
        impacts.append(market.price - 0.5)
    assert impacts[1] < impacts[0]


@pytest.mark.parametrize("shares_per_tick", [10.0, 40.0, 250.0])
@pytest.mark.parametrize("tick,payout", [(0.01, 1.0), (0.25, 100.0)])
def test_the_depth_calibration_delivers_the_depth_it_promises(
    shares_per_tick, tick, payout
):
    """The knob that makes a comparison against an order book mean anything.

    Asking for N shares at the touch has to actually produce N shares at the
    touch, or the two venues in Experiment 2 are being compared at different
    depths and the mechanism is not what is being measured.
    """
    from arena.market.lmsr import subsidy_for_depth

    subsidy = subsidy_for_depth(shares_per_tick, tick, payout)
    market = LmsrMarket(liquidity_for_subsidy(subsidy, payout), payout=payout)
    # Shares needed to move the marginal price one tick from even odds.
    moved = market.shares_for_price(payout / 2 + tick) - market.net
    assert moved == pytest.approx(shares_per_tick, rel=0.02)


# --------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------


def test_the_subsidy_is_exactly_the_worst_case_loss():
    for subsidy in (10.0, 250.0, 5_000.0):
        assert _market(subsidy).bounded_loss == pytest.approx(subsidy)


@pytest.mark.parametrize("payout", [1.0, 250.0])
def test_the_maker_never_loses_more_than_its_subsidy(payout):
    """Searched over random paths and both outcomes, not derived from the formula.

    The maker's profit is what traders paid minus what it owes the winning side.
    Nothing here may drive that below ``-subsidy``.
    """
    subsidy = 400.0
    rng = random.Random(17)
    worst = 0.0
    for _ in range(300):
        market = _market(subsidy, payout)
        received = 0.0
        for _ in range(rng.randint(1, 40)):
            quantity = rng.randint(-3_000, 3_000)
            if quantity:
                received += market.apply(quantity)
        # Traders hold `net` shares. If the event happens the maker pays
        # `net * payout`; if it does not, it pays nothing.
        for owed in (market.net * payout, 0.0):
            worst = min(worst, received - owed)
    assert worst >= -subsidy - 1e-6, f"lost {-worst:.4f} against a {subsidy} subsidy"


def test_liquidity_and_subsidy_are_inverses():
    b = liquidity_for_subsidy(750.0, payout=4.0)
    assert 4.0 * b * LN2 == pytest.approx(750.0)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_market_cannot_be_made_without_a_subsidy(bad):
    with pytest.raises(ValueError):
        liquidity_for_subsidy(bad, payout=1.0)


def test_a_zero_trade_is_rejected():
    with pytest.raises(ValueError):
        _market().apply(0)


def test_extreme_positions_do_not_overflow():
    """The stable softplus matters: exp(q/b) overflows long before this."""
    market = LmsrMarket(liquidity=1.0, payout=1.0)
    assert math.isfinite(market.cost(10_000))
    assert math.isfinite(market.cost(-10_000))
    market.apply(10_000)
    assert market.price == pytest.approx(1.0)


# --------------------------------------------------------------------------
# The venue
# --------------------------------------------------------------------------

from datetime import datetime, timedelta, timezone  # noqa: E402

from arena.contracts.payoff import Binary, Linear  # noqa: E402
from arena.contracts.spec import ContractSpec, DataPolicy, ObservationWindow  # noqa: E402
from arena.contracts.underlying import Single  # noqa: E402
from arena.exchange.events import Cancelled, Filled, Rejected, Submit  # noqa: E402
from arena.exchange.types import (  # noqa: E402
    AgentId,
    OrderType,
    Quantity,
    Side,
    TimeInForce,
)
from arena.market.instrument import Instrument  # noqa: E402
from arena.market.lmsr_venue import LMSR_MAKER_ID, LmsrVenue  # noqa: E402
from arena.portfolio.money import Money, from_money, to_money  # noqa: E402
from arena.worlds.circuit.metrics import metric_ref  # noqa: E402

UTC = timezone.utc
START = datetime(2026, 8, 31, tzinfo=UTC)
TRADER = AgentId("trader-1")


def _spec(payoff, tick: str = "0.01") -> ContractSpec:
    return ContractSpec(
        contract_id="B",
        underlying=Single(metric_ref("win_rate", "SUBJECT")),
        payoff=payoff,
        window=ObservationWindow(START, START + timedelta(days=28)),
        policy=DataPolicy(
            min_sample_size=1_000, min_stratum_battles=200, min_strata_coverage=0.80
        ),
        reference_id="ref-2026S09-v1",
        published_at=START - timedelta(days=1),
        tick_size=tick,
    )


def _venue(subsidy: float = 2_000.0) -> tuple[LmsrVenue, Instrument]:
    instrument = Instrument("B", _spec(Binary(">", 0.5, payout=1.0)))
    venue = LmsrVenue(starting_cash=1_000_000, subsidy=subsidy)
    venue.list_instrument(instrument)
    return venue, instrument


def _order(side: Side, quantity: int, price=None, tif=TimeInForce.IOC) -> Submit:
    return Submit(
        TRADER,
        side,
        Quantity(quantity),
        price,
        OrderType.MARKET if price is None else OrderType.LIMIT,
        tif,
    )


def _fill(events) -> Filled | None:
    return next((e for e in events if isinstance(e, Filled)), None)


def test_a_market_order_always_fills():
    """The defining property. A book cannot promise this; a scoring rule can."""
    venue, _ = _venue()
    for side in (Side.BUY, Side.SELL, Side.BUY):
        fill = _fill(venue.submit(TRADER, "B", _order(side, 5_000)))
        assert fill is not None and int(fill.quantity) == 5_000


def test_price_responds_to_flow_in_the_right_direction():
    venue, _ = _venue()
    opening = venue.mark_price("B")
    venue.submit(TRADER, "B", _order(Side.BUY, 800))
    higher = venue.mark_price("B")
    venue.submit(TRADER, "B", _order(Side.SELL, 1_600))
    assert higher > opening > venue.mark_price("B")


def test_value_is_conserved_exactly_through_random_flow():
    """The sharpest check on the whole venue, and it must be zero rather than small."""
    venue, _ = _venue()
    rng = random.Random(23)
    for _ in range(200):
        side = rng.choice([Side.BUY, Side.SELL])
        venue.submit(TRADER, "B", _order(side, rng.randint(1, 400)))
        assert venue.conservation_check() == 0


def test_the_traders_and_the_maker_hold_opposite_positions():
    venue, _ = _venue()
    rng = random.Random(5)
    for _ in range(40):
        venue.submit(
            TRADER, "B", _order(rng.choice([Side.BUY, Side.SELL]), rng.randint(1, 200))
        )
    trader = venue.account(TRADER).positions["B"].quantity
    maker = venue.account(LMSR_MAKER_ID).positions["B"].quantity
    assert trader == -maker != 0
    assert venue.engine("B").market.net == trader


def test_rounding_never_favours_the_trader():
    """Quantisation is what creates the spread, and it must go one way only.

    A buy books at or above the rule's average cost and a sell at or below it.
    Rounding the other way would leak the maker's bounded loss one tick at a
    time, and the guarantee would stop being a guarantee.
    """
    venue, instrument = _venue()
    rng = random.Random(11)
    tick = float(instrument.tick_size)
    for _ in range(120):
        side = rng.choice([Side.BUY, Side.SELL])
        quantity = rng.randint(1, 300)
        market = venue.engine("B").market
        signed = quantity if side is Side.BUY else -quantity
        exact = market.average_price(signed) / tick
        fill = _fill(venue.submit(TRADER, "B", _order(side, quantity)))
        assert fill is not None
        if side is Side.BUY:
            assert int(fill.price) >= math.floor(exact)
        else:
            assert int(fill.price) <= math.ceil(exact)


def test_a_limit_order_is_never_filled_through_its_price():
    venue, _ = _venue()
    venue.submit(TRADER, "B", _order(Side.BUY, 1_500))
    rng = random.Random(7)
    for _ in range(60):
        side = rng.choice([Side.BUY, Side.SELL])
        limit = rng.randint(30, 80)
        fill = _fill(venue.submit(TRADER, "B", _order(side, 200, limit)))
        if fill is None:
            continue
        if side is Side.BUY:
            assert int(fill.price) <= limit
        else:
            assert int(fill.price) >= limit


def test_an_unmarketable_limit_rests_nowhere_and_is_cancelled():
    """Nothing queues here, so an order that cannot trade now never trades."""
    venue, _ = _venue()
    events = venue.submit(TRADER, "B", _order(Side.BUY, 100, 5, TimeInForce.GTC))
    assert _fill(events) is None
    assert any(isinstance(e, Cancelled) for e in events)
    assert venue.engine("B").market.net == 0


def test_fill_or_kill_is_all_or_nothing():
    venue, _ = _venue()
    # A price one tick through the touch can supply only part of a large order.
    ask = int(venue.engine("B").book.best_price(Side.SELL))
    events = venue.submit(
        TRADER, "B", _order(Side.BUY, 100_000, ask, TimeInForce.FOK)
    )
    assert _fill(events) is None
    assert venue.engine("B").market.net == 0


def test_the_synthetic_book_is_never_crossed():
    venue, _ = _venue()
    rng = random.Random(31)
    for _ in range(80):
        venue.submit(
            TRADER, "B", _order(rng.choice([Side.BUY, Side.SELL]), rng.randint(1, 500))
        )
        snapshot = venue.engine("B").book.snapshot(6)
        if snapshot.best_bid is not None and snapshot.best_ask is not None:
            assert int(snapshot.best_bid) < int(snapshot.best_ask)
        for ladder in (snapshot.bids, snapshot.asks):
            assert all(int(q) > 0 for _p, q in ladder)


def test_the_ladder_matches_the_curve_it_claims_to_show():
    """Depth at the touch really is what it takes to move the price one tick."""
    venue, _ = _venue()
    book = venue.engine("B").book
    ask = book.best_price(Side.SELL)
    size = int(book.depth_at(Side.SELL, ask))
    assert size > 0
    before = venue.engine("B").market.price
    venue.submit(TRADER, "B", _order(Side.BUY, size))
    after = venue.engine("B").market.price
    # One tick of movement, give or take the rounding at the boundary.
    assert 0 < (after - before) <= 2 * 0.01


def test_a_non_binary_contract_cannot_be_listed():
    """A rule that prices two outcomes must not be asked to price a continuum."""
    venue = LmsrVenue(subsidy=1_000)
    with pytest.raises(ValueError, match="binary scoring rule"):
        venue.list_instrument(Instrument("F", _spec(Linear(10_000.0), tick="0.25")))


def test_the_maker_is_funded_to_exactly_its_bound():
    venue, _ = _venue(subsidy=3_000.0)
    assert from_money(venue.account(LMSR_MAKER_ID).starting_cash) == Decimal("3000")
    assert venue.engine("B").market.bounded_loss == pytest.approx(3_000.0)


def test_the_maker_never_loses_more_than_the_subsidy_in_a_real_session():
    """The bound, checked end to end through the ledger rather than the formula."""
    subsidy = 2_000.0
    venue, instrument = _venue(subsidy)
    rng = random.Random(19)
    for _ in range(150):
        venue.submit(
            TRADER, "B", _order(rng.choice([Side.BUY, Side.SELL]), rng.randint(1, 300))
        )
    maker = venue.account(LMSR_MAKER_ID)
    held = maker.positions["B"].quantity
    basis = int(maker.positions["B"].cost_basis)
    # Mark the maker's book at each outcome and take the worse one.
    worst = min(
        int(maker.cash) + held * int(to_money(Decimal(str(value)))) - basis
        for value in (Decimal("0"), Decimal("1"))
    )
    assert float(from_money(Money(worst))) >= -subsidy - 1.0


def test_a_settled_contract_pays_out_and_still_conserves():
    """Settlement is the inherited path, so this checks the two venues agree."""
    from arena.settlement.result import SettlementResult, SettlementStatus

    venue, instrument = _venue()
    venue.submit(TRADER, "B", _order(Side.BUY, 900))
    venue.close("B")
    venue.settle(
        "B",
        SettlementResult(
            contract_id="B",
            spec_digest=instrument.spec.spec_digest,
            status=SettlementStatus.SETTLED,
            settlement_value=Decimal("1"),
            underlying_level=0.7,
            resolutions=(),
        ),
    )
    assert venue.conservation_check() == 0
    assert venue.account(TRADER).positions.get("B") is None or (
        venue.account(TRADER).positions["B"].quantity == 0
    )
