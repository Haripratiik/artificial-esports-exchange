"""Position, account, and venue accounting.

The load-bearing test is ``test_value_is_conserved_across_random_trading``.
Trading moves value between participants; it does not create it. If total equity
across a closed market ever drifts from total starting capital, there is a leak
in the accounting and every PnL figure the project ever reports is wrong. It is
the sharpest single check available on this layer.

Second in importance is the position *flip*. A trade taking a position from +10
to -5 must close 10 and open 5, not blend a long and a short cost basis into a
number with no meaning.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from arena.contracts.payoff import Binary, Linear
from arena.contracts.spec import ContractSpec, DataPolicy, ObservationWindow
from arena.contracts.underlying import Difference, MetricRef, Single
from arena.exchange.events import Submit
from arena.exchange.types import AgentId, OrderType, Price, Quantity, RejectReason, Side, TimeInForce
from arena.market.instrument import Instrument, InstrumentClass
from arena.market.venue import Venue
from arena.portfolio.account import Account
from arena.portfolio.money import Money, from_money, to_money as M
from arena.portfolio.position import Position
from arena.settlement.result import SettlementResult, SettlementStatus

UTC = timezone.utc
D = Decimal


def make_spec(contract_id="C", payoff=None, underlying=None, tick="0.25") -> ContractSpec:
    window = ObservationWindow(
        datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 9, 30, tzinfo=UTC)
    )
    return ContractSpec(
        contract_id=contract_id,
        underlying=underlying or Single(MetricRef("win_rate", "EMBER")),
        payoff=payoff or Linear(scale=10_000.0),
        window=window,
        policy=DataPolicy(min_sample_size=1),
        reference_id="ref-1",
        published_at=window.start - timedelta(days=1),
        tick_size=tick,
    )


def future(symbol="EMBER_FUT") -> Instrument:
    return Instrument(symbol=symbol, spec=make_spec(symbol))


# --------------------------------------------------------------------------
# Settlement bounds: what makes collateral exact
# --------------------------------------------------------------------------


def test_linear_future_bounds_follow_the_metric_range():
    """A rate in [0,1] scaled by 10000 settles in [0, 10000]."""
    assert make_spec().settlement_bounds == (D("0"), D("10000"))


def test_binary_bounds_are_zero_to_payout():
    spec = make_spec(payoff=Binary(">", 0.55, payout=1.0), tick="0.01")
    assert spec.settlement_bounds == (D("0"), D("1"))


def test_spread_bounds_account_for_interval_subtraction():
    """A difference of two [0,1] rates ranges over [-1, 1], not [0, 0].

    Using (lo-lo, hi-hi) is the classic interval-arithmetic mistake and would
    report a spread as having no range at all, so a short would be allowed with
    zero collateral.
    """
    spread = Difference(
        Single(MetricRef("win_rate", "EMBER")),
        Single(MetricRef("win_rate", "RIFT")),
    )
    spec = make_spec(underlying=spread, payoff=Linear(scale=10_000.0))
    assert spec.settlement_bounds == (D("-10000"), D("10000"))


def test_negative_scale_flips_the_interval():
    spec = make_spec(payoff=Linear(scale=-10_000.0))
    low, high = spec.settlement_bounds
    assert low == D("-10000")
    assert high == D("0")


def test_collateral_is_the_exact_worst_case():
    """A short at 5100 on a [0, 10000] contract can lose exactly 4900 per lot."""
    spec = make_spec()
    assert spec.collateral_for(10, D("5100")) == D("51000")   # long: loses all of it
    assert spec.collateral_for(-10, D("5100")) == D("49000")  # short: to the top


def test_collateral_is_never_negative():
    """A position that cannot lose is charged nothing, not a credit.

    A short above the top of a claim's range makes money whatever happens, and
    the arithmetic ran straight past zero into negative territory: measured, a
    short of one lot at 12,000 on a contract bounded by [0, 10000] returned
    -2,000. That figure is *summed* with the rest of a portfolio's requirement,
    so a negative one does not merely misreport, it funds another position.
    """
    spec = make_spec()
    assert spec.collateral_for(-1, D("12000")) == D(0)
    assert spec.collateral_for(1, D("-500")) == D(0)

    bounds = (M("0"), M("10000"))
    assert Account.collateral_required(-1, M("12000"), bounds) == M("0")
    assert Account.collateral_required(1, M("-500"), bounds) == M("0")

    account = Account("a", M("100000"))
    account.apply_fill("X", -1, M("12000"), bounds)
    assert account.posted_collateral == M("0")
    assert account.free_cash == account.cash


# --------------------------------------------------------------------------
# Position accounting
# --------------------------------------------------------------------------


def test_average_price_is_volume_weighted():
    p = Position("X")
    p.apply_fill(10, M("100"))
    p.apply_fill(10, M("120"))
    assert p.quantity == 20
    assert p.average_price == D("110")


def test_closing_realises_against_the_average():
    p = Position("X")
    p.apply_fill(10, M("100"))
    record = p.apply_fill(-4, M("130"))

    assert record.realized == M("120")     # 4 lots * 30
    assert p.realized_pnl == M("120")
    assert p.quantity == 6
    assert p.average_price == D("100")     # untouched by a partial close


def test_shorts_profit_when_the_price_falls():
    p = Position("X")
    p.apply_fill(-10, M("100"))
    assert p.unrealized_pnl(M("90")) == M("100")
    assert p.unrealized_pnl(M("110")) == M("-100")


def test_a_flip_closes_then_opens():
    """The case that silently corrupts a naive ledger.

    +10 at 100, then sell 15 at 130: ten lots close and realise 300, and five
    lots open short with a basis of 130, not a blend of 100 and 130, which
    would be a cost basis for a position that never existed.
    """
    p = Position("X")
    p.apply_fill(10, M("100"))
    record = p.apply_fill(-15, M("130"))

    assert record.flipped is True
    assert record.closed == 10
    assert p.realized_pnl == M("300")
    assert p.quantity == -5
    assert p.average_price == D("130")
    # And the new short is marked from its own basis, not the old one.
    assert p.unrealized_pnl(M("130")) == M("0")


def test_closing_exactly_flat_resets_the_basis():
    p = Position("X")
    p.apply_fill(10, M("100"))
    p.apply_fill(-10, M("110"))
    assert p.is_flat
    assert p.average_price == D("0")
    assert p.realized_pnl == M("100")


def test_fees_reduce_realised_pnl():
    p = Position("X")
    p.apply_fill(10, M("100"), fee=M("5"))
    assert p.realized_pnl == M("-5")
    assert p.fees_paid == M("5")


def test_zero_quantity_fill_is_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        Position("X").apply_fill(0, M("100"))


# --------------------------------------------------------------------------
# Account and collateral
# --------------------------------------------------------------------------


def test_notional_is_not_debited_from_cash():
    """A future is a collateralised commitment, not a purchase.

    Debiting quantity * price would report a trader as broke the instant they
    opened a large position, which is spot accounting applied to a derivative.
    """
    account = Account("a", M("100000"))
    account.apply_fill("X", 10, M("5000"), (M("0"), M("10000")))

    assert account.cash == M("100000")            # unchanged
    assert account.posted_collateral == M("50000")  # but committed
    assert account.free_cash == M("50000")


def test_collateral_is_charged_against_the_basis_not_a_floored_average():
    """The average is a division and the basis is not.

    Two fills at different prices leave a basis that is not divisible by the
    quantity, and `apply_fill` used to derive ``cost_basis // quantity`` and
    charge against that. Floor division rounds a long's average *down*, so the
    requirement came out under the loss by whatever the basis left over.
    Measured on seven lots bought as three at 10.25 and four at 11.50: a basis
    of 76,750,000 minor units against 76,749,995 posted, five short. Under a
    minor unit a lot, and not zero, and the claim on this module is that its
    figures are exact rather than close.
    """
    account = Account("a", M("1000000"))
    bounds = (M("0"), M("10000"))
    account.apply_fill("X", 3, M("10.25"), bounds)
    account.apply_fill("X", 4, M("11.50"), bounds)

    position = account.position("X")
    # A long can lose everything it paid, down to the bottom of the range.
    assert int(account.collateral["X"]) == int(position.cost_basis)
    assert int(position.cost_basis) % position.quantity != 0  # the case that rounded

    # And the mirror: a short's exposure is the top of the range less its basis.
    short = Account("b", M("1000000"))
    short.apply_fill("Y", -3, M("10.25"), bounds)
    short.apply_fill("Y", -4, M("11.50"), bounds)
    seven_to_the_top = 7 * int(M("10000")) + int(short.position("Y").cost_basis)
    assert int(short.collateral["Y"]) == seven_to_the_top


def test_a_partial_close_recharges_the_remainder_exactly():
    """Collateral has to follow the basis left behind, remainder included.

    A proportional close leaves an integer remainder inside the position on
    purpose (that is what keeps the ledger conserving) and the requirement on
    what is left must be computed from that basis rather than reconstructed
    from an average, or the two disagree by the remainder.
    """
    account = Account("a", M("1000000"))
    bounds = (M("0"), M("10000"))
    account.apply_fill("X", 7, M("10.25"), bounds)
    account.apply_fill("X", -2, M("11.00"), bounds)

    position = account.position("X")
    assert position.quantity == 5
    assert int(account.collateral["X"]) == int(position.cost_basis)

    account.apply_fill("X", -5, M("11.00"), bounds)
    assert "X" not in account.collateral
    assert account.posted_collateral == M("0")


def test_collateral_is_released_when_a_position_closes():
    account = Account("a", M("100000"))
    bounds = (M("0"), M("10000"))
    account.apply_fill("X", 10, M("5000"), bounds)
    account.apply_fill("X", -10, M("5100"), bounds)

    assert account.posted_collateral == M("0")
    assert account.cash == M("101000")   # 10 lots * 100
    assert account.free_cash == M("101000")


def test_an_account_cannot_exceed_its_collateral():
    account = Account("a", M("10000"))
    bounds = (M("0"), M("10000"))
    assert account.can_afford("X", 2, M("5000"), bounds) is True
    assert account.can_afford("X", 3, M("5000"), bounds) is False


def test_adding_at_a_lower_price_cannot_outrun_the_cash():
    """The check has to price the position the fill creates, from its basis.

    Ten lots long at 5,000 is 50,000 of exposure. Ten more at 100 was checked
    as ``20 * 100 = 2,000``, because the resulting quantity was priced at the
    incoming trade rather than at what the position had actually paid. An
    account holding exactly 50,000 passed that check, came out with a basis of
    51,000, posted more collateral than it owned (`free_cash` at -1,000) and
    stood to owe a thousand it did not have if the contract settled at zero.
    """
    account = Account("a", M("50000"))
    bounds = (M("0"), M("10000"))
    account.apply_fill("X", 10, M("5000"), bounds)
    assert account.free_cash == M("0")

    assert account.can_afford("X", 10, M("100"), bounds) is False

    # The invariant the check exists to hold: whatever gets through, posted
    # collateral never exceeds the cash backing it.
    for quantity, price in ((1, M("100")), (4, M("2")), (10, M("100"))):
        if account.can_afford("X", quantity, price, bounds):
            account.apply_fill("X", quantity, price, bounds)
    assert int(account.free_cash) >= 0
    assert int(account.cash) + int(account.position("X").unrealized_pnl(M("0"))) >= 0


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8])
def test_the_projected_basis_matches_the_applied_one(seed):
    """`basis_after` duplicates `apply_fill`'s branches, so pin them together.

    Opening, adding, partial closes that leave an integer remainder, exact
    closes and flips: the projection has to agree on all five or the solvency
    check is answering about a position the ledger will not produce.
    """
    rng = random.Random(seed)
    position = Position("X")
    for _ in range(200):
        quantity = rng.choice([-9, -7, -3, -1, 1, 2, 5, 11])
        price = M(D(rng.randrange(1, 400)) * D("0.25"))
        projected = position.basis_after(quantity, price)
        position.apply_fill(quantity, price)
        assert int(position.cost_basis) == int(projected)


def test_reducing_a_position_is_always_affordable():
    """Otherwise an agent could be trapped in a losing position it cannot exit."""
    account = Account("a", M("50000"))
    bounds = (M("0"), M("10000"))
    account.apply_fill("X", 10, M("5000"), bounds)
    assert account.free_cash == M("0")
    assert account.can_afford("X", -5, M("4000"), bounds) is True


def test_settlement_realises_and_frees_collateral():
    account = Account("a", M("100000"))
    bounds = (M("0"), M("10000"))
    account.apply_fill("X", 10, M("5000"), bounds)
    realised = account.settle("X", M("5500"))

    assert realised == M("5000")
    assert account.cash == M("105000")
    assert account.posted_collateral == M("0")
    assert account.position("X").is_flat


def test_settling_twice_is_refused():
    """An expiry firing more than once is a plausible bug in any event system."""
    account = Account("a", M("100000"))
    account.apply_fill("X", 10, M("5000"), (M("0"), M("10000")))
    account.settle("X", M("5500"))
    with pytest.raises(ValueError, match="already settled"):
        account.settle("X", M("5500"))


def test_a_share_pays_down_its_range_and_closes_out_to_nothing():
    """The newest class, and the only one whose cash moves before settlement.

    Four payments of 467 against a claim that settles at nothing: the long
    receives, the short pays, and neither books a profit for it. What the two
    sides hold has to sum to exactly zero at every step (after each payment and
    after settlement) and both positions must end flat with no collateral and
    no residual basis. A share's terminal payoff is Linear(0), so its
    settlement bounds are [0, 0] and every guard downstream of settlement is
    vacuous on it; this is the check that it closes out anyway.
    """
    long, short = Account("l", M("1000000")), Account("s", M("1000000"))
    bounds = (M("0"), M("4000"))
    for account, quantity in ((long, 10), (short, -10)):
        account.apply_fill("EQ", quantity, M("1869"), bounds)

    paid = 0
    for _ in range(4):
        free_before = (int(long.free_cash), int(short.free_cash))
        paid += int(M("467"))
        remaining = (Money(int(bounds[0]) - paid), Money(int(bounds[1]) - paid))
        received = long.distribute("EQ", M("467"), remaining)
        owed = short.distribute("EQ", M("467"), remaining)
        assert int(received) + int(owed) == 0
        # A payment cannot make either side insolvent, and now exactly so: the
        # long's requirement rises by precisely the cash it received and the
        # short's falls by precisely the cash it paid, so free cash does not
        # move at all. Charged against a floored average this held only to
        # within the remainder the division threw away.
        assert (int(long.free_cash), int(short.free_cash)) == free_before
        # A payment moves cash and lowers the claim by the same amount, so the
        # short's requirement falls by exactly what it just handed over.
        assert int(short.collateral["EQ"]) == 10 * (4_000_000_000 - paid) - int(
            M("18690")
        )
        assert int(long.cash) + int(short.cash) == 2 * int(M("1000000"))

    # Nothing is left: it has all been paid out.
    for account in (long, short):
        account.settle("EQ", Money(int(M("0")) - paid))
        position = account.position("EQ")
        assert position.is_flat
        assert int(position.cost_basis) == 0
        assert account.posted_collateral == M("0")

    assert int(long.cash) + int(short.cash) == 2 * int(M("1000000"))
    assert int(long.equity({})) + int(short.equity({})) == 2 * int(M("1000000"))


def test_a_void_returns_collateral_without_realising():
    account = Account("a", M("100000"))
    account.apply_fill("X", 10, M("5000"), (M("0"), M("10000")))
    account.void("X")

    assert account.cash == M("100000")
    assert account.posted_collateral == M("0")
    assert account.position("X").is_flat


def test_equity_does_not_double_count_realised_pnl():
    """Cash already contains realised PnL; adding it again doubles the profit."""
    account = Account("a", M("100000"))
    bounds = (M("0"), M("10000"))
    account.apply_fill("X", 10, M("5000"), bounds)
    account.apply_fill("X", -10, M("5100"), bounds)

    assert account.realized_pnl == M("1000")
    assert account.equity({}) == M("101000")


# --------------------------------------------------------------------------
# The venue: the join that was missing
# --------------------------------------------------------------------------


def order(agent, side, price, qty, tif=TimeInForce.GTC) -> Submit:
    return Submit(agent, side, Quantity(qty), price, OrderType.LIMIT, tif)


def test_a_trade_becomes_a_position():
    """The loop that did not previously close: order -> fill -> position."""
    venue = Venue(starting_cash=D("1000000"))
    instrument = future()
    venue.list_instrument(instrument)
    maker, taker = AgentId("maker"), AgentId("taker")

    ticks = instrument.to_ticks(D("5000"))
    venue.submit(maker, "EMBER_FUT", order(maker, Side.SELL, ticks, 10))
    venue.submit(taker, "EMBER_FUT", order(taker, Side.BUY, ticks, 10))

    assert venue.account(taker).position("EMBER_FUT").quantity == 10
    assert venue.account(maker).position("EMBER_FUT").quantity == -10
    assert venue.account(taker).position("EMBER_FUT").average_price == D("5000")


def test_settlement_flows_all_the_way_to_pnl():
    """Define a contract, trade it, settle it, and see the cash move."""
    venue = Venue(starting_cash=D("1000000"))
    instrument = future()
    venue.list_instrument(instrument)
    maker, taker = AgentId("maker"), AgentId("taker")

    ticks = instrument.to_ticks(D("5000"))
    venue.submit(maker, "EMBER_FUT", order(maker, Side.SELL, ticks, 10))
    venue.submit(taker, "EMBER_FUT", order(taker, Side.BUY, ticks, 10))

    result = SettlementResult(
        contract_id="EMBER_FUT",
        spec_digest=instrument.spec.spec_digest,
        status=SettlementStatus.SETTLED,
        settlement_value=D("5500"),
        underlying_level=0.55,
        resolutions=(),
    )
    realised = venue.settle("EMBER_FUT", result)

    assert realised[taker] == M("5000")
    assert realised[maker] == M("-5000")
    assert venue.account(taker).cash == M("1005000")
    assert venue.account(maker).cash == M("995000")
    assert venue.conservation_check() == 0


def test_a_settlement_for_a_different_contract_is_refused():
    """The contract that settles must be the contract that traded."""
    venue = Venue()
    instrument = future()
    venue.list_instrument(instrument)
    result = SettlementResult(
        contract_id="EMBER_FUT",
        spec_digest="sha256:something-else",
        status=SettlementStatus.SETTLED,
        settlement_value=D("5500"),
        underlying_level=0.55,
        resolutions=(),
    )
    with pytest.raises(ValueError, match="not the contract that traded"):
        venue.settle("EMBER_FUT", result)


def test_orders_beyond_collateral_are_rejected_before_they_reach_the_book():
    """An exchange cannot unprint a trade it should not have allowed."""
    venue = Venue(starting_cash=D("10000"))
    instrument = future()
    venue.list_instrument(instrument)
    buyer = AgentId("buyer")

    events = venue.submit(
        buyer, "EMBER_FUT", order(buyer, Side.BUY, instrument.to_ticks(D("5000")), 100)
    )
    assert events[0].reason is RejectReason.INSUFFICIENT_COLLATERAL
    assert venue.engine("EMBER_FUT").book.snapshot().best_bid is None


def test_an_order_resting_in_one_book_is_paid_for_before_another_is_accepted():
    """Collateral is committed when an order is acknowledged, not when it fills.

    The affordability check charged the working orders in the symbol it was
    asked about and nothing resting anywhere else, so an account could park an
    order in one book and then spend the money backing it in another. Both were
    individually affordable. Whichever filled first took the cash, and the
    second left the account holding collateral it did not have.

    That is a failure in time rather than a simultaneous one, which is why a
    per-symbol scenario could not see it: the order is affordable when it is
    accepted, and the account is poorer by the time it fills. Measured on the
    live market at seed 17 and 90 simulated seconds, `fund-1` rested a sell of
    fifteen lots on a call, went on trading forty-five other contracts while it
    sat there, and on filling needed 77,977,500,000 minor units of collateral
    against free cash of 29,020,881,304, finishing at -31,486,674,402.

    Asserted against a control rather than against a number: the refused order
    is accepted on its own on an account of the same size, so what refuses it
    here is the order resting in the other book and nothing else.
    """
    venue = Venue(starting_cash=D("10000"))
    first, second = future("EMBER_FUT"), future("RIFT_FUT")
    venue.list_instrument(first)
    venue.list_instrument(second)
    who = AgentId("both-books")

    # A long of one lot at 6,000 on a contract that can settle at zero can lose
    # 6,000, which is most of what this account holds.
    parked = venue.submit(
        who, "EMBER_FUT", order(who, Side.BUY, first.to_ticks(D("6000")), 1)
    )
    assert getattr(parked[0], "reason", None) is None, parked
    assert venue._reserved[who] == int(M(D("6000")))

    blocked = venue.submit(
        who, "RIFT_FUT", order(who, Side.BUY, second.to_ticks(D("5000")), 1)
    )
    assert blocked[0].reason is RejectReason.INSUFFICIENT_COLLATERAL
    assert venue.engine("RIFT_FUT").book.snapshot().best_bid is None

    control = Venue(starting_cash=D("10000"))
    control.list_instrument(future("RIFT_FUT"))
    alone = control.submit(
        who, "RIFT_FUT", order(who, Side.BUY, second.to_ticks(D("5000")), 1)
    )
    assert getattr(alone[0], "reason", None) is None, alone


def test_cancelling_a_resting_order_hands_its_capital_back():
    """A reserve that only ever grew would be a slow leak with a hard stop.

    The account would refuse its own orders on the strength of exposure it no
    longer has, and nothing would ever say so: the collateral is not posted, so
    the ledger looks right and only the refusals are wrong.
    """
    venue = Venue(starting_cash=D("10000"))
    first, second = future("EMBER_FUT"), future("RIFT_FUT")
    venue.list_instrument(first)
    venue.list_instrument(second)
    who = AgentId("both-books")

    parked = venue.submit(
        who, "EMBER_FUT", order(who, Side.BUY, first.to_ticks(D("6000")), 1)
    )[0]
    refused = venue.submit(
        who, "RIFT_FUT", order(who, Side.BUY, second.to_ticks(D("5000")), 1)
    )
    assert refused[0].reason is RejectReason.INSUFFICIENT_COLLATERAL

    from arena.exchange.events import Cancel

    venue.submit(who, "EMBER_FUT", Cancel(who, parked.order_id))
    assert who not in venue._reserved

    accepted = venue.submit(
        who, "RIFT_FUT", order(who, Side.BUY, second.to_ticks(D("5000")), 1)
    )
    assert getattr(accepted[0], "reason", None) is None, accepted
    assert venue._reserved[who] == int(M(D("5000")))
    assert venue.conservation_check() == 0


def test_trading_stops_at_the_close_but_cancels_still_work():
    venue = Venue()
    instrument = future()
    venue.list_instrument(instrument)
    agent = AgentId("a")
    ticks = instrument.to_ticks(D("5000"))

    ack = venue.submit(agent, "EMBER_FUT", order(agent, Side.BUY, ticks, 1))[0]
    venue.close("EMBER_FUT")

    blocked = venue.submit(agent, "EMBER_FUT", order(agent, Side.BUY, ticks, 1))
    assert blocked[0].reason is RejectReason.ALREADY_TERMINAL

    from arena.exchange.events import Cancel

    tidied = venue.submit(agent, "EMBER_FUT", Cancel(agent, ack.order_id))
    assert not any(getattr(e, "reason", None) for e in tidied)


def test_relisting_a_symbol_is_refused():
    venue = Venue()
    venue.list_instrument(future())
    with pytest.raises(ValueError, match="already listed"):
        venue.list_instrument(future())


def test_off_grid_prices_are_rejected_not_rounded():
    """Rounding would fill an agent at a price it never chose."""
    instrument = future()
    with pytest.raises(ValueError, match="not a multiple"):
        instrument.to_ticks(D("5000.10"))   # tick is 0.25


def test_instrument_class_is_derived_from_the_contract():
    assert future().instrument_class == InstrumentClass.FUTURE

    binary = Instrument(
        "EMBER_GT55",
        make_spec("EMBER_GT55", payoff=Binary(">", 0.55), tick="0.01"),
    )
    assert binary.instrument_class == InstrumentClass.EVENT

    spread = Instrument(
        "EMBER_RIFT",
        make_spec(
            "EMBER_RIFT",
            underlying=Difference(
                Single(MetricRef("win_rate", "EMBER")),
                Single(MetricRef("win_rate", "RIFT")),
            ),
        ),
    )
    assert spread.instrument_class == InstrumentClass.SPREAD


def test_an_untraded_instrument_marks_at_the_middle_of_its_range():
    """Marking it at zero would report every short as instantly profitable."""
    venue = Venue()
    venue.list_instrument(future())
    assert venue.mark("EMBER_FUT") == M("5000")


# --------------------------------------------------------------------------
# Conservation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8, 13])
def test_value_is_conserved_across_random_trading(seed):
    """Trading moves value between participants; it does not create it.

    If total equity ever drifts from total starting capital, there is a leak in
    the accounting and every PnL figure the project reports is wrong.
    """
    rng = random.Random(seed)
    venue = Venue(starting_cash=D("1000000"))
    instrument = future()
    venue.list_instrument(instrument)
    agents = [AgentId(f"a{i}") for i in range(6)]
    for agent in agents:
        venue.account(agent)

    for _ in range(600):
        agent = rng.choice(agents)
        side = rng.choice([Side.BUY, Side.SELL])
        price = instrument.to_ticks(D(rng.randrange(4800, 5200, 25)) / 4 * 4)
        venue.submit(
            agent,
            "EMBER_FUT",
            order(agent, side, price, rng.randint(1, 8)),
        )

    assert venue.conservation_check() == 0


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_value_is_conserved_through_settlement(seed):
    """Including the moment positions turn into cash."""
    rng = random.Random(seed)
    venue = Venue(starting_cash=D("1000000"))
    instrument = future()
    venue.list_instrument(instrument)
    agents = [AgentId(f"a{i}") for i in range(5)]
    for agent in agents:
        venue.account(agent)

    for _ in range(300):
        agent = rng.choice(agents)
        side = rng.choice([Side.BUY, Side.SELL])
        price = instrument.to_ticks(D(rng.randrange(19200, 20800, 1)) * D("0.25"))
        venue.submit(agent, "EMBER_FUT", order(agent, side, price, rng.randint(1, 5)))

    result = SettlementResult(
        contract_id="EMBER_FUT",
        spec_digest=instrument.spec.spec_digest,
        status=SettlementStatus.SETTLED,
        settlement_value=D("5137.25"),
        underlying_level=0.513725,
        resolutions=(),
    )
    venue.settle("EMBER_FUT", result)
    assert venue.conservation_check() == 0
    assert all(a.posted_collateral == 0 for a in venue.accounts.values())


def test_multiple_instruments_trade_side_by_side():
    """A venue, not a book: several assets with independent order books."""
    venue = Venue(starting_cash=D("1000000"))
    fut = future("EMBER_FUT")
    binary = Instrument(
        "EMBER_GT55", make_spec("EMBER_GT55", payoff=Binary(">", 0.55), tick="0.01")
    )
    venue.list_instrument(fut)
    venue.list_instrument(binary)

    a, b = AgentId("a"), AgentId("b")
    venue.submit(a, "EMBER_FUT", order(a, Side.SELL, fut.to_ticks(D("5000")), 4))
    venue.submit(b, "EMBER_FUT", order(b, Side.BUY, fut.to_ticks(D("5000")), 4))
    venue.submit(a, "EMBER_GT55", order(a, Side.SELL, binary.to_ticks(D("0.60")), 20))
    venue.submit(b, "EMBER_GT55", order(b, Side.BUY, binary.to_ticks(D("0.60")), 20))

    assert venue.registry.symbols == ("EMBER_FUT", "EMBER_GT55")
    assert venue.account(b).position("EMBER_FUT").quantity == 4
    assert venue.account(b).position("EMBER_GT55").quantity == 20
    # Independent books: the binary's collateral is tiny next to the future's.
    assert venue.account(b).collateral["EMBER_GT55"] == M("12")
    assert venue.conservation_check() == 0
