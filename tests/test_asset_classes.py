"""Every asset class, and the venue lifecycle around it.

The four canonical instruments -- future, event contract, spread, index -- are
only the combinations someone thought to try. The underlying algebra composes
freely, so this exercises the combinations nobody designed for: a binary written
on a spread, a spread of spreads, an index of spreads, a long/short basket, an
inverse future. Each one has to price, trade, mark, collateralise and settle
without special-casing, or the algebra is not really an algebra.

The lifecycle tests cover the second half of the question: what happens around
the edges of a contract's life. Three real bugs were found here -- a replace that
was never collateral-checked, a replace that left the venue reserving against
stale values, and working-order state that survived settlement.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest

from arena.contracts.payoff import Binary, Call, Linear, Put
from arena.contracts.spec import ContractSpec, DataPolicy, ObservationWindow
from arena.contracts.underlying import Basket, Difference, MetricRef, Single
from arena.exchange.events import Replace, Submit
from arena.exchange.types import (
    AgentId,
    OrderType,
    Quantity,
    RejectReason,
    Side,
    TimeInForce,
)
from arena.market.instrument import Instrument, InstrumentClass
from arena.market.live import HUMAN_ID
from arena.market.venue import SymbolCommand, Venue
from arena.settlement.result import SettlementResult, SettlementStatus
from arena.sim.time import Timestamp, seconds

from dashboard.build_market import build

UTC = timezone.utc
WINDOW = ObservationWindow(
    datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 9, 30, tzinfo=UTC)
)
A, B = AgentId("a"), AgentId("b")
SYM = "SYM"


def wr(subject: str, bounds=(0.0, 1.0)) -> Single:
    return Single(MetricRef("win_rate", subject, bounds=bounds))


def make(underlying, payoff, tick="0.25") -> Instrument:
    spec = ContractSpec(
        contract_id=SYM,
        underlying=underlying,
        payoff=payoff,
        window=WINDOW,
        policy=DataPolicy(min_sample_size=1),
        reference_id="ref-1",
        published_at=WINDOW.start - timedelta(days=1),
        tick_size=tick,
    )
    return Instrument(SYM, spec)


def venue_with(instrument: Instrument, cash: int = 100_000_000) -> Venue:
    venue = Venue(starting_cash=cash)
    venue.list_instrument(instrument)
    return venue


def order(agent, side, ticks, quantity) -> Submit:
    return Submit(agent, side, Quantity(quantity), ticks, OrderType.LIMIT, TimeInForce.GTC)


# --------------------------------------------------------------------------
# The algebra composes
# --------------------------------------------------------------------------

COMBINATIONS = {
    "future": (wr("A"), Linear(10_000.0), "0.25", (D("0"), D("10000"))),
    "event": (wr("A"), Binary(">", 0.5), "0.01", (D("0"), D("1"))),
    "spread": (
        Difference(wr("A"), wr("B")),
        Linear(10_000.0),
        "0.25",
        (D("-10000"), D("10000")),
    ),
    "index": (
        Basket(((wr("A"), 0.5), (wr("B"), 0.5))),
        Linear(10_000.0),
        "0.25",
        (D("0"), D("10000")),
    ),
    "event on a spread": (
        Difference(wr("A"), wr("B")),
        Binary(">", 0.0),
        "0.01",
        (D("0"), D("1")),
    ),
    "spread of spreads": (
        Difference(Difference(wr("A"), wr("B")), Difference(wr("C"), wr("D"))),
        Linear(10_000.0),
        "0.25",
        (D("-20000"), D("20000")),
    ),
    "index of spreads": (
        Basket(
            ((Difference(wr("A"), wr("B")), 0.5), (Difference(wr("C"), wr("D")), 0.5))
        ),
        Linear(10_000.0),
        "0.25",
        (D("-10000"), D("10000")),
    ),
    "long/short basket": (
        Basket(((wr("A"), 1.0), (wr("B"), -1.0))),
        Linear(10_000.0),
        "0.25",
        (D("-10000"), D("10000")),
    ),
    "inverse future": (wr("A"), Linear(-10_000.0), "0.25", (D("-10000"), D("0"))),
    "call option": (
        wr("A"),
        Call(5_000.0, 10_000.0),
        "0.25",
        (D("0"), D("5000")),
    ),
    "put option": (
        wr("A"),
        Put(5_000.0, 10_000.0),
        "0.25",
        (D("0"), D("5000")),
    ),
    "future with offset": (
        wr("A"),
        Linear(10_000.0, offset=5_000.0),
        "0.25",
        (D("5000"), D("15000")),
    ),
}


@pytest.mark.parametrize("name", sorted(COMBINATIONS))
def test_bounds_propagate_through_the_algebra(name):
    """Interval arithmetic has to survive nesting and sign changes.

    The two traps: a difference inverts its right operand, and a negative weight
    or scale flips its interval. Either mistake understates the range, and an
    understated range under-collateralises every short in the instrument.
    """
    underlying, payoff, tick, expected = COMBINATIONS[name]
    assert make(underlying, payoff, tick).settlement_bounds == expected


@pytest.mark.parametrize("name", sorted(COMBINATIONS))
def test_every_class_trades_marks_and_settles(name):
    """One mechanism, no special cases, all the way through to cash."""
    underlying, payoff, tick, (low, high) = COMBINATIONS[name]
    instrument = make(underlying, payoff, tick)
    venue = venue_with(instrument)

    # Trade at a price a quarter of the way up the range, snapped to the grid.
    raw = low + (high - low) / 4
    step = instrument.tick_size
    price = (raw / step).to_integral_value() * step
    ticks = instrument.to_ticks(price)

    venue.submit(A, SYM, order(A, Side.SELL, ticks, 10))
    events = venue.submit(B, SYM, order(B, Side.BUY, ticks, 10))

    assert any(type(e).__name__ == "Traded" for e in events), f"{name} did not trade"
    assert venue.mark_price(SYM) == price
    assert venue.account(B).positions[SYM].quantity == 10
    assert venue.account(A).positions[SYM].quantity == -10

    settlement = (high / step).to_integral_value() * step
    result = SettlementResult(
        SYM, instrument.spec.spec_digest, SettlementStatus.SETTLED, settlement, 0.0, ()
    )
    realised = venue.settle(SYM, result)

    # The long gains exactly what the short loses, and nothing leaks.
    assert int(realised[A]) == -int(realised[B])
    assert venue.conservation_check() == 0
    assert all(int(a.posted_collateral) == 0 for a in venue.accounts.values())


OPTION_STRIKE = 5_000.0
OPTION_SCALE = 10_000.0


def test_options_are_classified_and_bounded():
    call = make(wr("A"), Call(OPTION_STRIKE, OPTION_SCALE))
    put = make(wr("A"), Put(OPTION_STRIKE, OPTION_SCALE))

    assert call.instrument_class == InstrumentClass.CALL
    assert put.instrument_class == InstrumentClass.PUT
    # Floored at zero by structure, capped by the best the underlying can do.
    assert call.settlement_bounds == (D("0"), D("5000"))
    assert put.settlement_bounds == (D("0"), D("5000"))


@pytest.mark.parametrize("level", [0.0, 0.1, 0.3, 0.4669, 0.5, 0.5001, 0.75, 1.0])
def test_put_call_parity_holds_exactly(level):
    """``C - P = F - K``, with no discount factor and no approximation.

    Both legs settle from the same metric at the same instant, so parity is an
    identity here rather than a no-arbitrage relationship that holds to within
    financing costs. If it ever fails, one of the two payoffs is wrong.
    """
    call = Call(OPTION_STRIKE, OPTION_SCALE).apply(level)
    put = Put(OPTION_STRIKE, OPTION_SCALE).apply(level)
    future = Linear(OPTION_SCALE).apply(level)

    assert call - put == pytest.approx(future - OPTION_STRIKE)


def test_an_option_never_settles_negative():
    """The whole point of buying one: downside bounded by structure."""
    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert Call(OPTION_STRIKE, OPTION_SCALE).apply(level) >= 0
        assert Put(OPTION_STRIKE, OPTION_SCALE).apply(level) >= 0


@pytest.mark.parametrize("payoff_name", ["call", "put"])
def test_options_trade_and_settle(payoff_name):
    payoff = (
        Call(OPTION_STRIKE, OPTION_SCALE)
        if payoff_name == "call"
        else Put(OPTION_STRIKE, OPTION_SCALE)
    )
    instrument = make(wr("A"), payoff)
    venue = venue_with(instrument)
    ticks = instrument.to_ticks(D("400"))  # a plausible premium

    venue.submit(A, SYM, order(A, Side.SELL, ticks, 10))
    events = venue.submit(B, SYM, order(B, Side.BUY, ticks, 10))
    assert any(type(e).__name__ == "Traded" for e in events)

    # A long option's worst case is exactly the premium paid -- it expires
    # worthless, and cannot do worse than that.
    assert venue.account(B).collateral[SYM] == int(D("400") * 10 * 1_000_000)

    result = SettlementResult(
        SYM, instrument.spec.spec_digest, SettlementStatus.SETTLED, D("0"), 0.0, ()
    )
    venue.settle(SYM, result)
    assert venue.conservation_check() == 0


def test_an_option_on_a_spread_is_an_option():
    """The payoff decides the class, not the shape of what it is written on."""
    instrument = make(
        Difference(wr("A"), wr("B")), Call(0.0, OPTION_SCALE)
    )
    assert instrument.instrument_class == InstrumentClass.CALL
    # A spread ranges over [-1, 1], so a call struck at zero can pay up to 10000.
    assert instrument.settlement_bounds == (D("0"), D("10000"))


def test_negative_prices_trade_correctly():
    """A spread is genuinely negative much of the time.

    Books, marks, collateral and settlement all have to work below zero, which
    is not true of an equity venue and is easy to leave untested.
    """
    instrument = make(Difference(wr("A"), wr("B")), Linear(10_000.0))
    venue = venue_with(instrument)
    ticks = instrument.to_ticks(D("-250"))

    venue.submit(A, SYM, order(A, Side.SELL, ticks, 10))
    venue.submit(B, SYM, order(B, Side.BUY, ticks, 10))

    assert venue.mark_price(SYM) == D("-250")
    # A long at -250 on a floor of -10000 risks only the distance to the floor.
    assert venue.account(B).collateral[SYM] == int(
        D("9750") * 10 * 1_000_000
    )

    result = SettlementResult(
        SYM, instrument.spec.spec_digest, SettlementStatus.SETTLED, D("-57"), 0.0, ()
    )
    realised = venue.settle(SYM, result)
    assert int(realised[B]) > 0  # bought at -250, settled at -57
    assert venue.conservation_check() == 0


# --------------------------------------------------------------------------
# Venue lifecycle
# --------------------------------------------------------------------------


def test_replace_is_collateral_checked():
    """A modification is a request for risk, exactly like an order.

    Guarding only new orders left a hole: work ten lots, then replace them with
    five hundred at a worse price, and take on exposure the account could never
    cover.
    """
    instrument = make(wr("A"), Linear(10_000.0))
    venue = venue_with(instrument, cash=100_000)
    ack = venue.submit(A, SYM, order(A, Side.BUY, instrument.to_ticks(D("4800")), 10))[0]

    events = venue.submit(
        A, SYM, Replace(A, ack.order_id, Quantity(5000), instrument.to_ticks(D("4900")))
    )
    assert events[0].reason is RejectReason.INSUFFICIENT_COLLATERAL
    # And the original order is untouched by the refusal.
    assert list(venue._working[(A, SYM)].values())[0][1] == 10


def test_replace_updates_the_reserved_amount():
    """Stale working-order state reserves against an order that no longer exists."""
    instrument = make(wr("A"), Linear(10_000.0))
    venue = venue_with(instrument)
    ack = venue.submit(A, SYM, order(A, Side.BUY, instrument.to_ticks(D("4800")), 10))[0]

    venue.submit(
        A, SYM, Replace(A, ack.order_id, Quantity(50), instrument.to_ticks(D("4900")))
    )
    side, quantity, price = list(venue._working[(A, SYM)].values())[0]
    assert quantity == 50
    assert price == int(D("4900") * 1_000_000)


def test_a_shrinking_replace_is_still_allowed():
    instrument = make(wr("A"), Linear(10_000.0))
    venue = venue_with(instrument)
    ack = venue.submit(A, SYM, order(A, Side.BUY, instrument.to_ticks(D("4800")), 10))[0]
    events = venue.submit(
        A, SYM, Replace(A, ack.order_id, Quantity(8), instrument.to_ticks(D("4800")))
    )
    assert any(type(e).__name__ == "Replaced" for e in events)


def test_settlement_clears_working_orders():
    """Nothing can be working on a settled contract."""
    instrument = make(wr("A"), Linear(10_000.0))
    venue = venue_with(instrument)
    venue.submit(A, SYM, order(A, Side.BUY, instrument.to_ticks(D("4000")), 10))

    result = SettlementResult(
        SYM, instrument.spec.spec_digest, SettlementStatus.SETTLED, D("5000"), 0.5, ()
    )
    venue.settle(SYM, result)

    assert not venue._working.get((A, SYM))
    assert int(venue.account(A).posted_collateral) == 0
    assert venue.conservation_check() == 0


def test_a_void_returns_everyone_to_where_they_started():
    """Nobody wins or loses when the world produced no evidence."""
    instrument = make(wr("A"), Linear(10_000.0))
    venue = venue_with(instrument)
    ticks = instrument.to_ticks(D("4000"))
    venue.submit(A, SYM, order(A, Side.BUY, ticks, 10))
    venue.submit(B, SYM, order(B, Side.SELL, ticks, 10))
    venue.submit(A, SYM, order(A, Side.BUY, instrument.to_ticks(D("3000")), 5))

    result = SettlementResult(
        SYM, instrument.spec.spec_digest, SettlementStatus.VOID, None, None, (), "no data"
    )
    venue.settle(SYM, result)

    for account in venue.accounts.values():
        assert int(account.cash) == int(account.starting_cash)
        assert int(account.posted_collateral) == 0
    assert venue.conservation_check() == 0


def test_a_settled_symbol_cannot_be_traded():
    instrument = make(wr("A"), Linear(10_000.0))
    venue = venue_with(instrument)
    result = SettlementResult(
        SYM, instrument.spec.spec_digest, SettlementStatus.SETTLED, D("5000"), 0.5, ()
    )
    venue.settle(SYM, result)

    events = venue.submit(A, SYM, order(A, Side.BUY, instrument.to_ticks(D("4000")), 1))
    assert events[0].reason is RejectReason.ALREADY_TERMINAL


def test_settling_twice_is_refused():
    instrument = make(wr("A"), Linear(10_000.0))
    venue = venue_with(instrument)
    result = SettlementResult(
        SYM, instrument.spec.spec_digest, SettlementStatus.SETTLED, D("5000"), 0.5, ()
    )
    venue.settle(SYM, result)
    with pytest.raises(ValueError, match="already settled"):
        venue.settle(SYM, result)


def test_an_account_that_never_traded_is_unaffected_by_settlement():
    instrument = make(wr("A"), Linear(10_000.0))
    venue = venue_with(instrument)
    venue.account(B)
    result = SettlementResult(
        SYM, instrument.spec.spec_digest, SettlementStatus.SETTLED, D("5000"), 0.5, ()
    )
    venue.settle(SYM, result)
    assert int(venue.account(B).cash) == int(venue.account(B).starting_cash)


# --------------------------------------------------------------------------
# Commodities: an amount delivered, not a proportion
# --------------------------------------------------------------------------


def test_a_quantity_metric_makes_a_commodity_not_a_future():
    """The class comes from what the contract measures, not from its name.

    A linear claim is a future when it is written on a rate and a commodity
    when it is written on an amount delivered. The metric declares which, so
    the layer doing the classifying never has to know what a competitor is.
    """
    from dashboard.build_market import instruments

    listed = {i.symbol: i for i in instruments()}
    assert listed["RIFT_SOLO_VOL_W1"].instrument_class == "commodity"
    assert listed["EMBER_OBJECTIVE_WR"].instrument_class == "future"


def test_a_commodity_has_a_term_structure():
    """Consecutive delivery weeks, differing in nothing else.

    That is what separates a commodity from a future on the same subject: the
    amount delivered in one week is a different thing from the amount delivered
    in the next, so the four of them form a curve rather than four copies.
    """
    from dashboard.build_market import instruments

    rungs = [
        i
        for i in instruments()
        if i.symbol.startswith("RIFT_SOLO_VOL_W") and "_GT" not in i.symbol
    ]
    assert len(rungs) >= 3

    windows = [(i.spec.window.start, i.spec.window.end) for i in rungs]
    assert len(set(windows)) == len(windows), "two rungs measure the same week"
    # Consecutive and non-overlapping: each week begins where the last ended.
    for (_, end), (start, _) in zip(sorted(windows), sorted(windows)[1:]):
        assert end == start, "the delivery weeks leave a gap or overlap"

    # Everything except the window is identical, or they are not one curve.
    payoffs = {i.spec.payoff.to_dict()["kind"] for i in rungs}
    subjects = {i.spec.underlying.ref.subject for i in rungs}
    assert payoffs == {"linear"} and len(subjects) == 1


def test_the_delivery_weeks_settle_at_different_amounts():
    """A curve that is flat by construction would prove nothing."""
    from dashboard.build_market import instruments, true_values

    values = true_values(instruments())
    curve = [values[f"RIFT_SOLO_VOL_W{n}"] for n in (1, 2, 3, 4)]
    assert len(set(curve)) > 1, f"the term structure is flat: {curve}"


def test_a_metric_must_declare_whether_it_is_a_rate_or_a_quantity():
    """The distinction decides the asset class, so a typo must not pass."""
    from arena.contracts.underlying import MetricRef

    with pytest.raises(ValueError, match="must be 'rate', 'quantity' or"):
        MetricRef(metric="match_volume", subject="RIFT", kind="amount")


def test_exactly_one_metric_is_a_quantity_and_it_is_the_commodity():
    """The whole asset class rests on one field of one metric declaration.

    A commodity is a linear claim on an amount delivered, and nothing decides
    which claims those are except `METRIC_KINDS`. Every other metric in this
    world is scale invariant in the length of the window, so a March contract
    and an April contract on it are the same contract measured twice; only
    `match_volume` doubles when the window doubles, and that is what gives the
    four delivery weeks a term structure rather than four copies of one number.

    Asserted as a count rather than as a lookup, because the failure this
    guards against is a second metric quietly acquiring the kind, which a
    lookup on the one name would not see.
    """
    from arena.worlds.circuit.metrics import METRIC_KINDS, METRICS

    quantities = [name for name, kind in METRIC_KINDS.items() if kind == "quantity"]
    assert quantities == ["match_volume"]
    assert set(METRIC_KINDS) == set(METRICS)
    assert METRICS["match_volume"].__doc__ is not None


# --------------------------------------------------------------------------
# Shares: a claim that pays before it settles
# --------------------------------------------------------------------------


def test_a_schedule_of_payments_makes_a_share_not_a_future():
    """Paying as it goes is the whole difference, so it decides the class."""
    from dashboard.build_market import instruments

    listed = {i.symbol: i for i in instruments()}
    assert listed["BASTION_SOLO_EQ"].instrument_class == "equity"
    assert listed["BASTION_SOLO_WR"].instrument_class == "future"
    # Written on the same metric, so nothing but the stream separates them.
    assert (
        listed["BASTION_SOLO_EQ"].spec.underlying.to_dict()
        == listed["BASTION_SOLO_WR"].spec.underlying.to_dict()
    )


def test_a_share_settles_at_nothing_and_is_worth_something_anyway():
    """The value is the stream, which is exactly what makes it a share.

    A contract valued only at its settlement would price this at zero, and be
    right about the last instant and wrong about every other one.
    """
    from dashboard.build_market import instruments, true_values

    listed = {i.symbol: i for i in instruments()}
    share = listed["BASTION_SOLO_EQ"]
    low, high = share.settlement_bounds
    assert low == high == 0

    values = true_values(list(listed.values()))
    assert values["BASTION_SOLO_EQ"] > 0
    # And its range covers the payments, or a short could owe more than it
    # posted.
    low, high = share.value_bounds
    assert float(share.from_ticks(int(values["BASTION_SOLO_EQ"]))) <= float(high)


def test_a_share_is_a_scaled_claim_on_the_same_thing_as_its_future():
    """Four weekly payments of 1,000 against one payment of 10,000.

    If the level were flat the share would be worth exactly 0.4 of the future,
    which is what makes the pair worth listing: any gap between them is either
    a mispricing or the price of getting collateral back early, and both are
    findings.
    """
    from dashboard.build_market import instruments, true_values

    values = true_values(instruments())
    ratio = values["BASTION_SOLO_EQ"] / values["BASTION_SOLO_WR"]
    # Not exactly 0.4: each week is measured on its own evidence, so the four
    # weekly rates do not average to the four-week rate once each is rounded to
    # the tick grid. Re-measured on this listing the ratio is 0.400205, two
    # hundredths of a percent out, which is that rounding.
    assert abs(ratio - 0.4) < 0.001, ratio


def test_paying_a_distribution_moves_cash_and_conserves_it():
    """Longs receive, shorts pay, and nothing is created."""
    from arena.portfolio.money import Money

    from arena.exchange.session import SessionState

    market = build(seed=7, human_cash=4_000_000)
    market.kernel.start()

    # Bought when the share is actually trading. At a fixed moment it may be
    # mid-auction -- either the opening call or a breaker pause -- and an order
    # into a halted book rests until the uncross rather than filling.
    symbol = "BASTION_SOLO_EQ"
    for moment in range(20, 300, 5):
        market.kernel.advance(until=seconds(moment))
        book = market.venue.engine(symbol).book.snapshot()
        if (
            market.venue.session(symbol) is SessionState.CONTINUOUS
            and book.best_ask is not None
        ):
            break
    # Retried, because seeing an offer and reaching it are different events.
    #
    # An order enqueued here travels the same latency link as an algorithm's,
    # and a maker is free to pull in between -- so a fill-or-cancel market
    # order that was aimed at a real offer arrives to an empty side and cancels
    # itself. Acknowledged, then Cancelled, with no position. That is the
    # market working, not a fault, and it began failing here only because
    # listing nineteen more contracts moved the trajectory at this seed.
    #
    # Holding the share is *setup* for this test; what is under test is that
    # `distribute` moves cash and conserves it. So the setup retries rather
    # than the assertion loosening.
    venue = market.venue
    position = None
    for _ in range(12):
        market.human.enqueue(
            SymbolCommand(
                symbol,
                Submit(
                    HUMAN_ID, Side.BUY, Quantity(40), None, OrderType.MARKET, TimeInForce.IOC
                ),
            )
        )
        market.kernel.advance(until=Timestamp(int(market.kernel.now) + int(seconds(2))))
        position = venue.account(HUMAN_ID).positions.get(symbol)
        if position is not None and position.quantity > 0:
            break

    assert position is not None and position.quantity > 0, (
        "twelve marketable orders in a row found no offer to take, which is a "
        "market with no sell side rather than a race"
    )

    before = int(venue.account(HUMAN_ID).cash)
    assert int(venue.conservation_check()) == 0
    moved = int(venue.distribute("BASTION_SOLO_EQ", Money(123 * 1_000_000)))
    assert moved > 0
    # The human was paid its share of the stream, to the unit.
    after = int(venue.account(HUMAN_ID).cash)
    assert after - before == position.quantity * 123 * 1_000_000
    # And it came from somewhere: value moved, none appeared.
    assert int(venue.conservation_check()) == 0


def test_a_distribution_leaves_no_account_short_of_collateral():
    """Paying what you owe cannot be what makes you insolvent.

    A short pays cash out, and the claim it is short is worth exactly that much
    less afterwards, so its requirement has to fall in the same instant. If the
    range it collateralises against did not narrow, meeting an obligation it
    always had would look like a margin breach.

    There is one exception, and it is worth stating exactly rather than
    exempting. A requirement cannot fall below zero. An account that sold the
    share *above* the ceiling which survives the payment has nothing left to
    give back: its requirement is already at the floor, the cash leaves anyway,
    and its headroom falls by precisely the part of the payment the requirement
    could not absorb.

    Those accounts used to carry a *negative* requirement instead, and since
    `posted_collateral` sums that dict, the credit was quietly funding
    positions on other underlyings. That is cross-underlying netting arriving
    through the back door, which is the one thing collateral here may never do
    Netting two different competitors assumes they move together, and a
    correlation is an estimate. Flooring the requirement at zero is what closed
    it, and this is the visible consequence.

    The live sweep finds whoever happens to be holding the share at t=40s, and
    which accounts those are depends on the whole market's composition. The
    exception itself is constructed below rather than waited for, which is why
    nothing here asserts a count of floored shorts.
    """
    market = build(seed=7)
    market.kernel.start()
    market.kernel.advance(until=seconds(40))

    venue = market.venue
    holders = [
        agent
        for agent, account in venue.accounts.items()
        if account.positions.get("BASTION_SOLO_EQ") and account.positions["BASTION_SOLO_EQ"].quantity
    ]
    assert holders, "nobody is holding the share; the test proves nothing"

    from arena.portfolio.money import Money

    # The change, not the level. An agent can be fully invested for reasons
    # that have nothing to do with this payment -- it is holding twenty-six
    # contracts -- and asserting a positive balance would be testing how much
    # capital the fixture happens to hand out. What must hold is that meeting
    # an obligation does not make anyone worse off: the cash goes out and the
    # requirement falls by exactly as much.
    payment = 123
    before = {
        agent: (
            int(venue.accounts[agent].free_cash),
            int(venue.accounts[agent].collateral.get("BASTION_SOLO_EQ", Money(0))),
            venue.accounts[agent].positions["BASTION_SOLO_EQ"].quantity,
        )
        for agent in holders
    }
    venue.distribute("BASTION_SOLO_EQ", Money(payment * 1_000_000))

    floored = 0
    for agent in holders:
        account = venue.accounts[agent]
        cash_before, required_before, quantity = before[agent]
        lost = cash_before - int(account.free_cash)
        if lost <= 1:
            continue
        # Headroom may only fall where the requirement had already run out of
        # room to fall, and then by exactly the remainder. Anything else is the
        # payment being charged twice.
        floored += 1
        assert int(account.collateral.get("BASTION_SOLO_EQ", Money(0))) == 0, (
            f"{agent} lost {lost} of headroom while its requirement could "
            "still have fallen"
        )
        assert lost == abs(quantity) * payment * 1_000_000 - required_before, (
            f"{agent} lost {lost} of headroom, against a shortfall of "
            f"{abs(quantity) * payment * 1_000_000 - required_before}"
        )

    # The live sweep above proves the rule for whoever happens to be holding.
    # The exception -- a requirement already at zero, so the payment has to come
    # out of headroom -- is then constructed rather than waited for.
    #
    # It used to be waited for, and that was a fixture depending on a bug.
    # `VenueAgent.top_of_book` was publishing the market-on-open sentinel, so
    # some account reliably ended up short the share at an absurd average, and
    # this test read that accident as its scenario. Fixing the feed removed the
    # accident and the test began reporting, accurately, that it was no longer
    # exercising the path it names. Measured: it passes on `best_price` and
    # fails on `best_priced`, which is the wrong way round for a test of
    # collateral arithmetic to depend on a market-data defect.
    instrument = venue.registry.require("BASTION_SOLO_EQ")
    _low, high = venue.bounds_in_minor(instrument)
    venue.open_account("floor-probe", 10_000_000)
    probe = venue.accounts["floor-probe"]
    # Short at the very top of what the claim can settle for, so the next
    # payment drops the ceiling below the basis and the requirement floors.
    probe.apply_fill(
        "BASTION_SOLO_EQ", -20, Money(int(high)), venue.bounds_in_minor(instrument)
    )
    cash_before = int(probe.free_cash)
    required_before = int(probe.collateral.get("BASTION_SOLO_EQ", Money(0)))
    venue.distribute("BASTION_SOLO_EQ", Money(payment * 1_000_000))

    assert int(probe.collateral.get("BASTION_SOLO_EQ", Money(0))) == 0, (
        "the constructed short was not pinned to the floor, so the exception "
        "is still untested"
    )
    lost = cash_before - int(probe.free_cash)
    assert lost == 20 * payment * 1_000_000 - required_before, (
        f"the floored short lost {lost} of headroom, against a shortfall of "
        f"{20 * payment * 1_000_000 - required_before}"
    )
    # Deliberately no conservation check on the probe. The position was booked
    # onto one account with no counterparty, which breaks the identity by
    # construction rather than by any fault in the ledger, and asserting it
    # here would be asserting that the fixture is a market. The conserving
    # path is covered by every test that trades through the venue.


def test_a_share_cannot_pay_more_than_it_promised():
    """Past the last scheduled payment there is nothing left to pay from."""
    from arena.portfolio.money import Money

    market = build(seed=7)
    venue = market.venue
    for _ in range(4):
        venue.distribute("BASTION_SOLO_EQ", Money(100 * 1_000_000))
    with pytest.raises(ValueError, match="already paid all 4"):
        venue.distribute("BASTION_SOLO_EQ", Money(100 * 1_000_000))


def test_only_a_share_can_pay_a_distribution():
    """A payment nobody collateralised is a payment nobody can make."""
    from arena.portfolio.money import Money

    market = build(seed=7)
    with pytest.raises(ValueError, match="declares no distribution schedule"):
        market.venue.distribute("EMBER_OBJECTIVE_WR", Money(1_000_000))


def test_the_payments_differ_from_week_to_week():
    """A flat stream would prove the windows were never measured separately."""
    from dashboard.build_market import _world, instruments
    from arena.settlement.engine import distributions

    oracle = _world()
    share = {i.symbol: i for i in instruments()}["BASTION_SOLO_EQ"]
    paid = distributions(share.spec, oracle)
    assert len(paid) == 4
    assert len(set(paid)) > 1, f"every week paid the same: {paid}"


def test_distribution_windows_must_lie_inside_the_contract_window():
    """A payment measured outside the window is measured on other evidence."""
    from datetime import UTC, datetime, timedelta

    from arena.contracts.payoff import Linear
    from arena.contracts.spec import (
        ContractSpec,
        DataPolicy,
        DistributionSchedule,
        ObservationWindow,
    )
    from arena.worlds.circuit.metrics import metric_ref
    from arena.contracts.underlying import Single

    window = ObservationWindow(
        datetime(2026, 8, 31, tzinfo=UTC), datetime(2026, 9, 7, tzinfo=UTC)
    )
    outside = ObservationWindow(window.start, window.end + timedelta(days=7))
    with pytest.raises(ValueError, match="outside the observation window"):
        ContractSpec(
            contract_id="BAD",
            underlying=Single(metric_ref("win_rate", "QUILL", modes=("solo",))),
            payoff=Linear(0.0),
            window=window,
            policy=DataPolicy(min_sample_size=1),
            reference_id="ref",
            published_at=window.start - timedelta(days=1),
            distribution=DistributionSchedule(windows=(outside,), payoff=Linear(1.0)),
        )


def test_overlapping_payment_windows_are_refused():
    """Overlapping periods would pay for the same battles twice."""
    from datetime import UTC, datetime

    from arena.contracts.payoff import Linear
    from arena.contracts.spec import DistributionSchedule, ObservationWindow

    first = ObservationWindow(
        datetime(2026, 8, 31, tzinfo=UTC), datetime(2026, 9, 10, tzinfo=UTC)
    )
    second = ObservationWindow(
        datetime(2026, 9, 7, tzinfo=UTC), datetime(2026, 9, 14, tzinfo=UTC)
    )
    with pytest.raises(ValueError, match="overlap"):
        DistributionSchedule(windows=(first, second), payoff=Linear(1.0))


# --------------------------------------------------------------------------
# Volatility: a claim on the second moment
# --------------------------------------------------------------------------


def test_a_dispersion_metric_makes_a_volatility_contract():
    """The class comes from which moment the metric reports."""
    from dashboard.build_market import instruments

    listed = {i.symbol: i for i in instruments()}
    assert listed["QUILL_SOLO_DISP"].instrument_class == "volatility"
    assert listed["EMBER_OBJECTIVE_WR"].instrument_class == "future"


def test_dispersion_and_level_are_different_questions_about_the_same_subject():
    """How well a competitor does and how evenly are not one number twice.

    Re-measured on the circuit listing, and the gap moved the other way round
    from the world this replaces. QUILL and BASTION settle 37.4% apart on their
    individual win rate and only 8.9% apart on placement dispersion, a factor
    of 4.2, where the retired pair sat within two percent on the level and
    nearly seventy percent apart on the spread.

    Either arrangement makes the same point, which is that the second moment
    does not follow the first through a fixed scale. It is worth being precise
    about how much this shows on its own: two subjects cannot separate a
    correlation from an identity, and the listing measures that separately at a
    residual of 0.0068 after a straight line in the win rate. What this asserts
    is the part visible from the two contracts the exchange actually lists.
    """
    from dashboard.build_market import instruments, true_values

    values = true_values(instruments())
    level_gap = (
        abs(values["QUILL_SOLO_WR"] - values["BASTION_SOLO_WR"])
        / values["BASTION_SOLO_WR"]
    )
    spread_gap = (
        abs(values["QUILL_SOLO_DISP"] - values["BASTION_SOLO_DISP"])
        / values["BASTION_SOLO_DISP"]
    )
    assert level_gap > 0.3, f"the two levels are not far apart: {level_gap:.2%}"
    assert spread_gap < 0.15, f"the two dispersions are not close: {spread_gap:.2%}"
    assert level_gap > 3 * spread_gap, (
        f"level gap {level_gap:.2%} against spread gap {spread_gap:.2%}: the "
        "two moments are moving together on one scale"
    )


def test_dispersion_is_bounded_so_collateral_stays_arithmetic():
    """A normalized placement lives in [0, 1], so its spread cannot exceed 0.5.

    That bound is not decoration. It is what lets a second-moment claim join an
    exchange whose whole collateral model is "every contract settles inside a
    known interval" -- without it, a short would need a variance estimate
    rather than a subtraction. The ceiling is reached only by a competitor that
    finishes first in half its matches and last in the other half, which is
    arithmetic rather than an observed maximum.
    """
    from arena.worlds.circuit.metrics import METRIC_BOUNDS

    assert METRIC_BOUNDS["placement_dispersion"] == (0.0, 0.5)

    from dashboard.build_market import instruments, true_values

    catalogue = instruments()
    listed = {i.symbol: i for i in catalogue}
    values = true_values(catalogue)
    for symbol in ("QUILL_SOLO_DISP", "BASTION_SOLO_DISP"):
        low, high = listed[symbol].tick_bounds
        assert int(low) <= values[symbol] <= int(high)


def test_dispersion_measures_the_same_matches_as_the_level():
    """Same window, same format, same appearances, a different moment.

    If it ran over different matches it would be a different measurement
    wearing the same contract's name, and the two would not be comparable. Both
    metrics start from `_appearances`, so the check is that the evidence trail
    they publish agrees rather than that the two happen to look similar.

    Measured on the prior window of seed 7: QUILL is drawn into 6,640 of the
    8,064 solo matches, wins 490 of them for a rate of 0.0738, and finishes
    with a placement dispersion of 0.3140 around a mean normalized placement of
    0.5638. One set of matches, two moments of it.
    """
    from arena.worlds.circuit.metrics import METRICS, metric_ref
    from dashboard.build_market import PRIOR_WINDOW, _world

    oracle = _world()
    level = oracle.resolve(
        metric_ref("win_rate", "QUILL", modes=("solo",)), PRIOR_WINDOW
    )
    spread = oracle.resolve(
        metric_ref("placement_dispersion", "QUILL", modes=("solo",)), PRIOR_WINDOW
    )
    assert "placement_dispersion" in METRICS
    assert level.sample_size == spread.sample_size == 6_640
    level_diagnostics = dict(level.diagnostics)
    spread_diagnostics = dict(spread.diagnostics)
    assert level_diagnostics["appearances"] == spread_diagnostics["appearances"]
    assert level_diagnostics["formats"] == spread_diagnostics["formats"] == "solo"
    assert level.value == pytest.approx(0.0738, abs=0.0001)
    assert spread.value == pytest.approx(0.3140, abs=0.0001)


def test_an_unknown_metric_kind_is_refused():
    from arena.contracts.underlying import MetricRef

    with pytest.raises(ValueError, match="must be 'rate', 'quantity' or 'dispersion'"):
        MetricRef(metric="placement_dispersion", subject="QUILL", kind="moment")



# --------------------------------------------------------------------------
# The clock a contract's window is measured against
# --------------------------------------------------------------------------


def test_a_live_market_reaches_expiry_and_settles():
    """Nothing ever settled in a running server, and that was the largest gap.

    There are two clocks here and they were not connected. The kernel counts
    simulated nanoseconds from zero; a contract's window closes on a calendar
    date. `Venue._enforce_lifecycle` asks `self._clock() >= instrument.expiry`,
    and in the live market `_clock` was `None`, so the question was never
    asked. Wiring it to a wall clock would not have helped either: the kernel
    would have had to run for a month of real time to reach the date.

    Measured before the calendar existed: after a simulated hour all 47 of the
    contracts then listed were still `continuous` and the settled set was
    empty. A position
    was marked forever and realised never, which leaves an algorithm no
    terminal event to score itself against.

    The settlement machinery was complete and heavily tested the whole time.
    `build_market.prior_levels` has always called `settle(spec, oracle)`
    successfully on every listing, which is the proof the oracle can answer.
    Nothing in the live path ever asked it.

    The mapping is the one the venue already used rather than a new one:
    `build_market` scales the breaker windows by `session_seconds / trading
    day`, so `session_seconds` of simulated time is already one trading day.

    The run is 600 simulated seconds and used to be 340, and the extra four
    minutes are the listing moving rather than the market getting slower. The
    contract calendar starts on publication day, which is the day before the
    circuit opens, and the observation window no longer opens there: it opens
    four weeks in, after the prior window every informed trader reads its
    opening belief off. So a four week contract now expires 57 trading days out
    rather than 29, which at a ten second day is 570 simulated seconds instead
    of 290.
    """
    market = build(seed=7, session_seconds=10.0)
    market.kernel.start()
    venue = market.venue

    assert market.calendar is not None, "the live market has no contract clock"
    opened = market.calendar.now()

    settled_at = {}
    for moment in range(20, 620, 20):
        market.kernel.advance(until=seconds(moment))
        market.calendar.advance_to(moment)
        for symbol in market.settle_due():
            settled_at.setdefault(symbol, moment)
        assert int(venue.conservation_check()) == 0, f"leaked at t={moment}"

    assert market.calendar.now() > opened, "the contract calendar never moved"

    listed = set(venue.registry.symbols)
    assert set(venue.settled_symbols) == listed, (
        f"{len(listed) - len(venue.settled_symbols)} contracts never settled"
    )

    # The weeklies expire first, and that ordering is the check that the
    # calendar is being compared against each contract's *own* window rather
    # than against one exchange-wide deadline.
    weekly = {s for s in listed if s.endswith(("_W1", "_W2"))}
    monthly = {s for s in listed if s.endswith("_WR")}
    assert weekly and monthly, "the fixture no longer lists both tenors"
    assert max(settled_at[s] for s in weekly) < max(settled_at[s] for s in monthly), (
        "a one-week contract outlived a four-week one"
    )


def test_nothing_trades_after_its_window_closes():
    """The outcome is determined, so anyone still trading knows the answer.

    This is the rule `_enforce_lifecycle` exists for, and until the live venue
    had a clock it could not fire there at all.
    """
    market = build(seed=7, session_seconds=10.0)
    market.kernel.start()
    for moment in range(20, 620, 20):
        market.kernel.advance(until=seconds(moment))
        market.calendar.advance_to(moment)
        market.settle_due()

    from arena.exchange.session import SessionState

    venue = market.venue
    for symbol in venue.registry.symbols:
        assert venue.session(symbol) is SessionState.CLOSED, (
            f"{symbol} is still open after its window closed"
        )
    assert int(venue.conservation_check()) == 0
