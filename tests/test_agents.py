"""Trading agents, and the market they make together.

Two of these are regression tests for bugs that only appeared once a real
market was run, and both are the dangerous kind: they produce plausible numbers
rather than crashes.

``test_messages_on_a_link_never_overtake`` guards the kernel: jittered latency
was allowed to reorder messages on the same link, so a fill could arrive before
the acknowledgement of the order that produced it.

``test_agent_position_matches_the_venue`` guards the consequence: an agent that
saw a fill for an order it had not been told about dropped the position update
entirely, and the late acknowledgement then left a phantom order working
forever. The agent believed it held +7 while the venue held 0.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from arena.agents.fundamental import FundamentalTrader
from arena.market.instrument import Instrument
from arena.agents.market_maker import MarketMaker
from arena.agents.noise import NoiseTrader
from arena.exchange.events import Submit
from arena.exchange.session import SessionState
from arena.exchange.types import AgentId, OrderType, Quantity, Side, TimeInForce
from arena.market.live import HUMAN_ID
from arena.portfolio.money import from_money
from arena.market.venue import SymbolCommand
from arena.sim.kernel import Kernel
from arena.sim.latency import PairwiseLatency
from arena.sim.time import Duration, Timestamp, micros, millis, seconds

from dashboard.build_market import build, instruments as build_instruments

SYMBOL = "EMBER_OBJECTIVE_WR"

# The first and last subject of the listing's own `SUBJECTS` tuple, on the
# individual format, and the format is the load-bearing half of that choice.
#
# A win rate contract opens at the midpoint of its range, which is 5,000, and
# the team format's neutral point is 0.500 exactly. So every team win rate
# future opens within a few hundred points of where it settles and has almost
# nothing to discover: measured on seed 17, BASTION_OBJECTIVE_WR opens 0.0 from
# its settlement, EMBER_OBJECTIVE_WR 38.2 and VANTA_OBJECTIVE_WR 106.5, against
# a tolerance of 1,000, so the test would pass or fail on which side of the
# answer the session happened to wander. The individual format's neutral point
# is 0.100, so its futures open 3,096 to 4,389 away from settlement and there is
# a real distance for the market to cover.
DISCOVERY = [("VANTA_SOLO_WR", 0.10), ("RIFT_SOLO_WR", 0.10)]


def _maker_position(market, symbol: str = SYMBOL) -> int:
    """The makers' net position, across all of them."""
    total = 0
    for agent_id in ("mm-1", "mm-2", "mm-3"):
        position = market.venue.account(agent_id).positions.get(symbol)
        if position is not None:
            total += int(position.quantity)
    return total


def _trading(market, symbol: str, until: int = 240, levels: int = 1) -> None:
    """Advance until ``symbol`` is trading two-sided, or give up at ``until``.

    A market with a circuit breaker spends part of its session halted, and the
    opening minutes are when it happens most: measured on seed 7, nine of
    twenty-six symbols are in an auction at sixty seconds and one to four are
    from three minutes on. A test that submits an order at a fixed moment is
    testing what the breaker happened to be doing.

    This is what a person hits too. An order into a halted book rests until the
    uncross rather than trading, which is the mechanism working.
    """
    from arena.exchange.session import SessionState

    moment = int(market.kernel.now // 1_000_000_000) + 5
    while moment <= until:
        market.kernel.advance(until=seconds(moment))
        engine = market.venue.engine(symbol)
        book = engine.book.snapshot()
        if (
            market.venue.session(symbol) is SessionState.CONTINUOUS
            and book.best_bid is not None
            and book.best_ask is not None
            and _reachable(engine, book.best_ask)
            and len(book.priced_asks) >= levels
        ):
            return
        moment += 5
    raise AssertionError(f"{symbol} never traded two-sided by t={until}")


def _reachable(engine, price) -> bool:
    """Whether a market order could actually reach this price.

    A collar stops an unpriced order at the edge of the band, so in a fast
    market (and this one is fast, because its evidence arrives over the
    session) the whole offer can sit beyond where a market order is allowed to
    go, and such an order fills nothing at all. That is the protection working,
    and it is why anyone in a hurry uses a limit order. A test that sweeps a
    book has to start from a moment when the book is sweepable.
    """
    band = engine.execution_band
    return band is None or band[0] <= int(price) <= band[1]



@pytest.fixture(scope="module")
def market():
    m = build(seed=17)
    m.kernel.start()
    m.kernel.advance(until=seconds(90))
    return m


# --------------------------------------------------------------------------
# Kernel: per-link ordering
# --------------------------------------------------------------------------


def test_messages_on_a_link_never_overtake():
    """An exchange session is a stream; a fill cannot precede its own ack.

    Independent per-message jitter models a datagram network, which is the
    wrong model for order entry and produced a real, silent divergence between
    an agent's position and the venue's.
    """
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class Sink:
        agent_id: AgentId = AgentId("sink")
        arrivals: list[tuple[int, int]] = field(default_factory=list)

        def on_start(self, ctx): pass
        def on_wakeup(self, ctx): pass
        def on_finish(self, ctx): pass
        def on_message(self, ctx, sender, message):
            self.arrivals.append((int(ctx.now), message))

    @dataclass
    class Spammer:
        agent_id: AgentId = AgentId("spammer")

        def on_start(self, ctx):
            for i in range(200):
                ctx.send(AgentId("sink"), i)

        def on_wakeup(self, ctx): pass
        def on_finish(self, ctx): pass
        def on_message(self, ctx, sender, message): pass

    kernel = Kernel(
        seed=5,
        latency=PairwiseLatency(default=millis(10), jitter_fraction=0.9, seed=5),
    )
    sink = Sink()
    kernel.add_all([sink, Spammer()])
    kernel.run(until=seconds(5))

    payloads = [m for _t, m in sink.arrivals]
    assert payloads == sorted(payloads), "messages overtook each other on one link"
    stamps = [t for t, _m in sink.arrivals]
    assert stamps == sorted(stamps)


def test_jitter_still_varies_delivery():
    """Ordering is preserved, but latency must not become constant."""
    from dataclasses import dataclass, field

    @dataclass
    class Sink:
        agent_id: AgentId = AgentId("sink")
        stamps: list[int] = field(default_factory=list)

        def on_start(self, ctx): pass
        def on_wakeup(self, ctx): pass
        def on_finish(self, ctx): pass
        def on_message(self, ctx, sender, message): self.stamps.append(int(ctx.now))

    @dataclass
    class Pinger:
        agent_id: AgentId = AgentId("pinger")
        sent: int = 0

        def on_start(self, ctx): ctx.request_wakeup(millis(100))
        def on_wakeup(self, ctx):
            self.sent += 1
            ctx.send(AgentId("sink"), self.sent)
            if self.sent < 30:
                ctx.request_wakeup(millis(100))
        def on_finish(self, ctx): pass
        def on_message(self, ctx, sender, message): pass

    kernel = Kernel(
        seed=5,
        latency=PairwiseLatency(default=millis(10), jitter_fraction=0.5, seed=5),
    )
    sink = Sink()
    kernel.add_all([sink, Pinger()])
    kernel.run(until=seconds(10))

    gaps = {sink.stamps[i + 1] - sink.stamps[i] for i in range(len(sink.stamps) - 1)}
    assert len(gaps) > 1, "jitter was flattened out entirely"


# --------------------------------------------------------------------------
# Agents keep an honest book
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_agent_position_matches_the_venue(seed):
    """An agent's belief about its own position must equal the truth.

    The failure this guards was silent: the agent reported +7 where the venue
    held 0, and nothing about either number looked wrong on its own.
    """
    m = build(seed=seed)
    m.kernel.start()
    m.kernel.advance(until=seconds(20))

    for i in range(10):
        m.human.enqueue(
            SymbolCommand(
                SYMBOL,
                Submit(
                    HUMAN_ID,
                    Side.BUY if i % 2 == 0 else Side.SELL,
                    Quantity(3),
                    None,
                    OrderType.MARKET,
                    TimeInForce.IOC,
                ),
            )
        )
        m.kernel.advance(until=seconds(20 + (i + 1) * 2))

    venue_quantity = m.venue.account(HUMAN_ID).positions[SYMBOL].quantity
    assert m.human.position[SYMBOL] == venue_quantity


def test_market_orders_leave_nothing_working():
    """An IOC never rests, so no phantom order may survive it."""
    m = build(seed=3)
    m.kernel.start()
    m.kernel.advance(until=seconds(20))
    for _ in range(6):
        m.human.enqueue(
            SymbolCommand(
                SYMBOL,
                Submit(HUMAN_ID, Side.BUY, Quantity(2), None, OrderType.MARKET, TimeInForce.IOC),
            )
        )
    m.kernel.advance(until=seconds(40))
    assert m.human.live_orders == {}


# --------------------------------------------------------------------------
# The market behaves like a market
# --------------------------------------------------------------------------


def test_a_large_order_moves_the_price():
    """Size has to have consequences, or the book is decorative.

    A sweep should walk the book, pay progressively worse prices, and leave the
    touch higher than it found it, then decay back as the maker refills and the
    fundamental agents push against it. Temporary impact and permanent impact
    are different things, and a market that shows neither is not a market.

    Sized against the depth that is standing rather than against a round
    number. This test used to sweep 5,000 lots, which happened to be more than
    the whole offer side: it cleared the book, left nothing at the touch, and
    then read the mark off whichever small print landed next. That measures
    luck, not impact, and it duly broke the first time an unrelated change
    moved the market's state at the sixty-second mark.
    """
    # Funded explicitly. A person's default account is deliberately small
    # enough to read a profit against, and this test is about what a *large*
    # order does to a book, so it asks for the capital it needs rather than
    # inheriting whatever the product happens to hand a new user.
    m = build(seed=7, human_cash=40_000_000)
    m.kernel.start()
    # Measured on a settled market rather than during the open. Evidence
    # arrives over the session, so the first minutes are a violent repricing:
    # the offer can fall three hundred points while a buy order is walking up
    # through it, and what that measures is the open rather than the order.
    m.kernel.advance(until=seconds(180))
    _trading(m, SYMBOL, until=420, levels=3)

    instrument = m.venue.registry.require(SYMBOL)
    book = m.venue.engine(SYMBOL).book.snapshot(levels=10_000)
    offered = sum(int(quantity) for _, quantity in book.asks)
    assert offered > 20, f"nothing to sweep: only {offered} offered"

    before = float(m.venue.mark_price(SYMBOL))
    best_ask = float(instrument.from_ticks(book.best_ask))
    # The whole visible offer, so it has to walk every level rather than
    # stopping on the first. The book is about sixty lots a side now (three
    # makers quoting thirty, twenty-two and fourteen) where it used to show
    # thousands, and those thousands were stale orders their owners had lost
    # track of rather than liquidity anyone would have honoured.
    sweep = offered

    before_makers = _maker_position(m)
    m.human.enqueue(
        SymbolCommand(
            SYMBOL,
            Submit(HUMAN_ID, Side.BUY, Quantity(sweep), None, OrderType.MARKET, TimeInForce.IOC),
        )
    )
    m.kernel.advance(until=Timestamp(int(m.kernel.now) + int(seconds(1))))

    position = m.venue.account(HUMAN_ID).positions[SYMBOL]
    assert position.quantity > 0, "the sweep did not fill at all"
    # It paid no better than the touch. Not *worse* than it, necessarily:
    # between reading the book and the order arriving twenty milliseconds
    # later the book can have thinned to a single level, and then a sweep of
    # everything visible is a sweep of one price. That is true of real markets
    # too, and the impact claim below is what actually carries this test.
    assert float(position.average_price) >= best_ask

    # And the market answered. Either the cheapest offer left is dearer than
    # the one it started with, or the move was large enough that the circuit
    # breaker stopped trading, which is the same statement about impact, made
    # by the venue instead of by the book.
    after = m.venue.engine(SYMBOL).book.snapshot()
    halted = m.venue.session(SYMBOL) is not SessionState.CONTINUOUS
    moved = (
        after.best_ask is None
        or float(instrument.from_ticks(after.best_ask)) > best_ask
    )
    assert halted or moved, (
        f"the sweep left the offer at {after.best_ask} against {best_ask} and "
        "did not trip anything"
    )

    # And then the market repairs itself, which it could not do when there was
    # one maker.
    #
    # This assertion used to say the opposite, and said so deliberately: with a
    # single maker, sweeping 60% of the offers left it short past the point its
    # collateral allowed it to quote, and the spread it left behind was still
    # ten times its opening width three minutes later. Not a slow repair: no
    # repair. The comment there said the test should fail on the day
    # replenishment worked, and it did: three makers differing in spread, size
    # and inventory limit mean the one that gets run over is not the only one
    # there, and the spread comes back from 1.25 to 2.50 rather than to 24.50.
    # The *change* across the makers, not their level. With three of them the
    # sweep is shared, and which one sells depends on whose quote was in front,
    # so asserting on `mm-1` alone was asserting on the queue. And a maker can
    # be long before the sweep and still long after selling into it, which is
    # why the level says nothing.
    after_makers = _maker_position(m)
    sold = before_makers - after_makers
    assert sold > 0, (
        f"the makers went from {before_makers} to {after_makers} and sold nothing"
    )

    before_spread = float(instrument.from_ticks(book.best_ask)) - float(
        instrument.from_ticks(book.best_bid)
    )
    _trading(m, SYMBOL, until=int(m.kernel.now // 1_000_000_000) + 120)
    healed = m.venue.engine(SYMBOL).book.snapshot()
    assert healed.best_ask is not None and healed.best_bid is not None
    after_spread = float(instrument.from_ticks(healed.best_ask)) - float(
        instrument.from_ticks(healed.best_bid)
    )
    assert after_spread <= 4 * before_spread, (
        f"the spread went from {before_spread} to {after_spread} and stayed there; "
        "replenishment has stopped working"
    )


def test_the_mark_never_sits_outside_the_touch():
    """A print is about the past; a resting order is about now.

    When they disagree the resting order wins, because it is a price someone
    will actually deal at. Without the clamp a single lot trading on the bid
    can drag the mark away from a touch that a thousand lots are standing at,
    and every open position is then valued at a number nobody is offering.
    """
    m = build(seed=7)
    m.kernel.start()
    m.kernel.advance(until=seconds(45))

    for symbol in m.venue.registry.symbols:
        instrument = m.venue.registry.require(symbol)
        book = m.venue.engine(symbol).book.snapshot()
        mark = float(m.venue.mark_price(symbol))
        if (
            book.best_bid is not None
            and book.best_ask is not None
            and int(book.best_bid) > int(book.best_ask)
        ):
            # A call phase accumulates orders without matching, so its book is
            # crossed on purpose and "inside the touch" is an empty interval.
            continue
        if book.best_bid is not None:
            assert mark >= float(instrument.from_ticks(book.best_bid)) - 1e-9, symbol
        if book.best_ask is not None:
            assert mark <= float(instrument.from_ticks(book.best_ask)) + 1e-9, symbol


def test_a_market_order_is_collateralised_against_the_book_not_the_range():
    """A market order can only trade against resting liquidity.

    Reserving against the far end of the settlement range instead (10,000 on a
    contract quoted near 4,700) rejects orders that could never have cost
    anything like that much, for a price they were structurally incapable of
    paying. The symptom was a large order silently vanishing.

    Funded explicitly: the point is that collateral is measured against the
    *book*, so the account has to be able to cover what the book actually
    holds. Inheriting a deliberately small default account would make this
    test pass or fail for the wrong reason.
    """
    m = build(seed=7, human_cash=40_000_000)
    m.kernel.start()
    _trading(m, SYMBOL)

    m.human.enqueue(
        SymbolCommand(
            SYMBOL,
            Submit(HUMAN_ID, Side.BUY, Quantity(5000), None, OrderType.MARKET, TimeInForce.IOC),
        )
    )
    m.kernel.advance(until=Timestamp(int(m.kernel.now) + int(seconds(2))))

    rejects = [e for e in m.human.log if e["type"] == "reject"]
    assert not rejects, f"order was rejected: {rejects[-1]['reason']}"
    assert m.venue.account(HUMAN_ID).positions[SYMBOL].quantity > 0


def test_an_order_beyond_all_liquidity_and_capital_is_still_refused():
    """The check must still bite when it should.

    Bounding a market order by the book must not become a way to take on
    exposure the account cannot cover: a resting limit order still reserves
    against its own price.
    """
    m = build(seed=7)
    m.kernel.start()
    m.kernel.advance(until=seconds(60))

    instrument = m.venue.registry.require(SYMBOL)
    m.human.enqueue(
        SymbolCommand(
            SYMBOL,
            Submit(
                HUMAN_ID,
                Side.BUY,
                Quantity(500_000),
                instrument.to_ticks(Decimal("4700")),
                OrderType.LIMIT,
                TimeInForce.GTC,
            ),
        )
    )
    m.kernel.advance(until=seconds(62))

    reasons = [e["reason"] for e in m.human.log if e["type"] == "reject"]
    assert "insufficient_collateral" in reasons


def test_the_market_trades(market):
    total = sum(len(market.venue.engine(s).tape) for s in market.venue.registry.symbols)
    assert total > 200, f"only {total} trades in 90 seconds"


def test_every_instrument_is_quoted_and_never_crossed(market):
    """Every book is live, and none of them cross.

    Not every book is two-sided, and that is correct rather than a defect. A
    maker at its position limit stops adding to the side that would breach it,
    so a contract the market has decided is worthless can end up offered-only.
    That is adverse selection visible in the book, which is a phenomenon to
    observe rather than to configure away.

    A book in a call phase is *expected* to be crossed; that is what a call
    phase is: orders accumulate without matching until the uncross clears them
    all at one price. So the check applies to books that are trading, and the
    ones that are not are skipped rather than excused.
    """
    from arena.exchange.session import SessionState

    for symbol in market.venue.registry.symbols:
        if market.venue.session(symbol) is not SessionState.CONTINUOUS:
            continue
        book = market.venue.engine(symbol).book.snapshot()
        assert book.bids or book.asks, f"{symbol} has no quotes at all"
        if book.best_bid is not None and book.best_ask is not None:
            assert book.best_bid < book.best_ask, f"{symbol} is crossed"


def test_the_futures_stay_two_sided(market):
    """The instruments carrying the research should keep a real market."""
    for symbol, _tolerance in DISCOVERY:
        book = market.venue.engine(symbol).book.snapshot()
        assert book.best_bid is not None and book.best_ask is not None


def test_no_account_is_over_committed(market):
    """Free cash below zero means the venue allowed a trade it could not back."""
    for agent_id, account in market.venue.accounts.items():
        assert int(account.free_cash) >= 0, f"{agent_id} is over-committed"


def test_value_is_conserved_in_a_live_market(market):
    assert market.venue.conservation_check() == 0



@pytest.mark.parametrize("symbol,tolerance", DISCOVERY)
def test_price_discovers_the_settlement_value(symbol, tolerance):
    """The market must move toward what the contract will actually pay.

    Opened deliberately at the midpoint of each contract's range rather than at
    the answer, so any convergence has to be produced by the fundamental agents
    trading against the maker. Without this the market would be liquid and
    volatile but anchored to nothing.

    Measured on seed 17 over the second half of a 360 second session:
    VANTA_SOLO_WR opens 3,602.5 from its settlement of 1,397.50 and averages
    822.76, closing 84.0% of the gap and finishing 0.0575 of its range out;
    RIFT_SOLO_WR opens 3,095.5 from 1,904.50 and averages 947.83, closing 69.1%
    and finishing 0.0957 out. The share of the gap closed is the assertion that
    carries the claim, because a residual expressed as a fraction of the range
    says nothing about how far the price travelled to get there.
    """
    from dashboard.build_market import instruments, true_values

    m = build(seed=17)
    m.kernel.start()
    truth = true_values(instruments())
    instrument = m.venue.registry.require(symbol)

    # Compared in price space, not ticks: a mark is a midpoint and can fall
    # between ticks, which to_ticks rightly refuses to round.
    low, high = instrument.settlement_bounds
    opening = float(low + high) / 2
    target = float(instrument.from_ticks(int(truth[symbol])))

    # Averaged over the second half rather than read off the end. The endpoint
    # of a market whose evidence arrives over the session is one draw from a
    # noisy process: a contract whose prior already equals its settlement has
    # nothing to discover, so all that is left to measure is the wandering. The
    # harness that produces this project's published results samples the same
    # way, for the same reason.
    samples = []
    for moment in range(20, 361, 20):
        m.kernel.advance(until=seconds(moment))
        if moment > 180:
            samples.append(float(m.venue.mark_price(symbol)))

    final = sum(samples) / len(samples)
    span = abs(float(high) - float(low))

    assert abs(final - target) < abs(opening - target), (
        f"{symbol} moved away from settlement: opened {opening}, "
        f"ended {final}, truth {target}"
    )
    closed = 1.0 - abs(final - target) / abs(opening - target)
    assert closed > 0.5, (
        f"{symbol} closed only {closed:.1%} of the gap between its opening "
        f"{opening} and its settlement {target}"
    )
    assert abs(final - target) / span < tolerance


def test_the_maker_respects_its_position_limit(market):
    """A position limit is soft, and the reason is worth stating.

    The maker stops quoting the side that would breach its limit, but its
    cancels are in flight while fills are still landing, so it can overshoot by
    roughly one quote. Any real market maker has the same exposure (an order
    already resting cannot be unsent) so the test allows the overshoot and
    checks it stays small rather than pretending it cannot happen.
    """
    maker = next(a for a in market.agents if isinstance(a, MarketMaker))
    slack = maker.quote_size * 2
    for symbol, quantity in maker.position.items():
        assert abs(quantity) <= maker.position_limit + slack, (
            f"{symbol} overshot its limit by more than one quote"
        )


def test_agent_populations_are_present(market):
    # By what an agent *is*, not by what it is called. The class-name version
    # of this broke the day a specialised maker was wired in, which is a fact
    # about the test rather than about the population.
    # Three, and that is the point: one maker is not a market, it is a
    # counterparty. Sweeping 60% of the offers against a single maker, it
    # absorbed 89% of the order and the spread it left never recovered.
    assert sum(isinstance(a, MarketMaker) for a in market.agents) == 3
    kinds = [type(a).__name__ for a in market.agents]
    # Six, log-spaced in precision. Two informed traders is not a
    # population: both pinned at their position limits a minute in and
    # the price simply stopped where their capital ran out.
    assert kinds.count("FundamentalTrader") == 6
    assert kinds.count("NoiseTrader") >= 10


# --------------------------------------------------------------------------
# The arbitrageur: consistency *between* books
# --------------------------------------------------------------------------


def test_relations_are_read_out_of_the_listed_contracts():
    """Nothing is configured by hand: the algebra comes from the contracts.

    This is what makes the agent connective tissue rather than a strategy: list
    a new spread and it becomes arbitrageable with no code change. If these
    relations were a hand-written table, the test would be checking that I can
    copy a list, which is not a property of the market.
    """
    from arena.agents.arbitrageur import derive_relations

    listed = {i.symbol: i for i in build_instruments()}
    relations = {r.name: r for r in derive_relations(listed)}

    spread = relations["spread:HALCYON_FORMAT_SPD"]
    assert spread.target == "HALCYON_FORMAT_SPD"
    assert dict(spread.legs) == {
        "HALCYON_SOLO_WR": 1.0,
        "HALCYON_OBJECTIVE_WR": -1.0,
    }
    assert spread.constant == 0.0

    # Put-call parity, C = P + F - K, with the strike as the constant.
    parity = relations["parity:EMBER_OBJECTIVE_C5000"]
    assert parity.target == "EMBER_OBJECTIVE_C5000"
    assert dict(parity.legs) == {
        "EMBER_OBJECTIVE_P5000": 1.0,
        "EMBER_OBJECTIVE_WR": 1.0,
    }
    assert parity.constant == -5_000.0


def test_a_relation_missing_a_leg_is_not_formed():
    """Take a leg away and the index relation goes; put it back and it returns.

    A relation traded against a proxy is a bet, not an arbitrage, so the agent
    has to decline it, and then form it the moment the leg is listed, with no
    code change, or the derivation is not really reading the contracts.

    Written by *removing* a listed leg rather than by relying on one being
    absent. An earlier version leaned on one component having no future of its
    own, which was true until that future was listed precisely so the index
    would have every leg; a test whose premise is an accident of the listing
    stops testing anything the day the listing improves.

    The index is equal weighted over the six subjects the exchange lists, one
    per archetype, so each leg carries a sixth and the six of them are the
    whole basket.
    """
    from arena.agents.arbitrageur import derive_relations

    listed = {i.symbol: i for i in build_instruments()}
    assert "CIRCUIT_SOLO_IDX" in listed
    leg = listed.pop("RIFT_SOLO_WR")
    assert not any(r.target == "CIRCUIT_SOLO_IDX" for r in derive_relations(listed))

    listed[leg.symbol] = leg
    index = next(r for r in derive_relations(listed) if r.target == "CIRCUIT_SOLO_IDX")
    assert set(dict(index.legs)) == {
        "VANTA_SOLO_WR",
        "QUILL_SOLO_WR",
        "BASTION_SOLO_WR",
        "EMBER_SOLO_WR",
        "HALCYON_SOLO_WR",
        "RIFT_SOLO_WR",
    }
    assert all(weight == pytest.approx(1 / 6) for _leg, weight in index.legs)


@pytest.fixture(scope="module")
def arb_market():
    """A market with the arbitrageur switched on.

    It is off by default. Across four paired seeds it improved spread
    consistency on three and worsened it on the fourth, and on one seed it
    took visible ask depth from 877 lots to 69: consistency bought with
    liquidity. So the tests below assert what it reliably *does* (derives the
    right identities, trades them, conserves value, respects its limits) and
    not the price convergence it does not reliably deliver. docs/GAPS.md
    carries the numbers.
    """
    m = build(seed=41, arbitrageur=True)
    m.kernel.start()
    m.kernel.advance(until=seconds(300))
    return m


def test_the_arbitrageur_actually_traded(arb_market):
    """Otherwise every assertion below is about an idle agent."""
    from arena.agents.arbitrageur import Arbitrageur

    arb = next(a for a in arb_market.agents if isinstance(a, Arbitrageur))
    assert arb.attempts > 0


def test_the_arbitrageur_leaves_conservation_exact(arb_market):
    """It adds volume, so it is the obvious place for the invariant to break."""
    assert arb_market.venue.conservation_check() == 0


def test_the_arbitrageur_never_legs_past_its_position_limit(arb_market):
    """A half-entered relation is a directional bet, so the limit is per leg."""
    from arena.agents.arbitrageur import Arbitrageur

    arb = next(a for a in arb_market.agents if isinstance(a, Arbitrageur))
    for symbol, quantity in arb.position.items():
        # The overshoot is bounded by one package on every leg, not by one
        # order: the agent checks its limits, then fires the whole package as
        # simultaneous IOCs, and each leg rounds its own size up. A relation
        # with three legs can therefore end one package plus rounding past the
        # limit on any of them.
        # One package of slack per relation the symbol is a leg of. The
        # agent checks its limits, then fires whole packages as simultaneous
        # IOCs, and a symbol that appears in a vertical, a butterfly and a
        # parity can be committed to by three of them in one wakeup before any
        # fill comes back.
        involved = sum(symbol in relation.symbols for relation in arb.relations)
        allowance = arb.position_limit + arb.base_size * max(1, involved) * 2
        assert abs(quantity) <= allowance, (
            f"{symbol}: {abs(quantity)} against a limit of {arb.position_limit} "
            f"with {involved} relations"
        )


def test_the_arbitrageur_sizes_to_the_liquidity_it_can_see(arb_market):
    """The participation cap is what stops it stripping the book bare.

    Without it this agent fired IOC orders on three legs every 400ms and took
    ask depth from 207 resting lots to 26, and a book that thin cannot absorb
    anyone else's order, so the damage was not confined to its own P&L.
    """
    from arena.agents.arbitrageur import Arbitrageur

    arb = next(a for a in arb_market.agents if isinstance(a, Arbitrageur))
    assert 0.0 < arb.max_participation <= 1.0
    for symbol in arb.instruments:
        for side in (Side.BUY, Side.SELL):
            book = arb.books[symbol]
            resting = book.ask_size if side is Side.BUY else book.bid_size
            assert arb._takeable(symbol, side) <= max(0, resting)


# --------------------------------------------------------------------------
# You can see who took the other side
# --------------------------------------------------------------------------


def test_your_counterparty_is_named_and_is_one_of_the_bots():
    """"Is anything actually on the other side of this?" deserves a real answer.

    A roster of agents does not answer it; naming the participant on the fill
    does. The population here is a market maker, two fundamental traders and
    fourteen noise traders, and a person's order has to end up against one of
    them rather than against nothing. Six informed traders now, but the point
    is unchanged: somebody real is on the other side.
    """
    market = build(seed=7, human_cash=250_000)
    market.start()
    _trading(market, SYMBOL)
    market.submit(SYMBOL, "buy", 12, None)
    market.kernel.advance(until=seconds(50))

    fills = market.venue.counterparties_for(HUMAN_ID)
    assert fills, "the order never traded against anyone"

    population = {str(a.agent_id) for a in market.agents}
    for fill in fills:
        assert fill["counterparty"] in population, fill["counterparty"]
        assert fill["counterparty"] != str(HUMAN_ID), "you cannot fill yourself"
        assert fill["symbol"] == SYMBOL
        assert fill["quantity"] > 0


def test_your_fills_survive_the_bots_flooding_the_tape():
    """The regression that made the feature useless without failing anything.

    The first version kept one rolling window of *every* fill on the venue. The
    bots print thousands a minute, so a person's single trade was evicted from
    it within seconds: the panel was empty in exactly the situation it existed
    for, and nothing anywhere reported a problem. Logs are per participant now.
    """
    market = build(seed=7, human_cash=250_000)
    market.start()
    _trading(market, SYMBOL)
    market.submit(SYMBOL, "buy", 12, None)
    market.kernel.advance(until=seconds(50))

    immediately = market.venue.counterparties_for(HUMAN_ID)
    assert immediately

    # Two more minutes of the population trading among themselves.
    market.kernel.advance(until=seconds(180))
    assert market.venue.counterparties_for(HUMAN_ID) == immediately


def test_a_person_starts_with_an_account_they_can_read():
    """A gain of a hundred against forty million teaches nothing.

    The bots keep a large balance because a maker quoting seven books at once
    genuinely needs one; the person does not, and their opening balance is
    deliberately a figure a profit is visible against.
    """
    from dashboard.build_market import HUMAN_STARTING_CASH

    market = build(seed=7)
    human = market.venue.account(HUMAN_ID)
    maker = next(a for a in market.agents if isinstance(a, MarketMaker))

    assert float(from_money(human.starting_cash)) == float(HUMAN_STARTING_CASH)
    assert human.starting_cash < market.venue.account(maker.agent_id).starting_cash
