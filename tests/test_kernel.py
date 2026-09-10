"""Discrete-event kernel tests.

Two properties carry the whole simulation layer:

**Determinism.** A seeded run must produce an identical event trace every time.
If it does not, no experiment is reproducible and no result is checkable.

**Honest information timing.** An agent must learn things when a message reaches
it, not when they happen. Every latency and information-asymmetry experiment the
project plans is a question about this and nothing else, so it is tested
directly rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timezone
from typing import Any

import pytest

from arena.exchange.events import Acknowledged, Filled, Submit
from arena.exchange.types import AgentId, OrderType, Price, Quantity, Side, TimeInForce
from arena.contracts.payoff import Linear
from arena.contracts.spec import ContractSpec, DataPolicy, ObservationWindow
from arena.contracts.underlying import MetricRef, Single
from arena.market.instrument import Instrument
from arena.market.venue import SymbolCommand, Venue
from arena.market.venue_agent import VenueAgent
from arena.sim.kernel import Kernel, SimulationContext
from arena.sim.latency import PairwiseLatency, UniformLatency
from arena.sim.messages import Feed, PrivateEvent, Subscribe, TopOfBook, TradePrint
from arena.sim.time import Duration, Timestamp, micros, millis, seconds

UTC = timezone.utc
EXCHANGE = AgentId("exchange")
SYM = "TEST_FUT"


def make_exchange() -> VenueAgent:
    """A one-symbol venue, so these tests stay about the kernel.

    The kernel does not care how many instruments a venue lists; these tests
    are about ordering, latency and delivery, so they use the smallest venue
    that can accept an order.
    """
    from datetime import datetime as _dt, timedelta as _td

    window = ObservationWindow(
        _dt(2026, 9, 1, tzinfo=UTC), _dt(2026, 9, 30, tzinfo=UTC)
    )
    spec = ContractSpec(
        contract_id=SYM,
        underlying=Single(MetricRef("win_rate", "EMBER")),
        payoff=Linear(scale=10_000.0),
        window=window,
        policy=DataPolicy(min_sample_size=1),
        reference_id="ref-1",
        published_at=window.start - _td(days=1),
        tick_size="1",
    )
    venue = Venue("test", starting_cash=50_000_000)
    venue.list_instrument(Instrument(SYM, spec))
    return VenueAgent(EXCHANGE, venue)


@dataclass
class Recorder:
    """An agent that records everything it sees, with the time it saw it."""

    agent_id: AgentId
    wake_every: Duration | None = None
    wakes: list[int] = field(default_factory=list)
    seen: list[tuple[int, Any]] = field(default_factory=list)
    subscribe_to: tuple[Feed, ...] = ()
    to_send: list[tuple[AgentId, Any]] = field(default_factory=list)
    # Delay before sending. Needed whenever a test wants other agents'
    # subscriptions to have landed first: a subscription is a message and is
    # subject to the sender's latency like any other, so a slow agent's request
    # to subscribe arrives slowly too.
    send_after: Duration | None = None

    def on_start(self, ctx: SimulationContext) -> None:
        for feed in self.subscribe_to:
            ctx.send(EXCHANGE, Subscribe(feed))
        if self.send_after is None:
            self._flush(ctx)
        else:
            ctx.request_wakeup(self.send_after)
        if self.wake_every is not None:
            ctx.request_wakeup(self.wake_every)

    def _flush(self, ctx: SimulationContext) -> None:
        for recipient, message in self.to_send:
            ctx.send(recipient, message)

    def on_wakeup(self, ctx: SimulationContext) -> None:
        self.wakes.append(int(ctx.now))
        if self.send_after is not None and len(self.wakes) == 1:
            self._flush(ctx)
            return
        if self.wake_every is not None and len(self.wakes) < 5:
            ctx.request_wakeup(self.wake_every)

    def on_message(self, ctx: SimulationContext, sender: AgentId, message: Any) -> None:
        self.seen.append((int(ctx.now), message))

    def on_finish(self, ctx: SimulationContext) -> None:
        pass


def limit(agent, side, price, qty) -> Submit:
    return Submit(agent, side, Quantity(qty), Price(price), OrderType.LIMIT, TimeInForce.GTC)


# --------------------------------------------------------------------------
# Ordering and determinism
# --------------------------------------------------------------------------


def test_events_are_processed_in_time_order():
    kernel = Kernel(seed=1, latency=UniformLatency(base=millis(1), jitter=Duration(0)))
    agent = Recorder(AgentId("a"), wake_every=seconds(1))
    kernel.add(agent)
    kernel.run(until=seconds(10))

    assert agent.wakes == sorted(agent.wakes)
    assert agent.wakes[0] == int(seconds(1))


def test_time_never_moves_backwards():
    kernel = Kernel(seed=1)
    kernel.record_trace = True
    a = Recorder(AgentId("a"), wake_every=micros(37))
    b = Recorder(AgentId("b"), wake_every=micros(11))
    kernel.add_all([a, b])
    kernel.run(until=millis(1))

    stamps = [int(e.timestamp) for e in kernel.trace]
    assert stamps == sorted(stamps)


def test_simultaneous_events_are_broken_by_insertion_order():
    """Two events at the same nanosecond still need a total order.

    Without the insertion counter a heap of (timestamp, payload) tuples would
    fall back to comparing payloads, which is either a TypeError or, worse, a
    silent ordering that depends on object contents.
    """
    kernel = Kernel(seed=1, latency=UniformLatency(base=millis(1), jitter=Duration(0)))
    a = Recorder(AgentId("a"), wake_every=millis(5))
    b = Recorder(AgentId("b"), wake_every=millis(5))
    kernel.record_trace = True
    kernel.add_all([a, b])
    kernel.run(until=millis(20))

    at_five = [e for e in kernel.trace if int(e.timestamp) == int(millis(5))]
    assert len(at_five) == 2
    assert [e.sequence for e in at_five] == sorted(e.sequence for e in at_five)


def _run_market(seed: int, latency=None) -> list[dict]:
    kernel = Kernel(seed=seed, latency=latency)
    exchange = make_exchange()
    kernel.add(exchange)

    for i, (side, price, qty) in enumerate(
        [
            (Side.BUY, 100, 10),
            (Side.SELL, 102, 8),
            (Side.BUY, 101, 5),
            (Side.SELL, 101, 12),
            (Side.BUY, 103, 7),
        ]
    ):
        name = AgentId(f"trader{i}")
        kernel.add(
            Recorder(name, subscribe_to=(Feed.TRADES,), to_send=[(EXCHANGE, SymbolCommand(SYM, limit(name, side, price, qty)))])
        )

    kernel.run(until=seconds(5))
    return [e.to_dict() for e in []] or [
        t.to_dict() for t in exchange.venue.engine(SYM).tape
    ]


@pytest.mark.parametrize("seed", [1, 7, 99])
def test_seeded_runs_are_reproducible(seed):
    """The property every experiment in this project depends on."""
    assert _run_market(seed) == _run_market(seed)


def test_different_seeds_can_differ_but_stay_valid():
    """Jitter is seeded, so seeds are genuinely independent streams."""
    a = _run_market(1)
    b = _run_market(2)
    # Both must be well-formed even where they diverge.
    for tape in (a, b):
        assert all(t["quantity"] > 0 for t in tape)


def test_agent_random_streams_are_independent_of_population():
    """Adding an agent must not shift another agent's draws.

    If it did, changing the population would silently change every agent's
    behaviour and two configurations could never be compared.
    """
    small = Kernel(seed=42)
    small.add(Recorder(AgentId("alice")))
    alice_draws = [small.rng_for(AgentId("alice")).random() for _ in range(5)]

    large = Kernel(seed=42)
    large.add_all([Recorder(AgentId(f"agent{i}")) for i in range(20)])
    large.add(Recorder(AgentId("alice")))
    alice_again = [large.rng_for(AgentId("alice")).random() for _ in range(5)]

    assert alice_draws == alice_again


# --------------------------------------------------------------------------
# Latency: the substrate for information asymmetry
# --------------------------------------------------------------------------


def test_messages_arrive_after_the_configured_latency():
    kernel = Kernel(seed=1, latency=UniformLatency(base=millis(5), jitter=Duration(0)))
    sender = Recorder(AgentId("a"), to_send=[(AgentId("b"), "hello")])
    receiver = Recorder(AgentId("b"))
    kernel.add_all([sender, receiver])
    kernel.run(until=seconds(1))

    assert len(receiver.seen) == 1
    assert receiver.seen[0][0] == int(millis(5))


def test_a_slow_agent_sees_the_same_event_later():
    """The core mechanism behind every latency experiment.

    Both agents subscribe to the same feed and the exchange broadcasts once.
    They receive it at different times purely because their latencies differ,
    which is what makes 'how much is faster information worth' a measurable
    question rather than a stipulated one.
    """
    latency = PairwiseLatency(
        default=millis(1),
        per_agent={AgentId("fast"): micros(10), AgentId("slow"): millis(100)},
        jitter_fraction=0.0,
    )
    kernel = Kernel(seed=1, latency=latency)
    exchange = make_exchange()
    fast = Recorder(AgentId("fast"), subscribe_to=(Feed.TRADES,))
    slow = Recorder(AgentId("slow"), subscribe_to=(Feed.TRADES,))
    # Both trade only after 500ms, by which time even the slow agent's
    # subscription (itself delayed 100ms) has reached the exchange.
    mover = Recorder(
        AgentId("mover"),
        to_send=[(EXCHANGE, SymbolCommand(SYM, limit(AgentId("mover"), Side.SELL, 100, 10)))],
        send_after=millis(500),
    )
    taker = Recorder(
        AgentId("taker"),
        to_send=[(EXCHANGE, SymbolCommand(SYM, limit(AgentId("taker"), Side.BUY, 100, 10)))],
        send_after=millis(600),
    )
    kernel.add_all([exchange, fast, slow, mover, taker])
    kernel.run(until=seconds(2))

    fast_prints = [(t, m) for t, m in fast.seen if isinstance(m, TradePrint)]
    slow_prints = [(t, m) for t, m in slow.seen if isinstance(m, TradePrint)]

    assert fast_prints and slow_prints
    assert fast_prints[0][1].price == slow_prints[0][1].price
    assert fast_prints[0][0] < slow_prints[0][0]
    # And the gap is the latency difference, not an artifact.
    assert slow_prints[0][0] - fast_prints[0][0] >= int(millis(90))


def test_trade_prints_carry_a_unique_sequence():
    """Subscribers need to identify a print, and price cannot do it.

    In a narrow market the same price prints hundreds of times, so anything
    aligning two subscribers' views by price will pair up different trades. The
    engine's match number is carried through unchanged for exactly this reason.
    """
    kernel = Kernel(seed=3, latency=UniformLatency(base=millis(1), jitter=Duration(0)))
    exchange = make_exchange()
    watcher = Recorder(AgentId("watcher"), subscribe_to=(Feed.TRADES,))
    maker, taker = AgentId("maker"), AgentId("taker")

    kernel.add_all(
        [
            exchange,
            watcher,
            Recorder(
                maker,
                to_send=[(EXCHANGE, SymbolCommand(SYM, limit(maker, Side.SELL, 100, 1))) for _ in range(5)],
                send_after=millis(10),
            ),
            Recorder(
                taker,
                to_send=[(EXCHANGE, SymbolCommand(SYM, limit(taker, Side.BUY, 100, 1))) for _ in range(5)],
                send_after=millis(20),
            ),
        ]
    )
    kernel.run(until=seconds(1))

    prints = [m for _, m in watcher.seen if isinstance(m, TradePrint)]
    assert len(prints) == 5
    # Every print is at the same price, so only the sequence distinguishes them.
    assert len({int(p.price) for p in prints}) == 1
    sequences = [int(p.sequence) for p in prints]
    assert len(set(sequences)) == len(sequences)
    assert sequences == sorted(sequences)


def test_latency_jitter_is_reproducible():
    model = PairwiseLatency(default=millis(1), jitter_fraction=0.2, seed=5)
    first = [model.delay(AgentId("a"), AgentId("b")) for _ in range(10)]
    again = PairwiseLatency(default=millis(1), jitter_fraction=0.2, seed=5)
    second = [again.delay(AgentId("a"), AgentId("b")) for _ in range(10)]

    assert first == second
    # Jitter must actually vary, or the test proves nothing.
    assert len(set(first)) > 1


def test_the_slower_leg_dominates_a_pair():
    """A message cannot arrive faster than its slowest hop."""
    model = PairwiseLatency(
        default=millis(1),
        per_agent={AgentId("fast"): micros(1), AgentId("slow"): millis(50)},
        jitter_fraction=0.0,
    )
    assert model.delay(AgentId("fast"), AgentId("slow")) == millis(50)


# --------------------------------------------------------------------------
# The exchange as an agent
# --------------------------------------------------------------------------


def test_orders_routed_through_the_kernel_reach_the_book():
    kernel = Kernel(seed=1, latency=UniformLatency(base=millis(1), jitter=Duration(0)))
    exchange = make_exchange()
    alice = AgentId("alice")
    kernel.add_all(
        [exchange, Recorder(alice, to_send=[(EXCHANGE, SymbolCommand(SYM, limit(alice, Side.BUY, 100, 10)))])]
    )
    kernel.run(until=seconds(1))

    assert exchange.venue.engine(SYM).book.snapshot().best_bid == 100


def test_an_agent_receives_its_own_fills_privately():
    kernel = Kernel(seed=1, latency=UniformLatency(base=millis(1), jitter=Duration(0)))
    exchange = make_exchange()
    maker, taker = AgentId("maker"), AgentId("taker")
    m = Recorder(maker, to_send=[(EXCHANGE, SymbolCommand(SYM, limit(maker, Side.SELL, 100, 10)))])
    t = Recorder(taker, to_send=[(EXCHANGE, SymbolCommand(SYM, limit(taker, Side.BUY, 100, 10)))])
    kernel.add_all([exchange, m, t])
    kernel.run(until=seconds(1))

    def fills(agent):
        return [
            msg.event
            for _, msg in agent.seen
            if isinstance(msg, PrivateEvent) and isinstance(msg.event, Filled)
        ]

    assert len(fills(m)) == 1
    assert len(fills(t)) == 1
    assert fills(m)[0].aggressor is False
    assert fills(t)[0].aggressor is True


def test_an_agent_cannot_act_for_another():
    """Trusting the agent_id field would let anyone cancel anyone's orders."""
    kernel = Kernel(seed=1, latency=UniformLatency(base=millis(1), jitter=Duration(0)))
    exchange = make_exchange()
    alice, mallory = AgentId("alice"), AgentId("mallory")
    # Mallory sends an order claiming to be Alice.
    bad = Recorder(mallory, to_send=[(EXCHANGE, SymbolCommand(SYM, limit(alice, Side.BUY, 100, 10)))])
    kernel.add_all([exchange, Recorder(alice), bad])
    kernel.run(until=seconds(1))

    assert exchange.venue.engine(SYM).book.snapshot().best_bid is None


def test_a_new_subscriber_gets_an_immediate_snapshot():
    """Otherwise a subscriber is blind until something happens to move the book."""
    kernel = Kernel(seed=1, latency=UniformLatency(base=millis(1), jitter=Duration(0)))
    exchange = make_exchange()
    seeder = AgentId("seeder")
    watcher = Recorder(AgentId("watcher"), subscribe_to=(Feed.TOP_OF_BOOK,))
    kernel.add_all(
        [
            exchange,
            Recorder(seeder, to_send=[(EXCHANGE, SymbolCommand(SYM, limit(seeder, Side.BUY, 100, 10)))]),
            watcher,
        ]
    )
    kernel.run(until=seconds(1))

    assert any(isinstance(m, TopOfBook) for _, m in watcher.seen)


def test_throttled_subscribers_receive_fewer_updates():
    """Models a subscriber that cannot consume every tick."""
    kernel = Kernel(seed=1, latency=UniformLatency(base=micros(10), jitter=Duration(0)))
    exchange = make_exchange()

    @dataclass
    class Churner:
        agent_id: AgentId
        sent: int = 0

        def on_start(self, ctx):
            ctx.request_wakeup(millis(1))

        def on_wakeup(self, ctx):
            self.sent += 1
            ctx.send(
                EXCHANGE,
                SymbolCommand(SYM, limit(self.agent_id, Side.BUY, 90 + self.sent % 5, 1)),
            )
            if self.sent < 40:
                ctx.request_wakeup(millis(1))

        def on_message(self, ctx, sender, message):
            pass

        def on_finish(self, ctx):
            pass

    live = Recorder(AgentId("live"), subscribe_to=())
    live.subscribe_to = (Feed.TOP_OF_BOOK,)
    lagged = Recorder(AgentId("lagged"))

    kernel.add_all([exchange, Churner(AgentId("churner")), live, lagged])
    # Subscribe the lagged agent with a throttle by hand.
    lagged.to_send = [(EXCHANGE, Subscribe(Feed.TOP_OF_BOOK, throttle=int(millis(10))))]
    kernel.run(until=seconds(1))

    live_updates = sum(1 for _, m in live.seen if isinstance(m, TopOfBook))
    lagged_updates = sum(1 for _, m in lagged.seen if isinstance(m, TopOfBook))

    assert live_updates > lagged_updates
    assert lagged_updates > 0


# --------------------------------------------------------------------------
# Safety
# --------------------------------------------------------------------------


def test_a_wakeup_cannot_be_scheduled_in_the_past():
    kernel = Kernel(seed=1)

    @dataclass
    class Naughty:
        agent_id: AgentId = AgentId("naughty")

        def on_start(self, ctx):
            with pytest.raises(ValueError, match="in the past"):
                ctx.request_wakeup(Duration(-1))

        def on_wakeup(self, ctx):
            pass

        def on_message(self, ctx, sender, message):
            pass

        def on_finish(self, ctx):
            pass

    kernel.add(Naughty())
    kernel.run(until=seconds(1))


def test_sending_to_an_unknown_agent_raises():
    kernel = Kernel(seed=1)
    kernel.add(Recorder(AgentId("a")))
    with pytest.raises(KeyError):
        kernel.send(AgentId("a"), AgentId("nobody"), "hello")


def test_duplicate_agent_ids_are_rejected():
    kernel = Kernel(seed=1)
    kernel.add(Recorder(AgentId("a")))
    with pytest.raises(ValueError, match="duplicate agent id"):
        kernel.add(Recorder(AgentId("a")))


def test_max_events_stops_a_runaway_loop():
    """An agent waking itself with no delay would otherwise spin forever."""
    kernel = Kernel(seed=1)

    @dataclass
    class Spinner:
        agent_id: AgentId = AgentId("spinner")

        def on_start(self, ctx):
            ctx.request_wakeup(Duration(0))

        def on_wakeup(self, ctx):
            ctx.request_wakeup(Duration(0))

        def on_message(self, ctx, sender, message):
            pass

        def on_finish(self, ctx):
            pass

    kernel.add(Spinner())
    kernel.run(max_events=500)
    assert kernel.processed == 500


def test_the_event_cap_does_not_strand_events_in_the_past():
    """Stopping early must not move the clock past what has not run yet.

    ``advance`` used to jump the clock to ``until`` unconditionally, including
    when it had bailed out on ``max_events``. Every event still queued behind
    ``until`` was then in the past, and the very next call raised "scheduled
    before current time" on events that were perfectly valid.

    The cap exists so a burst of activity cannot stall the loop serving the
    browser. So the corruption appeared only under load, which is the worst
    place for a simulation clock to be wrong.
    """
    from dataclasses import dataclass, field

    @dataclass
    class Ticker:
        agent_id: AgentId = AgentId("ticker")
        seen: list[int] = field(default_factory=list)

        def on_start(self, ctx):
            ctx.request_wakeup(millis(1))

        def on_wakeup(self, ctx):
            self.seen.append(int(ctx.now))
            ctx.request_wakeup(millis(1))

        def on_finish(self, ctx):
            pass

        def on_message(self, ctx, sender, message):
            pass

    kernel = Kernel(seed=1)
    ticker = Ticker()
    kernel.add(ticker)
    kernel.start()

    # Ask for ten seconds of a one-millisecond ticker (10,000 events) but allow
    # only 50 per slice, so the cap fires on every call.
    target = seconds(10)
    for _ in range(20):
        kernel.advance(until=target, max_events=50)

    assert ticker.seen, "nothing ran"
    assert int(kernel.now) <= int(target)
    # The clock must sit on the last event actually processed, not out at the
    # requested horizon with a thousand events stranded behind it.
    assert int(kernel.now) == ticker.seen[-1]
    assert ticker.seen == sorted(ticker.seen)


def test_an_uncapped_advance_still_reaches_its_horizon():
    """The fix must not stop a quiet market's clock from moving.

    With nothing left to run, time still has to pass; otherwise a market with
    no activity would freeze rather than idle.
    """
    kernel = Kernel(seed=1)
    kernel.start()
    kernel.advance(until=seconds(5))
    assert int(kernel.now) == int(seconds(5))
