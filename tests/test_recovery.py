"""A restart that does not lose the exchange.

Everything this venue knows lives in one process's memory, so until now a
restart discarded every account, position, working order and price. The journal
was written and tested against synthetic records long before anything wrote
real ones to it; this is the part that proves the recipe closes on the market
it was built for.

The claim being tested is narrow and total: replaying the exogenous inputs onto
a market built from the same seed reproduces the original exactly. Not
approximately, and not in aggregate. Same positions, same cash, same working
orders, same seats.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from arena.market.live import ENGINE_VERSION, LiveMarket, apply_input, rebuild
from arena.sim.journal import Journal, read_records
from arena.sim.time import seconds

from dashboard.build_market import build


def _drive(market: LiveMarket, seat_name: str = "alice"):
    """A short session of purely exogenous activity, of every kind recorded."""
    market.kernel.start()
    market.kernel.advance(until=seconds(20))
    market._sim_seconds = 20.0

    who = market.seat(seat_name)
    symbol = next(
        s
        for s in market.venue.registry.symbols
        if market.venue.registry.require(s).instrument_class == "future"
    )
    instrument = market.venue.registry.require(symbol)
    reference = market.venue.mark_price(symbol)
    tick = instrument.tick_size

    market._sim_seconds = 25.0
    market.kernel.advance(until=seconds(25))
    market.submit(symbol, "buy", 3, reference - tick * 20, trader=who)

    market._sim_seconds = 30.0
    market.kernel.advance(until=seconds(30))
    market.submit(symbol, "sell", 2, reference + tick * 20, trader=who)

    market._sim_seconds = 35.0
    market.kernel.advance(until=seconds(35))
    market.submit(symbol, "buy", 1, None, tif="ioc", trader=who)

    market._sim_seconds = 40.0
    market.kernel.advance(until=seconds(40))
    market.cancel_all(trader=who)

    market._sim_seconds = 45.0
    market.kernel.advance(until=seconds(45))
    return who, symbol


def _state(market: LiveMarket) -> dict:
    """Everything a rebuilt market has to agree with the original about."""
    return {
        "conservation": market.venue.conservation_check(),
        "seats": sorted(market.traders),
        "cash": {
            agent: int(account.cash)
            for agent, account in sorted(market.venue.accounts.items())
        },
        "positions": {
            agent: {
                symbol: position.quantity
                for symbol, position in sorted(account.positions.items())
                if position.quantity
            }
            for agent, account in sorted(market.venue.accounts.items())
        },
        "marks": {s: int(m) for s, m in sorted(market.venue.marks().items())},
    }


def test_a_rebuilt_market_is_the_same_market():
    """The whole recipe, end to end, on the venue it was written for.

    Journal the exogenous inputs, throw the market away, build a new one from
    the same seed, replay. Agent behaviour is not journalled and does not need
    to be: the kernel seeds every agent from `_stable_seed(seed, "agent", id)`
    and its draws do not depend on join order, so the population regenerates
    itself. Only what came from outside has to be recorded.
    """
    original = build(seed=7)
    original.journal = Journal.in_memory(
        engine_version=ENGINE_VERSION, metadata={"seed": 7}
    )
    who, symbol = _drive(original)
    before = _state(original)
    blob = original.journal.to_bytes()

    restored = build(seed=7)
    # The session was driven by hand rather than through `step`, so it wrote
    # no clock heartbeat and the journal's last record is the cancel at t=40.
    # Naming the stopping point is what the parameter is for; measured
    # without it, every account's positions differed by exactly the five
    # seconds of agent activity that nothing had recorded.
    result = rebuild(
        restored, blob, engine_version=ENGINE_VERSION, until_sim_seconds=45.0
    )

    assert result.applied > 0
    assert _state(restored) == before
    assert who in restored.traders
    assert restored.venue.conservation_check() == 0


def test_only_exogenous_input_is_recorded():
    """One record per outside command, and none for anything the seed can redo.

    The size argument for input journalling is the whole reason it is cheap: an
    order that sweeps two hundred resting levels is one record, not two hundred
    fills. A journal that grew with agent activity would be logging state, and
    the file would be larger than the market it describes.
    """
    market = build(seed=7)
    market.journal = Journal.in_memory(
        engine_version=ENGINE_VERSION, metadata={"seed": 7}
    )
    _drive(market)

    kinds = [record.kind for record in read_records(market.journal.to_bytes())]
    assert kinds == ["seat", "submit", "submit", "submit", "cancel_all"]


def test_replaying_does_not_journal_a_second_time():
    """Or every recovery doubles the file it just read.

    `submit` and its siblings record what they are asked to do, so a replay
    that left the journal attached would append each recovered input back onto
    the journal, and the next recovery would apply all of them twice.
    """
    original = build(seed=7)
    original.journal = Journal.in_memory(
        engine_version=ENGINE_VERSION, metadata={"seed": 7}
    )
    _drive(original)
    blob = original.journal.to_bytes()
    written = original.journal.records_written

    restored = build(seed=7)
    restored.journal = Journal.in_memory(
        engine_version=ENGINE_VERSION, metadata={"seed": 7}
    )
    rebuild(restored, blob, engine_version=ENGINE_VERSION)

    assert restored.journal.records_written == 0
    assert restored.journal is not None, "the journal must be reattached afterwards"
    assert written > 0


def test_a_market_without_a_journal_behaves_exactly_as_before():
    """Off by default, and off has to cost nothing and change nothing."""
    plain = build(seed=7)
    assert plain.journal is None
    _drive(plain)

    watched = build(seed=7)
    watched.journal = Journal.in_memory(
        engine_version=ENGINE_VERSION, metadata={"seed": 7}
    )
    _drive(watched)

    assert _state(plain) == _state(watched)


def test_an_input_this_market_cannot_apply_is_refused():
    """Skipping it would rebuild a market missing exactly those inputs.

    `journal.replay` documents this as the callback's job, and it is the one
    failure a recovery path must never have: a silent skip reports success on
    a market that is wrong by whatever it ignored.
    """
    market = build(seed=7)

    class Bogus:
        sequence, timestamp_ns, kind, payload = 1, 0, "teleport", {}

    with pytest.raises(ValueError, match="cannot apply"):
        apply_input(market, Bogus())


def test_the_clock_moves_before_the_command_lands():
    """An order has to meet the book it originally met.

    A record carries the simulated nanosecond its command arrived at. Applying
    it without advancing the agents to that instant first puts it into an
    earlier market, and every record after it compounds the difference.
    """
    market = build(seed=7)
    market.kernel.start()

    class At30:
        sequence, timestamp_ns, kind = 1, int(seconds(30)), "seat"
        payload = {"name": "bob"}

    assert market._sim_seconds < 30.0
    apply_input(market, At30())
    assert market._sim_seconds == pytest.approx(30.0)
    assert int(market.kernel.now) >= int(seconds(30))


def test_recording_cannot_fail_a_command_that_would_have_been_refused():
    """Instrumentation must not change what the venue does, including failing.

    `cancel` is reached with a `(symbol, id)` key as well as a bare id, and the
    lookup refuses the former on its own terms. Building the journal payload
    coerced it eagerly, so `int(("EMBER_OBJECTIVE_WR", 4))` raised and a refusal
    became a crash, on a path that had worked for as long as it existed, and on
    a market with no journal attached at all.

    Asserted both ways round, because the property is that the two markets
    behave identically rather than that either one succeeds.
    """
    plain = build(seed=7)
    plain.kernel.start()
    plain.kernel.advance(until=seconds(10))

    watched = build(seed=7)
    watched.journal = Journal.in_memory(
        engine_version=ENGINE_VERSION, metadata={"seed": 7}
    )
    watched.kernel.start()
    watched.kernel.advance(until=seconds(10))

    malformed = ("EMBER_OBJECTIVE_WR", 4)
    assert plain.cancel(malformed) == watched.cancel(malformed)
    assert plain.replace(malformed, 1) == watched.replace(malformed, 1)

    # And nothing unreplayable was written.
    assert [r.kind for r in read_records(watched.journal.to_bytes())] == []
