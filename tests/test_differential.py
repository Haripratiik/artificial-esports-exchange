"""Differential testing: the real engine against a naive one.

The strongest assurance available for a matching engine short of a proof. The
production engine earns its speed with a price heap, lazy removal of stale heap
entries, price-level buckets carrying running totals, and cancellation by
tombstone: four optimisations, each a place a subtle bug could live and none of
them visible in the output when they go wrong.

So it is checked against an implementation with no optimisations at all: a flat
list, re-sorted from scratch on every operation. Both are fed the same command
stream and must produce the same trades, in the same order, at the same prices,
and finish with the same resting depth.

This is also the harness the C++ port will be validated with. It exists before
the port rather than being improvised alongside it, which means the port's
acceptance criterion is already written down and already known to pass for one
implementation.
"""

from __future__ import annotations

import random

import pytest

from arena.exchange.engine import MatchingEngine
from arena.exchange.events import Cancel, Replace, Submit, Traded
from arena.exchange.types import (
    AgentId,
    OrderType,
    Price,
    Quantity,
    Side,
    TimeInForce,
)
from tests.reference_matcher import NaiveTrade, ReferenceMatcher

AGENTS = [AgentId("a"), AgentId("b"), AgentId("c"), AgentId("d")]


def random_stream(seed: int, count: int, *, price_span: int = 12) -> list:
    """A command stream with the shape real flow has.

    Cancels and replaces are heavily represented on purpose. Most orders
    in an electronic market are cancelled rather than filled, and
    cancellation is exactly the path the production engine optimises with
    tombstones, so a stream without them would exercise none of the code
    most likely to be wrong.
    """
    rng = random.Random(seed)
    commands: list = []
    known: list[int] = []
    next_id = 0

    for _ in range(count):
        agent = rng.choice(AGENTS)
        roll = rng.random()

        if roll < 0.55:
            side = rng.choice([Side.BUY, Side.SELL])
            tif = rng.choices(
                [TimeInForce.GTC, TimeInForce.IOC, TimeInForce.FOK], [0.75, 0.18, 0.07]
            )[0]
            commands.append(
                Submit(
                    agent,
                    side,
                    Quantity(rng.randint(1, 25)),
                    Price(rng.randint(100 - price_span, 100 + price_span)),
                    OrderType.LIMIT,
                    tif,
                )
            )
            next_id += 1
            known.append(next_id)
        elif roll < 0.68:
            side = rng.choice([Side.BUY, Side.SELL])
            commands.append(
                Submit(
                    agent,
                    side,
                    Quantity(rng.randint(1, 15)),
                    None,
                    OrderType.MARKET,
                    TimeInForce.IOC,
                )
            )
            next_id += 1
            known.append(next_id)
        elif roll < 0.88 and known:
            commands.append(Cancel(agent, rng.choice(known)))
        elif known:
            commands.append(
                Replace(
                    agent,
                    rng.choice(known),
                    Quantity(rng.randint(1, 25)),
                    Price(rng.randint(100 - price_span, 100 + price_span))
                    if rng.random() < 0.7
                    else None,
                )
            )
    return commands


def real_trades(commands) -> list[NaiveTrade]:
    engine = MatchingEngine()
    engine.apply_all(commands)
    return [
        NaiveTrade(
            price=int(t.price),
            quantity=int(t.quantity),
            aggressor_side=t.aggressor_side.value,
            buy_order_id=int(t.buy_order_id),
            sell_order_id=int(t.sell_order_id),
        )
        for t in engine.tape
    ], engine


def naive_trades(commands) -> tuple[list[NaiveTrade], ReferenceMatcher]:
    matcher = ReferenceMatcher()
    matcher.apply_all(commands)
    return matcher.trades, matcher


SEEDS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]


@pytest.mark.parametrize("seed", SEEDS)
def test_the_two_engines_print_the_same_trades(seed):
    """Same commands in, same tape out. Prices, sizes, order ids, and order."""
    commands = random_stream(seed, 600)
    fast, _engine = real_trades(commands)
    slow, _matcher = naive_trades(commands)

    assert len(fast) == len(slow), (
        f"trade counts diverge at seed {seed}: {len(fast)} vs {len(slow)}"
    )
    for index, (a, b) in enumerate(zip(fast, slow)):
        assert a == b, f"trade {index} differs at seed {seed}:\n  fast {a}\n  slow {b}"


@pytest.mark.parametrize("seed", SEEDS)
def test_the_two_engines_leave_the_same_book(seed):
    """Resting depth must agree, which is what catches a leaked tombstone.

    A cancelled order that is skipped for matching but still counted in a level
    total would never show up in the tape, only in the depth.
    """
    commands = random_stream(seed, 600)
    _fast, engine = real_trades(commands)
    _slow, matcher = naive_trades(commands)

    book = engine.book.snapshot(levels=1 << 20)
    real_depth = {
        ("buy", int(price)): int(quantity) for price, quantity in book.bids
    } | {("sell", int(price)): int(quantity) for price, quantity in book.asks}

    assert real_depth == matcher.depth, f"resting depth diverges at seed {seed}"


@pytest.mark.parametrize("seed", [7, 11, 17])
def test_they_agree_on_a_thin_book(seed):
    """A narrow price range means constant crossing and level exhaustion.

    Levels emptying and being recreated is precisely when lazy heap cleanup can
    go wrong, so it is worth forcing rather than waiting for.
    """
    commands = random_stream(seed, 800, price_span=2)
    fast, engine = real_trades(commands)
    slow, matcher = naive_trades(commands)
    assert fast == slow

    book = engine.book.snapshot(levels=1 << 20)
    real_depth = {("buy", int(p)): int(q) for p, q in book.bids} | {
        ("sell", int(p)): int(q) for p, q in book.asks
    }
    assert real_depth == matcher.depth


@pytest.mark.parametrize("seed", [4, 9])
def test_they_agree_when_almost_everything_is_cancelled(seed):
    """Real books are mostly cancellations, and tombstoning is the risk there."""
    rng = random.Random(seed)
    commands: list = []
    next_id = 0
    for _ in range(700):
        agent = rng.choice(AGENTS)
        if rng.random() < 0.35 or next_id == 0:
            commands.append(
                Submit(
                    agent,
                    rng.choice([Side.BUY, Side.SELL]),
                    Quantity(rng.randint(1, 12)),
                    Price(rng.randint(95, 105)),
                    OrderType.LIMIT,
                    TimeInForce.GTC,
                )
            )
            next_id += 1
        else:
            commands.append(Cancel(agent, rng.randint(1, next_id)))

    fast, engine = real_trades(commands)
    slow, matcher = naive_trades(commands)
    assert fast == slow

    book = engine.book.snapshot(levels=1 << 20)
    real_depth = {("buy", int(p)): int(q) for p, q in book.bids} | {
        ("sell", int(p)): int(q) for p, q in book.asks
    }
    assert real_depth == matcher.depth


def test_the_harness_can_actually_detect_a_difference():
    """A differential test that cannot fail proves nothing.

    Feeding the two engines *different* streams must produce a mismatch, or the
    comparison above would pass no matter how wrong either engine was.
    """
    fast, _ = real_trades(random_stream(1, 300))
    slow, _ = naive_trades(random_stream(2, 300))
    assert fast != slow
