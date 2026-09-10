"""A working market session on the exchange kernel.

    python examples/market_session.py

Not a research experiment: the agents here are deliberately trivial. It exists
to show the plumbing working end to end: orders routed through the kernel with
per-agent latency, matched by price-time priority, fills delivered privately,
prints broadcast publicly, and every subscriber seeing the same event at a
different time because their latencies differ.

That last point is the one worth watching in the output. The co-located maker
and the retail agent receive identical prints; the retail agent just receives
them ~100ms later, and that gap is what later phases turn into a measurable
dollar value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arena.exchange.events import Filled, Submit
from arena.exchange.types import AgentId, OrderType, Price, Quantity, Side, TimeInForce
from arena.sim.exchange_agent import ExchangeAgent
from arena.sim.kernel import Kernel, SimulationContext
from arena.sim.latency import PairwiseLatency
from arena.sim.messages import Feed, PrivateEvent, Subscribe, TopOfBook, TradePrint
from arena.sim.time import Duration, micros, millis, seconds, format_timestamp

EXCHANGE = AgentId("exchange")
FAIR_VALUE = 5000


def order(agent: AgentId, side: Side, price: int, qty: int) -> Submit:
    return Submit(agent, side, Quantity(qty), Price(price), OrderType.LIMIT, TimeInForce.GTC)


@dataclass
class SimpleMaker:
    """Quotes both sides around a fair value, skewing against its inventory.

    The simplest thing that deserves the name market maker: it earns the spread
    and it does not want to accumulate a position, so it shades its quotes to
    encourage the trade that flattens it. Nothing here is Avellaneda-Stoikov;
    that arrives with the real market-maker phase.
    """

    agent_id: AgentId
    half_spread: int = 3
    size: int = 20
    skew_per_lot: float = 0.05
    inventory: int = 0
    cash: int = 0
    fills: int = 0
    live: list[int] = field(default_factory=list)

    def on_start(self, ctx: SimulationContext) -> None:
        ctx.send(EXCHANGE, Subscribe(Feed.TOP_OF_BOOK))
        ctx.request_wakeup(millis(50))

    def on_wakeup(self, ctx: SimulationContext) -> None:
        from arena.exchange.events import Cancel

        for order_id in self.live:
            ctx.send(EXCHANGE, Cancel(self.agent_id, order_id))
        self.live.clear()

        skew = int(self.inventory * self.skew_per_lot)
        bid = FAIR_VALUE - self.half_spread - skew
        ask = FAIR_VALUE + self.half_spread - skew
        ctx.send(EXCHANGE, order(self.agent_id, Side.BUY, bid, self.size))
        ctx.send(EXCHANGE, order(self.agent_id, Side.SELL, ask, self.size))
        ctx.request_wakeup(millis(50))

    def on_message(self, ctx: SimulationContext, sender: AgentId, message: Any) -> None:
        if not isinstance(message, PrivateEvent):
            return
        event = message.event
        from arena.exchange.events import Acknowledged

        if isinstance(event, Acknowledged):
            self.live.append(int(event.order_id))
        elif isinstance(event, Filled):
            signed = int(event.quantity) if event.side is Side.BUY else -int(event.quantity)
            self.inventory += signed
            self.cash -= signed * int(event.price)
            self.fills += 1

    def on_finish(self, ctx: SimulationContext) -> None:
        pass

    @property
    def mark_to_market(self) -> int:
        return self.cash + self.inventory * FAIR_VALUE


@dataclass
class NoiseTrader:
    """Buys and sells at random. Provides the flow a maker earns its spread on.

    Not filler: without uninformed order flow there is nobody for a market maker
    to trade with profitably, and no camouflage for an informed trader to hide
    behind. Noise is load-bearing in every microstructure model this project
    cares about.
    """

    agent_id: AgentId
    interval: Duration = field(default_factory=lambda: millis(120))
    inventory: int = 0
    cash: int = 0
    trades: int = 0

    def on_start(self, ctx: SimulationContext) -> None:
        ctx.request_wakeup(Duration(int(ctx.rng.random() * int(self.interval)) + 1))

    def on_wakeup(self, ctx: SimulationContext) -> None:
        rng = ctx.rng
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        # Cross the spread: pay up to reach the maker's quote.
        price = FAIR_VALUE + (8 if side is Side.BUY else -8)
        ctx.send(EXCHANGE, order(self.agent_id, side, price, rng.randint(1, 6)))
        ctx.request_wakeup(self.interval)

    def on_message(self, ctx: SimulationContext, sender: AgentId, message: Any) -> None:
        if isinstance(message, PrivateEvent) and isinstance(message.event, Filled):
            event = message.event
            signed = int(event.quantity) if event.side is Side.BUY else -int(event.quantity)
            self.inventory += signed
            self.cash -= signed * int(event.price)
            self.trades += 1

    def on_finish(self, ctx: SimulationContext) -> None:
        pass

    @property
    def mark_to_market(self) -> int:
        return self.cash + self.inventory * FAIR_VALUE


@dataclass
class Watcher:
    """Subscribes and records nothing but arrival times. The latency probe."""

    agent_id: AgentId
    prints: list[tuple[int, int]] = field(default_factory=list)

    def on_start(self, ctx: SimulationContext) -> None:
        ctx.send(EXCHANGE, Subscribe(Feed.TRADES))

    def on_wakeup(self, ctx: SimulationContext) -> None:
        pass

    def on_message(self, ctx: SimulationContext, sender: AgentId, message: Any) -> None:
        if isinstance(message, TradePrint):
            self.prints.append((int(ctx.now), int(message.sequence), int(message.price)))

    def on_finish(self, ctx: SimulationContext) -> None:
        pass


def main() -> None:
    latency = PairwiseLatency(
        default=millis(2),
        per_agent={
            AgentId("maker"): micros(200),      # co-located
            AgentId("watcher_fast"): micros(200),
            AgentId("watcher_slow"): millis(100),  # retail, far away
        },
        jitter_fraction=0.1,
        seed=17,
    )
    kernel = Kernel(seed=17, latency=latency)

    exchange = ExchangeAgent(EXCHANGE, "VANTA_OBJECTIVE_WR")
    maker = SimpleMaker(AgentId("maker"))
    noise = [NoiseTrader(AgentId(f"noise{i}")) for i in range(8)]
    fast = Watcher(AgentId("watcher_fast"))
    slow = Watcher(AgentId("watcher_slow"))

    kernel.add_all([exchange, maker, fast, slow, *noise])
    kernel.run(until=seconds(30))

    book = exchange.snapshot()
    tape = exchange.tape
    prices = [int(t.price) for t in tape]

    print(f"instrument      {exchange.instrument}")
    print(f"session         {format_timestamp(kernel.now)}   {kernel.summary()}")
    print()
    print(f"trades printed  {len(tape)}")
    if prices:
        print(f"price range     {min(prices)} .. {max(prices)}   (fair value {FAIR_VALUE})")
        print(f"volume          {sum(int(t.quantity) for t in tape):,} lots")
    print(f"closing book    bid {book.best_bid}  ask {book.best_ask}  spread {book.spread}")
    print()

    print("market maker")
    print(f"  fills         {maker.fills}")
    print(f"  inventory     {maker.inventory:+d} lots")
    print(f"  cash          {maker.cash:+,}")
    print(f"  mark-to-mkt   {maker.mark_to_market:+,}")
    print()

    traded = sum(n.trades for n in noise)
    pnl = sum(n.mark_to_market for n in noise)
    print(f"noise traders   {len(noise)} agents, {traded} fills, "
          f"combined mark-to-market {pnl:+,}")
    print()

    print("latency probe: identical feed, different arrival times")
    print(f"  fast watcher  {len(fast.prints)} prints")
    print(f"  slow watcher  {len(slow.prints)} prints")

    # The slow watcher misses prints at BOTH ends, for different reasons worth
    # separating. At the open, its own subscription request took 100ms to reach
    # the exchange, so earlier trades were never sent to it. At the close,
    # prints dispatched before the bell were still in flight. Matching on the
    # feed's sequence number pairs the *same* trade on both sides. Pairing by
    # index, or by price, would silently compare different trades.
    fast_by_seq = {seq: t for t, seq, _price in fast.prints}
    slow_by_seq = {seq: t for t, seq, _price in slow.prints}
    shared = sorted(set(fast_by_seq) & set(slow_by_seq))

    missed_at_open = sum(1 for _, seq, _ in fast.prints if seq < min(slow_by_seq, default=0))
    lost_at_close = len(fast.prints) - len(shared) - missed_at_open

    if shared:
        gaps = sorted(slow_by_seq[seq] - fast_by_seq[seq] for seq in shared)
        print(f"  missed at open  {missed_at_open} (its subscription was still in flight)")
        print(f"  lost at close   {lost_at_close} (prints still in flight at the bell)")
        print(f"  matched prints  {len(shared)}")
        print(f"  median lag      {gaps[len(gaps) // 2] / 1e6:.1f} ms")
        print("  same prints, later arrival: the value of speed is now measurable")


if __name__ == "__main__":
    main()
