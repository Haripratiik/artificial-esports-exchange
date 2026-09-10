"""What every trading agent shares.

An agent's view of the world is exactly what has reached its mailbox. It does
not read the venue, the book, or another agent's state. If it wants to know the
best bid it must have subscribed, and what it knows is what arrived, at the
time it arrived. That restriction is the whole point: an agent that could peek
at the book would make every latency and information-asymmetry result
meaningless, and the peek would be invisible in the output.

So this base class maintains a *local* view, updated only by messages, and
offers the small vocabulary an agent needs: quote, cancel, and know its own
position. Anything domain-specific (what a competitor's win rate will be,
whether a patch matters) belongs to the subclass and arrives through its own
feed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from arena.exchange.events import (
    Acknowledged,
    Cancel,
    Filled,
    Rejected,
    Replaced,
    Submit,
)
from arena.exchange.types import (
    AgentId,
    OrderId,
    OrderType,
    Price,
    Quantity,
    RejectReason,
    Side,
    TimeInForce,
)
from arena.market.instrument import Instrument
from arena.market.venue import SymbolCommand
from arena.sim.kernel import SimulationContext
from arena.sim.messages import (
    Feed,
    PrivateEvent,
    Subscribe,
    TopOfBook,
    TradePrint,
    Unsubscribe,
)
from arena.sim.time import Duration, Timestamp

__all__ = ["TradingAgent", "LocalBook"]


# The refusals that mean the order itself is gone, as opposed to the ones that
# mean an *amendment* to it was refused while it carried on resting.
#
# The distinction cannot be read off a ``Rejected`` event, because the engine
# reports a refused Submit and a refused Replace with the same shape and the
# same order id. It can be read off what the agent sent, which is why
# ``note_replace`` exists. These three are the reasons that are true about the
# order rather than about the command: an id the engine has never heard of, an
# order already finished, and an order belonging to somebody else are all
# statements that there is nothing of ours resting under that id.
_ORDER_IS_GONE = frozenset(
    {
        RejectReason.UNKNOWN_ORDER,
        RejectReason.ALREADY_TERMINAL,
        RejectReason.NOT_ORDER_OWNER,
    }
)


@dataclass(slots=True)
class LocalBook:
    """An agent's own picture of one symbol, as of the last message it received.

    Deliberately allowed to be stale. A market maker quoting off a view that is
    100ms old *is* the adverse-selection problem; modelling it away would
    delete the phenomenon the experiments are about.
    """

    symbol: str
    bid: Price | None = None
    ask: Price | None = None
    bid_size: int = 0
    ask_size: int = 0
    last: Price | None = None
    updated_at: int = 0

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None:
            return float(self.last) if self.last is not None else None
        return (int(self.bid) + int(self.ask)) / 2.0

    @property
    def spread(self) -> int | None:
        if self.bid is None or self.ask is None:
            return None
        return int(self.ask) - int(self.bid)


def _on_grid(instrument, side: Side, price: Price) -> Price:
    """Move a price onto the increment its band requires, conservatively.

    A bid rounds down and an offer rounds up, so snapping to the grid never
    makes a quote more aggressive than the agent intended: it gives up a
    fraction of a tick rather than crossing something the agent had not meant
    to cross.

    Contracts with one increment everywhere, which is all but one of them,
    return unchanged after a single comparison.
    """
    ticks = int(price)
    # Repeated, because a single pass can round *into* a coarser band and land
    # off its grid. Reading the increment at the original price and rounding
    # once is the obvious implementation and is wrong for any table with two
    # tiers: with steps of 1.00 from 100 and 5.00 from 203, an offer at 202.75
    # snaps up to 203.00, which is in the five-point band and not a multiple of
    # five, so the agent's own snapped quote comes back INVALID_PRICE.
    #
    # Each pass moves in one direction only and lands on a multiple of a
    # strictly coarser increment, so this cannot cycle. Bounded anyway, because
    # a loop in an order path is not a thing to leave unbounded.
    for _ in range(8):
        step = instrument.increment_at(instrument.from_ticks(Price(ticks)))
        if step == instrument.tick_size:
            return Price(ticks)
        per_step = int(step / instrument.tick_size)
        remainder = ticks % per_step
        if not remainder:
            return Price(ticks)
        ticks = ticks - remainder if side is Side.BUY else ticks + per_step - remainder
    return Price(ticks)


class TradingAgent:
    """Base for agents that trade on a venue through the kernel."""

    def __init__(
        self,
        agent_id: AgentId,
        venue_id: AgentId,
        instruments: dict[str, Instrument],
        wake_interval: Duration,
    ) -> None:
        self.agent_id = agent_id
        self.venue_id = venue_id
        # Copied, not shared. `build_market` hands the same mapping to every
        # agent, and each of them keeps its own books, positions and
        # subscriptions, so a shared listing was one agent's view of the market
        # standing in for everybody's.
        #
        # Harmless for as long as the listing never changed after construction,
        # which stopped being true when matches began opening and closing. The
        # first agent to join a new contract put it in every other agent's
        # instruments too, and each of them then iterated a symbol it had no
        # book, no position and no subscription for. Measured: a fundamental
        # trader raised KeyError on `OBJECTIVE0_ELIM_BASTION_GT0`, a contract it
        # had never been told about, on the first wake after a match listed.
        self.instruments = dict(instruments)
        self.wake_interval = wake_interval
        self.books: dict[str, LocalBook] = {
            symbol: LocalBook(symbol) for symbol in instruments
        }
        # Orders this agent believes are live. Believes, not knows: an
        # acknowledgement may still be in flight, and a fill may not have
        # arrived yet. Reconciling that belief with reality is a real trading
        # problem and is left visible rather than assumed away. Working orders,
        # keyed by **(symbol, order id)**.
        #
        # Order ids come from the matching engine, and there is one engine per
        # symbol, so id 5 exists on every book at once. Keyed by id alone (as
        # this was), an agent working orders on twenty-six contracts loses
        # track of almost all of them: a new acknowledgement overwrites the
        # entry for a different symbol's order, and completing one puts its id
        # in ``_completed``, after which the acknowledgement for *another*
        # book's order with the same id is discarded as a late duplicate and
        # that order is never cancelled again.
        #
        # Measured before the fix, after two minutes: `mm-3` believed it had 8
        # working orders across the whole exchange and had **123 resting in one
        # book**. The stale ones formed a wall the price could not move
        # through, the maker's own position drifted from the venue's (it
        # believed +807 where the ledger said -63), and the cancel-to-trade
        # ratio (the thing docs/GAPS.md recorded as "nothing requotes fast
        # enough") was low because most quotes were never cancelled at all.
        self.live_orders: dict[tuple[str, OrderId], str] = {}
        # Which side each working order is on, so one side can be replaced
        # without disturbing the other.
        self.order_side: dict[tuple[str, OrderId], Side] = {}
        # The quote this agent believes it is working, per (symbol, side).
        self._intent: dict[tuple[str, Side], tuple[int, int]] = {}
        # Orders known to be finished, so a late message cannot revive one.
        self._completed: dict[tuple[str, OrderId], None] = {}
        # Amendments this agent has sent and not yet had answered, counted per
        # order. It is the only thing that tells a refused *replace*, after
        # which the original is still resting, from a refused *submit*, after
        # which nothing is. The engine reports both as ``Rejected`` carrying
        # the same order id, and an agent may not ask the book which it was.
        #
        # Measured, on an order resting for ten lots amended to a million: the
        # venue refused it INSUFFICIENT_COLLATERAL and left the order exactly
        # where it was. The engine held it, the venue held it and reserved
        # collateral against it, and the agent dropped it, after which
        # ``cancel`` answered "no such live order". Every guard on the replace
        # path ended in exposure its owner was no longer allowed to manage,
        # which is the outcome the venue's own rate limiter goes out of its way
        # to avoid, arriving by the other door.
        self._amending: dict[tuple[str, OrderId], int] = {}
        # Contracts that listed or settled while this agent was asleep.
        #
        # Written by whoever runs the listing and drained by the agent on its
        # own wakeup, because a subscription is recorded against whoever sent
        # it: if the operator subscribed on an agent's behalf, the feed would
        # be delivered to the operator and the agent would never see a price.
        # So the operator can only leave a note, and the agent acts on it with
        # its own context.
        self._joining: list[Instrument] = []
        self._leaving: list[str] = []
        self.position: dict[str, int] = dict.fromkeys(instruments, 0)
        self.fills = 0
        self.rejects = 0
        # Amendments answered, counted for the reason ``fills`` and ``rejects``
        # are: ``_numbered`` derives a monotonic, eviction-proof cursor from a
        # counter rather than from a position in a log that gets trimmed. A
        # replace is a third sequence and not a member of either of theirs:
        # fill 3, rejection 3 and amendment 3 are unrelated events.
        self.amendments = 0

    # -- lifecycle ---------------------------------------------------------

    def on_start(self, ctx: SimulationContext) -> None:
        # Quotes conflated to this agent's own decision cadence; trades not.
        #
        # An agent cannot act on a book update that arrives between two of its
        # wakeups (by the time it looks, the update has been superseded), so
        # sending every one is work nobody uses. Real feeds conflate for
        # exactly this reason, and unconflated ones are a product you pay extra
        # for.
        #
        # Measured: with twenty agents each subscribed to twenty-six books with
        # no conflation, market data alone was most of **887,000 events per
        # simulated minute**.
        #
        # Trades are not conflated. The tape is the record of what happened,
        # and the maker's anchor is an average over it: dropping prints would
        # not slow the market down, it would change what the market believes.
        throttle = int(self.wake_interval) // 2
        for symbol in sorted(self.instruments):
            ctx.send(self.venue_id, Subscribe(Feed.TOP_OF_BOOK, symbol, throttle))
            ctx.send(self.venue_id, Subscribe(Feed.TRADES, symbol))
        self.schedule_next(ctx)

    def note_listing(self, instrument: Instrument) -> None:
        """Tell this agent a contract exists. It joins on its own next wake."""
        if instrument.symbol not in self.instruments:
            self._joining.append(instrument)

    def note_delisting(self, symbol: str) -> None:
        """Tell this agent a contract has settled and will not trade again."""
        if symbol in self.instruments:
            self._leaving.append(symbol)

    def begin_trading(self, ctx: SimulationContext, instrument: Instrument) -> bool:
        """Start following a contract that listed after this agent started.

        Everything an agent knows about a symbol is built in `on_start`, which
        runs once, so a contract listed later is invisible to every agent
        already in the market: no local book, no subscription, and therefore no
        quote. That was fine while the listing was fixed for the life of a
        session. Matches open and close continuously, so it is not.

        Subscribed on the same terms as everything else, quotes conflated to
        half this agent's decision cadence and trades not conflated, because a
        symbol that arrived late is not a different kind of symbol.

        Returns whether this was new, so a caller can tell joining from
        re-joining without comparing dictionaries.
        """
        symbol = instrument.symbol
        if symbol in self.instruments:
            return False
        self.instruments[symbol] = instrument
        self.books[symbol] = LocalBook(symbol)
        self.position.setdefault(symbol, 0)
        throttle = int(self.wake_interval) // 2
        ctx.send(self.venue_id, Subscribe(Feed.TOP_OF_BOOK, symbol, throttle))
        ctx.send(self.venue_id, Subscribe(Feed.TRADES, symbol))
        return True

    def stop_trading(self, ctx: SimulationContext, symbol: str) -> bool:
        """Let a settled contract go, and stop paying for its feed.

        A match settles and never trades again, so an agent that kept its
        subscription would carry the whole history of the season in its books
        and pay the fan-out cost of every symbol that ever listed. Measured on
        this market, six extra active agents cost about a third of throughput;
        a thousand dead symbols would cost far more and buy nothing.
        """
        if symbol not in self.instruments:
            return False
        ctx.send(self.venue_id, Unsubscribe(Feed.TOP_OF_BOOK, symbol))
        ctx.send(self.venue_id, Unsubscribe(Feed.TRADES, symbol))
        self.instruments.pop(symbol, None)
        self.books.pop(symbol, None)
        return True

    def on_finish(self, ctx: SimulationContext) -> None:
        pass

    def schedule_next(self, ctx: SimulationContext) -> None:
        """Wake on a jittered interval.

        Jittered because a population of agents on an identical cadence would
        synchronise into artificial waves of order flow that no real market
        produces, and those waves would then show up in every microstructure
        measurement as if they were real.
        """
        jitter = ctx.rng.uniform(0.7, 1.3)
        ctx.request_wakeup(Duration(max(1, int(int(self.wake_interval) * jitter))))

    def on_wakeup(self, ctx: SimulationContext) -> None:
        self._drain_listings(ctx)
        self.act(ctx)
        self.schedule_next(ctx)

    def _drain_listings(self, ctx: SimulationContext) -> None:
        """Join and leave whatever changed while this agent was asleep.

        Before acting rather than after, so a contract that listed this tick is
        one this agent can already quote, rather than one it learns about a
        wake later. Leaving comes first: a symbol that listed and settled
        inside a single sleep should end where it started, not resting.
        """
        while self._leaving:
            self.stop_trading(ctx, self._leaving.pop())
        while self._joining:
            self.begin_trading(ctx, self._joining.pop())

    def act(self, ctx: SimulationContext) -> None:
        """What the agent does when it wakes. Subclasses implement this."""

    # -- messages ----------------------------------------------------------

    def on_message(self, ctx: SimulationContext, sender: AgentId, message: Any) -> None:
        if isinstance(message, TopOfBook):
            book = self.books.get(message.symbol)
            if book is not None:
                book.bid, book.ask = message.bid, message.ask
                book.bid_size = int(message.bid_size)
                book.ask_size = int(message.ask_size)
                book.updated_at = int(ctx.now)
            self.on_quote(ctx, message)
        elif isinstance(message, TradePrint):
            book = self.books.get(message.symbol)
            if book is not None:
                book.last = message.price
            self.on_print(ctx, message)
        elif isinstance(message, PrivateEvent):
            self._on_private(ctx, message.event, message.symbol)

    def _on_private(self, ctx: SimulationContext, event: Any, symbol: str) -> None:
        order_id = getattr(event, "order_id", None)
        key = None if order_id is None else (symbol, order_id)

        if isinstance(event, Acknowledged):
            # A late acknowledgement must not resurrect an order that has
            # already finished. The kernel now preserves per-link ordering so
            # this should not arise, but an agent whose book of working orders
            # depends on message ordering being perfect is an agent that will
            # eventually be wrong about its own risk.
            if key not in self._completed:
                self.live_orders[key] = symbol
                self.order_side[key] = event.side
                # What actually rested, which is not always what was asked
                # for: an order beyond the price band slides to its edge. The
                # intent has to record the *result*, or the agent believes it
                # is quoting where it wanted, sees no reason to requote, and
                # leaves a slid order sitting at the band for the rest of the
                # session.
                if event.price is not None:
                    self._intent[(symbol, event.side)] = (
                        int(event.price),
                        int(event.quantity),
                    )
        elif isinstance(event, Filled):
            self.fills += 1
            # The symbol comes from the event, not from a lookup in
            # live_orders. Looking it up would silently drop the fill if the
            # acknowledgement had not arrived yet, and a dropped fill means the
            # agent's position diverges from the venue's, which is the worst
            # class of bug this system can have.
            signed = int(event.quantity) * (1 if event.side is Side.BUY else -1)
            self.position[symbol] = self.position.get(symbol, 0) + signed
            if int(event.remaining) == 0:
                self._complete(key)
        elif isinstance(event, Replaced):
            # A replace keeps the order id and keeps the order. Falling through
            # to the branch below (which is where this landed, because
            # ``Replaced`` is neither an ack nor a fill nor a refusal) treated
            # a successful amendment as the end of the order.
            #
            # Measured, on a bid for ten amended to six at the same price: the
            # engine held it resting for six, the venue held it and reserved
            # collateral against it, and the agent's working orders went
            # **empty**. ``cancel`` and ``replace`` then both answered "no such
            # live order", the snapshot's blotter showed nothing, and
            # ``cancel_all`` walked a list the order was no longer in and left
            # it standing in the book.
            #
            # It is worse for a peg, which is the whole point of a peg: the
            # engine emits a ``Replaced`` every time the reference moves, so an
            # order pegged to the bid became uncancellable at its **first**
            # reprice. Measured, two lots resting at 4,889.00 with the venue
            # reserving against them and the agent tracking none of it.
            #
            # Counted before anything else, and unconditionally, because the
            # counter is a cursor over the log and the log gets one entry per
            # private event whatever this method then decides about it.
            self.amendments += 1
            self._resolve_amendment(key)
            if key is not None and key not in self._completed:
                self.live_orders[key] = symbol
                side = self.order_side.get(key)
                if side is not None:
                    # The result, not the request: the same argument the
                    # acknowledgement above makes. A peg that reprices has
                    # moved the quote this agent believes it is working, and an
                    # intent left pointing at the old price is an agent that
                    # sees no reason to requote an order that is no longer
                    # there.
                    self._intent[(symbol, side)] = (
                        int(event.price),
                        int(event.quantity),
                    )
        elif isinstance(event, Rejected):
            self.rejects += 1
            # A refusal is not always a removal, and which one it is depends on
            # what was refused rather than on anything in the event. The engine
            # refuses a replace it dislikes and leaves the original exactly
            # where it was, the same fact ``Venue._track_working`` records,
            # which it settles by asking the engine's book. An agent may not
            # ask the book, so it asks itself: an outstanding amendment on this
            # order, refused for a reason that is about the command rather than
            # about the order, leaves the order resting.
            #
            # ``key in self.live_orders`` covers the race the other way. A fill
            # that finished the order while the amendment was in flight has
            # already completed it, and a refusal arriving afterwards must not
            # bring it back.
            amended = self._resolve_amendment(key)
            if not (
                amended
                and key in self.live_orders
                and event.reason not in _ORDER_IS_GONE
            ):
                self._complete(key)
        else:
            self._complete(key)
        self.on_private(ctx, event)

    def note_replace(self, symbol: str, order_id: OrderId) -> None:
        """Record that this agent has sent an amendment for one order.

        **Anything that sends a ``Replace`` on this agent's behalf must call
        this.** ``HumanAgent.enqueue`` does it by reading the command it is
        about to send, which is why ``LiveMarket.replace`` does not have to
        remember to. An agent that skips it still trades correctly; what it
        loses is the ability to tell a refused *amendment* (after which its
        order is still resting) from a refused *order*, and it will drop a live
        order from its own book the first time a guard fires.

        Counted rather than flagged, so two amendments in flight on one order
        are answered by two events and the second answer does not read as an
        answer to a command nobody sent.
        """
        key = (symbol, order_id)
        self._amending[key] = self._amending.get(key, 0) + 1
        if len(self._amending) > 4096:
            # Bounded for the reason ``_completed`` is. An amendment goes
            # unanswered when the command never reached a book at all (an
            # unknown symbol is refused without an order id), and a long
            # session must not accumulate those without limit.
            for stale in list(self._amending)[:2048]:
                del self._amending[stale]

    def _resolve_amendment(self, key: Any) -> bool:
        """Consume one outstanding amendment. True if there was one to consume."""
        if key is None:
            return False
        outstanding = self._amending.get(key, 0)
        if outstanding <= 0:
            return False
        if outstanding > 1:
            self._amending[key] = outstanding - 1
        else:
            del self._amending[key]
        return True

    def _complete(self, key: Any) -> None:
        if key is None:
            return
        self.live_orders.pop(key, None)
        self.order_side.pop(key, None)
        # Nothing further can be answered about an order that is finished, so
        # any amendment still counted against it never will be. Dropped here
        # rather than left to the bound, or an order filled while two replaces
        # were in flight leaks an entry for the rest of the session.
        self._amending.pop(key, None)
        # A dict rather than a set, purely for its insertion order. Trimming a
        # set to "the last 2048" is a sentence with no meaning (``list(set)``
        # yields hash order), so the old bound discarded an arbitrary half of
        # the record and could resurrect a finished order at any time.
        self._completed.pop(key, None)
        self._completed[key] = None
        if len(self._completed) > 4096:
            # Bounded: a long session would otherwise grow this without limit,
            # and only recent ids can plausibly be re-acknowledged.
            for stale in list(self._completed)[:2048]:
                del self._completed[stale]

    def on_quote(self, ctx: SimulationContext, quote: TopOfBook) -> None:
        """A quote update arrived. Override to react."""

    def on_print(self, ctx: SimulationContext, print_: TradePrint) -> None:
        """A trade printed. Override to react."""

    def on_private(self, ctx: SimulationContext, event: Any) -> None:
        """A private event arrived. Override to react."""

    # -- actions -----------------------------------------------------------

    def quote(
        self,
        ctx: SimulationContext,
        symbol: str,
        side: Side,
        price: Price,
        quantity: int,
        tif: TimeInForce = TimeInForce.GTC,
    ) -> None:
        """Send a limit order, clamped to the instrument's settlement range.

        Clamping rather than rejecting: an agent quoting outside the range is
        asking for something the contract cannot pay, and the sensible venue
        behaviour is to refuse it. But an agent that computed a price slightly
        past the boundary is usually right about direction and wrong about
        magnitude, so pinning it to the boundary keeps it in the market instead
        of silently dropping its order.
        """
        instrument = self.instruments[symbol]
        low, high = instrument.tick_bounds
        bounded = Price(max(int(low), min(int(high), int(price))))
        bounded = _on_grid(instrument, side, bounded)
        ctx.send(
            self.venue_id,
            SymbolCommand(
                symbol,
                Submit(
                    self.agent_id,
                    side,
                    Quantity(max(1, quantity)),
                    bounded,
                    OrderType.LIMIT,
                    tif,
                ),
            ),
        )

    def post(
        self,
        ctx: SimulationContext,
        symbol: str,
        side: Side,
        price: Price,
        size: int,
        tif: TimeInForce = TimeInForce.GTC,
    ) -> None:
        """Work a quote, replacing what is there only if it has moved.

        Cancelling and replacing an order at a price it is already at is the
        worst thing an agent can do with a message: it gives up queue priority
        for nothing and pays for the privilege. No real participant does it.

        Doing it here cost more than realism. Every agent cancelled and
        reposted on all twenty-six contracts on every wakeup, which produced
        **1.6 million events per simulated minute**, the market spending its
        time on paperwork rather than on trading. It was invisible until agents
        could track their own orders, because before that they lost most of
        them and most of the cancels were never sent: the bug was quietly
        acting as a rate limiter.
        """
        # Compared on the price that will actually be SENT, not the one that
        # was asked for. :meth:`quote` clamps into the settlement range and
        # snaps to the tick grid, so two different requested prices routinely
        # become the same resting price, and comparing the requests instead
        # reads that as a change, cancels a perfectly good order and reposts it
        # at the price it was already at. Measured on seed 7 over five minutes:
        # 2,463 of 62,424 quotes, 3.9%, were exactly that. Clamping and
        # snapping here first costs one extra call and is idempotent, since
        # `quote` repeats both on a price that already satisfies them.
        instrument = self.instruments[symbol]
        low, high = instrument.tick_bounds
        final = _on_grid(
            instrument, side, Price(max(int(low), min(int(high), int(price))))
        )
        wanted = (int(final), int(size))
        working = any(
            order_symbol == symbol and self.order_side.get(key) is side
            for key, order_symbol in self.live_orders.items()
        )
        if working and self._intent.get((symbol, side)) == wanted:
            return
        self.cancel_side(ctx, symbol, side)
        self._intent[(symbol, side)] = wanted
        self.quote(ctx, symbol, side, final, size, tif)

    def withdraw(self, ctx: SimulationContext, symbol: str, side: Side) -> None:
        """Pull this agent's quote on one side, if it has one."""
        if self.cancel_side(ctx, symbol, side):
            self._intent.pop((symbol, side), None)

    def cancel_side(self, ctx: SimulationContext, symbol: str, side: Side) -> bool:
        """Pull this agent's working orders on one side of one book.

        Separate from :meth:`cancel_all` because replacing a quote should not
        disturb the other side of it: an agent that cancels both sides to move
        one gives up its queue position on an order it was perfectly happy
        with, and pays for it twice, once in priority and once in messages.
        """
        pulled = False
        for key, order_symbol in list(self.live_orders.items()):
            if order_symbol != symbol or self.order_side.get(key) is not side:
                continue
            ctx.send(
                self.venue_id,
                SymbolCommand(order_symbol, Cancel(self.agent_id, key[1])),
            )
            self.live_orders.pop(key, None)
            self.order_side.pop(key, None)
            pulled = True
        return pulled

    def take(
        self, ctx: SimulationContext, symbol: str, side: Side, quantity: int
    ) -> None:
        """Cross the spread immediately, taking whatever is available."""
        ctx.send(
            self.venue_id,
            SymbolCommand(
                symbol,
                Submit(
                    self.agent_id,
                    side,
                    Quantity(max(1, quantity)),
                    None,
                    OrderType.MARKET,
                    TimeInForce.IOC,
                ),
            ),
        )

    def cancel_all(self, ctx: SimulationContext, symbol: str | None = None) -> None:
        for key, order_symbol in list(self.live_orders.items()):
            if symbol is not None and order_symbol != symbol:
                continue
            ctx.send(
                self.venue_id,
                SymbolCommand(order_symbol, Cancel(self.agent_id, key[1])),
            )
            self.live_orders.pop(key, None)
