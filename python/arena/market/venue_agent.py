"""The venue as a participant in the simulation.

Gives a multi-instrument :class:`~arena.market.venue.Venue` a mailbox, so agents
reach it through the kernel and therefore through latency. Supersedes the
single-instrument ``ExchangeAgent``: same responsibilities, plus symbol routing
and the account layer.

The layering is unchanged and deliberate:

    matching        one MatchingEngine per symbol, still a pure function of its
                    own command stream, so the differential test against a
                    second implementation remains statable per symbol
    accounting      the Venue: positions, collateral, settlement
    timing          the kernel: latency, ordering, wakeups
    this module     the mailbox that joins them

Private events go out before public ones, as on a real venue: an agent learns of
its own fill before the tape learns a trade happened. The reverse would let a
subscriber react to a print before its counterparty knew it had traded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arena.exchange.events import (
    Acknowledged,
    Cancel,
    Cancelled,
    Event,
    Filled,
    Rejected,
    Replace,
    Replaced,
    Submit,
    Traded,
)
from arena.exchange.types import AgentId, Quantity, Side
from arena.market.venue import SymbolCommand, Venue
from arena.sim.kernel import SimulationContext
from arena.sim.messages import (
    DepthUpdate,
    Feed,
    PrivateEvent,
    Subscribe,
    TopOfBook,
    TradePrint,
    Unsubscribe,
)
from arena.sim.time import Duration, Timestamp

__all__ = ["VenueAgent"]


@dataclass(slots=True)
class _Subscription:
    feed: Feed
    symbol: str | None
    throttle: int
    last_sent: int = -1
    # The symbol whose update is being held back, if any. One slot, because
    # conflation keeps the *latest* state rather than a queue of stale ones.
    held: str | None = None

    def covers(self, symbol: str) -> bool:
        return self.symbol is None or self.symbol == symbol


class VenueAgent:
    """A multi-instrument venue with a mailbox and market-data feeds."""

    def __init__(
        self, agent_id: AgentId, venue: Venue, depth_levels: int = 8
    ) -> None:
        self.agent_id = agent_id
        self.venue = venue
        self.depth_levels = depth_levels
        self._subscriptions: dict[AgentId, list[_Subscription]] = {}
        # Every public event in order, for the research harness and the UI.
        # Agents never read this; they only see what reaches their mailbox.
        self.public_log: list[tuple[Timestamp, Any]] = []
        self.max_log = 5_000
        self._wakeup_pending = False

    # -- lifecycle ---------------------------------------------------------

    def on_start(self, ctx: SimulationContext) -> None:
        pass

    def on_finish(self, ctx: SimulationContext) -> None:
        pass

    def on_wakeup(self, ctx: SimulationContext) -> None:
        """Flush conflated feeds whose quiet period has elapsed.

        The only thing this agent does on its own schedule, and it has to: a
        conflated subscriber that has an update held back is owed it, and if
        the book never changes again nothing else will ever deliver it.
        """
        self._wakeup_pending = False
        due = int(ctx.now)
        soonest: int | None = None
        for recipient in sorted(self._subscriptions):
            for subscription in self._subscriptions[recipient]:
                if subscription.held is None:
                    continue
                ready = subscription.last_sent + subscription.throttle
                if ready <= due:
                    self._send_snapshot(
                        ctx, recipient, subscription.feed, subscription.held
                    )
                    subscription.last_sent = due
                    subscription.held = None
                elif soonest is None or ready < soonest:
                    soonest = ready
        if soonest is not None:
            self._schedule_flush(ctx, soonest - due)

    def _schedule_flush(self, ctx: SimulationContext, delay: int) -> None:
        """One outstanding wakeup at a time, however many feeds are waiting."""
        if self._wakeup_pending:
            return
        self._wakeup_pending = True
        ctx.request_wakeup(Duration(max(1, int(delay))))

    # -- mailbox -----------------------------------------------------------

    def on_message(self, ctx: SimulationContext, sender: AgentId, message: Any) -> None:
        if isinstance(message, Subscribe):
            self._subscriptions.setdefault(sender, []).append(
                _Subscription(message.feed, message.symbol, message.throttle)
            )
            # An immediate snapshot, so a new subscriber is not blind until
            # something happens to move a book.
            for symbol in self._symbols_for(message.symbol):
                self._send_snapshot(ctx, sender, message.feed, symbol)
            return

        if isinstance(message, Unsubscribe):
            existing = self._subscriptions.get(sender, [])
            self._subscriptions[sender] = [
                s
                for s in existing
                if not (s.feed is message.feed and s.symbol == message.symbol)
            ]
            return

        if isinstance(message, SymbolCommand):
            self._handle(ctx, sender, message.symbol, message.command)
            return

    def _symbols_for(self, symbol: str | None) -> tuple[str, ...]:
        if symbol is None:
            return self.venue.registry.symbols
        return (symbol,) if self.venue.registry.get(symbol) else ()

    def _handle(
        self, ctx: SimulationContext, sender: AgentId, symbol: str, command: Any
    ) -> None:
        if not isinstance(command, (Submit, Cancel, Replace)):
            return
        if command.agent_id != sender:
            # An agent may act only for itself. Trusting the field would let one
            # agent cancel another's orders by writing a different id.
            from arena.exchange.types import RejectReason, SequenceNumber

            ctx.send(
                sender,
                PrivateEvent(
                    Rejected(
                        SequenceNumber(0), sender, RejectReason.NOT_ORDER_OWNER, None
                    )
                ),
            )
            return

        events = self.venue.submit(sender, symbol, command)
        self.dispatch(ctx, symbol, events)

    def uncross(self, ctx: SimulationContext, symbol: str) -> Any:
        """Clear a call phase and tell the participants what happened to them.

        Routed through here rather than called on the venue directly, because
        this is the thing that owns the mailbox. An auction that books fills
        into the ledger without delivering them leaves every filled agent
        trading on a position it does not know it has.
        """
        result, events = self.venue.uncross_events(symbol)
        self.dispatch(ctx, symbol, events)
        return result

    def dispatch(self, ctx: SimulationContext, symbol: str, events: list[Event]) -> None:
        """Deliver private events to their owners, then publish the public ones."""
        for event in events:
            owner = _owner(event)
            if owner is not None:
                ctx.send(owner, PrivateEvent(event, symbol))

        traded = False
        for event in events:
            if isinstance(event, Traded):
                traded = True
                print_ = TradePrint(
                    symbol=symbol,
                    timestamp=ctx.now,
                    sequence=event.sequence,
                    price=event.price,
                    quantity=event.quantity,
                    aggressor_side=event.aggressor_side,
                )
                self._log(ctx.now, print_)
                self._broadcast(ctx, Feed.TRADES, symbol, print_)

        if events:
            self._publish_quotes(ctx, symbol, include_depth=True, traded=traded)

    # -- market data -------------------------------------------------------

    def _publish_quotes(
        self, ctx: SimulationContext, symbol: str, include_depth: bool, traded: bool
    ) -> None:
        top = self.top_of_book(symbol, ctx.now)
        self._log(ctx.now, top)
        self._broadcast(ctx, Feed.TOP_OF_BOOK, symbol, top)
        if include_depth:
            self._broadcast(
                ctx,
                Feed.DEPTH,
                symbol,
                DepthUpdate(
                    symbol,
                    ctx.now,
                    self.venue.engine(symbol).book.snapshot(self.depth_levels),
                ),
            )

    def top_of_book(self, symbol: str, now: Timestamp) -> TopOfBook:
        """The touch, as the market is entitled to see it.

        `best_priced`, not `best_price`. Market-on-open interest rests at a
        sentinel so that it crosses every candidate in the auction, which makes
        it the top of the book by a margin of 2^61 while naming no price at
        all. The matching engine must see it; a market-data subscriber must
        not, and `BookSnapshot.best_bid` has always agreed, so the depth feed
        was already correct and only the top-of-book feed was not.

        Measured on seed 7 over 20 simulated seconds before the fix: 78,742
        published touches sat outside the contract's own settlement range,
        across all 47 symbols, each one exactly 4,611,686,018,427,387,904
        ticks. Every agent in the market takes its `LocalBook` from this, so a
        strategy marking its book against it reported equity of 1.7e18 on an
        account of 400,000.
        """
        book = self.venue.engine(symbol).book
        bid = book.best_priced(Side.BUY)
        ask = book.best_priced(Side.SELL)
        return TopOfBook(
            symbol=symbol,
            timestamp=now,
            bid=bid,
            bid_size=book.depth_at(Side.BUY, bid) if bid is not None else Quantity(0),
            ask=ask,
            ask_size=book.depth_at(Side.SELL, ask) if ask is not None else Quantity(0),
        )

    def _broadcast(
        self, ctx: SimulationContext, feed: Feed, symbol: str, message: Any
    ) -> None:
        """Send to every subscriber, each after their own latency.

        Recipients are iterated in sorted order so the sequence of sends (and
        so the kernel sequence numbers that break timestamp ties) does not
        depend on the order agents happened to subscribe in.
        """
        for recipient in sorted(self._subscriptions):
            for subscription in self._subscriptions[recipient]:
                if subscription.feed is not feed or not subscription.covers(symbol):
                    continue
                if subscription.throttle > 0:
                    elapsed = int(ctx.now) - subscription.last_sent
                    if elapsed < subscription.throttle:
                        # Held, not dropped. A conflated feed sends the latest
                        # state at the next allowed moment; it does not decide
                        # the subscriber did not need to know.
                        #
                        # Dropping was the first version and it deadlocked the
                        # market. A maker that has not moved its quote sends no
                        # order, so nothing publishes; if the one update that
                        # announced the book was thrown away, every agent
                        # waiting for a price waited forever. Measured: an
                        # experiment trial went from 2,039 trades to **zero**.
                        subscription.held = symbol
                        self._schedule_flush(
                            ctx, subscription.throttle - elapsed
                        )
                        break
                subscription.last_sent = int(ctx.now)
                subscription.held = None
                ctx.send(recipient, message)
                break

    def _send_snapshot(
        self, ctx: SimulationContext, recipient: AgentId, feed: Feed, symbol: str
    ) -> None:
        if feed is Feed.TOP_OF_BOOK:
            ctx.send(recipient, self.top_of_book(symbol, ctx.now))
        elif feed is Feed.DEPTH:
            ctx.send(
                recipient,
                DepthUpdate(
                    symbol,
                    ctx.now,
                    self.venue.engine(symbol).book.snapshot(self.depth_levels),
                ),
            )

    def _log(self, now: Timestamp, message: Any) -> None:
        self.public_log.append((now, message))
        if len(self.public_log) > self.max_log:
            # Bounded: a live session runs indefinitely, and an unbounded log is
            # a memory leak with a long fuse.
            del self.public_log[: len(self.public_log) - self.max_log]


def _owner(event: Event) -> AgentId | None:
    if isinstance(event, (Acknowledged, Rejected, Filled, Cancelled, Replaced)):
        return event.agent_id
    return None
