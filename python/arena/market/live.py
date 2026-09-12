"""A market that runs in wall-clock time, with a human as one of the agents.

Everything else in the project runs the kernel as fast as it will go, which is
right for experiments and useless for watching. This drives the same kernel in
slices synchronised to a real clock, so a browser can see the book move and a
person can put an order into it.

Two properties are preserved deliberately, because losing them would make the
live view a different system from the one under test:

**The human is an agent.** Their orders travel through the kernel, with a
latency, and reach the venue the same way an algorithm's do. They get no
privileged read of the book, no instant fills, and no exemption from the
collateral check. Watching a market you are exempt from teaches nothing.

**The engine is untouched.** Stepping is a property of the *kernel* (see
``Kernel.start``/``advance``/``finish``), not a special mode. The same seed,
replayed headless, produces the same tape.

The one thing that genuinely differs is the clock: real time advances whether or
not the queue has work, so an idle market still moves forward. A batch run would
simply jump to the next event.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from arena.agents.base import TradingAgent
from arena.exchange.events import Cancel, Replace, Submit
from arena.exchange.types import (
    AgentId,
    OrderType,
    PegReference,
    Price,
    Quantity,
    Side,
    TimeInForce,
)
from arena.market.instrument import Instrument
from arena.market.venue import SymbolCommand, Venue
from arena.market.venue_agent import VenueAgent
from arena.portfolio.money import from_money
from arena.sim.kernel import Kernel, SimulationContext
from arena.sim.messages import Feed, PrivateEvent, Subscribe, TopOfBook, TradePrint
from arena.sim.time import Duration, Timestamp, millis, seconds

__all__ = ["LiveMarket", "HumanAgent"]

VENUE_ID = AgentId("venue")
HUMAN_ID = AgentId("you")


def _distribution(instrument) -> dict[str, object] | None:
    """The payment stream, for a contract that has one.

    Only shares do. What a trader needs is how many payments are left and what
    each is written on, because that, not the settlement, is what they are
    buying.
    """
    schedule = instrument.spec.distribution
    if schedule is None:
        return None
    return {
        "periods": len(schedule.windows),
        "payoff": schedule.payoff.to_dict(),
        "first": schedule.windows[0].end.strftime("%Y-%m-%d"),
        "last": schedule.windows[-1].end.strftime("%Y-%m-%d"),
    }


def _indicative_price(venue, instrument, symbol: str) -> str | None:
    """Where an auction in progress would clear, or ``None`` if none is."""
    from arena.exchange.session import SessionState

    if venue.session(symbol) is SessionState.CONTINUOUS:
        return None
    result = venue.indicative(symbol)
    if result is None or result.volume <= 0:
        return None
    return str(instrument.from_ticks(result.price))


# What a journal has to agree with before its records may be replayed here.
#
# Not a package version. It identifies the deterministic behaviour of this
# market, so it has to change whenever a replay of the same inputs would
# produce a different exchange: the agent population, the seeding, the order
# the kernel runs things in, or the meaning of a recorded input. A journal
# whose header disagrees is refused rather than replayed, because the failure
# it prevents is silent: the same commands land in a different market and the
# rebuilt state is wrong in ways nothing downstream can detect.
ENGINE_VERSION = "arena-live-1"


class HumanAgent(TradingAgent):
    """A person at a browser, as an ordinary participant.

    Orders arrive here from the UI and are forwarded on the next wakeup rather
    than injected directly, so a human's action enters the event queue exactly
    like an agent's. It also means a human cannot act faster than their wakeup
    cadence, which is the honest analogue of a person's reaction time.

    One of these per person, not one per exchange. Sharing a single agent meant
    two browser tabs were one trader: the same account, the same blotter, the
    same working orders, and each tab able to cancel the other's.
    """

    def __init__(
        self,
        venue_id: AgentId,
        instruments: dict[str, Instrument],
        agent_id: AgentId = HUMAN_ID,
        display_name: str = "You",
    ) -> None:
        super().__init__(agent_id, venue_id, instruments, millis(50))
        self.display_name = display_name
        self._outbox: list[SymbolCommand] = []
        self.log: list[dict[str, Any]] = []

    def enqueue(self, command: SymbolCommand) -> None:
        """Queue one command for the next wakeup, noting what it is.

        An amendment is declared to the base class as it goes out, because the
        answer to one is indistinguishable from the answer to a new order once
        it comes back: the engine refuses a Submit and a Replace with the same
        ``Rejected`` shape and the same order id, and only the sender knows
        which it asked for. See ``TradingAgent.note_replace``. Read off the
        command rather than passed in by ``LiveMarket.replace``, so a second
        caller that builds its own ``Replace`` cannot forget to say so.
        """
        inner = command.command
        if isinstance(inner, Replace):
            self.note_replace(command.symbol, inner.order_id)
        self._outbox.append(command)

    def act(self, ctx: SimulationContext) -> None:
        pending, self._outbox = self._outbox, []
        for envelope in pending:
            ctx.send(self.venue_id, envelope)

    def _on_private(self, ctx: SimulationContext, event: Any, symbol: str) -> None:
        """Record the event for the blotter, then let the base book it.

        Overridden at the underscore level rather than at ``on_private``
        because only this one carries the symbol. Without it a blotter can say
        a fill happened but not in what, and the price would be a raw tick
        count, which for a contract on a 0.25 grid is four times the number a
        person expects to read.
        """
        super()._on_private(ctx, event, symbol)
        entry: dict[str, Any] = {"t": int(ctx.now), "symbol": symbol, **event.to_dict()}
        instrument = self.instruments.get(symbol)
        ticks = entry.get("price")
        if ticks is not None and instrument is not None:
            entry["price"] = str(instrument.from_ticks(int(ticks)))
        self.log.append(entry)
        if len(self.log) > 200:
            del self.log[:-200]


@dataclass
class LiveMarket:
    """Owns the kernel, the venue, the agents, and the wall-clock loop."""

    venue: Venue
    kernel: Kernel
    venue_agent: VenueAgent
    human: HumanAgent
    agents: list[TradingAgent] = field(default_factory=list)
    speed: float = 1.0
    # Everyone trading here, by account id. The first entry is the account a
    # visitor gets before signing in, which keeps every existing caller (and
    # every test) working unchanged.
    traders: dict[AgentId, HumanAgent] = field(default_factory=dict)
    latency: Any = None
    seat_cash: int = 0
    # The clock a contract's window is measured against, and the function that
    # produces a settlement once one closes. Both optional so every existing
    # caller and every test keeps working unchanged: without them the market
    # behaves exactly as it did, which is to say nothing ever expires.
    calendar: Any = None
    settlement_source: Any = None
    # Where exogenous input is written so a restart can rebuild this market.
    #
    # Optional, and `None` by default, so every existing caller and every test
    # behaves exactly as before. Nothing is journalled unless somebody asks for
    # it, and asking is one field.
    #
    # Only *exogenous* input goes here, which is the whole reason this is cheap.
    # Agent behaviour regenerates from the seed, because the kernel seeds each
    # agent from `_stable_seed(seed, "agent", agent_id)` and its draws do not
    # depend on join order. What the seed cannot regenerate is what came from
    # outside: somebody taking a seat, and the orders they sent. Those are the
    # six methods below and nothing else.
    journal: Any = None
    # Simulated seconds at the last clock record, so the heartbeat below stays
    # coarse instead of writing one per browser frame.
    _journalled_seconds: float = -1.0
    _wall_last: float = 0.0
    _sim_seconds: float = 0.0
    _running: bool = False
    _settlement_log: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.traders.setdefault(self.human.agent_id, self.human)

    # -- people ------------------------------------------------------------

    def seat(self, name: str) -> AgentId:
        """Give a person their own account, agent and blotter. Returns the id.

        Joining a running market rather than reserving a fixed number of seats
        up front. A pool would have been simpler and would have made "the
        exchange is full" a thing that could happen to someone, which is not a
        property of an exchange; it is a property of a workaround.

        The id is derived from the display name only for readability; the
        server hands out an opaque one. Two people with the same name get two
        accounts, because a name is not an identity.
        """
        self._record("seat", {"name": name})
        index = len(self.traders)
        agent_id = AgentId(f"you-{index}")
        while agent_id in self.traders:
            index += 1
            agent_id = AgentId(f"you-{index}")

        self.venue.open_account(agent_id, self.seat_cash or self.venue.starting_cash)
        agent = HumanAgent(
            self.venue_agent.agent_id,
            dict(self.human.instruments),
            agent_id=agent_id,
            display_name=name,
        )
        if self.latency is not None:
            # The same distance from the exchange as anyone else at a browser.
            self.latency.per_agent[agent_id] = self.latency.per_agent.get(
                self.human.agent_id, millis(20)
            )
        self.traders[agent_id] = agent
        self.agents.append(agent)
        self.kernel.join(agent)
        return agent_id

    def trader(self, agent_id: AgentId | None) -> HumanAgent:
        """The agent for an account id, falling back to the shared one."""
        if agent_id is None:
            return self.human
        return self.traders.get(agent_id, self.human)

    def start(self) -> None:
        self.kernel.start()
        self._wall_last = time.monotonic()
        self._sim_seconds = 0.0
        self._running = True

    def _record(self, kind: str, payload: dict[str, Any]) -> None:
        """Write one exogenous input, stamped with the clock replay will use.

        Simulated nanoseconds, not wall time. Replay advances the kernel to
        this stamp and then applies the command, which is what makes the rebuilt
        market the same market rather than a similar one: the wall clock a
        session ran against is not reproducible and does not need to be, but
        the simulated instant a command landed at is both.
        """
        if self.journal is None:
            return
        self.journal.append(kind, int(self._sim_seconds * 1_000_000_000), payload)

    def step(self) -> int:
        """Advance simulated time to match the wall clock. Returns events run.

        Simulated time is *accumulated* from each slice rather than recomputed
        from the session start, so that changing speed applies from the moment
        it changes. Recomputing rescaled the whole history instead: raising the
        speed made the clock leap forward past events already scheduled, and
        lowering it would have moved the clock **backwards**, which the kernel
        rightly refuses to do.
        """
        if not self._running:
            return 0
        now = time.monotonic()
        # Clamped so a stalled event loop, a suspended laptop, or a debugger
        # breakpoint does not hand the kernel an hour of catch-up to run in one
        # slice and freeze the browser it is meant to be serving.
        delta = min(1.0, max(0.0, now - self._wall_last))
        self._wall_last = now
        self._sim_seconds += delta * self.speed
        # A clock record, at most one per simulated second.
        #
        # Without it a journal records commands and not the passage of time, so
        # a rebuild lands at the last command rather than where the session
        # actually stopped. Measured on a session whose final input was at t=40
        # and which then ran to t=45: every account's positions differed,
        # because five seconds of agent activity had no record saying they
        # happened. The commands were all there and the market was still wrong.
        #
        # One a second rather than one a step: `step` runs on the browser's
        # frame clock, and a record per frame would be logging the wall clock
        # rather than the market.
        if self.journal is not None and self._sim_seconds - self._journalled_seconds >= 1.0:
            self._journalled_seconds = self._sim_seconds
            self._record("clock", {})
        target = Timestamp(int(self._sim_seconds * 1_000_000_000))
        # The contract calendar moves with simulated time, not wall time. Done
        # before the kernel runs so a contract whose window closes during this
        # slice is closed to new orders for the whole of it, rather than taking
        # orders for a slice against an outcome that is already determined.
        if self.calendar is not None:
            self.calendar.advance_to(self._sim_seconds)
        # A cap per slice, so a burst of activity cannot stall the event loop
        # that is serving the browser. Anything left simply runs next slice.
        ran = self.kernel.advance(until=target, max_events=20_000)
        self.settle_due()
        return ran

    def settle_due(self) -> list[str]:
        """Settle every contract whose window has closed. Returns what settled.

        This is the step that was missing, and its absence was the largest
        remaining gap in the product. Measured before it existed: after a
        simulated hour all 47 contracts were still `continuous` and the settled
        set was empty, so a position was marked forever and realised never. An
        algorithm left running had no terminal event to score against.

        The settlement machinery itself was complete and heavily tested the
        whole time. `build_market.prior_levels` already calls
        `settle(spec, oracle)` successfully on every listed instrument, which is
        the proof the oracle can answer. Nothing in the live path ever asked.

        Failures are recorded rather than raised. A contract the oracle cannot
        answer for is a real outcome (the evidence was never collected) and it
        must not take the market down with it; the venue's own settlement
        already distinguishes a VOID from a payout for exactly this reason.
        """
        if self.settlement_source is None:
            return []
        done: list[str] = []
        for symbol in tuple(self.venue.registry.symbols):
            if symbol in self.venue.settled_symbols:
                continue
            instrument = self.venue.registry.get(symbol)
            if instrument is None:
                continue
            if self.calendar is None or self.calendar.now() < instrument.expiry:
                continue
            try:
                result = self.settlement_source(instrument.spec)
                self.venue.settle(symbol, result)
            except Exception as failure:  # noqa: BLE001 (recorded, never fatal)
                self._settlement_log.append((symbol, repr(failure)))
                continue
            self._settlement_log.append(
                (symbol, getattr(result.status, "value", str(result.status)))
            )
            done.append(symbol)
        return done

    # -- human actions -----------------------------------------------------

    def submit(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: Decimal | None,
        tif: str = "",
        trader: AgentId | None = None,
        stop: Decimal | None = None,
        display: int = 0,
        *,
        peg: str | PegReference | None = None,
        peg_offset: int = 0,
    ) -> dict[str, Any]:
        """Send one order. ``peg`` makes it a pegged one; everything else is derived.

        The peg arguments are keyword-only and default to "not a peg", so every
        existing caller (the browser ticket, the REST route, the tests) keeps
        its current signature and its current behaviour exactly.

        ``peg`` is the engine's own vocabulary rather than a second one: it
        takes a :class:`PegReference` or one of its values (``bid``, ``ask``,
        ``mid``), read off the enum rather than restated, so a fourth reference
        added to the exchange is reachable from here without anybody editing
        this method. ``peg_offset`` stays a signed count of *ticks* for the
        same reason it is one on ``Submit``: on a contract carrying a tick
        table the same decimal is a different number of ticks at different
        levels, so an offset expressed as a price would silently change size as
        the reference moved, which is precisely the thing a peg exists to stop
        happening.
        """
        self._record(
            "submit",
            {
                "symbol": symbol,
                "side": side,
                "quantity": int(quantity),
                # Prices go in as strings for the same reason they cross the
                # wire as strings: a Decimal that becomes a float on the way to
                # disk comes back as a different price.
                "price": None if price is None else str(price),
                "tif": tif,
                "trader": None if trader is None else str(trader),
                "stop": None if stop is None else str(stop),
                "display": int(display),
                "peg": None if peg is None else getattr(peg, "value", str(peg)),
                "peg_offset": int(peg_offset),
            },
        )
        instrument = self.venue.registry.get(symbol)
        if instrument is None:
            return {"ok": False, "error": f"unknown symbol {symbol}"}
        if quantity <= 0:
            return {"ok": False, "error": "quantity must be positive"}

        try:
            ticks = None if price is None else instrument.to_ticks(price)
        except ValueError as bad_price:
            return {"ok": False, "error": str(bad_price)}

        try:
            stop_ticks = None if stop is None else instrument.to_ticks(stop)
        except ValueError as bad_stop:
            return {"ok": False, "error": str(bad_stop)}
        if display < 0:
            return {"ok": False, "error": "display size cannot be negative"}

        reference: PegReference | None = None
        if peg is not None:
            if isinstance(peg, PegReference):
                reference = peg
            else:
                try:
                    reference = PegReference(str(peg).strip().lower())
                except ValueError:
                    return {
                        "ok": False,
                        "error": f"unknown peg reference {peg!r}; this venue tracks "
                        + ", ".join(sorted(choice.value for choice in PegReference)),
                    }
        try:
            offset = int(peg_offset)
        except (TypeError, ValueError):
            return {"ok": False, "error": "peg offset must be a whole number of ticks"}
        if reference is None and offset:
            # The engine refuses this as INVALID_PEG and is right to: an offset
            # from nothing is not an instruction. Refused here so the caller is
            # told which of the two fields it forgot, rather than watching the
            # order vanish into a blotter.
            return {"ok": False, "error": "a peg offset needs a peg reference"}

        # A market order can only ever be immediate; a limit order defaults to
        # resting. Anything else the caller asks for is honoured, so the browser
        # can reach post-only and fill-or-kill rather than only the two defaults.
        #
        # A peg is checked before the other two because it looks like a market
        # order from here (it names no price) and the unpriced branch would
        # have made it immediate-or-cancel, which is the one instruction a peg
        # cannot obey.
        if reference is not None:
            if ticks is not None:
                return {
                    "ok": False,
                    "error": "a pegged order names no price of its own: its price is "
                    "the reference plus the offset",
                }
            if stop_ticks is not None:
                return {"ok": False, "error": "a pegged order cannot also be a stop"}
            try:
                duration = TimeInForce(tif.lower()) if tif else TimeInForce.GTC
            except ValueError:
                return {"ok": False, "error": f"unknown time in force {tif!r}"}
            if duration in (TimeInForce.IOC, TimeInForce.FOK):
                return {
                    "ok": False,
                    "error": f"a pegged order cannot be {duration.value}: a peg is an "
                    "instruction to keep tracking and that one says not to rest",
                }
            kind = OrderType.PEGGED
        elif stop_ticks is not None:
            duration = TimeInForce.GTC
            kind = OrderType.STOP_LIMIT if ticks is not None else OrderType.STOP
        elif ticks is None:
            duration = TimeInForce.IOC
            kind = OrderType.MARKET
        else:
            try:
                duration = TimeInForce(tif.lower()) if tif else TimeInForce.GTC
            except ValueError:
                return {"ok": False, "error": f"unknown time in force {tif!r}"}
            kind = OrderType.LIMIT

        who = self.trader(trader)
        command = Submit(
            who.agent_id,
            Side.BUY if side.lower() == "buy" else Side.SELL,
            Quantity(quantity),
            ticks,
            kind,
            duration,
            display,
            stop_ticks,
            peg_to=reference,
            peg_offset=offset if reference is not None else 0,
        )
        who.enqueue(SymbolCommand(symbol, command))
        return {"ok": True}

    def cancel(
        self, order_id: int, trader: AgentId | None = None, symbol: str | None = None
    ) -> dict[str, Any]:
        """Pull one working order.

        The symbol is part of the address, not decoration. Order ids come from
        the matching engine and there is one per book, so id 5 exists on every
        contract at once: cancelling by id alone means cancelling whichever of
        them happened to be found first. The client sends both because the
        blotter it is reading already shows both; without one, the lookup falls
        back to a search and refuses if it is ambiguous.
        """
        # Only a well-formed address is recorded. Callers reach this with a
        # `(symbol, id)` key as well as a bare id, and the lookup below refuses
        # the former on its own terms; coercing it here raised instead, turning
        # a refusal into a crash on a path that had worked for as long as it
        # existed. An input the venue cannot even address is not one a replay
        # could apply, so it is left out rather than recorded unusably.
        if isinstance(order_id, int):
            self._record(
                "cancel",
                {
                    "order_id": int(order_id),
                    "trader": None if trader is None else str(trader),
                    "symbol": symbol,
                },
            )
        who = self.trader(trader)
        if symbol is None:
            matches = [key for key in who.live_orders if key[1] == order_id]
            if len(matches) != 1:
                return {"ok": False, "error": "no such live order"}
            symbol = matches[0][0]
        if (symbol, order_id) not in who.live_orders:
            # Someone else's order, or none. Reported the same way either way:
            # confirming that an id exists but belongs to another account tells
            # a stranger something about that account.
            return {"ok": False, "error": "no such live order"}
        who.enqueue(SymbolCommand(symbol, Cancel(who.agent_id, order_id)))
        return {"ok": True}

    def replace(
        self,
        order_id: int,
        quantity: int,
        price: Decimal | None = None,
        trader: AgentId | None = None,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        """Amend a working order's price, its size, or both, keeping its id.

        The alternative available until now was cancel-and-resubmit, and it is
        strictly worse in two ways that matter. It loses queue priority
        *unconditionally*, including in the one case the engine would have kept
        it. And the two commands race a fill: between the cancel and the
        resubmit the order is not in the book at all, so a client amending a
        quote in a moving market is repeatedly out of the market for a round
        trip, or, if the cancel loses the race, ends up holding both.

        **What it costs in queue priority, measured rather than assumed.** Two
        bids for ten at 100, ours first, then a sell of five sweeps the level:

            shrink 10 -> 6 at the same price    kept_priority=True   we fill
            grow   10 -> 14 at the same price   kept_priority=False  they fill
            10 -> 10 at the same price          kept_priority=False  they fill
            10 -> 6 at a different price        kept_priority=False  they fill

        So the usual summary ("raising size loses it, lowering it keeps it") is
        right about the two ends and silent about the middle, and the middle is
        the case a client hits by accident. Priority survives a **strict
        reduction at an unchanged price and nothing else**: an amendment that
        re-sends the size it already had goes to the back of the queue for
        asking for nothing. Send only what is changing.

        Quantity is counted the way the engine counts it, against ``remaining``
        rather than against the original size, so amending a partly filled order
        for 4 leaves 4 lots working and not 4 minus what already traded.

        Addressed and owner-checked exactly as :meth:`cancel` is, and refused
        identically for an order that is somebody else's, already filled, or
        never existed. Confirming that an id exists but belongs to another
        account tells a stranger something about that account.
        """
        # Same guard as `cancel`, for the same reason: instrumentation must not
        # be able to fail a command that would otherwise have been refused.
        if isinstance(order_id, int):
            self._record(
                "replace",
                {
                    "order_id": int(order_id),
                    "quantity": int(quantity),
                    "price": None if price is None else str(price),
                    "trader": None if trader is None else str(trader),
                    "symbol": symbol,
                },
            )
        who = self.trader(trader)
        if symbol is None:
            matches = [key for key in who.live_orders if key[1] == order_id]
            if len(matches) != 1:
                return {"ok": False, "error": "no such live order"}
            symbol = matches[0][0]
        if (symbol, order_id) not in who.live_orders:
            return {"ok": False, "error": "no such live order"}

        instrument = self.venue.registry.get(symbol)
        if instrument is None:
            return {"ok": False, "error": f"unknown symbol {symbol}"}
        if quantity <= 0:
            # The engine refuses this as INVALID_QUANTITY and leaves the order
            # exactly where it was, which is right: an amendment to zero is not
            # a cancel and must not be read as one.
            return {"ok": False, "error": "quantity must be positive"}
        try:
            ticks = None if price is None else instrument.to_ticks(price)
        except ValueError as bad_price:
            return {"ok": False, "error": str(bad_price)}

        who.enqueue(
            SymbolCommand(
                symbol, Replace(who.agent_id, order_id, Quantity(quantity), ticks)
            )
        )
        return {"ok": True}

    def cancel_all(self, trader: AgentId | None = None) -> dict[str, Any]:
        """Pull every working order. Distinct from flatten, which closes risk."""
        self._record("cancel_all", {"trader": None if trader is None else str(trader)})
        who = self.trader(trader)
        for (symbol, order_id), _ in list(who.live_orders.items()):
            who.enqueue(SymbolCommand(symbol, Cancel(who.agent_id, order_id)))
        return {"ok": True}

    def flatten(self, trader: AgentId | None = None) -> dict[str, Any]:
        """Close every position at market. The panic button."""
        self._record("flatten", {"trader": None if trader is None else str(trader)})
        who = self.trader(trader)
        account = self.venue.account(who.agent_id)
        for symbol, position in sorted(account.positions.items()):
            if position.quantity == 0:
                continue
            side = Side.SELL if position.quantity > 0 else Side.BUY
            who.enqueue(
                SymbolCommand(
                    symbol,
                    Submit(
                        who.agent_id,
                        side,
                        Quantity(abs(position.quantity)),
                        None,
                        OrderType.MARKET,
                        TimeInForce.IOC,
                    ),
                )
            )
        return {"ok": True}

    # -- reporting ---------------------------------------------------------

    def snapshot(
        self, trader: AgentId | None = None, *, statics: bool = True
    ) -> dict[str, Any]:
        """Everything the UI needs, in one message, for one person.

        The books, the tape and the clock are the same for everybody; the
        account, the blotter, the working orders and the counterparties are
        not. Before there was one of each, so two browsers were one trader:
        they shared a balance, and either could cancel the other's orders.

        ``statics`` carries the fields of a listing that cannot change while it
        is listed: what class it is, its tick, its value bounds and its
        contract terms. They are in here at all because a screen showing a
        price without the terms is a casino with extra steps, and they were
        being sent again twenty times a second for the life of every
        connection.

        Measured on the default market, 358 symbols: a snapshot is 234 KB and
        builds in 20.5ms, of which the contract terms alone are 159.4 KB and
        the four static fields together are 167.7 KB, or 72 per cent. The
        depth ladders every one of those bytes is nominally in aid of come to
        1.4 KB. At twenty ticks a second that was 41 per cent of a core and
        4.7 MB/s for each browser watching, which is why two open tabs left
        every HTTP handler on this server queued behind the event loop for
        eight to fourteen seconds.

        So the socket sends them once and the client keeps them. Nothing is
        lost: a listing is immutable for its whole life, and the one event
        that replaces every listing, a rebuild, changes the generation the
        client is already watching.
        """
        who = self.trader(trader)
        marks = self.venue.marks()
        account = self.venue.account(who.agent_id)
        symbols = self.venue.registry.symbols

        # A settled contract is a record, not a market. The registry is
        # append-only on purpose, because a symbol binds to one contract for
        # its whole life and relisting it would silently change what an open
        # position settles into, so nothing ever leaves it. That is right for
        # the ledger and wrong for a screen: matches settle every few minutes
        # and their contracts stayed on the board forever.
        #
        # Measured on the default market with matches listed: at 30 simulated
        # seconds 3 of 70 match contracts had settled, at 90 seconds 70 of 140,
        # and at 180 seconds 140 of 210. Two thirds of what a trader was being
        # shown had already resolved and could not be traded, and the board
        # grew without bound for as long as the market ran.
        #
        # They are dropped here rather than delisted, so the venue keeps its
        # record and the account keeps its history. What a holder of a settled
        # position needs is in the blotter and the fills, where settlement
        # already paid out; what they do not need is a dead book.
        settled = frozenset(self.venue.settled_symbols)
        books = {}
        for symbol in symbols:
            if symbol in settled:
                continue
            instrument = self.venue.registry.require(symbol)
            snap = self.venue.engine(symbol).book.snapshot(8)
            books[symbol] = {
                # Priced levels only. Market-on-open interest rests at a
                # sentinel so it crosses every candidate the auction weighs;
                # putting it on a screen published a bid of
                # 4,611,686,018,427,387,904 and a spread to match.
                "bids": [
                    [str(instrument.from_ticks(p)), int(q)] for p, q in snap.priced_bids
                ],
                "asks": [
                    [str(instrument.from_ticks(p)), int(q)] for p, q in snap.priced_asks
                ],
                "mark": str(from_money(marks[symbol])),
                # A crossed book has no spread, and during a call phase it is
                # *meant* to be crossed: orders accumulate without matching
                # until the uncross. Reporting the arithmetic difference
                # printed "-10,000.00" on the screen of anyone who looked
                # before the open, which is not a spread and not a number
                # anyone should have to interpret.
                "spread": (
                    str(instrument.from_ticks(Price(snap.spread)))
                    if snap.spread is not None and snap.spread >= 0
                    else None
                ),
                "session": self.venue.session(symbol).value,
                # What the call would clear at, published during it exactly as
                # real venues publish an indicative price. So an agent, or a
                # person, can respond to the auction rather than only to its
                # result.
                "indicative": _indicative_price(self.venue, instrument, symbol),
                "trades": len(self.venue.engine(symbol).tape),
            }
            if statics:
                books[symbol].update(
                    {
                        "class": instrument.instrument_class,
                        "tick": str(instrument.tick_size),
                        # What the claim can be worth, not only what it settles
                        # at. A share settles at nothing because it has paid
                        # everything out, so the settlement range would say a
                        # share is worth zero to zero, true at the last instant
                        # and useless before it.
                        "bounds": [str(b) for b in instrument.value_bounds],
                        # What the contract actually is, so a trader can see the
                        # terms rather than only the price. A market where you
                        # cannot read the contract is a casino with extra steps.
                        "contract": {
                            "id": instrument.spec.contract_id,
                            "payoff": instrument.spec.payoff.to_dict(),
                            "underlying": instrument.spec.underlying.to_dict(),
                            "expiry": instrument.expiry.strftime("%Y-%m-%d"),
                            "digest": instrument.spec.spec_digest[7:19],
                            "distribution": _distribution(instrument),
                        },
                    }
                )

        recent = []
        for stamp, message in self.venue_agent.public_log[-60:]:
            if isinstance(message, TradePrint):
                instrument = self.venue.registry.require(message.symbol)
                recent.append(
                    {
                        "t": int(stamp),
                        "symbol": message.symbol,
                        "price": str(instrument.from_ticks(message.price)),
                        "quantity": int(message.quantity),
                        "side": message.aggressor_side.value,
                    }
                )

        positions = []
        for symbol in symbols:
            position = account.positions.get(symbol)
            if position is None or (position.quantity == 0 and position.volume == 0):
                continue
            positions.append(
                {
                    "symbol": symbol,
                    "quantity": position.quantity,
                    "average_price": str(round(position.average_price, 4)),
                    "unrealized": str(from_money(position.unrealized_pnl(marks[symbol]))),
                    "realized": str(from_money(position.realized_pnl)),
                }
            )

        return {
            "clock": int(self.kernel.now),
            "events": self.kernel.processed,
            "books": books,
            "tape": recent[::-1],
            "account": {
                "cash": str(from_money(account.cash)),
                "free_cash": str(from_money(account.free_cash)),
                "collateral": str(from_money(account.posted_collateral)),
                "equity": str(from_money(account.equity(marks))),
                "pnl": str(
                    from_money(account.equity(marks)) - from_money(account.starting_cash)
                ),
                "positions": positions,
            },
            "orders": [
                {"order_id": oid, "symbol": sym}
                for (sym, oid), _ in sorted(who.live_orders.items())
            ],
            "log": who.log[-12:][::-1],
            "you": {"id": str(who.agent_id), "name": who.display_name},
            "traders": [
                {"id": str(a.agent_id), "name": a.display_name}
                for a in self.traders.values()
            ],
            # Who actually took the other side of your orders. A simulated
            # exchange should be able to answer that with names.
            "counterparties": self.venue.counterparties_for(who.agent_id, limit=25),
            "conservation": str(self.venue.conservation_check()),
        }


def apply_input(market: LiveMarket, record: Any) -> None:
    """Re-apply one journalled input to a market built from the same seed.

    The clock moves first. A record carries the simulated nanosecond its
    command landed at, and the agents around it have to have run up to that
    instant before it is applied, or the order meets a different book than the
    one it originally met and the rebuild diverges from the first record on.

    Raises on a kind it does not recognise, which :func:`journal.replay`
    documents as this callback's job. Silently skipping an unknown record
    would rebuild a market missing exactly those inputs and report success,
    which is the failure mode a recovery path must not have.
    """
    seconds_in = record.timestamp_ns / 1_000_000_000
    if seconds_in > market._sim_seconds:
        market._sim_seconds = seconds_in
        if market.calendar is not None:
            market.calendar.advance_to(seconds_in)
        market.kernel.advance(until=Timestamp(int(record.timestamp_ns)))
        market.settle_due()

    body = record.payload
    kind = record.kind
    if kind == "clock":
        # The advance above was the whole of it. A heartbeat carries no
        # command; it exists so that time itself is replayable.
        return
    if kind == "seat":
        market.seat(body["name"])
    elif kind == "submit":
        market.submit(
            body["symbol"],
            body["side"],
            int(body["quantity"]),
            None if body["price"] is None else Decimal(body["price"]),
            body.get("tif", ""),
            body.get("trader"),
            None if body.get("stop") is None else Decimal(body["stop"]),
            int(body.get("display", 0)),
            peg=body.get("peg"),
            peg_offset=int(body.get("peg_offset", 0)),
        )
    elif kind == "cancel":
        market.cancel(int(body["order_id"]), body.get("trader"), body.get("symbol"))
    elif kind == "replace":
        market.replace(
            int(body["order_id"]),
            int(body["quantity"]),
            None if body["price"] is None else Decimal(body["price"]),
            body.get("trader"),
            body.get("symbol"),
        )
    elif kind == "cancel_all":
        market.cancel_all(body.get("trader"))
    elif kind == "flatten":
        market.flatten(body.get("trader"))
    else:
        raise ValueError(f"journal holds an input this market cannot apply: {kind!r}")


def rebuild(
    market: LiveMarket,
    source: Any,
    *,
    engine_version: str,
    until_sim_seconds: float | None = None,
    **kwargs: Any,
):
    """Replay a journal onto a market, without journalling it a second time.

    ``until_sim_seconds`` runs the market on to a known stopping point after
    the last record. The clock heartbeat makes this unnecessary for a session
    driven through :meth:`step`, and it is here for one driven by hand.

    Detaching the journal for the duration is not tidiness. `submit` and its
    five siblings record what they are asked to do, so a replay that left the
    journal attached would append every recovered input back onto the file it
    was reading, and the next recovery would apply each of them twice.
    """
    from arena.sim.journal import replay as _replay

    # The population has to exist before the first record lands. `on_start` is
    # what schedules every agent's first wakeup, so a replay onto an unstarted
    # kernel advances a market in which nobody trades: measured, the rebuilt
    # marks came back at exactly the midpoint of every settlement range, which
    # is the price of a book that has never printed.
    #
    # `kernel.start` rather than `market.start`, which would also reset the
    # clock to zero and switch on wall-clock stepping. Idempotent, so a caller
    # that has already started is not punished for it.
    market.kernel.start()

    attached, market.journal = market.journal, None
    try:
        outcome = _replay(
            source,
            lambda record: apply_input(market, record),
            engine_version=engine_version,
            **kwargs,
        )
        if until_sim_seconds is not None and until_sim_seconds > market._sim_seconds:
            market._sim_seconds = until_sim_seconds
            if market.calendar is not None:
                market.calendar.advance_to(until_sim_seconds)
            market.kernel.advance(
                until=Timestamp(int(until_sim_seconds * 1_000_000_000))
            )
            market.settle_due()
        return outcome
    finally:
        market.journal = attached
