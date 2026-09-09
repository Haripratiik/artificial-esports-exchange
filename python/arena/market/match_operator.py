"""Matches that open, play out where everyone can watch, and settle as they go.

The statistical contracts on this exchange settle once, at the end of a four
week observation window. That is a real market and it is a slow one: nothing
resolves inside a session, so a trader has no terminal event to be scored
against and the exchange drains as contracts expire with nothing to replace
them.

A match fixes both. It opens, it runs for a few minutes, and it ends with an
outcome that is a fact. Matches are generated rather than collected, so there
is always another one, which is also why listing no longer runs out.

**The live part is not decoration, it is settlement arriving in pieces.** A
competitor who has been eliminated cannot win, so "that competitor wins" is
worth exactly zero and is settled the moment it happens, while everyone still
in keeps trading. Nothing here has to model an in-play market specially: an
elimination is a contract resolving early, and the venue has always been able
to resolve a contract. The mutually exclusive set still sums to one, because
the settled legs pay zero and the live ones carry the whole of it.

Only contracts whose value is already certain settle early. A competitor who is
out has definitively lost, and that is the whole of what an elimination tells
you: it says nothing final about where the survivors will place, so their
contracts stay open. Settling anything less than certain would be paying out a
forecast, which is the one thing a settlement may never be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from arena.exchange.types import AgentId
from arena.market.match_book import MatchBook, list_match, settlement
from arena.settlement.result import SettlementResult, SettlementStatus
from arena.sim.time import Duration, Timestamp, seconds
from arena.worlds.circuit.match import play
from arena.worlds.circuit.modes import FORMATS

__all__ = ["MatchOperator", "LiveMatch"]


@dataclass
class LiveMatch:
    """One match on the board, and how much of it has happened."""

    book: MatchBook
    format_name: str
    match_id: int
    opened_at: int
    closes_at: int
    # Finishing order worst-first, which is the order eliminations happen in.
    # Derived from the result rather than simulated a second time: the match is
    # already a pure function of its seed, and re-deriving it here would be a
    # second implementation that could disagree with the one that settles.
    knockout_order: tuple[str, ...]
    revealed: int = 0
    settled: set[str] = field(default_factory=set)

    @property
    def eliminated(self) -> tuple[str, ...]:
        return self.knockout_order[: self.revealed]


class MatchOperator:
    """Opens matches on a schedule, reveals them, and settles them.

    An agent rather than a venue feature, for the same reason `SessionOperator`
    is one: it acts on its own clock and everything it does is something a
    participant could in principle watch happen.
    """

    def __init__(
        self,
        agent_id: AgentId,
        venue: Any,
        seed: int,
        formats: tuple[str, ...] = ("solo", "objective"),
        match_seconds: float = 90.0,
        concurrent: int = 2,
        poll: Duration = seconds(2),
        venue_agent: Any = None,
    ) -> None:
        for name in formats:
            if name not in FORMATS:
                raise ValueError(f"no match format named {name!r} is registered")
        self.agent_id = agent_id
        self.venue = venue
        self.seed = seed
        self.formats = formats
        # Long enough that every agent wakes several times inside a match, or
        # the market is a single auction wearing a clock. The makers requote on
        # 300 to 480ms and the informed wake on 500 to 1350ms, so ninety
        # simulated seconds is dozens of decisions each.
        self.match_seconds = match_seconds
        self.concurrent = concurrent
        self.poll = poll
        self.venue_agent = venue_agent
        self.live: dict[str, LiveMatch] = {}
        self._next_id: dict[str, int] = {name: 0 for name in formats}
        self.opened = 0
        self.settled = 0

    # -- lifecycle ---------------------------------------------------------

    def on_start(self, ctx: Any) -> None:
        self._fill_board(ctx)
        ctx.request_wakeup(self.poll)

    def on_message(self, ctx: Any, sender: AgentId, message: Any) -> None:
        """Takes no instruction. Listed so the kernel can deliver to it."""

    def on_finish(self, ctx: Any) -> None:
        pass

    def on_wakeup(self, ctx: Any) -> None:
        now = int(ctx.now)
        for key in tuple(self.live):
            self._advance(ctx, self.live[key], now)
        self._fill_board(ctx)
        ctx.request_wakeup(self.poll)

    # -- the board ---------------------------------------------------------

    def _fill_board(self, ctx: Any) -> None:
        """Keep `concurrent` matches of each format open at all times.

        Topped up rather than scheduled, so a match that ends early or a
        session that runs long does not leave the board empty. An exchange with
        nothing listed is not a quiet exchange, it is a closed one.
        """
        now = int(ctx.now)
        for name in self.formats:
            live = sum(1 for m in self.live.values() if m.format_name == name)
            for _ in range(max(0, self.concurrent - live)):
                self._open(ctx, name, now)

    def _open(self, ctx: Any, format_name: str, now: int) -> None:
        match_id = self._next_id[format_name]
        self._next_id[format_name] = match_id + 1
        book = list_match(self.seed, match_id, format_name)
        result = play(self.seed, match_id, format_name)

        # Worst placement first, which is the order they went out in. A team
        # format eliminates nobody, so its board is empty and the whole match
        # settles at the end.
        if FORMATS[format_name].team_size == 1:
            order = tuple(
                sorted(result.placements, key=lambda k: -result.placements[k])
            )[:-1]
        else:
            order = ()

        for contract in book.contracts:
            self.venue.list_instrument(contract.instrument)

        self.live[book_key(format_name, match_id)] = LiveMatch(
            book=book,
            format_name=format_name,
            match_id=match_id,
            opened_at=now,
            closes_at=now + int(seconds(self.match_seconds)),
            knockout_order=order,
        )
        self.opened += 1

    def _advance(self, ctx: Any, match: LiveMatch, now: int) -> None:
        """Reveal whatever has happened by now, and close a finished match."""
        if now >= match.closes_at:
            self._close(ctx, match)
            return
        if not match.knockout_order:
            return

        # Eliminations spread evenly across the match's life. Evenly rather
        # than clustered because the interesting property is that information
        # keeps arriving, and a schedule that dumped it all at the end would
        # make the whole window a single pre-match auction.
        elapsed = now - match.opened_at
        span = max(1, match.closes_at - match.opened_at)
        due = int(len(match.knockout_order) * elapsed / span)
        while match.revealed < min(due, len(match.knockout_order)):
            self._knock_out(match, match.knockout_order[match.revealed])
            match.revealed += 1

    def _knock_out(self, match: LiveMatch, competitor: str) -> None:
        """Resolve what this elimination makes certain, and nothing else.

        Only the eliminated competitor's own winner contract. That one is
        certain: they are out, so they cannot win, so it is worth zero whatever
        happens next. Their placement and the survivors' contracts are not
        certain and stay open, because settling a contract whose value is still
        in doubt is paying out a forecast.
        """
        for contract in match.book.contracts:
            symbol = contract.instrument.symbol
            if symbol in match.settled:
                continue
            if contract.family != "winner":
                continue
            if contract.subject != competitor:
                continue
            self._settle(match, contract, Decimal(0))

    def _close(self, ctx: Any, match: LiveMatch) -> None:
        """The match is over, so everything still open has an answer now."""
        result = play(self.seed, match.match_id, match.format_name)
        values = settlement(match.book, result)
        for contract in match.book.contracts:
            symbol = contract.instrument.symbol
            if symbol in match.settled:
                continue
            self._settle(match, contract, Decimal(str(values[symbol])))
        self.live.pop(book_key(match.format_name, match.match_id), None)
        self.settled += 1

    def _settle(self, match: LiveMatch, contract: Any, value: Decimal) -> None:
        spec = contract.instrument.spec
        result = SettlementResult(
            contract_id=spec.contract_id,
            spec_digest=spec.spec_digest,
            status=SettlementStatus.SETTLED,
            settlement_value=value,
            underlying_level=None,
            resolutions=(),
        )
        self.venue.settle(contract.instrument.symbol, result)
        match.settled.add(contract.instrument.symbol)


def book_key(format_name: str, match_id: int) -> str:
    return f"{format_name}:{match_id}"
