"""The live feed, for programs rather than for a page.

    WS /v1/stream

The dashboard already has a websocket and it is the wrong shape for an
algorithm. ``/ws`` sends the whole world twenty times a second (every book, the
tape, the roster, the account, the blotter, the counterparties) because one
browser needs all of it at once to draw a page. A systematic trader usually
wants two contracts and its own fills, and on this venue that means it would
spend its entire decode budget parsing the other twenty-six instruments to
throw them away. Worse, it cannot tell whether anything it cares about actually
changed: every snapshot is a fresh copy of everything, so "did the book move"
is a diff the client has to perform against a message it did not ask for.

So this is a subscription feed in the shape Kalshi and Alpaca use: the client
names channels and receives only those, each frame labelled with the channel
that produced it.

Protocol, client to server (one JSON object per message)::

    {"op": "subscribe",   "channels": ["book.VANTA_OBJECTIVE_WR", "ticker.*"]}
    {"op": "unsubscribe", "channels": [...]}
    {"op": "get_snapshot", "channels": ["book.VANTA_OBJECTIVE_WR"]}
    {"op": "auth", "key_id": "...", "timestamp": "...", "signature": "..."}
    {"op": "resume", "session": "sn_...", "from_seq": 41}
    {"op": "ping"}

Any command may carry an ``"id"``, which is echoed on the reply to it. A client
with several commands in flight cannot otherwise tell which reply answered
which, and the tests in this repository use it as a fence: a ``pong`` for a
known id is a point in the stream after which everything earlier has certainly
been delivered.

Channels::

    ticker.<symbol>     mark, best bid and ask, change          public
    book.<symbol>       aggregated depth, both sides            public
    trades.<symbol>     prints, one frame each                  public
    lifecycle.<symbol>  listings, session changes, halts,       public
                        settlements
    orders              acks, cancels, rejects for this seat    private
    fills               this seat's executions                  private

``<symbol>`` may be ``*``, which follows the registry rather than a list fixed
at subscribe time, so an instrument listed later appears on the feed without
the client reconnecting, and no channel kind knows anything about any
particular asset class. Every contract the venue lists (future, event, index,
volatility, spread, share, call, put, commodity) streams through this one path.

Every server frame carries ``{"type", "channel", "seq"}``. ``seq`` counts
frames on **this session** and is gapless: a client that sees 41 after 39 knows
it lost one, which is the whole reason to number them. A connection is normally
a session, so this reads as "on this connection"; the exception is a connection
that resumed another session, where the numbering changes over once, at a point
the ``resumed`` frame names in ``next_seq`` and nowhere else. Frames that are
not about a channel (``hello``, ``pong``, ``subscribed``, ``unsubscribed``,
``snapshot``, ``auth``, ``resumed``, ``resume_failed``, ``reset``, ``error``)
carry ``"channel": null`` rather than inventing a channel to put them on.

A ``trade`` also carries ``exchange_seq``, and the two numbers are different
things: ``seq`` is this session's frame count, ``exchange_seq`` is the
matching engine's own match number for that symbol. A client aligning its view
against another subscriber's wants the second; a client checking it has missed
nothing wants the first.

Recovering, rather than only detecting
--------------------------------------

A number that reveals a gap is worth nothing on its own. A client that sees 41
after 39 knows its book is wrong and, without the three mechanisms below, its
only remedy is to tear the process down and start again, which is what a
systematic trader that drops a TCP connection had to do here until now.

**Re-snapshot without resubscribing.** ``{"op": "get_snapshot", "channels":
[...]}`` answers with a fresh full ``book`` (or ``ticker``) frame for each
named channel, marked ``"snapshot": true``, and **changes nothing**: not the
subscription list, not the conflation state, not the ``change`` anchor. It is
Kalshi's orderbook contract: a snapshot first, increments after, and a
``get_snapshot`` action that re-snapshots without resubscribing. Bybit states
the client's half of the same contract (a new snapshot means reset your local
book), which is why the flag is on the frame rather than implied by the
request.

**Resume across a reconnect.** ``hello`` carries a ``session`` id, and a client
that lost its socket may reconnect and send ``{"op": "resume", "session":
"...", "from_seq": N}``. This is SoupBinTCP's guarantee, and it is worth
stating in its own words: if the connection is broken the client re-logs in
naming the session and its next expected sequence number, and is then
guaranteed to receive every sequenced message in order despite the failure. A
successful resume answers ``resumed`` (which names ``next_seq``, exactly as
SoupBinTCP's Login Accepted packet does) then replays the retained frames
**with their original sequence numbers**, restores the subscriptions and the
cursors, and carries on. The cursors matter as much as the frames: a resume
that restored the subscription but reopened the tape cursor at *now* would drop
every print that landed while the client was away, and drop it silently,
because the sequence would still be gapless across the hole.

A resume that cannot be honoured is answered with a ``resume_failed`` frame
naming the reason, never by quietly starting a fresh sequence. That silent
case is the actual danger: a client that believes it resumed and did not is a
client trading on a book with a hole in it, and nothing downstream can tell.

What is retained is bounded and the bound is published. A session keeps its
last :data:`RESUME_FRAMES` frames (2,000, CME MDP 3.0's own cap on a TCP replay
request) for :data:`RESUME_SECONDS` after the socket goes, and at most
:data:`RESUME_SESSIONS` disconnected sessions are held at once. All three are
on the ``hello`` frame, because a client told the bound only at the moment it
exceeds one has already lost the data. Past the bound the answer is a refusal
and never a partial replay.

**A reset the client did not ask for.** ``reset`` is pushed whenever a cursor
this connection holds stops meaning anything: ``reconfigure`` discarding the
market and every account in it, a re-seat onto a different account, or the
seat's blotter sliding past the last entry forwarded. It says discard, and the
client re-snapshots.

Three things this module is built around, and the first two are measured facts
about this codebase rather than general principles:

**Conflation, not dropping.** :class:`~arena.market.venue_agent.VenueAgent`
records what dropping costs: an earlier feed threw away an update when a
subscriber was behind, a maker that has not moved its quote sends no order so
nothing republishes, and every agent waiting for a price waited forever: a
trial that traded 2,039 times traded 0. So nothing here is discarded. State
channels (``ticker``, ``book``) hold the *latest* state and send that on the
next tick, which is conflation: a superseded book is not an update anybody
needs. Event channels (``trades``, ``orders``, ``fills``) are not conflated at
all, because a print is a fact rather than a state and the next one does not
replace it.

The mechanism is that nothing is queued at production time. Each flush reads
the venue's own buffers (the engine's tape, the seat's blotter) from a cursor.
A client that stops reading for ten seconds therefore has no queue growing
behind it; it has a cursor that is ten seconds behind, and the next flush it
can absorb catches it up in order. That is also what keeps a slow client off
the market loop: the kernel is stepped by a different task, this one only
reads, and when a send blocks it blocks this connection and nothing else.

**Private channels re-resolve their seat every tick.** ``reconfigure`` discards
the market and every account in it, and :meth:`LiveMarket.trader` answers an id
it does not know with the **shared** account. A connection that captured an
agent id once would, after a rebuild, quietly stream someone else's fills,
which is exactly the failure that merged every browser visitor into one seat,
described at length in ``dashboard.server._seat_now``. An id is only meaningful
inside the generation that issued it, so what is held here is the key's opaque
seat token and a stable *name*, and the account is re-resolved every flush:
through the application's own resolver when :func:`configure` was given one,
and otherwise through ``runner.market.seat(name)`` whenever the generation
moves or the bound id stops being an account.

**A public channel can never carry private data.** The two are built by
different functions from different sources: a public frame is assembled from
the registry, the book and the tape, and there is no account in scope while it
is built. Subscribing to a private channel without a valid signature is
refused, and the refusal names ``auth_required`` so a client can tell it apart
from a channel that does not exist.

Mounting it::

    from arena.api import stream

    stream.configure(keys=api_keys, runner=runner, seat_now=_seat_now)
    app.add_api_websocket_route("/v1/stream", stream.stream_endpoint())

``keys`` and ``seat_now`` must be **the same two objects the REST half was
given**. A second key store would refuse credentials this one issued, and a
second seat resolver would put one credential on two accounts.

FastAPI is imported here, which the ``arena`` package otherwise avoids. It is
confined to this module and ``rest.py`` (the settlement core underneath still
imports nothing) and the alternative, hand-rolling the ASGI websocket
handshake, would be a second implementation of something the application
already depends on.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import sys
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from arena.api.errors import ERRORS, error_body
from arena.api.keys import KeyStore, SignatureError
from arena.exchange.session import SessionState
from arena.exchange.types import AgentId
from arena.portfolio.money import from_money

__all__ = [
    "configure",
    "stream_endpoint",
    "STREAM_PATH",
    "TICK_SECONDS",
    "DEPTH_LEVELS",
    "PUBLIC_KINDS",
    "PRIVATE_KINDS",
    "STATE_KINDS",
    "LIFECYCLE_EVENTS",
    "RESUME_FRAMES",
    "RESUME_SESSIONS",
    "RESUME_SECONDS",
]

# Where the endpoint is mounted, and therefore what an ``auth`` signature
# covers. Named rather than inlined because the signature is over the path: a
# client signing "/v1/stream" against a server that mounted this at "/stream"
# fails to authenticate with no clue as to why, and the two sides must not be
# free to drift apart on spelling.
STREAM_PATH = "/v1/stream"

# How often a connection flushes. The same 20Hz the browser socket uses, and
# for the same reason: the market is stepped at that cadence, so a faster flush
# would send the same state twice and a slower one would conflate away detail
# the venue actually published.
TICK_SECONDS = 0.05

# Depth published on a ``book`` channel. Eight is what the browser panel shows;
# ten here because a program is not laying it out and the marginal level costs
# a few bytes. A client that wants the whole ladder asks REST for it once
# rather than being sent it twenty times a second.
DEPTH_LEVELS = 10

# The channel kinds. Kinds, not symbols and not asset classes: every instrument
# the registry lists is reachable through the same four public kinds, and
# adding a contract type to the venue requires nothing here.
PUBLIC_KINDS = ("ticker", "book", "trades", "lifecycle")
PRIVATE_KINDS = ("orders", "fills")

# The kinds that carry *state* rather than events, and therefore the kinds a
# snapshot means anything for. A snapshot of ``trades`` would be a fiction: a
# tape has no current value, only a history, and the honest recovery for a
# missed print is the sequence number and ``resume`` rather than a re-read of
# something that was never a state. Refused by name for that reason instead of
# being answered with an empty frame that reads like "nothing has traded".
STATE_KINDS = ("ticker", "book")

# What a ``lifecycle`` frame can say. Kalshi publishes the same set under
# ``market_lifecycle_v2`` (created, activated, deactivated, determined,
# settled) and the mapping here is to what this venue actually has: a symbol
# appearing in the registry, its session moving between the four phases, the
# breaker or an operator stopping it, and its settlement paying out. An
# algorithm running unattended learns about a new listing from this rather than
# by polling the registry, and learns that a contract stopped trading before it
# works it out from a book that went quiet.
LIFECYCLE_EVENTS = ("listed", "halt", "session", "settled")

# Wildcard symbol. Resolved against the registry on every flush rather than
# expanded once at subscribe time, so a listing that appears mid-session
# appears on the feed.
ALL = "*"

# How far back a dropped connection may be replayed from, in frames.
#
# The bound has to exist and it has to be published, because the alternative is
# a server that retains everything for a client that is never coming back. Real
# venues all draw the line somewhere and say where: Nasdaq's SoupBinTCP caps a
# retransmission at what fits one packet, and CME's MDP 3.0 caps a TCP replay
# request at 2,000 packets. This is CME's number, and on this feed it is a
# useful amount of time rather than an arbitrary one: a connection watching a
# handful of contracts at 20Hz emits on the order of thirty frames a second, so
# 2,000 frames is roughly a minute of feed, which comfortably covers a TCP
# reconnect, a DNS stall or a process restart. A client watching every contract
# on the venue burns through it in a few seconds instead, which is exactly why
# the number is on the ``hello`` frame: the client can see the bound it is
# actually being given rather than assume a duration.
#
# Past the bound the answer is a refusal and not a partial replay. Sending the
# frames that *are* still held would leave the client with a hole it has been
# given no way to notice, which is the failure this whole section exists to
# remove.
RESUME_FRAMES = 2000

# How many *disconnected* sessions are retained at once, and for how long. A
# client in a reconnect loop would otherwise mint a session per attempt and
# leak every one of them: the frame buffer is bounded per session, so without
# these two the total is bounded only by how often a client can reconnect. A
# live connection is never evicted: it owns its session.
RESUME_SESSIONS = 64
RESUME_SECONDS = 120.0


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

_KEYS: KeyStore | None = None
_RUNNER: Any | None = None
_PATH: str = STREAM_PATH
_TICK: float = TICK_SECONDS
_SEAT_NOW: Callable[[str], Any | None] | None = None
_RESUME_FRAMES: int = RESUME_FRAMES


def configure(
    *,
    keys: KeyStore | None = None,
    runner: Any | None = None,
    path: str | None = None,
    tick_seconds: float | None = None,
    seat_now: Callable[[str], Any | None] | None = None,
    resume_frames: int | None = None,
) -> None:
    """Inject what the endpoint needs before it is mounted.

    Injected rather than imported. This module lives under ``arena``, the
    application that owns the running market lives under ``dashboard``, and an
    import in that direction would invert the layering and produce a cycle the
    moment ``dashboard.server`` mounts the endpoint.

    ``seat_now`` is how the application says where one of its own people is
    sitting *in the market running now*, given the seat token a key was issued
    against. **Pass the same one the REST half is given.** Without it the two
    halves of this API each seat the credential separately: the client places
    an order on one account and watches another's fills, sees nothing, and has
    no way to tell that from a broken feed. ``dashboard.server`` already has
    exactly this function, ``_seat_now``, and hands it to ``rest.configure``.

    Only what is passed is changed, so a caller can set the key store at import
    time and the runner later without having to restate either.

    Changing the runner clears the seat table, because a seat is an account in
    a particular market and one from the previous runner names nothing. It
    clears the resumable sessions for the same reason and a sharper one: a
    session holds cursors into an engine's tape and a seat's blotter, and
    replaying those against a different runner's market would answer a client
    with somebody else's history under its own sequence numbers.

    ``resume_frames`` overrides :data:`RESUME_FRAMES` for sessions opened after
    it is set. It exists so an operator can trade memory for reach knowingly,
    and so a test can overflow the buffer without producing two thousand frames
    to do it; the bound in force is always published on ``hello``.
    """
    global _KEYS, _RUNNER, _PATH, _TICK, _SEAT_NOW, _RESUME_FRAMES
    if keys is not None:
        _KEYS = keys
    if runner is not None and runner is not _RUNNER:
        _RUNNER = runner
        with _SEAT_LOCK:
            _SEATS.clear()
            _SEAT_NAMES.clear()
        with _SESSION_LOCK:
            _SESSIONS.clear()
    if path is not None:
        _PATH = path
    if tick_seconds is not None:
        _TICK = float(tick_seconds)
    if seat_now is not None:
        _SEAT_NOW = seat_now
    if resume_frames is not None:
        _RESUME_FRAMES = max(1, int(resume_frames))


def _resolve_runner(explicit: Any | None) -> Any:
    """The market this connection reads, or a refusal that says what is missing.

    The last resort reads ``runner`` off the application module *if it has
    already been imported*, via ``sys.modules`` rather than an import
    statement. Importing ``dashboard.server`` here would build a second market
    as a side effect of answering a websocket, which is a spectacular way to
    turn a missing call to :func:`configure` into a bug nobody can find.
    """
    if explicit is not None:
        return explicit
    if _RUNNER is not None:
        return _RUNNER
    for name in ("dashboard.server", "dashboard.state"):
        module = sys.modules.get(name)
        found = getattr(module, "runner", None) if module is not None else None
        if found is not None:
            return found
    raise RuntimeError(
        "arena.api.stream has no market to read: call configure(runner=...) "
        "before mounting the endpoint"
    )


# --------------------------------------------------------------------------
# Sessions: what outlives one socket
# --------------------------------------------------------------------------


@dataclass
class _Session:
    """A connection's stream, held apart from the socket carrying it.

    Everything here is what a reconnecting client needs and cannot reconstruct
    on its own. The frame buffer is the obvious half; the cursors are the half
    that is easy to leave out and disastrous to leave out. A resume that
    restored the subscriptions and then opened a fresh tape cursor at *now*
    would skip every print that landed while the client was disconnected, and
    the sequence numbers would still run 40, 41, 42 straight across the hole, a
    gap that is undetectable by the one mechanism provided for detecting gaps.

    ``token`` is the seat token of the credential the session authenticated
    with, or ``None`` if it never did. It is the reason a session id is not on
    its own enough to resume: the buffer of an authenticated session holds that
    seat's blotter, so resuming it is reading somebody's private feed and has
    to be proved the same way subscribing to it was. SoupBinTCP asks for the
    username and password again on a re-login for the same reason; the session
    id is a bookmark, not a credential.
    """

    session_id: str
    seq: int = 0
    frames: deque[dict[str, Any]] = field(default_factory=deque)
    channels: tuple[str, ...] = ()
    token: str | None = None
    generation: int | None = None
    # True while a socket holds this session. A second connection may not
    # resume it while a first is still writing into it: two writers on one
    # sequence produce two frames numbered 58 and no way to tell them apart.
    live: bool = True
    ended_at: float = 0.0

    # The cursors, mirrored out of the connection when it ends.
    tape_seen: dict[str, int] = field(default_factory=dict)
    published: dict[str, dict[str, Any]] = field(default_factory=dict)
    anchor: dict[str, Decimal] = field(default_factory=dict)
    private_mark: tuple[Any, ...] | None = None
    private_started: bool = False
    listed_seen: set[str] | None = None
    session_seen: dict[str, str] = field(default_factory=dict)
    halts_seen: int | None = None
    settled_seen: set[str] | None = None

    @property
    def retained_from(self) -> int:
        """The oldest sequence number this session can still answer for.

        When the buffer is empty that is the *next* number rather than the last
        one, so a client that was fully caught up when the socket died can
        still resume: it asks for ``seq + 1``, nothing is replayed, and it
        carries on. Answering ``seq`` there would refuse the one client that
        had missed nothing at all.
        """
        return int(self.frames[0]["seq"]) if self.frames else self.seq + 1


# Sessions by id, across every connection. Everything in here runs on the
# endpoint's event loop, so the lock is not guarding against the connections;
# it guards against ``configure`` clearing the table from whichever thread an
# application happens to start up on.
_SESSIONS: dict[str, _Session] = {}
_SESSION_LOCK = threading.Lock()


def _new_session(frames: int) -> _Session:
    """Open a session, and make room for it.

    The id is random rather than sequential. A session id is quoted back by a
    client to resume a stream, so a guessable one is a way to read a stream
    somebody else opened, and while an authenticated session is additionally
    protected by its seat token, a public one would be readable by anybody who
    could count.
    """
    session = _Session(
        session_id=f"sn_{secrets.token_hex(12)}",
        frames=deque(maxlen=max(1, int(frames))),
    )
    with _SESSION_LOCK:
        _evict_locked()
        _SESSIONS[session.session_id] = session
    return session


def _evict_locked() -> None:
    """Drop sessions nobody is coming back for. Never a live one.

    Both bounds are enforced here rather than by a sweeper task: a background
    task would have to be owned by something, and the only moments the size of
    this table can grow are the moments this runs.
    """
    now = time.monotonic()
    for session_id, session in list(_SESSIONS.items()):
        if not session.live and now - session.ended_at > RESUME_SECONDS:
            _SESSIONS.pop(session_id, None)
    idle = sorted(
        (s for s in _SESSIONS.values() if not s.live), key=lambda s: s.ended_at
    )
    while len(idle) > RESUME_SESSIONS:
        _SESSIONS.pop(idle.pop(0).session_id, None)


# --------------------------------------------------------------------------
# Seats
# --------------------------------------------------------------------------


@dataclass(slots=True)
class _Seat:
    """Which account a credential trades, and what it is called.

    The name is the durable half. The account id is only meaningful inside the
    generation that issued it (``reconfigure`` hands the same ``you-1`` to
    whoever seats first in the new market), so it is stamped with a generation
    and re-resolved rather than trusted.
    """

    name: str
    agent_id: AgentId | None
    generation: int


# Seats by **seat token**, shared across every connection. ``ApiKey.agent_id``
# is that token: an opaque, durable identifier for a person, which is what the
# REST half binds to as well. It is deliberately not an account id (an account
# lasts only as long as the market that issued it) and deliberately not a
# display name, because two traders called "Ash" are two traders.
#
# Keyed by token rather than by key id so that a person holding two keys, or
# one key on two connections, reads one blotter. That is how any real venue
# behaves, and per-connection seats would have handed the second connection its
# own account at the first rebuild.
_SEATS: dict[str, _Seat] = {}

# The name each token is seated under, remembered so that a rebuild re-seats a
# person under the name they chose rather than a fresh one.
_SEAT_NAMES: dict[str, str] = {}

# Seating is a read-modify-write on the market's roster and it is not atomic:
# ``LiveMarket.seat`` picks the next free id, opens an account under it and
# then registers the agent. Two threads through that window pick the same id
# and the second overwrites the first, which puts two credentials on one
# account: the failure this module exists to prevent, reached from the other
# side. ``dashboard.server`` holds its own lock over its own seating for
# exactly this reason.
_SEAT_LOCK = threading.Lock()


def _seat_for(runner: Any, key: Any, seat_now: Any | None) -> _Seat:
    """The account a key trades **in the market that is running now**.

    Called on every flush, not once per connection. That is the whole point:
    between two flushes the market may have been rebuilt, every account in it
    discarded, and a connection still holding the old id would be reading
    ``LiveMarket.trader``'s fallback (the shared account) and streaming a
    stranger's fills to a client that has no way of knowing.

    Two ways of answering, in this order.

    The application's own, when it has one. It knows where its browser sessions
    are sitting, and the REST half asks it the same question through the same
    hook, so a person's key, their cookie and their stream all land on one
    account, including across a rebuild, which is exactly where a naive binding
    falls back to the shared seat. The answer is checked against the venue's
    account table before it is trusted, because an application holding a stale
    id is the failure being guarded against, not a source of truth.

    Otherwise, the seat token's own binding, re-seated by **name** whenever the
    generation moves or the bound id is no longer an account. That last check
    matters more than the generation one: it tests the actual failure
    (``LiveMarket.trader`` answers an unknown id with the shared account)
    rather than a bookkeeping proxy for it.

    The token is never looked up as an account id. ``reconfigure`` starts the
    numbering again, so a token that happened to read ``you-1`` would find
    whoever seated first in the new market, and that person's fills would go
    out under this key. Both accounts exist, both have a blotter, and nothing
    raises: it is the same silent collision ``dashboard.server._Seat`` records
    between two browser visitors.
    """
    token = str(key.agent_id)
    with _SEAT_LOCK:
        market = runner.market
        name = _SEAT_NAMES.get(token) or str(getattr(key, "label", "") or token)
        _SEAT_NAMES[token] = name
        seat = _SEATS.get(token)
        if seat is None:
            seat = _Seat(name, None, -1)
            _SEATS[token] = seat

        if seat_now is not None:
            delegated = seat_now(token)
            if delegated is not None and delegated in market.venue.accounts:
                seat.agent_id = delegated
                seat.generation = runner.generation
                return seat

        if seat.generation != runner.generation or seat.agent_id not in market.traders:
            seat.agent_id = market.seat(seat.name)
            seat.generation = runner.generation
        return seat


# --------------------------------------------------------------------------
# Channels
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Channel:
    """One subscription: a kind, and the symbol it names if it names one."""

    kind: str
    symbol: str = ""

    @property
    def name(self) -> str:
        return f"{self.kind}.{self.symbol}" if self.symbol else self.kind

    @property
    def private(self) -> bool:
        return self.kind in PRIVATE_KINDS

    def concrete(self, symbol: str) -> str:
        """The channel label a frame for one symbol carries.

        A client subscribed to ``ticker.*`` receives frames labelled
        ``ticker.VANTA_OBJECTIVE_WR``, not ``ticker.*``. The wildcard is how
        you ask; it is not a thing any frame is actually about, and a client
        routing on the label would otherwise have to unpack the payload to find
        out which instrument moved.
        """
        return f"{self.kind}.{symbol}" if self.symbol else self.kind


class _BadChannel(ValueError):
    """A channel name that cannot be honoured, with the code to refuse it by."""

    def __init__(self, code: str, channel: str, **detail: Any) -> None:
        super().__init__(channel)
        self.code = code
        self.channel = channel
        self.detail = detail


def _parse_channel(raw: Any) -> _Channel:
    """One channel name, or the reason it is not one.

    Refused rather than guessed at. Quietly reinterpreting ``tickers.FOO`` as
    ``ticker.FOO`` would subscribe a client to something it did not ask for,
    and the first it would hear of the difference is a frame it cannot parse.
    """
    text = str(raw).strip() if isinstance(raw, str) else ""
    if not text:
        raise _BadChannel("invalid_request", str(raw))
    kind, _, symbol = text.partition(".")
    if kind in PRIVATE_KINDS:
        if symbol:
            # `fills.VANTA_OBJECTIVE_WR` reads like a filter and is not one.
            # Accepting it and ignoring the symbol would stream every fill to a
            # client that believes it asked for one instrument's.
            raise _BadChannel("invalid_request", text, reason="takes no symbol")
        return _Channel(kind)
    if kind in PUBLIC_KINDS:
        if not symbol:
            raise _BadChannel("invalid_request", text, reason="names no symbol")
        return _Channel(kind, symbol)
    raise _BadChannel("invalid_request", text, reason="unknown channel")


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------


def _error_frame(code: str, channel: str | None = None, **detail: Any) -> dict[str, Any]:
    """A refusal, in the one shape every refusal in this API takes.

    The sentence comes from :data:`arena.api.errors.ERRORS` rather than being
    written here, so the socket and the REST surface cannot end up describing
    the same refusal two different ways. The HTTP status in that table is left
    out: there is no status on a websocket frame, and publishing one would
    invite a client to branch on a number that means nothing here.
    """
    message = ERRORS.get(code, ("the request could not be read", 400))[0]
    return {"type": "error", "channel": channel, **error_body(code, message, **detail)}


def _price(instrument: Any, ticks: Any) -> str:
    """A tick count as the price it stands for, as a string.

    Every price on this feed goes through here. The settlement figure on the
    dashboard was published in ticks under a label that promised a price, and
    on a contract quoted on a 0.25 grid that is four times the number a reader
    expects: it did not look like a bug, it looked like a number. A string
    rather than a float because a price path with a float in it is a price path
    that eventually disagrees with the ledger.
    """
    return str(instrument.from_ticks(int(ticks)))


# --------------------------------------------------------------------------
# One connection
# --------------------------------------------------------------------------


class _Stream:
    """One websocket: its subscriptions, its cursors, and its sequence."""

    def __init__(
        self,
        socket: WebSocket,
        runner: Any,
        keys: KeyStore | None,
        path: str,
        tick_seconds: float,
        seat_now: Any | None = None,
        resume_frames: int | None = None,
    ) -> None:
        self._socket = socket
        self._runner = runner
        self._keys = keys
        self._path = path
        self._tick_seconds = tick_seconds
        self._seat_now = seat_now

        # The sequence counter lives on the session rather than on the
        # connection, because that is the thing a resume continues. A
        # connection that adopts a session adopts its numbering with it.
        self._session = _new_session(
            resume_frames if resume_frames is not None else _RESUME_FRAMES
        )
        self._resumed = False
        # Assigning a sequence number and writing the frame have to happen
        # together. Two tasks send here (the flush loop and the command
        # handler) and without the lock one can take number 6 while the other
        # is still awaiting the write of number 5, putting them on the wire out
        # of order. A client checking that its sequence increases by one would
        # then report a gap on a connection that never lost anything.
        self._writing = asyncio.Lock()
        self._closed = False

        # Ordered, so a flush emits channels in the order they were asked for
        # rather than in whatever order a set iterates.
        self._channels: dict[_Channel, None] = {}

        self._key: Any | None = None
        self._seat: _Seat | None = None

        self._generation: int | None = None
        # How far into each engine's tape this connection has read. Absent
        # means "not read yet", which starts at the end of the tape: a new
        # subscriber wants the market from here, not a replay of the session.
        self._tape_seen: dict[str, int] = {}
        # The last state published per concrete channel, so an unchanged book
        # is not sent again. This is the conflation: hold the latest, send it
        # when it differs.
        self._published: dict[str, dict[str, Any]] = {}
        # What ``change`` on a ticker is measured from, per symbol.
        self._anchor: dict[str, Decimal] = {}
        # Position in the seat's blotter, held as the identity of the last
        # entry forwarded rather than as an index (see :meth:`_drain_private`).
        self._private_mark: tuple[Any, ...] | None = None
        self._private_started = False

        # Lifecycle cursors. ``None`` means "not opened", which is how a
        # subscriber that arrives an hour in gets what happens next rather than
        # a replay of every halt since the open, the same rule the tape cursor
        # follows, for the same reason.
        self._listed_seen: set[str] | None = None
        self._session_seen: dict[str, str] = {}
        self._halts_seen: int | None = None
        self._settled_seen: set[str] | None = None

    # -- writing -----------------------------------------------------------

    async def _send(self, frame: dict[str, Any]) -> None:
        if self._closed:
            return
        async with self._writing:
            await self._emit_locked(frame)

    async def _emit_locked(self, frame: dict[str, Any]) -> None:
        """Number a frame, retain it, and write it. The lock must be held.

        Retained *before* the write is attempted and regardless of whether the
        write succeeds. The sequenced stream is what the server said, not what
        TCP managed to deliver: a frame lost to a socket that was already dying
        is precisely the frame the client will come back and ask for, and a
        buffer that only held the ones that made it out could not answer.
        """
        session = self._session
        session.seq += 1
        frame["seq"] = session.seq
        session.frames.append(frame)
        await self._write_locked(frame)

    async def _write_locked(self, frame: dict[str, Any]) -> None:
        """Put an already-numbered frame on the wire. The lock must be held.

        Used directly by a replay, which must not renumber or re-retain what it
        is sending: the whole point is that the client receives frame 41 as
        frame 41.
        """
        if self._closed:
            return
        try:
            await self._socket.send_json(frame)
        except (WebSocketDisconnect, RuntimeError):
            self._closed = True

    async def _refuse(
        self,
        code: str,
        channel: str | None = None,
        *,
        reply_id: Any = None,
        **detail: Any,
    ) -> None:
        """Refuse, carrying the id of the command being refused where there is one.

        A client with three commands in flight cannot otherwise tell which of
        them was rejected, and would have to guess from the channel, which is
        absent on exactly the errors where the guess matters, such as a
        malformed ``channels`` list.
        """
        await self._send(_tag(_error_frame(code, channel, **detail), reply_id))

    # -- lifecycle ---------------------------------------------------------

    async def run(self) -> None:
        # The outer ``finally`` covers the handshake as well as the loop. A
        # session is created before the socket is accepted and is held live
        # until it is released, and a live session is deliberately never
        # evicted, so a client that connects and aborts during the handshake
        # would otherwise leave a session in the table that nothing can ever
        # remove, once per attempt.
        try:
            await self._socket.accept()
            self._sync_generation()
            await self._send(self._hello())
            receiver = asyncio.create_task(self._receive())
            try:
                while not self._closed:
                    await self._flush()
                    await asyncio.sleep(self._tick_seconds)
            except (WebSocketDisconnect, RuntimeError):
                # A disconnect is how a connection ends and is not a fault.
                # Nothing else is caught: a stream that keeps a socket open
                # while quietly producing nothing is worse than one that
                # closes, because a client reconnects on a close and waits
                # forever on silence.
                pass
            finally:
                self._closed = True
                receiver.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receiver
        finally:
            self._closed = True
            self._release()

    def _release(self) -> None:
        """Hand the connection's state back to its session, for a resume.

        Run from ``finally`` rather than from the disconnect path, because the
        interesting way for a connection to end is the way nobody chose: a
        cancelled task, a socket that stopped answering, a process being shut
        down under it. Those are exactly the cases a client will reconnect
        from, and a session that only recorded its cursors on a clean close
        would be useless for every one of them.
        """
        session = self._session
        session.live = False
        session.ended_at = time.monotonic()
        session.channels = tuple(channel.name for channel in self._channels)
        session.generation = self._generation
        session.token = str(self._key.agent_id) if self._key is not None else None
        session.tape_seen = dict(self._tape_seen)
        session.published = dict(self._published)
        session.anchor = dict(self._anchor)
        session.private_mark = self._private_mark
        session.private_started = self._private_started
        session.listed_seen = (
            None if self._listed_seen is None else set(self._listed_seen)
        )
        session.session_seen = dict(self._session_seen)
        session.halts_seen = self._halts_seen
        session.settled_seen = (
            None if self._settled_seen is None else set(self._settled_seen)
        )
        with _SESSION_LOCK:
            _evict_locked()

    def _hello(self) -> dict[str, Any]:
        """What is on offer, so a client can start without a REST round trip.

        It also fixes ``seq`` at 1 before anything else is sent, which is what
        makes the gapless check startable: a client that joins mid-stream has
        nothing to compare its first number against.

        ``session`` and ``resume`` are the recovery half of the same idea, and
        they are here rather than in a document because a client cannot act on
        a document. SoupBinTCP's Login Accepted packet carries the session id
        and the next sequence number for the same reason: the client needs both
        of them before anything goes wrong, not after. ``resume.frames`` is the
        bound actually in force on this connection, so a client can work out
        for itself (from ``tick_ms`` and how much it subscribed to) roughly how
        long it has to reconnect within.
        """
        symbols = list(self._runner.market.venue.registry.symbols)
        return {
            "type": "hello",
            "channel": None,
            "generation": self._runner.generation,
            "tick_ms": int(round(self._tick_seconds * 1000)),
            "public_channels": list(PUBLIC_KINDS),
            "private_channels": list(PRIVATE_KINDS),
            "snapshot_channels": list(STATE_KINDS),
            "lifecycle_events": list(LIFECYCLE_EVENTS),
            "ops": [
                "subscribe",
                "unsubscribe",
                "get_snapshot",
                "auth",
                "resume",
                "ping",
            ],
            "session": self._session.session_id,
            "resume": {
                "frames": self._session.frames.maxlen,
                "sessions": RESUME_SESSIONS,
                "seconds": int(RESUME_SECONDS),
            },
            "symbols": symbols,
        }

    # -- commands ----------------------------------------------------------

    async def _receive(self) -> None:
        """Handle what the client sends. Never disconnects over a bad message.

        A malformed command is the client's problem to fix and it needs the
        connection to still be there to fix it on. Closing the socket on an
        unknown ``op`` would also make a typo indistinguishable from a network
        failure, and a reconnecting client would type it again.
        """
        while not self._closed:
            try:
                message = await self._socket.receive_json()
            except (WebSocketDisconnect, RuntimeError):
                self._closed = True
                return
            except (ValueError, TypeError, KeyError):
                await self._refuse("invalid_request", reason="expected JSON")
                continue

            if not isinstance(message, dict):
                await self._refuse("invalid_request", reason="expected a JSON object")
                continue

            reply_id = message.get("id")
            op = message.get("op")
            try:
                if op == "subscribe":
                    await self._subscribe(message, reply_id)
                elif op == "unsubscribe":
                    await self._unsubscribe(message, reply_id)
                elif op == "get_snapshot":
                    await self._get_snapshot(message, reply_id)
                elif op == "auth":
                    await self._auth(message, reply_id)
                elif op == "resume":
                    await self._resume(message, reply_id)
                elif op == "ping":
                    await self._send(_tag({"type": "pong", "channel": None}, reply_id))
                else:
                    await self._refuse("invalid_request", reply_id=reply_id, op=op)
            except _BadChannel as bad:
                await self._refuse(
                    bad.code, bad.channel or None, reply_id=reply_id, **bad.detail
                )

    def _requested(self, message: dict[str, Any]) -> list[_Channel]:
        raw = message.get("channels")
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list) or not raw:
            raise _BadChannel("invalid_request", "", reason="channels must be a list")
        return [_parse_channel(entry) for entry in raw]

    async def _subscribe(self, message: dict[str, Any], reply_id: Any) -> None:
        """Add channels, all of them or none of them.

        Atomic because a partial subscribe is the worst of both: the client
        gets an error *and* a feed, and has to work out which of the channels
        it asked for it is actually receiving. Refusing the whole command
        leaves it in a state it can reason about.
        """
        channels = self._requested(message)
        registry = self._runner.market.venue.registry
        for channel in channels:
            if channel.private and self._seat is None:
                raise _BadChannel("auth_required", channel.name)
            if channel.symbol and channel.symbol != ALL:
                if registry.get(channel.symbol) is None:
                    raise _BadChannel(
                        "invalid_symbol", channel.name, symbol=channel.symbol
                    )
        for channel in channels:
            self._channels[channel] = None
            self._open_cursor(channel)
        await self._send(
            _tag(
                {
                    "type": "subscribed",
                    "channel": None,
                    "channels": [c.name for c in channels],
                    "subscriptions": [c.name for c in self._channels],
                },
                reply_id,
            )
        )

    async def _unsubscribe(self, message: dict[str, Any], reply_id: Any) -> None:
        channels = self._requested(message)
        for channel in channels:
            self._channels.pop(channel, None)
            # The cursors go with it, so resubscribing later starts from the
            # market as it is then rather than replaying everything that
            # happened while nobody was listening.
            self._forget(channel)
        await self._send(
            _tag(
                {
                    "type": "unsubscribed",
                    "channel": None,
                    "channels": [c.name for c in channels],
                    "subscriptions": [c.name for c in self._channels],
                },
                reply_id,
            )
        )

    def _open_cursor(self, channel: _Channel) -> None:
        """Mark where an event subscription begins, at the instant it is asked for.

        Taken here rather than on the next flush, because the two are not the
        same instant: the flush loop runs behind a market step that can take
        tens of milliseconds, and a print that lands in that gap would be read
        as history and skipped. The same gap, on the private side, loses a
        client the acknowledgement of an order it sent immediately after
        subscribing, which is precisely what a systematic trader does.

        Every cursor here is opened with ``setdefault`` or an ``is None``
        guard, never overwritten. A resume restores the cursors it left off at
        and then re-opens its channels; a cursor that reset itself on the way
        through would put the client back at *now* and lose everything it
        reconnected in order to collect.
        """
        venue = self._runner.market.venue
        if channel.kind == "trades":
            for symbol in self._symbols_for(channel):
                self._tape_seen.setdefault(symbol, len(venue.engine(symbol).tape))
            return
        if channel.kind == "lifecycle":
            if self._halts_seen is None:
                self._halts_seen = len(getattr(venue, "halts", ()))
            if self._listed_seen is None:
                self._listed_seen = set(venue.registry.symbols)
            if self._settled_seen is None:
                self._settled_seen = set(getattr(venue, "settled_symbols", ()))
            for symbol in self._symbols_for(channel):
                self._session_seen.setdefault(symbol, venue.session(symbol).value)

    def _open_private_cursor(self) -> None:
        """Mark where this seat's blotter begins, at the instant it is bound."""
        self._private_started = True
        self._private_mark = None
        agent = None
        if self._seat is not None and self._seat.agent_id is not None:
            agent = self._runner.market.traders.get(self._seat.agent_id)
        if agent is not None and agent.log:
            self._private_mark = _identity(agent.log[-1])

    def _forget(self, channel: _Channel) -> None:
        for symbol in self._symbols_for(channel):
            self._published.pop(channel.concrete(symbol), None)
            if channel.kind == "trades":
                self._tape_seen.pop(symbol, None)
        # The lifecycle cursors are one set for the connection rather than one
        # per channel (the venue's halt log is venue-wide), so they are only
        # closed once the last lifecycle channel has gone. Closing them while
        # another lifecycle subscription was still open would silently stop
        # that one: the halts cursor would be reopened at the end of the log
        # and every event in between would belong to nobody.
        if channel.kind == "lifecycle" and not any(
            other.kind == "lifecycle" for other in self._channels
        ):
            self._halts_seen = None
            self._listed_seen = None
            self._settled_seen = None
            self._session_seen.clear()

    async def _get_snapshot(self, message: dict[str, Any], reply_id: Any) -> None:
        """A fresh full state for named channels, changing nothing.

        This is the message that turns gap *detection* into gap *recovery*. A
        client that sees 41 after 39 knows its book is wrong; before this, its
        only way back to a correct one was to tear the subscription down and
        rebuild it, and a client that reconnects to repair one contract loses
        its place on every other contract it was watching. Kalshi's orderbook
        channel has exactly this shape (a snapshot, then increments, and a
        ``get_snapshot`` action that re-snapshots without resubscribing) and
        the reason is the same.

        It is defined over state channels only. A ``trades`` snapshot would
        have to be either a lie or a replay: there is no current value of a
        tape, and quietly answering with an empty frame would read as "nothing
        has traded". A missed print is recovered by ``resume``, or by the
        cursor that never dropped it in the first place.

        Nothing here writes to ``_published`` or to ``_anchor``, and that is
        deliberate rather than an oversight. Recording a snapshot as published
        would let a channel the client is *not* subscribed to suppress the
        first live frame it receives when it later subscribes, a gap
        manufactured by the very command that exists to close one.

        A live frame from the flush loop can land in the middle of the answer,
        because the write lock is taken per frame and the market does not stop
        for a snapshot. The client is given two ways to tell them apart rather
        than a promise of adjacency it would be a lie to make: every snapshot
        frame carries ``"snapshot": true`` and echoes the command's ``id``, and
        the acknowledgement says how many of them are coming.
        """
        channels = self._requested(message)
        registry = self._runner.market.venue.registry
        for channel in channels:
            if channel.kind not in STATE_KINDS:
                raise _BadChannel(
                    "invalid_request",
                    channel.name,
                    reason=f"only {' and '.join(STATE_KINDS)} have a snapshot",
                )
            if channel.symbol != ALL and registry.get(channel.symbol) is None:
                raise _BadChannel("invalid_symbol", channel.name, symbol=channel.symbol)

        frames = self._snapshot_frames(channels)
        # The acknowledgement first, carrying the count and the subscription
        # list *unchanged*, so a client can see for itself that asking for a
        # snapshot did not sign it up for anything.
        await self._send(
            _tag(
                {
                    "type": "snapshot",
                    "channel": None,
                    "channels": [channel.name for channel in channels],
                    "frames": len(frames),
                    "subscriptions": [channel.name for channel in self._channels],
                },
                reply_id,
            )
        )
        for frame in frames:
            if self._closed:
                return
            await self._send(_tag(frame, reply_id))

    def _snapshot_frames(self, channels: list[_Channel]) -> list[dict[str, Any]]:
        """The current state of each named channel, marked as a snapshot.

        ``"snapshot": true`` is on the frame rather than left implicit in the
        request. Bybit states the client's half of this contract plainly (a new
        snapshot means reset your local orderbook) and a client can only act on
        that if a solicited full state is distinguishable from the conflated
        update that follows it.
        """
        venue = self._runner.market.venue
        frames: list[dict[str, Any]] = []
        for channel in channels:
            for symbol in self._symbols_for(channel):
                instrument = venue.registry.require(symbol)
                engine = venue.engine(symbol)
                book = engine.book.snapshot(DEPTH_LEVELS)
                if channel.kind == "ticker":
                    payload = self._ticker(instrument, book, engine.tape, remember=False)
                else:
                    payload = self._book(instrument, book)
                frames.append(
                    {
                        **payload,
                        "type": channel.kind,
                        "channel": channel.concrete(symbol),
                        "snapshot": True,
                    }
                )
        return frames

    async def _auth(self, message: dict[str, Any], reply_id: Any) -> None:
        """Prove the connection holds a credential, and seat it.

        Every failure answers ``auth_invalid`` with one sentence, which is the
        rule :class:`arena.api.keys.SignatureError` sets and the reason is
        worth restating: saying *which* part was wrong tells a caller holding
        no valid key which key ids exist, and a caller holding a valid one
        never needs the difference. A missing field is treated the same way for
        the same reason.
        """
        if self._keys is None:
            await self._refuse("auth_invalid", reply_id=reply_id)
            return
        try:
            key = self._keys.verify(
                key_id=str(message.get("key_id") or ""),
                timestamp=str(message.get("timestamp") or ""),
                signature=str(message.get("signature") or ""),
                method="GET",
                path=self._path,
                body=b"",
            )
        except SignatureError:
            await self._refuse("auth_invalid", reply_id=reply_id)
            return

        self._key = key
        self._seat = _seat_for(self._runner, key, self._seat_now)
        # Stamped on the session as well as on the connection. From here on
        # this session's buffer can hold this seat's blotter, so resuming it is
        # reading a private feed and has to be proved the same way subscribing
        # to one is (see :meth:`_resume`).
        self._session.token = str(key.agent_id)
        # A different seat has a different blotter, so the cursor into the old
        # one means nothing. Opened here rather than left for the flush loop:
        # a client that authenticates and immediately sends an order would
        # otherwise have its own acknowledgement arrive before the first flush,
        # be read as history, and never be delivered.
        self._open_private_cursor()
        await self._send(
            _tag(
                {
                    "type": "auth",
                    "channel": None,
                    "seat": self._seat_payload(),
                    "generation": self._runner.generation,
                },
                reply_id,
            )
        )

    def _seat_payload(self) -> dict[str, Any] | None:
        if self._seat is None or self._seat.agent_id is None:
            return None
        return {"id": str(self._seat.agent_id), "name": self._seat.name}

    # -- resuming ----------------------------------------------------------

    async def _resume(self, message: dict[str, Any], reply_id: Any) -> None:
        """Pick a dropped stream back up where it stopped, or say why not.

        SoupBinTCP's guarantee is the one being implemented, verbatim: if the
        connection is broken the client re-logs in naming the session and its
        next expected sequence number, and is then guaranteed to receive every
        sequenced message in order despite the failure. So the reply names
        ``next_seq`` the way Login Accepted does, the replay carries the
        original numbers, and the connection then continues that same sequence.

        The whole exchange happens under one hold of the write lock. Taking it,
        releasing it, and taking it again would let the flush loop (which is a
        separate task on the same event loop and is queued on the same lock)
        slip a live frame numbered 58 between the acknowledgement and the
        replay of 41, which is out of order on the wire and unrecoverable by
        the client that just asked to be made whole.

        Every refusal is a ``resume_failed`` frame naming a reason, and never a
        silent fresh start. A client told nothing would carry on believing it
        had been made whole and trade a book with a hole in it; that is the
        failure mode this command exists to remove, so it is the one outcome
        that must be impossible.
        """
        wanted = message.get("session")
        raw = message.get("from_seq")
        session_id = wanted if isinstance(wanted, str) else ""
        # ``bool`` is an ``int`` in Python and ``{"from_seq": true}`` would
        # otherwise resume from 1.
        from_seq = raw if isinstance(raw, int) and not isinstance(raw, bool) else None

        async with self._writing:
            session = _SESSIONS.get(session_id) if session_id else None
            reason = self._resume_refusal(session, session_id, from_seq)
            if reason is not None or session is None or from_seq is None:
                # How far the session reaches is told only to a caller that
                # was entitled to resume it and simply asked for too much. On
                # ``not_authorised`` those two numbers would measure somebody
                # else's stream for whoever came into possession of the id,
                # which is a smaller leak than the frames themselves and still
                # a leak.
                measurable = reason in ("beyond_buffer", "ahead_of_session")
                await self._emit_locked(
                    _tag(
                        {
                            "type": "resume_failed",
                            "channel": None,
                            "reason": reason or "malformed",
                            # The session now in force, which is *this*
                            # connection's own, so a client whose resume was
                            # refused knows what to quote if it drops again,
                            # and cannot mistake the refusal for a success.
                            "session": self._session.session_id,
                            "from_seq": from_seq,
                            "retained_from": (
                                session.retained_from
                                if session is not None and measurable
                                else None
                            ),
                            "retained_through": (
                                session.seq
                                if session is not None and measurable
                                else None
                            ),
                            "retained_frames": self._session.frames.maxlen,
                            "resnapshot": True,
                        },
                        reply_id,
                    )
                )
                return

            replay = [
                frame for frame in session.frames if int(frame["seq"]) >= from_seq
            ]
            # Worked out before the session is adopted, because the reply is
            # numbered in the sequence being left behind and has to be built
            # while that is still the one in force.
            restored = self._restorable(session)
            await self._emit_locked(
                _tag(
                    {
                        "type": "resumed",
                        "channel": None,
                        "session": session.session_id,
                        "from_seq": from_seq,
                        # Always equal to ``from_seq`` on success, and stated
                        # anyway: it is what the client asked to be told, and a
                        # client that checks it can tell a resume that honoured
                        # its request from one that quietly moved the goalposts.
                        "next_seq": from_seq,
                        "replayed": len(replay),
                        "subscriptions": [channel.name for channel in restored],
                        "generation": self._runner.generation,
                    },
                    reply_id,
                )
            )
            self._adopt(session, restored)
            for frame in replay:
                if self._closed:
                    return
                await self._write_locked(frame)

    def _resume_refusal(
        self, session: _Session | None, session_id: str, from_seq: int | None
    ) -> str | None:
        """Why this resume cannot be honoured, or ``None`` if it can.

        Separated out because the order of these checks is the security
        argument. The seat check comes before anything that would tell a caller
        how far a session reaches, so a session id that leaked cannot be used
        to measure somebody else's stream, and a caller holding no credential
        learns nothing beyond "no".
        """
        if not session_id or from_seq is None or from_seq < 1:
            return "malformed"
        if self._resumed:
            # One resume per connection. A second would abandon a sequence the
            # client is mid-way through reading and start another underneath
            # it, with no number the client could use to tell where the switch
            # happened.
            return "already_resumed"
        if self._channels:
            # Resuming replaces the subscription list with the resumed
            # session's, so doing it after subscribing would throw away
            # channels the client had already asked for on this connection,
            # quietly, and with a cursor the client believes is running.
            # Refusing says so. Authenticating first is fine and is in fact
            # required to resume a session that was authenticated.
            return "already_subscribed"
        if session is None:
            # Never issued, or aged out of the table. Indistinguishable on
            # purpose: which of the two it is says how long ago somebody else
            # was connected.
            return "unknown_session"
        if session.live:
            return "in_use"
        if session.token is not None and (
            self._key is None or str(self._key.agent_id) != session.token
        ):
            return "not_authorised"
        if from_seq > session.seq + 1:
            # The client claims to have seen frames this session never sent.
            # It is talking about a different stream, and replaying from the
            # end would hand it a sequence that jumps backwards.
            return "ahead_of_session"
        if from_seq < session.retained_from:
            return "beyond_buffer"
        return None

    def _restorable(self, session: _Session) -> list[_Channel]:
        """The subscriptions a resumed session may be given back.

        Re-checked rather than trusted. The market may have been rebuilt while
        the client was away and no longer list a symbol it was following, and a
        private channel is only restored to a connection that has proved the
        same seat: the token check in :meth:`_resume_refusal` establishes that,
        and this is where it is spent. A name that cannot be honoured is
        dropped rather than failing the resume: the frames are still worth
        replaying, and the reply reports what was actually restored so the
        client can diff it against what it asked for.
        """
        registry = self._runner.market.venue.registry
        restored: list[_Channel] = []
        for name in session.channels:
            try:
                channel = _parse_channel(name)
            except _BadChannel:
                continue
            if channel.private and self._seat is None:
                continue
            if (
                channel.symbol
                and channel.symbol != ALL
                and registry.get(channel.symbol) is None
            ):
                continue
            restored.append(channel)
        return restored

    def _adopt(self, session: _Session, restored: list[_Channel]) -> None:
        """Take over a session: its numbering, its buffer and its cursors.

        The cursors are the point. A resume that restored the subscription list
        and reopened the tape cursor at *now* would drop every print that
        landed between the old socket dying and the new one connecting, and
        drop it invisibly: the sequence would run straight across the hole, so
        the one mechanism this feed gives a client for noticing loss would
        report that nothing was lost.

        The generation is restored too, and deliberately not reconciled here.
        If the market was rebuilt while the client was away, the next flush
        sees the generation move and pushes a ``reset`` through the path that
        already exists for it, which is the correct answer, because the
        replayed frames describe a market that no longer exists.
        """
        with _SESSION_LOCK:
            # The connection's own session is abandoned rather than kept. It
            # holds a handful of frames the resumed stream never contained (the
            # hello, the auth) and leaving it in the table would let a confused
            # client resume its way into a two-frame stream.
            _SESSIONS.pop(self._session.session_id, None)
        self._session = session
        session.live = True
        self._resumed = True

        self._channels = dict.fromkeys(restored)
        self._generation = session.generation
        self._tape_seen = dict(session.tape_seen)
        self._published = dict(session.published)
        self._anchor = dict(session.anchor)
        self._private_mark = session.private_mark
        self._private_started = session.private_started
        self._listed_seen = (
            None if session.listed_seen is None else set(session.listed_seen)
        )
        self._session_seen = dict(session.session_seen)
        self._halts_seen = session.halts_seen
        self._settled_seen = (
            None if session.settled_seen is None else set(session.settled_seen)
        )
        # Only fills in cursors the restored session did not already carry:
        # every opener here is a ``setdefault`` or an ``is None`` guard.
        for channel in restored:
            self._open_cursor(channel)

    # -- flushing ----------------------------------------------------------

    async def _flush(self) -> None:
        """Collect one instant of the market, then write what it produced.

        Collection is synchronous on purpose. The kernel is stepped by another
        task on this same event loop, so a collection that never awaits sees
        one consistent instant, and an ``await`` in the middle of it would let
        the market advance between reading a book and reading the tape,
        publishing a state that never existed.
        """
        frames = self._collect()
        for frame in frames:
            if self._closed:
                return
            await self._send(frame)

    def _collect(self) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        rebuilt = self._sync_generation()

        reseated = False
        if self._key is not None:
            before = self._seat.agent_id if self._seat is not None else None
            self._seat = _seat_for(self._runner, self._key, self._seat_now)
            reseated = self._seat.agent_id != before
            if reseated:
                self._open_private_cursor()
        if rebuilt or reseated:
            # Said out loud rather than switched to silently. After a rebuild
            # the client's working orders are gone, its positions are gone and
            # its cash is back where it started; a client that is not told
            # reconciles an empty blotter against orders it believes are live
            # and concludes the venue lost them. A public subscriber needs it
            # too: every price on the feed is about to jump to a different
            # session, and the contract set may not even be the same.
            frames.append(
                {
                    "type": "reset",
                    "channel": None,
                    "reason": "rebuild" if rebuilt else "reseat",
                    # Null means the whole connection rather than a list of
                    # channels: after a rebuild every cursor this connection
                    # holds indexes something that no longer exists.
                    "channels": None,
                    # Bybit's contract, stated by the server rather than left
                    # for the client to infer: a reset means discard the local
                    # book and take a new snapshot. ``get_snapshot`` is how.
                    "resnapshot": True,
                    "generation": self._runner.generation,
                    "seat": self._seat_payload(),
                }
            )

        # Private before public, the order the venue itself dispatches in: an
        # agent learns of its own fill before the tape learns a trade happened.
        # The reverse would let a client react to a print of its own execution
        # before it had been told it executed.
        frames.extend(self._private_frames())
        frames.extend(self._public_frames())
        return frames

    def _sync_generation(self) -> bool:
        """Notice a rebuild, and forget everything that described the old market.

        Every cursor here indexes into something ``reconfigure`` threw away: an
        engine's tape, a seat's blotter, the last book published for a symbol.
        Carrying any of them across a rebuild reads the new market through the
        old market's bookmarks: a tape cursor at 400 against a tape that starts
        again at 0 publishes nothing until the new session has traded four
        hundred times.
        """
        generation = self._runner.generation
        if self._generation == generation:
            return False
        first = self._generation is None
        self._generation = generation
        # Symbols already being followed restart at the beginning of the new
        # market's tape, which is a complete record of it. Symbols not yet
        # followed keep no entry, so a later subscribe still starts at the end.
        self._tape_seen = dict.fromkeys(self._tape_seen, 0)
        self._published.clear()
        self._anchor.clear()
        self._private_started = False
        self._private_mark = None
        # The lifecycle cursors index the old venue's halt log and the old
        # venue's registry, so they are reopened against the new one. The halt
        # log starts again from empty, and a cursor left at 12 would swallow
        # the new market's first twelve halts. The listings and settlements are
        # reopened at what the new venue already has rather than replayed: a
        # rebuild is one new market, not twenty-eight new contracts, and the
        # ``reset`` frame going out alongside this already says the world
        # changed. Announcing every symbol as freshly listed would bury that.
        venue = self._runner.market.venue
        if self._halts_seen is not None:
            self._halts_seen = len(getattr(venue, "halts", ()))
        if self._listed_seen is not None:
            self._listed_seen = set(venue.registry.symbols)
        if self._settled_seen is not None:
            self._settled_seen = set(getattr(venue, "settled_symbols", ()))
        self._session_seen.clear()
        return not first

    # -- public channels ---------------------------------------------------

    def _symbols_for(self, channel: _Channel) -> tuple[str, ...]:
        if not channel.symbol:
            return ()
        registry = self._runner.market.venue.registry
        if channel.symbol == ALL:
            return tuple(registry.symbols)
        return (channel.symbol,) if registry.get(channel.symbol) is not None else ()

    def _public_frames(self) -> list[dict[str, Any]]:
        venue = self._runner.market.venue
        frames: list[dict[str, Any]] = []
        gone: list[_Channel] = []
        # Read once per symbol per flush and shared between the channels that
        # need them, rather than once per channel. ``MatchingEngine.tape``
        # copies the whole list on every read and ``OrderBook.snapshot`` walks
        # the ladder, and a client subscribed to ticker, book and trades on
        # twenty-eight contracts would otherwise pay for each of them three
        # times, twenty times a second.
        tapes: dict[str, tuple[Any, ...]] = {}
        books: dict[str, Any] = {}

        # Drained once for the connection, not once per channel. The venue's
        # halt log is one list for the whole venue, so a per-channel drain
        # would let the first lifecycle channel consume an event and leave the
        # second with nothing: a client subscribed to both ``lifecycle.*`` and
        # ``lifecycle.SOME_SYMBOL`` would see each halt on exactly one of them,
        # chosen by iteration order.
        events = (
            self._lifecycle_events()
            if any(channel.kind == "lifecycle" for channel in self._channels)
            else []
        )

        for channel in list(self._channels):
            if channel.private:
                continue
            symbols = self._symbols_for(channel)
            if not symbols and channel.symbol != ALL:
                # The instrument was listed when this was subscribed and is not
                # listed now, which happens when a rebuild changes the venue's
                # contract set. Silently streaming nothing would leave a client
                # waiting on a symbol that no longer exists.
                gone.append(channel)
                continue
            for symbol in symbols:
                if channel.kind == "lifecycle":
                    frames.extend(
                        {**payload, "type": "lifecycle", "channel": channel.concrete(symbol)}
                        for about, payload in events
                        if about == symbol
                    )
                    continue
                instrument = venue.registry.require(symbol)
                if channel.kind in ("trades", "ticker") and symbol not in tapes:
                    tapes[symbol] = venue.engine(symbol).tape
                if channel.kind in ("book", "ticker") and symbol not in books:
                    books[symbol] = venue.engine(symbol).book.snapshot(DEPTH_LEVELS)

                if channel.kind == "trades":
                    frames.extend(self._trade_frames(channel, instrument, tapes[symbol]))
                    continue
                if channel.kind == "ticker":
                    payload = self._ticker(instrument, books[symbol], tapes[symbol])
                else:
                    payload = self._book(instrument, books[symbol])
                label = channel.concrete(symbol)
                if self._published.get(label) == payload:
                    # Conflation with nothing left to conflate. The state has
                    # not moved, so there is no update being withheld, only a
                    # frame that would tell the client what it already knows.
                    continue
                self._published[label] = payload
                frames.append({"type": channel.kind, "channel": label, **payload})

        for channel in gone:
            self._channels.pop(channel, None)
            frames.append(
                _error_frame("invalid_symbol", channel.name, symbol=channel.symbol)
            )
        return frames

    def _ticker(
        self,
        instrument: Any,
        snapshot: Any,
        tape: tuple[Any, ...],
        *,
        remember: bool = True,
    ) -> dict[str, Any]:
        """Top of book, the mark, and what it has done since we started watching.

        ``change`` is measured from the first mark this connection published
        for the contract, and ``open`` names that anchor so the number is never
        ambiguous. The obvious alternative (change on the session) would mean
        reading ``MarketRunner.history``, which is kept in floats for charting.
        A chart may round; a price on this feed may not.

        No timestamp. A conflated state frame does not have an honest one: the
        book may have moved four times since the last flush, and stamping the
        frame with the moment it was assembled would claim a freshness the
        contents do not have. Event channels carry the clock instead, because
        an event happened at a time.

        ``remember=False`` is the snapshot path, and it reads the anchor
        without setting one. A ``get_snapshot`` on a contract this connection
        is not subscribed to would otherwise fix that contract's ``open`` at
        the moment of the snapshot, so a client that snapshotted first and
        subscribed second would be quoted a ``change`` measured from a
        different instant than the one it thinks: a command that promises to
        change nothing, silently changing what a later field means.
        """
        venue = self._runner.market.venue
        symbol = instrument.symbol
        bids, asks = snapshot.priced_bids, snapshot.priced_asks
        mark = from_money(venue.mark(symbol))
        if remember:
            anchor = self._anchor.setdefault(symbol, mark)
        else:
            anchor = self._anchor.get(symbol, mark)
        return {
            "symbol": symbol,
            "mark": str(mark),
            "open": str(anchor),
            "change": str(mark - anchor),
            "bid": _price(instrument, bids[0][0]) if bids else None,
            "bid_size": int(bids[0][1]) if bids else 0,
            "ask": _price(instrument, asks[0][0]) if asks else None,
            "ask_size": int(asks[0][1]) if asks else 0,
            "last": _price(instrument, tape[-1].price) if tape else None,
            "trades": len(tape),
            "session": venue.session(symbol).value,
        }

    def _book(self, instrument: Any, snapshot: Any) -> dict[str, Any]:
        """Aggregated depth, priced levels only.

        Market-on-open interest rests at a sentinel so that it crosses every
        candidate an auction weighs. It is the top of the book by a margin of
        2^61 and it names no price at all, and publishing it put a bid of
        4,611,686,018,427,387,904 on the dashboard. ``priced_bids`` is the
        filtered view and is what anything outside the matching engine reads.
        """
        venue = self._runner.market.venue
        symbol = instrument.symbol
        return {
            "symbol": symbol,
            "bids": [[_price(instrument, p), int(q)] for p, q in snapshot.priced_bids],
            "asks": [[_price(instrument, p), int(q)] for p, q in snapshot.priced_asks],
            "session": venue.session(symbol).value,
            "indicative": self._indicative(instrument),
        }

    def _indicative(self, instrument: Any) -> str | None:
        """Where a call in progress would clear, as a price.

        Published during an auction exactly as real venues publish one, so a
        program can respond to the auction rather than only to its result.
        ``AuctionResult.to_dict`` answers in ticks, which is right for the
        exchange and wrong for anything published, so only the converted price
        crosses this wire.
        """
        venue = self._runner.market.venue
        symbol = instrument.symbol
        if venue.session(symbol) is SessionState.CONTINUOUS:
            return None
        result = venue.indicative(symbol)
        if result is None or int(result.volume) <= 0:
            return None
        return _price(instrument, result.price)

    def _trade_frames(
        self, channel: _Channel, instrument: Any, tape: tuple[Any, ...]
    ) -> list[dict[str, Any]]:
        """Every print since the last flush, in order, none of them conflated.

        A print is a fact and the next one does not supersede it, so this
        channel holds a cursor rather than a latest value. The cursor is an
        index into the engine's own tape, which is per symbol and unbounded, so
        a client that stops reading for ten seconds is caught up in order
        rather than told a version of the session that skips.

        The venue's timestamped copy of the same prints, ``public_log``, would
        carry a clock, and is bounded at 5,000 entries shared across every
        contract the venue lists. On a market with twenty-eight of them a busy
        contract pushes a quiet one's print out of that window, which is
        dropping, which is the thing this module exists not to do. So the
        exchange's own sequence number identifies and orders a print here, and
        the wall clock is the price paid for delivering all of them.

        ``buy_order_id`` and ``sell_order_id`` are deliberately not published.
        The engine assigns the same ids that appear on an ``ack``, so a public
        print carrying them lets any observer match a trade to the private
        acknowledgement of whoever made it, and deanonymise the tape of a
        simulator built to study who knows what.
        """
        symbol = instrument.symbol
        seen = self._tape_seen.get(symbol)
        if seen is None:
            # A new subscriber gets the market from here. Replaying a session's
            # worth of prints at somebody who asked for a live feed is a denial
            # of service dressed as helpfulness.
            self._tape_seen[symbol] = len(tape)
            return []
        if seen >= len(tape):
            return []
        self._tape_seen[symbol] = len(tape)
        return [
            {
                "type": "trade",
                "channel": channel.concrete(symbol),
                "symbol": symbol,
                "price": _price(instrument, trade.price),
                "quantity": int(trade.quantity),
                "side": trade.aggressor_side.value,
                "exchange_seq": int(trade.sequence),
            }
            for trade in tape[seen:]
        ]

    # -- the lifecycle channel ---------------------------------------------

    def _lifecycle_events(self) -> list[tuple[str, dict[str, Any]]]:
        """What has happened *to* the contracts since the last flush.

        Not to their prices, but to the contracts themselves: one being listed,
        one moving between trading phases, one being stopped, one paying out.
        Kalshi publishes the same thing as ``market_lifecycle_v2`` and the
        reason is that an algorithm running unattended has no other way to find
        out. Without it, a new listing is invisible until somebody polls the
        registry, and a halt is indistinguishable from a contract that simply
        went quiet, so a maker keeps quoting into a book that is not going to
        trade, and a taker waits for a fill that cannot arrive.

        Returned as ``(symbol, payload)`` pairs rather than as frames because
        the caller decides which subscription each one belongs to, and one
        event can belong to more than one.

        The order within a flush is deliberate: a listing before anything that
        can be said about the contract, then the halt that explains a phase
        change, then the phase change, then the settlement. A client reading
        them in order never has to hold an event about a symbol it has not been
        told exists.
        """
        venue = self._runner.market.venue
        if self._listed_seen is None or self._halts_seen is None:
            return []
        events: list[tuple[str, dict[str, Any]]] = []

        for symbol in venue.registry.symbols:
            if symbol in self._listed_seen:
                continue
            self._listed_seen.add(symbol)
            instrument = venue.registry.require(symbol)
            state = venue.session(symbol).value
            # Seeded here as well, so the listing is not immediately followed
            # by a phase change out of nothing.
            self._session_seen[symbol] = state
            window = instrument.spec.window
            events.append(
                (
                    symbol,
                    {
                        "event": "listed",
                        "symbol": symbol,
                        "session": state,
                        # Kalshi's ``created`` carries the open and close of
                        # the market it is announcing, and for the same reason:
                        # a bot deciding whether to trade a new listing needs
                        # to know how long it has, and a second round trip to
                        # find out is a second round trip it will skip.
                        "open_time": _iso(window.start),
                        "close_time": _iso(window.end),
                        "instrument": instrument.to_dict(),
                    },
                )
            )

        halts = getattr(venue, "halts", ())
        if self._halts_seen < len(halts):
            for record in list(halts[self._halts_seen :]):
                symbol = str(record.get("symbol") or "")
                events.append((symbol, self._halt_payload(symbol, record)))
            self._halts_seen = len(halts)

        for symbol in venue.registry.symbols:
            state = venue.session(symbol).value
            was = self._session_seen.get(symbol)
            if was is None:
                self._session_seen[symbol] = state
            elif was != state:
                self._session_seen[symbol] = state
                events.append(
                    (
                        symbol,
                        {
                            "event": "session",
                            "symbol": symbol,
                            "from": was,
                            "to": state,
                        },
                    )
                )

        settled = set(getattr(venue, "settled_symbols", ()))
        fresh = settled - (self._settled_seen or set())
        if fresh:
            self._settled_seen = settled
            for symbol in sorted(fresh):
                events.append(
                    (
                        symbol,
                        {
                            "event": "settled",
                            "symbol": symbol,
                            "session": venue.session(symbol).value,
                        },
                    )
                )
        return events

    def _halt_payload(self, symbol: str, record: dict[str, Any]) -> dict[str, Any]:
        """A halt record, in units a client can read.

        The venue keeps these in its own units (the price it was banding is a
        tick count, the band itself is a fraction) and both of those are the
        settlement bug over again if they cross this wire untouched: a number
        four times out on a 0.25 grid reads as a number rather than as a
        mistake. So a price becomes a price, and anything left holding a float
        becomes a string, because a float in a price path is a price path that
        eventually disagrees with the ledger.

        Everything else in the record passes through rather than being picked
        from a list of the fields halts carry today. The breaker gains a field
        and the client sees it; a whitelist nobody remembered to extend would
        drop it silently, which is how the reason for a halt goes missing.
        """
        venue = self._runner.market.venue
        instrument = venue.registry.get(symbol)
        payload: dict[str, Any] = {}
        for key, value in record.items():
            if key == "symbol":
                continue
            if key in ("price", "reference"):
                if instrument is None:
                    # No tick grid to convert with, so there is no honest way
                    # to say this number. Dropped rather than published raw:
                    # 20012 under a field called ``price`` on a contract quoted
                    # on a 0.25 grid is four times the figure a reader expects
                    # and looks like a price rather than like a bug. The reason
                    # for the halt is the actionable part and it survives.
                    continue
                payload[key] = _price(instrument, value)
            elif isinstance(value, float):
                payload[key] = str(value)
            else:
                payload[key] = value
        # The envelope last, so nothing a halt record happens to be carrying
        # can overwrite the fields a client routes on.
        return {
            **payload,
            "event": "halt",
            "symbol": symbol,
            "session": venue.session(symbol).value if instrument is not None else None,
        }

    # -- private channels --------------------------------------------------

    def _private_frames(self) -> list[dict[str, Any]]:
        """This seat's own events, routed to ``fills`` or ``orders``.

        Read on every flush whether or not either channel is subscribed, so the
        cursor tracks the present. Without that, subscribing to ``orders`` an
        hour into a session would replay the hour.
        """
        if self._seat is None or self._seat.agent_id is None:
            return []
        market = self._runner.market
        agent = market.traders.get(self._seat.agent_id)
        if agent is None:
            return []

        wanted = {c.kind for c in self._channels if c.private}
        entries, lost = self._drain_private(list(agent.log))
        frames: list[dict[str, Any]] = []
        if lost and wanted:
            # Detected rather than assumed: the entry last forwarded is no
            # longer in the agent's window, so entries between then and now are
            # gone and no cursor can recover them. Saying so is the difference
            # between a client that knows to reconcile and one that believes it
            # has seen everything.
            #
            # A ``reset`` rather than an error, which is what this used to be.
            # An ``invalid_request`` tells a client to fix its request, and
            # there is nothing wrong with its request: the correct response is
            # to discard the private view and re-read it, which is what every
            # other cursor invalidation on this connection already says. A
            # client branching on error codes would have treated the one frame
            # that means "your positions may be wrong" as a typo.
            frames.append(
                {
                    "type": "reset",
                    "channel": None,
                    "reason": "blotter_gap",
                    "channels": sorted(wanted),
                    "resnapshot": True,
                    "generation": self._runner.generation,
                    "seat": self._seat_payload(),
                }
            )
        for entry in entries:
            kind = str(entry.get("type") or "")
            # Anything that is not an execution is news about an order. Routing
            # by what a fill *is*, rather than by a list of the event types that
            # exist today, means a new engine event reaches the client on the
            # orders channel instead of being silently dropped by a whitelist
            # nobody remembered to extend.
            channel = "fills" if kind == "fill" else "orders"
            if channel not in wanted:
                continue
            frames.append(self._private_frame(channel, kind, entry))
        return frames

    def _private_frame(
        self, channel: str, kind: str, entry: dict[str, Any]
    ) -> dict[str, Any]:
        payload = {k: v for k, v in entry.items() if k not in ("type", "sequence")}
        price = payload.get("price")
        if price is not None and not isinstance(price, str):
            # The blotter already converts ticks to a price, but this is a
            # price path and a raw tick count reaching a client under the name
            # ``price`` is the settlement bug over again: a number four times
            # out, on a 0.25 grid, that reads as a number rather than as a bug.
            instrument = self._runner.market.venue.registry.get(
                str(entry.get("symbol") or "")
            )
            payload["price"] = _price(instrument, price) if instrument else str(price)
        # The envelope is written last, so nothing a blotter entry happens to
        # be carrying can overwrite the fields a client routes on.
        return {
            **payload,
            "type": "fill" if channel == "fills" else "order",
            "channel": channel,
            "status": kind,
            "exchange_seq": entry.get("sequence"),
        }

    def _drain_private(
        self, log: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], bool]:
        """New blotter entries since the last flush, and whether any were lost.

        Held as the identity of the last entry forwarded rather than as an
        index, because ``HumanAgent`` keeps only the last 200 entries: an index
        into a list that slides is an index into the wrong entries, and it
        slides silently. Searching for the last entry forwarded answers both
        questions at once: where to resume, and whether the resume point is
        still there at all.

        ``(symbol, exchange sequence)`` identifies an entry: there is one
        matching engine per symbol and it numbers every event it emits in
        strict order, so the pair is unique even though ids repeat across
        books.
        """
        if not self._private_started:
            # A fallback. The cursor is normally opened the moment a seat is
            # bound, which is earlier and exact; this only fires if a flush
            # somehow reaches a seat that was never opened, and it starts at
            # the end rather than replaying a session at somebody.
            self._private_started = True
            self._private_mark = _identity(log[-1]) if log else None
            return [], False

        if self._private_mark is None:
            fresh = list(log)
        else:
            cut = None
            for index in range(len(log) - 1, -1, -1):
                if _identity(log[index]) == self._private_mark:
                    cut = index + 1
                    break
            if cut is None:
                self._private_mark = _identity(log[-1]) if log else None
                return list(log), True
            fresh = log[cut:]
        if fresh:
            self._private_mark = _identity(fresh[-1])
        return fresh, False


def _identity(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("symbol"),
        entry.get("sequence"),
        entry.get("type"),
        entry.get("order_id"),
    )


def _tag(frame: dict[str, Any], reply_id: Any) -> dict[str, Any]:
    """Echo a command's id onto the frame answering it, if it had one."""
    if reply_id is not None:
        frame["id"] = reply_id
    return frame


def _iso(moment: Any) -> str:
    """A contract's timestamp, in the one spelling this API publishes.

    The same format ``Instrument.to_dict`` stamps an expiry with, written here
    rather than borrowed so a listing frame cannot end up describing its open
    in one spelling and its close in another, which is a difference a client
    parsing both with one format string discovers at run time.
    """
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# The endpoint
# --------------------------------------------------------------------------


def stream_endpoint(
    *,
    keys: KeyStore | None = None,
    runner: Any | None = None,
    path: str | None = None,
    tick_seconds: float | None = None,
    seat_now: Callable[[str], Any | None] | None = None,
    resume_frames: int | None = None,
) -> Callable[[WebSocket], Awaitable[None]]:
    """Build the websocket endpoint, for the application to mount.

    A factory rather than a decorated module-level function, so that an
    application can mount it under a different path, and a test can mount it
    against its own market and key store without reaching into module state
    that another test is also using.

    Arguments given here win over :func:`configure`, and both are resolved when
    a client connects rather than when the endpoint is built, so mounting
    before the market exists is fine, which is the order an application
    starting up actually does things in.
    """

    async def endpoint(websocket: WebSocket) -> None:
        stream = _Stream(
            websocket,
            _resolve_runner(runner),
            keys if keys is not None else _KEYS,
            path if path is not None else _PATH,
            tick_seconds if tick_seconds is not None else _TICK,
            seat_now if seat_now is not None else _SEAT_NOW,
            resume_frames if resume_frames is not None else _RESUME_FRAMES,
        )
        await stream.run()

    return endpoint
