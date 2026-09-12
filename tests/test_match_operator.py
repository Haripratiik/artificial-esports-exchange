"""A board that fills itself, and settlement that arrives while you watch.

Two things were missing and this closes both. Nothing resolved inside a
session, so an algorithm left running had no terminal event to be scored
against. And nothing relisted, so the exchange drained as contracts expired.
Matches are generated rather than collected, so there is always another one.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from arena.market.match_operator import MatchOperator, book_key
from arena.market.match_book import list_match, settlement
from arena.sim.time import seconds
from arena.worlds.circuit.match import play

from dashboard.build_market import build


class _Ctx:
    """A clock the operator can read and a wakeup it can ask for.

    Driven by hand rather than by the kernel, so a test can step time to an
    exact instant and ask what had settled by then. The operator reads nothing
    else, which is most of why it is testable at all.
    """

    def __init__(self, now: int = 0) -> None:
        self.now = now
        self.wakeups: list[int] = []

    def request_wakeup(self, delay) -> None:
        self.wakeups.append(int(delay))


def _operator(seed: int = 7, **kwargs):
    market = build(seed=seed)
    operator = MatchOperator("matches", market.venue, seed=seed, **kwargs)
    return market, operator


# --------------------------------------------------------------------------
# The board
# --------------------------------------------------------------------------


def test_the_board_opens_full_and_stays_full():
    """An exchange with nothing listed is not quiet, it is closed."""
    market, operator = _operator(formats=("solo",), concurrent=2)
    ctx = _Ctx()
    operator.on_start(ctx)

    assert len(operator.live) == 2
    listed = [s for s in market.venue.registry.symbols if s.startswith("SOLO")]
    assert listed, "a match opened without listing anything"

    # Run past the end of both and check the board refilled rather than emptied.
    ctx.now = int(seconds(200))
    operator.on_wakeup(ctx)
    assert len(operator.live) == 2
    assert operator.opened > 2
    assert operator.settled >= 2


def test_a_symbol_is_never_relisted():
    """The registry refuses it, and it would silently change what a position
    settles into. Each match carries its own id into its symbols."""
    market, operator = _operator(formats=("solo",), concurrent=2)
    ctx = _Ctx()
    operator.on_start(ctx)
    for step in range(1, 6):
        ctx.now = int(seconds(step * 60))
        operator.on_wakeup(ctx)
    symbols = [s for s in market.venue.registry.symbols if s.startswith("SOLO")]
    assert len(symbols) == len(set(symbols))


# --------------------------------------------------------------------------
# Settlement arriving in pieces
# --------------------------------------------------------------------------


def test_an_elimination_settles_only_what_it_makes_certain():
    """Being out is definitive; where the survivors place is not.

    Settling more than this would be paying out a forecast, which is the one
    thing a settlement may never be.
    """
    market, operator = _operator(formats=("solo",), concurrent=1, match_seconds=100.0)
    ctx = _Ctx()
    operator.on_start(ctx)
    match = next(iter(operator.live.values()))

    ctx.now = int(seconds(50))
    operator.on_wakeup(ctx)
    assert 0 < match.revealed < len(match.knockout_order), "nothing was revealed"

    settled = market.venue.settled_symbols
    for competitor in match.eliminated:
        assert f"{match.book.contracts[0].symbol.split('_')[0]}_WIN_{competitor}" in settled

    # A survivor's winner contract is still open, because they might still win.
    survivors = [
        k for k in match.knockout_order[match.revealed :]
    ] + [match.book.sides[-1][0]]
    still_open = [
        c.symbol
        for c in match.book.contracts
        if c.family == "winner" and c.subject in survivors
    ]
    assert any(s not in settled for s in still_open)


def test_an_eliminated_competitor_settles_at_exactly_zero():
    """They cannot win, so the contract is worth nothing, exactly."""
    market, operator = _operator(formats=("solo",), concurrent=1, match_seconds=100.0)
    ctx = _Ctx()
    operator.on_start(ctx)
    match = next(iter(operator.live.values()))
    ctx.now = int(seconds(60))
    operator.on_wakeup(ctx)

    truth = settlement(match.book, play(7, match.match_id, "solo"))
    for competitor in match.eliminated:
        symbol = next(
            c.symbol
            for c in match.book.contracts
            if c.family == "winner" and c.subject == competitor
        )
        assert truth[symbol] == 0.0


def test_the_exclusive_set_still_sums_to_one_mid_match():
    """The settled legs pay zero and the live ones carry the whole of it.

    This is the property that makes early settlement safe. If an elimination
    removed value from the partition rather than moving it, the remaining
    contracts would price against a total that no longer added up.
    """
    market, operator = _operator(formats=("solo",), concurrent=1, match_seconds=100.0)
    ctx = _Ctx()
    operator.on_start(ctx)
    match = next(iter(operator.live.values()))
    ctx.now = int(seconds(70))
    operator.on_wakeup(ctx)

    truth = settlement(match.book, play(7, match.match_id, "solo"))
    winners = [c for c in match.book.contracts if c.family == "winner"]
    assert sum(truth[c.symbol] for c in winners) == pytest.approx(1.0)


def test_everything_open_is_resolved_when_the_match_ends():
    """A contract left unsettled is a position nobody can ever close."""
    market, operator = _operator(formats=("solo", "objective"), concurrent=1)
    ctx = _Ctx()
    operator.on_start(ctx)
    books = [m.book for m in operator.live.values()]

    ctx.now = int(seconds(120))
    operator.on_wakeup(ctx)

    settled = set(market.venue.settled_symbols)
    for book in books:
        for contract in book.contracts:
            assert contract.symbol in settled, contract.symbol


def test_settlement_agrees_with_the_match_that_happened():
    """Both paths settle from the same result, so they cannot disagree."""
    market, operator = _operator(formats=("solo",), concurrent=1)
    ctx = _Ctx()
    operator.on_start(ctx)
    match = next(iter(operator.live.values()))
    ctx.now = int(seconds(120))
    operator.on_wakeup(ctx)

    truth = settlement(match.book, play(7, match.match_id, "solo"))
    winners = [c for c in match.book.contracts if c.family == "winner"]
    assert sum(truth[c.symbol] for c in winners) == pytest.approx(1.0)
    assert sum(1 for c in winners if truth[c.symbol] == 1.0) == 1


def test_a_team_match_eliminates_nobody_and_settles_at_the_end():
    """The mode has no eliminations, so there is nothing to reveal early."""
    _market, operator = _operator(formats=("objective",), concurrent=1)
    ctx = _Ctx()
    operator.on_start(ctx)
    match = next(iter(operator.live.values()))
    assert match.knockout_order == ()
    ctx.now = int(seconds(60))
    operator.on_wakeup(ctx)
    assert match.revealed == 0


# --------------------------------------------------------------------------
# The invariants the venue already holds
# --------------------------------------------------------------------------


def test_running_matches_conserves_value_exactly():
    """Settling a hundred contracts early must not create or destroy a unit."""
    market, operator = _operator(formats=("solo", "objective"), concurrent=2)
    ctx = _Ctx()
    operator.on_start(ctx)
    for step in range(1, 8):
        ctx.now = int(seconds(step * 45))
        operator.on_wakeup(ctx)
        assert market.venue.conservation_check() == 0


def test_an_unregistered_format_is_refused_at_construction():
    """Rather than opening an empty board and reporting success."""
    market = build(seed=7)
    with pytest.raises(ValueError, match="format"):
        MatchOperator("matches", market.venue, seed=7, formats=("teleport",))


# --------------------------------------------------------------------------
# Joining a market that changes shape underneath you
# --------------------------------------------------------------------------


class _Watcher:
    """The two notes an operator leaves, recorded rather than acted on."""

    def __init__(self):
        self.joined, self.left, self.relations = [], [], []

    def note_listing(self, instrument):
        self.joined.append(instrument.symbol)

    def note_delisting(self, symbol):
        self.left.append(symbol)

    def add_relations(self, relations):
        self.relations.extend(relations)
        return len(relations)

    def drop_relations_for(self, symbols):
        before = len(self.relations)
        self.relations = [
            r
            for r in self.relations
            if not ({r.target, *(s for s, _ in r.legs)} & symbols)
        ]
        return before - len(self.relations)


def test_a_match_tells_the_market_it_exists():
    """A contract nobody was told about is a book nobody quotes.

    Everything an agent knows about a symbol is built once, at start, so a
    contract listed later is invisible to every agent already trading: no local
    book, no subscription, no quote. That was fine while the listing was fixed
    for a session and is not once matches open and close continuously.
    """
    market = build(seed=7)
    watcher = _Watcher()
    operator = MatchOperator(
        "matches", market.venue, seed=7, formats=("solo",), concurrent=1,
        participants=[watcher],
    )
    ctx = _Ctx()
    operator.on_start(ctx)

    match = next(iter(operator.live.values()))
    assert len(watcher.joined) == len(match.book.contracts)
    assert watcher.relations, "a match listed without handing over its identities"


def test_the_identities_arrive_with_the_match_and_leave_with_it():
    """47 of them for a solo match, in the arbitrageur's own vocabulary.

    So enforcing them needs no new execution path: an agent already built on
    these relations simply has a longer list. And they go when the match does,
    because a relation whose legs have settled can never be traded again and
    would price against a mark that no longer moves.

    It was 395 when a solo match listed 270 contracts. What the trim took out
    was mostly a ladder comparing each rung to the rung beside it, so the count
    fell further than the listing did; every kind of identity the operator hands
    over still arrives.
    """
    market = build(seed=7)
    watcher = _Watcher()
    operator = MatchOperator(
        "matches", market.venue, seed=7, formats=("solo",), concurrent=1,
        participants=[watcher],
    )
    ctx = _Ctx()
    operator.on_start(ctx)
    during = len(watcher.relations)
    assert during == 47

    ctx.now = int(seconds(200))
    operator.on_wakeup(ctx)
    settled_symbols = set(watcher.left)
    assert settled_symbols, "nothing was delisted"
    assert not any(
        {r.target, *(s for s, _ in r.legs)} & settled_symbols
        for r in watcher.relations
    ), "a relation outlived the contracts it reads"


def test_an_arbitrageur_takes_them_on_and_deduplicates():
    """Told twice about one match, it must not size every package twice."""
    from arena.agents.arbitrageur import Arbitrageur
    from arena.market.match_book import derive_match_relations, list_match

    listed = {i.symbol: i for i in _instruments()}
    arb = Arbitrageur("arb", "venue", listed)
    base = len(arb.relations)

    book = list_match(7, 0, "solo")
    relations = derive_match_relations(book)
    assert arb.add_relations(relations) == len(relations)
    assert arb.add_relations(relations) == 0
    assert len(arb.relations) == base + len(relations)

    dropped = arb.drop_relations_for({c.instrument.symbol for c in book.contracts})
    assert dropped == len(relations)
    assert len(arb.relations) == base


def _instruments():
    from dashboard.build_market import instruments

    return instruments()
