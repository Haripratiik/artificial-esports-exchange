"""Point the joint no-arbitrage test at a live match on this venue.

Three boards are tested, chosen because each has a state space small enough to
enumerate exactly and because together they cover every family the venue lists
on a match.

**The winner set.** States are the sides that can win, one each. Every
``WIN`` leg covers exactly one of them, and the venue publishes the partition
itself, so this is the classic mutually-exclusive-and-exhaustive test done as
an LP over the touch rather than as a sum of mids.

**The eliminations ladder, per competitor.** States are the elimination counts
that competitor can finish on. ``ELIM_k_GT{t}`` covers every state above ``t``,
so the whole ladder is tested at once: monotonicity, the gap between adjacent
rungs, and the level, in one question instead of three.

**The top-N ladder, per competitor.** States are the places that competitor can
finish in. ``TOP{n}_k`` covers places one to ``n``.

What is deliberately NOT attempted is one LP over every family at once. The
joint state of a ten-entrant match is a finishing order, and there are
3,628,800 of them; enumerating that to test 270 contracts would be a worse
answer than three exact ones, and a sampled version would report violations it
could not distinguish from its own sampling. The engine this came from has the
same boundary and draws it in the same place, capping its score grid and
putting the residue in a free state.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

from connectors.prediction_engine.client import ArenaClient, family_of, match_of
from connectors.prediction_engine.jointarb import Leg, check
from connectors.prediction_engine.models import CENTS, Market

__all__ = ["winner_board", "elimination_boards", "top_boards", "audit"]

_ELIM = re.compile(r"^(?P<tag>[A-Z]+\d+)_ELIM_(?P<who>.+)_GT(?P<threshold>\d+)$")
_TOP = re.compile(r"^(?P<tag>[A-Z]+\d+)_TOP(?P<rung>\d+)_(?P<who>.+)$")
_WIN = re.compile(r"^(?P<tag>[A-Z]+\d+)_WIN_(?P<who>.+)$")


# Only a book that is matching right now carries a price anybody could trade
# on. This filter is not fastidiousness, it is the difference between a result
# and an artefact, and leaving it out produced both failures worth recording.
#
# A contract on a settled match keeps its last touch forever, so a closed book
# reports whatever it happened to be quoted at when the match ended. And this
# venue opens every contract at the midpoint of its own settlement range
# rather than at an estimate, deliberately, so that convergence has to be
# produced by agents trading against the maker. Ten seconds after listing, all
# ten legs of a winner partition are therefore quoted near 0.50 and sum to
# 4.54 against a partition that must sum to 1. That reads as a 3.5 unit
# arbitrage and is nothing of the kind: it is a market that has not opened yet,
# measured as though it had.
_TRADEABLE = "continuous"


def _probability(market: Market) -> tuple[float, float] | None:
    """The touch as a pair of probabilities, or None if it cannot be traded.

    A book quoted one side only says nothing about where the market thinks the
    price is, and feeding it a half-spread in place of the missing side would
    manufacture the very consistency this is trying to test.
    """
    if market.status != _TRADEABLE or not market.is_quoted_both_sides:
        return None
    return market.yes_bid / CENTS, market.yes_ask / CENTS


def winner_board(markets: list[Market], tag: str) -> tuple[list[Leg], list[str]]:
    """Legs and states for one match's winner set."""
    legs, states = [], []
    for market in markets:
        found = _WIN.match(market.ticker)
        if found is None or found.group("tag") != tag:
            continue
        who = found.group("who")
        states.append(who)
        touch = _probability(market)
        if touch is not None:
            legs.append(Leg(market.ticker, touch[0], touch[1],
                            (lambda w: lambda state: state == w)(who)))
    return legs, states


def elimination_boards(markets: list[Market], tag: str) -> dict[str, tuple[list[Leg], list[int]]]:
    """One board per competitor: the whole over-under ladder on its count."""
    rungs: dict[str, list[tuple[int, Market]]] = defaultdict(list)
    for market in markets:
        found = _ELIM.match(market.ticker)
        if found is not None and found.group("tag") == tag:
            rungs[found.group("who")].append((int(found.group("threshold")), market))

    boards = {}
    for who, entries in rungs.items():
        highest = max(threshold for threshold, _ in entries)
        # One state above the highest rung, because "more than the top rung"
        # is a real outcome and leaving it out would force its probability to
        # zero, which is a constraint the venue never made.
        states = list(range(highest + 2))
        legs = []
        for threshold, market in sorted(entries):
            touch = _probability(market)
            if touch is not None:
                legs.append(Leg(market.ticker, touch[0], touch[1],
                                (lambda t: lambda state: state > t)(threshold)))
        boards[who] = (legs, states)
    return boards


def top_boards(markets: list[Market], tag: str, field_size: int) -> dict[str, tuple[list[Leg], list[int]]]:
    """One board per competitor: the top-N ladder on its finishing place."""
    rungs: dict[str, list[tuple[int, Market]]] = defaultdict(list)
    for market in markets:
        found = _TOP.match(market.ticker)
        if found is not None and found.group("tag") == tag:
            rungs[found.group("who")].append((int(found.group("rung")), market))

    boards = {}
    for who, entries in rungs.items():
        states = list(range(1, max(field_size, max(r for r, _ in entries)) + 1))
        legs = []
        for rung, market in sorted(entries):
            touch = _probability(market)
            if touch is not None:
                legs.append(Leg(market.ticker, touch[0], touch[1],
                                (lambda n: lambda place: place <= n)(rung)))
        boards[who] = (legs, states)
    return boards


def audit(base_url: str = "http://localhost:8000", *, tol: float = 0.0) -> int:
    """Run every board on every live match and print what the LP found."""
    with ArenaClient(base_url) as client:
        events, markets = client.fetch_universe()

    by_match: dict[str, list[Market]] = defaultdict(list)
    for market in markets:
        by_match[market.event_ticker].append(market)

    violations = 0
    checked = 0
    for event in events:
        tag = event.event_ticker
        here = by_match[tag]
        legs, states = winner_board(here, tag)
        field_size = len(states)
        boards = [(f"{tag} winner", legs, states)]
        for who, (elim_legs, elim_states) in sorted(elimination_boards(here, tag).items()):
            boards.append((f"{tag} eliminations {who}", elim_legs, elim_states))
        for who, (top_legs, top_states) in sorted(top_boards(here, tag, field_size).items()):
            boards.append((f"{tag} top-N {who}", top_legs, top_states))

        print(f"\n=== {tag}: {len(here)} markets, field of {field_size} ===")
        for name, board_legs, board_states in boards:
            result = check(name, board_legs, board_states, tol=tol)
            if result.n_legs < 3:
                continue
            checked += 1
            if result.arbitrage:
                violations += 1
                print(f"  ARBITRAGE  {name:38s} {result.n_legs:>3} legs  {result.detail}")
                # The legs, because a slack without them is not actionable. A
                # reader has to be able to see which quote is the odd one and
                # decide whether it is a dislocation or a bug in this file.
                for leg in board_legs:
                    print(f"      {leg.ticker:44s} {leg.bid:.2f} / {leg.ask:.2f}")
            else:
                print(f"  consistent {name:38s} {result.n_legs:>3} legs")

    print(f"\n{checked} boards tested, {violations} admitting no consistent distribution")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--tol", type=float, default=0.0,
        help="widen every spread by this many probability units before testing",
    )
    args = parser.parse_args()
    return 0 if audit(args.url, tol=args.tol) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
