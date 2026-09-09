"""What a match is, and how a new format gets added without touching anything.

The exchange needs events that resolve on their own, close together, and
produce numbers nobody has to invent. A match does all three: it starts, it is
watched, it ends with an outcome that is a fact rather than an estimate, and it
does so in minutes rather than in a four week observation window.

**This is a synthetic sport.** It is deliberately faithful to the shape of the
arena-brawler genre, because that shape is what makes the markets interesting:
a solo mode with one survivor produces a mutually exclusive winner set, and a
team mode produces a binary with a score attached. Formats and mechanics are
ideas and are free to model. Names are not, so every competitor, mode and
statistic here is our own.

**The settlement realism is arithmetic, not atmosphere.** In a ten-competitor
elimination there is exactly one survivor and exactly nine eliminations, so
"who won" and "how many did X get" resolve against numbers that must add up.
That is the same discipline the ledger already holds itself to, applied to the
world the contracts are written on: a match whose eliminations do not sum to
``entrants - 1`` is refused rather than settled, in the way a venue whose cash
does not conserve is a bug rather than a rounding difference.

Adding a format is one dataclass and one registry entry. Nothing in the venue,
the contracts or the agents knows how many modes exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = [
    "MatchFormat",
    "SoloElimination",
    "TeamObjective",
    "FORMATS",
    "register_format",
]


@runtime_checkable
class MatchFormat(Protocol):
    """One way of playing. Everything the market needs to list a match.

    Deliberately narrow. A format says how many compete, how they are grouped,
    and what a result has to satisfy to be a result. It does not say how the
    match is simulated, because that is `match.py`'s job, and it does not know
    what contracts exist, because that is the listing's.
    """

    name: str
    entrants: int
    team_size: int

    @property
    def teams(self) -> int:
        """How many sides contest one match."""
        ...

    def check(self, placements: dict[str, int], eliminations: dict[str, int]) -> None:
        """Refuse a result that cannot have happened. Raises on a bad one."""
        ...


@dataclass(frozen=True)
class SoloElimination:
    """Last one standing, everybody for themselves.

    The interesting one for prediction markets, because its outcome space is a
    partition: exactly one competitor survives, so the set of "X wins" contracts
    is mutually exclusive and exhaustive and its prices must sum to one. That is
    not a rule imposed on the market, it is a fact about the world, which is
    what makes enforcing it honest rather than cosmetic.
    """

    name: str = "solo"
    entrants: int = 10
    team_size: int = 1

    @property
    def teams(self) -> int:
        return self.entrants

    def check(self, placements: dict[str, int], eliminations: dict[str, int]) -> None:
        if len(placements) != self.entrants:
            raise ValueError(
                f"{self.name}: {len(placements)} placements for {self.entrants} entrants"
            )
        # A permutation of 1..n, so exactly one first place and no ties. Ties
        # would make "who won" ambiguous, and a market on an ambiguous question
        # cannot be settled by arithmetic.
        if sorted(placements.values()) != list(range(1, self.entrants + 1)):
            raise ValueError(f"{self.name}: placements are not a permutation")
        # Every entrant but the survivor was eliminated exactly once, so the
        # eliminations credited must add to one fewer than the field. This is
        # the conservation law of a match and it is checked, not assumed.
        total = sum(eliminations.values())
        if total != self.entrants - 1:
            raise ValueError(
                f"{self.name}: {total} eliminations credited, expected "
                f"{self.entrants - 1}"
            )
        if any(count < 0 for count in eliminations.values()):
            raise ValueError(f"{self.name}: an entrant was credited a negative count")


@dataclass(frozen=True)
class TeamObjective:
    """Two sides contest a target score. One of them gets there first.

    A different statistical object from the solo mode rather than a reskin of
    it: the outcome is binary, the margin is bounded and meaningful, and a
    competitor's contribution is separable from whether their side won. So it
    supports contracts the solo mode cannot, and the two worlds do not simply
    restate each other, which is the failure the previous listing had.
    """

    name: str = "objective"
    entrants: int = 6
    team_size: int = 3
    target: int = 3

    @property
    def teams(self) -> int:
        return self.entrants // self.team_size

    def check(self, placements: dict[str, int], eliminations: dict[str, int]) -> None:
        if len(placements) != self.entrants:
            raise ValueError(
                f"{self.name}: {len(placements)} placements for {self.entrants} entrants"
            )
        # Everyone on the winning side places first, everyone on the losing side
        # second. A team result is not a ranking of individuals and pretending
        # otherwise would invent a distinction the mode does not have.
        seen = sorted(set(placements.values()))
        if seen != [1, 2]:
            raise ValueError(f"{self.name}: expected two placements, got {seen}")
        for place in (1, 2):
            count = sum(1 for p in placements.values() if p == place)
            if count != self.team_size:
                raise ValueError(
                    f"{self.name}: {count} competitors placed {place}, "
                    f"expected {self.team_size}"
                )
        if any(count < 0 for count in eliminations.values()):
            raise ValueError(f"{self.name}: an entrant was credited a negative count")


# The registry a listing dispatches on. A format outside this mapping is
# refused when a match is drawn rather than producing a match nobody can price.
#
# Adding a mode is an entry here and a dataclass above. The venue, the
# contracts, the agents and the front end all read the registry, so none of
# them carries a list of modes that can fall out of date.
FORMATS: dict[str, MatchFormat] = {
    "solo": SoloElimination(),
    "objective": TeamObjective(),
}


def register_format(format_: MatchFormat) -> None:
    """Add a format at runtime. Refuses to replace one silently.

    Silent replacement is how two different worlds end up sharing a name and
    settling each other's contracts, which is unrecoverable after the fact
    because the match record says only which name it used.
    """
    if format_.name in FORMATS:
        raise ValueError(f"a format named {format_.name!r} is already registered")
    FORMATS[format_.name] = format_
