"""Who competes, and the one number that decides how often they win.

Every competitor here is invented. The names, the archetypes and the numbers
are ours, and none of them refers to anything that exists.

**Skill is latent and is never published.** A competitor carries a strength per
format, and nothing outside this module may read it: the oracle reports what
matches *did*, not what the roster *is*. That separation is the whole reason
the exchange is a forecasting problem at all. A market that could see the
parameter would be pricing a lookup, and every result measured on it would be a
result about arithmetic rather than about aggregation.

Strengths are drawn once per world from the world seed, so a given seed always
produces the same field, and two runs that differ only in the market are
comparing the same competitors.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

__all__ = ["Competitor", "ROSTER", "roster_for", "draw_strengths"]


@dataclass(frozen=True)
class Competitor:
    """One entrant, and what kind of thing it is good at.

    ``archetype`` exists so the field is not interchangeable. A roster of
    identical competitors makes every match a coin flip and every contract on
    it the same contract, which is the shallowness the previous listing had:
    forty-seven instruments that were four numbers wearing hats.
    """

    key: str
    name: str
    archetype: str


# Deliberately varied, because a market on an even field prices nothing. The
# archetypes differ in which format suits them, so a competitor that dominates
# the solo mode is not automatically the one to back in the team mode, and the
# two families of contracts stay genuinely different questions.
ROSTER: tuple[Competitor, ...] = (
    Competitor("VANTA", "Vanta", "duellist"),
    Competitor("QUILL", "Quill", "marksman"),
    Competitor("BASTION", "Bastion", "bulwark"),
    Competitor("EMBER", "Ember", "skirmisher"),
    Competitor("HALCYON", "Halcyon", "support"),
    Competitor("RIFT", "Rift", "controller"),
    Competitor("TALON", "Talon", "duellist"),
    Competitor("MIRE", "Mire", "controller"),
    Competitor("CINDER", "Cinder", "skirmisher"),
    Competitor("OBELISK", "Obelisk", "bulwark"),
    Competitor("WISP", "Wisp", "support"),
    Competitor("KESTREL", "Kestrel", "marksman"),
)

# How much an archetype is helped or hindered by a format.
#
# The point is that the two modes rank the field differently. A bulwark that
# survives a long solo elimination is not the same asset as a support that
# swings a three-a-side objective, so a trader who has learned one mode has
# only partly learned the other, and the second family of contracts carries
# information the first does not.
_ARCHETYPE_FIT: dict[str, dict[str, float]] = {
    "duellist": {"solo": 0.30, "objective": 0.05},
    "marksman": {"solo": 0.15, "objective": 0.15},
    "bulwark": {"solo": 0.25, "objective": -0.05},
    "skirmisher": {"solo": 0.10, "objective": 0.10},
    "support": {"solo": -0.25, "objective": 0.35},
    "controller": {"solo": -0.10, "objective": 0.25},
}


def draw_strengths(seed: int, format_name: str) -> dict[str, float]:
    """Latent strength per competitor for one format, fixed by the world seed.

    Log-scale, so the numbers combine multiplicatively when a match weighs one
    competitor against another and no draw can produce a negative weight. The
    spread is modest on purpose: a field where one entrant wins almost always
    makes its own winner market worthless, and a field with no spread makes
    every winner market identical. Measured over the default roster, this
    leaves the strongest solo competitor winning roughly three times as often
    as the weakest, which keeps every contract in the set worth pricing.
    """
    rng = random.Random(f"{seed}:strength:{format_name}")
    out: dict[str, float] = {}
    for competitor in ROSTER:
        fit = _ARCHETYPE_FIT.get(competitor.archetype, {}).get(format_name, 0.0)
        out[competitor.key] = rng.gauss(0.0, 0.35) + fit
    return out


def roster_for(format_name: str, entrants: int, seed: int, match_id: int) -> list[str]:
    """Which competitors are in one match, drawn without replacement.

    A different field each match, so a competitor's record accumulates over
    matches it was actually in. Sampling the field from the roster rather than
    running the same twelve every time is what gives a statistic like "win rate
    over the window" something to average over, and it is why two contracts on
    the same competitor in different weeks are not the same contract.
    """
    if entrants > len(ROSTER):
        raise ValueError(
            f"{format_name} wants {entrants} entrants but the roster holds "
            f"{len(ROSTER)}"
        )
    rng = random.Random(f"{seed}:field:{format_name}:{match_id}")
    return rng.sample([c.key for c in ROSTER], entrants)
