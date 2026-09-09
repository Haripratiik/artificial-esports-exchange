"""Playing one match out, deterministically, and refusing to return a bad one.

A match is drawn from the world seed and its own number, so the same seed gives
the same season and a replay reproduces every result exactly. Nothing here
reads a clock, and nothing carries state between matches: a match is a pure
function of the field, the format and the seed.

**Every result is checked before it is returned.** The format states what a
result must satisfy, and a draw that violates it raises rather than settling.
That is not defensive coding, it is the same rule the ledger holds: a match
whose eliminations do not sum to the number of entrants that lost is not a
match with a small error in it, it is a match that did not happen, and a
contract settled against one would be paying out on arithmetic that does not
close.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from arena.worlds.circuit.modes import FORMATS, MatchFormat
from arena.worlds.circuit.roster import draw_strengths, roster_for

__all__ = ["MatchResult", "play", "season"]


@dataclass(frozen=True)
class MatchResult:
    """What happened, and nothing about why.

    This is the whole of what the oracle may see. Strengths are not here, and
    that is deliberate: settlement resolves against what the match produced,
    so a contract can never be priced off the parameter that generated it.
    """

    match_id: int
    format_name: str
    field: tuple[str, ...]
    # 1 is the winner. In a team format every member of the winning side is 1.
    placements: dict[str, int]
    eliminations: dict[str, int]
    # Only meaningful where the format keeps score. Left empty otherwise rather
    # than filled with a zero that reads as a real result.
    scores: dict[str, int] = field(default_factory=dict)

    @property
    def winners(self) -> tuple[str, ...]:
        return tuple(sorted(k for k, v in self.placements.items() if v == 1))

    def placed_within(self, competitor: str, top: int) -> bool:
        return self.placements.get(competitor, 10**9) <= top


def _weights(field_: list[str], strengths: dict[str, float]) -> list[float]:
    """Softmax over latent strength, which keeps every entrant possible.

    Exponential rather than proportional so a strength difference has a
    consistent meaning wherever it sits on the scale, and so no weight can be
    zero. An entrant who cannot possibly win makes its own contract a constant,
    and a market that lists a constant is wasting a book on it.
    """
    return [math.exp(strengths.get(key, 0.0)) for key in field_]


def _draw_order(
    field_: list[str], strengths: dict[str, float], rng: random.Random
) -> list[str]:
    """Finishing order, strongest-biased, sampled without replacement.

    A weighted sample without replacement rather than a sort by strength plus
    noise. The distinction matters: sorting produces a field whose ordering is
    almost fixed, so the winner market converges to a certainty and stops being
    a forecast. Sampling keeps the tail live, which is what an upset is.
    """
    remaining = list(field_)
    weights = {k: w for k, w in zip(field_, _weights(field_, strengths))}
    order: list[str] = []
    while remaining:
        total = sum(weights[k] for k in remaining)
        cut = rng.random() * total
        running = 0.0
        for key in remaining:
            running += weights[key]
            if running >= cut:
                order.append(key)
                remaining.remove(key)
                break
        else:  # pragma: no cover - float drift only
            order.append(remaining.pop())
    return order


def play(seed: int, match_id: int, format_name: str = "solo") -> MatchResult:
    """One match. Deterministic in ``(seed, match_id, format_name)``."""
    fmt: MatchFormat = FORMATS[format_name]
    strengths = draw_strengths(seed, format_name)
    field_ = roster_for(format_name, fmt.entrants, seed, match_id)
    rng = random.Random(f"{seed}:match:{format_name}:{match_id}")

    if fmt.team_size == 1:
        order = _draw_order(field_, strengths, rng)
        placements = {key: place for place, key in enumerate(order, start=1)}
        # Every entrant but the survivor is eliminated exactly once, and the
        # credit goes to somebody still alive when it happened. Walking the
        # order from the back gives exactly `entrants - 1` credits by
        # construction rather than by a correction afterwards.
        eliminations = {key: 0 for key in field_}
        for position in range(len(order) - 1, 0, -1):
            alive = order[:position]
            weights = _weights(alive, strengths)
            eliminations[rng.choices(alive, weights=weights, k=1)[0]] += 1
        result = MatchResult(
            match_id, format_name, tuple(field_), placements, eliminations
        )
    else:
        sides = [
            field_[i : i + fmt.team_size]
            for i in range(0, fmt.entrants, fmt.team_size)
        ]
        power = [sum(math.exp(strengths.get(k, 0.0)) for k in side) for side in sides]
        target = getattr(fmt, "target", 3)
        scores = [0] * len(sides)
        eliminations = {key: 0 for key in field_}
        # A race to the target, one point at a time, each point going to a side
        # with probability proportional to its strength. The margin therefore
        # carries information the winner alone does not, which is what makes a
        # score contract on this mode a different bet from the winner contract.
        while max(scores) < target:
            total = sum(power)
            cut = rng.random() * total
            running = 0.0
            for index, weight in enumerate(power):
                running += weight
                if running >= cut:
                    scores[index] += 1
                    side = sides[index]
                    eliminations[
                        rng.choices(
                            side,
                            weights=[math.exp(strengths.get(k, 0.0)) for k in side],
                            k=1,
                        )[0]
                    ] += 1
                    break
        won = scores.index(max(scores))
        placements = {
            key: (1 if index == won else 2)
            for index, side in enumerate(sides)
            for key in side
        }
        result = MatchResult(
            match_id,
            format_name,
            tuple(field_),
            placements,
            eliminations,
            {key: scores[i] for i, side in enumerate(sides) for key in side},
        )

    # The format decides what a result has to satisfy, and it is asked every
    # time rather than in a test. A malformed match must not reach settlement.
    fmt.check(result.placements, result.eliminations)
    return result


def season(seed: int, matches: int, format_name: str = "solo") -> list[MatchResult]:
    """A run of matches, which is what a statistic is averaged over."""
    return [play(seed, n, format_name) for n in range(matches)]
