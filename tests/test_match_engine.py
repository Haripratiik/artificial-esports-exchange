"""Matches that settle on arithmetic rather than on atmosphere.

The exchange needs events that resolve on their own and produce numbers nobody
has to invent. A match does, and the reason its settlement can be trusted is
the same reason the ledger's can: the result has to add up, and a result that
does not is refused rather than rounded.
"""

from __future__ import annotations

from collections import Counter

import pytest

from arena.worlds.circuit.match import play, season
from arena.worlds.circuit.modes import (
    FORMATS,
    SoloElimination,
    TeamObjective,
    register_format,
)
from arena.worlds.circuit.roster import ROSTER, draw_strengths, roster_for


# --------------------------------------------------------------------------
# The conservation law of a match
# --------------------------------------------------------------------------


@pytest.mark.parametrize("format_name", sorted(FORMATS))
def test_every_match_is_a_legal_match(format_name):
    """Four thousand of them, each checked against its own format.

    The check is not a test fixture, it runs inside `play` on every match ever
    produced. A malformed result must not reach settlement, because a contract
    settled against one pays out on arithmetic that does not close, and nothing
    downstream can detect that afterwards.
    """
    runs = season(seed=7, matches=4000, format_name=format_name)
    assert len(runs) == 4000
    for result in runs:
        FORMATS[format_name].check(result.placements, result.eliminations)


def test_a_solo_match_credits_exactly_one_elimination_per_loser():
    """Nine, always, in a field of ten. Measured across 4,000 matches.

    This is the match's version of conservation. Ten enter, one survives, so
    nine eliminations happened and the credit for each has to go somewhere. A
    match crediting eight has lost one and a match crediting ten has invented
    one, and an over-under contract on a competitor's eliminations would settle
    against either without complaint.
    """
    totals = {sum(r.eliminations.values()) for r in season(7, 4000, "solo")}
    assert totals == {9}


def test_exactly_one_side_wins():
    """Which is what makes the winner contracts a partition rather than a list.

    A set of "X wins" contracts is mutually exclusive and exhaustive only if
    the world guarantees one winner. That guarantee is here, and the market
    layer is entitled to rely on it: their prices must sum to one because the
    outcomes do.
    """
    for format_name, expected in (("solo", 1), ("objective", 3)):
        for result in season(7, 500, format_name):
            assert len(result.winners) == expected


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_a_match_is_a_pure_function_of_its_seed():
    """Or a replay is a different season and every paired trial is unpaired."""
    for format_name in sorted(FORMATS):
        first = play(7, 41, format_name)
        again = play(7, 41, format_name)
        assert first.placements == again.placements
        assert first.eliminations == again.eliminations
        assert first.field == again.field


def test_a_different_seed_is_a_different_season():
    """The seed has to reach the outcome, not just the field.

    A world that drew a different field and then played it identically would
    look seeded and would not be: every contract on a competitor would settle
    the same way regardless of the seed, and a paired experiment across seeds
    would be comparing one season to itself.
    """
    a = [play(7, n, "solo").winners for n in range(200)]
    b = [play(11, n, "solo").winners for n in range(200)]
    assert a != b


# --------------------------------------------------------------------------
# A field worth pricing
# --------------------------------------------------------------------------


def test_the_field_is_uneven_but_nobody_is_hopeless():
    """Both halves matter, and each kills a different kind of market.

    A field with no spread makes every winner contract the same contract at
    1/n. A field with too much makes the favourite's contract a constant and
    the rest worthless. Measured over 4,000 solo matches on seed 7: win rate
    per appearance runs 0.036 to 0.178 against an even-field 0.100, a spread of
    4.9x, and every competitor wins sometimes.
    """
    runs = season(7, 4000, "solo")
    wins = Counter(k for r in runs for k in r.winners)
    appearances = Counter(k for r in runs for k in r.field)
    rates = {k: wins[k] / appearances[k] for k in appearances}

    assert len(rates) == len(ROSTER)
    assert min(rates.values()) > 0.0, "a competitor who never wins is a constant"
    assert 3.0 < max(rates.values()) / min(rates.values()) < 12.0


def test_the_two_formats_rank_the_field_differently():
    """Or the second family of contracts restates the first.

    That was the flaw in the previous listing: forty-seven instruments written
    on four numbers, so a trader who priced one had priced most of them.
    Measured over 4,000 matches per format on seed 7, the Spearman rank
    correlation between win rate in the solo mode and in the team mode is
    **-0.357**. Halcyon is last in one and first in the other.
    """

    def rates(format_name):
        runs = season(7, 4000, format_name)
        wins = Counter(k for r in runs for k in r.winners)
        seen = Counter(k for r in runs for k in r.field)
        return {k: wins[k] / seen[k] for k in seen}

    solo, objective = rates("solo"), rates("objective")
    keys = sorted(set(solo) & set(objective))
    by_solo = sorted(keys, key=lambda k: -solo[k])
    by_objective = sorted(keys, key=lambda k: -objective[k])
    n = len(keys)
    gap = sum((by_solo.index(k) - by_objective.index(k)) ** 2 for k in keys)
    rho = 1 - 6 * gap / (n * (n * n - 1))
    assert rho < 0.3, f"the two formats rank the field alike (rho={rho:+.3f})"


# --------------------------------------------------------------------------
# What the market may not see
# --------------------------------------------------------------------------


def test_a_result_carries_no_trace_of_latent_strength():
    """The parameter generates outcomes and must never be readable from one.

    If a price could be derived from the generator rather than from the record,
    every forecasting result measured on this world would be a result about
    arithmetic. The separation is structural: `MatchResult` has no field for it.
    """
    result = play(7, 0, "solo")
    for forbidden in ("strength", "strengths", "weights", "skill"):
        assert not hasattr(result, forbidden), forbidden
    assert set(draw_strengths(7, "solo")) == {c.key for c in ROSTER}


# --------------------------------------------------------------------------
# Adding a format
# --------------------------------------------------------------------------


def test_a_new_format_needs_no_change_anywhere_else():
    """The registry is the extension point, and this proves it from outside.

    Registered here rather than in the source, so the test fails if adding a
    mode ever starts requiring an edit to the engine.
    """
    added = SoloElimination(name="duel", entrants=2)
    try:
        register_format(added)
        result = play(7, 0, "duel")
        assert len(result.field) == 2
        assert sum(result.eliminations.values()) == 1
        assert len(result.winners) == 1
    finally:
        FORMATS.pop("duel", None)


def test_a_format_cannot_be_replaced_silently():
    """Two worlds sharing a name would settle each other's contracts."""
    with pytest.raises(ValueError, match="already registered"):
        register_format(SoloElimination())


def test_a_field_larger_than_the_roster_is_refused():
    """Rather than quietly running a smaller match than the format promised."""
    with pytest.raises(ValueError, match="roster"):
        roster_for("solo", len(ROSTER) + 1, seed=7, match_id=0)


@pytest.mark.parametrize(
    "placements,eliminations,reason",
    [
        ({f"P{i}": i + 1 for i in range(10)}, {"P0": 8}, "eliminations"),
        ({f"P{i}": 1 for i in range(10)}, {"P0": 9}, "permutation"),
        ({f"P{i}": i + 1 for i in range(9)}, {"P0": 9}, "placements"),
    ],
)
def test_an_impossible_result_is_refused(placements, eliminations, reason):
    """Each of these would settle a contract against something that did not
    happen, and none of them is detectable downstream."""
    with pytest.raises(ValueError):
        SoloElimination().check(placements, eliminations)


def test_a_team_result_is_two_placements_and_not_a_ranking():
    """A team mode does not rank its members, and inventing an order would
    invent a distinction the format does not have."""
    with pytest.raises(ValueError):
        TeamObjective().check({f"P{i}": i + 1 for i in range(6)}, {"P0": 3})
