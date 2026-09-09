"""The circuit world's statistics layer: what a window means and what it may see.

Two claims are load bearing here and everything else supports them.

The first is that an aggregate is a *measurement*. A metric that returns a
well formed zero for a competitor nobody entered, or that silently shortens a
window it cannot cover, or that hands back a number outside the range its
contract sized collateral against, is worse than one that refuses: the refusal
voids and everyone walks away whole, while the plausible number settles and
nothing downstream can tell. So the tests below spend most of their effort on
what the layer declines to answer.

The second is that the oracle cannot see the parameter that generated the
outcomes. ``roster.draw_strengths`` decides who wins; the oracle reports who
did. If those two were ever joined, every forecasting result measured on this
world would be a result about arithmetic rather than about aggregation, and the
separation would be an intention rather than a fact. It is tested here as a
fact, from the outside, three different ways.
"""

from __future__ import annotations

import inspect
import math
import statistics
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from arena.contracts.payoff import Linear
from arena.contracts.spec import ContractSpec, DataPolicy, ObservationWindow
from arena.contracts.underlying import ALL, MetricRef, Single
from arena.settlement.engine import settle
from arena.settlement.oracle import MetricUnavailable
from arena.settlement.result import SettlementStatus
from arena.worlds.circuit import metrics as circuit_metrics
from arena.worlds.circuit import oracle as circuit_oracle
from arena.worlds.circuit.match import MatchResult, play, season
from arena.worlds.circuit.metrics import (
    MAX_WINDOW_MATCHES,
    METRIC_BOUNDS,
    METRIC_KINDS,
    METRICS,
    InsufficientEvidence,
    match_volume,
    metric_ref,
    placement_dispersion,
    win_rate,
)
from arena.worlds.circuit.modes import FORMATS
from arena.worlds.circuit.oracle import CALENDAR, CircuitOracle, MatchCalendar
from arena.worlds.circuit.roster import ROSTER

UTC = timezone.utc
SEED = 7
KEYS = tuple(competitor.key for competitor in ROSTER)

# The first full week of the season, opening on the Monday after the epoch.
# 2,016 matches at the schedule's one every five minutes.
WEEK = ObservationWindow(
    start=datetime(2026, 9, 7, tzinfo=UTC), end=datetime(2026, 9, 14, tzinfo=UTC)
)
NEXT_WEEK = ObservationWindow(
    start=datetime(2026, 9, 14, tzinfo=UTC), end=datetime(2026, 9, 21, tzinfo=UTC)
)
FORTNIGHT = ObservationWindow(start=WEEK.start, end=NEXT_WEEK.end)
# The id the week opens on, used where a test needs a run of matches shorter
# than any sensible window.
WEEK_FIRST_MATCH = 1_728


@pytest.fixture(scope="module")
def oracle():
    return CircuitOracle(SEED)


@pytest.fixture(scope="module")
def long_seasons():
    """4,000 matches per format on seed 7, which is what every number here is
    measured over."""
    return {name: season(SEED, 4000, name) for name in sorted(FORMATS)}


def spearman(left: dict[str, float], right: dict[str, float]) -> float:
    keys = sorted(set(left) & set(right))
    size = len(keys)
    left_rank = {key: i for i, key in enumerate(sorted(keys, key=lambda k: left[k]))}
    right_rank = {key: i for i, key in enumerate(sorted(keys, key=lambda k: right[k]))}
    gap = sum((left_rank[key] - right_rank[key]) ** 2 for key in keys)
    return 1 - 6 * gap / (size * (size * size - 1))


def pearson(left: dict[str, float], right: dict[str, float]) -> float:
    keys = sorted(set(left) & set(right))
    xs = [left[key] for key in keys]
    ys = [right[key] for key in keys]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return numerator / denominator


# --------------------------------------------------------------------------
# Bounds, which are what collateral is sized against
# --------------------------------------------------------------------------


@pytest.mark.parametrize("format_name", sorted(FORMATS))
def test_every_metric_stays_inside_its_declared_bounds(long_seasons, format_name):
    """4,000 matches per format, every metric, every competitor.

    Settlement raises rather than voids when a resolved value falls outside the
    range its contract declared, because the collateral held against that
    contract was computed from the range. So an underestimate is not a cosmetic
    error, it is a solvency error that surfaces at the worst moment. Measured
    over this season the values sit well inside: the win rate runs 0.036 to
    0.178 in the solo mode and 0.425 to 0.645 in the objective mode against a
    declared [0, 1], eliminations per match 0.189 to 1.998 against [0, 9],
    volume 3,292 to 3,363 against [0, 8,640], placement dispersion 0.255 to
    0.318 against [0, 0.5] and the score margin -0.106 to +0.214 against
    [-1, 1].
    """
    runs = long_seasons[format_name]
    measured: dict[str, int] = {}
    for metric_name, metric in sorted(METRICS.items()):
        low, high = METRIC_BOUNDS[metric_name]
        for key in KEYS:
            try:
                value = metric(runs, key).value
            except InsufficientEvidence:
                continue
            assert low <= value <= high, (
                f"{metric_name} on {key} in {format_name} returned {value}, "
                f"outside its declared {(low, high)}"
            )
            measured[metric_name] = measured.get(metric_name, 0) + 1

    # A metric that refused every competitor would pass the loop above without
    # measuring anything, which is the shape of a test that guards nothing.
    assert measured, f"no metric produced a value in {format_name}"
    for metric_name, count in measured.items():
        assert count == len(KEYS), f"{metric_name} answered only {count} competitors"


def test_the_bounds_and_kinds_tables_cover_every_metric():
    """A metric with no declared bounds is a contract that cannot size its own
    collateral, and `metric_ref` refuses to build one rather than defaulting to
    the [0, 1] that every rate happens to use."""
    assert set(METRICS) == set(METRIC_BOUNDS) == set(METRIC_KINDS)
    for name, kind in METRIC_KINDS.items():
        assert kind in ("rate", "quantity", "dispersion"), f"{name}: {kind}"
    for name, (low, high) in METRIC_BOUNDS.items():
        assert low < high, name
    with pytest.raises(KeyError, match="declared bounds"):
        metric_ref("not_a_metric", "VANTA")


def test_metric_ref_carries_the_declared_bounds_and_kind():
    """Built from the table rather than by hand, because the bounds are both
    easy to get wrong and load bearing for collateral."""
    ref = metric_ref("match_volume", "VANTA", modes=("solo",))
    assert ref.bounds == METRIC_BOUNDS["match_volume"] == (0.0, float(MAX_WINDOW_MATCHES))
    assert ref.kind == "quantity"
    assert ref.modes == ("solo",)


def test_the_volume_bound_is_enforced_rather_than_hoped_for(oracle):
    """The ceiling on a window is what makes the volume bound a fact.

    A count has no arithmetic ceiling of its own, so the only honest way to
    declare one is to refuse a window that could exceed it. Thirty days at the
    schedule's five minutes a match is exactly 8,640 of them and resolves; one
    match more is refused, and the refusal names the bound it is protecting.
    """
    ceiling = ObservationWindow(
        start=CALENDAR.epoch, end=CALENDAR.epoch + timedelta(days=30)
    )
    assert len(CALENDAR.match_ids(ceiling)) == MAX_WINDOW_MATCHES

    resolution = oracle.resolve(
        metric_ref("match_volume", "VANTA", modes=("solo",)), ceiling
    )
    assert 0.0 <= resolution.value <= MAX_WINDOW_MATCHES
    # Ten of twelve compete in each solo match, so the expectation is 8,640 x
    # 10/12 = 7,200. Measured 7,267, which is what a bound has to sit above.
    assert 7_000 < resolution.value < 7_400

    too_long = ObservationWindow(
        start=CALENDAR.epoch,
        end=CALENDAR.epoch + timedelta(days=30) + CALENDAR.period,
    )
    with pytest.raises(MetricUnavailable, match="cannot be mapped"):
        oracle.resolve(metric_ref("match_volume", "VANTA", modes=("solo",)), too_long)


# --------------------------------------------------------------------------
# What a window is
# --------------------------------------------------------------------------


def test_a_window_is_a_half_open_run_of_match_ids():
    """Stated once here so the mapping is checked rather than described.

    Match n occupies the slot starting at epoch + n * period, and a window holds
    exactly the matches whose slot starts inside it. The week beginning
    2026-09-07 opens on the sixth day of the season, so it opens at match
    6 x 288 = 1,728 and runs 2,016 matches.
    """
    ids = CALENDAR.match_ids(WEEK)
    assert (ids.start, ids.stop) == (1_728, 3_744)
    assert CALENDAR.slot_start(ids.start) == WEEK.start
    assert CALENDAR.slot_start(ids.stop) == WEEK.end
    assert CALENDAR.per_day == 288.0


def test_consecutive_windows_tile_the_season_through_the_metric(oracle):
    """No match settles twice and none falls between two settlements.

    The tiling is easy to state on the ids and worth checking through the
    metric, because that is where losing it would show up as money: a volume
    contract on two adjacent weeks would double count a match at the seam, or
    miss one, and the fortnight would not add up.
    """
    first, second = CALENDAR.match_ids(WEEK), CALENDAR.match_ids(NEXT_WEEK)
    assert first.stop == second.start
    assert set(first).isdisjoint(second)
    assert set(CALENDAR.match_ids(FORTNIGHT)) == set(first) | set(second)

    ref = metric_ref("match_volume", "VANTA", modes=("solo",))
    week_one = oracle.resolve(ref, WEEK).value
    week_two = oracle.resolve(ref, NEXT_WEEK).value
    assert week_one + week_two == oracle.resolve(ref, FORTNIGHT).value


def test_two_windows_are_two_different_answers(oracle):
    """Or a window is decoration and every contract on the same subject is one
    contract.

    Measured over the two adjacent weeks beginning 2026-09-07 and 2026-09-14,
    all twelve competitors settle at a different solo win rate, ten of them by
    more than 0.001 and four by more than 0.01, the largest move being 0.036 on
    a field whose whole cross sectional spread is 0.036 to 0.178. A window that
    did not move the answer would mean the market had nothing to forecast
    between one settlement and the next.
    """
    moves = {}
    for key in KEYS:
        ref = metric_ref("win_rate", key, modes=("solo",))
        moves[key] = abs(
            oracle.resolve(ref, WEEK).value - oracle.resolve(ref, NEXT_WEEK).value
        )
    assert all(move > 0.0 for move in moves.values()), moves
    assert sum(1 for move in moves.values() if move > 0.001) == 10
    assert max(moves.values()) > 0.02


def test_a_window_that_opens_before_the_circuit_did_is_refused(oracle):
    """Rather than silently shortened, which would settle a volume contract
    against a season that never ran and would look like a quiet week."""
    early = ObservationWindow(
        start=CALENDAR.epoch - timedelta(days=1),
        end=CALENDAR.epoch + timedelta(days=1),
    )
    with pytest.raises(MetricUnavailable, match="before the circuit"):
        oracle.resolve(metric_ref("match_volume", "VANTA", modes=("solo",)), early)


def test_a_window_that_falls_between_matches_is_refused(oracle):
    """An empty run of ids is not a zero, it is an absence of evidence."""
    gap = ObservationWindow(
        start=CALENDAR.epoch + timedelta(minutes=1),
        end=CALENDAR.epoch + timedelta(minutes=2),
    )
    with pytest.raises(MetricUnavailable, match="no matches in window"):
        oracle.resolve(metric_ref("match_volume", "VANTA", modes=("solo",)), gap)


def test_the_listing_layer_and_the_oracle_share_one_clock():
    """Two clocks that never meet is a failure this repository has already had.

    ``market/match_book.py`` maps a match id forward to the minutes it occupies
    so it can list per match contracts; this oracle maps a window back to the
    ids inside it so it can settle aggregates. If the two disagreed about the
    epoch or the duration, a match id would name one moment to the listing layer
    and another to settlement, and nothing would raise: a contract listed on
    match 400 would quietly settle against whatever match 400 meant to the other
    side. That is the shape of the unwired clock that left the venue's lifecycle
    check running zero times in the live market while passing its own tests.

    Measured here rather than assumed: ``match_window(400)`` resolves to exactly
    ``range(400, 401)`` under this calendar, so a per match window names its own
    match and nothing else. The import is inside the test so that only this one
    fails if the market layer is mid-edit.
    """
    from arena.market import match_book

    assert CALENDAR.epoch == match_book.EPOCH
    assert CALENDAR.period == match_book.DURATION
    for match_id in (0, 400, 1_728, MAX_WINDOW_MATCHES - 1):
        window = match_book.match_window(match_id)
        assert CALENDAR.match_ids(window) == range(match_id, match_id + 1)


def test_the_calendar_refuses_a_shape_it_cannot_describe():
    with pytest.raises(ValueError, match="timezone-aware"):
        MatchCalendar(epoch=datetime(2026, 1, 1), period=timedelta(minutes=15))
    with pytest.raises(ValueError, match="positive"):
        MatchCalendar(epoch=CALENDAR.epoch, period=timedelta(0))
    with pytest.raises(ValueError, match="match ids start at zero"):
        CALENDAR.slot_start(-1)


# --------------------------------------------------------------------------
# Determinism, which is what makes a settlement defensible
# --------------------------------------------------------------------------


def test_the_oracle_is_a_pure_function_of_seed_window_metric_and_subject():
    """Two oracles, same seed, same everything down to the diagnostics order.

    Settlement records are content addressed, so a resolution that differed in
    its diagnostics ordering would produce a different digest for the same
    facts and make a dispute unresolvable. The memo one oracle builds and the
    other does not must therefore be invisible in the answer.
    """
    first, second = CircuitOracle(SEED), CircuitOracle(SEED)
    for metric_name in sorted(METRICS):
        modes = ("objective",) if metric_name == "score_margin" else ("solo",)
        if metric_name == "format_dispersion":
            ref = metric_ref(metric_name, "VANTA")
        else:
            ref = metric_ref(metric_name, "VANTA", modes=modes)
        left = first.resolve(ref, WEEK)
        # Resolved twice on the same instance, so the second call runs entirely
        # out of the memo and must not differ from the first.
        first.resolve(ref, WEEK)
        right = second.resolve(ref, WEEK)
        assert left.to_dict() == right.to_dict(), metric_name

    assert first.reference_id == second.reference_id
    assert first.reference_as_of == CALENDAR.epoch


def test_a_different_seed_is_a_different_season(oracle):
    """Or a paired experiment across seeds compares one season with itself."""
    other = CircuitOracle(SEED + 4)
    ref = metric_ref("win_rate", "VANTA", modes=("solo",))
    assert oracle.resolve(ref, WEEK).value != other.resolve(ref, WEEK).value
    assert oracle.reference_id != other.reference_id


def test_the_reference_can_never_postdate_a_window_the_oracle_answers(oracle):
    """The engine's lookahead guard and the calendar's refusal are one fact.

    A generated season was fixed when its seed was chosen, so nothing about it
    was estimated from inside a window it settles. The oracle says so by dating
    its reference at the epoch, and refuses any window opening earlier, so the
    engine's check that the reference predates the window start cannot fire on
    anything this oracle would answer.
    """
    assert oracle.reference_as_of <= WEEK.start
    for window in (WEEK, NEXT_WEEK, FORTNIGHT):
        assert oracle.reference_as_of <= window.start
        assert CALENDAR.match_ids(window).start >= 0


# --------------------------------------------------------------------------
# Absence is not zero
# --------------------------------------------------------------------------


def test_a_competitor_who_never_appeared_is_refused_rather_than_zeroed(oracle):
    """Zero is a real value here, so it may not double as "no evidence".

    A single solo match seats ten of the twelve on the roster, so two
    competitors have no record in that window at all. Their win rate is not
    0.0, it does not exist, and a contract settling at 0.0 would be paying out
    on a match its subject was never in.
    """
    one_match = ObservationWindow(
        start=CALENDAR.slot_start(WEEK_FIRST_MATCH),
        end=CALENDAR.slot_start(WEEK_FIRST_MATCH + 1),
    )
    seated = play(SEED, WEEK_FIRST_MATCH, "solo").field
    absent = [key for key in KEYS if key not in seated]
    assert len(seated) == FORMATS["solo"].entrants
    assert absent, "every competitor was seated, so this window proves nothing"

    for key in absent:
        with pytest.raises(MetricUnavailable, match="did not appear"):
            oracle.resolve(metric_ref("win_rate", key, modes=("solo",)), one_match)

    # And a name the roster has never held is refused the same way, rather than
    # resolving to a tidy zero for a subject that does not exist.
    with pytest.raises(MetricUnavailable, match="did not appear"):
        oracle.resolve(metric_ref("win_rate", "NOBODY", modes=("solo",)), WEEK)


def test_a_competitor_who_appeared_and_won_nothing_settles_at_zero(oracle):
    """The other half of the same rule, and the reason the first half matters.

    Nothing here is hardcoded to a competitor: the window is scanned for a real
    subject whose win rate is exactly 0.0, and the point is that such a subject
    exists and settles rather than voiding. Over the twenty matches the week
    opens with, two of the twelve entered 18 and 16 of them respectively and won
    none, so their zero rests on evidence where an absent competitor's would
    not.
    """
    short = ObservationWindow(
        start=CALENDAR.slot_start(WEEK_FIRST_MATCH),
        end=CALENDAR.slot_start(WEEK_FIRST_MATCH + 20),
    )
    winless = []
    for key in KEYS:
        try:
            resolution = oracle.resolve(metric_ref("win_rate", key, modes=("solo",)), short)
        except MetricUnavailable:
            continue
        if resolution.value == 0.0:
            winless.append((key, resolution.sample_size))

    assert winless, "no winless competitor in this window, so it proves nothing"
    for _key, sample_size in winless:
        assert sample_size > 0, "a settled zero must rest on matches actually entered"


def test_a_dispersion_over_a_single_appearance_is_refused(oracle):
    """The population standard deviation of one number is exactly 0.0.

    Which is well formed, inside its bounds, and a lie: it says the competitor
    is perfectly consistent when the truth is that it was measured once. This
    is the same failure as returning zero for an absent subject, wearing a
    number that looks even more legitimate.
    """
    one_match = ObservationWindow(
        start=CALENDAR.slot_start(WEEK_FIRST_MATCH),
        end=CALENDAR.slot_start(WEEK_FIRST_MATCH + 1),
    )
    seated = play(SEED, WEEK_FIRST_MATCH, "solo").field
    with pytest.raises(MetricUnavailable, match="too few appearances"):
        oracle.resolve(
            metric_ref("placement_dispersion", seated[0], modes=("solo",)), one_match
        )

    # Directly, so the refusal is the metric's and not the oracle's.
    single = season(SEED, 1, "solo")
    with pytest.raises(InsufficientEvidence, match="too few appearances"):
        placement_dispersion(single, single[0].field[0])


# --------------------------------------------------------------------------
# What the oracle may not see
# --------------------------------------------------------------------------


def test_the_oracle_cannot_reach_latent_strength():
    """Structural, not a convention anyone has to remember.

    Three separate statements of the same fact. The modules do not name the
    function that draws strengths, so it is not in their namespace to call. The
    metric signatures take a record and a subject and nothing else, so there is
    no argument a strength could arrive through even by accident. And the
    oracle's own state is a seed, a calendar, a threshold and a memo of played
    matches, none of which is a parameter of the generator.
    """
    for module in (circuit_metrics, circuit_oracle):
        # The function is named in the module docstrings, which is the point of
        # the note. What must not appear is code that reaches it.
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "draw_strengths(" not in source, module.__name__
        assert "from arena.worlds.circuit.roster import" not in source, module.__name__
        assert not hasattr(module, "draw_strengths"), module.__name__
        assert not hasattr(module, "ROSTER"), module.__name__

    for name, metric in sorted(METRICS.items()):
        parameters = list(inspect.signature(metric).parameters)
        assert parameters == ["results", "subject", "min_appearances"], name

    state = set(CircuitOracle(SEED).__dict__)
    assert state == {"_seed", "_calendar", "_min_appearances", "_memo"}


def test_a_metric_is_a_function_of_the_record_and_nothing_else():
    """Fed a record it could not have generated, a metric still answers it.

    The strongest available statement of the separation: these matches do not
    come from any seed, so any metric that consulted a generator would have
    nothing to consult. The record says the subject entered three matches and
    won two, so the win rate is 2/3 and the eliminations per match 1.0, and no
    strength anywhere could move either.
    """
    made_up = tuple(
        MatchResult(
            match_id=index,
            format_name="solo",
            field=("A", "B"),
            placements={"A": 1 if index < 2 else 2, "B": 2 if index < 2 else 1},
            eliminations={"A": 1 if index < 1 else 0, "B": 0 if index < 1 else 1},
        )
        for index in range(3)
    )
    assert win_rate(made_up, "A").value == pytest.approx(2 / 3)
    assert METRICS["eliminations_per_match"](made_up, "A").value == pytest.approx(1 / 3)
    assert match_volume(made_up, "A").value == 3.0


def test_a_resolution_carries_no_trace_of_a_strength(oracle):
    """The settlement record is published, so what it contains is what a
    counterparty can read back."""
    resolution = oracle.resolve(metric_ref("win_rate", "VANTA", modes=("solo",)), WEEK)
    published = str(resolution.to_dict())
    for forbidden in ("strength", "skill", "latent", "archetype"):
        assert forbidden not in published.lower(), forbidden


# --------------------------------------------------------------------------
# The two formats are two different questions
# --------------------------------------------------------------------------


def test_the_two_formats_rank_the_field_differently(long_seasons):
    """Measured through the metric layer, Spearman **-0.357**.

    This is the number that justifies listing a second family of contracts at
    all. The previous world's flaw was forty-seven instruments written on four
    numbers, so a trader who priced one had priced most of them. Here the solo
    win rate and the objective win rate rank the twelve competitors at Spearman
    -0.357 over 4,000 matches per format on seed 7, which is to say knowing one
    tells you close to nothing about the other, and it tells you the wrong
    thing if you assume otherwise.
    """
    solo = {key: win_rate(long_seasons["solo"], key).value for key in KEYS}
    objective = {key: win_rate(long_seasons["objective"], key).value for key in KEYS}

    rho = spearman(solo, objective)
    assert rho == pytest.approx(-0.357, abs=0.002), rho
    # Not merely uncorrelated by rank: the orderings disagree at the top, which
    # is where the money is. The best solo competitor is not the best objective
    # one, and neither is close.
    assert max(solo, key=solo.get) != max(objective, key=objective.get)


def test_format_dispersion_is_the_number_that_disagreement_makes_tradeable(
    long_seasons,
):
    """A specialist prices high, an all-rounder prices low.

    Measured over 4,000 matches per format on seed 7 this runs 0.0011 to 0.0914
    and correlates with the win rates it is built from at Pearson -0.267 in the
    solo mode and +0.631 in the objective mode, so it restates neither. The
    competitor that scores highest is the one furthest apart in the two modes,
    which is precisely the exposure nothing else here can be written on.
    """
    both = long_seasons["solo"] + long_seasons["objective"]
    spread = {key: METRICS["format_dispersion"](both, key).value for key in KEYS}
    solo = {key: win_rate(long_seasons["solo"], key).value for key in KEYS}
    objective = {key: win_rate(long_seasons["objective"], key).value for key in KEYS}

    assert min(spread.values()) == pytest.approx(0.0011, abs=0.0005)
    assert max(spread.values()) == pytest.approx(0.0914, abs=0.0005)
    assert pearson(spread, solo) == pytest.approx(-0.267, abs=0.01)
    assert pearson(spread, objective) == pytest.approx(+0.631, abs=0.01)

    widest = max(spread, key=spread.get)
    gap = abs((solo[widest] - 0.1) - (objective[widest] - 0.5))
    assert gap == pytest.approx(2 * spread[widest], abs=1e-12)


def test_a_single_format_window_has_no_format_dispersion(oracle):
    """Zero across one format is the schedule speaking, not the competitor."""
    with pytest.raises(MetricUnavailable, match="one format in the window"):
        oracle.resolve(
            metric_ref("format_dispersion", "VANTA", modes=("solo",)), WEEK
        )
    both = oracle.resolve(metric_ref("format_dispersion", "VANTA"), WEEK)
    assert both.value > 0.0


# --------------------------------------------------------------------------
# Metrics that would have been restatements, and the numbers that say so
# --------------------------------------------------------------------------


def test_placement_dispersion_refuses_a_team_format_it_would_only_restate(
    long_seasons,
):
    """Measured first, refused second, which is the order that matters.

    Where a format places every member of a side together, the normalized
    placement is an indicator, so its standard deviation is sqrt(w(1-w)) in the
    win rate w over the field size less one, exactly. Measured across the whole
    field over 4,000 objective matches the largest gap between the two is
    1.4e-17, which is float noise: the metric carries no information the win
    rate does not, and a number that restates another one is worse than no
    number because it looks like a hedge and is not.
    """
    runs = long_seasons["objective"]
    span = len(runs[0].field) - 1
    worst = 0.0
    for key in KEYS:
        rate = win_rate(runs, key).value
        places = [
            (result.placements[key] - 1) / span
            for result in runs
            if key in result.field
        ]
        worst = max(
            worst,
            abs(statistics.pstdev(places) - math.sqrt(rate * (1 - rate)) / span),
        )
    assert worst < 1e-15, worst

    with pytest.raises(InsufficientEvidence, match="does not rank individuals"):
        placement_dispersion(runs, KEYS[0])


def test_placement_dispersion_is_not_just_the_solo_win_rate_wearing_a_hat(
    long_seasons,
):
    """It correlates at -0.840, and what is left over is real.

    Most of a dispersion in a world with one latent scalar per format is the
    first moment showing through, and it would be dishonest to claim otherwise.
    What justifies the metric is the residual: fitting a straight line in the
    win rate leaves a standard deviation of 0.0094 against a sampling floor of
    sd/sqrt(2n) = 0.0037, a factor of 2.6, so there is structure beyond the win
    rate rather than noise dressed as one.
    """
    runs = long_seasons["solo"]
    rate = {key: win_rate(runs, key).value for key in KEYS}
    spread = {key: placement_dispersion(runs, key).value for key in KEYS}
    counts = {key: win_rate(runs, key).sample_size for key in KEYS}

    assert pearson(spread, rate) == pytest.approx(-0.840, abs=0.01)

    mean_rate = statistics.fmean(rate.values())
    mean_spread = statistics.fmean(spread.values())
    slope = sum(
        (rate[k] - mean_rate) * (spread[k] - mean_spread) for k in KEYS
    ) / sum((rate[k] - mean_rate) ** 2 for k in KEYS)
    residual = statistics.pstdev(
        [spread[k] - (mean_spread + slope * (rate[k] - mean_rate)) for k in KEYS]
    )
    floor = statistics.fmean([spread[k] / math.sqrt(2 * counts[k]) for k in KEYS])
    assert residual / floor > 2.0, (residual, floor)


def test_score_margin_needs_a_scored_format(oracle, long_seasons):
    """Which is the whole reason it is here.

    The solo mode produces a placement and no points, so a margin does not
    exist there. That is what stops the two families of contracts from being
    the same family twice. Its content within the objective mode is another
    matter and is documented on the metric: measured over 4,000 objective
    matches it correlates with the win rate at Pearson +0.998 and the residual
    after a straight line is 0.0047 against a sampling floor of 0.0153, so it
    adds no cross sectional signal about who is good, and it is listed for the
    exposure rather than the information.
    """
    with pytest.raises(MetricUnavailable, match="keeps no score"):
        oracle.resolve(metric_ref("score_margin", "VANTA", modes=("solo",)), WEEK)

    runs = long_seasons["objective"]
    margin = {key: METRICS["score_margin"](runs, key).value for key in KEYS}
    rate = {key: win_rate(runs, key).value for key in KEYS}
    assert pearson(margin, rate) == pytest.approx(0.998, abs=0.002)
    assert min(margin.values()) >= -1.0 and max(margin.values()) <= 1.0


def test_a_rate_refuses_a_mixed_format_window_and_a_count_does_not(oracle):
    """The asymmetry is the point rather than an inconsistency.

    A win in the solo mode sits against a neutral 0.100 and a win in the
    objective mode against 0.500, so a pooled rate would move when the schedule
    changed the mix even though nothing about the competitor changed. A count of
    appearances means the same thing whichever mode produced it, so it pools:
    measured over the week beginning 2026-09-07, Vanta entered 1,703 solo
    matches and 1,033 objective ones, and the pooled volume is 2,736.
    """
    for metric_name in ("win_rate", "eliminations_per_match", "score_margin"):
        with pytest.raises(MetricUnavailable, match="mixes formats"):
            oracle.resolve(metric_ref(metric_name, "VANTA"), WEEK)

    solo = oracle.resolve(metric_ref("match_volume", "VANTA", modes=("solo",)), WEEK)
    team = oracle.resolve(metric_ref("match_volume", "VANTA", modes=("objective",)), WEEK)
    pooled = oracle.resolve(metric_ref("match_volume", "VANTA"), WEEK)
    assert pooled.value == solo.value + team.value


def test_the_elimination_rate_is_scale_invariant_and_the_count_is_not(long_seasons):
    """Which is why it is declared a rate and not a quantity.

    The distinction is the one that cost this project the most: a settlement
    value re-dated onto a four week window without rescaling a quantity priced
    a contract at 274.92 against a fair 71.09, and a rate is scale invariant in
    window length where a count is not. Measured on seed 7 between a 672 match
    run and a 4,032 match run six times as long, the raw elimination count
    grows by 5.51x to 6.72x and appearances by 5.86x to 6.14x, while the per
    match average moves by 0.2% to 8.9%, which is sampling noise on a mean.
    """
    short = season(SEED, 672, "solo")
    long_run = season(SEED, 4_032, "solo")

    rate_drift = []
    count_ratio = []
    volume_ratio = []
    for key in KEYS:
        near = METRICS["eliminations_per_match"](short, key)
        far = METRICS["eliminations_per_match"](long_run, key)
        rate_drift.append(abs(near.value - far.value) / far.value)
        count_ratio.append(
            dict(far.diagnostics)["eliminations"] / dict(near.diagnostics)["eliminations"]
        )
        volume_ratio.append(
            match_volume(long_run, key).value / match_volume(short, key).value
        )

    assert max(rate_drift) < 0.10, max(rate_drift)
    assert min(count_ratio) > 5.0 and max(count_ratio) < 7.5
    assert min(volume_ratio) > 5.5 and max(volume_ratio) < 6.5
    assert METRIC_KINDS["eliminations_per_match"] == "rate"
    assert METRIC_KINDS["match_volume"] == "quantity"


# --------------------------------------------------------------------------
# Refusals the settlement engine relies on
# --------------------------------------------------------------------------


def test_an_unknown_metric_or_format_is_refused(oracle):
    """Rather than silently measuring something adjacent."""
    unknown_metric = MetricRef(metric="vibes", subject="VANTA", modes=("solo",))
    with pytest.raises(MetricUnavailable, match="unknown metric"):
        oracle.resolve(unknown_metric, WEEK)

    with pytest.raises(MetricUnavailable, match="unknown format"):
        oracle.resolve(metric_ref("win_rate", "VANTA", modes=("duel",)), WEEK)


def test_a_filter_from_the_other_world_is_refused_rather_than_ignored(oracle):
    """A filter that is ignored is a filter that does not exist.

    Maps and trophy buckets are the Brawl world's strata. A circuit match has
    no such dimension, and honouring the request as a no-op would settle
    something wider than the contract named while looking entirely normal,
    which is the shape of every guard in this repository that never fired.
    """
    ref = MetricRef(
        metric="win_rate", subject="VANTA", modes=("solo",), maps=("HardRockMine",)
    )
    with pytest.raises(MetricUnavailable, match="not meaningful in this world"):
        oracle.resolve(ref, WEEK)
    assert ref.trophy_buckets == (ALL,)


def test_a_contract_settles_reproducibly_through_the_engine(oracle):
    """The layer's real acceptance test: a contract, an oracle, one number.

    Everything above is about the statistics layer in isolation. This is the
    only test that runs the whole path the exchange actually uses, and it is
    what proves the declared bounds and the engine's own out of range guard
    agree: settlement raises rather than voids when a resolved value falls
    outside the contract's range, so a contract built from METRIC_BOUNDS must
    settle inside it.
    """
    ref = metric_ref("win_rate", "VANTA", modes=("solo",))
    spec = ContractSpec(
        contract_id="VANTA_WR_SOLO_2026W10",
        underlying=Single(ref),
        payoff=Linear(scale=10_000.0),
        window=WEEK,
        policy=DataPolicy(min_sample_size=100),
        reference_id=oracle.reference_id,
        published_at=WEEK.start - timedelta(days=1),
        tick_size="0.25",
    )

    first = settle(spec, oracle)
    second = settle(spec, CircuitOracle(SEED))
    assert first.status is SettlementStatus.SETTLED
    assert first.result_digest == second.result_digest

    low, high = spec.settlement_bounds
    assert (low, high) == (Decimal("0"), Decimal("1E+4"))
    assert low <= first.settlement_value <= high
    # Vanta entered 1,703 solo matches in the week and won 248 of them, so
    # 0.145625 at a scale of 10,000 quantized onto a quarter point tick.
    assert first.settlement_value == Decimal("1456.25")


def test_a_thin_window_voids_rather_than_settling(oracle):
    """A contract whose evidence never materialized should void loudly.

    The engine gates on the contract's own min_sample_size, and the sample size
    a circuit resolution reports is appearances rather than matches in the
    window, because a competitor that was not drawn into a match was not tested
    by it.
    """
    spec = ContractSpec(
        contract_id="VANTA_WR_SOLO_THIN",
        underlying=Single(metric_ref("win_rate", "VANTA", modes=("solo",))),
        payoff=Linear(scale=10_000.0),
        window=ObservationWindow(
            start=CALENDAR.slot_start(WEEK_FIRST_MATCH),
        end=CALENDAR.slot_start(WEEK_FIRST_MATCH + 20),
        ),
        policy=DataPolicy(min_sample_size=1_000),
        reference_id=oracle.reference_id,
        published_at=WEEK.start - timedelta(days=1),
        tick_size="0.25",
    )
    result = settle(spec, oracle)
    assert result.status is SettlementStatus.VOID
    assert "sample size" in result.void_reason
