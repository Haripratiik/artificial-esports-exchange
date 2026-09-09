"""Canonical metric definitions for the circuit world.

These are the quantities contracts settle on, defined once in code rather than
described in prose and reimplemented per experiment. A contract names a metric,
a subject and a window; the oracle turns the window into a run of matches and
hands the records here, and what comes back is a number plus the evidence trail
behind it.

**Every metric is a pure function of the match record.** Nothing in this module
imports :func:`arena.worlds.circuit.roster.draw_strengths`, and no metric takes
a seed. That is the separation the whole exchange rests on: strength is the
parameter that generates outcomes, and a metric that could read it would be
reporting the answer rather than measuring it. The signature is
``(results, subject)``, so there is no argument through which the parameter
could arrive even by accident.

Six metrics, and they are deliberately not six views of one number. The world
carries one latent scalar per competitor per format, so within a single format
every skill statistic ranks the field almost identically: measured over 4,000
solo matches on seed 7, eliminations per match ranks the field exactly as the
win rate does, Spearman +1.000. That is a fact about the world rather than a
defect in the metrics, and it is why the set is separated along the axes that
genuinely do carry different information. Those axes are the two formats, whose
win rates rank the field at Spearman -0.357; scheduling volume, which cannot
depend on strength because the field is drawn without reading it; and second
moments, which are not monotone in the scalar the first moments are monotone in.

    win_rate                how often the subject's side won, per appearance.

    eliminations_per_match  credited eliminations per appearance. A rate, not a
                            quantity, for the reason argued on the function.

    match_volume            appearances in the window. The one quantity here,
                            and the one number that is not about skill at all.

    placement_dispersion    how widely the subject finishes around its own mean
                            finishing position. Only meaningful where a format
                            ranks individuals.

    format_dispersion       how differently the subject performs across the
                            formats it played. The metric the -0.357 makes
                            tradeable, and the only one that needs more than
                            one format to mean anything.

    score_margin            average normalized points margin. Only meaningful
                            where a format keeps score.

**On pooling formats.** A win rate in a ten-way elimination sits against a
neutral point of 0.100 and a win rate in a three-a-side objective sits against
0.500, so an average over a window holding both is a function of how many of
each the schedule happened to run. That is the composition problem the Brawl
world solves by standardizing onto pinned weights, and here it has a cleaner
answer: refuse. Every per match average checks that the matches it was handed
all came from one format and raises otherwise, so a contract on a win rate has
to name its format. Counts do not have this problem, because a count of
appearances means the same thing whichever format produced it, so
:func:`match_volume` pools freely and says so.

**On absence.** A subject that never appeared raises rather than returning
zero. Zero is a real value here: a competitor that entered forty matches and
won none has a win rate of 0.0, and a competitor that entered none has no win
rate. Collapsing the two would settle a contract on evidence that does not
exist, and it is the exact shape of failure the venue refuses everywhere else.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from arena.determinism import stable_sum
from arena.worlds.circuit.match import MatchResult
from arena.worlds.circuit.modes import FORMATS

__all__ = [
    "MetricOutcome",
    "InsufficientEvidence",
    "MAX_WINDOW_MATCHES",
    "MAX_ELIMINATION_CREDIT",
    "win_rate",
    "eliminations_per_match",
    "match_volume",
    "placement_dispersion",
    "format_dispersion",
    "score_margin",
    "METRICS",
    "METRIC_BOUNDS",
    "METRIC_KINDS",
    "metric_ref",
]


# The longest run of matches a single observation window may cover. The oracle
# enforces it and refuses a longer window, which is what makes the ceiling
# `match_volume` declares a fact rather than a hope: a subject cannot appear in
# more matches than the window holds, so the bound cannot be exceeded by any
# schedule, any seed or any roster.
#
# 8,640 is thirty days at the schedule the circuit actually runs, one match
# every five minutes. Thirty days is also the longest observation window
# anything in this repository writes, the Brawl contracts settling on
# twenty-eight, so the ceiling costs nothing anybody wanted and stops a single
# settlement scanning an unbounded season. The constant lives here rather than
# with the calendar because it is the number the bound is made of, and a bound
# whose justification lives in another module is a bound nobody re-derives when
# the other module changes.
MAX_WINDOW_MATCHES = 8_640


class InsufficientEvidence(Exception):
    """The matches supplied cannot support this metric.

    A normal outcome rather than a failure. The settlement engine turns it into
    a void, which is the honest answer when the world did not produce the
    evidence a contract named.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason} ({detail})" if detail else reason)


@dataclass(frozen=True, slots=True)
class MetricOutcome:
    """A computed metric plus the evidence trail behind it."""

    value: float
    sample_size: int
    diagnostics: tuple[tuple[str, Any], ...]


def _max_credit(format_: Any) -> int:
    """The most eliminations one competitor can be credited in one match.

    In a last-one-standing format the eliminations conserve to exactly
    ``entrants - 1``, so that is the ceiling and it is reachable in principle by
    a single survivor. In a race to a target, a competitor is credited once per
    point its own side scores and the race stops at the target, so the ceiling
    is the target. A format that declares no target falls back to the
    elimination law, which is the wider of the two and therefore the safe way to
    be wrong: an overestimate costs collateral, an underestimate makes
    settlement raise.
    """
    if getattr(format_, "team_size", 1) == 1:
        return max(format_.entrants - 1, 0)
    return int(getattr(format_, "target", max(format_.entrants - 1, 0)))


# Derived from the registered formats rather than written down, so adding a mode
# that credits more cannot leave the bound behind. Over the default registry
# this is 9, from the ten-way elimination. Measured over 4,000 matches per
# format on seed 7, the largest credit anybody actually took in one match was 8
# in the solo mode and 3 in the objective mode, so the ceiling is reachable
# arithmetic rather than an observed maximum, which is the correct kind of
# bound: settlement refuses a value outside it, so it must hold for the match
# nobody has run yet.
#
# A format registered after import is not reflected here. That is a deliberate
# limitation of a module-level table and it fails loudly rather than quietly,
# because settlement rejects a resolved value outside the declared range.
MAX_ELIMINATION_CREDIT = max(_max_credit(format_) for format_ in FORMATS.values())


def _appearances(
    results: Sequence[MatchResult], subject: str
) -> tuple[MatchResult, ...]:
    """The matches ``subject`` was actually in, in a fixed order.

    Sorted by format and match id rather than left in the order the caller
    happened to build, so a metric cannot depend on how the oracle iterated its
    formats. Float addition is not associative and a reordered mean is a
    different number in the last bits, which is enough to make two runs of the
    same seed disagree on a digest.
    """
    return tuple(
        sorted(
            (result for result in results if subject in result.field),
            key=lambda result: (result.format_name, result.match_id),
        )
    )


def _require(appearances: Sequence[MatchResult], subject: str, minimum: int) -> None:
    """Refuse an empty or too thin record, naming which of the two it was."""
    if not appearances:
        raise InsufficientEvidence(
            "subject did not appear in the window",
            f"{subject!r} entered no match in the matches supplied, so it has no "
            "record here; zero is a value this metric can legitimately return "
            "for a competitor that did appear, and returning it for one that "
            "did not would settle a contract on evidence that does not exist",
        )
    if len(appearances) < minimum:
        raise InsufficientEvidence(
            "too few appearances",
            f"{subject!r} appeared in {len(appearances)} match(es), {minimum} "
            "required for this metric to mean anything",
        )


def _neutral(result: MatchResult) -> float:
    """The win rate an interchangeable field would produce in this match.

    Winning slots divided by all slots, which is 1/10 in the ten-way
    elimination and 3/6 in the three-a-side objective. Read off the record
    rather than looked up in the registry on purpose. The registry is mutable at
    runtime, so a format whose size changed after a match was played would
    retroactively change what that match's neutral point had been, and a
    settlement is supposed to be a statement about the match that happened. The
    record carries everything needed, so no metric consults ``FORMATS`` at
    resolution time; the module reads it once at import to derive an elimination
    ceiling and nowhere else.
    """
    return len(result.winners) / len(result.field)


def _one_format(appearances: Sequence[MatchResult]) -> str:
    """The single format these appearances came from, or a refusal.

    This is the guard that keeps a per match average from becoming a function of
    the schedule. A win in the solo mode sits against a neutral point of 0.100
    and a win in the objective mode against 0.500, and a solo match credits
    exactly 9 eliminations where an objective match credits 3 to 5, so a pooled
    average moves when the calendar changes the mix even though nothing about
    the competitor changed. That is the composition problem the Brawl world
    spends an entire standardization step on. Here the honest answer is to
    refuse, because the formats are ours and a contract can simply name one.
    """
    formats = sorted({result.format_name for result in appearances})
    if len(formats) > 1:
        levels = sorted({round(_neutral(result), 4) for result in appearances})
        raise InsufficientEvidence(
            "window mixes formats",
            f"formats {formats} sit at neutral levels {levels} and credit "
            "eliminations on different scales, so a pooled per match average "
            "would move with the schedule mix rather than with the competitor; "
            "name a single format on the contract",
        )
    return formats[0]


def win_rate(
    results: Sequence[MatchResult],
    subject: str,
    *,
    min_appearances: int = 0,
) -> MetricOutcome:
    """Matches won per match entered.

    A win is a first placement, which in the solo mode is the survivor and in
    the objective mode is every member of the side that reached the target. The
    denominator is appearances rather than matches in the window, because a
    competitor that was not drawn into a match was not tested by it, and
    counting those would make the number a function of how often the roster
    sampled the subject rather than of how it did.

    Measured over 4,000 matches per format on seed 7, this runs 0.036 to 0.178
    in the solo mode against a neutral 0.100, and 0.425 to 0.645 in the
    objective mode against a neutral 0.500. The two orderings agree at Spearman
    -0.357, which is the whole reason the second family of contracts is worth
    listing.
    """
    appearances = _appearances(results, subject)
    _require(appearances, subject, max(min_appearances, 1))
    _one_format(appearances)
    neutral = _neutral(appearances[0])

    wins = sum(1 for result in appearances if subject in result.winners)
    return MetricOutcome(
        value=wins / len(appearances),
        sample_size=len(appearances),
        diagnostics=(
            ("metric", "win_rate"),
            ("appearances", len(appearances)),
            ("wins", wins),
            ("neutral_level", neutral),
            ("formats", _format_label(appearances)),
        ),
    )


def eliminations_per_match(
    results: Sequence[MatchResult],
    subject: str,
    *,
    min_appearances: int = 0,
) -> MetricOutcome:
    """Credited eliminations divided by appearances.

    **This is a rate and not a quantity, and the distinction is the one that
    has cost this repository the most.** The test is not whether the number
    looks like a proportion, it is whether the number is scale invariant in the
    length of the window. Lengthening the window multiplies a count and leaves
    an average per match where it was: measured on seed 7 between a 672 match
    run and a 4,032 match run six times as long, the raw count of
    eliminations grows by a factor of 5.51 to 6.72 across the field and the
    count of appearances by 5.86 to 6.14, while the per match average moves by
    0.2% to 8.9%, which is sampling noise on a mean and not scale. The win rate
    over the same pair of windows moves by at most 0.024, so the two rates drift
    alike and the counts do not. So this has no term structure, the
    amount delivered in March is not a different thing from the amount delivered
    in April, and classifying it as a quantity would give it a delivery calendar
    it does not have. The unit bug that priced SPIKE_VOL_W1 at 274.92 against a
    settlement of 71.09 was exactly this distinction declared upstream and
    ignored downstream.

    It is a rate whose range is not [0, 1], which is fine and is why bounds are
    declared per metric rather than assumed. Measured over 4,000 matches per
    format on seed 7, this runs 0.189 to 1.998 in the solo mode and 0.434 to
    1.153 in the objective mode.

    Within one format it ranks the field exactly as the win rate does, Spearman
    +1.000 over the solo mode and +0.944 over the objective mode. That is not a
    reason to drop it: it settles against a different number with a different
    variance, and an over-under on eliminations is a bet a winner contract
    cannot express. It is a reason not to pretend it adds cross sectional
    information, and the number is here so nobody has to rediscover that.
    """
    appearances = _appearances(results, subject)
    _require(appearances, subject, max(min_appearances, 1))
    _one_format(appearances)
    neutral = _neutral(appearances[0])

    credited = sum(result.eliminations.get(subject, 0) for result in appearances)
    return MetricOutcome(
        value=credited / len(appearances),
        sample_size=len(appearances),
        diagnostics=(
            ("metric", "eliminations_per_match"),
            ("appearances", len(appearances)),
            ("eliminations", credited),
            ("neutral_level", neutral),
            ("formats", _format_label(appearances)),
        ),
    )


def match_volume(
    results: Sequence[MatchResult],
    subject: str,
    *,
    min_appearances: int = 0,
) -> MetricOutcome:
    """How many matches the subject entered in the window, in matches.

    A **quantity**, and the only one here. It scales with the window by
    definition, so a contract on it has a genuine term structure and a January
    contract is not a restatement of a February one.

    **It pools formats, and it is the only metric that may.** An appearance
    means the same thing whichever mode produced it, so adding a solo entry to
    an objective entry gives a number that is still a count of entries. Every
    per match average refuses that same window, because a solo win sits against
    a neutral 0.100 and an objective win against 0.500, and that asymmetry is
    the point rather than an inconsistency.

    **It is also the only metric here that is not about skill.** The field is
    drawn by ``roster.roster_for``, which seeds itself on the world seed, the
    format and the match id and never reads a strength, so appearances cannot
    depend on how good a competitor is. Measured over 4,000 matches per format
    on seed 7 the cross sectional correlation with the win rate is +0.31 in the
    solo mode and -0.41 in the objective mode, both inside the +/-0.58 that
    twelve independent points reach at the five percent level, so the apparent
    tilt is sampling noise sitting on top of a structural independence.

    Denominated in matches rather than thousands of them. The ceiling is 8,640,
    which is a level a person can hold in their head and quote on a tick grid;
    dividing by a thousand would put every realistic window under ten and make
    the tick size carry the whole number.
    """
    appearances = _appearances(results, subject)
    _require(appearances, subject, max(min_appearances, 1))

    formats = sorted({result.format_name for result in appearances})
    return MetricOutcome(
        value=float(len(appearances)),
        sample_size=len(appearances),
        diagnostics=(
            ("metric", "match_volume"),
            ("appearances", len(appearances)),
            ("formats", "|".join(formats)),
            ("matches_supplied", len(results)),
            ("basis", "entries in the scheduled season, not a census of play"),
        ),
    )


def placement_dispersion(
    results: Sequence[MatchResult],
    subject: str,
    *,
    min_appearances: int = 0,
) -> MetricOutcome:
    """How widely the subject finishes around its own mean finishing position.

    The population standard deviation of the normalized placement, where
    normalized placement is ``(place - 1) / (entrants - 1)`` so that first is
    0.0, last is 1.0 and the number does not change meaning when a format
    changes its field size. A rate lives in [0, 1], so the standard deviation of
    a set of them cannot exceed 0.5, reached only by a subject that finishes
    first in half its matches and last in the other half. That is what keeps
    collateral arithmetic rather than a variance estimate.

    It is a different kind of claim from a win rate rather than a different view
    of one. A competitor that always finishes fifth and one that alternates
    first and last have the same mean placement and are not the same thing to
    own. Measured over 4,000 solo matches on seed 7 this runs 0.255 to 0.318
    and correlates with the win rate at Pearson -0.840, so most of it is the
    first moment showing through; what is left is real rather than noise, with a
    residual standard deviation of 0.0094 after a straight line in the win rate
    against a sampling floor of 0.0037, a factor of 2.6.

    **A team format is refused rather than measured.** Where placements take
    only the values 1 and 2 the normalized placement is an indicator, so this
    collapses to ``sqrt(w(1-w))`` in the win rate w divided by the field size
    less one, exactly. That is an algebraic restatement and not a second number:
    measured across the whole field over 4,000 objective matches the largest gap
    between the two was 1.4e-17, which is float noise. A metric that restates
    another one is worse than no metric, because it looks like a hedge and is
    not, so the refusal is explicit.
    """
    appearances = _appearances(results, subject)
    # Two, because the population standard deviation of one value is exactly
    # 0.0, and a well formed zero from a single match is the same lie as a zero
    # from no matches at all.
    _require(appearances, subject, max(min_appearances, 2))
    _one_format(appearances)

    unrankable = [
        result
        for result in appearances
        if len(set(result.placements.values())) != len(result.placements)
    ]
    if unrankable:
        raise InsufficientEvidence(
            "format does not rank individuals",
            f"{unrankable[0].format_name!r} places every member of a side "
            "together, so its placement spread is sqrt(w(1-w)) in the win rate w "
            "over the field size less one, exactly, and carries no information "
            "the win rate does not; measured over 4,000 objective matches the "
            "worst gap between the two was 1.4e-17",
        )

    entrants = len(appearances[0].field)
    if entrants < 2:
        raise InsufficientEvidence(
            "a one entrant format has no placement spread",
            "every placement is first, so the normalization has no denominator",
        )

    # Normalized per match rather than against a field size read once, so a
    # format that ever varied its entrant count would still produce placements
    # on one scale instead of silently mixing two.
    places = [
        (result.placements[subject] - 1) / (len(result.field) - 1)
        for result in appearances
    ]
    mean = stable_sum(places) / len(places)
    variance = stable_sum((place - mean) ** 2 for place in places) / len(places)

    return MetricOutcome(
        value=variance**0.5,
        sample_size=len(appearances),
        diagnostics=(
            ("metric", "placement_dispersion"),
            ("appearances", len(appearances)),
            ("mean_normalized_placement", mean),
            ("variance", variance),
            ("entrants", entrants),
            ("formats", _format_label(appearances)),
        ),
    )


def format_dispersion(
    results: Sequence[MatchResult],
    subject: str,
    *,
    min_appearances: int = 0,
) -> MetricOutcome:
    """How differently the subject performs across the formats it played.

    For each format in the window, take the subject's win rate there and
    subtract that format's neutral point, which gives a lift that means the same
    thing in a ten-way elimination as in a three-a-side objective. The metric is
    the population standard deviation of those lifts.

    **This is the one metric that needs more than one format to exist, and it is
    the one that carries information neither format's win rate does.** The two
    modes rank the field at Spearman -0.357, so a competitor can be strong in
    one and weak in the other, and this is the number that says so. Measured
    over 4,000 matches per format on seed 7 it runs 0.0011 to 0.0914 and
    correlates with the underlying win rates at Pearson -0.267 against the solo
    mode and +0.631 against the objective mode, so it is not a restatement of
    either. A specialist prices high and an all-rounder prices low, and nothing
    else here can be written on that distinction.

    A lift is a win rate minus a neutral point and both live in [0, 1], so every
    lift lives in [-1, 1] and the population standard deviation of a set of them
    cannot exceed 1.0, reached only by a subject that never wins in one format
    and always wins in another. That is roughly eleven times the largest value
    measured, which is the correct direction to be wrong in: settlement refuses
    a value outside the declared range, so an underestimate fails loudly and an
    overestimate only costs collateral.

    **A window holding one format is refused rather than answered with zero.**
    The spread across a single format is 0.0 by construction, and a contract
    settling at 0.0 because the schedule ran one mode would be paying out on the
    calendar rather than on the competitor.
    """
    appearances = _appearances(results, subject)
    _require(appearances, subject, max(min_appearances, 2))

    by_format: dict[str, list[MatchResult]] = {}
    for result in appearances:
        by_format.setdefault(result.format_name, []).append(result)

    if len(by_format) < 2:
        raise InsufficientEvidence(
            "one format in the window",
            f"{subject!r} played only {sorted(by_format)}, and a spread across one "
            "format is zero by construction; a contract settling there would be "
            "paying out on the schedule rather than on the competitor",
        )

    lifts: list[tuple[str, float]] = []
    for format_name in sorted(by_format):
        entries = by_format[format_name]
        wins = sum(1 for result in entries if subject in result.winners)
        lifts.append(
            (format_name, wins / len(entries) - _neutral(entries[0]))
        )

    values = [lift for _name, lift in lifts]
    mean = stable_sum(values) / len(values)
    variance = stable_sum((lift - mean) ** 2 for lift in values) / len(values)

    return MetricOutcome(
        value=variance**0.5,
        sample_size=len(appearances),
        diagnostics=(
            ("metric", "format_dispersion"),
            ("appearances", len(appearances)),
            ("formats", "|".join(name for name, _lift in lifts)),
            ("lifts", tuple(round(lift, 6) for _name, lift in lifts)),
            ("mean_lift", mean),
            ("variance", variance),
        ),
    )


def score_margin(
    results: Sequence[MatchResult],
    subject: str,
    *,
    min_appearances: int = 0,
) -> MetricOutcome:
    """Average points margin per match, normalized by the target.

    In a scored format the subject's side finishes on some number of points and
    the other side on another, and the difference divided by the winning score
    gives a margin in [-1, 1]: a clean sweep is +1.0 or -1.0, a three to two is
    +1/3 or -1/3. Normalizing by the target rather than reporting raw points is
    what makes the bound independent of the format, so registering a race to
    five later cannot silently invalidate a declared range. The target is read
    off the record as the winning side's score rather than looked up, for the
    same reason the neutral point is.

    **Only a scored format supports this, which is the point.** The solo mode
    has no score, so the two families of contracts are genuinely different
    instruments rather than the same instrument twice, and a trader who has
    priced the elimination world has not thereby priced this one.

    **What it does not add, stated plainly.** Over a long window it is a linear
    restatement of the objective win rate in the cross section. Measured over
    4,000 objective matches on seed 7, the mean margin runs -0.106 to +0.214 and
    correlates with the win rate at Pearson +0.998, and the residual after
    fitting a straight line in the win rate has a standard deviation of 0.0047
    against a sampling floor on the mean margin of 0.0153, so the residual is
    below the noise and there is no measurable extra signal about who is good.
    That is a property of a world with one latent scalar per format rather than
    a defect here, and it does not make the contract a duplicate: measured
    across all 24,000 competitor slots in those matches the normalized margin
    has a per match standard deviation of 0.687 where a win indicator at a
    neutral 0.500 has 0.500, so over a short window the two settle with
    materially different precision on the same underlying question, and its
    range is signed where the win rate's is not, which changes what collateral a
    short has to post.
    """
    appearances = _appearances(results, subject)
    _require(appearances, subject, max(min_appearances, 1))
    _one_format(appearances)

    unscored = [result for result in appearances if not result.scores]
    if unscored:
        raise InsufficientEvidence(
            "format keeps no score",
            f"{unscored[0].format_name!r} produces a placement and no points, so "
            "there is no margin to average; a scored format is required",
        )

    margins: list[float] = []
    for result in appearances:
        mine = result.scores[subject]
        others = {
            score
            for key, score in result.scores.items()
            if result.placements[key] != result.placements[subject]
        }
        if len(others) != 1:
            raise InsufficientEvidence(
                "more than two scoring sides",
                f"match {result.match_id} in {result.format_name!r} shows "
                f"{len(others)} opposing scores, and a margin is only defined "
                "against a single opponent",
            )
        theirs = others.pop()
        target = max(mine, theirs)
        if target <= 0:
            raise InsufficientEvidence(
                "a match nobody scored in",
                f"match {result.match_id} finished nil all, so the margin has no "
                "scale to normalize against",
            )
        margins.append((mine - theirs) / target)

    mean = stable_sum(margins) / len(margins)
    swept = sum(1 for margin in margins if abs(margin) == 1.0)
    return MetricOutcome(
        value=mean,
        sample_size=len(appearances),
        diagnostics=(
            ("metric", "score_margin"),
            ("appearances", len(appearances)),
            ("sweeps", swept),
            ("formats", _format_label(appearances)),
        ),
    )


def _format_label(appearances: Sequence[MatchResult]) -> str:
    return "|".join(sorted({result.format_name for result in appearances}))


# The registry the oracle dispatches on. A contract naming a metric outside this
# mapping fails at resolution rather than silently measuring something adjacent.
METRICS = {
    "win_rate": win_rate,
    "eliminations_per_match": eliminations_per_match,
    "match_volume": match_volume,
    "placement_dispersion": placement_dispersion,
    "format_dispersion": format_dispersion,
    "score_margin": score_margin,
}

# The range each metric can take. A contract declares its own bounds, since the
# settlement layer must not import this module, but declaring them wrongly is
# easy and consequential: the bounds decide how much collateral a position
# needs, and settlement raises rather than voiding when a resolved value falls
# outside them. So every entry here is either arithmetic that cannot be breached
# or a ceiling the oracle enforces, never an observed maximum.
METRIC_BOUNDS: dict[str, tuple[float, float]] = {
    # Wins over appearances, both counts, the first no larger than the second.
    "win_rate": (0.0, 1.0),
    # Derived from the registered formats. 9 over the default registry, from the
    # ten-way elimination whose credits conserve to exactly nine. Measured over
    # 4,000 matches per format on seed 7 the largest single match credit anyone
    # took was 8 in the solo mode and 3 in the objective mode, and a window
    # average of 9 would need every one of those to go to one competitor.
    "eliminations_per_match": (0.0, float(MAX_ELIMINATION_CREDIT)),
    # A subject cannot enter more matches than the window holds, and the oracle
    # refuses a window holding more than MAX_WINDOW_MATCHES. Enforced rather
    # than observed: over a 2,016 match week a solo competitor appears 1,657 to
    # 1,710 times and an objective competitor 978 to 1,042, so the realistic
    # range is well inside this, but the bound has to hold for the window
    # nobody has written yet.
    "match_volume": (0.0, float(MAX_WINDOW_MATCHES)),
    # Normalized placements live in [0, 1], so their standard deviation cannot
    # exceed 0.5, reached only by finishing first in half the matches and last
    # in the other half. Measured range over 4,000 solo matches is 0.255 to
    # 0.318.
    "placement_dispersion": (0.0, 0.5),
    # A lift is a win rate minus a neutral point, so it lives in [-1, 1] and the
    # standard deviation of a set of lifts cannot exceed 1.0. Measured range
    # over 4,000 matches per format is 0.0011 to 0.0914, so the ceiling is
    # roughly eleven times the largest value seen; an overestimate costs
    # collateral where an underestimate makes settlement raise.
    "score_margin": (-1.0, 1.0),
    "format_dispersion": (0.0, 1.0),
}

# Whether each metric measures a proportion, an amount delivered, or a spread.
# A contract on an amount is a commodity: it has a delivery window that means
# something on its own and therefore a term structure, which a rate does not.
METRIC_KINDS: dict[str, str] = {
    "win_rate": "rate",
    # A rate despite not being a proportion, because the test that matters is
    # scale invariance in the window length rather than whether the number sits
    # in [0, 1]. Argued in full on the function.
    "eliminations_per_match": "rate",
    # The one quantity. Doubling the window doubles it, so a March contract and
    # an April contract are different contracts.
    "match_volume": "quantity",
    "placement_dispersion": "dispersion",
    "format_dispersion": "dispersion",
    # A signed average of a bounded per match number, scale invariant in the
    # window length, so a rate in the sense that matters even though it can be
    # negative.
    "score_margin": "rate",
}


def metric_ref(metric: str, subject: str, **filters):
    """Build a MetricRef with the metric's declared bounds already attached.

    Preferred over constructing one by hand, precisely because the bounds are
    both easy to get wrong and load-bearing for collateral. The ``modes`` filter
    carries format names in this world, so
    ``metric_ref("win_rate", "VANTA", modes=("solo",))`` is a contract on the
    elimination mode and nothing else.
    """
    from arena.contracts.underlying import MetricRef

    if metric not in METRIC_BOUNDS:
        raise KeyError(
            f"{metric!r} has no declared bounds; add it to METRIC_BOUNDS so that "
            "contracts written on it can size collateral correctly"
        )
    return MetricRef(
        metric=metric,
        subject=subject,
        bounds=METRIC_BOUNDS[metric],
        kind=METRIC_KINDS.get(metric, "rate"),
        **filters,
    )
