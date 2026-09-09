"""The circuit world's settlement oracle, and the calendar that makes it one.

The Brawl world collects matches and looks them up. This world generates them,
so the question a contract asks changes shape: "what happened between these two
dates" has no answer until something decides which matches those dates name.
That decision is the calendar below, and it is the whole of the mapping.

**A window is a half-open run of match ids.** Match ``n`` occupies the slot
starting at ``epoch + n * period``, and a window ``[start, end)`` holds exactly
the matches whose slot starts inside it. Both endpoints round the same way, up,
so consecutive windows tile the season without a match landing in two
settlements or falling between them. That is the same reason
:class:`~arena.contracts.spec.ObservationWindow` is half-open in the first
place, carried through to the one place it could have been lost.

**The oracle is a pure function of (seed, window, metric, subject).** The seed
fixes the season, the window fixes the match ids, the metric fixes the
arithmetic and the subject fixes whose record it runs over. Nothing reads a
clock, nothing accumulates between calls, and the only state the oracle holds
is a memo of matches it has already played, which cannot change an answer
because :func:`arena.worlds.circuit.match.play` is itself pure. Two runs of the
same seed therefore settle identically, which is the property every replay,
every paired experiment and every dispute rests on.

**The oracle cannot see latent strength, and that is structural rather than
observed.** This module does not import ``roster``, so
:func:`~arena.worlds.circuit.roster.draw_strengths` is not in its namespace and
cannot be reached from it; the metrics it dispatches to take
``(results, subject)`` and have no argument a strength could arrive through;
and :class:`~arena.worlds.circuit.match.MatchResult` carries no field holding
one. The oracle reports what matches did. If it could read the parameter that
generated them, every forecasting result measured on this world would be a
result about arithmetic rather than about aggregation, and the exchange would
be pricing a lookup.

**On the reference.** A generated world has no crawl to pin and no snapshot to
freeze, but the settlement engine still needs an identity to check a contract
against and a date to check for lookahead. Both fall out of the calendar: the
identity is a digest over the seed, the calendar and the shape of every
registered format, so a contract pinned to one season can never be settled by
an oracle configured for another; and the date is the epoch, because the
generator was fixed when the circuit opened and every window this oracle will
answer begins at or after that moment. The engine's lookahead check and the
oracle's own refusal of a pre-epoch window are therefore two statements of one
fact, and neither can fire without the other being wrong.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from arena.contracts.spec import ObservationWindow
from arena.contracts.underlying import ALL, MetricRef
from arena.determinism import digest
from arena.settlement.oracle import MetricResolution, MetricUnavailable, SourceRef
from arena.worlds.circuit.match import MatchResult, play
from arena.worlds.circuit.metrics import (
    MAX_WINDOW_MATCHES,
    METRICS,
    InsufficientEvidence,
)
from arena.worlds.circuit.modes import FORMATS

__all__ = ["MatchCalendar", "CALENDAR", "CircuitOracle"]

UTC = timezone.utc


def _ceil_slots(span: timedelta, period: timedelta) -> int:
    """``span / period`` rounded up, in exact integer arithmetic.

    Timedelta floor division is exact on microseconds, so this never sees a
    float. A window boundary a microsecond after a slot start must not round
    down onto that slot, and dividing two timedeltas as floats can decide that
    question wrongly for a window far enough from the epoch.
    """
    return -((-span) // period)


@dataclass(frozen=True, slots=True)
class MatchCalendar:
    """When each match id is played. The only bridge between dates and matches.

    Two constants and nothing else, because every additional degree of freedom
    here is a way for two participants to disagree about which matches a
    contract named. A calendar is part of the oracle's reference identity, so a
    contract cannot be settled by an oracle running a different one.
    """

    epoch: datetime
    period: timedelta

    def __post_init__(self) -> None:
        if self.epoch.tzinfo is None or self.epoch.utcoffset() != timedelta(0):
            raise ValueError("calendar epoch must be timezone-aware UTC")
        if self.period <= timedelta(0):
            raise ValueError("calendar period must be positive")

    @property
    def per_day(self) -> float:
        return timedelta(days=1) / self.period

    def slot_start(self, match_id: int) -> datetime:
        """When match ``match_id`` is played."""
        if match_id < 0:
            raise ValueError("match ids start at zero, when the circuit opened")
        return self.epoch + self.period * match_id

    def match_ids(self, window: ObservationWindow) -> range:
        """The half-open run of match ids a window names.

        Rounding both endpoints up is what makes windows tile. Match ``n`` is
        in ``[start, end)`` exactly when ``start <= slot_start(n) < end``, so the
        first id is the smallest slot at or after ``start`` and the stop is the
        smallest slot at or after ``end``, which is the first id belonging to
        the next window.

        Raises :class:`ValueError` for a window this calendar cannot describe:
        one opening before the circuit did, or one longer than the ceiling
        ``match_volume`` declares its bound against.
        """
        if window.start < self.epoch:
            raise ValueError(
                f"window opens at {window.start.isoformat()}, before the circuit "
                f"did at {self.epoch.isoformat()}; the matches it names do not "
                "exist and shortening it silently would settle a volume contract "
                "against a season that never ran"
            )
        first = _ceil_slots(window.start - self.epoch, self.period)
        stop = _ceil_slots(window.end - self.epoch, self.period)
        if stop - first > MAX_WINDOW_MATCHES:
            raise ValueError(
                f"window spans {stop - first} matches, more than the "
                f"{MAX_WINDOW_MATCHES} a settlement may scan. That ceiling is what "
                "makes the declared bound on match_volume a fact rather than a "
                "hope, so widening it means re-deriving the bound"
            )
        return range(first, stop)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch.isoformat().replace("+00:00", "Z"),
            "period_seconds": int(self.period.total_seconds()),
        }


# Matches run back to back from the first of September 2026, five minutes each.
#
# **These are not new numbers.** ``market/match_book.py`` already lists per match
# contracts against exactly this schedule, with the same epoch and the same
# duration, and it maps a match id forward to the minutes it occupies where this
# maps a window back to the ids inside it. Choosing a second cadence here would
# have produced two clocks that never meet, which is the failure that left the
# venue's own lifecycle check running zero times while passing its tests: a
# match id would name one moment to the listing layer and another to
# settlement, and a contract listed on match 400 would settle against whatever
# match 400 meant to somebody else. `test_circuit_world` asserts the two agree,
# so the pair cannot drift apart quietly.
#
# The cadence is also the right one for an aggregate. At 288 matches a day a one
# week window runs 2,016 of them, in which a solo competitor appears 1,657 to
# 1,710 times, so the standard error on a 0.100 win rate is 0.0073 against a
# cross sectional spread of 0.036 to 0.178. A weekly contract is therefore a
# measurement rather than mostly noise.
CALENDAR = MatchCalendar(
    epoch=datetime(2026, 9, 1, tzinfo=UTC),
    period=timedelta(minutes=5),
)

# How many played matches the memo holds before it is emptied. A memo cannot
# change an answer, because `play` is pure in (seed, match_id, format_name), so
# the only thing at stake is memory: at the thirty day ceiling across two
# formats a full scan is 17,280 results, and holding a few of those windows at
# once is the point of caching at all. Emptying wholesale rather than evicting
# by age keeps the policy something a reader can hold in their head.
_MEMO_LIMIT = 64_000


class CircuitOracle:
    """Resolves circuit metrics by replaying the matches a window names."""

    def __init__(
        self,
        seed: int,
        *,
        calendar: MatchCalendar = CALENDAR,
        min_appearances: int = 0,
    ) -> None:
        self._seed = int(seed)
        self._calendar = calendar
        # The evidential bar applied inside the metric. The engine separately
        # enforces the contract's min_sample_size on the finished number, which
        # is the one bar that genuinely is a post-hoc check. This one is not: a
        # dispersion over a single appearance is exactly 0.0 and well formed,
        # so the metric has to refuse it rather than hand back a number the
        # engine would happily settle.
        self._min_appearances = int(min_appearances)
        self._memo: dict[tuple[str, int], MatchResult] = {}

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def calendar(self) -> MatchCalendar:
        return self._calendar

    @property
    def reference_id(self) -> str:
        """Identity of the season this oracle answers for.

        A digest over the seed, the calendar and the declared shape of every
        registered format, so a contract pinned to one season cannot be settled
        by an oracle configured for another, and registering a format that
        changes what a match is moves the identity rather than quietly changing
        the answers. The roster is deliberately not in it: reading the roster
        would mean importing the module that draws strengths, and the identity
        is not worth that.
        """
        return "circuit-" + digest(
            {
                "seed": self._seed,
                "calendar": self._calendar.to_dict(),
                "formats": [
                    {
                        "name": name,
                        "entrants": FORMATS[name].entrants,
                        "team_size": FORMATS[name].team_size,
                        "teams": FORMATS[name].teams,
                    }
                    for name in sorted(FORMATS)
                ],
            }
        )[7:23]

    @property
    def reference_as_of(self) -> datetime:
        """When the generator was fixed, which is when the circuit opened.

        The season is a function of the seed alone, so nothing about it was
        estimated from inside any window it settles. Returning the epoch states
        that in the terms the engine checks, and the calendar's refusal of a
        pre-epoch window means the engine's lookahead guard can never fire on a
        window this oracle would answer.
        """
        return self._calendar.epoch

    def resolve(
        self,
        ref: MetricRef,
        window: ObservationWindow,
        policy_overrides: Mapping[str, Any] | None = None,
    ) -> MetricResolution:
        try:
            metric = METRICS[ref.metric]
        except KeyError:
            raise MetricUnavailable(
                ref,
                "unknown metric",
                f"{ref.metric!r} is not one of {sorted(METRICS)}",
            ) from None

        # `maps` and `trophy_buckets` describe the Brawl world's strata and have
        # no meaning to an aggregate over circuit matches. A contract that sets
        # one is asking for a narrowing this oracle cannot perform, and
        # honouring it as a no-op would settle something wider than the contract
        # named. Refusing is the difference between a filter that is absent and
        # a filter that is ignored.
        #
        # `market/match_book.py` repurposes `maps` to pin a match id on its per
        # match contracts, and those are resolved by its own `metric_levels`
        # against a single result rather than here. So a ref carrying one is not
        # malformed, it is addressed to the other resolver, and the message says
        # so rather than reading as a bug in the contract.
        for unsupported in ("maps", "trophy_buckets"):
            values = getattr(ref, unsupported)
            if values != (ALL,):
                raise MetricUnavailable(
                    ref,
                    f"{unsupported} filter is not meaningful in this world",
                    f"{ref.metric!r} was asked for {list(values)}, but a circuit "
                    "match has no such dimension to narrow on and settling it "
                    "unfiltered would measure something wider than the contract "
                    "named. A per match ref pinned this way is settled by "
                    "market.match_book.metric_levels against its own result, "
                    "not by this window oracle",
                )

        formats = sorted(FORMATS) if ref.modes == (ALL,) else list(ref.modes)
        unknown = [name for name in formats if name not in FORMATS]
        if unknown:
            raise MetricUnavailable(
                ref,
                "unknown format",
                f"{unknown} are not registered; {sorted(FORMATS)} are. The modes "
                "filter carries format names in this world",
            )

        try:
            ids = self._calendar.match_ids(window)
        except ValueError as bad_window:
            raise MetricUnavailable(
                ref, "window cannot be mapped to matches", str(bad_window)
            ) from None

        if not ids:
            raise MetricUnavailable(
                ref,
                "no matches in window",
                f"{window.start.isoformat()} to {window.end.isoformat()} falls "
                f"between scheduled matches at one every "
                f"{int(self._calendar.period.total_seconds())} seconds",
            )

        results = self._play(formats, ids)

        overrides = policy_overrides or {}
        try:
            outcome = metric(
                results,
                ref.subject,
                min_appearances=int(
                    overrides.get("min_appearances", self._min_appearances)
                ),
            )
        except InsufficientEvidence as thin:
            raise MetricUnavailable(ref, thin.reason, thin.detail) from None

        return MetricResolution(
            ref=ref,
            value=outcome.value,
            sample_size=outcome.sample_size,
            sources=tuple(
                SourceRef(
                    source_id=f"circuit:{self._seed}:{name}",
                    # Content address of the evidence rather than of a file: the
                    # season is generated, so what identifies it is the seed, the
                    # format and the exact run of match ids that were replayed.
                    # Two runs agreeing on this digest replayed the same matches.
                    digest=digest(
                        {
                            "seed": self._seed,
                            "format": name,
                            "first_match": ids.start,
                            "stop_match": ids.stop,
                        }
                    ),
                    rows=len(ids),
                )
                for name in formats
            ),
            diagnostics=(
                *outcome.diagnostics,
                ("window_start", window.start.isoformat()),
                ("window_end", window.end.isoformat()),
                ("first_match", ids.start),
                ("stop_match", ids.stop),
                ("matches_scanned", len(ids) * len(formats)),
                ("formats_requested", "|".join(formats)),
                ("calendar_period_seconds", int(self._calendar.period.total_seconds())),
                ("reference_id", self.reference_id),
            ),
        )

    def _play(self, formats: Sequence[str], ids: range) -> tuple[MatchResult, ...]:
        """Every match in the run, for every format asked for, memoized.

        Built in a fixed order, format by format and then by match id, so the
        sequence handed to a metric never depends on dictionary iteration or on
        what happened to be cached.
        """
        if len(self._memo) > _MEMO_LIMIT:
            self._memo.clear()
        played: list[MatchResult] = []
        for name in formats:
            for match_id in ids:
                key = (name, match_id)
                result = self._memo.get(key)
                if result is None:
                    result = play(self._seed, match_id, name)
                    self._memo[key] = result
                played.append(result)
        return tuple(played)
