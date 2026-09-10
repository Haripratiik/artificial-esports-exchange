"""The clock a contract's window is measured against.

There are two clocks in this simulator and they were never connected, which is
why nothing ever settled in a running server.

The kernel counts simulated nanoseconds from zero. A contract's window closes
on a *calendar date* (`2026-09-28` for the listings in `build_market`). So
`Venue._enforce_lifecycle` asks `self._clock() >= instrument.expiry`, and in
the live market `_clock` was `None`, which means the question was never asked.
Even wired to a wall clock it would not have helped: the kernel would have had
to run for a month of real time to reach the date.

Measured before this existed: after a simulated hour every one of the 47 listed
contracts was still `continuous` and the settled set was empty. Positions were
marked forever and realised never, so a systematic trader's P&L had no terminal
event to resolve against. The settlement machinery was complete and heavily
tested; it simply was not reachable from the live path.

**The mapping is the one this venue already uses.** `build_market` computes
``scale = session_seconds / trading_day`` with a 6.5 hour trading day, and
applies it to the circuit breaker's limit-state and pause windows. So the venue
already treats `session_seconds` of simulated time as one trading day. This
extends the same statement to the contract calendar rather than inventing a
second notion of time: one trading day of contract life costs
``session_seconds`` of simulated time, and a four-week window therefore runs its
course in twenty-eight of them.

That is a compression, and it is stated rather than hidden. Nothing about the
settlement arithmetic changes: the oracle answers from the same dataset over
the same window, and `settle()` returns exactly what it returns today when
`prior_levels` calls it. Only the moment the venue asks is different.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

__all__ = ["Calendar"]


@dataclass
class Calendar:
    """Maps simulated seconds onto contract time.

    Held as an object with a mutable cursor rather than a closure over the
    market, because the venue is constructed *before* the market that drives
    it: the venue needs `clock` at construction and the market needs the venue.
    A small mutable holder passed to both is the simplest thing that does not
    require either to know about the other.
    """

    start: datetime
    # Simulated seconds that pass for each day of contract time. Reusing
    # `session_seconds` keeps this consistent with the breaker windows, which
    # are already scaled against a 6.5 hour trading day.
    seconds_per_day: float
    _sim_seconds: float = field(default=0.0, repr=False)

    def advance_to(self, sim_seconds: float) -> None:
        """Move the cursor. Monotonic, because a clock that goes backwards
        would reopen a contract that had already closed."""
        if sim_seconds > self._sim_seconds:
            self._sim_seconds = sim_seconds

    def now(self) -> datetime:
        if self.seconds_per_day <= 0:
            return self.start
        return self.start + timedelta(days=self._sim_seconds / self.seconds_per_day)

    def seconds_until(self, moment: datetime) -> float:
        """Simulated seconds from here to a calendar moment, for reporting.

        Published so an operator or a client can answer "when does this
        expire" in the unit they are actually waiting in, rather than being
        handed a date and left to do the conversion.
        """
        remaining_days = (moment - self.now()).total_seconds() / 86_400.0
        return max(0.0, remaining_days * self.seconds_per_day)
