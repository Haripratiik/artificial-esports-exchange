"""Shared fixtures, all of them about the one world this repository still has.

The world used to be a collected one, so this file handed out a dataset, a
frozen reference snapshot and a row builder for hand-made scenarios. None of
those exist any more: `worlds/circuit` draws a season from a seed, so the
evidence behind a settlement is a run of match ids rather than a crawl, and the
thing a test needs to be given is the seed and a window the calendar can map.

Two consequences worth stating, because they are why the fixtures below are so
much smaller than the ones they replace.

The window opens on the calendar's own epoch. The oracle's reference is as of
that moment, and the calendar refuses a window that opens before the circuit
did, so a window starting anywhere earlier fails the engine's lookahead guard
rather than measuring thin evidence.

The policy carries `min_sample_size` and nothing else. `min_stratum_battles`
and `min_strata_coverage` guard the *composition* of a standardized rate over
strata, and a circuit metric refuses a window that mixes formats outright
rather than reweighting it, so there is no composition here for either to
guard. A bar whose input is never wired is a bar that silently does not exist,
which is the sixth bug class in `CONTRIBUTING.md`, so they are left at zero
rather than set to a number that reads as a check.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from arena.contracts.spec import DataPolicy, ObservationWindow
from arena.worlds.circuit.oracle import CALENDAR, CircuitOracle

# The season every fixture here answers for. The same integer
# `dashboard.build_market` lists against, so a test that reaches for both gets
# one world rather than two.
SEED = 7


@pytest.fixture
def policy() -> DataPolicy:
    """A real bar rather than a formality.

    A four week window holds 8,064 matches and seats a solo competitor in about
    6,700 of them, so 1,000 appearances is a bar a genuinely absent subject
    fails and a thin draw never does.
    """
    return DataPolicy(min_sample_size=1_000)


@pytest.fixture
def window() -> ObservationWindow:
    """The first four weeks of the season, opening when the circuit did."""
    return ObservationWindow(
        start=CALENDAR.epoch, end=CALENDAR.epoch + timedelta(weeks=4)
    )


@pytest.fixture
def published_at(window):
    return window.start - timedelta(days=1)


@pytest.fixture
def oracle() -> CircuitOracle:
    return CircuitOracle(SEED)
