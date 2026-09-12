"""Assemble the live market the dashboard serves.

The instruments settle against the circuit world, which generates its matches
from the world seed rather than collecting them, and the fundamental agents are
told what each will actually settle at, then given a *noisy* view of it, with a
different precision each. So the market has a true value to converge toward,
and whether it gets there is something you can watch rather than something
asserted.

Latency is heterogeneous on purpose. The market maker is effectively colocated,
the funds are a few milliseconds out, the noise traders are far away, and the
human is somewhere in between. That is the same configuration the research
experiments use; the dashboard is not a special case of the simulator, it is the
simulator with a browser attached.

**Why this listing is shaped the way it is.** The one it replaces carried 47
instruments of which 38 were written on the same four win rates and 20 of those
on a single one, so pricing one priced most of them and the exchange was four
numbers wearing hats. The cause was not carelessness about strikes, it was that
the world underneath carried one number per subject. This world carries one
latent scalar per competitor per format, and the consequence is measurable: over
the four week contract window on seed 7 a competitor's individual win rate ranks
the twelve competitor field almost exactly as its eliminations per match does,
Spearman +0.993, and the team pair agrees at +0.979. Listing a second strike on
a win rate therefore adds an instrument and no information.

The axes that do carry different information were measured over the same window
and the same field. The two formats rank the field at Spearman -0.378, so a
competitor strong in the elimination mode is not thereby strong in the team
mode. Scheduling volume ranks at -0.084 against the solo win rate and cannot
depend on skill at all, because the field for each match is drawn without
reading a strength. And the second moments are not restatements of the first:
placement dispersion correlates with the solo win rate at Pearson -0.908 but
keeps a residual of 0.0068 after a straight line in it, and format dispersion
sits at -0.271 against the solo win rate and +0.607 against the objective one,
so it is a restatement of neither.

So the listing spends its instruments on subjects and axes rather than on
strikes. 50 contracts across 22 distinct underlyings, against 47 across 10
before, and the largest cluster on any one underlying is 7 where the old
listing's heaviest win rate carried 20.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from functools import lru_cache

from arena.agents.arbitrageur import Arbitrageur
from arena.agents.flow import FlowTrader
from arena.agents.fundamental import FundamentalTrader
from arena.agents.market_maker import MarketMaker
from arena.agents.noise import NoiseTrader
from arena.agents.surface import SurfaceMarketMaker
from arena.contracts.payoff import Binary, Call, Linear, Put
from arena.contracts.spec import (
    ContractSpec,
    DataPolicy,
    DistributionSchedule,
    ObservationWindow,
)
from arena.contracts.underlying import Basket, Difference, Single, Underlying
from arena.exchange.types import AgentId
from arena.market.calendar import Calendar
from arena.market.instrument import Instrument, InstrumentClass
from arena.market.live import HUMAN_ID, VENUE_ID, HumanAgent, LiveMarket
from arena.market.fees import FREE, MAKER_TAKER, FeeSchedule
from arena.agents.match_maker import MatchMaker
from arena.market.match_operator import MatchOperator
from arena.market.operator import SessionOperator
from arena.market.lmsr_venue import LmsrVenue
from arena.market.venue import Venue
from arena.market.venue_agent import VenueAgent
from arena.settlement.engine import distributions, settle
from arena.sim.kernel import Kernel
from arena.sim.latency import PairwiseLatency
from arena.sim.time import micros, millis, seconds
from arena.worlds.circuit.metrics import metric_ref
from arena.worlds.circuit.modes import FORMATS
from arena.worlds.circuit.oracle import CALENDAR, CircuitOracle

# The competitor list, and nothing else from that module. `draw_strengths` is
# deliberately not imported: a listing has to know who competes, and any
# listing decision that read a latent strength would be writing the answer into
# the contract. Every threshold and strike below is derived from a format's own
# arithmetic or from the calendar, never from a settlement value.
from arena.worlds.circuit.roster import ROSTER

# Which season the statistical contracts settle against. The same integer
# `build` hands the kernel and the match operator, because a listing on one
# season and matches from another would be two different worlds sharing an
# exchange: `RIFT_SOLO_WR` and `SOLO0_WIN_RIFT` would be about different
# competitors and nothing would say so.
WORLD_SEED = 7

# What a person opens the exchange with. Large enough to take a real position
# in the futures, which settle between roughly 600 and 6,200 a contract, and
# small enough that a profit is a number you can see.
HUMAN_STARTING_CASH = 250_000

# The widest maker's mandate: how much it shows on each side of each book, and
# how much of any one contract it will carry. The other makers are scaled down
# from these, and `maker_capital` funds each one from its own pair, so the
# three numbers that decide what a maker is asked to do live in one place as
# the number that decides whether it can afford to do it.
MAKER_QUOTE_SIZE = 30
MAKER_POSITION_LIMIT = 1_200

# Which registered format plays which role, read off the registry rather than
# typed. It is not tidiness: `placement_dispersion` refuses a format that
# places a whole side together, and `score_margin` refuses one that keeps no
# score, so the two families below have to name the format that supports them
# and a mode registered later has to land in the right one on its own.
INDIVIDUAL = next(
    name for name, fmt in sorted(FORMATS.items()) if fmt.team_size == 1
)
TEAM = next(name for name, fmt in sorted(FORMATS.items()) if fmt.team_size > 1)

# A rate onto a price grid. A win rate of 0.1398 at this scale settles at 1,398,
# so one tick of 0.25 is a quotable amount rather than a rounding artifact.
RATE_SCALE = 10_000.0
# Eliminations per match run to about 2.0 with a ceiling of 9, so the same
# scale would put a typical contract at 14,000 inside a range of 90,000 and
# charge a maker ten times the collateral the contract needs.
ELIM_SCALE = 1_000.0
# What one week of the share pays per unit of win rate. Four of these add to
# 4,000, which is 0.4 of what the four week future pays, and the two are listed
# side by side so that relation can be watched.
WEEKLY_SCALE = 1_000.0

# Four weeks either side of the boundary between what a participant has seen
# and what the contracts are about.
#
# The circuit opened on the calendar's epoch and has run a match every five
# minutes since, so a window is a run of match ids rather than a query against a
# crawl. The first four weeks of that season are the record everyone can look
# at; the contracts observe the four weeks after it. That makes the prior below
# a measurement rather than a fixture, which it could not be if the contract
# window opened on the epoch. Measured: re-dated onto the four weeks before the
# epoch, all six subjects fail, on `ReferenceLookahead` first because the
# oracle's reference is as of the epoch, and on the calendar's own refusal of a
# pre-epoch window behind it. Both say the same thing, that those matches were
# never played, and `prior_levels` catches either, so every informed trader
# would have opened on nothing.
WEEKS = 4
PRIOR_WINDOW = ObservationWindow(
    CALENDAR.epoch, CALENDAR.epoch + timedelta(weeks=WEEKS)
)
WINDOW = ObservationWindow(
    PRIOR_WINDOW.end, PRIOR_WINDOW.end + timedelta(weeks=WEEKS)
)

# When the listing went up: the day before the circuit opened, so no contract
# here was written after any evidence about it existed. The venue's own clock
# starts here too, for a reason `build` sets out where it constructs it.
PUBLISHED_AT = CALENDAR.epoch - timedelta(days=1)

# Weekly delivery, the way a commodity is listed: the same deliverable across
# consecutive windows, so the curve across them is a forward curve rather than
# a set of unrelated contracts. They tile the observation window exactly, which
# is what makes the share below the sum of its legs rather than roughly it.
DELIVERY_WEEKS = [
    ObservationWindow(
        WINDOW.start + timedelta(weeks=n), WINDOW.start + timedelta(weeks=n + 1)
    )
    for n in range(WEEKS)
]

# How many appearances a week of window has to produce before a contract on it
# will settle.
#
# Read off the thinnest family rather than chosen round. A week holds 2,016
# matches, and the team format seats 6 of the 12 competitors, so a subject
# expects 1,008 appearances in it with a standard deviation of 22.4. A bar of
# 250 is a quarter of that and roughly thirty-four standard deviations below it,
# so it voids a subject the schedule genuinely never drew and never a thin draw.
# It is also a real bar rather than a formality: over the four week window it
# comes to 1,000 appearances, at which the standard error on a 0.5 win rate is
# 0.0158 against a cross sectional spread of 0.192 between the best and worst
# competitor, so a settlement that clears it is a measurement.
WEEKLY_SAMPLE_BAR = 250

# How far either side of a format's neutral point the listed strikes and rungs
# sit, in the units of a win rate. Four percentage points, a round number in the
# units of the metric and not read off any settlement. What it produced, once
# measured: at the individual format's rungs of 0.060, 0.100 and 0.140 the field
# splits 11 of 12, 5 of 12 and 1 of 12, so no rung on the board is a formality
# and none is a coin toss for every subject at once.
BAND = 0.04

def _one_per_archetype() -> tuple[str, ...]:
    """The first competitor the roster lists for each archetype.

    A subset rather than all twelve, and chosen by a rule rather than by who
    looks interesting. Archetype is the systematic reason the two formats
    disagree: the roster gives a support competitor a negative fit in the
    elimination mode and the largest positive fit in the team mode, so covering
    every archetype covers the axis the -0.378 comes from. Measured on the
    contract window, these six span 0.061 to 0.190 of the field's 0.038 to
    0.190 in the individual format and 0.441 to 0.622 of its 0.430 to 0.622 in
    the team format, so the rule buys nearly the whole cross section at half the
    subjects.
    """
    seen: set[str] = set()
    chosen: list[str] = []
    for competitor in ROSTER:
        if competitor.archetype in seen:
            continue
        seen.add(competitor.archetype)
        chosen.append(competitor.key)
    return tuple(chosen)


SUBJECTS: tuple[str, ...] = _one_per_archetype()

# Each family sits on a different competitor, taken in roster order and
# wrapping once. Nothing about a competitor earns it a family; the point is
# that no one subject carries the board, which is exactly what went wrong
# before. The largest cluster this leaves on a single underlying is 7 of 50
# contracts, against 20 of 47 on the old listing's heaviest win rate.
LADDER, CHAIN, SHARE, TEAM_CHAIN, SPECIALIST, VOLUME = SUBJECTS
ELIM_SUBJECT, PLACE_A, PLACE_B = SUBJECTS[0], SUBJECTS[1], SUBJECTS[2]


# Which seed produced a given reference id, filled in by `reference_id` itself.
#
# A memo of a pure function rather than mutable state: the id is a digest over
# the seed, the calendar and the registered formats, so nothing can change which
# seed produced one and an entry here can never go stale the way a captured
# account id can. It exists because a contract pins the digest and not the seed,
# so a caller holding a contract has no other way to ask for the oracle that can
# answer for it. Without it, every caller that does not already know the season
# would fall back to the default one, the engine would refuse the mismatch, and
# a dashboard running any seed but 7 would quietly lose its settlement column
# while every book on the page traded normally.
_SEASON_OF: dict[str, int] = {}


@lru_cache(maxsize=8)
def reference_id(seed: int) -> str:
    """Identity of the season a contract pins itself to.

    A function of the seed rather than a constant, because in a generated world
    the season *is* the seed: the oracle digests the seed, the calendar and the
    shape of every registered format into this string and the settlement engine
    refuses a contract whose reference does not match. So a contract listed on
    one season can never be settled by an oracle running another, which is the
    same protection the frozen snapshot id gave a collected world.
    """
    identity = CircuitOracle(seed).reference_id
    _SEASON_OF[identity] = seed
    return identity


def policy_for(window: ObservationWindow) -> DataPolicy:
    """Scale the evidence bar to the length of the window.

    A one week delivery window holds a quarter of the matches a four week
    observation window does, so holding both to the same minimum would void
    every weekly contract for thin evidence that is not actually thin, it is one
    week's worth.

    The stratum fields are left at zero deliberately. They guard the
    *composition* of a standardized rate over strata, and this world has no
    strata: a circuit metric refuses a window that mixes formats outright rather
    than reweighting it, so a contract naming a stratum bar would be declaring a
    check nothing can run. A guard whose input is never wired is a guard that
    silently does not exist.
    """
    weeks = max(1, round((window.end - window.start).days / 7))
    return DataPolicy(min_sample_size=WEEKLY_SAMPLE_BAR * weeks)


def neutral_level(format_name: str) -> float:
    """The win rate an interchangeable competitor posts in this format.

    Winning slots over all slots, which is 1 of 10 in the elimination mode and 3
    of 6 in the team mode. Derived from the registry rather than written down,
    so every strike and rung below moves with a format instead of having to be
    revisited when one changes. It is the one level about a format that is known
    before a single match is played, which is what makes it the honest place to
    hang a ladder: listing rungs around where a metric is known to settle is
    listing the answer.
    """
    fmt = FORMATS[format_name]
    winning_slots = 1 if fmt.team_size == 1 else fmt.team_size
    return winning_slots / fmt.entrants


def expected_appearances(window: ObservationWindow, format_name: str) -> float:
    """How many of a window's matches a competitor expects to be drawn into.

    Matches in the window times the field size over the roster size, because
    `roster_for` samples the field without replacement and without reading a
    strength. So this is the level a scheduling contract sits at for every
    competitor alike, and the only thing that moves a settlement away from it is
    the draw. Over one delivery week it comes to 1,680 in the individual format
    with a standard deviation of 16.7, and the four subjects measured came in at
    1,665, 1,676, 1,683 and 1,686.
    """
    return (
        len(CALENDAR.match_ids(window))
        * FORMATS[format_name].entrants
        / len(ROSTER)
    )


def _wr(subject: str, format_name: str = INDIVIDUAL) -> Single:
    """A subject's win rate in one format.

    The format is named rather than left open because the metric refuses a
    pooled average and says why at length: a win in the ten way elimination sits
    against a neutral point of 0.100 and a win in the three a side objective
    against 0.500, so an average over a window holding both would move with
    whatever mix the schedule happened to run.
    """
    return Single(metric_ref("win_rate", subject, modes=(format_name,)))


def _spec(
    contract_id: str,
    underlying: Underlying,
    payoff,
    tick: str = "0.25",
    window: ObservationWindow | None = None,
    distribution: DistributionSchedule | None = None,
    tick_table: tuple[tuple[str, str], ...] = (),
    seed: int = WORLD_SEED,
) -> ContractSpec:
    """A contract on the default window unless one is given.

    The window is a parameter because a commodity needs it to be. Every rate
    contract here measures the same four weeks, so the window would have been a
    constant, but the whole point of a delivery month is that the amount
    delivered in one is a different thing from the amount delivered in the next,
    and a term structure is a set of contracts that differ in nothing else.
    """
    measured = window or WINDOW
    return ContractSpec(
        contract_id=contract_id,
        underlying=underlying,
        payoff=payoff,
        window=measured,
        policy=policy_for(measured),
        reference_id=reference_id(seed),
        published_at=PUBLISHED_AT,
        tick_size=tick,
        tick_table=tick_table,
        distribution=distribution,
    )


def _listed(symbol: str, underlying: Underlying, payoff, **spec_kwargs) -> Instrument:
    """One tradeable symbol and the contract behind it, named once."""
    return Instrument(symbol, _spec(symbol, underlying, payoff, **spec_kwargs))


def instruments(seed: int = WORLD_SEED) -> list[Instrument]:
    """Every statistical contract the exchange lists, for one season.

    Nine asset classes, because the venue's claim is nine on one matching
    engine, and 22 distinct underlyings across 50 contracts, because the claim
    the old listing could not make is that pricing one of them does not price
    the rest.
    """
    listed: list[Instrument] = []

    def add(symbol: str, underlying: Underlying, payoff, **kwargs) -> None:
        listed.append(_listed(symbol, underlying, payoff, seed=seed, **kwargs))

    # ── futures on the two axes the world actually carries ────────────────
    #
    # Twelve futures on twelve distinct underlyings, which is the whole of the
    # answer to what went wrong before. The two formats rank this field at
    # Spearman -0.378 over the contract window, so the pair on one competitor
    # is two questions rather than one asked twice, and a trader who has priced
    # the elimination mode has priced roughly none of the team mode.
    #
    # One of them carries a tiered tick, so the rule is exercised rather than
    # merely available, and it is the team one because that is where the coarse
    # increment is reachable: a team win rate settles near 5,000 and an
    # individual one near 1,100, so a table thresholded at 4,000 would never
    # fire on the individual side. A quarter of a point below 4,000 and a whole
    # point above it: fine enough at the bottom for a spread to narrow to what
    # the market knows, coarse enough at the top that a resting order cannot be
    # stepped in front of for a rounding error.
    for subject in SUBJECTS:
        add(
            f"{subject}_{INDIVIDUAL.upper()}_WR",
            _wr(subject, INDIVIDUAL),
            Linear(RATE_SCALE),
        )
    for n, subject in enumerate(SUBJECTS):
        add(
            f"{subject}_{TEAM.upper()}_WR",
            _wr(subject, TEAM),
            Linear(RATE_SCALE),
            tick_table=(("4000.00", "1.00"),) if n == 0 else (),
        )

    # ── the same competitor's eliminations, in both formats ───────────────
    #
    # Listed knowing exactly what it does not add. Within a format this ranks
    # the field as the win rate does, Spearman +0.993 in the elimination mode
    # and +0.979 in the team mode, so it carries no cross sectional information
    # and nobody should rediscover that by trading it. What it does carry is a
    # different number with a different variance, and an over-under on a count
    # is a bet a winner contract cannot express at all. The pair also states the
    # cross format axis on a second metric: an elimination in the solo mode is a
    # knockout and in the team mode it is a point scored, which is two different
    # economies wearing one metric's name.
    for format_name in (INDIVIDUAL, TEAM):
        add(
            f"{ELIM_SUBJECT}_{format_name.upper()}_ELIM",
            Single(
                metric_ref(
                    "eliminations_per_match",
                    ELIM_SUBJECT,
                    modes=(format_name,),
                )
            ),
            Linear(ELIM_SCALE),
        )

    # ── the signed one ────────────────────────────────────────────────────
    #
    # Average points margin, normalized by the target, so a clean sweep is +1.0
    # and a loss by one point in a race to three is -1/3. It restates the team
    # win rate in the cross section and the metric says so: measured over the
    # contract window the two correlate at Pearson +0.998. It is listed anyway
    # for the two things that are not restatements. Its range is signed, so a
    # short posts collateral against a different interval from anything else
    # here, and the metric records a per match standard deviation of 0.687
    # against a win indicator's 0.500, so over a short window the two settle
    # with materially different precision on the same question.
    add(
        f"{SPECIALIST}_{TEAM.upper()}_MARGIN",
        Single(metric_ref("score_margin", SPECIALIST, modes=(TEAM,))),
        Linear(RATE_SCALE),
    )

    # ── second moments ────────────────────────────────────────────────────
    #
    # A claim on how unevenly a competitor performs rather than on how well. Two
    # competitors with the same win rate and different spreads are not the same
    # thing to own: one finishes mid table every time, the other alternates
    # first and last, and only this contract can tell you which you are holding.
    #
    # Placement dispersion is measured in the individual format because the
    # metric refuses the other one, and the refusal is the interesting part:
    # where a whole side places together the spread of placements is
    # sqrt(w(1-w)) over the field size less one, exactly, so a team version
    # would be the win rate wearing a second name. Two subjects rather than one,
    # because there is a cross section here to hold: the residual after a
    # straight line in the win rate is 0.0068 on this window, against a sampling
    # floor of 0.30 over the square root of twice 6,720 appearances, which is
    # 0.0026. A factor of 2.6, and the metric records the same factor over its
    # own shorter run.
    for subject in (PLACE_A, PLACE_B):
        add(
            f"{subject}_{INDIVIDUAL.upper()}_DISP",
            Single(
                metric_ref(
                    "placement_dispersion", subject, modes=(INDIVIDUAL,)
                )
            ),
            Linear(RATE_SCALE),
        )

    # Format dispersion is the one metric that needs both formats to exist, and
    # the one that carries information neither format's win rate does: -0.271
    # against the individual win rate and +0.607 against the team one on this
    # window. A specialist prices high and an all rounder prices low, and
    # nothing else on the board is written on that distinction. Its declared
    # ceiling of 1.0 is roughly twelve times the largest value measured, which
    # is the correct direction to be wrong in, since settlement refuses a value
    # outside the range and an overestimate only costs collateral.
    add(
        f"{SPECIALIST}_FORMAT_DISP",
        Single(metric_ref("format_dispersion", SPECIALIST)),
        Linear(RATE_SCALE),
    )

    # ── event contracts ───────────────────────────────────────────────────
    #
    # A ladder, not a single threshold. One binary struck where a rate is known
    # to settle is a foregone conclusion, and a prediction market with no
    # uncertainty in it is a countdown. These rungs sit at the individual
    # format's neutral point and one band either side, so they are derived from
    # the format's own arithmetic before a match is played. Some will be near
    # certain and some genuinely open, and which is which is what the market is
    # for.
    individual_neutral = neutral_level(INDIVIDUAL)
    for threshold in (
        individual_neutral - BAND,
        individual_neutral,
        individual_neutral + BAND,
    ):
        add(
            f"{LADDER}_{INDIVIDUAL.upper()}_GT{round(threshold * 1000):03d}",
            _wr(LADDER, INDIVIDUAL),
            Binary(">", threshold, payout=1.0),
            tick="0.01",
        )

    # The team format's neutral point is 0.500 exactly, so "did this competitor
    # beat an interchangeable one" is a threshold the format hands you rather
    # than one anybody picked. Listed across every subject: six different
    # underlyings and one payoff, which is the opposite shape from six strikes
    # on one underlying. Declining to list it for the subjects that look decided
    # would be listing the answer, and on this window one of the six lands on
    # 0.500000 exactly, 2,024 wins from 4,048 appearances, and settles at
    # nothing because the comparison is strict, while another clears it by
    # 0.122.
    team_neutral = neutral_level(TEAM)
    for subject in SUBJECTS:
        add(
            f"{subject}_{TEAM.upper()}_GT{round(team_neutral * 1000):03d}",
            _wr(subject, TEAM),
            Binary(">", team_neutral, payout=1.0),
            tick="0.01",
        )

    # And one event contract that cannot be about skill at all. The field for
    # every match is drawn without replacement and without reading a strength,
    # so appearances are a pure draw around a level the schedule fixes: 2,016
    # matches in a delivery week seating 10 of 12 competitors is 1,680
    # appearances expected, with a standard deviation of 16.7. Rounded because
    # the metric is a count of matches and a fractional rung could never be
    # crossed by a fraction. It and the volume curve below are the only claims
    # here whose prices should not move on anything a trader learns about how
    # good a competitor is.
    volume_week = DELIVERY_WEEKS[0]
    volume_rung = round(expected_appearances(volume_week, INDIVIDUAL))
    add(
        f"{VOLUME}_{INDIVIDUAL.upper()}_VOL_W1_GT{volume_rung}",
        Single(metric_ref("match_volume", VOLUME, modes=(INDIVIDUAL,))),
        Binary(">", float(volume_rung), payout=1.0),
        tick="0.01",
        window=volume_week,
    )

    # ── options ───────────────────────────────────────────────────────────
    #
    # Payoffs on the same underlying as the future, so they settle from the same
    # metric at the same instant and need no separate machinery, and put-call
    # parity holds at the shared strike as an exact identity rather than to
    # within a discount factor.
    #
    # Strikes are laid on a grid around the format's neutral point rather than
    # around where the metric is known to settle. Calls take the rungs at and
    # below neutral, puts those at and above it, and both take the neutral rung,
    # so parity is exercised at a strike that exists on both sides and a chain
    # carries strikes either side of neutral whichever side the subject turns
    # out to sit. Two chains rather than one wide one, and on the two formats,
    # because a fourth strike on one win rate is an instrument and no
    # information where a chain on the other format is a different question.
    def strike_at(format_name: str, step: float) -> float:
        return round((neutral_level(format_name) + BAND * step) * RATE_SCALE, 2)

    for subject, format_name, call_steps, put_steps in (
        (CHAIN, INDIVIDUAL, (-1.0, -0.5, 0.0), (0.0, 0.5, 1.0)),
        (TEAM_CHAIN, TEAM, (-0.5, 0.0), (0.0, 0.5)),
    ):
        tag = format_name.upper()
        for step in call_steps:
            strike = strike_at(format_name, step)
            add(
                f"{subject}_{tag}_C{round(strike)}",
                _wr(subject, format_name),
                Call(strike, RATE_SCALE),
            )
        for step in put_steps:
            strike = strike_at(format_name, step)
            add(
                f"{subject}_{tag}_P{round(strike)}",
                _wr(subject, format_name),
                Put(strike, RATE_SCALE),
            )

    # ── the spread ────────────────────────────────────────────────────────
    #
    # One competitor's individual win rate less its team win rate, which is the
    # -0.378 written as a single tradeable number instead of inferred from two
    # books. It hedges out whatever moves both formats together and leaves the
    # thing that does not, and it is arbitrage linked to the two futures already
    # listed on its legs, so the relation is exact rather than approximate.
    add(
        f"{SPECIALIST}_FORMAT_SPD",
        Difference(_wr(SPECIALIST, INDIVIDUAL), _wr(SPECIALIST, TEAM)),
        Linear(RATE_SCALE),
    )

    # ── the index ─────────────────────────────────────────────────────────
    #
    # Equal weights over the individual win rates of every listed subject, which
    # makes it the field's level rather than any competitor's. Weights are
    # pinned in the spec and therefore covered by its digest, because an index
    # whose weights could be recomputed from data inside its own observation
    # window would leak the future into its own settlement. All six legs are
    # listed as futures, so the arbitrageur can form the index relation at all;
    # a basket priced against a subset of its components is a bet on the rest.
    add(
        f"CIRCUIT_{INDIVIDUAL.upper()}_IDX",
        Basket(
            tuple(
                (_wr(subject, INDIVIDUAL), 1.0 / len(SUBJECTS))
                for subject in SUBJECTS
            )
        ),
        Linear(RATE_SCALE),
    )

    # ── commodities ───────────────────────────────────────────────────────
    #
    # A claim on an amount rather than on a proportion: matches entered over one
    # delivery week. That makes the window part of the contract rather than a
    # detail, and gives the four of them a term structure with its own shape.
    #
    # **Each one names a format, and that is a bound and not a preference.**
    # `match_volume` is the only metric that may pool formats, because a count
    # of entries means the same thing whichever mode produced it, and its
    # declared ceiling is the 8,640 matches a window may hold. That ceiling is
    # derived for one format. Measured over the four week window on seed 7, a
    # pooled count comes to between 10,635 and 10,847 appearances across the
    # twelve competitor field, because two formats each seat the competitor.
    # Settlement refuses a value outside the declared range and raises rather
    # than voiding, so a pooled four week volume contract would not merely
    # mislead, it would fail loudly at expiry. Naming the individual format
    # holds the same subject to 6,700, comfortably inside.
    for n, week in enumerate(DELIVERY_WEEKS):
        add(
            f"{VOLUME}_{INDIVIDUAL.upper()}_VOL_W{n + 1}",
            Single(metric_ref("match_volume", VOLUME, modes=(INDIVIDUAL,))),
            Linear(1.0),
            window=week,
        )
    # A second subject on the front week, so the curve has a cross sectional
    # partner and the scheduling axis is two points rather than one.
    add(
        f"{LADDER}_{INDIVIDUAL.upper()}_VOL_W1",
        Single(metric_ref("match_volume", LADDER, modes=(INDIVIDUAL,))),
        Linear(1.0),
        window=DELIVERY_WEEKS[0],
    )

    # ── the weekly legs a share is made of ────────────────────────────────
    #
    # One future per delivery week, settling at 1,000 times that week's win
    # rate, which is precisely what the share pays at the end of that week. So
    # the share is the sum of these four exactly and not approximately: same
    # metric, same windows, same evidential bar, so the two sides of that
    # equation resolve from the same numbers.
    #
    # Listed for that reason. Without them the only relation available is "the
    # share is worth 0.4 times the four week future", which is *not* an
    # identity, since the four weekly rates are each appearance weighted and do
    # not average to the four week rate. The arbitrageur enforces identities
    # only, so an approximation it cannot tell from an identity is the dangerous
    # kind of relation to leave lying about.
    for n, week in enumerate(DELIVERY_WEEKS):
        add(
            f"{SHARE}_{INDIVIDUAL.upper()}_W{n + 1}",
            _wr(SHARE, INDIVIDUAL),
            Linear(WEEKLY_SCALE),
            window=week,
        )

    # ── the share ─────────────────────────────────────────────────────────
    #
    # A claim that pays as it goes: 1,000 a week times that week's win rate,
    # four weeks, then nothing left. The stream is what makes it a share rather
    # than a future, and the weekly measurement is what makes the stream
    # interesting, since a bad week is a smaller payment rather than a smaller
    # number at the end.
    #
    # This is not a perpetual and does not pretend to be. Every contract here
    # settles inside a known interval, which is what makes collateral arithmetic
    # rather than an estimate; a claim with no last payment has no such interval,
    # and the funding rate machinery real perpetuals use to live without one is a
    # different risk model. What is here is the honest finite version.
    #
    # It also carries a relation worth watching. Four payments at 1,000 add to
    # 4,000 times the same rate the four week future pays 10,000 times, so the
    # share should be worth 0.4 of that future if only the level mattered. The
    # one thing that should separate them is that a share hands collateral back
    # as it pays and capital is the binding constraint here, so it ought to
    # trade at a premium. Nothing has been done to make that happen; it is a
    # prediction, and the two are listed side by side so it can be checked.
    add(
        f"{SHARE}_{INDIVIDUAL.upper()}_EQ",
        _wr(SHARE, INDIVIDUAL),
        # Nothing is left at the end: it has all been paid out.
        Linear(0.0),
        distribution=DistributionSchedule(
            windows=tuple(DELIVERY_WEEKS), payoff=Linear(WEEKLY_SCALE)
        ),
    )

    return listed


@lru_cache(maxsize=8)
def _world(seed: int = WORLD_SEED) -> CircuitOracle:
    """The settlement oracle for one season, built once.

    No dataset and no reference snapshot, because this world's matches are
    generated from the seed rather than collected: the season is a pure function
    of `(seed, calendar, formats)` and the oracle's own reference id is a digest
    over exactly those. Cached for the same reason the collected world's loader
    was, and with more to gain: the oracle memoizes the matches it has replayed.
    Measured on seed 7, the first build of a season plays 32,256 matches across
    the contract window and the prior one and takes 3.0 seconds; a later build
    of the same season takes 0.9, because only the metric arithmetic re-runs.
    The memo those matches sit in costs 34.5 MB a season, so the eight seasons
    this cache will hold come to roughly 280 MB, which is what the cache size is
    chosen against rather than by taste.

    `min_appearances` is left at zero on purpose. It is a flat bar applied
    inside the metric regardless of how long the window is, and the bar that
    should scale with the window lives on the contract, in `policy_for`. The
    metrics enforce their own structural minimums underneath either of them: a
    dispersion over one appearance is exactly 0.0 and well formed, so the metric
    refuses it rather than handing back a number the engine would settle.
    """
    return CircuitOracle(seed)


def _oracle_for(spec: ContractSpec, seed: int | None) -> CircuitOracle:
    """The oracle that can answer for one contract.

    A named seed wins, because a caller that knows which season it is asking
    about is stating something this cannot infer. Otherwise the season is read
    back off the reference the contract pins, which is the only thing a contract
    carries that identifies its world.

    Falling back to the default season for an unrecognised reference is
    deliberate and safe: the engine compares the reference before it resolves
    anything, so a wrong guess raises `ReferenceMismatch` rather than returning
    a well formed number about the wrong world.
    """
    if seed is not None:
        return _world(seed)
    return _world(_SEASON_OF.get(spec.reference_id, WORLD_SEED))


def true_levels(
    listed: list[Instrument], seed: int | None = None
) -> dict[str, float]:
    """The true *metric level* each contract is written on.

    Not the settlement value: agents are given a view on the underlying rate and
    derive what it implies for each contract themselves, which is both how a
    fundamental analyst actually works and what makes a single noise parameter
    meaningful across a future, an option and an event contract alike.
    """
    levels: dict[str, float] = {}
    for instrument in listed:
        result = settle(instrument.spec, _oracle_for(instrument.spec, seed))
        if result.settled and result.underlying_level is not None:
            levels[instrument.symbol] = float(result.underlying_level)
    return levels


def _prior_scale(spec: ContractSpec) -> float | None:
    """Carrying a level measured over the prior window onto a contract's own.

    A rate is intensive in window length: a win rate over four weeks and over
    one week are the same kind of number, so re-dating measures it directly and
    the answer needs no conversion. A quantity is not, and ``MetricRef.kind``
    already says so in as many words: "the amount delivered in March is a
    different thing from the amount delivered in April". The prior path declared
    that distinction upstream and then ignored it here.

    Measured before this existed, on the world this listing replaced: the two
    weekly commodities observed one week each and every informed trader was
    handed a four week count as though it were a one week count. The contract
    settled at 71.09 and its prior said 274.92, off by 3.87x against a window
    ratio of 3.87, while the four win rate futures came back at 1.00 to 1.04x on
    the same run, which is what said the error was the unit and not the fixture.
    The market wore it exactly as you would expect: at t=180s the contract traded
    171.93 against a fair 71.10, because six informed agents all believed the
    same wrong number.

    The conversion is a belief, not a measurement, and that is the right standard
    for this function: a prior is what somebody thinks on day one, and this whole
    fixture exists because opening on history opens wrong. It would not be an
    acceptable standard anywhere near collateral.

    ``None`` for an underlying that mixes kinds: a difference between a rate and
    a count is not homogeneous in window length, so there is no conversion to
    make, and a prior that cannot be converted is no prior.
    """
    kinds = {ref.kind for ref in spec.underlying.atoms()}
    if kinds <= {"rate", "dispersion"}:
        return 1.0
    if kinds != {"quantity"}:
        return None
    prior_span = PRIOR_WINDOW.end - PRIOR_WINDOW.start
    if prior_span <= timedelta(0):
        return None
    return (spec.window.end - spec.window.start) / prior_span


def prior_levels(
    listed: list[Instrument], seed: int | None = None
) -> dict[str, float]:
    """Where each contract's underlying sat *before* its window opened.

    This is what an informed participant knows on day one, and it is not the
    answer. On this season the difference is the whole point: the first subject
    ran at 0.1469 in the individual format over the four weeks before the window
    and settles at 0.1398, and every one of the twelve moves, so a market that
    opens on history opens wrong and has something to discover.

    It is also a real measurement rather than a fixture, which is why the
    contract window opens four weeks after the circuit does rather than on the
    epoch. A prior window before the epoch names matches that were never played,
    the oracle refuses it, and every informed trader would open on nothing.

    Built by re-dating each contract onto the prior window rather than by a
    separate calculation, so a listing that changes cannot leave this behind. A
    contract whose prior window has too little evidence simply gets no prior, and
    the agent falls back to the truth, vaguely, since it will still be swamped by
    its own uncertainty at the open.
    """
    levels: dict[str, float] = {}
    for instrument in listed:
        # The distribution schedule goes with the window, because a payment
        # measured outside the window a contract observes is measured on
        # evidence that contract never claimed to be about and the spec refuses
        # it, correctly. Re-dated proportionally: the same number of periods,
        # tiling the prior window.
        schedule = instrument.spec.distribution
        if schedule is not None:
            periods = len(schedule.windows)
            span = (PRIOR_WINDOW.end - PRIOR_WINDOW.start) / periods
            schedule = DistributionSchedule(
                windows=tuple(
                    ObservationWindow(
                        PRIOR_WINDOW.start + span * n,
                        PRIOR_WINDOW.start + span * (n + 1),
                    )
                    for n in range(periods)
                ),
                payoff=schedule.payoff,
            )
        spec = replace(
            instrument.spec,
            window=PRIOR_WINDOW,
            policy=policy_for(PRIOR_WINDOW),
            published_at=PRIOR_WINDOW.start - timedelta(days=1),
            distribution=schedule,
        )
        try:
            result = settle(spec, _oracle_for(instrument.spec, seed))
        except Exception:  # noqa: BLE001 - a prior that cannot be measured is no prior
            continue
        if not (result.settled and result.underlying_level is not None):
            continue
        scale = _prior_scale(instrument.spec)
        if scale is None:
            continue
        levels[instrument.symbol] = float(result.underlying_level) * scale
    return levels


def true_values(
    listed: list[Instrument], seed: int | None = None
) -> dict[str, float]:
    """What each contract is actually worth over its life, in ticks.

    Settlement plus every payment it makes on the way, which for everything that
    pays once is just the settlement. A share settles at nothing, so reporting
    only the settlement would say a share is worth nothing, and it would be right
    about the last instant and wrong about every other one.
    """
    values: dict[str, float] = {}
    for instrument in listed:
        oracle = _oracle_for(instrument.spec, seed)
        result = settle(instrument.spec, oracle)
        if not (result.settled and result.settlement_value is not None):
            continue
        total = result.settlement_value + sum(distributions(instrument.spec, oracle))
        values[instrument.symbol] = float(instrument.to_ticks(total))
    return values


def match_maker_capital(
    formats: tuple[str, ...], seed: int, concurrent: int, lots: int = 40
) -> int:
    """What it costs to be the market in every contract on every live match.

    Derived from a real book rather than named, for the reason `maker_capital`
    gives: a figure is right only for the list it was written against, and a
    solo match lists 50 contracts against the statistical listing's 50. Measured
    at build time off match zero of each format, which is representative because
    every match of a format lists the same families over the same field size.

    Cheaper per contract than the statistical listing by a wide margin, and that
    is the bounded payoff doing its work: a winner contract settles in [0, 1]
    where a win rate future settles in [0, 10000], so a whole match book costs
    less to make a market in than a handful of futures. That was the argument
    when a match listed 270 of them and it did not depend on the count, which is
    why trimming the listing to 50 changed this function not at all.
    """
    from arena.market.match_book import list_match

    total = 0.0
    for name in formats:
        book = list_match(seed, 0, name)
        for contract in book.contracts:
            low, high = contract.instrument.spec.value_bounds
            total += float(high - low) * lots
    return int(total * concurrent) + 1


def maker_capital(
    listed: list[Instrument], position_limit: int, quote_size: int
) -> int:
    """What it costs to be the market in every listed contract at once.

    Read off the contracts and the maker's own mandate rather than named as a
    figure, because a figure is right only for the list it was written against.
    Every position here is collateralised against the whole range its contract
    can settle in, so a maker that shows ``quote_size`` on both sides of a book
    and will carry ``position_limit`` in it needs that many lots times that
    range, and it needs it in every book simultaneously, because that is what
    quoting a market means.

    The alternative is what was there before and what this replaces: forty
    million, chosen when the exchange listed fewer contracts and never revised.
    Listing two dispersion futures took the requirement past it, and the
    consequence was not that the makers quoted a little less. Measured on seed 7
    against the listing this replaces, whose front future settled at 4,669: **63,345 orders rejected for collateral**, the makers
    out of the future's book entirely, the front future carrying a spread of
    1,277 points where it had carried 6, and the option chain, which is priced
    off the midpoint of that spread, swinging between 3,629 and 5,701 and
    inverting across strikes. Funded from the list instead: 245 rejects, the
    spread back to 3, and the chain monotone at every sampled moment.

    So this is not a softer limit. Collateral is still exact and still posted in
    full; what changes is that the makers arrive with enough of it to do the job
    they were given, and that listing one more contract funds itself instead of
    quietly starving the ones already there.
    """
    lots = position_limit + quote_size
    total = 0.0
    for instrument in listed:
        low, high = instrument.tick_bounds
        total += float(instrument.from_ticks(int(high) - int(low))) * lots
    return int(total)


def build(
    seed: int = WORLD_SEED,
    speed: float = 1.0,
    arbitrageur: bool = False,
    recycle_capital: bool = True,
    flow_traders: int = 0,
    fees: FeeSchedule = MAKER_TAKER,
    price_band: float | None = 0.05,
    human_cash: int = HUMAN_STARTING_CASH,
    surface: bool = True,
    makers: int = 3,
    opening_auction: bool = True,
    session_seconds: float = 600.0,
    mechanism: str = "book",
    information_flow: bool = True,
    informed: int = 6,
    netting: bool = False,
    # How many books each uninformed trader touches per wake.
    #
    # Six rather than one, because one produced a market that could not support
    # market making at all. Measured over 180 simulated seconds on seed 7 with
    # matches listed: uninformed flow was 0.5% of volume at an informed ratio
    # of 42.3 to 1, and Glosten-Milgrom's own conclusion is that past a high
    # enough informed share no spread both trades and profits. Six takes it to
    # 3.0% and 6.4 to 1.
    #
    # Not higher, and the reason is cost rather than taste. The full curve, same
    # run: 6 books across 40 traders reaches 7.7% and 2.3 to 1 but runs at 0.18x
    # real time, and 14 across 40 reaches a realistic 16.0% and 0.9 to 1 at
    # 0.10x. Realistic composition is affordable for a headless research run and
    # not for a market anybody watches, which is the clearest statement of where
    # this simulator's performance budget actually binds.
    noise_breadth: int = 6,
    noise_traders: int = 14,
    matches: bool = False,
    match_formats: tuple[str, ...] = ("solo", "objective"),
    match_seconds: float = 90.0,
    concurrent_matches: int = 1,
) -> LiveMarket:
    # The scoring rule prices a binary and nothing else, so choosing it
    # narrows the exchange to its event contracts. That is not a limitation
    # worked around: a logarithmic market scoring rule is defined on a
    # partition of outcomes, and there is no honest way to quote a future on it.
    #
    # It also removes the market makers, which is the entire point: on this
    # mechanism the venue *is* the maker, and it subsidises the market rather
    # than trying to profit from it. Experiment 2 compared the two on 200
    # paired trials and found the mechanism explains none of the difference in
    # information aggregation; this is that comparison, reachable by anyone.
    scoring_rule = mechanism == "scoring-rule"
    listed = instruments(seed)
    if scoring_rule:
        listed = [i for i in listed if i.instrument_class == InstrumentClass.EVENT]
        makers = 0
        opening_auction = False
        price_band = None
    by_symbol = {i.symbol: i for i in listed}
    levels = true_levels(listed, seed)

    # Sized for the contracts on offer, not picked round, and now computed from
    # them rather than remembered. A maker is asked to be both sides of every
    # book at once, full collateralisation means that capital is genuinely
    # committed rather than notional, and too little of it means every agent
    # spends the session rejected, which looks like a broken market rather than
    # a poor one. See `maker_capital` for what that cost is and what it looked
    # like when the figure stopped covering it.
    # Floored at one, because the dashboard will run eight of these and the
    # ladder ran out at four: the fifth maker was configured to show -2 lots
    # and carry -50, which the quoting path silently turned back into 1 and
    # which this function would have turned into a negative opening balance.
    # The three the market is built with are unaffected.
    maker_ids = [AgentId(f"mm-{n + 1}") for n in range(max(0, makers))]
    maker_limits = [max(1, MAKER_POSITION_LIMIT - 250 * n) for n in range(len(maker_ids))]
    maker_sizes = [max(1, MAKER_QUOTE_SIZE - 8 * n) for n in range(len(maker_ids))]

    # The breaker's three time constants, scaled from the rule they model.
    #
    # Limit up-limit down uses a fifteen-second limit state, a five-minute
    # pause, and a five-minute trailing reference, against a six-and-a-half
    # hour session. Those are ratios, not durations: a session here lasts
    # minutes and does a day's price discovery in the first one, so real
    # durations make the reference stale for the whole session and the breaker
    # spends its time policing the walk to fair value. Measured with the
    # literal figures: twelve of twenty-six symbols halted at once. Scaled by
    # the same fraction of the session, the reference keeps up and the breaker
    # fires on dislocations instead.
    trading_day = 6.5 * 60 * 60
    scale = session_seconds / trading_day

    # The contract calendar, on the same footing as the breaker windows above:
    # `session_seconds` of simulated time is one trading day, so a four week
    # observation window runs its course in twenty-eight of them.
    #
    # Without this the live venue had no clock at all, so
    # `Venue._enforce_lifecycle` never asked whether a window had closed and
    # nothing ever settled. Measured before it existed: after a simulated hour
    # every listed contract was still `continuous` and the settled set was
    # empty. A position was marked forever and realised never, which leaves an
    # algorithm no terminal event to score itself against.
    #
    # It starts on publication day rather than at the observation window,
    # because two families of contracts share this one clock and they run at
    # different rates. A match is pinned to the circuit's own schedule of five
    # minutes each from the epoch, and this calendar covers a day every
    # `session_seconds`, which is 144 contract seconds per simulated second
    # against a match schedule advancing 3.3 of them. So a clock started at the
    # observation window would already be four weeks past every match ever
    # listed, and `_enforce_lifecycle` would close each match contract in the
    # same instant the operator listed it. Starting a day before the epoch buys
    # the whole default session, and costs the statistical contracts nothing
    # that matters: they expire 57 trading days out, which is 34,200 simulated
    # seconds at the default session length and so past any session anyone runs.
    calendar = Calendar(start=PUBLISHED_AT, seconds_per_day=session_seconds)

    venue_class = LmsrVenue if scoring_rule else Venue
    venue = venue_class(
        "arena-lmsr" if scoring_rule else "arena",
        clock=calendar.now,
        netting=netting,
        starting_cash=40_000_000,
        fees=fees,
        price_band=price_band,
        limit_state_ns=max(1, int(15 * scale * 1e9)),
        pause_ns=max(1, int(300 * scale * 1e9)),
        reference_window_ns=max(1, int(300 * scale * 1e9)),
        # A person starts with an account they can actually read, and each
        # maker with what its own mandate costs to collateralise. Everyone else
        # keeps the default: the informed traders are bounded by their position
        # limits rather than by their cash, so funding them from the listing
        # would change how much informed capital the market has, which is the
        # quantity Experiment 1 is a measurement of, and not something to move
        # as a side effect of paying the makers properly.
        balances={
            HUMAN_ID: human_cash,
            **{
                agent_id: maker_capital(listed, limit, size)
                for agent_id, limit, size in zip(maker_ids, maker_limits, maker_sizes)
            },
        },
    )
    for instrument in listed:
        venue.list_instrument(instrument)

    maker_id = maker_ids[0] if maker_ids else AgentId("mm-none")
    operator_id = AgentId("exchange")
    arb_id = AgentId("arb-1")
    fund_ids = [AgentId(f"fund-{n}") for n in range(informed)]
    noise_ids = [AgentId(f"noise-{i:02d}") for i in range(noise_traders)]
    flow_ids = [AgentId(f"flow-{i:02d}") for i in range(flow_traders)]

    latency = PairwiseLatency(
        default=millis(4),
        per_agent={
            **{a: micros(150 + 40 * n) for n, a in enumerate(maker_ids)},
            operator_id: micros(1),                      # the venue itself
            arb_id: millis(2),
            # Sharper agents are also closer, which is what being a serious
            # participant looks like: information and speed are bought together.
            **{a: millis(3 + 2 * n) for n, a in enumerate(fund_ids)},
            HUMAN_ID: millis(20),                        # a person on a browser
            **{a: millis(45) for a in noise_ids},        # retail, far away
            **{a: millis(6) for a in flow_ids},          # brokers' algos
        },
        jitter_fraction=0.15,
        seed=seed,
    )

    kernel = Kernel(seed=seed, latency=latency)
    # The breaker measures elapsed time, so it needs something that elapses.
    # Without this its limit-state timer never advanced and a symbol could sit
    # outside its band forever without ever pausing: the breaker recorded 241
    # excursions in three minutes and halted for none of them.
    venue.sim_clock = lambda: int(kernel.now)
    venue_agent = VenueAgent(VENUE_ID, venue)
    human = HumanAgent(VENUE_ID, by_symbol)

    # Options priced off one distribution on the underlying rather than each
    # book on its own. `surface=False` restores the plain maker, which is what
    # the before-and-after in docs/GAPS.md was measured against, and the
    # comparison has to stay runnable or the numbers in it are just claims.
    maker_class = SurfaceMarketMaker if surface else MarketMaker
    opening = {s: float(sum(i.tick_bounds) / 2) for s, i in by_symbol.items()}

    # More than one, because one was measured to be the whole other side of the
    # market. Sweeping 60% of the offers, the single maker absorbed 89% of the
    # order, ended short past the point its collateral let it quote, and the
    # spread it left behind was still ten times its opening width three minutes
    # later. Not a slow repair: no repair, because the maker that got run over
    # was the only one there.
    #
    # They differ in the three parameters that decide who gets run over first:
    # a tighter maker is hit sooner and fills its inventory faster, a wider one
    # is still quoting when the tight one has stopped. Identical makers would be
    # one maker with three times the capital, which is not what was missing.
    makers_list = [
        maker_class(
            agent_id,
            VENUE_ID,
            by_symbol,
            wake_interval=millis(300 + 90 * n),
            half_spread=5 + 3 * n,
            quote_size=maker_sizes[n],
            max_skew_fraction=0.10,
            position_limit=maker_limits[n],
            # The middle of each contract's range, never its true value: if a
            # maker started on the answer there would be nothing to discover.
            #
            # It is a wider opening error on this world than on the one it
            # replaces, and the reason is structural rather than a choice. A
            # collected win rate sits near 0.5 by construction, so the middle of
            # [0, 10000] was 5,000 against a settlement of 4,669. An elimination
            # win rate sits at the format's neutral 0.100, so the same anchor
            # opens 3,900 above a settlement near 1,100. That is the informed
            # traders' work rather than a mispricing to be papered over, and it
            # is why they are given a prior measured over the season's first
            # four weeks rather than the midpoint.
            #
            # Withdrawing this during the opening call was tried and was worse.
            # The theory was sound, since a maker that turns up to the auction
            # with a guess makes the guess the official opening price, but with
            # only two informed agents and a crowd of random market orders, an
            # auction with no maker in it cleared the front future at 9,377
            # against a fair value of 4,669, both figures from the listing this
            # replaces. A mediocre anchored open beats a
            # wild unanchored one, and the informed interest still pulls the
            # clearing price toward fair.
            reference=opening,
        )
        for n, agent_id in enumerate(maker_ids)
    ]

    # They bring interest to the opening call rather than waiting for a price,
    # which on a venue that opens with an auction is the difference between a
    # market and an empty book: every other agent here reacts to a price, so
    # with nobody posting first the auction cleared nothing and the exchange
    # stayed empty for the whole session.
    # Their evidence arrives over the session rather than all at t=0.
    #
    # With everything known at the open the market has an information *stock*:
    # it converges within seconds and then nothing can move it, because there
    # is nothing left to arrive. Measured that way, the realised dispersion of
    # the old front future over ten minutes was 14.6 on a price near 4,670:
    # options
    # were worth their intrinsic value and nothing more, and every binary was a
    # foregone conclusion inside a minute. A market whose subject is
    # disagreement needs something to keep disagreeing about.
    #
    # The experiment harnesses pass nothing here, so every published result was
    # produced under the old model and stays reproducible.
    reveal = seconds(int(session_seconds)) if information_flow else None
    priors = prior_levels(listed, seed) if information_flow else {}

    # Six of them, log-spaced in precision, rather than two.
    #
    # Two informed traders is not a population, it is an anecdote, and it had a
    # measurable consequence rather than an aesthetic one. Both ran into their
    # position limits about a minute in and the price simply stopped there: the
    # sharper one believed 4,687 against a true 4,669 on the old listing, was
    # short its full 900
    # lots, and could do nothing while the market printed 5,005. The makers had
    # absorbed 1,560 lots between them and had capacity for 1,300 more. The
    # price is where informed capital runs out, which is Experiment 1's finding,
    # but a market where informed capital is two agents is measuring the fixture
    # rather than the mechanism.
    #
    # Log-spaced because that is how information is actually distributed: a few
    # who know a great deal, more who know a little. Their limits are smaller
    # than the two they replace, so no one of them can move the price alone.
    funds = [
        FundamentalTrader(
            agent_id, VENUE_ID, by_symbol, levels,
            wake_interval=millis(500 + 170 * n),
            precision=0.6 * (1.45 ** n),
            base_size=10 + 2 * n,
            max_position=350 + 90 * n,
            open_interest=opening_auction,
            reveal_over=reveal,
            prior_level=priors,
        )
        for n, agent_id in enumerate(fund_ids)
    ]
    noise = [
        NoiseTrader(
            a,
            VENUE_ID,
            by_symbol,
            wake_interval=millis(1_100),
            symbols_per_wake=noise_breadth,
        )
        for a in noise_ids
    ]

    # Off unless asked for. These agents *assume* power-law sizes and bursty
    # arrivals, so any claim that this market produces fat tails emergently is
    # only meaningful with them absent. See arena/agents/flow.py.
    flow = [
        FlowTrader(agent_id, VENUE_ID, by_symbol, wake_interval=millis(500))
        for agent_id in flow_ids
    ]

    agents = [*makers_list, *funds, *noise, *flow]
    if opening_auction:
        # The market opens with a call rather than with whoever arrives first.
        agents.append(SessionOperator(operator_id, venue, venue_agent=venue_agent))
    if arbitrageur:
        # Off by default, on the evidence. It derives the right identities and
        # trades them, but measured across four paired seeds it improved spread
        # consistency on three and made it worse on the fourth, and on one seed
        # it took visible ask depth from 877 lots to 69. Enforcing a relation by
        # repeatedly lifting the book buys consistency with liquidity, and a
        # market that cannot absorb an order is broken more fundamentally than
        # one carrying a stale spread. See docs/GAPS.md for the numbers.
        agents.append(
            Arbitrageur(
                arb_id,
                VENUE_ID,
                by_symbol,
                wake_interval=millis(400),
                recycle_capital=recycle_capital,
            )
        )
    match_operator = None
    if matches:
        # A market maker of its own, because a match family is priced off one
        # distribution over outcomes and the statistical makers are not. Given
        # no instruments at construction: every contract it quotes arrives with
        # a match, through the same note-and-join path as every other agent.
        match_maker = MatchMaker(
            AgentId("mm-match"),
            VENUE_ID,
            {},
            wake_interval=millis(360),
        )
        agents.append(match_maker)
        venue.open_account(
            match_maker.agent_id, match_maker_capital(match_formats, seed, concurrent_matches)
        )

        # Told about every listing, so the arbitrageur takes on each match's
        # identities and every trader can see the book. The operator is not in
        # its own participant list: it lists contracts, it does not trade them.
        match_operator = MatchOperator(
            AgentId("matches"),
            venue,
            seed=seed,
            formats=match_formats,
            match_seconds=match_seconds,
            concurrent=concurrent_matches,
            venue_agent=venue_agent,
            participants=list(agents),
        )
        agents.append(match_operator)

    kernel.add(venue_agent)
    kernel.add(human)
    kernel.add_all(agents)

    return LiveMarket(
        venue=venue,
        kernel=kernel,
        venue_agent=venue_agent,
        human=human,
        agents=agents,
        speed=speed,
        calendar=calendar,
        # Bound to the same oracle every other settlement in this module uses,
        # so a contract that settles live gets exactly the value `prior_levels`
        # and `true_values` would have computed for it, and to the same season
        # the matches are played from.
        # Two settlement sources, dispatched on who owns the contract. A match
        # settles against the match that was played and the statistical
        # contracts against the oracle, and neither can answer for the other:
        # asking the oracle about a match contract fails rather than returning
        # something wrong, which is the right failure but not a useful one.
        # The operator settles its own as it goes, so anything reaching here
        # from a match is a contract it has already closed.
        settlement_source=lambda spec: settle(spec, _world(seed)),
        # So a person who signs in gets an account they can read a profit
        # against, at the same distance from the exchange as anyone else at a
        # browser, rather than the bots' balance sheet and the default wire.
        latency=latency,
        seat_cash=human_cash,
    )
