"""Assemble the live market the dashboard serves.

The instruments settle from the real fixture dataset, and the fundamental
agents are told what each will actually settle at -- then given a *noisy* view of
it, with a different precision each. So the market has a true value to converge
toward, and whether it gets there is something you can watch rather than
something asserted.

Latency is heterogeneous on purpose. The market maker is effectively colocated,
the funds are a few milliseconds out, the noise traders are far away, and the
human is somewhere in between. That is the same configuration the research
experiments use; the dashboard is not a special case of the simulator, it is the
simulator with a browser attached.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

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
from arena.contracts.underlying import Basket, Difference, Single
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
from arena.worlds.brawl.dataset import CanonicalDataset
from arena.worlds.brawl.metrics import metric_ref
from arena.worlds.brawl.oracle import BrawlOracle
from arena.worlds.brawl.reference import load_reference

REPO = Path(__file__).resolve().parents[1]
REFERENCE_ID = "ref-2026S09-v1"

# What a person opens the exchange with. Large enough to take a real
# position in the futures, which settle around 4,700 a contract, and small
# enough that a profit is a number you can see.
HUMAN_STARTING_CASH = 250_000

# The widest maker's mandate: how much it shows on each side of each book, and
# how much of any one contract it will carry. The other makers are scaled down
# from these, and `maker_capital` funds each one from its own pair -- so the
# three numbers that decide what a maker is asked to do live in one place as
# the number that decides whether it can afford to do it.
MAKER_QUOTE_SIZE = 30
MAKER_POSITION_LIMIT = 1_200
UTC = timezone.utc

POLICY = DataPolicy(
    min_sample_size=1_000, min_stratum_battles=200, min_strata_coverage=0.80
)
WINDOW = ObservationWindow(
    datetime(2026, 8, 31, tzinfo=UTC), datetime(2026, 9, 28, tzinfo=UTC)
)


def _wr(subject: str):
    return Single(metric_ref("adjusted_win_rate", subject))


def _spec(
    contract_id: str,
    underlying,
    payoff,
    tick: str = "0.25",
    window: ObservationWindow | None = None,
    distribution: DistributionSchedule | None = None,
    tick_table: tuple[tuple[str, str], ...] = (),
) -> ContractSpec:
    """A contract on the default window unless one is given.

    The window is a parameter because a commodity needs it to be. Every rate
    contract here measures the same four weeks, so the window was a constant --
    but the whole point of a delivery month is that the amount delivered in one
    is a different thing from the amount delivered in the next, and a term
    structure is a set of contracts that differ in nothing else.
    """
    measured = window or WINDOW
    return ContractSpec(
        contract_id=contract_id,
        underlying=underlying,
        payoff=payoff,
        window=measured,
        policy=policy_for(measured),
        reference_id=REFERENCE_ID,
        published_at=measured.start - timedelta(days=1),
        tick_size=tick,
        tick_table=tick_table,
        distribution=distribution,
    )


def policy_for(window: ObservationWindow) -> DataPolicy:
    """Scale the evidence bar to the length of the window.

    A one-week delivery window contains roughly a quarter of the battles a
    four-week observation window does, so holding both to the same minimum
    sample size would void every weekly contract for thin data that is not
    actually thin -- it is one week's worth.
    """
    weeks = max(1, round((window.end - window.start).days / 7))
    return DataPolicy(
        min_sample_size=250 * weeks,
        min_stratum_battles=POLICY.min_stratum_battles,
        min_strata_coverage=POLICY.min_strata_coverage,
    )


# Weekly delivery, the way a commodity is listed: the same deliverable across
# consecutive windows, so the curve across them is a forward curve rather than
# a set of unrelated contracts.
DELIVERY_WEEKS = [
    ObservationWindow(
        datetime(2026, 8, 31, tzinfo=UTC) + timedelta(weeks=n),
        datetime(2026, 9, 7, tzinfo=UTC) + timedelta(weeks=n),
    )
    for n in range(4)
]


def _chain(subject: str, calls: tuple[int, ...], puts: tuple[int, ...]) -> list[Instrument]:
    """An option ladder on one subject.

    Strikes are listed across the plausible band an adjusted win rate occupies,
    the way an exchange lists strikes, and deliberately not around where the
    metric is known to settle -- listing the answer is not listing a market.
    Some rungs will be near-certain and some genuinely open, and which is which
    is the thing the market is for.

    Ladders were thin before this. `SPIKE` carried three calls and two puts,
    `CROW` and `ELPRIMO` carried none, so a chain that the surface maker,
    the arbitrageur's vertical and butterfly relations and the whole
    static-arbitrage apparatus exist to price had almost nothing to price. The
    exchange is also the wrong shape at that size: real venues reach hundreds
    of listings mostly through strikes and expiries on subjects they already
    carry, which is exactly what the markets screen was rebuilt to absorb.
    """
    return [
        Instrument(
            f"{subject}_C{strike}",
            _spec(f"{subject}_C{strike}", _wr(subject), Call(float(strike), 10_000.0),
                  tick="0.25"),
        )
        for strike in calls
    ] + [
        Instrument(
            f"{subject}_P{strike}",
            _spec(f"{subject}_P{strike}", _wr(subject), Put(float(strike), 10_000.0),
                  tick="0.25"),
        )
        for strike in puts
    ]


def instruments() -> list[Instrument]:
    return [
        Instrument("SPIKE_WR_FUT", _spec("SPIKE_WR_FUT", _wr("SPIKE"), Linear(10_000.0))),
        Instrument("CROW_WR_FUT", _spec("CROW_WR_FUT", _wr("CROW"), Linear(10_000.0))),
        # A ladder, not a single threshold.
        #
        # One binary struck at 0.48 was a foregone conclusion: the rate settles
        # near 0.467, so the answer was known and the price sat pinned at three
        # cents for the whole session. A prediction market with no uncertainty
        # in it is a countdown, not a market.
        #
        # These thresholds are chosen from the plausible band a win rate lives
        # in, the way an exchange lists strikes -- not from the settlement value,
        # which would be listing the answer. Some rungs will be near-certain and
        # some genuinely open; which is which is exactly what the market is for.
        *[
            Instrument(
                f"SPIKE_GT{int(threshold * 100)}",
                _spec(
                    f"SPIKE_GT{int(threshold * 100)}",
                    _wr("SPIKE"),
                    Binary(">", threshold, payout=1.0),
                    tick="0.01",
                ),
            )
            for threshold in (0.44, 0.46, 0.47, 0.48)
        ],
        Instrument(
            "CROW_GT47",
            _spec("CROW_GT47", _wr("CROW"), Binary(">", 0.47, payout=1.0), tick="0.01"),
        ),
        # Listed so the index has all three of its legs. Without it the
        # arbitrageur declines to form the index relation at all -- correctly,
        # since a basket priced against two of its three components is a bet on
        # the third -- so ASSASSIN_IDX floated free of the things it is defined
        # as. One contract closes that.
        # The one contract with a tiered tick, so the rule is exercised rather
        # than merely available. A quarter of a point up to 4,000 and a whole
        # point above it: fine enough at the bottom of the range for a spread
        # to narrow to what the market knows, coarse enough at the top that a
        # resting order cannot be stepped in front of for a rounding error.
        Instrument(
            "PIPER_WR_FUT",
            _spec(
                "PIPER_WR_FUT",
                _wr("PIPER"),
                Linear(10_000.0),
                tick_table=(("4000.00", "1.00"),),
            ),
        ),
        Instrument(
            "SPIKE_CROW",
            _spec("SPIKE_CROW", Difference(_wr("SPIKE"), _wr("CROW")), Linear(10_000.0)),
        ),
        # Options are payoffs on the same underlying as the future, so they
        # settle from the same metric at the same instant and need no separate
        # machinery. Struck either side of where SPIKE actually settles, so one
        # expires worthless and the other in the money.
        # Struck as a ladder for the same reason. Both options used to sit at
        # 4,700 against a rate settling near 4,669, so both were worth almost
        # nothing and their books were two-sided about two percent of the time.
        # An option nobody can quote is not an asset class, it is a row.
        *_chain("SPIKE", (4_550, 4_600, 4_650, 4_700, 4_750),
                (4_600, 4_650, 4_700, 4_750, 4_800)),
        # CROW carried a future and one binary and no options at all, so half
        # the chain machinery had nothing to act on for the second-largest
        # subject on the exchange.
        *_chain("CROW", (4_650, 4_700, 4_750), (4_700, 4_750)),
        # ── ELPRIMO ──────────────────────────────────────────────────────
        #
        # A fourth subject, and it was already in the data. The fixture carries
        # CROW, ELPRIMO, PIPER and SPIKE, and ELPRIMO was the one with no
        # market written on it -- so this lists a Brawler the oracle can
        # already settle rather than inventing one it cannot. Prior level
        # 0.4674, which is where its strikes are banded from.
        Instrument(
            "ELPRIMO_WR_FUT",
            _spec("ELPRIMO_WR_FUT", _wr("ELPRIMO"), Linear(10_000.0)),
        ),
        *[
            Instrument(
                f"ELPRIMO_GT{int(threshold * 100)}",
                _spec(
                    f"ELPRIMO_GT{int(threshold * 100)}",
                    _wr("ELPRIMO"),
                    Binary(">", threshold, payout=1.0),
                    tick="0.01",
                ),
            )
            for threshold in (0.45, 0.47, 0.49)
        ],
        *_chain("ELPRIMO", (4_600, 4_650, 4_700), (4_650, 4_700)),
        # ── volatility ───────────────────────────────────────────────────
        #
        # A claim on how unevenly a Brawler performs across the maps and modes
        # it plays, rather than on how well. Two Brawlers with the same
        # adjusted win rate and different spreads are not the same thing to
        # own: one wins everywhere, the other wins on half the maps and loses
        # on the rest, and only this contract can tell you which you are
        # holding.
        #
        # It is the first claim here on a *second* moment, and it joins the
        # exchange on the same terms as everything else: the standard deviation
        # of a set of rates cannot exceed 0.5, so collateral stays arithmetic.
        *[
            Instrument(
                f"{subject}_DISP",
                _spec(
                    f"{subject}_DISP",
                    Single(metric_ref("stratum_dispersion", subject)),
                    Linear(10_000.0),
                    tick="0.25",
                ),
            )
            for subject in ("SPIKE", "CROW")
        ],
        # ── the weekly legs a share is made of ───────────────────────────
        #
        # One future per delivery week, settling at 1,000 times that week's
        # adjusted win rate -- which is precisely what SPIKE_EQ pays at the end
        # of that week. So the share is the sum of these four, exactly, and not
        # approximately: same metric, same window, same evidential bar, so the
        # two sides of that equation resolve from the same numbers.
        #
        # Listed for that reason. Before them the only relation available was
        # "the share is worth 0.4 times the four-week future", which is *not*
        # an identity: the four weekly rates are each battle-weighted, so they
        # do not average to the four-week rate. Measured, the two differ by
        # 0.08% -- small, and small is exactly what makes it dangerous to trade
        # as though it were exact. With the weekly legs listed there is a real
        # identity to enforce, and the arbitrageur enforces identities only.
        #
        # CROW deliberately has no legs, so one share is arbitrage-linked and
        # one is not. That is a control, not an oversight.
        *[
            Instrument(
                f"SPIKE_WR_W{n + 1}",
                _spec(
                    f"SPIKE_WR_W{n + 1}",
                    _wr("SPIKE"),
                    Linear(1_000.0),
                    tick="0.25",
                    window=week,
                ),
            )
            for n, week in enumerate(DELIVERY_WEEKS)
        ],
        # ── shares ───────────────────────────────────────────────────────
        #
        # A claim on a Brawler's performance that pays as it goes: 1,000 a week
        # times its adjusted win rate that week, four weeks, then nothing left.
        # The stream is what makes it a share rather than a future, and the
        # weekly measurement is what makes the stream interesting -- a bad week
        # is a smaller payment, not a smaller number at the end.
        #
        # This is not a perpetual and does not pretend to be. Every contract
        # here settles inside a known interval, which is what makes collateral
        # arithmetic rather than an estimate; a claim with no last payment has
        # no such interval, and the funding-rate machinery that lets real
        # perpetuals live without one is a different risk model. What is here
        # is the honest finite version, and docs/GAPS.md says so.
        #
        # It also carries a relation worth watching. Four weekly payments at
        # 1,000 add to 4,000 times the same rate the four-week future pays
        # 10,000 times, so SPIKE_EQ should be worth 0.4 x SPIKE_WR_FUT if the
        # only thing that mattered were the level. The one thing that should
        # separate them is that a share hands collateral back as it pays, and
        # capital is the binding constraint in this market -- so the share
        # ought to trade at a premium to the future. Nothing has been done to
        # make that happen; it is a prediction, and the two are listed side by
        # side so it can be checked.
        *[
            Instrument(
                f"{subject}_EQ",
                _spec(
                    f"{subject}_EQ",
                    _wr(subject),
                    # Nothing is left at the end: it has all been paid out.
                    Linear(0.0),
                    tick="0.25",
                    distribution=DistributionSchedule(
                        windows=tuple(DELIVERY_WEEKS),
                        payoff=Linear(1_000.0),
                    ),
                ),
            )
            for subject in ("SPIKE", "CROW")
        ],
        # ── commodities ────────────────────────────────────────────────
        #
        # A claim on an amount delivered, not on a proportion. Battles played,
        # in thousands, over one delivery week -- which makes the window part of
        # the contract rather than a detail, and gives the four of them a term
        # structure with its own shape.
        #
        # Volume is measured in the canonical corpus rather than in the game,
        # and the metric says so at length. A wider crawl sees more battles.
        *[
            Instrument(
                f"SPIKE_VOL_W{n + 1}",
                _spec(
                    f"SPIKE_VOL_W{n + 1}",
                    Single(metric_ref("battle_volume", "SPIKE")),
                    Linear(1.0),
                    tick="0.05",
                    window=week,
                ),
            )
            for n, week in enumerate(DELIVERY_WEEKS)
        ],
        Instrument(
            "CROW_VOL_W1",
            _spec(
                "CROW_VOL_W1",
                Single(metric_ref("battle_volume", "CROW")),
                Linear(1.0),
                tick="0.05",
                window=DELIVERY_WEEKS[0],
            ),
        ),
        Instrument(
            "ASSASSIN_IDX",
            _spec(
                "ASSASSIN_IDX",
                Basket(((_wr("SPIKE"), 0.5), (_wr("CROW"), 0.3), (_wr("PIPER"), 0.2))),
                Linear(10_000.0),
            ),
        ),
    ]


@lru_cache(maxsize=1)
def _world():
    """The dataset and reference snapshot, loaded once.

    Pure functions of two committed files, so caching changes nothing about
    what any run produces -- it only stops every market build from re-reading a
    CSV and re-running settlement, which dominated the test suite.
    """
    dataset = CanonicalDataset.from_csv(REPO / "data" / "fixtures" / "brawl_aggregates.csv")
    reference = load_reference(REPO / "data" / "reference" / f"{REFERENCE_ID}.json")
    return dataset, reference, BrawlOracle(dataset, reference, POLICY)


def true_levels(listed: list[Instrument]) -> dict[str, float]:
    """The true *metric level* each contract is written on.

    Not the settlement value: agents are given a view on the underlying rate
    and derive what it implies for each contract themselves, which is both how
    a fundamental analyst actually works and what makes a single noise
    parameter meaningful across a future, an option and an event contract
    alike.
    """
    _dataset, _reference, oracle = _world()

    levels: dict[str, float] = {}
    for instrument in listed:
        result = settle(instrument.spec, oracle)
        if result.settled and result.underlying_level is not None:
            levels[instrument.symbol] = float(result.underlying_level)
    return levels


# The four weeks immediately before the contract window. Everything in it
# exists on the day the contracts are published, so a belief anchored here is
# lookahead-free by construction rather than by care.
PRIOR_WINDOW = ObservationWindow(WINDOW.start - timedelta(weeks=4), WINDOW.start)


def _prior_scale(spec: ContractSpec) -> float | None:
    """Carrying a level measured over the prior window onto a contract's own.

    A rate is intensive in window length: a win rate over four weeks and over
    one week are the same kind of number, so re-dating measures it directly and
    the answer needs no conversion. A quantity is not, and ``MetricRef.kind``
    already says so in as many words -- "the amount delivered in March is a
    different thing from the amount delivered in April". The prior path
    declared that distinction upstream and then ignored it here.

    Measured before this existed, against the four-week prior window: the two
    weekly commodities observe one week each, and every informed trader was
    handed a four-week count as though it were a one-week count. SPIKE_VOL_W1
    settles at 71.09 and its prior said 274.92, off by 3.87x against a window
    ratio of 3.87; CROW_VOL_W1, 56.75 against 214.96, 3.79x. The four win-rate
    futures came back at 1.00 to 1.04x on the same run, which is what says the
    error was the unit and not the fixture. The market wore it exactly as you
    would expect: at t=180s SPIKE_VOL_W1 traded 171.93 against a fair 71.10,
    because six informed agents all believed the same wrong number.

    The conversion is a belief, not a measurement, and that is the right
    standard for this function -- a prior is what somebody thinks on day one,
    and this whole fixture exists because opening on history opens wrong. It
    would not be an acceptable standard anywhere near collateral.

    ``None`` for an underlying that mixes kinds: a difference between a rate
    and a count is not homogeneous in window length, so there is no conversion
    to make, and a prior that cannot be converted is no prior.
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


def prior_levels(listed: list[Instrument]) -> dict[str, float]:
    """Where each contract's underlying sat *before* its window opened.

    This is what an informed participant knows on day one, and it is not the
    answer. On this fixture the difference is the whole point: SPIKE ran at
    0.4839 for the twelve weeks before the window and settles at 0.4669, so a
    market that opens on history opens wrong and has something to discover.

    Built by re-dating each contract onto the prior window rather than by a
    separate calculation, so a listing that changes cannot leave this behind.
    A contract whose prior window has too little evidence simply gets no prior,
    and the agent falls back to the truth -- vaguely, since it will still be
    swamped by its own uncertainty at the open.
    """
    _dataset, _reference, oracle = _world()

    levels: dict[str, float] = {}
    for instrument in listed:
        # The distribution schedule goes with the window, because a payment
        # measured outside the window a contract observes is measured on
        # evidence that contract never claimed to be about -- the spec refuses
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
            result = settle(spec, oracle)
        except Exception:  # noqa: BLE001 - a prior that cannot be measured is no prior
            continue
        if not (result.settled and result.underlying_level is not None):
            continue
        scale = _prior_scale(instrument.spec)
        if scale is None:
            continue
        levels[instrument.symbol] = float(result.underlying_level) * scale
    return levels


def true_values(listed: list[Instrument]) -> dict[str, float]:
    """What each contract is actually worth over its life, in ticks.

    Settlement plus every payment it makes on the way, which for everything
    that pays once is just the settlement. A share settles at nothing, so
    reporting only the settlement would say a share is worth nothing -- and it
    would be right about the last instant and wrong about every other one.
    """
    _dataset, _reference, oracle = _world()

    values: dict[str, float] = {}
    for instrument in listed:
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
    match lists 270 contracts where the statistical listing lists 47. Measured
    at build time off match zero of each format, which is representative
    because every match of a format lists the same families over the same
    field size.

    Cheaper per contract than the statistical listing by a wide margin, and
    that is the bounded payoff doing its work: a winner contract settles in
    [0, 1] where a win-rate future settles in [0, 10000], so 270 match
    contracts cost less to make a market in than a handful of futures.
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
    range -- and it needs it in every book simultaneously, because that is what
    quoting a market means.

    The alternative is what was there before and what this replaces: forty
    million, chosen when the exchange listed fewer contracts and never revised.
    Listing the two dispersion futures took the requirement past it, and the
    consequence was not that the makers quoted a little less. Measured on
    seed 7 with the flat figure: **63,345 orders rejected for collateral**, the
    makers out of the future's book entirely, `SPIKE_WR_FUT` carrying a spread
    of 1,277 points where it had carried 6, and the option chain -- which is
    priced off the midpoint of that spread -- swinging between 3,629 and 5,701
    and inverting across strikes. Funded from the list instead: 245 rejects,
    the spread back to 3, and the chain monotone at every sampled moment.

    So this is not a softer limit. Collateral is still exact and still posted
    in full; what changes is that the makers arrive with enough of it to do the
    job they were given, and that listing a twenty-ninth contract funds itself
    instead of quietly starving the twenty-eight already there.
    """
    lots = position_limit + quote_size
    total = 0.0
    for instrument in listed:
        low, high = instrument.tick_bounds
        total += float(instrument.from_ticks(int(high) - int(low))) * lots
    return int(total)


def build(
    seed: int = 7,
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
    matches: bool = False,
    match_formats: tuple[str, ...] = ("solo", "objective"),
    match_seconds: float = 90.0,
    concurrent_matches: int = 1,
) -> LiveMarket:
    # The scoring rule prices a binary and nothing else, so choosing it
    # narrows the exchange to its event contracts. That is not a limitation
    # worked around -- a logarithmic market scoring rule is defined on a
    # partition of outcomes, and there is no honest way to quote a future on it.
    #
    # It also removes the market makers, which is the entire point: on this
    # mechanism the venue *is* the maker, and it subsidises the market rather
    # than trying to profit from it. Experiment 2 compared the two on 200
    # paired trials and found the mechanism explains none of the difference in
    # information aggregation; this is that comparison, reachable by anyone.
    scoring_rule = mechanism == "scoring-rule"
    listed = instruments()
    if scoring_rule:
        listed = [i for i in listed if i.instrument_class == InstrumentClass.EVENT]
        makers = 0
        opening_auction = False
        price_band = None
    by_symbol = {i.symbol: i for i in listed}
    levels = true_levels(listed)

    # Sized for the contracts on offer, not picked round -- and now computed
    # from them rather than remembered. A maker is asked to be both sides of
    # every book at once, full collateralisation means that capital is
    # genuinely committed rather than notional, and too little of it means
    # every agent spends the session rejected, which looks like a broken market
    # rather than a poor one. See `maker_capital` for what that cost is and
    # what it looked like when the figure stopped covering it.
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
    # `session_seconds` of simulated time is one trading day, so a four-week
    # observation window runs its course in twenty-eight of them.
    #
    # Without this the live venue had no clock at all, so
    # `Venue._enforce_lifecycle` never asked whether a window had closed and
    # nothing ever settled. Measured before it existed: after a simulated hour
    # all 47 contracts were still `continuous` and the settled set was empty.
    # A position was marked forever and realised never, which leaves an
    # algorithm no terminal event to score itself against.
    calendar = Calendar(start=WINDOW.start, seconds_per_day=session_seconds)

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
        # would change how much informed capital the market has -- which is the
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
    noise_ids = [AgentId(f"noise-{i:02d}") for i in range(14)]
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
    # outside its band forever without ever pausing -- the breaker recorded 241
    # excursions in three minutes and halted for none of them.
    venue.sim_clock = lambda: int(kernel.now)
    venue_agent = VenueAgent(VENUE_ID, venue)
    human = HumanAgent(VENUE_ID, by_symbol)

    # Options priced off one distribution on the underlying rather than each
    # book on its own. `surface=False` restores the plain maker, which is what
    # the before-and-after in docs/GAPS.md was measured against -- the comparison
    # has to stay runnable or the numbers in it are just claims.
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
            # Withdrawing this during the opening call was tried and was worse.
            # The theory was sound -- a maker that turns up to the auction with
            # a guess makes the guess the official opening price -- but with
            # only two informed agents and a crowd of random market orders, an
            # auction with no maker in it cleared `SPIKE_WR_FUT` at 9,377
            # against a fair value of 4,669. A mediocre anchored open beats a
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
    # `SPIKE_WR_FUT` over ten minutes was 14.6 on a price near 4,670 -- options
    # were worth their intrinsic value and nothing more, and every binary was a
    # foregone conclusion inside a minute. A market whose subject is
    # disagreement needs something to keep disagreeing about.
    #
    # The experiment harnesses pass nothing here, so every published result was
    # produced under the old model and stays reproducible.
    reveal = seconds(int(session_seconds)) if information_flow else None
    priors = prior_levels(listed) if information_flow else {}

    # Six of them, log-spaced in precision, rather than two.
    #
    # Two informed traders is not a population, it is an anecdote -- and it had
    # a measurable consequence rather than an aesthetic one. Both ran into their
    # position limits about a minute in and the price simply stopped there:
    # `fund-sharp` believed 4,687 against a true 4,669, was short its full 900
    # lots, and could do nothing while the market printed 5,005. The makers had
    # absorbed 1,560 lots between them and had capacity for 1,300 more. The
    # price is where informed capital runs out, which is Experiment 1's finding
    # -- but a market where informed capital is two agents is measuring the
    # fixture rather than the mechanism.
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
        NoiseTrader(a, VENUE_ID, by_symbol, wake_interval=millis(1_100))
        for a in noise_ids
    ]

    # Off unless asked for. These agents *assume* power-law sizes and bursty
    # arrivals, so any claim that this market produces fat tails emergently is
    # only meaningful with them absent -- see arena/agents/flow.py.
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
        # so a contract that settles live gets exactly the value
        # `prior_levels` and `true_values` would have computed for it.
        # Two settlement sources, dispatched on who owns the contract. A match
        # settles against the match that was played and the statistical
        # contracts against the oracle, and neither can answer for the other:
        # asking the oracle about a match contract fails rather than returning
        # something wrong, which is the right failure but not a useful one.
        # The operator settles its own as it goes, so anything reaching here
        # from a match is a contract it has already closed.
        settlement_source=lambda spec: settle(spec, _world()[2]),
        # So a person who signs in gets an account they can read a profit
        # against, at the same distance from the exchange as anyone else at a
        # browser -- rather than the bots' balance sheet and the default wire.
        latency=latency,
        seat_cash=human_cash,
    )
