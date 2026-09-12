"""The statistical listing, and the four ways a listing can be wrong.

The listing this replaces was not wrong in any way a test could see. Every one
of its 47 contracts settled, every asset class was represented, and the market
built and traded. What it was, measured, is shallow: 38 of the 47 were written
on four win rates and 20 of those on a single one, so a trader who priced that
one subject had priced two fifths of the exchange, and an experiment that called
the remainder independent observations was counting the same number over again.

So these tests check the properties that failure had no way of tripping. That a
contract settles at all and lands inside the range its collateral was computed
from, since settlement raises rather than voids on a bound declared too tight
and a short would have been undercollateralised the whole time. That the nine
asset classes are actually all there rather than nearly. That no two contracts
are the same claim under two names. That the statistical symbols and the per
match symbols cannot be confused for one another on the one venue that lists
both. And the specific bound this world makes it easy to get wrong: a volume
contract that pools two formats settles at 10,635 against a declared ceiling of
8,640, and the only reason nothing on the board does that is that every volume
contract names a format.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import pytest

from arena.contracts.payoff import Binary, Linear
from arena.contracts.underlying import Single
from arena.determinism import canonical_json
from arena.exchange.session import SessionState
from arena.market.instrument import InstrumentClass
from arena.market.match_book import list_match
from arena.settlement.engine import (
    ReferenceMismatch,
    SettlementOutOfBounds,
    distributions,
    settle,
)
from arena.sim.time import seconds
from arena.worlds.circuit.metrics import metric_ref
from arena.worlds.circuit.modes import FORMATS
from arena.worlds.circuit.oracle import CALENDAR, CircuitOracle
from arena.worlds.circuit.roster import ROSTER

from dashboard.build_market import (
    BAND,
    DELIVERY_WEEKS,
    INDIVIDUAL,
    SHARE,
    VOLUME,
    WINDOW,
    WORLD_SEED,
    _spec,
    build,
    instruments,
    prior_levels,
    true_levels,
    true_values,
)


@pytest.fixture(scope="module")
def listed():
    return instruments()


@pytest.fixture(scope="module")
def circuit_oracle():
    """An oracle built here rather than borrowed from the listing module.

    Constructed independently on purpose: if the contracts only settled against
    the module's own cached oracle, the reference identity every spec pins would
    be a formality. Settling against a separately constructed one for the same
    season is the statement that the pin means something.
    """
    return CircuitOracle(WORLD_SEED)


def _shape(instrument) -> str:
    """What a contract is, with its name taken off.

    Underlying, payoff, window and distribution schedule, which is the whole of
    what decides a settlement. Two contracts agreeing on all four are the same
    claim listed twice under different tickers.
    """
    spec = instrument.spec
    return canonical_json(
        {
            "underlying": spec.underlying.to_dict(),
            "payoff": spec.payoff.to_dict(),
            "window": spec.window.to_dict(),
            "distribution": (
                None if spec.distribution is None else spec.distribution.to_dict()
            ),
        }
    )


def _underlying(instrument) -> str:
    return canonical_json(instrument.spec.underlying.to_dict())


# ──────────────────────────────────────────────────────────────────────────
# Settlement
# ──────────────────────────────────────────────────────────────────────────


def test_every_instrument_settles_inside_its_declared_bounds(listed, circuit_oracle):
    """Nothing voids, nothing raises, and nothing lands outside its range.

    The bounds are not documentation. A position's collateral is the distance
    from its price to the end of this interval, so a bound declared too wide
    only overcharges a short while one declared too narrow means every short in
    the contract was undercollateralised and the engine finds out at expiry, by
    raising. Checking the whole listing rather than a sample is the only version
    of this test worth having, because the one contract nobody checked is the
    one whose bound was copied from a metric it is not written on.

    Measured over the four week window on seed 7, the extremes are
    `HALCYON_FORMAT_SPD` at -5,610.50 inside [-10000, 10000] and
    `RIFT_SOLO_VOL_W4` at 1,686 inside [0, 8640]. Every one of the 50 settles.
    """
    assert len(listed) == 50, "the recorded measurements below are for 50 contracts"

    for instrument in listed:
        result = settle(instrument.spec, circuit_oracle)
        assert result.settled, (
            f"{instrument.symbol} voided: {result.void_reason}. A listing that "
            "cannot be settled is a listing whose prices resolve to nothing."
        )
        low, high = instrument.spec.settlement_bounds
        assert low <= result.settlement_value <= high, (
            f"{instrument.symbol} settled at {result.settlement_value} outside "
            f"[{low}, {high}]"
        )
        # On the grid the exchange can represent, not merely inside the range.
        assert result.settlement_value % instrument.tick_size == 0


def test_a_share_pays_a_stream_that_stays_inside_its_schedule(listed, circuit_oracle):
    """Four payments, each measured on its own week, none of them equal.

    A flat stream would say the four windows were never resolved separately, and
    a share whose periods cannot differ is a future with extra steps. Measured:
    123.00, 126.50, 123.75 and 115.50 against a schedule declaring [0, 1000] a
    period, so the weeks differ by up to 9% and every one is well inside.

    The range matters more here than anywhere else, because a payment moves cash
    before settlement and `Venue.distribute` lowers the bounds every short is
    collateralised against by whatever was paid. A payment outside the declared
    range would move those bounds silently, and unlike a settlement it cannot be
    walked back.
    """
    shares = [i for i in listed if i.spec.distribution is not None]
    assert len(shares) == 1
    share = shares[0]

    paid = distributions(share.spec, circuit_oracle)
    assert len(paid) == len(DELIVERY_WEEKS) == 4
    assert len(set(paid)) > 1, f"every week paid the same: {paid}"

    low, high = share.spec.distribution.payoff.bounds(share.spec.underlying.bounds())
    for amount in paid:
        assert low <= float(amount) <= high


def test_the_share_is_exactly_the_sum_of_its_weekly_legs(listed, circuit_oracle):
    """An identity, not an approximation, and that is why the legs are listed.

    Without them the only relation available is "the share is worth 0.4 times
    the four week future", which is false in the last digits: the four weekly
    rates are each appearance weighted, so they do not average to the four week
    rate. Measured on this season the share pays 488.75 where 0.4 times the four
    week future is 488.50, a gap of 0.25 on a claim quoted in quarters. Small,
    and small is exactly what makes it dangerous to trade as though it were
    exact. Against the legs the equation closes to the cent.
    """
    by_symbol = {i.symbol: i for i in listed}
    share = by_symbol[f"{SHARE}_{INDIVIDUAL.upper()}_EQ"]
    legs = [
        by_symbol[f"{SHARE}_{INDIVIDUAL.upper()}_W{n + 1}"]
        for n in range(len(DELIVERY_WEEKS))
    ]

    paid = distributions(share.spec, circuit_oracle)
    for amount, leg in zip(paid, legs):
        settled = settle(leg.spec, circuit_oracle)
        assert settled.settlement_value == amount, (
            f"{leg.symbol} settles at {settled.settlement_value} where the share "
            f"pays {amount} for the same week"
        )

    four_week = by_symbol[f"{SHARE}_{INDIVIDUAL.upper()}_WR"]
    approximate = 0.4 * float(settle(four_week.spec, circuit_oracle).settlement_value)
    assert sum(float(a) for a in paid) != approximate


# ──────────────────────────────────────────────────────────────────────────
# Breadth
# ──────────────────────────────────────────────────────────────────────────


def test_the_listing_spans_every_asset_class_the_venue_claims(listed):
    """Nine classes on one matching engine, and the nine are read off the code.

    Derived from `InstrumentClass` rather than typed out, so a tenth class added
    to the venue fails here until the listing carries one. That is the right
    pressure: the claim the exchange makes is about the matching engine handling
    every class, and a class nothing is listed in is a claim nobody can check.

    Measured on this listing: 19 futures, 10 event contracts, 5 calls, 5 puts, 5
    commodities, 3 volatility contracts, and one each of spread, index and
    equity.
    """
    declared = {
        value
        for name, value in vars(InstrumentClass).items()
        if not name.startswith("_") and isinstance(value, str)
    }
    assert len(declared) == 9

    present = Counter(i.instrument_class for i in listed)
    assert set(present) == declared, f"missing: {sorted(declared - set(present))}"
    assert min(present.values()) >= 1


def test_no_two_contracts_are_the_same_claim_under_two_names(listed):
    """The redundancy that made the old listing shallow, checked directly.

    Two contracts sharing an underlying is normal and is what an option chain
    is. Two sharing an underlying *and* a payoff is a duplicate unless they
    differ in the window they observe, which is what a term structure is: the
    volume delivered in one week is a different thing from the volume delivered
    in the next, and the four weekly commodities are four contracts for exactly
    that reason.

    So the check is in two parts. Every (underlying, payoff) pair that repeats
    has to repeat across different windows, and no (underlying, payoff, window,
    distribution) may repeat at all. Measured on this listing: 50 contracts, 50
    distinct shapes, and the only repeated (underlying, payoff) pairs are the
    two volume curves and the share's weekly legs.
    """
    shapes = defaultdict(list)
    for instrument in listed:
        shapes[_shape(instrument)].append(instrument.symbol)
    duplicated = {tuple(v) for v in shapes.values() if len(v) > 1}
    assert not duplicated, f"the same claim listed twice: {duplicated}"

    pairs = defaultdict(list)
    for instrument in listed:
        key = canonical_json(
            {
                "underlying": instrument.spec.underlying.to_dict(),
                "payoff": instrument.spec.payoff.to_dict(),
            }
        )
        pairs[key].append(instrument)
    for group in pairs.values():
        if len(group) == 1:
            continue
        windows = {i.spec.window for i in group}
        assert len(windows) == len(group), (
            "contracts sharing an underlying and a payoff must differ in the "
            f"window they observe: {[i.symbol for i in group]}"
        )


def test_the_listing_is_written_on_many_underlyings_rather_than_many_strikes(listed):
    """Breadth measured, because breadth is the whole point of the rewrite.

    This world carries one latent scalar per competitor per format, so within a
    format every skill statistic ranks the field almost identically: over the
    contract window on seed 7 the solo win rate and solo eliminations per match
    agree at Spearman +0.993. A second strike on a win rate therefore adds an
    instrument and no information, and the previous listing was built almost
    entirely out of second strikes.

    Measured, old against new: 47 contracts on 10 underlyings with 20 of them on
    a single win rate, against 50 contracts on 22 underlyings with 7 on the
    largest. The thresholds below are the measurements with room to move, not
    targets; if a change moves them, re-measure and record the new number here
    rather than loosening them.
    """
    by_underlying = defaultdict(list)
    for instrument in listed:
        by_underlying[_underlying(instrument)].append(instrument.symbol)

    assert len(by_underlying) >= 20, (
        f"{len(by_underlying)} distinct underlyings across {len(listed)} "
        "contracts; measured at 22"
    )
    largest = max(by_underlying.values(), key=len)
    assert len(largest) <= len(listed) // 5, (
        f"{len(largest)} contracts sit on one underlying: {largest}. Measured at "
        "7 of 50, against 20 of 47 before."
    )


# ──────────────────────────────────────────────────────────────────────────
# Bounds that this world makes it easy to get wrong
# ──────────────────────────────────────────────────────────────────────────


def test_a_volume_contract_that_pooled_formats_would_breach_its_bound(circuit_oracle):
    """Why every volume contract on the board names a format.

    `match_volume` is the one metric allowed to pool formats, because a count of
    entries means the same thing whichever mode produced it, and its declared
    ceiling is the 8,640 matches a window may hold. That ceiling is derived for
    one format: a subject cannot enter more matches than the window holds. With
    two formats running the same schedule it is simply false, because each seats
    the competitor separately.

    Measured over the four week window on seed 7: a pooled count comes to 10,635
    appearances against a declared 8,640, and settlement raises rather than
    voiding, because a value outside the declared range means the collateral
    held against the contract was computed from a false premise. Across the
    whole twelve competitor field the pooled count runs 10,635 to 10,847, so
    this is not one unlucky subject. Naming the individual format holds the same
    subject to 6,700.

    This is a live hazard rather than a hypothetical, and it is the reason the
    listing names a format on every volume contract. The declared bound lives in
    `arena/worlds/circuit/metrics.py`, whose comment derives it for a single
    format without saying so.
    """
    pooled = _spec(
        "POOLED_VOLUME",
        Single(metric_ref("match_volume", VOLUME)),
        Linear(1.0),
        window=WINDOW,
    )
    with pytest.raises(SettlementOutOfBounds):
        settle(pooled, circuit_oracle)

    named = _spec(
        "NAMED_VOLUME",
        Single(metric_ref("match_volume", VOLUME, modes=(INDIVIDUAL,))),
        Linear(1.0),
        window=WINDOW,
    )
    result = settle(named, circuit_oracle)
    low, high = named.settlement_bounds
    assert result.settled and low <= result.settlement_value <= high


def test_a_quantity_prior_is_carried_over_in_the_right_unit(listed):
    """The units trap, on the one metric here that has units.

    A rate is scale invariant in window length, so re-dating a win rate onto the
    prior window measures the same kind of number and needs no conversion. A
    count is not, and the last time that distinction was declared upstream and
    ignored here the market wore it: a one week commodity settling at 71.09 was
    handed a prior of 274.92, off by 3.87x against a window ratio of 3.87, and
    every informed trader traded the artifact.

    Measured on this listing: the weekly volume contracts settle between 1,648
    and 1,686 and their prior is 1,679, which is the four week count of 6,716
    divided by four. A prior of four thousand something would be the same bug
    wearing a new world's clothes.
    """
    levels = true_levels(listed)
    priors = prior_levels(listed)

    weekly = [i for i in listed if i.instrument_class == InstrumentClass.COMMODITY]
    assert weekly, "the listing carries no commodity to check the unit on"
    for instrument in weekly:
        settled = levels[instrument.symbol]
        prior = priors[instrument.symbol]
        assert 0.5 * settled < prior < 2.0 * settled, (
            f"{instrument.symbol} settles at {settled} and its prior says "
            f"{prior}; a quantity prior off by the window ratio is the unit bug"
        )

    # And the rates are carried straight across, which is the other half of the
    # same statement: a scale applied to a rate would be just as wrong.
    rates = [i for i in listed if i.instrument_class == InstrumentClass.FUTURE]
    for instrument in rates:
        atoms = instrument.spec.underlying.atoms()
        if {a.kind for a in atoms} != {"rate"}:
            continue
        assert 0.5 * levels[instrument.symbol] < priors[instrument.symbol]


def test_the_prior_is_a_real_measurement_and_is_not_the_answer(listed):
    """Every contract opens on history, and history is wrong.

    Both halves matter. A listing with no prior leaves the informed traders with
    nothing to open on, which is what happens if the contract window opens on
    the calendar epoch: the four weeks before it name matches that were never
    played and the oracle refuses them. A prior equal to the settlement leaves
    the market nothing to discover.

    Measured over the season's first four weeks against its second four: the
    front subject's individual win rate ran 0.1469 and settles at 0.1398, and
    every one of the 50 contracts gets a prior while none of the 22 underlyings
    lands on its settlement exactly.
    """
    levels = true_levels(listed)
    priors = prior_levels(listed)

    missing = [i.symbol for i in listed if i.symbol not in priors]
    assert not missing, f"no prior for {missing}"

    identical = [s for s, level in levels.items() if priors[s] == level]
    assert not identical, (
        f"{identical} opened on their own settlement, so there is nothing in "
        "them to discover"
    )


# ──────────────────────────────────────────────────────────────────────────
# Two families, one venue
# ──────────────────────────────────────────────────────────────────────────


def test_statistical_symbols_cannot_collide_with_match_symbols(listed):
    """One venue lists both families and a symbol is the only key it has.

    A match names itself by format and match id, `SOLO0_WIN_WISP` and
    `OBJECTIVE0_ELIM_BASTION_GT0`, and the operator lists a fresh match every
    ninety simulated seconds for as long as the session runs. So the collision
    is not a one time check against a fixed set: any statistical symbol shaped
    like a format tag followed by digits would collide with whichever match
    eventually carries that id, and it would collide by silently replacing a
    contract in the registry rather than by raising.

    Both halves are checked. The listed symbols are disjoint from the books of
    several real matches in every registered format, and no listed symbol has
    the shape a match tag has, which covers the ids nobody listed yet.
    """
    symbols = {i.symbol for i in listed}
    assert len(symbols) == len(listed), "the listing repeats a symbol"

    for format_name in FORMATS:
        for match_id in (0, 1, 2, 17, 400):
            book = list_match(WORLD_SEED, match_id, format_name)
            clash = symbols & set(book.symbols)
            assert not clash, f"{format_name}{match_id} collides on {clash}"

    tag = re.compile(
        r"^(" + "|".join(sorted(name.upper() for name in FORMATS)) + r")\d+_"
    )
    shaped = [symbol for symbol in symbols if tag.match(symbol)]
    assert not shaped, (
        f"{shaped} are shaped like a match tag, so a match with that id would "
        "overwrite them when the operator lists it"
    )


def test_a_build_with_matches_carries_both_families_without_conflict():
    """The two families run on one clock that ticks at two different rates.

    A statistical contract observes four weeks and a match occupies five
    minutes, and the venue has one calendar for both. That calendar covers a
    contract day every `session_seconds`, which is 144 contract seconds per
    simulated second against a match schedule advancing 3.3 of them, so a clock
    started at the observation window would already be four weeks past every
    match ever listed and `_enforce_lifecycle` would close each match contract
    in the instant the operator listed it. Starting it on publication day, a day
    before the circuit opens, is what buys the session.

    Measured on seed 7: 120 symbols listed, 50 statistical and 70 across the two
    open matches, none of either family closed at the open, and conservation
    exactly zero after two simulated seconds of trading.

    It was 358 and 308 before the match listing was trimmed, and the balance is
    the point of the number rather than the number itself. Two open matches used
    to be six times the whole statistical listing, so a board built to show nine
    asset classes was a match board with the other eight in the margin. Now the
    two families are within a half of each other, and the assertion is an exact
    count rather than a floor because a floor is what let the old figure drift
    without anybody noticing it had.
    """
    statistical = {i.symbol for i in instruments()}
    market = build(seed=WORLD_SEED, matches=True)
    market.kernel.start()

    live = set(market.venue.registry.symbols)
    match_symbols = live - statistical
    assert statistical <= live
    assert len(match_symbols) == 70, (
        f"{len(match_symbols)} match contracts listed; the operator opens one "
        "match per format, and those carry 50 and 20"
    )
    assert len(live) == len(statistical) + len(match_symbols) == 120

    closed = [
        symbol for symbol in live if market.venue.session(symbol) is SessionState.CLOSED
    ]
    assert not closed, (
        f"{len(closed)} contracts were closed the moment they listed, which is "
        "the venue clock running past a family's expiry rather than a market"
    )

    market.kernel.advance(until=seconds(2))
    assert market.venue.conservation_check() == 0


def test_a_contract_cannot_be_settled_against_another_season(listed):
    """The season is the seed, and the reference pin is what says so.

    In a collected world the reference is a frozen snapshot and pinning it stops
    a contract being settled against a different crawl. Here the season is
    generated, so the same protection has to come from the seed: the oracle
    digests the seed, the calendar and the shape of every registered format into
    its reference id, and the engine refuses a contract that does not match.

    Without the pin the failure would be silent and plausible. A contract listed
    on one season settled against another returns a perfectly well formed number
    about a competitor who never played those matches.
    """
    other = CircuitOracle(WORLD_SEED + 1)
    assert other.reference_id != CircuitOracle(WORLD_SEED).reference_id

    with pytest.raises(ReferenceMismatch):
        settle(listed[0].spec, other)


def test_a_season_other_than_the_default_still_gets_its_values(listed):
    """The other half of pinning the season to the seed.

    Refusing a mismatched oracle is only half a design. The other half is that a
    caller holding a contract has to be able to find the oracle that can answer
    for it, and a contract pins a digest rather than a seed. The dashboard is
    exactly such a caller: it settles instruments one at a time and drops the
    ones it cannot answer for, so on any seed but the default it would have
    shown a full board of live books with the settlement column silently empty
    and nothing raising anywhere.

    Measured: `true_values` with no season named values all 50 contracts of
    season 11 as well as all 50 of season 7, and only 6 of the 50 come out the
    same in both, which is what says these are two seasons rather than one
    answered twice.
    """
    other_season = instruments(WORLD_SEED + 4)
    assert {i.symbol for i in other_season} == {i.symbol for i in listed}

    ours = true_values(listed)
    theirs = true_values(other_season)
    assert len(ours) == len(theirs) == len(listed)

    agreed = sum(1 for symbol in ours if ours[symbol] == theirs.get(symbol))
    assert agreed < len(listed) // 2, (
        f"{agreed} of {len(listed)} contracts settle identically in two "
        "different seasons, so one of them is being answered by the other's "
        "oracle"
    )

    # And the dashboard's own shape: one at a time, dropping refusals.
    one_by_one: dict[str, float] = {}
    for instrument in other_season:
        try:
            one_by_one.update(true_values([instrument]))
        except Exception:  # noqa: BLE001 - the caller this imitates does the same
            continue
    assert len(one_by_one) == len(other_season)


def test_an_event_contract_hangs_on_a_threshold_the_format_supplies(listed):
    """No rung on this board was read off a settlement value.

    A binary struck where a rate is known to settle is a countdown rather than a
    market, and the way that happens is somebody choosing a threshold after
    looking. Every threshold here comes from arithmetic that exists before a
    match is played: a format's neutral point is its winning slots over all its
    slots, 1 of 10 in the elimination mode and 3 of 6 in the team mode, and a
    scheduling contract sits at the window's matches times the field size over
    the roster size.

    Checked structurally rather than by reading the numbers back, so a threshold
    quietly nudged toward an answer fails here. Measured on this listing: the
    team rungs sit at 0.500 exactly and split the six subjects 3 to 3, and the
    individual rungs at 0.060, 0.100 and 0.140 settle 1, 1 and 0 on a subject
    that came in at 0.1398.
    """
    rungs = [i for i in listed if isinstance(i.spec.payoff, Binary)]
    assert len(rungs) == 10

    # Rebuilt here from the registry and the calendar rather than imported from
    # the listing, so a threshold moved by hand has nothing to agree with.
    allowed = set()
    for fmt in FORMATS.values():
        winning_slots = 1 if fmt.team_size == 1 else fmt.team_size
        neutral = winning_slots / fmt.entrants
        allowed.update({neutral - BAND, neutral, neutral + BAND})
        for week in DELIVERY_WEEKS:
            matches = len(CALENDAR.match_ids(week))
            allowed.add(float(round(matches * fmt.entrants / len(ROSTER))))

    for instrument in rungs:
        threshold = instrument.spec.payoff.threshold
        assert any(abs(threshold - candidate) < 1e-9 for candidate in allowed), (
            f"{instrument.symbol} is struck at {threshold}, which is not a level "
            "the format or the calendar hands you"
        )
