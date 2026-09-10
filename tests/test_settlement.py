"""End-to-end settlement tests.

The deliverable is one sentence: given a contract and a season, settlement is
deterministic. These tests are what that sentence means operationally: not
just that the number is stable, but that it is provably tied to the exact spec
and to the exact run of matches that produced it, and that it refuses to appear
at all when the evidence is too thin.

What "the exact evidence" means changed when the world did. A collected world
pinned a dataset's bytes and a frozen reference snapshot; a generated one pins a
seed, a calendar and the shape of every registered format, and the oracle
digests those three into its `reference_id`. So the contract still names the
evidence it will be settled against, and settling it against another season is
still a hard error rather than a plausible number.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from arena.contracts.payoff import Binary, Linear
from arena.contracts.spec import (
    ContractSpec,
    DataPolicy,
    DistributionSchedule,
    ObservationWindow,
)
from arena.contracts.underlying import Basket, Difference, MetricRef, Single
from arena.settlement.engine import (
    ReferenceLookahead,
    ReferenceMismatch,
    SettlementOutOfBounds,
    distributions,
    settle,
)
from arena.settlement.oracle import MetricResolution
from arena.settlement.result import SettlementStatus
from arena.worlds.circuit.oracle import CALENDAR, CircuitOracle

SEED = 7
# The identity of the season every spec below pins itself to. Written out
# rather than computed, so a change to the calendar or to a registered format
# shows up here as a failing test rather than as two things agreeing with each
# other about a value nobody stated.
REFERENCE = "circuit-40cffa27511c1621"

# The team format, whose neutral point is 0.500, so a win rate on it sits in the
# middle of its own range and a binary struck at 0.30 is a real question rather
# than a formality. EMBER settles at 0.4933 over the fixture window and RIFT at
# 0.5315, which is what the spread and basket tests below difference.
EMBER_WR = MetricRef(metric="win_rate", subject="EMBER", modes=("objective",))
RIFT_WR = MetricRef(metric="win_rate", subject="RIFT", modes=("objective",))
HALCYON_WR = MetricRef(metric="win_rate", subject="HALCYON", modes=("objective",))


def make_spec(window, published_at, **overrides):
    defaults = dict(
        contract_id="EMBER_OBJECTIVE_WR_2026W36",
        underlying=Single(EMBER_WR),
        payoff=Linear(scale=10_000.0),
        window=window,
        # `min_sample_size` alone. The two stratum bars guard the composition of
        # a standardized rate, and a circuit metric refuses a window that mixes
        # formats rather than reweighting it, so there is nothing here for them
        # to check. `conftest.policy` argues this at length.
        policy=DataPolicy(min_sample_size=1_000),
        reference_id=REFERENCE,
        published_at=published_at,
        tick_size="0.25",
    )
    defaults.update(overrides)
    return ContractSpec(**defaults)


# --------------------------------------------------------------------------
# The core deliverable
# --------------------------------------------------------------------------


def test_settlement_is_deterministic(oracle, window, published_at):
    spec = make_spec(window, published_at)
    first = settle(spec, oracle)
    second = settle(spec, oracle)

    assert first.status == SettlementStatus.SETTLED
    assert first.result_digest == second.result_digest
    assert first.to_dict() == second.to_dict()


def test_a_second_oracle_on_the_same_seed_settles_identically(window, published_at):
    """Determinism across processes, not merely across two calls to one memo.

    The oracle caches the matches it has replayed, so settling twice against one
    instance could agree because nothing was recomputed. A fresh oracle replays
    all 8,064 matches from the seed and has to arrive at the same digest.
    """
    spec = make_spec(window, published_at)
    first = settle(spec, CircuitOracle(SEED))
    second = settle(spec, CircuitOracle(SEED))
    assert first.result_digest == second.result_digest


def test_settlement_lands_on_the_tick_grid(oracle, window, published_at):
    spec = make_spec(window, published_at, tick_size="0.25")
    result = settle(spec, oracle)
    assert result.settlement_value is not None
    assert result.settlement_value % Decimal("0.25") == 0


def test_settlement_value_tracks_the_underlying(oracle, window, published_at):
    """A 10000x linear future on a team win rate near 0.49 settles near 4,933.

    Measured on seed 7 over the first four weeks of the season: EMBER is drawn
    into 3,967 of the 8,064 objective matches and wins 1,957 of them, a rate of
    0.493320, which the quarter-point grid takes to 4,933.25.
    """
    result = settle(make_spec(window, published_at), oracle)
    assert result.underlying_level is not None
    assert 0.45 < result.underlying_level < 0.65
    assert result.settlement_value == Decimal("4933.25")
    assert result.settlement_value == pytest.approx(
        Decimal(str(round(result.underlying_level * 10_000, 2))), abs=Decimal("0.5")
    )


def test_result_carries_full_provenance(oracle, window, published_at):
    """The evidence is a run of match ids, and the record has to name it.

    A collected world's provenance was a file digest. A generated one's is the
    content address of the matches themselves: the seed, the format and the
    exact half-open run of ids that were replayed. Two runs agreeing on this
    digest replayed the same matches, which is the whole of what a source
    reference has to promise.
    """
    result = settle(make_spec(window, published_at), oracle)
    (resolution,) = result.resolutions
    source_ids = {source.source_id for source in resolution.sources}

    assert source_ids == {f"circuit:{SEED}:objective"}
    assert all(source.digest.startswith("sha256:") for source in resolution.sources)
    assert all(source.rows == 8_064 for source in resolution.sources)

    diagnostics = dict(resolution.diagnostics)
    assert diagnostics["reference_id"] == REFERENCE
    assert diagnostics["first_match"] == 0
    assert diagnostics["stop_match"] == 8_064
    assert diagnostics["formats_requested"] == "objective"


# --------------------------------------------------------------------------
# Digest sensitivity: the record must notice when anything material changed
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("payoff", Linear(scale=10_000.0, offset=1.0)),
        ("tick_size", "0.5"),
        ("policy", DataPolicy(min_sample_size=2_000)),
        ("contract_id", "SOMETHING_ELSE"),
    ],
)
def test_changing_the_spec_changes_its_digest(window, published_at, field, value):
    base = make_spec(window, published_at)
    mutated = make_spec(window, published_at, **{field: value})
    assert base.spec_digest != mutated.spec_digest


def test_the_same_contract_on_two_seasons_is_two_records(window, published_at):
    """Settling the same claim against different evidence must be visible.

    The season *is* the evidence in a generated world, so this is the same
    check a collected world made by trimming a dataset. Measured on the fixture
    window: EMBER's objective win rate is 0.493320 on seed 7 and 0.419168 on
    seed 41, so the two records differ in the number as well as in the digest,
    and a digest that matched across them would mean the record was not
    covering the evidence at all.
    """
    seven = CircuitOracle(SEED)
    forty_one = CircuitOracle(41)
    assert seven.reference_id != forty_one.reference_id

    on_seven = settle(make_spec(window, published_at), seven)
    on_forty_one = settle(
        make_spec(window, published_at, reference_id=forty_one.reference_id),
        forty_one,
    )

    assert on_seven.result_digest != on_forty_one.result_digest
    assert on_seven.underlying_level != on_forty_one.underlying_level


def test_reference_snapshot_is_covered_by_the_spec_digest(window, published_at):
    base = make_spec(window, published_at)
    other = make_spec(window, published_at, reference_id="circuit-somethingelse")
    assert base.spec_digest != other.spec_digest


def test_oracle_pinned_to_a_different_reference_is_rejected(
    oracle, window, published_at
):
    """Wiring the experiment up wrong raises; it does not quietly void."""
    spec = make_spec(window, published_at, reference_id="circuit-SOMETHING-ELSE")
    with pytest.raises(ReferenceMismatch, match="pinned to reference"):
        settle(spec, oracle)


def test_a_window_that_opens_before_the_circuit_did_is_rejected(oracle):
    """The most dangerous lookahead channel, because it improves results.

    A reference estimated from data inside its own observation window has seen
    the outcome, so its weights and priors encode the answer. The settlement it
    produces would look *better*, not broken, which is exactly why this has to
    be a hard error rather than something a reviewer is expected to notice.

    In a generated world the two ends of that check meet. The oracle's reference
    is as of the epoch, because the generator was fixed when the circuit opened,
    and the calendar refuses a window opening before the epoch because the
    matches it names were never played. So the engine's lookahead guard and the
    calendar's own refusal are two statements of one fact, and this is the guard
    firing first.
    """
    early = ObservationWindow(
        start=CALENDAR.epoch - timedelta(days=1),
        end=CALENDAR.epoch + timedelta(days=27),
    )
    spec = make_spec(early, early.start - timedelta(days=1))

    with pytest.raises(ReferenceLookahead, match="after the window had already begun"):
        settle(spec, oracle)


def test_a_window_that_opens_exactly_when_the_circuit_did_is_allowed(oracle):
    """The boundary is inclusive: a season fixed *at* the open has seen nothing."""
    boundary = ObservationWindow(
        start=CALENDAR.epoch, end=CALENDAR.epoch + timedelta(days=28)
    )
    spec = make_spec(boundary, boundary.start - timedelta(days=1))
    assert settle(spec, oracle).settled


# --------------------------------------------------------------------------
# Voiding: refusing to settle is a feature
# --------------------------------------------------------------------------


def test_voids_when_sample_size_is_below_the_bar(oracle, window, published_at):
    spec = make_spec(
        window, published_at, policy=DataPolicy(min_sample_size=10**12)
    )
    result = settle(spec, oracle)

    assert result.status == SettlementStatus.VOID
    assert result.settlement_value is None
    assert "sample size" in result.void_reason


def test_voids_when_no_matches_fall_inside_the_window(oracle, published_at):
    """A window narrower than the gap between matches names nothing.

    The collected world reached this state with a window past the end of the
    crawl. A generated season has no end, so the way to name no evidence is to
    name an interval the schedule steps over: matches run every five minutes, so
    a four minute window starting a minute after one has been played holds none
    of them. The oracle says so rather than widening the window to the nearest
    match, which would settle something the contract did not name.
    """
    gap = ObservationWindow(
        start=CALENDAR.epoch + timedelta(seconds=60),
        end=CALENDAR.epoch + timedelta(seconds=300),
    )
    spec = make_spec(gap, published_at)
    result = settle(spec, oracle)

    assert result.status == SettlementStatus.VOID
    assert "no matches in window" in result.void_reason


def test_voids_on_unknown_subject(oracle, window, published_at):
    """A competitor nobody entered is refused, never returned as a zero.

    A zero is a well formed number that settles, and a contract on somebody who
    never played would pay out on it.
    """
    spec = make_spec(
        window,
        published_at,
        underlying=Single(
            MetricRef(metric="win_rate", subject="NOBODY", modes=("objective",))
        ),
    )
    result = settle(spec, oracle)
    assert result.status == SettlementStatus.VOID
    assert result.settlement_value is None


def test_voids_on_unknown_metric(oracle, window, published_at):
    spec = make_spec(
        window,
        published_at,
        underlying=Single(
            MetricRef(metric="vibes", subject="EMBER", modes=("objective",))
        ),
    )
    result = settle(spec, oracle)
    assert result.status == SettlementStatus.VOID
    assert "unknown metric" in result.void_reason


def test_voids_on_a_format_that_was_never_registered(oracle, window, published_at):
    """The modes filter carries format names in this world, so a typo is a void.

    Settling it unfiltered would measure a pooled average over both formats,
    which is a different number: a win in the ten way elimination sits against a
    neutral point of 0.100 and a win in the three a side objective against
    0.500.
    """
    spec = make_spec(
        window,
        published_at,
        underlying=Single(
            MetricRef(metric="win_rate", subject="EMBER", modes=("duos",))
        ),
    )
    result = settle(spec, oracle)
    assert result.status == SettlementStatus.VOID
    assert "unknown format" in result.void_reason


def test_void_record_keeps_the_evidence_it_did_gather(oracle, window, published_at):
    """A spread whose second leg fails still records the first leg's resolution.

    The missing subject is named to sort *after* EMBER. Atoms resolve in
    canonical order, so a name like "NOBODY" would fail before EMBER was ever
    reached and the test would pass vacuously with zero resolutions.
    """
    spec = make_spec(
        window,
        published_at,
        underlying=Difference(
            left=Single(EMBER_WR),
            right=Single(
                MetricRef(
                    metric="win_rate", subject="ZZ_MISSING", modes=("objective",)
                )
            ),
        ),
    )
    result = settle(spec, oracle)

    assert result.status == SettlementStatus.VOID
    assert len(result.resolutions) == 1
    assert result.resolutions[0].ref.subject == "EMBER"


# --------------------------------------------------------------------------
# The underlying algebra: one mechanism, three instrument families
# --------------------------------------------------------------------------


def test_binary_contract_settles_to_payout_or_zero(oracle, window, published_at):
    low = make_spec(
        window, published_at, payoff=Binary(">", threshold=0.30), tick_size="0.01"
    )
    high = make_spec(
        window, published_at, payoff=Binary(">", threshold=0.99), tick_size="0.01"
    )
    assert settle(low, oracle).settlement_value == Decimal("1")
    assert settle(high, oracle).settlement_value == Decimal("0")


def test_spread_equals_the_difference_of_its_legs(oracle, window, published_at):
    ember = settle(make_spec(window, published_at, underlying=Single(EMBER_WR)), oracle)
    rift = settle(make_spec(window, published_at, underlying=Single(RIFT_WR)), oracle)
    spread = settle(
        make_spec(
            window,
            published_at,
            underlying=Difference(Single(EMBER_WR), Single(RIFT_WR)),
        ),
        oracle,
    )
    assert spread.underlying_level == pytest.approx(
        ember.underlying_level - rift.underlying_level
    )


def test_basket_equals_its_weighted_components(oracle, window, published_at):
    legs = ((Single(EMBER_WR), 0.6), (Single(RIFT_WR), 0.4))
    index = settle(make_spec(window, published_at, underlying=Basket(legs)), oracle)

    ember = settle(make_spec(window, published_at, underlying=Single(EMBER_WR)), oracle)
    rift = settle(make_spec(window, published_at, underlying=Single(RIFT_WR)), oracle)

    assert index.underlying_level == pytest.approx(
        0.6 * ember.underlying_level + 0.4 * rift.underlying_level
    )


def test_basket_leg_order_does_not_change_the_settlement_value(
    oracle, window, published_at
):
    """Float addition is not associative, so this is a real risk, not a ritual."""
    forward = (
        (Single(EMBER_WR), 0.5),
        (Single(RIFT_WR), 0.3),
        (Single(HALCYON_WR), 0.2),
    )
    reverse = tuple(reversed(forward))

    a = settle(make_spec(window, published_at, underlying=Basket(forward)), oracle)
    b = settle(make_spec(window, published_at, underlying=Basket(reverse)), oracle)

    assert a.underlying_level == b.underlying_level
    assert a.settlement_value == b.settlement_value


# --------------------------------------------------------------------------
# Payments, which cannot be walked back
# --------------------------------------------------------------------------


class _FixedOracle:
    """Returns one level for every reference and window, whatever was asked.

    Enough to drive the distribution path past the point where a real oracle
    would have refused, which is the only way to reach the guard below: a win
    rate is wins over appearances and cannot leave [0, 1], so the metric never
    produces an out-of-range value, and a guard that has never fired is a guard
    nobody has checked.
    """

    def __init__(self, value, as_of, reference_id=REFERENCE):
        self._value = value
        self._as_of = as_of
        self._reference_id = reference_id

    @property
    def reference_id(self):
        return self._reference_id

    @property
    def reference_as_of(self):
        return self._as_of

    def resolve(self, ref, window, policy_overrides=None):
        return MetricResolution(
            ref=ref, value=self._value, sample_size=10_000, sources=()
        )


def _share(window, published_at, periods=4):
    span = (window.end - window.start) / periods
    return make_spec(
        window,
        published_at,
        contract_id="EMBER_OBJECTIVE_EQ",
        payoff=Linear(scale=0.0),
        distribution=DistributionSchedule(
            windows=tuple(
                ObservationWindow(
                    window.start + span * n, window.start + span * (n + 1)
                )
                for n in range(periods)
            ),
            payoff=Linear(scale=1_000.0),
        ),
    )


def test_each_period_is_measured_on_its_own_evidence(oracle, window, published_at):
    """A share is worth the stream, and the stream is only interesting if it moves."""
    paid = distributions(_share(window, published_at), oracle)
    assert len(paid) == 4
    assert len(set(paid)) > 1
    assert all(Decimal(0) <= amount <= Decimal(1_000) for amount in paid)


def test_a_payment_outside_the_schedules_range_is_a_hard_error(window, published_at):
    """The one guard a share never had, on the one contract whose cash moves early.

    A share's terminal payoff is Linear(0), so its settlement bounds are [0, 0]
    and the out-of-range check in `settle` can never fire on it -- yet the
    payments happen *before* settlement and `Venue.distribute` lowers the range
    every short is collateralised against by whatever was paid. So a payment
    computed from a level the contract never contemplated would silently move
    the bounds that back the whole contract, and nothing downstream would
    notice. A rate of 1.5 against a declared [0, 1] pays 1,500 on a schedule
    whose range is [0, 1000].
    """
    spec = _share(window, published_at)
    fabulist = _FixedOracle(1.5, published_at)
    with pytest.raises(SettlementOutOfBounds, match="outside the range"):
        distributions(spec, fabulist)

    # And the honest end of the same range still passes, so the guard is not
    # simply refusing everything.
    assert distributions(spec, _FixedOracle(1.0, published_at)) == (Decimal(1_000),) * 4


def test_repeated_leg_is_resolved_once(oracle, window, published_at):
    """Atom deduplication keeps oracle work proportional to distinct metrics."""
    spec = make_spec(
        window,
        published_at,
        underlying=Basket(((Single(EMBER_WR), 0.5), (Single(EMBER_WR), 0.5))),
    )
    result = settle(spec, oracle)
    assert len(result.resolutions) == 1
    assert result.underlying_level == pytest.approx(result.resolutions[0].value)
