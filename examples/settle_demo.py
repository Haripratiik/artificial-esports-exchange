"""Settle a small book of contracts across two adjacent weeks of the circuit.

Run:  python examples/settle_demo.py

Shows the three instrument families settling through one mechanism, the metric
diagnostics that make a settlement auditable, and the same contract moving
between two adjacent observation windows because the matches inside them are
different matches.

Nothing here is collected. The circuit is generated from a seed, so the season
this settles against is reproducible by anyone holding the seed, and running
this file twice prints the same digest.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from arena.contracts.payoff import Binary, Linear
from arena.contracts.spec import ContractSpec, DataPolicy, ObservationWindow
from arena.contracts.underlying import Difference, Single
from arena.settlement.engine import settle
from arena.worlds.circuit.metrics import metric_ref
from arena.worlds.circuit.oracle import CircuitOracle

UTC = timezone.utc

SEED = 7
POLICY = DataPolicy(min_sample_size=500)

# Both legs name a format. A win rate sits at a neutral 0.100 in the ten-way
# elimination and 0.500 in the three-a-side objective, so an average over a
# window holding both would move with whatever the schedule happened to run
# rather than with the competitor. The oracle refuses the pooled version rather
# than returning a number that looks fine, which is why the filter is here and
# not a convention.
VANTA = Single(metric_ref("win_rate", "VANTA", modes=("objective",)))
QUILL = Single(metric_ref("win_rate", "QUILL", modes=("objective",)))

# Signed, and bounded in [-1, 1] rather than the [0, 1] a rate defaults to.
# Built through metric_ref for exactly that reason: declaring the range wrongly
# would size a short position's collateral against the wrong worst case.
MARGIN = Single(metric_ref("score_margin", "VANTA", modes=("objective",)))


def spec(contract_id, underlying, payoff, window, reference_id) -> ContractSpec:
    return ContractSpec(
        contract_id=contract_id,
        underlying=underlying,
        payoff=payoff,
        window=window,
        policy=POLICY,
        reference_id=reference_id,
        published_at=window.start - timedelta(days=1),
        tick_size="0.25",
    )


def main() -> None:
    oracle = CircuitOracle(seed=SEED)
    calendar = oracle.calendar
    reference_id = oracle.reference_id

    print(f"season    seed {SEED}, reference {reference_id}")
    print(
        f"calendar  one match every {calendar.period.total_seconds() / 60:.0f} minutes "
        f"from {calendar.epoch.date()}, {calendar.per_day:.0f} per day per format"
    )

    windows = {
        "week 1": ObservationWindow(
            datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 9, 8, tzinfo=UTC)
        ),
        "week 2": ObservationWindow(
            datetime(2026, 9, 8, tzinfo=UTC), datetime(2026, 9, 15, tzinfo=UTC)
        ),
    }

    print()
    for label, window in windows.items():
        future = settle(
            spec("VANTA_OBJECTIVE_WR", VANTA, Linear(scale=10_000.0), window, reference_id),
            oracle,
        )
        binary = settle(
            spec("VANTA_OBJECTIVE_GT480", VANTA, Binary(">", 0.48), window, reference_id), oracle
        )
        spread = settle(
            spec(
                "VANTA_QUILL_OBJECTIVE_SPD",
                Difference(VANTA, QUILL),
                Linear(scale=10_000.0),
                window,
                reference_id,
            ),
            oracle,
        )
        margin = settle(
            spec("VANTA_OBJECTIVE_MARGIN", MARGIN, Linear(scale=10_000.0), window, reference_id),
            oracle,
        )
        diagnostics = dict(future.resolutions[0].diagnostics)

        print(
            f"{label}    future {future.settlement_value:>9}"
            f"   binary(>0.48) {binary.settlement_value:>4}"
            f"   spread {spread.settlement_value:>9}"
            f"   margin {margin.settlement_value:>9}"
        )
        print(
            f"          level {future.underlying_level:.6f}"
            f"   entries {future.resolutions[0].sample_size:>6,}"
            f"   wins {diagnostics['wins']:>5,}"
            f"   neutral {diagnostics['neutral_level']:.3f}"
            f"   matches scanned {diagnostics['matches_scanned']:>6,}"
        )

    window = windows["week 1"]
    once = settle(
        spec("VANTA_OBJECTIVE_WR", VANTA, Linear(scale=10_000.0), window, reference_id), oracle
    )
    twice = settle(
        spec("VANTA_OBJECTIVE_WR", VANTA, Linear(scale=10_000.0), window, reference_id), oracle
    )
    print(f"\ndeterministic: {once.result_digest == twice.result_digest}")
    print(f"result digest: {once.result_digest}")
    print(f"spec digest:   {once.spec_digest}")


if __name__ == "__main__":
    main()
