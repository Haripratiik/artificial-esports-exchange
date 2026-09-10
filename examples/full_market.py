"""The complete loop: define contracts, trade them, settle them, see the PnL.

    python examples/full_market.py

Until now the project had two halves that never met: a contract layer that
could settle a future, and an exchange that could match orders in an abstract
instrument. Nothing said *this symbol settles by that contract*, so a trade
could never become a settlement and a settlement could never reach anyone's
account.

This runs the whole cycle on four instruments at once, all settling against the
same run of matches on the circuit:

    VANTA_OBJECTIVE_WR         linear performance future
    VANTA_OBJECTIVE_GT480      binary event contract
    VANTA_QUILL_OBJECTIVE_SPD  relative-value spread
    CIRCUIT_OBJECTIVE_IDX      weighted index

Two things are worth watching. **Collateral is exact**, because every contract
settles inside a known interval, so a short on the future is charged what it can
actually lose rather than a volatility estimate. And **value is conserved to the
unit** through trading and settlement both, which is the check that makes any
PnL figure here believable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from arena.contracts.payoff import Binary, Linear
from arena.contracts.spec import ContractSpec, DataPolicy, ObservationWindow
from arena.contracts.underlying import Basket, Difference, Single
from arena.exchange.events import Submit
from arena.exchange.types import AgentId, OrderType, Quantity, Side, TimeInForce
from arena.market.instrument import Instrument
from arena.market.venue import Venue
from arena.portfolio.money import from_money
from arena.settlement.engine import settle
from arena.worlds.circuit.metrics import metric_ref
from arena.worlds.circuit.oracle import CircuitOracle

UTC = timezone.utc
SEED = 7
POLICY = DataPolicy(min_sample_size=500)

WINDOW = ObservationWindow(
    datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 9, 29, tzinfo=UTC)
)

# Every leg names its format. A win rate is neutral at 0.500 in the three-a-side
# objective and at 0.100 in the ten-way elimination, so a contract that pooled
# them would settle on the schedule as much as on the competitor, and the oracle
# refuses to answer one rather than returning a number that reads as fine.
WR = lambda who: Single(  # noqa: E731
    metric_ref("win_rate", who, modes=("objective",))
)


def spec(contract_id, underlying, payoff, reference_id, tick="0.25") -> ContractSpec:
    return ContractSpec(
        contract_id=contract_id,
        underlying=underlying,
        payoff=payoff,
        window=WINDOW,
        policy=POLICY,
        reference_id=reference_id,
        published_at=WINDOW.start - timedelta(days=1),
        tick_size=tick,
    )


def build_instruments(reference_id: str) -> list[Instrument]:
    return [
        Instrument(
            "VANTA_OBJECTIVE_WR",
            spec("VANTA_OBJECTIVE_WR", WR("VANTA"), Linear(10_000.0), reference_id),
        ),
        Instrument(
            "VANTA_OBJECTIVE_GT480",
            spec(
                "VANTA_OBJECTIVE_GT480",
                WR("VANTA"),
                Binary(">", 0.48, payout=1.0),
                reference_id,
                tick="0.01",
            ),
        ),
        Instrument(
            "VANTA_QUILL_OBJECTIVE_SPD",
            spec(
                "VANTA_QUILL_OBJECTIVE_SPD",
                Difference(WR("VANTA"), WR("QUILL")),
                Linear(10_000.0),
                reference_id,
            ),
        ),
        Instrument(
            "CIRCUIT_OBJECTIVE_IDX",
            spec(
                "CIRCUIT_OBJECTIVE_IDX",
                Basket(((WR("VANTA"), 0.5), (WR("TALON"), 0.3), (WR("EMBER"), 0.2))),
                Linear(10_000.0),
                reference_id,
            ),
        ),
    ]


def order(agent, side, ticks, qty) -> Submit:
    return Submit(agent, side, Quantity(qty), ticks, OrderType.LIMIT, TimeInForce.GTC)


def main() -> None:
    oracle = CircuitOracle(seed=SEED)

    venue = Venue("arena", starting_cash=1_000_000)
    instruments = build_instruments(oracle.reference_id)
    for instrument in instruments:
        venue.list_instrument(instrument)

    print("LISTED")
    for instrument in instruments:
        low, high = instrument.settlement_bounds
        print(
            f"  {instrument.symbol:<26} {instrument.instrument_class:<8} "
            f"tick {instrument.tick_size:<6} settles in [{low}, {high}]"
        )

    # Two traders take opposite sides of everything, at prices near where each
    # contract will actually settle.
    bull, bear = AgentId("bull"), AgentId("bear")
    quotes = {
        "VANTA_OBJECTIVE_WR": (Decimal("4800"), 40),
        "VANTA_OBJECTIVE_GT480": (Decimal("0.35"), 500),
        "VANTA_QUILL_OBJECTIVE_SPD": (Decimal("100"), 25),
        "CIRCUIT_OBJECTIVE_IDX": (Decimal("4900"), 30),
    }

    print("\nTRADING")
    for symbol, (price, qty) in quotes.items():
        instrument = venue.registry.require(symbol)
        ticks = instrument.to_ticks(price)
        venue.submit(bear, symbol, order(bear, Side.SELL, ticks, qty))
        venue.submit(bull, symbol, order(bull, Side.BUY, ticks, qty))
        bull_collateral = venue.account(bull).collateral.get(symbol, 0)
        bear_collateral = venue.account(bear).collateral.get(symbol, 0)
        print(
            f"  {symbol:<26} {qty:>4} lots @ {price:<8} "
            f"collateral  long {from_money(bull_collateral):>10,}  "
            f"short {from_money(bear_collateral):>10,}"
        )

    print(f"\n  bull free cash {from_money(venue.account(bull).free_cash):>12,}")
    print(f"  bear free cash {from_money(venue.account(bear).free_cash):>12,}")
    print(f"  conservation   {venue.conservation_check()}")

    print("\nSETTLEMENT (against the matches the window names)")
    for instrument in instruments:
        result = settle(instrument.spec, oracle)
        realised = venue.settle(instrument.symbol, result)
        value = result.settlement_value
        entry = quotes[instrument.symbol][0]
        print(
            f"  {instrument.symbol:<26} settled {str(value):>10}  "
            f"(traded at {entry})   "
            f"bull {from_money(realised[bull]):>+12,}   "
            f"bear {from_money(realised[bear]):>+12,}"
        )

    print("\nFINAL")
    for agent in (bull, bear):
        account = venue.account(agent)
        pnl = from_money(account.cash) - from_money(account.starting_cash)
        print(
            f"  {agent:<6} cash {from_money(account.cash):>14,}   "
            f"PnL {pnl:>+12,}   collateral held {from_money(account.posted_collateral)}"
        )

    residual = venue.conservation_check()
    print(f"\n  conservation check: {residual}   ({'exact' if residual == 0 else 'LEAK'})")
    print("  trading moves value between participants; it never creates it")


if __name__ == "__main__":
    main()
