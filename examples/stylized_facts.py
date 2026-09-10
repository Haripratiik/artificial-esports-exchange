"""Does this market behave like a market?

    python examples/stylized_facts.py

Runs a long session, records prices, and measures them against the statistical
regularities real order-book markets exhibit. Nothing here feeds back into the
agents: the point is to find out what the market does, not to make it do
something.

Expectations are declared in ``arena.research.stylized`` before any number is
computed, and some of them are that a fact should be *absent*. This market's
underlying is a bounded statistic that settles at a value fixed for the whole
session, so mechanisms that depend on continuous information arrival or on
metaorder splitting have nothing to work with here. Reporting those as absent
is the model being honest, not failing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), "python"]

from arena.research.recorder import MarketRecorder
from arena.research.stylized import analyse, variance_signature
from arena.sim.time import Timestamp, millis, seconds
from dashboard.build_market import build, instruments, true_values

SAMPLE_EVERY = millis(250)
SESSION = seconds(3_600)


def pick_focus(listed) -> list[str]:
    """Two outright futures and a spread, chosen by class rather than by name.

    Named tickers were hardcoded here and went stale the moment the listing was
    rebuilt, which turns a diagnostic into a KeyError. The diagnostics below are
    written about outrights and the spread between them, so that is what the
    example asks the listing for.
    """
    futures = [i.symbol for i in listed if i.instrument_class == "future"]
    spreads = [i.symbol for i in listed if i.instrument_class == "spread"]
    return futures[:2] + spreads[:1]


def main() -> None:
    market = build(seed=11)
    market.kernel.start()
    recorder = MarketRecorder(market.venue)
    truth = true_values(instruments())

    print(f"running {int(SESSION) / 1e9:.0f} simulated seconds, "
          f"sampling every {int(SAMPLE_EVERY) / 1e6:.0f}ms ...")
    step = int(SAMPLE_EVERY)
    for tick in range(1, int(SESSION) // step + 1):
        now = Timestamp(tick * step)
        market.kernel.advance(until=now)
        recorder.sample(now)

    print(f"done: {market.kernel.processed:,} events\n")

    focus = pick_focus(instruments())
    for symbol in focus:
        history = recorder.history[symbol]
        if len(history.trade_prices) < 100:
            print(f"{symbol}: too few trades ({len(history.trade_prices)}) to analyse\n")
            continue
        report = analyse(
            symbol, history.mid_array, history.trade_array, history.sign_array
        )
        print(report)
        print(f"  trades: {len(history.trade_prices):,}\n")

    # --- the facts this market has that equities do not --------------------
    print("=" * 78)
    print("properties specific to a settling contract")
    print("=" * 78)

    for symbol in focus:
        history = recorder.history[symbol]
        instrument = market.venue.registry.require(symbol)
        mid = history.mid_array
        if mid.size < 100 or symbol not in truth:
            continue

        target = float(truth[symbol])
        span = abs(float(instrument.tick_bounds[1]) - float(instrument.tick_bounds[0]))

        quarters = np.array_split(mid, 4)
        errors = [abs(float(q.mean()) - target) / span * 100 for q in quarters]
        vols = [float(np.std(np.diff(q))) for q in quarters]

        print(f"\n{symbol}   settles at {instrument.from_ticks(int(target))}")
        print("  pricing error, % of contract range, by quarter of session:")
        print("    " + "  ".join(f"Q{i+1} {e:6.2f}%" for i, e in enumerate(errors)))
        print("  realised volatility of mid, by quarter:")
        print("    " + "  ".join(f"Q{i+1} {v:7.3f}" for i, v in enumerate(vols)))

        converged = errors[-1] < errors[0]
        settled = vols[-1] < vols[0]
        print(f"  price converged toward settlement : {'yes' if converged else 'NO'}")
        print(f"  volatility decayed as it settled  : {'yes' if settled else 'NO'}")

    print()
    print("=" * 78)
    print("how to read this")
    print("=" * 78)
    print(
        "  Facts marked NOT expected are ones this market has no mechanism to\n"
        "  produce. Volatility clustering needs information to arrive in bursts,\n"
        "  and here the truth is fixed for the whole session. Long-memory order\n"
        "  flow needs metaorder splitting, and no agent here splits anything.\n"
        "  Seeing them absent is the model being honest; seeing them strong\n"
        "  would mean something is generating them artificially."
    )


if __name__ == "__main__":
    main()
