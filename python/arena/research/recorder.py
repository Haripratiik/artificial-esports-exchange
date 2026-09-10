"""Capture a market's price and order-flow history for analysis.

An observer, not a participant. It reads the venue's books directly and never
sends a message, so nothing it does can perturb what it is measuring, which
matters because every agent in this simulation influences the market simply by
being in it.

Two series per symbol, because they answer different questions:

    mid prices     the market's view of value. Efficiency shows up here
    trade prices   what actually printed, carrying the bid-ask bounce, which is
                   a property of execution rather than of value

Trade signs come from the aggressor side, which the engine records and which
cannot be recovered from prices afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arena.exchange.types import Side
from arena.market.venue import Venue
from arena.sim.time import Timestamp

__all__ = ["MarketRecorder", "SymbolHistory"]


@dataclass
class SymbolHistory:
    symbol: str
    times: list[int] = field(default_factory=list)
    mids: list[float] = field(default_factory=list)
    trade_prices: list[float] = field(default_factory=list)
    trade_signs: list[int] = field(default_factory=list)

    @property
    def mid_array(self) -> np.ndarray:
        return np.asarray(self.mids, dtype=float)

    @property
    def trade_array(self) -> np.ndarray:
        return np.asarray(self.trade_prices, dtype=float)

    @property
    def sign_array(self) -> np.ndarray:
        return np.asarray(self.trade_signs, dtype=float)


class MarketRecorder:
    """Samples mid prices on a schedule and drains the tape as it grows."""

    def __init__(self, venue: Venue) -> None:
        self.venue = venue
        self.history: dict[str, SymbolHistory] = {
            symbol: SymbolHistory(symbol) for symbol in venue.registry.symbols
        }
        self._tape_seen: dict[str, int] = dict.fromkeys(venue.registry.symbols, 0)

    def sample(self, now: Timestamp) -> None:
        """Record one observation per symbol.

        Sampled on a clock rather than on every event, because event-time and
        clock-time series have different statistical properties and mixing them
        would make the autocorrelations meaningless. Clock time is what the
        stylized-fact literature measures.
        """
        for symbol, history in self.history.items():
            engine = self.venue.engine(symbol)
            instrument = self.venue.registry.require(symbol)

            book = engine.book.snapshot()
            if book.best_bid is not None and book.best_ask is not None:
                mid = (int(book.best_bid) + int(book.best_ask)) / 2.0
            elif history.mids:
                # Carry the last mid rather than dropping the observation, so
                # the series stays evenly spaced in time. A gap would be read
                # as a large return by any autocorrelation estimate.
                mid = history.mids[-1]
            else:
                low, high = instrument.tick_bounds
                mid = (int(low) + int(high)) / 2.0

            history.times.append(int(now))
            history.mids.append(mid)

            tape = engine.tape
            seen = self._tape_seen[symbol]
            for trade in tape[seen:]:
                history.trade_prices.append(float(int(trade.price)))
                history.trade_signs.append(1 if trade.aggressor_side is Side.BUY else -1)
            self._tape_seen[symbol] = len(tape)
