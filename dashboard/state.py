"""The running market, its configuration, and the history the UI charts.

Kept apart from the HTTP layer because it has a life of its own: the market runs
whether or not anyone is watching, it can be rebuilt with a different
configuration while the server stays up, and it accumulates a price history that
outlives any one WebSocket connection.

Rebuilding rather than mutating is deliberate. Half the point of this venue is
that a run is reproducible from its seed, and an agent population edited
mid-flight would produce a session that no seed could ever reproduce. Changing
the configuration therefore starts a new market, and the UI says so.

Two records are kept, and they are for two different readers.

The **sampled path** -- ``Series.stamps`` and ``Series.mids`` -- is for the
chart and for the stylized-fact estimators. It is floats, because a chart is a
float and ``analyse`` wants a numpy array.

The **candles** are for a program. They are integers in the exchange's own tick
domain, they are aggregated as each period closes rather than recomputed on
demand, and they carry three separate OHLC blocks: the trade price, the best
bid and the best ask. Candling the quotes separately is the whole point on a
book this thin. Nine of the forty-seven contracts the default configuration
lists went a full simulated second without a print in the measurement that
motivated this, and on a symbol that did not trade, the last trade price is a
fact about some earlier minute while the bid and the ask are facts about the
period -- so a backtester reconstructing what was actually *quotable* needs the
quote candles and cannot get there from the tape.
"""

from __future__ import annotations

import time
from array import array
from collections import deque
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any, NamedTuple

import numpy as np

from arena.exchange.session import SessionState
from arena.exchange.types import AgentId
from arena.market.fees import FREE, MAKER_TAKER, FeeSchedule
from arena.market.live import HUMAN_ID, LiveMarket
from arena.portfolio.money import from_money
from arena.research.stylized import analyse
from dashboard.build_market import build, true_values

__all__ = [
    "MarketConfig",
    "MarketRunner",
    "FEE_SCHEDULES",
    "Candle",
    "CandleRing",
    "CANDLE_PERIODS",
    "CANDLE_DEPTH",
]

# Named schedules, so the browser sends a name rather than raw basis points and
# cannot invent a venue that pays out more than it takes without meaning to.
FEE_SCHEDULES: dict[str, FeeSchedule] = {
    "free": FREE,
    "maker-taker": MAKER_TAKER,
    "taker-only": FeeSchedule(taker_bps=3.0, maker_bps=0.0),
    "generous": FeeSchedule(taker_bps=1.0, maker_bps=-2.0),
}

# How many samples of each symbol's price to keep, and how many server ticks
# pass between samples.
#
# The arithmetic is written out because the comment that used to sit here was
# wrong by a factor of two. It said "sampling every fourth tick gives 5Hz, so
# this buffer covers six minutes", and it was true when it was written -- then
# SAMPLE_EVERY went from 4 to 2 and nobody came back to it. What the buffer
# actually held was:
#
#     1 / TICK_SECONDS = 1 / 0.05        = 20 ticks a second
#     20 / SAMPLE_EVERY = 20 / 2         = 10 samples a second
#     HISTORY / 10 = 1,800 / 10          = 180 seconds
#
# Three minutes, under a comment promising six, and nothing older than that
# existed anywhere in this process. That is not a chart bug -- a chart wants a
# few hundred points and gets them -- it is why the API's history endpoint was
# useless to anything systematic: three minutes is shorter than the shortest
# thing a backtest measures.
#
# HISTORY is now 3,600, which at 10 samples a second is 360 seconds: the six
# minutes the old comment claimed, actually delivered. It stays this size
# because this buffer's job is the chart and the estimators, not the queryable
# window -- that job now belongs to the candles below.
#
# Bytes: four deques a symbol, one entry per sample (`trades` and `signs` fill
# per print rather than per sample, so they are the loose bound rather than the
# typical one). A CPython float is 24 bytes of object plus an 8-byte slot in the
# deque's block, so about 32 bytes an entry, 128 bytes a sample across the four:
#
#     3,600 samples x 128 bytes         = 461 KB a symbol
#     x 47 contracts on the default cfg = 21.7 MB
HISTORY = 3_600
# Every second tick rather than every fourth. A step costs 0.12ms against a
# 50ms budget, so the old rate was leaving a chart to fill slowly for no
# reason anyone could point at.
SAMPLE_EVERY = 2

# The candle periods, in **seconds of simulated time**, and the number of closed
# candles retained per period per symbol.
#
# A closed enum, refused rather than rounded when a client asks for something
# else, which is Kalshi's behaviour: their candlestick endpoint takes 1, 60 and
# 1440 and answers anything else with a refusal rather than the nearest one it
# does keep. Rounding would hand back a series whose bars are not the bars that
# were asked for, and nothing downstream would notice.
#
# The values are one scale below Kalshi's minute/hour/day because the clock they
# are cut against is this simulator's, not a calendar's. A session here is
# minutes to hours of simulated time, sampled ten times a second:
#
#     1 second   -- 10 samples a candle, the finest bar the sampler can fill
#     10 seconds -- 100 samples
#     60 seconds -- one simulated minute
#
# One second is the floor on purpose. A period shorter than 10 samples would
# produce candles whose high and low are one or two observations, which is not
# a range, it is noise with a box drawn round it.
#
# Depth, and the arithmetic that sizes it. Each candle is 17 int64 fields --
# end, volume, trades, notional, open interest, and four each of price, bid and
# ask -- stored flat in an ``array('q')`` ring rather than as objects. Flat
# because the object form was measured at 564 bytes for a busy candle and 340
# for a quiet one, against exactly 136 for the flat one, and 4x on a structure
# this repetitive is the difference between a window worth querying and one that
# is not:
#
#     17 fields x 8 bytes                  = 136 bytes a candle
#     x CANDLE_DEPTH = 1,800               = 245 KB a period a symbol
#     x 3 periods                          = 734 KB a symbol
#     x 47 contracts on the default cfg    = 34.5 MB
#
# and what that buys, per symbol:
#
#     1s  x 1,800 =   1,800 s = 30 minutes
#     10s x 1,800 =  18,000 s = 5 hours
#     60s x 1,800 = 108,000 s = 30 hours
#
# So the retained span went from 180 seconds of mids with no volume and no
# quotes, to 30 hours of gap-free OHLC with both, for 34.5 MB. The depth is the
# same 1,800 the sampled buffer used to hold, which is a coincidence worth
# naming rather than hiding: 1,800 of something is a useful number of bars and
# was never a useful number of ticks.
CANDLE_PERIODS: tuple[int, ...] = (1, 10, 60)
CANDLE_DEPTH = 1_800

# 17 int64s a candle, and the value that means "there was none".
#
# A sentinel rather than a parallel bitmap because the flat store has no room
# for ``None`` and the absence is real: a book with nothing resting on one side
# has no bid, and a period before the first print ever has no close. The value
# is the minimum an int64 can hold, which no price on this venue can collide
# with -- the widest settlement range listed is a few thousand contract units,
# and even the market-on-open interest that rests at +2^62 so it crosses every
# candidate an auction weighs is a large *positive* number, nowhere near this.
_CANDLE_FIELDS = 17
_ABSENT = -(1 << 63)
_NANOS = 1_000_000_000


@dataclass(frozen=True, slots=True)
class MarketConfig:
    """Everything that decides what market is running."""

    seed: int = 7
    speed: float = 1.0
    arbitrageur: bool = False
    flow_traders: int = 0
    fees: str = "maker-taker"
    price_band: float | None = 0.05
    makers: int = 3
    opening_auction: bool = True
    surface: bool = True
    mechanism: str = "book"
    # Whether live matches run alongside the statistical contracts.
    #
    # On, because a match is the only thing here that resolves while somebody
    # is watching: the statistical contracts settle at the end of a four week
    # observation window and nothing about them changes inside a session.
    #
    # One at a time per format rather than more. Measured over 180 simulated
    # seconds on seed 7, matches take the listing from 47 symbols to 971 and
    # the market from 1.85x real time to 0.75x, so the board is kept small
    # enough that a session anybody watches keeps up.
    matches: bool = True
    concurrent_matches: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "speed": self.speed,
            "arbitrageur": self.arbitrageur,
            "flow_traders": self.flow_traders,
            "fees": self.fees,
            "price_band": self.price_band,
            "makers": self.makers,
            "opening_auction": self.opening_auction,
            "surface": self.surface,
            "mechanism": self.mechanism,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MarketConfig":
        """Build from untrusted JSON, clamping rather than trusting.

        Everything here arrives from a browser, so every field is bounded. An
        unbounded agent count or speed would let a page freeze the server that
        is serving it.
        """
        band = payload.get("price_band", 0.05)
        fees = str(payload.get("fees", "maker-taker"))
        return cls(
            seed=int(payload.get("seed", 7)) % (2**31),
            speed=max(0.0, min(50.0, float(payload.get("speed", 1.0)))),
            arbitrageur=bool(payload.get("arbitrageur", False)),
            flow_traders=max(0, min(24, int(payload.get("flow_traders", 0)))),
            fees=fees if fees in FEE_SCHEDULES else "free",
            price_band=(
                None if band in (None, "", "none") else max(0.001, min(5.0, float(band)))
            ),
            makers=max(1, min(8, int(payload.get("makers", 3)))),
            opening_auction=bool(payload.get("opening_auction", True)),
            surface=bool(payload.get("surface", True)),
            mechanism=(
                "scoring-rule"
                if str(payload.get("mechanism", "book")) == "scoring-rule"
                else "book"
            ),
        )


class Candle(NamedTuple):
    """One closed period, in the exchange's own integers.

    Ticks and lots, never prices, because this is the storage layer and the
    conversion to a decimal price belongs where the instrument is in scope. The
    same rule the rest of this codebase keeps: a raw internal unit may travel,
    but only under a name that says what unit it is.

    ``notional`` is ``sum(price_in_ticks x quantity)`` over the period's prints,
    exact and integral. It is carried rather than only a mean because a mean is
    a quotient and a quotient of two integers is not always a decimal -- so the
    number a client can check the arithmetic with has to be the numerator.
    """

    # End of the period, in simulated nanoseconds. Kalshi stamps a candle with
    # the end rather than the start for the reason that matters to anything
    # streaming: a bar stamped with its start is a bar you cannot tell from the
    # one still being built.
    end: int
    volume: int
    trades: int
    notional: int
    open_interest: int
    price_open: int | None
    price_high: int | None
    price_low: int | None
    price_close: int | None
    bid_open: int | None
    bid_high: int | None
    bid_low: int | None
    bid_close: int | None
    ask_open: int | None
    ask_high: int | None
    ask_low: int | None
    ask_close: int | None


def _present(value: int) -> int | None:
    return None if value == _ABSENT else value


class CandleRing:
    """A fixed-size ring of closed candles for one symbol at one period.

    Flat storage, for the reason the module constants do the arithmetic on: an
    object per candle costs 340 to 564 bytes measured, and 136 exactly. At the
    depths that make this window worth querying that is the difference between
    34 MB and well over a hundred.

    Candles go in strictly ascending ``end`` and exactly one per period, gaps
    included, so the ring is ordered by construction and needs no sort and no
    index. That ordering is the whole reason the gap filling below is not
    optional.
    """

    __slots__ = ("period", "period_ns", "depth", "_cells", "_count", "_next")

    def __init__(self, period: int, depth: int = CANDLE_DEPTH) -> None:
        self.period = period
        self.period_ns = period * _NANOS
        self.depth = depth
        self._cells = array("q", bytes(8 * _CANDLE_FIELDS * depth))
        self._count = 0
        self._next = 0

    def __len__(self) -> int:
        return self._count

    @property
    def retains_ns(self) -> int:
        """The widest window this ring can answer for, full or not."""
        return self.depth * self.period_ns

    def append(self, row: tuple[int, ...]) -> None:
        base = self._next * _CANDLE_FIELDS
        self._cells[base : base + _CANDLE_FIELDS] = array("q", row)
        self._next = (self._next + 1) % self.depth
        self._count = min(self._count + 1, self.depth)

    def _slot(self, ordinal: int) -> int:
        """Where the ordinal-th oldest retained candle physically sits."""
        return (self._next - self._count + ordinal) % self.depth

    def _at(self, ordinal: int) -> Candle:
        base = self._slot(ordinal) * _CANDLE_FIELDS
        cells = self._cells
        return Candle(
            cells[base],
            cells[base + 1],
            cells[base + 2],
            cells[base + 3],
            cells[base + 4],
            *(_present(cells[base + i]) for i in range(5, _CANDLE_FIELDS)),
        )

    def span(self) -> tuple[int, int] | None:
        """The oldest and newest period ends held, or None while empty."""
        if not self._count:
            return None
        oldest = self._cells[self._slot(0) * _CANDLE_FIELDS]
        newest = self._cells[self._slot(self._count - 1) * _CANDLE_FIELDS]
        return (oldest, newest)

    def window(self, start_ns: int, end_ns: int, limit: int) -> list[Candle]:
        """Candles whose period ended within ``[start_ns, end_ns]``, oldest first.

        Ascending, always, because a series that a client has to sort is a
        series that one client will forget to sort. When more fall in the range
        than ``limit`` allows, the *newest* are kept: a client that asked for a
        window wider than its page size wants the recent end of it, and the
        endpoint refuses a range wider than this ring can hold anyway, so this
        only ever trims a page rather than hiding a gap.

        Only closed periods are here. The one being accumulated is deliberately
        not published: a partial bar has a high that is not the period's high,
        and a backtester that mixed one in would be reading the future's
        opening quote as the past's close.
        """
        cells = self._cells
        picked = []
        for ordinal in range(self._count):
            end = cells[self._slot(ordinal) * _CANDLE_FIELDS]
            if start_ns <= end <= end_ns:
                picked.append(ordinal)
        return [self._at(ordinal) for ordinal in picked[-limit:]]


class _PeriodBook:
    """The ring for one period, plus the candle currently being accumulated.

    Aggregated forward as samples arrive rather than recomputed from the tape on
    request. The tape is the only other source and it holds prints, not quotes:
    the best bid at 14:03:07 is not recoverable from a list of trades, so a
    quote candle has to be built while the quote is observable or not at all.
    """

    __slots__ = (
        "ring",
        "index",
        "interest",
        "volume",
        "trades",
        "notional",
        "p_open",
        "p_high",
        "p_low",
        "p_close",
        "b_open",
        "b_high",
        "b_low",
        "b_close",
        "a_open",
        "a_high",
        "a_low",
        "a_close",
        "carry_close",
        "carry_bid",
        "carry_ask",
    )

    def __init__(self, period: int, depth: int = CANDLE_DEPTH) -> None:
        self.ring = CandleRing(period, depth)
        self.index: int | None = None
        # Carried rather than reset, all four: open interest and the three
        # closes are the state a quiet period is filled from, and a quiet
        # period is exactly the one that has nothing of its own to fill from.
        self.interest = 0
        self.carry_close = _ABSENT
        self.carry_bid = _ABSENT
        self.carry_ask = _ABSENT
        self._reset()

    def _reset(self) -> None:
        self.volume = 0
        self.trades = 0
        self.notional = 0
        self.p_open = self.p_high = self.p_low = self.p_close = _ABSENT
        self.b_open = self.b_high = self.b_low = self.b_close = _ABSENT
        self.a_open = self.a_high = self.a_low = self.a_close = _ABSENT

    def observe(
        self,
        now_ns: int,
        bid: int | None,
        ask: int | None,
        prints: Any,
        interest: int,
    ) -> None:
        """Fold one sample, and whatever printed since the last one, into the bar."""
        period_ns = self.ring.period_ns
        index = now_ns // period_ns
        if self.index is None:
            self.index = index
        elif index > self.index:
            self._close()
            # Every period between the one that just closed and the one now
            # opening gets a bar of its own. Skipping them is what makes a
            # client reindex, and a client that reindexes a price series is a
            # client writing its own interpolation policy in a hurry.
            for skipped in range(self.index + 1, index):
                self._quiet((skipped + 1) * period_ns)
            self.index = index

        self.interest = interest
        if bid is not None:
            if self.b_open == _ABSENT:
                self.b_open = self.b_high = self.b_low = bid
            elif bid > self.b_high:
                self.b_high = bid
            elif bid < self.b_low:
                self.b_low = bid
            self.b_close = bid
        if ask is not None:
            if self.a_open == _ABSENT:
                self.a_open = self.a_high = self.a_low = ask
            elif ask > self.a_high:
                self.a_high = ask
            elif ask < self.a_low:
                self.a_low = ask
            self.a_close = ask
        for trade in prints:
            price = int(trade.price)
            quantity = int(trade.quantity)
            if self.p_open == _ABSENT:
                self.p_open = self.p_high = self.p_low = price
            elif price > self.p_high:
                self.p_high = price
            elif price < self.p_low:
                self.p_low = price
            self.p_close = price
            self.volume += quantity
            self.trades += 1
            self.notional += price * quantity

    # -- closing -----------------------------------------------------------

    def _block(self, o: int, h: int, low: int, c: int, carry: int) -> tuple[int, ...]:
        """One OHLC block, or four copies of the last value if nothing arrived."""
        return (o, h, low, c) if c != _ABSENT else (carry, carry, carry, carry)

    def _close(self) -> None:
        assert self.index is not None
        traded = self.p_close != _ABSENT
        self.ring.append(
            (
                (self.index + 1) * self.ring.period_ns,
                self.volume,
                self.trades,
                self.notional,
                self.interest,
                *self._block(
                    self.p_open, self.p_high, self.p_low, self.p_close, self.carry_close
                ),
                *self._block(
                    self.b_open, self.b_high, self.b_low, self.b_close, self.carry_bid
                ),
                *self._block(
                    self.a_open, self.a_high, self.a_low, self.a_close, self.carry_ask
                ),
            )
        )
        if traded:
            self.carry_close = self.p_close
        if self.b_close != _ABSENT:
            self.carry_bid = self.b_close
        if self.a_close != _ABSENT:
            self.carry_ask = self.a_close
        self._reset()

    def _quiet(self, end_ns: int) -> None:
        """A period no sample landed in at all: flat, zero volume, carried closes.

        Distinct from the ordinary no-trade period, which still has real bid and
        ask candles because the sampler still saw the book ten times a second.
        This one only happens when the server itself stalled through a whole
        period, and it is filled rather than skipped for the same reason: the
        series is gap-free or it is not.
        """
        carried = self.carry_close
        self.ring.append(
            (
                end_ns,
                0,
                0,
                0,
                self.interest,
                carried,
                carried,
                carried,
                carried,
                *(self.carry_bid,) * 4,
                *(self.carry_ask,) * 4,
            )
        )


def _new_books() -> dict[int, _PeriodBook]:
    return {period: _PeriodBook(period, CANDLE_DEPTH) for period in CANDLE_PERIODS}


def _open_interest(venue: Any) -> dict[str, int]:
    """Contracts outstanding per symbol: the sum of the long side.

    The textbook definition, and it is computable here rather than approximated
    because this venue's positions net to zero by construction -- every contract
    long is a contract short, so counting one side counts the contracts and
    counting both would count each twice.

    Computed once per sample and shared across all forty-seven symbols rather
    than once per symbol, because ``Venue.accounts`` builds a freshly sorted
    copy of the account table on every read: asking it per symbol would be
    forty-seven sorts ten times a second to answer one question.
    """
    outstanding: dict[str, int] = {}
    for account in venue.accounts.values():
        for symbol, position in account.positions.items():
            if position.quantity > 0:
                outstanding[symbol] = outstanding.get(symbol, 0) + position.quantity
    return outstanding


@dataclass
class Series:
    """One symbol's recent price path: sampled mids for a chart, candles for a program."""

    stamps: deque[int] = field(default_factory=lambda: deque(maxlen=HISTORY))
    mids: deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY))
    trades: deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY))
    signs: deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY))
    books: dict[int, _PeriodBook] = field(default_factory=_new_books)

    def to_dict(self) -> dict[str, list[Any]]:
        return {
            "t": list(self.stamps),
            "mid": [None if np.isnan(v) else v for v in self.mids],
        }

    # -- candles -----------------------------------------------------------

    @property
    def periods(self) -> tuple[int, ...]:
        """The closed enum of periods this series keeps, in seconds.

        Published as a property rather than as a constant an API layer imports,
        because the API layer is ``arena.api.rest`` and it must not import the
        application it is mounted into. It reads the enum off the runner it was
        handed, the same way it reads the history cap off the buffer, so the
        number a client is refused against and the number this file keeps are
        one number.
        """
        return tuple(sorted(self.books))

    def ring(self, period: int) -> CandleRing | None:
        """The closed candles at one period, or None if that is not a period."""
        book = self.books.get(period)
        return None if book is None else book.ring

    def observe(
        self,
        now_ns: int,
        bid: int | None,
        ask: int | None,
        prints: Any,
        interest: int,
    ) -> None:
        for book in self.books.values():
            book.observe(now_ns, bid, ask, prints, interest)


def _role(agent) -> str:
    """What an agent does, for a reader who does not know the class names.

    Published beside the class so the page can say "market maker" while the
    diagnostics keep the exact type. It is derived by isinstance rather than
    by name, so a specialised maker is still a maker -- which is the thing two
    tests got wrong when the options maker arrived and they went looking for
    the string "MarketMaker".
    """
    from arena.agents.arbitrageur import Arbitrageur
    from arena.agents.bayesian import BayesianFundamental
    from arena.agents.flow import FlowTrader
    from arena.agents.fundamental import FundamentalTrader
    from arena.agents.market_maker import MarketMaker
    from arena.agents.noise import NoiseTrader

    for kind, label in (
        (MarketMaker, "market maker"),
        (Arbitrageur, "arbitrageur"),
        (BayesianFundamental, "informed"),
        (FundamentalTrader, "informed"),
        (FlowTrader, "order flow"),
        (NoiseTrader, "uninformed"),
    ):
        if isinstance(agent, kind):
            return label
    return "participant"


class MarketRunner:
    """Owns the live market, steps it, and records what happened."""

    def __init__(self, config: MarketConfig | None = None) -> None:
        self.config = config or MarketConfig()
        self.market: LiveMarket = self._build(self.config)
        self.history: dict[str, Series] = {}
        self.generation = 0
        self.started_at = time.time()
        self._tape_seen: dict[str, int] = {}
        self._ticks = 0
        self._reset_history()

    # -- lifecycle ---------------------------------------------------------

    def _build(self, config: MarketConfig) -> LiveMarket:
        market = build(
            seed=config.seed,
            speed=config.speed,
            arbitrageur=config.arbitrageur,
            flow_traders=config.flow_traders,
            fees=FEE_SCHEDULES[config.fees],
            price_band=config.price_band,
            makers=config.makers,
            opening_auction=config.opening_auction,
            surface=config.surface,
            mechanism=config.mechanism,
            matches=config.matches,
            concurrent_matches=config.concurrent_matches,
        )
        return market

    def _reset_history(self) -> None:
        self.history = {s: Series() for s in self.market.venue.registry.symbols}
        self._tape_seen = dict.fromkeys(self.market.venue.registry.symbols, 0)

    def start(self) -> None:
        self.market.start()

    def reconfigure(self, config: MarketConfig) -> dict[str, Any]:
        """Swap in a new market. The old one is discarded, not paused.

        A configuration change starts a fresh session precisely so the result
        stays reproducible from its seed.
        """
        self.config = config
        self.market = self._build(config)
        self.generation += 1
        self.started_at = time.time()
        self._reset_history()
        self.start()
        return {"ok": True, "generation": self.generation, "config": config.to_dict()}

    def set_speed(self, speed: float) -> dict[str, Any]:
        """Speed is the one setting that does not need a rebuild.

        It changes how fast simulated time tracks the wall clock, not what any
        agent decides, so the session stays the one the seed describes.
        """
        value = max(0.0, min(50.0, float(speed)))
        self.market.speed = value
        self.config = replace(self.config, speed=value)
        return {"ok": True, "speed": value}

    # -- stepping ----------------------------------------------------------

    def step(self) -> int:
        events = self.market.step()
        self._ticks += 1
        if self._ticks % SAMPLE_EVERY == 0:
            self._record()
        return events

    def _record(self) -> None:
        venue = self.market.venue
        now = int(self.market.kernel.now)
        interest = _open_interest(venue)
        # Take up whatever listed since the last tick.
        #
        # History was built once from the symbols present at construction,
        # which was right while the listing was fixed for a session. Matches
        # open and close continuously, so a chart keyed on the original listing
        # simply never draws them: no crash, no warning, and a market that
        # looks empty in the one place anybody is watching it.
        for symbol in venue.registry.symbols:
            if symbol not in self.history:
                self.history[symbol] = Series()
                self._tape_seen[symbol] = 0

        for symbol, series in list(self.history.items()):
            if venue.registry.get(symbol) is None:
                continue
            instrument = venue.registry.require(symbol)
            engine = venue.engine(symbol)
            # Two levels, not one. Market-on-open interest rests at a sentinel
            # price so that it crosses every candidate an auction weighs, which
            # makes it the top of the book by a margin of 2^61 -- and a
            # one-level snapshot of a book holding any can therefore contain
            # nothing but the sentinel, at which point ``best_bid`` filters it
            # out and this samples NaN through an entire call phase. It is the
            # same two levels ``rest.py::TOUCH_LEVELS`` reads, for the same
            # reason, and reading one was quietly blanking the opening auction
            # out of every chart and every candle.
            snapshot = engine.book.snapshot(2)
            mid = snapshot.mid
            series.stamps.append(now)
            series.mids.append(
                float("nan") if mid is None else float(instrument.from_ticks(int(mid)))
            )
            tape = engine.tape
            seen = self._tape_seen.get(symbol, 0)
            fresh = tape[seen:]
            for trade in fresh:
                series.trades.append(float(instrument.from_ticks(trade.price)))
                series.signs.append(1.0 if trade.aggressor_side.value == "buy" else -1.0)
            self._tape_seen[symbol] = len(tape)
            # The candles take the prints in ticks rather than the converted
            # floats above. The chart's copy is a float because a chart is a
            # float; a bar a backtester will difference must be the integer the
            # engine matched at, or the whole exact-arithmetic argument stops at
            # the first `high - low`.
            bid, ask = snapshot.best_bid, snapshot.best_ask
            series.observe(
                now,
                None if bid is None else int(bid),
                None if ask is None else int(ask),
                fresh,
                interest.get(symbol, 0),
            )

    # -- reporting ---------------------------------------------------------

    def session_state(self) -> dict[str, Any]:
        venue = self.market.venue
        return {
            "generation": self.generation,
            "uptime": time.time() - self.started_at,
            "config": self.config.to_dict(),
            "fees": venue.fees.to_dict(),
            # Through `from_money`, like every other money field on this
            # payload. It was published in raw *minor* units -- a factor of a
            # million -- and the browser silently corrected it with `/ 1e6` at
            # the point of display. That is the same defect shape as the
            # settlement value that read 18,677 against a real 4,663 and the
            # equity column that rendered a seat as "143745.00M": a number
            # crossing the wire in units its label does not claim, with one
            # consumer compensating and every other consumer wrong.
            #
            # It matters more now than it did, because the browser is no longer
            # the only reader. A programmatic client asking `GET /v1/exchange`
            # has no `/ 1e6` to apply and no reason to suspect it needs one.
            # The compensation was also `Number(...)`, which puts a float in a
            # money path in a project whose whole claim is exact arithmetic.
            "fees_collected": str(from_money(venue.fees_collected)),
            "price_band": venue.price_band,
            "halts": [self._readable_halt(halt) for halt in venue.halts[-20:]],
            "sessions": {
                symbol: venue.session(symbol).value
                for symbol in venue.registry.symbols
            },
        }

    def _readable_halt(self, halt: dict[str, Any]) -> dict[str, Any]:
        """One breaker record, with its prices as prices.

        The venue records a halt in the unit it matches in -- ticks -- which is
        right for the venue and wrong for a screen. The Halts table draws
        ``price`` and ``reference`` straight into columns headed as such, so a
        band break on a contract quoted on a 0.25 grid printed 1,989 against a
        real price of 497.25, on a claim whose whole settlement range is 0 to
        1,000. A number four times outside the range the same page publishes
        does not read as a bug; it reads as a number, which is exactly how the
        settlement figure survived so long.
        """
        instrument = self.market.venue.registry.get(halt.get("symbol", ""))
        if instrument is None:
            return dict(halt)
        readable = dict(halt)
        for field_name in ("price", "reference"):
            ticks = halt.get(field_name)
            if ticks is not None:
                readable[field_name] = str(instrument.from_ticks(int(ticks)))
        return readable

    def agents(self) -> list[dict[str, Any]]:
        """Who is in the market. The population is the experiment."""
        roster = []
        for agent in self.market.agents:
            account = self.market.venue.accounts.get(agent.agent_id)
            roster.append(
                {
                    "id": str(agent.agent_id),
                    "kind": type(agent).__name__,
                    "role": _role(agent),
                    "halted": agent.agent_id
                    in self.market.venue.halted_participants,
                    "fills": getattr(agent, "fills", 0),
                    "rejects": getattr(agent, "rejects", 0),
                    "positions": {
                        s: q for s, q in sorted(getattr(agent, "position", {}).items()) if q
                    },
                    # In price units, like every other money figure on the
                    # wire. `Account.equity` answers in *minor* units -- the
                    # integer the ledger is kept in -- and publishing that
                    # straight put the Participants table a million times out:
                    # a maker worth 113,125,513.21 was drawn as
                    # "113125513.21M", and a person's own seat showed
                    # "143745.00M" beside a header reading "143.7k". The same
                    # shape as the settlement bug: a raw internal unit under a
                    # label that promises a price.
                    "equity": (
                        str(from_money(account.equity(self.market.venue.marks())))
                        if account is not None
                        else None
                    ),
                }
            )
        return roster

    def diagnostics(self, symbol: str) -> dict[str, Any]:
        """Stylized-fact diagnostics for one symbol, from the live history.

        The same estimators the research harness uses, so a number seen in the
        browser is the number a paper would quote rather than a second
        implementation that might disagree.
        """
        series = self.history.get(symbol)
        if series is None:
            return {"error": f"unknown symbol {symbol}"}
        mids = np.array([v for v in series.mids if not np.isnan(v)], dtype=float)
        trades = np.asarray(series.trades, dtype=float)
        signs = np.asarray(series.signs, dtype=float)
        if mids.size < 60:
            return {
                "symbol": symbol,
                "observations": int(mids.size),
                "pending": True,
                "verdicts": [],
            }
        report = analyse(symbol, mids, trades, signs)
        return report.to_dict()

    @staticmethod
    def _true_values(listed: list[Any]) -> dict[str, Any]:
        """What each contract will be worth, for the contracts this can answer.

        `true_values` settles every instrument against the statistical world's
        oracle, and a match contract belongs to a different world: it is pinned
        to its own reference and the oracle refuses it, correctly, because
        answering would mean measuring one world against another's evidence.

        One refusal took the whole route down. `/api/instruments` raised
        ReferenceMismatch on the first match contract it reached, so enabling
        matches by default turned the entire front end into a 500 while every
        underlying book was trading normally. The settlement path already knew
        better: `settle_due` records a failure and carries on, because a
        contract nobody can answer for is a real outcome rather than a fault in
        the venue.

        Asked one at a time so a contract this cannot value costs only its own
        line, and left out of the mapping rather than filled with a zero, since
        zero is a price and "no answer" is not.
        """
        out: dict[str, Any] = {}
        for instrument in listed:
            try:
                out.update(true_values([instrument]))
            except Exception:  # noqa: BLE001 - an unanswerable contract is not a fault
                continue
        return out

    def instruments(self) -> list[dict[str, Any]]:
        venue = self.market.venue
        settle = self._true_values(
            [venue.registry.require(s) for s in venue.registry.symbols]
        )
        out = []
        for symbol in venue.registry.symbols:
            instrument = venue.registry.require(symbol)
            payload = instrument.to_dict()
            payload["session"] = venue.session(symbol).value
            payload["trades"] = len(venue.engine(symbol).tape)
            # What it will actually be worth, **as a price**. Only meaningful
            # because this is a simulation, and the whole reason the market can
            # be scored rather than merely watched.
            #
            # In ticks until now, while every price on the page beside it was
            # in contract units. So the one number whose entire job is to be
            # compared against the market was in a different unit from the
            # market: SPIKE_WR_FUT marked at 4,663 and revealed a "settlement"
            # of 18,677, and the chart drew that as a target line four times
            # off the top of the series. It looked like a number rather than
            # like a bug, which is why it survived.
            ticks = settle.get(symbol)
            payload["settles_at"] = (
                None if ticks is None else float(instrument.from_ticks(round(ticks)))
            )
            out.append(payload)
        return out

    def halt(self, symbol: str) -> dict[str, Any]:
        venue = self.market.venue
        if venue.registry.get(symbol) is None:
            return {"ok": False, "error": f"unknown symbol {symbol}"}
        venue.halt(symbol, reason="manual")
        return {"ok": True, "session": venue.session(symbol).value}

    def kill(self, agent_id: str) -> dict[str, Any]:
        """Stop a participant. The bluntest control an exchange has.

        Reported back with the symbols it pulled orders in, rather than a bare
        acknowledgement: an operator reaching for this needs to know what
        actually came out of the book, and "done" is not that.
        """
        venue = self.market.venue
        if agent_id not in venue.accounts:
            return {"ok": False, "error": f"unknown participant {agent_id}"}
        touched = venue.kill(AgentId(agent_id))
        return {"ok": True, "agent_id": agent_id, "symbols": touched}

    def revive(self, agent_id: str) -> dict[str, Any]:
        venue = self.market.venue
        venue.revive(AgentId(agent_id))
        return {"ok": True, "agent_id": agent_id}

    def uncross(self, symbol: str) -> dict[str, Any]:
        venue = self.market.venue
        if venue.registry.get(symbol) is None:
            return {"ok": False, "error": f"unknown symbol {symbol}"}
        if venue.session(symbol) is SessionState.CONTINUOUS:
            return {"ok": False, "error": f"{symbol} is already trading continuously"}
        result = venue.uncross(symbol)
        return {
            "ok": True,
            "session": venue.session(symbol).value,
            "auction": None if result is None else result.to_dict(),
        }

    def indicative(self, symbol: str) -> dict[str, Any] | None:
        """Where an auction in progress would clear, as a price.

        ``AuctionResult.to_dict`` answers in ticks, which is right for the
        exchange and wrong for anything published. The socket already converts
        this same figure -- ``books[symbol].indicative`` is "5003.00" -- while
        the ladder endpoint was handing out 20012 for the same auction, on the
        same contract, at the same instant. One name, two units, four times
        apart. Nothing draws the ladder's copy yet, which is the only reason
        it never appeared on a screen.
        """
        venue = self.market.venue
        instrument = venue.registry.get(symbol)
        if instrument is None:
            return None
        result = venue.indicative(symbol)
        if result is None:
            return None
        payload = result.to_dict()
        payload["price"] = str(instrument.from_ticks(int(result.price)))
        return payload
