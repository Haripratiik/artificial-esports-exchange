"""A constant half-spread, skewed by inventory. The control in every comparison.

This is the simplest thing that is still a market maker, and it is here to be
the baseline every other strategy in this package is measured against. It shows
a fixed width around a reference price, shades the centre against its own
position so that the trade which flattens it is the more attractive one, and
stops quoting a side altogether once that position reaches its limit. There is
no model of the counterparty in it at all, which is the point: what a model of
the counterparty buys is only visible against something that has none.

    centre = reference - skew * position
    bid    = centre - half_spread
    ask    = centre + half_spread

The one choice worth explaining is that the half-spread is a fraction of the
contract's *settlement range* rather than a count of ticks or a percentage of
the price. Every contract here settles inside a known interval and those
intervals differ by three orders of magnitude: `HALCYON_FORMAT_SPD` spans
80,000 ticks and `VANTA_SOLO_GT100` spans 100. The incumbent makers use a
constant number of ticks, so mm-1's five-tick half-spread is 0.00625% of the
range on the spread contract and 5.0% of it on a binary, a factor of 800
between two books quoted by one parameter. A percentage of the price is no
better, because a binary trading at 0.02 and a future at 4,670 cannot share
one, and a contract worth nothing has no percentage at all. A fraction of the
range is the only one of the three that means the same thing on every contract
in this market, and it is available because every contract here has a range
that is written down.

Three things this baseline does that a first strategy usually forgets, and each
of them was measured on this market rather than reasoned about.

It never posts a price that would cross the book it can see. A limit order sent
through the touch is a market order with extra steps: it takes, it pays the
spread instead of earning it, and it books as an aggressive fill. Measured on
seed 7 over 180 seconds, the three incumbent makers were the aggressor on 42.7%,
41.2% and 44.4% of their own fills, and a maker that takes that often is not
making. Capping the bid one tick below the visible offer and flooring the ask
one tick above the visible bid costs nothing while the quote is already inside
the touch, which is where a maker's quote is supposed to be, and takes this
strategy's own aggressive share to 30.6% on the same run.

That is a defence rather than a guarantee, and it cannot be more than one. The
book this strategy reads is conflated to half its wake interval and arrives over
a wire, so the touch it caps against is 160ms old and the market moves inside
that. The exchange has the order type that would make crossing impossible,
`TimeInForce.POST_ONLY`, and `StrategyAgent` sends every quote GTC with no way
for a strategy to ask for anything else. Measured by making it ask, on seed 7
over 180 seconds, the aggressive share of this strategy's own fills goes from
30.7% to 1.4%. It is not free: it also refused 31,804 orders that would have
crossed and took the fill count from 26,131 to 4,872.

It ignores a touch that is outside the contract's own settlement range, because
that is not a price. Market-on-open interest rests at 2^62 so that it crosses
every candidate in the auction, and the quote feed publishes that as if it were
a quote: measured on seed 7 over 120 seconds, 479 of 2,256 top-of-book samples
carried it, 21.2%, across 46 of the 47 books. A strategy that reads `best_bid`
and believes it will bid 4,611,686,018,427,387,904 for something that settles
under 10,000. The range test needs no knowledge of that number and would catch
any other impossible price the same way.

It refuses to anchor on a mid that is its own reflection. With a position on the
centre is the reference less the skew, so if this strategy is both sides of the
touch then the mid it reads back is the centre it just posted, and the next
requote subtracts the skew again from a number that already had it subtracted.
Against an otherwise identical strategy that always anchors on the mid, run on
seed 7 for 180 seconds under the same agent id and so the same wake jitter, the
guard is worth 16.2M: +17.3M of attributed P&L against +1.1M, on 5% fewer
fills. When the touch is this strategy's own on both sides the anchor falls back
to the last print, which is what the incumbent makers anchor on, for this same
reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from arena.exchange.types import Side
from arena.strategies.base import MarketView, Quote, SymbolView, TwoSided, snap

__all__ = ["FixedSpread", "priced_touch"]


def priced_touch(v: SymbolView) -> tuple[Decimal | None, Decimal | None]:
    """The best bid and offer, keeping only the ones that are prices.

    A quote outside the contract's settlement range is not a price the contract
    could ever pay, so nothing that trades on it can be true. In this market the
    concrete case is market-on-open interest, which rests at 2^62 so that it
    crosses every candidate in the opening auction and is then published on the
    quote feed as the best bid. `SnapshotBook` filters it out of everything a
    person looks at, through `_first_priced`, and `VenueAgent.top_of_book` does
    not, so an agent's own book carries it: measured on seed 7, 21.2% of
    top-of-book samples over the first 120 seconds, in 46 of 47 books.

    Written as a bounds test rather than as a test against that number, since
    the strategy has no business knowing what the exchange uses to mean "any
    price" and the bounds are already the thing it is not allowed to quote
    outside of.
    """
    low, high = v.bounds
    bid = v.best_bid if v.best_bid is not None and low <= v.best_bid <= high else None
    ask = v.best_ask if v.best_ask is not None and low <= v.best_ask <= high else None
    return bid, ask


class FixedSpread:
    """Two-sided quoting at a constant width, with a linear inventory skew."""

    def __init__(
        self,
        half_spread_fraction: float = 0.0004,
        skew_fraction: float = 0.02,
        position_limit: int = 250,
        quote_size: int = 10,
        min_half_spread_ticks: int = 1,
    ) -> None:
        # Half the quoted width, as a fraction of the contract's settlement
        # range. 0.0004 is the width the market around this strategy actually
        # shows: measured on seed 7 over 120 simulated seconds, 1,043 two-sided
        # samples across all 47 books, the incumbent makers' median touch
        # spread was 0.00075 of the range, and half of that is 0.00038.
        #
        # Where a fraction of the range is least convincing is the binaries,
        # and it is worth naming rather than hiding. A range of 1.00 with a
        # 0.01 tick is 100 ticks wide, so 0.0004 of it is a twenty-fifth of a
        # tick and the floor below does all the work: this maker shows two
        # ticks on a book where the incumbents show twelve. That is a real cost
        # of the parameterisation rather than an accident of the number, and it
        # is the first thing a strategy built on this one should improve.
        self.half_spread_fraction = half_spread_fraction
        # How far a full position moves the centre, again as a fraction of the
        # range. Set to shade by the same amount per lot as the widest
        # incumbent: mm-1 moves its reservation price by 0.10 of the range over
        # a 1,200 lot limit, which is 8.33e-5 of the range per lot, and 0.02
        # over 250 lots is 8.0e-5. The same shading, expressed against a limit
        # this strategy's capital can actually reach.
        self.skew_fraction = skew_fraction
        # The hard stop. Past it the strategy shows one side only, which says
        # something a very wide two-sided quote does not: this book is closed to
        # more risk in that direction at any price.
        #
        # 250 rather than the incumbents' 1,200 because a strategy account here
        # opens with 20M and being two-sided in 47 books is what that has to
        # cover. Measured over 180 seconds with these defaults, peak posted
        # collateral was 13.8M on seed 7 and 15.3M on seed 3 against 20M of
        # cash, and not one of the 2,732 refusals in a third run of the same
        # thing was for want of collateral: every one was a cancel arriving
        # after the order it named had already filled.
        self.position_limit = position_limit
        self.quote_size = quote_size
        # A quote narrower than a tick is not a quote. The floor binds only
        # where the tick is coarse against the range, which in this market
        # means the eight binaries and nothing else.
        self.min_half_spread_ticks = min_half_spread_ticks

    def symbols(self, view: MarketView) -> Sequence[str]:
        """Everything the agent lists. A baseline that chose would not be one."""
        return list(view.symbols)

    def quote(self, view: MarketView, symbol: str) -> TwoSided:
        v = view[symbol]
        instrument = v.instrument
        tick = instrument.tick_size
        low, high = v.bounds
        span = float(high - low)
        bid_touch, ask_touch = priced_touch(v)

        reference = self._reference(v, bid_touch, ask_touch)
        half = max(
            self.half_spread_fraction * span,
            self.min_half_spread_ticks * float(tick),
        )
        skew = self.skew_fraction * span * v.position / max(1, self.position_limit)
        centre = reference - skew

        # One tick inside the visible touch is the most aggressive price that is
        # still passive. The book here is the stale one this strategy would
        # really have had, so this is not a guarantee against ever crossing,
        # only against doing it on purpose.
        bid_price = centre - half
        if ask_touch is not None:
            bid_price = min(bid_price, float(ask_touch - tick))
        ask_price = centre + half
        if bid_touch is not None:
            ask_price = max(ask_price, float(bid_touch + tick))

        bid = snap(instrument, Side.BUY, min(max(bid_price, float(low)), float(high)))
        ask = snap(instrument, Side.SELL, min(max(ask_price, float(low)), float(high)))
        pair = self._separate(bid, ask, low, high, tick)
        if pair is None:
            # The range has no room for a two-sided quote at this width, which
            # can only happen on a contract whose whole range is a tick or two.
            return TwoSided()
        bid, ask = pair

        # The limit is enforced through the size rather than through the price,
        # so the strategy never sends an order that would have to be refused. A
        # side with no room left is simply absent, which `TwoSided` says with
        # ``None`` rather than with a zero it cannot represent.
        bid_size = min(self.quote_size, self.position_limit - v.position)
        ask_size = min(self.quote_size, self.position_limit + v.position)
        return TwoSided(
            bid=Quote(bid, bid_size) if bid_size > 0 else None,
            ask=Quote(ask, ask_size) if ask_size > 0 else None,
        )

    # -- the pieces --------------------------------------------------------

    def _reference(
        self, v: SymbolView, bid: Decimal | None, ask: Decimal | None
    ) -> float:
        """Where this strategy thinks the market is, without asking itself.

        The mid, unless this strategy is the touch on both sides, in which case
        the mid is the centre it posted last time and reading it back turns the
        inventory skew into a ratchet. The last print is the answer then, for
        the same reason the incumbent makers anchor on prints rather than on the
        book: an execution happened to somebody, and a quote did not.

        The midpoint of the settlement range is the last resort, for a book that
        has never traded and has no quote either. Someone has to name an opening
        price, and on a venue with no prior session that someone is a maker.
        """
        own_touch = v.working_bid == bid and v.working_ask == ask
        if bid is not None and ask is not None and not own_touch:
            return float(bid + ask) / 2.0
        if v.last is not None:
            return float(v.last)
        if bid is not None and ask is not None:
            return float(bid + ask) / 2.0
        low, high = v.bounds
        return float(low + high) / 2.0

    def _separate(
        self, bid: Decimal, ask: Decimal, low: Decimal, high: Decimal, tick: Decimal
    ) -> tuple[Decimal, Decimal] | None:
        """Keep the two sides at least a tick apart, inside the range.

        Both quotes are pinned into the settlement range independently, so on a
        contract whose fair value sits against a boundary they can land on the
        same tick, and this strategy would then be trading with itself.
        Widening wherever there is room is the same repair the surface maker
        makes, and giving up a book entirely is better than crossing in it.
        """
        if ask > bid:
            return bid, ask
        if bid - tick >= low:
            return bid - tick, ask
        if ask + tick <= high:
            return bid, ask + tick
        return None
