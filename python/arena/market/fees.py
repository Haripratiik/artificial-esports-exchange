"""Maker-taker fees, and where the money goes.

Fees are not a cosmetic surcharge. They change market making qualitatively,
because they decide the *minimum* spread a maker can quote and still survive: a
maker earning a rebate can quote inside a maker paying a fee, and a taker fee
sets the size of mispricing an arbitrageur must see before acting at all. The
no-arbitrage band, the effective spread, and whether a strategy is profitable at
all are all downstream of this file.

Conservation
------------

Value still has to be conserved exactly. A fee does not evaporate: it moves from
the trader to a venue account, which is a real account holding real cash and
included in the conservation check like any other. Netting fees against nothing,
or discarding them, would make the ledger's central invariant a lie the moment
fees were switched on.

That also makes venue revenue *measurable* rather than notional. With a rebate
schedule the venue can genuinely lose money, and this is what shows it.

Rounding
--------

Fees are computed on integer minor units and always round **toward the venue**:
a charge rounds up, a rebate rounds down in magnitude. Rounding the other way
would let a strategy of many tiny fills extract a fraction of a unit per trade,
which is exactly the kind of leak the integer ledger exists to make impossible.
"""

from __future__ import annotations

from dataclasses import dataclass

from arena.portfolio.money import Money

__all__ = ["FeeSchedule", "FREE", "MAKER_TAKER"]

BASIS_POINTS = 10_000


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """Charged on notional, in basis points. Negative means a rebate.

    ``taker_bps`` is paid by the aggressor, the order that crossed the spread.
    ``maker_bps`` is paid by the resting side, and is usually negative, because
    a venue pays for the liquidity that makes it worth trading on.
    """

    taker_bps: float = 0.0
    maker_bps: float = 0.0
    # What an auction fill pays, on both sides. ``None`` means the taker rate.
    #
    # An auction has no aggressor, so the obvious thing is to bill every fill
    # at the maker rate. That was done, and it is wrong in a way only running
    # an auction shows. A maker rate is usually a rebate, so a venue billing
    # both sides of its own opening cross *pays out* on every share it crosses.
    # Measured on the first session that ran one: 26 opening auctions took
    # venue revenue to **minus 1,251**, which is a venue paying people to let
    # it open. Exchanges charge for cross executions rather than paying for
    # them, and this defaults to the taker rate for that reason.
    auction_bps: float | None = None

    @property
    def free(self) -> bool:
        return (
            self.taker_bps == 0.0
            and self.maker_bps == 0.0
            and not self.auction_bps
        )

    def rate(self, aggressor: bool, auction: bool = False) -> float:
        if auction:
            return self.taker_bps if self.auction_bps is None else self.auction_bps
        return self.taker_bps if aggressor else self.maker_bps

    def charge(
        self, notional: int | Money, aggressor: bool, auction: bool = False
    ) -> Money:
        """Fee on ``notional`` minor units. Positive is charged, negative paid.

        Both directions round toward the venue, so a participant can never end
        up ahead on the rounding no matter how it slices its orders.
        """
        rate = self.rate(aggressor, auction)
        if rate == 0.0:
            return Money(0)
        exact = abs(int(notional)) * rate / BASIS_POINTS
        if exact > 0:
            # A charge: round up, so the venue is never short.
            return Money(int(-((-exact) // 1)))
        # A rebate: round down in magnitude, for the same reason.
        return Money(-int((-exact) // 1))

    def to_dict(self) -> dict[str, float]:
        return {
            "taker_bps": self.taker_bps,
            "maker_bps": self.maker_bps,
            "auction_bps": self.rate(False, auction=True),
        }


# No fees. The default everywhere, so that switching them on is a deliberate
# act and every existing measurement keeps its meaning.
FREE = FeeSchedule()

# A conventional maker-taker schedule: the taker pays, the maker is paid rather
# less, and the difference is what the venue keeps. Set so the venue's net take
# is positive on every trade; a schedule where rebates exceed fees is a venue
# paying people to trade with each other, which is a real thing that happens
# and a real way to go bankrupt.
MAKER_TAKER = FeeSchedule(taker_bps=2.0, maker_bps=-1.0)
