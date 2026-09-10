"""Gueant, Lehalle and Fernandez-Tapia, with the measured adverse selection in.

The paper is "Dealing with the inventory risk: a solution to the market making
problem", arXiv:1105.3115. It solves the same control problem Avellaneda and
Stoikov set up, but with a hard inventory bound ``Q`` and without a terminal
date, and the payoff for doing so is that the optimal quotes have a closed
asymptotic form rather than needing a lattice solved backwards from ``T``. That
form is what is implemented here, together with the section 5.2 extension in
which each trade moves the reference price by ``xi`` and the maker charges half
of that on each side:

    c   = sqrt( s^2 g / (2 k A) * (1 + g/k)^(1 + k/g) )
    d_b = (1/g) ln(1 + g/k) + ((2q + 1)/2) c + xi/2
    d_a = (1/g) ln(1 + g/k) - ((2q - 1)/2) c + xi/2

with ``g`` the risk aversion, ``k`` and ``A`` the decay and the level of the
order arrival intensity ``A exp(-k d)``, ``s`` the volatility and ``q`` the
inventory in lots. The bid distance grows and the ask distance shrinks as
inventory rises, so both quotes move down together: the maker pays to be
flattened. The first term is the whole of the width, the second is the whole of
the skew, and the third is the price of being traded with by somebody who knows
something.

``xi`` is not invented here. ``StrategyAgent`` measures the drift of the mid
over the second after each of this strategy's own fills, per side, and hands it
back as ``view[symbol].markout``, which is exactly what a desk computes from
its own trades and exactly what section 5.2 calls for. The sign convention on
the way in is the adapter's: positive means the mid moved the strategy's way,
so the damage this model wants is the negative part, and a side that has not
been hurt contributes nothing.

Where the model is applied is a measurement, not the paper
---------------------------------------------------------

CONTRIBUTING.md records two attempts to defend the makers on this venue by width,
both of which made the loss worse: clamping quotes so they never cross the
touch was 9.4x worse, and widening each side by half the measured markout was
2.2x worse. The second of those is literally the ``xi/2`` term above, so it was
worth finding out whether the finding transfers before inheriting it.

**It does not transfer, and what the width term does is not what the headline
number suggests.** Measured over 300s on seeds 7 and 11 with this strategy on a
20,000,000 seat, turning it on takes the total from -12,559,906 and -12,962,477
to -2,191,340 and -3,167,307, which is 5.7x and 4.1x better.

Read per lot instead and the improvement goes away: -43.3 and -46.5 become
-46.3 and -61.5. Every lot this strategy trades is slightly *worse* with the
width term on. The whole of the gain is that it trades 290,158 and 278,941 lots
without it and 47,377 and 51,512 with it, a sixth as many. So the honest
description is not that widening improves the quote, it is that widening is a
brake, and on a venue where adverse selection runs at 124% and 140% of this
strategy's own loss the most valuable thing a maker can do is trade less. Worth
saying plainly, because "5.7x better" invites the wrong fix.

The two findings agree about the mechanism. CONTRIBUTING.md gives the reason widening
failed there, and the reason is a feedback loop: the three incumbents are
roughly 98% of the book, so widening *them* widens the mid the informed traders
price off, and those traders size by ``|edge| / uncertainty``, so the wider mid
makes them trade bigger and the brake never engages. This strategy is one
participant in a book it does not set, holding about a twenty-fifth of the
lots, and cannot move that mid, so the brake engages. Width is a bad defence
for the maker who *is* the market and an ordinary one for everybody else.

Even so ``xi`` is decomposed here into the part that widens and the part that
skews, so the two can be priced separately and so that finding stays checkable.
Writing ``m = (xi_b + xi_a)/4`` and ``t = (xi_b - xi_a)/4``, the paper's
``xi_b/2`` and ``xi_a/2`` are ``m + t`` and ``m - t``. The ``m`` half is
symmetric and is pure width. The ``t`` half is antisymmetric and moves the
centre of the market down when buying is being punished more than selling,
which is skew and costs nothing in width. ``xi_width`` scales the first and
``xi_skew`` the second, and both default to one, which is the paper exactly.
Both earn it: with the width term on, dropping the skew half cost 1.5x and
1.05x when it was measured, -2,763,780 and -3,940,565 becoming -4,108,570 and
-4,136,353.

The third channel is size, which the paper does not have and which CONTRIBUTING.md
points at directly: a fill at distance ``d`` on a side whose measured markout
is ``xi`` earns ``d - xi``, so a side whose edge has gone is a side to show
less on, or none at all. ``TwoSided`` is built for that. Its own docstring says
a ``None`` side is "a legitimate and frequently correct answer", and it is
distinct from a size of zero, which is not representable. Measured with the
other two channels off, it is worth 1.14x on the mean of the two seeds.

What is kept and what is replaced
---------------------------------

**Kept.** The closed form above, the hard bound ``Q`` applied by refusing to
add to the side that would breach it, and the interpretation of every
parameter. Unlike Avellaneda-Stoikov this model has no terminal date to lose:
the asymptotic regime the closed form comes from is the stationary one, so
there is no ``(T - t)`` here to be unable to compute. That is the one respect
in which GLFT survives the move to bounded payoffs better than the model it
generalises.

**Replaced.** ``s`` is state-dependent and estimated rather than constant,
through :class:`~arena.strategies.making.avellaneda.BoundedVolatility`, for the
reason that module gives: a constant volatility on a claim that must settle
inside ``spec.value_bounds`` lets the price diffuse out of the interval, and
the standard construction for a bounded claim scales the diffusion by ``(p -
low) * (high - p)`` over the range so that it vanishes at both bounds. That is
also what stops the skew term running away in the option books, where a
contract pinned at zero has no volatility to be paid for carrying.

Units are the same as in :mod:`~arena.strategies.making.avellaneda`: fractions
of the contract's own value range everywhere, except ``k``, which is given in
ticks and converted per contract, because the measured half-spread of this book
is a tick quantity (3.5 to 10.0 ticks across 47 contracts, against 0.00007 to
0.075 as fractions of range).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from arena.exchange.types import Side
from arena.strategies.base import MarketView, Quote, TwoSided
from arena.strategies.making.avellaneda import (
    BoundedVolatility,
    place,
    reference_price,
)

__all__ = ["Distances", "GueantLehalleFT"]


@dataclass(frozen=True)
class Distances:
    """One symbol's solved quote distances, before the grid.

    In fractions of the contract's value range, signed as the paper writes
    them: both are distances *from* the reference price, so the bid sits at
    ``reference - bid`` and the ask at ``reference + ask``. Either can go
    negative, which is the model asking to quote through the reference on one
    side because inventory makes that trade worth having, and is not an error.
    """

    reference: float
    bid: float
    ask: float
    base: float
    skew_per_lot: float
    sigma: float
    inventory: int
    xi_bid: float
    xi_ask: float


class GueantLehalleFT:
    """The closed-form asymptotic quotes, with a bound and a measured ``xi``."""

    def __init__(
        self,
        gamma: float = 40.0,
        k_ticks: float = 0.2,
        arrival_rate: float = 40.0,
        quote_size: int = 12,
        inventory_bound: int = 90,
        *,
        xi_width: float = 1.0,
        xi_skew: float = 1.0,
        xi_size: bool = True,
        min_half_spread_ticks: float = 0.5,
        cross_touch: bool = False,
        quote_without_reference: bool = False,
        volatility: BoundedVolatility | None = None,
    ) -> None:
        # Every one of these divides something in the closed form, and a zero
        # would surface as a ZeroDivisionError several frames away from the
        # parameter that caused it. Both papers require all three positive:
        # gamma is a CARA coefficient, and `A exp(-k d)` is not an intensity
        # without a positive `A` and `k`.
        if gamma <= 0 or k_ticks <= 0 or arrival_rate <= 0:
            raise ValueError("gamma, k_ticks and arrival_rate must be positive")
        if quote_size < 1 or inventory_bound < 1:
            raise ValueError("a maker showing no lots is not a maker")
        # Risk aversion per unit of the contract's value range, the same number
        # and the same calibration as `AvellanedaStoikov` uses, so the two are
        # comparable: the incumbent makers here shift their reservation price
        # by 0.0000833 of the range per lot, and at the measured median
        # normalised volatility of 0.00075 per root second that is what gamma =
        # 40 reproduces. The two models spend it very differently, which is the
        # point of running both. On a win rate future at its measured
        # volatility of 0.0024 the skew here is 1.3 ticks a lot against
        # Avellaneda-Stoikov's 34, because this one is the stationary answer
        # and that one is a variance accumulating over a horizon.
        self.gamma = gamma
        # Order arrival decay per tick, so the base half-spread `(1/gamma) ln(1
        # + gamma/k)` is close to `1/k_ticks`, which is 5.0 ticks here against
        # a measured median book half-spread of 6.0 (p10 4.5, p90 9.0, 47
        # contracts, seed 7, 600s).
        self.k_ticks = k_ticks
        # `A` in `A exp(-k d)`, in lots a second at zero distance. Forty,
        # against a measured median traded volume of 37.6 lots a second per
        # contract on seed 7 over 600s. It enters only through `1 / sqrt(A)` in
        # the skew, so the model is not sensitive to it: a factor of four in A
        # is a factor of two in the per-lot skew.
        self.arrival_rate = arrival_rate
        # The same mandate `AvellanedaStoikov` carries, deliberately, so that a
        # comparison between the two is a comparison of models rather than of
        # sizes. Both are set by what a seat can collateralise: `maker_capital`
        # over the 47 contracts listed here is 19,941,816 at 12 lots and a 90
        # lot bound, just inside a 20,000,000 account, against the 240,474,840
        # the incumbents are funded for their 30 and 1,200.
        self.quote_size = quote_size
        self.inventory_bound = inventory_bound
        # Whether to spend the measured markout on width, which is the paper's
        # own prescription and is the intervention CONTRIBUTING.md records as 2.2x
        # worse when it was applied to the incumbent makers. Measured again for
        # this strategy on a 20,000,000 seat over 300s, and it is 5.7x and 4.1x
        # *better* on seeds 7 and 11: -12,559,906 and -12,962,477 become
        # -2,191,340 and -3,167,307.
        #
        # The mechanism is not the one that number suggests, and the module
        # docstring says so at length. Per lot the width term is slightly
        # worse, -43.3 and -46.5 going to -46.3 and -61.5. All of the gain is
        # volume: 290,158 and 278,941 lots become 47,377 and 51,512. Width here
        # is a brake, and a brake pays because adverse selection is 124% and
        # 140% of this strategy's loss.
        self.xi_width = xi_width
        # The antisymmetric half, which moves the centre rather than the width.
        # This is the channel the same evidence says to prefer, and it is free:
        # a maker being picked off harder on one side has learnt where the
        # informed flow is, and a lower centre is the answer that does not
        # widen the mid everyone else prices off.
        self.xi_skew = xi_skew
        # The third channel, which the paper does not have. A fill at distance
        # `d` on a side whose measured markout is `xi` earns `d - xi`, so size
        # is shown in proportion to the edge that survives and a side with none
        # left is not quoted.
        self.xi_size = xi_size
        self.min_half_spread_ticks = min_half_spread_ticks
        # Whether a quote may be posted through the other side of the book.
        # The closed form has no notion of a touch and its distances go
        # negative once `|q|` passes `base / c`, which with the measured
        # per-lot skew of 1.15 ticks on a win rate future and a base of 5.0 is
        # five lots. A twelve lot quote clears that on its first fill, so left
        # true this strategy takes rather than makes.
        #
        # Measured with the width term on, over 300s on seeds 7 and 11:
        # crossing costs 5.1x and 3.6x, taking the total from -2,763,780 and
        # -3,940,565 to -19,592,181 and -14,020,819, the passive share of its
        # own lots from 75.3% and 71.5% to 14.9% and 14.8%, and the orders the
        # venue refuses for collateral from 127 and 165 to 12,019 and 3,344.
        # CONTRIBUTING.md records clamping to the touch costing the incumbents 9.4x
        # and that is a different experiment: they are the book, and a maker
        # that cannot cross has nowhere to unwind into.
        self.cross_touch = cross_touch
        self.quote_without_reference = quote_without_reference
        self.volatility = volatility or BoundedVolatility()

    # -- the model ---------------------------------------------------------

    def solve(self, view: MarketView, symbol: str) -> Distances | None:
        """The closed form, on this contract's normalised scale."""
        symbol_view = view.get(symbol)
        if symbol_view is None:
            return None
        low, high = (float(bound) for bound in symbol_view.bounds)
        span = high - low
        if span <= 0:
            return None
        price = reference_price(symbol_view)
        if price is None:
            if not self.quote_without_reference:
                return None
            level = 0.5
        else:
            level = (float(price) - low) / span

        self.volatility.observe(symbol, view.now, level)
        sigma = self.volatility.sigma(symbol, level)

        instrument = symbol_view.instrument
        span_ticks = float(span / float(instrument.tick_size))
        k = self.k_ticks * span_ticks
        gamma = self.gamma
        ratio = gamma / k

        base = math.log1p(ratio) / gamma
        # `(1 + g/k)^(1 + k/g)` through logs. Written directly it is a small
        # number raised to a very large power: on a future here `g/k` is
        # 0.00026 and the exponent is 3,847, and the whole thing is 2.718 to
        # three places because that limit is `e`. log1p keeps the small end
        # exact instead of losing it to the 1.
        bracket = math.exp((1.0 + 1.0 / ratio) * math.log1p(ratio))
        skew_per_lot = math.sqrt(
            sigma * sigma * gamma / (2.0 * k * self.arrival_rate) * bracket
        )

        inventory = max(-self.inventory_bound, min(self.inventory_bound,
                                                   symbol_view.position))
        bid = base + ((2 * inventory + 1) / 2.0) * skew_per_lot
        ask = base - ((2 * inventory - 1) / 2.0) * skew_per_lot

        xi_bid = self._xi(symbol_view, Side.BUY, span)
        xi_ask = self._xi(symbol_view, Side.SELL, span)
        # The paper's `xi/2` on each side, split into the symmetric half that
        # is width and the antisymmetric half that is skew, so the two can be
        # priced separately. At `xi_width = xi_skew = 1` this is `xi_b/2` and
        # `xi_a/2` exactly.
        common = 0.25 * (xi_bid + xi_ask)
        tilt = 0.25 * (xi_bid - xi_ask)
        bid += self.xi_width * common + self.xi_skew * tilt
        ask += self.xi_width * common - self.xi_skew * tilt

        floor = self.min_half_spread_ticks / span_ticks
        # Applied to the pair rather than to each side, because pinning one
        # side up to a floor on its own would move the centre of the market,
        # and in this model the centre is where the inventory information is.
        # Only the crossed case is repaired, and it is repaired symmetrically.
        shortfall = 2.0 * floor - (bid + ask)
        if shortfall > 0:
            bid += shortfall / 2.0
            ask += shortfall / 2.0
        return Distances(
            reference=level,
            bid=bid,
            ask=ask,
            base=base,
            skew_per_lot=skew_per_lot,
            sigma=sigma,
            inventory=inventory,
            xi_bid=xi_bid,
            xi_ask=xi_ask,
        )

    @staticmethod
    def _xi(symbol_view, side: Side, span: float) -> float:
        """The measured damage per lot on one side, in fractions of range.

        ``markout`` arrives in contract price units per lot and signed so that
        positive means the mid moved the strategy's way. GLFT's ``xi`` is the
        size of the move *against* the maker, so only the negative part is
        adverse selection and a side that has been rewarded contributes zero
        rather than a subsidy. ``None`` means not enough of this strategy's own
        fills have matured yet, which is the honest answer early on and is
        treated as no evidence rather than as no damage.
        """
        measured = symbol_view.markout.get(side)
        if measured is None:
            return 0.0
        return max(0.0, -float(measured)) / span

    # -- the strategy interface --------------------------------------------

    def symbols(self, view: MarketView) -> Sequence[str]:
        return sorted(view.symbols)

    def quote(self, view: MarketView, symbol: str) -> TwoSided:
        solved = self.solve(view, symbol)
        if solved is None:
            return TwoSided()
        symbol_view = view[symbol]
        low, high = (float(bound) for bound in symbol_view.bounds)
        span = high - low

        position = symbol_view.position
        rooms = {
            Side.BUY: self.inventory_bound - position,
            Side.SELL: self.inventory_bound + position,
        }
        distances = {Side.BUY: solved.bid, Side.SELL: solved.ask}
        xis = {Side.BUY: solved.xi_bid, Side.SELL: solved.xi_ask}
        quotes: dict[Side, Quote | None] = {Side.BUY: None, Side.SELL: None}
        for side in (Side.BUY, Side.SELL):
            room = rooms[side]
            if room <= 0:
                continue
            distance = distances[side]
            size = min(self.quote_size, room)
            # The size channel, and only where it is about edge. A distance
            # that has already gone negative is the inventory term asking to
            # pay to be flattened, which is a risk decision rather than a
            # profitable one, and pulling that side would strand the position
            # the skew exists to unwind. Measured before this was separated
            # out, the gate fired on every flattening quote and the strategy
            # showed one side of the market for 92% of its lots.
            if self.xi_size and xis[side] > 0 and distance > 0:
                edge = distance - xis[side]
                if edge <= 0:
                    # Nothing left to earn on this side. Pulling it is the
                    # cheapest defence there is and the only one that does not
                    # widen the mid the informed traders price off.
                    continue
                size = max(1, round(size * (edge / distance)))
            offset = -distance if side is Side.BUY else distance
            # Each side snapped and clamped on its own. Clamping the centre
            # into the range instead is in the CONTRIBUTING.md table of things that
            # look like improvements and are not: it makes the mid a function
            # of the half-spread, so two contracts worth the same amount price
            # 40 points apart.
            price = place(
                symbol_view,
                side,
                low + (solved.reference + offset) * span,
                cross_touch=self.cross_touch,
            )
            if price is None:
                continue
            quotes[side] = Quote(price=price, size=size)
        return TwoSided(bid=quotes[Side.BUY], ask=quotes[Side.SELL])
