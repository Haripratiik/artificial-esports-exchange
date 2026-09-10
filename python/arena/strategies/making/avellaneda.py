"""Avellaneda-Stoikov, adapted to a claim that settles inside a known bound.

The paper is Avellaneda and Stoikov, "High-frequency trading in a limit order
book", Quantitative Finance 8(3):217-224, 2008. Two equations are the whole of
it, and both are implemented here literally:

    reservation price   r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
    optimal spread      d_a + d_b  = gamma * sigma^2 * (T - t)
                                     + (2 / gamma) * ln(1 + gamma / k)

The first says where to centre the market: away from your inventory, so that
the trade which flattens you is the more attractive one. The second says how
wide, and the thing worth noticing about it is what it does *not* contain.
There is no ``q`` in it. Avellaneda and Stoikov say so explicitly of their
equation 25, and it is not an approximation: with exponentially distributed
arrival intensities the two first-order conditions separate, so inventory moves
the centre of the market and never its width.

**This is the opposite of what the makers already in this repository do.**
``arena.agents.market_maker.MarketMaker`` computes ``half = half_spread * (1 +
2 * pressure)`` with ``pressure`` the fraction of its position limit used, so
its spread widens with inventory and its centre moves as well. That is a
defensible desk heuristic and it is not this model. Which of the two is right
here is a measurement, not an argument, and keeping the paper's behaviour
intact is what makes the measurement mean anything.

A related measurement is worth putting at the top, because it reverses
something this repository already believed. CONTRIBUTING.md records widening backfiring
twice on this venue, by 9.4x and by 2.2x, and explains why: three makers are
roughly 98% of the book, so widening *them* widens the mid everybody else
prices off, and the informed traders size by ``|edge| / uncertainty`` and
therefore trade bigger. That mechanism needs the maker to be the market. This
strategy is one participant among four holding a fiftieth of the lots, and for
it width behaves the way the papers say. Capping the width here at the measured
p90 half-spread of the book, which looked obviously right, is 2.1x *worse* on a
seat that can fund the mandate, so ``max_half_spread_ticks`` defaults to
``None`` and the equation above runs uncapped. The number is with the parameter.

What is kept from the paper
---------------------------

The reservation price and the spread, in the exact algebraic form above. The
independence of spread from inventory. The interpretation of ``k`` as the decay
of order arrival intensity in distance from the mid, and of ``gamma`` as
constant absolute risk aversion.

What is replaced, and why
-------------------------

**The ``(T - t)`` liquidation term does not survive.** Two reasons, and the
second is the interesting one.

The shallow reason is that it cannot be computed. ``SymbolView.seconds_to_expiry``
is documented as "``None`` where the calendar has not been wired", and
``StrategyAgent._symbol_view`` passes ``seconds_to_expiry=None`` unconditionally,
so a strategy running under the adapter never learns ``T``. A model whose
central term is an input nobody supplies is a guard whose input was never
wired, which is the sixth bug class in CONTRIBUTING.md.

The deep reason is that ``(T - t)`` is time to *liquidation at the mid*, and
nothing here is liquidated at the mid. Every contract on this venue settles as
a known function of one bounded scalar, so inventory carried to the end is paid
out at a settlement level inside ``spec.value_bounds`` and not at whatever the
book happens to show. What the term is actually doing in the paper is naming
the variance of the terminal price, and that quantity is available here in
closed form without any clock at all: a price is a martingale on ``[low,
high]``, and a martingale on a bounded interval sitting at normalised level
``u`` has terminal variance at most ``u * (1 - u)``, attained by the two-point
distribution that puts all its mass on the bounds. So the substitution is

    sigma^2 * (T - t)   ->   min( sigma(u)^2 * h ,  u * (1 - u) )

with ``h`` the horizon over which this maker expects to carry inventory before
it recycles rather than the time to expiry. The cap is the settlement bound
entering the model directly. It binds when the horizon is long or the price is
near a bound, it vanishes at both bounds exactly as the volatility does, and it
is what makes the model refuse to charge more inventory risk than the contract
can possibly deliver. Setting ``horizon_seconds=None`` drops the first branch
and quotes the pure hold-to-settlement variance, which is the honest answer for
a maker that intends to carry the position all the way.

The cap earns its place empirically as well as structurally. The mid series in
the deep option books here is violently jumpy: measured over 600s on seed 7,
one deep out of the money put had a median normalised level of 0.000 and a
maximum of 1.000, and its RMS volatility of 0.218 of the range per root second
is two thousand times its median-based estimate of 0.0001. That run predates
the circuit, and the same shape survives it: on the current listing twenty of
the fifty contracts have a median move of exactly zero against an RMS of 0.027
to 0.342. Uncapped, ``sigma^2 * h`` on such a book asks for an inventory
penalty larger than the contract is worth.

**Volatility is state-dependent and estimated, not constant.** A constant sigma
on a bounded claim is incoherent: it lets the price diffuse out of the interval
it must settle inside. The standard construction for a bounded claim is a
diffusion coefficient proportional to ``(p - low) * (high - p)``, normalised by
the range, and that is what :class:`BoundedVolatility` produces. It vanishes at
both bounds, which is the only behaviour under which the interval is absorbing
at the right places and the price cannot leave it.

**A hard inventory bound is added.** The paper has none, and a maker without
one accumulates without bound in a trending market. The bound is expressed as a
refusal to add to the side that would breach it, which is what the incumbent
makers do and for the reason their code gives: quoting and letting the venue
reject burns order ids and hides the constraint from the strategy's own logic.

Units
-----

Everything inside the model is in *fractions of the contract's own value
range*, so ``u = 0`` is the floor of ``spec.value_bounds`` and ``u = 1`` is the
cap. This is not a convenience. The contracts here span 100 ticks (a binary)
to 80,000 ticks (a spread), and a parameter in absolute price units means
completely different things across them. It is also the unit the bounded
construction is written in.

``k`` is the exception and is given in ticks, because the measurement says the
book's own width is a tick quantity. Over 47 contracts on seed 7 for 600s, the
median half-spread ranges from 3.5 to 10.0 ticks, a factor of 2.9, while the
same half-spreads as fractions of range range from 0.00007 to 0.075, a factor
of 1,091. So ``k`` is converted per contract as ``k_ticks * span_in_ticks``,
which is read off the instrument's own public grid and special-cases nothing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal

from arena.exchange.types import Side
from arena.market.instrument import Instrument
from arena.strategies.base import MarketView, Quote, SymbolView, TwoSided, snap

__all__ = [
    "AvellanedaStoikov",
    "BoundedVolatility",
    "Reservation",
    "bounded_shape",
    "on_increment",
    "place",
    "reference_price",
]


def bounded_shape(u: float) -> float:
    """The volatility profile of a claim that must settle inside its bounds.

    ``4 * u * (1 - u)`` on the normalised level, which is ``(p - low) * (high -
    p)`` divided by the square of the half-range. Dimensionless, one at the
    midpoint and zero at both bounds, so a diffusion carrying it as a factor
    cannot push the price out of the interval. Clamped rather than allowed
    negative, because a mid outside the bounds is a book in an odd state and
    not a licence to model a negative variance.
    """
    return max(0.0, 4.0 * u * (1.0 - u))


def reference_price(view: SymbolView) -> Decimal | None:
    """Where this contract is, or ``None`` if the book has never said.

    Deliberately not :attr:`SymbolView.reference`, which is written as ``self.mid
    or self.last or midpoint`` and therefore reads a mid of exactly zero as no
    price at all, falling through to the middle of the settlement range.

    Honestly, that difference does not fire today: measured over 300s on seed
    7, not one of the 47 books had a mid of exactly zero at any of 1,200
    samples, because a bid resting at the floor against an offer one tick above
    it gives a mid of half a tick rather than nothing. But seven of the option
    books sit at a median normalised level of 0.000, which is one cancelled
    offer away from the state that does fire, and the value it would fall
    through to on one of them is 2,350 on a put that is worth nothing. So the
    checks here are against ``None``, and a contract with no price at all gets
    no quote rather than a guessed one.
    """
    mid = view.mid
    if mid is not None:
        return mid
    if view.last is not None:
        return view.last
    return None


def on_increment(instrument: Instrument, side: Side, price: Decimal) -> Decimal:
    """Move a snapped price onto the increment its own price band requires.

    :func:`~arena.strategies.base.snap` reads ``instrument.tick_size`` and
    nothing else, but the grid a contract is actually quoted on is
    ``instrument.increment_at``, which can coarsen above a threshold written
    into the contract. One of the 50 contracts listed here has such a table:
    ``VANTA_OBJECTIVE_WR`` steps in 1.00 above 4,000 and in 0.25 below it, so a
    modelled price of 5,232.25 comes back from ``snap`` off the grid the venue
    will accept. ``TradingAgent.quote`` repairs it before the order is sent, so
    nothing breaks today, but a strategy that emits a price its own contract
    cannot carry is wrong on its own terms and would be wrong for any other
    consumer.

    Repeated rather than done once, for the reason ``TradingAgent._on_grid``
    gives: a single pass can round *into* a coarser band and land off that
    band's grid. Each pass moves one way onto a multiple of a strictly coarser
    increment, so it cannot cycle, and it is bounded anyway because a loop in
    an order path is not a thing to leave unbounded.
    """
    low, high = instrument.value_bounds
    for _ in range(8):
        if instrument.on_grid(price):
            return price
        step = instrument.increment_at(price)
        steps = price / step
        floor = steps.to_integral_value(rounding=ROUND_FLOOR)
        chosen = floor if side is Side.BUY or steps == floor else floor + 1
        price = max(low, min(high, chosen * step))
    return price


def place(
    view: SymbolView, side: Side, price: float, *, cross_touch: bool
) -> Decimal | None:
    """One side of a modelled quote, on the grid and inside the range.

    :func:`~arena.strategies.base.snap` does the tick grid and the range,
    rounding away from the touch so the order never rests a tick better than
    the model asked for, and :func:`on_increment` finishes the grid where the
    contract carries a tick table. What is added on top is the option of
    refusing to quote through the other side of the book, and it is applied to
    each side separately: the CONTRIBUTING.md table records that clamping the
    *centre* of a quote makes the mid a function of the half-spread, so two
    worthless calls priced 40 points apart.

    ``None`` comes back when the side cannot be placed without crossing, which
    happens when the opposite touch is already at the boundary of the
    settlement range and there is no tick left inside it. Not quoting is a
    legitimate answer and :class:`~arena.strategies.base.TwoSided` is built to
    carry it.
    """
    instrument = view.instrument
    if not cross_touch:
        tick = float(instrument.tick_size)
        if side is Side.BUY and view.best_ask is not None:
            price = min(price, float(view.best_ask) - tick)
        elif side is Side.SELL and view.best_bid is not None:
            price = max(price, float(view.best_bid) + tick)
    snapped = on_increment(instrument, side, snap(instrument, side, price))
    if not cross_touch:
        if side is Side.BUY and view.best_ask is not None and snapped >= view.best_ask:
            return None
        if side is Side.SELL and view.best_bid is not None and snapped <= view.best_bid:
            return None
    return snapped


@dataclass
class _Track:
    """One symbol's running volatility state, in normalised units."""

    at: float
    level: float
    mean_absolute: float
    mean_shape: float
    samples: int


class BoundedVolatility:
    """A state-dependent volatility estimated from the strategy's own history.

    The model is the one a bounded claim forces, written on the normalised
    level ``u = (p - low) / (high - low)``:

        du = sigma_hat * u * (1 - u) * dW

    so the local volatility ``sigma(u) = sigma_hat * u * (1 - u)`` vanishes at
    both bounds and is largest in the middle. What is estimated is
    ``sigma_hat``, the level-free scale, and it is estimated from increments
    this strategy has actually seen through its own view rather than assumed.

    Two decisions in here are measurements rather than taste.

    **The estimator is a mean absolute deviation, not a mean square.** A price
    series in this market jumps. Re-measured over 600s on seed 7 across the
    circuit listing, sampling every 250ms: on ``QUILL_SOLO_WR`` the RMS
    normalised volatility is 0.0729 per root second against a median-based
    0.0019, a factor of 39. On twenty of the fifty contracts the median move is
    exactly zero while the RMS runs 0.027 to 0.342, which says the same thing
    more sharply: the ordinary interval on this market has no move in it at
    all, and the whole of a squared estimate is a handful of excursions.
    Squaring hands those excursions the estimate. The figure this docstring
    carried before, a factor of 15, was measured on the retired statistics
    world and is superseded rather than contradicted, and
    ``tests/test_strategies_optimal.py`` holds the re-measurement. So this
    accumulates ``|du|`` and multiplies by ``sqrt(pi / 2)``, which is the
    Gaussian relation ``E|X| = sigma * sqrt(2 / pi)`` and is the standard
    robust substitute.

    **The estimate is transported between levels rather than de-modulated in
    place.** Dividing each increment by the local shape recovers ``sigma_hat``
    directly and is what the SDE literally says to do, and it is unusable here:
    measured across 47 contracts and 2,400 samples on seed 7, the smallest
    shape observed is 0.0004, so that division amplifies a one-tick move by
    four orders of magnitude and one deep out of the money call reported a
    level-free volatility of 0.215 against a raw 0.0007. Instead the running
    shape is tracked alongside the running increment, and the ratio of the
    shape now to the shape then transports the estimate. When the price has not
    moved between the window and now, which is the ordinary case, the ratio is
    one and the answer is exactly the observed volatility.
    """

    def __init__(
        self,
        *,
        gain: float = 0.05,
        sample_interval: float = 0.25,
        warmup: int = 8,
        prior: float = 0.00075,
        transport_cap: float = 4.0,
        shape_floor: float = 1e-3,
    ) -> None:
        # A slow EWMA, matching the shape of the adapter's own markout estimate
        # in `strategy_agent.MARKOUT_GAIN`: the quantity is a running average of
        # a noisy per-sample number and anything more elaborate would be fitting.
        self.gain = gain
        # Increments are only taken this far apart. A strategy is asked to
        # quote again the instant it is filled, so without a floor on the
        # sampling interval the estimator divides a zero increment by a
        # microsecond and reports either nothing or everything.
        self.sample_interval = sample_interval
        self.warmup = warmup
        # What to assume before there is any history. 0.00075 of the range per
        # root second is the median of this estimator's own reading across the
        # 47 contracts listed here, taken at the end of a 300s run on seed 7,
        # where the spread across contracts was p10 0.0000 and p90 0.0067.
        self.prior = prior
        # How far the estimate may be scaled when the price has moved between
        # the estimation window and now. Four, because the transport ratio is
        # unbounded above when the window sat on a bound and the price has
        # since come off it, and an unbounded multiplier on a volatility is an
        # unbounded quote.
        self.transport_cap = transport_cap
        # A division guard, not a model parameter. Measured on seed 7 over
        # 300s, the running shape this estimator actually holds has a tenth
        # percentile of 0.0017 across the 47 contracts and a median of 0.44, so
        # a floor of 0.001 binds only on a book pinned at its bound.
        self.shape_floor = shape_floor
        self._by_symbol: dict[str, _Track] = {}

    def observe(self, symbol: str, now: float, level: float) -> None:
        """Record where this contract is, if enough time has passed to matter."""
        track = self._by_symbol.get(symbol)
        if track is None:
            self._by_symbol[symbol] = _Track(now, level, 0.0, bounded_shape(level), 0)
            return
        dt = now - track.at
        if dt < self.sample_interval:
            return
        absolute = abs(level - track.level) / math.sqrt(dt)
        shape = bounded_shape((level + track.level) / 2.0)
        if track.samples == 0:
            track.mean_absolute = absolute
            track.mean_shape = shape
        else:
            track.mean_absolute += self.gain * (absolute - track.mean_absolute)
            track.mean_shape += self.gain * (shape - track.mean_shape)
        track.at = now
        track.level = level
        track.samples += 1

    def sigma(self, symbol: str, level: float) -> float:
        """Local volatility at ``level``, in fractions of range per root second."""
        track = self._by_symbol.get(symbol)
        if track is None or track.samples < self.warmup:
            base, observed_shape = self.prior, 1.0
        else:
            base = track.mean_absolute * math.sqrt(math.pi / 2.0)
            observed_shape = track.mean_shape
        transport = bounded_shape(level) / max(observed_shape, self.shape_floor)
        return base * min(transport, self.transport_cap)


@dataclass(frozen=True)
class Reservation:
    """One symbol's solved quote, before it is snapped onto the grid.

    Every field is in fractions of the contract's value range, which is the
    unit the model works in throughout. Exposed because the property worth
    testing about this strategy is that ``half_spread`` does not depend on
    ``inventory`` while ``reservation`` does, and that property is about the
    numbers the model produced rather than about what survived tick rounding.
    """

    reference: float
    reservation: float
    half_spread: float
    sigma: float
    variance: float
    inventory: int


class AvellanedaStoikov:
    """Inventory-optimal quoting: the centre moves, the width does not."""

    def __init__(
        self,
        gamma: float = 40.0,
        k_ticks: float = 0.2,
        quote_size: int = 12,
        inventory_bound: int = 90,
        *,
        horizon_seconds: float | None = 3.6,
        min_half_spread_ticks: float = 0.5,
        max_half_spread_ticks: float | None = None,
        cross_touch: bool = False,
        quote_without_reference: bool = False,
        volatility: BoundedVolatility | None = None,
    ) -> None:
        # Both divide something in the paper's own equations, and a zero would
        # surface as a ZeroDivisionError several frames from the parameter that
        # caused it. The paper requires both positive: gamma is a CARA
        # coefficient and `A exp(-k d)` is not an intensity without a positive
        # `k`.
        if gamma <= 0 or k_ticks <= 0:
            raise ValueError("gamma and k_ticks must be positive")
        if quote_size < 1 or inventory_bound < 1:
            raise ValueError("a maker showing no lots is not a maker")
        # Risk aversion, per unit of the contract's value range. Calibrated so
        # that the per-lot reservation shift `gamma * variance` matches what
        # the incumbent makers here are configured to apply: they use
        # `max_skew_fraction=0.10` at `position_limit=1200`, which is 0.0000833
        # of the range per lot. At the measured median normalised volatility of
        # 0.00075 per root second and the 3.6s horizon below, the variance is
        # 0.00000202, so gamma = 0.0000833 / 0.00000202 = 41, rounded to 40.
        #
        # The skew this produces is very sensitive to volatility, because the
        # variance is quadratic in it and the estimate spans two orders of
        # magnitude across the listing. At the median it is the incumbents' 3.3
        # ticks a lot on a future; on the win rate future at the top of that
        # spread, whose measured volatility is 0.0024, it is 34. That is the
        # model rather than a miscalibration: a contract whose price moves ten
        # times as much is one where carrying a lot costs a hundred times as
        # much, and a maker that reprices hard after a fill is a maker that
        # stays flat. Adverse selection is 117% of the incumbents' loss here,
        # so staying flat is the medicine.
        self.gamma = gamma
        # Order arrival decay, per tick. The base half-spread the model quotes
        # is `(1 / gamma) * ln(1 + gamma / k)`, which for small `gamma / k` is
        # `1 / k`, so 0.2 asks for 5.0 ticks a side. The measured median
        # half-spread of this book over 47 contracts and 600s on seed 7 is 6.0
        # ticks, p10 4.5 and p90 9.0, so this quotes just inside the median
        # rather than wider than it. CONTRIBUTING.md records both attempts at
        # widening on this venue making the loss worse.
        self.k_ticks = k_ticks
        # Twelve lots against the incumbents' 30, 22 and 14, and a 90 lot bound
        # against their 1,200, 950 and 700. Both are set by what a seat can
        # actually collateralise rather than by preference: `maker_capital`
        # over the 47 contracts listed here is 19,941,816 at 12 and 90, which
        # is just inside a 20,000,000 account, while the incumbents are funded
        # 240,474,840 for their own mandate.
        self.quote_size = quote_size
        self.inventory_bound = inventory_bound
        # How long this maker expects to carry inventory before it recycles,
        # standing in for the paper's time to liquidation. Measured by Little's
        # law on the incumbent makers, mean absolute position over lots traded
        # a second: 3.6s median across the 47 contracts on seed 7 over 300s,
        # p10 0.21s on the futures where the volume is and p90 57s in the
        # binaries where there is none. `None` means carry to settlement, which
        # drops this branch entirely and leaves the bounded terminal variance
        # `u * (1 - u)`.
        self.horizon_seconds = horizon_seconds
        # Half a tick a side, so the two quotes are at least one tick apart
        # after snapping. Below that the bid and the ask round to the same
        # price and the strategy posts a locked market against itself, which
        # the engine resolves by cancelling one of its own orders.
        self.min_half_spread_ticks = min_half_spread_ticks
        # An optional cap on width. `None`, which is the paper exactly, because
        # capping it was tried and measured worse.
        #
        # The reason to want one is real. `gamma * sigma^2 * (T - t)` is
        # unbounded in the horizon and quadratic in a volatility estimate that
        # spans two orders of magnitude across this listing: at the median
        # estimate of 0.00075 the term asks for 1.7 ticks a side, and at the
        # measured p90 of 0.0067 it asks for 129 on a book whose whole spread
        # is 11. That is the substantive difference between the two papers,
        # since Gueant, Lehalle and Fernandez-Tapia take the horizon to
        # infinity and get a finite width where this one does not.
        #
        # Capping it at nine ticks, the measured p90 half-spread of this book,
        # looked obviously right and is 2.1x worse. Over 300s on seeds 7 and 11
        # on a seat funded for the mandate, the cap gives -49,095,913 and
        # -37,322,149 against -24,300,671 and -15,937,203 uncapped, and it also
        # doubles the volume, 190,570 and 184,171 lots against 94,522 and
        # 99,330. On the 20,000,000 seat the two look identical, which is the
        # trap: both were refusing 34,914 and 22,517 orders for collateral
        # there, so the number being compared was the seat rather than the
        # model.
        self.max_half_spread_ticks = max_half_spread_ticks
        # Whether a quote is allowed through the other side of the book.
        #
        # The paper emits a price and has no notion of a touch, and once the
        # reservation price has moved further than the half-spread the quote it
        # asks for is on the wrong side of the market. With the default gamma
        # and that same measured volatility on a win rate future the
        # reservation moves 34 ticks a lot against a capped half-spread of 9,
        # so a single lot of inventory is enough to put it there. Left true,
        # this strategy takes rather than makes, which
        # `arena/strategies/base.py` names as the specific failure that
        # produced the makers here aggressive on 61% of their fills.
        #
        # CONTRIBUTING.md records clamping to the touch costing the incumbents 9.4x,
        # so this was measured rather than inherited. Over 300s on seeds 7 and
        # 11, crossing takes the passive share of this strategy's own lots from
        # 66.3% and 61.1% down to 19.1% and 19.5%, and the lots it trades from
        # 34,762 and 45,813 up to 116,357 and 90,431 while the total stays
        # within half a percent. It is buying four times the volume at a price
        # that leaves it no better off, which is a taker wearing a maker's
        # name. False by default for that reason. Each side is then held one
        # tick inside the opposite touch, independently, never by moving the
        # centre.
        #
        # Not the same thing as `Quote.post_only`, and the difference decides
        # the sign. `post_only` asks the venue to *refuse* an order that would
        # cross, and its own docstring records the cost: 31,804 orders refused
        # and fills down from 26,131 to 4,872. This reprices instead, so the
        # quote still rests, one tick inside, and the strategy keeps a market
        # in the symbol it was about to run over.
        self.cross_touch = cross_touch
        self.quote_without_reference = quote_without_reference
        self.volatility = volatility or BoundedVolatility()

    # -- the model ---------------------------------------------------------

    def solve(self, view: MarketView, symbol: str) -> Reservation | None:
        """The paper's two equations, on this contract's normalised scale."""
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

        # `sigma^2 * (T - t)` with the paper's terminal date replaced by a
        # carrying horizon, capped by the largest variance a martingale pinned
        # at this level inside these bounds can have. Neither branch needs a
        # clock, which is what makes the model runnable here at all.
        ceiling = max(0.0, level * (1.0 - level))
        variance = (
            ceiling
            if self.horizon_seconds is None
            else min(sigma * sigma * self.horizon_seconds, ceiling)
        )

        inventory = max(-self.inventory_bound, min(self.inventory_bound,
                                                   symbol_view.position))
        reservation = level - inventory * self.gamma * variance

        # Equation 25, verbatim, and the reason this class exists: there is no
        # inventory in it.
        instrument = symbol_view.instrument
        span_ticks = float(span / float(instrument.tick_size))
        k = self.k_ticks * span_ticks
        spread = self.gamma * variance + (2.0 / self.gamma) * math.log1p(
            self.gamma / k
        )
        half = spread / 2.0
        if self.max_half_spread_ticks is not None:
            half = min(half, self.max_half_spread_ticks / span_ticks)
        floor = self.min_half_spread_ticks / span_ticks
        return Reservation(
            reference=level,
            reservation=reservation,
            half_spread=max(half, floor),
            sigma=sigma,
            variance=variance,
            inventory=inventory,
        )

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

        # Each side is placed on its own. Clamping the *centre* into the range
        # was tried on this venue and is in the CONTRIBUTING.md table of things that
        # look like improvements: it makes the mid a function of the
        # half-spread, so two contracts worth the same amount price 40 points
        # apart.
        position = symbol_view.position
        rooms = {
            Side.BUY: self.inventory_bound - position,
            Side.SELL: self.inventory_bound + position,
        }
        levels = {
            Side.BUY: solved.reservation - solved.half_spread,
            Side.SELL: solved.reservation + solved.half_spread,
        }
        quotes: dict[Side, Quote | None] = {Side.BUY: None, Side.SELL: None}
        for side in (Side.BUY, Side.SELL):
            room = rooms[side]
            if room <= 0:
                continue
            price = place(
                symbol_view,
                side,
                low + levels[side] * span,
                cross_touch=self.cross_touch,
            )
            if price is None:
                continue
            quotes[side] = Quote(price=price, size=min(self.quote_size, room))
        return TwoSided(bid=quotes[Side.BUY], ask=quotes[Side.SELL])
