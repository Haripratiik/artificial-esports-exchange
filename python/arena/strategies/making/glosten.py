"""Quotes conditioned on the order that is about to arrive.

Glosten and Milgrom (1985), "Bid, ask and transaction prices in a specialist
market with heterogeneously informed traders", JFE 14(1):71-100. A single value
``V`` is drawn. Traders arrive one at a time; with probability ``mu`` the
arriving trader knows ``V`` and with probability ``1 - mu`` it trades for
reasons of its own. A competitive risk-neutral maker breaks even on every trade
rather than on average, so it quotes

    ask = E[V | history, the next order is a BUY]
    bid = E[V | history, the next order is a SELL]

and that is the whole model. The spread is strictly positive whenever
``mu > 0`` and exactly zero when ``mu = 0``, and none of it is inventory cost or
order processing cost, because the model has neither. Every tick of it is
payment for the trades that will be against somebody who knew more.

Two consequences separate this from every other maker in this repository, and
they are the reason it is worth having.

**Being filled is news, so the quote has to move.** The ask is defined as the
expectation conditional on a buy, so the moment a buy arrives that expectation
becomes the unconditional one and the next ask is computed from a belief that
has already moved up. A maker whose ask does not rise after it is lifted is
offering the same information at the same price twice, which is a thing that is
measurable here: 17% of the incumbent makers' passive fills are a second fill at
the same price inside 500ms, because they requote on a timer. The adapter asks
this strategy again the instant it trades, and this strategy answers with a
different number.

**An order is worth what it tells you, not one vote.** The incumbents run a
plain exponentially-weighted average of prints, which moves the same distance
for a 30-lot lift as for a 2-lot print. Here the likelihood of an order of ``q``
lots is the one-lot likelihood raised to ``q`` over the size of an ordinary
order, which the strategy measures from its own fills as it goes. Easley and
O'Hara (1987) is the standard treatment of why size belongs in the update:
somebody who knows something trades as much as they can get away with, so a
large order is more likely to be one of theirs.

**The belief lives on the settlement range.** Every contract on this venue
settles as a known function of one bounded scalar and the bounds are written
into the contract, so the posterior is a distribution over that interval rather
than over the whole line. The model needs a prior with support and this market
hands it one, which is a fit good enough to be worth saying out loud: the
"bounded support" that Glosten-Milgrom needs for its conditional expectations to
exist is not an approximation here, it is the collateral rule.

The belief is assembled at each requote from two pieces:

    posterior  proportional to  public(v) * exp(tilt(v))

``public`` is what the rest of the market thinks, a bell centred on the mid with
the width the mid has actually been wandering by, both measured. ``tilt`` is
what this strategy alone knows, the accumulated log-likelihood of the orders it
has personally seen arrive, decayed at the rate the public price absorbs the
same information. Neither piece is optional. Without the public piece the belief
on a contract this strategy has not traded stays wherever the prior left it and
the quote is nowhere near the market: on a uniform prior over a future's range
with ``mu = 0.5`` the model quotes 3,659 at 6,341 on a contract worth 4,670,
which is not a market, it is a refusal to make one. Without the tilt there is no
private information to be paid for and the ask does not move when it is lifted,
which is the property this class exists to have.

The decay on the tilt is the one thing here that is a time constant rather than
a measurement, and it has to be. An order is informative until the price has
absorbed it, and after that, tilting against a mid that has already moved counts
the same news twice. The adapter's markout horizon is one second because this
market's adverse selection "is positive through the first half second and only
turns over between one and five"; five seconds is the far end of that, which is
where an order has stopped being news.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np

from arena.exchange.types import Side
from arena.strategies.base import MarketView, Quote, SymbolView, TwoSided, snap

# The two guards every maker on this venue needs and neither of which is about
# a model. They belong beside `snap` in `arena.strategies.base`; they live in
# `fixed.py` because that is the module this change was allowed to create.
from arena.strategies.making.fixed import priced_touch

__all__ = ["GlostenMilgrom"]

# How fast the belief's centre and its width follow the mid. The pair is the
# one `SurfaceMarketMaker` already uses on prints, for the reason given there:
# a mean can be tracked from a dozen observations and a variance cannot, so the
# width has to forget an order of magnitude more slowly than the level or every
# quote in the book moves at once.
ANCHOR_GAIN = 0.15
WIDTH_GAIN = 0.02


@dataclass
class _Belief:
    """One contract's posterior over where it settles, and what fed it."""

    grid: np.ndarray
    # Log-likelihood of every order this strategy has personally seen arrive,
    # decayed. Only differences across the grid matter, so it is renormalised
    # by its own maximum and carries no absolute meaning.
    tilt: np.ndarray
    at: float
    mid: float | None = None
    anchor: float | None = None
    width: float = 0.0
    spread: float | None = None
    position: int = 0
    lots: float | None = None
    # The informed share, derived from the two lines below rather than stored
    # in its own right, so it can never disagree with what fed it.
    mu: float = 0.5
    # Exponentially weighted probability that an arriving order has the same
    # direction as the one before it, and the direction of that one.
    same: float | None = None
    buying: bool | None = None
    # Posterior mass this strategy's own quoted interval does not cover, which
    # is how much of the flow the count above can say anything about.
    outside: float = 1.0
    # Exponentially weighted markout, as a fraction of the half-spread it ate.
    slip: float = 0.0
    half: float | None = None
    markout: dict[Side, float] = field(default_factory=dict)


class GlostenMilgrom:
    """A competitive maker that prices the order it is about to be given."""

    def __init__(
        self,
        nodes: int = 1025,
        mu_prior: float = 0.5,
        mu_gain: float = 0.05,
        mu_cap: float = 0.99,
        tilt_halflife: float = 5.0,
        size_gain: float = 0.10,
        weight_cap: float = 6.0,
        position_limit: int = 250,
        quote_size: int = 10,
    ) -> None:
        # How finely the settlement range is cut. The grid has to resolve the
        # belief's own width or the posterior is one cell wide and the quote is
        # quantised to a step of the range. Measured on seed 7 over 180
        # seconds, the dispersion of the mid around a 0.15-gain average of
        # itself runs from 0.0012 of the range on one weekly leg of a share to
        # 0.52 on a deep out of the money put, median 0.081. At 1,025 points
        # the narrowest of those is 1.3 steps and the median is 83, which is
        # why the grid is this fine and why the width below is floored at one
        # step.
        self.nodes = int(nodes)
        # Where the informed share starts before this strategy has any
        # experience of its own. A coin flip, which is the least committal
        # thing to say about a counterparty you have never traded with.
        self.mu_prior = mu_prior
        # How fast both halves of the estimate forget. See :meth:`_recompute_mu`
        # for what they are and :meth:`_update_mu` for what the markout adds.
        self.mu_gain = mu_gain
        # The smallest posterior mass outside this strategy's own quote that
        # the flow count is allowed to be divided by. Orders arriving on a
        # value the quote already straddles are noise whatever the informed
        # share is, so they dilute the count, and the correction for that is a
        # division: it has to stop somewhere or an estimate becomes a division
        # by zero. Measured on seed 7 over 180 seconds, 34,055 quotes, the mass
        # outside was below 0.05 on 6.7% of them against a median of 0.47, so
        # the floor caps the amplification at twenty and binds on the tail.
        self.mass_floor = 0.05
        # Glosten-Milgrom's own conclusion is that at a high enough informed
        # share no spread is profitable and the market shuts. That is a real
        # prediction and the cap is not here to prevent it, it is here because
        # at exactly one the likelihood of an order against the values it rules
        # out is zero, and a posterior with no mass anywhere is not a
        # distribution. It has to sit above the answer, and the answer is high:
        # measured over 180 seconds on seeds 7 and 3, the share of this
        # strategy's own passive flow that came from a fundamental trader
        # rather than from another maker was 0.974 on both.
        self.mu_cap = mu_cap
        # Seconds for the private tilt to lose half its weight. See the module
        # docstring: an order stops being news once the public price has
        # absorbed it, and on this market that has happened by five seconds.
        self.tilt_halflife = tilt_halflife
        self.size_gain = size_gain
        # An order this many times an ordinary one is not a large trade, it is
        # a broken input. Measured on seed 7 over 120 seconds, the incumbent
        # makers' fills ran from 1 to 72 lots against a median of 13, so a
        # ratio of 6 is past anything this market produces and the cap never
        # binds on real flow.
        self.weight_cap = weight_cap
        # Glosten-Milgrom has no inventory term at all, so this is a risk
        # control on the account and never a price. Past the limit the strategy
        # shows one side only; what it shows is still exactly E[V | order], not
        # a shaded version of it. Sized like the baseline's, against the 20M a
        # strategy account opens with and 47 books to be two-sided in.
        self.position_limit = position_limit
        self.quote_size = quote_size

        self._beliefs: dict[str, _Belief] = {}
        self._cash: Decimal | None = None

    # -- what a caller can read --------------------------------------------

    def estimated_mu(self, symbol: str | None = None) -> float:
        """The informed share this strategy currently believes it faces.

        Pooled over the contracts that have given it something to count when
        asked without one, and a book with no flow of its own is left out
        rather than reported at the prior: an average that counts 47 books when
        five of them have traded is mostly a restatement of the prior. A symbol
        answers with its own, which is the more useful number, since the
        informed share is not the same thing on a binary that has already
        resolved and on a future that is still moving.
        """
        if symbol is not None:
            belief = self._beliefs.get(symbol)
            return self.estimated_mu() if belief is None else belief.mu
        counted = [b.mu for b in self._beliefs.values() if b.same is not None]
        if not counted:
            return self.mu_prior
        # Clamped, because the mean of values that are each at or below the cap
        # can still land above it: beliefs all pinned at 0.99 average to
        # 0.9900000000000001 in binary floating point. An ulp of representation
        # error is not an opinion this strategy holds, and a caller reading the
        # bound the class advertises is entitled to have it hold.
        return min(self.mu_cap, sum(counted) / len(counted))

    def conditional_prices(self, view: MarketView, symbol: str) -> tuple[float, float]:
        """``(E[V | next order is a SELL], E[V | next order is a BUY])``.

        The model itself, before the tick grid and before any risk control.
        This is the pair that collapses to a single number when ``mu`` is zero
        and widens toward the whole settlement range as ``mu`` approaches one.

        It moves no belief forward in time and observes nothing, so a caller
        may ask as often as it likes. It does record how much posterior mass
        its own answer straddles, because the informed-share count needs it and
        this is the only place both numbers exist at once.
        """
        v = view[symbol]
        belief = self._beliefs.get(symbol)
        if belief is None:
            belief = self._open(v, view.now)
        posterior = self._posterior(belief)
        ask = _conditional_mean(belief.grid, posterior, belief.mu)
        bid = -_conditional_mean(-belief.grid[::-1], posterior[::-1], belief.mu)
        straddled = (belief.grid >= bid) & (belief.grid <= ask)
        belief.outside = max(0.0, 1.0 - float(posterior[straddled].sum()))
        return bid, ask

    # -- the strategy ------------------------------------------------------

    def symbols(self, view: MarketView) -> Sequence[str]:
        """Everything the agent lists, each with its own belief and its own mu.

        Nothing here is shared across contracts except the starting point for a
        book that has not traded yet, because the settlement value of one
        contract is not evidence about another and pretending otherwise would
        be the correlation this venue's collateral engine refuses to estimate.
        """
        return list(view.symbols)

    def quote(self, view: MarketView, symbol: str) -> TwoSided:
        v = view[symbol]
        instrument = v.instrument
        tick = instrument.tick_size
        low, high = v.bounds
        bid_touch, ask_touch = priced_touch(v)

        self._orders(view, symbol)
        belief = self._beliefs[symbol]
        self._decay(belief, view.now)
        self._watch(belief, v, bid_touch, ask_touch)

        bid_price, ask_price = self.conditional_prices(view, symbol)
        belief.half = max((ask_price - bid_price) / 2.0, float(tick))
        self._update_mu(belief, v)

        # A resting quote that crosses the book is a taker paying the spread it
        # is here to earn, and the model has nothing to say about that: the
        # competitive price is what it is whether or not somebody is already
        # showing it. One tick inside the visible touch is the most aggressive
        # price that is still passive.
        if ask_touch is not None:
            bid_price = min(bid_price, float(ask_touch - tick))
        if bid_touch is not None:
            ask_price = max(ask_price, float(bid_touch + tick))

        bid = snap(instrument, Side.BUY, min(max(bid_price, float(low)), float(high)))
        ask = snap(instrument, Side.SELL, min(max(ask_price, float(low)), float(high)))
        if ask <= bid:
            # ``mu`` near zero is a one-price market, which the model is
            # entitled to quote and a book is not. A tick apart is the tightest
            # thing that is expressible, and giving up a side beats crossing.
            if bid - tick >= low:
                bid = bid - tick
            elif ask + tick <= high:
                ask = ask + tick
            else:
                return TwoSided()

        bid_size = min(self.quote_size, self.position_limit - v.position)
        ask_size = min(self.quote_size, self.position_limit + v.position)
        return TwoSided(
            bid=Quote(bid, bid_size) if bid_size > 0 else None,
            ask=Quote(ask, ask_size) if ask_size > 0 else None,
        )

    # -- the belief --------------------------------------------------------

    def _open(self, v: SymbolView, now: float) -> _Belief:
        """A flat belief over the settlement range, which assumes nothing.

        Uniform is the maximum-entropy distribution on a bounded support, and
        it is the honest state before any evidence. It does not stay uniform
        for long: the first mid this strategy sees narrows it to whatever the
        rest of the market is showing.
        """
        low, high = v.bounds
        grid = np.linspace(float(low), float(high), self.nodes)
        belief = _Belief(
            grid=grid,
            tilt=np.zeros(self.nodes),
            at=now,
            position=v.position,
            mu=self.estimated_mu(),
        )
        self._beliefs[v.symbol] = belief
        return belief

    def _decay(self, belief: _Belief, now: float) -> None:
        """Forget private information at the rate the public price absorbs it."""
        elapsed = max(0.0, now - belief.at)
        belief.at = now
        if elapsed and self.tilt_halflife > 0:
            belief.tilt *= 0.5 ** (elapsed / self.tilt_halflife)

    def _watch(
        self,
        belief: _Belief,
        v: SymbolView,
        bid: Decimal | None,
        ask: Decimal | None,
    ) -> None:
        """Track where the rest of the market is and how far it wanders.

        Only when the mid has actually moved. Resampling an unchanged quote on
        a timer would be counting one opinion once per wakeup, and the width is
        a variance: fed the same number 1,875 times it goes to zero and the
        belief becomes a spike on a price nobody has confirmed.

        The mid is ignored when this strategy is both sides of it, because then
        it is this strategy's own quote coming back, and a belief that conditions
        on its own output has stopped conditioning on anything.
        """
        own = v.working_bid == bid and v.working_ask == ask
        if bid is not None and ask is not None and not own:
            # The width the rest of the market is showing, kept for the floor
            # under the belief's own. Not recorded when the touch is this
            # strategy's own on both sides: the floor would then be half of
            # whatever this strategy last quoted, so a narrow quote would
            # justify a narrow belief which would justify a narrow quote.
            belief.spread = float(ask - bid)
            mid = float(bid + ask) / 2.0
        elif v.last is not None:
            mid = float(v.last)
        elif belief.mid is not None:
            return
        else:
            low, high = v.bounds
            mid = float(low + high) / 2.0

        if belief.mid is not None and mid == belief.mid:
            return
        belief.mid = mid
        if belief.anchor is None:
            belief.anchor = mid
            return
        deviation = mid - belief.anchor
        belief.width += WIDTH_GAIN * (deviation * deviation - belief.width)
        belief.anchor += ANCHOR_GAIN * deviation

    def _sigma(self, belief: _Belief) -> float:
        """How wide the belief is, in price units.

        The dispersion the mid has actually shown, floored twice. Once at a
        grid step, because a belief cannot be sharper than the grid it is
        written on. Once at the market's own quoted half-spread, because the
        market has already said in public that it does not know the price
        closer than that, and a maker claiming to know it better on no evidence
        is a maker that will quote inside everyone and find out why.
        """
        floor = float(belief.grid[1] - belief.grid[0])
        if belief.spread is not None and belief.spread > 0.0:
            floor = max(floor, belief.spread / 2.0)
        return max(belief.width**0.5, floor)

    def _posterior(self, belief: _Belief) -> np.ndarray:
        """What the market says, tilted by what this strategy alone has seen."""
        if belief.mid is None:
            weights = np.exp(belief.tilt - belief.tilt.max())
            return weights / weights.sum()
        sigma = self._sigma(belief)
        z = (belief.grid - belief.mid) / sigma
        logp = belief.tilt - 0.5 * z * z
        weights = np.exp(logp - logp.max())
        return weights / weights.sum()

    # -- what its own fills said -------------------------------------------

    def _orders(self, view: MarketView, symbol: str) -> None:
        """Turn this strategy's own position changes into arriving orders.

        The interface hands a maker a view and nothing else, so a fill is
        observed as the difference between the position now and the position at
        the last look. That is enough, because the direction is the whole of
        what Glosten-Milgrom conditions on: a position that fell means somebody
        bought from this strategy, which is a BUY arriving, and the belief has
        to move up.

        The price is recovered from the cash the fill moved, which is exact
        where only one book traded between two looks, and taken from the
        working quote on the side that traded otherwise. It matters that it is
        the traded price rather than the current quote: the informed trader
        bought at the price it paid, so that price is the threshold the
        indicator in the likelihood is written against.
        """
        known = self._beliefs.get(symbol)
        if (
            self._cash is not None
            and view.cash == self._cash
            and known is not None
            and known.position == view[symbol].position
        ):
            return

        opening = self._cash is None
        moved = float(view.cash - self._cash) if not opening else 0.0
        self._cash = view.cash

        changed: list[tuple[SymbolView, _Belief, int]] = []
        for v in view:
            belief = self._beliefs.get(v.symbol)
            if belief is None:
                belief = self._open(v, view.now)
            signed = v.position - belief.position
            belief.position = v.position
            if signed and not opening:
                changed.append((v, belief, signed))

        for v, belief, signed in changed:
            price = self._fill_price(
                v, belief, signed, moved if len(changed) == 1 else None
            )
            two_sided = v.working_bid is not None and v.working_ask is not None
            self._arrival(belief, signed < 0, abs(signed), price, two_sided)

    def _fill_price(
        self, v: SymbolView, belief: _Belief, signed: int, moved: float | None
    ) -> float:
        """What the counterparty paid, from the only two places it is visible."""
        if moved is not None and signed:
            price = -moved / signed
            low, high = v.bounds
            if float(low) <= price <= float(high):
                return price
        working = v.working_ask if signed < 0 else v.working_bid
        if working is not None:
            return float(working)
        if belief.mid is not None:
            return belief.mid
        low, high = v.bounds
        return float(low + high) / 2.0

    def _arrival(
        self, belief: _Belief, buy: bool, lots: int, price: float, two_sided: bool
    ) -> None:
        """One order, weighted by how much of an order it was.

        The likelihood of a single unit order is the model's own: a buyer is
        informed with probability ``mu`` and then only if the value is above
        what it paid, and is trading for its own reasons with probability
        ``1 - mu`` and then buys half the time. An order of ``q`` lots is
        treated as ``q`` of those, in units of an ordinary order, which the
        strategy is measuring as it goes rather than being told.
        """
        belief.lots = (
            float(lots)
            if belief.lots is None
            else belief.lots + self.size_gain * (lots - belief.lots)
        )
        weight = min(self.weight_cap, lots / max(belief.lots, 1e-9))
        noise = (1.0 - belief.mu) / 2.0
        informed = belief.grid > price if buy else belief.grid < price
        likelihood = np.where(informed, belief.mu + noise, noise)
        belief.tilt += weight * np.log(likelihood)
        belief.tilt -= belief.tilt.max()

        # The same order, counted for what it says about how many of them are
        # informed. See :meth:`_recompute_mu`.
        #
        # Counted only across a stretch where this strategy was showing both
        # sides, and that condition is most of what makes the count an estimate
        # rather than a mirror. A maker at its position limit shows one side,
        # so every order it then sees arrives in the same direction and a run
        # of them says nothing about who sent them. Measured without the
        # condition on seed 7 over 180 seconds, consecutive orders agreed
        # 96.5%, 96.7%, 96.5% and 96.7% of the time on the four weekly legs of
        # one share, which reads as an informed share of one on the four
        # quietest books in the market.
        if not two_sided:
            belief.buying = None
            return
        if belief.buying is not None:
            repeat = 1.0 if buy is belief.buying else 0.0
            belief.same = (
                repeat
                if belief.same is None
                else belief.same + self.mu_gain * (repeat - belief.same)
            )
        belief.buying = buy
        self._recompute_mu(belief)

    def _recompute_mu(self, belief: _Belief) -> None:
        """The informed share, counted off the flow and audited by the markout.

        Under the model, orders are independent given ``V`` and correlated only
        because they share it. An informed trader buys wherever the value is
        above the ask and sells wherever it is below the bid; everybody else
        buys half the time. So for a value the quote does not straddle, two
        consecutive orders agree with probability

            P(same) = ((1 + mu) / 2)^2 + ((1 - mu) / 2)^2 = (1 + mu^2) / 2

        and for a value it does straddle, the informed trader does not trade at
        all and consecutive orders agree exactly half the time. Writing ``w``
        for the posterior mass outside this strategy's own quote,

            P(same) = 1/2 + w * mu^2 / 2      so      mu = sqrt((2P - 1) / w)

        which is an estimate of the informed share out of nothing but the
        directions of the orders that arrived. It needs no scale, no
        half-spread and no units, which is what makes it usable on a future and
        a binary at the same time, and ``w`` comes from the strategy's own
        posterior rather than from an assumption.

        The markout is the audit on that count, and a desk uses it the same
        way: the count says how many of the arriving orders knew something, and
        the markout says whether the spread that count implies actually covered
        what they took. It should be zero on a competitive quote, because the
        ask *is* the expectation given a buy and there is nothing left to drift
        toward. When it is not zero the count has missed something, and it does
        not matter to the quote whether what it missed was size, speed or this
        strategy's own 160ms-stale view of the book. It is added rather than
        multiplied so that it can move the estimate in either direction from
        wherever the count left it, including all the way to nothing. Measured
        on seed 7 over 180 seconds, one binary struck on a competitor's win
        rate had a flow count of 0.54 and a correction of -0.61, so the
        estimate there is zero: the audit overruled the count on a book where
        consecutive orders did lean one way but every fill was still making
        money by the time the mid caught up.
        """
        if belief.same is None:
            flow = self.mu_prior
        else:
            mass = max(belief.outside, self.mass_floor)
            flow = min(1.0, max(0.0, (2.0 * belief.same - 1.0) / mass)) ** 0.5
        belief.mu = min(self.mu_cap, max(0.0, flow + belief.slip))

    def _update_mu(self, belief: _Belief, v: SymbolView) -> None:
        """Fold one matured markout into the correction on the flow count.

        Applied once per markout that has changed, not once per requote. The
        adapter's markout is itself an exponentially-weighted average that
        persists between fills, so acting on it every time would be reading one
        fill 1,875 times. Measured that way on seed 7 over 60 seconds, with the
        estimate a plain running sum of those steps, every one of its ten
        highest values was pinned at the 0.990 cap and its four lowest at
        0.000: a random walk into the bounds rather than an estimate.

        The step is a fraction of this strategy's own half-spread, so it reads
        as one: a markout of a full half-spread means the entire quoted width
        went to the counterparty, and the correction moves toward one. Clipped
        there because a single fill is a single observation however bad it was.
        """
        scale = belief.half or float(v.instrument.tick_size)
        for side in (Side.BUY, Side.SELL):
            markout = v.markout.get(side)
            if markout is None or markout == belief.markout.get(side):
                continue
            belief.markout[side] = markout
            observed = max(-1.0, min(1.0, -markout / scale))
            belief.slip += self.mu_gain * (observed - belief.slip)
        self._recompute_mu(belief)


def _conditional_mean(values: np.ndarray, weights: np.ndarray, mu: float) -> float:
    """``E[V | the next order is a BUY]``, which is Glosten-Milgrom's ask.

    A buy arrives from an informed trader, who buys only where the value is
    above the price it would pay, or from anybody else, who buys half the time
    whatever the value is. So the order tilts the belief by

        P(BUY | V = v) = mu * 1[v > a] + (1 - mu) / 2

    and the ask ``a`` is the mean of the belief under that tilt, which makes it
    a fixed point rather than an average: the threshold in the indicator is the
    answer being computed. Solved by evaluating the tilted mean at every
    threshold the grid offers, which is one pass of cumulative sums, and taking
    the first threshold the answer falls below. That threshold is the crossing,
    because the tilted mean rises with it and the grid rises faster.

    At ``mu = 0`` the tilt is flat and this returns the unconditional mean, so
    the bid and the ask are the same number and the spread is zero. That is the
    model's central claim rather than an edge case, and it is why this is
    written as one function of ``mu`` instead of as a spread around a centre.
    """
    mean = float(values @ weights)
    if mu <= 0.0:
        return mean
    noise = (1.0 - mu) / 2.0
    tail_mass = np.cumsum(weights[::-1])[::-1]
    tail_value = np.cumsum((values * weights)[::-1])[::-1]
    candidate = (mu * tail_value + noise * mean) / (mu * tail_mass + noise)
    below = candidate < values
    index = int(np.argmax(below)) if below.any() else values.size - 1
    return float(candidate[index])
