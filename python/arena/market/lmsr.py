"""Hanson's logarithmic market scoring rule, as a venue mechanism.

Experiment 1 found the limit order book aggregating no better than an unweighted
average of its agents' beliefs. Two explanations survived the ablations: the
*mechanism* limits it (a book needs a counterparty for every trade, so a
well-informed agent moves price only as fast as someone takes the other side),
or the *population* limits it (every agent has identical size, limits and cash,
so nothing lets the better-informed one push harder).

This module exists to separate them. Same agents, same information, same seeds,
different mechanism. And this mechanism has no counterparty problem at all,
because the market maker is a function rather than a participant.

The rule
--------

For a binary outcome with ``q`` shares held net-long by traders, and liquidity
parameter ``b``:

    C(q) = payout * b * ln(1 + exp(q / b))          the cost function
    p(q) = payout / (1 + exp(-q / b))               the marginal price

Buying ``d`` shares costs ``C(q + d) - C(q)``. Three properties matter here:

* **It always quotes.** There is a price for any size at any time, which is the
  whole reason real prediction markets use this rather than a book.
* **Loss is bounded, exactly.** The maker's worst case is ``payout * b * ln 2``,
  and the proof is two lines: ``C(q) - q_i = b ln(sum_j exp((q_j - q_i)/b)) >= 0``
  because the ``j = i`` term alone contributes 1, so the maker's profit is never
  worse than ``-C(0) = -b ln 2``. That is an arithmetic bound, not a
  value-at-risk estimate, which is the same standard the rest of this codebase
  holds collateral to.
* **It is path independent.** Cost depends only on where ``q`` started and
  ended, so an agent cannot manufacture profit by splitting an order.

Liquidity is parameterised by the **subsidy**, not by ``b``. The subsidy is what
the venue is willing to lose to make the market, which is the quantity anyone
actually decides; ``b`` is then whatever satisfies the bound. Picking ``b``
directly would be picking a number for a knob with no units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["LmsrMarket", "liquidity_for_subsidy", "subsidy_for_depth"]

LN2 = math.log(2.0)


def _softplus(x: float) -> float:
    """``ln(1 + exp(x))`` without overflowing on either tail."""
    if x > 0.0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def liquidity_for_subsidy(subsidy: float, payout: float) -> float:
    """The ``b`` whose worst-case loss is exactly ``subsidy``.

    Inverts ``subsidy = payout * b * ln 2``. Stated this way round because the
    subsidy is the decision (how much the venue will spend to make a market)
    and ``b`` is only its consequence.
    """
    if subsidy <= 0.0:
        raise ValueError("a scoring-rule market needs a positive subsidy")
    if payout <= 0.0:
        raise ValueError("payout must be positive")
    return subsidy / (payout * LN2)


def subsidy_for_depth(shares_per_tick: float, tick_size: float, payout: float) -> float:
    """The subsidy that puts ``shares_per_tick`` at the touch of a fresh market.

    This is the calibration that makes a comparison against an order book mean
    anything. Depth is what decides how far a given amount of informed trading
    moves the price, so two venues quoting different depth are not being
    compared on mechanism at all; the deeper one will simply look less
    responsive, whatever its rule.

    From ``p = sigmoid(q/b)``, ``dq/dp = b / (p (1 - p))`` in payout units, which
    at even odds is ``4b``. One tick therefore absorbs ``4 b tick / payout``
    shares, and inverting that gives ``b``.
    """
    if shares_per_tick <= 0.0 or tick_size <= 0.0 or payout <= 0.0:
        raise ValueError("depth, tick size and payout must all be positive")
    liquidity = shares_per_tick * payout * 0.25 / tick_size
    return payout * liquidity * LN2


@dataclass(slots=True)
class LmsrMarket:
    """The scoring rule for one binary contract. Pure arithmetic, no state but ``net``."""

    liquidity: float
    payout: float = 1.0
    net: int = 0

    def __post_init__(self) -> None:
        if self.liquidity <= 0.0:
            raise ValueError("liquidity must be positive")
        if self.payout <= 0.0:
            raise ValueError("payout must be positive")

    # -- pricing -----------------------------------------------------------

    def price_at(self, net: int | float) -> float:
        """Marginal price with ``net`` shares outstanding, in price units."""
        return self.payout * _sigmoid(net / self.liquidity)

    @property
    def price(self) -> float:
        return self.price_at(self.net)

    def _cost_function(self, net: float) -> float:
        return self.payout * self.liquidity * _softplus(net / self.liquidity)

    def cost(self, quantity: int) -> float:
        """Cost to move the book by ``quantity`` shares. Signed.

        Positive to buy, and negative when selling returns money. Computed as a
        difference of the cost function rather than by integrating the price, so
        it is exact and path independent by construction.
        """
        return self._cost_function(self.net + quantity) - self._cost_function(self.net)

    def average_price(self, quantity: int) -> float:
        """Cost per share for ``quantity`` shares: what the trader really pays."""
        if quantity == 0:
            raise ValueError("a trade must have non-zero quantity")
        return self.cost(quantity) / quantity

    def shares_for_price(self, price: float) -> float:
        """The ``net`` at which the marginal price would be ``price``.

        The inverse of :meth:`price_at`, used to render the curve as a ladder of
        price levels: how much has to be bought to push the price to a given
        tick is exactly the size sitting at that level.
        """
        fraction = min(1.0 - 1e-12, max(1e-12, price / self.payout))
        return self.liquidity * math.log(fraction / (1.0 - fraction))

    # -- risk --------------------------------------------------------------

    @property
    def bounded_loss(self) -> float:
        """Worst case for the maker, over every path and every outcome."""
        return self.payout * self.liquidity * LN2

    def apply(self, quantity: int) -> float:
        """Execute ``quantity`` shares and return the signed cost."""
        if quantity == 0:
            raise ValueError("a trade must have non-zero quantity")
        charged = self.cost(quantity)
        self.net += quantity
        return charged
