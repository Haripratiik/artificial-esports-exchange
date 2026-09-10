"""Money as integers.

The same discipline the matching engine applies to prices, applied to cash. And
for the same reason: exactness.

Decimal is not enough here, and the failure is instructive. Average-cost
accounting has to split a position's basis proportionally when part of it
closes, and a split like ``33875 * 2 / 7`` needs 29 significant digits. Once the
result is rounded, ``basis - closed_basis`` no longer reconstructs the original,
and roughly 1e-24 of value evaporates per fill. Invisible in any single figure,
and fatal to the one check that makes a PnL statement believable: that total
equity in a closed market equals the capital that entered it. Raising the
precision does not fix it, it only moves the leak further down.

Integers have no such problem. A proportional split is ``basis * c // q`` with
the remainder left in the position, and subtraction reconstructs the total
exactly, because integer arithmetic is exact by construction. Where the split
lands at the sub-unit level is arbitrary; that it conserves is not.

So all money is counted in **minor units**, a fixed fraction of one price unit.
Conversion to a human-readable Decimal happens at the reporting boundary and
nowhere else, exactly as tick-to-price conversion happens at the exchange
boundary and nowhere else.
"""

from __future__ import annotations

from decimal import Decimal
from typing import NewType

__all__ = ["Money", "MONEY_SCALE", "to_money", "from_money", "tick_in_minor"]

# One price unit is this many minor units. 1e-6 resolution: fine enough that
# rounding a real quantity to it is meaningless, coarse enough that a century of
# trading stays far inside a 64-bit integer.
MONEY_SCALE = 1_000_000

Money = NewType("Money", int)


def to_money(amount: Decimal | int | str) -> Money:
    """Convert a price-unit amount to minor units.

    Refuses amounts finer than the minor unit rather than rounding them. A
    silently rounded input is how an exact ledger stops being exact.
    """
    value = amount if isinstance(amount, Decimal) else Decimal(str(amount))
    scaled = value * MONEY_SCALE
    if scaled != scaled.to_integral_value():
        raise ValueError(
            f"{amount} is finer than the minor unit (1/{MONEY_SCALE}); rounding it "
            "here would break exact conservation"
        )
    return Money(int(scaled))


def from_money(amount: Money | int) -> Decimal:
    """Convert minor units back to a price-unit Decimal, for reporting."""
    return Decimal(int(amount)) / MONEY_SCALE


def tick_in_minor(tick_size: Decimal) -> int:
    """How many minor units one tick is worth.

    Required to be a whole number of minor units. A tick finer than the money
    resolution would make a single fill unrepresentable, which is a
    configuration error rather than something to round away.
    """
    scaled = tick_size * MONEY_SCALE
    if scaled != scaled.to_integral_value() or scaled <= 0:
        raise ValueError(
            f"tick size {tick_size} is not a positive whole number of minor units "
            f"(1/{MONEY_SCALE})"
        )
    return int(scaled)
