"""Position accounting, in exact integer money.

Average-cost semantics, which is what derivatives venues use:

    unrealized = quantity * mark - cost_basis
    realized   = crystallised when lots close, against that same basis

Two decisions carry the correctness of this module.

**The basis is stored exactly and never reconstructed from an average.** Storing
a rounded average price and rebuilding the basis from it leaks a fraction of a
unit per fill, and after enough fills the total equity in a closed market no
longer equals the capital that entered it.

**Everything is integer minor units.** See :mod:`arena.portfolio.money`. A
proportional close needs ``basis * c // q``, and in any fixed-precision decimal
that division rounds so that ``basis - closed`` no longer reconstructs ``basis``.
In integers it does, exactly. Where a sub-unit remainder lands is arbitrary;
that it stays inside the position rather than escaping into the market's total
is not.

The remaining case worth naming is the **flip**. A trade taking a position from
+10 to -5 closes 10 lots and opens 5. It does not blend a long and a short cost
basis into a number describing a position that never existed, which is a
classic way to build a backtest reporting profits nobody could have made.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from arena.portfolio.money import Money, from_money

__all__ = ["Position", "FillRecord"]


@dataclass(frozen=True, slots=True)
class FillRecord:
    """What a position learned from one execution. Amounts in minor units."""

    quantity: int
    price: Money
    realized: Money
    closed: int
    opened: int
    flipped: bool


@dataclass(slots=True)
class Position:
    """A signed position in one instrument, with exact cost-basis accounting."""

    symbol: str
    quantity: int = 0
    # Signed, exact, authoritative. Positive for a long, negative for a short.
    # Every other figure is derived from it.
    cost_basis: Money = Money(0)
    realized_pnl: Money = Money(0)
    fees_paid: Money = Money(0)
    volume: int = 0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def average_price(self) -> Decimal:
        """Derived for display and collateral. Never the source of truth."""
        if self.quantity == 0:
            return Decimal(0)
        return from_money(self.cost_basis) / Decimal(self.quantity)

    def unrealized_pnl(self, mark: Money) -> Money:
        """Mark-to-market on the open position.

        Signed throughout, so one expression serves both directions: a short has
        negative quantity and a negative basis, so a falling mark makes the
        difference positive.
        """
        if self.quantity == 0:
            return Money(0)
        return Money(self.quantity * int(mark) - int(self.cost_basis))

    def equity(self, mark: Money) -> Money:
        return Money(int(self.realized_pnl) + int(self.unrealized_pnl(mark)))

    def basis_after(self, quantity: int, price: Money) -> Money:
        """What :meth:`apply_fill` would leave the basis at, without applying it.

        Needed because a solvency check has to price the position the fill
        would *create*, and that position's exposure comes from its basis
        rather than from the price of the trade that finished it. Pricing the
        whole resulting quantity at the incoming price understates an add:
        measured on ten lots long at 5,000 followed by ten more at 100, the
        check saw ``20 * 100 = 2,000`` where the position it produced carries a
        basis of 51,000. An account with 50,000 of cash passed that check and
        came out holding a position that can lose 51,000, owing a thousand it
        does not have, which is the one thing full collateralisation is
        supposed to make impossible.

        Every branch mirrors :meth:`apply_fill` line for line, and
        ``test_the_projected_basis_matches_the_applied_one`` holds them
        together over random fill sequences rather than trusting that they were
        written to match.
        """
        if self.quantity == 0 or _same_sign(self.quantity, quantity):
            return Money(int(self.cost_basis) + quantity * int(price))

        remaining = self.quantity + quantity
        if abs(quantity) > abs(self.quantity):
            return Money(remaining * int(price))
        if remaining == 0:
            return Money(0)

        closing = min(abs(quantity), abs(self.quantity))
        closed_basis = int(self.cost_basis) * closing // abs(self.quantity)
        return Money(int(self.cost_basis) - closed_basis)

    def apply_fill(
        self, quantity: int, price: Money, fee: Money = Money(0)
    ) -> FillRecord:
        """Apply an execution. ``quantity`` is signed: positive buys.

        * **Opening or adding**: same sign, or from flat. The basis grows by
          ``quantity * price``. Exact, no division.
        * **Reducing**: opposite sign, smaller magnitude. A proportional slice
          of the basis closes and realises against the trade price. The slice
          is *subtracted* from the basis rather than the basis being
          recomputed, so the integer remainder stays in the position.
        * **Flipping**: opposite sign, larger magnitude. The whole basis
          closes, which needs no proportion and so cannot round at all, then
          the remainder opens at the trade price.
        """
        if quantity == 0:
            raise ValueError("a fill must have non-zero quantity")

        self.fees_paid = Money(int(self.fees_paid) + int(fee))
        self.realized_pnl = Money(int(self.realized_pnl) - int(fee))
        self.volume += abs(quantity)

        if self.quantity == 0 or _same_sign(self.quantity, quantity):
            self.cost_basis = Money(int(self.cost_basis) + quantity * int(price))
            self.quantity += quantity
            return FillRecord(quantity, price, Money(0), 0, abs(quantity), False)

        closing = min(abs(quantity), abs(self.quantity))
        flipping = abs(quantity) > abs(self.quantity)

        if flipping:
            # The entire basis closes: no proportion, no remainder, no rounding.
            closed_basis = int(self.cost_basis)
        else:
            closed_basis = int(self.cost_basis) * closing // abs(self.quantity)

        direction = 1 if self.quantity > 0 else -1
        realized = direction * closing * int(price) - closed_basis
        self.realized_pnl = Money(int(self.realized_pnl) + realized)

        # Subtracted, not recomputed. This is what keeps the remainder inside
        # the position instead of leaking it into the market's total value.
        self.cost_basis = Money(int(self.cost_basis) - closed_basis)
        remaining = self.quantity + quantity

        if flipping:
            opened = abs(remaining)
            self.cost_basis = Money(remaining * int(price))
        else:
            opened = 0
            if remaining == 0:
                self.cost_basis = Money(0)

        self.quantity = remaining
        return FillRecord(
            quantity=quantity,
            price=price,
            realized=Money(realized),
            closed=closing,
            opened=opened,
            flipped=flipping,
        )

    def to_dict(self, mark: Money | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "cost_basis": str(from_money(self.cost_basis)),
            "average_price": str(self.average_price),
            "realized_pnl": str(from_money(self.realized_pnl)),
            "fees_paid": str(from_money(self.fees_paid)),
            "volume": self.volume,
        }
        if mark is not None:
            payload["mark"] = str(from_money(mark))
            payload["unrealized_pnl"] = str(from_money(self.unrealized_pnl(mark)))
            payload["equity"] = str(from_money(self.equity(mark)))
        return payload


def _same_sign(a: int, b: int) -> bool:
    return (a > 0) == (b > 0)
