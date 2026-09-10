"""An agent's account: cash, positions, collateral, and settlement.

The collateral model is **full collateralisation**, in the sense event-contract
venues use it: the venue holds the entire contingent payout, nobody is naked
short, and an account can lose everything it committed but can never owe more
than it has. Margin and leverage arrive later as a deliberate experiment rather
than as the default.

What makes it exact here, and unusual, is that **every instrument settles inside
a known interval**. A win-rate future scaled by ten thousand settles somewhere in
[0, 10000], so a short at 5100 loses at most 4900 per lot. That worst case is
arithmetic, not a value-at-risk estimate; an ordinary future on an unbounded
price cannot say the same and needs a volatility model to guess at it. So
solvency can be enforced exactly:

    an order is admissible if the worst case of the position it would create is
    still covered by free cash

All amounts are integer minor units (see :mod:`arena.portfolio.money`), so the
ledger conserves value exactly rather than nearly.

The conservatism worth knowing about: an agent holding a long and a short in
economically related contracts posts collateral on both, because nothing here
yet understands that they hedge. Portfolio netting is a later phase, and is one
of the things the margin experiments are *about*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from arena.portfolio.money import Money, from_money
from arena.portfolio.position import Position

__all__ = ["Account", "InsufficientCollateral"]


class InsufficientCollateral(Exception):
    """The account cannot cover the worst case of a proposed position."""


@dataclass(slots=True)
class Account:
    """One agent's book. Cash, positions, and collateral held against them."""

    agent_id: str
    starting_cash: Money
    cash: Money = Money(0)
    positions: dict[str, Position] = field(default_factory=dict)
    # Collateral posted per symbol: held, not spendable, released on close.
    collateral: dict[str, Money] = field(default_factory=dict)
    settled: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if int(self.cash) == 0:
            self.cash = Money(int(self.starting_cash))

    # -- views -------------------------------------------------------------

    def position(self, symbol: str) -> Position:
        position = self.positions.get(symbol)
        if position is None:
            position = Position(symbol=symbol)
            self.positions[symbol] = position
        return position

    @property
    def posted_collateral(self) -> Money:
        return Money(sum(int(v) for v in self.collateral.values()))

    @property
    def free_cash(self) -> Money:
        """Cash not currently backing a position."""
        return Money(int(self.cash) - int(self.posted_collateral))

    def unrealized_pnl(self, marks: dict[str, Money]) -> Money:
        total = 0
        for symbol in sorted(self.positions):
            mark = marks.get(symbol)
            if mark is not None:
                total += int(self.positions[symbol].unrealized_pnl(mark))
        return Money(total)

    @property
    def realized_pnl(self) -> Money:
        return Money(sum(int(p.realized_pnl) for p in self.positions.values()))

    def equity(self, marks: dict[str, Money]) -> Money:
        """Total account value: cash plus mark-to-market on open positions.

        ``cash`` already contains realised PnL, because realising is what moves
        cash. Adding ``realized_pnl`` here too would double-count it, and make a
        strategy look twice as profitable as it was.
        """
        return Money(int(self.cash) + int(self.unrealized_pnl(marks)))

    # -- trading -----------------------------------------------------------

    @staticmethod
    def collateral_required(
        quantity: int, price: Money, bounds: tuple[Money, Money]
    ) -> Money:
        """Worst-case loss on a position of ``quantity`` lots opened at ``price``.

        For a position not yet taken, where a price is all there is. Once it
        exists, charge :meth:`collateral_for_basis` against what it actually
        paid: an average price is a division and the basis is not.
        """
        return Account.collateral_for_basis(
            quantity, Money(quantity * int(price)), bounds
        )

    @staticmethod
    def collateral_for_basis(
        quantity: int, cost_basis: Money, bounds: tuple[Money, Money]
    ) -> Money:
        """Worst-case loss on a position holding exactly this basis.

        One expression covers both directions, because a short is a negative
        quantity carrying a negative basis. The position will be worth
        ``quantity * value`` and it paid ``cost_basis`` for that, so the loss is
        ``cost_basis - quantity * edge`` where the edge is the bottom of the
        claim's range for a long and the top of it for a short.

        Charged against the basis rather than against an average derived from
        it, because the average needs a division and the basis does not.
        `apply_fill` used to post ``collateral_required(quantity, cost_basis //
        quantity, bounds)``, and floor division rounds a long's average *down*:
        measured on seven lots bought as three at 10.25 and four at 11.50, the
        basis is 76,750,000 minor units and the collateral posted was
        76,749,995, five short of what the position can lose. Under a minor
        unit per lot, and not zero, and this module's whole claim is that the
        figure is exact rather than close.

        Never negative. A long opened below the least the claim can settle for
        cannot lose anything, and reporting that as a negative requirement would
        hand the account a credit against its other positions, since
        `posted_collateral` simply sums the dictionary. Measured before the
        clamp: a short of one lot at 12,000 on a contract bounded by [0, 10000]
        posted -2,000, which is 2,000 of spending power conjured out of a
        position that cannot make money.
        """
        if quantity == 0:
            return Money(0)
        edge = int(bounds[0]) if quantity > 0 else int(bounds[1])
        return Money(max(0, int(cost_basis) - quantity * edge))

    def can_afford(
        self, symbol: str, quantity: int, price: Money, bounds: tuple[Money, Money]
    ) -> bool:
        """Whether the account could cover the position this fill would produce.

        Evaluated on the *resulting* position rather than the incremental trade,
        so a trade that reduces exposure stays admissible even with no free
        cash. It must: otherwise an agent becomes unable to close a losing
        position at precisely the moment it needs to.

        And on the resulting position's *basis*, not on its quantity priced at
        the incoming trade. The two differ whenever a fill adds to a position
        at a different price, and they differ in the dangerous direction:
        measured on an account holding 50,000 of cash and ten lots long at
        5,000, a further ten lots at 100 was checked as needing ``20 * 100 =
        2,000`` and produced a position carrying a basis of 51,000. The account
        passed, then posted more collateral than it owned (`free_cash` went to
        -1,000) and stood to owe a thousand it did not have if the contract
        settled at the bottom of its range. Full collateralisation says an
        account can lose everything it committed and never more; that check let
        it be more.
        """
        position = self.positions.get(symbol)
        current = position.quantity if position else 0
        resulting = current + quantity
        if resulting == 0:
            return True

        basis = (
            position.basis_after(quantity, price)
            if position is not None
            else Money(quantity * int(price))
        )
        required = int(self.collateral_for_basis(resulting, basis, bounds))
        released = int(self.collateral.get(symbol, Money(0)))
        return int(self.free_cash) + released >= required

    def apply_fill(
        self,
        symbol: str,
        quantity: int,
        price: Money,
        bounds: tuple[Money, Money],
        fee: Money = Money(0),
    ) -> None:
        """Book an execution, moving cash and re-posting collateral."""
        position = self.position(symbol)
        record = position.apply_fill(quantity, price, fee)

        # Realised PnL and fees are the only things that move cash before
        # settlement. The notional does not: a futures position is a
        # collateralised commitment, not a purchase, so debiting
        # quantity * price would be spot accounting and would report a trader as
        # broke the instant they opened a large position.
        self.cash = Money(int(self.cash) + int(record.realized) - int(fee))

        if position.quantity == 0:
            self.collateral.pop(symbol, None)
        else:
            # Charged against the position's own basis, which is what it would
            # lose. Not against an average derived from it: that division
            # floors, and on a long it floors the average downwards, so the
            # requirement came out under the loss by whatever the basis left
            # over, up to one minor unit a lot, every time two fills went on at
            # different prices.
            self.collateral[symbol] = self.collateral_for_basis(
                position.quantity, position.cost_basis, bounds
            )

    def distribute(
        self, symbol: str, per_unit: Money, bounds: tuple[Money, Money]
    ) -> Money:
        """Pay (or charge) a distribution on this symbol, and re-post collateral.

        Longs receive, shorts pay, and the two are the same line of arithmetic
        because a short position is a negative quantity. Nothing is realised:
        the holder gets cash and the contract is worth exactly that much less,
        so equity does not move. That is the correct accounting and it is also
        the thing most likely to be got wrong. Booking a dividend as profit
        would report a holder as making money for holding.

        ``bounds`` are the claim's range *after* this payment, which is what
        makes the collateral work out. Paying ``d`` per unit lowers both ends of
        what is left by ``d``, so a short's requirement falls by exactly the
        cash it just paid and a long's rises by exactly the cash it just
        received. Neither can be made insolvent by a payment it was always
        going to make.
        """
        position = self.positions.get(symbol)
        if position is None or position.quantity == 0:
            return Money(0)

        amount = Money(position.quantity * int(per_unit))
        self.cash = Money(int(self.cash) + int(amount))
        self.collateral[symbol] = self.collateral_for_basis(
            position.quantity, position.cost_basis, bounds
        )
        return amount

    # -- settlement --------------------------------------------------------

    def settle(self, symbol: str, settlement_value: Money) -> Money:
        """Settle an expired contract: realise against the final value, free collateral.

        Idempotent by symbol: settling twice would pay a position out twice,
        and an expiry firing more than once is a plausible bug in any
        event-driven system.
        """
        if symbol in self.settled:
            raise ValueError(f"{symbol} has already settled for {self.agent_id}")
        self.settled.add(symbol)

        position = self.positions.get(symbol)
        if position is None or position.quantity == 0:
            self.collateral.pop(symbol, None)
            return Money(0)

        realized = position.unrealized_pnl(settlement_value)
        position.realized_pnl = Money(int(position.realized_pnl) + int(realized))
        position.quantity = 0
        position.cost_basis = Money(0)
        self.cash = Money(int(self.cash) + int(realized))
        self.collateral.pop(symbol, None)
        return realized

    def void(self, symbol: str) -> None:
        """Release a voided contract's collateral without realising anything.

        A void means the world never produced the evidence the contract needed.
        Nobody wins or loses; collateral comes back and the position ceases to
        exist. Paying out a guess instead would make the guess indistinguishable
        from a measurement the moment it reached a PnL statement.
        """
        self.settled.add(symbol)
        position = self.positions.get(symbol)
        if position is not None:
            position.quantity = 0
            position.cost_basis = Money(0)
        self.collateral.pop(symbol, None)

    def to_dict(self, marks: dict[str, Money] | None = None) -> dict[str, Any]:
        marks = marks or {}
        return {
            "agent_id": self.agent_id,
            "cash": str(from_money(self.cash)),
            "starting_cash": str(from_money(self.starting_cash)),
            "posted_collateral": str(from_money(self.posted_collateral)),
            "free_cash": str(from_money(self.free_cash)),
            "realized_pnl": str(from_money(self.realized_pnl)),
            "unrealized_pnl": str(from_money(self.unrealized_pnl(marks))),
            "equity": str(from_money(self.equity(marks))),
            "positions": [
                self.positions[s].to_dict(marks.get(s))
                for s in sorted(self.positions)
                if self.positions[s].quantity != 0 or self.positions[s].volume > 0
            ],
        }
