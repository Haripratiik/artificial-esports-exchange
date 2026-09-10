"""A collection of strategies run as a firm, rather than as a pile of bots.

A strategy that is right about a price is not yet a business. What turns one
into a business is the thing that decides how much capital it is allowed to
consume, what happens when two of them want the same capital at the same
instant, and what the answer is when the account is down. That is this module.
:class:`Firm` satisfies both strategy protocols by delegating to the strategies
it holds, so it drops into :class:`~arena.agents.strategy_agent.StrategyAgent`
exactly where a single strategy would, and everything it adds is arithmetic on
quantities the venue already charges in.

**The budget is denominated in collateral, and the unit of budgeting is the
netting group.** Measured on the live catalogue, the exchange is not 50
instruments, it is 22 groups keyed exactly the way ``Venue._underlying_of``
keys them: win_rate on QUILL in the solo format with 7 members, on BASTION in
the solo format with 6, on EMBER in the objective format with 6 and on VANTA in
the solo format with 4, match_volume on RIFT with 5, five further objective
win_rate groups with 2 apiece, and twelve groups holding a single contract, one
of which is a difference and one a basket. A group is exactly the set within
which capital is released,
because it is exactly the set :func:`~arena.portfolio.netting.worst_case` will
accept, so it is the only granularity at which a capital budget means anything
at all. The key is derived from the contract's underlying rather than from its
symbol, so a contract listed tomorrow lands in the right group without anybody
editing a list.

**Collateral is a better budgeting primitive than the industry's, not a worse
one.** The usual buy-side budget is expressed in value-at-risk, and VaR is not
subadditive, so per-strategy VaR budgets need not sum to a portfolio bound.
Artzner, Delbaen, Eber and Heath's own counterexample for that is a pair of
digital options, which this venue lists ten of. The worst case here is exact
arithmetic over a bounded scalar, it is subadditive by construction, and a sum
of per-group budgets is therefore a real bound rather than a hopeful one. No
covariance matrix is estimated across the 22 groups, and none should be: from
minutes of one seed it would be noise, and it would smuggle an estimate back
into the one place this project refuses one.

**Two sizing rules the literature would suggest are wrong here, and neither is
implemented.** Volatility targeting levers to sigma_target over sigma_hat, and
variance in this world shrinks deterministically as the observation window
fills with evidence, so such a rule would mechanically lever *up* into expiry,
which is backwards. Stop-losses cap an otherwise unbounded loss or head off a
margin call, and there is neither here: the loss is already capped by the
collateral posted, there is no margin call, and stopping out converts a
bounded, already-funded loss into a realised one plus the spread. The one
legitimate flatten is a package that has become half-legged, which is an
execution failure rather than a view being wrong, and that one is here.

**The venue charges gross today.** ``Venue.__init__`` takes ``netting: bool =
False`` and no call site sets it True, so the live venue posts per-contract
collateral. The group budget is therefore run against the netted figure, which
is what the firm can actually lose, while the cash reserve is run against the
gross figure, which is what the account is actually charged. Both are reported
side by side through :func:`~arena.portfolio.netting.netting_benefit`, so the
capital a clearing house would release is a number rather than an assertion.

**A firm sees no more than a strategy does.** Everything here is computed from
the :class:`~arena.strategies.base.MarketView` and from the intents the firm
itself emitted, so this risk management is available to somebody testing a
strategy without real money, which is the point of the whole package.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

from arena.contracts.payoff import Binary, Call, Linear, Payoff, Put
from arena.contracts.spec import ContractSpec
from arena.contracts.underlying import Basket, Difference, Single, Underlying
from arena.determinism import canonical_json, digest
from arena.exchange.types import Side
from arena.market.instrument import Instrument
from arena.portfolio.netting import netting_benefit, worst_case
from arena.strategies.base import MarketView, Quote, SymbolView, Take, TwoSided

__all__ = [
    "Firm",
    "FirmReport",
    "GroupReport",
    "StrategyReport",
    "group_key",
    "group_label",
    "metric_sensitivity",
]


# How many of its own fills a strategy needs before the firm will charge it a
# cost of capital. Below this its measured edge is `None`, which is the honest
# answer early on, and an unmeasured strategy is never refused for its record.
#
# Eight, and the number deserves less confidence than the rest of this module,
# because the measurement that was supposed to justify it said the opposite of
# what was expected. Running the three strategies of `tests/test_firm.py` over
# 120 simulated seconds on seeds 7, 13 and 41 and recording the sign of each
# one's edge per unit of collateral after every fill, the sign was still
# flipping at the end of every run: last flip at fill 173 of 174, at 98 of 98,
# at 100 of 114, at 77 of 78. There is no fill count inside a session after
# which this ratio settles, so this bar cannot be and is not the point the
# measurement becomes stable.
#
# What it is is a floor under the denominator. The ratio is P&L over mean
# committed collateral, and both halves are noisy after two prints, so a bar of
# one gates a strategy on a single fill. Eight is roughly one wake's worth of
# activity for the strategies measured above and is small enough that the
# charge still binds inside a minute.
#
# The bar is not identifiable from what it earns either, and the second
# measurement is worth recording because it says so plainly. Twelve live runs
# of sixty simulated seconds, three seeds by four settings, firm P&L and the
# number of intents the charge refused:
#
#     bar        seed 7            seed 13           seed 41
#       1    -71,234 / 337      -4,329 / 650    +715,447 / 374
#       8    -87,954 / 388      -8,968 / 577    +158,190 / 469
#      32    -99,576 / 446      +4,327 / 563    +786,992 / 448
#     off    -55,593 /   0     -21,845 /   0    +615,715 /   0
#
# Neither column moves with the bar. Seed 41 earns 158,190 at eight, 786,992 at
# thirty-two and 615,715 with the gate switched off entirely, and the refusal
# count is not monotone either, falling from 650 to 563 on seed 13 as the bar
# rises. What that spread measures is trajectory divergence rather than the
# setting: refusing one intent moves every subsequent fill. So this constant is
# not tuned to P&L and must not be, since tuning it to a table like that one is
# fitting three numbers. Conservation was exactly zero in all twelve runs.
#
# A caller who does not want a gate whose input is this noisy should pass
# `cost_of_capital=False`, and the honest reading of the measurement above is
# that this is a defensible thing to want.
MIN_FILLS_FOR_EDGE = 8

# How long an order gets after the market has traded past it before the firm
# writes it off, in wakes. It sets both the moment a take stops claiming lots
# it will never get and the moment a package is judged half-legged, and those
# have to be the same number: a package is judged on whether its legs still
# have orders outstanding, so a claim that outlives its order would hold a
# package open forever and one that dies early would call it half-legged while
# a leg was still on the wire.
#
# One, because `StrategyAgent._apply_take` sends every take as IOC or as a
# market order, so a leg that is going to resolve resolves within one round
# trip: the live market's default pairwise latency is 4ms against a wake
# interval of 300ms, which is seventy-five round trips of slack. The clock does
# not start until the symbol prints without us, so a book that has stopped
# trading never starts it. See `Firm._reap` for what happened when it did.
PACKAGE_WAKES = 1

# Passes of the water-filling loop that redistributes a concentration-capped
# group's budget across the others. Bounded rather than run to convergence:
# each pass can only newly cap a group that was uncapped before, and the live
# catalogue has ten.
_ALLOCATION_PASSES = 12


def group_key(instrument: Instrument) -> str:
    """The netting group a contract belongs to, keyed the way the venue keys it.

    Byte-identical to ``Venue._underlying_of``, and deliberately so. A budget
    enforced on a grouping finer than the venue's would refuse packages the
    venue funds; one on a coarser grouping would try to net across two
    underlyings, and :func:`~arena.portfolio.netting.worst_case` raises rather
    than answering, because netting those needs a correlation and a correlation
    is an estimate.
    """
    return canonical_json(instrument.spec.underlying.to_dict())


def group_label(underlying: Underlying) -> str:
    """A readable name for a group, derived from its structure, never its symbol.

    Reporting only. The identity a budget is enforced against is always
    :func:`group_key`. Two distinct underlyings can share a label, so
    :class:`Firm` disambiguates a collision with a digest suffix rather than
    letting two groups share one row of a report.
    """
    if isinstance(underlying, Single):
        return f"{underlying.ref.metric}:{underlying.ref.subject}"
    if isinstance(underlying, Difference):
        return "difference"
    if isinstance(underlying, Basket):
        return "basket"
    return type(underlying).__name__.lower()


def _slope(payoff: Payoff, level: float) -> float:
    """``d payoff / d level``, exactly, at one level of the underlying.

    Four shapes and four answers. Linear gives its scale everywhere. A call
    gives its scale above the strike and nothing below, a put the negative of
    its scale below the strike and nothing above, and a binary gives zero
    almost everywhere: its whole value sits in a jump at the threshold, which
    has no derivative and is not exposure in the sense a delta limit is about.
    The jump is not ignored elsewhere, it is exactly what
    :func:`~arena.portfolio.netting.kinks_of` hands the collateral arithmetic
    as a candidate level, so the risk is charged for even though the
    sensitivity is zero.

    The kink itself is assigned to the flat branch, so a call struck exactly at
    the level contributes nothing. Either convention is defensible at a single
    point and this one is the right-hand derivative of the payoff.
    """
    if isinstance(payoff, Linear):
        return payoff.scale
    if isinstance(payoff, Call):
        return payoff.scale if payoff.scale * level > payoff.strike else 0.0
    if isinstance(payoff, Put):
        return -payoff.scale if payoff.scale * level < payoff.strike else 0.0
    if isinstance(payoff, Binary):
        return 0.0
    raise TypeError(
        f"{type(payoff).__name__} has no declared slope, so the metric-space "
        "exposure of a portfolio holding it cannot be computed exactly. Teach "
        "_slope the shape rather than differencing it numerically: a difference "
        "quotient taken across a kink reports a sensitivity the payoff does not "
        "have anywhere."
    )


def metric_sensitivity(spec: ContractSpec, quantity: int, level: float) -> float:
    """``dV/dtheta`` for ``quantity`` lots of one contract, in metric space.

    The whole claim, so a contract that pays a stream as well as a settlement
    is differentiated through both. Exact rather than bumped: every payoff here
    declares its own slope, and the level the slope is taken at is read off the
    market rather than assumed.
    """
    slope = _slope(spec.payoff, level)
    if spec.distribution is not None:
        slope += _slope(spec.distribution.payoff, level) * len(spec.distribution.windows)
    return quantity * slope


def _affine_claim(spec: ContractSpec) -> tuple[float, float] | None:
    """``(a, b)`` where the whole claim is ``a * level + b``, or None.

    Only a linear settlement with a linear stream is invertible, which is what
    lets a price be turned back into the level the market is pricing. An option
    or a binary is not, and recovering a level from one would need exactly the
    volatility model this project does without.
    """
    if not isinstance(spec.payoff, Linear):
        return None
    a, b = spec.payoff.scale, spec.payoff.offset
    if spec.distribution is not None:
        if not isinstance(spec.distribution.payoff, Linear):
            return None
        windows = len(spec.distribution.windows)
        a += spec.distribution.payoff.scale * windows
        b += spec.distribution.payoff.offset * windows
    return (a, b) if a else None


def _gross_of(holdings: Sequence[tuple[ContractSpec, int, Decimal]]) -> Decimal:
    """What charging every contract in a portfolio separately costs."""
    return sum(
        (spec.collateral_for(quantity, price) for spec, quantity, price in holdings),
        start=Decimal(0),
    )


@dataclass(frozen=True)
class GroupReport:
    """One netting group's capital, risk and price, at one instant."""

    key: str
    label: str
    members: int
    budget: Decimal
    committed: Decimal
    gross: Decimal
    net: Decimal
    shadow_price: float
    shadow_price_mean: float
    level: float
    implied: bool
    delta: float
    collateral_seconds: float
    binding: Mapping[str, int]

    @property
    def benefit(self) -> Decimal:
        """Capital a clearing house would release against these holdings."""
        return self.gross - self.net


@dataclass(frozen=True)
class StrategyReport:
    """One strategy's contribution, in the units the firm charges it in."""

    name: str
    pnl: Decimal
    realised: Decimal
    unrealised: Decimal
    fills: int
    lots: int
    collateral_seconds: float
    # P&L per unit of collateral committed, dimensionless, so it is directly
    # comparable with a group's shadow price. `None` below MIN_FILLS_FOR_EDGE
    # fills rather than a number, because a strategy is not measurably anything
    # after two prints and refusing it capital on that basis would be a coin
    # toss wearing a limit.
    edge_per_collateral: float | None
    refused: Mapping[str, int]

    @property
    def pnl_per_collateral_second(self) -> float:
        """The venue-native efficiency metric, and exact here.

        Collateral is arithmetic rather than an estimate, so the denominator is
        a measured quantity of capital genuinely tied up rather than a modelled
        one. A strategy making the same money on half the collateral for half
        the time is four times the business, and no other ratio available here
        says so.
        """
        if self.collateral_seconds <= 0.0:
            return 0.0
        return float(self.pnl) / self.collateral_seconds


@dataclass(frozen=True)
class FirmReport:
    """Everything the firm knows about itself, assembled from one view."""

    now: float
    capital: Decimal
    equity: Decimal
    high_water_mark: Decimal
    floor: Decimal
    gross: Decimal
    net: Decimal
    unattributed: Decimal
    # Fills that arrived against no claim at all, which is the only way the
    # residue above becomes anything but a rounding. There is exactly one
    # mechanism for it and a strategy cannot see it coming: a withdrawal is a
    # Cancel in flight, and a Cancel can lose the race to a fill. Reported as a
    # count rather than folded into the money, because "0.7% of P&L" and "one
    # fill in a session" are different facts and the second is the diagnostic.
    unattributed_fills: int
    groups: Mapping[str, GroupReport]
    strategies: Mapping[str, StrategyReport]
    equity_curve: Sequence[tuple[float, Decimal]]
    binding: Mapping[str, int]
    # The limit that bound first, and the simulated second it did. Which limit
    # binds *most* is a different question from which one the firm met first,
    # and the second is the one that says what the mandate is really
    # constrained by rather than which check happens to be asked most often.
    first_binding: tuple[str, float] | None

    @property
    def benefit(self) -> Decimal:
        """Gross collateral minus netted, across every group."""
        return self.gross - self.net

    @property
    def pnl(self) -> Decimal:
        """The firm's whole P&L, which the per-strategy figures sum to exactly."""
        return self.equity - self.capital

    def reconciles(self) -> bool:
        """Whether attribution accounts for every unit of the firm's P&L.

        An identity rather than a tolerance: the per-strategy figures plus the
        unattributed residue are the firm's total, in Decimal, exactly. The
        residue exists because one observation covering fills in more than one
        symbol cannot recover a per-symbol price from a single cash number, and
        it is reported rather than spread over strategies that did not earn it.
        """
        attributed = sum((s.pnl for s in self.strategies.values()), start=Decimal(0))
        return attributed + self.unattributed == self.pnl


@dataclass
class _Claim:
    """One order the firm has out, and who it is out for.

    The firm's whole record of its own working interest, and one object rather
    than two because the two jobs have to agree: it says who the lots that
    arrive belong to, and it is what the group budget projects against, so an
    order the firm has forgotten can be neither misattributed nor spent twice.
    """

    name: str
    lots: int
    price: Decimal
    # A quote rests until it is replaced. A take does not, and dies a wake
    # after the market first trades past it without filling it.
    resting: bool
    # The symbol's last print when the order was sent, and the wake the firm
    # first saw a print newer than that. Together they are how a dead take is
    # told from one that is merely waiting for a market to reopen.
    last: Decimal | None
    dead_from: int | None = None


@dataclass
class _Ledger:
    """One strategy's own book, rebuilt from what the firm sent and then saw.

    A desk reconciles its own blotter against the clearer's rather than being
    told its position, and this is that: the firm tags every intent with the
    strategy that asked for it, watches the position and the cash the view
    reports, and books the difference against the tag. Nothing here reads a
    fill event, because a strategy cannot.
    """

    cash: Decimal = Decimal(0)
    lots: dict[str, int] = field(default_factory=dict)
    basis: dict[str, Decimal] = field(default_factory=dict)
    fills: int = 0
    collateral_seconds: float = 0.0
    refused: dict[str, int] = field(default_factory=dict)

    def apply(self, symbol: str, quantity: int, cash: Decimal) -> None:
        """Book ``quantity`` lots whose total cash flow was ``cash``.

        Average cost, and the flip is the case that gets it wrong: closing four
        lots and opening eleven the other way is two events, and a basis that
        treats it as one carries the closed lots' cost into the new position.
        ``Venue._survives`` documents the same mistake costing an account
        190,370,000 minor units of free cash on the venue's own side of the
        ledger.
        """
        held = self.lots.get(symbol, 0)
        self.basis[symbol] = _basis_after(
            held, self.basis.get(symbol, Decimal(0)), quantity, cash
        )
        self.lots[symbol] = held + quantity
        self.cash += cash
        self.fills += 1

    def mark_value(self, marks: Mapping[str, Decimal]) -> Decimal:
        return sum(
            (
                Decimal(quantity) * marks[symbol]
                for symbol, quantity in self.lots.items()
                if quantity and symbol in marks
            ),
            start=Decimal(0),
        )

    def carried_basis(self) -> Decimal:
        return sum(
            (b for s, b in self.basis.items() if self.lots.get(s)), start=Decimal(0)
        )

    def pnl(self, marks: Mapping[str, Decimal]) -> Decimal:
        return self.cash + self.mark_value(marks)


def _basis_after(
    held: int, basis: Decimal, quantity: int, cash: Decimal
) -> Decimal:
    """The cost basis ``quantity`` lots at a total cash flow of ``cash`` leaves.

    Adding to a position carries its cost in. Reducing one releases the closed
    lots' share of the basis and leaves the rest. Flipping does both, and the
    remainder opens at the price the fill actually paid rather than at the
    average of a position that no longer exists.
    """
    if held == 0 or (held > 0) == (quantity > 0):
        return basis - cash
    closed = min(abs(quantity), abs(held))
    basis -= basis * Decimal(closed) / Decimal(abs(held))
    if abs(quantity) > abs(held):
        opened = abs(quantity) - abs(held)
        price = -cash / Decimal(quantity)
        basis = price * Decimal(opened if quantity > 0 else -opened)
    return basis


class Firm:
    """Strategies, a capital budget, and the limits that make them a firm.

    ``Firm(budget=Decimal(250_000), strategies={"mm": maker, "rv": taker})`` is
    the whole constructor anyone needs. The firm satisfies both protocols, so
    it goes into the agent as both::

        StrategyAgent(agent_id, VENUE_ID, instruments,
                      maker=firm, taker=firm, starting_cash=budget)

    ``starting_cash`` matters. The agent keeps its own shadow ledger and the
    firm reads its capital changes from it, so an agent told nothing starts at
    zero and the firm's banked cash reads as a loss it never took.

    Passing the firm as the *maker* matters too, even for a firm holding only
    takers. ``StrategyAgent`` asks its maker to quote again the instant it is
    filled, which hands the firm one fill at a time, and one fill at a time is
    what makes that fill's price exactly recoverable from the cash the view
    reports. Attribution still reconciles without it, with the price residue
    reported as ``unattributed`` instead of being spread over the strategies.
    """

    def __init__(
        self,
        budget: Decimal | int,
        strategies: Mapping[str, Any],
        *,
        reserve: float = 0.10,
        concentration: float = 0.40,
        max_drawdown: float = 0.20,
        drawdown_multiple: float = 3.0,
        gross_metric_limit: float | None = None,
        net_metric_limit: float | None = None,
        cost_of_capital: bool = True,
        equity_curve_limit: int = 4096,
    ) -> None:
        if not strategies:
            raise ValueError("a firm with no strategies is not a firm")
        if not 0.0 <= reserve < 1.0:
            raise ValueError(f"reserve must lie in [0, 1), got {reserve}")
        if not 0.0 < concentration <= 1.0:
            raise ValueError(f"concentration must lie in (0, 1], got {concentration}")
        if not 0.0 < max_drawdown <= 1.0:
            raise ValueError(f"max_drawdown must lie in (0, 1], got {max_drawdown}")
        if drawdown_multiple <= 0.0:
            raise ValueError(
                f"drawdown_multiple must be positive, got {drawdown_multiple}"
            )

        self.budget = Decimal(budget)
        self.strategies = dict(strategies)
        self.reserve = reserve
        self.concentration = concentration
        self.max_drawdown = max_drawdown
        self.drawdown_multiple = drawdown_multiple
        # No default for either metric limit, deliberately. A delta cap's
        # natural scale is the book's own: a win-rate future here has a slope
        # of 10,000 per unit of metric and a battle-volume contract 500, so any
        # figure this module picked would be a constant nobody measured. Off
        # unless the caller names one, in the units the payoffs are written in.
        self.gross_metric_limit = gross_metric_limit
        self.net_metric_limit = net_metric_limit
        self.cost_of_capital = cost_of_capital
        self.equity_curve_limit = equity_curve_limit

        self._ledgers: dict[str, _Ledger] = {n: _Ledger() for n in self.strategies}
        # Lots the firm holds that no strategy claimed. Its P&L is carried in
        # `unattributed` rather than being handed to whichever strategy asked
        # last, so the reconciliation identity stays an identity.
        self._orphan = _Ledger()

        self._label: dict[str, str] = {}
        self._members: dict[str, list[str]] = {}
        self._of_symbol: dict[str, str] = {}
        self._universe: tuple[str, ...] = ()
        self._budgets: dict[str, Decimal] = {}

        self._lots: dict[str, int] = {}
        self._basis: dict[str, Decimal] = {}
        self._marks: dict[str, Decimal] = {}
        self._specs: dict[str, ContractSpec] = {}
        self._levels: dict[str, tuple[float, bool]] = {}
        self._cash = Decimal(0)
        self._first_cash: Decimal | None = None
        # The agent's total posted collateral as of the last observation. It is
        # the only public number that carries a position's *basis*, and the
        # basis is what collateral is charged against, so this is what a
        # strategy has instead of a blotter.
        self._collateral_seen = Decimal(0)
        self._residue = Decimal(0)

        # Every order the firm has out, per side of per symbol, consumed
        # first in first out. Both the attribution key and the thing the budget
        # projects against: a maker and a taker buying one symbol in one wake
        # are not confused for each other, and neither is funded twice.
        self._claims: dict[tuple[str, Side], list[_Claim]] = {}
        self._wanted: dict[str, frozenset[str]] = {}

        self._shadow: dict[str, float] = {}
        self._shadow_now: dict[str, float] = {}
        self._shadow_seconds: dict[str, float] = {}
        self._committed: dict[str, Decimal] = {}
        self._gross: dict[str, Decimal] = {}
        self._group_seconds: dict[str, float] = {}
        # Cash the firm has moved in each group, so a group's P&L is its whole
        # P&L rather than the part of it still sitting in open inventory.
        self._group_cash: dict[str, Decimal] = {}
        self._binding: dict[str, int] = {}
        self._group_binding: dict[str, dict[str, int]] = {}
        self._first_binding: tuple[str, float] | None = None

        self._now = 0.0
        self._started: float | None = None
        self._wake = 0
        self._high_water: Decimal | None = None
        self._equity_curve: list[tuple[float, Decimal]] = []
        self._packages: dict[str, tuple[int, dict[str, int], dict[str, int]]] = {}

    # -- the two protocols --------------------------------------------------

    def symbols(self, view: MarketView) -> Sequence[str]:
        """Every contract any maker in the firm wants to quote, deduplicated.

        Also the once-a-wake hook where each maker is asked what it wants, so
        that :meth:`quote` can tell a maker that asked for a symbol from one
        that is only being requoted because it was filled in it.
        """
        self._observe(view)
        wanted: dict[str, None] = {}
        for name, strategy in self.strategies.items():
            if not callable(getattr(strategy, "quote", None)):
                continue
            chooser = getattr(strategy, "symbols", None)
            asked = tuple(view.symbols) if chooser is None else tuple(chooser(view))
            self._wanted[name] = frozenset(asked)
            for symbol in asked:
                if symbol in view:
                    wanted[symbol] = None
        return tuple(wanted)

    def quote(self, view: MarketView, symbol: str) -> TwoSided:
        """What the firm wants resting in one symbol, after the capital contest.

        Every maker that asked for the symbol is polled, the two sides are
        settled separately because the venue collateralises them as separate
        scenarios, and the winner of a side is the strategy with the highest
        edge per unit of collateral. That ordering is the internal cost of
        capital: a strategy below the group's shadow price loses the seat to
        one above it, which is what a Sharpe-ranked allocation is trying to
        approximate and what this can do exactly.
        """
        self._observe(view)
        symbol_view = view.get(symbol)
        if symbol_view is None:
            return TwoSided()

        bids: list[tuple[float, int, str, Quote]] = []
        asks: list[tuple[float, int, str, Quote]] = []
        for order, (name, strategy) in enumerate(self.strategies.items()):
            quoter = getattr(strategy, "quote", None)
            if quoter is None:
                continue
            asked = self._wanted.get(name)
            if asked is not None and symbol not in asked:
                continue
            wanted = quoter(view, symbol)
            if wanted.bid is not None:
                edge = self._edge(name, symbol_view, Side.BUY, wanted.bid)
                bids.append((edge, order, name, wanted.bid))
            if wanted.ask is not None:
                edge = self._edge(name, symbol_view, Side.SELL, wanted.ask)
                asks.append((edge, order, name, wanted.ask))

        return TwoSided(
            bid=self._settle_side(view, symbol_view, Side.BUY, bids),
            ask=self._settle_side(view, symbol_view, Side.SELL, asks),
        )

    def orders(self, view: MarketView) -> Sequence[Take]:
        """Everything the firm wants executed now, in the order it wants it.

        Flattening legs first, because a half-legged package is the one
        position here whose risk is nobody's intent, and it is the only unwind
        this firm ever does. Then each taker's intents, ranked and clipped by
        the same capital contest the quotes go through, with a strategy that
        declares itself packaged admitted whole or not at all.
        """
        self._observe(view)
        self._wake += 1
        # Reaped again on the new wake number rather than only on the old one,
        # because a claim's whole lifetime is a wake and the package sitting on
        # it is resolved in the next line. Left to the next observation, a
        # half-legged package waited a wake longer than it had to.
        self._reap(view)
        out: list[Take] = list(self._resolve_packages(view))

        ranked: list[tuple[float, int, str, Take, SymbolView]] = []
        packaged: dict[str, list[tuple[Take, SymbolView]]] = {}
        for order, (name, strategy) in enumerate(self.strategies.items()):
            lister = getattr(strategy, "orders", None)
            if lister is None:
                continue
            for intent in lister(view):
                symbol_view = view.get(intent.symbol)
                if symbol_view is None:
                    continue
                if getattr(strategy, "packaged", False):
                    packaged.setdefault(name, []).append((intent, symbol_view))
                    continue
                price = self._take_price(symbol_view, intent)
                edge = self._edge(
                    name, symbol_view, intent.side, Quote(price, intent.size)
                )
                ranked.append((edge, order, name, intent, symbol_view))

        for _edge, _order, name, intent, symbol_view in sorted(
            ranked, key=lambda row: (-row[0], row[1], row[3].symbol)
        ):
            price = self._take_price(symbol_view, intent)
            size = self._admit(name, symbol_view, intent.side, price, intent.size)
            if size <= 0:
                continue
            self._claim(symbol_view, intent.side, name, size, price)
            out.append(Take(intent.symbol, intent.side, size, intent.limit))

        out.extend(self._admit_packages(packaged))
        return tuple(out)

    # -- the capital contest ------------------------------------------------

    def _settle_side(
        self,
        view: MarketView,
        symbol_view: SymbolView,
        side: Side,
        candidates: list[tuple[float, int, str, Quote]],
    ) -> Quote | None:
        """One side of one symbol, awarded to the best bidder for the capital."""
        del view
        queue = self._claims.get((symbol_view.symbol, side))
        if not candidates:
            # No quote coming means this side is withdrawn, and a withdrawal is
            # a Cancel in flight rather than an order that has already gone.
            # The claim stops being a resting one and becomes reapable, exactly
            # like a take: it still catches a fill that crosses the cancel on
            # the wire, and it dies once the market prints without us. Deleting
            # it outright lost the four lots of a win rate future the uncross
            # booked against an empty queue on seed 7.
            if queue is not None:
                for claim in queue:
                    claim.resting = False
            return None
        candidates.sort(key=lambda row: (-row[0], row[1]))
        _edge, _order, name, quote = candidates[0]
        for losing in candidates[1:]:
            self._refuse(losing[2], "outbid")
        size = self._admit(name, symbol_view, side, quote.price, quote.size, resting=True)
        if size <= 0:
            return None
        self._claim(symbol_view, side, name, size, quote.price, resting=True)
        # Clipped by `replace` rather than rebuilt, so that the size is the only
        # thing the firm changes about a maker's quote. Rebuilding it silently
        # dropped `post_only` the day that field was added, and a maker that
        # asked never to cross would have started crossing because its risk
        # manager resized it.
        return replace(quote, size=size)

    def _admit(
        self,
        name: str,
        symbol_view: SymbolView,
        side: Side,
        price: Decimal,
        size: int,
        resting: bool = False,
    ) -> int:
        """The largest part of an intent every limit will fund, in lots.

        The group budget is checked against :func:`worst_case` on the holdings
        the intent would create, not against a linear proxy for it, and that
        matters in the direction a proxy is always wrong: a hedging leg
        *reduces* a group's worst case, so a per-contract approximation would
        refuse the one order that makes the book safer. The bisection is exact
        rather than a search over an unknown shape, because ``worst_case`` is a
        maximum of affine functions of the size and is therefore convex in it,
        so the fundable sizes are an interval starting at zero.
        """
        if size <= 0:
            return 0
        group = self._of_symbol.get(symbol_view.symbol)
        if group is None:
            return 0

        # A quote replaces this side's working order, so the order it
        # supersedes is left out of the projection. A take adds to whatever is
        # already resting, so nothing is.
        base = self._projected(group, side, symbol_view.symbol if resting else None)
        spec = symbol_view.instrument.spec
        signed = 1 if side is Side.BUY else -1
        # An order that lowers the group's worst case is never refused, for any
        # reason, and this branch is why the firm is not a stop-loss machine
        # wearing a budget. Measured without it, over twenty simulated seconds
        # on seed 7: a maker whose marked inventory turned its edge negative
        # was refused on 122 of 264 side decisions, and since a refused quote
        # is a *withdrawn* quote, that took away the only mechanism it had for
        # working out of the inventory that priced it out. Being over a limit
        # has to leave the way back under it open, or the limit is a trap
        # rather than a control.
        if worst_case([*base, (spec, signed * size, price)]) <= worst_case(base):
            return size

        if self._priced_out(name, group):
            self._refuse(name, "cost-of-capital")
            self._note_binding(group, "cost-of-capital")
            return 0

        net_cap, net_limit, gross_cap = self._caps(group)

        def fits(lots: int) -> tuple[bool, str]:
            if lots <= 0:
                return (True, "")
            holdings = [*base, (spec, signed * lots, price)]
            if worst_case(holdings) > net_cap:
                return (False, net_limit)
            if _gross_of(holdings) > gross_cap:
                return (False, "reserve")
            return self._within_metric_limits(group, spec, signed * lots)

        if fits(size)[0]:
            return size
        low, high = 0, size
        while low < high:
            middle = (low + high + 1) // 2
            if fits(middle)[0]:
                low = middle
            else:
                high = middle - 1
        # The limit that binds at the *margin*, which is the one worth
        # reporting. Reading it off the full-size attempt instead named
        # whichever check happens to run first: an intent for forty lots that
        # the budget stops at twenty and a metric cap stops at five was
        # recorded against the budget, and the budget was not what decided it.
        limit = fits(low + 1)[1]
        self._refuse(name, limit)
        self._note_binding(group, limit)
        self._price_capital(group, name, symbol_view, side, price)
        return low

    def _caps(self, group: str) -> tuple[Decimal, str, Decimal]:
        """The most collateral this group may carry, netted and gross.

        Four limits, each in the unit the venue actually charges in. Three of
        them bound what the group can *lose*, which is the netted worst case;
        the reserve bounds what the account is *charged*, which is the gross
        per-contract figure the live venue posts because ``Venue.netting`` is
        False everywhere. Mixing the two would be the units bug this repository
        has hit four times, so they are returned separately and compared
        separately. The binding one is recorded rather than merged away,
        because which limit bound is the only diagnostic that says what to
        change.
        """
        capital = self._capital()
        deployable = capital * (Decimal(1) - Decimal(str(self.reserve)))
        others_net = sum(
            (v for k, v in self._committed.items() if k != group), start=Decimal(0)
        )
        others_gross = sum(
            (v for k, v in self._gross.items() if k != group), start=Decimal(0)
        )

        caps: list[tuple[Decimal, str]] = [
            (self._budgets.get(group, Decimal(0)), "budget"),
            # A share of the collateral the firm has committed to its
            # strategies, which is the sum of the budgets and equals the
            # deployable capital by construction. Measured against currently
            # *posted* collateral the limit would be unsatisfiable for a firm
            # holding one group, since one group is then a hundred per cent of
            # what is posted, and that is a statement about arithmetic rather
            # than about risk. What justifies the limit is not a correlation
            # argument: a group IS one scalar, so everything in it can be wrong
            # at once, which is a fact and not an estimate.
            (deployable * Decimal(str(self.concentration)), "concentration"),
        ]
        if self._high_water is not None:
            # CPPI, or Grossman and Zhou's constant-proportion rule: exposure
            # is a multiple of the cushion above a floor that ratchets with the
            # high-water mark. Expressed in committed collateral so it composes
            # with the other three rather than being a separate kind of limit
            # that has to be reconciled with them by hand.
            floor = self._high_water * (Decimal(1) - Decimal(str(self.max_drawdown)))
            cushion = max(Decimal(0), self._equity() - floor)
            caps.append(
                (Decimal(str(self.drawdown_multiple)) * cushion - others_net, "drawdown")
            )
        cap, limit = min(caps, key=lambda row: (row[0], row[1]))
        # `Venue._survives` records an account reaching free cash of
        # -190,370,000 because flipping a position realises its P&L the instant
        # it books. The reserve is the honest version of unencumbered
        # liquidity: gross collateral across every group stays under the
        # deployable fraction of capital, so what is left over is genuinely
        # unencumbered rather than merely unspent.
        return (max(Decimal(0), cap), limit, max(Decimal(0), deployable - others_gross))

    def _within_metric_limits(
        self, group: str, spec: ContractSpec, quantity: int
    ) -> tuple[bool, str]:
        """Whether the added lots keep metric-space exposure inside its limits.

        ``sum |dV/dtheta_g|`` across groups and ``dV/dtheta_g`` within one, both
        exactly computable from the payoffs and neither needing an estimate.
        The level each slope is taken at is read out of the market by inverting
        the group's own linear contracts, which is reading a price rather than
        fitting a model.
        """
        if self.gross_metric_limit is None and self.net_metric_limit is None:
            return (True, "")
        deltas = self._deltas()
        level = self._levels.get(group, (0.0, False))[0]
        deltas[group] = deltas.get(group, 0.0) + metric_sensitivity(spec, quantity, level)
        if self.net_metric_limit is not None and (
            abs(deltas[group]) > self.net_metric_limit
        ):
            return (False, "net-metric")
        if self.gross_metric_limit is not None and (
            sum(abs(d) for d in deltas.values()) > self.gross_metric_limit
        ):
            return (False, "gross-metric")
        return (True, "")

    def _priced_out(self, name: str, group: str) -> bool:
        """Whether this strategy's measured edge is below the group's shadow price.

        The venue-native replacement for ranking strategies by Sharpe. A Sharpe
        ratio needs a variance, a variance here is an estimate over minutes of
        one seed, and what a capital allocator actually wants is value per unit
        of the scarce resource. The scarce resource is collateral, the shadow
        price of collateral is what one more unit of budget would have earned,
        and both sides of that comparison are exact.

        This is an allocation decision and not a stop-loss. It declines to put
        *new* capital behind a strategy that is measurably destroying it; it
        never closes a position, because a loss on an open position here is
        already bounded by the collateral posted and stopping out would convert
        a funded, bounded loss into a realised one plus the spread.
        """
        if not self.cost_of_capital:
            return False
        edge = self._measured_edge(name)
        return edge is not None and edge < self._shadow.get(group, 0.0)

    def _price_capital(
        self, group: str, name: str, symbol_view: SymbolView, side: Side, price: Decimal
    ) -> None:
        """Record the shadow price of collateral implied by a refused intent.

        Complementary slackness, in the only form available without solving an
        LP: intents are admitted in descending order of edge per unit of
        collateral, so the first one a budget turns away is the marginal
        activity and its ratio is what one more unit of budget would buy. A
        group whose budget never bound has a shadow price of zero, which is the
        right answer rather than a missing measurement.

        Floored at zero, so it is a lower bound rather than a valuation. A take
        that names no limit is collateralised at the far end of its range, so
        its distance from the reference is negative by construction and it can
        never set a positive price: a firm of nothing but unpriced takers
        therefore reports its collateral as free. That is the honest ex-ante
        answer, because what a cross is worth depends on a view the firm cannot
        see, and it is another reason `Take` tells strategies to name a limit.
        """
        edge = self._edge(name, symbol_view, side, Quote(price, 1))
        self._shadow_now[group] = max(self._shadow_now.get(group, 0.0), edge)

    def _edge(
        self, name: str, symbol_view: SymbolView, side: Side, quote: Quote
    ) -> float:
        """What one unit of collateral spent on this intent is worth, measured.

        Two measurements, and the better one wins. Where a strategy has enough
        of its own fills the answer is its realised and marked P&L per unit of
        collateral committed, which is a fact about this strategy on this
        market. Below that bar the answer is how far the intent rests from the
        reference, per unit of collateral it would consume, which is exact for
        a maker: it is the spread the quote earns if it is filled. For a taker
        crossing the spread that figure is negative by construction, which is
        the honest ex-ante statement, since whether the cross is worth it
        depends on a view the firm cannot see and should not invent. It is used
        to rank, never on its own to refuse.
        """
        measured = self._measured_edge(name)
        if measured is not None:
            return measured
        reference = self._marks.get(symbol_view.symbol)
        if reference is None:
            return 0.0
        spec = symbol_view.instrument.spec
        signed = 1 if side is Side.BUY else -1
        size = max(1, quote.size)
        collateral = spec.collateral_for(signed * size, quote.price)
        if collateral <= 0:
            return 0.0
        return float((reference - quote.price) * signed * Decimal(size) / collateral)

    def _measured_edge(self, name: str) -> float | None:
        """This strategy's P&L per unit of collateral committed, or None.

        Collateral-seconds divided by elapsed seconds is the mean collateral
        the strategy has actually had tied up, so the ratio is a value per unit
        of collateral and is directly comparable with a shadow price. Dividing
        by collateral-seconds instead gives a rate, which is the right
        efficiency metric and the wrong thing to compare against a price.
        """
        ledger = self._ledgers[name]
        if ledger.fills < MIN_FILLS_FOR_EDGE or ledger.collateral_seconds <= 0.0:
            return None
        elapsed = self._elapsed()
        if elapsed <= 0.0:
            return None
        mean_committed = ledger.collateral_seconds / elapsed
        if mean_committed <= 0.0:
            return None
        return float(ledger.pnl(self._marks)) / mean_committed

    # -- packages -----------------------------------------------------------

    def _admit_packages(
        self, packaged: Mapping[str, list[tuple[Take, SymbolView]]]
    ) -> list[Take]:
        """All the legs of a package, or none of them.

        A firm that clips one leg of a hedge and funds the other has
        manufactured the exact position the package existed to avoid, so a
        package that does not fit whole does not go at all. What remains after
        that is genuine execution risk, which is the venue's to impose and this
        firm's to unwind rather than to prevent.

        Projected as one portfolio and not leg by leg, which is the difference
        between funding a package and refusing it. A leg on its own is charged
        its own worst case, and for a vertical the long leg alone is the whole
        premium: measured on ten spreads between 4,600 and 4,650 at midpoint
        prices, the long leg by itself is 27,000 against a group budget of
        5,400, while the package it belongs to can lose 250. Charging the leg
        would refuse a position that is safer than most single orders the same
        firm sends without comment.
        """
        out: list[Take] = []
        for name, legs in packaged.items():
            admitted = [
                (intent, self._take_price(symbol_view, intent), symbol_view)
                for intent, symbol_view in legs
            ]
            fits, limit = self._package_fits(name, admitted)
            if not fits:
                self._refuse(name, "package-unfunded")
                for group in {
                    self._of_symbol.get(v.symbol) for _i, _p, v in admitted
                }:
                    if group is not None and limit:
                        self._note_binding(group, limit)
                continue
            legs_sent: dict[str, int] = {}
            for intent, price, symbol_view in admitted:
                size = intent.size
                self._claim(symbol_view, intent.side, name, size, price)
                signed = size if intent.side is Side.BUY else -size
                legs_sent[intent.symbol] = legs_sent.get(intent.symbol, 0) + signed
                out.append(Take(intent.symbol, intent.side, size, intent.limit))
            if legs_sent:
                before = {
                    symbol: self._ledgers[name].lots.get(symbol, 0) for symbol in legs_sent
                }
                self._packages[name] = (self._wake, legs_sent, before)
        return out

    def _package_fits(
        self, name: str, legs: Sequence[tuple[Take, Decimal, SymbolView]]
    ) -> tuple[bool, str]:
        """Whether every group the package touches funds all of it at once.

        Both sides together, unlike a single order, because a package executes
        as one thing: the buy scenario and the sell scenario of
        ``Venue._survives`` are alternatives for one order and simultaneous for
        a package, and projecting them apart would charge the firm for a
        position it will never hold.
        """
        by_group: dict[str, list[tuple[ContractSpec, int, Decimal]]] = {}
        for intent, price, symbol_view in legs:
            group = self._of_symbol.get(symbol_view.symbol)
            if group is None:
                return (False, "unlisted")
            signed = intent.size if intent.side is Side.BUY else -intent.size
            by_group.setdefault(group, []).append(
                (symbol_view.instrument.spec, signed, price)
            )

        for group, extra in by_group.items():
            base = self._projected(group, None, None)
            holdings = base + extra
            net = worst_case(holdings)
            if net <= worst_case(base):
                continue
            if self._priced_out(name, group):
                return (False, "cost-of-capital")
            net_cap, net_limit, gross_cap = self._caps(group)
            if net > net_cap:
                return (False, net_limit)
            if _gross_of(holdings) > gross_cap:
                return (False, "reserve")
            for spec, signed, _price in extra:
                allowed, limit = self._within_metric_limits(group, spec, signed)
                if not allowed:
                    return (False, limit)
        return (True, "")

    def _resolve_packages(self, view: MarketView) -> list[Take]:
        """Flatten anything that came back with some legs missing, immediately.

        The one unwind this firm performs, and the reason it is not a
        stop-loss: a package's legs are chosen so that the package cannot lose
        whatever the outcome, so a subset of them is a position nobody sized,
        nobody has a view on, and nobody is being paid to hold. Leaving it
        rather than paying the spread to close it is not patience, it is
        carrying an exposure that was never anybody's intent.
        """
        out: list[Take] = []
        for name in sorted(self._packages):
            wake, legs, before = self._packages[name]
            if self._wake - wake < PACKAGE_WAKES or self._outstanding(name, legs):
                continue
            del self._packages[name]
            filled = {
                symbol: self._filled(name, symbol, before[symbol], wanted)
                for symbol, wanted in legs.items()
            }
            done = [s for s, got in filled.items() if got]
            if not done or len(done) == len(legs):
                continue
            for symbol in sorted(done):
                got = filled[symbol]
                if view.get(symbol) is None:
                    continue
                side = Side.SELL if got > 0 else Side.BUY
                symbol_view = view[symbol]
                intent = Take(symbol, side, abs(got))
                out.append(intent)
                self._claim(
                    symbol_view,
                    side,
                    name,
                    abs(got),
                    self._take_price(symbol_view, intent),
                )
            self._refuse(name, "half-legged")
        return out

    def _outstanding(self, name: str, legs: Mapping[str, int]) -> bool:
        """Whether any leg of this package still has an order the firm has not seen.

        A wake counter is not enough on its own. It is right in continuous
        trading, where an IOC is over within a round trip, and wrong during an
        opening call, where a market order rests until the uncross: this
        venue's call runs for ten simulated seconds against a 300ms wake, so a
        package sent into it would be judged thirty-three wakes before its legs
        had a chance to fill, forgotten, and then not flattened when the
        uncross filled one leg of it. The claim is the record of an unresolved
        order, so asking it is asking the right question.
        """
        for symbol, wanted in legs.items():
            side = Side.BUY if wanted > 0 else Side.SELL
            for claim in self._claims.get((symbol, side), ()):
                if claim.name == name and not claim.resting:
                    return True
        return False

    def _filled(self, name: str, symbol: str, before: int, wanted: int) -> int:
        """How much of one leg this strategy actually got, capped at the intent.

        Read off the strategy's own ledger rather than off the account, because
        two strategies can hold opposite positions in one symbol and the
        account shows only their sum. Measured as a change from what the
        strategy held when the package was sent, so a package trader that was
        not flat is not credited with the position it already had.
        """
        moved = self._ledgers[name].lots.get(symbol, 0) - before
        if wanted > 0:
            return max(0, min(wanted, moved))
        return min(0, max(wanted, moved))

    # -- observation and attribution ---------------------------------------

    def _observe(self, view: MarketView) -> None:
        """Reconcile the firm's own book against what the view now reports.

        Called on every entry point rather than on a timer, and idempotent by
        construction: a view showing the same cash and the same positions
        produces no deltas, so being asked to quote twenty symbols off one view
        books one set of fills and not twenty.
        """
        self._catalogue(view)
        for symbol_view in view:
            self._marks[symbol_view.symbol] = _mark(symbol_view)

        if self._first_cash is None:
            self._first_cash = view.cash
            self._cash = view.cash
            self._collateral_seen = view.posted_collateral
            self._lots = {s.symbol: s.position for s in view}
            # A firm that opens onto a book it did not build has no basis for
            # it and no way to recover one: the collateral inversion below is a
            # difference, and there is nothing to difference against. Marked in
            # at the reference, which is the only defensible opening figure,
            # and it is exact for the ordinary case of starting flat.
            self._basis = {
                symbol: Decimal(lots) * self._marks[symbol]
                for symbol, lots in self._lots.items()
                if lots and symbol in self._marks
            }
            self._now = view.now
            self._started = view.now
            self._refresh_levels(view)
            self._budgets = self._allocate()
            self._recompute_collateral()
            self._high_water = self._equity()
            self._sample(view.now)
            return

        if view.now != self._now:
            self._accrue(max(0.0, view.now - self._now))
            self._shadow = dict(self._shadow_now)
            self._shadow_now = {}
            self._now = view.now
            self._refresh_levels(view)

        cash_delta = view.cash - self._cash
        collateral_delta = view.posted_collateral - self._collateral_seen
        moved = {
            s.symbol: s.position - self._lots.get(s.symbol, 0)
            for s in view
            if s.position != self._lots.get(s.symbol, 0)
        }
        if moved:
            self._book(view, moved, cash_delta, collateral_delta)
            for symbol in moved:
                self._lots[symbol] = view[symbol].position
        elif cash_delta:
            self._residue += cash_delta
        self._cash = view.cash
        self._collateral_seen = view.posted_collateral

        self._reap(view)
        self._recompute_collateral()
        equity = self._equity()
        if self._high_water is None or equity > self._high_water:
            self._high_water = equity
        self._sample(view.now)

    def _charged(self, symbol: str) -> Decimal:
        """What the venue has posted against this symbol, from the firm's basis.

        A restatement of ``Account.collateral_for_basis``, clamp and all, so
        that the difference taken in :meth:`_book` is a difference between two
        figures computed the same way. The edge is the bottom of the claim's
        range for a long and the top of it for a short, because that is the end
        the position loses towards.
        """
        lots = self._lots.get(symbol, 0)
        spec = self._specs.get(symbol)
        if not lots or spec is None:
            return Decimal(0)
        low, high = spec.value_bounds
        edge = low if lots > 0 else high
        return max(Decimal(0), self._basis.get(symbol, Decimal(0)) - Decimal(lots) * edge)

    def _recover_basis(
        self, symbol: str, lots_after: int, charged_after: Decimal
    ) -> Decimal:
        """The basis a position must hold to be charged exactly this much.

        ``Account.collateral_for_basis`` is ``max(0, basis - quantity * edge)``,
        which is one equation in one unknown once the quantity is known, and
        the quantity is in the view. So the inverse is arithmetic rather than
        an estimate, which is the standard the rest of this module is held to.

        The clamp is the one place it is not injective, and it is harmless
        there: a requirement of zero says the position cannot lose anything at
        any level, and the basis this returns is the one at which that becomes
        true, so the worst case computed from it is the same zero.
        """
        if lots_after == 0:
            return Decimal(0)
        spec = self._specs.get(symbol)
        if spec is None:
            return Decimal(0)
        low, high = spec.value_bounds
        edge = low if lots_after > 0 else high
        return charged_after + Decimal(lots_after) * edge

    def _book(
        self,
        view: MarketView,
        moved: Mapping[str, int],
        cash_delta: Decimal,
        collateral_delta: Decimal,
    ) -> None:
        """Turn a position change into per-strategy fills at the price they filled.

        The price is recovered rather than assumed, and it has to be recovered
        because a strategy is not told it. ``StrategyAgent`` keeps its shadow
        book with ``Account``, and ``Account.apply_fill`` moves cash by realised
        P&L and fees alone, since a futures position is a collateralised
        commitment rather than a purchase. So an opening fill moves cash by
        exactly nothing, and reading a price out of the cash delta reads zero.

        That was not a small error. Measured on seed 7 over sixty simulated
        seconds while this method still differenced cash: every position in the
        firm's book carried a basis of 0.00 against the venue's 4,919.19 on a
        win rate future, and since ``collateral_for`` charges a long the
        distance from its price down to the bottom of its range, a long booked
        at zero reports a worst case of zero. Every group looked empty, the
        budget never bound on anything the firm already held, and that future's
        netting group finished at 255,797.90 against a budget of 36,000.
        A budget checked before every order is worth nothing if the book it is
        checked against is blank.

        What does carry the price is the collateral. ``posted_collateral`` is
        charged against each position's own basis and moves only when a fill
        moves that basis, so when one symbol has moved, the whole change
        belongs to it and inverts to a basis exactly. The fill's notional then
        falls out of the identity ``Venue._survives`` is built on,
        ``basis_after = basis_before + quantity * price + realised``, with the
        realised part being the cash delta the view just reported.

        More than one symbol moving in one observation cannot be unpicked from
        one collateral number, so each is valued at its mark and the difference
        is carried as a residue. That case is rare by construction rather than
        by luck: ``StrategyAgent`` requotes the instant it is filled, so a firm
        installed as the maker sees one fill at a time.
        """
        symbols = sorted(moved)
        flows: dict[str, Decimal] = {}
        recovered: dict[str, Decimal] = {}
        if len(symbols) == 1:
            symbol = symbols[0]
            lots_after = view[symbol].position
            charged_after = self._charged(symbol) + collateral_delta
            basis_before = self._basis.get(symbol, Decimal(0))
            basis_after = self._recover_basis(symbol, lots_after, charged_after)
            recovered[symbol] = basis_after
            # Spot terms, which is what the ledgers keep: a buy is cash out.
            # The identity gives the notional, and the notional is the price.
            flows[symbol] = -(basis_after - basis_before - cash_delta)
        else:
            for symbol in symbols:
                flows[symbol] = -Decimal(moved[symbol]) * self._marks[symbol]
            self._residue += cash_delta - sum(flows.values(), start=Decimal(0))

        for symbol in symbols:
            quantity = moved[symbol]
            side = Side.BUY if quantity > 0 else Side.SELL
            group = self._of_symbol.get(symbol)
            if group is not None:
                self._group_cash[group] = (
                    self._group_cash.get(group, Decimal(0)) + flows[symbol]
                )
            self._split(symbol, side, quantity, flows[symbol])
            if symbol in recovered:
                self._basis[symbol] = recovered[symbol]
            else:
                self._basis[symbol] = _basis_after(
                    self._lots.get(symbol, 0),
                    self._basis.get(symbol, Decimal(0)),
                    quantity,
                    flows[symbol],
                )

    def _split(self, symbol: str, side: Side, quantity: int, cash: Decimal) -> None:
        """Hand the lots and the cash to whoever asked for them, first in first out.

        The last claimant absorbs any lots beyond what the firm intended and
        the last share absorbs the cash remainder, so the parts sum to the
        whole exactly rather than to within a rounding of it.

        Lots that arrive against no claim at all go to the orphan book, and
        there is one mechanism that produces them: a withdrawal is a Cancel in
        flight, and a Cancel can lose the race to a fill. The claim it belonged
        to survives that race for a wake, so the loser is only the order whose
        cancel and whose fill straddle a whole wake, and no strategy could have
        known. Measured over sixty simulated seconds on seed 7, one fill of
        three lots of a win rate future landed here out of 113, carrying 58.88
        of a P&L of -8,713.38, which is 0.68% of it. Giving those lots to
        whoever asked last would be a guess dressed as attribution, so they are
        reported as their own line instead.
        """
        claims = self._claims.get((symbol, side), [])
        shares: list[list[Any]] = []
        remaining = abs(quantity)
        while remaining > 0 and claims:
            claim = claims[0]
            taken = min(remaining, claim.lots)
            if taken > 0:
                shares.append([claim.name, taken])
                claim.lots -= taken
                remaining -= taken
            if claim.lots <= 0:
                claims.pop(0)
        signed = 1 if quantity > 0 else -1
        if remaining > 0:
            if shares:
                shares[-1][1] += remaining
            else:
                self._orphan.apply(symbol, quantity, cash)
                return

        booked = Decimal(0)
        for index, (name, lots) in enumerate(shares):
            if index == len(shares) - 1:
                share = cash - booked
            else:
                share = cash * Decimal(lots) / Decimal(abs(quantity))
                booked += share
            self._ledgers[name].apply(symbol, signed * lots, share)

    def _catalogue(self, view: MarketView) -> None:
        """Group the listed universe the way the venue does, once per universe.

        A label collision is broken with a digest of the key rather than with a
        counter, so a report's row names do not depend on the order symbols
        happened to arrive in.
        """
        universe = tuple(view.symbols)
        if universe == self._universe:
            return
        self._universe = universe
        self._of_symbol = {}
        self._members = {}
        self._specs = {}
        taken: dict[str, str] = {}
        for symbol in sorted(universe):
            symbol_view = view.get(symbol)
            if symbol_view is None:
                continue
            instrument = symbol_view.instrument
            key = group_key(instrument)
            self._of_symbol[symbol] = key
            self._members.setdefault(key, []).append(symbol)
            self._specs[symbol] = instrument.spec
            if key not in self._label:
                base = group_label(instrument.spec.underlying)
                if taken.get(base, key) != key:
                    base = f"{base}#{digest(instrument.spec.underlying.to_dict())[:6]}"
                taken.setdefault(base, key)
                self._label[key] = base

    # -- capital ------------------------------------------------------------

    def _allocate(self) -> dict[str, Decimal]:
        """Split the deployable capital across the groups, by measured edge.

        A group's weight is its measured P&L per collateral-second, which is
        the only edge figure available here that is exact. Before anything has
        been measured every group weighs the same, and an equal split is the
        honest prior rather than a placeholder: a firm on its first wake has no
        evidence, and inventing a tilt out of instrument counts or notionals
        would be exactly the unmeasured constant this codebase does without.
        :meth:`rebalance` is where the measurement arrives.

        The concentration limit is applied here as well as at order time, so
        the allocation cannot propose a book the order path would have to
        refuse. A capped group's excess is water-filled across the rest, and
        with fewer than ``1 / concentration`` groups the remainder stays
        uncommitted, which is what the limit means rather than a rounding
        failure.
        """
        groups = sorted(self._members)
        if not groups:
            return {}
        deployable = self._capital() * (Decimal(1) - Decimal(str(self.reserve)))
        cap = deployable * Decimal(str(self.concentration))
        weights = {g: self._group_score(g) for g in groups}
        total = sum(weights.values())
        if total <= 0.0:
            weights = {g: 1.0 for g in groups}
            total = float(len(groups))

        budgets = {g: deployable * Decimal(str(weights[g] / total)) for g in groups}
        for _ in range(_ALLOCATION_PASSES):
            spill = Decimal(0)
            free = []
            for g in groups:
                if budgets[g] > cap:
                    spill += budgets[g] - cap
                    budgets[g] = cap
                elif budgets[g] < cap:
                    free.append(g)
            if spill <= 0 or not free:
                break
            headroom = sum((cap - budgets[g] for g in free), start=Decimal(0))
            if headroom <= 0:
                break
            share = min(spill, headroom)
            for g in free:
                budgets[g] += share * (cap - budgets[g]) / headroom
        return budgets

    def _group_score(self, group: str) -> float:
        """A group's measured P&L per collateral-second, floored at zero.

        Cash moved plus what is still held, which is the group's whole P&L. The
        first version scored only the marked value of open inventory, and that
        is the wrong quantity for an allocator: a group that traded profitably
        all session and finished flat scored exactly zero and would have been
        defunded in favour of one that had merely not closed anything.

        Floored because a negative weight is a short position in a budget,
        which is not a thing a budget can express. A group that lost money
        takes none of the next allocation rather than a negative share of it.
        """
        seconds = self._group_seconds.get(group, 0.0)
        if seconds <= 0.0:
            return 0.0
        pnl = self._group_cash.get(group, Decimal(0))
        for symbol in self._members.get(group, ()):
            lots = self._lots.get(symbol, 0)
            mark = self._marks.get(symbol)
            if lots and mark is not None:
                pnl += Decimal(lots) * mark
        return max(0.0, float(pnl) / seconds)

    def rebalance(self) -> dict[str, Decimal]:
        """Re-split the budgets on what the last session measured. Between sessions.

        Between and not during, and the reason is not caution. Rebalancing
        inside a session moves a budget in response to marks that the firm's
        own working orders are part of, so the allocator would be reading its
        own reflection. Between sessions the measurement is over a closed
        interval and the marks belong to somebody else.
        """
        self._budgets = self._allocate()
        return dict(self._budgets)

    def _capital(self) -> Decimal:
        """The firm's capital: what it was given, plus what it has banked.

        Banked, not spent. ``MarketView.cash`` moves by realised P&L and fees
        alone, so the change in it since the first observation is money the
        firm has actually made or lost and not the notional of what it holds.
        That is the right base for the reserve, because it is the same quantity
        the venue's own ``free_cash`` is measured against.

        Derived from the declared budget and that change rather than from the
        cash figure itself, so a firm handed a budget smaller than the account
        it sits in stays inside its mandate instead of quietly spending the
        whole account.
        """
        if self._first_cash is None:
            return self.budget
        return self.budget + (self._cash - self._first_cash)

    def _unrealised(self) -> Decimal:
        """What the open book has made and not yet banked, at the firm's marks."""
        return sum(
            (
                Decimal(lots) * self._marks[symbol] - self._basis.get(symbol, Decimal(0))
                for symbol, lots in self._lots.items()
                if lots and symbol in self._marks
            ),
            start=Decimal(0),
        )

    def _equity(self) -> Decimal:
        """Capital banked plus what is still open, which is the whole of it.

        The marked value of the book is not the second term and was, until the
        adapter's shadow ledger stopped debiting notional on every fill. Under
        a purchase ledger, capital had already been reduced by what the
        positions cost, so adding their value back gave equity; under a
        realised-P&L ledger it has not, so adding their value counts the whole
        position twice and reports a firm holding twenty-five lots of a
        contract marked near 4,900 as being 123,000 richer than it is.
        """
        return self._capital() + self._unrealised()

    def _elapsed(self) -> float:
        return 0.0 if self._started is None else max(0.0, self._now - self._started)

    def _accrue(self, elapsed: float) -> None:
        """Add collateral-seconds, at the commitment that stood over the interval.

        Left endpoint, so the capital charged for an interval is the capital
        that was actually tied up during it rather than the capital the fills
        at its end produced.
        """
        if elapsed <= 0.0:
            return
        for group, committed in self._committed.items():
            self._group_seconds[group] = (
                self._group_seconds.get(group, 0.0) + float(committed) * elapsed
            )
        # Time-weighted, because complementary slackness makes the instant
        # figure zero whenever the budget happens to be slack, and reading the
        # price of collateral off one such instant would say it is free. The
        # mean over the session is what a caller wants when deciding whether a
        # group is worth more capital.
        for group, price in self._shadow.items():
            self._shadow_seconds[group] = (
                self._shadow_seconds.get(group, 0.0) + price * elapsed
            )
        for name, share in self._strategy_collateral().items():
            self._ledgers[name].collateral_seconds += float(share) * elapsed

    def _strategy_collateral(self) -> dict[str, Decimal]:
        """Split each group's committed collateral across the strategies in it.

        By the gross per-contract figure, because the gross figure is a sum and
        is therefore the only one of the two that can be split at all. The
        netted figure is the worst case of the whole group and is not additive
        by construction, which is exactly what makes it worth reporting
        separately rather than dividing up.
        """
        out: dict[str, Decimal] = dict.fromkeys(self.strategies, Decimal(0))
        for group, committed in self._committed.items():
            weights: dict[str, Decimal] = {}
            total = Decimal(0)
            for symbol in self._members.get(group, ()):
                mark = self._marks.get(symbol)
                spec = self._specs.get(symbol)
                if mark is None or spec is None:
                    continue
                for name, ledger in self._ledgers.items():
                    lots = ledger.lots.get(symbol, 0)
                    if not lots:
                        continue
                    gross = spec.collateral_for(lots, mark)
                    weights[name] = weights.get(name, Decimal(0)) + gross
                    total += gross
            if total <= 0:
                continue
            for name, weight in weights.items():
                out[name] += committed * weight / total
        return out

    def _recompute_collateral(self) -> None:
        self._committed = {}
        self._gross = {}
        for group, symbols in self._members.items():
            holdings = self._holdings(symbols)
            if not holdings:
                continue
            gross, net = netting_benefit(holdings)
            self._gross[group] = gross
            self._committed[group] = net

    def _holdings(
        self, symbols: Sequence[str]
    ) -> list[tuple[ContractSpec, int, Decimal]]:
        holdings = []
        for symbol in symbols:
            lots = self._lots.get(symbol, 0)
            spec = self._specs.get(symbol)
            if not lots or spec is None:
                continue
            basis = self._basis.get(symbol)
            price = (
                basis / Decimal(lots)
                if basis is not None
                else self._marks.get(symbol, Decimal(0))
            )
            holdings.append((spec, lots, price))
        return holdings

    def _projected(
        self, group: str, side: Side | None, exclude: str | None
    ) -> list[tuple[ContractSpec, int, Decimal]]:
        """The group's holdings if every order it has out on one side filled.

        One side at a time, mirroring ``Venue._survives``, which evaluates a
        buy scenario and a sell scenario separately rather than assuming both
        happen at once. The symbol being requoted is excluded because its
        working order is about to be replaced by the one under consideration,
        and counting both would charge an amendment as though it were exposure
        on top of what it supersedes.

        ``side`` of ``None`` asks the other question, and only a package asks
        it: what the group holds if *everything* the firm has out fills, which
        is what executing both legs of one package means.
        """
        symbols = self._members.get(group, ())
        holdings = self._holdings(symbols)
        for (symbol, order_side), queue in self._claims.items():
            if self._of_symbol.get(symbol) != group:
                continue
            if side is not None and order_side is not side:
                continue
            signed = 1 if order_side is Side.BUY else -1
            spec = self._specs.get(symbol)
            if spec is None:
                continue
            for claim in queue:
                if claim.resting and symbol == exclude:
                    continue
                holdings.append((spec, signed * claim.lots, claim.price))
        return holdings

    def _deltas(self) -> dict[str, float]:
        """``dV/dtheta`` per group, from the positions the firm actually holds."""
        out: dict[str, float] = {}
        for group, symbols in self._members.items():
            level = self._levels.get(group, (0.0, False))[0]
            total = 0.0
            for symbol in symbols:
                lots = self._lots.get(symbol, 0)
                spec = self._specs.get(symbol)
                if lots and spec is not None:
                    total += metric_sensitivity(spec, lots, level)
            if total:
                out[group] = total
        return out

    def _refresh_levels(self, view: MarketView) -> None:
        """The level the market is pricing each group's metric at, once a wake.

        Inverted from the group's own linear contracts, which is reading a
        price rather than fitting a model: a linear claim is ``a * theta + b``
        with both constants written in the spec, so one quote is one equation
        in one unknown. Averaged across every invertible contract in the group,
        because they are all opinions about the same scalar. A group with no
        linear contract falls back to the midpoint of the metric's declared
        range and says so, because a delta reported off an invented level would
        look like a measurement.
        """
        levels: dict[str, tuple[float, bool]] = {}
        for group, symbols in self._members.items():
            found: list[float] = []
            bounds = (0.0, 1.0)
            for symbol in symbols:
                spec = self._specs.get(symbol)
                symbol_view = view.get(symbol)
                if spec is None or symbol_view is None:
                    continue
                bounds = spec.underlying.bounds()
                affine = _affine_claim(spec)
                reference = _priced(_mark(symbol_view), symbol_view.bounds)
                if affine is None or reference is None:
                    continue
                a, b = affine
                found.append((float(reference) - b) / a)
            low, high = bounds
            if found:
                level = sum(found) / len(found)
                levels[group] = (min(high, max(low, level)), True)
            else:
                levels[group] = ((low + high) / 2.0, False)
        self._levels = levels

    # -- bookkeeping helpers ------------------------------------------------

    def _take_price(self, symbol_view: SymbolView, intent: Take) -> Decimal:
        """What to collateralise a take at, the way the venue would.

        The named limit where there is one, because a limit order cannot pay
        more than it says. An order that names no price is charged the far end
        of the contract's range, which is what the venue charges a market order
        it cannot see liquidity for, and it is not conservatism: a market order
        sweeps, and the touch is one price level of protection.

        Using the touch instead was measured on seed 7 over sixty simulated
        seconds and it defeated the budget entirely. A packaged vertical
        projected at a best ask of 0.00 for its long leg and 0.00 for its
        short is arithmetically riskless, since a lower strike is worth at
        least as much as a higher one at every level, so the package was
        admitted twenty-four times as risk-reducing. The fills came back at
        5,400.00 and 3.00, the two ends of the range, leaving 54 lots whose
        real worst case was 291,438.00 against a group budget of 36,000.00 and
        a strategy 322,420.50 down on a 400,000 account. The venue itself
        never had this hole: it walks the book for what a market order could
        actually pay, and charges the range end when the walk finds nothing.
        A strategy cannot walk the book, so it takes the range end always,
        which is why `Take` tells strategies that naming a limit is strictly
        safer.
        """
        if intent.limit is not None:
            return intent.limit
        low, high = symbol_view.bounds
        return high if intent.side is Side.BUY else low

    def _claim(
        self,
        symbol_view: SymbolView,
        side: Side,
        name: str,
        lots: int,
        price: Decimal,
        resting: bool = False,
    ) -> None:
        """Record an order the firm has out: who wants it and what it costs.

        A quote replaces the side's resting claim, because it replaces the
        side's working order. Wiping the whole queue instead was measured to
        lose a take: the opening uncross booked two lots of a win rate future
        at 9,968.50 against an empty queue, because the maker had requoted
        between the take going out and the auction clearing, and 1,276.00 of
        P&L went to the unattributed bucket rather than to the strategy that
        earned it.

        A take goes in front of the resting claim, because a take crosses now
        and a quote waits to be hit.
        """
        queue = self._claims.setdefault((symbol_view.symbol, side), [])
        claim = _Claim(
            name, lots, price, resting, _priced(symbol_view.last, symbol_view.bounds)
        )
        if resting:
            # A new quote supersedes the old one atomically at the venue, since
            # `TradingAgent.post` amends rather than cancelling and reposting,
            # so the claim it replaces is genuinely gone rather than in flight.
            queue[:] = [c for c in queue if not c.resting]
            queue.append(claim)
            return
        queue.insert(sum(1 for c in queue if not c.resting), claim)

    def _reap(self, view: MarketView) -> None:
        """Drop takes the market has traded past and left behind for a wake.

        A take cannot be aged out on a timer.
        ``StrategyAgent._apply_take`` sends every take as IOC, which in
        continuous trading is over within one round trip, so a one-wake
        lifetime looks right. It is wrong wherever a book stops trading. A
        market order rests through an opening call, and this venue's runs for
        ten simulated seconds against a 300ms wake, which is thirty-three wakes
        of resting; measured with a bare timer on seed 7, the uncross filled
        four lots of a win rate future against a projection of two and that
        future's netting group finished committing 56,013.50 against a budget
        of 36,000. It also
        rests through a circuit-breaker pause, which is the same shape and is
        not confined to the start of the session: measured on seed 7 with the
        timer conditioned on the symbol having ever printed, five fills came
        back at 20.0s, 40.0s and 52.0s with no claim left to book them against,
        two lots and then four of one call at 5,400.00 and six of the next
        strike up the same chain at 0.00, and 97,200.00 of P&L belonged to no
        strategy.

        Nor can it be aged out on the print alone. The uncross *is* the
        symbol's first print after a pause and it arrives in the same instant
        as the fills it produced, so reaping on the print dropped the claims
        for lots that were still on their way and the firm projected against a
        book it no longer knew it had: on seed 7 the netting group behind that
        chain finished at 39,143.69 against a budget of 36,000.

        So the clock starts at the *first print that is not ours*, and runs for
        one wake. A symbol that has stopped trading never starts it, whatever
        stopped it and however long for.
        """
        for key in list(self._claims):
            symbol, _side = key
            symbol_view = view.get(symbol)
            if symbol_view is None:
                continue
            last = _priced(symbol_view.last, symbol_view.bounds)
            kept: list[_Claim] = []
            for claim in self._claims[key]:
                if claim.resting:
                    kept.append(claim)
                    continue
                if claim.dead_from is None and last is not None and last != claim.last:
                    claim.dead_from = self._wake
                if (
                    claim.dead_from is None
                    or self._wake - claim.dead_from < PACKAGE_WAKES
                ):
                    kept.append(claim)
            if kept:
                self._claims[key] = kept
            else:
                del self._claims[key]

    def _refuse(self, name: str, reason: str) -> None:
        ledger = self._ledgers[name]
        ledger.refused[reason] = ledger.refused.get(reason, 0) + 1

    def _note_binding(self, group: str, limit: str) -> None:
        if not limit:
            return
        if self._first_binding is None:
            self._first_binding = (limit, self._now)
        self._binding[limit] = self._binding.get(limit, 0) + 1
        per_group = self._group_binding.setdefault(group, {})
        per_group[limit] = per_group.get(limit, 0) + 1

    def _sample(self, now: float) -> None:
        equity = self._equity()
        if self._equity_curve and self._equity_curve[-1][0] == now:
            self._equity_curve[-1] = (now, equity)
            return
        self._equity_curve.append((now, equity))
        if len(self._equity_curve) > self.equity_curve_limit:
            del self._equity_curve[: len(self._equity_curve) // 2]

    # -- reporting ----------------------------------------------------------

    def report(self) -> FirmReport:
        """Everything the firm knows about itself, as of the last view it saw.

        Assembled rather than accumulated, so a caller reading it twice reads
        the same numbers, and so nothing in the reporting path can move a
        figure the trading path depends on.
        """
        elapsed = self._elapsed()
        groups: dict[str, GroupReport] = {}
        for key in sorted(self._members):
            label = self._label.get(key, key)
            level, implied = self._levels.get(key, (0.0, False))
            delta = 0.0
            for symbol in self._members[key]:
                spec = self._specs.get(symbol)
                lots = self._lots.get(symbol, 0)
                if spec is not None and lots:
                    delta += metric_sensitivity(spec, lots, level)
            groups[label] = GroupReport(
                key=key,
                label=label,
                members=len(self._members[key]),
                budget=self._budgets.get(key, Decimal(0)),
                committed=self._committed.get(key, Decimal(0)),
                gross=self._gross.get(key, Decimal(0)),
                net=self._committed.get(key, Decimal(0)),
                shadow_price=self._shadow.get(key, 0.0),
                shadow_price_mean=(
                    self._shadow_seconds.get(key, 0.0) / elapsed if elapsed else 0.0
                ),
                level=level,
                implied=implied,
                delta=delta,
                collateral_seconds=self._group_seconds.get(key, 0.0),
                binding=dict(self._group_binding.get(key, {})),
            )

        strategies: dict[str, StrategyReport] = {}
        for name, ledger in self._ledgers.items():
            pnl = ledger.pnl(self._marks)
            unrealised = ledger.mark_value(self._marks) - ledger.carried_basis()
            strategies[name] = StrategyReport(
                name=name,
                pnl=pnl,
                # Realised as the residual, so the two halves add to the total
                # exactly. A basis carries a division and a division carries a
                # rounding; putting it in the split rather than in the sum
                # keeps the identity the report is checked against intact.
                realised=pnl - unrealised,
                unrealised=unrealised,
                fills=ledger.fills,
                lots=sum(abs(v) for v in ledger.lots.values()),
                collateral_seconds=ledger.collateral_seconds,
                edge_per_collateral=self._measured_edge(name),
                refused=dict(ledger.refused),
            )

        high_water = self._high_water if self._high_water is not None else Decimal(0)
        return FirmReport(
            now=self._now,
            capital=self.budget,
            equity=self._equity(),
            high_water_mark=high_water,
            floor=high_water * (Decimal(1) - Decimal(str(self.max_drawdown))),
            gross=sum(self._gross.values(), start=Decimal(0)),
            net=sum(self._committed.values(), start=Decimal(0)),
            unattributed=self._residue + self._orphan.pnl(self._marks),
            unattributed_fills=self._orphan.fills,
            groups=groups,
            strategies=strategies,
            equity_curve=tuple(self._equity_curve),
            binding=dict(self._binding),
            first_binding=self._first_binding,
        )

    # -- introspection a caller and the tests both want ---------------------

    def budgets(self) -> dict[str, Decimal]:
        """The collateral budget per group, keyed by the venue's own key."""
        return dict(self._budgets)

    def committed(self) -> dict[str, Decimal]:
        """Netted collateral currently committed, per group."""
        return dict(self._committed)

    def basis(self) -> dict[str, Decimal]:
        """The cost basis the firm believes it holds, per symbol.

        Recovered from posted collateral rather than told to the firm, so it is
        worth being able to check against the account it is meant to mirror.
        Everything the budget is enforced on is computed from this, and when it
        was blank the budget was enforced on nothing.
        """
        return {s: b for s, b in self._basis.items() if self._lots.get(s)}

    def shadow_prices(self) -> dict[str, float]:
        """The last completed wake's shadow price of collateral, per group."""
        return dict(self._shadow)

    def label_of(self, group: str) -> str:
        return self._label.get(group, group)

    def group_of(self, symbol: str) -> str | None:
        return self._of_symbol.get(symbol)


def _priced(
    price: Decimal | None, bounds: tuple[Decimal, Decimal]
) -> Decimal | None:
    """``price``, or None where it is not a price this contract could pay.

    The bounds are the contract's own, public and already the thing collateral
    is charged against, so this is applying the contract rather than clamping
    an inconvenience away.

    It is also a live defect guard, and the defect is in the venue rather than
    here. ``OrderBook.best_priced`` and ``BookSnapshot.best_bid`` both drop the
    sentinel price that market-on-open interest rests at, and
    ``VenueAgent.top_of_book`` calls ``best_price``, which does not. Measured
    on seed 7 over twenty simulated seconds, a plain strategy's view carried
    9,920 sentinel-valued touches, 5,352 bids and 4,568 asks, across 34 of the
    47 listed symbols at 84 distinct wake timestamps from 0.9346s to 19.8666s,
    every one of them exactly 1,152,921,504,606,846,976, which is ``1 << 62``
    ticks on a 0.25 grid. Marking a position at that number took this firm's
    reported equity to 1.7e18 on a 400,000 account. A price the contract cannot
    settle at is not a price, whatever put it in the field.

    Fixed at the source since this was written: ``top_of_book`` now calls
    ``best_priced``, and measured over the same twenty seconds the count is
    zero. The guard stays, because it is a bounds test rather than a sentinel
    test and it costs one comparison. The two figures above are one number in
    two units and it is worth saying which is which, because they differ by a
    factor of four and both appear around the repository: the raw sentinel is a
    tick index of ``1 << 62``, or 4,611,686,018,427,387,904, and the price it
    becomes on a 0.25 grid is ``1 << 60``.
    """
    if price is None:
        return None
    low, high = bounds
    return price if low <= price <= high else None


def _mark(symbol_view: SymbolView) -> Decimal:
    """What a position is worth, from the strategy's own stale view.

    The mid where the book is two-sided, the last print otherwise, and the
    midpoint of the contract's range where it has never traded. The same ladder
    ``SymbolView.reference`` uses, restated here so that the firm's marks and
    its P&L cannot drift apart if that property ever changes meaning, and with
    every rung filtered through :func:`_priced` first.
    """
    bounds = symbol_view.bounds
    bid = _priced(symbol_view.best_bid, bounds)
    ask = _priced(symbol_view.best_ask, bounds)
    if bid is not None and ask is not None:
        return (bid + ask) / 2
    last = _priced(symbol_view.last, bounds)
    if last is not None:
        return last
    return (bounds[0] + bounds[1]) / 2
