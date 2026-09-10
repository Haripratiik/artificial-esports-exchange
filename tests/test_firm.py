"""A firm, rather than a pile of bots: budgets, limits and attribution.

Every limit here is tested in the state where it matters. That is not a style
preference, it is the fourth bug class in `CONTRIBUTING.md`: the kill switch was rate
limited and a participant worth killing is at its message cap by definition, so
the control was correct and inert. A concentration limit checked on a firm with
a full account and no losses is the same shape of test, and it passes on a
control that does nothing. So the concentration and drawdown tests below first
put the firm through a loss, and then ask.

The strategies are trivial and defined here rather than imported. That is
deliberate too: `arena.strategies.making` and `arena.strategies.taking` are the
library, and a test of the risk machinery that depended on them would be
measuring two things at once and would fail for either reason.
"""

from __future__ import annotations

from decimal import Decimal as D

import pytest
from dashboard.build_market import build, instruments

from arena.agents.strategy_agent import StrategyAgent
from arena.contracts.payoff import Binary, Call, Linear, Put
from arena.exchange.types import AgentId, Side
from arena.market.fees import FREE
from arena.market.live import VENUE_ID
from arena.portfolio.account import Account
from arena.portfolio.money import from_money, to_money
from arena.portfolio.netting import netting_benefit, worst_case
from arena.sim.time import millis, seconds
from arena.strategies.base import MarketView, Quote, SymbolView, Take, TwoSided
from arena.strategies.firm import (
    Firm,
    group_key,
    group_label,
    metric_sensitivity,
)


@pytest.fixture(scope="module")
def listed():
    return {i.symbol: i for i in instruments()}


# --------------------------------------------------------------------------
# Trivial strategies, and a view to run them against
# --------------------------------------------------------------------------


class Quoter:
    """Rests a fixed size a fixed distance off the reference, on both sides."""

    def __init__(
        self,
        symbols,
        half_spread=D(20),
        size=1_000,
        sides=("bid", "ask"),
        post_only=False,
    ):
        self._symbols = tuple(symbols)
        self.half_spread = D(half_spread)
        self.size = size
        self.sides = sides
        self.post_only = post_only

    def symbols(self, view):
        return self._symbols

    def quote(self, view, symbol):
        symbol_view = view[symbol]
        low, high = symbol_view.bounds
        reference = symbol_view.reference
        bid = Quote(
            max(low, reference - self.half_spread), self.size, self.post_only
        )
        ask = Quote(
            min(high, reference + self.half_spread), self.size, self.post_only
        )
        return TwoSided(
            bid=bid if "bid" in self.sides else None,
            ask=ask if "ask" in self.sides else None,
        )


class Lifter:
    """Wants the same take every wake, at the touch."""

    def __init__(self, symbol, size, side=Side.BUY, limit=None):
        self.symbol = symbol
        self.size = size
        self.side = side
        self.limit = limit

    def orders(self, view):
        if view.get(self.symbol) is None:
            return ()
        return (Take(self.symbol, self.side, self.size, self.limit),)


class Spread:
    """A packaged taker: both legs of a vertical, or neither.

    The legs offset, so the package is exactly the shape netting exists for and
    exactly the shape a half-legged fill ruins.

    Both legs name a limit, because `Take` says naming one is strictly safer
    and a package trader is precisely who that is aimed at: an order with no
    limit is collateralised against the far end of the contract's range, so an
    unpriced vertical is charged as though it bought at the top and sold at the
    bottom, which on this catalogue is 61,200 of collateral for six lots that
    can lose 1,200.
    """

    packaged = True

    def __init__(self, long_symbol, short_symbol, size, once=False):
        self.long_symbol = long_symbol
        self.short_symbol = short_symbol
        self.size = size
        self.once = once
        self.sent = 0

    def orders(self, view):
        if self.once and self.sent:
            return ()
        long_view = view.get(self.long_symbol)
        short_view = view.get(self.short_symbol)
        if long_view is None or short_view is None:
            return ()
        buy = long_view.best_ask or long_view.reference
        sell = short_view.best_bid or short_view.reference
        if buy is None or sell is None:
            return ()
        low, high = long_view.bounds
        buy = min(high, max(low, buy))
        low, high = short_view.bounds
        sell = min(high, max(low, sell))
        self.sent += 1
        return (
            Take(self.long_symbol, Side.BUY, self.size, buy),
            Take(self.short_symbol, Side.SELL, self.size, sell),
        )


class Book:
    """The agent's shadow book, kept with the venue's own `Account`.

    Deliberately not a second implementation of it. `StrategyAgent` keeps
    exactly this, so a fixture that worked cash or collateral out its own way
    would be testing the firm against a venue that does not exist, and the
    difference is not cosmetic: `Account.apply_fill` moves cash by realised
    P&L and fees alone, because a futures position is a collateralised
    commitment rather than a purchase, so an opening fill moves cash by
    nothing at all. The firm reads a position's basis back out of the posted
    collateral, which is the only public figure that carries it, so a fixture
    that reported the old purchase-ledger cash and no collateral would hand it
    a blank book and every budget would look free.
    """

    def __init__(self, listed, cash=D(1_000_000)):
        self.listed = listed
        self.account = Account("firm", to_money(cash))

    def fill(self, symbol, quantity, price):
        instrument = self.listed[symbol]
        self.account.apply_fill(
            symbol, quantity, to_money(D(price)), instrument.bounds_in_minor
        )
        return self

    @property
    def cash(self):
        return from_money(self.account.cash)

    @property
    def posted(self):
        return from_money(self.account.posted_collateral)

    @property
    def positions(self):
        return {s: p.quantity for s, p in self.account.positions.items() if p.quantity}


def _view(
    listed, *, now=0.0, cash=D(1_000_000), prices=None, positions=None, book=None
):
    """A MarketView over the whole live catalogue, priced and positioned to order.

    Prices default to the midpoint of each contract's own range, so a symbol
    the test does not care about still has a reference and still consumes
    collateral in the same arithmetic as one it does.

    Pass a :class:`Book` for anything that drives a sequence of fills. Cash,
    positions and posted collateral then all come from one `Account` and agree
    with each other, which the three of them given separately do not: the
    firm recovers a fill price by differencing the collateral against the cash,
    and two figures that were never produced by the same book do not difference
    into a price.
    """
    prices = prices or {}
    posted = D(0)
    if book is not None:
        cash, positions, posted = book.cash, book.positions, book.posted
    positions = positions or {}
    by_symbol = {}
    for symbol, instrument in listed.items():
        low, high = instrument.value_bounds
        price = D(prices.get(symbol, (low + high) / 2))
        by_symbol[symbol] = SymbolView(
            symbol=symbol,
            instrument=instrument,
            best_bid=price,
            best_ask=price,
            last=price,
            position=positions.get(symbol, 0),
            working_bid=None,
            working_ask=None,
            seconds_to_expiry=None,
        )
    mark_value = sum(
        (D(v.position) * v.last for v in by_symbol.values() if v.position), start=D(0)
    )
    return MarketView(
        now=now,
        symbols=tuple(listed),
        cash=cash,
        free_cash=cash - posted,
        posted_collateral=posted,
        equity=cash + mark_value,
        _by_symbol=by_symbol,
    )


def _wake(firm, view, symbols=()):
    """One wake of the adapter's own shape: symbols, then quotes, then orders."""
    firm.symbols(view)
    quotes = {symbol: firm.quote(view, symbol) for symbol in symbols}
    return quotes, firm.orders(view)


# --------------------------------------------------------------------------
# Grouping: the unit a budget is denominated in
# --------------------------------------------------------------------------


def test_the_catalogue_is_twenty_two_netting_groups(listed):
    """50 instruments, 22 groups, keyed the way the venue keys them.

    The whole design rests on this: a budget per instrument would be 50
    budgets that do not correspond to anything capital is released within, and
    a budget for the account would be one number that cannot be attributed.

    Re-measured on the circuit listing, the group sizes are 7 on QUILL's
    individual win rate, 6 on BASTION's, 6 on EMBER's team win rate, 5 on
    RIFT's match volume, 4 on VANTA's individual win rate, five pairs and
    twelve singles. The previous listing put 20 of its 47 contracts in one
    group and this one puts 7 of 50, which is the same measurement the listing
    itself is argued from.
    """
    groups: dict[str, list[str]] = {}
    for symbol, instrument in listed.items():
        groups.setdefault(group_key(instrument), []).append(symbol)
    assert len(listed) == 50
    assert len(groups) == 22
    assert sorted((len(v) for v in groups.values()), reverse=True) == [
        7, 6, 6, 5, 4, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]
    labels = {group_label(listed[v[0]].spec.underlying) for v in groups.values()}
    assert "win_rate:EMBER" in labels
    assert {"difference", "basket"} <= labels


def test_a_group_is_the_set_worst_case_will_accept(listed):
    """Grouping any other way is refused by the arithmetic, not merely wrong.

    Two contracts from one group net; one from each of two groups raises,
    because netting them would need a correlation. So the grouping is not a
    convention this module chose, it is the only one `worst_case` answers for.
    """
    ember = listed["EMBER_OBJECTIVE_WR"].spec
    rift = listed["RIFT_OBJECTIVE_WR"].spec
    assert worst_case([(ember, 4, D(5038)), (ember, -4, D(5038))]) == 0
    with pytest.raises(ValueError, match="not written on the same underlying"):
        worst_case([(ember, 4, D(5038)), (rift, -4, D(5038))])


def test_metric_sensitivity_is_read_off_the_payoff(listed):
    """Linear gives the scale, a call gives it only above the strike, a binary none.

    Exact rather than bumped. A difference quotient across the strike of
    EMBER_OBJECTIVE_C5000 would report a slope of 10,000 at a level where the
    option is worth nothing and moves not at all.

    The levels straddle 0.50, which is where this chain is struck: at 0.51 the
    call is in the money and the put is not, at 0.49 the reverse, and at 0.50
    itself the kink is assigned to the flat branch.
    """
    future = listed["EMBER_OBJECTIVE_WR"].spec
    call = listed["EMBER_OBJECTIVE_C5000"].spec
    put = listed["EMBER_OBJECTIVE_P5000"].spec
    binary = listed["EMBER_OBJECTIVE_GT500"].spec
    assert isinstance(future.payoff, Linear)
    assert isinstance(call.payoff, Call)
    assert isinstance(put.payoff, Put)
    assert isinstance(binary.payoff, Binary)

    assert metric_sensitivity(future, 3, 0.40) == 30_000.0
    assert metric_sensitivity(future, 3, 0.55) == 30_000.0
    assert metric_sensitivity(call, 3, 0.51) == 30_000.0
    assert metric_sensitivity(call, 3, 0.49) == 0.0
    assert metric_sensitivity(put, 3, 0.49) == -30_000.0
    assert metric_sensitivity(put, 3, 0.51) == 0.0
    assert metric_sensitivity(binary, 3, 0.51) == 0.0


# --------------------------------------------------------------------------
# The budget
# --------------------------------------------------------------------------


def test_a_group_budget_is_never_exceeded(listed):
    """Checked against `worst_case` itself, and tight to the lot.

    The strategy asks for a thousand lots. What comes back is the largest size
    whose worst case fits the group's budget, and one lot more does not, so the
    clip is the budget's own arithmetic rather than an approximation that
    happens to be conservative.
    """
    firm = Firm(budget=D(1_000_000), strategies={"mm": Quoter(["EMBER_OBJECTIVE_WR"])})
    view = _view(listed)
    quotes, _ = _wake(firm, view, ["EMBER_OBJECTIVE_WR"])
    bid = quotes["EMBER_OBJECTIVE_WR"].bid
    assert bid is not None and 0 < bid.size < 1_000

    spec = listed["EMBER_OBJECTIVE_WR"].spec
    budget = firm.budgets()[firm.group_of("EMBER_OBJECTIVE_WR")]
    assert worst_case([(spec, bid.size, bid.price)]) <= budget
    assert worst_case([(spec, bid.size + 1, bid.price)]) > budget


def test_the_budgets_sum_inside_the_reserve(listed):
    """`sum(budgets) <= cash * (1 - reserve)`, to inside one minor unit.

    The equality used to be exact and this listing is what stopped it being so.
    `Firm._allocate` divides the deployable capital by the group count through
    a float before it reaches Decimal, and the previous catalogue had 10
    netting groups, where a tenth is exact in binary. This one has 22, where a
    twenty-second is not: measured on 750,000 of deployable capital across 22
    groups the sum comes back at 750,000.000000000024, which is 2.4e-11 above
    the reserve line.

    Money here is integer minor units at a scale of a million, so that excess
    is 2.4e-5 of one minor unit and cannot be posted, collected or conserved;
    `to_money` refuses it outright rather than rounding. It is pinned two-sided
    rather than waved through, so if the allocator ever drifts by an amount
    that could actually move, this says so.
    """
    firm = Firm(
        budget=D(1_000_000),
        strategies={"mm": Quoter(["EMBER_OBJECTIVE_WR"])},
        reserve=0.25,
    )
    firm.symbols(_view(listed))
    budgets = firm.budgets()
    assert len(budgets) == 22
    excess = sum(budgets.values()) - D(1_000_000) * D("0.75")
    assert D(0) <= excess < D("0.000001"), excess
    assert sum(budgets.values()) == pytest.approx(D(750_000), abs=1)


def test_two_strategies_in_one_group_compete_for_one_budget(listed):
    """And two in different groups do not, which is the point of grouping at all.

    Both takers ask for five hundred lots. Inside one group the two admitted
    sizes together have to fit one budget, so the second is clipped by what the
    first took; across two groups each gets its own budget and neither sees the
    other. Re-measured on a budget of 1,000,000 with a reserve of 0.10, now
    split across 22 groups rather than 10: the paired EMBER legs come back at 4
    and 0 lots against 4 and 4 when the second leg is moved to RIFT.
    """
    together = Firm(
        budget=D(1_000_000),
        strategies={
            "a": Lifter("EMBER_OBJECTIVE_WR", 500),
            "b": Lifter("EMBER_OBJECTIVE_C5000", 500),
        },
    )
    view = _view(listed)
    _, takes = _wake(together, view)
    shared = {t.symbol: t.size for t in takes}

    apart = Firm(
        budget=D(1_000_000),
        strategies={
            "a": Lifter("EMBER_OBJECTIVE_WR", 500),
            "b": Lifter("RIFT_OBJECTIVE_WR", 500),
        },
    )
    _, takes = _wake(apart, _view(listed))
    split = {t.symbol: t.size for t in takes}

    assert split["EMBER_OBJECTIVE_WR"] == split["RIFT_OBJECTIVE_WR"] > 0
    assert shared.get("EMBER_OBJECTIVE_WR", 0) == split["EMBER_OBJECTIVE_WR"]
    assert shared.get("EMBER_OBJECTIVE_C5000", 0) < split["RIFT_OBJECTIVE_WR"]

    ember = together.group_of("EMBER_OBJECTIVE_WR")
    assert together.group_of("EMBER_OBJECTIVE_C5000") == ember
    assert apart.group_of("RIFT_OBJECTIVE_WR") != ember
    holdings = [
        (listed[symbol].spec, size, listed[symbol].value_bounds[1] / 2)
        for symbol, size in shared.items()
    ]
    assert worst_case(holdings) <= together.budgets()[ember]


def test_clipping_a_quote_changes_its_size_and_nothing_else(listed):
    """The firm resizes a maker's quote. It does not get to restate it.

    `post_only` is the case that makes the point: a maker that says it will
    never cross has said something about how it wants to trade, not how much,
    and a risk manager that rebuilds the quote from a price and a size drops it
    silently. The clip is a `replace`, so every field the strategy set survives
    and so will the next one anybody adds.
    """
    firm = Firm(
        budget=D(1_000_000),
        strategies={"mm": Quoter(["EMBER_OBJECTIVE_WR"], post_only=True)},
    )
    quotes, _ = _wake(firm, _view(listed), ["EMBER_OBJECTIVE_WR"])
    bid = quotes["EMBER_OBJECTIVE_WR"].bid
    assert bid is not None
    assert bid.size < 1_000, "the budget has to bite, or this proves nothing"
    assert bid.post_only is True
    assert quotes["EMBER_OBJECTIVE_WR"].ask.post_only is True


def test_an_order_that_lowers_the_worst_case_is_never_refused(listed):
    """Being over a limit has to leave the way back under it open.

    A firm already at its budget still lets a strategy sell what it is long,
    because that order reduces the group's worst case rather than adding to it.
    Without this the firm withdrew the maker's quote the moment its inventory
    priced it out, which removed the only mechanism it had for working out of
    the inventory: measured on seed 7 over twenty simulated seconds, 122 of 264
    side decisions refused, and the position never came down.
    """
    long_lots = 400
    firm = Firm(
        budget=D(20_000),
        strategies={"out": Lifter("EMBER_OBJECTIVE_WR", long_lots, Side.SELL)},
    )
    view = _view(listed, positions={"EMBER_OBJECTIVE_WR": long_lots})
    _, takes = _wake(firm, view)
    assert [(t.symbol, t.side, t.size) for t in takes] == [
        ("EMBER_OBJECTIVE_WR", Side.SELL, long_lots)
    ]


# --------------------------------------------------------------------------
# The limits, each in the state where it binds
# --------------------------------------------------------------------------


def _drawn_down(listed, firm, symbol, lots, opened_at, marked_at, close=0):
    """Run the firm into a loss: buy at one price, then mark at a lower one.

    Two observations rather than one, because a high-water mark that has never
    seen a higher equity is not a drawdown, and the whole question is what the
    firm does *after* it has been ahead.

    ``close`` sells part of the position back at the marked price, which is the
    difference between a loss the firm has taken and one it is merely carrying.
    It matters to exactly one of the limits below: cash moves on realised P&L
    alone, so an unrealised loss leaves capital untouched and only the limits
    written against *equity* can see it.
    """
    book = Book(listed)
    _wake(firm, _view(listed, now=0.0, book=book), [symbol])
    book.fill(symbol, lots, opened_at)
    _wake(firm, _view(listed, now=1.0, book=book, prices={symbol: opened_at}), [symbol])
    if close:
        book.fill(symbol, -close, marked_at)
    return _view(listed, now=2.0, book=book, prices={symbol: marked_at})


def test_the_concentration_limit_binds_once_capital_has_fallen(listed):
    """Budgets are set at allocation time; concentration is measured live.

    That difference is the limit's whole job. A budget struck when the firm had
    a million is a stale number after the firm has taken a loss, and the
    concentration limit is the one that still knows: it is a share of the
    capital as it stands, so it falls with the capital and overtakes the stale
    budget. Asked on a calm firm it never fires, which is exactly the shape of
    control this repository has shipped inert before.

    The loss has to be a realised one, and that is not a detail of the fixture.
    Cash moves on realised P&L alone, so a position merely marked down leaves
    capital exactly where it was; what an unrealised loss moves is equity,
    which is what the drawdown limit below is written against. The two limits
    read different numbers on purpose, and this is the state that tells them
    apart.

    A group IS one scalar, so everything in it can be wrong at once. That is
    what justifies the limit, and it is a fact rather than a correlation
    estimate.
    """
    firm = Firm(
        budget=D(1_000_000),
        strategies={"mm": Quoter(["EMBER_OBJECTIVE_WR"])},
        concentration=0.03,
        # Wide enough that the cushion above the floor is not the tightest cap
        # here. Two limits that both bind prove nothing about either.
        max_drawdown=0.40,
        cost_of_capital=False,
    )
    calm = _view(listed)
    _wake(firm, calm, ["EMBER_OBJECTIVE_WR"])
    assert "concentration" not in firm.report().binding

    marked = _drawn_down(listed, firm, "EMBER_OBJECTIVE_WR", 60, 5_000, 500, close=30)
    _wake(firm, marked, ["EMBER_OBJECTIVE_WR"])
    label = firm.label_of(firm.group_of("EMBER_OBJECTIVE_WR"))
    binding = firm.report().groups[label].binding
    assert binding.get("concentration", 0) > 0, binding


def test_the_drawdown_limit_binds_below_the_high_water_mark(listed):
    """CPPI: committed collateral is a multiple of the cushion above the floor.

    Grossman and Zhou's constant-proportion rule, expressed in the same unit
    every other limit here is in, which is what lets the four be compared with
    a `min` rather than reconciled by hand. Tested after the loss that creates
    the cushion, because with equity at its high-water mark the cushion is the
    whole drawdown allowance and the limit cannot bind.
    """
    firm = Firm(
        budget=D(1_000_000),
        strategies={"mm": Quoter(["EMBER_OBJECTIVE_WR"])},
        max_drawdown=0.10,
        drawdown_multiple=1.0,
        cost_of_capital=False,
    )
    calm = _view(listed)
    _wake(firm, calm, ["EMBER_OBJECTIVE_WR"])
    assert "drawdown" not in firm.report().binding

    marked = _drawn_down(listed, firm, "EMBER_OBJECTIVE_WR", 60, 5_000, 500)
    _wake(firm, marked, ["EMBER_OBJECTIVE_WR"])
    report = firm.report()
    assert report.equity < report.high_water_mark
    label = firm.label_of(firm.group_of("EMBER_OBJECTIVE_WR"))
    assert report.groups[label].binding.get("drawdown", 0) > 0
    assert report.floor == report.high_water_mark * D("0.90")


def test_the_drawdown_limit_stops_new_risk_at_the_floor(listed):
    """At the floor the cushion is zero, so the allowance is zero. Nothing is sold.

    A stop-loss would flatten here and this deliberately does not. The loss on
    an open position is already bounded by the collateral posted and there is
    no margin call, so selling converts a funded, bounded loss into a realised
    one plus the spread. What the floor stops is *new* risk.
    """
    firm = Firm(
        budget=D(1_000_000),
        strategies={"mm": Quoter(["EMBER_OBJECTIVE_WR"])},
        max_drawdown=0.01,
        drawdown_multiple=1.0,
        cost_of_capital=False,
    )
    marked = _drawn_down(listed, firm, "EMBER_OBJECTIVE_WR", 150, 9_000, 100)
    quotes, takes = _wake(firm, marked, ["EMBER_OBJECTIVE_WR"])
    assert quotes["EMBER_OBJECTIVE_WR"].bid is None
    assert takes == ()
    assert firm.committed()[firm.group_of("EMBER_OBJECTIVE_WR")] > 0


def test_the_metric_exposure_limit_binds_in_the_units_the_payoffs_declare(listed):
    """`sum |dV/dtheta_g|` and `dV/dtheta_g`, computed rather than estimated.

    There is no default for either, and there should not be: a win-rate future
    here has a slope of 10,000 per unit of metric and a match-volume commodity
    1, so any figure this module chose would be a constant nobody measured.
    With a net cap of 50,000 the firm stops at five lots of
    EMBER_OBJECTIVE_WR, which is 50,000 exactly, and refuses the sixth. The
    same firm without the cap stops at 10, where its 102,272 group budget runs
    out against the 10,000 a lot an unpriced market buy is collateralised at,
    so the metric limit is genuinely the thing deciding and not the budget
    wearing another name.

    The capital is 2,500,000 rather than the 1,000,000 the rest of this file
    uses, and that is the point rather than an inflation. The circuit listing
    is 22 netting groups against the previous 10, so a flat split of 1,000,000
    leaves 40,909 a group and stops this taker at 4 lots on the budget alone.
    A limit tested in a state where a different limit binds first is the inert
    control this file exists to avoid.
    """
    firm = Firm(
        budget=D(2_500_000),
        strategies={"a": Lifter("EMBER_OBJECTIVE_WR", 40)},
        net_metric_limit=50_000.0,
    )
    _, takes = _wake(firm, _view(listed))
    assert [(t.symbol, t.size) for t in takes] == [("EMBER_OBJECTIVE_WR", 5)]
    assert firm.report().binding.get("net-metric", 0) > 0

    unlimited = Firm(
        budget=D(2_500_000), strategies={"a": Lifter("EMBER_OBJECTIVE_WR", 40)}
    )
    _, takes = _wake(unlimited, _view(listed))
    assert takes[0].size == 10
    assert unlimited.report().binding == {"budget": 1}


def test_the_reserve_is_charged_on_the_gross_the_venue_actually_posts(listed):
    """Netting is off on the live venue, so free cash is a gross question.

    `Venue.__init__` takes `netting: bool = False` and no call site sets it
    True, so an account is charged per contract however well its package
    offsets. Budgeting the *risk* on the netted figure and the *cash* on the
    gross one is the only pair that is right about both, and mixing them would
    be the units bug this repository has hit four times.
    """
    def spread(size):
        firm = Firm(
            budget=D(60_000),
            strategies={"rv": Spread("EMBER_OBJECTIVE_C4800", "EMBER_OBJECTIVE_C5000", size)},
            reserve=0.10,
            concentration=1.0,
        )
        _, takes = _wake(firm, _view(listed))
        # The two prices `_view` hands the strategy: the midpoint of each
        # leg's own range, 2,600 on a call whose value tops out at 5,200 and
        # 2,500 on one that tops out at 5,000.
        holdings = [
            (listed["EMBER_OBJECTIVE_C4800"].spec, size, D(2_600)),
            (listed["EMBER_OBJECTIVE_C5000"].spec, -size, D(2_500)),
        ]
        return firm, takes, netting_benefit(holdings)

    deployable = D(60_000) * D("0.90")
    _firm, takes, (gross, net) = spread(10)
    assert len(takes) == 2
    assert (gross, net) == (D(51_000), D(1_000))
    assert gross <= deployable

    # One more lot, and nothing about the risk has changed: the net requirement
    # goes from 1,000 to 1,100 against a group budget of 2,454. What refuses it
    # is the gross figure crossing 54,000, which is the number the venue would
    # actually post, and the package goes out whole or not at all.
    firm, takes, (gross, net) = spread(11)
    assert takes == ()
    assert net == D(1_100) < firm.budgets()[firm.group_of("EMBER_OBJECTIVE_C4800")]
    assert gross == D(56_100) > deployable


# --------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------


def test_a_package_that_does_not_fit_whole_is_not_sent_at_all(listed):
    """Half a hedge is the position the hedge existed to avoid.

    A firm that clips one leg and funds the other has manufactured naked risk
    out of a package that could not lose. So a package is admitted whole or
    refused, and the refusal is recorded against the strategy that asked.
    """
    firm = Firm(
        budget=D(4_000),
        strategies={"rv": Spread("EMBER_OBJECTIVE_C4800", "EMBER_OBJECTIVE_C5000", 5_000)},
    )
    _, takes = _wake(firm, _view(listed))
    assert takes == ()
    assert firm.report().strategies["rv"].refused["package-unfunded"] == 1


def test_a_half_legged_package_is_flattened_immediately(listed):
    """The one unwind this firm performs, and it is not a stop-loss.

    Both legs go out; only the long comes back. What is left is a naked long
    nobody sized, nobody has a view on and nobody is being paid to hold, so it
    goes at the next wake. A view is worth waiting on; an execution failure is
    not.

    The short leg's symbol prints at a new price in the second view, and that
    is the whole point rather than scene-setting: an unfilled order is not dead
    because time passed, it is dead because the market traded past it. A leg
    that has not filled in a book that has not traded either is still working,
    which is what a halted symbol and an opening call both look like, and
    unwinding against one of those would be inventing an execution failure.
    """
    strategy = Spread("EMBER_OBJECTIVE_C4800", "EMBER_OBJECTIVE_C5000", 20, once=True)
    firm = Firm(budget=D(1_000_000), strategies={"rv": strategy})
    _, takes = _wake(firm, _view(listed, now=0.0))
    assert sorted(t.symbol for t in takes) == ["EMBER_OBJECTIVE_C4800", "EMBER_OBJECTIVE_C5000"]

    quiet = _view(
        listed,
        now=0.3,
        cash=D(1_000_000) - D(20) * D(2_600),
        positions={"EMBER_OBJECTIVE_C4800": 20},
    )
    _, takes = _wake(firm, quiet)
    assert takes == ()

    half = _view(
        listed,
        now=0.6,
        cash=D(1_000_000) - D(20) * D(2_600),
        positions={"EMBER_OBJECTIVE_C4800": 20},
        prices={"EMBER_OBJECTIVE_C5000": D(2_400)},
    )
    _, takes = _wake(firm, half)
    assert [(t.symbol, t.side, t.size) for t in takes] == [
        ("EMBER_OBJECTIVE_C4800", Side.SELL, 20)
    ]
    assert firm.report().strategies["rv"].refused["half-legged"] == 1


def test_a_package_that_fills_whole_is_left_alone(listed):
    """The same state one leg away, and the firm does nothing.

    Without this the flatten test would pass on a firm that unwinds every
    package it ever sends, which is a different and much worse control.
    """
    strategy = Spread("EMBER_OBJECTIVE_C4800", "EMBER_OBJECTIVE_C5000", 20, once=True)
    firm = Firm(budget=D(1_000_000), strategies={"rv": strategy})
    _wake(firm, _view(listed, now=0.0))
    whole = _view(
        listed,
        now=0.3,
        cash=D(1_000_000),
        positions={"EMBER_OBJECTIVE_C4800": 20, "EMBER_OBJECTIVE_C5000": -20},
    )
    _, takes = _wake(firm, whole)
    assert takes == ()
    assert "half-legged" not in firm.report().strategies["rv"].refused


def test_netting_releases_capital_on_the_packages_these_strategies_produce(listed):
    """Gross against net on the vertical this firm actually trades.

    Twenty spreads between 4,800 and 5,000 can lose at most two hundred points
    a lot, so at the settlement prices of the two legs the netted requirement
    is 4,000 against a gross of 104,000: netting would release 100,000, which
    is 96.2% of what the venue charges today. The gross figure is dominated by
    the short leg, which is charged the whole distance from its price to the
    top of its range whatever it is held against. That is the reason the report
    carries both numbers rather than one.
    """
    holdings = [
        (listed["EMBER_OBJECTIVE_C4800"].spec, 20, D("238.25")),
        (listed["EMBER_OBJECTIVE_C5000"].spec, -20, D("38.25")),
    ]
    gross, net = netting_benefit(holdings)
    assert net == D(4_000)
    assert gross == D(104_000)
    assert gross - net == D(100_000)
    assert (gross - net) / gross > D("0.96")


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------


def test_the_firm_recovers_the_basis_it_is_never_told(listed):
    """The one number everything else here is computed from, and it is inferred.

    A strategy is not told what it paid. `Account.apply_fill` moves cash by
    realised P&L and fees alone, because a futures position is a collateralised
    commitment rather than a purchase, so an opening fill moves cash by exactly
    nothing and there is no price in the cash delta to read. What does carry
    the price is the collateral: it is charged against the position's own basis
    and moves only when a fill moves that basis, so one symbol moving means the
    whole change belongs to it and inverts to a basis exactly.

    This is a regression test in the strict sense. The firm used to difference
    cash, and when the adapter's ledger stopped being a purchase ledger, every
    position in the firm's book silently took a basis of 0.00. A long booked at
    zero is charged the distance from zero down to the bottom of its range,
    which is nothing, so every group reported a committed collateral of zero,
    no budget ever bound on anything already held, and the live run finished
    255,797.90 into a group budgeted 36,000. Nothing raised. The budget check
    ran before every order exactly as designed, against a blank book.

    Four fills, chosen so that each branch of the basis arithmetic is used: an
    open, an add at a different price, a partial close that realises, and a
    flip through zero.
    """
    firm = Firm(budget=D(1_000_000), strategies={"a": Lifter("EMBER_OBJECTIVE_WR", 1)})
    book = Book(listed)
    fills = [(10, 4_600), (10, 4_800), (-15, 4_900), (-10, 4_500)]
    _wake(firm, _view(listed, now=0.0, book=book))
    for step, (lots, price) in enumerate(fills, start=1):
        book.fill("EMBER_OBJECTIVE_WR", lots, price)
        _wake(firm, _view(listed, now=step * 0.3, book=book,
                          prices={"EMBER_OBJECTIVE_WR": D(price)}))
        held = book.account.positions["EMBER_OBJECTIVE_WR"]
        assert firm.basis() == {
            "EMBER_OBJECTIVE_WR": from_money(held.cost_basis)
        }, (step, lots, price)

    # And the collateral computed from it is the venue's own figure, not
    # merely a self-consistent one.
    assert firm.committed()[firm.group_of("EMBER_OBJECTIVE_WR")] == worst_case(
        [(listed["EMBER_OBJECTIVE_WR"].spec, held.quantity, from_money(held.cost_basis)
          / D(held.quantity))]
    )
    assert book.posted > 0


def test_attribution_across_strategies_sums_to_the_firm_exactly(listed):
    """An identity in Decimal, not a tolerance.

    Two strategies, two symbols, and a fill sequence that opens, adds and
    partly closes. The per-strategy figures plus the unattributed residue are
    the firm's whole P&L, exactly, and each strategy's realised and unrealised
    halves are exactly its own total.
    """
    firm = Firm(
        budget=D(1_000_000),
        strategies={
            "a": Lifter("EMBER_OBJECTIVE_WR", 10),
            "b": Lifter("RIFT_OBJECTIVE_WR", 10, Side.SELL),
        },
    )
    cash = D(1_000_000)
    positions: dict[str, int] = {}
    prices: dict[str, D] = {}
    for step, (symbol, lots, price) in enumerate(
        [
            ("EMBER_OBJECTIVE_WR", 10, D(4_600)),
            ("RIFT_OBJECTIVE_WR", -10, D(4_700)),
            ("EMBER_OBJECTIVE_WR", 10, D(4_800)),
            ("EMBER_OBJECTIVE_WR", -15, D(4_900)),
        ]
    ):
        _wake(firm, _view(listed, now=step * 0.3, cash=cash, positions=dict(positions),
                          prices=dict(prices)))
        cash -= D(lots) * price
        positions[symbol] = positions.get(symbol, 0) + lots
        prices[symbol] = price
    view = _view(listed, now=9.0, cash=cash, positions=positions, prices=prices)
    _wake(firm, view)

    report = firm.report()
    assert report.reconciles()
    attributed = sum((s.pnl for s in report.strategies.values()), start=D(0))
    assert attributed + report.unattributed == report.pnl
    for strategy in report.strategies.values():
        assert strategy.realised + strategy.unrealised == strategy.pnl
    assert report.strategies["a"].fills > 0
    assert report.strategies["b"].fills > 0
    assert report.unattributed == 0


def test_collateral_seconds_are_the_efficiency_metric(listed):
    """P&L per collateral-second, and exact because collateral is arithmetic.

    A strategy that makes the same money on half the collateral for half the
    time is four times the business. No ratio built on a volatility estimate
    can say that here without smuggling in the estimate this venue does
    without.
    """
    firm = Firm(budget=D(1_000_000), strategies={"a": Lifter("EMBER_OBJECTIVE_WR", 10)})
    book = Book(listed)
    _wake(firm, _view(listed, now=0.0, book=book))
    book.fill("EMBER_OBJECTIVE_WR", 10, 4_600)
    _wake(firm, _view(listed, now=10.0, book=book, prices={"EMBER_OBJECTIVE_WR": D(4_600)}))
    _wake(firm, _view(listed, now=20.0, book=book, prices={"EMBER_OBJECTIVE_WR": D(4_700)}))
    report = firm.report()
    strategy = report.strategies["a"]
    assert strategy.collateral_seconds > 0
    assert strategy.pnl == D(1_000)
    assert strategy.pnl_per_collateral_second == pytest.approx(
        float(strategy.pnl) / strategy.collateral_seconds
    )
    group = report.groups[firm.label_of(firm.group_of("EMBER_OBJECTIVE_WR"))]
    # Ten lots long at 4,600 is 46,000 of collateral, held for the ten
    # seconds between the second observation and the third.
    assert group.collateral_seconds == pytest.approx(460_000.0)


def test_an_unmeasured_strategy_is_never_priced_out(listed):
    """`None` until there is evidence, which is the honest answer early on.

    A cost of capital charged against two prints is a coin toss wearing a
    limit, so the measured edge does not exist below the evidence bar and an
    intent is never refused for a record the firm has not got.
    """
    firm = Firm(budget=D(1_000_000), strategies={"a": Lifter("EMBER_OBJECTIVE_WR", 5)})
    _wake(firm, _view(listed))
    assert firm.report().strategies["a"].edge_per_collateral is None
    assert "cost-of-capital" not in firm.report().binding


def test_the_shadow_price_is_zero_until_a_budget_binds(listed):
    """Complementary slackness, and it is the right answer rather than a gap.

    A group whose budget was never in the way has a shadow price of zero,
    because one more unit of capital there would have bought nothing. A group
    that turned an intent away prices the capital at the ratio of the intent it
    turned away, which is what the next unit of budget would have earned.
    Published a wake late because that is when the wake it belongs to is over,
    and a price for a wake still in progress would move as the wake did.

    Priced off a maker, and that is not incidental. A take that names no limit
    is collateralised at the far end of the contract's range, so its distance
    from the reference is negative by construction and it can never set a
    positive price: the floor at zero holds, and a firm of pure unpriced takers
    reports collateral as free. That is the honest ex-ante answer, since what a
    cross is worth depends on a view the firm cannot see, and it is why the
    shadow price is a lower bound rather than a valuation.
    """
    quoter = Quoter(["EMBER_OBJECTIVE_WR"], half_spread=D(20), size=1_000)
    small = Firm(budget=D(20_000), strategies={"a": quoter})
    _wake(small, _view(listed, now=0.0), ["EMBER_OBJECTIVE_WR"])
    _wake(small, _view(listed, now=0.3), ["EMBER_OBJECTIVE_WR"])
    ember = small.group_of("EMBER_OBJECTIVE_WR")
    assert small.shadow_prices()[ember] > 0
    assert small.group_of("RIFT_OBJECTIVE_WR") not in small.shadow_prices()

    roomy = Firm(
        budget=D(1_000_000),
        strategies={"a": Quoter(["EMBER_OBJECTIVE_WR"], half_spread=D(20), size=1)},
    )
    _wake(roomy, _view(listed, now=0.0), ["EMBER_OBJECTIVE_WR"])
    _wake(roomy, _view(listed, now=0.3), ["EMBER_OBJECTIVE_WR"])
    assert roomy.shadow_prices() == {}
    roomy_label = roomy.label_of(roomy.group_of("EMBER_OBJECTIVE_WR"))
    assert roomy.report().groups[roomy_label].shadow_price == 0.0


def test_a_strategy_below_the_shadow_price_is_charged_and_stops(listed):
    """The internal cost of capital, in the state where a strategy has a record.

    Eight fills at prices the market immediately marks against it, and the
    strategy's edge per unit of collateral is measurably negative. The shadow
    price of collateral is never below zero, so the strategy is asking for
    capital at a price it cannot pay, and the firm stops funding it. This is
    the venue-native replacement for a Sharpe ranking, and it is an allocation
    decision rather than a stop: the position it already holds is left alone,
    still funded and still bounded by the collateral posted.

    The capital is 2,000,000 because the limit has to be the thing that binds.
    At 1,000,000 across the circuit listing's 22 netting groups a group gets
    40,909, an unpriced market buy is collateralised at 10,000 a lot, and the
    budget refuses six of the twelve wakes: measured, seven fills and an edge
    of `None`, so there was no record for a cost of capital to be charged
    against and the refusals all read `budget`. At 2,000,000 the taker gets
    eight fills, an edge of -0.255 per unit of collateral, and five refusals,
    every one of them `cost-of-capital`.
    """
    firm = Firm(budget=D(2_000_000), strategies={"a": Lifter("EMBER_OBJECTIVE_WR", 1)})
    book = Book(listed)
    for step in range(12):
        price = D(5_000) - D(100) * step
        _wake(
            firm,
            _view(listed, now=step * 0.3, book=book, prices={"EMBER_OBJECTIVE_WR": price}),
        )
        book.fill("EMBER_OBJECTIVE_WR", 1, price)

    strategy = firm.report().strategies["a"]
    assert strategy.fills >= 8
    assert strategy.edge_per_collateral is not None
    assert strategy.edge_per_collateral < 0
    _, takes = _wake(
        firm, _view(listed, now=9.0, book=book, prices={"EMBER_OBJECTIVE_WR": D(3_800)})
    )
    assert takes == ()
    assert firm.report().strategies["a"].refused["cost-of-capital"] > 0
    assert firm.committed()[firm.group_of("EMBER_OBJECTIVE_WR")] > 0

    unpriced = Firm(
        budget=D(2_000_000),
        strategies={"a": Lifter("EMBER_OBJECTIVE_WR", 1)},
        cost_of_capital=False,
    )
    book = Book(listed)
    for step in range(12):
        price = D(5_000) - D(100) * step
        _wake(
            unpriced,
            _view(listed, now=step * 0.3, book=book, prices={"EMBER_OBJECTIVE_WR": price}),
        )
        book.fill("EMBER_OBJECTIVE_WR", 1, price)
    _, takes = _wake(
        unpriced, _view(listed, now=9.0, book=book, prices={"EMBER_OBJECTIVE_WR": D(3_800)})
    )
    assert [t.size for t in takes] == [1]


# --------------------------------------------------------------------------
# Allocation
# --------------------------------------------------------------------------


def test_the_first_split_is_equal_and_the_rebalance_is_measured(listed):
    """No evidence means no tilt; a session of evidence means a measured one.

    Inventing an initial tilt out of instrument counts or notionals would be
    the unmeasured constant this codebase does without, so the prior is flat.
    After a session in which one group earned and the others did not, the
    rebalance puts the capital where the P&L per collateral-second was.
    """
    firm = Firm(budget=D(1_000_000), strategies={"a": Lifter("EMBER_OBJECTIVE_WR", 10)})
    book = Book(listed)
    _wake(firm, _view(listed, now=0.0, book=book))
    first = firm.budgets()
    assert len(set(first.values())) == 1

    book.fill("EMBER_OBJECTIVE_WR", 10, 4_600)
    _wake(firm, _view(listed, now=5.0, book=book, prices={"EMBER_OBJECTIVE_WR": D(4_600)}))
    _wake(firm, _view(listed, now=10.0, book=book, prices={"EMBER_OBJECTIVE_WR": D(5_000)}))
    after = firm.rebalance()
    ember = firm.group_of("EMBER_OBJECTIVE_WR")
    assert after[ember] > first[ember]
    assert sum(after.values()) <= firm.budget * D("0.90")
    assert after[ember] <= firm.budget * D("0.90") * D("0.40")


def test_budgeting_by_collateral_is_not_budgeting_by_notional(listed):
    """The same capital, split two ways, and the two disagree by a lot.

    Notional is what a contract is worth; collateral is what holding it can
    cost. On this catalogue the call struck at 5,000 marks at 40 and can lose
    40, while the future it is written on marks at 5,038 and can lose 5,038
    long or 4,962 short. A notional budget spends almost nothing on the option
    and almost everything on the future; a collateral budget prices them by
    what they actually consume, which is what the venue charges and what runs
    out.

    The three prices are where this chain sits: the team win rate settles at
    5,038.25, its 5,000 call at 38.25 and its 5,000 put at nothing, so 5,038,
    40 and 10 are that chain rounded to prices a book would show.
    """
    lots = 100
    rows = [
        ("EMBER_OBJECTIVE_WR", D(5_038)),
        ("EMBER_OBJECTIVE_C5000", D(40)),
        ("EMBER_OBJECTIVE_P5000", D(10)),
    ]
    notional = {s: p * lots for s, p in rows}
    collateral = {
        s: listed[s].spec.collateral_for(lots, p) for s, p in rows
    }
    assert notional["EMBER_OBJECTIVE_WR"] == D(503_800)
    assert collateral["EMBER_OBJECTIVE_WR"] == D(503_800)
    assert notional["EMBER_OBJECTIVE_C5000"] == D(4_000)
    assert collateral["EMBER_OBJECTIVE_C5000"] == D(4_000)

    short = {s: listed[s].spec.collateral_for(-lots, p) for s, p in rows}
    assert short["EMBER_OBJECTIVE_WR"] == D(496_200)
    assert short["EMBER_OBJECTIVE_C5000"] == D(496_000)
    # The whole disagreement, in one number. A hundred short
    # EMBER_OBJECTIVE_C5000 is 4,000 of notional and 496,000 of collateral, a
    # factor of 124, because a short call is charged the distance from its
    # price to the top of its range. The short put, which marks at 10 against a
    # range of 5,000, is a factor of 499, and the future is 0.98. So a notional
    # budget would fund 124 times the short call position the capital can
    # actually pay for, and would size the future almost right: the two rules
    # disagree least on exactly the instrument where collateral is easiest to
    # guess, and most where it is not.
    assert short["EMBER_OBJECTIVE_C5000"] / notional["EMBER_OBJECTIVE_C5000"] == D(124)
    assert short["EMBER_OBJECTIVE_P5000"] / notional["EMBER_OBJECTIVE_P5000"] == D(499)
    assert short["EMBER_OBJECTIVE_WR"] / notional["EMBER_OBJECTIVE_WR"] == pytest.approx(
        D("0.9849"), abs=D("0.0001")
    )


# --------------------------------------------------------------------------
# On the venue
# --------------------------------------------------------------------------


def test_a_firm_runs_on_the_live_market_and_conserves_exactly():
    """Sixty simulated seconds, three strategies, conservation exactly zero.

    The end-to-end check, and the only one that exercises the adapter, the
    latency, the opening auction, a circuit-breaker pause and the venue's own
    collateral refusals at once. Measured on seed 7 with fees off: 113 fills,
    conservation exactly 0, the budget binding first at 0.294 simulated
    seconds, and 112 of the 113 fills attributed to the strategy that asked for
    them, the last one carrying 58.88 of a P&L of -8,713.38.

    Only the properties are asserted, not those figures. Everything here
    depends on the whole market's composition, so a listing or a change to a
    maker moves them legitimately, and a test that fails for that is a test
    nobody can read. Attribution is the exception, because it is the one thing
    here that is a property of this module rather than of the market. The
    identity is exact and asserted as such; the coverage behind it is asserted
    as a share of fills, since the money attached to an orphaned lot is
    whatever that lot happened to be worth.

    An earlier version of the reaping rule left 107,196.00 unattributed, which
    is where the coverage assertion earns its place: a circuit-breaker pause
    leaves an order resting exactly as an opening call does, and five fills
    came back at 20.0s, 40.0s and 52.0s against claims the firm had already
    aged out on a timer.

    Fees are off because the agent's shadow ledger books fills and not fees, so
    with a schedule on, its cash and the venue's diverge by exactly the fees
    paid, and a reconciliation that has to allow for that stops being able to
    catch anything else.
    """
    market = build(seed=7, fees=FREE)
    cash = D(400_000)
    firm = Firm(
        budget=cash,
        strategies={
            "mm": Quoter(["EMBER_OBJECTIVE_WR", "RIFT_OBJECTIVE_WR"], half_spread=D(12), size=4),
            "rv": Lifter("EMBER_OBJECTIVE_WR", 2),
            "sp": Spread("EMBER_OBJECTIVE_C4800", "EMBER_OBJECTIVE_C5000", 6),
        },
    )
    agent_id = AgentId("firm-1")
    market.venue.open_account(agent_id, cash)
    agent = StrategyAgent(
        agent_id,
        VENUE_ID,
        {s: market.venue.registry.require(s) for s in market.venue.registry.symbols},
        wake_interval=millis(300),
        maker=firm,
        taker=firm,
        starting_cash=cash,
    )
    market.kernel.add(agent)
    market.kernel.start()
    market.kernel.advance(until=seconds(60))

    assert int(market.venue.conservation_check()) == 0

    report = firm.report()
    assert report.reconciles()
    attributed = sum((s.pnl for s in report.strategies.values()), start=D(0))
    assert attributed + report.unattributed == report.pnl

    # The identity above is exact and structural. This is the quality of the
    # attribution behind it, and it is a count rather than an amount because
    # the amount depends on what the orphaned lot happened to be worth. There
    # is one way a fill reaches nobody: a withdrawal is a Cancel in flight and
    # a Cancel can lose the race to a fill, which no strategy can see coming.
    # Measured on this run, one fill in ninety-nine.
    fills = sum(s.fills for s in report.strategies.values())
    assert fills > 0
    assert report.unattributed_fills < fills // 20
    assert len(report.equity_curve) > 1
    assert report.first_binding is not None
    # Netting on the packages these strategies actually produced, which is the
    # figure the report carries both halves of: the venue charges the gross
    # because `Venue.netting` is False everywhere, and the difference is what a
    # clearing house would hand back.
    assert report.gross >= report.net
    assert report.benefit == report.gross - report.net

    # The budget, re-derived from the venue's own positions rather than from
    # anything the firm recorded, so this cannot pass on a firm that miscounts
    # its own book.
    account = market.venue.account(agent_id)
    holdings: dict[str, list] = {}
    for symbol, position in account.positions.items():
        if not position.quantity:
            continue
        instrument = market.venue.registry.require(symbol)
        average = D(int(position.cost_basis)) / D(position.quantity) / 1_000_000
        holdings.setdefault(group_key(instrument), []).append(
            (instrument.spec, int(position.quantity), average)
        )
    budgets = firm.budgets()
    for group, rows in holdings.items():
        assert worst_case(rows) <= budgets[group], firm.label_of(group)

    # And the reason it holds: the firm's own book is the venue's book. Asserted
    # rather than implied, because the budget breach that led here was not a
    # sizing error at all, it was the firm checking a budget against positions
    # it thought it had bought for nothing.
    mine = firm.basis()
    theirs = {
        symbol: D(int(position.cost_basis)) / 1_000_000
        for symbol, position in account.positions.items()
        if position.quantity
    }
    assert mine == theirs


def test_the_firm_is_both_protocols_at_once():
    """It has to be, or `StrategyAgent(maker=firm, taker=firm)` is not a thing.

    Checked structurally rather than by running it, because a firm that
    satisfies only one of the two would silently become half a firm: the agent
    accepts a maker, a taker or both, so the failure is a missing method rather
    than an error anybody sees.
    """
    from arena.strategies.base import MakerStrategy, TakerStrategy

    firm = Firm(budget=D(1_000), strategies={"a": Lifter("X", 1)})
    assert isinstance(firm, MakerStrategy)
    assert isinstance(firm, TakerStrategy)
