"""A venue whose market maker is a function rather than a participant.

Everything an agent can observe here has the shape it has on the order-book
venue: a two-sided book with depth at successive ticks, top-of-book and trade
feeds, private fills, accounts, collateral and settlement. The only thing that
changed is where the liquidity comes from (a cost function instead of resting
orders), which is precisely the variable Experiment 2 needs to isolate.

Rendering the curve as a ladder
-------------------------------

An automated market maker *is* a book; it is just defined by a curve rather than
by orders someone left behind. Level ``T`` on the ask holds the shares that move
the marginal price from tick ``T`` to tick ``T+1``, which is exactly what
:meth:`LmsrMarket.shares_for_price` inverts. A limit order then walks that
ladder the same way it walks a real one, and no agent needs to know which venue
it is trading on.

Two consequences are worth being explicit about, because they are where this
venue is *not* a book:

* **Nothing rests.** There is no queue and no time priority, so every order is
  immediate-or-cancel in effect: the marketable part fills against the curve and
  the remainder is cancelled rather than left working. Agents already handle
  that lifecycle, since it is what an IOC does.
* **The spread is made by rounding.** Raw LMSR is path independent, so a round
  trip at the same net position is exactly free: it has no spread at all.
  Quantising to the tick grid, always in the maker's favour, is what produces
  one. That keeps the maker's bounded-loss guarantee intact (it can only ever
  be charged less than the rule says) and it keeps the ledger exact, because
  both sides of a fill book at the same integer tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable

from arena.contracts.payoff import Binary
from arena.exchange.book import BookSnapshot
from arena.exchange.events import (
    Acknowledged,
    Cancel,
    Cancelled,
    Command,
    Event,
    Filled,
    Rejected,
    Replace,
    Submit,
    Traded,
)
from arena.exchange.types import (
    AgentId,
    OrderId,
    OrderType,
    Price,
    Quantity,
    RejectReason,
    SequenceNumber,
    Side,
    TimeInForce,
)
from arena.market.instrument import Instrument
from arena.market.lmsr import LmsrMarket, liquidity_for_subsidy
from arena.exchange.session import SessionState
from arena.market.venue import Venue
from arena.portfolio.money import Money, to_money

__all__ = ["LmsrVenue", "LmsrPool", "LmsrBook", "LMSR_MAKER_ID"]

# The scoring rule needs somewhere to book the other side of every trade. It is
# a real account with real positions, so conservation stays checkable: value
# moves between traders and this account, and nothing is created.
LMSR_MAKER_ID = AgentId("lmsr-maker")


class LmsrBook:
    """The cost curve, presented as an L2 book.

    Sizes are the shares that move the marginal price across one tick, so the
    ladder is the market's real liquidity rather than a decoration. Walking it
    and applying the cost function give the same fills.
    """

    def __init__(self, market: LmsrMarket, tick_size: Decimal, bounds: tuple[int, int]):
        self.market = market
        self.tick = float(tick_size)
        self.low, self.high = bounds

    # -- tick arithmetic ---------------------------------------------------

    def _ticks(self, price: float) -> float:
        return price / self.tick

    def _clamp(self, ticks: int) -> int:
        return max(self.low, min(self.high, ticks))

    @property
    def fair_ticks(self) -> float:
        return self._ticks(self.market.price)

    def best_price(self, side: Side) -> Price | None:
        """The touch. BUY is what the venue bids; SELL is what it offers.

        Rounded outward from fair value so the venue is never the one giving
        value away on the rounding, and forced at least one tick apart; two
        equal touches would be a crossed quote to anyone reading the feed.
        """
        fair = self.fair_ticks
        if side is Side.BUY:
            bid = self._clamp(math.floor(fair))
            if bid >= self._clamp(math.ceil(fair)):
                bid = self._clamp(bid - 1)
            return Price(bid) if bid >= self.low else None
        ask = self._clamp(math.ceil(fair))
        if ask <= self._clamp(math.floor(fair)):
            ask = self._clamp(ask + 1)
        return Price(ask) if ask <= self.high else None

    def best_priced(self, side: Side) -> Price | None:
        """The same touch, because there is nothing here to leave out.

        The order book's version of this filters market-on-open interest, which
        rests at a sentinel so that it crosses everything in the call. A
        scoring-rule market holds no orders at all (the touch is a function of
        the outstanding quantity), so there is no sentinel to hide and the two
        questions have one answer.

        Present so that the two venues answer the same question by the same
        name. `VenueAgent.top_of_book` serves both and had to move to
        `best_priced` to stop publishing the sentinel; without this it raised
        `AttributeError` on every LMSR market instead.
        """
        return self.best_price(side)

    def shares_to_reach(self, ticks: int) -> float:
        """Net position at which the marginal price would be ``ticks``."""
        return self.market.shares_for_price(ticks * self.tick)

    def depth_at(self, side: Side, price: Price | None) -> Quantity:
        """Shares available before the price moves past ``price``."""
        if price is None:
            return Quantity(0)
        level = int(price)
        if side is Side.SELL:
            # Buying pushes net up; this level is exhausted at its own tick.
            available = self.shares_to_reach(level) - self.market.net
        else:
            available = self.market.net - self.shares_to_reach(level)
        return Quantity(max(0, int(available)))

    def snapshot(self, levels: int = 5) -> BookSnapshot:
        bids: list[tuple[Price, Quantity]] = []
        asks: list[tuple[Price, Quantity]] = []

        best_bid = self.best_price(Side.BUY)
        best_ask = self.best_price(Side.SELL)

        if best_ask is not None:
            previous = float(self.market.net)
            for offset in range(levels):
                tick = int(best_ask) + offset
                if tick > self.high:
                    break
                edge = self.shares_to_reach(tick)
                size = int(edge - previous)
                previous = max(previous, edge)
                if size > 0:
                    asks.append((Price(tick), Quantity(size)))

        if best_bid is not None:
            previous = float(self.market.net)
            for offset in range(levels):
                tick = int(best_bid) - offset
                if tick < self.low:
                    break
                edge = self.shares_to_reach(tick)
                size = int(previous - edge)
                previous = min(previous, edge)
                if size > 0:
                    bids.append((Price(tick), Quantity(size)))

        return BookSnapshot(bids=tuple(bids), asks=tuple(asks))


@dataclass(slots=True)
class LmsrPool:
    """Stands where a matching engine stands, so the venue layer is unchanged."""

    symbol: str
    market: LmsrMarket
    book: LmsrBook
    _tape: list[Traded] = field(default_factory=list)
    _sequence: int = 0

    @property
    def tape(self) -> tuple[Traded, ...]:
        return tuple(self._tape)

    def next_sequence(self) -> SequenceNumber:
        self._sequence += 1
        return SequenceNumber(self._sequence)


class LmsrVenue(Venue):
    """The order-book venue with its matching replaced by a scoring rule.

    Subclassed rather than rewritten on purpose: accounts, collateral,
    settlement, expiry and the conservation check are the *same code* on both
    venues. If they were reimplemented here, a difference between the two
    experiments could be an accounting difference, and the comparison would be
    worthless.
    """

    def __init__(
        self,
        name: str = "arena-lmsr",
        starting_cash: Decimal | int = 1_000_000,
        subsidy: Decimal | int | float = 2_000,
        clock: Callable[[], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, starting_cash=starting_cash, clock=clock, **kwargs
        )
        self.subsidy = float(subsidy)
        self._pools: dict[str, LmsrPool] = {}

    # -- listing -----------------------------------------------------------

    def list_instrument(self, instrument: Instrument) -> None:
        payoff = instrument.spec.payoff
        if not isinstance(payoff, Binary):
            raise ValueError(
                f"{instrument.symbol} is a {type(payoff).__name__}. This venue runs a "
                "binary scoring rule, and listing a contract it cannot price would "
                "quote a number with no meaning behind it."
            )
        self.registry.list_instrument(instrument)
        self._phase[instrument.symbol] = SessionState.CONTINUOUS
        market = LmsrMarket(
            liquidity_for_subsidy(self.subsidy, float(payoff.payout)),
            payout=float(payoff.payout),
        )
        low, high = instrument.tick_bounds
        self._pools[instrument.symbol] = LmsrPool(
            symbol=instrument.symbol,
            market=market,
            book=LmsrBook(market, instrument.tick_size, (int(low), int(high))),
        )
        # The maker's account is opened here so its starting capital is on the
        # books from the beginning, and the conservation check has something to
        # balance against from the first trade.
        self.account(LMSR_MAKER_ID)

    def engine(self, symbol: str) -> LmsrPool:  # type: ignore[override]
        return self._pools[symbol]

    @property
    def pools(self) -> dict[str, LmsrPool]:
        return dict(sorted(self._pools.items()))

    def account(self, agent_id: AgentId):
        """The maker is funded to its bound, and only to its bound.

        Its worst case is the subsidy, exactly, so that is what it gets. Funding
        it arbitrarily would make "the venue spent this much making a market" an
        unanswerable question.
        """
        if agent_id == LMSR_MAKER_ID and agent_id not in self._accounts:
            from arena.portfolio.account import Account

            account = Account(
                agent_id=str(agent_id),
                starting_cash=to_money(Decimal(str(self.subsidy))),
            )
            self._accounts[agent_id] = account
            return account
        return super().account(agent_id)

    # -- marking -----------------------------------------------------------

    def mark(self, symbol: str) -> Money:
        """Fair value straight off the curve.

        The book venue has to fall back through last-trade and mid-range when
        nothing is quoted; here the rule always has an opinion, which is the
        whole point of it.
        """
        instrument = self.registry.require(symbol)
        pool = self._pools[symbol]
        return to_money(
            Decimal(str(pool.market.price)).quantize(instrument.tick_size)
        )

    # -- trading -----------------------------------------------------------

    def submit(self, agent_id: AgentId, symbol: str, command: Command) -> list[Event]:
        instrument = self.registry.get(symbol)
        if instrument is None or symbol not in self._pools:
            return [Rejected(SequenceNumber(0), agent_id, RejectReason.UNKNOWN_ORDER)]

        if self._clock is not None and self._clock() >= instrument.expiry:
            self._set_phase(symbol, SessionState.CLOSED)

        if isinstance(command, Cancel):
            # Nothing rests, so there is never anything to cancel. Answering
            # with a Cancelled rather than a rejection keeps the agent's own
            # bookkeeping consistent: it asked for the order to be gone, and it
            # is gone.
            return [
                Cancelled(
                    self._pools[symbol].next_sequence(),
                    agent_id,
                    command.order_id,
                    Quantity(0),
                )
            ]

        if not isinstance(command, Submit):
            if isinstance(command, Replace):
                return [
                    Rejected(
                        SequenceNumber(0),
                        agent_id,
                        RejectReason.UNKNOWN_ORDER,
                        command.order_id,
                    )
                ]
            return [Rejected(SequenceNumber(0), agent_id, RejectReason.UNKNOWN_ORDER)]

        if not self.session(symbol).accepts_orders:
            return [
                Rejected(SequenceNumber(0), agent_id, RejectReason.ALREADY_TERMINAL)
            ]
        if self.session(symbol) is SessionState.AUCTION:
            # A scoring rule has no call phase to accumulate into: its price is
            # a function of net flow, and there is no book to clear. A halted
            # AMM is therefore simply shut, which is the honest behaviour rather
            # than pretending it can hold an auction.
            return [
                Rejected(SequenceNumber(0), agent_id, RejectReason.ALREADY_TERMINAL)
            ]

        return self._execute(agent_id, symbol, instrument, command)

    def _execute(
        self, agent_id: AgentId, symbol: str, instrument: Instrument, command: Submit
    ) -> list[Event]:
        pool = self._pools[symbol]
        order_id = OrderId(int(pool.next_sequence()))
        events: list[Event] = [
            Acknowledged(
                pool.next_sequence(),
                agent_id,
                order_id,
                command.side,
                command.quantity,
                command.price,
            )
        ]

        quantity = self._fillable(pool, instrument, command)
        if quantity <= 0:
            events.append(Cancelled(pool.next_sequence(), agent_id, order_id, command.quantity))
            return events

        if command.time_in_force is TimeInForce.FOK and quantity < int(command.quantity):
            events.append(Cancelled(pool.next_sequence(), agent_id, order_id, command.quantity))
            return events

        signed = quantity if command.side is Side.BUY else -quantity
        ticks = self._fill_ticks(pool, instrument, signed)

        if not self._affordable_at(agent_id, instrument, command.side, quantity, ticks):
            return [
                Rejected(
                    SequenceNumber(0),
                    agent_id,
                    RejectReason.INSUFFICIENT_COLLATERAL,
                    order_id,
                )
            ]

        pool.market.apply(signed)

        # Both sides book at the same integer tick, which is what makes the
        # ledger conserve exactly rather than nearly.
        bounds = instrument.bounds_in_minor
        price = instrument.price_in_minor(ticks)
        self.account(agent_id).apply_fill(symbol, signed, price, bounds)
        self.account(LMSR_MAKER_ID).apply_fill(symbol, -signed, price, bounds)

        events.append(
            Filled(
                pool.next_sequence(),
                agent_id,
                order_id,
                command.side,
                Quantity(quantity),
                ticks,
                # Every order here crosses: there is nothing resting to be the
                # passive side, so the taker flag is always true.
                True,
                Quantity(max(0, int(command.quantity) - quantity)),
            )
        )
        trade = Traded(
            pool.next_sequence(),
            Quantity(quantity),
            ticks,
            command.side,
            order_id,
            OrderId(0),
        )
        pool._tape.append(trade)
        self._last[symbol] = ticks
        events.append(trade)

        if quantity < int(command.quantity):
            events.append(
                Cancelled(
                    pool.next_sequence(),
                    agent_id,
                    order_id,
                    Quantity(int(command.quantity) - quantity),
                )
            )
        return events

    # -- sizing and pricing ------------------------------------------------

    def _fillable(self, pool: LmsrPool, instrument: Instrument, command: Submit) -> int:
        """How much of this order the curve will fill at an acceptable price.

        A market order takes the whole size: the rule always has a price. A
        limit order walks the ladder and stops where the marginal price passes
        its limit, exactly as it would on a real book.
        """
        wanted = int(command.quantity)
        if command.order_type is OrderType.MARKET or command.price is None:
            return wanted

        limit = int(command.price)
        book = pool.book
        if command.side is Side.BUY:
            if limit < int(book.best_price(Side.SELL) or limit + 1):
                return 0
            # Everything up to and including the limit tick is available.
            reachable = book.shares_to_reach(limit + 1) - pool.market.net
        else:
            if limit > int(book.best_price(Side.BUY) or limit - 1):
                return 0
            reachable = pool.market.net - book.shares_to_reach(limit)
        return max(0, min(wanted, int(reachable)))

    def _fill_ticks(self, pool: LmsrPool, instrument: Instrument, signed: int) -> Price:
        """The whole-tick price both sides book at.

        Rounded against the trader (up when buying, down when selling) so the
        maker is charged no more than the scoring rule says and its bounded
        loss survives the quantisation.
        """
        average = pool.market.average_price(signed)
        ticks = average / float(instrument.tick_size)
        low, high = instrument.tick_bounds
        rounded = math.ceil(ticks) if signed > 0 else math.floor(ticks)
        return Price(max(int(low), min(int(high), rounded)))

    def _affordable_at(
        self,
        agent_id: AgentId,
        instrument: Instrument,
        side: Side,
        quantity: int,
        ticks: Price,
    ) -> bool:
        """Exact worst case for the resulting position, as on the book venue."""
        account = self.account(agent_id)
        position = account.positions.get(instrument.symbol)
        held = position.quantity if position is not None else 0
        signed = quantity if side is Side.BUY else -quantity
        resulting = held + signed
        if resulting == 0:
            return True

        price = instrument.price_in_minor(ticks)
        basis = int(position.cost_basis) if position is not None else 0
        blended = (basis + signed * int(price)) // resulting if resulting else int(price)
        required = account.collateral_required(
            resulting, Money(blended), instrument.bounds_in_minor
        )
        committed = sum(
            int(value)
            for symbol, value in account.collateral.items()
            if symbol != instrument.symbol
        )
        return int(account.cash) - committed >= int(required)
