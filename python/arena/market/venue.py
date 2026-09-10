"""A multi-instrument venue: books, accounts, collateral, and settlement.

This is the component that makes the project a market rather than a book. It
holds one matching engine per symbol, an account per agent, and the instrument
registry that says what each symbol settles into, so a trade can finally become
a position, a position can be marked, and an expiry can turn it into cash.

Responsibilities, and what is deliberately *not* here:

    here          symbols, accounts, collateral checks, marks, settlement
    not here      matching (one MatchingEngine per symbol, untouched)
    not here      latency and scheduling (the kernel's job)
    not here      what a metric means (the world's job)

Keeping matching out is what preserves the differential acceptance test: each
engine remains a pure function of its own command stream, so a second
implementation can be validated by replaying one symbol's commands through both.

**Risk is checked before an order reaches the book, not after it fills.** An
exchange that discovers insolvency after the trade has printed cannot unprint
it, and a simulation that allows it produces PnL nobody could have earned. Every
instrument here settles inside a known interval, so the check is exact: the
worst case of the resulting position is arithmetic, not an estimate.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from arena.contracts.spec import ContractSpec
from arena.exchange.engine import MatchingEngine
from arena.exchange.events import (
    Acknowledged,
    Cancel,
    Cancelled,
    Command,
    Event,
    Filled,
    Rejected,
    Replace,
    Replaced,
    Submit,
    Traded,
)
from arena.exchange.types import (
    AgentId,
    OrderId,
    Price,
    Quantity,
    RejectReason,
    SequenceNumber,
    Side,
)
from arena.exchange.session import AuctionResult, SessionState, indicative_auction
from arena.market.fees import FREE, FeeSchedule
from arena.market.instrument import Instrument
from arena.portfolio.account import Account
from arena.portfolio.money import Money, from_money, to_money, MONEY_SCALE
from arena.settlement.result import SettlementResult, SettlementStatus

__all__ = ["Venue", "SymbolCommand", "InstrumentRegistry"]

ZERO = Decimal(0)


@dataclass(frozen=True, slots=True)
class SymbolCommand:
    """A command addressed to one symbol.

    The engine's commands carry no symbol (deliberately, since each engine
    serves exactly one book). Routing is the venue's job, so it is expressed as
    an envelope rather than by widening the engine's own message types.
    """

    symbol: str
    command: Command


class InstrumentRegistry:
    """The listed universe. Immutable once a symbol is listed."""

    def __init__(self) -> None:
        self._instruments: dict[str, Instrument] = {}

    def list_instrument(self, instrument: Instrument) -> None:
        if instrument.symbol in self._instruments:
            raise ValueError(
                f"{instrument.symbol} is already listed. A symbol binds to one "
                "contract for its whole life; relisting it would silently change "
                "what open positions settle into."
            )
        self._instruments[instrument.symbol] = instrument

    def get(self, symbol: str) -> Instrument | None:
        return self._instruments.get(symbol)

    def require(self, symbol: str) -> Instrument:
        instrument = self._instruments.get(symbol)
        if instrument is None:
            raise KeyError(f"{symbol} is not listed on this venue")
        return instrument

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._instruments))

    def by_class(self, instrument_class: str) -> tuple[Instrument, ...]:
        return tuple(
            self._instruments[s]
            for s in self.symbols
            if self._instruments[s].instrument_class == instrument_class
        )


# Where fees land. A real account, so the conservation check sees it.
FEE_ACCOUNT_ID = AgentId("venue-treasury")


def _underlying_of(instrument) -> str:
    """What a contract is written on, as a key positions can be grouped by.

    The window is deliberately *not* part of it. Two contracts on the same
    competitor over different weeks settle on different numbers, so strictly
    they do not net, but `claim_value` prices every one of them off a single
    level, which is the same assumption an agent with one view of a competitor
    makes, and it is the assumption under which the strip identity holds
    exactly. Grouping them together is therefore consistent with how everything
    else here values them; grouping them apart would charge collateral against
    a package the exchange itself treats as riskless.
    """
    from arena.determinism import canonical_json

    return canonical_json(instrument.spec.underlying.to_dict())


class Venue:
    """Books, accounts, and settlement across many instruments."""

    def __init__(
        self,
        name: str = "arena",
        starting_cash: Decimal | int = 1_000_000,
        clock: Callable[[], datetime] | None = None,
        sim_clock: Callable[[], int] | None = None,
        fees: FeeSchedule = FREE,
        price_band: float | None = None,
        limit_state_ns: int = 15_000_000_000,
        pause_ns: int = 300_000_000_000,
        reference_window_ns: int = 300_000_000_000,
        min_reference_prints: int = 20,
        message_rate: int | None = None,
        netting: bool = False,
        balances: dict[AgentId, Decimal | int] | None = None,
    ) -> None:
        self.name = name
        # Supplies wall-clock or simulated calendar time, so expiries can be
        # enforced. Left None when a caller wants to trade a contract outside
        # its window deliberately, which every unit test does, since a fixture
        # contract's window is a fixed date in the past or future.
        self._clock = clock
        # Elapsed simulated time in nanoseconds, which is a different question
        # from what the calendar says and is answered by a different thing. The
        # calendar decides whether a contract has expired; elapsed time decides
        # how long a symbol has been in a limit state and when its pause is
        # over. Conflating them would ask the kernel what year it is.
        #
        # Settable after construction because the venue is built before the
        # kernel that will drive it.
        self.sim_clock = sim_clock
        self.registry = InstrumentRegistry()
        # Converted once, here. Everything downstream is integer minor units,
        # so the ledger conserves exactly rather than nearly.
        self.starting_cash = to_money(starting_cash)
        # Per-agent opening balances, for participants who should not start with
        # the same capital as everyone else. A market maker needs a balance
        # sized to quote every book at once; a person needs one they can read a
        # profit against. Forty million in an account makes a gain of a hundred
        # invisible, which is a bad way to learn what your trade did.
        self._balances: dict[AgentId, Money] = {
            agent_id: to_money(amount) for agent_id, amount in (balances or {}).items()
        }
        # The venue's own account opens empty, because it is not a participant
        # and nobody funded it. Falling through to the default opening balance
        # gave it capital it never received and hid the one thing it exists to
        # show: on a schedule that pays out more than it takes, the venue lost
        # **930,000,000** minor units over two hundred fills and its own
        # account still read 39,999,070,000,000, comfortably solvent, and a
        # measurement of nothing. Starting at zero makes the balance equal to
        # what was collected, so a venue paying people to trade with each other
        # shows up as a negative number rather than a slow drain on capital
        # that was never there.
        self._balances.setdefault(FEE_ACCOUNT_ID, Money(0))
        self._engines: dict[str, MatchingEngine] = {}
        self._accounts: dict[AgentId, Account] = {}
        self._phase: dict[str, SessionState] = {}
        # Symbols that have settled. Tracked here rather than only on the
        # accounts, because a symbol nobody held would otherwise settle twice
        # without complaint. And a settlement firing more than once is a
        # plausible bug in any event-driven system.
        self._settled: set[str] = set()
        # Orders each agent has working, per symbol: order_id -> (side, qty,
        # price in minor units). Collateral is reserved against these, not only
        # against filled positions.
        self._working: dict[tuple[AgentId, str], dict[OrderId, tuple[Side, int, int]]] = {}
        # What those working orders would cost the account if the worse side of
        # each of them filled, per symbol and totalled per agent.
        #
        # `_affordable` has always charged an agent's working orders in the
        # symbol being checked. It never charged the ones resting everywhere
        # else, and the money backing an order is spent by whatever fills
        # first. Measured on seed 17 at 90 simulated seconds: `fund-1` rested a
        # sell of fifteen lots on a call, kept trading forty-five other
        # contracts while it sat there, and when it finally filled the position
        # it created needed 77,977,500,000 minor units of collateral against
        # free cash of 29,020,881,304. The account came out at
        # **-31,486,674,402**, which is the venue holding collateral it does
        # not have.
        #
        # So the requirement is carried from the moment the order is
        # acknowledged rather than discovered when it fills. Kept as a running
        # total rather than summed per admission because an agent quoting
        # thirty symbols would otherwise re-price thirty scenarios on every
        # order it sends.
        self._reserve: dict[tuple[AgentId, str], int] = {}
        self._reserved: dict[AgentId, int] = {}
        # Last traded price per symbol, in ticks. The mark of last resort when
        # a book has no two-sided quote.
        self._last: dict[str, Price] = {}
        # Distributions already paid, per symbol, per unit and cumulative.
        #
        # A contract that pays as it goes is worth less afterwards by exactly
        # what it paid, so this is subtracted from both ends of its range. That
        # is not bookkeeping tidiness: collateral is computed from those ends,
        # so without it a short would keep reserving against a stream it has
        # already paid out, and would look insolvent for having met its
        # obligations.
        self._distributed: dict[str, Money] = {}
        self._distributions_paid: dict[str, int] = {}
        # Fees move value; they do not destroy it. Whatever traders pay lands
        # in the venue's own account, which is checked for conservation like
        # any other. So switching fees on cannot quietly break the ledger's
        # central invariant, and venue revenue becomes a number rather than an
        # idea.
        self.fees = fees
        self.fees_collected = Money(0)
        # How far price may travel from the reference before trading is
        # suspended, as a fraction. None disables the breaker entirely, which
        # is the default so existing measurements are unchanged.
        self.price_band = price_band
        # How long price must stay outside the band before trading pauses, and
        # how long the pause lasts, in nanoseconds.
        #
        # Both exist because a breaker without them is a different mechanism.
        # The rule this models (the US limit up-limit down plan) does not halt
        # on a single print outside the band: the symbol enters a *limit state*
        # and only pauses if it is still there fifteen seconds later, because
        # most excursions are one bad order and halting on those turns a fat
        # finger into an outage. The first version here halted immediately on
        # one print, which is the crude version of the same idea and would have
        # made the breaker fire far more often than the thing it is named after.
        self.limit_state_ns = limit_state_ns
        self.pause_ns = pause_ns
        self._reference: dict[str, Price] = {}
        # Prints inside the reference window, oldest first, as (time, price).
        # The reference is their mean rather than the last trade or the open:
        # a single print can be anywhere, and a breaker referenced to one
        # measures the print rather than the market.
        self._recent: dict[str, deque[tuple[int, Price]]] = {}
        # When each symbol entered its current limit state, if it is in one.
        self._limit_since: dict[str, int] = {}
        # How far back the reference mean looks, and how many prints it needs
        # before the band is enforced at all.
        #
        # The second is not a softening of the rule, it is the rule's own
        # premise. A band is a statement about distance from a *reliable*
        # price, and one print is not one. Enforced from the first trade, the
        # breaker measured every symbol's walk away from its opening print
        # (which on a thinly-attended opening auction is exactly the walk it
        # should be making) and paused 24 of 26 symbols inside a minute.
        self.reference_window_ns = reference_window_ns
        self.min_reference_prints = min_reference_prints
        # Most commands the venue will take from one participant per second, or
        # ``None`` for no limit.
        #
        # Not politeness. An algorithm that malfunctions emits orders faster
        # than anything downstream can process them, and a venue with no limit
        # is the one that goes down with it. It is also the only defence
        # against a participant that discovers it can profit by simply sending
        # more messages than everyone else.
        self.message_rate = message_rate
        # Command timestamps per participant, inside the last second.
        self._messages: dict[AgentId, deque[int]] = {}
        # Participants that have been stopped, and why.
        #
        # A kill switch is the control an exchange reaches for when a
        # participant is doing something nobody wants to reason about at the
        # time. It pulls everything they have working and refuses everything
        # new, and it is deliberately blunt: the point of a kill switch is that
        # it is the one control that always works.
        self.halted_participants: dict[AgentId, str] = {}
        # True while the venue is acting on its own behalf rather than relaying
        # a participant's command, so its own housekeeping is not charged to
        # the participant's message allowance.
        self._internal = False
        # Whether collateral may be netted across contracts on one underlying.
        #
        # Off by default, and the reason is stronger than caution: **as built,
        # this flag only relaxes the gate and does not change what is charged**,
        # so the two disagree.
        #
        # `_portfolio_affords` runs as a fallback when the per-contract check
        # refuses an order, and it asks the netted question: can this account
        # cover the worst case of the whole group. But `Account.apply_fill`
        # keys collateral by *symbol* and always posts `collateral_for_basis`
        # for that one contract, so what the ledger takes is gross whatever
        # this flag says.
        #
        # Measured on seed 7 over 300 simulated seconds with it on: the
        # fallback ran 2,535 times and admitted 1,301 orders the per-contract
        # check had refused, and total posted collateral went *up*, from
        # 321,704,201 to 341,327,876, because more positions were admitted and
        # every one of them was still charged in full. An order admitted on
        # netted risk and then charged gross is exactly how an account ends up
        # holding more than it can back.
        #
        # Making it sound means posting per netting group rather than per
        # symbol, so the gate and the ledger answer the same question. Until
        # then this stays off, and `netting.py`'s released-capital figures
        # describe `worst_case` as a function rather than anything this venue
        # charges.
        self.netting = netting
        # When each paused symbol may reopen.
        self._reopen_at: dict[str, int] = {}
        # Every breach, for the record: a halt that leaves no trace is
        # indistinguishable from a market that simply went quiet.
        self.halts: list[dict[str, Any]] = []
        # Who filled whom, per participant, most recent last. A trade event
        # carries order ids rather than agents, so the pairing is resolved
        # while both orders still resolve; afterwards the ids are just numbers.
        #
        # Kept per agent rather than as one shared window, which was the first
        # attempt and was quietly useless: the bots print thousands of fills a
        # minute, so a person's single trade was evicted from a shared buffer
        # within seconds of making it. The question being answered is "who
        # filled *me*", and that has to survive everyone else's activity.
        self.fills_log: dict[str, list[dict[str, Any]]] = {}

        # An optional observer of every print, with both sides named. `None`
        # by default and costing one attribute test per trade, because a market
        # should not pay for a measurement nobody asked for.
        #
        # It exists because identities cannot be recovered afterwards. Prices
        # and mids can be sampled from outside at any time: that is what
        # `research.recorder` does, and being outside is what makes it unable
        # to perturb what it measures. But an order id resolves to an agent
        # only while the order is still in the book, so *who traded* has to be
        # taken here or not at all, which is the same reason
        # `_record_counterparties` runs where it does.
        #
        # The entry passed is the one in `fills_log` and is shared with the
        # other side of the trade. An observer must not mutate it.
        self.trade_observer: Callable[[dict[str, Any]], None] | None = None

    # -- listing and accounts ---------------------------------------------

    def list_instrument(self, instrument: Instrument) -> None:
        self.registry.list_instrument(instrument)
        self._engines[instrument.symbol] = MatchingEngine(instrument.symbol)
        # Listed instruments open continuous, because that is what every
        # existing caller expects. A venue that wants an opening auction asks
        # for one with `begin_session`.
        self._phase[instrument.symbol] = SessionState.CONTINUOUS

    def engine(self, symbol: str) -> MatchingEngine:
        return self._engines[symbol]

    def account(self, agent_id: AgentId) -> Account:
        account = self._accounts.get(agent_id)
        if account is None:
            account = Account(
                agent_id=str(agent_id),
                starting_cash=self._balances.get(agent_id, self.starting_cash),
            )
            self._accounts[agent_id] = account
        return account

    def open_account(self, agent_id: AgentId, balance: Decimal | int) -> Account:
        """Create an account with its own opening balance.

        Refused once the account exists. An account's ``starting_cash`` is what
        every profit-and-loss figure and the conservation check are measured
        against, so changing it after the fact would rewrite history rather
        than add capital.
        """
        if agent_id in self._accounts:
            raise ValueError(
                f"{agent_id} already has an account; its opening balance is what "
                "its whole PnL is measured against and cannot be restated"
            )
        self._balances[agent_id] = to_money(balance)
        return self.account(agent_id)

    @property
    def accounts(self) -> dict[AgentId, Account]:
        return dict(sorted(self._accounts.items()))

    # -- marking -----------------------------------------------------------

    def mark(self, symbol: str) -> Money:
        """The price open positions are valued at, in minor units.

        Mid when the book is two-sided, otherwise the last trade, otherwise the
        midpoint of the contract's settlement bounds, and in every case held
        inside whatever side of the touch is standing.

        The final fallback matters more than it looks: a contract that has never
        traded still has to be marked, and marking it at zero would report every
        short as instantly profitable.

        The clamp matters for a subtler reason. A last trade is a fact about
        the past and a resting order is an offer about the present, so when
        they disagree the resting order wins: if someone is bidding 4682 right
        now, the position is worth at least 4682 whatever the tape says,
        because that is a price you can hit. Without it a one-lot print can
        drag the mark away from a touch that a thousand lots are standing at,
        which is exactly what a stale or unrepresentative print is, and why
        real venues bound settlement prices by the book rather than taking the
        last price as given.
        """
        instrument = self.registry.require(symbol)
        tick = instrument.tick_in_minor
        book = self._engines[symbol].book.snapshot()
        crossed = (
            book.best_bid is not None
            and book.best_ask is not None
            and int(book.best_bid) > int(book.best_ask)
        )
        if not crossed and book.best_bid is not None and book.best_ask is not None:
            # Averaged in minor units rather than in ticks, so a one-tick spread
            # marks at the true midpoint instead of being floored to the bid.
            return Money((int(book.best_bid) * tick + int(book.best_ask) * tick) // 2)

        last = self._last.get(symbol)
        if last is not None:
            reference = int(last) * tick
        else:
            low, high = self.bounds_in_minor(instrument)
            reference = (int(low) + int(high)) // 2
        # Not clamped into a crossed touch. A book in a call phase is crossed
        # on purpose (orders accumulate without matching) so "inside the touch"
        # is an empty interval, and forcing a number into it produced a mark
        # *below* the standing bid, which is the opposite of what the clamp
        # exists to prevent.
        if not crossed:
            if book.best_bid is not None:
                reference = max(reference, int(book.best_bid) * tick)
            if book.best_ask is not None:
                reference = min(reference, int(book.best_ask) * tick)
        return Money(reference)

    def bounds_in_minor(self, instrument: Instrument) -> tuple[Money, Money]:
        """The claim's remaining range, after whatever it has already paid.

        The instrument declares the range it had when it was written. The venue
        is the only thing that knows how much of it has since been handed over,
        so the adjustment lives here rather than on the contract, which must
        stay immutable, because every price ever printed against it was an
        opinion about the terms as written.
        """
        low, high = instrument.bounds_in_minor
        paid = int(self._distributed.get(instrument.symbol, Money(0)))
        if not paid:
            return (low, high)
        return (Money(int(low) - paid), Money(int(high) - paid))

    def marks(self) -> dict[str, Money]:
        return {symbol: self.mark(symbol) for symbol in self.registry.symbols}

    def mark_price(self, symbol: str) -> Decimal:
        """The mark as a human-readable price. Reporting only."""
        return from_money(self.mark(symbol))

    # -- trading -----------------------------------------------------------

    def submit(self, agent_id: AgentId, symbol: str, command: Command) -> list[Event]:
        """Route a command to a symbol's book, after checking it is affordable."""
        instrument = self.registry.get(symbol)
        if instrument is None or symbol not in self._engines:
            return [
                Rejected(SequenceNumber(0), agent_id, RejectReason.UNKNOWN_ORDER)
            ]

        # A stopped participant may still cancel. Refusing that too would trap
        # it in the orders it already has, which is the opposite of what
        # stopping it is for.
        if agent_id in self.halted_participants and isinstance(
            command, (Submit, Replace)
        ):
            return [
                Rejected(
                    SequenceNumber(0),
                    agent_id,
                    RejectReason.PARTICIPANT_HALTED,
                    getattr(command, "order_id", None),
                )
            ]
        # Priced off the grid its band requires. Checked here rather than in
        # the engine because it is a listing rule (a fact about this contract
        # on this venue) and not a property of matching. Replace as well as
        # Submit. Guarding only new orders leaves a hole wide enough to drive
        # through, the same argument the collateral check twenty lines below
        # already makes, and for the same reason: a modification is a request
        # for a price exactly as an order is. An accepted bid could be replaced
        # onto a price its band forbids and rest there with nothing rejected.
        priced = getattr(command, "price", None)
        if isinstance(command, Replace):
            priced = command.new_price
        if isinstance(command, (Submit, Replace)) and priced is not None:
            if not instrument.on_grid(instrument.from_ticks(priced)):
                return [
                    Rejected(
                        SequenceNumber(0),
                        agent_id,
                        RejectReason.INVALID_PRICE,
                        getattr(command, "order_id", None),
                    )
                ]
            # And inside the range the contract can settle in, which nothing
            # checked at all. Measured: a bid at -400 ticks on a contract
            # bounded by [0, 40,000] was acknowledged and rested, and a bid at
            # 40,400 was acknowledged and *traded*, printing at 10,100 on a
            # claim that can be worth at most 10,000, and dragging the mark of
            # every position in the symbol to 10,100,000,000 with it.
            #
            # Collateral structurally cannot catch this, which is why it needs
            # a listing rule of its own. The requirement is the worst case over
            # the settlement range, so a bid *below* the floor scores as the
            # safest order on the book: it can only gain. The venue's central
            # safety mechanism rates the impossible order as the safe one.
            #
            # Checked against the range as it stands now, in the same minor
            # units the collateral is computed in, rather than against the
            # range the contract was written with. A share that has paid out
            # part of its value is worth less by exactly what it paid, and
            # quoting the un-narrowed range would let an order be entered above
            # what the claim can still deliver.
            #
            # Inclusive at both ends: a settlement value can legitimately land
            # on a bound, so a price there is a price the contract can pay.
            #
            # It governs entry and nothing else. An order already resting when
            # a distribution narrows the range underneath it stays exactly
            # where it is: the venue does not reprice an order whose owner
            # named a price (the same argument that keeps a limit order out of
            # the circuit breaker's collar) and pulling it would close a
            # position its owner never asked to close. What follows the range
            # down instead is the collateral, which `distribute` re-posts
            # against the new bounds in the same instant it pays.
            low, high = self.bounds_in_minor(instrument)
            if not int(low) <= int(instrument.price_in_minor(priced)) <= int(high):
                return [
                    Rejected(
                        SequenceNumber(0),
                        agent_id,
                        RejectReason.INVALID_PRICE,
                        getattr(command, "order_id", None),
                    )
                ]

        # Sized on the lot the contract is listed in, which is the other half
        # of the same listing rule and was the half nobody enforced. The
        # instrument declares "tick / lot: the grid the exchange enforces" and
        # only the tick was ever checked: measured on a contract listed in lots
        # of ten, an order for **seven** was acknowledged and rested, and would
        # have traded. A quantity that cannot exist is exactly as unquotable as
        # a price that cannot exist, and a rule the venue does not enforce is
        # documentation rather than a listing rule.
        sized = getattr(command, "quantity", None)
        if isinstance(command, Replace):
            sized = command.new_quantity
        if isinstance(command, (Submit, Replace)) and sized is not None:
            if int(sized) % instrument.lot_size:
                return [
                    Rejected(
                        SequenceNumber(0),
                        agent_id,
                        RejectReason.INVALID_QUANTITY,
                        getattr(command, "order_id", None),
                    )
                ]

        # A cancel is counted against the allowance but is never refused for
        # it. Refusing one traps the participant in the orders it already has
        # (unable to place, unable to withdraw, holding exposure nobody is
        # permitted to manage), which is exactly the outcome the kill switch
        # thirty lines above goes out of its way to avoid, arriving by the
        # other control. Measured before this: a participant at a cap of five
        # sent five orders, then every attempt to pull one came back
        # RATE_LIMITED, and fifty lots stayed standing in the book through five
        # retries. A cancel also only ever *reduces* the venue's work and the
        # participant's risk, so refusing it is the one refusal that makes both
        # sides worse.
        if self._rate_limited(agent_id, reducing=isinstance(command, Cancel)):
            return [
                Rejected(
                    SequenceNumber(0),
                    agent_id,
                    RejectReason.RATE_LIMITED,
                    getattr(command, "order_id", None),
                )
            ]
        # A contract's own terms say when it stops trading, and until now
        # nothing enforced them: the expiry sat on the instrument as
        # documentation while the book carried on past it. Once the observation
        # window has closed the outcome is determined, so anyone still trading
        # is trading against an answer that already exists.
        self._enforce_lifecycle(symbol)

        if not self.session(symbol).accepts_orders and isinstance(
            command, (Submit, Replace)
        ):
            # Cancels stay legal after the close so an agent can tidy up; new
            # risk cannot be taken once the outcome is determined.
            return [Rejected(SequenceNumber(0), agent_id, RejectReason.ALREADY_TERMINAL)]

        # A call phase is defined by the fact that nothing matches in it, and a
        # replace is the one command that breaks that definition. The engine's
        # replace pulls the old order and re-runs the match on the replacement,
        # unconditionally (it never consults the phase) so a halted book
        # traded: measured, a replace during a halt printed 20 lots at 17,000
        # against an order that was only resting there because the auction had
        # not run yet.
        #
        # Worse, market-on-open orders rest at the sentinel price so that they
        # cross every candidate the auction considers. A replace during a call
        # phase matches against those in continuous fashion, so it printed
        # trades at **-4,611,686,018,427,387,904**, the same catastrophe the
        # engine's unfilled-market-order sweep was written to prevent, reached
        # through a door that sweep does not cover.
        #
        # Refused rather than accumulated, which is what the engine already
        # does with the other instructions that mean "right now" in a phase
        # that has no right now. A participant that wants different terms in an
        # auction cancels and re-enters.
        if isinstance(command, Replace) and not self.session(symbol).matches_continuously:
            return [
                Rejected(
                    SequenceNumber(0),
                    agent_id,
                    RejectReason.NOT_ACCEPTED_IN_AUCTION,
                    command.order_id,
                )
            ]

        # Replace is checked as well as Submit. Guarding only new orders leaves
        # a hole wide enough to drive through: an agent could work ten lots,
        # then replace them with five hundred at a worse price and take on
        # exposure the account was never able to cover. A modification is a
        # request for risk exactly like an order is.
        if isinstance(command, (Submit, Replace)) and not self._affordable(
            agent_id, instrument, command
        ):
            return [
                Rejected(
                    SequenceNumber(0),
                    agent_id,
                    RejectReason.INSUFFICIENT_COLLATERAL,
                    getattr(command, "order_id", None),
                )
            ]

        self._apply_band(symbol)
        events = self._engines[symbol].apply(command)
        self._check_limit_state(symbol)
        self._book_fills(symbol, instrument, events)
        # Delivered to everyone the events belong to, not only to the agent who
        # sent the command. A trade has two sides, and the resting one is
        # somebody else's: booking the batch under the sender alone meant the
        # passive side's fill was looked up in the wrong agent's record, found
        # nothing, and did nothing. So a maker's order stayed in the venue's
        # working book forever after the engine had removed it.
        #
        # Measured on a maker quoting two lots a round and getting lifted every
        # round: after 120 rounds the venue believed it was working **120
        # orders for 240 lots** while the engine's book held none, and
        # collateral was reserved against every one of them. With a million in
        # capital the maker was refused for insufficient collateral at round
        # 47, holding 497,100 of free cash and nothing at all in the book: the
        # account charged twice for a risk it holds once.
        self._track_all_working(symbol, events, default_owner=agent_id)
        return events

    def _affordable(
        self, agent_id: AgentId, instrument: Instrument, command: Submit
    ) -> bool:
        """Could the account survive every one of its working orders filling?

        Checking only the position an order would create is not enough, and the
        gap is not academic: a market maker works a bid and an ask at once, each
        individually affordable, and ends up over-committed when both fill. The
        symptom is an account with negative free cash, which is a venue that
        allowed a trade it could not collateralise.

        So the test is over *scenarios*, not over one order. Collateral is
        reserved against the worse of the two directional extremes:

            every working buy fills   ->  position + total working buy quantity
            every working sell fills  ->  position - total working sell quantity

        Only one of those can be the adverse one, so the maximum of the two is
        both sufficient and not over-conservative. Working orders are priced at
        their own limits; a market order is priced at the far end of the
        settlement range, because it can fill anywhere and the only honest
        assumption is the worst price it could get.
        """
        symbol = instrument.symbol
        account = self.account(agent_id)
        bounds = self.bounds_in_minor(instrument)
        working = dict(self._working.get((agent_id, symbol), {}))

        if isinstance(command, Replace):
            existing = working.pop(command.order_id, None)
            if existing is None:
                # Nothing of ours to replace. Let the engine reject it for the
                # right reason rather than pre-empting with a collateral error.
                return True
            side = existing[0]
            quantity = int(command.new_quantity)
            price = (
                instrument.price_in_minor(command.new_price)
                if command.new_price is not None
                else Money(existing[2])
            )
        elif getattr(command, "stop_price", None) is not None:
            # A stop is reserved for at its trigger, from the moment it is
            # parked. It has to be: the engine releases a triggered stop
            # inside its own matching, which never passes back through this
            # check, so an unreserved stop would create a position the account
            # had never been asked to cover.
            #
            # A plain stop can still fill through its trigger in a fast market
            # (that is the risk its owner takes in reality too) but the collar
            # on unpriced orders bounds how far through.
            side = command.side
            price = instrument.price_in_minor(command.stop_price)
            quantity = int(command.quantity)
        elif command.price is not None:
            side = command.side
            price = instrument.price_in_minor(command.price)
            quantity = int(command.quantity)
        elif self.session(symbol).matches_continuously:
            side = command.side
            # A market order can only ever trade against resting liquidity, so
            # its exposure is bounded by the book rather than by the contract's
            # settlement range. Reserving against the far end of the range
            # (10,000 on a win-rate future quoted near 4,700) rejects orders
            # that could never have cost anything like that much, and does it
            # for a price the order was structurally incapable of paying.
            quantity, price = self._market_exposure(
                symbol, instrument, side, int(command.quantity)
            )
            if quantity == 0:
                # No liquidity to hit. The order will cancel unfilled and can
                # create no position, so there is nothing to collateralise.
                return True
        else:
            # In a call phase the book says nothing about what this will fill
            # against, and the bound above is only sound because the engine
            # walks the same levels in the same instant. A market-on-open order
            # does not trade on arrival at all: it rests until the uncross and
            # then trades against liquidity that had not arrived yet, at a
            # price the auction picks. So the honest assumption is the one the
            # rest of this method makes about an order that named no price: the
            # far end of the contract's range.
            #
            # Measured before this: an account holding 10,000 sent a
            # market-on-open buy for 100,000 lots into an empty ask side. The
            # book walk found nothing to hit, concluded there was nothing to
            # collateralise, and let it rest. Sellers then arrived, the auction
            # cleared it in full, and the account came out of its own opening
            # auction 45,000 times underwater: 450,000,000 of collateral
            # required against free cash of **-449,990,000,000,000** minor
            # units. A venue that discovers insolvency after the trade has
            # printed cannot unprint it.
            side = command.side
            quantity = int(command.quantity)
            price = Money(int(bounds[1]) if side is Side.BUY else int(bounds[0]))

        position = account.positions.get(symbol)
        current = position.quantity if position else 0

        # `working` already has the order being replaced removed, so a
        # modification is measured as the difference it makes rather than as
        # additional exposure on top of what it supersedes.
        buys = sum(q for s, q, _p in working.values() if s is Side.BUY)
        sells = sum(q for s, q, _p in working.values() if s is Side.SELL)
        if side is Side.BUY:
            buys += quantity
        else:
            sells += quantity

        # Price each scenario at the worst working price on that side, so a
        # cheap order cannot subsidise an expensive one.
        prices = [p for _s, _q, p in working.values()] + [int(price)]
        worst_buy = Money(max(prices))
        worst_sell = Money(min(prices))

        # Each scenario is charged against the *basis* the position would end
        # up holding, not against its quantity priced at the trade that
        # finished it. The two differ whenever a fill adds to a position at a
        # different price, and they differ in the dangerous direction: the cost
        # of what is already held is simply missing from ``resulting *
        # incoming_price``.
        #
        # Measured through this order path, on an account holding 50,500 with
        # bounds of [0, 10,000]: ten lots long at 5,000, then ten more at 100.
        # The scenario was evaluated as ``20 * 100 = 2,000`` against a position
        # whose basis came out at 51,000, the order was accepted, and the
        # account finished with collateral of 51,000,000,000 minor units
        # against 50,500,000,000 of cash, **free cash of -500,000,000**, which
        # is an account owing money it does not have on a venue whose whole
        # claim is that it cannot.
        #
        # `basis_after` is a pure projection of `Position.apply_fill`, held to
        # it by a property test over random fill sequences, so the figure
        # checked here is the figure the fill will produce rather than a second
        # implementation of it.
        released = int(account.collateral.get(symbol, Money(0)))
        # The fee the fill will cost, at the aggressor rate. A resting order
        # may well earn the maker rebate instead, but which side of the trade
        # this order ends up on is not knowable when it is accepted, and the
        # check has to hold for the worse of the two.
        buy_fee = int(self.fees.charge(abs(buys * int(worst_buy)), aggressor=True))
        sell_fee = int(self.fees.charge(abs(sells * int(worst_sell)), aggressor=True))
        # Orders this agent has resting in other symbols are already spoken
        # for. Their collateral is not posted yet, because nothing has filled,
        # but it is not available to this order either.
        elsewhere = self._reserved_elsewhere(agent_id, symbol)
        if self._survives(
            account, position, buys, worst_buy, current + buys, released, bounds,
            buy_fee, elsewhere,
        ) and self._survives(
            account, position, -sells, worst_sell, current - sells, released, bounds,
            sell_fee, elsewhere,
        ):
            return True
        if not self.netting:
            return False

        # The per-contract figure says no. Ask the portfolio.
        #
        # Charging each contract its own worst case assumes the world can be
        # simultaneously terrible for all of them, and for contracts on the
        # same underlying it cannot: they are functions of the same number. The
        # arbitrageur pays this most, because its whole business is holding
        # packages that offset. A conversion posts collateral on all three legs
        # and can lose nothing at any level.
        #
        # This is exact rather than a model. See `arena/portfolio/netting.py`.
        return self._portfolio_affords(agent_id, instrument, side, quantity, price)

    @classmethod
    def _survives(
        cls,
        account: Account,
        position: Any,
        quantity: int,
        price: Money,
        resulting: int,
        released: int,
        bounds: tuple[Money, Money],
        fee: int,
        reserved: int = 0,
    ) -> bool:
        """Whether the account still covers itself after one scenario fills.

        Three things move, and for a long time only two of them were counted.
        The scenario posts collateral against the position it creates (that was
        counted) and it *realises* whatever the closing part of it made or
        lost, which comes straight out of cash the moment the fill books.
        Checking the requirement against the cash the account has now compares
        it with money the trade is about to take away.

        Measured over four hundred fills sweeping the whole settlement range:
        an account short four lots at an average of 50 bought eleven at 9,500.
        The flip realised a loss of **37,800,000,000** minor units, which the
        check never saw, so 66,500,000,000 of collateral was approved against
        104,130,530,000 of cash that became 66,309,630,000 the instant it
        filled, free cash of **-190,370,000**. Nine of thirty random runs
        finished with some account underwater.

        The third thing is the fee, and it was missing for the same reason the
        realised figure once was: ``apply_fill`` does ``cash += realised -
        fee``, so a fill approved at exactly zero free cash goes negative the
        instant it books. Measured before this counted it, on a 300 second
        run of the default market, ``fund-5`` finished holding 40,000,000 with
        free cash of **-8,358**. Small, and the size is not the point: an
        account whose posted collateral exceeds its cash is an account whose
        collateral is not there, on a venue whose entire claim is that it is.

        Charged at the aggressor rate. A resting order may earn the maker
        rebate instead, and under ``MAKER_TAKER`` that rebate is negative, so
        assuming it would make the check *less* conservative than the outcome.
        Which side of the trade an order lands on is not knowable when it is
        accepted, so the check takes the worse one.

        The realised figure is derived from ``basis_after`` rather than
        computed alongside it, and the derivation is an identity that holds in
        every branch of ``apply_fill`` (opening, reducing and flipping alike):

            basis_after  =  basis_before  +  quantity * price  +  realised

        Reading it the other way round gives the line below. So there is still
        exactly one projection of the fill here, and it is the one the
        portfolio layer pins to ``apply_fill`` with a property test.
        """
        return int(account.free_cash) - reserved >= cls._need(
            account, position, quantity, price, resulting, released, bounds, fee
        )

    @classmethod
    def _need(
        cls,
        account: Account,
        position: Any,
        quantity: int,
        price: Money,
        resulting: int,
        released: int,
        bounds: tuple[Money, Money],
        fee: int,
    ) -> int:
        """Cash this scenario consumes: what it must post, net of what it frees.

        The same arithmetic `_survives` has always done, named and returned
        instead of compared, because an order resting in another symbol needs
        the figure rather than the verdict. Positive means the scenario costs
        the account money it has to have already; zero or less means the
        scenario releases more than it takes, which a closing order does.
        """
        projected = cls._basis_after(position, quantity, price)
        basis_now = int(position.cost_basis) if position is not None else 0
        realised = int(projected) - quantity * int(price) - basis_now
        required = int(account.collateral_for_basis(resulting, projected, bounds))
        return required - released - realised + fee

    def _reserve_for(self, agent_id: AgentId, symbol: str) -> int:
        """What this agent's working orders in one symbol have to be backed by.

        The worse of the two directional scenarios, which is the same choice
        `_affordable` makes and for the same reason: only one of them can be
        the adverse one, so charging both would charge a two-sided quote twice
        for a risk it runs once.

        Floored at zero. An order that would close a position releases
        collateral rather than consuming it, and letting that show up as a
        negative reserve would let a resting exit *fund* new exposure
        elsewhere, on the strength of a fill that has not happened.
        """
        working = self._working.get((agent_id, symbol))
        if not working:
            return 0
        instrument = self.registry.get(symbol)
        account = self._accounts.get(agent_id)
        if instrument is None or account is None:
            return 0
        bounds = self.bounds_in_minor(instrument)
        position = account.positions.get(symbol)
        current = position.quantity if position else 0
        released = int(account.collateral.get(symbol, Money(0)))
        buys = sum(q for s, q, _p in working.values() if s is Side.BUY)
        sells = sum(q for s, q, _p in working.values() if s is Side.SELL)
        prices = [p for _s, _q, p in working.values()]

        need = 0
        if buys:
            worst = Money(max(prices))
            fee = int(self.fees.charge(abs(buys * int(worst)), aggressor=True))
            need = max(
                need,
                self._need(
                    account, position, buys, worst, current + buys, released,
                    bounds, fee,
                ),
            )
        if sells:
            worst = Money(min(prices))
            fee = int(self.fees.charge(abs(sells * int(worst)), aggressor=True))
            need = max(
                need,
                self._need(
                    account, position, -sells, worst, current - sells, released,
                    bounds, fee,
                ),
            )
        return max(0, need)

    def _restate_reserve(self, agent_id: AgentId, symbol: str) -> None:
        """Re-price one symbol's reserve and carry the difference to the total.

        Called wherever the three things it depends on move: the working orders
        themselves, and the position and posted collateral a fill changes. Every
        one of those arrives as an event batch that `_track_working` sees, so
        this hangs off that rather than off three separate call sites that could
        each be forgotten independently.
        """
        key = (agent_id, symbol)
        previous = self._reserve.get(key, 0)
        current = self._reserve_for(agent_id, symbol)
        if current == previous:
            return
        if current:
            self._reserve[key] = current
        else:
            self._reserve.pop(key, None)
        total = self._reserved.get(agent_id, 0) + current - previous
        if total:
            self._reserved[agent_id] = total
        else:
            self._reserved.pop(agent_id, None)

    def _reserved_elsewhere(self, agent_id: AgentId, symbol: str) -> int:
        """The agent's open-order reserve outside the symbol being checked.

        The symbol's own reserve is excluded because `_affordable` puts those
        orders into its scenario directly; counting them here as well would
        charge them twice and refuse a maker its second quote.
        """
        return self._reserved.get(agent_id, 0) - self._reserve.get(
            (agent_id, symbol), 0
        )

    @staticmethod
    def _basis_after(position: Any, quantity: int, price: Money) -> Money:
        """The basis a scenario would leave behind, from flat or from a holding.

        A thin wrapper so the two scenarios read as one line each, and so an
        account with nothing in the symbol yet takes the same path as one that
        already holds something. From flat the answer is just what the fill
        pays, which is what the old figure assumed was always true.
        """
        if position is None:
            return Money(quantity * int(price))
        return position.basis_after(quantity, price)

    def _portfolio_affords(
        self,
        agent_id: AgentId,
        instrument: Instrument,
        side: Side,
        quantity: int,
        price: Money,
    ) -> bool:
        """Whether the account covers its worst case with this order added.

        Grouped by underlying, because only positions on the same underlying
        are functions of the same number. Two competitors are two numbers, and
        netting across them would need a correlation, which would be an
        estimate, and would give away the one thing this collateral model has.

        This asks its question about *positions*, so it does not see the
        open-order reserve the per-contract check applies, and an order that
        check refuses can still be admitted here past orders the account has
        resting elsewhere. That is one more reason `netting` is off by default
        rather than a separate defect: the fix is to charge per netting group
        in the gate and the ledger alike, at which point working orders belong
        inside the group's worst case rather than beside it. Patching this
        method on its own would leave two collateral models disagreeing about
        the same account. docs/GAPS.md carries the measurements.
        """
        from arena.portfolio.netting import worst_case

        account = self.account(agent_id)
        signed = quantity if side is Side.BUY else -quantity

        groups: dict[str, list[tuple[Any, int, Decimal]]] = {}
        for symbol, position in account.positions.items():
            listed = self.registry.get(symbol)
            if listed is None or position.quantity == 0:
                continue
            average = Decimal(int(position.cost_basis) // position.quantity) / MONEY_SCALE
            groups.setdefault(_underlying_of(listed), []).append(
                (listed.spec, int(position.quantity), average)
            )

        groups.setdefault(_underlying_of(instrument), []).append(
            (instrument.spec, signed, Decimal(int(price)) / MONEY_SCALE)
        )

        needed = sum(worst_case(holdings) for holdings in groups.values())
        return Decimal(int(account.cash)) / MONEY_SCALE >= needed

    def _market_exposure(
        self, symbol: str, instrument: Instrument, side: Side, quantity: int
    ) -> tuple[int, Money]:
        """How much a market order could fill, and the worst price it could pay.

        Walks the opposite side of the book. The answer is exact rather than
        conservative because the engine will walk exactly the same levels in
        exactly the same order: a market order cannot reach a price nobody is
        quoting, and cannot fill more than is resting.
        """
        book = self._engines[symbol].book.snapshot(levels=1 << 20)
        levels = book.asks if side is Side.BUY else book.bids

        filled = 0
        worst: Price | None = None
        for price, available in levels:
            filled += min(quantity - filled, int(available))
            worst = price
            if filled >= quantity:
                break

        if filled == 0 or worst is None:
            return 0, Money(0)
        return filled, instrument.price_in_minor(worst)

    def _track_all_working(
        self,
        symbol: str,
        events: list[Event],
        default_owner: AgentId | None = None,
    ) -> None:
        """Book a batch of events into the working record of whoever owns them.

        An event batch belongs to everybody who was in it, which is obvious for
        an auction and was missed for continuous trading: a match produces a
        fill for the incoming order and one for the resting order, and those
        are two different participants.
        """
        by_owner: dict[AgentId, list[Event]] = (
            {default_owner: []} if default_owner is not None else {}
        )
        for event in events:
            owner = getattr(event, "agent_id", None)
            if owner is not None:
                by_owner.setdefault(owner, []).append(event)
        for owner, owned in by_owner.items():
            self._track_working(owner, symbol, owned)

    def _track_working(
        self, agent_id: AgentId, symbol: str, events: list[Event]
    ) -> None:
        """Maintain the book of orders this agent has working.

        Reserving against working orders is the whole point of the affordability
        scenario above, so this has to stay in step with the engine's view.
        """
        book = self._working.setdefault((agent_id, symbol), {})
        for event in events:
            if isinstance(event, Acknowledged):
                priced = self._working_price(symbol, event)
                if priced is not None:
                    book[event.order_id] = (
                        event.side,
                        int(event.quantity),
                        priced,
                    )
            elif isinstance(event, Filled):
                existing = book.get(event.order_id)
                if existing is not None:
                    remaining = int(event.remaining)
                    if remaining <= 0:
                        book.pop(event.order_id, None)
                    else:
                        book[event.order_id] = (existing[0], remaining, existing[2])
            elif isinstance(event, Replaced):
                # The engine keeps the order id across a replace, whether or
                # not queue priority survived, so the entry is updated in place.
                # Leaving the old size and price here would reserve collateral
                # against an order that no longer exists.
                #
                # The side comes from the event when there is no entry to read
                # it from. Defaulting to BUY was a guess that is wrong half the
                # time, and it had a real path to it: a pegged order with no
                # reference acknowledges no price, so nothing is tracked, and
                # its first reprice then booked a sell as a buy, reserving
                # against the wrong end of the range entirely.
                book[event.order_id] = (
                    self._side_of(symbol, event.order_id, book),
                    int(event.quantity),
                    int(self.registry.require(symbol).price_in_minor(event.price)),
                )
            elif isinstance(event, Cancelled):
                book.pop(getattr(event, "order_id", None), None)
            elif isinstance(event, Rejected):
                # A refusal is not a removal, and treating it as one lost the
                # venue's only record of a live order. The engine refuses a
                # replace it dislikes and leaves the original exactly where it
                # was: `Replace(order, quantity=0)` came back INVALID_QUANTITY,
                # the order stayed resting for 30 lots, and the venue forgot it
                # existed. `kill` then reported no symbols and left the order
                # standing in the book, which is the one control that is meant
                # to always work doing nothing at all.
                #
                # The engine is the authority on whether anything went away, so
                # ask it. A rejection that *did* terminate an order (post-only
                # that would have crossed, fill-or-kill that could not, an
                # immediate order refused by a call phase) is acknowledged
                # before it is refused, so the record must still be dropped for
                # those or it becomes a phantom.
                order_id = getattr(event, "order_id", None)
                if order_id is not None and order_id in book:
                    order = self._engines[symbol].book.get(order_id)
                    if order is not None and not order.is_resting:
                        book.pop(order_id, None)
        self._restate_reserve(agent_id, symbol)

    def _working_price(self, symbol: str, event: Acknowledged) -> int | None:
        """What to reserve an acknowledged order against, in minor units.

        Its own price when it named one. An order that named none is either
        parked with no price yet (a peg with no reference, which is not in any
        book and can create no position until it is) or a market-on-open order,
        which is resting right now and will trade at whatever the auction
        clears at. The book tells the two apart, and the second is reserved
        against the far end of the contract's range because that is the worst
        price an order that named none could get.
        """
        instrument = self.registry.require(symbol)
        if event.price is not None:
            return int(instrument.price_in_minor(event.price))
        order = self._engines[symbol].book.get(event.order_id)
        if order is None or not order.is_resting:
            return None
        low, high = self.bounds_in_minor(instrument)
        return int(high) if event.side is Side.BUY else int(low)

    def _side_of(self, symbol: str, order_id: OrderId, book: dict) -> Side:
        """Which side a replaced order is on, asked of whoever still knows.

        The venue's own record first, then the engine's book. Defaulting to BUY
        was a guess that is wrong half the time, and there is a real path to
        it: a pegged order with no reference acknowledges no price, so the
        venue tracks nothing, and its first reprice then booked a sell as a
        buy, reserving against the wrong end of the settlement range entirely.
        """
        existing = book.get(order_id)
        if existing is not None:
            return existing[0]
        engine = self._engines.get(symbol)
        order = engine.book.get(order_id) if engine is not None else None
        return order.side if order is not None else Side.BUY

    def _book_fills(
        self,
        symbol: str,
        instrument: Instrument,
        events: list[Event],
        auction: bool = False,
    ) -> None:
        bounds = self.bounds_in_minor(instrument)
        charged = 0
        for event in events:
            if isinstance(event, Filled):
                signed = int(event.quantity) * (1 if event.side is Side.BUY else -1)
                price = instrument.price_in_minor(event.price)
                fee = self.fees.charge(
                    abs(int(price) * int(event.quantity)), event.aggressor, auction
                )
                charged += int(fee)
                self.account(event.agent_id).apply_fill(
                    symbol, signed, price, bounds, fee
                )
            elif isinstance(event, Traded):
                self._last[symbol] = event.price
                self._record_counterparties(symbol, event)
                self._check_price_band(symbol, event.price)
        if charged:
            # The exact counterpart of what the participants paid, so the two
            # sides of every fee cancel to the unit.
            treasury = self.account(FEE_ACCOUNT_ID)
            treasury.cash = Money(int(treasury.cash) + charged)
            self.fees_collected = Money(int(self.fees_collected) + charged)

    # -- lifecycle ---------------------------------------------------------

    def _record_counterparties(self, symbol: str, trade: Traded) -> None:
        """Name both sides of a print, while the order ids still resolve."""
        book = self._engines[symbol].book
        buyer = book.get(trade.buy_order_id)
        seller = book.get(trade.sell_order_id)
        if buyer is None or seller is None:
            return
        entry = {
            "symbol": symbol,
            "quantity": int(trade.quantity),
            "price": int(trade.price),
            "buyer": str(buyer.agent_id),
            "seller": str(seller.agent_id),
            "aggressor": trade.aggressor_side.value,
        }
        for side in (entry["buyer"], entry["seller"]):
            log = self.fills_log.setdefault(side, [])
            log.append(entry)
            if len(log) > 60:
                del log[:-60]
        if self.trade_observer is not None:
            self.trade_observer(entry)

    def counterparties_for(self, agent_id: AgentId, limit: int = 40) -> list[dict[str, Any]]:
        """The other side of this agent's recent fills, most recent first."""
        who = str(agent_id)
        out: list[dict[str, Any]] = []
        for entry in reversed(self.fills_log.get(who, ())):
            mine_is_buy = entry["buyer"] == who
            instrument = self.registry.get(entry["symbol"])
            out.append(
                {
                    "symbol": entry["symbol"],
                    "side": "buy" if mine_is_buy else "sell",
                    "quantity": entry["quantity"],
                    "price": (
                        str(instrument.from_ticks(Price(entry["price"])))
                        if instrument is not None
                        else str(entry["price"])
                    ),
                    "counterparty": entry["seller"] if mine_is_buy else entry["buyer"],
                }
            )
            if len(out) >= limit:
                break
        return out

    # -- sessions ----------------------------------------------------------

    def session(self, symbol: str) -> SessionState:
        return self._phase.get(symbol, SessionState.CONTINUOUS)

    def _set_phase(self, symbol: str, state: SessionState) -> None:
        self._phase[symbol] = state
        # A venue whose mechanism is not a matching engine (the scoring rule)
        # keeps the phase without an engine to push it into.
        engine = self._engines.get(symbol)
        if engine is not None:
            engine.phase = state

    def _enforce_lifecycle(self, symbol: str) -> bool:
        """Close a symbol the contract no longer permits trading in.

        The rule itself is old (once the observation window has closed the
        outcome is determined, so anyone still trading is trading against an
        answer that already exists) but it was enforced on exactly one path,
        the arrival of an order. Every other way the venue acts on a symbol
        skipped it, and a halted symbol is precisely the one nobody sends
        orders to.

        Measured: a symbol paused by the circuit breaker whose window closed
        while it was paused was still reported by `reopen_due`, and the reopen
        auction printed 40 lots at 18,800 on a contract whose outcome was
        already known, then put it back into continuous trading.

        A settled symbol is the same statement made by the other clock. Nothing
        may trade in a contract that has already paid out: it can never settle
        again, so the position would be marked forever and realised never.
        """
        if self.session(symbol) is SessionState.CLOSED:
            return True
        if symbol in self._settled:
            self._set_phase(symbol, SessionState.CLOSED)
            return True
        instrument = self.registry.get(symbol)
        if (
            instrument is not None
            and self._clock is not None
            and self._clock() >= instrument.expiry
        ):
            self._set_phase(symbol, SessionState.CLOSED)
            return True
        return False

    def begin_session(self, symbol: str) -> None:
        """Put a symbol into its opening call phase. Orders rest, nothing trades."""
        self.registry.require(symbol)
        if symbol in self._settled:
            # Loud, because this one can only be a mistake. Opening a call
            # phase on a contract that has already paid out let orders
            # accumulate and the next uncross crossed them: measured, a
            # participant came out of it holding 10 lots of a contract that can
            # never settle again, because `settle` refuses to fire twice.
            raise ValueError(
                f"{symbol} has already settled; opening a session on it would "
                "let positions be taken in a contract that can never pay them out"
            )
        if self._enforce_lifecycle(symbol):
            # Expired rather than mistaken. The calendar closed it, and a call
            # phase cannot reopen what the calendar closed.
            return
        self._set_phase(symbol, SessionState.PRE_OPEN)

    def indicative(self, symbol: str) -> AuctionResult | None:
        """What the auction would clear at right now, changing nothing.

        Published during a call phase as real venues publish indicative prices
        and opening imbalances, so an agent can respond to the auction rather
        than only to its result.
        """
        return indicative_auction(
            self._engines[symbol].book, self._reference.get(symbol)
        )

    def uncross(self, symbol: str) -> AuctionResult | None:
        """Clear the call phase and return to continuous trading."""
        result, _events = self.uncross_events(symbol)
        return result

    def uncross_events(self, symbol: str) -> tuple[AuctionResult | None, list[Event]]:
        """The same, and the events it produced, so they can be delivered.

        An auction fills real orders belonging to real participants, and until
        this existed nobody told them. The ledger moved, the agents did not
        hear, and every one of them then traded on a position it did not have:
        measured after two minutes, **342 of 483** (agent, symbol) pairs had an
        agent's own belief about its position disagreeing with the venue's
        record. A market maker skewing its quotes off an inventory that is not
        its inventory is not managing risk, it is guessing.

        The fills go through exactly the same account path as continuous ones,
        so collateral, fees and conservation cannot diverge between the two: an
        auction that settled through its own accounting would be the ideal
        place for a leak to hide.
        """
        instrument = self.registry.require(symbol)
        if self._enforce_lifecycle(symbol):
            # Nothing to cross. A contract past its window or already settled
            # has an answer, and an auction is a mechanism for discovering a
            # price; running one here would print trades against a number
            # everybody could already look up.
            return None, []
        engine = self._engines[symbol]
        result = indicative_auction(engine.book, self._reference.get(symbol))
        events = engine.uncross(self._reference.get(symbol))
        self._book_fills(symbol, instrument, events, auction=True)
        # Grouped by owner, because an auction's events belong to everyone who
        # was in it. Skipping this left collateral reserved against orders the
        # uncross had already filled or cancelled.
        self._track_all_working(symbol, events)
        if result is not None and result.volume > 0:
            # The reference moves to the cleared price, but the *window* is
            # already carrying it: `_book_fills` records every `Traded` the
            # uncross produced, and an auction that clears has to produce at
            # least one. Appending here as well counted a single auction print
            # twice (measured, one print at 18,600 and two entries at 18,600 in
            # the window), which double-weights the auction in the trailing
            # mean and inflates the count the halting rule waits for.
            self._reference[symbol] = result.price
        self._reopen_at.pop(symbol, None)
        self._limit_since.pop(symbol, None)
        self._set_phase(symbol, SessionState.CONTINUOUS)
        return result, events + self._release_stops_after(symbol, instrument, events)

    def _release_stops_after(
        self, symbol: str, instrument: Instrument, events: list[Event]
    ) -> list[Event]:
        """Fire the stops the auction's own prints set off.

        An uncross prints, and a print is a print whoever made it. Measured on
        the same book twice: a sell stop parked at 18,000 with a bid of 20 at
        17,900 behind it. Traded continuously the tape read ``[(10, 18000),
        (10, 17900)]`` and the stop was gone; cleared by an auction the tape
        read ``[(10, 18000)]`` and the stop was still parked. So an auction
        jumped straight over a stop, and an auction is the event most likely to
        gap through one, which is exactly why the stop was there.

        **After the phase flip, and that is the whole reason this lives here
        rather than in the engine.** `MatchingEngine.uncross` runs while the
        symbol is still in its call phase, so a stop released inside it becomes
        a market order arriving at a book that does not match: it would be
        accumulated at the sentinel price instead. That is the order that once
        printed trades at -4,611,686,018,427,387,904 and billed 4.8e22 in fees.
        The venue owns the phase, so the venue is the only place the sequence
        can be got right: uncross, reopen, then release into a continuous book.

        The band is recomputed first, because a cascade is a market order and
        market orders are collared. The reference has just moved to the cleared
        price, and collaring the cascade against the price the auction *left*
        is the whole point of having a band at all.
        """
        engine = self._engines[symbol]
        self._apply_band(symbol)
        released = engine._release_after(events)
        if not released:
            return []
        # Booked at the taker rate, not the auction rate: these orders crossed
        # a spread in continuous trading, which is what an aggressor is. Billing
        # them as part of the auction would charge a cascade at whatever the
        # cross costs, and the cascade is not part of the cross.
        self._book_fills(symbol, instrument, released)
        self._track_all_working(symbol, released)
        self._check_limit_state(symbol)
        return released

    def halt(self, symbol: str, reason: str = "manual") -> None:
        """Suspend trading. Orders keep arriving; the reopen is an auction.

        Resuming straight into continuous trading would hand the whole
        dislocation to whichever order arrived first, which is the outcome a
        halt exists to prevent.
        """
        self.registry.require(symbol)
        if self._enforce_lifecycle(symbol):
            return
        # A manual halt has no timer, so it drops whatever timer was running.
        # Without this the breaker's stale reopen time survived an operator's
        # halt and reopened it: a symbol paused by the band and then halted for
        # news was still listed by `reopen_due` the moment the *band's* pause
        # ran out, so the operator's halt quietly expired on a schedule the
        # operator never set. A halt somebody decided on ends when somebody
        # decides it ends.
        self._reopen_at.pop(symbol, None)
        self._limit_since.pop(symbol, None)
        self._set_phase(symbol, SessionState.AUCTION)
        self.halts.append({"symbol": symbol, "reason": reason})

    def _now(self) -> int:
        """Elapsed simulated nanoseconds, or zero when nothing is driving time.

        Zero is a deliberate answer rather than an error: a venue used directly
        in a unit test has no kernel, and a breaker whose clock never advances
        simply never times anything out, which is the right behaviour for a
        venue that is being poked rather than run.
        """
        return int(self.sim_clock()) if self.sim_clock is not None else 0

    def _reference_price(self, symbol: str) -> Price | None:
        """The mean of prints inside the reference window.

        The opening print until the window has filled, which is what a venue
        uses before it has five minutes of trading to average, and it is the
        auction price, so it is a price size actually transacted at.
        """
        window = self._recent.get(symbol)
        cutoff = self._now() - self.reference_window_ns
        while window and window[0][0] < cutoff:
            window.popleft()
        if window:
            return Price(sum(int(price) for _t, price in window) // len(window))
        return self._quoted_reference(symbol)

    def _quoted_reference(self, symbol: str) -> Price | None:
        """Where the market is, when nothing has traded recently.

        The mid, and only then the last cleared price. Falling straight back to
        the stored one was the obvious thing and it strands the band: a symbol
        that goes quiet keeps a reference from whenever it last printed, the
        market walks away from it, and every unpriced order is then collared
        against a price that no longer exists. Measured, the band on
        a win rate future sat at 6,392 while the book was quoting 4,760, a
        third of the way across the contract's range, and no market order
        could trade at all.

        A quote is weaker evidence than a trade, which is why it is the
        fallback rather than the rule. It is much better evidence than a price
        from a minute ago.
        """
        book = self._engines[symbol].book.snapshot()
        if book.best_bid is not None and book.best_ask is not None:
            if int(book.best_bid) <= int(book.best_ask):
                return Price((int(book.best_bid) + int(book.best_ask)) // 2)
        return self._reference.get(symbol)

    def _check_price_band(self, symbol: str, price: Price) -> None:
        """Record a print in the reference window. Nothing else.

        Prints cannot leave the band any more (the engine refuses to match
        outside it) so a print is no longer evidence of anything except where
        the market is. The limit state is judged from the *quote* instead, in
        :meth:`_check_limit_state`, which is what the rule actually says: a
        symbol is in a limit state when the best bid or offer is *at* a band,
        not when a trade has already happened beyond it.
        """
        self._reference.setdefault(symbol, price)
        if self.price_band is None:
            # The window exists to serve the band and nothing else, and it is
            # only ever trimmed by the two methods the band calls, both of
            # which return immediately when there is no band. So a venue with
            # the breaker switched off appended a print per trade and dropped
            # none: measured, 400 prints retained inside a window five minutes
            # wide that nothing was ever going to look at. This venue bounds
            # its trade log and its public log for exactly this reason.
            return
        window = self._recent.setdefault(symbol, deque())
        window.append((self._now(), price))
        cutoff = self._now() - self.reference_window_ns
        while window and window[0][0] < cutoff:
            window.popleft()

    def _check_limit_state(self, symbol: str) -> None:
        """Enter a limit state while the quote presses against a band, and pause
        if it stays there.

        Three states rather than two, which is the whole point of modelling
        this properly: quoting inside the band, pressing against it, and
        paused. One order that reaches the edge is not an outage (it is one
        order) so reaching it starts a clock and only staying there stops the
        market.
        """
        if self.price_band is None or self.session(symbol) is not SessionState.CONTINUOUS:
            return
        now = self._now()

        # A locked book is a limit state by definition, whatever the reference
        # says. Bid above offer while trading means interest that wants to
        # cross and is not allowed to, which happens when an order slid to a
        # band edge and the band later moved away from it. Nothing in
        # continuous trading can clear that; an auction can, and clearing it at
        # one price is exactly what an auction is for.
        book = self._engines[symbol].book.snapshot()
        if (
            book.best_bid is not None
            and book.best_ask is not None
            and int(book.best_bid) > int(book.best_ask)
        ):
            since = self._limit_since.get(symbol)
            if since is None:
                self._limit_since[symbol] = now
                self.halts.append(
                    {"symbol": symbol, "reason": "limit_state", "locked": True}
                )
                return
            if now - since < self.limit_state_ns:
                return
            self._limit_since.pop(symbol, None)
            self._reopen_at[symbol] = now + self.pause_ns
            self.halts.append(
                {"symbol": symbol, "reason": "price_band", "locked": True}
            )
            self._set_phase(symbol, SessionState.AUCTION)
            return

        window = self._recent.get(symbol)
        reference = self._reference_price(symbol)
        if reference is None or not window or len(window) < self.min_reference_prints:
            # Pausing a market is disruptive, so it waits for a reference
            # several prints deep. Banding an execution is protective and does
            # not: the worst it does is leave an order unfilled.
            return

        # A fraction of what the contract can be *worth*, not of what it costs.
        #
        # A percentage of price is the rule real equity venues use, and it is
        # meaningless here. A binary trading at fifty cents gets a band of two
        # and a half cents, which any ordinary tick of opinion breaks, so the
        # breaker paused every event contract on the exchange repeatedly while
        # leaving the future (whose 5% is 233 points, sixteen standard
        # deviations) untouched. Percentage bands assume a price with no
        # natural scale. Every contract here has one: its settlement range. The
        # same reasoning already governs the maker's inventory skew, and it
        # makes one parameter mean the same thing on a future and on a coin
        # flip.
        low, high = self.registry.require(symbol).tick_bounds
        allowed = abs(int(high) - int(low)) * self.price_band
        book = self._engines[symbol].book.snapshot()
        # Pressing against a band means there is interest that cannot trade: a
        # bid at or above the top of the band, or an offer at or below the
        # bottom of it. Either is the market saying it wants to be somewhere
        # the venue will not let it go.
        price = None
        if book.best_bid is not None and int(book.best_bid) - int(reference) >= allowed:
            price = book.best_bid
        elif book.best_ask is not None and int(reference) - int(book.best_ask) >= allowed:
            price = book.best_ask
        if price is None:
            self._limit_since.pop(symbol, None)
            return

        since = self._limit_since.get(symbol)
        if since is None:
            self._limit_since[symbol] = now
            self.halts.append(
                {
                    "symbol": symbol,
                    "reason": "limit_state",
                    "reference": int(reference),
                    "price": int(price),
                    "band": self.price_band,
                }
            )
            return
        if now - since < self.limit_state_ns:
            return

        self._limit_since.pop(symbol, None)
        self._reopen_at[symbol] = now + self.pause_ns
        self.halts.append(
            {
                "symbol": symbol,
                "reason": "price_band",
                "reference": int(reference),
                "price": int(price),
                "band": self.price_band,
            }
        )
        self._set_phase(symbol, SessionState.AUCTION)

    def kill(self, agent_id: AgentId, reason: str = "operator") -> list[str]:
        """Stop a participant: pull its working orders and refuse it more.

        Returns the symbols it had orders in, so an operator can see what was
        pulled rather than being told "done".
        """
        self.halted_participants[agent_id] = reason
        touched: list[str] = []
        # The venue's own cancels do not count against the participant's
        # message allowance, and that is the whole point of the exception.
        #
        # A runaway algorithm is by definition at its cap at the moment someone
        # reaches for the kill switch, so routing these through the ordinary
        # path meant every one of them came back RATE_LIMITED: the one control
        # that is meant to always work was the one the limiter disabled, and
        # `kill` reported the symbols as pulled while both orders were still
        # standing in the book.
        self._internal = True
        try:
            for symbol in self.registry.symbols:
                engine = self._engines.get(symbol)
                if engine is None:
                    continue
                # The book first, and the venue's own record second. Asking
                # only the record was asking the wrong authority: a
                # market-on-open order acknowledges no price, so nothing was
                # ever recorded for it, and the kill switch walked straight
                # past one. Measured, `kill` reported the symbol as pulled
                # while a 40-lot market-on-open buy stayed standing. And the
                # stopped participant then took 40 lots in the very auction it
                # had been stopped before.
                #
                # A parked stop is the case the book cannot answer: it is
                # deliberately off the book so that nobody can see where the
                # cascade starts, and it is still an order this participant is
                # working. So both are asked, and the union is pulled.
                order_ids = {
                    order.order_id
                    for order in engine.book.resting_orders
                    if order.agent_id == agent_id
                }
                order_ids |= set(self._working.get((agent_id, symbol), {}))
                if not order_ids:
                    continue
                touched.append(symbol)
                for order_id in sorted(order_ids):
                    self.submit(agent_id, symbol, Cancel(agent_id, order_id))
        finally:
            self._internal = False
        return sorted(set(touched))

    def revive(self, agent_id: AgentId) -> None:
        """Let a stopped participant back in. It starts with nothing working."""
        self.halted_participants.pop(agent_id, None)

    def _rate_limited(self, agent_id: AgentId, reducing: bool = False) -> bool:
        """Whether this participant has already used its second's allowance.

        A rolling second rather than a fixed one, so a burst cannot be split
        across a boundary and counted as two quiet windows.

        ``reducing`` marks a command that can only take risk *out* of the
        market. It is still counted (a burst of them is still traffic, and
        still costs the sender its ability to add anything) but it is never
        refused, because a participant that cannot withdraw is a participant
        holding exposure nobody is permitted to manage.
        """
        if self.message_rate is None or self._internal:
            return False
        now = self._now()
        window = self._messages.setdefault(agent_id, deque())
        cutoff = now - 1_000_000_000
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= self.message_rate and not reducing:
            # Not appended, so a refusal cannot lengthen the lockout: a client
            # that retries would otherwise keep its own lockout alive by trying
            # to get out of it.
            return True
        window.append(now)
        return False

    def _apply_band(self, symbol: str) -> None:
        """Tell the engine where trades may print, before it matches anything.

        Recomputed per command because the reference is a trailing mean and
        moves with the tape. Cheap: it is two integers and a lookup.

        Applied as soon as there is any reference at all, which is a lower bar
        than the one for *halting*, and deliberately. Banding an execution is
        protective: the worst it does is leave an order unfilled. Halting is
        disruptive, so it waits for a reference several prints deep. Holding
        both to the strict bar meant a market order could still walk a thin
        book to the floor for as long as the tape was quiet, which is exactly
        when a thin book is walkable.
        """
        engine = self._engines.get(symbol)
        if engine is None:
            return
        if self.price_band is None:
            engine.execution_band = None
            return
        reference = self._reference_price(symbol)
        if reference is None:
            engine.execution_band = None
            return
        low, high = self.registry.require(symbol).tick_bounds
        allowed = abs(int(high) - int(low)) * self.price_band
        engine.execution_band = (
            int(max(int(low), int(reference) - allowed)),
            int(min(int(high), int(reference) + allowed)),
        )

    def reopen_due(self) -> tuple[str, ...]:
        """Symbols whose pause has run its course, in canonical order.

        The venue does not reopen them itself. Something has to decide *when*
        time passes, and that is the simulation's business rather than the
        ledger's, so this reports and the caller uncrosses.
        """
        now = self._now()
        due: list[str] = []
        for symbol, at in sorted(self._reopen_at.items()):
            if now < at:
                continue
            # The calendar is asked here too, because this is the one path that
            # reaches a symbol nobody is sending orders to, which is what a
            # paused symbol is. A contract whose window closed while it was
            # paused has an answer, and reopening it would cross resting orders
            # against a number that already exists.
            if self._enforce_lifecycle(symbol):
                self._reopen_at.pop(symbol, None)
                continue
            if self.session(symbol) is SessionState.AUCTION:
                due.append(symbol)
        return tuple(due)

    def close(self, symbol: str) -> None:
        """Stop trading. The outcome is determined; only settlement remains."""
        self.registry.require(symbol)
        self._set_phase(symbol, SessionState.CLOSED)

    @property
    def closed_symbols(self) -> tuple[str, ...]:
        return tuple(
            sorted(s for s, p in self._phase.items() if p is SessionState.CLOSED)
        )

    @property
    def settled_symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._settled))

    def settle(self, symbol: str, result: SettlementResult) -> dict[AgentId, Decimal]:
        """Apply a settlement to every account holding the symbol.

        A VOID releases collateral without realising anything: the world never
        produced the evidence the contract required, so nobody wins and nobody
        loses. Paying out a guess instead would make the guess indistinguishable
        from a measurement the moment it hit a PnL statement.
        """
        instrument = self.registry.require(symbol)
        if result.spec_digest != instrument.spec.spec_digest:
            raise ValueError(
                f"settlement for {symbol} carries digest {result.spec_digest} but the "
                f"listed instrument is {instrument.spec.spec_digest}. The contract that "
                "settled is not the contract that traded."
            )
        if symbol in self._settled:
            raise ValueError(
                f"{symbol} has already settled. Settling twice would pay every "
                "position out twice, and an expiry firing more than once is a "
                "plausible bug rather than an impossible one."
            )
        self._settled.add(symbol)
        self._set_phase(symbol, SessionState.CLOSED)
        # Nothing can be working on a settled contract. Leaving entries behind
        # would keep reserving collateral against orders that can never fill.
        for key in [k for k in self._working if k[1] == symbol]:
            del self._working[key]
        for agent_id in [k[0] for k in self._reserve if k[1] == symbol]:
            self._restate_reserve(agent_id, symbol)

        realised: dict[AgentId, Decimal] = {}
        for agent_id in sorted(self._accounts):
            account = self._accounts[agent_id]
            if result.status == SettlementStatus.VOID:
                account.void(symbol)
                realised[agent_id] = Money(0)
            else:
                assert result.settlement_value is not None
                realised[agent_id] = account.settle(
                    symbol, to_money(result.settlement_value)
                )
        return realised

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        marks = self.marks()
        return {
            "venue": self.name,
            "instruments": [
                self.registry.require(s).to_dict() for s in self.registry.symbols
            ],
            "marks": {s: str(from_money(m)) for s, m in marks.items()},
            "accounts": [
                self._accounts[a].to_dict(marks) for a in sorted(self._accounts)
            ],
            "total_equity": str(
                from_money(
                    Money(sum(int(a.equity(marks)) for a in self._accounts.values()))
                )
            ),
        }

    def distribute(self, symbol: str, per_unit: Money) -> Money:
        """Pay ``per_unit`` to every holder of ``symbol``, charged to every short.

        Returns the gross amount that moved, which is a measurement rather than
        a total: the net is always zero, because every contract outstanding has
        someone long it and someone short it, and the same integer is added on
        one side and subtracted on the other. That is why this cannot break
        conservation even in principle, and the test still checks it.

        The remaining range narrows by exactly what was paid, so collateral
        follows the cash in the same instant rather than a moment later.
        """
        instrument = self.registry.require(symbol)
        if instrument.spec.distribution is None:
            raise ValueError(
                f"{symbol} declares no distribution schedule; paying one would be "
                "a payment its holders never agreed to and its shorts never "
                "collateralised"
            )
        if symbol in self._settled:
            raise ValueError(f"{symbol} has already settled")

        scheduled = len(instrument.spec.distribution.windows)
        already = self._distributions_paid.get(symbol, 0)
        if already >= scheduled:
            raise ValueError(
                f"{symbol} has already paid all {scheduled} of its scheduled "
                f"distributions; a further payment would take the claim below "
                "the range its collateral was computed from"
            )

        self._distributed[symbol] = Money(
            int(self._distributed.get(symbol, Money(0))) + int(per_unit)
        )
        self._distributions_paid[symbol] = already + 1
        bounds = self.bounds_in_minor(instrument)

        moved = 0
        for account in self._accounts.values():
            moved += abs(int(account.distribute(symbol, per_unit, bounds)))
        return Money(moved)

    def conservation_check(self) -> Money:
        """Total equity minus total starting capital, in minor units.

        Must be **exactly** zero: trading moves value between participants, it
        does not create it. A non-zero figure means an accounting leak, and
        this is the single sharpest check available on the whole portfolio
        layer, which is why the ledger runs on integers, so "exactly" can be
        meant literally.

        Fees do not change that. They are a transfer to the venue's own account,
        which is counted here alongside everyone else's, so a schedule with any
        rates at all still nets to zero. If it ever did not, fees would be
        creating or destroying value rather than moving it.
        """
        marks = self.marks()
        equity = sum(int(a.equity(marks)) for a in self._accounts.values())
        started = sum(int(a.starting_cash) for a in self._accounts.values())
        return Money(equity - started)
