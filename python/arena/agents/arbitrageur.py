"""A cross-instrument arbitrageur: the market's connective tissue.

Without it, related instruments are priced by unrelated crowds. Measured on the
live market before this agent existed: put-call parity was violated by ~360
ticks, and the spread contract was genuinely mean-reverting (variance ratio
0.38) because nothing tied it to the two futures it is defined by. Those are
not tuning problems: no amount of better market making on each book separately
creates consistency *between* books.

The relations it enforces are mechanical identities of the contracts, not
statistical estimates, which is what makes this agent connective tissue rather
than a strategy:

    parity      C - P = F - K            exact at settlement, proven in tests
    spread      S = LEG_A - LEG_B        the spread's own definition
    index       I = sum(w_i * LEG_i)     the basket's own weights
    strip       SHARE = sum(WEEK_i)      a share is its own payment schedule
    vertical    0 <= C(K1) - C(K2) <= K2 - K1     for K1 < K2
    butterfly   C(K2) <= w1*C(K1) + w3*C(K3)      for K1 < K2 < K3

The last two are *bounds* rather than equalities, and they are what makes an
option chain internally consistent. A set of call prices is free of static
arbitrage exactly when it is decreasing and convex in strike with slope in
[-1, 0] (Davis and Hobson 2007; Carr and Madan 2005), and those two families
are that condition written as portfolios. A violated bound is still a riskless
trade (buy the cheap side of a vertical and the payoff can never be negative),
so the same execution path handles both; only the definition of "mispriced"
changes, from "not zero" to "outside the band".

Relations are *derived from the listed instruments* at construction, from each
contract's underlying algebra and payoff, rather than configured by hand, so
listing a new spread or option chain makes it arbitrageable with no code
change, and a relation with a missing leg (an index component with no listed
future) is simply not formed.

Honest limitations, kept on purpose:

  * It trades on its *local* view, which is stale by its latency. Both legs are
    sent as IOC market orders simultaneously, but fills are not atomic: it can
    get one leg and miss the other. That legging risk is real on real venues
    too, and the cost hurdle plus position limits are the classical
    mitigations, not a magic exemption.
  * It pays the spread on every leg, so it only acts when the mispricing
    exceeds the round-trip cost times a safety multiple. Small violations
    persist; that is the no-arbitrage *band*, which is what real markets have.
"""

from __future__ import annotations

from dataclasses import dataclass

from arena.agents.base import TradingAgent
from arena.contracts.payoff import Call, Linear, Put
from arena.determinism import canonical_json
from arena.exchange.types import AgentId, Side
from arena.market.instrument import Instrument
from arena.sim.kernel import SimulationContext
from arena.sim.time import Duration, millis

__all__ = ["Arbitrageur", "Relation", "derive_relations"]


@dataclass(frozen=True, slots=True)
class Relation:
    """A linear pricing band: ``price(target) - sum(coef * leg) - const`` in
    ``[lower, upper]``.

    An identity is the special case where the band has zero width, which is
    every relation this agent started with. The general form exists because the
    conditions that make an option chain consistent are inequalities: a call
    must cost more than the call above it, but not more than the difference in
    their strikes, and neither statement is an equation.

    Everything in *price* units (not ticks), because the legs of one relation
    can trade on different tick grids.
    """

    name: str
    target: str
    legs: tuple[tuple[str, float], ...]
    constant: float = 0.0
    lower: float = 0.0
    upper: float = 0.0

    def excess(self, target_price: float, leg_prices: dict[str, float]) -> float:
        """How far outside the band this relation currently sits, signed.

        Zero inside it. Positive means the target is too dear relative to the
        package, negative too cheap. That is the same convention an identity
        had, so everything downstream is unchanged.
        """
        theoretical = self.constant
        for symbol, coefficient in self.legs:
            theoretical += coefficient * leg_prices[symbol]
        raw = target_price - theoretical
        return raw - min(max(raw, self.lower), self.upper)

    @property
    def symbols(self) -> tuple[str, ...]:
        return (self.target, *(symbol for symbol, _ in self.legs))


def _underlying_key(instrument: Instrument) -> str:
    """What a contract is written on, *and over what period*.

    The window belongs in the key. Without it two contracts that differ only by
    the week they measure are indistinguishable here, so the last one listed
    silently wins the lookup and a relation gets formed against the wrong leg:
    an identity between two things that are not the same thing, traded as
    though it were free money. Nothing was mispriced by it yet, because no
    spread or index referenced a weekly contract at the time; listing weekly
    futures is precisely what would have made it start being wrong.
    """
    return _key(instrument.spec.underlying.to_dict(), instrument.spec.window.to_dict())


def _key(underlying: dict, window: dict) -> str:
    """One lookup key for "this thing, measured over this period".

    Takes plain dictionaries because half the callers have an object and half
    have the serialized shape of a leg they read out of a composite contract.
    """
    return canonical_json({"underlying": underlying, "window": window})


def derive_relations(instruments: dict[str, Instrument]) -> list[Relation]:
    """Read the pricing identities out of the listed contracts.

    A relation is only formed when every leg it needs is actually listed. An
    index over a component with no future is left alone rather than
    approximated, because a relation traded against a proxy is a bet, and this
    agent only trades identities.
    """
    # Atomic price references: plain futures (Linear, no offset) per underlying.
    futures: dict[str, tuple[str, float]] = {}
    for symbol, instrument in sorted(instruments.items()):
        payoff = instrument.spec.payoff
        # A contract that pays as it goes is not a price reference for one
        # that does not, however linear its settlement looks: two claims on the
        # same underlying that pay at different times are different things, and
        # the relation between them is a real one this agent does not yet know
        # how to trade.
        if instrument.spec.distribution is not None:
            continue
        if isinstance(payoff, Linear) and payoff.offset == 0.0:
            futures[_underlying_key(instrument)] = (symbol, payoff.scale)

    relations: list[Relation] = []

    # -- spreads and indices: linearity of the underlying algebra ------------
    for symbol, instrument in sorted(instruments.items()):
        payoff = instrument.spec.payoff
        if instrument.spec.distribution is not None:
            continue
        if not (isinstance(payoff, Linear) and payoff.offset == 0.0):
            continue
        shape = instrument.spec.underlying.to_dict()
        window = instrument.spec.window.to_dict()

        if shape["kind"] == "difference":
            left = futures.get(_key(shape["left"], window))
            right = futures.get(_key(shape["right"], window))
            if left and right and left[1] == payoff.scale == right[1]:
                relations.append(
                    Relation(
                        name=f"spread:{symbol}",
                        target=symbol,
                        legs=((left[0], 1.0), (right[0], -1.0)),
                    )
                )

        elif shape["kind"] == "basket":
            legs: list[tuple[str, float]] = []
            for entry in shape["legs"]:
                component = futures.get(_key(entry["leg"], window))
                if component is None or component[1] != payoff.scale:
                    legs = []
                    break
                legs.append((component[0], float(entry["weight"])))
            if legs:
                relations.append(
                    Relation(name=f"index:{symbol}", target=symbol, legs=tuple(legs))
                )

    # -- put-call parity: C = P + F - K --------------------------------------
    calls: dict[tuple[str, float, float], str] = {}
    puts: dict[tuple[str, float, float], str] = {}
    for symbol, instrument in sorted(instruments.items()):
        payoff = instrument.spec.payoff
        key = _underlying_key(instrument)
        if isinstance(payoff, Call):
            calls[(key, payoff.strike, payoff.scale)] = symbol
        elif isinstance(payoff, Put):
            puts[(key, payoff.strike, payoff.scale)] = symbol

    for (key, strike, scale), call_symbol in sorted(calls.items()):
        put_symbol = puts.get((key, strike, scale))
        future = futures.get(key)
        if put_symbol is None or future is None or future[1] != scale:
            continue
        relations.append(
            Relation(
                name=f"parity:{call_symbol}",
                target=call_symbol,
                legs=((put_symbol, 1.0), (future[0], 1.0)),
                constant=-strike,
            )
        )

    # -- strips: a share is the sum of the weeks it pays ---------------------
    #
    # Exact, not approximate. The share's payment for a week and the future on
    # that week resolve the same metric over the same window under the same
    # evidential bar, so they are the same number. Formed only when *every*
    # week is listed; a strip missing a leg is a directional bet on the leg
    # that is missing.
    for symbol, instrument in sorted(instruments.items()):
        schedule = instrument.spec.distribution
        payoff = instrument.spec.payoff
        if schedule is None or not isinstance(schedule.payoff, Linear):
            continue
        if not isinstance(payoff, Linear) or payoff.offset != 0.0:
            continue
        if schedule.payoff.offset != 0.0:
            continue

        legs: list[tuple[str, float]] = []
        for window in schedule.windows:
            leg = futures.get(
                _key(instrument.spec.underlying.to_dict(), window.to_dict())
            )
            if leg is None or leg[1] != schedule.payoff.scale:
                legs = []
                break
            legs.append((leg[0], 1.0))
        if not legs:
            continue
        # Anything left at expiry is one more leg, on the contract's own window.
        if payoff.scale != 0.0:
            terminal = futures.get(
                _key(
                    instrument.spec.underlying.to_dict(),
                    instrument.spec.window.to_dict(),
                )
            )
            if terminal is None or terminal[1] != payoff.scale:
                continue
            legs.append((terminal[0], 1.0))
        relations.append(
            Relation(name=f"strip:{symbol}", target=symbol, legs=tuple(legs))
        )

    # -- the option chain: decreasing and convex in strike -------------------
    #
    # These two families are the whole of static arbitrage freedom for calls at
    # one maturity. They are inequalities, so they are the reason Relation has
    # a band: a call may be dearer than the call above it by anything from
    # nothing up to the gap in their strikes, and both ends of that are
    # tradeable when breached.
    chains: dict[tuple[str, float, str], list[tuple[float, str]]] = {}
    for symbol, instrument in sorted(instruments.items()):
        payoff = instrument.spec.payoff
        if isinstance(payoff, Call):
            kind = "call"
        elif isinstance(payoff, Put):
            kind = "put"
        else:
            continue
        chains.setdefault(
            (_underlying_key(instrument), payoff.scale, kind), []
        ).append((payoff.strike, symbol))

    for (_key_, _scale, kind), rungs in sorted(chains.items()):
        rungs.sort()
        if len(rungs) < 2:
            continue

        # Monotone, and by no more than the strikes differ. For a call the
        # lower strike is the dearer one; for a put it is the higher.
        for (low_strike, low_symbol), (high_strike, high_symbol) in zip(rungs, rungs[1:]):
            width = high_strike - low_strike
            dear, cheap = (
                (low_symbol, high_symbol) if kind == "call" else (high_symbol, low_symbol)
            )
            relations.append(
                Relation(
                    name=f"vertical:{dear}/{cheap}",
                    target=dear,
                    legs=((cheap, 1.0),),
                    lower=0.0,
                    upper=width,
                )
            )

        # Convex: the middle strike cannot cost more than the straight line
        # between its neighbours. Weighted for uneven spacing, so a ladder does
        # not have to be evenly struck to be checked.
        for first, second, third in zip(rungs, rungs[1:], rungs[2:]):
            (k1, s1), (k2, s2), (k3, s3) = first, second, third
            span = k3 - k1
            if span <= 0:
                continue
            w1, w3 = (k3 - k2) / span, (k2 - k1) / span
            relations.append(
                Relation(
                    name=f"butterfly:{s2}",
                    target=s2,
                    legs=((s1, w1), (s3, w3)),
                    lower=float("-inf"),
                    upper=0.0,
                )
            )

    return relations


class Arbitrageur(TradingAgent):
    """Trades mechanical mispricings between related instruments."""

    def __init__(
        self,
        agent_id: AgentId,
        venue_id: AgentId,
        instruments: dict[str, Instrument],
        wake_interval: Duration = millis(400),
        base_size: int = 6,
        edge_multiple: float = 1.5,
        position_limit: int = 300,
        recycle_capital: bool = True,
        exit_fraction: float = 0.5,
        max_participation: float = 0.25,
        scale_in_multiple: float = 1.5,
    ) -> None:
        super().__init__(agent_id, venue_id, instruments, wake_interval)
        self.relations = derive_relations(instruments)

        self.base_size = base_size
        # The mispricing must exceed the round-trip cost by this multiple
        # before acting. Below it lies the no-arbitrage band, where violations
        # persist because closing them would cost more than they are worth.
        self.edge_multiple = edge_multiple
        self.position_limit = position_limit
        # Whether to take a converged package back off. Holding to settlement
        # realises the whole gap and pays no exit cost, which is the better
        # trade in isolation, but it consumes the balance sheet permanently,
        # and a fully-invested arbitrageur cannot correct the *next*
        # dislocation. Measured rather than assumed; see the ablation.
        self.recycle_capital = recycle_capital
        self.exit_fraction = exit_fraction
        # Never take more than this share of the depth it can see at the touch.
        # Without it this agent strips the book: firing IOC orders on three legs
        # every 400ms against a maker that refills every 300ms cut visible ask
        # depth from 207 lots to 26, and a market that thin stops being able to
        # absorb anyone else's order. Every real execution algorithm carries a
        # participation cap for the same reason.
        self.max_participation = max_participation
        # How much wider the gap must get before adding to an existing package.
        self.scale_in_multiple = scale_in_multiple
        self.attempts = 0
        self.unwinds = 0
        self.starved = 0
        # Packages held per relation, signed: +1 is long the target and short
        # the replicating legs. Tracked per relation because one symbol can be
        # a leg of several, so its net position says nothing about which
        # relation put it there.
        self._packages: dict[str, int] = {}
        # The gap each relation's current package was put on at, so adding to it
        # can be made conditional on the dislocation actually getting worse.
        self._entry_gap: dict[str, float] = {}

    # -- pricing helpers ---------------------------------------------------

    def add_relations(self, relations: list[Relation]) -> int:
        """Take on identities that listed after this agent was built.

        `derive_relations` runs once, over the contracts that existed at
        construction, which was right while the listing was fixed for a
        session. Matches open and close continuously and bring their own
        identities with them, in this agent's own vocabulary, so enforcing
        them needs no new execution path: the list simply gets longer.

        Deduplicated on the relation itself rather than trusting the caller,
        because an operator that told this agent twice about one match would
        otherwise have it size every package on that match twice.
        """
        known = set(self.relations)
        added = [r for r in relations if r not in known]
        self.relations = [*self.relations, *added]
        return len(added)

    def drop_relations_for(self, symbols: set[str]) -> int:
        """Forget identities whose legs have settled.

        A relation over a settled contract can never be traded again, and one
        whose legs are gone would price against a mark that no longer moves.
        Dropped when any leg goes, not only when all of them do: a package
        missing a leg is a directional bet, which is the failure this agent
        already flattens for elsewhere.
        """
        before = len(self.relations)
        self.relations = [
            r for r in self.relations if not (set(_legs_of(r)) & symbols)
        ]
        return before - len(self.relations)

    def _mid_price(self, symbol: str) -> float | None:
        """The agent's local mid, converted to price units."""
        book = self.books[symbol]
        if book.mid is None or book.updated_at == 0:
            return None
        return book.mid * float(self.instruments[symbol].tick_size)

    def _takeable(self, symbol: str, side: Side) -> int:
        """Lots this agent is willing to lift from the touch it can see.

        Read off its own stale local book, not the venue's, so it is sizing
        against the same out-of-date picture it is pricing against.
        """
        book = self.books[symbol]
        resting = book.ask_size if side is Side.BUY else book.bid_size
        return int(max(0, resting) * self.max_participation)

    def _half_spread(self, symbol: str) -> float | None:
        """Half the touch, in price units: the cost of taking one leg."""
        book = self.books[symbol]
        if book.spread is None:
            return None
        return book.spread / 2.0 * float(self.instruments[symbol].tick_size)

    # -- the loop ----------------------------------------------------------

    def act(self, ctx: SimulationContext) -> None:
        for relation in self.relations:
            self._check(ctx, relation)

    def _check(self, ctx: SimulationContext, relation: Relation) -> None:
        target_mid = self._mid_price(relation.target)
        target_cost = self._half_spread(relation.target)
        if target_mid is None or target_cost is None:
            return

        cost = target_cost
        leg_prices: dict[str, float] = {}
        for symbol, coefficient in relation.legs:
            leg_mid = self._mid_price(symbol)
            leg_cost = self._half_spread(symbol)
            if leg_mid is None or leg_cost is None:
                return
            leg_prices[symbol] = leg_mid
            # Taking a leg costs its half-spread regardless of the weight's
            # sign; the |coefficient| scales how many lots that cost is paid on.
            cost += abs(coefficient) * leg_cost

        # How far outside the band, which for an identity is simply the gap.
        gap = relation.excess(target_mid, leg_prices)
        held = self._packages.get(relation.name, 0)

        if abs(gap) <= cost * self.edge_multiple:
            # Inside the no-arbitrage band: the relation is fair, so there is
            # nothing to correct. If a package is still on from when it was not
            # fair, this is where it comes off, but only once the gap has
            # converged well past the band, or the exit would give back more
            # than the entry captured.
            if self.recycle_capital and held and abs(gap) <= cost * self.exit_fraction:
                self._execute(ctx, relation, -1 if held > 0 else 1, abs(held))
                self.unwinds += 1
            return

        # Rich target: sell it, buy the replicating package. Cheap: reverse.
        units = -1 if gap > 0 else 1

        # Do not average into a trade that is not working. Holding a package in
        # this direction already, the dislocation has to have *widened*
        # materially before adding to it; otherwise the agent re-enters the
        # same position on every wakeup for as long as the gap persists, which
        # is what it did: firing on 93% of wakeups, ~1,400 times a session, and
        # taking the book down from 207 resting lots to 26. A market that thin
        # cannot absorb anyone else's order, so the damage was not confined to
        # this agent's own P&L.
        if held and (held > 0) == (units > 0):
            entry = self._entry_gap.get(relation.name, 0.0)
            if abs(gap) < entry * self.scale_in_multiple:
                return
        self._entry_gap[relation.name] = abs(gap)
        self._execute(ctx, relation, units, self.base_size)

    def _execute(self, ctx: SimulationContext, relation: Relation, units: int, lots: int) -> None:
        """Trade ``lots`` packages in direction ``units`` (+1 long the target).

        All legs go out together as IOC market orders. Not atomic. Legging risk
        is real and deliberately kept; see the module docstring.
        """
        lots = min(lots, self.base_size)
        if lots <= 0:
            return

        target_side = Side.BUY if units > 0 else Side.SELL
        sides: list[tuple[str, Side, float]] = [(relation.target, target_side, 1.0)]
        for symbol, coefficient in relation.legs:
            # Long the target means short the replicating legs, and a negative
            # coefficient flips that leg again.
            long_leg = (coefficient > 0) != (units > 0)
            sides.append((symbol, Side.BUY if long_leg else Side.SELL, abs(coefficient)))

        # Size to the thinnest leg. Taking the full size on a deep leg and a
        # fraction of it on a thin one would leave the relation half-on, which
        # is a directional bet, so the whole package shrinks to what the
        # scarcest book can supply.
        for symbol, side, weight in sides:
            lots = min(lots, int(self._takeable(symbol, side) / max(weight, 1e-9)))
        if lots <= 0:
            self.starved += 1
            return

        orders = [
            (symbol, side, max(1, round(lots * weight))) for symbol, side, weight in sides
        ]

        # Respect position limits on every leg before sending anything: a
        # relation half-entered is a directional bet, so if any leg is at its
        # limit the whole relation is skipped this round.
        for symbol, side, quantity in orders:
            signed = quantity if side is Side.BUY else -quantity
            if abs(self.position.get(symbol, 0) + signed) > self.position_limit:
                return

        self.attempts += 1
        self._packages[relation.name] = self._packages.get(relation.name, 0) + units * lots
        for symbol, side, quantity in orders:
            self.take(ctx, symbol, side, quantity)


def _legs_of(relation: Relation) -> tuple[str, ...]:
    """Every symbol a relation reads, the target and each of its legs."""
    return (relation.target, *(symbol for symbol, _weight in relation.legs))
