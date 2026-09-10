"""A market maker that quotes a whole option chain from one distribution.

The defect this exists to fix was measured on the live market and is a riskless
arbitrage sitting in the book: on one chain of calls over a win rate future,
the strike at 4,700 marked at 72.7 while the strike at 4,600 on the same
underlying marked at 59.1. A call struck higher cannot be worth more than one
struck lower (the lower strike pays whatever the higher one pays and sometimes
more), so anyone could buy the 4,600 and sell the 4,700 and never lose.
Put-call parity was out by 35 ticks at the same time.

Strikes in the four thousands anywhere in this file belong to the retired
statistics world. Those measurements predate the circuit and the listing it
carries, and they are kept as measured because every one of them is a fact
about the quoting rule rather than about which contract it was quoting.

There was one cause. The plain maker anchors each book on its own trade prints
and opens it at the middle of its own settlement range, so every strike was
priced by a process that had never heard of the strike next to it. Nothing
related them, so nothing kept them related.

The fix is the one real desks use: quote the whole ladder off a **single
distribution** of where the underlying will settle, and the ladder is
arbitrage-free by construction. A set of call prices at one maturity is free of
static arbitrage exactly when it is decreasing and convex in strike with slope
in [-1, 0] (Davis and Hobson 2007; Carr and Madan 2005), and every one of those
conditions is automatic for prices of the form ``E[(F - K)+]`` under any fixed
law: monotone because the payoff is, convex because a maximum of affine
functions is, slope ``-P(F > K)`` which lives in [-1, 0] by definition. Parity
holds because the put is *defined* here as ``C - (E[F] - K)``.

Two things it deliberately does not do, because they would make it price the
answer rather than a belief:

  * The distribution is centred on the **market's** view of the underlying (the
    maker's own anchor for the future, which is a slow average of where that
    future has actually traded) and never on the settlement value. If the
    future is mispriced, the whole chain is consistently mispriced, which is
    right: consistency is the property being fixed, not correctness.
  * Its width is **estimated from the tape**, not configured. A constant would
    have been a number chosen to make option prices look plausible, and it
    would have frozen the one quantity an option market is actually about.
    Instead the maker keeps an exponentially-weighted variance of how far
    prints in the underlying land from its own anchor, the same estimator a
    risk desk runs, and converts it into a Beta concentration by matching
    moments: for a rate ``m`` scaled by ``S``, a standard deviation of ``s`` in
    price implies ``kappa = m(1-m)S^2/s^2 - 1``.

    So the width follows the market. When the underlying is quiet, options
    price near intrinsic; when it moves, they carry time value. That is a
    belief the maker can still be wrong about (realised dispersion is not
    settlement uncertainty, and the difference is the adverse selection an
    options market maker actually faces), but it is wrong in a way that
    responds to evidence rather than in a way that was typed in.

The event contracts are on the same ladder, for the same reason and by the same
arithmetic. A binary struck at a level is a digital on the forward the calls
are written on, so ``P(F > K)`` prices it and that is the same quantity
``call_delta`` already computes for the ladder. Left off it, they were quoted
by the plain maker's exponential average of their own prints, which on a claim
settling at zero or one is an average of a quantity that takes neither value
until it resolves, and which stops being an estimator at all once the book
stops trading. Measured on seed 7 over 600s: every event contract was dead
inside two minutes, one binary struck on a competitor's win rate spent the
remaining eight marked at 0.40 against a settlement of 1.00, and another struck
at the same level on a different competitor took every one of its passive fills
on the same side.

Inventory is skewed in the underlying, not per strike, and that is not a
detail. The plain maker skews each book by a fraction of *that contract's*
settlement range, which is sensible for a future worth half its range and
absurd for an option worth one percent of it: a full inventory in a call a
hundred ticks out of the money would move its quote by 540, against a fair
value near 70. Here the chain's net delta shifts the underlying anchor and the
whole ladder reprices from the shifted anchor, so inventory control happens in
the units the risk is actually denominated in, and the ladder stays consistent
while it happens.
"""

from __future__ import annotations

from scipy.special import betainc

from arena.agents.market_maker import MarketMaker
from arena.agents.base import TradePrint
from arena.contracts.payoff import Binary, Call, Linear, Put
from arena.contracts.spec import ObservationWindow
from arena.contracts.underlying import Underlying
from arena.determinism import canonical_json
from arena.exchange.types import AgentId
from arena.market.instrument import Instrument
from arena.sim.kernel import SimulationContext
from arena.sim.time import Duration, millis, seconds

__all__ = [
    "SurfaceMarketMaker",
    "ChainMember",
    "derive_chains",
    "option_value",
    "digital_value",
    "call_delta",
]


def _key(underlying: Underlying, window: ObservationWindow) -> str:
    return canonical_json(
        {"underlying": underlying.to_dict(), "window": window.to_dict()}
    )


class ChainMember:
    """One derivative, and the future whose price it is quoted off.

    ``strike`` is always in the future's price units, which is what lets a
    binary sit in the same ladder as the options. An option's strike is
    already quoted that way; a binary's threshold is stated in the metric's
    own units, so it is carried onto the price grid by the future's scale.
    That is not a convention chosen here, it is the same identity the payoff
    classes already use: a future settles at ``scale * level``, so the level
    crossing ``threshold`` is the future crossing ``threshold * scale``.
    """

    __slots__ = (
        "underlying_symbol",
        "strike",
        "scale",
        "is_call",
        "is_digital",
        "above",
        "payout",
    )

    def __init__(
        self,
        underlying_symbol: str,
        strike: float,
        scale: float,
        is_call: bool,
        is_digital: bool = False,
        above: bool = True,
        payout: float = 0.0,
    ):
        self.underlying_symbol = underlying_symbol
        self.strike = strike
        self.scale = scale
        self.is_call = is_call
        self.is_digital = is_digital
        # Which side of the threshold pays, read off the contract's own
        # comparison rather than assumed. Under a continuous law ``>`` and
        # ``>=`` are the same probability, so only the direction survives.
        self.above = above
        self.payout = payout


def derive_chains(instruments: dict[str, Instrument]) -> dict[str, ChainMember]:
    """Match every listed derivative to the listed future it is written on.

    Read out of the contracts rather than configured, so listing a new strike
    makes it quotable with no code change, and an option whose underlying
    future is not listed is simply left to the plain maker, because there is
    nothing to anchor it to.

    Binaries belong here for the same reason the calls do, and leaving them out
    was the same defect arriving through a different door. A binary struck at a
    level is a digital on the same forward as the call struck at that level
    times the scale, so a distribution that prices one prices the other; a
    process that has never heard of the strike next to it prices neither.
    Measured on seed 7 before they joined, over 600s: every event contract in
    the market stopped trading inside two minutes, the maker's print average
    froze wherever it happened to be when the last trade printed, and one
    binary struck on a competitor's win rate spent the remaining eight minutes
    marked at 0.40 against a settlement of 1.00. Its passive flow imbalance was
    +0.43 and another binary struck at the same level on a different competitor
    was +1.00, meaning every passive fill on it landed on the same side without
    exception. A print average is not merely a poor estimator of a probability,
    it is not an estimator at all once the book it feeds on has stopped
    trading, and a book quoted at 0.40 against a certainty is what stops it.
    """
    futures: dict[str, tuple[str, float]] = {}
    for symbol, instrument in sorted(instruments.items()):
        payoff = instrument.spec.payoff
        if instrument.spec.distribution is not None:
            continue
        if isinstance(payoff, Linear) and payoff.offset == 0.0:
            futures[_key(instrument.spec.underlying, instrument.spec.window)] = (
                symbol,
                payoff.scale,
            )

    chain: dict[str, ChainMember] = {}
    for symbol, instrument in sorted(instruments.items()):
        payoff = instrument.spec.payoff
        if not isinstance(payoff, (Call, Put, Binary)):
            continue
        found = futures.get(_key(instrument.spec.underlying, instrument.spec.window))
        if found is None:
            continue
        underlying_symbol, scale = found
        if isinstance(payoff, Binary):
            # A binary carries no scale of its own, because its settlement
            # range is the payout and says nothing about the underlying. The
            # future supplies the only scale in the relationship, and the
            # threshold is stated in the same units the future's scale
            # multiplies, so the two compose without a conversion.
            if not scale or payoff.payout == 0.0:
                continue
            chain[symbol] = ChainMember(
                underlying_symbol=underlying_symbol,
                strike=payoff.threshold * scale,
                scale=scale,
                # Nothing reads ``is_call`` on a digital: every path that
                # branches on it checks ``is_digital`` first. It is set rather
                # than left to a default so the field never carries a claim
                # that a reader could take at face value.
                is_call=True,
                is_digital=True,
                above=payoff.comparison.startswith(">"),
                payout=payoff.payout,
            )
            continue
        if scale != payoff.scale:
            continue
        chain[symbol] = ChainMember(
            underlying_symbol=underlying_symbol,
            strike=payoff.strike,
            scale=scale,
            is_call=isinstance(payoff, Call),
        )
    return chain


def call_delta(forward: float, strike: float, scale: float, concentration: float) -> float:
    """P(the option finishes in the money), which is also dC/dF.

    Zero or one outside the metric's range: a call struck above the largest
    value the metric can take can never pay, and there is no distribution to
    consult about it.
    """
    mean = forward / scale
    if not 0.0 < mean < 1.0:
        return 1.0 if forward > strike else 0.0
    threshold = strike / scale
    if threshold <= 0.0:
        return 1.0
    if threshold >= 1.0:
        return 0.0
    a, b = concentration * mean, concentration * (1.0 - mean)
    return float(1.0 - betainc(a, b, threshold))


def digital_value(
    forward: float,
    strike: float,
    scale: float,
    concentration: float,
    above: bool,
    payout: float,
) -> float:
    """What a contract paying ``payout`` on one side of ``strike`` is worth.

    The same distribution the calls are priced off, read at one point instead
    of integrated over a tail, so the digital and the call ladder cannot
    disagree about the probability of the same event. That is the property the
    whole class exists to hold: a digital struck at K priced independently of
    the call struck at K is two beliefs about one question, and the difference
    between them is free money in whichever direction it happens to fall.

    It is also the reason a binary can be quoted at all here. The plain maker
    prices it from an average of its own prints, which on a contract settling
    at zero or one is an average of a quantity that only ever takes two values
    and mostly takes neither, because the book stops trading long before it
    resolves. This is a probability computed from a forward that is still
    trading, so it keeps moving when the binary's own book does not.
    """
    probability = call_delta(forward, strike, scale, concentration)
    return payout * (probability if above else 1.0 - probability)


def option_value(
    forward: float, strike: float, scale: float, concentration: float, is_call: bool
) -> float:
    """``E[(F - K)+]`` for a call, and its parity partner for a put.

    The underlying is a proportion in [0, 1] scaled onto a price grid, so the
    law is a Beta with the given mean and concentration. The truncated mean has
    a closed form: for ``L ~ Beta(a, b)`` and ``k`` in (0, 1),

        E[(L - k)+] = m * (1 - I_k(a + 1, b)) - k * (1 - I_k(a, b))

    where ``I`` is the regularized incomplete beta and ``m = a / (a + b)``. It
    follows from ``L * f_{a,b}(L) = m * f_{a+1,b}(L)``, so no quadrature and no
    simulation is needed, which matters because this runs on every requote of
    every strike.

    The put is defined by parity rather than integrated separately. Computing
    it independently would leave the two agreeing only up to floating point,
    and the whole purpose of this maker is that they agree exactly.
    """
    mean = forward / scale
    threshold = strike / scale
    if not 0.0 < mean < 1.0 or threshold >= 1.0 or threshold <= 0.0:
        # Degenerate: no distribution to speak of, so fall back on intrinsic
        # value, which is the correct limit and never negative.
        call = max(0.0, forward - strike)
    else:
        a, b = concentration * mean, concentration * (1.0 - mean)
        above_mass = 1.0 - float(betainc(a, b, threshold))
        above_mean = mean * (1.0 - float(betainc(a + 1.0, b, threshold)))
        call = scale * (above_mean - threshold * above_mass)
        call = max(0.0, call)
    if is_call:
        return call
    return call - (forward - strike)


class SurfaceMarketMaker(MarketMaker):
    """The plain maker, with its option books priced off one distribution."""

    def __init__(
        self,
        agent_id: AgentId,
        venue_id: AgentId,
        instruments: dict[str, Instrument],
        wake_interval: Duration = millis(300),
        vol_weight: float = 0.02,
        vol_halflife: Duration = seconds(60),
        min_prints: int = 25,
        skew_sigmas: float = 1.0,
        delta_limit: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(agent_id, venue_id, instruments, wake_interval, **kwargs)
        self.chain = derive_chains(instruments)
        # How quickly the variance estimate forgets. Slower than the anchor's
        # own weight on purpose: a mean can be tracked from a dozen prints, a
        # variance cannot, and a jumpy width would move every strike at once.
        self.vol_weight = vol_weight
        # How long it takes a realised-variance estimate to lose half its
        # weight when nothing is printing.
        #
        # Without it the estimate only moved on prints, so a quiet market kept
        # whatever width it last had. Forever. Measured: with the future pinned
        # at 4,800 for six straight minutes the makers were still quoting the
        # call struck a hundred ticks below it at 153 against an intrinsic
        # value of 100, the informed agents sold it to them all session, all
        # three reached their position limits and stopped bidding, and the
        # strike had no bid at 17 of 19 sampled moments. Volatility is a
        # statement about now; an estimate of it has to forget at the rate time
        # passes, not at the rate trades happen.
        self.vol_halflife = vol_halflife
        # Below this many prints the estimate is noise, and the chain is left
        # to the plain maker rather than quoted off a number that is not one.
        self.min_prints = min_prints
        # How far a full inventory shades the forward, in standard deviations
        # of the underlying's own realised dispersion.
        #
        # Not a fraction of the settlement range, which is what the plain maker
        # uses and what this class first copied. A range-sized skew is sensible
        # for a future worth half its range and ruinous for an option worth one
        # percent of it: a net delta of 66 lots shifted the forward by 165 and
        # every call in the chain went to zero while the puts tripled. In
        # dispersion units the same inventory shades the forward by about one
        # standard deviation, which is a statement a trader would recognise.
        self.skew_sigmas = skew_sigmas
        # Net delta, in underlying-equivalent lots, at which that skew is
        # reached. Defaults to a full position in the underlying itself.
        self.delta_limit = float(delta_limit or self.position_limit)
        # There is no vega limit here, and one was written, measured and taken
        # back out rather than left as an omission. The gap it was aimed at is
        # real and is recorded in :meth:`_net_options`: a delta limit is
        # structurally blind to a short straddle, and this maker finishes short
        # 30,768 option lots on seed 7 over 600s, short on 56 of the 60
        # contract and maker pairs it holds, pinned at the per contract limit
        # on several strikes and therefore showing no offer on them at all.
        #
        # Skewing the dispersion on net option lots, one over ``1 + k * pull``
        # with ``pull`` the net position over a limit, does not touch it. With
        # the limit at a full position in one contract, 1,200 lots, and a net
        # option position near -15,000, the pull is clamped to -1 on every
        # requote from the first minute onward, so it is not a limit at all,
        # it is a constant multiplier on the width. Measured with ``k = 0.5``,
        # a doubling: the net option position moved from -14,618 to -14,278,
        # a 2% dent, and `test_every_strike_stays_quotable` failed on a
        # different strike rather than passing.
        #
        # It saturates because the miss is not a couple of standard deviations,
        # it is a factor of six, and it is a factor of six for a reason no
        # inventory signal can supply. Realised print dispersion is a one step
        # forecast error; settlement uncertainty is that error compounded over
        # the steps remaining. The correction is a square root of a horizon,
        # and the horizon is a calendar quantity while everything this agent
        # can see is kernel nanoseconds. Those are the two clocks nothing
        # connects. A limit set to bind would have to be a number chosen to
        # land on the informed traders' own posterior width, which is fitting
        # the answer.
        self._variance: dict[str, float] = {}
        self._prints: dict[str, int] = {}
        self._variance_at: dict[str, int] = {}

    # -- what the tape says the width is -----------------------------------

    def on_print(self, ctx: SimulationContext, print_: TradePrint) -> None:
        """Track dispersion around the anchor, then let the anchor move.

        Order matters. Measuring the deviation *before* the anchor absorbs this
        print is what makes it a deviation; afterwards the anchor has already
        moved toward the print and the estimate shrinks toward zero on exactly
        the days it should be growing.
        """
        anchor = self._anchor.get(print_.symbol)
        if anchor is not None:
            deviation = float(int(print_.price)) - anchor
            current = self._variance.get(print_.symbol, deviation * deviation)
            self._variance[print_.symbol] = current + self.vol_weight * (
                deviation * deviation - current
            )
            self._variance_at[print_.symbol] = int(ctx.now)
            self._prints[print_.symbol] = self._prints.get(print_.symbol, 0) + 1
        super().on_print(ctx, print_)

    def dispersion_for(self, symbol: str, now: int = 0) -> float | None:
        """Realised standard deviation of prints in ``symbol``, in price units.

        Decayed by how long it has been since the last print, because
        volatility is a statement about the present. An estimate that only
        moved when a trade happened would keep a quiet market's last width
        indefinitely, and a maker quoting yesterday's volatility into today's
        silence is selling insurance against a risk that has gone away.

        ``None`` until there is enough tape to say anything, which is the
        honest answer at the open.
        """
        if self._prints.get(symbol, 0) < self.min_prints:
            return None
        variance = self._variance.get(symbol)
        if not variance or variance <= 0.0:
            return None
        elapsed = max(0, int(now) - self._variance_at.get(symbol, int(now)))
        if elapsed and self.vol_halflife:
            variance *= 0.5 ** (elapsed / float(int(self.vol_halflife)))
        if variance <= 0.0:
            return None
        return variance**0.5 * float(self.instruments[symbol].tick_size)

    def concentration_for(
        self,
        symbol: str,
        forward: float,
        scale: float,
        now: int = 0,
        sigma: float | None = None,
    ) -> float | None:
        """Beta concentration implied by that dispersion.

        ``sigma`` overrides the tape's own estimate, so a caller that has
        already shaded the width for its inventory gets a concentration that
        agrees with the width it is quoting. Omitted, it reads the tape, which
        is what every caller outside :meth:`_requote` wants.
        """
        if sigma is None:
            sigma = self.dispersion_for(symbol, now)
        if sigma is None:
            return None
        level = forward / scale
        if not 0.0 < level < 1.0 or sigma <= 0.0:
            return None
        # Moment matching: Var[S*L] = S^2 * m(1-m)/(kappa+1).
        concentration = level * (1.0 - level) * scale * scale / (sigma * sigma) - 1.0
        # A concentration at or below zero is not a distribution. It means the
        # tape is wider than a rate on [0, 1] can be, which is a broken market
        # rather than a very uncertain one.
        return concentration if concentration > 1.0 else None

    # -- the surface -------------------------------------------------------

    def _underlying_anchor(self, symbol: str) -> float | None:
        """Where the underlying is trading now, in price units.

        The **live mid** of the future's book, which is a different choice from
        the one the plain maker makes for its own quotes and deliberately so.
        That maker anchors on prints because it is both sides of its own mid,
        so quoting around the mid would be quoting around itself and the price
        could never move. An option is not in that loop: this maker's option
        quotes do not touch the future's book, so the future's mid is an
        outside price as far as the chain is concerned.

        It also matters that it is fast. Anchoring the chain on the slow print
        average was measured and was clearly wrong: the underlying converges
        from its opening reference to fair value within seconds, and a chain
        priced off a stale forward showed put-call parity out by 715 while
        every quote in it was internally exact. The maker was consistent with a
        price that no longer existed.

        Falls back to prints, then to the opening reference, so a book that has
        not opened still gets a chain rather than nothing.
        """
        book = self.books[symbol]
        ticks = book.mid if book.mid is not None and book.updated_at else None
        if ticks is None:
            ticks = self._anchor.get(symbol, self.reference.get(symbol))
        if ticks is None:
            return None
        return float(ticks) * float(self.instruments[symbol].tick_size)

    def _net_options(self, underlying_symbol: str) -> float:
        """Net option lots held on this underlying, long minus short.

        Reported and not acted on. It is the exposure a delta limit cannot
        see, because a call and a put struck at the same price have deltas of
        opposite sign and identical sensitivity to width: the put here is
        *defined* as ``C - (F - K)`` and neither ``F`` nor ``K`` moves with the
        dispersion, so a short straddle nets to roughly zero delta while
        carrying the whole of the width risk. Measured on seed 7 over 600s
        this reaches -14,618 lots on one maker against a per contract limit of
        1,200, which is the shape of the surface being too narrow rather than
        the shape of a position anybody chose. See the constructor for what
        was tried against it and why it was taken out.

        Digitals are left out and it is not an oversight. An option is worth
        more the wider the law is, whatever its strike, so its exposure to
        width has one sign. A digital's does not: struck out of the money it
        gains from a wider law and struck in the money it loses, so adding one
        here would cancel an option position that carries real risk against a
        binary that carries the opposite.
        """
        total = 0.0
        for symbol, member in self.chain.items():
            if member.underlying_symbol != underlying_symbol or member.is_digital:
                continue
            total += float(self.position.get(symbol, 0))
        return total

    def _net_delta(self, underlying_symbol: str, forward: float, now: int = 0) -> float:
        """Delta of everything held on this underlying, in future-equivalents.

        Digitals are counted at zero rather than at ``call_delta``, and the
        difference is not small. A digital's sensitivity to the forward is the
        density at its strike, not the probability above it, so charging it the
        probability would read a position of 300 binary lots as roughly 200
        future lots. In future-equivalents the real figure is nearer nothing: a
        lot of a win rate future is worth up to 10,000 currency and a lot of a
        binary written on the same rate is worth at most one, so the whole
        event ladder is a rounding error against a single future position and
        pretending otherwise would swing the skew on the option chain for no
        risk.
        """
        total = float(self.position.get(underlying_symbol, 0))
        for symbol, member in self.chain.items():
            if member.underlying_symbol != underlying_symbol or member.is_digital:
                continue
            held = self.position.get(symbol, 0)
            if not held:
                continue
            concentration = self.concentration_for(
                underlying_symbol, forward, member.scale, now
            )
            if concentration is None:
                continue
            delta = call_delta(forward, member.strike, member.scale, concentration)
            total += held * (delta if member.is_call else delta - 1.0)
        return total

    def _requote(self, ctx: SimulationContext, symbol: str) -> None:
        member = self.chain.get(symbol)
        if member is None:
            super()._requote(ctx, symbol)
            return

        forward = self._underlying_anchor(member.underlying_symbol)
        if forward is None:
            super()._requote(ctx, symbol)
            return
        sigma = self.dispersion_for(member.underlying_symbol, int(ctx.now))
        concentration = self.concentration_for(
            member.underlying_symbol, forward, member.scale, int(ctx.now), sigma
        )

        # Inventory moves the underlying, and the whole ladder follows. Skewing
        # each strike on its own would be the defect this class exists to fix,
        # only arriving through a different door.
        pull = max(
            -1.0,
            min(
                1.0,
                self._net_delta(member.underlying_symbol, forward, int(ctx.now))
                / self.delta_limit,
            ),
        )
        if sigma is None:
            shifted = forward
        else:
            shifted = forward - pull * self.skew_sigmas * sigma

        instrument = self.instruments[symbol]
        if concentration is None:
            # Not enough tape to have a view on width, so quote intrinsic: the
            # zero-volatility limit, and the one answer that needs no estimate.
            # Falling back to the plain maker here was worse than doing
            # nothing. It anchored each strike at the middle of its own
            # settlement range, so a call far out of the money opened near
            # 2,650 against a fair value under 20 and put-call parity was out
            # by a thousand. Intrinsic is wrong about time value and right
            # about everything else, including being monotone and convex across
            # strikes.
            #
            # A digital's zero-volatility limit is the whole payout on one side
            # of the strike and nothing on the other, which is the same
            # statement: it is what the contract settles at if the forward is
            # where it will end.
            if member.is_digital:
                pays = shifted > member.strike
                fair = member.payout if pays is member.above else 0.0
            else:
                fair = max(0.0, shifted - member.strike)
                if not member.is_call:
                    fair = max(0.0, member.strike - shifted)
        elif member.is_digital:
            fair = digital_value(
                shifted,
                member.strike,
                member.scale,
                concentration,
                member.above,
                member.payout,
            )
        else:
            fair = option_value(
                shifted, member.strike, member.scale, concentration, member.is_call
            )
        self._quote_around(ctx, symbol, fair / float(instrument.tick_size), abs(pull))

    def _quote_around(
        self, ctx: SimulationContext, symbol: str, fair_ticks: float, pressure: float
    ) -> None:
        """Post a two-sided quote around a fair value already in ticks.

        ``pressure`` is how much risk the maker is already carrying, on a scale
        where one is a full book, and it decides how wide the quote is. It is
        supplied by the caller rather than read off this contract's own
        position, and that is the same argument the skew makes one level up.

        The plain maker widens on ``inventory / position_limit`` in the book it
        is quoting, and copying that here reintroduced per-strike inventory
        into an option price by a route the skew had already been moved to
        avoid. It is invisible while the fair values differ and unmistakable
        once they do not. Every quote is clamped into its contract's settlement
        range, so when the fair value sits at or under the floor the bid pins
        to the floor while the ask stays at ``fair + half``, and the mid is
        then half the spread, which is to say a pure function of the position.
        Measured on seed 7 with the future at 4,264, all three calls on one
        chain were worth nothing and marked 1.88, 2.00 and 2.50 in ascending
        strike: a chain of worthless options getting *more* valuable the
        further out of the money they were, and free money to anyone who read
        it as a price.

        Driving the width from the chain's net delta instead gives one width to
        the whole ladder at each requote, so the quotes stay monotone in strike
        whatever the maker is holding, and the widening still means what it is
        supposed to mean: a desk deep in risk is more likely to be on the wrong
        side of whatever is moving the market, and charges for it. It measures
        the risk in the units that risk is actually denominated in. A maker
        long calls and long puts is short nothing in particular and quotes
        tight, which is right, and the per-strike position limit is still there
        to stop it accumulating an unbounded amount of either.
        """
        from arena.exchange.types import Price, Side

        inventory = self.position.get(symbol, 0)

        low, high = self.instruments[symbol].tick_bounds
        half = self.half_spread * (1.0 + 2.0 * pressure)

        # Clamped into the contract's own range, one side at a time. A quote
        # outside it is one the venue would refuse to collateralise, and an
        # option's fair value can sit within a hair of zero.
        #
        # Clamping the *centre* instead was a defect worth naming, because it
        # was invisible on any single book. Pushing the centre up to
        # ``floor + half`` keeps the bid legal, but it also makes the mid a
        # function of the half-spread, and the half-spread here widens with
        # inventory. Two worthless calls then price differently for no reason
        # other than how much of each the maker happens to hold: with the
        # future at 3,784 the calls struck at 4,600 and 4,650 were worth
        # nothing, and they marked at 1.63 and 68.38 respectively, purely
        # because one book carried a heavier position than the other. That is
        # an inverted chain (buy the 4,600, sell the 4,650, collect 133 that
        # settlement cannot take back) manufactured by the quoting rule rather
        # than by any view.
        #
        # Clamping each side independently keeps the same guarantee (no quote
        # leaves the settlement range) without letting spread width leak into
        # price. A call worth nothing bids at the floor and offers just above
        # it, however much of it the maker is holding.
        centre = min(max(fair_ticks, float(int(low))), float(int(high)))
        bid = Price(max(int(low), int(centre - half)))
        ask = Price(min(int(high), int(centre + half) + 1))
        if ask <= bid:
            # The range is too narrow to hold a two-sided quote at this width.
            # Give up a side rather than cross itself.
            if bid > int(low):
                bid = Price(int(bid) - 1)
            elif ask < int(high):
                ask = Price(int(ask) + 1)
            else:
                return

        if inventory < self.position_limit:
            self.post(ctx, symbol, Side.BUY, bid,
                       min(self.quote_size, self.position_limit - inventory))
        else:
            self.withdraw(ctx, symbol, Side.BUY)
        if -inventory < self.position_limit:
            self.post(ctx, symbol, Side.SELL, ask,
                       min(self.quote_size, self.position_limit + inventory))
        else:
            self.withdraw(ctx, symbol, Side.SELL)
