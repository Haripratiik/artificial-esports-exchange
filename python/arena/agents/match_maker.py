"""A market maker that quotes every contract on a match from one belief.

The defect this exists to prevent has already been measured on this exchange's
standalone prediction markets: a violation in 3 of 3 runs, by 0.23 and 0.31 on
a contract bounded in [0, 1]. On a claim that can only ever pay zero or one,
0.31 is a third of the whole instrument sitting in the book as free money. The
option chain, priced by :class:`arena.agents.surface.SurfaceMarketMaker`,
measures 0 of 8 vertical violations over the same runs, and the reason is not
that the chain got lucky. It is that every strike is an expectation under one
law, so no two quotes are able to disagree.

This is that idea applied to a match. A match has an outcome set that is
mutually exclusive and exhaustive by construction, which `modes.SoloElimination`
states plainly: exactly one competitor survives a ten-way elimination, so the
set of "X wins" claims partitions the world and its prices must sum to one.
That is a fact about the world rather than a rule imposed on the market, which
is what makes enforcing it honest.

The maker holds one weight per competitor. For a match whose outcome set is
``O``, the belief is ``p_o = W_o / sum_o W_o`` with ``W_o`` the total weight of
the competitors that outcome is composed of, which is Luce's rule and is
exactly the law `match._draw_order` samples its finishing order from. Every
contract on that match declares what it pays at each outcome, and its fair
value is ``p . payoff``. Nothing is priced on its own.

Arbitrage-freeness is then a two-line argument rather than a check run
afterwards. Suppose a client buys ``q_i >= 0`` of each contract at the maker's
ask and sells ``r_i >= 0`` at its bid. Because each side is clamped
independently, ``bid_i <= f_i <= ask_i`` with ``f_i = p . payoff_i``, so the
package's expected profit under ``p`` is
``sum_i q_i (f_i - ask_i) + sum_i r_i (bid_i - f_i) <= 0``. A package that won
at every outcome would have strictly positive expectation under any measure of
full support. The belief has full support, because the update below is
multiplicative and exp is never zero, and because the inventory tilt is also
multiplicative. So no such package exists. The property holds because of how
the quotes are produced, and the enumeration in `tests/test_match_maker.py`
confirms rather than establishes it.

Three details carry the argument and each of them is a place the property is
easy to lose.

The **clamp is per side**, and one confident claim about why was wrong and the
measurement said so. The expectation was that clamping the centre would break
the bracket outright and hand a client the whole exhaustive set for more than
the unit it pays. It does not: pushing the centre to ``floor + half`` leaves
the bid *at* the floor, which is still under the fair value, so the bracket
survives and the linear program in `tests/test_match_maker.py` finds nothing.
What the centre clamp destroys is the mark. Measured on a ten way partition
with six claims worth 0.0049 and a half-spread of 0.0866, all six mark at
0.0900 whatever they are worth and the ten marks sum to 1.5000 on a set certain
to pay 1.0000, which is the same defect `CONTRIBUTING.md` records marking two
worthless calls 1.63 and 68.38. The per side clamp marks the same six at 0.0500
for a sum of 1.2600, all of it truncation. Rounding goes outward for the same
reason, floor for a bid and ceiling for an offer, which `TradingAgent.quote`
already does through `_on_grid` and which this maker repeats before the clamp so
the two cannot fight.

The bracket does get broken, and by the thing this maker is really about. A
per contract inventory skew, which is what `MarketMaker` does and which is
correct on one book, shades a claim by a fixed fraction of *that contract's*
settlement range. Short a count contract declared over [0, 9] on a field that
can only produce 0.70 or 4.00, 12% of the range is 108 ticks, and the maker
bids 4.48 for a claim that cannot pay more than 4.00. Measured: a package worth
0.480 at its worst outcome, riskless. It cannot happen here because the bid is
at most the fair value and the fair value is a convex combination of the
payoffs, so no quote can sit outside what the contract can actually pay.

**Inventory tilts the measure rather than shading the price.** Being long every
competitor in a mutually exclusive set is a constant payoff and therefore close
to riskless; being long one is a directional bet; and a maker that skews each
contract by its own position cannot tell those apart. So the exposure is
computed at match level as the portfolio's payoff at each outcome,
``Pi_o = sum_i position_i * payoff_i(o)``, and the maker prices off
``q_o proportional to p_o * exp(-lambda * Pi_o)``, the entropic indifference
measure of a desk with exponential utility. A flat book across the set leaves
``Pi`` constant, the exponential factors out of the normalisation and ``q = p``
exactly, so a riskless position produces **zero** skew rather than a small one.
A long position in one name down-weights that outcome and its quote falls,
which is the sign a maker wants. Critically ``q`` is still a probability with
full support, so tilting costs nothing in the argument above. This is the same
move `SurfaceMarketMaker` makes when it shifts the forward and reprices the
whole ladder instead of shading each strike.

**The width follows how far the match has resolved**, and the shape of that was
measured rather than assumed. A solo match removes one competitor at a time and
each removal is a repricing: the departing name's contract goes to zero and
every survivor's is renormalised up. Measured on seed 7 over 2,000 solo
matches, the mean L1 repricing per elimination runs 0.1401, 0.1787, 0.2154,
0.2568, 0.3090, 0.3818, 0.4789, 0.6578, 0.9900 from the first elimination to
the last, a growth of 7.07x. A maker holding one width across a match is
quoting the first elimination's risk into the last one's, and the last one is
where it gets picked off. The statistic it can compute from its own belief with
no extra input is the effective number of outcomes still live,
``exp(H(p))``, and ``len(outcomes) / exp(H(p))`` tracks the realised repricing
at a correlation of 0.8605 sample by sample, rising 4.79x over the same span.
It under-responds by a third at the very last elimination, which is recorded
here rather than tuned away: a gain chosen to recover 7.07 would be a gain
fitted to seed 7. Two alternatives were measured and are worse or dearer.
Twice the Simpson concentration correlates 0.8537. The exact reverse-Luce
expectation, which is available in closed form because a Plackett-Luce law over
finishing order implies a ``1/w`` law over who goes out next, correlates 0.8567
and needs the extra assumption that the next event is an elimination, so it
buys nothing for the assumption it costs.

The belief **learns from the tape**, because a maker that starts at the even
field and stays there is not aggregating anything. The rule is exponentiated
gradient on the squared pricing error, which is mirror descent under the
simplex geometry (Kivinen and Warmuth 1997): a print at ``x`` on a contract
with payoff ``v`` and settlement span ``s`` multiplies each outcome's weight by
``exp(eta * (x - f) * (v_o - f) / s^2)`` with ``f = p . v``, and the field's
weights are then rescaled to the total they had. Three reasons for that rule
and not another. It cannot leave the simplex or reach zero, so the full support
the arbitrage argument needs is a property of the update rather than of a
clamp, and measured over 4,000 matches the smallest normalised weight it ever
produced was 0.0185 on seed 7 and 0.0185 to 0.0335 across four, nowhere near a
floor and never needing one. Dividing by the span makes ``eta``
dimensionless, so one learning rate means the same thing on a binary bounded in
[0, 1] and on an eliminations contract bounded in [0, 9]. And the rescaling is
what lets a per-competitor weight outlive a match: a probability cannot be
carried to the next match because the next match has a different field, a
weight can, and rescaling the field to its former total leaves the competitors
who sat this one out exactly where they were, which is the right invariance
since the match said nothing about them.

Measured convergence, seed 7, 4,000 solo matches, learning from settlement
prints alone and starting from an even field. The realised win rate per
appearance ranges 0.036 to 0.178 against an even-field 0.100, and the maker's
implied rate per appearance tracks it to a **maximum absolute error of 0.0029
and a mean of 0.0010**, with the twelve competitors ranked in exactly the
realised order. Across seeds 7, 3, 11 and 41 the maximum error is 0.0029,
0.0033, 0.0030 and 0.0021, and the rolling error is under 0.02 within 400 to
600 matches. So the number is not seed 7 luck.

The ranking is exact on two of those four seeds and swaps one adjacent pair on
the other two, and the pairs are worth naming rather than glossing: HALCYON and
WISP on seed 3 finish 0.0003 apart against a standard error of 0.0063 on 3,330
appearances each, and RIFT and WISP on seed 11 finish 0.0004 apart. At that
separation the realised order is not evidence about which is stronger, so the
swap is the sample's noise rather than the maker's error. Every pair separated
by more than one standard error is ordered correctly on all four seeds.

What this maker deliberately does not do is read `roster.draw_strengths`. The
strengths are latent and unpublished for the reason `roster.py` gives, and a
maker that could see them would be pricing a lookup rather than aggregating a
tape, which would make every result measured on it a result about arithmetic.

**What a listing module has to supply**, which is the whole of the coupling
between this file and whatever lists the contracts. One `MatchListing` per live
match, carrying a stable key, the outcome set as a tuple of tuples of competitor
keys, and a `MatchLeg` per listed symbol whose ``payoff`` is that contract's
settlement at each outcome in the same order, with ``low`` and ``high`` equal to
the instrument's own ``value_bounds``. Then `list_match` when the match opens
and `close_match` when it settles, and nothing else: the maker subscribes to the
tape itself and needs no fills, no clock and no oracle. A leg whose declared
range disagrees with its instrument's is refused at listing rather than quoted,
because that disagreement is the only way the per side clamp could put a bid
above a fair value.

A contract that is not a function of the outcome supplies its expected
settlement given each outcome, and reinstalls the leg when that expectation
moves. Nothing here computes those, because what a book lists is the book's
business and a maker that enumerated contract kinds would need editing every
time one was added.

`arena.market.match_book` supplies exactly that, and the adapter is about twenty
lines and was measured rather than assumed. Its ensemble is a bag of drawn
`MatchResult` objects and its `settlement` scores the whole book against one of
them, so bucketing the ensemble by which side won and averaging the settlements
inside each bucket is the conditional expectation this wants, in the book's own
units. Measured on `list_match(7, 0, "solo")`, all 50 contracts at 4,000 draws:
the listing builds in 0.43s, the maker's fair values sit within 0.0100 of
`coherent_prices` on the same ensemble, which is exactly the sampling gap
between the Luce winner marginal and the ensemble's own frequency that
`match_book` records at that draw count, and every one of its `exclusive_sets`
sums to its declared total to nine decimals: the winner set to 1.000000000 and
the top five set to 5.000000000. Quoted mids sum to 1.000000000 and 5.005000000,
the second being half a tick spread over ten contracts rather than a gap in the
measure. Over 20 random beliefs and inventory states on that book the exhaustive
enumeration finds a maximum arbitrage of 0. The figures moved when `match_book`
cut a match from 270 contracts to 50; nothing here changed, which is the
coupling working. This module imports none of it: the
adapter belongs to whatever wires the two together, so a change to either side's
internals cannot reach across.
"""

from __future__ import annotations

import math

from arena.agents.market_maker import MarketMaker
from arena.exchange.types import AgentId, Price, Side
from arena.market.instrument import Instrument
from arena.sim.kernel import SimulationContext
from arena.sim.messages import TradePrint
from arena.sim.time import Duration, millis

__all__ = [
    "MatchLeg",
    "MatchListing",
    "MatchMaker",
    "solo_outcomes",
    "team_outcomes",
    "winner_legs",
    "field_leg",
]


class MatchLeg:
    """One contract on a match, and what it pays at each outcome.

    ``payoff`` is a vector aligned with its listing's ``outcomes``, in the
    contract's own settlement units. That is the whole of the interface: the
    maker never asks what kind of contract this is, so a listing may add a new
    shape of claim without this module learning about it, in the same way
    `derive_chains` makes a new strike quotable with no code change.

    ``low`` and ``high`` are the contract's declared settlement range, which is
    not the same thing as the range of ``payoff``. A contract crediting
    eliminations is bounded [0, 9] by its own terms whatever this particular
    field can produce, and it is the declared range that decides what a quote is
    allowed to be and what the span in the learning rule means. They default to
    the payoff's own extremes so a leg built by hand is still well formed.

    A payoff that is a function of the outcome states the contract exactly. A
    contract that is not a function of the outcome, which on a solo match means
    anything reading a placement other than first or a count of eliminations,
    states its expected settlement conditional on each outcome instead. The
    maker's coherence guarantee is with respect to the vector it is given, which
    is the correct scope: a conditional expectation that is wrong about the
    world is a mispricing, and a mispricing is not an arbitrage. It is also the
    conservative direction, because a package that pays at every full outcome
    also pays at every conditional expectation of those outcomes, so the
    enumeration over the small outcome set cannot miss a violation that the
    large one would have shown.
    """

    __slots__ = ("symbol", "payoff", "low", "high")

    def __init__(
        self,
        symbol: str,
        payoff: tuple[float, ...],
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        if not payoff:
            raise ValueError(f"{symbol}: a leg must pay something at every outcome")
        self.symbol = symbol
        self.payoff = tuple(float(v) for v in payoff)
        self.low = float(min(self.payoff)) if low is None else float(low)
        self.high = float(max(self.payoff)) if high is None else float(high)
        if self.high < self.low:
            raise ValueError(f"{symbol}: settlement range {self.low}..{self.high} is inverted")
        # A payoff outside the declared range is a contract that pays something
        # it says it cannot, and it would put the maker's clamp and its fair
        # value on opposite sides of the boundary. Refused at construction, in
        # the way `modes.check` refuses a match whose eliminations do not sum.
        for value in self.payoff:
            if not self.low <= value <= self.high:
                raise ValueError(
                    f"{symbol}: pays {value} outside its declared range "
                    f"{self.low}..{self.high}"
                )

    @property
    def span(self) -> float:
        return self.high - self.low


class MatchListing:
    """One match, its outcome set, and every contract written on it.

    ``outcomes`` is a tuple of tuples: each outcome is the group of competitors
    that wins together. A solo elimination has ten outcomes of one competitor
    each and a team objective has two of three, which is the only difference
    between the two formats as far as this maker is concerned. Grouping them
    this way is what lets a weight stay attached to a competitor rather than to
    a side, so the same learned weight prices a name in a solo match and inside
    a team, and `match.play` composes team strength the same way by summing
    ``exp(strength)`` over the side.

    The outcome set has to be mutually exclusive and exhaustive or the belief is
    not a belief about anything. Both formats supply one by construction and
    `modes.MatchFormat.check` is what enforces it on the result.
    """

    __slots__ = ("key", "outcomes", "legs")

    def __init__(
        self,
        key: str,
        outcomes: tuple[tuple[str, ...], ...],
        legs: tuple[MatchLeg, ...] = (),
    ) -> None:
        if len(outcomes) < 2:
            raise ValueError(f"{key}: a match with fewer than two outcomes is settled")
        seen: set[str] = set()
        for outcome in outcomes:
            if not outcome:
                raise ValueError(f"{key}: an outcome has no competitors in it")
            for competitor in outcome:
                if competitor in seen:
                    raise ValueError(
                        f"{key}: {competitor} appears in two outcomes, so the set is "
                        "not mutually exclusive and its prices cannot sum to one"
                    )
                seen.add(competitor)
        self.key = key
        self.outcomes = outcomes
        self.legs: dict[str, MatchLeg] = {}
        for leg in legs:
            self.add(leg)

    def add(self, leg: MatchLeg) -> None:
        if len(leg.payoff) != len(self.outcomes):
            raise ValueError(
                f"{leg.symbol}: {len(leg.payoff)} payoffs for {len(self.outcomes)} "
                f"outcomes of {self.key}"
            )
        self.legs[leg.symbol] = leg

    @property
    def competitors(self) -> tuple[str, ...]:
        return tuple(c for outcome in self.outcomes for c in outcome)


# ----------------------------------------------------------------------
# building the standard outcome sets and legs
# ----------------------------------------------------------------------
#
# These exist so the maker is usable and testable before `match_book` lists
# anything, and so the shape the book has to produce is written down somewhere
# executable rather than only in prose.


def solo_outcomes(field: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Ten entrants, ten outcomes, one survivor."""
    return tuple((competitor,) for competitor in field)


def team_outcomes(
    field: tuple[str, ...], team_size: int
) -> tuple[tuple[str, ...], ...]:
    """Sides in the order `match.play` cuts them, which is contiguous blocks."""
    if team_size <= 0 or len(field) % team_size:
        raise ValueError(f"{len(field)} entrants do not divide into sides of {team_size}")
    return tuple(
        tuple(field[i : i + team_size]) for i in range(0, len(field), team_size)
    )


def winner_legs(
    outcomes: tuple[tuple[str, ...], ...],
    symbol_for,
    payout: float = 1.0,
) -> tuple[MatchLeg, ...]:
    """The mutually exclusive exhaustive set: one claim per outcome.

    ``symbol_for`` maps an outcome to the symbol listing it, so this module
    never has to know how a book names its contracts.
    """
    legs = []
    for index, outcome in enumerate(outcomes):
        payoff = tuple(
            payout if other == index else 0.0 for other in range(len(outcomes))
        )
        legs.append(MatchLeg(symbol_for(outcome), payoff, 0.0, payout))
    return tuple(legs)


def field_leg(
    symbol: str,
    outcomes: tuple[tuple[str, ...], ...],
    members: frozenset[str],
    payout: float = 1.0,
) -> MatchLeg:
    """A claim that pays if any of a subset wins.

    Worth listing because it is the cheapest way to make a package interesting:
    a field claim and the individual claims inside it are a relation a client
    can trade against, and the maker's guarantee is that the two cannot be put
    together into a free lunch.
    """
    payoff = tuple(
        payout if any(c in members for c in outcome) else 0.0 for outcome in outcomes
    )
    return MatchLeg(symbol, payoff, 0.0, payout)


class MatchMaker(MarketMaker):
    """The plain maker, with every match contract priced off one belief."""

    def __init__(
        self,
        agent_id: AgentId,
        venue_id: AgentId,
        instruments: dict[str, Instrument],
        listings: tuple[MatchListing, ...] = (),
        wake_interval: Duration = millis(300),
        learn_rate: float = 0.1,
        skew_strength: float = 1.0,
        resolve_gain: float = 1.0,
        prior: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(agent_id, venue_id, instruments, wake_interval, **kwargs)
        # How far one print moves the belief. Dimensionless, because the update
        # divides by the contract's settlement span.
        #
        # 0.1 from the convergence measurement in the module docstring: over
        # 4,000 solo matches on seed 7 it reaches a maximum implied-rate error
        # of 0.0029 against realised rates spanning 0.036 to 0.178, and gets the
        # rolling error under 0.02 inside 600 matches. 0.02 is too slow, at
        # 0.0134 after the same 4,000 and with the twelve competitors ranked
        # wrongly. Above 0.2 the steady-state error stops improving, 0.0016 at
        # 0.2 against 0.0013 at 0.4, while every print moves the book further,
        # so the extra rate buys noise rather than speed.
        self.learn_rate = learn_rate
        # How hard a full inventory tilts the measure. One means a full position
        # limit in a single outcome multiplies that outcome's probability by
        # ``1/e`` before renormalisation.
        self.skew_strength = skew_strength
        # How hard the width follows the resolution of the match. One is the
        # measured relationship used unmodified; see the module docstring for
        # what it recovers and what it does not.
        self.resolve_gain = resolve_gain
        # The prior over the field. An even field unless told otherwise, which
        # is the only honest starting point: the strengths that decide the
        # answer are latent by design and nothing here may read them.
        self.weights: dict[str, float] = dict(prior or {})
        self.listings: dict[str, MatchListing] = {}
        # Symbol to listing, so a print or a requote finds its match in one
        # lookup rather than a scan over every live match.
        self.listing_of: dict[str, MatchListing] = {}
        # Prints that moved the belief, counted per match. A maker whose belief
        # never moved is one whose feed was never wired, which is the failure
        # mode `CONTRIBUTING.md` calls a guard that never fires, and a counter
        # is how it becomes visible instead of silent.
        self.learned: dict[str, int] = {}
        for listing in listings:
            self.list_match(listing)

    # ----------------------------------------------------------------------
    # listing
    # ----------------------------------------------------------------------

    def list_match(self, listing: MatchListing) -> None:
        """Register a match and everything written on it.

        A leg that claims to pay outside what its instrument says it can settle
        at is refused rather than quoted. It is the only way the per side clamp
        could put a bid above a fair value, since the clamp uses the
        instrument's bounds and the fair value uses the leg's payoffs, and it is
        an instance of the failure `CONTRIBUTING.md` counts five times: a number
        crossing a boundary in a unit its label does not claim, with one
        consumer compensating and the rest wrong.
        """
        for symbol, leg in listing.legs.items():
            instrument = self.instruments.get(symbol)
            if instrument is None:
                continue
            low, high = instrument.value_bounds
            if leg.low < float(low) - 1e-12 or leg.high > float(high) + 1e-12:
                raise ValueError(
                    f"{symbol}: the leg pays over {leg.low}..{leg.high} and the "
                    f"instrument settles over {float(low)}..{float(high)}"
                )
        self.listings[listing.key] = listing
        for competitor in listing.competitors:
            self.weights.setdefault(competitor, 1.0)
        for symbol in listing.legs:
            self.listing_of[symbol] = listing
        self.learned.setdefault(listing.key, 0)

    def close_match(self, key: str) -> None:
        """Drop a settled match, so its weights stop being consulted.

        The learned weights survive, which is the point of holding a weight per
        competitor rather than a probability per match.
        """
        listing = self.listings.pop(key, None)
        if listing is None:
            return
        for symbol in listing.legs:
            self.listing_of.pop(symbol, None)
        self.learned.pop(key, None)

    # ----------------------------------------------------------------------
    # the belief
    # ----------------------------------------------------------------------

    def belief(self, key: str) -> tuple[float, ...]:
        """The probability of each outcome, from the weights alone.

        Luce's rule over the outcome's total weight, which is the law
        `match._draw_order` samples from and the way `match.play` composes a
        side's strength. No inventory in it: this is what the maker thinks the
        world is, separately from what it is holding.
        """
        listing = self.listings[key]
        totals = [
            sum(self.weights.get(c, 1.0) for c in outcome) for outcome in listing.outcomes
        ]
        # Every outcome carries mass, and it is refused here rather than assumed.
        # The multiplicative update cannot produce a zero and neither can the
        # tilt, so the only way one arrives is a caller handing in a prior that
        # declares an outcome impossible. That is a different maker: a package
        # is arbitrage free against this one because the measure it prices under
        # has full support, and an outcome at zero is one the quotes stop paying
        # attention to while the contract still settles on it.
        for total, outcome in zip(totals, listing.outcomes):
            if total <= 0.0:
                raise ValueError(
                    f"{key}: outcome {'+'.join(outcome)} carries no weight, so the "
                    "belief has no opinion about a thing that can still happen"
                )
        mass = sum(totals)
        return tuple(t / mass for t in totals)

    def exposure(self, key: str) -> tuple[float, ...]:
        """What the maker's whole book on this match pays, outcome by outcome.

        The one number that tells a riskless position from a directional one.
        Long every claim in the exhaustive set is a constant vector; long one
        name is a spike. Computing it per contract instead is what makes a maker
        misprice the difference, and it is the same error
        `SurfaceMarketMaker._net_delta` exists to avoid on the option chain.
        """
        listing = self.listings[key]
        payoff = [0.0] * len(listing.outcomes)
        for symbol, leg in listing.legs.items():
            held = self.position.get(symbol, 0)
            if not held:
                continue
            for index, value in enumerate(leg.payoff):
                payoff[index] += held * value
        return tuple(payoff)

    def measure(self, key: str) -> tuple[float, ...]:
        """The belief tilted by inventory. Every quote on the match comes off it.

        ``q_o proportional to p_o * exp(-lambda * Pi_o)``, the entropic
        indifference measure of a desk with exponential utility, with
        ``lambda = skew_strength / position_limit`` so a full position in one
        outcome tilts by ``skew_strength``. Two properties matter and both are
        exact rather than approximate. A constant exposure factors out of the
        normalisation, so a book long the whole mutually exclusive set is
        quoted at zero skew rather than at a small one. And ``q`` is a strictly
        positive probability whatever the book is, so the whole no-arbitrage
        argument survives the tilt untouched.
        """
        p = self.belief(key)
        payoff = self.exposure(key)
        lam = self.skew_strength / max(1, self.position_limit)
        tilt = [-lam * value for value in payoff]
        # Centred before exponentiating, which is a no-op on the result because
        # a constant cancels in the normalisation, and is what keeps the
        # exponent in range. Clipped at 40 below the top: a tilt that assigns an
        # outcome 4e-18 of the mass is no longer a maker expressing a view about
        # its inventory, and letting it run to an underflow would be the one way
        # this measure could lose the full support the argument depends on.
        top = max(tilt)
        weighted = [
            prob * math.exp(max(-40.0, value - top)) for prob, value in zip(p, tilt)
        ]
        mass = sum(weighted)
        return tuple(w / mass for w in weighted)

    def resolution(self, key: str) -> float:
        """How far the match has collapsed, as a width multiplier at or above 1.

        ``len(outcomes) / exp(H(p))`` is the opening effective field size over
        the current one, and the module docstring records what it was measured
        against. Read off the belief rather than the tilted measure, because how
        far the match has run is a statement about the world and not about what
        the maker is holding.
        """
        p = self.belief(key)
        entropy = -sum(v * math.log(v) for v in p if v > 0.0)
        live = math.exp(entropy)
        return (len(p) / live) ** self.resolve_gain

    def pressure(self, key: str) -> float:
        """How much risk the book on this match is carrying, on a scale to one.

        The spread of the tilt across outcomes, not the size of any one
        position. A flat book across the exhaustive set scores zero however
        large it is, which is right, and a full limit in one name scores
        ``skew_strength``. It is the same quantity that moves the price, so the
        maker cannot widen for a risk it is not also skewing for.
        """
        payoff = self.exposure(key)
        lam = self.skew_strength / max(1, self.position_limit)
        return min(1.0, lam * (max(payoff) - min(payoff)))

    # ----------------------------------------------------------------------
    # pricing
    # ----------------------------------------------------------------------

    def fair(self, symbol: str) -> float | None:
        """What one contract is worth, in its own settlement units.

        ``None`` for anything this maker has no listing for, which is how a
        symbol falls through to the plain maker rather than being priced off a
        belief that does not cover it.
        """
        listing = self.listing_of.get(symbol)
        if listing is None:
            return None
        return self._value(self.measure(listing.key), listing.legs[symbol])

    @staticmethod
    def _value(measure: tuple[float, ...], leg: MatchLeg) -> float:
        return sum(q * v for q, v in zip(measure, leg.payoff))

    def quote_ticks(self, symbol: str, half: float | None = None) -> tuple[int, int] | None:
        """The two-sided quote in integer ticks, or ``None`` if not a match leg.

        Public because it is the thing worth testing. Everything the maker
        promises is a statement about this pair: that it brackets the fair
        value, that it never leaves the contract's settlement range, and that
        every leg of a match produced one from the same measure.
        """
        listing = self.listing_of.get(symbol)
        if listing is None:
            return None
        if half is None:
            half = self.half_spread * (
                1.0 + 2.0 * self.pressure(listing.key)
            ) * self.resolution(listing.key)
        fair = self._value(self.measure(listing.key), listing.legs[symbol])
        return self._bracket(symbol, fair, half)

    def _bracket(self, symbol: str, fair: float, half: float) -> tuple[int, int]:
        """Round outward, then clamp each side, and never the other way round.

        Both steps only ever move a bid down and an offer up, so
        ``bid <= fair <= ask`` holds at the end whatever the half-spread is and
        wherever the fair value sits in the range. That inequality is the whole
        no-arbitrage argument, and every line here exists to keep it.

        Clamping is per side, which is the rule `CONTRIBUTING.md` states after
        the centre clamp marked two worthless calls 40 points apart. It matters
        more here than it did there, because on a mutually exclusive exhaustive
        set the mid is not decoration. A centre pushed to ``floor + half``
        detaches the mid from the fair value entirely: measured on a ten way
        partition with six claims worth 0.0049 apiece and a half-spread of
        0.0866, all six mark at 0.0900 whatever they are worth, and the ten
        marks sum to 1.5000 on a set certain to pay 1.0000. The same belief
        under the per side clamp marks them at 0.0500 and sums to 1.2600, which
        is the truncation and nothing else.

        Rounding is outward for the same reason and has a hazard of its own.
        ``int()`` truncates toward zero, so on a contract whose range dips below
        zero it rounds a negative bid *up*, which is the wrong way; ``floor``
        and ``ceil`` do not care where zero is.

        The epsilon is the float representation of the tick grid and nothing
        else. A fair value of 0.10 on a one cent tick is 9.999999999999998
        ticks, not 10, so an exact ``floor`` puts the bid a whole tick below
        where the arithmetic says, and on a ten way partition that is half a
        tick of mid per claim: measured on an even field, the mids summed to
        0.95 rather than 1.00 before this was here. It is bounded by ``half`` so
        it can never move a bid past the centre, which keeps the bracket exact
        rather than merely usually right.
        """
        instrument = self.instruments[symbol]
        tick = float(instrument.tick_size)
        low, high = instrument.tick_bounds
        centre = fair / tick
        grid = min(1e-9, half)
        bid = max(int(low), math.floor(centre - half + grid))
        ask = min(int(high), math.ceil(centre + half - grid))
        if ask <= bid:
            # The range cannot hold a two-sided quote this wide. Give a side up
            # rather than cross, and only ever by moving outward, because moving
            # inward is the one repair that could break the bracket.
            if bid > int(low):
                bid -= 1
            elif ask < int(high):
                ask += 1
        return bid, ask

    # ----------------------------------------------------------------------
    # learning from the tape
    # ----------------------------------------------------------------------

    def on_print(self, ctx: SimulationContext, print_: TradePrint) -> None:
        """A trade in a match contract is evidence about the field.

        Converted to settlement units here and nowhere else, which is the rule
        `CONTRIBUTING.md` states for a unit crossing a boundary. The engine
        matches in integer ticks and the belief is a float model, so the one
        conversion lives at the edge and every quote goes back out through
        ``Price(int(...))``.

        It learns from its own prints too, and that is deliberate rather than
        an oversight. These makers are most of the book, so a print is usually
        at this maker's own bid or offer and the residual is minus or plus the
        half-spread by construction, which `MarketMaker.on_print` records as the
        reason its own anchor gain is fixed. What survives that is the side: a
        print at the offer is somebody paying up and a print at the bid is
        somebody hitting out, so a belief moved by the signed residual is moved
        by the flow imbalance, which is the sequential-trade channel a maker is
        supposed to learn from. What it is not is a fast estimator of the level,
        and the convergence figure in the module docstring is measured on
        settlement prints, where the residual is the outcome rather than the
        spread.
        """
        listing = self.listing_of.get(print_.symbol)
        if listing is not None:
            tick = float(self.instruments[print_.symbol].tick_size)
            self.learn(print_.symbol, float(int(print_.price)) * tick)
        super().on_print(ctx, print_)

    def learn(self, symbol: str, price: float) -> None:
        """Move the weights toward a print. Exponentiated gradient, renormalised.

        Takes a price in settlement units and no context, so a study can drive
        it over a season of matches without a kernel, which is how the
        convergence number in the module docstring was measured.
        """
        listing = self.listing_of.get(symbol)
        if listing is None:
            return
        leg = listing.legs[symbol]
        span = leg.span
        if span <= 0.0:
            # A contract that pays the same at every outcome is a constant and
            # carries no information about the field. Its print is not evidence,
            # and dividing by its span would be dividing by zero.
            return
        p = self.belief(listing.key)
        fair = self._value(p, leg)
        # Clipped to one span. The venue clamps every quote into the settlement
        # range so a print cannot legitimately sit outside it, which makes this
        # a guard against a bad feed rather than a limit on the rule.
        residual = max(-span, min(span, price - fair))
        step = self.learn_rate * residual / (span * span)

        outcomes = listing.outcomes
        # Multiplied on the outcome and applied to each competitor in it, so the
        # outcome's total weight scales by exactly the multiplier and a weight
        # stays attached to a competitor rather than to a side.
        multiplier = [math.exp(step * (value - fair)) for value in leg.payoff]
        before = sum(
            sum(self.weights.get(c, 1.0) for c in outcome) for outcome in outcomes
        )
        after = sum(
            m * sum(self.weights.get(c, 1.0) for c in outcome)
            for m, outcome in zip(multiplier, outcomes)
        )
        if after <= 0.0:
            return
        rescale = before / after
        for m, outcome in zip(multiplier, outcomes):
            for competitor in outcome:
                self.weights[competitor] = (
                    self.weights.get(competitor, 1.0) * m * rescale
                )
        self.learned[listing.key] = self.learned.get(listing.key, 0) + 1

    # ----------------------------------------------------------------------
    # quoting
    # ----------------------------------------------------------------------

    def act(self, ctx: SimulationContext) -> None:
        """Quote every match from one measure, then everything else as before.

        Grouped by match rather than looped over symbols, and that is not an
        optimisation. Computing the measure once per match per wakeup is what
        makes "no two quotes on this match disagree" structural: every leg is
        priced from the same vector at the same instant, so the property cannot
        be lost to a belief that moved between two contracts.
        """
        done: set[str] = set()
        for key in sorted(self.listings):
            listing = self.listings[key]
            measure = self.measure(key)
            half = self.half_spread * (
                1.0 + 2.0 * self.pressure(key)
            ) * self.resolution(key)
            for symbol in sorted(listing.legs):
                if symbol not in self.instruments:
                    continue
                self._quote_leg(ctx, symbol, listing.legs[symbol], measure, half)
                done.add(symbol)
        for symbol in sorted(self.instruments):
            if symbol not in done and symbol not in self.listing_of:
                super()._requote(ctx, symbol)

    def _requote(self, ctx: SimulationContext, symbol: str) -> None:
        """A single-symbol requote, for callers that still work that way.

        Falls through to the plain maker for anything with no listing, in the
        way `SurfaceMarketMaker._requote` does for a strike whose future is not
        listed. A match contract with no belief behind it would otherwise be
        anchored at the middle of its own settlement range, which on a claim
        that pays zero or one is a standing quote at 0.50 for the session.
        """
        listing = self.listing_of.get(symbol)
        if listing is None:
            super()._requote(ctx, symbol)
            return
        half = self.half_spread * (
            1.0 + 2.0 * self.pressure(listing.key)
        ) * self.resolution(listing.key)
        self._quote_leg(
            ctx, symbol, listing.legs[symbol], self.measure(listing.key), half
        )

    def _quote_leg(
        self,
        ctx: SimulationContext,
        symbol: str,
        leg: MatchLeg,
        measure: tuple[float, ...],
        half: float,
    ) -> None:
        bid, ask = self._bracket(symbol, self._value(measure, leg), half)
        inventory = self.position.get(symbol, 0)
        # The per contract limit stays, and it is doing a different job from the
        # tilt. The tilt moves the price so the trade that flattens the book is
        # the attractive one; the limit stops any single claim accumulating
        # without bound while the tilt is busy being zero because the rest of
        # the set happens to offset it.
        if inventory < self.position_limit:
            self.post(
                ctx,
                symbol,
                Side.BUY,
                Price(bid),
                min(self.quote_size, self.position_limit - inventory),
            )
        else:
            self.withdraw(ctx, symbol, Side.BUY)
        if -inventory < self.position_limit:
            self.post(
                ctx,
                symbol,
                Side.SELL,
                Price(ask),
                min(self.quote_size, self.position_limit + inventory),
            )
        else:
            self.withdraw(ctx, symbol, Side.SELL)
