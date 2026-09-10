"""What a portfolio can lose, exactly.

Collateral here is charged per contract. An account holding a long future and a
short call on the same competitor posts against the worst case of each
separately, as though the world could be simultaneously terrible for both. And
it cannot, because both are functions of the same number. The arbitrageur feels
this most: its whole business is holding offsetting packages, and it pays
collateral on every leg of every one of them.

Netting them is what a clearing house is for. The usual objection is that
portfolio margining means a risk *model*, and a model is an estimate, and an
estimate is exactly what this project's collateral is not. That objection does
not apply here, and the reason is the same one that makes single-contract
collateral exact: every instrument settles as a known function of a bounded
scalar. So the worst case of a portfolio is

    min over level in [0, 1] of  sum_i quantity_i * payoff_i(level)

which is a minimisation of a **piecewise-linear function of one bounded
variable**. Its minimum is attained at an endpoint or at a kink, every kink is
known in advance (a call's strike, a put's strike, a binary's threshold), and
there are a handful of them. Evaluating at each is not an approximation of the
answer. It is the answer.

Two consequences worth being explicit about:

* **Only same-underlying positions net.** ``VANTA_SOLO_WR`` and
  ``QUILL_SOLO_WR`` are functions of *different* numbers, and nothing here
  knows how those two move together. Netting them would require a correlation,
  which would be an estimate, and then the whole guarantee is gone. They are
  collateralised separately, and that is not a limitation to be fixed later; it
  is the line between arithmetic and modelling.
* **Netting can only ever reduce the requirement**, never raise it, because the
  gross figure is the sum of per-contract worst cases and the net is the worst
  case of the sum. Anything else would be an arithmetic error.

And one place where the guarantee is genuinely weaker than the paragraph above,
stated here rather than left to be discovered. What settles is not
``payoff(level)`` but ``quantize_to_tick(payoff(level))``, and quantization
turns each leg into a staircase whose steps do not line up across legs of
different scale. Legs that share a scale and a tick cancel exactly: half-even
rounding is an odd function, so ``quantize(x) + quantize(K - x) = K`` whenever
``K`` is on the grid, which is why put-call parity and the four weekly legs of
a share still net to zero here. Legs of *different* scale do not: measured, a
long of one four week win rate future (scale 10,000) against a short of ten of
its first weekly leg (scale 1,000) is riskless before quantization and loses
1.25 after it, at a level of 0.00013. The loss is bounded above by ``sum
|quantity| * tick / 2`` (1.375 for that package), and the tight answer would
mean enumerating every level where any leg crosses a half-tick boundary, 40,000
of them for a single scale-10,000 contract on a 0.25 grid, which is not
something an order-entry check can afford. So this module reports the exact
minimum of the *unquantized* claim, and the residue above is a known, bounded
gap rather than a rounding error nobody measured.
"""

from __future__ import annotations

import math
from decimal import Decimal

from arena.contracts.payoff import Binary, Call, Linear, Payoff, Put
from arena.contracts.spec import ContractSpec
from arena.determinism import canonical_json

__all__ = ["kinks_of", "worst_case", "netting_benefit"]


def kinks_of(payoff: Payoff, bounds: tuple[float, float]) -> list[float]:
    """Levels where this payoff stops being the straight line it just was.

    A linear payoff has none. An option has one, at its strike. A binary's step
    is not a kink but a jump, so what is offered there is the last level on one
    branch and the first on the other; which of the two is adverse depends on
    the sign of the position, and the caller does not have to know which.

    Raises for a shape whose kinks are not known, rather than guessing at them.
    An approximate candidate set is an approximate worst case, and collateral
    here is arithmetic.
    """
    low, high = bounds
    found: list[float] = []

    if isinstance(payoff, (Call, Put)):
        if payoff.scale:
            found.append(payoff.strike / payoff.scale)
    elif isinstance(payoff, Binary):
        # A step: the value differs on either side of the threshold and there
        # is no level at which it is between them. Both branches have to be
        # offered because which of them is adverse depends on the sign of the
        # position, and the caller does not have to know which.
        #
        # Two things were wrong with offering `threshold` and a fixed nudge
        # past it, and both under-charged rather than over-charged.
        #
        # The nudge was `threshold * (1 + 1e-12) + 1e-12`, which lands *below*
        # a negative threshold as soon as its magnitude passes one. Measured on
        # a metric bounded by [-2, 2] with a step at -1.5, the "far side"
        # candidate came out at -1.5000000000005 (the same side as the
        # threshold), and a package holding that binary short against a long
        # linear leg was charged 1,500 against a loss of 1,999.998.
        #
        # And the pair only spans both branches when the comparison leaves the
        # threshold on the low branch. `>` and `<=` do; `>=` and `<` put it on
        # the high branch, so both candidates sat inside one branch and the
        # other was never evaluated at all. Measured on a short Linear(1000) at
        # 467 against a short `<` binary paying 1,000: 33 charged against a
        # loss of 533, the entire payout missing.
        #
        # So take the threshold plus whichever neighbour lies on the *other*
        # branch, found by asking the payoff rather than by assuming which way
        # the comparison points. Two candidates is all a two-branch function
        # has, and stepping one representable float cannot collide with the
        # threshold or jump over a neighbouring one the way a fixed epsilon can.
        below = math.nextafter(payoff.threshold, -math.inf)
        above = math.nextafter(payoff.threshold, math.inf)
        here = payoff.apply(payoff.threshold)
        other = below if payoff.apply(below) != here else above
        found.extend((payoff.threshold, other))
    elif not isinstance(payoff, Linear):
        # Sampling an unknown shape was not conservative, it was wrong. The old
        # fallback took 63 evenly spaced levels and called the smallest of them
        # the answer. Measured against a payoff that dips to -1,000 inside a
        # window 0.004 wide, all 63 samples missed the dip and the portfolio
        # was charged nothing for a position that loses 1,000, and a miss like
        # that is unbounded, so no sample count makes it safe.
        #
        # An exact worst case needs the kinks, and the kinks are a property of
        # the shape. A shape this module has never seen has to say where they
        # are before anything can be collateralised against it.
        raise TypeError(
            f"{type(payoff).__name__} has no declared kinks, so the worst case of a "
            "portfolio holding it cannot be computed exactly. Teach kinks_of the "
            "shape rather than sampling it: sampling can miss a minimum between "
            "samples by an unbounded amount, and collateral here is arithmetic "
            "rather than an estimate."
        )

    return [level for level in found if low <= level <= high]


def _claim_kinks(spec: ContractSpec, bounds: tuple[float, float]) -> list[float]:
    """Every kink in what the whole claim pays, settlement and stream alike.

    `claim_value` is settlement *plus* every scheduled payment, so a contract
    whose schedule pays an option-shaped amount has a kink the settlement payoff
    knows nothing about. Measured on a package short one Call(4600) against two
    shares each paying Call(4700) once: the only kink enumerated was 0.46, the
    minimum sits at 0.47, and the package was charged nothing for a loss of 100.
    """
    found = kinks_of(spec.payoff, bounds)
    if spec.distribution is not None:
        found = found + kinks_of(spec.distribution.payoff, bounds)
    return found


def worst_case(
    holdings: list[tuple[ContractSpec, int, Decimal]],
) -> Decimal:
    """The most this portfolio can lose, over every level the metric can take.

    ``holdings`` is ``(spec, signed quantity, price paid)``. Every spec must be
    written on the same underlying; grouping is the caller's job, because only
    the caller knows what "the same underlying" means for its world. A caller
    that gets it wrong is refused rather than answered.

    Returns a non-negative loss. Zero means the portfolio cannot lose anything
    at any level, which a fully hedged package genuinely cannot.
    """
    if not holdings:
        return Decimal(0)

    first = holdings[0][0]
    bounds = first.underlying.bounds()
    # The single-underlying rule used to be a sentence in this docstring and
    # nothing else. Measured, breaking it was silent and expensive: a long of 4
    # of one competitor's win rate future at 4,670 against a short of 4 of a
    # second competitor's at the same price netted to *zero* against a gross of
    # 40,000, because both are Linear(10000) and the arithmetic happily treated
    # two competitors as one number. That is a perfect-correlation assumption,
    # which is a risk model, which is the one thing this collateral is supposed
    # to be free of. Same subject is not enough either: a long future against a
    # short dispersion contract on the same competitor netted 26,030 against a
    # gross of 41,030, a win rate and a standard deviation being different
    # scalars that can be adverse at once.
    #
    # Keyed on the canonical form rather than on object identity so that the
    # rule enforced here is exactly the rule the venue groups by.
    written_on = canonical_json(first.underlying.to_dict())
    candidates = {float(bounds[0]), float(bounds[1])}
    for spec, _quantity, _price in holdings:
        if canonical_json(spec.underlying.to_dict()) != written_on:
            raise ValueError(
                f"{spec.contract_id} is not written on the same underlying as "
                f"{first.contract_id}, so the two are functions of different "
                "numbers. Netting them would need to know how those numbers move "
                "together, which is a correlation, which is an estimate. Group the "
                "holdings by underlying and call this once per group."
            )
        candidates.update(_claim_kinks(spec, bounds))

    worst = Decimal(0)
    for level in sorted(candidates):
        value = Decimal(0)
        for spec, quantity, price in holdings:
            settlement = Decimal(str(spec.claim_value(level)))
            value += Decimal(quantity) * (settlement - price)
        worst = min(worst, value)
    return -worst


def netting_benefit(
    holdings: list[tuple[ContractSpec, int, Decimal]],
) -> tuple[Decimal, Decimal]:
    """``(gross, net)`` collateral for the same holdings, for reporting.

    Gross is what charging each contract separately costs; net is what the
    portfolio can actually lose. The difference is the capital a clearing house
    hands back, and it is worth being able to quote rather than assert.
    """
    gross = Decimal(0)
    for spec, quantity, price in holdings:
        gross += spec.collateral_for(quantity, price)
    return gross, worst_case(holdings)
