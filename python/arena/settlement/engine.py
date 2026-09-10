"""Deterministic settlement.

The whole engine is one function. That is the point: settlement should be small
enough to read in one sitting and verify by inspection, because every research
claim this project makes eventually reduces to trusting it.

Order of operations matters and is fixed:

    1. check the oracle is the one the contract pinned
    2. resolve every atom, in canonical order
    3. apply the evidential bar (sample size), per atom
    4. evaluate the underlying algebra
    5. apply the payoff
    6. quantize onto the tick grid

Steps 3 and 6 are the ones people skip. Skipping 3 means settling on evidence
the contract said was insufficient; skipping 6 means printing a closing value
the exchange cannot represent.
"""

from __future__ import annotations

from decimal import Decimal

from arena.contracts.spec import ContractSpec
from arena.determinism import quantize_to_tick
from arena.settlement.oracle import MetricResolution, MetricUnavailable, Oracle
from arena.settlement.result import SettlementResult, SettlementStatus

__all__ = [
    "distributions",
    "settle",
    "ReferenceMismatch",
    "ReferenceLookahead",
    "SettlementOutOfBounds",
]


class SettlementOutOfBounds(Exception):
    """A contract settled outside the interval it declared it could settle in."""


class ReferenceMismatch(Exception):
    """The oracle's standardization snapshot is not the one the contract pinned.

    This is a hard error rather than a void. A void says "the world did not
    supply enough evidence"; this says "you wired the experiment up wrong", and
    silently voiding would hide a configuration bug behind a plausible-looking
    market outcome.
    """


class ReferenceLookahead(Exception):
    """The standardization snapshot post-dates the window it is settling.

    The most dangerous failure this engine can have, because it does not look
    like a failure. A snapshot fitted on data from inside its own observation
    window has seen the outcome, so its weights and priors encode the answer,
    and the settlement it produces will be *better* than an honest one, which
    is precisely why nothing about the result would invite suspicion.
    """


def distributions(spec: ContractSpec, oracle: Oracle) -> tuple[Decimal, ...]:
    """What this contract pays at each of its distribution dates, in order.

    Each window is resolved on its own evidence, which is the entire point: a
    share is worth the stream, and the stream is only interesting because its
    periods differ. Resolving the whole life once and dividing would produce a
    flat series that no amount of news could move.

    Empty for a contract that pays once at the end, which is every contract
    that is not a share.

    Raises rather than voiding when a period cannot be measured. A settlement
    can void because the world failed to produce evidence and everyone walks
    away whole; a payment cannot, because by the time it is due, earlier
    payments have already moved cash between accounts and there is no longer a
    state to walk back to. Deciding what a missing period should pay is a
    contract-terms question, and it is left open rather than guessed.
    """
    if spec.distribution is None:
        return ()

    # The same out-of-range guard `settle` applies, for the same reason, and
    # the only place a share ever gets one. A share's terminal payoff is
    # Linear(0), so its settlement bounds are [0, 0] and the check below
    # `settle` can never fire on it. Yet a share is the one instrument whose
    # cash moves *before* settlement, and `Venue.distribute` lowers the range
    # collateral is charged against by whatever was paid. So a payment outside
    # the range the schedule declared would silently move the bounds every
    # short in the contract is collateralised against, and nothing downstream
    # would notice. On the fixture a share's four weekly payments came out at
    # 468.5, 464.75, 468.75 and 467.25 against a declared range of [0, 1000],
    # so this is a guard rather than a change of behaviour.
    floor, ceiling = spec.distribution.payoff.bounds(spec.underlying.bounds())
    lowest = quantize_to_tick(floor, spec.tick_size)
    highest = quantize_to_tick(ceiling, spec.tick_size)

    paid: list[Decimal] = []
    for window in spec.distribution.windows:
        values = {ref: oracle.resolve(ref, window).value for ref in spec.atoms()}
        level = spec.underlying.evaluate(values)
        amount = quantize_to_tick(
            spec.distribution.payoff.apply(level), spec.tick_size
        )
        if not lowest <= amount <= highest:
            raise SettlementOutOfBounds(
                f"{spec.contract_id} would pay {amount} for the period beginning "
                f"{window.start.isoformat()}, outside the range [{lowest}, {highest}] "
                "its schedule declared. Either the oracle returned a metric outside "
                "its stated bounds or those bounds are wrong, and unlike a "
                "settlement, a payment cannot be walked back once it has moved cash."
            )
        paid.append(amount)
    return tuple(paid)


def settle(spec: ContractSpec, oracle: Oracle) -> SettlementResult:
    """Settle ``spec`` against ``oracle``, deterministically.

    Returns a VOID result when the evidence does not meet the contract's stated
    bar. Raises only when the *setup* is wrong, never when the world is.
    """
    if oracle.reference_id != spec.reference_id:
        raise ReferenceMismatch(
            f"{spec.contract_id} is pinned to reference {spec.reference_id!r} but the "
            f"oracle is configured with {oracle.reference_id!r}"
        )

    if oracle.reference_as_of > spec.window.start:
        raise ReferenceLookahead(
            f"{spec.contract_id} opens its observation window at "
            f"{spec.window.start.isoformat()}, but reference {spec.reference_id!r} was "
            f"estimated as of {oracle.reference_as_of.isoformat()}, after the window "
            "had already begun. Its weights and priors may encode the outcome. "
            "Estimate a new snapshot dated on or before the window start."
        )

    resolutions: list[MetricResolution] = []
    # spec.atoms() is sorted, so resolution order (and therefore the order of
    # resolutions in the record and its digest) does not depend on how the
    # underlying happened to be nested.
    for ref in spec.atoms():
        try:
            resolution = oracle.resolve(ref, spec.window)
        except MetricUnavailable as unavailable:
            return _void(spec, resolutions, unavailable.args[0])

        if resolution.sample_size < spec.policy.min_sample_size:
            return _void(
                spec,
                resolutions,
                f"{ref.key}: sample size {resolution.sample_size} below required "
                f"{spec.policy.min_sample_size}",
            )
        resolutions.append(resolution)

    values = {resolution.ref: resolution.value for resolution in resolutions}
    level = spec.underlying.evaluate(values)
    settlement_value = quantize_to_tick(spec.payoff.apply(level), spec.tick_size)

    # A settlement outside the contract's declared range means the oracle
    # returned something the contract never contemplated: a metric out of its
    # stated bounds, or bounds declared wrongly. Either way the collateral
    # posted against this contract was computed from a false premise, so this
    # is a hard error rather than a void: voiding would hide a solvency problem
    # behind a normal-looking outcome.
    low, high = spec.settlement_bounds
    if not low <= settlement_value <= high:
        raise SettlementOutOfBounds(
            f"{spec.contract_id} settled at {settlement_value}, outside its declared "
            f"range [{low}, {high}]. Either the oracle returned a metric outside its "
            "stated bounds, or the contract's bounds are wrong. In both cases the "
            "collateral held against this contract was computed from a false premise."
        )

    return SettlementResult(
        contract_id=spec.contract_id,
        spec_digest=spec.spec_digest,
        status=SettlementStatus.SETTLED,
        settlement_value=settlement_value,
        underlying_level=level,
        resolutions=tuple(resolutions),
    )


def _void(
    spec: ContractSpec,
    resolutions: list[MetricResolution],
    reason: str,
) -> SettlementResult:
    """Build a void record that still carries whatever evidence was gathered.

    The partial resolutions are kept on purpose. "Voided because this
    competitor's sample was thin" is a far more useful record than "voided",
    especially when the same contract template is about to be reused for the
    next window.
    """
    return SettlementResult(
        contract_id=spec.contract_id,
        spec_digest=spec.spec_digest,
        status=SettlementStatus.VOID,
        settlement_value=None,
        underlying_level=None,
        resolutions=tuple(resolutions),
        void_reason=reason,
    )
