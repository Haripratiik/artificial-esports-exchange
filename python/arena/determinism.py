"""Primitives that make settlement byte-for-byte reproducible.

Every settlement artifact in the Artificial Esports Exchange is
content-addressed. If two runs disagree about a contract's settlement value,
the digests must disagree too. Otherwise a silent change in a metric
definition, a reference weight set, or a data source could rewrite history
without leaving a trace.

Three rules are enforced here and relied on everywhere else:

1. Serialization is canonical (sorted keys, no insignificant whitespace), so
   the digest of a structure depends only on its content.
2. Aggregation order is explicit, never dictionary insertion order. Float
   addition is not associative; a reordered sum is a different number.
3. Money-like values are quantized onto the contract's tick grid with
   banker's rounding, so a settlement price always lands on a tradeable tick.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Any

__all__ = [
    "canonical_json",
    "digest",
    "digest_of",
    "quantize_to_tick",
    "stable_sum",
    "stable_weighted_mean",
]

# Digests are prefixed so a bare hex string can never be mistaken for one.
_DIGEST_PREFIX = "sha256:"

# Settlement arithmetic runs at higher precision than float64 so that the
# quantization step, not accumulated Decimal error, decides the final tick.
_DECIMAL_PRECISION = 60


def canonical_json(value: Any) -> str:
    """Serialize ``value`` so that equal content always yields equal text."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_encode_unsupported,
    )


def _encode_unsupported(value: Any) -> Any:
    # Tuples already serialize as arrays; this catches the types our specs use
    # that json does not handle natively. Anything else is a bug, not a value we
    # should silently coerce into a string.
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, frozenset):
        return sorted(value)
    raise TypeError(f"{type(value).__name__} is not canonically serializable")


def digest(value: Any) -> str:
    """Content address of ``value``: ``sha256:<hex>`` over its canonical JSON."""
    payload = canonical_json(value).encode("utf-8")
    return _DIGEST_PREFIX + hashlib.sha256(payload).hexdigest()


def digest_of(payload: bytes) -> str:
    """Content address of raw bytes, for provenance over source files."""
    return _DIGEST_PREFIX + hashlib.sha256(payload).hexdigest()


def stable_sum(values: Iterable[float]) -> float:
    """Sum in a fixed order, smallest magnitude first, to bound rounding drift.

    Callers must still pass a deterministically ordered iterable; sorting by
    magnitude only makes the accumulation better conditioned, it does not
    rescue a nondeterministic input order.
    """
    ordered = sorted(values, key=abs)
    total = 0.0
    for value in ordered:
        total += value
    return total


def stable_weighted_mean(
    pairs: Sequence[tuple[str, float, float]],
) -> float:
    """Weighted mean over ``(key, weight, value)`` triples, ordered by key.

    Weights are renormalized over exactly the triples supplied, so a caller
    that has dropped low-sample strata gets a mean over the strata that
    remain rather than a silently deflated one.
    """
    if not pairs:
        raise ValueError("cannot take a weighted mean of zero terms")
    ordered = sorted(pairs, key=lambda item: item[0])
    total_weight = stable_sum(weight for _, weight, _ in ordered)
    if total_weight <= 0.0:
        raise ValueError("total weight must be positive")
    return stable_sum(weight * value for _, weight, value in ordered) / total_weight


def quantize_to_tick(value: float | Decimal, tick_size: str | Decimal) -> Decimal:
    """Snap ``value`` onto the ``tick_size`` grid using banker's rounding.

    Settlement values must be expressible as an integer number of ticks or the
    exchange cannot represent the closing print. Half-even rounding keeps
    repeated settlements from drifting upward the way half-up would.
    """
    tick = Decimal(str(tick_size))
    if tick <= 0:
        raise ValueError(f"tick_size must be positive, got {tick_size}")
    with localcontext() as ctx:
        ctx.prec = _DECIMAL_PRECISION
        amount = value if isinstance(value, Decimal) else Decimal(repr(value))
        ticks = (amount / tick).quantize(Decimal(1), rounding=ROUND_HALF_EVEN)
        return (ticks * tick).normalize()


def freeze_mapping(mapping: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    """Turn a mapping into a hashable, deterministically ordered tuple."""
    return tuple(sorted(mapping.items()))
