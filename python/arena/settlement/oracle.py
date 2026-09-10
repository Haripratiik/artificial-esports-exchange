"""The boundary between the financial system and the outside world.

Everything on the exchange side of this interface is generic: it knows about
contracts, prices, and settlement. Everything on the far side knows about
matches, formats, and competitors. The Oracle protocol is the only thing that
crosses, which is what lets a second data-generating world be plugged into the
same exchange later.

An oracle answers exactly one question: *what was this metric, over this
window, according to the evidence?* It does not know what contract is asking or
what the answer is worth. That separation is deliberate: an oracle that could
see the payoff could, in principle, be tuned to it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from arena.contracts.spec import ObservationWindow
from arena.contracts.underlying import MetricRef

__all__ = [
    "SourceRef",
    "MetricResolution",
    "MetricUnavailable",
    "Oracle",
]


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Provenance for one input that fed a resolved metric.

    The digest is over the source bytes as ingested. If a settlement is ever
    disputed, this is what makes the argument finite: either the bytes hash to
    the recorded digest or the record is not describing the data that was used.
    """

    source_id: str
    digest: str
    rows: int

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "digest": self.digest, "rows": self.rows}


@dataclass(frozen=True, slots=True)
class MetricResolution:
    """A measured value, plus everything needed to defend it."""

    ref: MetricRef
    value: float
    sample_size: int
    sources: tuple[SourceRef, ...]
    # Free-form, but ordered and serialized into the settlement record: the
    # per-stratum counts, the coverage achieved, the shrinkage applied. This is
    # what turns a settlement from an assertion into an audit trail.
    diagnostics: tuple[tuple[str, Any], ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref.to_dict(),
            "value": self.value,
            "sample_size": self.sample_size,
            "sources": [source.to_dict() for source in self.sources],
            "diagnostics": [list(pair) for pair in self.diagnostics],
        }


class MetricUnavailable(Exception):
    """Raised when a metric cannot be measured to the standard the contract set.

    This is a normal outcome, not a failure. A contract whose evidence never
    materialized should void loudly rather than settle on whatever happened to
    be in the table.
    """

    def __init__(self, ref: MetricRef, reason: str, detail: str = "") -> None:
        self.ref = ref
        self.reason = reason
        self.detail = detail
        message = f"{ref.key}: {reason}"
        if detail:
            message = f"{message} ({detail})"
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {"ref": self.ref.to_dict(), "reason": self.reason, "detail": self.detail}


class Oracle(Protocol):
    """Resolves metric references against evidence.

    Implementations must be pure with respect to their inputs: the same
    reference, window, and underlying dataset must always produce the same
    resolution, including the same diagnostics ordering.
    """

    @property
    def reference_id(self) -> str:
        """Identity of the frozen standardization snapshot this oracle uses.

        The engine checks this against the contract so that a spec pinned to one
        snapshot can never be settled by an oracle configured with another.
        """
        ...

    @property
    def reference_as_of(self) -> datetime:
        """When the standardization snapshot was estimated.

        Exposed so the engine can enforce the last lookahead invariant: a
        snapshot fitted on data from inside the window it settles would have
        seen the outcome it is helping to price. The contract pins an id, but
        only the oracle knows the date behind it, so the check has to live
        here rather than in the spec.
        """
        ...

    def resolve(
        self,
        ref: MetricRef,
        window: ObservationWindow,
        policy_overrides: Mapping[str, Any] | None = None,
    ) -> MetricResolution:
        """Measure ``ref`` over ``window`` or raise :class:`MetricUnavailable`."""
        ...
