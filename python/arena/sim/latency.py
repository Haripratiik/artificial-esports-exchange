"""How long a message takes to get from one agent to another.

Latency is not a detail of this simulation, it is one of its subjects. Several
of the project's research questions (how much is faster information worth, does
a co-located market maker earn its edge, how quickly does a patch get priced)
are questions about the latency matrix and nothing else. So it is a first-class,
configurable object rather than a constant buried in the kernel.

The design follows ABIDES: a pairwise matrix plus a noise model, applied to every
message. Pairwise rather than per-agent because the interesting asymmetries are
directional: an agent can receive market data quickly while its orders still
take the slow path, which is exactly the situation a colocation experiment wants
to create.

Jitter is drawn from a stream seeded per ordered pair, so adding a new agent to a
configuration cannot perturb the latencies experienced by existing ones. Without
that, changing the agent population would silently change every other agent's
random draws and no two experiments would be comparable.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from arena.exchange.types import AgentId
from arena.sim.time import Duration, micros, millis

__all__ = ["LatencyModel", "UniformLatency", "PairwiseLatency"]


def _stable_seed(*parts: object) -> int:
    """A seed derived reproducibly from its parts.

    Python's built-in ``hash`` is randomized per process for strings, so using it
    here would make runs irreproducible across processes: the precise failure
    this module exists to avoid.
    """
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class LatencyModel:
    """Base: how long a message from ``sender`` to ``recipient`` takes."""

    def delay(self, sender: AgentId, recipient: AgentId) -> Duration:
        raise NotImplementedError


@dataclass
class UniformLatency(LatencyModel):
    """One latency for everyone. The neutral baseline.

    Useful as a control: any difference between agents in a run using this model
    cannot be a latency effect, which makes it the right comparison for the
    colocation experiments.
    """

    base: Duration = field(default_factory=lambda: millis(1))
    jitter: Duration = field(default_factory=lambda: micros(50))
    seed: int = 0

    def delay(self, sender: AgentId, recipient: AgentId) -> Duration:
        if self.jitter <= 0:
            return self.base
        rng = random.Random(_stable_seed(self.seed, sender, recipient))
        return Duration(max(1, int(self.base) + rng.randint(-int(self.jitter), int(self.jitter))))


@dataclass
class PairwiseLatency(LatencyModel):
    """Per-agent latencies to and from the exchange, with per-pair overrides.

    The common configuration is a hub: every agent talks to one exchange, and
    what differs is how far away each agent is. So the ordinary case is
    expressed as a per-agent number, with the full pairwise matrix available for
    experiments that need agents talking to each other.
    """

    default: Duration = field(default_factory=lambda: millis(1))
    per_agent: dict[AgentId, Duration] = field(default_factory=dict)
    overrides: dict[tuple[AgentId, AgentId], Duration] = field(default_factory=dict)
    jitter_fraction: float = 0.05
    seed: int = 0
    # Draw counter per ordered pair, so repeated messages between the same two
    # agents get different jitter rather than an identical delay every time.
    _draws: dict[tuple[AgentId, AgentId], int] = field(
        default_factory=dict, repr=False, compare=False
    )

    def delay(self, sender: AgentId, recipient: AgentId) -> Duration:
        pair = (sender, recipient)
        base = self.overrides.get(pair)
        if base is None:
            # A message involving an agent with a declared latency takes that
            # agent's latency; if both have one, the slower dominates, because a
            # message cannot arrive faster than its slowest leg.
            candidates = [
                self.per_agent[agent] for agent in pair if agent in self.per_agent
            ]
            base = max(candidates) if candidates else self.default

        if self.jitter_fraction <= 0:
            return Duration(max(1, int(base)))

        draw = self._draws.get(pair, 0)
        self._draws[pair] = draw + 1
        rng = random.Random(_stable_seed(self.seed, sender, recipient, draw))
        spread = int(int(base) * self.jitter_fraction)
        offset = rng.randint(-spread, spread) if spread > 0 else 0
        return Duration(max(1, int(base) + offset))
