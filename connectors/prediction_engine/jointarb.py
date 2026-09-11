"""Joint no-arbitrage across every market that shares a state space.

Ported from a prediction-market engine written against Kalshi, with one
change: there, a state is a final score and `covers` takes two integers,
because the venue it was written for lists football. Here a state is whatever
the contracts are written over, so it is an opaque object and `covers` takes
one. Nothing else about the argument moves.

The question is the engine's and is worth restating, because it is stronger
than the pairwise checks it subsumes:

    does ANY probability distribution over the states price every market
    inside its own bid-ask spread, simultaneously?

If none does then, by LP duality, a portfolio exists whose payoff is
non-negative in every state and whose cost is negative. That is riskless
profit, and unlike a statistical edge it needs no forecast to collect.

WHY IT IS WORTH RUNNING AGAINST THIS VENUE
------------------------------------------
This exchange prices every contract on a match as an expectation under one
distribution, so its *model* prices cannot disagree with each other. That is a
real guarantee and it is not the one this tests. Quotes are not model prices:
a maker widens for inventory, skews for adverse selection, and reprices off a
book it saw 160ms ago. Whether the prices actually standing on the screen
admit a consistent distribution is a different question, and nothing in this
repository asked it before.

CONSERVATIVE BY CONSTRUCTION
----------------------------
Prices enter as the touch that could actually be traded: a market may be
bought at its ask and sold at its bid, so it constrains its own probability to
[bid, ask]. Any state the contracts do not pin down sits in a free `OTHER`
state that no leg claims, which can only make the LP easier to satisfy. A
violation reported here is a lower bound on the real inconsistency, never an
artefact of how the space was cut.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import linprog

__all__ = ["Leg", "JointResult", "check"]


@dataclass(frozen=True, slots=True)
class Leg:
    """One quoted market, as an indicator over the state space."""

    ticker: str
    bid: float                       # probability units, 0..1
    ask: float
    covers: Callable[[Any], bool] = field(repr=False)


@dataclass(frozen=True, slots=True)
class JointResult:
    board: str
    n_legs: int
    feasible: bool
    slack: float                     # how far outside its spread the worst leg sits
    detail: str = ""

    @property
    def arbitrage(self) -> bool:
        """Infeasible means no consistent distribution exists, so arbitrage."""
        return not self.feasible


def check(board: str, legs: Iterable[Leg], states: Sequence[Any], *,
          tol: float = 0.0) -> JointResult:
    """Is there a distribution over `states` pricing every leg inside its spread?

    `tol` widens every spread by that many probability units before testing, so
    a violation has to exceed it to be reported. It exists because a one-tick
    numerical wobble is not an arbitrage, and because the alternative is a
    scanner reporting thousands of half-cent opportunities.

    Fewer than three legs is reported feasible rather than tested. Two legs on
    one state space can always be priced by some distribution, so a "pass"
    there would be arithmetic rather than evidence.
    """
    legs = [leg for leg in legs if leg.covers is not None]
    if len(legs) < 3:
        return JointResult(board, len(legs), True, 0.0, "fewer than 3 legs")

    n = len(states) + 1                    # + the free OTHER state
    other = n - 1

    # Feasibility as a minimisation of the worst spread violation. The
    # variables are the state probabilities plus one scalar slack; if the
    # optimal slack is zero then a consistent distribution exists.
    cost = np.zeros(n + 1)
    cost[-1] = 1.0

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for leg in legs:
        indicator = np.zeros(n + 1)
        for index, state in enumerate(states):
            if leg.covers(state):
                indicator[index] = 1.0
        upper = indicator.copy()
        upper[-1] = -1.0                   # sum(p over covered) - slack <= ask
        rows.append(upper)
        rhs.append(leg.ask + tol)
        lower = -indicator
        lower[-1] = -1.0                   # -sum(p over covered) - slack <= -bid
        rows.append(lower)
        rhs.append(-(leg.bid - tol))

    equality = np.zeros((1, n + 1))
    equality[0, :n] = 1.0                  # the probabilities sum to one
    bounds = [(0.0, 1.0)] * n + [(0.0, None)]
    bounds[other] = (0.0, 1.0)

    solved = linprog(
        cost, A_ub=np.array(rows), b_ub=np.array(rhs),
        A_eq=equality, b_eq=np.array([1.0]), bounds=bounds, method="highs",
    )
    if not solved.success:
        return JointResult(board, len(legs), True, 0.0, f"solver: {solved.message}")

    slack = float(solved.x[-1])
    feasible = slack <= 1e-9
    detail = "consistent" if feasible else f"worst leg is {slack:.4f} outside its spread"
    return JointResult(board, len(legs), feasible, slack, detail)
