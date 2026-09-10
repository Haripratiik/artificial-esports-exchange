"""The exchange's own operations: opening the market, and reopening it.

Everything a venue does to *itself* rather than in response to an order
(running the opening call, watching a paused symbol until its pause has run,
uncrossing it back into continuous trading) happens on a schedule, and a
schedule needs a participant in the simulation to keep it. That is this.

It is a participant rather than a method on the venue for the same reason the
venue is a participant rather than a function: something has to decide when
time passes, and in a discrete-event simulation the only thing that can decide
that is an agent with a wakeup. Putting a clock inside the ledger would make
the ledger's behaviour depend on when it happened to be asked a question.

Two jobs, both of them things real venues do and this one could not:

* **the open.** Symbols start in ``PRE_OPEN``. Orders accumulate and nothing
  matches; at the opening time every book uncrosses at a single price. A
  continuous book with no opening auction gives the first order to arrive the
  whole of the overnight information, which is precisely what an opening
  auction exists to prevent.
* **the reopen.** A symbol paused by the circuit breaker comes back through an
  auction once its pause has run, never straight into continuous trading.

Neither of these is a market participant. It holds no position, posts no
quotes, and its account never moves, so it does not appear in the conservation
check as anything but a zero.
"""

from __future__ import annotations

from typing import Any

from arena.exchange.session import SessionState
from arena.exchange.types import AgentId
from arena.market.venue import Venue
from arena.sim.kernel import SimulationContext
from arena.sim.time import Duration, seconds

__all__ = ["SessionOperator"]


class SessionOperator:
    """Runs the opening call, and reopens whatever the breaker paused."""

    def __init__(
        self,
        agent_id: AgentId,
        venue: Venue,
        opens_at: Duration = seconds(10),
        poll: Duration = seconds(1),
        venue_agent: Any = None,
    ) -> None:
        self.agent_id = agent_id
        self.venue = venue
        # The mailbox the uncross's fills go out through. Without it the
        # auction books fills into the ledger and tells nobody.
        self.venue_agent = venue_agent
        # How long the opening call runs before it clears. Long enough that
        # every agent has woken at least once and had something to say, or the
        # auction clears on whoever happened to be quickest, which is the
        # problem it exists to solve.
        self.opens_at = opens_at
        self.poll = poll
        self.opened = False
        self.reopens: list[dict[str, Any]] = []

    # -- lifecycle ---------------------------------------------------------

    def on_start(self, ctx: SimulationContext) -> None:
        for symbol in self.venue.registry.symbols:
            self.venue.begin_session(symbol)
        ctx.request_wakeup(self.opens_at)

    def on_finish(self, ctx: SimulationContext) -> None:
        pass

    def on_message(self, ctx: SimulationContext, sender: AgentId, message: Any) -> None:
        """Nothing addresses the operator. It acts on time alone."""

    def on_wakeup(self, ctx: SimulationContext) -> None:
        if not self.opened:
            self.opened = True
            for symbol in self.venue.registry.symbols:
                if self.venue.session(symbol) is SessionState.PRE_OPEN:
                    result = self._uncross(ctx, symbol)
                    self.reopens.append(
                        {
                            "symbol": symbol,
                            "reason": "open",
                            "price": None if result is None else int(result.price),
                            "volume": 0 if result is None else int(result.volume),
                        }
                    )
        else:
            for symbol in self.venue.reopen_due():
                result = self._uncross(ctx, symbol)
                self.reopens.append(
                    {
                        "symbol": symbol,
                        "reason": "reopen",
                        "price": None if result is None else int(result.price),
                        "volume": 0 if result is None else int(result.volume),
                    }
                )
        ctx.request_wakeup(self.poll)

    def _uncross(self, ctx: SimulationContext, symbol: str) -> Any:
        if self.venue_agent is not None:
            return self.venue_agent.uncross(ctx, symbol)
        return self.venue.uncross(symbol)
