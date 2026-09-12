"""How everyone on this exchange is actually doing, ranked.

Backs the leaderboard. The question it answers is the comparative one: not
"what is this agent doing right now", which `MarketRunner.agents` answers a row
per kernel agent, and not "what jobs are being done here", which
`dashboard.participants.directory` answers a row per role. This one puts a
market maker, a person at a browser and an outside system that has never
imported this package on the same table and sorts them by return.

**The roster is `venue.accounts`, not `market.agents`, and that is the whole
point.** `MarketRunner.agents` walks `self.market.agents`, so it can only ever
report participants this market's own kernel constructed. The two systems in
`connectors/` reach the venue over the published HTTP API and hold nothing but
an account, and they are precisely what a leaderboard exists to compare the
resident agents against. Walking the accounts also picks up the venue's own
treasury, which no roster built from agents can see: it has an account and it
has never had an agent.

The trade in the other direction is real and is the right one. An agent seated
without an account does not appear here, because the venue opens accounts
lazily and an account is where every figure on this table comes from. A row
with no cash, no equity and no return is not a competitor, it is an empty line.

**Money is a string in price units.** `Account.equity` answers in the integer
*minor* units the ledger is kept in, and publishing that raw has cost this
project real bugs twice: a maker worth 113,125,513.21 drawn on the participants
table as "113125513.21M", and a person's own seat drawn as "143745.00M" beside
a header reading "143.7k". `dashboard/state.py` records both. Everything here
goes out through `arena.portfolio.money.from_money`, as a string, exactly as
`Account.to_dict` and every other payload in this package does.

**A missing number is `None` and never "0.00".** They are different facts and
the difference is the leaderboard's whole job: `"0.00"` says a book is flat,
`None` says nobody could value it. The same rule covers `return_pct` on an
account that opened at zero, which the venue's treasury does by construction,
and `fills` for an account with no agent behind it, which every API client is.

Nothing here raises on a degenerate market. A venue that cannot produce marks,
an account whose positions cannot be read, a market with no kernel: each one
costs the fields it actually covers and leaves the rest standing. An operator
screen that returns 500 because one agent is unaccounted for is worse than one
showing a gap where that agent's equity should be.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from arena.market.venue import FEE_ACCOUNT_ID
from arena.portfolio.money import from_money
from arena.sim.time import NANOS_PER_SECOND

__all__ = ["standings"]

# The three groups, fixed here so a header can draw a total for each of them
# even on a market where nobody is in one. A missing key and a zero are the
# same fact for a count, unlike for a price.
KINDS = ("resident", "seat", "operator")


# -- classification --------------------------------------------------------
#
# Agent classes are imported inside the functions that test against them, the
# way `state.py` and `participants.py` both do it. A reporting surface has no
# business dragging every agent implementation into the import graph of
# whatever asked it a question.


def _operator_label(agent_id: str, agent: Any) -> str | None:
    """The venue's own machinery, or None if this is a participant.

    Tested first, before anything else, because two of the three are seated in
    `market.agents` alongside the traders. They are there because in a discrete
    event simulation only a participant with a wakeup can decide that time has
    passed, not because they trade, and a leaderboard that filed the session
    operator as a competing agent would be reporting the referee's score.

    The treasury goes by id because it is the one that has no agent at all: it
    is an account and only an account, deliberately, so the conservation check
    counts the fees it holds.
    """
    if agent_id == str(FEE_ACCOUNT_ID):
        return "Venue treasury"
    if agent is None:
        return None
    try:
        from arena.market.match_operator import MatchOperator
        from arena.market.operator import SessionOperator
        from arena.market.venue_agent import VenueAgent

        for kind, label in (
            (SessionOperator, "Session operator"),
            (MatchOperator, "Match operator"),
            (VenueAgent, "Venue"),
        ):
            if isinstance(agent, kind):
                return label
    except Exception:  # pragma: no cover - an agent module that will not import
        return None
    return None


def _resident_label(agent: Any) -> str:
    """What a resident agent does, in words, capitalised for a column.

    The ladder and its order are `dashboard/state.py`'s `_role`, which derives
    the answer by isinstance rather than by class name for a reason worth
    repeating: two tests went looking for the string "MarketMaker" and quietly
    stopped finding the options maker the day it arrived. A specialised maker
    is still a maker, so `SurfaceMarketMaker` and `MatchMaker` both read
    "Market maker" here, and the `id` column is where they are told apart.

    Falls back to the class name rather than to the id, because a reader who
    does not recognise `FlowTrader` still learns more from it than from
    `flow-03`.
    """
    try:
        from arena.agents.arbitrageur import Arbitrageur
        from arena.agents.bayesian import BayesianFundamental
        from arena.agents.flow import FlowTrader
        from arena.agents.fundamental import FundamentalTrader
        from arena.agents.market_maker import MarketMaker
        from arena.agents.noise import NoiseTrader

        for kind, label in (
            (MarketMaker, "Market maker"),
            (Arbitrageur, "Arbitrageur"),
            (BayesianFundamental, "Informed"),
            (FundamentalTrader, "Informed"),
            (FlowTrader, "Order flow"),
            (NoiseTrader, "Uninformed"),
        ):
            if isinstance(agent, kind):
                return label
    except Exception:  # pragma: no cover - an agent this ladder cannot classify
        pass
    return type(agent).__name__


def _seat_label(agent_id: str, seat: Any) -> str:
    """A person's own name, or the id when there is nobody to ask.

    An API client that holds an account without ever having taken a seat has no
    name anywhere in the market, and inventing one would be worse than showing
    the id it authenticates as.
    """
    name = str(getattr(seat, "display_name", "") or "").strip()
    return name or agent_id


def _classify(agent_id: str, agent: Any, seat: Any) -> tuple[str, str]:
    """Which group an account is in, and what to call it.

    **Seat is tested before resident, and the order is load bearing.**
    `LiveMarket.seat` appends the new `HumanAgent` to `market.agents` and joins
    it to the kernel, so a browser trader and an API client are kernel agents
    in every sense that matters to a roster built by walking that list. Asking
    "is this a kernel agent" first would file every person on the exchange as a
    resident bot, which is the one distinction this table exists to draw.

    The fall through is `seat` rather than `resident` for the same reason. An
    account with no agent object and no entry in `market.traders` is what an
    outside client looks like from in here: it reached the venue over the API,
    it holds an account, and nothing in this process built it.
    """
    operator = _operator_label(agent_id, agent)
    if operator is not None:
        return "operator", operator
    if seat is not None:
        return "seat", _seat_label(agent_id, seat)
    if agent is not None:
        return "resident", _resident_label(agent)
    return "seat", agent_id


# -- the figures -----------------------------------------------------------


def _money(minor: Any) -> str | None:
    """Minor units as a price-unit string, or None if they will not convert."""
    try:
        return str(from_money(int(minor)))
    except Exception:  # pragma: no cover - a balance that is not an integer
        return None


def _equity_minor(account: Any, marks: dict[str, Any] | None) -> int | None:
    """Equity in minor units, or None when this account cannot be valued.

    Kept in minor units here because `pnl` and `return_pct` are both derived
    from it, and deriving them from the rounded string would put a float back
    in the middle of a ledger whose entire argument is that it does not have
    one.
    """
    if marks is None:
        return None
    try:
        return int(account.equity(marks))
    except Exception:  # pragma: no cover - a position with no mark to value it
        return None


def _unrealized(account: Any, marks: dict[str, Any] | None) -> str | None:
    if marks is None:
        return None
    try:
        return _money(account.unrealized_pnl(marks))
    except Exception:  # pragma: no cover - a position with no mark to value it
        return None


def _attribute(account: Any, name: str) -> str | None:
    """One money field off the account, as a price-unit string."""
    try:
        return _money(getattr(account, name))
    except Exception:  # pragma: no cover - an account missing a ledger field
        return None


def _return_pct(equity: int | None, starting: int | None) -> float | None:
    """Return on opening capital, in percent.

    None on an account that opened at zero, which is not a degenerate case: the
    venue's treasury opens empty on purpose so its balance equals what the fee
    schedule collected. A percentage of nothing is undefined rather than
    infinite, and a leaderboard is not the place to decide which.

    Computed on the integer minor units and converted once at the end, so the
    ranking is a fact about the ledger rather than about two roundings.
    """
    if equity is None or not starting:
        return None
    try:
        return float(Decimal(equity - starting) / Decimal(starting) * 100)
    except Exception:  # pragma: no cover - a starting balance that will not divide
        return None


def _held(account: Any) -> int | None:
    """Symbols this account has a position in, counted on non-zero quantity.

    Not `len(account.positions)`. `Account.position` creates a flat `Position`
    on first enquiry and leaves it in the dict, so the raw length counts every
    symbol ever asked about rather than every symbol held. Measured on seed 7
    after sixty simulated seconds: 1,127 position entries across the venue's
    24 accounts against 1,092 actually held, with `noise-09` carrying 50
    entries for 43 positions. A trader that had gone home flat would read as
    still holding everything it ever touched.
    """
    try:
        positions = account.positions
    except Exception:  # pragma: no cover - an account with no position book
        return None
    try:
        return sum(1 for p in positions.values() if getattr(p, "quantity", 0))
    except Exception:  # pragma: no cover - a position book that will not walk
        return None


def _fills(agent: Any) -> int | None:
    """Fills this participant has taken, or None where nobody counted.

    `TradingAgent` counts its own, so a resident and a browser seat both have a
    real figure. An account reached over the API has no agent in this process
    and therefore no counter, and the honest answer there is that this market
    does not know rather than zero.
    """
    if agent is None:
        return None
    value = getattr(agent, "fills", None)
    try:
        return None if value is None else int(value)
    except Exception:  # pragma: no cover - a counter that is not a number
        return None


def _row(
    agent_id: str,
    account: Any,
    agent: Any,
    seat: Any,
    marks: dict[str, Any] | None,
    halted: set[str] | None,
) -> dict[str, Any]:
    """One competitor.

    Every field is guarded on its own, because one unreadable account is not a
    reason to lose the other twenty three rows.
    """
    kind, label = _classify(agent_id, agent, seat)
    equity = _equity_minor(account, marks)
    try:
        starting = int(account.starting_cash)
    except Exception:  # pragma: no cover - an account with no opening balance
        starting = None
    return {
        "id": agent_id,
        "label": label,
        "kind": kind,
        "starting_cash": None if starting is None else _money(starting),
        "cash": _attribute(account, "cash"),
        "equity": None if equity is None else _money(equity),
        "pnl": (
            None
            if equity is None or starting is None
            else _money(equity - starting)
        ),
        "return_pct": _return_pct(equity, starting),
        "realized": _attribute(account, "realized_pnl"),
        "unrealized": _unrealized(account, marks),
        "collateral": _attribute(account, "posted_collateral"),
        "free_cash": _attribute(account, "free_cash"),
        "positions": _held(account),
        "fills": _fills(agent),
        # None rather than False when the venue cannot produce its stop list.
        # "Not halted" is a claim about a participant and this would be a fact
        # about the venue, and a leaderboard that quietly reports one as the
        # other is the same defect as a missing price drawn as zero.
        "halted": None if halted is None else agent_id in halted,
    }


def _order(row: dict[str, Any]) -> tuple:
    """Best return first, unrankable last, ties broken on id.

    The tiebreak is what keeps the table still. Without it, two accounts on an
    identical return swap places between polls at Python's discretion and the
    rows move under the cursor of whoever is reading them, which looks exactly
    like the market doing something.
    """
    pct = row["return_pct"]
    return (pct is None, 0.0 if pct is None else -pct, row["id"])


# -- the payload -----------------------------------------------------------


def standings(market: Any) -> dict[str, Any]:
    """Every account on this venue, ranked by return on opening capital.

    Takes a `LiveMarket` and never raises on one.

    `kind` says where a participant came from rather than what it trades.
    **resident** is an agent this market's own kernel constructed and wakes.
    **seat** is a person at a browser or a system calling the published API,
    which are the same thing from in here: an account somebody outside this
    process is trading. **operator** is the venue's own machinery, which is
    ranked alongside everyone else because its numbers are as real as theirs,
    and which will generally sit wherever a non-trader sits.

    `conserved` is the venue's own conservation check reduced to a flag: total
    equity minus total starting capital, which has to be exactly zero because
    trading moves value between participants rather than creating it. It is
    `None` when the check could not be run, since a leaderboard claiming the
    books balance when nothing verified it is worse than one saying so.
    """
    venue = getattr(market, "venue", None)

    accounts: dict[str, Any] = {}
    if venue is not None:
        try:
            accounts = {str(k): v for k, v in venue.accounts.items()}
        except Exception:  # pragma: no cover - a venue that cannot list accounts
            accounts = {}

    # Marks once for the whole table, not once per account. `Venue.marks`
    # prices every listed symbol, and calling it per row would price the
    # listing twenty four times over to draw one screen.
    marks = None
    if venue is not None:
        try:
            marks = venue.marks()
        except Exception:  # pragma: no cover - a book with nothing to mark against
            marks = None

    agents: dict[str, Any] = {}
    for agent in list(getattr(market, "agents", None) or []):
        agent_id = getattr(agent, "agent_id", None)
        if agent_id is not None:
            agents[str(agent_id)] = agent
    # The venue agent is added straight to the kernel rather than to
    # `market.agents`, so it is invisible to the loop above. It is included so
    # that if it ever does hold an account it is classified as the venue's own
    # machinery instead of falling through to the outside-client default.
    venue_agent = getattr(market, "venue_agent", None)
    if venue_agent is not None:
        venue_agent_id = getattr(venue_agent, "agent_id", None)
        if venue_agent_id is not None:
            agents.setdefault(str(venue_agent_id), venue_agent)

    seats: dict[str, Any] = {}
    try:
        seats = {str(k): v for k, v in market.traders.items()}
    except Exception:  # pragma: no cover - a market with no seat register
        seats = {}

    halted: set[str] | None
    try:
        halted = {str(a) for a in venue.halted_participants}
    except Exception:  # pragma: no cover - a venue with no stop list
        halted = None

    rows = [
        _row(agent_id, account, agents.get(agent_id), seats.get(agent_id), marks, halted)
        for agent_id, account in accounts.items()
    ]
    rows.sort(key=_order)
    for position, row in enumerate(rows, 1):
        row["rank"] = position

    counts = dict.fromkeys(KINDS, 0)
    for row in rows:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1

    as_of: float | None
    try:
        as_of = int(market.kernel.now) / NANOS_PER_SECOND
    except Exception:  # pragma: no cover - a market with no kernel to ask
        as_of = None

    conserved: bool | None
    try:
        conserved = int(venue.conservation_check()) == 0
    except Exception:  # pragma: no cover - a venue that cannot value its accounts
        conserved = None

    return {
        "standings": rows,
        "as_of": as_of,
        "counts": counts,
        "conserved": conserved,
    }
