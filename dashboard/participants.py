"""Everyone on this exchange: the agents it seats, and the systems that call it.

Backs the Players screen. The other operator views answer "what is this agent
doing right now": `MarketRunner.agents` publishes a class name, a role word,
fills, rejects and equity, one row per agent id. That is the right payload for
a table somebody is watching and the wrong one for a reader who does not yet
know what a `SurfaceMarketMaker` is, or why three makers were seated instead of
one. This module answers the other question, and answers it by **role** rather
than by id: three market makers are one row carrying a count, because the
interesting fact about them is the job rather than that their ids run mm-1 to
mm-3.

**Every sentence below is read off the module it describes, or off a
measurement recorded in one.** The `measured` field carries a figure somebody
actually ran and names the file it was recorded in, and it is `None` where
there is no such figure rather than a number invented to fill the column. One
row carries `None` today, the portfolio manager, whose own documentation
records a design and a count of methods and no measurement.

**Nothing here decides whether an agent exists.** Families are matched against
`market.agents` by isinstance, so an agent that was not seated reports
`count: 0` and `live: False` and keeps the whole of its description. That is
deliberate rather than incidental. The arbitrageur is off by default on
evidence recorded in `arena/agents/arbitrageur.py` and `dashboard/build_market.py`,
and a screen that simply omitted it would make a considered default look like
an absence. The venue lookups are guarded for the same reason: an operator
screen that returns 500 because the arbitrageur is switched off is worse than
one that says it is switched off.

**`live` means two different things and the difference is load bearing.** For a
resident agent it means "seated in this market", which `market.agents` answers
exactly. For an outside system it would have to mean "is an API seat currently
authenticated", and the store that could answer that (`arena.api.keys.KeyStore`)
is held by `dashboard.server`, not by the market. Nothing reachable from a
`LiveMarket` knows. So both connected rows report `False` and the field is not
guessed at; a connector that is plainly described as unreachable from here is
better than one this module pretends to have measured.

Money leaves this payload the way it leaves every other one in this package:
through `arena.portfolio.money.from_money`, as a string, in price units rather
than in the ledger's integer minor units. `MarketRunner.session_state` records
what the alternative cost, which was a person's own seat drawn as
"143745.00M" beside a header reading "143.7k".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from arena.market.venue import FEE_ACCOUNT_ID
from arena.portfolio.money import from_money

__all__ = ["directory"]


@dataclass(frozen=True)
class _Family:
    """One role, and the test for whether anybody is doing it here.

    `match` is an isinstance test rather than a name or an id prefix, for the
    reason `MarketRunner._role` gives after two tests went looking for the
    string "MarketMaker" and stopped finding the options maker: a specialised
    maker is still a maker, and the class hierarchy already says so.
    """

    id: str
    name: str
    role: str
    strategy: str
    why: str
    source: str
    measured: str | None
    match: Callable[[Any], bool]


# -- what each family is, in its own module's words ------------------------
#
# Imported inside the function rather than at module scope, as `state.py` does
# for the same set. The Players screen is a reporting surface and has no
# business dragging every agent implementation into the import graph of
# whatever asks it a question.


def _resident_families() -> list[_Family]:
    from arena.agents.arbitrageur import Arbitrageur
    from arena.agents.bayesian import BayesianFundamental
    from arena.agents.flow import FlowTrader
    from arena.agents.fundamental import FundamentalTrader
    from arena.agents.market_maker import MarketMaker
    from arena.agents.match_maker import MatchMaker
    from arena.agents.noise import NoiseTrader
    from arena.agents.surface import SurfaceMarketMaker

    def plain_maker(agent: Any) -> bool:
        # Both specialised makers subclass `MarketMaker`, so the plain one is
        # the residue. Written as an exclusion rather than as `type(a) is` so
        # that a third specialisation lands in its own row instead of silently
        # inflating this one.
        return isinstance(agent, MarketMaker) and not isinstance(
            agent, (SurfaceMarketMaker, MatchMaker)
        )

    return [
        _Family(
            id="mm-surface",
            name="Options market maker",
            role=(
                "Quotes a whole option chain off one distribution, and is paid the "
                "spread for carrying the inventory that leaves it with"
            ),
            strategy=(
                "Prices every strike on an underlying as an expectation under a single "
                "law centred on its own anchor for that underlying, so the ladder is "
                "decreasing and convex in strike by arithmetic rather than by policy. "
                "The width of that law is estimated from the tape, as an exponentially "
                "weighted variance of how far prints land from the anchor, converted to "
                "a Beta concentration by moment matching. Inventory shifts the anchor "
                "and the whole ladder reprices, rather than shading each strike."
            ),
            why=(
                "The plain maker anchors each book on its own prints, so every strike "
                "was priced by a process that had never heard of the strike next to it. "
                "Nothing related them, so nothing kept them related, and the result was "
                "a riskless trade sitting in the book. The binaries are on the same "
                "ladder because P(F > K) is the digital's price and the same quantity "
                "the chain already computes."
            ),
            source="python/arena/agents/surface.py",
            measured=(
                "The defect it removes, measured on the live market: a call struck at "
                "4,700 marked at 72.7 while the same underlying's 4,600 strike marked "
                "at 59.1, which is free money in the book, and put-call parity was out "
                "by 35 ticks at the same moment (arena/agents/surface.py)."
            ),
            match=lambda agent: isinstance(agent, SurfaceMarketMaker),
        ),
        _Family(
            id="mm-plain",
            name="Plain market maker",
            role="Quotes both sides of one book at a time and is paid for the risk of doing so",
            strategy=(
                "Quotes around a reference and shades those quotes against its own "
                "position, so the trade that flattens it is the more attractive one: "
                "reservation = reference - inventory * skew, bid and ask a half spread "
                "either side. It respects a position limit, and widens when its view of "
                "the book is stale or the market has moved against its inventory."
            ),
            why=(
                "It is the control. `surface=False` restores it in place of the options "
                "maker, which is what the before and after in docs/GAPS.md was measured "
                "against, and a comparison has to stay runnable or the numbers in it are "
                "only claims. Deliberately not Avellaneda-Stoikov: inventory held to "
                "expiry here faces settlement risk rather than liquidation risk, and "
                "building that adaptation on a maker whose basic inventory control had "
                "not been watched working would be premature."
            ),
            source="python/arena/agents/market_maker.py",
            measured=(
                "It waits for a reference before quoting, because a maker that names the "
                "opening price itself paused every one of 26 symbols inside the first "
                "minute by tripping the circuit breaker (arena/agents/market_maker.py)."
            ),
            match=plain_maker,
        ),
        _Family(
            id="mm-match",
            name="Match market maker",
            role="Quotes every contract on a live match off one belief about who wins",
            strategy=(
                "Holds one weight per competitor and prices each contract as p . payoff "
                "under Luce's rule, which is the same law the match itself samples its "
                "finishing order from, so no two quotes on a match can disagree. "
                "Inventory tilts the measure rather than shading the price, by the "
                "entropic indifference measure of a desk with exponential utility, so a "
                "flat book across a mutually exclusive set produces exactly zero skew. "
                "The belief learns from the tape by exponentiated gradient on squared "
                "pricing error, which cannot leave the simplex or reach zero."
            ),
            why=(
                "A match's outcome set is mutually exclusive and exhaustive by "
                "construction, so its prices must sum to one. That is a fact about the "
                "world rather than a rule imposed on the market, which is what makes "
                "enforcing it honest. Given no instruments when it is built: every "
                "contract it quotes arrives with a match, through the same note and join "
                "path every other agent uses."
            ),
            source="python/arena/agents/match_maker.py",
            measured=(
                "Seed 7, 4,000 solo matches, learning from settlement prints alone and "
                "starting from an even field: its implied win rate per appearance tracks "
                "the realised one to a maximum absolute error of 0.0029 and a mean of "
                "0.0010, with the twelve competitors in exactly the realised order "
                "(arena/agents/match_maker.py)."
            ),
            match=lambda agent: isinstance(agent, MatchMaker),
        ),
        _Family(
            id="fund",
            name="Informed trader",
            role="Holds a private view of where a contract settles and trades the gap",
            strategy=(
                "Told the true settlement value, then given a noisy view of it scaled by "
                "a precision, and it trades on edge relative to its own uncertainty "
                "rather than on raw distance from price. Sizing by conviction is what "
                "stops a vague agent dominating a book by being wrong loudly. Its "
                "evidence arrives across the session rather than all at the open, and it "
                "starts from a prior measured over the season's first four weeks."
            ),
            why=(
                "This is the agent that makes prices mean something. Makers and noise "
                "traders give a market liquidity and volatility but no anchor; the "
                "informed population supplies the force that drags price toward the "
                "value the contract will actually settle at, which is what turns \"does "
                "this market aggregate information?\" into a question with an answer."
            ),
            source="python/arena/agents/fundamental.py",
            measured=(
                "Six of them, log spaced in precision, because two was an anecdote: both "
                "ran into their position limits about a minute in, the sharper one short "
                "its full 900 lots believing 4,687 against a true 4,669 while the market "
                "printed 5,005 (dashboard/build_market.py)."
            ),
            match=lambda agent: isinstance(agent, FundamentalTrader),
        ),
        _Family(
            id="fund-bayes",
            name="Bayesian informed trader",
            role=(
                "The same job as the informed trader, with its information counted in "
                "battles rather than in a free parameter"
            ),
            strategy=(
                "Given n battles drawn from the process the contract settles on, so its "
                "belief is Beta(a0 + k, b0 + n - k) with the prior taken from the "
                "reference snapshot's own mode prior and shrinkage strength. It values a "
                "contract as E[payoff(theta)] under that posterior and never as "
                "payoff(E[theta]): for a linear future the two coincide, for an option "
                "the difference is the entire time value."
            ),
            why=(
                "\"Precision 3.0\" means nothing. It cannot be compared across contracts, "
                "it has no natural prior, and it leaves \"how much is better information "
                "worth?\" with no denominator. One more battle observed is a quantity, so "
                "the question gets an answer in dollars per battle. The research harness "
                "in arena/research/experiment.py seats this one; the dashboard market "
                "seats the noise-scaled trader above, so this row normally reads as not "
                "seated and that is correct rather than a gap."
            ),
            source="python/arena/agents/bayesian.py",
            measured=(
                "Its error must fall as 1/sqrt(n) in battles observed. That is a hard, "
                "checkable property of a conjugate posterior rather than a fitted one, "
                "and the test suite checks it; a noise knob has no such constraint, so "
                "nothing about it could ever be wrong (arena/agents/bayesian.py)."
            ),
            match=lambda agent: isinstance(agent, BayesianFundamental),
        ),
        _Family(
            id="noise",
            name="Noise trader",
            role="Uninformed flow: the other side a maker earns a spread from",
            strategy=(
                "Random direction with a trend chasing bias, because a population of "
                "purely random traders is too benign: real uninformed flow chases and "
                "clusters, which is what produces the autocorrelated order flow and the "
                "occasional runs a maker actually has to survive. Each one touches six "
                "books per wake."
            ),
            why=(
                "Not filler. Without uninformed flow there is nobody for a maker to earn "
                "a spread from and no camouflage for an informed trader to hide behind, "
                "so both the market making and the information asymmetry questions "
                "become degenerate. Kyle's insider is profitable only because the maker "
                "cannot tell them apart from the noise."
            ),
            source="python/arena/agents/noise.py",
            measured=(
                "Breadth is six books a wake rather than one because one book a wake "
                "thins as the exchange grows: over 180 simulated seconds on seed 7, "
                "noise volume fell from 6,904 to 4,522 when matches took the listing "
                "from 47 symbols to 971 and the informed to uninformed ratio went from "
                "25.6 to 1 up to 42.3 to 1. Six takes uninformed flow to 3.0% of volume "
                "at 6.4 to 1 (arena/agents/noise.py, dashboard/build_market.py)."
            ),
            match=lambda agent: isinstance(agent, NoiseTrader),
        ),
        _Family(
            id="flow",
            name="Order flow trader",
            role="Order flow with the size, placement and burstiness real order flow has",
            strategy=(
                "Power law order sizes with a tail exponent near 2.3 to 2.5, limit "
                "prices placed a power law distance from the touch with an exponent near "
                "1.5, and self exciting arrivals. The parameters are the literature's "
                "(Gopikrishnan et al. 2000, Zovko and Farmer 2002, Filimonov and "
                "Sornette 2012) rather than fitted to make this market's output look "
                "like anything."
            ),
            why=(
                "Zero by default, and the reason is a warning rather than a preference. "
                "This agent assumes what the rest of the project measures: the market "
                "already reproduces fat tails, volatility clustering and long memory "
                "order flow emergently, from agents with none of those distributions "
                "built in, and that is the stronger result. A statistic gathered with an "
                "assumed size distribution present is a claim about a market that was "
                "told what its order sizes are. What it is genuinely for: queue dynamics "
                "under heavy cancellation, and whether the engine holds up under bursty "
                "load."
            ),
            source="python/arena/agents/flow.py",
            measured=(
                "The original warning here said switching it on would inflate the "
                "measured tails. Over three paired seeds that was wrong and in the "
                "opposite direction: Hill tail index 1.86 to 2.06, excess kurtosis 152.9 "
                "to 131.4, and the bid-ask bounce moved from +0.13 to -0.05, which is "
                "the sign a real tape has (arena/agents/flow.py)."
            ),
            match=lambda agent: isinstance(agent, FlowTrader),
        ),
        _Family(
            id="arb-1",
            name="Arbitrageur",
            role="Enforces the mechanical identities between related contracts",
            strategy=(
                "Derives its relations from the listed instruments at construction, off "
                "each contract's own underlying algebra and payoff, so listing a new "
                "spread or chain makes it arbitrageable with no code change: parity "
                "C - P = F - K, a spread against its legs, an index against its weights, "
                "a share against its strip, and the vertical and butterfly bounds that "
                "are the no-arbitrage condition on a call chain written as portfolios. "
                "It sends both legs as simultaneous IOC market orders and acts only when "
                "the mispricing clears the round-trip cost times a safety multiple."
            ),
            why=(
                "Off by default, on the evidence, which is why this row usually reads as "
                "not seated. Without it, related instruments are priced by unrelated "
                "crowds, and no amount of better market making on each book separately "
                "creates consistency between books. But enforcing a relation by "
                "repeatedly lifting the book buys that consistency with liquidity, and a "
                "market that cannot absorb an order is broken more fundamentally than "
                "one carrying a stale spread. Switch it on in the market configuration "
                "to seat it."
            ),
            source="python/arena/agents/arbitrageur.py",
            measured=(
                "Before it existed, put-call parity on the live market was violated by "
                "about 360 ticks and the spread contract genuinely mean-reverted at a "
                "variance ratio of 0.38. Against that, across four paired seeds it "
                "improved spread consistency on three and made it worse on the fourth, "
                "and on one seed took visible ask depth from 877 lots to 69 "
                "(arena/agents/arbitrageur.py, dashboard/build_market.py)."
            ),
            match=lambda agent: isinstance(agent, Arbitrageur),
        ),
    ]


def _operator_families() -> list[_Family]:
    from arena.market.match_operator import MatchOperator
    from arena.market.operator import SessionOperator

    return [
        _Family(
            id="exchange",
            name="Session operator",
            role="Not a trader. Runs the opening call and brings paused symbols back",
            strategy=(
                "Symbols start in PRE_OPEN, orders accumulate, nothing matches, and at "
                "the opening time every book uncrosses at a single price. A symbol the "
                "circuit breaker paused comes back through an auction once its pause has "
                "run, never straight into continuous trading."
            ),
            why=(
                "Everything a venue does to itself rather than in response to an order "
                "happens on a schedule, and in a discrete event simulation a schedule "
                "needs a participant to keep it. Putting a clock inside the ledger would "
                "make the ledger's behaviour depend on when it happened to be asked a "
                "question. It holds no position, posts no quotes and its account never "
                "moves, so it appears in the conservation check as nothing but a zero."
            ),
            source="python/arena/market/operator.py",
            measured=(
                "Its reopen job is sized by breaker windows scaled to the session rather "
                "than copied from the real rule. With the literal figures from limit up "
                "limit down, twelve of twenty-six symbols were halted at once "
                "(dashboard/build_market.py)."
            ),
            match=lambda agent: isinstance(agent, SessionOperator),
        ),
        _Family(
            id="matches",
            name="Match operator",
            role=(
                "Not a trader. Opens live matches, lists their contracts, and settles "
                "them as they play"
            ),
            strategy=(
                "A match opens, runs for a few minutes and ends with an outcome that is "
                "a fact. Eliminations settle in pieces while everyone still in keeps "
                "trading: a competitor who is out cannot win, so that contract is worth "
                "exactly zero and resolves the moment it happens. Only contracts whose "
                "value is already certain settle early, because settling anything less "
                "would be paying out a forecast."
            ),
            why=(
                "The statistical contracts settle once, at the end of a four week "
                "observation window. That is a real market and a slow one: nothing "
                "resolves inside a session, so a trader has no terminal event to be "
                "scored against and the listing drains as contracts expire with nothing "
                "to replace them. Matches are generated rather than collected, so there "
                "is always another one. It is not in its own participant list: it lists "
                "contracts, it does not trade them."
            ),
            source="python/arena/market/match_operator.py",
            measured=(
                "It tells every participant to drop a settled match's symbols rather "
                "than keep feeding them, because six extra active agents cost about a "
                "third of throughput on this market and a season of dead symbols on "
                "every feed would cost far more and buy nothing "
                "(arena/market/match_operator.py)."
            ),
            match=lambda agent: isinstance(agent, MatchOperator),
        ),
    ]


# -- the two systems that reach this exchange from outside -----------------
#
# Described here rather than discovered, because there is nothing to discover:
# neither imports the exchange, both speak HTTP against the published API, and
# by design the venue cannot tell one of them from any other signed client.

_CONNECTED: tuple[dict[str, Any], ...] = (
    {
        "id": "prediction-engine",
        "name": "Prediction market research engine",
        "count": 1,
        "role": (
            "An outside client, Kalshi shaped, that reads this venue's markets and "
            "audits their prices for joint no-arbitrage"
        ),
        "strategy": (
            "Presents this exchange as the three files a Kalshi shaped engine already "
            "keeps a venue behind: one data model, one request signer, one client, "
            "method for method. On top of that sits the engine's own joint no-arbitrage "
            "LP, which asks whether any probability distribution over the states prices "
            "every market inside its own bid-ask spread simultaneously. If none does "
            "then by LP duality a portfolio exists with non-negative payoff in every "
            "state and negative cost, which is riskless and needs no forecast."
        ),
        "why": (
            "It talks HTTP as an outsider on purpose. A test that shared the venue's own "
            "model of what a coherent price is could not falsify that model, so nothing "
            "in the package imports the exchange. And the audit is not redundant with "
            "the maker's guarantee: every contract on a match is priced as an "
            "expectation under one ensemble, so the model prices cannot disagree with "
            "each other, measured at zero violations across 22,650 relation checks in "
            "exact rational arithmetic. Quotes are not model prices. A maker widens for "
            "inventory, skews for adverse selection, and reprices off a book it last saw "
            "160ms ago, and whether the prices actually standing on the screen admit any "
            "consistent distribution is a different question that nothing in this "
            "repository asked before."
        ),
        "source": "connectors/prediction_engine/",
        "measured": (
            "It has already found one. Asked of the pricing function directly, 0 of 16 "
            "elimination ladders are non-monotone, so the model's guarantee is exactly "
            "true. Asked of the live book it fails: on one match a ladder stood at "
            "P(X > 1) offered at 0.49 and P(X > 2) bid at 0.60, and buying the first "
            "against the second collects 0.11 for a payoff that cannot be negative. "
            "The arbitrageur halves those, 64.6 per cent of tradeable ladders down to "
            "29.2 per cent on paired seeds, and does not clear them "
            "(connectors/prediction_engine/README.md)."
        ),
    },
    {
        "id": "portfolio-manager",
        "name": "Equities portfolio manager",
        "count": 1,
        "role": (
            "An outside allocator, Alpaca shaped, that trades here through the broker "
            "interface it already calls"
        ),
        "strategy": (
            "`ArenaBroker` implements the same nine methods and one property that an "
            "Alpaca-shaped `BrokerClient` declares, so an allocator, a sizing model and "
            "a bot loop written against that base class trade here without a branch. "
            "Three things genuinely differ and are handled in that one file: this venue "
            "accepts an order asynchronously so `orderId` is the client order id "
            "throughout, an order is keyed by (symbol, id) because there is one matching "
            "engine per symbol, and a short is an ordinary collateralised position whose "
            "exact worst case is charged up front rather than a margin agreement."
        ),
        "why": (
            "A small file with an outsized consequence: a system built to trade equities "
            "through a real broker can trade a synthetic esports exchange without a "
            "branch, and every order it sends is settled by a ledger whose conservation "
            "is integer zero rather than by a paper trading endpoint that approximates "
            "one. Prices cross the wire as strings and the float conversion happens in "
            "that file and only there, which is the honest place for it: a float is what "
            "the allocator's arithmetic needs, and it is not what the ledger is made of."
        ),
        "source": "connectors/portfolio_manager/",
        "measured": None,
    },
)


# -- reading the market ----------------------------------------------------


def _accounts_for(venue: Any, agents: list[Any]) -> list[Any]:
    """The accounts belonging to a family, skipping anyone who has none.

    An agent can be seated without an account having been opened for it, and
    that is ordinary rather than exceptional: the venue opens one lazily and
    `build` funds only the makers and the person at the browser explicitly.
    """
    try:
        accounts = venue.accounts
    except Exception:  # pragma: no cover - a venue that cannot list its accounts
        return []
    found = []
    for agent in agents:
        account = accounts.get(getattr(agent, "agent_id", None))
        if account is not None:
            found.append(account)
    return found


def _equity(venue: Any, accounts: list[Any]) -> str | None:
    """Summed equity across a family, in price units, or None.

    None rather than "0.00" when nobody in the family holds an account,
    because the two are different facts and the screen should be able to tell
    a flat book from an absent one.
    """
    if not accounts:
        return None
    try:
        marks = venue.marks()
        total = sum(int(account.equity(marks)) for account in accounts)
    except Exception:  # pragma: no cover - a mark this family cannot be valued at
        return None
    return str(from_money(total))


def _open_symbols(accounts: list[Any]) -> int | None:
    """How many distinct symbols the family has a position in.

    Counted on non-zero quantity. `Account.position` creates a flat `Position`
    on first enquiry and leaves it in the dict, so the raw length of
    `positions` counts every symbol ever asked about rather than every symbol
    held, and a maker that had quoted the whole listing and gone home flat
    would read as holding all of it.
    """
    if not accounts:
        return None
    symbols = set()
    for account in accounts:
        for symbol, position in getattr(account, "positions", {}).items():
            if getattr(position, "quantity", 0):
                symbols.add(symbol)
    return len(symbols)


def _player(family: _Family, seated: list[Any], venue: Any) -> dict[str, Any]:
    """One row. Its description holds whether or not anybody is doing the job."""
    members = [agent for agent in seated if _matches(family, agent)]
    accounts = _accounts_for(venue, members) if venue is not None else []
    return {
        "id": family.id,
        "name": family.name,
        "count": len(members),
        "role": family.role,
        "strategy": family.strategy,
        "why": family.why,
        "source": family.source,
        "live": bool(members),
        "equity": _equity(venue, accounts) if venue is not None else None,
        "positions": _open_symbols(accounts),
        "measured": family.measured,
    }


def _matches(family: _Family, agent: Any) -> bool:
    """The family's own test, with a failure treated as a non-match.

    An isinstance test cannot normally raise, but this one runs over whatever
    happens to be in `market.agents`, and the whole point of this module is
    that it keeps answering when the population is not what it expected.
    """
    try:
        return bool(family.match(agent))
    except Exception:  # pragma: no cover - an agent this family cannot classify
        return False


def _treasury(venue: Any) -> dict[str, Any]:
    """Where fees land. An account rather than a counter, on purpose."""
    account = None
    if venue is not None:
        try:
            account = venue.accounts.get(FEE_ACCOUNT_ID)
        except Exception:  # pragma: no cover - a venue that cannot list its accounts
            account = None
    accounts = [account] if account is not None else []
    return {
        "id": str(FEE_ACCOUNT_ID),
        "name": "Venue treasury",
        "count": 1,
        "role": "Not a trader. The account every fee and every rebate passes through",
        "strategy": (
            "It holds cash and never a position. The venue charges the taker and pays "
            "the maker under the fee schedule, and the difference lands here, so the "
            "balance is exactly what the schedule collected."
        ),
        "why": (
            "A real account rather than a running total, so the conservation check sees "
            "it: total equity in a closed market has to equal the capital that entered "
            "it, and a fee that left the participants without arriving anywhere would "
            "break that identity silently. It opens empty rather than at the venue's "
            "default balance, because it is not a participant and nobody funded it."
        ),
        "source": "python/arena/market/venue.py",
        # True by construction. `FEE_ACCOUNT_ID` is given a zero balance when the
        # venue is built, so the treasury is always a real account here even
        # before the first fee materialises the `Account` object, which is why
        # `equity` can be None on a market that has not traded yet while `live`
        # stays True.
        "live": True,
        "equity": _equity(venue, accounts) if venue is not None else None,
        "positions": _open_symbols(accounts),
        "measured": (
            "Funded at the default opening balance instead, it read "
            "39,999,070,000,000 and looked comfortably solvent while the schedule had "
            "actually paid out 930,000,000 minor units more than it took over two "
            "hundred fills. Starting at zero makes a venue that pays people to trade "
            "with each other show up as a negative number (arena/market/venue.py)."
        ),
    }


# -- the payload -----------------------------------------------------------


def directory(market: Any) -> dict[str, Any]:
    """Everyone on this exchange, grouped by where they come from.

    Three groups, and the distinction between them is about who built the
    participant rather than about what it trades. **resident** agents are
    constructed and woken by this market's own kernel. **operator** entries are
    the venue's own machinery, which is seated alongside the agents because in
    a discrete event simulation only a participant with a wakeup can decide
    that time has passed; none of them trades. **connected** systems reach the
    exchange from outside over the published HTTP API and are not in
    `market.agents` at all, by design.

    `counts` carries three totals a header can draw without walking the groups:
    `resident` is how many resident agents are actually seated, `connected` is
    how many outside systems are described, and `seats` is how many people hold
    an account at this market, which starts at one because the shared browser
    account exists before anybody signs in.

    Takes a `LiveMarket` and never raises on one. A market missing an agent,
    a venue that cannot value a position, an agent seated without an account:
    each of those produces a row saying so rather than an exception, because
    the arbitrageur being switched off is a configuration and not a fault.
    """
    seated = list(getattr(market, "agents", None) or [])
    venue = getattr(market, "venue", None)

    residents = [_player(f, seated, venue) for f in _resident_families()]
    operators = [_player(f, seated, venue) for f in _operator_families()]
    operators.append(_treasury(venue))

    connected = []
    for entry in _CONNECTED:
        row = dict(entry)
        # Flat False, and the docstring at the top of this module says why: the
        # key store that knows whether an API seat is authenticated belongs to
        # the server, not to the market, so nothing reachable from here can
        # answer the question. A guess would be worse than an honest no.
        row["live"] = False
        row["equity"] = None
        row["positions"] = None
        connected.append(row)

    try:
        seats = len(market.traders)
    except Exception:  # pragma: no cover - a market with no seat register
        seats = 0

    return {
        "groups": [
            {
                "kind": "resident",
                "title": "Agents that live in the market",
                "blurb": (
                    "Built and woken by this market's own kernel, each one seated at a "
                    "measured distance from the exchange. The population is the "
                    "experiment: what these agents are and how many of them there are "
                    "is the thing every result here is a result about."
                ),
                "players": residents,
            },
            {
                "kind": "operator",
                "title": "The exchange running itself",
                "blurb": (
                    "The venue's own machinery. It is seated alongside the agents "
                    "because only a participant with a wakeup can decide that time has "
                    "passed, but none of it trades: it lists, opens, pauses, reopens and "
                    "settles, and it holds no position while doing so."
                ),
                "players": operators,
            },
            {
                "kind": "connected",
                "title": "Systems that trade here from outside",
                "blurb": (
                    "Written against other venues and pointed at this one. Neither "
                    "imports the exchange; both speak HTTP against the published API "
                    "exactly as any outside client does, which is what makes what they "
                    "find worth reading."
                ),
                "players": connected,
            },
        ],
        "counts": {
            "resident": sum(row["count"] for row in residents),
            "connected": len(connected),
            "seats": seats,
        },
    }
