"""Where a market maker's money actually went.

"The makers lost 42.5M" is not a finding, it is a symptom, and every fix
proposed for it is a guess until the loss is split into the terms a desk can
act on separately. Those terms are not a matter of taste: total trading P&L
telescopes exactly, for any fill sequence and any choice of horizon ``h``,

    Σ qᵢ(M_T − Pᵢ)  =  Σ qᵢ(Mᵢ − Pᵢ)          spread captured
                     + Σ qᵢ(M_{i+h} − Mᵢ)      adverse selection over h
                     + Σ qᵢ(M_T − M_{i+h})     inventory and residual drift

with ``qᵢ`` the signed lots, ``Pᵢ`` the fill price, ``Mᵢ`` the mid prevailing
before the fill, and ``M_T`` the mark at the end. Nothing is assumed and
nothing is estimated (the middle two terms are added and subtracted), so this
belongs in a repository whose collateral is arithmetic rather than a model. The
first two terms are the Huang-Stoll effective and realized spread; their
difference is what the literature means by adverse selection, and what
Hasbrouck reads as the permanent, information-bearing part of a trade's impact.

**What it is for.** The three terms fail in different directions and want
opposite fixes. Negative spread capture means the quote was behind the market
and is being run over, and the answer is a faster anchor. Negative adverse
selection means the quote was picked off by somebody who knew more, and the
answer is size, skew, or not quoting that side. A negative residual is
inventory, and the answer is a limit or a hedge. Widening the spread is the
reflex for all three and is only ever right for the second, and on a market
this thin, not even always then, because a maker that is most of the book
widens the mid it is measured against.

**Horizons are read as a curve, not a number.** A jump inside the first
hundred milliseconds and flat after is a stale quote being arbitraged, which is
a latency problem. A drift that keeps going for seconds is information, which
is not. The two look identical in a single-horizon summary and want completely
different work, so the report gives the ladder and lets the shape say which.

**On being outside.** :class:`~arena.research.recorder.MarketRecorder` reads
the books and sends nothing, and that is what makes it unable to perturb what
it measures. This is *almost* that. It samples its own mid series exactly the
same way, and takes only one thing from inside the venue: who was on each side
of a print. That cannot be recovered afterwards, because an order id resolves
to an agent only while the order is still in the book. So the venue offers a
notification and this listens to it, and the honest cost of the arrangement is
that the mid attributed to a fill is the one sampled most recently before it:
stale by up to one sampling interval, which the caller chooses.

Floats, unlike everywhere else in this project. A mid sits half a tick between
two integers, every quantity here is a difference of two prices rather than a
balance anybody is owed, and nothing computed in this module is ever paid to
anyone. The ledger stays integer; this is a report about it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from arena.exchange.types import Side
from arena.market.venue import Venue
from arena.sim.time import Timestamp, seconds

__all__ = ["TradeAttribution", "Decomposition", "DEFAULT_HORIZONS"]


# Half a decade of a maker's requote interval up to well past the point the
# drift stops. Chosen to bracket the two failure modes rather than to name a
# convention: the short end is latency, the long end is information, and the
# whole argument for a ladder is that a single horizon cannot tell them apart.
DEFAULT_HORIZONS: tuple[int, ...] = (
    seconds(0.1),
    seconds(0.5),
    seconds(1),
    seconds(5),
    seconds(30),
)


@dataclass
class Decomposition:
    """One agent's P&L, split into terms that want different fixes.

    Every field is in the ledger's minor units, signed from the agent's point
    of view, and computed over the fills that had reached the horizon by the
    time the report was taken. ``fills`` says how big that subset is, because a
    decomposition over three trades is not evidence of anything.
    """

    agent_id: str
    horizon: int
    fills: int = 0
    lots: int = 0
    spread_captured: float = 0.0
    adverse_selection: float = 0.0
    residual: float = 0.0
    passive_lots: int = 0
    aggressive_lots: int = 0
    by_counterparty: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        """The three terms, which is the whole trading P&L on these fills."""
        return self.spread_captured + self.adverse_selection + self.residual

    @property
    def realized_spread(self) -> float:
        """What the quote actually earned once the drift is paid for.

        Huang-Stoll: effective spread minus adverse selection. The number that
        says whether the quote is priced right, as opposed to whether it is
        wide.
        """
        return self.spread_captured + self.adverse_selection

    def per_lot(self, value: float, instrument_tick_in_minor: float) -> float:
        """A term as ticks per lot, which is how a desk reads these."""
        if not self.lots or not instrument_tick_in_minor:
            return 0.0
        return value / self.lots / instrument_tick_in_minor

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "horizon": self.horizon,
            "fills": self.fills,
            "lots": self.lots,
            "spread_captured": self.spread_captured,
            "adverse_selection": self.adverse_selection,
            "realized_spread": self.realized_spread,
            "residual": self.residual,
            "total": self.total,
            "passive_lots": self.passive_lots,
            "aggressive_lots": self.aggressive_lots,
            "by_counterparty": dict(self.by_counterparty),
        }


@dataclass
class _Fill:
    """One side of one print, waiting for its horizons to elapse."""

    at: int
    symbol: str
    agent_id: str
    counterparty: str
    signed: int
    price_minor: float
    mid_before: float
    passive: bool
    # Horizon -> the mid once that much time had passed. Filled in as the
    # sampler reaches each one, which is why a horizon that never elapses
    # before the report simply does not appear rather than being guessed at.
    mid_at: dict[int, float] = field(default_factory=dict)


class TradeAttribution:
    """Splits every participant's P&L into spread, adverse selection and drift.

    Attach it, drive :meth:`sample` on the same clock that drives any other
    recorder, and ask for :meth:`report` whenever. It never sends a message.
    """

    def __init__(
        self,
        venue: Venue,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        agents: frozenset[str] | None = None,
    ) -> None:
        if not horizons:
            raise ValueError("attribution needs at least one horizon")
        self.venue = venue
        self.horizons = tuple(sorted(int(h) for h in horizons))
        # A filter, not a requirement. Restricting to the makers keeps the
        # record small on a long run; leaving it open is what makes the
        # counterparty breakdown add up.
        self.agents = agents
        self._mid: dict[str, float] = {}
        self._minor_per_tick: dict[str, int] = {}
        self._now = 0
        self._fills: list[_Fill] = []
        self._open: list[_Fill] = []
        self._attached = False

    # -- wiring ------------------------------------------------------------

    def attach(self) -> None:
        """Start listening. Refuses to displace an observer already there.

        Silently replacing one would leave the first caller with a recorder
        that has stopped recording and no way to find out, which is the same
        shape as a guard whose input was never wired.
        """
        if self.venue.trade_observer is not None and not self._attached:
            raise RuntimeError("this venue already has a trade observer")
        self.venue.trade_observer = self._on_print
        self._attached = True

    def detach(self) -> None:
        if self._attached:
            self.venue.trade_observer = None
            self._attached = False

    def __enter__(self) -> TradeAttribution:
        self.attach()
        return self

    def __exit__(self, *_: object) -> None:
        self.detach()

    # -- collection --------------------------------------------------------

    def _tick_in_minor(self, symbol: str) -> int:
        cached = self._minor_per_tick.get(symbol)
        if cached is None:
            cached = int(self.venue.registry.require(symbol).tick_in_minor)
            self._minor_per_tick[symbol] = cached
        return cached

    def _current_mid(self, symbol: str) -> float:
        """The mid in ticks, carrying the last one over an empty book.

        Carried rather than dropped for the same reason the recorder carries
        it: a hole would be read as a price move, and here it would be read as
        adverse selection against whoever happened to trade next.
        """
        book = self.venue.engine(symbol).book.snapshot()
        if book.best_bid is not None and book.best_ask is not None:
            return (int(book.best_bid) + int(book.best_ask)) / 2.0
        carried = self._mid.get(symbol)
        if carried is not None:
            return carried
        low, high = self.venue.registry.require(symbol).tick_bounds
        return (int(low) + int(high)) / 2.0

    def _on_print(self, entry: dict[str, Any]) -> None:
        """Record both sides of a print. Called by the venue; sends nothing."""
        symbol = entry["symbol"]
        quantity = int(entry["quantity"])
        if quantity <= 0:
            return
        minor = self._tick_in_minor(symbol)
        price_minor = float(int(entry["price"]) * minor)
        # The mid sampled most recently before this print, in the same units.
        # Falls back to the print itself on the very first trade in a symbol,
        # which is the honest answer: with no prior quote there is no spread to
        # have captured, and charging the maker one would be inventing it.
        mid_ticks = self._mid.get(symbol)
        mid_before = float(mid_ticks * minor) if mid_ticks is not None else price_minor

        aggressor_is_buy = entry["aggressor"] == Side.BUY.value
        buyer, seller = str(entry["buyer"]), str(entry["seller"])
        for agent_id, other, signed, passive in (
            (buyer, seller, quantity, not aggressor_is_buy),
            (seller, buyer, -quantity, aggressor_is_buy),
        ):
            if self.agents is not None and agent_id not in self.agents:
                continue
            self._open.append(
                _Fill(
                    at=self._now,
                    symbol=symbol,
                    agent_id=agent_id,
                    counterparty=other,
                    signed=signed,
                    price_minor=price_minor,
                    mid_before=mid_before,
                    passive=passive,
                )
            )

    def sample(self, now: Timestamp | int) -> None:
        """Refresh the quote series and mature whatever horizons have elapsed.

        Order matters and is the whole correctness argument. Mids are read
        first, so a fill that matures on this tick is marked against the price
        as of *now*; ``_now`` moves last, so a print arriving before the next
        sample is attributed to the mid this call just recorded, which is the
        quote that was standing when it traded.
        """
        stamp = int(now)
        for symbol in self.venue.registry.symbols:
            self._mid[symbol] = self._current_mid(symbol)

        longest = self.horizons[-1]
        still_open: list[_Fill] = []
        for fill in self._open:
            minor = self._tick_in_minor(fill.symbol)
            mid_now = self._mid[fill.symbol] * minor
            for horizon in self.horizons:
                if horizon not in fill.mid_at and stamp >= fill.at + horizon:
                    fill.mid_at[horizon] = mid_now
            if stamp >= fill.at + longest:
                self._fills.append(fill)
            else:
                still_open.append(fill)
        self._open = still_open
        self._now = stamp

    # -- reporting ---------------------------------------------------------

    def report(self, horizon: int | None = None) -> dict[str, Decomposition]:
        """The decomposition per agent at one horizon.

        Only fills that had reached the horizon are included, because the other
        kind would need a mid that has not happened yet. That makes the three
        terms sum to the trading P&L *on this subset* exactly, rather than
        approximately over everything.
        """
        want = self.horizons[-1] if horizon is None else int(horizon)
        if want not in self.horizons:
            raise ValueError(f"horizon {want} was never collected")

        # The venue's own mark, not this module's mid. They differ, and the
        # venue's is the one the ledger settles against, so using it is what
        # lets the three terms be checked against `equity - starting_cash`
        # instead of merely resembling it. An attribution that reconciles to
        # something other than the books is not an attribution.
        marks = {
            symbol: float(int(self.venue.mark(symbol)))
            for symbol in self.venue.registry.symbols
        }
        out: dict[str, Decomposition] = {}
        for fill in [*self._fills, *self._open]:
            mid_at = fill.mid_at.get(want)
            if mid_at is None:
                continue
            row = out.get(fill.agent_id)
            if row is None:
                row = out[fill.agent_id] = Decomposition(fill.agent_id, want)
            lots = abs(fill.signed)
            row.fills += 1
            row.lots += lots
            if fill.passive:
                row.passive_lots += lots
            else:
                row.aggressive_lots += lots
            row.spread_captured += fill.signed * (fill.mid_before - fill.price_minor)
            row.adverse_selection += fill.signed * (mid_at - fill.mid_before)
            contribution = fill.signed * (marks[fill.symbol] - mid_at)
            row.residual += contribution
            row.by_counterparty[fill.counterparty] = (
                row.by_counterparty.get(fill.counterparty, 0.0)
                + fill.signed * (marks[fill.symbol] - fill.price_minor)
            )
        return out

    def curve(self, agent_id: str) -> dict[int, float]:
        """One agent's adverse selection at every horizon, in minor units.

        The shape is the diagnosis. Flat after the first step is a stale quote;
        still moving after seconds is information.
        """
        return {
            horizon: self.report(horizon).get(
                agent_id, Decomposition(agent_id, horizon)
            ).adverse_selection
            for horizon in self.horizons
        }

    def informed_share(self, agent_id: str, informed: frozenset[str]) -> float:
        """The fraction of an agent's passive lots that came from a given set.

        Glosten-Milgrom's ``μ``. It is the parameter that decides whether any
        spread can be profitable at all, and it is directly observable here
        while being unobservable on a real venue, which is most of the reason
        to run a market you built.
        """
        passive = matched = 0
        for fill in [*self._fills, *self._open]:
            if fill.agent_id != agent_id or not fill.passive:
                continue
            lots = abs(fill.signed)
            passive += lots
            if fill.counterparty in informed:
                matched += lots
        return matched / passive if passive else 0.0

    def flow_imbalance(self, agent_id: str) -> dict[str, float]:
        """Net passive lots over gross, per symbol. Zero is a healthy quote.

        The cheapest defect detector available: inventory that swings around
        zero is risk, but a maker that is persistently taken on the same side
        of the same contract is not unlucky, it is priced wrong. This is what
        catches a surface whose volatility is too low, because a delta limit is
        structurally blind to being short calls and puts at once.
        """
        net: dict[str, int] = defaultdict(int)
        gross: dict[str, int] = defaultdict(int)
        for fill in [*self._fills, *self._open]:
            if fill.agent_id != agent_id or not fill.passive:
                continue
            net[fill.symbol] += fill.signed
            gross[fill.symbol] += abs(fill.signed)
        return {
            symbol: net[symbol] / gross[symbol] for symbol in gross if gross[symbol]
        }
