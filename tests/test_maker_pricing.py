"""Two ways a maker's quote can be at the wrong price, and how to tell.

Neither of these is about risk. A maker on this venue loses money to informed
flow whatever it quotes, and `arena.research.attribution` exists to keep that
term separate. What these test is narrower and answerable: whether the price in
the middle of the quote is a price, and whether the flow arriving at it is
two-sided.

One market run serves the whole file, because a run is expensive and the three
questions are about the same session. Everything is read out of
:class:`~arena.research.attribution.TradeAttribution` or off the book directly,
and nothing here sends a message.
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from arena.exchange.types import Side
from arena.market.venue import Venue
from arena.research.attribution import TradeAttribution
from arena.sim.time import seconds

from dashboard.build_market import build

SEED = 7
SESSION = 300.0
# Coarse on purpose, and it costs nothing here. The effective half spread below
# is taken from the touch standing before each print rather than off this grid,
# and `flow_imbalance` reads fills and never reads a mid at all, so the ladder
# exists only so the recorder's own bookkeeping runs at all.
GRID = seconds(2.0)
HORIZONS = (seconds(1), seconds(5))
MINUTE = seconds(60)


class Session:
    """One driven market, with everything the three tests below need."""

    def __init__(self, market, attribution, edges, volume, skipped):
        self.market = market
        self.attribution = attribution
        # symbol -> [signed edge in tick-lots, lots, fills, negative fills],
        # counting only prints that had a real two-sided uncrossed touch to be
        # measured against.
        self.edges = edges
        # symbol -> minute -> lots printed
        self.volume = volume
        # Why prints were left out, so a shrinking sample cannot pass quietly.
        self.skipped = skipped
        self.makers = frozenset(
            a.agent_id for a in market.agents if "MarketMaker" in type(a).__name__
        )
        self.instrument_class = {
            symbol: market.venue.registry.require(symbol).instrument_class
            for symbol in market.venue.registry.symbols
        }

    def events(self) -> list[str]:
        return sorted(
            symbol
            for symbol, kind in self.instrument_class.items()
            if kind == "event"
        )

    def by_class(self) -> dict[str, list]:
        totals: dict[str, list] = defaultdict(lambda: [0.0, 0, 0, 0])
        for symbol, row in self.edges.items():
            into = totals[self.instrument_class[symbol]]
            for index in range(4):
                into[index] += row[index]
        return totals

    def imbalance(self) -> dict[str, float]:
        """Net over gross passive lots per symbol, across all three makers.

        Summed over the makers rather than reported per maker, because the
        three differ only in width and a contract priced wrong is priced wrong
        for all of them. Per maker it is the same finding split three ways and
        noisier for it.
        """
        net: dict[str, int] = defaultdict(int)
        gross: dict[str, int] = defaultdict(int)
        for fill in [*self.attribution._fills, *self.attribution._open]:
            if fill.agent_id not in self.makers or not fill.passive:
                continue
            net[fill.symbol] += fill.signed
            gross[fill.symbol] += abs(fill.signed)
        return {s: net[s] / gross[s] for s in gross if gross[s]}


@pytest.fixture(scope="module")
def session() -> Session:
    """Drive one market and watch it two ways.

    The effective half spread is Huang and Stoll's: the mid prevailing at the
    time of the trade, less the price, signed by the maker's side. The phrase
    that decides everything here is "at the time of the trade".
    :class:`TradeAttribution` says plainly that it can only offer the mid it
    sampled most recently before the print, stale by up to one sampling
    interval, because an order id resolves to an agent only while the order is
    still in the book and the recorder has to listen from outside. This wraps
    ``Venue.submit`` instead and reads the touch immediately before the command
    that caused the print, which carries no staleness at all. The wrapper only
    reads a book snapshot, so the market it measures is the market that would
    have run without it.

    The difference between the two readings is far larger than it sounds.
    Measured on seed 7 over 300s, on one run: 139 passive fills carried a
    negative effective half spread against the exact mid and 2,284 did against
    a mid sampled every 250ms, a factor of sixteen. What the coarse reading is
    mostly catching is the maker requoting between two samples on the books
    that move fastest, which on this fixture are the commodities, whose opening
    reference is the middle of their declared range against a front week volume
    contract that settles at 1,685.
    """
    market = build(seed=SEED)
    attribution = TradeAttribution(market.venue, horizons=HORIZONS)
    attribution.attach()

    makers = frozenset(
        a.agent_id for a in market.agents if "MarketMaker" in type(a).__name__
    )
    touch: dict[str, tuple[int, int]] = {}
    edges: dict[str, list] = defaultdict(lambda: [0.0, 0, 0, 0])
    volume: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    skipped: dict[str, int] = defaultdict(int)
    original = Venue.submit
    recorder = market.venue.trade_observer

    def submit(self, agent_id, symbol, command):
        book = self._engines[symbol].book.snapshot()
        bid, ask = book.best_bid, book.best_ask
        # Recorded on every command, including as *absent*. Keeping the last
        # two-sided touch instead was the defect this fixture used to carry: a
        # book quoted on one side only still answered with the mid it had when
        # it was last quoted on both, however long ago that was, and every
        # print in between was scored against a price no longer on the screen.
        touch[symbol] = (
            (int(bid), int(ask)) if bid is not None and ask is not None else None
        )
        return original(self, agent_id, symbol, command)

    def observe(entry) -> None:
        symbol = entry["symbol"]
        quantity, price = int(entry["quantity"]), int(entry["price"])
        volume[symbol][int(market.kernel.now) // MINUTE] += quantity
        seen = touch.get(symbol)
        if seen is None:
            # Nothing to measure against. A one-sided book has no mid, and
            # scoring a fill against half of one is not a weaker reading of the
            # same quantity, it is a different quantity.
            skipped["no two-sided book"] += 1
        elif seen[0] > seen[1]:
            # A locked or crossed book puts its own mid above its own offer, so
            # an offer that trades there is below the mid by construction and
            # for no reason having to do with the quote. Legitimate here: a
            # book in a call phase accumulates without matching, which
            # `Venue.mark` says in as many words.
            skipped["locked or crossed"] += 1
        else:
            mid = (seen[0] + seen[1]) / 2.0
            aggressor_buy = entry["aggressor"] == Side.BUY.value
            buyer, seller = str(entry["buyer"]), str(entry["seller"])
            for agent, other, signed, passive in (
                (buyer, seller, quantity, not aggressor_buy),
                (seller, buyer, -quantity, aggressor_buy),
            ):
                # Maker against maker is left out. The same trade appears once
                # on each side with opposite sign, so it nets to nothing while
                # doubling the count, which flatters both the mean and the
                # share of negatives.
                if agent not in makers or not passive or other in makers:
                    continue
                edge = signed * (mid - price)
                row = edges[symbol]
                row[0] += edge
                row[1] += quantity
                row[2] += 1
                row[3] += 1 if edge < 0 else 0
        recorder(entry)

    Venue.submit = submit
    market.venue.trade_observer = observe
    try:
        market.kernel.start()
        now = 0
        while now < seconds(SESSION):
            now += GRID
            market.kernel.advance(until=now)
            attribution.sample(now)
    finally:
        Venue.submit = original
    attribution.detach()
    return Session(market, attribution, edges, volume, dict(skipped))


# --------------------------------------------------------------------------
# The quote is where the market is
# --------------------------------------------------------------------------


def test_a_resting_quote_is_not_systematically_run_over(session):
    """A passive fill worse than the prevailing mid is a quote left behind.

    It is not adverse selection, which is the second term of the decomposition
    and is paid for by the informed share rather than by the price. A resting
    offer that executes *below* the mid standing when it traded has been
    overtaken by the market, and no amount of informed flow can produce that on
    a quote that is where the market is: hitting a maker's bid at ``m - h``
    when the mid is ``m`` earns the maker ``h``, whatever the buyer knew.

    So the question is only asked where it has an answer. A print is measured
    when the book carried a two-sided uncrossed touch at the instant the
    command that caused it arrived, and is set aside otherwise, counted in
    ``session.skipped`` rather than dropped silently. Both exclusions are
    states the venue is entitled to be in: a one-sided book has no mid, and a
    book in a call phase is crossed on purpose, because orders accumulate there
    without matching.

    Asked that way the answer is not a small number, it is none. Measured over
    300s on seeds 7, 3 and 41: 28,869, 28,566 and 26,958 passive maker fills
    against a real touch, and **zero** of them executed worse than the mid on
    any of the three. Every negative in the session is on a locked book, 19,
    31 and 12 of them, where the mid sits above the offer by construction and
    an offer trading at the offer is below it for arithmetic reasons rather
    than for anything to do with the quote.

    This used to read as a bound of three per cent against a worst measured
    1.73, and both figures were artefacts of how the mid was sampled rather
    than facts about the quotes. The fixture kept the last two-sided touch it
    had seen and scored every later print against it, so a book that had been
    one-sided for two minutes was still being compared with the mid it had when
    it was last quoted on both sides. On the world as it now lists, that stale
    carry-forward read 5.28%. Read against the touch that was actually
    standing, the same session reads 0.00%.
    """
    totals = session.by_class()
    assert totals, "nothing traded, so the test measured nothing"

    fills = sum(row[2] for row in totals.values())
    negative = sum(row[3] for row in totals.values())
    assert fills > 5_000, f"only {fills} passive fills; the sample is too thin"
    # Exactly none, on three seeds. A bound with room in it would be room for
    # the defect to come back, and there is no measurement here asking for any.
    assert negative == 0, (
        f"{negative} of {fills} passive fills executed worse than the mid that "
        f"was standing when they traded; skipped {session.skipped}"
    )


# --------------------------------------------------------------------------
# The flow arriving at it is two-sided
# --------------------------------------------------------------------------


def test_passive_flow_on_the_event_ladder_is_not_one_sided(session):
    """Net over gross passive lots. Persistently off zero is a wrong price.

    Inventory that swings around zero is risk and is supposed to happen. A
    maker taken on the *same* side of the same contract every time is not
    unlucky, because luck does not have a sign, and the only thing that can
    produce it is a price the whole market disagrees with in one direction.

    Measured over 300s before the binaries were priced off the surface, on the
    ladder of eight the exchange then listed: seed 7 averaged 0.898 with three
    of the eight at exactly 1.00, and seed 3 averaged 0.960 with four at 1.00.
    Exactly 1.00 means every passive fill on that contract landed on the same
    side, without a single exception, which no distribution of luck produces.
    Priced off the same law as the option chain the same runs give 0.556 and
    0.611.

    The bound is on the mean and not on the worst contract, because the worst
    contract was still 1.00 on seed 3 after the fix: it traded 11 lots in the
    final minute and its ratio is a statement about eleven lots. A per contract
    bound would be dominated by whichever binary is closest to resolved and
    therefore least traded, which is the opposite of what this is asking.
    """
    imbalance = session.imbalance()
    ladder = {
        symbol: imbalance[symbol]
        for symbol in session.events()
        if symbol in imbalance
    }
    assert len(ladder) >= 6, f"only {len(ladder)} event contracts traded at all"

    mean = sum(abs(v) for v in ladder.values()) / len(ladder)
    # 0.70 sits between a measured 0.898 and 0.960 before and 0.556 and 0.611
    # after, which is the widest gap the two seeds leave.
    assert mean <= 0.70, (
        f"the event ladder averaged {mean:.3f} one-sided, on "
        + ", ".join(f"{s} {v:+.2f}" for s, v in sorted(ladder.items()))
    )


def test_every_event_contract_is_still_trading_at_the_end_of_the_session(session):
    """A book that stops trading has stopped having a price, and it shows.

    This is the other face of the same defect and the reason it was
    self-sustaining. The plain maker prices a binary from an exponential
    average of its own prints, so when the book stops trading the estimate
    stops moving, and the quote that stopped it is exactly the quote that keeps
    it stopped.

    Measured on seed 7 before the fix, in lots printed per minute: one rung of
    the ladder traded 145, 21, 0, 0, 0 and another 211, 0, 0, 0, 0. Every one
    of the eight event contracts then listed was dead inside two minutes while
    the futures were still printing thousands of lots a minute, and the maker
    spent the rest of the session quoting a rung around 0.40 against a
    settlement of 1.00. Priced off the surface the same run gives the first of
    those 806, 865, 761, 779, 730 and every contract trading in every
    minute.

    The mechanism that killed them is worth naming, because it is not the
    anchor alone. The venue collars a market order to a band around a trailing
    reference, and with no recent prints that reference is the mid of the book,
    which on these contracts is the maker's own quote. A binary spans 100
    ticks, so the band is five, and the maker's own two sided quote is eleven
    ticks wide or more. Its own quotes then sit outside a band centred on its
    own mid, and measured on seed 7 the engine refused 7,406 of the 7,563
    market orders that reached an event contract, 97.9%, against 6.1% on the
    futures. Nothing could trade, so nothing printed, so the print average
    never moved again.

    Asserted on the last minute rather than on the total, because a total is
    passed by a contract that traded once at the open and never again, which is
    exactly the state this is here to catch. Before the fix seven of eight
    contracts on seed 7 and eight of eight on seed 3 printed nothing at all in
    the final minute.
    """
    last = int(seconds(SESSION)) // MINUTE - 1
    for symbol in session.events():
        traded = session.volume[symbol].get(last, 0)
        assert traded > 0, (
            f"{symbol} printed nothing in the final minute of the session, so "
            "its price is whatever it was when it stopped"
        )


def test_the_surface_reports_the_exposure_a_delta_limit_cannot_see(session):
    """Net option lots, which is the risk a short straddle hides.

    A call and a put struck at the same price have deltas of opposite sign, so
    a book short both nets to roughly zero delta and `delta_limit` reads it as
    flat, while the width risk on it is doubled rather than cancelled. Measured
    on seed 7 over 600s the three makers finish short 30,768 option lots and
    short on 56 of the 60 contract and maker pairs they hold, and over 300s
    `mm-1` alone is short 14,618 on this accessor.

    The consequence is visible in the suite already, and it is a seed lottery
    rather than a fixed property: `test_every_strike_stays_quotable` requires
    all twenty strikes to be two-sided at 60% of the moments they trade, and on
    the unmodified maker that holds on seeds 7 and 41 and fails on seeds 3 and
    11. Every strike that fails does so with all three makers at exactly their
    short position limit, -1,200, -950 and -700, at which a maker stops adding
    to a side and the strike has no offer at all. On seed 7 the weakest link
    passed by one sampled moment, at 13 of 21 against the 12.6 the threshold
    needs.

    This asserts only that the accessor measures what it says, because the
    quantity itself is a defect and not an invariant: the surface's implied
    dispersion has a median of 9.45 price points against a median remaining
    distance to settlement of 60.92, so it sells every strike too cheap and
    ends short every strike. Closing that needs a settlement horizon, and the
    kernel counts nanoseconds while a contract expires on a calendar.
    """
    makers = [
        a for a in session.market.agents if "MarketMaker" in type(a).__name__
    ]
    chain = makers[0].chain
    underlyings = sorted({m.underlying_symbol for m in chain.values()})
    assert underlyings, "no chain was derived, so there is nothing to report on"

    for maker in makers:
        for underlying in underlyings:
            direct = sum(
                maker.position.get(symbol, 0)
                for symbol, member in chain.items()
                if member.underlying_symbol == underlying and not member.is_digital
            )
            assert maker._net_options(underlying) == float(direct)


def test_the_session_conserved_value_exactly(session):
    """Integer zero, and the reason it is asserted next to the two above.

    Both changes move where a quote sits, which moves what is collateralised
    against it. A pricing fix that leaked a minor unit would be a worse defect
    than either of the ones it fixed.
    """
    assert session.market.venue.conservation_check() == 0
