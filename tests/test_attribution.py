"""Splitting a maker's loss into terms that want different fixes.

The decomposition is an arithmetic identity, not a model. The two middle terms
are added and subtracted. So most of what is worth testing is that the identity
holds and that measuring does not change the thing measured.
"""

from __future__ import annotations

import pytest

from arena.research.attribution import DEFAULT_HORIZONS, TradeAttribution
from arena.sim.time import seconds

from dashboard.build_market import build

HORIZONS = (seconds(0.1), seconds(1), seconds(5))


def _run(seed: int = 7, until: float = 90.0, attach: bool = True, step: float = 0.25):
    """One market, optionally watched, driven on a fixed grid."""
    market = build(seed=seed)
    attribution = TradeAttribution(market.venue, horizons=HORIZONS)
    if attach:
        attribution.attach()
    market.kernel.start()
    now = 0
    while now < seconds(until):
        now += seconds(step)
        market.kernel.advance(until=now)
        if attach:
            attribution.sample(now)
    attribution.detach()
    return market, attribution


def _roles(market):
    makers = frozenset(
        a.agent_id for a in market.agents if "MarketMaker" in type(a).__name__
    )
    informed = frozenset(
        a.agent_id
        for a in market.agents
        if type(a).__name__ in ("FundamentalTrader", "BayesianFundamental")
    )
    return makers, informed


# --------------------------------------------------------------------------
# The identity
# --------------------------------------------------------------------------


def test_the_three_terms_sum_to_the_trading_pnl_on_those_fills():
    """Spread plus adverse selection plus residual is the whole of it.

    Not an approximation and not a fit: ``M`` and ``M_h`` are each added once
    and subtracted once, so the sum telescopes to ``Σ q(M_T − P)`` for any
    fills and any horizon. A failure here is an accounting bug in the report,
    which is the only kind of bug this identity can have.
    """
    market, attribution = _run()
    marks = {s: float(int(market.venue.mark(s))) for s in market.venue.registry.symbols}

    for horizon in HORIZONS:
        rows = attribution.report(horizon)
        assert rows, "no fills matured, so the test measured nothing"
        for agent_id, row in rows.items():
            direct = sum(
                fill.signed * (marks[fill.symbol] - fill.price_minor)
                for fill in [*attribution._fills, *attribution._open]
                if fill.agent_id == agent_id and horizon in fill.mid_at
            )
            assert row.total == pytest.approx(direct, rel=1e-9, abs=1.0), (
                agent_id,
                horizon,
            )


def test_realized_spread_is_effective_spread_less_adverse_selection():
    """Huang-Stoll, which is a definition rather than a claim about markets."""
    _market, attribution = _run()
    for row in attribution.report(seconds(1)).values():
        assert row.realized_spread == pytest.approx(
            row.spread_captured + row.adverse_selection
        )


# --------------------------------------------------------------------------
# Not perturbing what it measures
# --------------------------------------------------------------------------


def test_watching_the_market_does_not_change_it():
    """The load-bearing property, and the reason the hook does no arithmetic.

    An observer that moved a single price would make every number it produced
    a statement about a market nobody else is trading. Same seed, same grid,
    once watched and once not: every mark, every position and the conservation
    figure must agree exactly.
    """
    watched, _ = _run(attach=True)
    plain, _ = _run(attach=False)

    assert watched.venue.conservation_check() == 0
    assert plain.venue.conservation_check() == 0
    assert watched.venue.marks() == plain.venue.marks()
    for agent in plain.agents:
        mine = watched.venue.account(agent.agent_id)
        theirs = plain.venue.account(agent.agent_id)
        assert int(mine.cash) == int(theirs.cash), agent.agent_id
        assert mine.positions == theirs.positions, agent.agent_id


def test_a_second_observer_is_refused_rather_than_silently_displacing_the_first():
    """Because the first one would keep reporting, on nothing.

    The same failure shape as a guard whose input was never wired: it does not
    raise, it just stops being true.
    """
    market = build(seed=3)
    first = TradeAttribution(market.venue, horizons=HORIZONS)
    first.attach()
    with pytest.raises(RuntimeError, match="already has a trade observer"):
        TradeAttribution(market.venue, horizons=HORIZONS).attach()
    first.detach()
    TradeAttribution(market.venue, horizons=HORIZONS).attach()


def test_detaching_stops_the_recording():
    market = build(seed=3)
    attribution = TradeAttribution(market.venue, horizons=HORIZONS)
    attribution.attach()
    attribution.detach()
    assert market.venue.trade_observer is None


# --------------------------------------------------------------------------
# What the terms mean
# --------------------------------------------------------------------------


def test_the_horizon_ladder_is_collected_in_full():
    """A single horizon cannot tell a stale quote from an informed one.

    Measured on seed 7 over 600s: the largest maker's adverse selection is
    *positive* through the first half second and only turns over between one
    and five seconds. A pick-off by somebody faster shows the opposite shape
    (all of the damage inside the first hundred milliseconds and flat after),
    so the sign of the short end is what rules latency out. Neither is visible
    without the ladder, which is why the ladder is the default.
    """
    _market, attribution = _run()
    row = next(iter(attribution.report(HORIZONS[0]).values()))
    curve = attribution.curve(row.agent_id)
    assert set(curve) == set(attribution.horizons)


def test_a_maker_quoting_against_noise_alone_is_not_adversely_selected():
    """Adverse selection is a claim about who traded, not about losing money.

    ``mu`` is Glosten-Milgrom's informed share, and it is directly observable
    here while being unobservable on a real venue. Against uninformed flow it
    should be near zero, and it is the parameter that decides whether any
    spread can be profitable at all.
    """
    market, attribution = _run()
    makers, informed = _roles(market)
    noise = frozenset(
        a.agent_id for a in market.agents if type(a).__name__ == "NoiseTrader"
    )
    for maker in makers:
        share = attribution.informed_share(maker, informed)
        assert 0.0 <= share <= 1.0
        assert attribution.informed_share(maker, informed | noise) >= share


def test_informed_share_of_a_stranger_is_zero():
    market, attribution = _run()
    makers, _ = _roles(market)
    assert attribution.informed_share(next(iter(makers)), frozenset({"nobody"})) == 0.0
    assert attribution.informed_share("nobody", frozenset({"anybody"})) == 0.0


def test_one_sided_passive_flow_is_reported_per_symbol():
    """Net over gross. Persistently off zero is a pricing error, not bad luck.

    Measured on seed 7: the makers' passive flow on the binaries reaches +1.00
    and -0.93, every passive fill on a contract landing on the same side,
    without exception. Inventory noise cannot do that; only a quote that is
    wrong in one direction can. The equivalent delta limit is structurally
    blind to it, which is what makes this worth reporting separately.
    """
    market, attribution = _run()
    makers, _ = _roles(market)
    for maker in makers:
        for symbol, ratio in attribution.flow_imbalance(maker).items():
            assert -1.0 <= ratio <= 1.0, (maker, symbol, ratio)


def test_an_unknown_horizon_is_refused_rather_than_interpolated():
    """There is no mid at a time nobody sampled, and inventing one would lie."""
    market = build(seed=3)
    attribution = TradeAttribution(market.venue, horizons=HORIZONS)
    with pytest.raises(ValueError, match="never collected"):
        attribution.report(seconds(999))


def test_a_recorder_with_no_horizons_is_refused():
    market = build(seed=3)
    with pytest.raises(ValueError, match="at least one horizon"):
        TradeAttribution(market.venue, horizons=())


def test_the_default_ladder_spans_latency_and_information():
    """Sub-second at one end, tens of seconds at the other.

    The two failure modes live at opposite ends and a ladder that reached only
    one of them would answer the wrong question confidently.
    """
    assert min(DEFAULT_HORIZONS) <= seconds(0.1)
    assert max(DEFAULT_HORIZONS) >= seconds(30)
