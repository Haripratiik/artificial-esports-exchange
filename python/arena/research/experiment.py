"""One trial of the information-aggregation experiment, and its manifest.

A trial is a purpose-built minimal market rather than the seven-instrument
dashboard world. That is a methodological choice, not a shortcut: the dashboard
market has spreads, options and an index whose prices all move together, so a
difference in the binary's accuracy could come from any of them. One contract,
one population, one question.

What a trial asks
-----------------

A contract pays 1 if a Brawler's measured win rate over the observation window
exceeds a threshold. The true rate ``p*`` is fixed by the trial. The window
holds ``window_battles`` battles, so the *measured* rate is a draw, and

    truth = P(Binomial(N, p*) / N > theta)

is a genuine probability in (0, 1) that is known exactly. Every forecast in the
experiment -- the market's and each baseline's -- is scored against this number.

Real prediction-market studies cannot do this. They score against a single
realised outcome, so most of what they measure is Bernoulli noise rather than
skill; separating a good forecaster from a lucky one takes thousands of
questions. Here the noise is removed by construction, and the sampled outcome is
kept only as a secondary, deliberately noisier check that the primary metric is
not an artefact of knowing the answer.

Each informed agent sees ``n_j`` battles drawn from the same process, forms a
Beta posterior, and forecasts the Beta-Binomial tail -- which is *exactly*
Bayes-optimal for what it has seen. So the population the market is scored
against is not a crowd of sloppy forecasters. Every one of them is individually
optimal, and the only thing left for the market to add is aggregation.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from arena.agents.bayesian import (
    BayesianFundamental,
    posterior_for,
    predictive_probability,
    settlement_probability,
)
from arena.agents.market_maker import MarketMaker
from arena.agents.noise import NoiseTrader
from arena.contracts.payoff import Binary
from arena.contracts.spec import ContractSpec, DataPolicy, ObservationWindow
from arena.contracts.underlying import Single
from arena.determinism import canonical_json
from arena.exchange.types import AgentId, Side
from arena.market.instrument import Instrument
from arena.market.lmsr_venue import LmsrVenue
from arena.market.venue import Venue
from arena.market.venue_agent import VenueAgent
from arena.sim.kernel import Kernel, SimulationContext
from arena.sim.latency import PairwiseLatency
from arena.sim.time import Duration, Timestamp, micros, millis, seconds
from arena.worlds.circuit.metrics import metric_ref

__all__ = ["TrialConfig", "TrialResult", "run_trial", "draw_trials", "manifest_digest"]

UTC = timezone.utc
VENUE_ID = AgentId("venue")
SYMBOL = "WR_BINARY"


@dataclass(frozen=True, slots=True)
class TrialConfig:
    """Everything that determines a trial, and nothing that does not.

    Frozen and fully serialisable, so a run is reproducible from its manifest
    alone. If a field is not here it cannot influence the result.
    """

    seed: int
    truth: float
    threshold: float
    battles: tuple[int, ...]
    window_battles: int = 2_000
    prior_mean: float = 0.5
    prior_strength: float = 20.0
    noise_traders: int = 8
    duration: int = int(seconds(300))
    measure_fraction: float = 0.10
    maker_half_spread: int = 2
    maker_quote_size: int = 40
    starting_cash: int = 4_000_000
    heterogeneous_latency: bool = True
    # How large a position an informed agent may carry. This is the channel
    # through which a view reaches the price, so it is a first-class knob
    # rather than a constant buried in the builder.
    position_limit: int = 800
    # Which market mechanism runs the trial. "clob" is a limit order book with a
    # quoting market maker; "lmsr" is a logarithmic scoring rule, where the
    # maker is a cost function rather than a participant. Experiment 2 varies
    # only this.
    venue_kind: str = "clob"
    # What the scoring-rule venue is willing to lose making the market.
    # Calibrated so a fresh scoring-rule book shows the same depth at the touch
    # as the order-book maker quotes -- 40 lots a tick. Depth decides how far a
    # given amount of informed trading moves the price, so two venues quoting
    # different depth are not being compared on mechanism at all. Derived by
    # subsidy_for_depth(40, 0.01, 1.0) rather than chosen.
    subsidy: float = 693.1

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "truth": self.truth,
            "threshold": self.threshold,
            "battles": list(self.battles),
            "window_battles": self.window_battles,
            "prior_mean": self.prior_mean,
            "prior_strength": self.prior_strength,
            "noise_traders": self.noise_traders,
            "duration": self.duration,
            "measure_fraction": self.measure_fraction,
            "maker_half_spread": self.maker_half_spread,
            "maker_quote_size": self.maker_quote_size,
            "starting_cash": self.starting_cash,
            "heterogeneous_latency": self.heterogeneous_latency,
            "position_limit": self.position_limit,
            "venue_kind": self.venue_kind,
            "subsidy": self.subsidy,
        }

    @property
    def truth_probability(self) -> float:
        """The number every forecast is scored against."""
        return settlement_probability(self.truth, self.window_battles, self.threshold)


@dataclass(frozen=True, slots=True)
class TrialResult:
    config: TrialConfig
    truth_probability: float
    market_forecast: float
    closing_forecast: float
    agent_forecasts: tuple[float, ...]
    agent_battles: tuple[int, ...]
    outcome: float
    trades: int
    quoted_fraction: float
    conservation: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "truth_probability": self.truth_probability,
            "market_forecast": self.market_forecast,
            "closing_forecast": self.closing_forecast,
            "agent_forecasts": list(self.agent_forecasts),
            "agent_battles": list(self.agent_battles),
            "outcome": self.outcome,
            "trades": self.trades,
            "quoted_fraction": self.quoted_fraction,
            "conservation": self.conservation,
        }


# --------------------------------------------------------------------------
# Building one trial's market
# --------------------------------------------------------------------------


def _spec(threshold: float, window_battles: int) -> ContractSpec:
    """A binary on a synthetic subject.

    The subject is synthetic on purpose. Reading a real Brawler's rate out of
    the fixture would fix ``p*`` to one value, and the experiment needs it to
    vary across trials so the forecasts span the probability range instead of
    clustering wherever that Brawler happens to sit.
    """
    start = datetime(2026, 8, 31, tzinfo=UTC)
    return ContractSpec(
        contract_id=SYMBOL,
        underlying=Single(metric_ref("win_rate", "TRIAL_SUBJECT")),
        payoff=Binary(">", threshold, payout=1.0),
        window=ObservationWindow(start, start + timedelta(days=28)),
        policy=DataPolicy(
            min_sample_size=window_battles,
            min_stratum_battles=200,
            min_strata_coverage=0.80,
        ),
        reference_id="ref-2026S09-v1",
        published_at=start - timedelta(days=1),
        tick_size="0.01",
    )


def _latency(config: TrialConfig, agents: list[AgentId], maker: AgentId) -> PairwiseLatency:
    """Colocated maker, informed agents spread out, noise traders far away.

    Homogeneous latency is available as an ablation because it is the control:
    if aggregation only works when some agents are faster, that is a fact about
    the speed hierarchy rather than about the market.
    """
    if not config.heterogeneous_latency:
        return PairwiseLatency(default=millis(5), jitter_fraction=0.15, seed=config.seed)
    per_agent = {maker: micros(150)}
    for index, agent_id in enumerate(agents):
        per_agent[agent_id] = millis(2 + 3 * index)
    return PairwiseLatency(
        default=millis(40), per_agent=per_agent, jitter_fraction=0.15, seed=config.seed
    )


def run_trial(config: TrialConfig) -> TrialResult:
    """Run one trial and return the market's forecast alongside the agents'."""
    instrument = Instrument(SYMBOL, _spec(config.threshold, config.window_battles))
    by_symbol = {SYMBOL: instrument}
    levels = {SYMBOL: config.truth}

    if config.venue_kind == "lmsr":
        venue = LmsrVenue(
            "arena-lmsr",
            starting_cash=config.starting_cash,
            subsidy=config.subsidy,
        )
    elif config.venue_kind == "clob":
        venue = Venue("arena", starting_cash=config.starting_cash)
    else:
        raise ValueError(f"unknown venue kind {config.venue_kind!r}")
    venue.list_instrument(instrument)

    maker_id = AgentId("mm-1")
    fund_ids = [AgentId(f"fund-{i:02d}") for i in range(len(config.battles))]
    noise_ids = [AgentId(f"noise-{i:02d}") for i in range(config.noise_traders)]

    kernel = Kernel(seed=config.seed, latency=_latency(config, fund_ids, maker_id))
    venue_agent = VenueAgent(VENUE_ID, venue)

    low, high = instrument.tick_bounds
    # The scoring rule *is* the market maker, so adding a quoting agent beside
    # it would be running two makers and calling it one mechanism.
    makers = (
        []
        if config.venue_kind == "lmsr"
        else [
            MarketMaker(
                maker_id,
                VENUE_ID,
                by_symbol,
                wake_interval=millis(250),
                half_spread=config.maker_half_spread,
                quote_size=config.maker_quote_size,
                max_skew_fraction=0.15,
                position_limit=2_000,
                # Opens at the middle of the range, not at the answer: the maker
                # must not be the thing that already knows.
                reference={SYMBOL: float((int(low) + int(high)) / 2)},
            )
        ]
    )

    funds = [
        BayesianFundamental(
            agent_id,
            VENUE_ID,
            by_symbol,
            levels,
            battles=battles,
            prior_mean=config.prior_mean,
            prior_strength=config.prior_strength,
            window_battles=config.window_battles,
            wake_interval=millis(500 + 130 * index),
            base_size=12,
            max_position=config.position_limit,
        )
        for index, (agent_id, battles) in enumerate(zip(fund_ids, config.battles))
    ]
    noise = [
        NoiseTrader(agent_id, VENUE_ID, by_symbol, wake_interval=millis(900))
        for agent_id in noise_ids
    ]

    kernel.add(venue_agent)
    kernel.add_all([*makers, *funds, *noise])
    kernel.start()

    # Sample the mid on a fixed grid so the "time-weighted" average really is
    # time-weighted rather than weighted by how often the book happened to move.
    step = int(millis(250))
    total_steps = max(1, config.duration // step)
    mids: list[float | None] = []
    engine = venue.engine(SYMBOL)
    for i in range(1, total_steps + 1):
        kernel.advance(until=Timestamp(i * step))
        # `best_priced`, not `best_price`. Market-on-open interest rests at a
        # sentinel so that it crosses everything in the call, and this loop is
        # sampling the mid that *becomes the market's forecast*. A single
        # sentinel sample would put 2^62 into the average and the trial would
        # report it as the market's opinion.
        bid = engine.book.best_priced(Side.BUY)
        ask = engine.book.best_priced(Side.SELL)
        mids.append(None if bid is None or ask is None else (int(bid) + int(ask)) / 2)
    kernel.finish()

    payout = float(instrument.spec.payoff.payout)

    def to_probability(ticks: float) -> float:
        return float(instrument.from_ticks(int(round(ticks)))) / payout

    tail = max(1, int(total_steps * config.measure_fraction))
    observed = [m for m in mids[-tail:] if m is not None]
    fallback = [m for m in mids if m is not None]
    if observed:
        market_forecast = to_probability(sum(observed) / len(observed))
    elif fallback:
        market_forecast = to_probability(fallback[-1])
    else:
        market_forecast = float("nan")
    closing = to_probability(fallback[-1]) if fallback else float("nan")

    forecasts = []
    for agent in funds:
        value = agent.forecast(SimulationContext(kernel, agent.agent_id), SYMBOL)
        forecasts.append(float("nan") if value is None else value)

    # The realised outcome, for the secondary (noisier) Brier score. Drawn from
    # its own generator so that adding or removing this line cannot shift the
    # simulation's own random stream.
    draw = random.Random(_stable_int(config.seed, "outcome"))
    measured = draw.binomialvariate(config.window_battles, config.truth)
    outcome = 1.0 if measured / config.window_battles > config.threshold else 0.0

    return TrialResult(
        config=config,
        truth_probability=config.truth_probability,
        market_forecast=market_forecast,
        closing_forecast=closing,
        agent_forecasts=tuple(forecasts),
        agent_battles=tuple(config.battles),
        outcome=outcome,
        trades=len(engine.tape),
        quoted_fraction=sum(1 for m in mids if m is not None) / max(1, len(mids)),
        conservation=int(venue.conservation_check()),
    )


# --------------------------------------------------------------------------
# Designing the trial set
# --------------------------------------------------------------------------


def _stable_int(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def draw_trials(
    count: int,
    seed: int = 0,
    agents: int = 8,
    min_battles: int = 50,
    max_battles: int = 5_000,
    window_battles: int = 2_000,
    prior_mean: float = 0.5,
    prior_strength: float = 20.0,
    **overrides: Any,
) -> list[TrialConfig]:
    """Draw a set of trials whose answers span the probability range.

    Thresholds are chosen to place the truth roughly uniformly over [0.05,
    0.95] rather than at random. Random thresholds would put almost every trial
    at 0 or 1 -- a window of 2,000 battles measures a rate tightly, so a
    threshold even slightly away from ``p*`` makes the answer a foregone
    conclusion. A set of foregone conclusions cannot separate any two
    forecasters, so it would be a null experiment dressed as a real one.

    Information is log-spaced across agents because that is what makes the
    dispersion wide enough to matter: eight agents all seeing about the same
    number of battles is one forecaster with extra steps.
    """
    rng = random.Random(_stable_int(seed, "design"))
    # Log-spaced battle counts, fixed across trials so the population's
    # information profile is a constant of the experiment rather than a nuisance
    # source of variation between trials.
    span = math.log(max_battles / min_battles)
    battles = tuple(
        int(round(min_battles * math.exp(span * i / max(1, agents - 1))))
        for i in range(agents)
    )

    trials: list[TrialConfig] = []
    for index in range(count):
        target = 0.05 + 0.90 * (index + 0.5) / count
        truth = rng.uniform(0.35, 0.65)
        threshold = _threshold_for(truth, window_battles, target)
        trials.append(
            TrialConfig(
                seed=_stable_int(seed, f"trial-{index}") % (2**31),
                truth=truth,
                threshold=threshold,
                battles=battles,
                window_battles=window_battles,
                prior_mean=prior_mean,
                prior_strength=prior_strength,
                **overrides,
            )
        )
    return trials


def _threshold_for(truth: float, window_battles: int, target: float) -> float:
    """The threshold making P(measured rate > threshold) equal ``target``.

    Inverted by bisection on a monotone function rather than by a normal
    approximation, because the approximation is worst exactly in the tails
    where the interesting trials live.
    """
    low, high = 0.0, 1.0
    for _ in range(60):
        mid = (low + high) / 2
        if settlement_probability(truth, window_battles, mid) > target:
            low = mid
        else:
            high = mid
    return round((low + high) / 2, 6)


def bayes_optimal_forecast(config: TrialConfig, battles: int, seed_label: str) -> float:
    """What a single agent with this much evidence would say, for reference."""
    rng = random.Random(_stable_int(config.seed, seed_label))
    a, b = posterior_for(
        config.truth, battles, config.prior_mean, config.prior_strength, rng
    )
    return predictive_probability(a, b, config.window_battles, config.threshold)


def manifest_digest(payload: dict[str, Any]) -> str:
    """A stable digest of a run's configuration and results.

    Determinism is only a claim until something checks it, and this is what the
    determinism test compares between two runs of the same manifest.
    """
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()
