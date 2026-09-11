"""The engine's data model, as this venue fills it in.

Deliberately a copy rather than an import. The engine that defines these
shapes is a separate repository with its own dependencies, and a connector
that could only be built by installing it would be a connector nobody runs.
What is copied is the shape a prediction market has anywhere: a series, an
event, and the markets written on it. Field names match the engine's so that
a value built here drops into it unchanged.

Two conversions live here because they are where the venues genuinely differ.

**Money.** The engine holds prices as integer cents, 1 to 99, because that is
what a Kalshi contract is. This exchange holds integer minor units at a scale
of 1,000,000 and lists nine asset classes over five settlement ranges. The
conversion is exact for the class that matters: an event contract here is
bounded [0, 1] on a 0.01 tick, so 0.47 is 47 cents with nothing lost. It is
refused for every other class rather than approximated, because a future
bounded [0, 10,000] is not a probability and rounding it into one would put a
number in the engine that means nothing.

**Exhaustiveness.** Kalshi marks a mutually exclusive event with a flag and
the engine verifies it. Here the venue publishes the partition itself, so the
flag is not a claim to be checked but a fact to be read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

__all__ = [
    "Venue",
    "Side",
    "Series",
    "Event",
    "Market",
    "CENTS",
    "to_cents",
    "PriceOutOfModel",
]

# One contract's price range, in the engine's units. A probability in cents.
CENTS = 100


class Venue(StrEnum):
    """The engine's venue enum, with this exchange added.

    The value matters: it is what a manifest records and what a KPI digest
    groups by, so it has to be stable and it has to be this venue's own name.
    """

    ARENA = "arena"


class Side(StrEnum):
    """YES-referenced, as the engine is throughout."""

    YES = "yes"
    NO = "no"


class PriceOutOfModel(ValueError):
    """A price that cannot be a probability, offered to something expecting one.

    Raised rather than clamped. A future on this venue settles anywhere in
    [0, 10,000] and a spread can settle below zero; squeezing either into
    1 to 99 cents produces a number that is the right type and a lie, and the
    engine downstream has no way left to notice.
    """


def to_cents(price: Decimal | float | str, bounds: tuple[Decimal, Decimal]) -> int:
    """A price on this venue, as the engine's integer cents.

    Exact for a contract bounded [0, 1], which is every contract the engine is
    able to reason about. `bounds` is carried rather than assumed so that the
    refusal below is about the contract rather than about the number: 0.47 is
    a perfectly good price on a future too, and it is not 47 cents there.
    """
    low, high = Decimal(bounds[0]), Decimal(bounds[1])
    if (low, high) != (Decimal(0), Decimal(1)):
        raise PriceOutOfModel(
            f"a contract bounded [{low}, {high}] is not a probability, so its "
            "price has no reading in cents"
        )
    return int((Decimal(str(price)) * CENTS).to_integral_value())


@dataclass(frozen=True, slots=True)
class Series:
    """A family of events written on the same question.

    On this venue a family is a contract family on a match: who wins, who
    finishes top N, how many eliminations, and the head to head. The engine
    uses a series to carry fee terms and settlement sources, and this venue
    charges maker-taker rather than Kalshi's quadratic fee, which is why the
    fee fields are named rather than defaulted.
    """

    ticker: str
    title: str = ""
    category: str = ""
    fee_type: str = "maker_taker"
    fee_multiplier: float = 1.0
    settlement_sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Event:
    """One thing that resolves, and the markets written on it.

    Here that is one live match. `mutually_exclusive` is read from the venue's
    own published partition rather than inferred from the tickers, and
    `exhaustive_verified` is true for the same reason: the venue states which
    contracts sum to a known constant, so there is nothing left to verify.
    """

    venue: Venue = Venue.ARENA
    event_ticker: str = ""
    series_ticker: str = ""
    category: str = ""
    title: str = ""
    mutually_exclusive: bool = False
    exhaustive_verified: bool = False
    # The partition this event publishes, as {name: (symbols, total)}. Empty
    # where the venue declares none, which is every non-match contract.
    partitions: dict[str, tuple[tuple[str, ...], float]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Market:
    """One contract, quoted.

    Prices are the touch as the engine reads it: `yes_bid` is what this market
    can be sold at and `yes_ask` what it can be bought at, both in cents, and
    either may be absent on a book quoted one side only.
    """

    venue: Venue = Venue.ARENA
    ticker: str = ""
    event_ticker: str = ""
    series_ticker: str = ""
    title: str = ""
    status: str = ""
    yes_bid: int | None = None
    yes_ask: int | None = None
    yes_bid_size: float = 0.0
    yes_ask_size: float = 0.0
    volume: float = 0.0
    open_interest: float = 0.0
    close_at_us: int | None = None
    rules_primary: str = ""

    @property
    def is_quoted_both_sides(self) -> bool:
        return self.yes_bid is not None and self.yes_ask is not None

    @property
    def mid(self) -> float | None:
        if not self.is_quoted_both_sides:
            return None
        return (self.yes_bid + self.yes_ask) / 2.0
