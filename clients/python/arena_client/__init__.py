"""Client library for the Artificial Esports Exchange trading API.

Ten lines from nothing to a working order::

    from decimal import Decimal
    from arena_client import ArenaClient

    client = ArenaClient("http://localhost:8000", key_id="ak_...", secret="...")
    book = client.book("VANTA_OBJECTIVE_WR", depth=5)
    best_bid = book["bids"][0].price          # Decimal('4689.00'), not 4689.0
    order = client.place_order(
        "VANTA_OBJECTIVE_WR", "buy", 1, price=best_bid, time_in_force="gtc"
    )

The signing is automatic once a key is present, and every price and balance
arrives as :class:`decimal.Decimal`. See :mod:`arena_client.client` for why both
of those are load-bearing rather than stylistic, and ``docs/API.md`` for the
endpoint reference and a worked signing vector you can check another language's
implementation against.

The venue is a simulation. No real money and no real securities are involved.
"""

from arena_client.client import (
    HEADER_KEY,
    HEADER_SIGNATURE,
    HEADER_TIMESTAMP,
    MAX_SKEW_SECONDS,
    ArenaClient,
    ArenaError,
    AuthError,
    ClientError,
    InvalidRequest,
    Level,
    NotFound,
    RateLimited,
    Rejected,
    amount,
    body_bytes,
    canonical_request,
    sign,
)

__all__ = [
    "ArenaClient",
    "ArenaError",
    "AuthError",
    "ClientError",
    "InvalidRequest",
    "Level",
    "NotFound",
    "RateLimited",
    "Rejected",
    "HEADER_KEY",
    "HEADER_SIGNATURE",
    "HEADER_TIMESTAMP",
    "MAX_SKEW_SECONDS",
    "amount",
    "body_bytes",
    "canonical_request",
    "sign",
]

__version__ = "0.0.1"
