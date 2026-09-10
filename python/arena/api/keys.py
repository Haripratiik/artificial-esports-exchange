"""API credentials, and the signature that proves a request came from one.

A browser session and a programmatic client are different problems. The
dashboard's cookie identifies whoever is holding the browser and is scoped to
one machine; an algorithm needs a credential it can carry, rotate, and be
refused for. So the two live side by side: :mod:`dashboard.identity` for
people, this for programs, and both resolve to the same seat at the venue, so a
key trades the same account its owner sees on screen.

**Requests are signed, not merely labelled.** A bare bearer token in a header
is replayable by anything that observes it once, and it says nothing about the
request it accompanies. Each request here carries a timestamp, and the
signature covers the timestamp, the method, the path and the body together, so
a captured signature cannot be moved onto a different order.

The scheme is HMAC-SHA256 over a shared secret, which is what Alpaca and
Coinbase do and what the standard library gives us without a dependency. It is
worth naming the cost rather than implying there is none: the server has to
hold the secret in order to verify a signature, so anyone who can read the key
store can sign as the key's owner. Kalshi avoids this by taking an RSA *public*
key and verifying a signature it cannot itself produce, which is strictly
better and needs a crypto library this project does not otherwise want. The
store is therefore a secret at rest, and the tradeoff is written down here
rather than discovered later.
"""

from __future__ import annotations

import hmac
import json
import secrets
import time
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any

__all__ = [
    "ApiKey",
    "KeyStore",
    "SignatureError",
    "canonical_request",
    "sign",
    "body_bytes",
    "MAX_SKEW_SECONDS",
    "HEADER_KEY",
    "HEADER_TIMESTAMP",
    "HEADER_SIGNATURE",
]

# How far a request's timestamp may sit from the venue's clock. A signature is
# replayable only inside this window, and the window has to be wide enough to
# survive ordinary clock drift between two machines. Thirty seconds is what
# most venues settle on, for the same reason.
MAX_SKEW_SECONDS = 30

# Headers a signed request carries. Named rather than inlined so that a client
# and the server cannot drift apart on spelling.
HEADER_KEY = "arena-key-id"
HEADER_TIMESTAMP = "arena-timestamp"
HEADER_SIGNATURE = "arena-signature"


class SignatureError(ValueError):
    """A request did not prove it came from the key it claims.

    Deliberately one exception for every failure (unknown key, bad signature,
    stale timestamp, revoked key) and one message for all of them. Saying
    *which* of those went wrong tells a caller holding no valid key which key
    ids exist, and a caller holding a valid one never needs the difference.
    """


def canonical_request(
    method: str, path: str, timestamp: str, body: bytes = b""
) -> bytes:
    """The exact bytes a signature covers.

    Newline-separated rather than concatenated, because concatenation lets two
    different requests produce identical bytes: path ``/v1/orders`` with body
    ``x`` and path ``/v1/order`` with body ``sx`` are indistinguishable once
    joined. A separator that cannot appear in a method removes that whole class
    of collision.

    The path is taken verbatim, query string included, so a signature cannot be
    lifted from one filter onto another.
    """
    return b"\n".join(
        [
            timestamp.encode("utf-8"),
            method.upper().encode("utf-8"),
            path.encode("utf-8"),
            body or b"",
        ]
    )


def sign(secret: str, method: str, path: str, timestamp: str, body: bytes = b"") -> str:
    """The hex signature for one request. The same function serves both sides."""
    message = canonical_request(method, path, timestamp, body)
    return hmac.new(secret.encode("utf-8"), message, sha256).hexdigest()


def body_bytes(payload: Any) -> bytes:
    """Serialise a body the way both sides have to serialise it.

    A signature covers bytes, so client and server must produce the same ones
    from the same object. Sorted keys and no incidental whitespace make that
    true regardless of dict ordering at either end.
    """
    if payload is None:
        return b""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class ApiKey:
    """One credential, and the seat it trades."""

    key_id: str
    secret: str
    agent_id: str
    label: str = ""
    created_at: float = 0.0
    revoked: bool = False

    def public(self) -> dict[str, Any]:
        """Everything about the key except the part that must never be shown."""
        return {
            "key_id": self.key_id,
            "agent_id": self.agent_id,
            "label": self.label,
            "created_at": self.created_at,
            "revoked": self.revoked,
        }


@dataclass
class KeyStore:
    """Issues, verifies and revokes keys.

    In memory, because the market it authorises is in memory too. A key that
    outlived the exchange it trades on would authenticate against an account
    that no longer exists, so a rebuild is expected to invalidate every key and
    :meth:`clear` says so explicitly rather than leaving it to chance.
    """

    keys: dict[str, ApiKey] = field(default_factory=dict)
    # Signatures already accepted, and when they stop being worth remembering.
    #
    # The skew window is the hole this closes. A signature covers the request
    # it accompanies, so a captured one cannot be moved onto a different order,
    # but nothing stopped it being sent again *as itself* for up to thirty
    # seconds. `DELETE /v1/orders` is in that window, and so is any order worth
    # replaying twice.
    _spent: dict[str, float] = field(default_factory=dict, repr=False)

    def issue(self, agent_id: str, label: str = "") -> ApiKey:
        """Mint a credential for one seat.

        The secret is returned here and never again. There is nowhere else to
        read it from, which is the property that makes a leaked store the only
        way it escapes.
        """
        key = ApiKey(
            key_id=f"ak_{secrets.token_hex(8)}",
            secret=secrets.token_hex(32),
            agent_id=agent_id,
            label=label,
            created_at=time.time(),
        )
        self.keys[key.key_id] = key
        return key

    def revoke(self, key_id: str) -> bool:
        """Retire a key without forgetting it.

        Kept rather than deleted so a revoked id can never be reissued to
        somebody else and inherit the first owner's history.
        """
        existing = self.keys.get(key_id)
        if existing is None or existing.revoked:
            return False
        self.keys[key_id] = replace(existing, revoked=True)
        return True

    def for_agent(self, agent_id: str) -> list[ApiKey]:
        return [key for key in self.keys.values() if key.agent_id == agent_id]

    def clear(self) -> None:
        self.keys.clear()
        # The spent set goes with them. Every signature in it was made against
        # a secret that no longer exists, so it can never be presented again,
        # and keeping it would leak a little memory on every rebuild.
        self._spent.clear()

    def verify(
        self,
        key_id: str,
        timestamp: str,
        signature: str,
        method: str,
        path: str,
        body: bytes = b"",
        now: float | None = None,
    ) -> ApiKey:
        """The key behind a request, or :class:`SignatureError`.

        Order matters. The timestamp is checked before the signature so a stale
        request is cheap to refuse, and the comparison is
        :func:`hmac.compare_digest` so the time it takes carries no information
        about how much of the signature was correct.
        """
        key = self.keys.get(key_id)
        if key is None or key.revoked:
            raise SignatureError("could not authenticate this request")

        try:
            sent_at = float(timestamp)
        except (TypeError, ValueError):
            raise SignatureError("could not authenticate this request") from None

        clock = time.time() if now is None else now
        if abs(clock - sent_at) > MAX_SKEW_SECONDS:
            raise SignatureError("could not authenticate this request")

        expected = sign(key.secret, method, path, timestamp, body)
        if not hmac.compare_digest(expected, signature or ""):
            raise SignatureError("could not authenticate this request")

        # Authentic, and now also spent.
        #
        # Checked here rather than with the timestamp, even though checking it
        # earlier would be cheaper, because an unauthentic signature must not be
        # able to put anything in this table. Remembering every forged signature
        # a stranger sends is a memory exhaustion attack that needs no
        # credential at all, and the cheap-refusal ordering that makes the skew
        # check come first is exactly what would have caused it.
        if signature in self._spent:
            raise SignatureError("could not authenticate this request")
        # Nothing outside the skew window can be replayed anyway, so an entry
        # older than that is dead weight. Swept on insert rather than on a timer
        # because there is no timer here, and bounded by a size check so the
        # sweep cost is amortised rather than paid on every request.
        if len(self._spent) > 512:
            self._spent = {
                seen: expires for seen, expires in self._spent.items() if expires > clock
            }
        self._spent[signature] = sent_at + MAX_SKEW_SECONDS
        return key
