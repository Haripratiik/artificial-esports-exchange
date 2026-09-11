"""Request signing for this venue, in the shape the engine's signer has.

Kalshi signs `timestamp + METHOD + path` with RSA-PSS and sends three headers.
This venue signs the same four fields with HMAC-SHA256 and sends three headers.
The interface is therefore the same and the primitive is not, which is exactly
the sort of difference a venue package exists to absorb.

Two differences are worth stating rather than discovering.

**The body is signed here and is not on Kalshi.** A captured signature on this
venue cannot be moved onto a different order, because the order is inside the
signed bytes. It also means client and server have to serialise a body to
identical bytes, which is why `body_bytes` sorts keys and strips incidental
whitespace rather than calling `json.dumps` and hoping.

**The query string is signed here and is not on Kalshi.** Kalshi's signed path
excludes it, so a signature there covers the route but not the filter. Here
the path signed is the path sent, query and all.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any

__all__ = [
    "KEY_HEADER",
    "TS_HEADER",
    "SIG_HEADER",
    "MAX_SKEW_SECONDS",
    "body_bytes",
    "signing_string",
    "sign_message",
    "ArenaSigner",
]

KEY_HEADER = "arena-key-id"
TS_HEADER = "arena-timestamp"
SIG_HEADER = "arena-signature"

# How far this venue lets a request's clock sit from its own. Mirrored here so
# a client can fail loudly on skew rather than collecting mysterious refusals,
# which is the failure mode the Kalshi signer documents for the same reason.
MAX_SKEW_SECONDS = 30


def body_bytes(payload: Any) -> bytes:
    """Serialise a body the way both sides must, or the signature will not match."""
    if payload is None:
        return b""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def signing_string(method: str, path: str, timestamp: str, body: bytes = b"") -> bytes:
    """The exact bytes this venue signs: timestamp, method, path, then body.

    Newline separated with the body last, so no field can be slid into its
    neighbour: path ``/v1/orders`` with body ``x`` and path ``/v1/order`` with
    body ``sx`` are indistinguishable once concatenated.

    The field order is the venue's and is not guessable. Written the obvious
    way round, method first, this produced signatures that verified against
    nothing: a client that could read every public endpoint and authenticate
    on none of them. It is pinned against the venue's own `canonical_request`
    by a test rather than by reading.
    """
    return b"\n".join([
        timestamp.encode("utf-8"),
        method.upper().encode("utf-8"),
        path.encode("utf-8"),
        body or b"",
    ])


def sign_message(secret: str, method: str, path: str, timestamp: str,
                 body: bytes = b"") -> str:
    return hmac.new(
        secret.encode("utf-8"),
        signing_string(method, path, timestamp, body),
        hashlib.sha256,
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ArenaSigner:
    """Holds one key pair and stamps requests with it.

    `key_id` and `secret` come from the venue's own key endpoint. Unlike
    Kalshi there is no PEM file to load, because the secret is symmetric: the
    venue knows it too, which is what makes HMAC possible and what makes a
    leaked secret a full compromise rather than a forged signature.
    """

    key_id: str
    secret: str

    def headers(self, method: str, path: str, body: bytes = b"",
                now: float | None = None) -> dict[str, str]:
        stamp = repr(time.time() if now is None else now)
        return {
            KEY_HEADER: self.key_id,
            TS_HEADER: stamp,
            SIG_HEADER: sign_message(self.secret, method, path, stamp, body),
        }

    def ws_auth(self, path: str = "/v1/stream", now: float | None = None) -> dict[str, Any]:
        """The ``auth`` frame that upgrades a socket to the private channels.

        A socket cannot carry headers on every message, so the venue takes one
        signed frame over the connection instead. It is signed as a GET of the
        stream path with an empty body, so one implementation of the signature
        serves both transports.
        """
        stamp = repr(time.time() if now is None else now)
        return {
            "op": "auth",
            "key_id": self.key_id,
            "timestamp": stamp,
            "signature": sign_message(self.secret, "GET", path, stamp, b""),
        }
