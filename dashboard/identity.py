"""Who is at the browser, and how the server knows.

The exchange had no answer to that question. Every connection traded the same
account, so two tabs were one trader: they shared a balance and a blotter, and
either could cancel the other's working orders. That is not a small gap in a
thing whose whole point is that people trade against each other.

What this is
------------

A signed session cookie. The cookie carries an account id and a display name;
the signature is an HMAC over both with a key the process holds, so a browser
can read what it is but cannot write itself a different account. Nothing is
stored server-side beyond the accounts the market already has.

What this is deliberately not
-----------------------------

**There are no passwords.** Signing in means choosing a name, and a name is not
an identity: two people called Ada get two accounts, and losing the cookie
means losing the account. That is the right shape for a paper-trading exchange
where the capital is imaginary, and it is stated here rather than implied,
because the difference between "signed in" and "authenticated" is exactly the
kind of thing that is comfortable to leave vague.

A real one would need credentials to store, which means storing password
hashes, a reset path, and a way to be wrong about all of it. None of that is
built, and pretending otherwise by putting a password box on the page would be
worse than not having one.

The signing key
---------------

Generated per process unless ``ARENA_SECRET`` is set. A restart therefore
invalidates every session, which is correct for a simulator whose market does
not survive the restart either: the accounts those cookies name are gone too.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import secrets
from hashlib import sha256
from typing import Any

__all__ = ["COOKIE", "sign", "verify", "new_identity", "display_name"]

COOKIE = "arena_session"
_SECRET = os.environ.get("ARENA_SECRET", "").encode() or secrets.token_bytes(32)

# Names for people who have not chosen one. Ordinary words rather than
# "trader-7": the counterparty panel names whoever took the other side, and a
# list of serial numbers tells a reader nothing about there being people there.
_ADJECTIVES = (
    "quiet", "patient", "brisk", "steady", "sharp", "idle", "wry", "keen",
    "spare", "candid", "blunt", "gentle", "sly", "plain", "eager",
)
_NOUNS = (
    "otter", "heron", "marten", "finch", "vole", "shrike", "lynx", "swift",
    "raven", "pike", "ibis", "stoat", "grebe", "adder", "kite",
)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign(payload: dict[str, Any]) -> str:
    """A cookie value the browser can read and cannot forge."""
    body = _b64(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    mac = _b64(hmac.new(_SECRET, body.encode(), sha256).digest())
    return f"{body}.{mac}"


def verify(value: str | None) -> dict[str, Any] | None:
    """The payload if the signature holds, otherwise ``None``.

    Compared with :func:`hmac.compare_digest` rather than ``==``. The timing
    difference is not a plausible attack on a paper-trading exchange, and
    writing the comparison the other way in code people read and copy is how
    the habit spreads.
    """
    if not value or value.count(".") != 1:
        return None
    body, mac = value.split(".")
    expected = _b64(hmac.new(_SECRET, body.encode(), sha256).digest())
    if not hmac.compare_digest(mac, expected):
        return None
    try:
        payload = json.loads(_unb64(body))
    except (ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def new_identity(rng: secrets.SystemRandom | None = None) -> str:
    """A display name for someone who has not chosen one."""
    chooser = rng or secrets.SystemRandom()
    return f"{chooser.choice(_ADJECTIVES)} {chooser.choice(_NOUNS)}"


def display_name(raw: str | None) -> str:
    """A name safe to put beside a price.

    Trimmed, length-capped, and stripped of control characters. It is rendered
    as text on every other trader's screen, so what arrives here is the sort of
    string that has to be assumed hostile.
    """
    if not raw:
        return new_identity()
    cleaned = "".join(c for c in raw if c.isprintable()).strip()
    return cleaned[:24] or new_identity()
