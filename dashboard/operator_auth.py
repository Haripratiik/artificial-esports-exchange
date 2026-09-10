"""Who is allowed to reach for the controls that affect everybody else.

There are two kinds of route on this server and they were being treated as one.
Most of them affect only the caller: place an order, read your positions,
rename yourself. A handful affect *every* participant at once, and those had no
guard on them at all.

The one that matters most is ``POST /api/config``. It calls
``MarketRunner.reconfigure``, whose own docstring says "The old one is
discarded, not paused": every account, every position, every working order and
every price series, for everyone connected, gone. Any visitor could send it.
``POST /api/participant/{agent_id}/kill`` was worse in a quieter way: it takes
an **arbitrary** agent id, so one visitor could reach across and disable
another human's seat, pulling their working orders while they watched.

That is survivable while the exchange is one person's demo. It is the end of
the product the moment two strangers are trading against each other, which is
exactly what this exchange is for.

**The token is generated, not defaulted.** A default like "admin" or an empty
string that means "allow" is the shape of every embarrassing breach, because
the deployment that forgets to override it looks identical to the one that
did. If ``ARENA_OPERATOR_TOKEN`` is unset a fresh random token is minted at
startup and printed once to the console. An operator running locally copies it
from their terminal; a deployment sets the variable. Neither ends up open by
accident.

Comparison worth recording: no venue surveyed exposes its operator controls at
all. Interactive Brokers' paper-account reset is a Client Portal form
processed the next business day, and Alpaca removed its reset endpoint
entirely in favour of create-and-delete. The convention is that destructive
controls are not an API, and gating them behind a token is already more
permissive than the industry.
"""

from __future__ import annotations

import hmac
import os
import secrets
from typing import Any

__all__ = ["OPERATOR_HEADER", "operator_token", "is_operator", "token_was_generated"]

# Named rather than inlined so a client and this module cannot disagree on
# spelling, the same argument `arena.api.keys` makes for its own headers.
OPERATOR_HEADER = "arena-operator-token"

_ENV = "ARENA_OPERATOR_TOKEN"

_configured = os.environ.get(_ENV, "").strip()
_generated = not _configured

# A token exists either way. The only question is whether the deployment chose
# it, and `token_was_generated()` answers that so startup can say so out loud.
_TOKEN = _configured or secrets.token_urlsafe(24)


def operator_token() -> str:
    return _TOKEN


def token_was_generated() -> bool:
    """True when nobody set one, so the caller can print it once at startup."""
    return _generated


def is_operator(request_or_socket: Any) -> bool:
    """Whether this request carries the operator token.

    Compared with :func:`hmac.compare_digest` so the time taken carries no
    information about how much of the token was correct. That matters more
    here than it looks: an operator token is a bearer credential with no
    timestamp and no request binding, so a timing oracle would be the whole
    attack rather than one step of it.
    """
    headers = getattr(request_or_socket, "headers", None)
    if headers is None:
        return False
    sent = headers.get(OPERATOR_HEADER) or ""
    if not sent:
        return False
    return hmac.compare_digest(sent, _TOKEN)
