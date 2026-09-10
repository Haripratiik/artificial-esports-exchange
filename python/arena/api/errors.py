"""One shape for every refusal.

A trading client cannot branch on prose. If a rejection arrives sometimes as
``{"detail": "..."}``, sometimes as ``{"ok": false, "error": "..."}`` and
sometimes as a bare 500, the only thing an algorithm can reliably do with a
failure is stop, which is the wrong response to "you are one tick off the grid"
and the right one to "your signature is invalid", and the client cannot tell
those apart.

So every failure carries a stable machine code, a sentence for a human, and an
HTTP status, and the three never disagree. The codes are the contract; the
sentences are free to improve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["ApiError", "ERRORS", "error_body"]


@dataclass(frozen=True)
class ApiError(Exception):
    """A refusal a client can act on."""

    code: str
    message: str
    status: int = 400
    detail: dict[str, Any] | None = None

    def body(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"error": {"code": self.code, "message": self.message}}
        if self.detail:
            payload["error"]["detail"] = self.detail
        return payload


def error_body(code: str, message: str, **detail: Any) -> dict[str, Any]:
    return ApiError(code, message, detail=detail or None).body()


# The catalogue. Grouped by what a client should *do* about it, because that is
# the only grouping that helps an algorithm decide.
#
#   auth_*        stop and fix the credential; retrying cannot help
#   invalid_*     the request is malformed; fix it and resend
#   rejected_*    the venue understood and refused; the market may allow it later
#   not_found     the thing addressed does not exist
#   rate_limited  back off and retry; this one is temporary by definition
ERRORS: dict[str, tuple[str, int]] = {
    "auth_required": ("this endpoint needs a signed request", 401),
    "auth_invalid": ("could not authenticate this request", 401),
    "not_found": ("no such resource", 404),
    "invalid_request": ("the request could not be read", 400),
    "invalid_symbol": ("no such instrument", 400),
    "invalid_side": ("side must be buy or sell", 400),
    "invalid_quantity": ("quantity must be a positive whole number", 400),
    "invalid_price": ("price must be a number on the instrument's tick grid", 400),
    "invalid_time_in_force": ("unknown time in force", 400),
    "invalid_order_type": ("unknown order type", 400),
    "rejected_by_venue": ("the venue refused this order", 422),
    "rate_limited": ("too many requests", 429),
}
