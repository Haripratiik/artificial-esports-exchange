"""A signature is good once.

Signing a request covers the request, so a captured signature cannot be moved
onto a different order. It said nothing about the same order being sent twice,
and the skew window left thirty seconds in which to do it.
"""

from __future__ import annotations

import time

import pytest

from arena.api.keys import MAX_SKEW_SECONDS, KeyStore, SignatureError, body_bytes, sign


def _signed(store, key, method="POST", path="/v1/orders", payload=None, at=None):
    now = time.time() if at is None else at
    stamp = repr(now)
    body = body_bytes(payload)
    return {
        "key_id": key.key_id,
        "timestamp": stamp,
        "signature": sign(key.secret, method, path, stamp, body),
        "method": method,
        "path": path,
        "body": body,
        "now": now,
    }


def test_a_captured_request_cannot_be_sent_again():
    """The window it had was the whole skew window, and DELETE /v1/orders is in it."""
    store = KeyStore()
    key = store.issue("agent-1")
    request = _signed(store, key)

    assert store.verify(**request) is key
    with pytest.raises(SignatureError):
        store.verify(**request)


def test_the_same_order_at_a_different_moment_is_not_a_replay():
    """Two identical orders are an ordinary thing to want.

    They differ only in their timestamp, which is what the signature covers, so
    a client that sends the same body twice is refused only if it also claims
    the same instant. Nothing here forces a client to invent a nonce.
    """
    store = KeyStore()
    key = store.issue("agent-1")
    now = time.time()
    payload = {"symbol": "EMBER_OBJECTIVE_WR", "side": "buy", "quantity": 5}

    first = _signed(store, key, payload=payload, at=now)
    second = _signed(store, key, payload=payload, at=now + 0.001)

    assert first["signature"] != second["signature"]
    assert store.verify(**first) is key
    assert store.verify(**second) is key


def test_a_forged_signature_cannot_fill_the_table():
    """Which is why the spent check runs after verification, not before it.

    Refusing cheaply first is the ordering the rest of this function uses, and
    here it would be the bug: remembering every signature a stranger makes up
    is a memory exhaustion attack that needs no credential at all.
    """
    store = KeyStore()
    store.issue("agent-1")
    key = store.issue("agent-2")
    request = _signed(store, key)

    for n in range(50):
        forged = dict(request, signature=f"{n:0>64x}")
        with pytest.raises(SignatureError):
            store.verify(**forged)
    assert store._spent == {}


def test_a_signature_too_old_to_replay_is_forgotten():
    """Bounded memory. Nothing outside the skew window can be presented anyway."""
    store = KeyStore()
    key = store.issue("agent-1")
    base = time.time()

    for n in range(600):
        request = _signed(store, key, path=f"/v1/orders/{n}", at=base)
        store.verify(**request)
    assert len(store._spent) == 600

    later = _signed(store, key, path="/v1/late", at=base + MAX_SKEW_SECONDS * 3)
    store.verify(**later)
    # The sweep runs on insert once the table is worth sweeping, and everything
    # from `base` is now further away than a request is allowed to be.
    assert len(store._spent) == 1


def test_rebuilding_the_exchange_forgets_the_spent_signatures_too():
    """They were made against secrets that no longer exist."""
    store = KeyStore()
    key = store.issue("agent-1")
    store.verify(**_signed(store, key))
    assert store._spent

    store.clear()
    assert store._spent == {}


def test_replay_protection_does_not_leak_which_failure_occurred():
    """One message for every refusal, as everywhere else in this module.

    A distinct "already used" would tell a caller holding no valid key that the
    signature it just guessed was, at some point, real.
    """
    store = KeyStore()
    key = store.issue("agent-1")
    request = _signed(store, key)
    store.verify(**request)

    with pytest.raises(SignatureError) as replayed:
        store.verify(**request)
    with pytest.raises(SignatureError) as unknown:
        store.verify(**dict(request, key_id="ak_nope"))
    assert str(replayed.value) == str(unknown.value)
