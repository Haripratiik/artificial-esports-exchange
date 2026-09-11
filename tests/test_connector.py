"""The connector that presents this venue to an outside prediction engine.

A connector is the one component that cannot be checked by reading, because
its whole job is to agree with something written somewhere else. These tests
hold it against the venue's own implementation rather than against a
description of it.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from arena.api.keys import KeyStore, body_bytes as venue_body_bytes, canonical_request
from arena.api.keys import sign as venue_sign

from connectors.prediction_engine.audit import (
    elimination_boards,
    top_boards,
    winner_board,
)
from connectors.prediction_engine.auth import ArenaSigner, body_bytes, sign_message
from connectors.prediction_engine.client import family_of, match_of
from connectors.prediction_engine.jointarb import Leg, check
from connectors.prediction_engine.models import PriceOutOfModel, Market, to_cents

CASES = [
    ("GET", "/v1/instruments", "1234.5", b""),
    ("post", "/v1/orders", "1700000000.125", b'{"a":1}'),
    ("DELETE", "/v1/orders/SOLO0_WIN_VANTA/7", "0.0", b""),
    ("GET", "/v1/instruments?class=event&limit=1000", "9.75", b""),
]


@pytest.mark.parametrize("method,path,stamp,body", CASES)
def test_the_connector_signs_exactly_what_the_venue_verifies(method, path, stamp, body):
    """Byte identical, not merely both HMAC.

    The field order is the venue's and there is no way to infer it. Written the
    obvious way round, method before timestamp, this signer produced a client
    that could read every public endpoint and authenticate on none of them,
    which is a failure that looks like a credentials problem and is not.
    """
    assert canonical_request(method, path, stamp, body) == __import__(
        "connectors.prediction_engine.auth", fromlist=["signing_string"]
    ).signing_string(method, path, stamp, body)
    assert venue_sign("secret", method, path, stamp, body) == sign_message(
        "secret", method, path, stamp, body
    )


def test_a_signed_request_is_accepted_by_the_venues_own_keystore():
    """The end the connector actually has to satisfy.

    Signing the same bytes is necessary and is not the claim. The claim is that
    the venue accepts what this produces, so the test asks the venue.
    """
    store = KeyStore()
    key = store.issue("engine")
    signer = ArenaSigner(key.key_id, key.secret)

    payload = {"symbol": "SOLO0_WIN_VANTA", "side": "buy", "quantity": 3}
    body = body_bytes(payload)
    headers = signer.headers("POST", "/v1/orders", body)

    accepted = store.verify(
        key_id=headers["arena-key-id"],
        timestamp=headers["arena-timestamp"],
        signature=headers["arena-signature"],
        method="POST",
        path="/v1/orders",
        body=body,
        now=float(headers["arena-timestamp"]),
    )
    assert accepted is key


def test_the_body_is_serialised_the_way_the_venue_serialises_it():
    """A signature covers bytes, so two dict orderings must produce one string."""
    assert body_bytes({"b": 2, "a": 1}) == venue_body_bytes({"a": 1, "b": 2})
    assert body_bytes(None) == b""


def test_a_contract_that_is_not_a_probability_is_refused_rather_than_rounded():
    """A future bounded [0, 10,000] has no reading in cents.

    Clamping it into 1 to 99 would put a number downstream that is the right
    type and a lie, and every strategy after it would have no way to notice.
    """
    assert to_cents(Decimal("0.47"), (Decimal(0), Decimal(1))) == 47
    assert to_cents(Decimal("1"), (Decimal(0), Decimal(1))) == 100
    with pytest.raises(PriceOutOfModel):
        to_cents(Decimal("4700"), (Decimal(0), Decimal("10000")))


def test_a_symbol_says_which_match_and_which_question_it_belongs_to():
    assert match_of("SOLO0_WIN_VANTA") == "SOLO0"
    assert family_of("SOLO0_WIN_VANTA") == "WIN"
    assert family_of("OBJECTIVE1_ELIM_MIRE_GT2") == "ELIM"
    assert family_of("SOLO0_TOP3_QUILL") == "TOP3"
    assert family_of("SOLO0_H2H_RIFT_OVER_WISP") == "H2H"
    # A statistical contract belongs to no match and is not given a fake one.
    assert match_of("BASTION_OBJECTIVE_GT500") is None
    assert family_of("BASTION_OBJECTIVE_GT500") is None


def _market(ticker, bid, ask, status="continuous"):
    return Market(ticker=ticker, event_ticker="SOLO0", status=status,
                  yes_bid=bid, yes_ask=ask)


def test_a_partition_priced_inside_one_is_consistent():
    """Three legs summing to one, each with a spread around its share."""
    markets = [
        _market("SOLO0_WIN_A", 30, 36),
        _market("SOLO0_WIN_B", 30, 36),
        _market("SOLO0_WIN_C", 30, 36),
    ]
    legs, states = winner_board(markets, "SOLO0")
    assert len(legs) == 3 and len(states) == 3
    assert not check("partition", legs, states).arbitrage


def test_a_partition_whose_bids_sum_past_one_is_an_arbitrage():
    """Sell all three for 1.35 and pay out 1.00, whatever happens.

    This is the direction that matters: bids above one is a portfolio that is
    paid to exist. The LP finds it without being told the legs partition
    anything, which is the property that makes it stronger than a sum.
    """
    markets = [
        _market("SOLO0_WIN_A", 45, 50),
        _market("SOLO0_WIN_B", 45, 50),
        _market("SOLO0_WIN_C", 45, 50),
    ]
    legs, states = winner_board(markets, "SOLO0")
    result = check("rich partition", legs, states)
    assert result.arbitrage
    # The slack is per leg, not the total excess, which is worth knowing before
    # reading one: three bids of 0.45 sum to 1.35, and the cheapest way to
    # satisfy all three at once is to move each one by 0.35/3. A reader who
    # expects the 0.35 will think the LP has understated the violation when it
    # has only shared it out.
    assert result.slack == pytest.approx(0.35 / 3, abs=1e-9)


def test_a_ladder_that_rises_with_its_threshold_is_an_arbitrage():
    """P(X > 2) cannot exceed P(X > 1): the second event contains the first.

    Nobody encodes that relation here. It falls out of the states, which is
    the whole argument for asking the question as a distribution rather than
    as a list of pairwise rules.
    """
    rising = [
        _market("SOLO0_ELIM_TALON_GT0", 10, 12),
        _market("SOLO0_ELIM_TALON_GT1", 30, 32),
        _market("SOLO0_ELIM_TALON_GT2", 60, 62),
    ]
    boards = elimination_boards(rising, "SOLO0")
    legs, states = boards["TALON"]
    assert check("rising ladder", legs, states).arbitrage

    falling = [
        _market("SOLO0_ELIM_TALON_GT0", 60, 62),
        _market("SOLO0_ELIM_TALON_GT1", 30, 32),
        _market("SOLO0_ELIM_TALON_GT2", 10, 12),
    ]
    legs, states = elimination_boards(falling, "SOLO0")["TALON"]
    assert not check("falling ladder", legs, states).arbitrage


def test_a_book_that_cannot_be_traded_is_not_evidence_about_prices():
    """The filter that separates a finding from an artefact.

    A settled match keeps its last touch forever, and this venue opens every
    contract at the midpoint of its range on purpose, so an unfiltered audit
    reads a market that has not opened yet as a 3.5 unit arbitrage. Measured
    on a live board at ten seconds: ten winner legs quoted near 0.50, summing
    to 4.54 against a partition that must sum to one.
    """
    closed = [
        _market("SOLO0_WIN_A", 45, 50, status="closed"),
        _market("SOLO0_WIN_B", 45, 50, status="closed"),
        _market("SOLO0_WIN_C", 45, 50, status="closed"),
    ]
    legs, states = winner_board(closed, "SOLO0")
    assert legs == []
    # And with no legs there is nothing to conclude, rather than a pass.
    assert check("closed board", legs, states).detail == "fewer than 3 legs"


def test_one_sided_quotes_are_left_out_rather_than_completed():
    """Inventing the missing side would manufacture the consistency under test."""
    half = [
        _market("SOLO0_WIN_A", 30, None),
        _market("SOLO0_WIN_B", None, 36),
        _market("SOLO0_WIN_C", 30, 36),
    ]
    legs, states = winner_board(half, "SOLO0")
    assert [leg.ticker for leg in legs] == ["SOLO0_WIN_C"]
    # The states still name the whole field, because a leg being unquoted does
    # not mean its outcome cannot happen.
    assert len(states) == 3


def test_the_top_ladder_is_read_over_finishing_places():
    markets = [
        _market("SOLO0_TOP2_WISP", 20, 24),
        _market("SOLO0_TOP3_WISP", 30, 34),
        _market("SOLO0_TOP4_WISP", 40, 44),
    ]
    legs, states = top_boards(markets, "SOLO0", field_size=10)["WISP"]
    assert len(legs) == 3
    assert states == list(range(1, 11))
    assert not check("top ladder", legs, states).arbitrage


def test_fewer_than_three_legs_is_reported_rather_than_passed():
    """Two legs can always be priced by some distribution, so a pass is arithmetic."""
    result = check("thin", [Leg("A", 0.1, 0.2, lambda s: True)], ["x"])
    assert result.feasible
    assert result.detail == "fewer than 3 legs"
