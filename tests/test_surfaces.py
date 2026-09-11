"""Two front ends over one exchange, and what separates them.

`/` is the exchange as a trader meets it. `/desk` is the same exchange plus
the controls that rebuild and measure it, which is a different job rather than
a power user's version of the same one.

The split is a **product** boundary, not a security one, and these tests are
written to say which is which. What stops a trader rebuilding the market is
the operator token the server checks on every control. What the split stops is
a trader being offered a lever that would refuse them, and being left to guess
whether they did something wrong.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# The live-server fixture, reused rather than rebuilt. It starts a market and
# runs it until it has actually traded, which costs several seconds, and two
# copies of that in one suite is a cost paid for nothing: these tests ask the
# same server different questions.
from tests.test_dashboard import client  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "dashboard" / "static" / "js" / "main.js"
INDEX = ROOT / "dashboard" / "static" / "index.html"

OPERATOR_ONLY = ("players", "research", "lab")
EVERYONE = ("markets", "trade", "portfolio")


@pytest.fixture(scope="module")
def source() -> str:
    return MAIN.read_text(encoding="utf-8")


def test_both_paths_serve_the_page(client):
    """One file, two routes. The surface is read from the path, not the file.

    Serving a second copy would mean the header, the socket, the watchlist and
    the order ticket existed twice, and a fix to one of them would land in one
    place. The screens that differ are chosen at runtime instead.
    """
    for path in ("/", "/desk"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert "<div class=\"nav-links\" id=\"nav\"></div>" in response.text


def test_the_rail_is_not_written_into_the_page(client):
    """It is painted from the view registry, so the two cannot disagree.

    A rail written in HTML and a registry written in JavaScript are two
    statements of the same fact, and the failure mode is a button that renders
    and goes nowhere.
    """
    markup = INDEX.read_text(encoding="utf-8")
    assert 'data-view=' not in markup
    assert 'id="nav"' in markup


def test_the_trader_registry_offers_no_operator_screen(source):
    """The public surface mounts three views and knows nothing of the others."""
    public = re.search(r"const PUBLIC_VIEWS = \{([^}]*)\}", source)
    assert public is not None, "the public registry has been renamed"
    named = {name.strip() for name in public.group(1).split(",") if name.strip()}
    assert named == set(EVERYONE)
    for view in OPERATOR_ONLY:
        assert view not in named


def test_the_desk_registry_is_the_public_one_plus_the_operator_screens(source):
    """Built from the public registry rather than listed again.

    Spelled out twice, a view added for traders would silently be missing from
    the operator surface, which is the one place somebody would be looking for
    it.
    """
    desk = re.search(r"\? \{ \.\.\.PUBLIC_VIEWS, ([^}]*)\}", source)
    assert desk is not None, "the desk registry has been restructured"
    extra = {name.strip() for name in desk.group(1).split(",") if name.strip()}
    assert extra == set(OPERATOR_ONLY)


def test_a_view_this_surface_does_not_offer_cannot_be_typed_into_the_url(source):
    """`?view=lab` on the public surface has to fall back, not render the lab.

    The registry is the boundary, so a link, a bookmark and a typed URL all get
    the same answer. Asserted on the source because the alternative is a
    browser test for a one line guard, and the guard is the whole mechanism:
    `readUrl` consults `VIEWS`, and `VIEWS` is whichever registry this surface
    got.
    """
    assert "if (view && view in VIEWS) store.view = view;" in source
    assert "const VIEWS = SURFACE === 'desk'" in source


def test_the_surface_is_decided_by_the_path(source):
    assert "const SURFACE =" in source
    assert "location.pathname" in source


def test_the_directory_is_only_fetched_where_it_is_shown(source):
    """A trader's browser should not poll for a screen it cannot open."""
    assert "if (SURFACE === 'desk') wanted.push(json('/api/players'));" in source


def test_the_players_endpoint_answers_for_this_market(client):
    """The directory is about the market that is running, not about the code.

    The distinction is the point of the endpoint. Anyone can read what a market
    maker does in the repository; what they cannot read is whether one is
    seated here, how many, and what the arbitrageur is doing, which on this
    venue is nothing unless it was switched on.
    """
    payload = client.get("/api/players").json()
    assert payload["groups"], payload
    kinds = [group["kind"] for group in payload["groups"]]
    assert "resident" in kinds
    assert "connected" in kinds

    everyone = [
        player for group in payload["groups"] for player in group["players"]
    ]
    assert everyone
    for player in everyone:
        assert player["id"]
        assert player["name"]
        # Every resident claim has to be traceable to the file that makes it.
        assert "live" in player


def test_the_connected_systems_are_described_as_outside_the_exchange(client):
    """They reach this venue through the published API, which is the claim.

    An audit written by something that imported the exchange would be asking
    the venue whether it agrees with itself, so the directory says plainly that
    these two do not.
    """
    payload = client.get("/api/players").json()
    connected = next(
        group for group in payload["groups"] if group["kind"] == "connected"
    )
    assert len(connected["players"]) >= 2
    sources = " ".join(str(player.get("source", "")) for player in connected["players"])
    assert "connectors/" in sources
