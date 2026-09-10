"""Colour contrast, as arithmetic against the real stylesheet.

A dark palette drifts under the WCAG thresholds very easily and never looks
obviously wrong while doing it, which is exactly why this is computed rather
than judged. The values are parsed out of ``terminal.css`` itself, so editing a
token to something prettier fails here rather than shipping.

Two distinctions matter, and getting them wrong produces either false alarms or
false comfort:

* **Which surface a token actually appears on.** Checking every colour against
  every background is easy and wrong: control borders never sit on the ladder's
  hover tint, so measuring them there invents a failure that cannot occur.
* **Decoration versus component.** WCAG 1.4.11 covers what is "required to
  identify" a control. A hairline dividing two panels identifies nothing, and
  holding it to 3:1 would mean a terminal ruled in bright grey. The edge of an
  input is a different thing, and gets the full requirement.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CSS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "css" / "terminal.css"

# Text under 18px (and under 14px bold) needs 4.5:1; UI components need 3:1.
# Everything in this terminal is small, so text is held to 4.5 throughout.
SMALL_TEXT = 4.5
COMPONENT = 3.0


def tokens() -> dict[str, str]:
    """Every custom property defined on :root, read from the stylesheet."""
    text = CSS.read_text(encoding="utf-8")
    root = text.split(":root", 1)[1].split("}", 1)[0]
    return {
        name: value
        for name, value in re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})\s*;", root)
    }


def luminance(colour: str) -> float:
    raw = colour.lstrip("#")
    channels = [int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def ratio(foreground: str, background: str) -> float:
    low, high = sorted((luminance(foreground), luminance(background)))
    return (high + 0.05) / (low + 0.05)


# (token, surfaces it is actually drawn on, minimum, what it carries)
CASES = [
    ("ink", ("void", "sunk", "panel", "raised", "hover"), SMALL_TEXT, "body text and prices"),
    ("ink-dim", ("sunk", "panel", "raised", "hover"), SMALL_TEXT, "secondary text and nav"),
    ("ink-faint", ("sunk", "panel", "raised", "hover"), SMALL_TEXT, "table headers, labels"),
    ("amber", ("void", "sunk", "panel", "raised"), SMALL_TEXT, "accent text, active nav"),
    ("up", ("panel", "raised", "hover"), SMALL_TEXT, "positive prices"),
    ("down", ("panel", "raised", "hover"), SMALL_TEXT, "negative prices"),
    ("halt", ("panel", "sunk"), SMALL_TEXT, "halted-session badge"),
    ("closed", ("panel", "sunk"), SMALL_TEXT, "closed-session badge"),
    # Controls live on panels and the sunk header, never on the row-hover tint.
    ("control-edge", ("sunk", "panel"), COMPONENT, "input and button borders"),
    ("amber", ("sunk", "panel", "raised", "hover"), COMPONENT, "the focus ring"),
]


@pytest.mark.parametrize("token,surfaces,minimum,role", CASES)
def test_contrast_meets_wcag(token, surfaces, minimum, role):
    palette = tokens()
    colour = palette[token]
    for surface in surfaces:
        measured = ratio(colour, palette[surface])
        assert measured >= minimum, (
            f"--{token} ({colour}) carries {role} and is {measured:.2f}:1 on "
            f"--{surface} ({palette[surface]}); WCAG needs {minimum}:1"
        )


def test_the_primary_button_label_is_readable():
    """Amber is bright, so its label must be dark; white on it is 2.0:1."""
    palette = tokens()
    assert ratio("#16120a", palette["amber"]) >= SMALL_TEXT
    assert ratio("#ffffff", palette["amber"]) < SMALL_TEXT, (
        "white on amber would pass this test for the wrong reason"
    )


def test_direction_colours_stay_distinguishable():
    """Green and red are the only carriers of direction, so they must not blur
    into each other for anyone reading them side by side in a ladder."""
    palette = tokens()
    assert ratio(palette["up"], palette["down"]) >= 1.5


def test_every_palette_token_is_a_six_digit_hex():
    """Keeps the parser honest: a token in another notation would be silently
    skipped, and skipping is how a failing colour passes."""
    text = CSS.read_text(encoding="utf-8")
    root = text.split(":root", 1)[1].split("}", 1)[0]
    declared = set(re.findall(r"--([\w-]+):", root))
    parsed = set(tokens())
    non_colour = {"mono", "display", "rail", "tick", "amber-sunk", "up-sunk", "down-sunk"}

    # The type scale lives in the same block, under a `t-` namespace. Adding
    # those seven names to `non_colour` would widen the very hole this test
    # exists to close, so they are checked rather than skipped, just against a
    # different rule. Every one has to be a whole-pixel length, which is what
    # stops the half-pixel steps creeping back: the stylesheet used to carry
    # 8.5, 9.5, 10.5, 11.5 and 13.5px, and a half-pixel difference is not
    # hierarchy a reader can perceive, only noise they cannot name.
    type_tokens = {name for name in declared if name.startswith("t-")}
    assert type_tokens, "the type scale has gone from :root"
    for name in sorted(type_tokens):
        value = re.search(rf"--{name}:\s*([^;]+);", root).group(1).strip()
        assert re.fullmatch(r"\d+px", value), (
            f"--{name} is {value!r}; the scale is whole pixels only"
        )

    assert declared - parsed - non_colour - type_tokens == set()


def test_no_text_is_smaller_than_the_scale_allows():
    """Ten pixels is the floor, and it is a floor rather than a target.

    The stylesheet reached 71 font-size declarations carrying 19 distinct
    values, nine of them below 12px and the smallest at 8px. Measured in the
    browser on the static shell alone, sixteen elements rendered under 12px,
    including the primary navigation at 11px and every header stat label at
    8.5px, while the largest text on the page was the 15px brand. An interface
    whose entire type range is 8 to 15px has no hierarchy available to it:
    everything arrives at the reader with the same weight, and half of it is
    too small to read without leaning in.

    Density on a trading screen comes from tight spacing and hairline rules,
    not from shrinking the text until it fits.
    """
    text = CSS.read_text(encoding="utf-8")
    sizes = [float(v) for v in re.findall(r"font-size:\s*([\d.]+)px", text)]
    assert sizes, "no font sizes found; the parser has drifted from the file"

    too_small = sorted({s for s in sizes if s < 10})
    assert not too_small, f"text below the 10px floor: {too_small}"

    fractional = sorted({s for s in sizes if s != int(s)})
    assert not fractional, (
        f"half-pixel sizes are nudges rather than decisions: {fractional}"
    )


def test_the_figures_outrank_their_labels():
    """A number a person came to read must beat the caption describing it.

    Both of these led by less than half a pixel over their own label. The
    header stat was 12px against an 8.5px caption, and in the market rail the
    price was 11.5px against a 9px category tag, so a list a person scans for
    prices gave the price no more presence than the word 'future' beside it.
    """
    text = CSS.read_text(encoding="utf-8")

    def size_of(selector: str) -> str:
        block = text.split(selector, 1)[1].split("}", 1)[0]
        found = re.search(r"font-size:\s*([^;]+);", block)
        assert found, f"{selector} no longer sets a font size"
        return found.group(1).strip()

    # Both resolve through the scale rather than a literal, which is the point:
    # a figure that is sized by hand drifts away from its caption again.
    assert size_of(".stat b {") == "var(--t-lg)"
    assert size_of(".watch .px {") == "var(--t-lg)"


def test_spacing_lands_on_a_grid():
    """Padding, gap and margin came in seventeen distinct values.

    1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 24, 56 for padding, and
    twelve more for gap. The odd numbers are the tell: 7, 9, 11 and 13 are
    nudges made one at a time, and a reader registers the absence of rhythm
    without being able to name it. Snapped onto 2/4/6/8/10/12/16/24, which
    leaves the two deliberate one-off indents alone.
    """
    text = CSS.read_text(encoding="utf-8")
    allowed = {1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 39, 56, 156}
    seen: dict[int, str] = {}
    for match in re.finditer(r"\b(padding|gap|margin)(-[a-z]+)?\s*:([^;}]*)", text):
        for value in re.findall(r"(\d+)px", match.group(3)):
            seen.setdefault(int(value), match.group(0)[:48])

    stray = {v: seen[v] for v in sorted(set(seen) - allowed)}
    assert not stray, f"spacing off the grid: {stray}"


def test_the_depth_bars_are_scaled_and_not_widthed():
    """A percentage width on an absolutely positioned bar measures the wrong box.

    The ladder's depth bars sit either side of a 78px price column, so each owns
    `50% - 39px` of the row. They were sized with an inline `width: N%`, which
    resolves against the whole containing block instead, so every bar was drawn
    against a reference nearly twice its own space. Measured: a 46% ask bar
    rendered 307px into 295px and pushed the row 12px past its panel; at full
    depth it would have overrun by 373px.

    Scaling also puts the animation on the one property that composites.
    Transitioning `width` relayouts the row on every tick, twenty times a
    second, for every level in the book.
    """
    css = CSS.read_text(encoding="utf-8")
    bar = css.split(".lad-row .bar {", 1)[1].split("}", 1)[0]
    assert "width: calc(50% - 39px)" in bar, "the bar no longer spans its own half"
    assert "transition: transform" in bar, "a width transition relayouts the ladder"

    views = CSS.parent.parent / "js" / "views.js"
    emitted = views.read_text(encoding="utf-8")
    assert "class=\"bar bid\" style=\"transform:scaleX(" in emitted
    assert "class=\"bar ask\" style=\"transform:scaleX(" in emitted


def test_flexible_grid_columns_can_actually_shrink():
    """`1fr` is a suggestion until the child is told it may shrink.

    A grid child defaults to `min-width: auto`, which refuses to go below its
    own min-content width, so a `1fr 1fr` row stops being two equal columns the
    moment either holds text that will not wrap small enough. Measured in the
    ticket: 348px of content in a 304px panel, 44px hanging off the edge.
    """
    css = CSS.read_text(encoding="utf-8")
    for selector in (".row2 > *", ".lad-row > *"):
        block = css.split(selector, 1)
        assert len(block) > 1, f"{selector} has gone"
        assert "min-width: 0" in block[1].split("}", 1)[0], (
            f"{selector} can be pushed past its container again"
        )


def test_the_interface_carries_no_em_dashes():
    """A house style rule, held where it is easy to break it again.

    Twelve of these were sitting in the front end after the docs had been swept
    of them, because they were HTML entities rather than literal characters and
    a search for the character found nothing. Most were doing a comma's job.
    """
    root = CSS.parent.parent
    offenders = {}
    for path in sorted(list((root / "js").glob("*.js")) + [root / "index.html"]):
        if "vendor" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        hits = re.findall(r"&mdash;|—", text)
        if hits:
            offenders[path.name] = len(hits)
    assert not offenders, f"em dashes in the interface: {offenders}"


# The three shapes the stand-in takes. Built from pieces rather than written
# out, because a literal one here would be a violation of the rule this test
# exists to enforce, and the file would flag itself.
_GAP = " " + "-" * 2 + " "
_STANDIN = (
    re.compile(re.escape(_GAP)),
    re.compile(re.escape(_GAP.rstrip()) + "$"),
    re.compile(r"^\s*(?:#\s*)?" + re.escape(_GAP.lstrip())),
)
_DIVIDER = re.compile("-" * 4)
_SWEPT = ("python", "tests", "dashboard", "experiments", "docs")
_ALSO = ("README.md", "AGENTS.md", "CONTRIBUTING.md")
_SUFFIXES = {".py", ".md", ".js", ".css", ".html"}


def _is_divider(line: str) -> bool:
    """A dash drawing a rule across the page, not punctuating a sentence.

    Two forms, and both are layout. The ordinary one rules off a section and is
    recognised by its run of hyphens. The second is the same idiom with a label
    too long to leave room for the run: `match_book.py` carries a series of
    twelve headings, and the three with the longest labels trail two hyphens,
    three, and none at all. A comment that *opens* with the pair is heading its
    section either way, so it is read as one.
    """
    return bool(_DIVIDER.search(line)) or line.strip().startswith("# --")


def _house_style_files():
    """Everything this repository wrote, and nothing it merely vendored."""
    root = Path(__file__).resolve().parents[1]
    seen = []
    for name in _SWEPT:
        for path in sorted((root / name).rglob("*")):
            if path.suffix not in _SUFFIXES or not path.is_file():
                continue
            parts = set(path.parts)
            if "__pycache__" in parts or "vendor" in parts or ".agents" in parts:
                continue
            seen.append(path)
    seen.extend(root / name for name in _ALSO if (root / name).exists())
    # This file holds the patterns it searches for, so it cannot search itself.
    return [p for p in seen if p != Path(__file__).resolve()]


def test_nothing_in_the_repository_stands_in_for_an_em_dash():
    """The same house style rule, over everything rather than over the front end.

    The interface was swept first and the test above only ever watched the
    interface, so the rule was enforced across four thousand lines of it and
    unenforced across seventy thousand of everything else. Measured when the
    scope was widened: **1,022** spaced double hyphens doing a dash's job, in
    83 files, against zero em dashes. The character had been kept out and the
    habit had not.

    Three shapes, because a search for the obvious one finds about nine in ten:
    the stand-in also turns up at the end of a line, and at the start of the
    line that continues it, whenever a paragraph was rewrapped around it. That
    last shape is the one that has to be told apart from a section heading, and
    `_is_divider` is where that judgement lives.

    Written to name the file and the line, because a bare count tells you the
    rule broke and not where, and this is a rule that breaks a paragraph at a
    time.
    """
    offenders = []
    for path in _house_style_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # pragma: no cover - not a text file after all
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if _is_divider(line):
                continue
            if any(pattern.search(line) for pattern in _STANDIN):
                offenders.append(f"{path.name}:{number}")
    assert not offenders, (
        f"{len(offenders)} double hyphens standing in for a dash: "
        f"{offenders[:12]}"
    )


def test_the_narrow_layout_reaches_every_view():
    """Five views, and two of them were off the side of a phone.

    The rail's group heading is a device for a vertical list. Laid out
    horizontally it becomes a dead tab that reads as a disabled view, and it
    took 90px of a 375px screen: measured, the tab row wanted 537px, which put
    Lab entirely off-screen and left one pixel of Research showing, on a
    scroller with no visible affordance. The heading is dropped at that width
    and the tabs tightened, which seats all five.
    """
    css = CSS.read_text(encoding="utf-8")
    narrow = css.split("@media (max-width: 860px)", 1)
    assert len(narrow) > 1, "the narrow breakpoint has gone"
    block = narrow[1]
    assert ".nav-group { display: none; }" in block, (
        "the group heading is back in the tab row"
    )
    assert "min-height: 44px" in block, "touch targets are unconstrained again"
