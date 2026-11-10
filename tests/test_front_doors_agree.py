"""The two front doors are one document with two addresses.

`docs/index.md` is canonical: it is the page mkdocs builds, and the tutorial
links into it for its install section. The root README mirrors it verbatim for
the shared sections, so a fact gets corrected in one place and copied, not
re-invented in the other. Sections only the README needs -- `Results`, `Scope`,
`Documentation` -- are outside the contract.

The badges at the top of the README are GitHub furniture, so they are dropped
before the two openings are compared. What is left of the opening is shared: a
reader arriving from the package index and a reader arriving from the site meet
the same four paragraphs.

Link destinations are the one thing that cannot be copied: `docs/index.md`
addresses its neighbours from inside `docs/`, the README from the root, so the
same sentence carries `explanation/how-it-works.md` in one and
`docs/explanation/how-it-works.md` in the other. Text is compared with the
destinations masked, and their resolution is checked separately: the sentence is
one fact, the two addresses are two.
"""

import difflib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = "docs/index.md"
MIRROR = "README.md"
SHARED = ("Install", "Minimal run", "What's inside")
LINK = re.compile(r"\]\(([^)\s]+)\)")


def _text(path: str) -> str:
    """Read one of the two front doors from the repository root."""
    return (ROOT / path).read_text(encoding="utf-8")


def _sections(text: str) -> dict[str, str]:
    """Split a document into its `## ` sections, keyed by heading.

    The preamble sits under the empty key, so an opening is addressable the same
    way a section is. Deeper headings stay inside the body they appear in: the
    shared sections are flat, and their subheads belong to them.
    """
    sections: dict[str, list[str]] = {"": []}
    heading = ""
    for line in text.splitlines():
        if line.startswith("## "):
            heading = line[3:].strip()
            sections[heading] = []
        else:
            sections[heading].append(line)
    return {name: "\n".join(body).strip("\n") for name, body in sections.items()}


def _opening(text: str) -> str:
    """Return the preamble without the badge lines GitHub renders above it.

    Dropping the badges leaves the blank line that separated them, so runs of
    blank lines collapse: the title's distance from the first paragraph is
    layout, and both pages lay it out as one blank line.
    """
    kept = [
        line for line in _sections(text)[""].splitlines() if not line.startswith("[![")
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip("\n")


def _assert_copied(mirror: str, canonical: str, what: str) -> None:
    """Fail with the difference, not just with a claim that there is one."""
    diff = list(
        difflib.unified_diff(
            _masked(canonical).splitlines(),
            _masked(mirror).splitlines(),
            fromfile=f"{CANONICAL} ({what})",
            tofile=f"{MIRROR} ({what})",
            lineterm="",
        )
    )
    assert not diff, "\n".join(diff)


def _masked(text: str) -> str:
    """Drop every link destination, keeping the link text."""
    return LINK.sub("]", text)


def _slug(heading: str) -> str:
    """Approximate the anchor a heading gets, on either renderer."""
    return re.sub(r"[\s]+", "-", re.sub(r"[^\w\s-]", "", heading.lower()).strip())


def _headings(path: Path) -> set[str]:
    """Every anchor a document defines, at any heading level."""
    return {
        _slug(line.lstrip("#").strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("#")
    }


def test_the_openings_are_the_same() -> None:
    """Both entrances open on the same paragraphs, badges aside."""
    _assert_copied(_opening(_text(MIRROR)), _opening(_text(CANONICAL)), "opening")


@pytest.mark.parametrize("heading", SHARED)
def test_a_shared_section_is_copied_verbatim(heading: str) -> None:
    """A section both pages carry is the canonical text, not a paraphrase."""
    assert heading in _sections(_text(MIRROR)), f"{MIRROR} lacks ## {heading}"
    assert heading in _sections(_text(CANONICAL)), f"{CANONICAL} lacks ## {heading}"
    _assert_copied(
        _sections(_text(MIRROR))[heading],
        _sections(_text(CANONICAL))[heading],
        heading,
    )


@pytest.mark.parametrize("path", [MIRROR, CANONICAL])
def test_every_relative_link_resolves(path: str) -> None:
    """A front door's links reach their file, and their anchor its heading."""
    base = (ROOT / path).parent
    for target in LINK.findall(_text(path)):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        file, _, anchor = target.partition("#")
        resolved = base / file if file else ROOT / path
        assert resolved.exists(), f"{path}: {target} is not a path"
        if anchor and resolved.is_file():
            assert _slug(anchor) in _headings(resolved), (
                f"{path}: {target} points at no heading"
            )
