"""Every reference a docstring cites is one the reference list already carries.

The full entries live in one place - the `## References` list in
`docs/site/explanation/how-it-works.md` - and a docstring carries only a locator
(first author, year, venue). That split is what makes the list reviewable: one
surface holds the author lists, page ranges and URLs, and nothing else restates
them. This test keeps the split honest: a docstring may not cite a paper the
list never states.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "soar_lite"
PAGE = ROOT / "docs" / "site" / "explanation" / "how-it-works.md"

# `- Mathesen, L., Pedrielli, G., ... (2021).` and `- Conn et al. (2000), ...`
ENTRY = re.compile(r"^-\s+(?P<surname>[A-Z][A-Za-z'\-]+)")
YEAR = re.compile(r"\((?P<year>\d{4})\)")


def _key(line: str) -> tuple[str, str] | None:
    """Return the (surname, year) a reference entry names, or None."""
    entry = ENTRY.match(line.strip())
    year = YEAR.search(line)
    if entry is None or year is None:
        return None
    return entry.group("surname"), year.group("year")


def _page_entries() -> list[str]:
    """Return the reference list's entry lines, stopping at the next heading."""
    entries: list[str] = []
    inside = False
    for line in PAGE.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            inside = line.strip() == "## References"
            continue
        if inside and line.startswith("- "):
            entries.append(line)
    assert entries, f"no entries parsed out of {PAGE}'s reference list"
    return entries


def _docstring_citations() -> list[tuple[Path, int, str, str]]:
    """Collect (file, line, surname, year) for every citation in a docstring."""
    found: list[tuple[Path, int, str, str]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        section_indent: int | None = None
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            if section_indent is not None and indent <= section_indent:
                section_indent = None  # left the section
            if line.strip() == "References:":
                section_indent = indent
                continue
            if section_indent is None or not line.strip().startswith("- "):
                continue
            key = _key(line)
            assert key is not None, (
                f"{path}:{number}: unparsable reference {line.strip()!r}"
            )
            found.append((path, number, *key))
    return found


def test_no_docstring_cites_a_reference_the_list_lacks() -> None:
    """A docstring locator resolves against the one list, or the test fails."""
    listed = {key for key in map(_key, _page_entries()) if key is not None}
    missing = [
        f"{path}:{number}: {surname} ({year})"
        for path, number, surname, year in _docstring_citations()
        if (surname, year) not in listed
    ]
    assert not missing, (
        f"cited in a docstring but absent from the list in {PAGE.name}:\n  "
        + "\n  ".join(missing)
    )


def test_the_list_itself_has_no_duplicate_entry() -> None:
    """One paper, one entry: a second copy is a second thing to correct."""
    keys = [key for key in map(_key, _page_entries()) if key is not None]
    assert len(keys) == len(set(keys)), f"duplicate entries in {PAGE.name}: {keys}"
