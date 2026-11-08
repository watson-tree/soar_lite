"""The trace file is a record: a run at another budget must not replace it.

A trace is named ``(problem, arm)`` and the budget it was measured at lives
inside it, so without this guard a typo in ``--seeds`` silently replaces a
20-seed record with a 2-seed one and no later reader can tell. Needs the ``exp``
dependency group (the harness imports matplotlib and optax), so a plain ``dev``
test run skips this file instead of failing it.
"""

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
pytest.importorskip("matplotlib")
pytest.importorskip("optax")

from experiments import compare_baselines as harness

PROBLEM = "some_problem"
ARM = "some-arm"
RECORD = {"seeds": 20, "max_steps": 3000}


@pytest.fixture
def roots(tmp_path: Path) -> harness.Roots:
    """Stand a temp dir in for the record both roots name."""
    return harness.Roots(results=tmp_path, figures=tmp_path)


def _seed_record(root: Path, *, seeds: int, max_steps: int) -> Path:
    """Write a trace file whose meta claims the given budget, and return it."""
    path = root / PROBLEM / f"{ARM}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"problem": PROBLEM, "arm": ARM, "seeds": seeds, "max_steps": max_steps}
    path.write_text(json.dumps({"meta": meta, "seeds": []}))
    return path


def test_a_run_at_another_budget_is_refused_and_writes_nothing(
    tmp_path: Path,
    roots: harness.Roots,
) -> None:
    """The whole point: 2 x 300 does not get to replace 20 x 3000."""
    path = _seed_record(tmp_path, **RECORD)
    before = path.read_text()

    with pytest.raises(SystemExit) as caught:
        harness.refuse_budget_change(roots, PROBLEM, ARM, n_seeds=2, max_steps=300)

    message = str(caught.value)
    assert "20 seeds x 3000 batches" in message
    assert "2 x 300" in message
    assert path.read_text() == before


def test_the_same_budget_is_allowed(
    tmp_path: Path,
    roots: harness.Roots,
) -> None:
    """Re-measuring an arm at its own budget is the documented rerun."""
    _seed_record(tmp_path, **RECORD)

    harness.refuse_budget_change(roots, PROBLEM, ARM, n_seeds=20, max_steps=3000)


def test_nothing_readable_on_disk_is_allowed(
    tmp_path: Path,
    roots: harness.Roots,
) -> None:
    """No record, or one this code cannot read, means there is nothing to guard."""
    harness.refuse_budget_change(roots, PROBLEM, ARM, n_seeds=2, max_steps=300)

    path = tmp_path / PROBLEM / f"{ARM}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    harness.refuse_budget_change(roots, PROBLEM, ARM, n_seeds=2, max_steps=300)
