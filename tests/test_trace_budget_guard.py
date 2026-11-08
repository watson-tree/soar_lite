"""The trace file is a record: a run at another budget must not replace it.

A trace is named ``(problem, arm)`` and the budget it was measured at lives
inside it, so without this guard a typo in ``--seeds`` silently replaces a
20-seed record with a 2-seed one and no later reader can tell.

Only a *missing* file may be written. A file that exists but cannot say what it
holds - truncated JSON, no ``meta``, no budget keys - is refused too, because
"could not read the budget" read as "nothing to guard" is the same overwrite
with a different cause. The one exception is deliberate: a record written while
the flag was ``--max-steps`` still names its budget, so it is read at its own
value instead of becoming a file no run may ever replace.

Needs the ``exp`` dependency group (the harness imports matplotlib and optax),
so a plain ``dev`` test run skips this file instead of failing it.
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


@pytest.fixture
def roots(tmp_path: Path) -> harness.Roots:
    """Stand a temp dir in for the record both roots name."""
    return harness.Roots(results=tmp_path, figures=tmp_path)


def _seed_record(
    root: Path,
    *,
    seeds: int,
    batches: int,
    budget_key: str = "max_batches",
) -> Path:
    """Write a trace file whose meta claims the given budget, and return it."""
    path = root / PROBLEM / f"{ARM}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "problem": PROBLEM,
        "arm": ARM,
        "seeds": seeds,
        budget_key: batches,
    }
    path.write_text(json.dumps({"meta": meta, "seeds": []}))
    return path


def test_a_run_at_another_budget_is_refused_and_writes_nothing(
    tmp_path: Path,
    roots: harness.Roots,
) -> None:
    """The whole point: 2 x 300 does not get to replace 20 x 3000."""
    path = _seed_record(tmp_path, seeds=20, batches=3000)
    before = path.read_text()

    with pytest.raises(SystemExit) as caught:
        harness.refuse_budget_change(roots, PROBLEM, ARM, n_seeds=2, max_batches=300)

    message = str(caught.value)
    assert "20 seeds x 3000 batches" in message
    assert "2 x 300" in message
    assert path.read_text() == before


def test_the_same_budget_is_allowed(
    tmp_path: Path,
    roots: harness.Roots,
) -> None:
    """Re-measuring an arm at its own budget is the documented rerun."""
    _seed_record(tmp_path, seeds=20, batches=3000)

    harness.refuse_budget_change(roots, PROBLEM, ARM, n_seeds=20, max_batches=3000)


def test_no_file_at_all_is_allowed(roots: harness.Roots) -> None:
    """Nothing has measured this arm, so a first run has nothing to protect."""
    harness.refuse_budget_change(roots, PROBLEM, ARM, n_seeds=2, max_batches=300)


@pytest.mark.parametrize(
    "body",
    [
        "{not json",
        '{"seeds": []}',
        '{"meta": {"problem": "some_problem"}, "seeds": []}',
    ],
)
def test_a_file_with_no_readable_budget_is_refused(
    tmp_path: Path,
    roots: harness.Roots,
    body: str,
) -> None:
    """Truncated, meta-less, or budget-less: none of them is "nothing to guard"."""
    path = tmp_path / PROBLEM / f"{ARM}.json"
    path.parent.mkdir(parents=True)
    path.write_text(body)

    with pytest.raises(SystemExit, match="unreadable budget"):
        harness.refuse_budget_change(roots, PROBLEM, ARM, n_seeds=2, max_batches=300)

    assert path.read_text() == body


def test_a_record_from_before_the_rename_is_read_at_its_own_budget(
    tmp_path: Path,
    roots: harness.Roots,
) -> None:
    """An old trace still says what it measured, so its own budget may rerun it."""
    _seed_record(tmp_path, seeds=20, batches=3000, budget_key="max_steps")

    harness.refuse_budget_change(roots, PROBLEM, ARM, n_seeds=20, max_batches=3000)


def test_a_record_from_before_the_rename_is_still_guarded(
    tmp_path: Path,
    roots: harness.Roots,
) -> None:
    """The rename must not turn every old record into a file nothing can hit."""
    _seed_record(tmp_path, seeds=20, batches=3000, budget_key="max_steps")

    with pytest.raises(SystemExit, match="20 seeds x 3000 batches"):
        harness.refuse_budget_change(roots, PROBLEM, ARM, n_seeds=2, max_batches=300)
