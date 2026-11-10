"""The tuning records are inputs: written by a sweep, joined to the roster by name.

A pick is ``grid[argmin(fin)]``, so the file proves itself - these tests check
the record against its own rule (the pick really is the argmin of the curve
beside it) and against the roster (a rename reads as a missing entry plus an
orphaned one). The budget guard is exercised too, since it is the only thing
between a quick sweep and a committed record.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.harness import panels, tuning


def _artifact() -> dict[str, Any]:
    """Read the committed adam record as the file it is."""
    text = (tuning.TUNING / "adam.json").read_text(encoding="utf-8")
    return json.loads(text)


def _curves() -> dict[str, dict[str, Any]]:
    """Every problem's recorded entry: its pick and the curve it came from."""
    return _artifact()["problems"]


def _roster() -> set[str]:
    """Every problem the harness builds."""
    return {p.name for p in panels.every_problem()}


def test_every_problem_has_a_pick() -> None:
    """A new problem cannot reach a panel at a rate nobody measured."""
    missing = _roster() - set(tuning.picks("adam"))
    assert not missing, f"no adam pick for {sorted(missing)}; run --lr-sweep"


def test_no_pick_names_a_problem_the_harness_does_not_build() -> None:
    """A rename leaves the old name behind here; that half is the visible one."""
    stale = set(tuning.picks("adam")) - _roster()
    assert not stale, f"stale adam picks: {sorted(stale)}"


def test_each_pick_is_the_argmin_of_the_curve_beside_it() -> None:
    """The pick is re-derivable; a hand-edited number would part from its curve."""
    grid = tuning.grid("adam")
    for name, entry in _curves().items():
        fins = entry["fin"]
        assert entry["lr"] == grid[fins.index(min(fins))], name


def test_every_curve_has_one_point_per_rate() -> None:
    """The curve is aligned with the grid, so an index means the same in both."""
    n_rates = len(tuning.grid("adam"))
    for name, entry in _curves().items():
        assert len(entry["fin"]) == n_rates, name


def test_the_grid_is_ascending_and_positive() -> None:
    """The sweep plots it on a log axis, and the tie-break reads its order."""
    grid = tuning.grid("adam")
    assert grid == sorted(grid)
    assert all(lr > 0 for lr in grid)


def test_an_unknown_problem_raises_with_the_refresh_command() -> None:
    """The miss has to say how to fix itself, since there is no fallback."""
    with pytest.raises(KeyError, match="--lr-sweep"):
        tuning.pick("adam", "no_such_problem")


@pytest.fixture
def record_at_two_seeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand a temp dir in for the tuning tree, holding one 2-seed record."""
    monkeypatch.setattr(tuning, "TUNING", tmp_path)
    record = {"method": "adam", "grid": [1e-2], "seeds": 2, "max_batches": 3000}
    (tmp_path / "adam.json").write_text(json.dumps(record), encoding="utf-8")


def test_a_sweep_at_another_budget_is_refused(
    record_at_two_seeds: None,  # noqa: ARG001 - the fixture writes the record
) -> None:
    """A quick sweep would replace the picks with 1-seed ones."""
    with pytest.raises(SystemExit, match="budget conflict"):
        tuning.refuse_other_budget("adam", seeds=1, max_batches=3000)


def test_a_sweep_at_the_recorded_budget_is_allowed(
    record_at_two_seeds: None,  # noqa: ARG001 - the fixture writes the record
) -> None:
    """The guard is about the budget, not about refusing every rerun."""
    assert tuning.refuse_other_budget("adam", seeds=2, max_batches=3000) is None
