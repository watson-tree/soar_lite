"""The Adam lr table is a claim about every problem, and a miss must be refused.

`ADAM_LR` is keyed by problem name - a string this table and `problems.build`
agree on by hand. A rename therefore reads as a brand-new problem: the arm key
becomes `adam-lr0.01`, a figure is drawn at a lr nobody grid-tuned, and the only
trace of it is the number in a file name. The table had a 1e-2 fallback, which
is exactly what made the rename invisible.

Needs the ``exp`` dependency group (the harness imports matplotlib and optax), so
a plain ``dev`` test run skips this file instead of failing it.
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
pytest.importorskip("matplotlib")
pytest.importorskip("optax")

from experiments import compare_baselines as harness
from experiments.harness import problems, record


@pytest.fixture(autouse=True)
def _isolated_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the refusal away from the real results and figures trees."""
    monkeypatch.setattr(record, "RECORD_RESULTS", tmp_path / "results")
    monkeypatch.setattr(record, "RECORD_FIGURES", tmp_path / "figures")


def test_every_problem_the_harness_builds_has_a_tuned_lr() -> None:
    """`build` is the roster; the table is a claim about every entry in it."""
    missing = [
        prob.name for prob in problems.build("all") if prob.name not in harness.ADAM_LR
    ]

    assert not missing


def test_a_problem_with_no_tuned_lr_is_refused_before_anything_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Dropping one entry is what a rename looks like from the table's side."""
    monkeypatch.delitem(harness.ADAM_LR, "rastrigin_d100")

    with pytest.raises(SystemExit) as caught:
        harness.main(["--plot-only"])

    assert "no tuned Adam lr for rastrigin_d100" in str(caught.value)
    assert not (tmp_path / "figures").exists()
