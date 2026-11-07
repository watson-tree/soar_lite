"""`--quick` decides the budget in one place, for every mode that measures.

The sweep used to read `args.max_steps` directly, so `--quick --lr-sweep` ran
the full grid at 3000 batches while the flag promised a smoke check. The seam
is `_budget`, and `main` is asked what it hands the sweep.

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


@pytest.fixture(autouse=True)
def _isolated_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the smoke-root rebind from leaking into the rest of the session."""
    monkeypatch.setattr(harness, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(harness, "FIGURES", tmp_path / "figures")


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--quick", "--lr-sweep"], (2, 300)),
        (["--lr-sweep"], (2, 3000)),
        (["--quick"], (2, 300)),
        (["--seeds", "3", "--max-steps", "50"], (3, 50)),
    ],
)
def test_quick_decides_the_budget_the_sweep_measures_at(
    argv: list[str],
    expected: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever the mode, one budget: what `--quick` promises or what was asked."""
    handed: dict[str, int] = {}

    def _record(n_seeds: int, max_steps: int) -> None:
        handed.update(seeds=n_seeds, max_steps=max_steps)

    monkeypatch.setattr(harness, "run_lr_sweep", _record)
    monkeypatch.setattr(harness, "run_invariance", _record)
    monkeypatch.setattr(
        harness,
        "run_panel",
        lambda _group, n_seeds, max_steps, **_rest: _record(n_seeds, max_steps),
    )
    monkeypatch.setattr(harness, "plot_panel", lambda *_args: None)

    harness.main(argv)

    assert (handed["seeds"], handed["max_steps"]) == expected
