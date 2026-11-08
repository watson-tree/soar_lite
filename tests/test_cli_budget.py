"""Where the harness measures, and how much: `--quick`, `--out`, `--seeds`.

The sweep used to read `args.max_steps` directly, so `--quick --lr-sweep` ran
the full grid at 3000 batches while the flag promised a smoke check; it also
hard-coded 2 seeds, so `--seeds` moved every number except the sweep's. The
seam is `_budget`, and `main` is asked what it hands the sweep.

`--out` is the other half of the same idea: it decides which roots a run writes
into, so a demo cannot overwrite the record. Both are checked here rather than
measured - every runner below is replaced by a recorder.

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
    """Keep the root rebind from leaking, or writing, into the rest of the session."""
    monkeypatch.setattr(harness, "_BASE_RESULTS", tmp_path / "results")
    monkeypatch.setattr(harness, "_BASE_FIGURES", tmp_path / "figures")
    monkeypatch.setattr(harness, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(harness, "FIGURES", tmp_path / "figures")


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--quick", "--lr-sweep"], (2, 300)),
        (["--lr-sweep"], (20, 3000)),
        (["--lr-sweep", "--seeds", "2"], (2, 3000)),
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


def test_out_moves_both_roots_and_survives_a_second_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both roots follow the name, and a re-run cell lands in the same place."""
    drawn: list[Path] = []
    monkeypatch.setattr(harness, "run_panel", lambda *_args, **_rest: None)
    monkeypatch.setattr(harness, "plot_panel", lambda _panel, out: drawn.append(out))

    for _ in range(2):
        harness.main(["--groups", "smooth", "--out", "demo"])

    assert tmp_path / "results" / "demo" == harness.RESULTS
    assert [tmp_path / "figures" / "demo" / "smooth"] * 2 == drawn


@pytest.mark.parametrize("out", ["a/b", ".", ".."])
def test_out_refuses_a_path(
    out: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prefix is one directory name; anything else would escape both roots."""
    monkeypatch.setattr(harness, "plot_panel", lambda *_args: None)

    with pytest.raises(SystemExit, match="one directory name"):
        harness.main(["--groups", "smooth", "--out", out])

    assert tmp_path / "results" == harness.RESULTS


def test_quick_and_out_are_two_answers_to_one_question(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two namespaces for one run is refused, not silently resolved."""
    monkeypatch.setattr(harness, "run_panel", lambda *_args, **_rest: None)
    monkeypatch.setattr(harness, "plot_panel", lambda *_args: None)

    with pytest.raises(SystemExit, match="pick one"):
        harness.main(["--quick", "--out", "demo"])

    assert tmp_path / "results" == harness.RESULTS
