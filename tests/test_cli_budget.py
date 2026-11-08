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
    """Keep the record's roots out of the repo, and out of the rest of the session."""
    monkeypatch.setattr(harness, "_RECORD_RESULTS", tmp_path / "results")
    monkeypatch.setattr(harness, "_RECORD_FIGURES", tmp_path / "figures")


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

    def _record(_roots: harness.Roots, n_seeds: int, max_steps: int) -> None:
        handed.update(seeds=n_seeds, max_steps=max_steps)

    monkeypatch.setattr(harness, "run_lr_sweep", _record)
    monkeypatch.setattr(harness, "run_invariance", _record)
    monkeypatch.setattr(
        harness,
        "run_panel",
        lambda _roots, _group, n_seeds, max_steps, **_rest: _record(
            _roots,
            n_seeds,
            max_steps,
        ),
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
    monkeypatch.setattr(
        harness,
        "plot_panel",
        lambda roots, panel: drawn.append(roots.panel_dir(panel)),
    )

    runs = [harness.main(["--groups", "smooth", "--out", "demo"]) for _ in range(2)]

    assert (
        runs[0]
        == runs[1]
        == harness.Roots(
            results=tmp_path / "results" / "demo",
            figures=tmp_path / "figures" / "demo",
        )
    )
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

    assert not (tmp_path / "results").exists()


def test_quick_and_out_are_two_answers_to_one_question(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two namespaces for one run is refused, not silently resolved."""
    monkeypatch.setattr(harness, "run_panel", lambda *_args, **_rest: None)
    monkeypatch.setattr(harness, "plot_panel", lambda *_args: None)

    with pytest.raises(SystemExit, match="pick one"):
        harness.main(["--quick", "--out", "demo"])

    assert not (tmp_path / "results").exists()
