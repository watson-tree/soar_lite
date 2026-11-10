"""The panel's cut is a ruler, and a chain that converged early is not one.

Every curve on a subplot ends at the same x, taken as the shortest arm's extent,
so that no curve outlives the others on an axis it never shared. An arm that
stopped before its batch budget - a converged chain, at machine precision within
a handful of batches on the diagonal quadratic - has no more data, not a shorter
run, and taking its stop as the panel's cut hides every arm still spending its
budget. It is drawn to its own last point instead.

Needs the ``exp`` dependency group (the harness imports matplotlib and optax).

# ruff: noqa: SLF001 - the cut is a plotting rule the figure's own call site
# does not expose, so a test of it has to reach the private names.
"""

# pyright: reportPrivateUsage=false

import os

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
pytest.importorskip("matplotlib")
pytest.importorskip("optax")

import matplotlib.pyplot as plt

from experiments.harness.figures import _cut, _up_to, plot_cohort
from experiments.harness.methods import Trace
from experiments.harness.record import Loaded

BUDGET = 300


def _arm(n: int, *, points_per_batch: int = 1) -> Loaded:
    """One arm's single-seed trace, stopping after ``n`` batches."""
    batches = list(range(1, n + 1))
    return Loaded(
        traces=[
            Trace(
                points=[i * points_per_batch for i in batches],
                batches=batches,
                best=[1.0] * n,
                fun=1.0,
            ),
        ],
        seeds=1,
        max_batches=BUDGET,
    )


def test_a_panel_of_whole_runs_cuts_at_the_shortest_arm() -> None:
    """With every arm completing, the cut is the panel's own shortest extent."""
    loads = [_arm(BUDGET), _arm(BUDGET), _arm(BUDGET, points_per_batch=16)]

    assert _cut(loads, 0) == BUDGET  # points: the wide arm cannot stretch it
    assert _cut(loads, 1) == BUDGET


def test_a_converged_arm_does_not_clip_the_rest() -> None:
    """The arm that stopped at batch 3 has no say in where the others end."""
    loads = [_arm(3), _arm(BUDGET), _arm(BUDGET)]

    assert _cut(loads, 0) == BUDGET
    assert _cut(loads, 1) == BUDGET


def test_a_panel_of_all_early_stops_still_cuts_somewhere() -> None:
    """No ruler left is not a crash: the shortest of the stopped arms decides."""
    assert _cut([_arm(3), _arm(40)], 1) == 3


def test_the_converged_arm_keeps_its_whole_trace() -> None:
    """Cutting at the panel's budget leaves that arm's three points intact."""
    stopped = _arm(3).traces[0]

    kept = _up_to(stopped, "batches", BUDGET)

    assert kept.batches == [1, 2, 3]
    assert kept.best == [1.0, 1.0, 1.0]


def test_the_step_a_run_stopped_on_is_drawn() -> None:
    """The last step is an early run's answer, so the shared grid must sample it."""
    converged = Trace(
        points=[1, 2, 3],
        batches=[1, 2, 3],
        best=[1e4, 1e4, 1e-28],
        fun=1e-28,
    )
    fig, ax = plt.subplots()
    try:
        plot_cohort(
            ax,
            [converged],
            "batches",
            label="converged at 3",
            style={"color": "C0"},
            xmax=BUDGET,
        )
        drawn = ax.get_lines()[0].get_ydata()
    finally:
        plt.close(fig)

    assert np.nanmin(drawn) == pytest.approx(1e-16)  # the floor a best is clipped to
