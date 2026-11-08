"""A run that never sees a finite value records nan, not inf.

`CONTEXT.md` gives nan to "no answer" and leaves inf to mean a value an
objective actually took, so the three drivers have to agree on what a trace
holds when nothing finite arrives. The mirror half matters just as much: a
nan-seeded best-so-far must not blind the arms that track one, since `v < nan`
is False and `jnp.minimum` hands the nan back. Both halves are pinned here -
an objective that answers nan at every point, and a finite one that would go
nan for the rest of its run if the sentinel leaked into the comparison.

Needs the ``exp`` dependency group: the harness imports matplotlib and optax.
"""

import math
import os
from functools import partial

import jax.numpy as jnp
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
pytest.importorskip("matplotlib")
pytest.importorskip("optax")

from matplotlib import pyplot as plt

from experiments.harness import figures, problems
from experiments.harness.methods import Trace, run_adam, run_lbfgs, run_soar_lite

BUDGET = 4

RUNS = {
    "soar-lite": partial(
        run_soar_lite,
        n_paths=1,
        max_batches=BUDGET,
        seed=0,
        restart_stalled=False,
    ),
    "adam": partial(run_adam, max_batches=BUDGET, seed=0, lr=0.1),
    "lbfgs": partial(run_lbfgs, max_batches=BUDGET, seed=0),
}


def _every_value_nan(d: int = 3) -> problems.Problem:
    """Build a problem whose objective never answers."""

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        return jnp.full((x.shape[0],), jnp.nan), jnp.zeros_like(x), None

    return problems.Problem(
        name="all_nan",
        d=d,
        bounds=jnp.asarray([[-1.0, 1.0]] * d),
        eval_f_g=eval_f_g,
        f_star=math.nan,
        description="every value is nan",
    )


@pytest.mark.parametrize("label", list(RUNS))
def test_a_run_with_no_finite_value_records_nan(label: str) -> None:
    """No answer is nan, so it cannot be mistaken for a value of +inf."""
    trace: Trace = RUNS[label](_every_value_nan())

    assert trace.best, "a run records one entry per batch whatever it finds"
    assert all(math.isnan(v) for v in trace.best)
    assert math.isnan(trace.fun)


@pytest.mark.parametrize("label", list(RUNS))
def test_a_nan_seed_still_tracks_a_finite_best(label: str) -> None:
    """The sentinel must not cost a healthy run its best-so-far."""
    trace: Trace = RUNS[label](problems.diag_quadratic(d=5, cond=10.0))

    assert all(math.isfinite(v) for v in trace.best)
    assert trace.best == sorted(trace.best, reverse=True)  # best-so-far never rises
    assert math.isfinite(trace.fun)


def test_a_cohort_of_nan_traces_draws_without_a_crash() -> None:
    """Drawing is the one place a nan reaches numpy; it must pass through."""
    trace: Trace = RUNS["adam"](_every_value_nan())
    fig, ax = plt.subplots()
    try:
        figures.plot_cohort(
            ax, [trace], "batches", label="never answers", style={"color": "C0"}
        )
    finally:
        plt.close(fig)
