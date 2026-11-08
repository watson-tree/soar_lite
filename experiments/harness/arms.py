"""Arm identity, runners and presentation: what to run, and how to draw it.

The arm key spells out the arm's whole configuration
(`soar-sample-np1-mu0-restart-off`), never a path through a default it could
have inherited. An engine default moving therefore changes the key and the file
path: the old trace stays as a record of the old configuration.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

import jax.numpy as jnp

from experiments.harness import problems
from experiments.harness.methods import (
    Trace,
    run_adam,
    run_lbfgs,
    run_soar_lite,
    surrogate_minimize,
)
from soar_lite import (
    BLRGlobalModel,
    DiagonalPreconditioner,
    GlobalModel,
    IdentityProjection,
    SphereProjection,
)

Runner: TypeAlias = Callable[..., Trace]


@dataclass(frozen=True)
class Arm:
    """One curve: what to run, and how to draw it.

    ``key`` is the arm's identity on disk, so it names every attribute the
    runner pins - including the ones the runner could have left to a default.
    ``label`` and ``style`` are presentation only: two panels may draw the same
    key under different labels, and do.
    """

    key: str
    label: str
    runner: Runner
    style: dict[str, Any]


# The preconditioner's own momentum default. Arms that take no knobs read it
# from the class instead of restating it, so a default move re-keys them (a new
# file) rather than silently changing what an existing file means.
DEFAULT_MU = DiagonalPreconditioner().mu


def soar_arm(  # noqa: PLR0913 - one keyword per attribute the key must name
    *,
    n_paths: int,
    mu: float,
    label: str,
    style: dict[str, Any],
    model: str = "sample",
    restart_stalled: bool = False,
) -> Arm:
    """Build one soar-lite arm, its key derived from exactly these attributes."""
    key = (
        f"soar-{model}-np{n_paths}-mu{mu:g}"
        f"-restart-{'on' if restart_stalled else 'off'}"
    )

    def runner(prob: problems.Problem, *, max_batches: int, seed: int) -> Trace:
        return run_soar_lite(
            prob,
            n_paths=n_paths,
            max_batches=max_batches,
            seed=seed,
            restart_stalled=restart_stalled,
            mu=mu,
            model=_model_for(model),
        )

    return Arm(key=key, label=label, runner=runner, style=style)


# Per-problem Adam learning rate, the pick the panels use (see --lr-sweep).
# The picks span over two orders of magnitude across the panels, and lr=1e-4
# costs 52 orders of magnitude on the diagonal quadratic against its pick.
# Tuning is itself budget the y axis never shows -- every trial is more
# objective points served, and it is manual, not adaptive; soar-lite has no
# learning rate at all. Two optima sit at the sweep grid's right edge (the
# rotated quadratic and the oscillator keep improving to lr=2.0). On the
# diagonal quadratic every grid point from 1e-2 up is a floor or a plateau, so
# that pick is one point of a tie, not a strict argmin; on
# rastrigin the floor is flat over 1e-3..2e-1 and 0.5 sits ~5% below it. The
# compartment chain is the one strict interior argmin: its curve falls from
# 5.0 at lr=1e-2 to 1.1e-3 at 1e-1 and back up to 2.8e-2 at 2e-1, so there the
# pick is a real optimum rather than an edge or a tie.
ADAM_LR: dict[str, float] = {
    "oscillator_n20_F20": 2.0,
    "pk_chain_n21_d61": 1e-1,
    "diag_quad_d100_cond10000": 1e-2,
    "rot_quad_d100_cond1e+06": 2.0,
    "rastrigin_d100": 0.5,
}


def adam_lr(prob: problems.Problem) -> float:
    """Return the grid-tuned Adam lr for this problem.

    There is no fallback: `_refuse_untuned_problems` keeps the table a claim
    about the whole roster, so a miss here is a bug in that guard, not a case
    to paper over with a default - the arm key and the sweep's C0 diamond both
    read this number, and neither would say it had been guessed.
    """
    return ADAM_LR[prob.name]


def adam_arm(prob: problems.Problem) -> Arm:
    """Adam at this problem's grid-tuned lr; the lr is part of the identity."""
    lr = adam_lr(prob)
    key = f"adam-lr{lr:g}"

    def runner(p: problems.Problem, *, max_batches: int, seed: int) -> Trace:
        return run_adam(p, lr=lr, max_batches=max_batches, seed=seed)

    return Arm(
        key=key,
        label="Adam (grid lr)",
        runner=runner,
        style={"color": "C3", "lw": 1.5},
    )


LBFGS_ARM = Arm(
    key="lbfgs",
    label="L-BFGS-B",
    runner=run_lbfgs,
    style={"color": "C7", "lw": 1.2, "ls": ":"},
)


def _model_for(model: str) -> Callable[[jnp.ndarray], GlobalModel] | None:
    """Return the restart-model factory a key's model part names.

    ``sample`` is the engine's own uniform draw, so it answers None.
    """
    if model == "sample":
        return None
    projections = {
        "blr-identity": IdentityProjection,
        "blr-sphere": SphereProjection,
    }
    try:
        projection = projections[model]
    except KeyError:
        msg = f"unknown restart model {model!r}"
        raise SystemExit(msg) from None
    return lambda key: BLRGlobalModel(
        projection_factory=projection,
        minimize=surrogate_minimize,
        key=key,
    )
