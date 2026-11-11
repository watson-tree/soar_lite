"""Arm identity, runners and presentation: what to run, and how to draw it.

The arm key spells out the arm's whole configuration
(`soar_lite-sample-n_paths1-mu0-restart-off`), never a path through a default it could
have inherited. An engine default moving therefore changes the key and the file
path: the old trace stays as a record of the old configuration.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

import jax.numpy as jnp

from experiments.harness import problems, tuning
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
        f"soar_lite-{model}-n_paths{n_paths}-mu{mu:g}"
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


def adam_arm(prob: problems.Problem) -> Arm:
    """Adam at this problem's tuned lr (`tuning/adam.json`); the lr is in the key."""
    lr = tuning.pick("adam", prob.name)
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
