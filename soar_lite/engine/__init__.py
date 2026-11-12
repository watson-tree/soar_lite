"""The engine package: the contracts a plugin is written against, and the machine.

This package is two modules behind one facade.
`soar_lite.engine.contracts` is what a plugin author reads: the two plugin
contracts (`LocalSolver`, `GlobalModel`), the data they exchange (`Point`,
`ValuePoint`, `PathPoint`, `Work`, `NPathResult`, `NPathStatus`), the
evaluation-seam aliases, and `pointwise`. `soar_lite.engine.schedule` is the
machine: the `n_path_schedule` loop and the four private holders a run's state
lives in.
This module re-exports the public half, so `from soar_lite.engine import
LocalSolver` names the contract without a caller knowing which file holds it.

The dependency direction is one-way - plugins import these types, never the
reverse:

    engine/contracts.py   <- LocalSolver, GlobalModel, Point, ValuePoint, PathPoint
    engine/schedule.py    <- n_path_schedule and the four holders
    domain.py             <- Domain, Bounds (the box; a leaf)
    local_solver.py       <- DiagonalPreconditioner (implements LocalSolver)
    global_model/         <- SampleGlobalModel, BLRGlobalModel and its projections

The two concrete defaults are bound lazily inside `n_path_schedule`, which keeps
that graph acyclic. The engine is first-order (value and gradient),
coordinate-free, and owns the flock, the domain and its box test, the shared horizon
and the restart cadence; the local solver owns direction, step size, acceptance
and its step's enumeration, and the global model only where a restart lands.

Restarting is two plugin seams and one flag, not a third plugin: `_Paths` marks a
path stalled when its method gave up or its point was not usable,
`restart_stalled` says whether such a path is re-drawn, and `GlobalModel.propose`
says where.
"""

from soar_lite.engine.contracts import (
    EvalF,
    EvalFG,
    GlobalModel,
    LocalSolver,
    NPathResult,
    NPathStatus,
    PathPoint,
    Point,
    ServeTrials,
    ValuePoint,
    Work,
    pointwise,
)
from soar_lite.engine.schedule import n_path_schedule

__all__ = [
    "EvalF",
    "EvalFG",
    "GlobalModel",
    "LocalSolver",
    "NPathResult",
    "NPathStatus",
    "PathPoint",
    "Point",
    "ServeTrials",
    "ValuePoint",
    "Work",
    "n_path_schedule",
    "pointwise",
]
