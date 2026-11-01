"""soar_lite: first-order n-path descent with stall restarts.

The public face is deliberate and small. The two plugin contracts
(`LocalSolver`, `GlobalModel`) and the data types they exchange are re-exported
from the engine so callers can write their own methods and restart-point models
against them; the concrete implementations (`DiagonalPreconditioner`,
`SampleGlobalModel`, `BLRGlobalModel`) are re-exported for convenience.
`n_path_schedule` is the whole algorithm entry point.

Whether a stalled path is re-drawn is the `restart_stalled` knob on
`n_path_schedule`, not a plugin: *when* a path has stopped is the local solver's
own stopping test, and *where* it restarts is the global model's, so the engine
needs no third seam to join them.

Internal helpers (the box/argmin utilities, the run's private state holders) are
not re-exported here on purpose; they are `_`-private in their own modules. A
test or a contributor may still import them, but nothing outside the package is
meant to.
"""

from soar_lite.domain import Bounds, Domain, DrawFn
from soar_lite.engine import (
    EvalF,
    EvalFG,
    GlobalModel,
    LocalSolver,
    NPathResult,
    NPathStatus,
    PathPoint,
    Verdict,
    n_path_schedule,
    pointwise,
)
from soar_lite.global_model import (
    AcquisitionFn,
    BLRGlobalModel,
    IdentityProjection,
    MinimizeFn,
    Projection,
    SampleGlobalModel,
    SphereProjection,
)
from soar_lite.local_solver import DiagonalPreconditioner

__all__ = [
    "AcquisitionFn",
    "BLRGlobalModel",
    "Bounds",
    "DiagonalPreconditioner",
    "Domain",
    "DrawFn",
    "EvalF",
    "EvalFG",
    "GlobalModel",
    "IdentityProjection",
    "LocalSolver",
    "MinimizeFn",
    "NPathResult",
    "NPathStatus",
    "PathPoint",
    "Projection",
    "SampleGlobalModel",
    "SphereProjection",
    "Verdict",
    "n_path_schedule",
    "pointwise",
]
