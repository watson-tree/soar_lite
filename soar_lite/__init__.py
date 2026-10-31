"""soar_lite: first-order n-path descent with stall restarts.

The public face is deliberate and small. The two plugin contracts
(`LocalSolver`, `GlobalModel`) and the data types they exchange are re-exported
from the engine so callers can write their own methods and restart-point models
against them; the concrete implementations (`DiagonalPreconditioner`,
`UniformGlobalModel`, `BLRGlobalModel`) are re-exported for convenience.
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

from soar_lite.engine import (
    Bounds,
    EvalF,
    EvalFG,
    GlobalModel,
    LocalSolver,
    NPathResult,
    NPathStatus,
    PathPoint,
    Verdict,
    n_path_schedule,
)
from soar_lite.global_model import UniformGlobalModel
from soar_lite.local_solver import DiagonalPreconditioner
from soar_lite.spherical_linear import (
    AcquisitionFn,
    BLRGlobalModel,
    DrawFn,
    IdentityProjection,
    MinimizeFn,
    Projection,
    SphereProjection,
)

__all__ = [
    "AcquisitionFn",
    "BLRGlobalModel",
    "Bounds",
    "DiagonalPreconditioner",
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
    "SphereProjection",
    "UniformGlobalModel",
    "Verdict",
    "n_path_schedule",
]
