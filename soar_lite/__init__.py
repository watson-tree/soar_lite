"""soar_lite: first-order n-path descent with restart rules.

The public face is deliberate and small. The two plugin contracts
(`LocalSolver`, `RestartRule`) and the data types they exchange are re-exported
from the engine so callers can write their own methods/rules against them; the
concrete implementations (`DiagonalPreconditioner`, `NoRestartRule`,
`StallRestartRule`) are re-exported for convenience. `n_path_schedule` is the
whole algorithm entry point, and `GlobalModel` is the shape of its restart-point
seam.

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
    RestartRule,
    Verdict,
    n_path_schedule,
)
from soar_lite.restart import NoRestartRule, StallRestartRule
from soar_lite.solver import DiagonalPreconditioner

__all__ = [
    "Bounds",
    "DiagonalPreconditioner",
    "EvalF",
    "EvalFG",
    "GlobalModel",
    "LocalSolver",
    "NPathResult",
    "NPathStatus",
    "NoRestartRule",
    "PathPoint",
    "RestartRule",
    "StallRestartRule",
    "Verdict",
    "n_path_schedule",
]
