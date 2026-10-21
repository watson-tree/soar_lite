"""soar_lite: first-order n-path descent with sample-based restarts.

The public face is deliberate and small. The two plugin contracts
(`LocalSolver`, `RestartRule`) and the data types they exchange are re-exported
from the engine so callers can write their own methods/rules against them; the
concrete defaults (`DiagonalPreconditioner`, `EIRestartRule`) are re-exported
for convenience. `n_path_schedule` is the whole algorithm entry point.

Internal helpers (the EI closed forms, the box/argmin utilities) are not
re-exported here on purpose; they are `_`-private in their own modules. A test
or a contributor may still import them, but nothing outside the package is
meant to.
"""

from soar_lite.engine import (
    BatchEvaluator,
    Bounds,
    LocalSolver,
    NPathResult,
    NPathStatus,
    PathPoint,
    RestartRule,
    Sampler,
    Verdict,
    n_path_schedule,
)
from soar_lite.restart import EIRestartRule
from soar_lite.solver import DiagonalPreconditioner

__all__ = [
    "BatchEvaluator",
    "Bounds",
    "DiagonalPreconditioner",
    "EIRestartRule",
    "LocalSolver",
    "NPathResult",
    "NPathStatus",
    "PathPoint",
    "RestartRule",
    "Sampler",
    "Verdict",
    "n_path_schedule",
]
