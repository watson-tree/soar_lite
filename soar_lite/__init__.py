"""soar_lite: first-order K-path descent with sample-based restarts.

The public face is deliberate and small. The two plugin contracts
(`LocalSolver`, `RestartRule`) and the data types they exchange are re-exported
from the engine so callers can write their own methods/rules against them; the
concrete defaults (`DiagonalPreconditioner`, `EIRestartRule`) are re-exported
for convenience. `kpath_schedule` is the whole algorithm entry point.

Internal helpers (the EI closed forms, the box/argmin utilities) are not
re-exported here on purpose: they are an implementation detail, reachable only
from `soar_lite.restart` / `soar_lite.engine` when a test or a contributor
really needs them.
"""

from soar_lite.engine import (
    BatchEvaluator,
    Bounds,
    KPathResult,
    KPathStatus,
    LocalSolver,
    PathPoint,
    RestartRule,
    Sampler,
    Verdict,
    kpath_schedule,
)
from soar_lite.restart import EIRestartRule
from soar_lite.solver import DiagonalPreconditioner

__all__ = [
    "BatchEvaluator",
    "Bounds",
    "DiagonalPreconditioner",
    "EIRestartRule",
    "KPathResult",
    "KPathStatus",
    "LocalSolver",
    "PathPoint",
    "RestartRule",
    "Sampler",
    "Verdict",
    "kpath_schedule",
]
