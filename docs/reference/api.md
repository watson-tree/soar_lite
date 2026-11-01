# API Reference

Generated from the docstrings in `soar_lite/`. The public surface is deliberate
and small, in four groups: the entry point, the data types it hands back and
through, the two plugin contracts, and the concrete defaults.

## Entry point

::: soar_lite.n_path_schedule
::: soar_lite.pointwise

## Data types

::: soar_lite.NPathResult
::: soar_lite.NPathStatus
::: soar_lite.PathPoint
::: soar_lite.Verdict
::: soar_lite.Bounds
::: soar_lite.Domain
::: soar_lite.DrawFn

## Plugin contracts

Code against these to replace a piece; the engine calls them, never the reverse.

::: soar_lite.LocalSolver
::: soar_lite.GlobalModel
::: soar_lite.EvalFG
::: soar_lite.EvalF
::: soar_lite.Projection
::: soar_lite.AcquisitionFn
::: soar_lite.MinimizeFn

## Concrete defaults

Pass one by name to swap it, or implement the contract above to replace it.

::: soar_lite.DiagonalPreconditioner
::: soar_lite.SampleGlobalModel
::: soar_lite.BLRGlobalModel
::: soar_lite.IdentityProjection
::: soar_lite.SphereProjection