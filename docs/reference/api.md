# API Reference

Generated from the docstrings in `soar_lite/`. The public surface is deliberate
and small, in four groups: the entry point, the data types it hands back and
through, the two plugin contracts, and the concrete defaults.

A few names carry fixed meanings across the page. A **path** is one chain of
points; the **flock** is all `n_paths` of them. A **seam** is a function the
engine calls through - an injected objective (`EvalFG`, `EvalF`) or one of the
two plugins. A **fused pass** is one call that returns value and gradient
together. A **judged step** serves its candidates for their values first and lets
`LocalSolver.verdict` rule them; an **unjudged** one commits its proposal after
the engine's finite check alone. An **arm** is one variant of a plugin, and an
**ask** is one `propose` call. `docs/explanation/how-it-works.md` tells the whole
story.

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
::: soar_lite.ProjectionFactory

## Concrete defaults

Pass one by name to swap it, or implement the contract above to replace it.

::: soar_lite.DiagonalPreconditioner
::: soar_lite.SampleGlobalModel
::: soar_lite.BLRGlobalModel
::: soar_lite.IdentityProjection
::: soar_lite.SphereProjection