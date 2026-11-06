# API Reference

Generated from the docstrings in `soar_lite/`. The public surface is
`soar_lite.__all__`, grouped here in six: the entry point, the results, the
domain, the type aliases, the plugin contracts, and the concrete defaults.

Starting out? [The tutorial](../tutorial/getting-started.md) builds a run end to
end. This page gives the contract for each name; [How it
works](../explanation/how-it-works.md) gives the reasoning behind it.

A few names carry fixed meanings across the page. A **path** is one chain of
points; the **flock** is all `n_paths` of them. A **seam** is a function the
engine calls through, named by what it carries: the two **evaluation seams** are
`eval_f_g` (the fused pass) and `eval_f` (value only), and the two **plugin
seams** are `LocalSolver` and `GlobalModel`. A **fused pass** is one call that
returns value and gradient together. A **judged step** has the engine serve each
candidate for its value alone and lets `LocalSolver.verdict` rule it, buying
gradients only for the winner; an **unjudged** one serves its single proposal
fused with its gradient and commits it after the engine's finite check alone. An
**arm** is one variant of a plugin, and an **ask** is one `propose` call.

## Entry point

::: soar_lite.n_path_schedule
::: soar_lite.pointwise

## Results

One run's end state, and the point type the plugins exchange.

::: soar_lite.NPathResult
::: soar_lite.PathPoint

## The domain

The box, its per-dimension scales and its draws: everything a restart model may
read, and the object a projection is built on.

::: soar_lite.Domain

## Type aliases

The shapes a caller writes down when it plugs something in.

::: soar_lite.NPathStatus
::: soar_lite.Verdict
::: soar_lite.Bounds
::: soar_lite.DrawFn
::: soar_lite.EvalFG
::: soar_lite.EvalF
::: soar_lite.AcquisitionFn
::: soar_lite.MinimizeFn

## Plugin contracts

Code against these to replace a piece. The engine calls `LocalSolver` and
`GlobalModel`, never the reverse; `Projection` is the model's own extension point,
read by `BLRGlobalModel`, and comes with the factory alias a caller writes to
build one.

::: soar_lite.LocalSolver
::: soar_lite.GlobalModel
::: soar_lite.Projection
::: soar_lite.ProjectionFactory

## Concrete defaults

Pass one by name to swap it, or implement the contract above to replace it.

::: soar_lite.DiagonalPreconditioner
::: soar_lite.SampleGlobalModel
::: soar_lite.BLRGlobalModel
::: soar_lite.IdentityProjection
::: soar_lite.SphereProjection
