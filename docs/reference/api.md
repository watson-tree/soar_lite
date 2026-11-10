# API Reference

Generated from the docstrings in `soar_lite/`. The public surface is
`soar_lite.__all__`, grouped here in seven: the entry point, the results, the
domain, the evaluation seams, the local solvers, the global models, and the
projections.

Starting out? [The tutorial](../tutorial/getting-started.md) builds a run end to
end. This page gives the contract for each name; [How it
works](../explanation/how-it-works.md) gives the reasoning behind it.

A few names carry fixed meanings across the page. A **point** is one candidate:
coordinates and the value there, with the gradient once it has been bought; the
point a path commits — coordinates, value and gradient — is a `PathPoint`. A
**path** is one sequence of points, one per iteration, and a restart replaces its
current point, so the sequence is continuous in identity, not in space. The
**flock** is the points the paths hold at one moment — one per path, hence
`n_paths` of them; `flock_x` is the flock as the run ended. A **seam** is a
function the engine calls through, named by what it carries: the two
**evaluation seams** are `eval_f_g` (the fused pass) and `eval_f` (value only),
and the two **plugin seams** are `LocalSolver` and `GlobalModel`. A **fused
pass** is one call that returns value and gradient together. A **trial** is one
point a step looks at, served through `serve_trials`: when `eval_f` is
connected a trial costs its value alone, and without it every trial is itself a
fused pass. A **step** is one `LocalSolver.propose` call for one cohort: the
method reads as many trials as it wants and returns a **winner** per path, or
`None` to stop that path where it stands. A winner that was read travels as a
`Pending` with its gradient still **owed**, and whatever the step owes is bought
in one fused batch over the whole iteration (`settle`) — so with `eval_f`
connected an iteration adds one fused pass however many trials it read. A winner that does not survive that
batch's finite check, or that was read as non-finite, stalls the path rather than
being committed. An **arm** is one variant of a plugin, and an **ask** is one
`propose` call.

## Entry point

::: soar_lite.n_path_schedule
::: soar_lite.pointwise

## Results

One run's end state — the result, the work it billed, and the status that names
why it ended — and the two point types the plugins exchange: a `PathPoint` is a
committed point, a `Pending` is one still owed a half.

::: soar_lite.NPathResult
::: soar_lite.Work
::: soar_lite.NPathStatus
::: soar_lite.PathPoint
::: soar_lite.Pending

## The domain

The box, its per-dimension scales and its draws: everything a restart model may
read, and the object a projection is built on. `Bounds` is the box in scipy's
spelling, which `Domain` normalizes; `DrawFn` is one dimension's draw, the
sampler a caller injects.

::: soar_lite.Domain
::: soar_lite.Bounds
::: soar_lite.DrawFn

## Evaluation seams

The objective at the seam, in three shapes: the fused pass a run uses by
default, the value-only half a caller may add, and the trial hook the engine
hands a step in place of the caller's own function.

::: soar_lite.EvalFG
::: soar_lite.EvalF
::: soar_lite.ServeTrials

## Local solvers

`LocalSolver` is the engine's local seam: the engine calls it, never the
reverse. `DiagonalPreconditioner` is the solver the engine runs when the caller
names no other.

::: soar_lite.LocalSolver
::: soar_lite.DiagonalPreconditioner

## Global models

`GlobalModel` is the engine's restart seam, called by the engine and never
calling back. `SampleGlobalModel` draws from the domain; `BLRGlobalModel` fits a
posterior over the features its `Projection` hands it. `AcquisitionFn` and
`MinimizeFn` are the pair its inner minimizer works in: what it minimizes, and
the callable it injects to minimize it.

::: soar_lite.GlobalModel
::: soar_lite.SampleGlobalModel
::: soar_lite.BLRGlobalModel
::: soar_lite.AcquisitionFn
::: soar_lite.MinimizeFn

## Projections

The geometry a global model reads its features through: the model's own
extension point, not an engine seam. `SphereProjection` bends the coordinates
toward the pole, `IdentityProjection` reads them as they are, and a
`ProjectionFactory` is the recipe a caller hands the model to pick one per ask.

::: soar_lite.Projection
::: soar_lite.ProjectionFactory
::: soar_lite.SphereProjection
::: soar_lite.IdentityProjection
