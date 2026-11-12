# API Reference

Generated from the docstrings in `soar_lite/`. The public surface is
`soar_lite.__all__`, grouped here in seven: the entry point, the results, the
domain, the evaluation seams, the local solvers, the global models, and the
projections.

Starting out? [The tutorial](../tutorial/getting-started.md) builds a run end to
end, and the Example under `n_path_schedule` below is a whole run you can paste.
This page gives the contract for each name; [How it
works](../explanation/how-it-works.md) gives the reasoning behind it.

## Vocabulary

A few names carry fixed meanings across the page. These are worth reading once;
everything after them assumes them.

**The run.** A **path** is one sequence of points, one per iteration, and a
restart replaces its current point, so the sequence is continuous in identity,
not in space. The **flock** is the points the paths hold at one moment - one per
path, hence `n_paths` of them; `flock_x` is the flock as the run ended.

**What gets evaluated.** A **point** is one candidate: coordinates and the value
there, with the gradient once it has been bought; the point a path commits -
coordinates, value and gradient - is a `PathPoint`. A **batch** is the `(M, d)`
points one call serves. A **fused pass** is one call
that returns value and gradient together. A **trial** is one point a step looks
at, served through `serve_trials`; one **round** of that hook is one trial per
path in the cohort and one batch. When `eval_f` is connected a trial costs its
value alone, and without it each round is itself a fused pass. A **step** is
one `LocalSolver.propose` call for one cohort: the method reads as many trials
as it wants and returns a **winner** per path, or `None` to stop that path where
it stands. A winner that was read travels as a `ValuePoint` with its gradient still
**owed**, and whatever the step owes is bought in one fused batch over the whole
iteration - the engine's *settle* phase - so with `eval_f` connected an iteration
adds one fused pass however many trials it read. A winner that does not survive
that batch's finite check, or that was read as non-finite, stalls the path rather
than being committed.

The other refusal is not a stall. A **coordinate** the run cannot hold - not
finite, or outside `domain` - is a bug in whoever produced it, so the engine
raises `ValueError` at the seam that handed it over. And the settle batch is
conditional: it serves whatever still owes a gradient, so an iteration that
decides nothing buys nothing.

**The seams.** A **seam** is a name the engine calls through, named by what it
carries: an **evaluation seam** is a function handed in (`eval_f_g`, the fused
pass, and `eval_f`, value only), while a **plugin seam** is a protocol whose
methods the engine calls (`LocalSolver`, `GlobalModel`). An **arm** is one
variant of a plugin, an **ask** is one `propose` call, and a **cohort** is the
set of paths one call covers - the stepping paths for the method's `propose`,
the restarting ones for the model's.

Names with an entry below are exported from `soar_lite`. The words *settle*,
*owed*, *arm*, *ask* and *cohort* are this page's vocabulary for how the engine
works -
there is no such import.


## Entry point

::: soar_lite.n_path_schedule
::: soar_lite.pointwise

## Results

One run's end state — the result, the work it billed, and the status that names
why it ended — and the two point types the plugins exchange: a `PathPoint` is a
committed point, a `ValuePoint` is one whose value is read and whose gradient is
still owed, and a bare `Point` is one nothing has been read from yet.

::: soar_lite.NPathResult
::: soar_lite.Work
::: soar_lite.NPathStatus
    options:
      show_attribute_values: true
      show_signature_annotations: false
::: soar_lite.Point
::: soar_lite.ValuePoint
::: soar_lite.PathPoint

## The domain

The box, its per-dimension scales and its draws: everything a restart model may
read, and the object a projection is built on. `Bounds` is the box in scipy's
spelling, which `Domain` normalizes; `DrawFn` is one dimension's draw, the
sampler a caller injects.

::: soar_lite.Domain
::: soar_lite.Bounds
    options:
      show_attribute_values: true
      show_signature_annotations: false
::: soar_lite.DrawFn
    options:
      show_attribute_values: true
      show_signature_annotations: false

## Evaluation seams

The objective at the seam, in three shapes: the fused pass a run uses by
default, the value-only half a caller may add, and the trial hook the engine
hands a step in place of the caller's own function.

::: soar_lite.EvalFG
    options:
      show_attribute_values: true
      show_signature_annotations: false
::: soar_lite.EvalF
    options:
      show_attribute_values: true
      show_signature_annotations: false
::: soar_lite.ServeTrials
    options:
      show_attribute_values: true
      show_signature_annotations: false

## Local solvers

`LocalSolver` is the engine's local seam: the engine calls it, never the
reverse. `DiagonalPreconditioner` is the solver the engine runs when the caller
names no other.

::: soar_lite.LocalSolver
::: soar_lite.DiagonalPreconditioner

## Global models

`GlobalModel` is the engine's restart seam, called by the engine and never
calling back. `SampleGlobalModel` draws from the domain; `BLRGlobalModel` fits a
posterior over the features its `Projection` hands it and descends its
acquisition with a `MinimizeFn`, defaulting to `descent_minimize`.
`AcquisitionFn` is what that minimizer minimizes, `MinimizeFn` the callable's
shape, and `descent_minimize` the one the package ships.

::: soar_lite.GlobalModel
::: soar_lite.SampleGlobalModel
::: soar_lite.BLRGlobalModel
::: soar_lite.AcquisitionFn
    options:
      show_attribute_values: true
      show_signature_annotations: false
::: soar_lite.MinimizeFn
    options:
      show_attribute_values: true
      show_signature_annotations: false

::: soar_lite.descent_minimize

## Projections

The geometry a global model reads its features through: the model's own
extension point, not an engine seam. `SphereProjection` bends the coordinates
toward the pole, `IdentityProjection` reads them as they are, and a
`ProjectionFactory` is the recipe a caller hands the model to pick one per ask.

::: soar_lite.Projection
::: soar_lite.ProjectionFactory
    options:
      show_attribute_values: true
      show_signature_annotations: false
::: soar_lite.SphereProjection
::: soar_lite.IdentityProjection
