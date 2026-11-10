# soar-lite

A JAX optimizer for objectives that are differentiable but expensive to
evaluate. What matters is how many evaluations a run needs, not how long each one
takes: a single call can take seconds, so every choice here aims at needing fewer
of them. The caller passes value and gradient together, so there is no Hessian to
assemble and no second-order system to solve — each coordinate gets its own step
length instead.

**Use it when** the objective is differentiable, one evaluation costs real time
or money, and the search space is moderate- to high-dimensional.

**Skip it when** there is no gradient to pass — a gradient-free objective is out
of scope — or when a plain multistart local solve is affordable: if a few
scrambled starts plus a quasi-Newton method find the answer cheaply, that costs
you less. See [when this fits](explanation/how-it-works.md#when-this-fits) for
the reasoning.

A run moves `n_paths` paths in lockstep, each following its own descent.
Iteration 0 is the exception on the schedule — it places them, serving one batch
of starts before any step is taken (the engine's move logic itself needs no
special case: a path with no point simply restarts). After that an iteration is
usually one fused pass — a single call returning value and gradient for every
served point — unless the method enumerates candidates, which spends trial
batches of its own first and one fused batch grading the winner (see
[How it works](explanation/how-it-works.md)). On a
GPU a wider batch barely costs more wall-clock than a narrow one, which is why
the engine is built around batched evaluations; on a CPU the same run works,
just without that saving.

## Install

```bash
pip install soar-lite                 # CPU build: no GPU needed
pip install soar-lite 'jax[cuda12]'   # GPU: CUDA 12 driver
pip install soar-lite 'jax[cuda13]'   # GPU: CUDA 13 driver
```

`soar-lite` declares plain `jax` (the CPU build) so an install never needs a GPU,
and holds no GPU-specific code — only the JAX backend differs. The CUDA build is a
JAX extra, so the CUDA version is your choice, not one this package pins. JAX
picks its backend from what is installed; to pin one inside a process, call
`jax.config.update("jax_platform_name", "cpu")` — or `"cuda"` — before any JAX
work.

## Minimal run

```python
import jax.numpy as jnp
from soar_lite import Domain, n_path_schedule, pointwise


def squares(x):  # one point (d,) in, one scalar out
    return jnp.sum((x - 3.0) ** 2)


# pointwise returns two functions: eval_f_g for value and gradient in one pass,
# and eval_f for the value alone. This run needs only the first; `eval_f` is the
# value-only seam for a step's trials, not a gradient-free entry.
eval_f_g, _ = pointwise(squares)

result = n_path_schedule(
    eval_f_g,
    domain=Domain([(-5.0, 5.0), (-5.0, 5.0)]),  # (lower, upper) per dimension of x
    n_paths=4,
    max_iters=30,
)
print(result.x)  # x ≈ [3. 3]
print(result.fun)  # fun ≈ 3e-5
```

`x` is the best point the run evaluated, shape `(d,)` — the argmin over the
feasible points that were served, a feasible `-inf` naming itself — and `fun` is
its value. Do not read it as a row of `flock_x`: the flock is the points the
paths hold at one moment, one per path, so `flock_x` is the flock as the run
ended, and the best point may be one a path has left behind. Field names follow
scipy's
`OptimizeResult` (`x` the minimizer, `fun` the minimum); the extra fields are the
flock pair, four work counts (points and batches, with and without a gradient)
and a `status`.

## What's inside

Two plugin seams — extension points you can swap — under one engine that
schedules the work, and nothing beyond JAX and the standard library.

- **The engine, `n_path_schedule`** — runs the `n_paths` paths in lockstep and
  owns the domain, the shared horizon (`max_iters`), and the restart schedule: a
  path stalls when its `propose` answers `None` or its point fails the finite
  gate, and the engine redraws it — `restart_stalled` on — from a fresh point.
- **The local method, `LocalSolver`** — one path's descent step,
  `DiagonalPreconditioner` by default: a step length per coordinate re-estimated
  from successive gradients, no matrix and no scalar learning rate to tune. It
  takes one whole step, read by no value; a method that does enumerate
  candidates names its own winner, and the engine grades that winner alone.
- **The restart source, `GlobalModel`** — where a restarted path is placed, plus
  the record of everything evaluated so far. `SampleGlobalModel` draws from the
  domain; `BLRGlobalModel` is a Bayesian surrogate for the same contract.

The two seams are keyword arguments: pass `local_solver=` or `global_model=` to
replace either.

What ships is not a general-purpose toolkit but a small set of defaults picked by
judgment — which is what `-lite` signals. They are aimed at scientific numerical
optimization, where the variables carry physical units and the objective's scale
is rarely yours to pick: once the first secant pair exists the step is
invariant to rescaling of the objective, so one set of knobs serves any units,
and nobody has to non-dimensionalize the problem by hand first. The warm-up step
is the exception — its whole-step length is still chosen for the units in play.

The name nods to the **adaptive restart** framework of
[Mathesen et al. (2021)](explanation/how-it-works.md#references): a local method
runs between restarts, and a global side decides where each start comes from.
This library borrows that division of labor, not its code —
[how it works](explanation/how-it-works.md#soar-attribution-and-distinction) says
where the two depart.

## Where to go next

- **[Tutorial](tutorial/getting-started.md)** — when you have one run to get
  right: the objective contract, the knobs, and the two plugin seams.
- **[Surrogate restarts](how-to/surrogate-restarts.md)** — when `BLRGlobalModel`
  is in play and its acquisition function has to be descended by the engine
  itself.
- **[API Reference](reference/api.md)** — when you have a name and need its
  contract; generated from the docstrings.
- **[How it works](explanation/how-it-works.md)** — when you want the reasons:
  the derivation behind the local step, how cost is counted, and how this library
  departs from SOAR.

## Citing

If this library does useful work in something you publish, cite the software:
the entry is `CITATION.cff` at the repository root, which is what GitHub's
*Cite this repository* button reads. Its `version` field names the release this
tree is at, so pin the citation to the version you ran — a result then names the
code that produced it.
