# soar-lite

[![CI](https://github.com/watson-tree/soar_lite/actions/workflows/ci.yml/badge.svg)](https://github.com/watson-tree/soar_lite/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/watson-tree/soar_lite/blob/main/notebooks/run_panels.ipynb)

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
you less. See [when this fits](docs/explanation/how-it-works.md#when-this-fits) for
the reasoning.

A run moves `n_paths` paths in lockstep, each following its own descent.
Iteration 0 is the exception on the schedule — it places them, serving one batch
of starts before any step is taken (the engine's move logic itself needs no
special case: a path with no point simply restarts). After that an iteration is
usually one fused pass — a single call returning value and gradient for every
served point — unless the method enumerates candidates, which spends trial
batches of its own first and one fused batch grading the winner (see
[How it works](docs/explanation/how-it-works.md)). On a
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

## Results

Each method gets the same objective, the same seeds and the same batch budget.
A curve is best-so-far, the line a median over 20 seeds and the band its Q1-Q3
spread. Click a panel for the full size.

[![The two real inverse problems, best-so-far against batches served](docs/figures/headline-batches.png)](docs/figures/headline-batches.png)

**The two real inverse problems** - a mixed-unit oscillator chain and a
non-convex compartment chain - on the batches axis: one batch is one parallel
pass, the wall clock of a GPU setting, which is what this library is built for.
20 seeds x 3000 batches. A wide `n_paths=16` arm is shown whole here and pays 16
objective points per batch for it. It is not ahead on both axes: the same runs
on the points axis, where that width is charged per point, are in
[`docs/figures/`](docs/figures/).

[![Two preconditioned quadratics, one where a diagonal metric is exact](docs/figures/smooth-batches.png)](docs/figures/smooth-batches.png)

**Two quadratics, 100 dimensions each** - one diagonal and conditioned at `1e4`,
where a per-coordinate estimate is exact, one rotated and conditioned at `1e6`,
where it is not - on the batches axis. 20 seeds x 3000 batches.

[![Rastrigin in 100 dimensions](docs/figures/multimodal-batches.png)](docs/figures/multimodal-batches.png)

**A multimodal landscape** - Rastrigin in 100 dimensions, where the basin a run
falls into decides how far it gets - on the batches axis. 20 seeds x 3000
batches.

[![One physics measured in two unit systems, with no retuning](docs/figures/invariance-units.png)](docs/figures/invariance-units.png)

**Unit invariance.** One physics measured in two unit systems (`x' = 10x`) with
nothing retuned: the per-coordinate secant overlaps, while a single global
learning rate separates because the effective step rescales with the unit. 20
seeds x 3000 batches.

Both axes of all four panels, and the Adam lr sweep, are in
[`docs/figures/`](docs/figures/) - the points axis is the one that charges
`n_paths=16` for its width, and it is worth a look before the batches axis is
read as a ranking. The medians behind every curve and the caveats that qualify
them are in `experiments/RESULTS.md`; what the comparison measures is in
`experiments/DESIGN.md`. The commands that redraw the whole set are in
[`docs/figures/README.md`](docs/figures/README.md).

Those files are named by path rather than linked because the harness under
`experiments/` is a benchmark sub-library, not a dependency: it imports
soar-lite and cites this documentation, nothing in the package or these docs
depends on it, and it ships in neither the wheel nor the sdist. The figures
above are a snapshot of one run, so reading a result does not need the harness.

To run any panel yourself, open
[`notebooks/run_panels.ipynb`](notebooks/run_panels.ipynb) (what the Colab badge
above opens): one section per panel, with the figure each run writes.

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
[Mathesen et al. (2021)](docs/explanation/how-it-works.md#references): a local method
runs between restarts, and a global side decides where each start comes from.
This library borrows that division of labor, not its code —
[how it works](docs/explanation/how-it-works.md#soar-attribution-and-distinction) says
where the two depart.

## Scope

This package is the optimizer only. The evaluation budget it consumes is the
caller's business: `eval_f_g` takes a batch of points and returns f values,
gradients and an optional feasibility mask, so the same engine drives a scalar
objective, a batched one, or a penalized transform of a constrained problem. The
envelope (working-space reparameterization, the penalty schedule, submission and
scoring plumbing) lives outside.

The mask is a gate, not a guide: a point it masks out is still served and still
pulls its path, it just never becomes `best`, so `fun` is a minimum over feasible
points only. A constraint the domain cannot state steers nothing until you fold
it into the objective as a differentiable penalty, and that schedule is yours to
choose. Both halves of the contract, plus what becomes of a non-finite value or
gradient, are set out in [the tutorial's first
section](docs/tutorial/getting-started.md#1-write-the-batched-objective-and-constraints).

## Documentation

Published at <https://watson-tree.github.io/soar_lite/>.

- [Tutorial](docs/tutorial/getting-started.md) — one run end to end: the
  objective contract, the knobs, and the two swap points.
- [Surrogate restarts](docs/how-to/surrogate-restarts.md) — when
  `BLRGlobalModel` is in play and its acquisition function has to be descended
  by the engine itself.
- [How it works](docs/explanation/how-it-works.md) — the derivation behind the
  local step, the two cost axes, and what this library does and does not take
  from SOAR.
- [API Reference](docs/reference/api.md) — generated from the docstrings.

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) holds the checks a change has to pass
(pytest, ruff, pyright, `mkdocs build --strict`), the sync contract between these
two front doors, and where the benchmark suite lives.

## Citing

If this library does useful work in something you publish, cite the software:
the entry is `CITATION.cff` at the repository root, which is what GitHub's
*Cite this repository* button reads. Its `version` field names the release this
tree is at, so pin the citation to the version you ran — a result then names the
code that produced it.
