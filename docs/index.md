# soar-lite

A JAX optimizer for objectives that are differentiable but expensive to
evaluate. The binding constraint is the number of evaluations, not wall-clock
time: one call can cost seconds, and every choice here is aimed at needing fewer
of them. The caller hands in value and gradient together, so no Hessian is ever
assembled and no second-order system is solved — the diagonal preconditioner is a
biased estimate of the diagonal of `1/H` (`H` the Hessian), not Newton's method.

**Use it when** the objective is differentiable, one evaluation costs real time
or money, and the search space is moderate- to high-dimensional.

**Skip it when** there is no gradient to hand over — a gradient-free objective is
outside the entry contract — or the dimension is low:
[when this fits](explanation/how-it-works.md#when-this-fits) says why a cheap
Hessian makes a quasi-Newton method the better call there.

A run moves `n_paths` paths in lockstep, one local descent trajectory each, and
calls the objective on all their points in a single fused pass per iteration: one
call that returns value and gradient together and shares the work in between. On
a GPU a wider batch barely costs more wall-clock than a narrow one, which is why
the method is shaped that way; on a CPU the same run works, without the dividend.

## Install

```bash
pip install soar-lite                 # CPU JAX: works anywhere
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


# pointwise returns two functions: eval_f_g for the value and gradient in one
# pass, and eval_f for the value alone. This run needs only the first.
eval_f_g, _ = pointwise(squares)

result = n_path_schedule(
    eval_f_g,
    domain=Domain([(-5.0, 5.0), (-5.0, 5.0)]),
    n_paths=4,
    max_iters=30,
)
print(result.x, result.fun)  # ≈[3. 3.] 3e-5
```

`result` follows scipy's `OptimizeResult` (`x`, `fun`) and adds the whole flock,
the four evaluation counts and a `status`.

## What's inside

Two plugin seams under one engine, and no dependencies beyond JAX and the
standard library. The engine schedules the work; each plugin seam owns half of
the search, and either can be replaced by passing an instance in.

- **The engine, `n_path_schedule`** — runs the `n_paths` paths in lockstep, one
  fused pass per iteration, and owns the domain, the shared horizon
  (`max_iters`), and the stall state machine that decides when a path restarts.
- **The local method, `LocalSolver`** — one path's descent step. The default
  `DiagonalPreconditioner` takes a per-coordinate secant step `-d ⊙ m`, where `d`
  is `1/H` for that coordinate, estimated from successive gradients, and `m` is an
  EMA-smoothed gradient history (`m = g` at `mu = 0`). A gradient-norm floor
  keeps `d` finite as the curvature estimate goes to zero, and an optional alpha
  grid (`n_alpha`) is a set of candidate step lengths the engine compares by
  value and commits the lowest.
  No matrix, no linear solve, and no scalar learning rate to tune: the step is
  invariant to rescaling of the objective's units, so one set of unitless knobs
  serves any units.
- **The restart source, `GlobalModel`** — one fresh point for each path that
  restarts, and the record of everything evaluated so far. The default
  `SampleGlobalModel` draws whatever the `domain` admits; `BLRGlobalModel` is the
  Bayesian linear-regression surrogate for the same contract.

What ships is a lightweight set of defaults picked by judgment, not a
general-purpose toolkit — which is what `-lite` says. Both plugin seams are open,
so a more general implementation can drop in later; what is implemented today is
aimed at scientific numerical optimization.

The name is a nod to the adaptive restart framework of Mathesen et al. (2021): a
local method runs between restarts, and the **adaptive restart process** decides
where each next start comes from. Its main lever is a **sampling criterion** — an
improvement function such as expected improvement (EI) or probability of
improvement (PI) — scored against a global model fitted on what has been
evaluated. This library keeps that outer division, a global side and a local
side, and folds the fitted model and the sampling criterion into the one
`GlobalModel` contract, so the question it answers is where to restart. It borrows
the framework's division of labor, not its code;
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
  the derivation behind the local step, the two cost axes, and how this library
  departs from SOAR.
