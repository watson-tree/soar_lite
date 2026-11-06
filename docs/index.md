# soar-lite

A JAX optimizer for objectives that are differentiable but expensive: the number
of evaluations, not wall-clock, is the binding constraint. The caller hands in
value and gradient together, so the library fits no Hessian and never solves
second-order information directly — the diagonal preconditioner is a biased
`1/H` estimate, not Newton.

**Use it when** the objective is differentiable, one evaluation costs real time
or money, and the search space is moderate- to high-dimensional.

**Skip it when** there is no gradient to hand over (a zero-order objective is not
this library's seam), or the dimension is low — see
[when this fits](explanation/how-it-works.md#when-this-fits) for what goes wrong
there.

`n_paths` paths advance in lockstep, one fused batch per iteration, which is a
GPU pattern: a batch's wall-clock barely grows as it widens.

## Install

```bash
pip install soar-lite                 # CPU JAX: works anywhere
pip install soar-lite 'jax[cuda12]'   # GPU: CUDA 12 driver
pip install soar-lite 'jax[cuda13]'   # GPU: CUDA 13 driver
```

`soar-lite` declares plain `jax` (the CPU build) so an install never needs a GPU,
and holds no GPU-specific code — only the JAX backend differs. The CUDA build is a
JAX extra, so which one you get stays yours to name rather than a version this
package picks. JAX picks its backend from what is installed; to pin one inside a
process, call `jax.config.update("jax_platform_name", "cpu")` — or `"cuda"` —
before any JAX work.

## Minimal run

```python
import jax.numpy as jnp
from soar_lite import Domain, n_path_schedule, pointwise


def squares(x):  # one point (d,) in, one scalar out
    return jnp.sum((x - 3.0) ** 2)


eval_f_g, _ = pointwise(squares)  # (fused, value-only); the second is unused here

result = n_path_schedule(
    eval_f_g,
    domain=Domain([(-5.0, 5.0), (-5.0, 5.0)]),
    n_paths=4,
    max_iters=30,
)
print(result.x, result.fun)  # ≈[3. 3.] 3e-5
```

## What's inside

Two plugin seams under one engine, and no dependencies beyond JAX and the
standard library. The engine schedules the work; each seam owns one half of the
search, and either can be replaced by passing an instance in.

- **The engine, `n_path_schedule`** — runs the `n_paths` paths in lockstep, one
  fused batch per iteration, and owns the domain, the shared horizon and the
  stall state machine.
- **The local method, `LocalSolver`** — one path's descent step. The default
  `DiagonalPreconditioner` is a per-coordinate secant step `-d ⊙ m`, where `m` is
  an EMA-smoothed gradient history (`g` itself at `mu = 0`), with a gradient-norm
  floor and an optional alpha grid (`n_alpha`) whose argmin the engine commits. No
  matrix, no linear solve, no scalar learning rate.
- **The restart source, `GlobalModel`** — one fresh point per restarting path, and
  the run's evidence sink. The default `SampleGlobalModel` draws whatever the
  `domain` admits; `BLRGlobalModel` is the Bayesian linear-regression surrogate for
  the same contract.

What ships is a lightweight set of defaults picked by judgment, not a
general-purpose toolkit — which is what `-lite` says. Both seams are open, so a
more general implementation can drop in later; what is implemented today is aimed
at scientific numerical optimization.

The name is a nod to the adaptive restart framework of Mathesen et al. (2021). A
local method runs between restarts, and the **adaptive restart process** decides
where each next start comes from — mainly by selecting a **sampling criterion**,
an improvement function such as expected improvement or probabilistic
improvement, scored against a global model fitted on what has been evaluated.
This library keeps that outer division — a global side and a local side — and
folds the fitted model and the sampling criterion into the one `GlobalModel`
contract, so the question it answers is where to restart. The framework is the
idea; the code here is this library's own, and
[how it works](explanation/how-it-works.md#soar-attribution-and-distinction) says
where the two depart.

## Where to go next

- **[Tutorial](tutorial/getting-started.md)** — when you have one run to get
  right: the objective contract, the knobs, and the two swap points.
- **[Surrogate restarts](how-to/surrogate-restarts.md)** — when `BLRGlobalModel`
  is in play and its acquisition has to be descended by the engine itself.
- **[API Reference](reference/api.md)** — when you have a name and need its
  contract; generated from the docstrings.
- **[How it works](explanation/how-it-works.md)** — when you want the reasons:
  the derivation behind the local step, the two cost axes, and how this library
  departs from SOAR.
