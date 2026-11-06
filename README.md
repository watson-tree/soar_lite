# soar-lite

[![CI](https://github.com/watson-tree/soar_lite/actions/workflows/ci.yml/badge.svg)](https://github.com/watson-tree/soar_lite/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/watson-tree/soar_lite/blob/main/notebooks/quickstart.ipynb)

A JAX optimiser for objectives that are differentiable but expensive: the number
of evaluations, not wall-clock, is the binding constraint. The caller hands in
value and gradient together, so the library fits no Hessian and never solves
second-order information directly — the diagonal preconditioner is a biased
`1/H` estimate, not Newton.

**Use it when** the objective is differentiable, one evaluation costs real time
or money, and the search space is moderate- to high-dimensional.

**Skip it when** there is no gradient to hand over (a zero-order objective is not
this library's seam), or the dimension is low — with few coordinates several
restarts collapse into the same basin, and a plain multistart local solve is the
cheaper call.

`n_paths` paths advance in lockstep, one fused batch per iteration, which is a
GPU pattern: a batch's wall-clock barely grows as it widens. On a CPU the same
run works, but it buys a best-of-n variance reduction per point rather than a
wall-clock speedup.

## What's inside

Two plugin seams and the engine that joins them, a few modules
(`soar_lite/engine.py`, `soar_lite/local_solver.py`, `soar_lite/domain.py`,
`soar_lite/global_model/`), no dependencies beyond JAX and the standard library:

- **`n_path_schedule`** — the engine. Runs `n_paths` paths in lockstep: an
  iteration decides every move on values and buys the gradients it kept in one
  batch. Owns the domain, the shared horizon and the stall state machine.
  `n_paths = 1` is the single-path baseline.
- **`LocalSolver`** — one path's descent step. The default
  `DiagonalPreconditioner` reads the secant equation one coordinate at a time,
  `d_i = 1/B_ii = s_i/y_i`, smooths the secant pair with an EMA, floors the
  denominator with a gradient-norm term, and steps `-d ⊙ m`, where `m` is the
  momentum (the gradient-history blend; the bare gradient `g` at `mu = 0`). No
  matrix, no linear solve, no scalar learning rate, and unit-invariant: scaling a
  coordinate by `c` scales its step by `c`, and rescaling `f` leaves the step
  alone, so one set of unitless knobs serves mixed units.
- **`GlobalModel`** — the restart points, and the run's evidence sink. The engine
  asks it for one fresh point per restarting path and clips it into the domain.
  The default `SampleGlobalModel` draws whatever the `domain` admits (uniform
  inside a bounded dimension, `N(0, scale²)` where it is free);
  `BLRGlobalModel` is the Bayesian linear-regression surrogate for the same
  contract.
- **`restart_stalled`** — the *whether* of a restart, a boolean on
  `n_path_schedule`. *When* a path has stopped is the local solver's own stopping
  test; *where* it goes is the model's, so there is no third seam. `True` (the
  default) re-draws a stalled path; `False` keeps the flock the caller starts as
  the whole run, and the run then ends on `all_stalled` once every path parks.

The name is a nod to the adaptive restart framework of Mathesen et al. (2021). A
local method runs between restarts, and the **adaptive restart process** decides
where each next start comes from — mainly by selecting a **sampling criterion**,
an improvement function such as expected improvement or probabilistic
improvement, scored against a global model fitted on what has been evaluated.
This library keeps that outer division — a global side and a local side — and
folds the fitted model and the sampling criterion into the one `GlobalModel`
contract, so the question it answers is where to restart. The framework is the
idea; the code here is this library's own, and
[How it works](docs/explanation/how-it-works.md#soar-attribution-and-distinction)
says where the two depart.

What ships is a lightweight set of defaults picked by judgement, not a
general-purpose toolkit — which is what `-lite` says. Both seams are open, so a
more general implementation can drop in later; what is implemented today is aimed
at scientific numerical optimisation.

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

## Development

```bash
uv sync                       # jax
uv run python -m pytest -q    # all tests in one go
uv run --no-sync python -m tests.test_local_solver   # or one module at a time,
uv run --no-sync python -m tests.test_global_model   # each module also runs
uv run --no-sync python -m tests.test_scheduler      # as a plain script
uv run --no-sync python -m tests.test_plugin_seam
uv run ruff check && uv run pyright
```

The tests are assert-style self-checks: plain `assert` in `test_*` functions,
pytest-collectable, each module also runnable as a script — no fixtures, no
recorded golden numbers that would encode this repository's history.

## Scope

This package is the optimiser only. The evaluation budget it consumes is the
caller's business: `eval_f_g` takes a batch of points and returns f values,
gradients and an optional feasibility mask, so the same engine drives a scalar
objective, a batched one, or a penalized transform of a constrained problem. The
envelope (working-space reparameterization, the penalty schedule, submission and
scoring plumbing) lives outside.

## Documentation

Published at <https://watson-tree.github.io/soar_lite/>.

- [Tutorial](docs/tutorial/getting-started.md) — one run end to end: the
  objective contract, the knobs, and the two swap points.
- [How it works](docs/explanation/how-it-works.md) — the derivation behind the
  local step, the two cost axes, and what this library does and does not take
  from SOAR.
- [API Reference](docs/reference/api.md) — generated from the docstrings.
