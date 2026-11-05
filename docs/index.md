# soar-lite

First-order n-path descent with stalled-path restarts, for objectives
expensive enough that the evaluation budget is the binding constraint. The
caller returns value and gradient together: the library fits no Hessian and
does not solve second-order information directly (the diagonal preconditioner
is a biased `1/H` estimate, not Newton), and a zero-order objective — value
only, no gradient — is not its seam. It is another realisation of the SOAR
framework (Mathesen et al., 2021): the core is the same idea (a global
restart-point model and a local method each own their half, joined by one
mechanism), while the details are this library's own — not a port of the
reference implementation, not interface-compatible, and no claim to reproduce
its results.

It is built for moderate- to high-dimensional search spaces and is
deliberately not tuned for low dimension: with few coordinates several
restarts collapse into the same basin, and the default sampler draws
independent uniform/normal points rather than a low-discrepancy sequence
because the mid-to-high regime is the target. The `n_paths` paths advance in
lockstep, one fused batch per iteration — a GPU pattern: a batch's wall-clock
barely grows as it widens. On a CPU the same run works, but it only trades
evaluations for a best-of-n reduction in variance, not a wall-clock speedup.

`-lite` is the same decision stated as scope: the shipped pieces are a
lightweight set of defaults, chosen by judgment rather than fitted for
generality — not a general-purpose toolkit. The local method and the
restart-point model are both plugin points — write a `LocalSolver` or a
`GlobalModel` and pass it in — so a more general implementation can drop in
later; what this repository currently implements is aimed at scientific
numerical optimisation. The default local solver, `DiagonalPreconditioner`,
is a per-coordinate secant preconditioner that is unit-invariant: scaling a
coordinate `x` by `c` scales its step by `c`, and rescaling `f` leaves the
step alone, so one set of unitless knobs serves mixed units (kg, N/m,
N·s/m, …). Unit invariance is a property of the preconditioner itself, not of
the warm-up step, which is by design not unit-invariant — so the
invariance experiment shows a small, self-correcting offset there, not a
different descent rate (the contrast is Adam, whose single learning rate is
per-unit-system).

Two plugin seams under one engine, no dependencies beyond JAX and the standard
library:

- **`n_path_schedule`** — the engine: runs `n_paths` paths in lockstep, one
  batched evaluation per iteration, and owns the domain, the shared horizon and the
  stall state machine. A stalled path is re-drawn unless the run sets
  `restart_stalled=False`, the run's one restart knob.
- **`DiagonalPreconditioner`** — the default local method: a per-coordinate
  diagonal secant step `-d ⊙ m`, where `m` is the momentum (the gradient-history
  blend; the bare gradient `g` at `mu = 0`), EMA-smoothed, with a gradient-norm
  floor and an optional alpha grid (`n_alpha`) whose argmin the engine commits.
  No matrix, no linear solve, no scalar learning rate.
- **`SampleGlobalModel`** / **`BLRGlobalModel`** — the restart-point source and the
  run's evidence sink: the engine asks it for one fresh point per restarting path.
  The default `SampleGlobalModel` draws whatever the `domain` admits; `BLRGlobalModel`
  is the Bayesian linear-regression surrogate for the same contract.

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


eval_f_g, _ = pointwise(squares)

result = n_path_schedule(
    eval_f_g,
    domain=Domain([(-5.0, 5.0), (-5.0, 5.0)]),
    n_paths=4,
    max_iters=30,
)
print(result.x, result.fun)  # ≈[3. 3.] 3e-5
```

## Where to go next

- **[Tutorial](tutorial/getting-started.md)** — one run end to end: the
  objective contract, the knobs, and the two swap points.
- **[API Reference](reference/api.md)** — generated from the docstrings.
- **[How it works](explanation/how-it-works.md)** — the derivation: what each
  piece does, the two cost axes, and what this library does and does not take
  from SOAR.
