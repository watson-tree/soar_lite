# soar-lite

First-order diagonal-preconditioned n-path descent with stalled-path restarts
drawn from a `GlobalModel`, for objectives expensive enough that the
evaluation budget is the binding constraint. The caller returns value and
gradient together; the library fits no Hessian and no curvature model — the
default restarts are sampled from the domain, a surrogate is opt-in
(`BLRGlobalModel`).

Three seams, no dependencies beyond JAX and the standard library:

- **`n_path_schedule`** — the engine: runs `n_paths` paths in lockstep, one
  batched evaluation per iteration, and owns the domain, the shared horizon and the
  stall state machine. A stalled path is re-drawn only when `restart_stalled=True`,
  the run's one restart knob.
- **`DiagonalPreconditioner`** — the default local method: a per-coordinate
  diagonal secant step `-d ⊙ g`, EMA-smoothed, with a gradient-norm floor and an
  optional alpha grid (`n_alpha`) whose argmin the engine commits. No matrix, no
  linear solve, no scalar learning rate.
- **`SampleGlobalModel`** / **`BLRGlobalModel`** — the restart-point source and the
  run's evidence sink: the engine asks it for one fresh point per restarting path.
  The default `SampleGlobalModel` draws whatever the `domain` admits; `BLRGlobalModel`
  is the Bayesian linear-regression surrogate for the same contract.

## Install

```bash
pip install soar-lite
```

## Minimal run

```python
import jax.numpy as jnp
from soar_lite import Domain, n_path_schedule


def eval_f_g(x):
    """(M, d) points in; (M,) f values, (M, d) gradients, (M,)|None mask out."""
    return jnp.sum((x - 3.0) ** 2, axis=1), 2.0 * (x - 3.0), None


result = n_path_schedule(
    eval_f_g,
    domain=Domain([(-5.0, 5.0), (-5.0, 5.0)]),
    n_paths=4,
    max_iters=30,
)
print(result.x, result.fun)  # [3. 3.] 0.0
```

## Where to go next

- **[Tutorial](tutorial/getting-started.md)** — one run end to end: the
  objective contract, the knobs, and the two swap points.
- **[API Reference](reference/api.md)** — generated from the docstrings.
- `ALGORITHM.md` in the repository is the derivation: what each piece does, the
  invariants that must survive a redesign, and what this library does and does
  not take from SOAR.
