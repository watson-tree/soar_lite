# soar-lite

First-order diagonal-preconditioned n-path descent with sample-based
(expected-improvement) restarts, for objectives expensive enough that the
evaluation budget is the binding constraint. The caller returns value and
gradient together; the library fits no Hessian, no curvature model and no
surrogate.

Three pieces, no dependencies beyond JAX and the standard library:

- **`n_path_schedule`** — the engine: runs `n_paths` paths in lockstep, one
  batched evaluation per iteration, and owns the box, the shared horizon, the restart
  verdicts and the stall state machine.
- **`DiagonalPreconditioner`** — the default local method: a per-coordinate
  diagonal secant step `-d ⊙ g`, EMA-smoothed, with a gradient-norm floor and an
  optional alpha grid (`n_alpha`) whose argmin the engine commits. No matrix, no
  linear solve, no scalar learning rate.
- **`EIRestartRule`** — the default restart rule: compares the expected
  improvement of continuing a path against the expected improvement of
  restarting it, both read off the path's own recent f changes and a pool of
  group-mins. Each side is a scott_mad Gaussian-kernel KDE over its own
  samples, so neither assumes a single Gaussian.
- **`NoRestartRule`** — the same seam with the probes turned off after the
  initialization one: a run is pure multi-start, or a plain serial chain at
  `n_paths = 1`.

## Install

```bash
pip install soar-lite
```

## Minimal run

```python
import jax.numpy as jnp
from soar_lite import n_path_schedule


def eval_f_g(x):
    """(M, d) points in; (M,) f values, (M, d) gradients, (M,)|None mask out."""
    return jnp.sum((x - 3.0) ** 2, axis=1), 2.0 * (x - 3.0), None


result = n_path_schedule(
    eval_f_g,
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
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
