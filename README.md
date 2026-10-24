# soar-lite

[![CI](https://github.com/watson-tree/soar_lite/actions/workflows/ci.yml/badge.svg)](https://github.com/watson-tree/soar_lite/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/watson-tree/soar_lite/blob/main/notebooks/quickstart.ipynb)

First-order diagonal-preconditioned n-path descent with sample-based
(expected-improvement) restarts, for minimizing objectives that are expensive
enough to make the evaluation budget the binding constraint. Value and gradient
come back together from the caller; no Hessian, no curvature model, no surrogate
model.

Three pieces, three modules (`soar_lite/engine.py`, `soar_lite/solver.py`,
`soar_lite/restart.py`), no dependencies beyond JAX and the standard library:

- **`DiagonalPreconditioner`** — the local method. Reads the secant equation one
  coordinate at a time, `d_i = 1/B_ii = s_i/y_i`, smooths the secant pair with an
  EMA, floors the denominator with a gradient-norm term, and steps `-d ⊙ g`. No
  matrix, no linear solve, no scalar learning rate, unit-invariant.
- **`EIRestartRule`** — sample-based restarts. Compares the expected improvement of
  continuing a path against the expected improvement of restarting it, both read
  off the path's own recent loss changes and a pool of restart payoffs shared by
  every path. No surrogate model is fitted.
- **`n_path_schedule`** — the engine. Runs `n_paths` paths in lockstep, one batched
  evaluation per iteration, and owns the box, the shared budget, the restart verdicts
  and the stall state machine. `n_paths = 1` is the single-path baseline.

`ALGORITHM.md` is the derivation: what each piece does, why the floor is a floor
and not an addend, the two cost axes, what may be redesigned freely, the
seven invariants that must survive a redesign, and what this library does and does
not take from SOAR.

## Install

```bash
pip install soar-lite
```

```python
from soar_lite import DiagonalPreconditioner, EIRestartRule, n_path_schedule
```

## Run

```bash
uv sync                       # jax
uv run python -m tests.test_solver        # the local method
uv run python -m tests.test_restart       # the restart rule
uv run python -m tests.test_scheduler     # the engine, driven by a stub method
uv run python -m tests.test_solver_seam   # the injection seams
uv run --no-sync python -m pytest -q   # or all 66 in one go
uv run ruff check && uv run pyright
```

The tests are assert-style self-checks: plain `assert` in `test_*` functions,
pytest-collectable, each module also runnable as a script — no fixtures, no
recorded golden numbers that would encode this repository's history.

## Scope

This package is the optimizer only. The evaluation budget it consumes is the
caller's business: `evaluate_batch` takes a batch of points and returns losses,
gradients and an optional feasibility mask, so the same engine drives a scalar
objective, a batched one, or a penalized transform of a constrained problem. The
envelope (working-space reparameterization, the penalty schedule, submission and
scoring plumbing) lives outside.