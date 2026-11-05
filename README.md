# soar-lite

[![CI](https://github.com/watson-tree/soar_lite/actions/workflows/ci.yml/badge.svg)](https://github.com/watson-tree/soar_lite/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/watson-tree/soar_lite/blob/main/notebooks/quickstart.ipynb)

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
step alone, so one set of dimensionless knobs serves mixed units (kg, N/m,
N·s/m, …). Unit invariance is a property of the preconditioner itself, not of
the warm-up step, which is by design not coordinate-invariant — so the
invariance experiment shows a small, self-correcting offset there, not a
different descent rate (the contrast is Adam, whose single learning rate is
per-unit-system).

Three seams, a few modules (`soar_lite/engine.py`, `soar_lite/local_solver.py`,
`soar_lite/domain.py`, `soar_lite/global_model/`), no dependencies beyond JAX
and the standard library:

- **`DiagonalPreconditioner`** — the local method. Reads the secant equation one
  coordinate at a time, `d_i = 1/B_ii = s_i/y_i`, smooths the secant pair with an
  EMA, floors the denominator with a gradient-norm term, and steps `-d ⊙ g`. No
  matrix, no linear solve, no scalar learning rate, unit-invariant.
- **`SampleGlobalModel`** / **`BLRGlobalModel`** — the restart points. A model
  supplies **where** a restart lands and is the run's evidence sink; the engine
  asks it for one fresh point per restarting path and clips it into the domain. The
  default `SampleGlobalModel` draws whatever the `domain` admits (uniform inside
  a bounded dimension, `N(0, scale²)` where it is free); `BLRGlobalModel` is the
  Bayesian linear-regression surrogate for the same contract.
- **`restart_stalled`** — the *whether* of a restart, a boolean on
  `n_path_schedule`. *When* a path has stopped is the local solver's own
  stopping test; *where* it goes is the model's, so there is no third seam.
  `True` (the default) re-draws a stalled path; `False` keeps the flock the
  caller starts as the whole run, and the run then ends on `all_stalled` once
  every path parks.
- **`n_path_schedule`** — the engine. Runs `n_paths` paths in lockstep: an
  iteration decides every move on values and buys the gradients it kept in one
  batch. Owns the domain, the shared horizon and the stall state machine.
  `n_paths = 1` is the single-path baseline.

`ALGORITHM.md` is the derivation: what each piece does, why the floor is a floor
and not an addend, the two cost axes, what may be redesigned freely, the
seven invariants that must survive a redesign, and what this library does and does
not take from SOAR.

## Install

```bash
pip install soar-lite
```

```python
from soar_lite import DiagonalPreconditioner, SampleGlobalModel, n_path_schedule
```

## Run

```bash
uv sync                       # jax
uv run python -m tests.test_local_solver   # the local method
uv run python -m tests.test_global_model   # the restart-point models
uv run python -m tests.test_scheduler      # the engine, driven by a stub method
uv run python -m tests.test_plugin_seam    # the injection seams
uv run --no-sync python -m pytest -q   # or all tests in one go
uv run ruff check && uv run pyright
```

The tests are assert-style self-checks: plain `assert` in `test_*` functions,
pytest-collectable, each module also runnable as a script — no fixtures, no
recorded golden numbers that would encode this repository's history.

## Scope

This package is the optimizer only. The evaluation budget it consumes is the
caller's business: `eval_f_g` takes a batch of points and returns f values,
gradients and an optional feasibility mask, so the same engine drives a scalar
objective, a batched one, or a penalized transform of a constrained problem. The
envelope (working-space reparameterization, the penalty schedule, submission and
scoring plumbing) lives outside.