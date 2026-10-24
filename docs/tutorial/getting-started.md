# Getting started

This page walks one run end to end: the objective contract, the call, what the
result fields mean, and where the two swap points are.

## 1. Write the batched objective

The engine sees the objective through exactly one function: a batch of points
in, losses and gradients out together. One fused `jax.value_and_grad` pass gives
you both.

| Position | Shape | Meaning |
| --- | --- | --- |
| input `x` | `(M, d)` | `M` points, `d` coordinates, already inside the box |
| output 1 | `(M,)` | losses |
| output 2 | `(M, d)` | gradients `∂f/∂x` |
| output 3 | `(M,)` or `None` | feasibility mask; `None` means all feasible |

Constraints reach the engine by two independent routes:

- **The box.** `bounds` clips every candidate into the box *before* it is
  served, so the objective never sees a point outside it.
- **The mask.** A point masked `False` is still served - its loss and gradient
  come back normally and are used - but it never becomes the incumbent. So
  `result.fun` is a minimum over feasible points only, and is `inf` if nothing
  feasible was ever served.

```python
import jax.numpy as jnp


def evaluate_batch(x):
    loss = jnp.sum((x - 3.0) ** 2, axis=1)
    grad = 2.0 * (x - 3.0)
    return loss, grad, None
```

## 2. Run it

`evaluate_batch` is the only positional argument. Everything else is a keyword,
and just one of them is conditionally required: the engine draws its starts from
a uniform prior over a finite `bounds`, so an unbounded or absent box needs your
own `sample=`. With neither, the call raises `ValueError`.

```python
from soar_lite import n_path_schedule

result = n_path_schedule(
    evaluate_batch,
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
    n_paths=4,
    max_evals=30,
)
print(result.x, result.fun, result.status)
# [3. 3.] 0.0 all_stalled
```

`all_stalled` is not a failure: every path reached its own per-coordinate
stopping test and no restart verdict was reachable before `max_evals` ran out,
so the engine ended the run instead of spinning iterations that serve no
evaluation.

## 3. Read the result

`NPathResult` follows `scipy.optimize.OptimizeResult` where the concepts
correspond:

| Field | Meaning |
| --- | --- |
| `x` | argmin, in the coordinates you handed the engine |
| `fun` | its value: the incumbent over feasible finite points, `inf` if none |
| `n_points` | points served - the serial-cost count |
| `n_batches` | batches served, one per `evaluate_batch` call - the parallel-cost count |
| `nit` | engine iterations attempted; one iteration may serve several batches |
| `status` | `max_evals`, `budget_exhausted` or `all_stalled` |

There is no `success` or `message`: the engine has no convergence criterion of
its own. Stopping is always a budget or a stall.

## 4. Turn the knobs you actually need

| Knob | What it buys |
| --- | --- |
| `bounds` | the search box, scipy-style `(lb, ub)` pairs per dim; `None`/`±inf` is unbounded |
| `n_paths` | engine width: paths advanced in lockstep each iteration (default 16) |
| `max_evals` | the engine's budget, counted in served batches (default 1000) |
| `max_ls` | trials one step may serve before the path stalls (default 20) |
| `n_probe_evals` | probe depth: draws per start, and the horizon a continue sample spans (default 2) |
| `sample` | your own prior `(key, n) -> (n, d)`; required when the box is unbounded |
| `budget_check` | an extra whole-batch gate for a budget the engine cannot see |
| `key` | PRNG seed; `None` uses a fixed default so unseeded runs stay deterministic |

Three gotchas worth knowing up front:

- **Unbounded boxes need a sampler.** With `bounds=None` or infinite sides there
  is no uniform prior to draw starts from, so pass `sample=`; the engine rejects
  the run rather than guessing.
- **`n_paths = 1` is the single-path baseline.** Same code, no restart
  competition.
- **The budget counts batches, not points.** It can only be exact in the unit
  it is charged in: one `evaluate_batch` call costs one of `max_evals`, whatever
  the batch holds. The point count is reported, not budgeted - read `n_points`.
  `budget_check` is consulted once per batch too, so an external budget derived
  from points can over-run by less than one batch, never more.

## 5. What a batch and a point cost

The two result fields answer different questions, and which one matters depends
on whether your objective is really parallel:

- **On a GPU.** A batch's wall-clock is dominated by kernel launches, so it
  barely grows as the batch widens, up to the device's parallel limit. Set
  `n_paths` to that limit and measure progress in `n_batches`: it is very nearly
  your wall-clock. Wider paths are then almost free, so raising `n_paths` (and
  `n_probe_evals` with it) shrinks the spread of the incumbent and buys a larger
  improvement for the same number of batches.
- **On a CPU.** Batches run mostly serially, so the sampling work per point is
  what costs, and `n_points` is the honest number: raising `n_paths` or
  `n_probe_evals` improves the incumbent per point far less than it improves it
  per batch.

One sentence to keep them straight: a **batch** is one call (wall-clock, GPU), a
**point** is one row of data (serial cost, CPU) - and a batch is a
`(M, d)` array whose rows are those points.

## 6. Swap a piece

The engine owns the schedule; the method and the restart decision are plugins.
Both defaults are bound lazily, so you replace either one by name:

```python
from soar_lite import DiagonalPreconditioner, EIRestartRule, n_path_schedule

result = n_path_schedule(
    evaluate_batch,
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
    solver=DiagonalPreconditioner(n_alpha=4, backtrack=0.5),
    restart_rule=EIRestartRule(),
)
```

To replace one outright, implement its protocol and pass the instance:

- **`LocalSolver`** — the search method: `init`, `restart`, `propose`,
  `verdict`, `fold`. You own direction, step size and acceptance; the engine
  owns projection, the budget and the per-step argmin.
- **`RestartRule`** — the continue-versus-restart decision: `observe_continue`,
  `observe_restart`, `should_restart`, `samples_until_verdict`, `restart`.

Both protocols are documented in the [API Reference](../reference/api.md), which
is the contract to code against.

## 7. Next

`ALGORITHM.md` in the repository is the derivation: why the floor is a floor and
not an addend, the two cost axes, what may be redesigned freely, and the
invariants that must survive a redesign.
