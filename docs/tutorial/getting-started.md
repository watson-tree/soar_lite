# Getting started

This page walks one run end to end: the objective contract, the call, what the
result fields mean, and where the two swap points are.

## 1. Write the batched objective

The engine sees the objective through exactly one function: a batch of points
in, f values and gradients out together. One fused `jax.value_and_grad` pass gives
you both.

| Position | Shape | Meaning |
| --- | --- | --- |
| input `x` | `(M, d)` | `M` points, `d` coordinates, already inside the box |
| output 1 | `(M,)` | f values |
| output 2 | `(M, d)` | gradients `∂f/∂x` |
| output 3 | `(M,)` or `None` | feasibility mask; `None` means all feasible |

Constraints reach the engine by two independent routes:

- **The box.** `bounds` clips every candidate into the box *before* it is
  served, so the objective never sees a point outside it.
- **The mask.** A point masked `False` is still served - its f value and gradient
  come back normally and are used - but it never becomes the best. So
  `result.fun` is a minimum over feasible points only, and is `inf` if nothing
  feasible was ever served.

```python
import jax.numpy as jnp


def eval_f_g(x):
    f_vals = jnp.sum((x - 3.0) ** 2, axis=1)
    grad = 2.0 * (x - 3.0)
    return f_vals, grad, None
```

## 2. Run it

`eval_f_g` is the only positional argument. Everything else is a keyword,
and three of them are conditionally required: the engine draws its starts from a
uniform prior over a finite `bounds`, so an unbounded or absent box needs your
own `sample=`; a probe deeper than one draw needs an `eval_f=`; and a run
needs a horizon, `max_iters` or `time_limit`. Each is refused by `ValueError`
rather than guessed.

```python
from soar_lite import n_path_schedule

result = n_path_schedule(
    eval_f_g,
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
    n_paths=4,
    max_iters=30,
)
print(result.x, result.fun, result.status)
# [3. 3.] 0.0 max_iters
```

The status says only why the loop stopped, never that it converged: it is
`max_iters` here because the run reached the horizon you gave it. The paths
themselves had finished earlier - each one had already reached its own
per-coordinate stopping test and was waiting for a restart verdict that never
came due - so the last iterations served nothing and the run idled to the
horizon. That is the trade a deterministic stop makes: the iteration count is
yours, the work inside it is not. `n_batches` and `n_points` report what was
actually billed (value+gradient passes).

## 3. Read the result

`NPathResult` follows `scipy.optimize.OptimizeResult` where the concepts
correspond:

| Field | Meaning |
| --- | --- |
| `x` | argmin, in the coordinates you handed the engine |
| `fun` | its value: the best over feasible finite points, `inf` if none |
| `n_points` | points billed - the serial-cost count |
| `n_batches` | batches billed, one per `eval_f_g` call - the parallel-cost count |
| `n_iters` | engine iterations completed (iteration 0 is the initialization probe); one iteration may serve several batches |
| `status` | `max_iters`, `time_limit` or `budget_exhausted` |

There is no `success` or `message`: the engine has no convergence criterion of
its own. Stopping is always a horizon or an extra gate.

## 4. Turn the knobs you actually need

| Knob | What it buys |
| --- | --- |
| `bounds` | the search box, scipy-style `(lb, ub)` pairs per dim; `None`/`±inf` is unbounded |
| `n_paths` | engine width: paths advanced in lockstep each iteration (default 16) |
| `max_iters` | the horizon, counted in iterations (iteration 0 is the init probe); at least one of `max_iters`/`time_limit` must be given |
| `time_limit` | a wall-clock horizon in seconds, for when an iteration's cost is too uneven to budget in iterations |
| `n_draws` | probe depth: draws per start (default 1); above 1 requires `eval_f` |
| `sample` | your own prior `(key, n) -> (n, d)`; required when the box is unbounded |
| `budget_check` | an extra whole-batch gate for a budget the engine cannot see |
| `key` | PRNG seed; `None` uses a fixed default so unseeded runs stay deterministic |

Three gotchas worth knowing up front:

- **Unbounded boxes need a sampler.** With `bounds=None` or infinite sides there
  is no uniform prior to draw starts from, so pass `sample=`; the engine rejects
  the run rather than guessing.
- **A probe depth above one needs `eval_f`.** Each extra draw would compute
  its own gradient against a step's single one, so the engine refuses the depth
  unless you also supply the value-only `eval_f`; with it the probe computes
  one gradient, at its winners.
- **`n_paths = 1` is the single-path baseline.** Same code, no restart
  competition.
- **The horizon counts iterations, not work.** What one iteration costs is the
  method's business: it serves from zero batches (a fully stalled flock) up to a
  probe plus however many trials the method's own line search serves. So
  `max_iters` is a deterministic stop, not
  a cost estimate - read `n_batches` and `n_points` for what the run actually
  did, and pass `time_limit` if wall-clock is what you care about. `budget_check`
  is an extra gate consulted before every billed batch, so an external budget
  derived from batches or points can over-run by less than one batch, never more
  (inside a multi-draw probe the blind spot is one probe: its f-only draws are
  cheap and are neither gated nor counted).

## 5. What a batch and a point cost

The two result fields answer different questions, and which one matters depends
on whether your objective is really parallel:

- **On a GPU.** A batch's wall-clock is dominated by kernel launches, so it
  barely grows as the batch widens, up to the device's parallel limit. Set
  `n_paths` to that limit and measure progress in `n_batches`: it is very nearly
  your wall-clock. Wider paths are then almost free, so raising `n_paths` (and
  `n_draws` with it, which needs `eval_f`) shrinks the spread of the
  best and buys a larger
  improvement for the same number of batches.
- **On a CPU.** Batches run mostly serially, so the sampling work per point is
  what costs, and `n_points` is the honest number: raising `n_paths` or
  `n_draws` improves the best per point far less than it improves it
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
    eval_f_g,
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
    max_iters=30,
    solver=DiagonalPreconditioner(n_alpha=4, backtrack=0.5),
    restart_rule=EIRestartRule(),
)
```

To replace one outright, implement its protocol and pass the instance:

- **`LocalSolver`** — the search method: `restart`, `propose`, `verdict`,
  `fold`. You own direction, step size and acceptance; the engine owns
  projection, the horizon and the per-step argmin. `verdict` is handed a
  trial's coords and value, not its gradient — the engine buys gradients only
  for the points it keeps, and `fold` is where a step lands. `restart` is the
  only cold-start call, so it runs at initialization and after every restart.
- **`RestartRule`** — the continue-versus-restart decision: `observe_continue`,
  `observe_restart`, `should_restart`, `restart`.

Both protocols are documented in the [API Reference](../reference/api.md), which
is the contract to code against.

## 7. Next

`ALGORITHM.md` in the repository is the derivation: why the floor is a floor and
not an addend, the two cost axes, what may be redesigned freely, and the
invariants that must survive a redesign.
