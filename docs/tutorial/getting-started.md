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

The mask is the only way to express a constraint. Points masked `False` still
move along the objective, but they never become the incumbent, so `result.fun`
is a minimum over feasible points only and is `inf` if nothing feasible was ever
served.

```python
import jax.numpy as jnp


def evaluate_batch(x):
    loss = jnp.sum((x - 3.0) ** 2, axis=1)
    grad = 2.0 * (x - 3.0)
    return loss, grad, None
```

## 2. Run it

The one required argument is the evaluator. Everything else is a knob with a
default.

```python
from soar_lite import n_path_schedule

result = n_path_schedule(
    evaluate_batch,
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
    n_paths=4,
    max_steps=30,
)
print(result.x, result.fun, result.status)
# [3. 3.] 0.0 all_stalled
```

`all_stalled` is not a failure: every path reached its own per-coordinate
stopping test and no restart verdict was reachable before `max_steps`, so the
engine ended the run instead of spending rounds on no evaluation.

## 3. Read the result

`NPathResult` follows `scipy.optimize.OptimizeResult` where the concepts
correspond:

| Field | Meaning |
| --- | --- |
| `x` | argmin, in the coordinates you handed the engine |
| `fun` | its value: the incumbent over feasible finite points, `inf` if none |
| `nfev` | points evaluated, or batches under `per_round_evals` |
| `nit` | engine rounds; one round may hold several linesearch rounds |
| `status` | `max_steps`, `budget_exhausted` or `all_stalled` |

There is no `success` or `message`: the engine has no convergence criterion of
its own. Stopping is always a budget or a stall.

## 4. Turn the knobs you actually need

| Knob | What it buys |
| --- | --- |
| `bounds` | the search box, scipy-style `(lb, ub)` pairs per dim; `None`/`±inf` is unbounded |
| `n_paths` | engine width: paths advanced in lockstep each round (default 16) |
| `max_steps` | the engine's batch budget, counted in rounds |
| `max_ls` | per-step round cap after which a path stalls |
| `sample` | your own prior `(key, n) -> (n, d)`; required when the box is unbounded |
| `per_round_evals` | bill batches instead of points, a parallel-cost proxy |
| `key` | PRNG seed; `None` uses a fixed default so unseeded runs stay deterministic |

Three gotchas worth knowing up front:

- **Unbounded boxes need a sampler.** With `bounds=None` or infinite sides there
  is no uniform prior to draw starts from, so pass `sample=`; the engine rejects
  the run rather than guessing.
- **`n_paths = 1` is the single-path baseline.** Same code, no restart
  competition.
- **The budget is rounds, not evaluations.** With one batched call per round,
  `max_steps` bounds the number of calls you make.

## 5. Swap a piece

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

## 6. Next

`ALGORITHM.md` in the repository is the derivation: why the floor is a floor and
not an addend, the two billing notions, what may be redesigned freely, and the
invariants that must survive a redesign.
