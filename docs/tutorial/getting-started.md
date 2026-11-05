# Getting started

This page walks one run end to end: the objective contract and its constraints,
the call, what the result fields mean, and where the two swap points are.

## 1. Write the batched objective and constraints

The engine sees the objective through exactly one function: a batch of points
in, f values and gradients out together. You do not write that gradient by
hand: write the objective once, per point, and let `pointwise` build the batch
for you.

| Position | Shape | Meaning |
| --- | --- | --- |
| input `x` | `(M, d)` | `M` points, `d` coordinates, already inside the domain |
| output 1 | `(M,)` | f values |
| output 2 | `(M, d)` | gradients `∂f/∂x` |
| output 3 | `(M,)` or `None` | feasibility mask: 0/1 (or booleans); `None` means all feasible |

### 1.1 The objective

```python
import jax.numpy as jnp
from soar_lite import pointwise


def squares(x):  # one point (d,) in, one scalar out
    return jnp.sum((x - 3.0) ** 2)


eval_f_g, eval_f = pointwise(squares)
```

`pointwise` `vmap`s `jax.value_and_grad` for the fused seam and the plain
forward pass for the optional one, so the only function you write is the
per-point scalar - hand-writing `∂f/∂x` is fine for a sum of squares and is how
a sign or a factor slips in anywhere else. The second return is the `eval_f`
seam of §4: pass it as `eval_f=eval_f` to serve judged trials for their value
alone.

### 1.2 Constraints

Constraints ride two independent routes, split by kind rather than by taste:

- **The boundary is the domain's job.** `domain` declares the region every
  proposal is clipped into *before* it is served, so the objective never sees a
  point outside it.
- **Every other constraint is the mask's job.** Pass `feasible` to `pointwise`:
  a per-point predicate that returns `True` where the point is allowed. A point
  masked `False` is still served - its f value and gradient come back normally
  and are used - but it never becomes the best. So `result.fun` is a minimum
  over feasible points only, and is `nan` if nothing feasible was ever served.
  The mask is read as 0/1 and nothing else: only `0` marks a point infeasible,
  so a signed margin (negative where violated) or a `NaN` raises a `ValueError`
  instead of counting as feasible.

```python
from soar_lite import Domain


def below_diagonal(x):  # one point (d,) in, one boolean out
    return x[0] + x[1] <= 7.0


domain = Domain([(-5.0, 5.0), (-5.0, 5.0)])
eval_f_g, eval_f = pointwise(squares, feasible=below_diagonal)
```

`domain` carries only bounded sides here; the `Domain` of §4 adds a
per-dimension `scale`, unbounded sides, and the sampler a restart draws from.
`feasible` is how a constraint that is no side of the domain reaches the
engine: `below_diagonal` is a diagonal half-space no `Domain` can state, and it
leaves this run's answer alone because `(3, 3)` sits well inside it.

Remember neither route steers: `domain` only clips each proposal into its
sides, and the mask is a gate, not a guide - masked-out points are still served
and still pull the paths, so it decides who may be `best` and does not sit a
path on a boundary it violates. To make a constraint the domain cannot state
actually pull the paths, fold it into the objective as a differentiable penalty
and let the gradient guide:
`f(x) + c · max(0, violation(x))²` - in JAX,
`squares(x) + c * jnp.maximum(0.0, violation(x)) ** 2`. That buys guidance at
the price of soft feasibility: a finite `c` gives an approximately feasible
answer (too small and the paths still cross, too large and the objective
distorts), so keep `feasible` as the hard gate on what may become `best`. The
penalty schedule is your envelope, like any constraint transform; the engine
only climbs the `f` you hand it.

### 1.3 An objective that misbehaves: non-finite values and gradients

The engine never refuses a hard objective, it just stops trusting a point - and
two places trust differently. `best` is the looser read: it keeps the minimum
over *finite* `f` values among feasible points and never looks at `g`. So a
`nan` or `+inf` value never becomes `best`, but a finite value whose gradient
has a `nan` somewhere still can: the run reports that point even though no path
ever stands on it. A path commits more strictly - a point is `usable` only when
its value *and* every gradient component are finite, because the next step is
proposed from `-g` and folded on gradient differences. A point that is not
`usable` stalls its path in place: re-drawn by default, parked under
`restart_stalled=False` (see §2). The run carries on.

`-inf` is the one case to decide yourself. A *feasible* point priced at `-inf`
is not a value to compare but the answer, so the run stops at once with
`status == "unbounded"` (§3 gives the result semantics). If that `-inf` is not
the answer you want but a sign the objective was handed a point outside its real
domain, mark those points infeasible - a `domain` side or the `feasible`
predicate - and the short-circuit never fires: an infeasible `-inf` is skipped
like any other non-finite value.

`pointwise` differentiates with `jax.value_and_grad`, and that is where an
objective's seams show. A function that cannot be traced (a Python `if` on a
value, indexing by a traced integer) raises there; a function that is not
differentiable at a point yields whatever subgradient JAX picks, or a `nan` -
the `nan` stalls the path, a finite-but-wrong one steers it silently. When
either matters, write the gradient by hand and hand the engine a fused
`(f, g, feas)` seam instead of `pointwise` - the batch shape in the table above.

## 2. Run it

`eval_f_g` is the only positional argument; everything else is keyword-only.
`domain` and `n_paths` are required, and a run needs a stop - at least one of
the horizon `max_iters` or the deadline `time_limit`. `eval_f` is optional
(the value-only seam), handed in here for the budget it saves when the method
enumerates trials; the default single-sample step leaves it unbilled this run.
Each missing required argument is refused by `ValueError` rather
than guessed.

```python
from soar_lite import n_path_schedule

result = n_path_schedule(
    eval_f_g,
    domain=domain,
    n_paths=4,
    max_iters=30,
    eval_f=eval_f,
)
print(result.x, result.fun, result.status)
# ≈[3. 3.] 3e-5 max_iters
```

The status says only why the loop stopped, never that it converged: it is
`max_iters` here because the run reached the horizon you gave it - and the
horizon counts iterations, not work. The four paths get close to the minimiser
within a few iterations, and every later stop is a fresh start: by default
(`restart_stalled=True`) a stalled path reloads a draw from the global model each
time its own stopping test fires, so the flock keeps working until the horizon.
Ask for the frozen version instead and the same run ends much sooner: with
`restart_stalled=False` a converged path is parked for good - it holds its point
and still counts toward `best`, but never moves again - and once every path is
parked the run has nothing left to do and stops with status `all_stalled`. The
four counters report what either run actually served, which is why they, not
`n_iters`, are the cost.

## 3. Read the result

`NPathResult` follows `scipy.optimize.OptimizeResult` where the concepts
correspond:

| Field | Meaning |
| --- | --- |
| `x` | argmin, in the coordinates you handed the engine |
| `fun` | its value: the best over feasible points, `nan` if none, `-inf` if the objective has no floor |
| `n_f_g_points` | fused points billed - the serial-cost count |
| `n_f_g_batches` | fused batches billed, one per `eval_f_g` call - the parallel-cost count |
| `n_f_points` | value-only points billed (the `eval_f` seam) |
| `n_f_batches` | value-only batches billed (the `eval_f` seam) |
| `n_iters` | engine iterations completed (iteration 0 places the flock); one iteration may serve several batches |
| `status` | `max_iters`, `time_limit`, `budget_exhausted`, `unbounded` or `all_stalled` |

There is no `success` or `message`: the engine has no convergence criterion of
its own. It stops on a gate - the horizon (`max_iters`), the deadline
(`time_limit`), the extra budget gate - or before the horizon on an answer in
hand: `all_stalled` (the whole flock stalled with restarts off) and `unbounded`
(a feasible point priced the objective at `-inf`). Neither is a criterion the
search met.

## 4. Turn the knobs you actually need

| Knob | What it buys |
| --- | --- |
| `n_paths` | engine width: paths advanced in lockstep each iteration |
| `domain` | the search region, a `Domain`: its sides, per-dimension `scale`, and its injected draws (a side may be unbounded) |
| `x0` | the flock's start, one point per path `(n_paths, d)`: placed by iteration 0 exactly where you ask; `None` lets the global model draw it - hand it in for the starts a run is measured on |
| `eval_f` | the optional value-only seam: points in, f values out, no gradient - pass it to save budget, omit it in a fused-only comparison |
| `max_iters` | the horizon, counted in iterations (iteration 0 places the flock); at least one of `max_iters`/`time_limit` must be given |
| `time_limit` | the deadline, a wall-clock stop in seconds, for when an iteration's cost is too uneven to budget in iterations |
| `budget_check` | an extra whole-batch gate for a budget the engine cannot see |
| `restart_stalled` | whether a stalled path is re-drawn (default `True`); `False` freezes the flock, and the run ends on `all_stalled` once every path parks |
| `global_model` | the restart-point source and evidence sink; `None` binds `SampleGlobalModel` |
| `local_solver` | the search method; `None` binds `DiagonalPreconditioner` |
| `key` | PRNG seed; `None` uses a fixed default so unseeded runs stay deterministic |

Three gotchas worth knowing up front:

- **Sides may be unbounded.** A `Domain` side of `(None, None)` draws from
  `N(0, scale²)`, a half-side draws the same normal truncated, and any side can
  take its own injected sampler. There is no uniform prior over an infinite
  side, so the engine never invents one - it falls back to those draws.
- **`eval_f` splits judged steps into value-only trials and one fused settle.**
  The method reads each trial's f cheaply, then the single winner's gradient is
  bought once. Without `eval_f` every trial is itself a fused batch, so the
  count lands on the `f_g` pair instead of the `f` pair.
- **`n_paths = 1` is the single-path baseline.** Same code, one path - and with
  one path the parallelism is gone while the default restart bit still fires:
  every stop becomes a random redraw, which on a unimodal problem is a restart
  into the same basin until the horizon. Pass `restart_stalled=False` for the
  single chain that stops where it stalled.
- **The horizon counts iterations, not work.** What one iteration costs is the
  method's business: it serves the redraw starts plus however many trials the
  method's own line search serves, and a flock with nothing left to do ends the
  run (`all_stalled`) instead of entering a batchless iteration. So `max_iters`
  is a deterministic stop, not
  a cost estimate - read the four counters for what the run actually
  did, and pass the deadline (`time_limit`) if wall-clock is what you care about. `budget_check`
  is an extra gate consulted at the loop head each iteration and again before
  every billed batch (value-only included), so an external budget derived from
  batches or points can over-run by less than one batch, never more.

## 5. What a batch and a point cost

The two result fields answer different questions, and which one matters depends
on whether your objective is really parallel:

- **On a GPU.** A batch's wall-clock is dominated by kernel launches, so it
  barely grows as the batch widens, up to the device's parallel limit. Set
  `n_paths` to that limit and measure progress in batches: it is very nearly
  your wall-clock. Wider paths are then almost free, so raising `n_paths` (and,
  with `eval_f`, buying trial values by cheap value-only passes) shrinks the
  spread of the best and buys a larger improvement for the same number of
  batches.
- **On a CPU.** Batches run mostly serially, so the point count is the honest
  cost axis, and there width thins every path instead of speeding anything up.
  The flock advances in lockstep — every path takes one step an iteration — so
  at a fixed point budget each path gets about `1/n_paths` of the steps a
  single path would. Best-of-n only shrinks the spread across the starts; it
  cannot put the missing steps back. So whenever the budget runs out before a
  path converges — the regime this library is for — that lost depth dominates,
  and a single path ends better per point than a wide one. Width on a CPU buys
  variance reduction, not a speedup; it pays only where the paths are nearly
  free, which is the GPU's line, not the CPU's.

One sentence to keep them straight: a **batch** is one call (wall-clock, GPU), a
**point** is one row of data (serial cost, CPU) - and a batch is a
`(M, d)` array whose rows are those points.

## 6. Swap a piece

The engine owns the schedule; the method and the restart-point source are
plugins.
Both defaults are bound lazily, so you replace either one by name:

```python
from soar_lite import DiagonalPreconditioner, Domain, n_path_schedule

result = n_path_schedule(
    eval_f_g,
    domain=Domain([(-5.0, 5.0), (-5.0, 5.0)]),
    max_iters=30,
    local_solver=DiagonalPreconditioner(n_alpha=4, backtrack=0.5),
    restart_stalled=True,
)
```

To replace one outright, implement its protocol and pass the instance:

- **`LocalSolver`** — the search method: `restart`, `needs_verdict`, `propose`,
  `verdict`, `fold`. You own direction, step size and acceptance; the engine owns
  projection, the horizon and the per-step argmin. `verdict` is handed a
  trial's coords and value, not its gradient — the engine buys gradients only
  for the points it keeps, and `fold` is where a step lands. `restart` is the
  only cold-start call, so it runs at initialization and after every restart.
- **`GlobalModel`** — the restart-point source and evidence sink: `observe`,
  `propose`. The shipped pair is `SampleGlobalModel` (draws starts from the
  domain, fits nothing — the default) and `BLRGlobalModel` (fits a Bayesian
  linear posterior and proposes the `-log EI` minimum).

Both protocols are documented in the [API Reference](../reference/api.md), which
is the contract to code against.

## 7. Next

`ALGORITHM.md` in the repository is the derivation: why the floor is a floor and
not an addend, the cost counters, what may be redesigned freely, and the
invariants that must survive a redesign.
