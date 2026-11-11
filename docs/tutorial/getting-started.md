# Getting started

This page walks one run end to end: the objective contract and its constraints,
the call, what the result fields mean, and where the two swap points are. It
assumes `soar-lite` and JAX are installed — see [Install](../index.md#install) on
the home page.

## 1. Write the batched objective and constraints

The engine sees the objective through one required function — the batch function
it calls, `eval_f_g` below, whose signature is the table. It takes a batch of
points in, f values and gradients out together. You do not write that gradient by
hand: write the objective once, per point, and let `pointwise` build the batch
for you.

| Position | Shape | Meaning |
| --- | --- | --- |
| input `x` | `(M, d)` | `M` points, `d` coordinates, already inside the domain |
| output 1 | `(M,)` | f values |
| output 2 | `(M, d)` | gradients `∂f/∂x` |
| output 3 | `(M,)` or `None` | feasibility mask: 0/1 (or booleans); `None` means all feasible |

Every point served is inside `domain` already: the engine clips the restart
starts it draws, and a method clips its own step.

### 1.1 The objective

```python
import jax.numpy as jnp
from soar_lite import pointwise


def squares(x):  # one point (d,) in, one scalar out
    return jnp.sum((x - 3.0) ** 2)


eval_f_g, eval_f = pointwise(squares)
```

`pointwise` `vmap`s `jax.value_and_grad` for the fused seam — the function the
engine calls, once per batch — and the plain forward pass for the optional
value-only seam (the second return value), so the only function you write is the
per-point scalar. Hand-writing `∂f/∂x` is safe for a sum of squares, but
elsewhere it is an easy way to introduce a sign or a scale error. That second
return is the seam §4 uses: pass it as `eval_f=eval_f` to serve a step's trials
for their value alone. Without `feasible`, the third slot is still there, as `None`
— the same `None` the hand-written seam in §1.3 returns.

### 1.2 Constraints

Constraints ride two independent routes, split by kind rather than by
preference:

- **The boundary is the domain's job.** `domain` declares the region; nothing
  you write has to re-check it.
- **Every other constraint is the mask's job.** Pass `feasible` to `pointwise`:
  a per-point predicate that returns `True` where the point is allowed. A point
  masked `False` is still served — its f value and gradient come back normally
  and are used — but it never becomes the best. So `result.f` is a minimum
  over feasible points only, and is `nan` when no feasible point carried a
  finite value.
  The mask is read as 0/1, booleans included (`True == 1`): a `1` marks the
  point feasible, a `0` infeasible, and any other value — a signed margin, a
  `NaN` — raises a `ValueError` instead of being read.

```python
from soar_lite import Domain


def below_diagonal(x):  # one point (d,) in, one boolean out
    return x[0] + x[1] <= 7.0


domain = Domain([(-5.0, 5.0), (-5.0, 5.0)])
eval_f_g, eval_f = pointwise(squares, feasible=below_diagonal)
```

`domain` carries only bounded sides here; §4 has the rest of what a `Domain`
can hold. `feasible` is how a constraint the domain cannot state reaches the engine:
`below_diagonal` is a diagonal half-space, not a box side, and it leaves this
run's answer alone because `(3, 3)` sits well inside it.

Remember neither route steers: `domain` only bounds each proposal by its
sides, and the mask is a gate, not a guide — masked-out points are still served
and still pull the paths, so it decides who may be `best` but does not steer a
path away from infeasible regions. To make a constraint the domain cannot state
actually pull the paths, fold it into the objective as a differentiable penalty
and let the gradient guide:
`f(x) + c · max(0, violation(x))²`, where `violation(x)` is zero where the
constraint holds and positive where it is violated — in JAX,
`squares(x) + c * jnp.maximum(0.0, violation(x)) ** 2`. That buys guidance at
the price of soft feasibility: a finite `c` gives an approximately feasible
answer (too small and the paths still cross, too large and the objective
distorts), so keep `feasible` as the hard gate on what may become `best`. The
penalty schedule is yours to choose, like any constraint transform; the engine
only climbs the `f` you hand it.

### 1.3 An objective that misbehaves: non-finite values and gradients

The engine never refuses a hard objective, it just stops trusting a point. Two
places trust differently: `best` reads `f` alone, while a path needs a finite
`g` as well.

`best` is the looser read — the minimum over *finite* `f` values among feasible
points — so a `nan` or `+inf` value never becomes `best`, but a finite value whose
gradient has a `nan` somewhere still can become `best`: the run reports that point
even though no path ever stands on it. A path commits more strictly: a point is
`is_usable` only when its value *and* every gradient component are finite, because the
next step is proposed from `-g` and the secant is built from gradient differences.
A point that is not `is_usable` stalls its path in place. A path that holds a
point is re-drawn by default and parked under `restart_stalled=False` (see §2);
a path holding none is re-drawn either way. The run carries on.

`-inf` is the one case to decide yourself. A *feasible* point priced at `-inf`
is not a value to compare but the answer, so the run stops at once with
`status == "unbounded"` (§3 gives the result semantics). If that `-inf` is not
the answer you want but a sign the objective was handed a point outside its real
domain, mark those points infeasible with `pointwise`'s `feasible` predicate,
and the short-circuit never fires: an infeasible `-inf` is skipped like any other
non-finite value. Narrowing the domain's sides instead masks nothing — the engine
clips into the box, so such a point is never served at all.

`pointwise` differentiates with `jax.value_and_grad`, and that is where the
entry contract shows through. A function that cannot be traced (a Python `if` on a
value, indexing by a traced integer) raises there; a function that is not
differentiable at a point yields whatever subgradient JAX picks, or a `nan` —
the `nan` stalls the path, a finite-but-wrong one steers it silently. When
either matters, write the gradient by hand and hand the engine the fused
`(f, g, feas)` evaluation seam instead of the `eval_f_g` that `pointwise` builds,
with the shapes of the table above:

```python
import jax


def squares_f_g(x):  # one point (d,) in, value and gradient out
    return jax.value_and_grad(squares)(x)


def squares_batch(x_batch):  # the fused seam: (M, d) -> (M,), (M, d), (M,)
    f, g = jax.vmap(squares_f_g)(x_batch)
    return f, g, x_batch[:, 0] + x_batch[:, 1] <= 7.0  # the third slot is the mask
```

`squares_batch` goes where `eval_f_g` went; the third slot takes `None` when
there is no mask. What you buy is control over the gradient, and what you owe is
every shape in that table.

## 2. Run it

The fused seam — `eval_f_g` here, whatever you name it — is the only positional
argument; everything else is keyword-only.
`domain` and `n_paths` are required, and a run needs a stop — at least one of
the horizon `max_iters` or the deadline `time_limit`. Leaving out a required
keyword is a `TypeError` from the signature, and giving neither stop is a
`ValueError` at the entry — checked, rather than guessed.

`eval_f` is optional: hand it in when the method enumerates trials for their
values alone; §4 says when that saves budget. The default solver reads no trial
at all — it takes one whole step and reads no value — so this run never calls it.
It is passed here only to show where it goes.

```python
from soar_lite import n_path_schedule

# eval_f_g, eval_f and domain come from §1.2
result = n_path_schedule(
    eval_f_g,
    domain=domain,
    n_paths=4,
    max_iters=30,
    eval_f=eval_f,
)
print(result.x, result.f, result.status)
# x ≈ [3. 3], f ≈ 3e-5, status = max_iters
```

The status says only why the loop stopped, never that it converged: it is
`max_iters` here because the run reached the horizon you gave it — and the
horizon counts iterations, not work. The four paths get close to the minimizer
within a few iterations, and every later stop is a fresh start: by default
(`restart_stalled=True`) a stalled path reloads a draw from the global model each
time its own stopping test fires, so the flock — the points the paths hold — keeps
working until the horizon.
Ask for the frozen version instead and the same run ends much sooner: with
`restart_stalled=False` a converged path is parked for good — it holds its point
and still counts toward `best`, but never moves again — and once every path is
parked the run has nothing left to do and stops with status `all_stalled`. The
four counters of §3 report what either run actually served, which is why the
cost is read off them and not off `n_iters`.

## 3. Read the result

`NPathResult` spells its answer the way a point does — `x` for the
coordinates, `f` for the value — and prefixes the flock's own pair with `flock_`:

| Field | Meaning |
| --- | --- |
| `x` | argmin over the served points that were feasible with a finite value — a feasible `-inf` names itself — in the coordinates you handed the engine |
| `f` | its value: the best finite feasible value, `nan` when no feasible point had one, `-inf` if the objective has no floor |
| `flock_x` | each path's held coordinates as the run ended, `(n_paths, d)`; `nan` where the path never held a usable point |
| `flock_f` | each path's held value, `(n_paths,)`; `nan` where the path holds no usable point |
| `work` | a `Work`: the single home of the four counts below |
| `n_f_g_points` | fused points billed — the serial-cost count |
| `n_f_g_batches` | fused batches billed, one per `eval_f_g` call — the parallel-cost count |
| `n_f_points` | value-only points billed (the `eval_f` seam) |
| `n_f_batches` | value-only batches billed (the `eval_f` seam) |
| `n_iters` | engine iterations completed (iteration 0 places the flock); one iteration may serve several batches |
| `status` | `max_iters`, `time_limit`, `budget_exhausted`, `unbounded` or `all_stalled` |

There is no `success` or `message`: the engine has no convergence criterion of
its own. It stops on a gate — the horizon (`max_iters`), the deadline
(`time_limit`), the extra budget gate — or before the horizon on an answer in
hand: `all_stalled` (the whole flock stalled with restarts off) and `unbounded`
(a feasible point priced the objective at `-inf`). Neither of those two is a
criterion the search met.

The flock pair is not a ranking: `flock_x` and `flock_f` are each path's
held point as the run ended, one row per path, unsorted and unfiltered by
feasibility — the value is the one that path stopped on, not a fresh
evaluation, so a path frozen by `restart_stalled=False` reports the value it
stalled with. Nothing gets you the flock of an earlier iteration, and `x`/`f`
need not be a row of this one: the best point may come from a path that was
restarted afterwards.

## 4. Turn the knobs you actually need

| Knob | What it buys |
| --- | --- |
| `n_paths` | engine width: paths advanced in lockstep each iteration |
| `domain` | the search region, a `Domain`: its sides, per-dimension `scale`, and its injected draws (a side may be unbounded) |
| `x0` | the flock's start, one point per path `(n_paths, d)`: placed by iteration 0 exactly where you ask, and it must lie inside `domain` — an outside row is refused, not clipped; `None` lets the global model draw it — hand it in when runs have to be compared on the same starts |
| `eval_f` | the optional value-only seam: points in, `(f, feas?)` out, no gradient — pass it to save budget, omit it in a fused-only comparison |
| `max_iters` | the horizon, counted in iterations (iteration 0 places the flock); at least one of `max_iters`/`time_limit` must be given |
| `time_limit` | the deadline, a wall-clock stop in seconds, for when an iteration's cost is too uneven to budget in iterations |
| `budget_check` | an extra gate for a budget the engine cannot see, checked each iteration and before every batch it serves; it is handed the run's `Work`, so it reads the engine's own counts instead of keeping a parallel tally |
| `restart_stalled` | whether a stalled path is re-drawn (default `True`); `False` freezes the flock, and the run ends on `all_stalled` once every path parks |
| `global_model` | the restart-point source and evidence sink; `None` binds `SampleGlobalModel` |
| `local_solver` | the search method; `None` binds `DiagonalPreconditioner` |
| `key` | PRNG seed; `None` uses a fixed default so unseeded runs stay deterministic; ignored when `global_model` is supplied, since that model owns its key |

Four gotchas worth knowing up front:

- **Sides may be unbounded.** A `Domain` side of `(None, None)` draws from
  `N(0, scale²)`, a half-side draws the same normal truncated, and any side can
  take its own injected sampler. There is no uniform prior over an infinite
  side, so the engine never invents one — it falls back to those draws.
- **`eval_f` serves a step's trials value-only; the winner's gradient still
  costs the iteration's one fused `settle` batch.** A method that reads its
  candidates (§6) gets each trial's f cheaply and pays that one gradient pass
  for the winner — so reading more candidates costs serial work, never another
  gradient pass. Without `eval_f` every trial is itself a fused batch and
  already carries its gradient, so the count lands on the `f_g` pair instead of
  the `f` pair.
- **`n_paths = 1` is the single-path baseline.** Same code, one path — and with
  one path the parallelism is gone while the default restart bit still fires:
  every stop becomes a random redraw, which on a unimodal problem is a restart
  into the same basin until the horizon. Pass `restart_stalled=False` for the
  single path that stops where it stalled.
- **The horizon counts iterations, not work.** What one iteration costs is the
  method's business: it serves the redraw starts plus however many trials the
  method's own line search serves, and a flock with nothing left to do ends the
  run (`all_stalled`) instead of entering a batchless iteration. So `max_iters`
  is a deterministic stop, not a cost estimate: read the counters for what the
  run actually did, and pass the deadline (`time_limit`) if wall-clock is what
  you care about. An external budget derived from batches or points, checked
  through `budget_check`, is a ceiling the engine cannot cross: before each
  batch the gate is handed the `Work` that batch would leave, and the batch is
  charged only if the gate allows it. Write the comparison with `<=` — the
  count already includes the batch — and remember that a batch is served whole
  or not at all, so the run meets the ceiling from below, within one batch
  width of it.

## 5. What a batch and a point cost

The two axes answer different questions, and which one matters depends on
whether your objective is really parallel:

- **On a GPU.** A batch's wall-clock barely grows as it widens, so measure
  progress in batches: `n_paths` set to the device's parallel limit makes the
  batch count very nearly your wall-clock. Wider paths are close to free, so
  raising `n_paths` — and buying trial values by cheap value-only passes with
  `eval_f` — shrinks the spread of the best across runs for the same number of
  batches.
- **On a CPU.** Batches run mostly serially, so the point count is the honest axis,
  and width spreads the same point budget thinner across paths instead of speeding
  anything up. The flock advances in lockstep — every path gets one move an
  iteration (a redraw or a step; a stalled path moves nowhere) — so at a fixed
  point budget each path gets about `1/n_paths` of the steps a single path
  would, and best-of-`n_paths` only shrinks the spread
  across starts; it cannot put the missing steps back. Whenever the budget runs
  out before a path converges — the regime this library is for — that lost depth
  dominates, and a single path ends better per point than a wide one.

One sentence to keep them straight: a **batch** is one call (wall-clock, GPU), a
**point** is one row of data (serial cost, CPU) — and a batch is a
`(M, d)` array whose rows are those points. A result reports both counters, so
say which axis you are reading.

## 6. Swap a piece

The engine owns the schedule; the method and the restart-point source are
plugins — the two plugin seams, objects you swap, unlike §1's seams, which were
functions you hand in. Either default can be replaced by passing an instance of
your own class. Continuing §1's example:

```python
from soar_lite import DiagonalPreconditioner, Domain, n_path_schedule

result = n_path_schedule(
    eval_f_g,  # §1's fused seam
    domain=Domain([(-5.0, 5.0), (-5.0, 5.0)]),
    n_paths=4,
    max_iters=30,
    # a shorter whole step: `alpha0` damps the scaled direction
    local_solver=DiagonalPreconditioner(alpha0=0.5),
    restart_stalled=True,
)
```

To replace one outright, implement its protocol and pass the instance:

- **`LocalSolver`** — the search method: `restart`, `propose`, `step`. You own
  direction, step size, acceptance, how many candidates a step looks at and the
  clip of your own step into the domain; the engine owns the horizon, the finite
  gate and the one fused batch that settles what a step keeps. `propose` gets the
  cohort's indices, the run's
  `domain` and `serve_trials`, and answers a winner (a `Pending` point, read or
  unread) or `None` to stop that path. With `eval_f` connected a trial is read
  for its value alone, never for its gradient; without it the same call brings
  both. `restart` is the only cold-start call, so it runs at
  initialization and after every restart; `step` is the only place a landed step
  is announced.
- **`GlobalModel`** — the restart-point source and evidence sink: `observe`,
  `propose`. The shipped pair is `SampleGlobalModel` (draws starts from the
  domain, fits nothing — the default) and `BLRGlobalModel` (fits a Bayesian
  linear posterior and proposes the `-log EI` minimum).

Both protocols are documented in the [API Reference](../reference/api.md), which
is the contract to code against.

Replacing a piece stops at the protocol. Making `BLRGlobalModel` descend its
own acquisition — the engine inside the engine — is a separate job, and has
[its own page](../how-to/surrogate-restarts.md).

## 7. Next

[How it works](../explanation/how-it-works.md) is the derivation: why the
denominator floor is anchored to the gradient, the cost counters, and what this
library does and does not take from SOAR.
[Surrogate restarts](../how-to/surrogate-restarts.md) is the one advanced page:
fitting a model on the evidence, and descending the acquisition it proposes.
