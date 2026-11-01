"""Method drivers: soar-lite, Adam, L-BFGS-B, all on one trace format.

Every driver returns a :class:`Trace` - best-so-far against both cost axes:

* ``points`` - objective points served (one per served coordinate vector), the
  fair information unit shared by an n-path method and a single chain;
* ``batches`` - sequential batched calls, the budget unit: for soar-lite one
  served batch (n_paths points at once), for a single-chain method one point.
  This is the wall-clock axis of a batched/GPU setting, where an n-path batch
  costs one sequential unit - the parallel claim lives here, but a batch is not
  free, which is why the points axis is read off the same runs.

All single-chain drivers start from one uniform draw over the box; soar-lite
draws its n_paths starts the same way - uniformly over the box, in the driver -
and hands them to the engine as the entry's `x0`, so no method
gets a start it did not pay for.
"""

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import minimize

from experiments.problems import Problem
from soar_lite import (
    AcquisitionFn,
    DiagonalPreconditioner,
    Domain,
    n_path_schedule,
)

Carry = tuple[jnp.ndarray, optax.OptState, jnp.ndarray]

# Inner horizon of `surrogate_minimize`: one restart-free single-path descent,
# long enough for the acquisition to reach its local minimum from a uniform
# start, short of the outer run's own horizon.
_INNER_ITERS = 40


@dataclass
class Trace:
    """Best-so-far trajectory of one run, keyed by both cost axes."""

    points: list[int] = field(default_factory=list)
    batches: list[int] = field(default_factory=list)
    best: list[float] = field(default_factory=list)
    fun: float = math.inf  # final best

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (points, batches, best) as float arrays for aggregation."""
        return (
            np.asarray(self.points, dtype=float),
            np.asarray(self.batches, dtype=float),
            np.asarray(self.best, dtype=float),
        )


# --- soar-lite: the n-path engine --------------------------------------------


def run_soar_lite(
    problem: Problem,
    *,
    n_paths: int,
    max_steps: int,
    seed: int,
    restart_stalled: bool = False,
) -> Trace:
    """Drive the n-path engine, recording best-so-far after every batch.

    The budget is ``max_steps`` **batches** - one eval of parallel evaluations,
    the axis a single chain pays one point per step on. The engine's own stop
    counts iterations, not batches, so the ceiling is held by a batch gate
    (``budget_check``): a probe or a rejected trial batch costs one each, and
    the run stops within one batch of it. ``max_iters`` is the horizon the
    engine requires - deliberately non-binding, so it never ends the run first.

    ``restart_stalled`` is whether a path the engine marks stalled is re-drawn;
    ``False`` keeps the flock frozen. The n_paths=1 arms leave it off so their
    single chain is the pure serial descent, which is what the panels compare a
    batched run against.

    The flock is drawn here, uniformly over the box, so the experiment's
    distribution is the caller's: the engine only ever places the points it is
    handed. It arrives as the entry's `x0`, which is also why the key is split
    once - the flock's half draws the starts and the other half seeds the
    model's restarts, so the two do not share a stream.
    """
    trace = Trace()
    n_points = 0
    n_batches = 0
    best = math.inf

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        nonlocal n_points, n_batches, best
        f, g, _ = problem.eval_f_g(x)
        n_batches += 1
        n_points += int(x.shape[0])
        for v in f.tolist():
            if math.isfinite(v) and v < best:
                best = v
        trace.points.append(n_points)
        trace.batches.append(n_batches)
        trace.best.append(best)
        return f, g, None

    key, sub = jax.random.split(jax.random.PRNGKey(seed))
    domain = Domain(problem.bounds)
    x0 = jax.random.uniform(
        sub,
        (n_paths, problem.d),
        minval=domain.lo,
        maxval=domain.hi,
    )
    result = n_path_schedule(
        eval_f_g,
        n_paths=n_paths,
        domain=domain,
        x0=x0,
        max_iters=max_steps,
        budget_check=lambda: n_batches < max_steps,
        local_solver=DiagonalPreconditioner(),
        restart_stalled=restart_stalled,
        key=key,
    )
    trace.fun = float(result.fun)
    return trace


def surrogate_minimize(
    objective: AcquisitionFn,
    bounds: jnp.ndarray,
    key: jnp.ndarray,
    n_points: int,
) -> jnp.ndarray:
    """Minimize a batched acquisition with `n_points` single-path engine runs.

    The injected inner minimizer `BLRGlobalModel.propose` asks for: the
    model hands out its own acquisition and this driver answers with the
    library's own descent, one serial path per start - the same restart-free
    recipe the n_paths=1 arms use, so the acquisition is optimized by the method
    it will later drive. Starts are uniform over the box, matching the default
    model's draw; each run stops at `_INNER_ITERS`.

    Args:
        objective: the acquisition to descend, ``(m, d) -> ((m,), (m, d))``:
            values and their per-row gradients, the engine's `EvalFG` shape
            without the feasibility column.
        bounds: the search box, a ``(d, 2)`` array.
        key: the PRNG key the starts are split from.
        n_points: how many local minimizers to return, one per row.

    Returns:
        A ``(n_points, d)`` array, one minimizer per row.

    """
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    domain = Domain(bounds)

    def eval_g(x_batch: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        f, g = objective(x_batch)
        return f, g, None

    outs = []
    for _ in range(n_points):
        key, sub = jax.random.split(key)
        x0 = jax.random.uniform(sub, (1, d), minval=lo, maxval=hi)
        result = n_path_schedule(
            eval_g,
            n_paths=1,
            domain=domain,
            x0=x0,
            max_iters=_INNER_ITERS,
            local_solver=DiagonalPreconditioner(),
            restart_stalled=False,
            key=sub,
        )
        outs.append(result.x)
    return jnp.stack(outs)


# --- Adam: one compiled scan per run --------------------------------


def run_adam(problem: Problem, *, max_steps: int, seed: int, lr: float) -> Trace:
    """Plain Adam, one evaluation per step, the whole loop in one compiled scan."""
    lo, hi = problem.bounds[:, 0], problem.bounds[:, 1]
    _, sub = jax.random.split(jax.random.PRNGKey(seed))
    x0 = jax.random.uniform(sub, (problem.d,), minval=lo, maxval=hi)
    opt = optax.adam(lr)
    state = opt.init(x0)

    def body(carry: Carry, _t: jnp.ndarray) -> tuple[Carry, jnp.ndarray]:
        x, opt_state, best = carry
        f, g, _ = problem.eval_f_g(x[None, :])
        updates, opt_state = opt.update(g[0], opt_state)
        x = jnp.clip(jnp.asarray(optax.apply_updates(x, updates)), lo, hi)
        fval = jnp.where(jnp.isfinite(f[0]), f[0], jnp.inf)
        best = jnp.minimum(best, fval)
        return (x, opt_state, best), best

    _, bests = jax.lax.scan(
        body,
        (x0, state, jnp.asarray(jnp.inf)),
        jnp.arange(1, max_steps + 1),
    )
    values = np.asarray(bests.tolist(), dtype=float)
    grid = np.arange(1, max_steps + 1)
    return Trace(
        points=grid.tolist(),
        batches=grid.tolist(),
        best=values.tolist(),
        fun=float(values[-1]),
    )


# --- L-BFGS-B: the smooth-panel ceiling reference ----------------------------


def run_lbfgs(problem: Problem, *, max_steps: int, seed: int) -> Trace:
    """Scipy L-BFGS-B with jac=True; the serial full-memory reference.

    Not a head-to-head baseline: it is full-memory (not per-coordinate) and
    serial line-search (n_paths=1, serial). It calibrates the smooth panel - how
    much the diagonal approximation costs against an exact-curvature method.
    """
    lo, hi = np.asarray(problem.bounds[:, 0]), np.asarray(problem.bounds[:, 1])
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(lo, hi)
    trace = Trace()
    n_points = 0
    best = math.inf

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal n_points, best
        f, g, _ = problem.eval_f_g(jnp.asarray(x)[None, :])
        fval = float(f[0])
        n_points += 1
        if math.isfinite(fval) and fval < best:
            best = fval
        trace.points.append(n_points)
        trace.batches.append(n_points)
        trace.best.append(best)
        return fval, np.asarray(g[0])

    result = minimize(
        fun,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(lo, hi, strict=True)),
        options={"maxfun": max_steps, "maxiter": max_steps},
    )
    trace.fun = float(best) if math.isfinite(best) else float(result.fun)
    return trace
