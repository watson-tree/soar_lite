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
keeps the engine's own init (min of two probes, two points, both counted), so
no method gets a start it did not pay for.
"""

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import minimize

from experiments.problems import Problem
from soar_lite import DiagonalPreconditioner, n_path_schedule

Carry = tuple[jnp.ndarray, optax.OptState, jnp.ndarray]


@dataclass
class Trace:
    """Best-so-far trajectory of one run, keyed by both cost axes."""

    points: list[int] = field(default_factory=list)
    batches: list[int] = field(default_factory=list)
    best: list[float] = field(default_factory=list)
    fun: float = math.inf  # final incumbent

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
) -> Trace:
    """Drive the n-path engine, recording best-so-far after every batch.

    The budget is ``max_steps`` **batches** - one round of parallel evaluations,
    the axis a single chain pays one point per step on. The engine charges a
    batch against that budget before it serves it, so the ceiling is exact: a
    probe or a rejected trial batch costs one each, and the run stops early on
    its own convergence criterion.
    """
    trace = Trace()
    n_points = 0
    n_batches = 0
    best = math.inf

    def evaluate_batch(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        nonlocal n_points, n_batches, best
        f, g, _ = problem.evaluate_batch(x)
        n_batches += 1
        n_points += int(x.shape[0])
        for v in f.tolist():
            if math.isfinite(v) and v < best:
                best = v
        trace.points.append(n_points)
        trace.batches.append(n_batches)
        trace.best.append(best)
        return f, g, None

    result = n_path_schedule(
        evaluate_batch,
        bounds=problem.bounds,
        n_paths=n_paths,
        max_steps=max_steps,
        solver=DiagonalPreconditioner(),
        key=jax.random.PRNGKey(seed),
    )
    trace.fun = float(result.fun)
    return trace


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
        f, g, _ = problem.evaluate_batch(x[None, :])
        updates, opt_state = opt.update(g[0], opt_state)
        x = jnp.clip(jnp.asarray(optax.apply_updates(x, updates)), lo, hi)
        fval = jnp.where(jnp.isfinite(f[0]), f[0], jnp.inf)
        best = jnp.minimum(best, fval)
        return (x, opt_state, best), best

    _, bests = jax.lax.scan(
        body, (x0, state, jnp.asarray(jnp.inf)), jnp.arange(1, max_steps + 1)
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
        f, g, _ = problem.evaluate_batch(jnp.asarray(x)[None, :])
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
