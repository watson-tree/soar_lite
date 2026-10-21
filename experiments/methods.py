"""Method drivers: soar-lite, Adam, NAAdam, L-BFGS-B, all on one trace format.

Every driver returns a :class:`Trace` - best-so-far against both cost axes:

* ``points`` - objective points served (one per served coordinate vector), the
  fair information unit shared by an n-path method and a single chain;
* ``rounds`` - sequential batched calls. For soar-lite that is one engine round
  (n_paths points served at once); for a single-chain method it equals ``points``.
  This is the wall-clock axis for a batched/GPU setting, where an n-path
  round costs one sequential unit - the parallelism claim lives here, but it is
  not free, which is why the points axis is the primary one.

NAAdam is Adam plus decaying Gaussian noise, ``sigma_t = sigma0/sqrt(t)``, the
brute-force escape mechanism the restart proposition is compared against. All
single-chain drivers start from one uniform draw over the box; soar-lite keeps
the engine's own init (min of two probes, two points, both counted), so no
method gets a start it did not pay for.
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

Carry = tuple[jnp.ndarray, optax.OptState, jnp.ndarray, jnp.ndarray]


@dataclass
class Trace:
    """Best-so-far trajectory of one run, keyed by both cost axes."""

    points: list[int] = field(default_factory=list)
    rounds: list[int] = field(default_factory=list)
    best: list[float] = field(default_factory=list)
    fun: float = math.inf  # final incumbent

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (points, rounds, best) as float arrays for aggregation."""
        return (
            np.asarray(self.points, dtype=float),
            np.asarray(self.rounds, dtype=float),
            np.asarray(self.best, dtype=float),
        )


# --- soar-lite: the n-path engine --------------------------------------------


def run_soar_lite(
    problem: Problem,
    *,
    n_paths: int,
    max_evals: int,
    seed: int,
) -> Trace:
    """Drive the n-path engine, recording best-so-far after every batch."""
    trace = Trace()
    n_points = 0
    n_rounds = 0
    best = math.inf

    def evaluate_batch(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        nonlocal n_points, n_rounds, best
        f, g, _ = problem.evaluate_batch(x)
        n_rounds += 1
        n_points += int(x.shape[0])
        for v in f.tolist():
            if math.isfinite(v) and v < best:
                best = v
        trace.points.append(n_points)
        trace.rounds.append(n_rounds)
        trace.best.append(best)
        return f, g, None

    result = n_path_schedule(
        evaluate_batch,
        bounds=problem.bounds,
        n_paths=n_paths,
        max_steps=max_evals,
        budget_check=lambda: n_points < max_evals,
        solver=DiagonalPreconditioner(),
        key=jax.random.PRNGKey(seed),
    )
    trace.fun = float(result.fun)
    return trace


# --- Adam / NAAdam: one compiled scan per run --------------------------------


def run_adam(problem: Problem, *, max_evals: int, seed: int, lr: float) -> Trace:
    """Plain Adam, one evaluation per step."""
    return _single_chain(problem, max_evals=max_evals, seed=seed, lr=lr, sigma0=0.0)


def run_naadam(
    problem: Problem,
    *,
    max_evals: int,
    seed: int,
    lr: float,
    sigma0: float = 1e-2,
) -> Trace:
    """Adam plus decaying Gaussian noise (the brute-force escape baseline)."""
    return _single_chain(problem, max_evals=max_evals, seed=seed, lr=lr, sigma0=sigma0)


def _single_chain(
    problem: Problem,
    *,
    max_evals: int,
    seed: int,
    lr: float,
    sigma0: float,
) -> Trace:
    """Adam, plus noise when ``sigma0 > 0``; the whole loop runs in one scan."""
    lo, hi = problem.bounds[:, 0], problem.bounds[:, 1]
    key = jax.random.PRNGKey(seed)
    key, sub = jax.random.split(key)
    x0 = jax.random.uniform(sub, (problem.d,), minval=lo, maxval=hi)
    opt = optax.adam(lr)
    state = opt.init(x0)

    def body(carry: Carry, t: jnp.ndarray) -> tuple[Carry, jnp.ndarray]:
        x, opt_state, best, noise_key = carry
        f, g, _ = problem.evaluate_batch(x[None, :])
        updates, opt_state = opt.update(g[0], opt_state)
        x = jnp.asarray(optax.apply_updates(x, updates))
        x = jnp.clip(x, lo, hi)
        if sigma0 > 0.0:
            noise_key, nk = jax.random.split(noise_key)
            noise = (sigma0 / jnp.sqrt(t)) * jax.random.normal(nk, x.shape)
            x = jnp.clip(x + noise, lo, hi)
        fval = jnp.where(jnp.isfinite(f[0]), f[0], jnp.inf)
        best = jnp.minimum(best, fval)
        return (x, opt_state, best, noise_key), best

    _, bests = jax.lax.scan(
        body, (x0, state, jnp.asarray(jnp.inf), key), jnp.arange(1, max_evals + 1)
    )
    values = np.asarray(bests.tolist(), dtype=float)
    grid = np.arange(1, max_evals + 1)
    return Trace(
        points=grid.tolist(),
        rounds=grid.tolist(),
        best=values.tolist(),
        fun=float(values[-1]),
    )


# --- L-BFGS-B: the smooth-panel ceiling reference ----------------------------


def run_lbfgs(problem: Problem, *, max_evals: int, seed: int) -> Trace:
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
        trace.rounds.append(n_points)
        trace.best.append(best)
        return fval, np.asarray(g[0])

    result = minimize(
        fun,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(lo, hi, strict=True)),
        options={"maxfun": max_evals, "maxiter": max_evals},
    )
    trace.fun = float(best) if math.isfinite(best) else float(result.fun)
    return trace
