"""n-path scheduler self-checks.

Seam (拍板): the engine consumes the flock, callables, and the box it searches

    eval_f_g(x (M,d)) -> (f_vals (M,), grads_x (M,d), feas (M,)|None)
    budget_check() -> bool
    bounds -> scipy-style (lb, ub) pairs | (d, 2) array | None (no box)
    x0 (n_paths, d) -> the flock; the sole authority on n_paths and d
    global_model(key, n_points) -> (n_points, d) restart points; not consulted
        unless a restart rule fires

The caller composes x->θ and the chain rule grad_x = grad_θ * dθ_dx(x) into
eval_f_g; the engine owns clip-to-bounds, and the flock is the caller's - with
the arguments in the order a run reads them: `x0` first, then the rule that
decides when to leave a point and the model that decides where to go. With
`global_model=None` the engine falls back to i.i.d. uniform over a finite
`bounds`, so an unbounded box (or none) without a model is a wiring error.
Unbounded means a `None`/±inf side, and `clip` is then the identity.
`test_solver_seam.py`'s evaluator is that composed form.
"""

# pyright: reportPrivateUsage=false
# This suite is the spec for two `_`-private engine helpers; importing them
# here is the intended read, not a boundary crossing.

import math
from collections.abc import Callable
from typing import TypeAlias

import jax
import jax.numpy as jnp

from soar_lite.engine import (
    NPathResult,
    PathPoint,
    Verdict,
    _argmin_finite,
    n_path_schedule,
)
from soar_lite.restart import NoRestartRule, StallRestartRule
from soar_lite.solver import DiagonalPreconditioner

jax.config.update("jax_enable_x64", val=True)

Pointwise: TypeAlias = Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]


def unit_bounds(d: int = 1) -> jnp.ndarray:
    """Return the engine's box in working coords: x ∈ [0,1]^d."""
    return jnp.stack([jnp.zeros(d), jnp.ones(d)], axis=1)


def flock(n_paths: int, d: int = 1) -> jnp.ndarray:
    """Return `n_paths` deterministic starting points inside [0,1]^d.

    Spread evenly and off the walls rather than drawn: these are the engine's
    tests, so the starts must not depend on a PRNG the caller now owns, and an
    interior row keeps a steepest-descent step from being pinned at a face
    before the test has said anything.
    """
    row = jnp.linspace(0.0, 1.0, n_paths + 2)[1:-1]
    return jnp.repeat(row[:, None], d, axis=1)


class HalfStep:
    """A deterministic local method: half a steepest-descent step, strict decrease.

    These tests are the engine's, so they must not depend on which local method
    is plugged in - the engine owns the paths, the box, the shared budget and the
    EI cadence, and the method owns only direction, step size, acceptance and
    how long its own enumeration runs. This one is deliberately naive: it
    proposes `x - 0.5*g` and accepts only a strict decrease, giving up after
    `trial_cap` rejects, so a path stalls for an engine reason (a rejected trial,
    typically one pinned at the box wall) and never because a curvature model
    went quiet. What the real method does is `test_solver.py`'s subject.
    """

    def __init__(self, trial_cap: int = 20) -> None:
        """Bound one step's enumeration at `trial_cap` rejects (the method's own)."""
        self.trial_cap = trial_cap
        self.n_rejects: dict[int, int] = {}

    def restart(self, i: int, p: PathPoint) -> None:  # noqa: ARG002
        """Take no per-path state beyond the reject counter, which restarts."""
        self.n_rejects[i] = 0

    def propose(self, i: int, p: PathPoint) -> jnp.ndarray:  # noqa: ARG002
        """Return half a steepest-descent step from p."""
        return p.x - 0.5 * p.g

    def needs_verdict(self, i: int) -> bool:  # noqa: ARG002
        """Rule every trial: the strict-decrease test is this method's acceptance."""
        return True

    def verdict(self, i: int, p: PathPoint, x: jnp.ndarray, f: float) -> Verdict:  # noqa: ARG002
        """Accept a strict decrease; give up once the trial cap is spent."""
        if f < p.f:
            self.n_rejects[i] = 0  # the cap is per step, like an enumeration
            return "accept"
        self.n_rejects[i] = self.n_rejects.get(i, 0) + 1
        return "retry" if self.n_rejects[i] < self.trial_cap else "give_up"

    def fold(self, i: int, p: PathPoint, best: PathPoint) -> None:  # noqa: ARG002
        """Fold nothing: this method keeps no model."""
        return


class Recorder:
    """Wraps a pointwise (f, grad) fn as the batched seam; records everything."""

    def __init__(self, pointwise: Pointwise) -> None:
        """Map a single point θ (d,) to (f, grad_θ (d,))."""
        self.pointwise = pointwise
        self.all_f_vals: list[float] = []  # every f value ever returned, per point
        self.batch_shapes: list[tuple[int, int]] = []  # shape of each seam call
        self.evals = 0

    def __call__(self, theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        """Assert the (M,d) batch shape, record it, and vmap the pointwise fn."""
        assert theta.ndim == 2, f"eval_f_g expects (M,d) batch, got {theta.shape}"
        self.batch_shapes.append((int(theta.shape[0]), int(theta.shape[1])))
        f_vals, grads = jax.vmap(self.pointwise)(theta)
        self.all_f_vals.extend(float(v) for v in f_vals)
        self.evals += int(theta.shape[0])
        return f_vals, grads, None

    def limit_check(self, limit: int) -> Callable[[], bool]:
        """Return a batch-level budget_check() False once `limit` evals served.

        The core consults it before each batch (and once at the head of each
        iteration), it reads the evals already served, and the terminal batch
        therefore over-runs by less than one batch width.
        """

        def check() -> bool:
            return self.evals < limit

        return check


def test_batch_shape_and_point_counting() -> None:
    """eval_f_g receives (n_paths,d) batches; each point counts exactly 1 eval.

    reported evals == points the seam actually served.
    """

    def quartic(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # A quartic, not a quadratic: one half-step of steepest descent lands a
        # 1-D quadratic exactly on its minimiser, where the gradient is zero and
        # every further trial is rejected - the run would stall long before the
        # eval budget is spent, and this test is about the budget accounting.
        d = theta - 0.7
        return jnp.sum(d**4 + d**2), 4.0 * d**3 + 2.0 * d

    rec = Recorder(quartic)
    res = n_path_schedule(
        rec,
        x0=flock(4),
        bounds=unit_bounds(),
        max_iters=50,
        budget_check=rec.limit_check(500),
        solver=HalfStep(),
        key=jax.random.PRNGKey(0),
    )
    assert isinstance(res, NPathResult)
    assert res.n_points == rec.evals, (res.n_points, rec.evals)
    assert res.n_points <= 500
    # n_iters counts engine iterations, not the trials inside them (the method's
    # own trial cap bounds one iteration), so it stays far below the eval
    # count and never passes the horizon.
    assert 0 < res.n_iters <= 50
    # With no rule firing, a path serves at most one g point per iteration, and
    # every g batch serves at least one: the two counts bracket each other.
    assert res.n_points <= res.n_iters * 4, (res.n_points, res.n_iters)
    assert res.n_batches <= res.n_points, (res.n_batches, res.n_points)
    # batched concurrency: at least one call served >1 point at once
    assert any(m > 1 for (m, _d) in rec.batch_shapes), rec.batch_shapes
    # all batches carry the same width d
    assert all(d == 1 for (_m, d) in rec.batch_shapes)


def test_budget_exhaustion_stops_everything() -> None:
    """budget_check() False -> global stop; terminal batch over-runs < n_paths."""

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum((theta - 0.7) ** 2), 2.0 * (theta - 0.7)

    rec = Recorder(quad)
    res = n_path_schedule(
        rec,
        x0=flock(8),
        bounds=unit_bounds(),
        max_iters=100,
        budget_check=rec.limit_check(30),
        solver=HalfStep(),
        key=jax.random.PRNGKey(1),
    )
    assert res.n_points == rec.evals
    assert 30 <= res.n_points < 30 + 8  # at most one batch past the limit
    assert 0 < res.n_iters < 100  # the gate stopped it, not the horizon
    assert res.status == "budget_exhausted"

    # gate already False on entry: nothing evaluated at all
    rec0 = Recorder(quad)
    res0 = n_path_schedule(
        rec0,
        x0=flock(4),
        bounds=unit_bounds(),
        max_iters=100,
        budget_check=rec0.limit_check(0),
        solver=HalfStep(),
        key=jax.random.PRNGKey(2),
    )
    assert res0.n_points == 0
    assert res0.n_iters == 0  # never got past the first gate check
    assert res0.status == "budget_exhausted"

    # Budget for iteration 0 only: it completes, so the flock is placed and
    # n_iters is 1 - a refused first step is distinguishable from a refused
    # start (n_iters = 0 above).
    rec1 = Recorder(quad)
    res1 = n_path_schedule(
        rec1,
        x0=flock(4),
        bounds=unit_bounds(),
        max_iters=1,
        solver=HalfStep(),
        key=jax.random.PRNGKey(3),
    )
    assert res1.n_points == 4  # iteration 0 served every path
    assert res1.n_iters == 1
    assert res1.status == "max_iters"


def test_fun_is_true_min_over_history() -> None:
    """Pin that fun is the min over every finite f the seam returned.

    Not the final point, not a compressed history.
    """

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum((theta - 0.7) ** 2), 2.0 * (theta - 0.7)

    rec = Recorder(quad)
    res = n_path_schedule(
        rec,
        x0=flock(4),
        bounds=unit_bounds(),
        max_iters=50,
        budget_check=rec.limit_check(500),
        solver=HalfStep(),
        key=jax.random.PRNGKey(3),
    )
    assert rec.all_f_vals
    assert res.fun == min(rec.all_f_vals)
    # 0.7 is the true minimizer; the engine should land essentially on it
    assert res.fun < 1e-8


_TWO_WELL_SIGMA = 0.04


def _two_well_1d(theta: jnp.ndarray) -> jnp.ndarray:
    """Two-well 1-D: global -8 @ θ=0.38, local -2 @ θ=0.62, saddle near 0.505."""
    return -2.0 * jnp.exp(-(((theta - 0.62) / _TWO_WELL_SIGMA) ** 2)) - 8.0 * jnp.exp(
        -(((theta - 0.38) / _TWO_WELL_SIGMA) ** 2)
    )


def _two_well_grad(theta: jnp.ndarray) -> jnp.ndarray:
    """Return d/dθ of `_two_well_1d`."""
    u_l, u_g = (theta - 0.62) / _TWO_WELL_SIGMA, (theta - 0.38) / _TWO_WELL_SIGMA
    return (
        4.0 * u_l * jnp.exp(-(u_l**2)) / _TWO_WELL_SIGMA
        + 16.0 * u_g * jnp.exp(-(u_g**2)) / _TWO_WELL_SIGMA
    )


def _two_well(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pointwise (f, ∇f) on the two-well landscape, θ (d,) -> f scalar."""
    return _two_well_1d(theta[0]), _two_well_grad(theta)


def test_k_start_beats_single_path_on_multimodal() -> None:
    """Equal eval budget: 16 starts strictly beat 1 start on a two-well 1-D.

    Global well -8 @ θ=0.38, local well -2 @ θ=0.62, saddle ~ 0.505 (the deep
    well is 4x deeper, so its basin reaches just past mid-domain). The identity
    map makes θ = x, so the single path is placed in the shallow well and the
    sixteen are spread across the box; both arms run the shipped method and are
    pinned to `NoRestartRule`, so the only difference is the start count. What a
    rule adds on top of width is the next test's subject.
    """
    # independent literal grid check: global min is -8 (brute force)
    grid = jnp.linspace(0.05, 0.95, 901)
    grid_min = float(jnp.min(jax.vmap(_two_well_1d)(grid)))
    assert abs(grid_min - (-8.0)) < 0.15

    budget = 320
    rec_k = Recorder(_two_well)
    res_k = n_path_schedule(
        rec_k,
        x0=flock(16),
        bounds=unit_bounds(),
        max_iters=100,
        budget_check=rec_k.limit_check(budget),
        solver=DiagonalPreconditioner(),
        restart_rule=NoRestartRule(),
        key=jax.random.PRNGKey(2),
    )
    rec_1 = Recorder(_two_well)
    res_1 = n_path_schedule(
        rec_1,
        x0=jnp.array([[0.62]]),  # planted in the shallow well
        bounds=unit_bounds(),
        max_iters=100,
        budget_check=rec_1.limit_check(budget),
        solver=DiagonalPreconditioner(),
        restart_rule=NoRestartRule(),
        key=jax.random.PRNGKey(2),
    )
    assert res_1.fun > -3.0, res_1.fun  # stuck near local well (-2)
    assert res_k.fun < -7.0, res_k.fun  # found the global well (-8)
    assert res_k.fun < res_1.fun  # strictly better, equal budget


def test_the_stall_rule_escapes_a_shallow_basin() -> None:
    """Same starts, same budget, one rule swapped: the rule is what escapes.

    Four starts planted in the shallow well (-2 @ θ=0.62) and the same method in
    both arms, so every step is identical until the first verdict. The frozen
    flock descends to the bottom of the well it was given and idles there; the
    stall rule reads a converged path as no progress, fires once its window is
    full, and the engine's default global model re-draws it - some draw lands in
    the deep basin (-8 @ θ=0.38). Both runs end on the budget, not the horizon,
    so the comparison is equal work.
    """
    budget = 320
    x0_shallow = jnp.array([[0.60], [0.64], [0.58], [0.66]])

    def run(rule: NoRestartRule | StallRestartRule) -> NPathResult:
        rec = Recorder(_two_well)
        return n_path_schedule(
            rec,
            x0=x0_shallow,
            bounds=unit_bounds(),
            max_iters=100,
            budget_check=rec.limit_check(budget),
            solver=DiagonalPreconditioner(),
            restart_rule=rule,
            key=jax.random.PRNGKey(9),
        )

    frozen = run(NoRestartRule())
    revived = run(StallRestartRule(window=10))
    assert frozen.status == "budget_exhausted", frozen
    assert revived.status == "budget_exhausted", revived
    assert frozen.fun > -3.0, frozen.fun  # four shallow starts, all stuck
    assert revived.fun < -7.0, revived.fun  # a re-draw reached the deep basin
    assert revived.fun < frozen.fun


def test_projected_descent_reaches_wall_optimum() -> None:
    """Clip-in-backtracking lands on a wall optimum (projected descent).

    f(θ) = (θ0 - 1.5)^2 + θ1^2 on θ ∈ [0,1]^2: the unconstrained min sits
    outside the box, so the constrained min is (1, 0) on the θ0=1 face with
    f = 0.25. Guards the projection-with-backtracking contract: the trial is
    clipped into [0,1] and the search still reaches the boundary point (a
    regression here would stall x1 away from 0).
    """

    def wall_quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return (theta[0] - 1.5) ** 2 + theta[1] ** 2, jnp.array(
            [2.0 * (theta[0] - 1.5), 2.0 * theta[1]]
        )

    rec = Recorder(wall_quad)
    res = n_path_schedule(
        rec,
        x0=flock(1, 2),
        bounds=unit_bounds(2),
        restart_rule=NoRestartRule(),  # pure projected descent, no restarts
        max_iters=100,
        budget_check=rec.limit_check(500),
        solver=HalfStep(),
        key=jax.random.PRNGKey(4),
    )
    assert res.fun < 0.4, res.fun  # must slide near (1, 0), f=0.25


class _RecordingBest:
    """A rule that logs the raw best each verdict is handed, and never fires.

    Recording only: the run is `NoRestartRule`'s plus a log of what `f⋆` each
    verdict read, which is how the engine's reporting can be told apart from its
    trajectory.
    """

    def __init__(self) -> None:
        """Start with an empty log."""
        self.bs: list[float] = []

    def observe_step(self, i: int, delta: float | None) -> None:
        """No-op: the log is of verdicts, not of steps."""

    def should_restart(self, i: int, f_k: float, best_fun_raw: float) -> bool:  # noqa: ARG002
        """Log the raw best the verdict reads, then keep the path."""
        self.bs.append(best_fun_raw)
        return False


def test_feasible_only_best() -> None:
    """The reported best (`fun`) is the feasible min; `f⋆` is raw.

    f = (θ-0.2)^2 has its global min at θ=0.2, which the feas mask marks
    infeasible (θ <= 0.5); the best feasible design is θ=0.5+ (f 0.09). The
    engine must report the feasible min even though the raw min over all evals
    sits lower (in the infeasible basin).
    """
    thr = 0.5

    def pointwise(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return (theta[0] - 0.2) ** 2.0, 2.0 * (theta - 0.2)

    all_f_vals: list[float] = []
    all_feas: list[bool] = []
    evals = 0

    def eval_f_g(
        theta: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        nonlocal evals
        f_vals, grads = jax.vmap(pointwise)(theta)
        feas = theta[:, 0] > thr
        all_f_vals.extend(float(v) for v in f_vals)
        all_feas.extend(bool(v) for v in feas)
        evals += int(theta.shape[0])
        return f_vals, grads, feas

    def budget_check() -> bool:
        return evals < 600

    rule = _RecordingBest()
    res = n_path_schedule(
        eval_f_g,
        x0=flock(4),
        bounds=unit_bounds(),
        max_iters=100,
        budget_check=budget_check,
        solver=HalfStep(),
        restart_rule=rule,
        key=jax.random.PRNGKey(5),
    )

    feasible_f_vals = [L for L, ok in zip(all_f_vals, all_feas, strict=True) if ok]
    assert feasible_f_vals, "no feasible point ever served"
    # the raw min sits in the infeasible basin, below every feasible point
    assert min(all_f_vals) < min(feasible_f_vals)
    # the reported answer is the best feasible point, never the raw min
    assert res.fun == min(feasible_f_vals)
    assert res.x[0] > thr  # the reported best lies on the feasible side
    # ... and the verdict reads the raw min: an `f⋆` below every feasible value
    # can only have come from an infeasible point.
    assert rule.bs, "a verdict was never asked"
    assert min(rule.bs) < min(feasible_f_vals)


def test_a_feasible_minus_inf_ends_the_run_on_that_point() -> None:
    """A served feasible -inf is the answer, not a value to compare.

    The objective has no floor above `floor`, and the four starts spread across
    the box, so iteration 0 serves one of them in that region. Monotone descent
    cannot beat -inf, so spending the horizon after it would be pure waste - and
    dropping it from `fun` (`_argmin_finite` requires a finite value) would
    report a larger value than the run had already served.
    """
    floor = 0.7

    def eval_f_g(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        f_vals = jnp.where(theta[:, 0] > floor, -jnp.inf, (theta[:, 0] - 0.2) ** 2.0)
        return f_vals, 2.0 * (theta - 0.2), None

    res = n_path_schedule(
        eval_f_g,
        x0=jnp.linspace(0.05, 0.95, 4)[:, None],
        bounds=unit_bounds(),
        max_iters=50,
        solver=HalfStep(),
        restart_rule=NoRestartRule(),
        key=jax.random.PRNGKey(0),
    )

    assert res.fun == -math.inf
    assert res.status == "unbounded"
    assert res.x[0] > floor  # the point that priced it, not a later one
    assert res.n_iters == 0  # stopped inside iteration 0: none completed


def test_an_infeasible_minus_inf_does_not_end_the_run() -> None:
    """Only a feasible -inf is the answer; an infeasible one is just evidence.

    Same floor, but the mask puts it outside the feasible set, so the -inf
    points are served and must change nothing: the run keeps searching and
    reports a finite feasible `fun`. Feasibility is what separates "an answer no
    later trial can beat" from "a trial the caller cannot use", and it is the
    reason the stop lives in the engine rather than in the restart rule, which
    sees unfiltered group-min samples.
    """
    floor = 0.7
    served: list[float] = []

    def eval_f_g(
        theta: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        f_vals = jnp.where(theta[:, 0] > floor, -jnp.inf, (theta[:, 0] - 0.2) ** 2.0)
        served.extend(float(v) for v in f_vals)
        return f_vals, 2.0 * (theta - 0.2), theta[:, 0] <= floor

    res = n_path_schedule(
        eval_f_g,
        x0=jnp.linspace(0.05, 0.95, 4)[:, None],
        bounds=unit_bounds(),
        max_iters=20,
        solver=HalfStep(),
        restart_rule=NoRestartRule(),
        key=jax.random.PRNGKey(0),
    )

    assert any(v == -math.inf for v in served), "no -inf point was ever served"
    assert res.status == "max_iters"
    assert math.isfinite(res.fun)  # the answer stays on the feasible side
    assert res.x[0] <= floor


def test_nan_first_candidate_does_not_mask_batch_min() -> None:
    """A NaN-ranked start must not swallow the batch's real best point.

    Iteration 0 serves [NaN, 0.01] (both feasible), so the engine must report
    best_f == 0.01. `min(candidates, key=f_vals_list.__getitem__)` kept the NaN
    (NaN < x is False): the batch's real minimum - and its x - was discarded and
    best_f stayed inf.
    """
    served = [0]

    def eval_f_g(
        theta: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        served[0] += int(theta.shape[0])
        row0 = jnp.arange(theta.shape[0]) == 0
        nan = jnp.asarray(jnp.nan, dtype=theta.dtype)
        f_vals = jnp.where(row0, nan, jnp.asarray(0.01, dtype=theta.dtype))
        return f_vals, jnp.zeros_like(theta), jnp.ones(theta.shape[0], dtype=bool)

    res = n_path_schedule(
        eval_f_g,
        x0=flock(2),
        bounds=unit_bounds(),
        max_iters=5,
        budget_check=lambda: served[0] < 40,
        solver=HalfStep(),
        key=jax.random.PRNGKey(6),
    )
    assert res.fun == 0.01, res.fun
    assert bool(jnp.isfinite(res.x[0])), res.x


def test_the_best_skips_nonfinite_values() -> None:
    """`_argmin_finite` is how `fun` ignores the non-finite points it served."""
    assert _argmin_finite([2.0, 1.0]) == 1
    assert _argmin_finite([math.nan, 0.01, math.inf]) == 1
    assert _argmin_finite([math.nan, math.inf]) is None


def test_a_dead_path_serves_nothing_after_its_start() -> None:
    """A start whose served values are non-finite stalls on arrival.

    It has no anchor to descend from, so it must not burn budget sitting there:
    under `NoRestartRule` a dead row is served once, in iteration 0's fused
    batch, and never again, while the live row costs exactly one point per
    iteration.
    """
    n_paths, n_iterations = 2, 3
    widths: list[int] = []

    def eval_f_g(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        widths.append(int(theta.shape[0]))
        d = theta - 0.7
        f_vals = jnp.sum(d**4 + d**2, axis=1)
        grads = 4.0 * d**3 + 2.0 * d
        dead = jnp.arange(theta.shape[0]) == 0  # row 0 never comes back finite
        return (
            jnp.where(dead, jnp.inf, f_vals),
            jnp.where(dead[:, None], jnp.nan, grads),
            None,
        )

    res = n_path_schedule(
        eval_f_g,
        x0=flock(n_paths),
        bounds=unit_bounds(),
        max_iters=n_iterations + 1,
        budget_check=lambda: len(widths) < 100,
        solver=HalfStep(),
        restart_rule=NoRestartRule(),
        key=jax.random.PRNGKey(0),
    )
    assert res.n_iters == n_iterations + 1, res
    assert widths[0] == n_paths, widths  # iteration 0 serves the whole flock
    assert all(w == 1 for w in widths[1:]), widths  # then the live path alone


def test_a_dead_start_is_redrawn_by_a_restart() -> None:
    """A rule that fires on a dead path recovers it; a frozen flock cannot.

    The dead row is dead only in iteration 0, so the path recovers as soon as it
    is given a new start: `StallRestartRule` reads the stalls as no progress,
    fires once its window is full, the engine draws the start from the default
    global model (uniform over the box) and the path descends. `NoRestartRule`
    is the control on the same objective - same dead start, no re-draw, fun
    stays inf - which is what makes the recovery the verdict's doing.
    """

    def make_evaluator() -> Callable[
        [jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, None]
    ]:
        calls = [0]

        def eval_f_g(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
            calls[0] += 1
            if calls[0] == 1:  # iteration 0's batch: nothing finite
                nan = jnp.full((theta.shape[0],), jnp.nan)
                return nan, jnp.full_like(theta, jnp.nan), None
            d = theta - 0.7
            return jnp.sum(d**2, axis=1), 2.0 * d, None

        return eval_f_g

    def run(rule: NoRestartRule | StallRestartRule) -> NPathResult:
        return n_path_schedule(
            make_evaluator(),
            x0=flock(1),
            bounds=unit_bounds(),
            max_iters=20,
            solver=DiagonalPreconditioner(),
            restart_rule=rule,
            key=jax.random.PRNGKey(0),
        )

    revived = run(StallRestartRule(window=2))
    assert math.isfinite(revived.fun), revived
    assert run(NoRestartRule()).fun == math.inf


def test_a_step_serves_as_many_trials_as_the_method_asks_for() -> None:
    """The engine has no trial cap: the enumeration's length is the method's.

    A method that wants 30 rejects before its first acceptance must get all 31
    trials in one iteration. An engine-side cap would have stalled the path
    mid-errand, turning a converging step into a stall that only a restart could
    undo - a stopping rule the method never asked for.
    """
    n_rejects = 30

    class Patient:
        """Propose uphill until the 30th reject, then the descent step."""

        def __init__(self) -> None:
            self.calls: dict[int, int] = {}

        def restart(self, i: int, p: PathPoint) -> None:  # noqa: ARG002
            self.calls[i] = 0

        def propose(self, i: int, p: PathPoint) -> jnp.ndarray:
            self.calls[i] = self.calls.get(i, 0) + 1
            return p.x + p.g if self.calls[i] <= n_rejects else p.x - p.g

        def needs_verdict(self, i: int) -> bool:  # noqa: ARG002
            """Rule every trial: this stub's whole point is a long enumeration."""
            return True

        def verdict(self, i: int, p: PathPoint, x: jnp.ndarray, f: float) -> Verdict:  # noqa: ARG002
            return "accept" if f < p.f else "retry"

        def fold(self, i: int, p: PathPoint, best: PathPoint) -> None:  # noqa: ARG002
            return

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum(theta**2), 2.0 * theta

    rec = Recorder(quad)
    res = n_path_schedule(
        rec,
        x0=flock(1),  # one path: a single batch per trial keeps the count exact
        bounds=unit_bounds(),
        max_iters=2,  # iteration 0 places x0, then this one step
        solver=Patient(),
        key=jax.random.PRNGKey(11),
    )
    assert len(rec.batch_shapes) == 1 + n_rejects + 1, rec.batch_shapes  # x0 + trials
    assert res.n_iters == 2, res
    assert rec.all_f_vals[-1] < rec.all_f_vals[0], rec.all_f_vals  # it committed


def test_bounds_spellings_are_the_same_box() -> None:
    """scipy-style pairs and the (d, 2) array run byte-identically.

    One key stream, one flock: if the two spellings normalized differently, the
    clips would land elsewhere and every recorded f would differ. This is also
    what pins both spellings to one normalization: the (d, 2) array and the pair
    list are the same box.
    """

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum((theta - 0.7) ** 2), 2.0 * (theta - 0.7)

    def run(
        bounds: jnp.ndarray | list[tuple[float, float]],
    ) -> tuple[list[float], float]:
        rec = Recorder(quad)
        res = n_path_schedule(
            rec,
            x0=flock(4, 2),
            bounds=bounds,
            budget_check=rec.limit_check(80),
            max_iters=10,
            solver=HalfStep(),
            key=jax.random.PRNGKey(3),
        )
        return rec.all_f_vals, res.fun

    f_vals_arr, fun_arr = run(unit_bounds(2))
    f_vals_pairs, fun_pairs = run([(0.0, 1.0)] * 2)
    assert f_vals_pairs == f_vals_arr
    assert fun_pairs == fun_arr


def test_unbounded_or_absent_box_needs_a_global_model() -> None:
    """global_model=None can only draw a finite box; anything else is an error.

    An unbounded side has no distribution to draw from unless the caller brings
    one, so the engine refuses instead of inventing a box (or a d) of its own.
    One-sided is the quiet case: half the box still exists, so the run would
    silently be a different algorithm than the one the caller thinks it wired.
    The check is eager - a model that is never asked is still required, because
    the rule can change without the box changing.
    """

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum(theta**2), 2.0 * theta

    for bounds in ([(-jnp.inf, 1.0)], [(None, None)], None):
        rec = Recorder(quad)
        try:
            n_path_schedule(
                rec,
                x0=flock(2),
                bounds=bounds,
                max_iters=1,
                solver=HalfStep(),
                key=jax.random.PRNGKey(0),
            )
        except ValueError:
            pass
        else:  # pragma: no cover - the raise is the assertion
            msg = f"no global model, yet {bounds} was accepted"
            raise AssertionError(msg)
        assert rec.evals == 0  # rejected before any point was served


def test_the_flock_is_the_only_authority_on_d() -> None:
    """A bare None box + a flock is a well-posed run: d is the flock's width.

    `bounds=None` and an all-unbounded pair list are the same box (∓inf), so
    they must also be the same run.
    """

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum((theta - 0.7) ** 2), 2.0 * (theta - 0.7)

    def draw(sub: jnp.ndarray, n: int) -> jnp.ndarray:
        return jax.random.logistic(sub, (n, 3))

    rec_none = Recorder(quad)
    res = n_path_schedule(
        rec_none,
        x0=draw(jax.random.PRNGKey(4), 2),
        bounds=None,
        global_model=draw,
        budget_check=rec_none.limit_check(60),
        max_iters=3,
        solver=HalfStep(),
        key=jax.random.PRNGKey(4),
    )
    assert all(d == 3 for _m, d in rec_none.batch_shapes), rec_none.batch_shapes
    assert res.x.shape == (3,)

    rec_pairs = Recorder(quad)
    n_path_schedule(
        rec_pairs,
        x0=draw(jax.random.PRNGKey(4), 2),
        bounds=[(None, None)] * 3,
        global_model=draw,
        budget_check=rec_pairs.limit_check(60),
        max_iters=3,
        solver=HalfStep(),
        key=jax.random.PRNGKey(4),
    )
    assert rec_pairs.all_f_vals == rec_none.all_f_vals

    # a flock and a box that disagree on d is the wiring bug this catches
    try:
        n_path_schedule(
            Recorder(quad),
            x0=flock(2, 3),
            bounds=unit_bounds(2),
            global_model=draw,
            max_iters=1,
            solver=HalfStep(),
            key=jax.random.PRNGKey(4),
        )
    except ValueError:
        pass
    else:  # pragma: no cover - the raise is the assertion
        msg = "a d=3 flock was accepted against a d=2 box"
        raise AssertionError(msg)


def test_an_unbounded_box_is_really_unbounded() -> None:
    """Unbounded means unbounded: a minimizer outside [0,1] is reached.

    θ* = 3.0 sits outside that box, so a clipped finite-box run could never
    serve it; with ∓inf bounds the clip is the identity and the descent is free
    to land on it.
    """
    seen: list[float] = []  # max |x| per served batch

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        seen.append(float(jnp.max(jnp.abs(x))))
        return jnp.sum((x - 3.0) ** 2, axis=-1), 2.0 * (x - 3.0), None

    def draw(sub: jnp.ndarray, n: int) -> jnp.ndarray:
        return jax.random.logistic(sub, (n, 1)) + 3.0  # starts near the well

    res = n_path_schedule(
        eval_f_g,
        x0=draw(jax.random.PRNGKey(5), 4),
        bounds=[(None, None)],
        global_model=draw,
        budget_check=lambda: True,
        max_iters=40,
        solver=HalfStep(),
        key=jax.random.PRNGKey(5),
    )
    assert max(seen) > 1.0, seen  # a finite [0,1] box could not have gone there
    assert res.fun < 1e-6, res.fun


def test_out_of_range_knobs_are_rejected_at_the_entry() -> None:
    """A bad budget knob fails once, at the entry, naming the knob.

    None of these shapes can run: zero paths advance nothing, and a negative
    horizon means no iteration at all. The caller should see the knob, not an
    empty result later.
    """

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum(theta**2), 2.0 * theta

    def rejects(**knobs: float) -> str:
        try:
            # the knob under test is deliberately out of contract
            n_path_schedule(
                Recorder(quad),
                x0=flock(1),
                bounds=[(0.0, 1.0)],
                **knobs,  # pyright: ignore[reportArgumentType]
            )
        except ValueError as e:
            return str(e)
        msg = f"accepted: {knobs}"
        raise AssertionError(msg)

    assert "max_iters" in rejects(max_iters=-1)
    assert "time_limit" in rejects(time_limit=0.0)
    # no horizon at all is refused too, and the message names both ways to stop
    assert "max_iters" in rejects()


def test_a_malformed_flock_is_rejected_at_the_entry() -> None:
    """The flock is validated where it is handed over, not at the first clip.

    Four defects, one place: not a batch, no paths, no dims, and a non-finite
    start. The fifth is the box agreement - d is the flock's second axis, so a
    flock that disagrees with the box is the wiring bug, not a run to guess at.
    """

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum(theta**2), 2.0 * theta

    def rejects(x0: jnp.ndarray) -> str:
        try:
            n_path_schedule(Recorder(quad), x0=x0, bounds=[(0.0, 1.0)], max_iters=1)
        except ValueError as e:
            return str(e)
        msg = f"accepted: {x0!r}"
        raise AssertionError(msg)

    assert "x0 must be a (n_paths, d) array" in rejects(jnp.zeros(3))
    assert "at least one point" in rejects(jnp.zeros((0, 1)))
    assert "at least one dim" in rejects(jnp.zeros((2, 0)))
    assert "x0 must be finite" in rejects(jnp.array([[0.0], [jnp.nan]]))
    assert "d=2" in rejects(flock(2, 2))  # the box is d=1


def test_a_seam_dial_that_is_not_callable_is_refused_at_the_entry() -> None:
    """A mistyped seam fails where it is passed, naming that argument.

    Same gate as the two Protocol objects, at the other end: without it a wrong
    callable reaches the first batch and fails there as `'int' object is not
    callable`, a symptom a frame away from the argument that caused it.
    """

    def eval_f_g(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        return jnp.sum(theta**2, axis=1), 2.0 * theta, None

    def rejects(first: object, **knobs: object) -> str:
        try:
            # the dial under test is deliberately out of contract
            n_path_schedule(
                first,  # pyright: ignore[reportArgumentType]
                x0=flock(1),
                bounds=[(0.0, 1.0)],
                max_iters=1,
                **knobs,  # pyright: ignore[reportArgumentType]
            )
        except TypeError as e:
            return str(e)
        msg = f"accepted: {first!r} {knobs}"
        raise AssertionError(msg)

    assert "eval_f_g must be callable" in rejects(42)
    assert "eval_f must be callable" in rejects(eval_f_g, eval_f=7)
    assert "global_model must be callable or None" in rejects(eval_f_g, global_model=3)
    assert "budget_check must be callable" in rejects(eval_f_g, budget_check=0)


def test_malformed_bounds_are_rejected_at_the_entry() -> None:
    """A box that is not a box fails here, not inside a clip.

    Both spellings are checked for the same three defects: an inverted dim, a
    NaN side (which would silently poison every clip instead of failing), and
    an array whose second axis is not the pair.
    """

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum(theta**2), 2.0 * theta

    def rejects(bounds: jnp.ndarray | list[tuple[float | None, float | None]]) -> str:
        try:
            n_path_schedule(Recorder(quad), x0=flock(1), bounds=bounds, max_iters=1)
        except ValueError as e:
            return str(e)
        msg = f"accepted: {bounds!r}"
        raise AssertionError(msg)

    assert "lo=1.0 > hi=0.0" in rejects([(1.0, 0.0)])
    assert "NaN" in rejects([(0.0, float("nan"))])
    assert "(d, 2)" in rejects(jnp.zeros((3, 3)))


def test_a_seam_object_without_its_methods_is_a_type_error() -> None:
    """The two seams are checked for presence once, before any point is served.

    Static typing already refuses these; the runtime check catches the caller
    who reaches the engine from untyped code, and it names which seam objected.
    """

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum(theta**2), 2.0 * theta

    def rejects(**knobs: object) -> str:
        try:
            # the knob under test is deliberately out of contract
            n_path_schedule(
                Recorder(quad),
                x0=flock(1),
                bounds=[(0.0, 1.0)],
                max_iters=1,
                **knobs,  # pyright: ignore[reportArgumentType]
            )
        except TypeError as e:
            return str(e)
        msg = f"accepted: {knobs}"
        raise AssertionError(msg)

    assert "solver" in rejects(solver=object())
    assert "restart_rule" in rejects(restart_rule=object())


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"all {len(fns)} passed")
