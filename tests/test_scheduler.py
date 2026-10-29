"""n-path scheduler self-checks.

Seam (拍板): the engine consumes its width, the callables, and the box it searches

    eval_f_g(x (M,d)) -> (f_vals (M,), grads_x (M,d), feas (M,)|None)
    budget_check() -> bool
    bounds -> scipy-style (lb, ub) pairs | (d, 2) array | None (no box)
    n_paths -> the run's width, the engine's own structure
    global_model -> the restart-point source and the evidence sink: fed every
        served batch, asked for points only when a restart fires; its `d` is
        the run's dimension

The caller composes x->θ and the chain rule grad_x = grad_θ * dθ_dx(x) into
eval_f_g; the engine owns clip-to-bounds, and the starts are the model's - and
the two plugins are spelled `global_model`, `local_solver`
in that order wherever the engine names them. Whether a stall is re-drawn is
the `restart_stalled` knob, not a plugin. With
`global_model=None` the engine falls back to a `UniformGlobalModel` drawing
i.i.d. uniformly over a finite `bounds`; an unbounded box (or none) without a
model is a wiring error, since the engine then has nothing of its own to draw
from. Unbounded means a `None`/±inf side, and `clip` is then the identity.
`test_plugin_seam.py`'s evaluator is that composed form.

Every run below plants its flock through `flock_kwargs`, i.e. through the
shipped default model's first proposal: the engine has no start argument, and a
test that wants known starts hands them to a model.
"""

# pyright: reportPrivateUsage=false
# This suite is the spec for two `_`-private engine helpers; importing them
# here is the intended read, not a boundary crossing.

import math
from collections.abc import Callable
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp

from soar_lite.engine import (
    Bounds,
    NPathResult,
    PathPoint,
    Verdict,
    _argmin_finite,
    n_path_schedule,
)
from soar_lite.global_model import UniformGlobalModel
from soar_lite.local_solver import DiagonalPreconditioner

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


def flock_kwargs(
    x0: jnp.ndarray,
    bounds: Bounds,
    *,
    key: jnp.ndarray | None = None,
) -> dict[str, Any]:
    """Return the entry's two flock kwargs: the run's width and its starts.

    The engine takes its width as `n_paths` and every point a path starts from as
    the global model's first proposal, so a test that chooses its flock says both
    here. The model is the shipped default one: it hands `x0` back on the first
    `propose` and draws any later re-draw from `key`.
    """
    return {
        "n_paths": int(x0.shape[0]),
        "global_model": UniformGlobalModel(
            bounds,
            key=jax.random.PRNGKey(0) if key is None else key,
            x0=x0,
        ),
    }


class FixedModel:
    """A global model over a fixed list of points: the flock, then the same rows.

    For the runs with no finite box at all: `UniformGlobalModel` needs one to
    draw from, and a model-free test that only needs starts of a known `d` says
    so with a list.
    """

    def __init__(self, points: jnp.ndarray) -> None:
        """Store the points every ask is answered from."""
        self.points = points
        self.d = int(points.shape[1])

    def observe(self, x: jnp.ndarray, f: list[float], g: jnp.ndarray | None) -> None:
        """Ignore the evidence: this model fits nothing."""

    def propose(self, n_points: int) -> jnp.ndarray:
        """Return n_points of the stored rows, tiling them if more are asked."""
        rows = int(self.points.shape[0])
        reps = -(-n_points // rows)
        return jnp.concatenate([self.points] * reps)[:n_points]


class HalfStep:
    """A deterministic local method: half a steepest-descent step, strict decrease.

    These tests are the engine's, so they must not depend on which local method
    is plugged in - the engine owns the paths, the box, the shared budget and the
    EI cadence, and the method owns only direction, step size, acceptance and
    how long its own enumeration runs. This one is deliberately naive: it
    proposes `x - 0.5*g` and accepts only a strict decrease, giving up after
    `trial_cap` rejects, so a path stalls for an engine reason (a rejected trial,
    typically one pinned at the box wall) and never because a curvature model
    went quiet. What the real method does is `test_local_solver.py`'s
    subject.
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

    def needs_verdict(self, i: int, p: PathPoint) -> bool:  # noqa: ARG002
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
        **flock_kwargs(flock(4), unit_bounds(), key=jax.random.PRNGKey(0)),
        bounds=unit_bounds(),
        max_iters=50,
        budget_check=rec.limit_check(500),
        local_solver=HalfStep(),
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
        **flock_kwargs(flock(8), unit_bounds(), key=jax.random.PRNGKey(1)),
        bounds=unit_bounds(),
        max_iters=100,
        budget_check=rec.limit_check(30),
        local_solver=HalfStep(),
    )
    assert res.n_points == rec.evals
    assert 30 <= res.n_points < 30 + 8  # at most one batch past the limit
    assert 0 < res.n_iters < 100  # the gate stopped it, not the horizon
    assert res.status == "budget_exhausted"

    # gate already False on entry: nothing evaluated at all
    rec0 = Recorder(quad)
    res0 = n_path_schedule(
        rec0,
        **flock_kwargs(flock(4), unit_bounds(), key=jax.random.PRNGKey(2)),
        bounds=unit_bounds(),
        max_iters=100,
        budget_check=rec0.limit_check(0),
        local_solver=HalfStep(),
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
        **flock_kwargs(flock(4), unit_bounds(), key=jax.random.PRNGKey(3)),
        bounds=unit_bounds(),
        max_iters=1,
        local_solver=HalfStep(),
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
        **flock_kwargs(flock(4), unit_bounds(), key=jax.random.PRNGKey(3)),
        bounds=unit_bounds(),
        max_iters=50,
        budget_check=rec.limit_check(500),
        local_solver=HalfStep(),
    )
    assert rec.all_f_vals
    assert res.fun == min(rec.all_f_vals)
    # 0.7 is the true minimizer; the engine should land essentially on it
    assert res.fun < 1e-8


_TWO_WELL_SIGMA = 0.04


def _two_well_1d(theta: jnp.ndarray) -> jnp.ndarray:
    """Two-well 1-D: global -8 @ θ=0.38, local -2 @ θ=0.62, saddle near 0.505."""
    return -2.0 * jnp.exp(-(((theta - 0.62) / _TWO_WELL_SIGMA) ** 2)) - 8.0 * jnp.exp(
        -(((theta - 0.38) / _TWO_WELL_SIGMA) ** 2),
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
    sixteen are spread across the box; both arms run the shipped method with
    restart off, so the only difference is the start count. What restarting adds
    on top of width is the next test's subject.
    """
    # independent literal grid check: global min is -8 (brute force)
    grid = jnp.linspace(0.05, 0.95, 901)
    grid_min = float(jnp.min(jax.vmap(_two_well_1d)(grid)))
    assert abs(grid_min - (-8.0)) < 0.15

    budget = 320
    rec_k = Recorder(_two_well)
    res_k = n_path_schedule(
        rec_k,
        **flock_kwargs(flock(16), unit_bounds(), key=jax.random.PRNGKey(2)),
        bounds=unit_bounds(),
        max_iters=100,
        budget_check=rec_k.limit_check(budget),
        local_solver=DiagonalPreconditioner(),
    )
    rec_1 = Recorder(_two_well)
    res_1 = n_path_schedule(
        rec_1,
        **flock_kwargs(
            jnp.array([[0.62]]),
            unit_bounds(),
            key=jax.random.PRNGKey(2),
        ),  # planted in the shallow well
        bounds=unit_bounds(),
        max_iters=100,
        budget_check=rec_1.limit_check(budget),
        local_solver=DiagonalPreconditioner(),
    )
    assert res_1.fun > -3.0, res_1.fun  # stuck near local well (-2)
    assert res_k.fun < -7.0, res_k.fun  # found the global well (-8)
    assert res_k.fun < res_1.fun  # strictly better, equal budget


def test_a_stall_hands_a_redraw_to_the_global_model() -> None:
    """A stalled path is re-placed where the model says, not where it stood.

    Synthetic on purpose. The value depends only on which side of 0.5 the point
    lies on, and the gradient is zero everywhere, so `HalfStep` can never find a
    strict decrease: it gives up (its own trial cap) and the engine marks the
    path stalled. Nothing in either run moves except by a restart, which is
    exactly what `restart_stalled` wires together - the engine's own stall bit
    says *when*, the model says *where*. The model places the shallow flock on
    its first ask and the deep well on every later one, so the comparison is
    deterministic: the frozen arm parks the flock at -2, the restarting arm
    hands each stop to the model and collects -8.
    """
    shallow, deep = 0.62, 0.38

    def eval_f_g(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        f = jnp.where(theta[:, 0] < 0.5, -8.0, -2.0)
        return f, jnp.zeros_like(theta), None

    class FirstShallowThenDeep:
        """Answer ask 0 with the shallow well, every later ask with the deep one."""

        d = 1

        def __init__(self) -> None:
            """Start on the shallow first ask."""
            self.asked = 0

        def observe(
            self,
            x: jnp.ndarray,
            f: list[float],
            g: jnp.ndarray | None,
        ) -> None:
            """Ignore the evidence: this model fits nothing."""

        def propose(self, n_points: int) -> jnp.ndarray:
            """Return n_points of the well this ask is assigned to."""
            centre = shallow if self.asked == 0 else deep
            self.asked += 1
            return jnp.full((n_points, 1), centre)

    def run(*, restart_stalled: bool) -> NPathResult:
        return n_path_schedule(
            eval_f_g,
            n_paths=2,
            bounds=unit_bounds(),
            max_iters=6,
            local_solver=HalfStep(),
            restart_stalled=restart_stalled,
            global_model=FirstShallowThenDeep(),
        )

    frozen = run(restart_stalled=False)
    revived = run(restart_stalled=True)
    assert frozen.fun == -2.0, frozen.fun  # nothing ever moved
    assert float(frozen.x[0]) == shallow
    assert revived.fun == -8.0, revived.fun  # the model's re-draws landed deep
    assert float(revived.x[0]) == deep


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
            [2.0 * (theta[0] - 1.5), 2.0 * theta[1]],
        )

    rec = Recorder(wall_quad)
    res = n_path_schedule(
        rec,
        **flock_kwargs(flock(1, 2), unit_bounds(2), key=jax.random.PRNGKey(4)),
        bounds=unit_bounds(2),
        max_iters=100,
        budget_check=rec.limit_check(500),
        local_solver=HalfStep(),
    )
    assert res.fun < 0.4, res.fun  # must slide near (1, 0), f=0.25


def test_feasible_only_best() -> None:
    """The reported best (`fun`) is the feasible min, never the raw min.

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

    res = n_path_schedule(
        eval_f_g,
        **flock_kwargs(flock(4), unit_bounds(), key=jax.random.PRNGKey(5)),
        bounds=unit_bounds(),
        max_iters=100,
        budget_check=budget_check,
        local_solver=HalfStep(),
    )

    feasible_f_vals = [L for L, ok in zip(all_f_vals, all_feas, strict=True) if ok]
    assert feasible_f_vals, "no feasible point ever served"
    # the raw min sits in the infeasible basin, below every feasible point
    assert min(all_f_vals) < min(feasible_f_vals)
    # the reported answer is the best feasible point, never the raw min
    assert res.fun == min(feasible_f_vals)
    assert res.x[0] > thr  # the reported best lies on the feasible side


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
        **flock_kwargs(
            jnp.linspace(0.05, 0.95, 4)[:, None],
            unit_bounds(),
            key=jax.random.PRNGKey(0),
        ),
        bounds=unit_bounds(),
        max_iters=50,
        local_solver=HalfStep(),
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
        **flock_kwargs(
            jnp.linspace(0.05, 0.95, 4)[:, None],
            unit_bounds(),
            key=jax.random.PRNGKey(0),
        ),
        bounds=unit_bounds(),
        max_iters=20,
        local_solver=HalfStep(),
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
        **flock_kwargs(flock(2), unit_bounds(), key=jax.random.PRNGKey(6)),
        bounds=unit_bounds(),
        max_iters=5,
        budget_check=lambda: served[0] < 40,
        local_solver=HalfStep(),
    )
    assert res.fun == 0.01, res.fun
    assert bool(jnp.isfinite(res.x[0])), res.x


def test_the_best_skips_nonfinite_values() -> None:
    """`_argmin_finite` is how `fun` ignores the non-finite points it served."""
    assert _argmin_finite([2.0, 1.0]) == 1
    assert _argmin_finite([math.nan, 0.01, math.inf]) == 1
    assert _argmin_finite([math.nan, math.inf]) is None


def test_a_dead_path_never_widens_a_batch() -> None:
    """A row that never holds a point never widens a batch past one point.

    Row 0 is served non-finite values forever, so it never becomes a point: it
    joins no cohort and contributes no winner, and iteration 0's flock is
    therefore the run's only fused batch - every later batch carries exactly one
    point, the live row's alone. The dead row is not idle, and this test does not
    claim it is: holding no point, it is owed a fresh start every iteration, and
    that start is one of those width-1 batches. What it never does is make the
    live row's batches wider, which is the cost pinned here.
    """
    n_paths, n_iterations = 2, 3
    widths: list[int] = []
    asks: list[int] = []
    inner = flock_kwargs(flock(n_paths), unit_bounds(), key=jax.random.PRNGKey(0))[
        "global_model"
    ]

    class AskSpy:
        """The flock's model, recording how many starts each ask wanted."""

        def __init__(self, inner: UniformGlobalModel) -> None:
            self.inner = inner
            self.d = inner.d

        def observe(
            self,
            x: jnp.ndarray,
            f: list[float],
            g: jnp.ndarray | None,
        ) -> None:
            self.inner.observe(x, f, g)

        def propose(self, n_points: int) -> jnp.ndarray:
            asks.append(n_points)
            return self.inner.propose(n_points)

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
        n_paths=n_paths,
        global_model=AskSpy(inner),
        bounds=unit_bounds(),
        max_iters=n_iterations + 1,
        budget_check=lambda: len(widths) < 100,
        local_solver=HalfStep(),
    )
    assert res.n_iters == n_iterations + 1, res
    assert widths[0] == n_paths, widths  # iteration 0 serves the whole flock
    assert all(w == 1 for w in widths[1:]), widths  # then the live path alone
    # One ask for the flock, then one re-draw per iteration for the dead row: the
    # start it is owed, never merged into the live row's batch.
    assert asks == [n_paths, *([1] * n_iterations)], asks


def test_a_dead_start_is_redrawn_without_asking_the_rule() -> None:
    """A path holding no point is owed a start: the engine draws one, unasked.

    Iteration 0's only batch is all-NaN, so the path holds no point - an empty
    `x` - and `restart_stalled` is never consulted about it: that knob governs a
    path that holds a point and stalled on it, while a path with no point is owed
    a start whatever the knob says. Nothing else could recover it, and gating it
    on the knob would leave iteration 0 with nothing to place, so the difference
    is not a policy choice. The re-draw comes from the global model, whose second
    answer is a uniform point over the box, and that start descends: the dead row
    costs one batch, not the run.
    """

    def make_evaluator() -> Callable[
        [jnp.ndarray],
        tuple[jnp.ndarray, jnp.ndarray, None],
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

    def run(*, restart_stalled: bool) -> NPathResult:
        return n_path_schedule(
            make_evaluator(),
            **flock_kwargs(flock(1), unit_bounds(), key=jax.random.PRNGKey(0)),
            bounds=unit_bounds(),
            max_iters=20,
            local_solver=DiagonalPreconditioner(),
            restart_stalled=restart_stalled,
        )

    # restart off is the control: a path with no point is re-drawn by the
    # engine's own "no point" rule either way, so both arms must survive.
    assert math.isfinite(run(restart_stalled=False).fun)
    assert math.isfinite(run(restart_stalled=True).fun)


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

        def needs_verdict(self, i: int, p: PathPoint) -> bool:  # noqa: ARG002
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
        **flock_kwargs(
            flock(1),
            unit_bounds(),
            key=jax.random.PRNGKey(11),
        ),  # one path: a single batch per trial keeps the count exact
        bounds=unit_bounds(),
        max_iters=2,  # iteration 0 places x0, then this one step
        local_solver=Patient(),
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
            **flock_kwargs(flock(4, 2), bounds, key=jax.random.PRNGKey(3)),
            bounds=bounds,
            budget_check=rec.limit_check(80),
            max_iters=10,
            local_solver=HalfStep(),
        )
        return rec.all_f_vals, res.fun

    f_vals_arr, fun_arr = run(unit_bounds(2))
    f_vals_pairs, fun_pairs = run([(0.0, 1.0)] * 2)
    assert f_vals_pairs == f_vals_arr
    assert fun_pairs == fun_arr


def test_the_default_model_needs_a_finite_box() -> None:
    """With no model, a finite box is required: it is where the draws come from.

    `UniformGlobalModel` draws uniformly, so an unbounded side has no
    distribution to draw from - the engine refuses instead of inventing one.
    One-sided is the quiet case: half the box still exists, so the run would
    silently be a different algorithm than the one the caller thinks it wired.
    The check is eager - the model is built before any point is served, because
    the first restart is iteration 0's placement.
    """

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum(theta**2), 2.0 * theta

    for bounds in ([(-jnp.inf, 1.0)], [(None, None)], None):
        rec = Recorder(quad)
        try:
            n_path_schedule(
                rec,
                n_paths=2,
                bounds=bounds,
                max_iters=1,
                local_solver=HalfStep(),
            )
        except ValueError:
            pass
        else:  # pragma: no cover - the raise is the assertion
            msg = f"no global model, yet {bounds} was accepted"
            raise AssertionError(msg)
        assert rec.evals == 0  # rejected before any point was served


def test_the_model_is_the_only_authority_on_d() -> None:
    """With no box at all, d is the model's to declare.

    A surrogate knows d before the run starts and `bounds` may be absent
    entirely, so the model is the authority and the box only cross-checks it.
    `bounds=None` and an all-unbounded pair list are the same box (∓inf), so
    they must also be the same run.
    """

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum((theta - 0.7) ** 2), 2.0 * (theta - 0.7)

    starts = jax.random.logistic(jax.random.PRNGKey(4), (2, 3))

    rec_none = Recorder(quad)
    res = n_path_schedule(
        rec_none,
        n_paths=2,
        bounds=None,
        global_model=FixedModel(starts),
        budget_check=rec_none.limit_check(60),
        max_iters=3,
        local_solver=HalfStep(),
    )
    assert all(d == 3 for _m, d in rec_none.batch_shapes), rec_none.batch_shapes
    assert res.x.shape == (3,)

    rec_pairs = Recorder(quad)
    n_path_schedule(
        rec_pairs,
        n_paths=2,
        bounds=[(None, None)] * 3,
        global_model=FixedModel(starts),
        budget_check=rec_pairs.limit_check(60),
        max_iters=3,
        local_solver=HalfStep(),
    )
    assert rec_pairs.all_f_vals == rec_none.all_f_vals

    # a model and a box that disagree on d is the wiring bug this catches
    try:
        n_path_schedule(
            Recorder(quad),
            n_paths=2,
            bounds=unit_bounds(2),
            global_model=FixedModel(flock(2, 3)),
            max_iters=1,
            local_solver=HalfStep(),
        )
    except ValueError:
        pass
    else:  # pragma: no cover - the raise is the assertion
        msg = "a d=3 model was accepted against a d=2 box"
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

    starts = jax.random.logistic(jax.random.PRNGKey(5), (4, 1)) + 3.0

    res = n_path_schedule(
        eval_f_g,
        n_paths=4,
        bounds=[(None, None)],
        global_model=FixedModel(starts),  # starts near the well, box-free
        budget_check=lambda: True,
        max_iters=40,
        local_solver=HalfStep(),
    )
    assert max(seen) > 1.0, seen  # a finite [0,1] box could not have gone there
    assert res.fun < 1e-6, res.fun


def test_out_of_range_knobs_are_rejected_at_the_entry() -> None:
    """A bad budget knob fails once, at the entry, naming the knob.

    Neither shape can run: a negative horizon means no iteration at all, and a
    zero time limit no second of it. The caller should see the knob, not an
    empty result later.
    """

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum(theta**2), 2.0 * theta

    def rejects(**knobs: float) -> str:
        try:
            # the knob under test is deliberately out of contract
            n_path_schedule(
                Recorder(quad),
                **flock_kwargs(flock(1), [(0.0, 1.0)]),  # pyright: ignore[reportArgumentType]
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


def test_a_nonpositive_width_is_refused_at_the_entry() -> None:
    """The run's width is the engine's own structure, checked with the model's d.

    A model knows d but not how many paths a caller wants, so `n_paths` is
    passed on its own and validated beside the model: zero paths advance nothing
    and a bool is not a count.
    """

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum(theta**2), 2.0 * theta

    def rejects(width: object) -> str:
        try:
            n_path_schedule(
                Recorder(quad),
                n_paths=width,  # pyright: ignore[reportArgumentType]
                bounds=[(0.0, 1.0)],
                max_iters=1,
            )
        except ValueError as e:
            return str(e)
        msg = f"accepted: {width!r}"
        raise AssertionError(msg)

    assert "n_paths" in rejects(0)
    assert "n_paths" in rejects(-2)
    assert "n_paths" in rejects(width=True)


def test_a_malformed_flock_is_refused_by_the_model() -> None:
    """A flock a model cannot hand back is refused where it is handed over.

    Four defects, one place, in the model that carries it: not a batch, no
    paths, no dims, and a non-finite start. The shape check names the d it
    wanted, because the model is what knows the run's dimension.
    """

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum(theta**2), 2.0 * theta

    def rejects(x0: jnp.ndarray) -> str:
        try:
            n_path_schedule(
                Recorder(quad),
                **flock_kwargs(x0, [(0.0, 1.0)]),
                bounds=[(0.0, 1.0)],
                max_iters=1,
            )
        except ValueError as e:
            return str(e)
        msg = f"accepted: {x0!r}"
        raise AssertionError(msg)

    assert "x0 must be a (n_paths, 1) array" in rejects(jnp.zeros(3))
    assert "at least one finite start" in rejects(jnp.zeros((0, 1)))
    assert "x0 must be a (n_paths, 1) array" in rejects(jnp.zeros((2, 0)))
    assert "at least one finite start" in rejects(jnp.array([[0.0], [jnp.nan]]))


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
                **flock_kwargs(flock(1), [(0.0, 1.0)]),  # pyright: ignore[reportArgumentType]
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
            n_path_schedule(
                Recorder(quad),
                **flock_kwargs(flock(1), bounds),
                bounds=bounds,
                max_iters=1,
            )
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
                **flock_kwargs(flock(1), [(0.0, 1.0)]),  # pyright: ignore[reportArgumentType]
                bounds=[(0.0, 1.0)],
                max_iters=1,
                **knobs,  # pyright: ignore[reportArgumentType]
            )
        except TypeError as e:
            return str(e)
        msg = f"accepted: {knobs}"
        raise AssertionError(msg)

    assert "local_solver" in rejects(local_solver=object())
    assert "global_model" in rejects(global_model=object())


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"all {len(fns)} passed")
