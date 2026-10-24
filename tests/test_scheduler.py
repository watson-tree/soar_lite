"""n-path scheduler self-checks.

Seam (拍板): the engine consumes callables plus the box it searches

    evaluate_batch(x (M,d)) -> (f_vals (M,), grads_x (M,d), feas (M,)|None)
    budget_check() -> bool
    bounds -> scipy-style (lb, ub) pairs | (d, 2) array | None (no box)
    sample(key, n) -> (n, d) engine starts; the sole authority on d

The caller composes x->θ and the chain rule grad_x = grad_θ * dθ_dx(x) into
evaluate_batch; the engine owns clip-to-bounds, and the prior is
injected as `sample` - with `sample=None` it falls back to i.i.d. uniform over
a finite `bounds` (the historical path, byte-identical), so an unbounded box
without a sampler is a wiring error. Unbounded means a
`None`/±inf side, and `clip` is then the identity.
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
from soar_lite.restart import EIRestartRule

jax.config.update("jax_enable_x64", val=True)

Pointwise: TypeAlias = Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]


def unit_bounds(d: int = 1) -> jnp.ndarray:
    """Return the engine's box in working coords: x ∈ [0,1]^d."""
    return jnp.stack([jnp.zeros(d), jnp.ones(d)], axis=1)


class HalfStep:
    """A deterministic local method: half a steepest-descent step, strict decrease.

    These tests are the engine's, so they must not depend on which local method
    is plugged in - the engine owns the paths, the box, the shared budget and the
    EI cadence, and the method owns only direction, step size and acceptance.
    This one is deliberately naive: it proposes `x - 0.5*g` and accepts only a
    strict decrease, so a path stalls for an engine reason (a rejected trial,
    typically one pinned at the box wall) and never because a curvature model
    went quiet. What the real method does is `test_solver.py`'s subject.
    """

    def init(self, i: int, p: PathPoint) -> None:  # noqa: ARG002
        """Take no per-path state: the step depends only on the current point."""
        return

    def restart(self, i: int, p: PathPoint) -> None:  # noqa: ARG002
        """Re-seed: nothing to reset, so this is `init` again."""
        return

    def propose(self, i: int, p: PathPoint) -> jnp.ndarray:  # noqa: ARG002
        """Return half a steepest-descent step from p."""
        return p.x - 0.5 * p.g

    def verdict(self, i: int, p: PathPoint, trial: PathPoint) -> Verdict:  # noqa: ARG002
        """Accept a strict decrease, retry otherwise."""
        return "accept" if trial.f < p.f else "retry"

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
        assert theta.ndim == 2, f"evaluate_batch expects (M,d) batch, got {theta.shape}"
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
    """evaluate_batch receives (n_paths,d) batches; each point counts exactly 1 eval.

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
        bounds=unit_bounds(),
        n_paths=4,
        max_evals=50,
        budget_check=rec.limit_check(500),
        solver=HalfStep(),
        key=jax.random.PRNGKey(0),
    )
    assert isinstance(res, NPathResult)
    assert res.n_points == rec.evals, (res.n_points, rec.evals)
    assert res.n_points <= 500
    # nit counts engine iterations, not the trials inside them (one iteration
    # may serve up to max_ls trials per path), so it is bounded by the step cap
    # and stays far below the eval count - but it is not the cap itself: a run
    # that ends on all_stalled stops counting short of it.
    assert 0 < res.nit <= 50
    assert res.n_points >= res.nit * 4  # a batch serves at least the n_paths points
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
        bounds=unit_bounds(),
        n_paths=8,
        max_evals=100,
        budget_check=rec.limit_check(30),
        solver=HalfStep(),
        key=jax.random.PRNGKey(1),
    )
    assert res.n_points == rec.evals
    assert 30 <= res.n_points < 30 + 8  # at most one batch past the limit
    assert 0 < res.nit < 100  # the gate stopped it, not the step cap
    assert res.status == "budget_exhausted"

    # gate already False on entry: nothing evaluated at all
    rec0 = Recorder(quad)
    res0 = n_path_schedule(
        rec0,
        bounds=unit_bounds(),
        n_paths=4,
        budget_check=rec0.limit_check(0),
        solver=HalfStep(),
        key=jax.random.PRNGKey(2),
    )
    assert res0.n_points == 0
    assert res0.nit == 0  # never got past the first gate check
    assert res0.status == "budget_exhausted"


def test_fun_is_true_min_over_history() -> None:
    """Pin that fun is the min over every finite f the seam returned.

    Not the final point, not a compressed history.
    """

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum((theta - 0.7) ** 2), 2.0 * (theta - 0.7)

    rec = Recorder(quad)
    res = n_path_schedule(
        rec,
        bounds=unit_bounds(),
        n_paths=4,
        max_evals=50,
        budget_check=rec.limit_check(500),
        solver=HalfStep(),
        key=jax.random.PRNGKey(3),
    )
    assert rec.all_f_vals
    assert res.fun == min(rec.all_f_vals)
    # 0.7 is the true minimizer; the engine should land essentially on it
    assert res.fun < 1e-8


def test_k_start_beats_single_path_on_multimodal() -> None:
    """Equal eval budget: 16 starts strictly beat 1 start on a two-well 1-D.

    Global well -8 @ θ=0.38, local well -2 @ θ=0.62, saddle ~ 0.505 (the
    deep well is 4x deeper, so its basin reaches just past mid-domain).
    x0 ~ uniform(0,1) and the identity map makes θ = x, so the single
    path's fate depends on its pair-min start (min of 2 draws) relative
    to the saddle.  Cold-start
    evidence asymmetry: n_paths=1 feeds one group-min -> sigma_r = 0 -> degenerate
    EI_restart = 0 -> the single path never restarts; n_paths=16 feeds a real
    spread -> sigma_r > 0 -> stalled paths may probe their way to the
    global well.
    """
    sigma_g, sigma_l = 0.04, 0.04

    def f1d(theta: jnp.ndarray) -> jnp.ndarray:
        return -2.0 * jnp.exp(-(((theta - 0.62) / sigma_l) ** 2)) - 8.0 * jnp.exp(
            -(((theta - 0.38) / sigma_g) ** 2)
        )

    def grad1d(theta: jnp.ndarray) -> jnp.ndarray:
        u_l, u_g = (theta - 0.62) / sigma_l, (theta - 0.38) / sigma_g
        return (
            4.0 * u_l * jnp.exp(-(u_l**2)) / sigma_l
            + 16.0 * u_g * jnp.exp(-(u_g**2)) / sigma_g
        )

    # independent literal grid check: global min is -8 (brute force)
    grid = jnp.linspace(0.05, 0.95, 901)
    grid_min = float(jnp.min(jax.vmap(f1d)(grid)))
    assert abs(grid_min - (-8.0)) < 0.15

    def pointwise(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return f1d(theta[0]), grad1d(theta)

    budget = 320
    rec_k = Recorder(pointwise)
    res_k = n_path_schedule(
        rec_k,
        bounds=unit_bounds(),
        n_paths=16,
        max_evals=100,
        budget_check=rec_k.limit_check(budget),
        solver=HalfStep(),
        key=jax.random.PRNGKey(2),
    )
    rec_1 = Recorder(pointwise)
    res_1 = n_path_schedule(
        rec_1,
        bounds=unit_bounds(),
        n_paths=1,
        max_evals=100,
        budget_check=rec_1.limit_check(budget),
        solver=HalfStep(),
        key=jax.random.PRNGKey(2),
    )
    # the single path must be trapped in the shallow well: replicate the
    # engine's n_paths=1 init key chain (split -> probe eval 1, split -> eval 2)
    # and verify the pair-min start lands in the shallow basin (x > 0.505)
    k = jax.random.PRNGKey(2)
    k, k1 = jax.random.split(k)
    k, k2 = jax.random.split(k)
    xa = float(jax.random.uniform(k1, (1, 1), minval=0.0, maxval=1.0)[0, 0])
    xb = float(jax.random.uniform(k2, (1, 1), minval=0.0, maxval=1.0)[0, 0])
    x0_1 = xa if float(f1d(jnp.asarray(xa))) <= float(f1d(jnp.asarray(xb))) else xb
    assert 0.505 < x0_1 < 1.0, f"single-path seed moved: x0={x0_1} (xa={xa}, xb={xb})"
    assert res_1.fun > -3.0, res_1.fun  # stuck near local well (-2)
    assert res_k.fun < -7.0, res_k.fun  # found the global well (-8)
    assert res_k.fun < res_1.fun  # strictly better, equal budget


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
        bounds=unit_bounds(2),
        n_paths=1,  # n_paths=1 -> sigma_r=0 -> never restart, pure projected descent
        max_evals=100,
        budget_check=rec.limit_check(500),
        solver=HalfStep(),
        key=jax.random.PRNGKey(4),
    )
    assert res.fun < 0.4, res.fun  # must slide near (1, 0), f=0.25


def test_feasible_only_incumbent() -> None:
    """B (incumbent) is the min over *feasible* points, not the raw min.

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

    def evaluate_batch(
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
        evaluate_batch,
        bounds=unit_bounds(),
        n_paths=4,
        max_evals=100,
        budget_check=budget_check,
        solver=HalfStep(),
        key=jax.random.PRNGKey(5),
    )

    feasible_f_vals = [L for L, ok in zip(all_f_vals, all_feas, strict=True) if ok]
    assert feasible_f_vals, "no feasible point ever served"
    # the raw min sits in the infeasible basin, below every feasible point
    assert min(all_f_vals) < min(feasible_f_vals)
    # B must be exactly the best feasible point, never the raw min
    assert res.fun == min(feasible_f_vals)
    assert res.x[0] > thr  # the reported best lies on the feasible side


def test_nan_first_candidate_does_not_mask_batch_min() -> None:
    """A NaN probe ranked first must not swallow the batch's real best point.

    The init probe serves [NaN, 0.01] (both feasible), so the engine must
    report best_f == 0.01. `min(candidates, key=f_vals_list.__getitem__)`
    kept the NaN (NaN < x is False): the batch's real minimum — and its x —
    was discarded and best_f stayed inf.
    """
    served = [0]

    def evaluate_batch(
        theta: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        served[0] += int(theta.shape[0])
        row0 = jnp.arange(theta.shape[0]) == 0
        nan = jnp.asarray(jnp.nan, dtype=theta.dtype)
        f_vals = jnp.where(row0, nan, jnp.asarray(0.01, dtype=theta.dtype))
        return f_vals, jnp.zeros_like(theta), jnp.ones(theta.shape[0], dtype=bool)

    res = n_path_schedule(
        evaluate_batch,
        bounds=unit_bounds(),
        n_paths=2,
        max_evals=5,
        budget_check=lambda: served[0] < 40,
        solver=HalfStep(),
        key=jax.random.PRNGKey(6),
    )
    assert res.fun == 0.01, res.fun
    assert bool(jnp.isfinite(res.x[0])), res.x


def run_probe_pass_with_nan_draw(nan_draw: int) -> None:
    """One n_paths=2 init pass serving NaN for the whole f-only draw `nan_draw`."""
    n_paths = 2
    draws: list[jnp.ndarray] = []
    winners: list[jnp.ndarray] = []

    def evaluate_f(theta: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        draws.append(theta)
        # The finite draw only has to be finite and distinguishable; it is made
        # clearly worse than the real objective below so the winners' re-served
        # value reads as an improvement rather than a worsening.
        value = jnp.nan if len(draws) == nan_draw else 1.0e3
        return jnp.full((theta.shape[0],), value, dtype=theta.dtype), None

    def evaluate_batch(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        winners.append(theta)
        d = theta - 0.7
        return jnp.sum(d**4 + d**2, axis=1), 4.0 * d**3 + 2.0 * d, None

    res = n_path_schedule(
        evaluate_batch,
        evaluate_f=evaluate_f,
        bounds=unit_bounds(),
        n_paths=n_paths,
        # group-min needs two draws to skip between; this test pins the skip,
        # not the default, so it states the probe size it assumes.
        n_probe_evals=2,
        max_evals=6,  # two f-only draws, their winners' batch, then steps
        solver=HalfStep(),
        key=jax.random.PRNGKey(7),
    )
    assert winners, "the engine never served the probe's winners"
    assert res.status == "max_evals", (nan_draw, res.status)  # no NaN stall
    finite_draw = draws[1] if nan_draw == 1 else draws[0]
    for j in range(n_paths):
        assert jnp.allclose(winners[0][j], finite_draw[j]), (
            f"nan_draw={nan_draw}: start {j} is not the finite draw's point"
        )


def test_probe_group_min_skips_nan() -> None:
    """A NaN probe draw must not win the group-min: starts stay finite.

    One whole f-only draw serves NaN for every point while the other serves
    1.0e3, so every start must be its point from the finite draw - asserted per
    column against the recorded draws, since each path takes the group-min of
    its own column (no re-derivation of the engine's PRNG chain). Picking the
    NaN draw's x would set that path's ZH reference to NaN: Armijo can never
    accept against NaN, so the path would stall for the rest of the run. Running
    the NaN draw first and last also kills an "always take the last draw"
    mis-implementation.
    """
    run_probe_pass_with_nan_draw(1)
    run_probe_pass_with_nan_draw(2)
    # helper contract: skip non-finite, None when nothing is finite
    assert _argmin_finite([2.0, 1.0]) == 1
    assert _argmin_finite([math.nan, 0.01, math.inf]) == 1
    assert _argmin_finite([math.nan, math.inf]) is None


class ForcedWindow(EIRestartRule):
    """The delivery rule with its verdict window pinned (a test/arm knob).

    The rule derives its own window from the payoff pool, so a test
    that wants a literal cadence for an engine run states it here: it stays the
    same estimator and the same gate, only the length is forced. The delivery
    rule has no such parameter - a fixed cadence is a variant of it, not a knob
    on it.
    """

    def __init__(self, window: int) -> None:
        """Pin every path's stage to `window` samples."""
        super().__init__()
        self._window = window

    def _stage_window(self) -> int:
        """Return the pinned window (the pool's size is not consulted)."""
        return self._window


def test_probe_inf_group_min_does_not_silence_restarts() -> None:
    """An all-+inf probe group must not disable every restart in the flock.

    Column 0 is +inf in both init probe evals (a group with no finite
    member), so the shared init group-min list carries an inf sample. Fitting
    it naively gives mu_r = inf, sigma_r = NaN and EI_restart = NaN, and
    `NaN > EI_continue` is False forever - no path would ever restart again
    even though B is finite. The payoff pool proves the verdicts still fire:
    it is n_paths long after init and every restart event appends its own
    entries, so a run with verdicts grows past n_paths while the same run with
    the window pushed out of reach does not.
    """
    n_paths = 3
    # The flat field makes every step a rejecting line search, so one iteration
    # can burn max_ls trials; the budget only has to carry init plus the
    # iterations the verdicts need to come due.
    max_evals = 60
    init_f_vals = jnp.array([jnp.inf, 1.0, 3.0])  # row 0 = column 0 = dead

    def seam_run(window: int) -> tuple[NPathResult, ForcedWindow]:
        calls = [0]

        def evaluate_batch(
            theta: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray, None]:
            calls[0] += 1
            if calls[0] <= 2:  # init probe evals: shared cold-start evidence
                f_vals = init_f_vals
            else:  # everything afterwards is finite and flat (Δ = 0, sigma_p = 0)
                f_vals = jnp.full((theta.shape[0],), 1.0, dtype=theta.dtype)
            return f_vals, jnp.zeros_like(theta), None

        rule = ForcedWindow(window)
        res = n_path_schedule(
            evaluate_batch,
            bounds=unit_bounds(),
            n_paths=n_paths,
            max_evals=max_evals,
            solver=HalfStep(),
            restart_rule=rule,
            key=jax.random.PRNGKey(0),
        )
        return res, rule

    res_off, rule_off = seam_run(10**9)  # verdicts never due
    res_on, rule_on = seam_run(2)  # verdict once two samples land
    assert res_off.fun == 1.0, res_off.fun  # B is finite, so EI_restart fits
    assert res_on.fun == 1.0, res_on.fun
    assert len(rule_off.group_mins) == n_paths, rule_off.group_mins  # init only
    assert len(rule_on.group_mins) > n_paths, rule_on.group_mins  # restarts fired


def test_nonfinite_start_waits_for_ei_instead_of_stepping() -> None:
    """A start whose whole probe group is non-finite is stalled on arrival.

    It has no finite reference to descend from, so it must not burn budget
    while waiting for an EI verdict: with verdicts effectively off (huge
    window) exactly one eval per step is served after the two init
    iterations - the live path only.
    """
    n_paths = 2
    # the budget is in batches and the init probe is billed, so it is the init
    # batch plus the three trial batches this test counts
    n_probe_evals = 1
    n_iterations = 3
    max_evals = n_probe_evals + n_iterations
    served = [0]
    calls = [0]

    def evaluate_batch(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        calls[0] += 1
        served[0] += int(theta.shape[0])
        # Column 1 needs a real objective: on a flat field it rejects max_ls
        # times per step instead of committing one, and the per-step accounting
        # this test pins would measure the line search, not the stall.
        d = theta - 0.7
        f_vals = jnp.sum(d**4 + d**2, axis=1)
        grads = 4.0 * d**3 + 2.0 * d
        if calls[0] <= n_probe_evals:  # the init probe: column 0 gets no evidence
            f_vals = f_vals.at[0].set(jnp.inf)
        return f_vals, grads, None

    res = n_path_schedule(
        evaluate_batch,
        bounds=unit_bounds(),
        n_paths=n_paths,
        n_probe_evals=n_probe_evals,
        max_evals=max_evals,
        budget_check=lambda: served[0] < 60,
        solver=HalfStep(),
        restart_rule=ForcedWindow(10**9),
        key=jax.random.PRNGKey(0),
    )
    # one point per live path per iteration, not two evals per step
    assert res.n_points == n_probe_evals * n_paths + n_iterations, res.n_points


def test_stalled_paths_wait_for_ei_then_restart() -> None:
    """max_ls exhaustion stalls a path; only a due EI verdict revives it.

    Init probes return f = x[0] (a real spread, so sigma_r > 0 and B stays
    small); every later evaluation returns the same bad constant with a zero
    gradient, so ZH backtracking rejects max_ls times and the paths stall. With
    window = 2 (the shortest stage that has a spread at all) and one sample per
    iteration, the stalled flock's second sample lands at iteration 2 and its
    verdict is due at iteration 3, with EI_continue = 0
    (f_k >= B) and EI_restart > 0, so the flock is revived by a restart probe;
    with verdicts off (huge window) the very same run serves nothing after the
    stall. The two traces differ by exactly the restart
    events - this is the engine-level wiring (stall -> EI -> restart) the unit
    tests of _ei_restart/EI_continue cannot cover. The huge-period arm also pins
    the engine's stop for it: every path stalled and the next verdict lies
    ~10**9 iterations out, past max_evals, so the run must end there (status
    "all_stalled", one attempted iteration) rather than spin out the eval budget.
    """
    n_paths, max_ls, n_probe_evals = 4, 3, 1
    # Batches: the init probe, the single rejecting iteration (max_ls trials),
    # the restart probe, and one more rejecting iteration of the revived flock.
    max_evals = 2 * (n_probe_evals + max_ls)
    bad = 10.0

    def seam_run(window: int) -> tuple[list[tuple[int, int]], int, NPathResult]:
        shapes: list[tuple[int, int]] = []
        evals = [0]

        def evaluate_batch(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
            shapes.append((int(theta.shape[0]), int(theta.shape[1])))
            evals[0] += int(theta.shape[0])
            if len(shapes) <= n_probe_evals:  # init probe evals: real objective
                return theta[:, 0], jnp.ones_like(theta), None
            return (
                jnp.full((theta.shape[0],), bad, dtype=theta.dtype),
                jnp.zeros_like(theta),
                None,
            )

        res = n_path_schedule(
            evaluate_batch,
            bounds=unit_bounds(),
            n_paths=n_paths,
            n_probe_evals=n_probe_evals,
            max_evals=max_evals,
            max_ls=max_ls,
            # the step cap ends both runs
            budget_check=lambda: evals[0] < 10**6,
            solver=HalfStep(),
            restart_rule=ForcedWindow(window),
            key=jax.random.PRNGKey(8),
        )
        return shapes, evals[0], res

    shapes_off, evals_off, res_off = seam_run(10**9)  # verdicts never due
    shapes_on, evals_on, res_on = seam_run(2)  # verdict once two samples land
    # without verdicts: init + max_ls rejecting trials per path, then nothing
    pre = n_probe_evals + max_ls
    assert len(shapes_off) == pre, shapes_off
    assert evals_off == (n_probe_evals + max_ls) * n_paths
    # Iteration 5 is the verdict iteration - no eval, then a restart probe of every
    # path. The tail is left as an inequality on purpose - what the revived
    # paths do after the probe (accept, or reject again on the same flat field
    # until the step cap) is the injected method's business.
    assert len(shapes_on) - pre >= n_probe_evals, shapes_on
    assert res_off.status == "all_stalled"  # nothing can revive the flock
    assert res_on.status == "max_evals"  # revived, then capped
    assert all(m == n_paths for m, _d in shapes_on), shapes_on  # no shrinking batches
    assert evals_on == len(shapes_on) * n_paths > evals_off
    assert res_on.status == "max_evals", res_on  # a due verdict keeps it alive
    assert res_off.status == "all_stalled", res_off  # not "budget_exhausted"
    # The unreachable verdict ends the run at the stall instead of spinning the
    # remaining iterations: the productive iteration folds the stall's own
    # sample, and only then can the rule state a cadence for the shortcut to
    # read. So exactly one iteration completes, and the next one serves nothing
    # before the shortcut ends the run.
    assert res_off.nit == 1, res_off
    assert len(shapes_on) == max_evals, shapes_on  # the revived run spends it all


def test_bounds_spellings_are_the_same_box() -> None:
    """scipy-style pairs and the historical (d, 2) array run byte-identically.

    One key stream, one draw sequence: if the two spellings normalized
    differently, the probes would land elsewhere and every recorded f would
    differ. This is also what pins the new entry to the historical engine
    (the (d, 2) array is what the caller passed then).
    """

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum((theta - 0.7) ** 2), 2.0 * (theta - 0.7)

    def run(
        bounds: jnp.ndarray | list[tuple[float, float]],
    ) -> tuple[list[float], float]:
        rec = Recorder(quad)
        res = n_path_schedule(
            rec,
            bounds=bounds,
            budget_check=rec.limit_check(80),
            n_paths=4,
            max_evals=10,
            solver=HalfStep(),
            key=jax.random.PRNGKey(3),
        )
        return rec.all_f_vals, res.fun

    f_vals_arr, fun_arr = run(unit_bounds(2))
    f_vals_pairs, fun_pairs = run([(0.0, 1.0)] * 2)
    assert f_vals_pairs == f_vals_arr
    assert fun_pairs == fun_arr


def test_unbounded_or_absent_box_needs_an_injected_sampler() -> None:
    """sample=None can only draw a finite box; anything else is a wiring error.

    An unbounded side has no distribution to draw from unless the caller brings
    one, so the engine refuses instead of inventing a box (or a d) of its own.
    One-sided is the quiet case: half the box still exists, so the run would
    silently be a different algorithm than the one the caller thinks it wired.
    """

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum(theta**2), 2.0 * theta

    for bounds in ([(-jnp.inf, 1.0)], [(None, None)], None):
        rec = Recorder(quad)
        try:
            n_path_schedule(
                rec,
                bounds=bounds,
                n_paths=2,
                max_evals=1,
                solver=HalfStep(),
                key=jax.random.PRNGKey(0),
            )
        except ValueError:
            pass
        else:  # pragma: no cover - the raise is the assertion
            msg = f"no sampler, yet {bounds} was accepted"
            raise AssertionError(msg)
        assert rec.evals == 0  # rejected before any point was served


def test_the_sampler_is_the_only_authority_on_d() -> None:
    """A bare None box + a sampler is a well-posed run: d is whatever it draws.

    `bounds=None` and an all-unbounded pair list are the same box (∓inf), so
    they must also be the same run.
    """

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum((theta - 0.7) ** 2), 2.0 * (theta - 0.7)

    def sample(sub: jnp.ndarray, n: int) -> jnp.ndarray:
        return jax.random.logistic(sub, (n, 3))

    rec_none = Recorder(quad)
    res = n_path_schedule(
        rec_none,
        bounds=None,
        sample=sample,
        budget_check=rec_none.limit_check(60),
        n_paths=2,
        max_evals=3,
        solver=HalfStep(),
        key=jax.random.PRNGKey(4),
    )
    assert all(d == 3 for _m, d in rec_none.batch_shapes), rec_none.batch_shapes
    assert res.x.shape == (3,)

    rec_pairs = Recorder(quad)
    n_path_schedule(
        rec_pairs,
        bounds=[(None, None)] * 3,
        sample=sample,
        budget_check=rec_pairs.limit_check(60),
        n_paths=2,
        max_evals=3,
        solver=HalfStep(),
        key=jax.random.PRNGKey(4),
    )
    assert rec_pairs.all_f_vals == rec_none.all_f_vals

    # a box and a sampler that disagree on d is the wiring bug this catches
    try:
        n_path_schedule(
            Recorder(quad),
            bounds=unit_bounds(2),
            sample=sample,
            n_paths=2,
            max_evals=1,
            solver=HalfStep(),
            key=jax.random.PRNGKey(4),
        )
    except ValueError:
        pass
    else:  # pragma: no cover - the raise is the assertion
        msg = "a d=3 sampler was accepted against a d=2 box"
        raise AssertionError(msg)


def test_an_injected_sampler_searches_outside_the_old_box() -> None:
    """Unbounded means unbounded: a minimizer outside [0,1] is reached.

    θ* = 3.0 sits outside that box, so a clipped finite-box run could never
    serve it; with ∓inf bounds the clip is the identity and the descent is free
    to land on it.
    """
    seen: list[float] = []  # max |x| per served batch

    def evaluate_batch(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        seen.append(float(jnp.max(jnp.abs(x))))
        return jnp.sum((x - 3.0) ** 2, axis=-1), 2.0 * (x - 3.0), None

    def sample(sub: jnp.ndarray, n: int) -> jnp.ndarray:
        return jax.random.logistic(sub, (n, 1)) + 3.0  # starts near the well

    res = n_path_schedule(
        evaluate_batch,
        bounds=[(None, None)],
        sample=sample,
        budget_check=lambda: True,
        n_paths=4,
        max_evals=40,
        solver=HalfStep(),
        key=jax.random.PRNGKey(5),
    )
    assert max(seen) > 1.0, seen  # a finite [0,1] box could not have gone there
    assert res.fun < 1e-6, res.fun


def test_out_of_range_knobs_are_rejected_at_the_entry() -> None:
    """A bad budget knob fails once, at the entry, naming the knob.

    None of these shapes can run: zero paths advance nothing, and a negative
    step cap means no iteration at all. The caller should see the knob, not an
    empty result later.
    """

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum(theta**2), 2.0 * theta

    def rejects(**knobs: int) -> str:
        try:
            # the knob under test is deliberately out of contract
            n_path_schedule(
                Recorder(quad),
                bounds=[(0.0, 1.0)],
                **knobs,  # pyright: ignore[reportArgumentType]
            )
        except ValueError as e:
            return str(e)
        msg = f"accepted: {knobs}"
        raise AssertionError(msg)

    for bad in (
        {"n_paths": 0},
        {"n_probe_evals": 0},
        {"max_ls": 0},
        {"max_evals": -1},
    ):
        assert next(iter(bad)) in rejects(**bad), bad


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
            n_path_schedule(Recorder(quad), bounds=bounds)
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
                bounds=[(0.0, 1.0)],
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
