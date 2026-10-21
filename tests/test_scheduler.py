"""n-path scheduler self-checks.

Seam (拍板): the engine consumes callables plus the box it searches

    evaluate_batch(x (M,d)) -> (losses (M,), grads_x (M,d), feas (M,)|None)
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

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp

from soar_lite.engine import (
    NPathResult,
    PathPoint,
    Verdict,
    argmin_finite,  # helper contract pinned directly
    n_path_schedule,
)
from soar_lite.restart import EIRestartRule

jax.config.update("jax_enable_x64", val=True)

Pointwise = Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]


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
    """Wraps a pointwise (loss, grad) fn as the batched seam; records everything."""

    def __init__(self, pointwise: Pointwise) -> None:
        """Map a single point θ (d,) to (loss, grad_θ (d,))."""
        self.pointwise = pointwise
        self.all_losses: list[float] = []  # every loss ever returned, per point
        self.batch_shapes: list[tuple[int, int]] = []  # shape of each seam call
        self.evals = 0

    def __call__(self, theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        """Assert the (M,d) batch shape, record it, and vmap the pointwise fn."""
        assert theta.ndim == 2, f"evaluate_batch expects (M,d) batch, got {theta.shape}"
        self.batch_shapes.append((int(theta.shape[0]), int(theta.shape[1])))
        losses, grads = jax.vmap(self.pointwise)(theta)
        self.all_losses.extend(float(v) for v in losses)
        self.evals += int(theta.shape[0])
        return losses, grads, None

    def limit_check(self, limit: int) -> Callable[[], bool]:
        """Return a batch-level budget_check() False once `limit` evals served.

        The core consults it once per batch, it reads the evals already served,
        and the terminal batch therefore over-runs by less than one batch width.
        """

        def check() -> bool:
            return self.evals < limit

        return check


def test_batch_shape_and_per_point_counting() -> None:
    """evaluate_batch receives (n_paths,d) batches; each point counts exactly 1 eval.

    reported evals == points the seam actually served.
    """

    def quartic(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # A quartic, not a quadratic: one half-step of steepest descent lands a
        # 1-D quadratic exactly on its minimiser, where the gradient is zero and
        # every further trial is rejected - the run would stall long before the
        # round budget is spent, and this test is about the budget accounting.
        d = theta - 0.7
        return jnp.sum(d**4 + d**2), 4.0 * d**3 + 2.0 * d

    rec = Recorder(quartic)
    res = n_path_schedule(
        rec,
        unit_bounds(),
        jax.random.PRNGKey(0),
        budget_check=rec.limit_check(500),
        n_paths=4,
        max_steps=50,
        solver=HalfStep(),
    )
    assert isinstance(res, NPathResult)
    assert res.nfev == rec.evals, (res.nfev, rec.evals)
    assert res.nfev <= 500
    # nit counts engine rounds, not the linesearch rounds inside them (one round
    # may serve up to max_ls points per path), so it is bounded by the step cap
    # and stays far below the eval count - but it is not the cap itself: a run
    # that ends on all_stalled stops counting short of it.
    assert 0 < res.nit <= 50
    assert res.nfev >= res.nit * 4  # a round serves at least the n_paths paths
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
        unit_bounds(),
        jax.random.PRNGKey(1),
        budget_check=rec.limit_check(30),
        n_paths=8,
        max_steps=100,
        solver=HalfStep(),
    )
    assert res.nfev == rec.evals
    assert 30 <= res.nfev < 30 + 8  # at most one batch past the limit
    assert 0 < res.nit < 100  # the gate stopped it, not the step cap
    assert res.status == "budget_exhausted"

    # gate already False on entry: nothing evaluated at all
    rec0 = Recorder(quad)
    res0 = n_path_schedule(
        rec0,
        unit_bounds(),
        jax.random.PRNGKey(2),
        budget_check=rec0.limit_check(0),
        n_paths=4,
        solver=HalfStep(),
    )
    assert res0.nfev == 0
    assert res0.nit == 0  # never got past the first gate check
    assert res0.status == "budget_exhausted"


def test_fun_is_true_min_over_history() -> None:
    """Pin that fun is the min over every finite loss the seam returned.

    Not the final point, not a compressed history.
    """

    def quad(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.sum((theta - 0.7) ** 2), 2.0 * (theta - 0.7)

    rec = Recorder(quad)
    res = n_path_schedule(
        rec,
        unit_bounds(),
        jax.random.PRNGKey(3),
        budget_check=rec.limit_check(500),
        n_paths=4,
        max_steps=50,
        solver=HalfStep(),
    )
    assert rec.all_losses
    assert res.fun == min(rec.all_losses)
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
        unit_bounds(),
        jax.random.PRNGKey(2),
        budget_check=rec_k.limit_check(budget),
        n_paths=16,
        max_steps=100,
        solver=HalfStep(),
    )
    rec_1 = Recorder(pointwise)
    res_1 = n_path_schedule(
        rec_1,
        unit_bounds(),
        jax.random.PRNGKey(2),
        budget_check=rec_1.limit_check(budget),
        n_paths=1,
        max_steps=100,
        solver=HalfStep(),
    )
    # the single path must be trapped in the shallow well: replicate the
    # engine's n_paths=1 init key chain (split -> probe round 1, split -> round 2)
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
        unit_bounds(2),
        jax.random.PRNGKey(4),
        budget_check=rec.limit_check(500),
        n_paths=1,  # n_paths=1 -> sigma_r=0 -> never restart, pure projected descent
        max_steps=100,
        solver=HalfStep(),
    )
    assert res.fun < 0.4, res.fun  # must slide near (1, 0), f=0.25


def test_feasible_only_incumbent() -> None:
    """B (incumbent) is the min over *feasible* points, not the raw min.

    loss = (θ-0.2)^2 has its global min at θ=0.2, which the feas mask marks
    infeasible (θ <= 0.5); the best feasible design is θ=0.5+ (loss 0.09). The
    engine must report the feasible min even though the raw min over all evals
    sits lower (in the infeasible basin).
    """
    thr = 0.5

    def pointwise(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return (theta[0] - 0.2) ** 2.0, 2.0 * (theta - 0.2)

    all_losses: list[float] = []
    all_feas: list[bool] = []
    evals = 0

    def evaluate_batch(
        theta: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        nonlocal evals
        losses, grads = jax.vmap(pointwise)(theta)
        feas = theta[:, 0] > thr
        all_losses.extend(float(v) for v in losses)
        all_feas.extend(bool(v) for v in feas)
        evals += int(theta.shape[0])
        return losses, grads, feas

    def budget_check() -> bool:
        return evals < 600

    res = n_path_schedule(
        evaluate_batch,
        unit_bounds(),
        jax.random.PRNGKey(5),
        budget_check=budget_check,
        n_paths=4,
        max_steps=100,
        solver=HalfStep(),
    )

    feasible_losses = [L for L, ok in zip(all_losses, all_feas, strict=True) if ok]
    assert feasible_losses, "no feasible point ever served"
    # the raw min sits in the infeasible basin, below every feasible point
    assert min(all_losses) < min(feasible_losses)
    # B must be exactly the best feasible point, never the raw min
    assert res.fun == min(feasible_losses)
    assert res.x[0] > thr  # the reported best lies on the feasible side


def test_nan_first_candidate_does_not_mask_batch_min() -> None:
    """A NaN probe ranked first must not swallow the batch's real best point.

    The init probe serves [NaN, 0.01] (both feasible), so the engine must
    report best_loss == 0.01. `min(candidates, key=losses_list.__getitem__)`
    kept the NaN (NaN < x is False): the batch's real minimum — and its x —
    was discarded and best_loss stayed inf.
    """
    served = [0]

    def evaluate_batch(
        theta: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        served[0] += int(theta.shape[0])
        row0 = jnp.arange(theta.shape[0]) == 0
        nan = jnp.asarray(jnp.nan, dtype=theta.dtype)
        losses = jnp.where(row0, nan, jnp.asarray(0.01, dtype=theta.dtype))
        return losses, jnp.zeros_like(theta), jnp.ones(theta.shape[0], dtype=bool)

    res = n_path_schedule(
        evaluate_batch,
        unit_bounds(),
        jax.random.PRNGKey(6),
        budget_check=lambda: served[0] < 40,
        n_paths=2,
        max_steps=5,
        solver=HalfStep(),
    )
    assert res.fun == 0.01, res.fun
    assert bool(jnp.isfinite(res.x[0])), res.x


def run_probe_pass_with_nan_round(nan_round: int) -> None:
    """One n_paths=2 init pass serving NaN for the whole probe round `nan_round`."""
    n_paths = 2
    calls = [0]
    served = [0]
    batches: list[jnp.ndarray] = []
    starts: list[jnp.ndarray] = []

    def evaluate_batch(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        calls[0] += 1
        served[0] += int(theta.shape[0])
        if calls[0] <= 2:  # the two init probe rounds
            batches.append(theta)
        if calls[0] <= 2:  # probe rounds: one flat, one NaN (a whole round of it)
            # The finite probe round only has to be finite and distinguishable;
            # it is made clearly worse than the real objective below so the
            # first post-probe trial reads as an improvement (a probe value near
            # the real one would make it look like a worsening and be rejected).
            value = jnp.nan if calls[0] == nan_round else 1.0e3
            losses = jnp.full((theta.shape[0],), value, dtype=theta.dtype)
            grads = jnp.zeros_like(theta)
        else:  # past the probe: sloped, or every path rejects and burns the budget
            d = theta - 0.7
            losses = jnp.sum(d**4 + d**2, axis=1)
            grads = 4.0 * d**3 + 2.0 * d
        if calls[0] == 3:  # first post-init step: it must serve the starts
            starts.append(theta)
        return losses, grads, None

    res = n_path_schedule(
        evaluate_batch,
        unit_bounds(),
        jax.random.PRNGKey(7),
        budget_check=lambda: served[0] < 40,
        n_paths=n_paths,
        max_steps=5,
        solver=HalfStep(),
    )
    assert starts, "the engine never served a post-init step"
    assert res.status == "max_steps", (nan_round, res.status)  # no NaN stall
    finite_round = batches[0] if nan_round == 2 else batches[1]
    for j in range(n_paths):
        assert any(jnp.allclose(starts[0][j], pt) for pt in finite_round), (
            f"nan_round={nan_round}: start {j} is not the finite round's point"
        )


def test_probe_group_min_skips_nan() -> None:
    """A NaN probe round must not win the group-min: starts stay finite.

    One whole probe round serves NaN for every point while the other serves
    0.01, so every start must be its point from the finite round (asserted by
    membership in the recorded finite probe batch - no re-derivation of the
    engine's PRNG chain). Picking the NaN round's x would set that path's ZH
    reference to NaN: Armijo can never accept against NaN, so the path would
    stall for the rest of the run. Running the NaN round first and last also
    kills an "always take the last round" mis-implementation.
    """
    run_probe_pass_with_nan_round(1)
    run_probe_pass_with_nan_round(2)
    # helper contract: skip non-finite, None when nothing is finite
    assert argmin_finite([2.0, 1.0]) == 1
    assert argmin_finite([math.nan, 0.01, math.inf]) == 1
    assert argmin_finite([math.nan, math.inf]) is None


class ForcedWindow(EIRestartRule):
    """The delivery rule with its verdict window pinned (a test/arm knob).

    The rule derives its own window from the payoff pool, so a test
    that wants a literal cadence for an engine run states it here: it stays the
    same estimator and the same gate, only the length is forced. The delivery
    rule has no such parameter - a fixed cadence is a variant of it, not a knob
    on it.
    """

    def __init__(self, window: int) -> None:
        """Pin every path's stage to `window` rounds."""
        super().__init__()
        self._window = window

    def stage_window(self) -> int:
        """Return the pinned window (the pool's size is not consulted)."""
        return self._window


def test_probe_inf_group_min_does_not_silence_restarts() -> None:
    """An all-+inf probe group must not disable every restart in the flock.

    Column 0 is +inf in both init probe rounds (a group with no finite
    member), so the shared init group-min list carries an inf sample. Fitting
    it naively gives mu_r = inf, sigma_r = NaN and EI_restart = NaN, and
    `NaN > EI_continue` is False forever - no path would ever restart again
    even though B is finite. The extra probe batches served after the init
    phase prove the verdicts still fire (> the no-restart call count).
    """
    n_paths = 3
    max_steps = 3
    calls = [0]
    served = [0]
    batches: list[tuple[int, int]] = []
    init_losses = jnp.array([jnp.inf, 1.0, 3.0])  # row 0 = column 0 = dead

    def evaluate_batch(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        calls[0] += 1
        served[0] += int(theta.shape[0])
        batches.append((int(theta.shape[0]), int(theta.shape[1])))
        if calls[0] <= 2:  # init probe rounds: shared cold-start evidence
            losses = init_losses
        else:  # everything afterwards is finite and flat (Δ = 0, sigma_p = 0)
            losses = jnp.full((theta.shape[0],), 1.0, dtype=theta.dtype)
        return losses, jnp.zeros_like(theta), None

    res = n_path_schedule(
        evaluate_batch,
        unit_bounds(),
        jax.random.PRNGKey(0),
        budget_check=lambda: served[0] < 60,
        n_paths=n_paths,
        max_steps=max_steps,
        restart_rule=ForcedWindow(2),
        solver=HalfStep(),
    )
    assert res.fun == 1.0, res.fun  # B is finite
    # No restarts would serve exactly 2 init batches + max_steps lockstep ones
    # (measured: 5 at max_steps = 3 with the verdicts off).
    assert len(batches) > 2 + max_steps, (len(batches), batches)


def test_nonfinite_start_waits_for_ei_instead_of_stepping() -> None:
    """A start whose whole probe group is non-finite is stalled on arrival.

    It has no finite reference to descend from, so it must not burn budget
    while waiting for an EI verdict: with verdicts effectively off (huge
    window) exactly one eval per step is served after the two init
    rounds - the live path only.
    """
    n_paths = 2
    max_steps = 3
    served = [0]
    calls = [0]

    def evaluate_batch(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        calls[0] += 1
        served[0] += int(theta.shape[0])
        # Column 1 needs a real objective: on a flat field it rejects max_ls
        # times per step instead of committing one, and the per-step accounting
        # this test pins would measure the line search, not the stall.
        d = theta - 0.7
        losses = jnp.sum(d**4 + d**2, axis=1)
        grads = 4.0 * d**3 + 2.0 * d
        if calls[0] <= 2:  # the two init probe rounds: column 0 gets no evidence
            losses = losses.at[0].set(jnp.inf)
        return losses, grads, None

    res = n_path_schedule(
        evaluate_batch,
        unit_bounds(),
        jax.random.PRNGKey(0),
        budget_check=lambda: served[0] < 60,
        n_paths=n_paths,
        max_steps=max_steps,
        restart_rule=ForcedWindow(10**9),
        solver=HalfStep(),
    )
    assert res.nfev == 2 * n_paths + max_steps, res.nfev  # not 2 evals/step


def test_stalled_paths_wait_for_ei_then_restart() -> None:
    """max_ls exhaustion stalls a path; only a due EI verdict revives it.

    Init probes return loss = x[0] (a real spread, so sigma_r > 0 and B stays
    small); every later evaluation returns the same bad constant with a zero
    gradient, so ZH backtracking rejects max_ls times and the paths stall. With
    window = 2 (the shortest stage that has a spread at all) the stalled
    paths are verdict-due on the next even round with EI_continue = 0
    (f_k >= B) and EI_restart > 0, so the flock is revived by a restart probe;
    with verdicts off (huge window) the very same run serves nothing after the
    stall. The two traces differ by exactly the restart
    events - this is the engine-level wiring (stall -> EI -> restart) the unit
    tests of ei_restart/EI_continue cannot cover. The huge-period arm also pins
    the engine's stop for it: every path stalled and the next verdict lies
    ~10**9 rounds out, past max_steps, so the run must end there (status
    "all_stalled", one attempted round) rather than spin out the step budget.
    """
    n_paths, max_ls, max_steps, n_probe_rounds = 4, 3, 3, 2
    bad = 10.0

    def seam_run(window: int) -> tuple[list[tuple[int, int]], int, NPathResult]:
        shapes: list[tuple[int, int]] = []
        evals = [0]

        def evaluate_batch(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
            shapes.append((int(theta.shape[0]), int(theta.shape[1])))
            evals[0] += int(theta.shape[0])
            if len(shapes) <= n_probe_rounds:  # init probe rounds: real objective
                return theta[:, 0], jnp.ones_like(theta), None
            return (
                jnp.full((theta.shape[0],), bad, dtype=theta.dtype),
                jnp.zeros_like(theta),
                None,
            )

        res = n_path_schedule(
            evaluate_batch,
            unit_bounds(),
            jax.random.PRNGKey(8),
            # the step cap ends both runs
            budget_check=lambda: evals[0] < 10**6,
            n_paths=n_paths,
            max_steps=max_steps,
            max_ls=max_ls,
            restart_rule=ForcedWindow(window),
            n_probe_rounds=n_probe_rounds,
            solver=HalfStep(),
        )
        return shapes, evals[0], res

    shapes_off, evals_off, res_off = seam_run(10**9)  # verdicts never due
    shapes_on, evals_on, res_on = seam_run(2)  # verdict every second step
    # without verdicts: init + max_ls rejecting trials per path, then nothing
    pre = n_probe_rounds + max_ls
    assert len(shapes_off) == pre, shapes_off
    assert evals_off == (n_probe_rounds + max_ls) * n_paths
    # step 2 is due, so the flock is revived: a verdict round, then a restart
    # probe of every path. The tail is left as an inequality on purpose - what
    # the revived paths do after the probe (accept, or reject again on the same
    # flat field until the step cap) is the injected method's business.
    assert len(shapes_on) - pre >= 1 + n_probe_rounds, shapes_on
    assert res_off.status == "all_stalled"  # nothing can revive the flock
    assert res_on.status == "max_steps"  # revived, then capped
    assert all(m == n_paths for m, _d in shapes_on), shapes_on  # no shrinking batches
    assert evals_on == len(shapes_on) * n_paths > evals_off
    assert res_on.status == "max_steps", res_on  # a due verdict keeps it alive
    assert res_off.status == "all_stalled", res_off  # not "budget_exhausted"
    # The unreachable verdict ends the run at the stall instead of spinning the
    # 3 - 1 = 2 remaining rounds: the stall is the one productive round.
    assert res_off.nit == 1, res_off
    assert res_on.nit == max_steps, res_on


def test_bounds_spellings_are_the_same_box() -> None:
    """scipy-style pairs and the historical (d, 2) array run byte-identically.

    One key stream, one draw sequence: if the two spellings normalized
    differently, the probes would land elsewhere and every recorded loss would
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
            bounds,
            jax.random.PRNGKey(3),
            budget_check=rec.limit_check(80),
            n_paths=4,
            max_steps=10,
            solver=HalfStep(),
        )
        return rec.all_losses, res.fun

    losses_arr, fun_arr = run(unit_bounds(2))
    losses_pairs, fun_pairs = run([(0.0, 1.0)] * 2)
    assert losses_pairs == losses_arr
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
                bounds,
                jax.random.PRNGKey(0),
                n_paths=2,
                max_steps=1,
                solver=HalfStep(),
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
        None,
        jax.random.PRNGKey(4),
        sample=sample,
        budget_check=rec_none.limit_check(60),
        n_paths=2,
        max_steps=3,
        solver=HalfStep(),
    )
    assert all(d == 3 for _m, d in rec_none.batch_shapes), rec_none.batch_shapes
    assert res.x.shape == (3,)

    rec_pairs = Recorder(quad)
    n_path_schedule(
        rec_pairs,
        [(None, None)] * 3,
        jax.random.PRNGKey(4),
        sample=sample,
        budget_check=rec_pairs.limit_check(60),
        n_paths=2,
        max_steps=3,
        solver=HalfStep(),
    )
    assert rec_pairs.all_losses == rec_none.all_losses

    # a box and a sampler that disagree on d is the wiring bug this catches
    try:
        n_path_schedule(
            Recorder(quad),
            unit_bounds(2),
            jax.random.PRNGKey(4),
            sample=sample,
            n_paths=2,
            max_steps=1,
            solver=HalfStep(),
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
        [(None, None)],
        jax.random.PRNGKey(5),
        sample=sample,
        budget_check=lambda: True,
        n_paths=4,
        max_steps=40,
        solver=HalfStep(),
    )
    assert max(seen) > 1.0, seen  # a finite [0,1] box could not have gone there
    assert res.fun < 1e-6, res.fun


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"all {len(fns)} passed")
