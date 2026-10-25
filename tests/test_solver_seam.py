"""Solver-seam self-checks.

The seam's contract is only proven by injection: a stub `LocalSolver` and a stub
`RestartRule` are plugged into the engine, which must drive them instead of its
defaults, and a foreign box (`bounds` as data) plus the caller's own x->θ map
composed into `evaluate_batch` must be honoured too. These also pin the two
documented zero-caller hooks — a method
returning `"give_up"` and a run with no extra gate (`budget_check=None` ->
`status == "max_iters"`, the horizon alone).

Run with:

    uv run python -m tests.test_solver_seam
"""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp

from soar_lite.engine import PathPoint, Verdict, n_path_schedule

jax.config.update("jax_enable_x64", val=True)

PROBE_EVALS = 1  # the shipping probe depth: sizes above 1 need the f-only seam


def unit_bounds(d: int = 1) -> jnp.ndarray:
    """Return the engine's box in working coords: x ∈ [0,1]^d."""
    return jnp.stack([jnp.zeros(d), jnp.ones(d)], axis=1)


def pointwise(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """F = (θ - 0.25)^2, grad = 2(θ - 0.25)."""
    return jnp.sum((theta - 0.25) ** 2, axis=-1), 2.0 * (theta - 0.25)


def evaluator() -> tuple[
    Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, None]],
    list[tuple[int, int]],
]:
    """Return (evaluate_batch, recorded batch shapes)."""
    shapes: list[tuple[int, int]] = []

    def evaluate_batch(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        shapes.append((int(x.shape[0]), int(x.shape[1])))
        f_vals, grads = jax.vmap(pointwise)(x)
        return f_vals, grads, None

    return evaluate_batch, shapes


def f_only(
    evaluate_batch: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, None]],
) -> Callable[[jnp.ndarray], tuple[jnp.ndarray, None]]:
    """Derive the f-only sibling, so a test needs one objective, not two."""

    def evaluate_f(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        f_vals, _grads, feas = evaluate_batch(x)
        return f_vals, feas

    return evaluate_f


def batches_below(shapes: list[tuple[int, int]], limit: int) -> Callable[[], bool]:
    """Return a batch gate that closes once `limit` batches have been served."""
    return lambda: len(shapes) < limit


class StubSolver:
    """Accepts every trial on its first verdict, or gives up instead.

    Records the calls the engine makes, so the test can prove the engine drove
    the injected method rather than its default `BBZhangHager`.
    """

    def __init__(self, *, give_up: bool = False) -> None:
        """Store the verdict policy; give_up stalls on the first verdict."""
        self.give_up = give_up
        self.inited: list[int] = []
        self.verdicts: list[int] = []
        self.folds: list[int] = []

    def restart(self, i: int, p: PathPoint) -> None:  # noqa: ARG002
        """Record path i's cold start (the engine's only seed call)."""
        self.inited.append(i)

    def propose(self, i: int, p: PathPoint) -> jnp.ndarray:  # noqa: ARG002
        """Return a halved steepest-descent step from p."""
        return p.x - 0.5 * p.g

    def verdict(self, i: int, p: PathPoint, trial: PathPoint) -> Verdict:  # noqa: ARG002
        """Record the verdict: accept, or give up on the first trial."""
        self.verdicts.append(i)
        return "give_up" if self.give_up else "accept"

    def fold(self, i: int, p: PathPoint, best: PathPoint) -> None:  # noqa: ARG002
        """Record the folded committed step; this stub folds nothing."""
        self.folds.append(i)


class RecordingRule:
    """Records the engine's rule-side calls; restarts only the cold start."""

    def __init__(self) -> None:
        """Start with empty evidence, no restarts and no verdicts."""
        self.group_mins: list[float] = []
        self.deltas: list[tuple[int, float]] = []
        self.restarts: list[int] = []
        self.verdicts: list[int] = []

    def observe_continue(self, i: int, delta: float) -> None:
        """Record one f change delta for path i."""
        self.deltas.append((i, delta))

    def observe_restart(self, r: float) -> None:
        """Record one restart event's payoff r (the pool carries no path index)."""
        self.group_mins.append(r)

    def should_restart(self, i: int, f_k: float, incumbent: float) -> bool:  # noqa: ARG002
        """Record the verdict call; restart only while the pool is empty."""
        self.verdicts.append(i)
        return not self.group_mins  # the engine's initialization probe

    def restart(self, i: int) -> None:
        """Record path i's restart notification."""
        self.restarts.append(i)


class AlwaysRestartRule(RecordingRule):
    """Same recorder, but every verdict says restart."""

    def should_restart(self, i: int, f_k: float, incumbent: float) -> bool:  # noqa: ARG002
        """Record the verdict call for path i and always restart."""
        self.verdicts.append(i)
        return True


def test_injected_solver_is_driven() -> None:
    """The engine calls the injected method, never its default."""
    k = 3
    evaluate_batch, shapes = evaluator()
    solver = StubSolver()
    res = n_path_schedule(
        evaluate_batch,
        bounds=unit_bounds(),
        budget_check=lambda: True,
        n_paths=k,
        max_iters=PROBE_EVALS + 1,  # iteration 0 is the init probe
        n_probe_evals=PROBE_EVALS,
        solver=solver,
        restart_rule=RecordingRule(),
        key=jax.random.PRNGKey(0),
    )
    assert solver.inited == [0, 1, 2], solver.inited
    assert solver.verdicts == [0, 1, 2], solver.verdicts  # one trial, then accept
    assert res.n_points == sum(w for w, _d in shapes) == PROBE_EVALS * k + k
    assert res.status == "max_iters"


def test_give_up_stalls_a_path() -> None:
    """A "give_up" verdict stalls that path at once: no further trial for it."""
    k = 3
    evaluate_batch, shapes = evaluator()
    solver = StubSolver(give_up=True)
    res = n_path_schedule(
        evaluate_batch,
        bounds=unit_bounds(),
        budget_check=lambda: True,
        n_paths=k,
        max_iters=PROBE_EVALS + 1,  # iteration 0 is the init probe
        n_probe_evals=PROBE_EVALS,
        solver=solver,
        restart_rule=RecordingRule(),
        key=jax.random.PRNGKey(1),
    )
    assert solver.verdicts == [0, 1, 2]  # one verdict each, then stalled
    # init probe + the single stalled iteration; the rest serve nothing
    assert res.n_points == sum(w for w, _d in shapes) == PROBE_EVALS * k + k
    assert res.status == "max_iters"


def test_injected_rule_is_driven() -> None:
    """The engine feeds its one injected rule instance all the bookkeeping."""
    k = 2
    evaluate_batch, _shapes = evaluator()
    rule = RecordingRule()
    res = n_path_schedule(
        evaluate_batch,
        bounds=unit_bounds(),
        budget_check=lambda: True,
        n_paths=k,
        max_iters=PROBE_EVALS + PROBE_EVALS,  # init probe + one step
        n_probe_evals=PROBE_EVALS,
        solver=StubSolver(),
        restart_rule=rule,
        key=jax.random.PRNGKey(2),
    )
    n_iterations = PROBE_EVALS  # max_iters minus the init probe's iteration
    assert len(rule.group_mins) == k  # cold start: the n_paths init group-mins, once
    # Iteration 0 is the initialization probe, so the cold-start ask restarts
    # every path before any step; the one iteration the budget still buys steps.
    assert rule.verdicts == [0, 1] * (n_iterations + 1), rule.verdicts
    assert rule.restarts == [0, 1], rule.restarts  # the cold start is one too
    assert [i for i, _delta in rule.deltas] == [0, 1] * n_iterations  # Δf per step
    assert res.status == "max_iters"


def test_rule_can_force_restarts() -> None:
    """A rule that always says restart re-probes every iteration, never steps."""
    k = 2
    evaluate_batch, shapes = evaluator()
    solver = StubSolver()
    rule = AlwaysRestartRule()
    res = n_path_schedule(
        evaluate_batch,
        bounds=unit_bounds(),
        budget_check=lambda: True,
        n_paths=k,
        # Each iteration restarts every path, so each one buys a probe
        # (PROBE_EVALS batches) and no step: a restarted path is skipped by that
        # iteration's step.
        max_iters=3,  # the init probe plus two restart iterations
        n_probe_evals=PROBE_EVALS,
        solver=solver,
        restart_rule=rule,
        key=jax.random.PRNGKey(3),
    )
    # init probe + two restart events, and nothing else
    assert res.n_points == sum(w for w, _d in shapes)
    assert res.n_points == PROBE_EVALS * k + 2 * (PROBE_EVALS * k)
    # The cold start is a restart event too: every path restarts on
    # iteration 0 and on each of the two iterations after it.
    assert rule.restarts == [0, 1] * 3  # every iteration, cold start first
    assert len(rule.group_mins) == 3 * k  # init + two restart events
    assert solver.verdicts == []  # restarted, not stepped
    assert res.status == "max_iters"


def test_the_gate_counts_g_batches_not_f_only_draws() -> None:
    """Probes are billed, so a restart storm cannot serve past the gate.

    The expensive event in the engine is the restart probe, and the deepest
    accounting bug this engine had was that the probes were never charged: a
    rule that restarts every iteration would then have served far more batches
    than the caller allowed. What is charged now is the fused value+gradient
    pass - the half that dominates - and a probe's extra f-only draws are
    cheaper work that is neither billed nor gated. A storm is still bounded,
    because every probe pays exactly one charged batch.
    """
    k, probe_evals = 3, 2  # a probe serves 2 f-only draws, then 1 fused batch

    def run(limit: int) -> tuple[int, int, int, str]:
        """Serve under one gate limit; return (billed, drawn, n_points, status)."""
        billed = 0  # g batches: the only billed calls
        drawn = 0  # f-only draws: served, never billed

        def evaluate_batch(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
            nonlocal billed
            billed += 1
            f_vals, grads = jax.vmap(pointwise)(x)
            return f_vals, grads, None

        def evaluate_f(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
            nonlocal drawn
            drawn += 1
            f_vals, _grads = jax.vmap(pointwise)(x)
            return f_vals, None

        res = n_path_schedule(
            evaluate_batch,
            evaluate_f=evaluate_f,
            bounds=unit_bounds(),
            budget_check=lambda: billed < limit,
            n_paths=k,
            max_iters=10,  # not the binding stop: the gate closes first
            n_probe_evals=probe_evals,
            solver=StubSolver(),
            restart_rule=AlwaysRestartRule(),
            key=jax.random.PRNGKey(7),
        )
        return billed, drawn, res.n_points, res.status

    for limit in (1, 2, 4, 7):
        billed, drawn, n_points, status = run(limit)
        # The gate closes on the billed count alone: never one g batch past it.
        assert billed == limit, (limit, billed)
        assert n_points == limit * k  # every billed batch covers all k paths
        assert status == "budget_exhausted", (limit, status)
        # The draws did happen - one batch per probe - and stayed out of both
        # counters: only the fused pass is billed.
        assert drawn == limit * probe_evals, (limit, drawn)


def test_a_run_without_an_extra_gate_reports_max_iters() -> None:
    """No gate passed at all (the keyword defaults to None): the horizon only."""
    k = 2
    evaluate_batch, shapes = evaluator()
    res = n_path_schedule(
        evaluate_batch,
        bounds=unit_bounds(),
        n_paths=k,
        max_iters=PROBE_EVALS + 3,  # the init probe plus three steps
        n_probe_evals=PROBE_EVALS,
        solver=StubSolver(),
        restart_rule=RecordingRule(),
        key=jax.random.PRNGKey(4),
    )
    assert res.status == "max_iters", res
    assert res.n_points == sum(w for w, _d in shapes) == PROBE_EVALS * k + 3 * k


def test_box_is_data_not_a_unit_cube_assumption() -> None:
    """A foreign box [1,2] plus the caller's own θ = x + 1 map: both are honoured.

    The engine must sample and clip inside the box it was handed, and must never
    see the caller's parameterization (that map is composed into
    `evaluate_batch`). Every trial here lands outside the box - the landscape's
    minimizer x = 3 is not in it - so the clip is load-bearing, not incidental.
    """
    lo, hi = 1.0, 2.0
    served: list[float] = []

    def evaluate_batch(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        """Serve θ = x + 1, f = (θ - 4)²: the caller's map and landscape."""
        served.extend(float(v) for v in x[:, 0])
        theta = x + 1.0
        return jnp.sum((theta - 4.0) ** 2, axis=-1), 2.0 * (theta - 4.0), None

    bounds = jnp.stack([jnp.full(1, lo), jnp.full(1, hi)], axis=1)
    res = n_path_schedule(
        evaluate_batch,
        bounds=bounds,
        budget_check=None,
        n_paths=2,
        max_iters=3,
        n_probe_evals=PROBE_EVALS,
        solver=StubSolver(),  # step = x - 0.5·g = 3 exactly: outside the box
        restart_rule=RecordingRule(),
        key=jax.random.PRNGKey(5),
    )
    assert served, "nothing was ever served"
    assert min(served) >= lo, min(served)
    assert max(served) <= hi, max(served)
    # the clip really fired (the raw trial was 3.0) and the best point sits on it
    assert abs(max(served) - hi) < 1e-12, max(served)
    assert abs(float(res.x[0]) - hi) < 1e-12, res.x
    assert res.status == "max_iters", res


def test_a_box_uniform_sampler_reproduces_the_default_prior() -> None:
    """Pin that handing over the engine's own finite-box draw changes nothing.

    The engine's *default* prior is an i.i.d. uniform draw over `bounds`, and a
    caller may inject its own with `sample=`. This pins that a sampler spelling
    `default_sample` verbatim - same key, same call - reproduces the run
    *exactly*, not just in distribution: the injected prior then *is* the
    default one, so a finite-box number recorded with the default stays
    comparable with one recorded after a sampler was handed over. Every restart
    verdict says restart, so both draw sites (init probe and restart) are
    exercised.
    """
    box = unit_bounds(2)
    lo, hi = box[:, 0], box[:, 1]

    def draw(key: jnp.ndarray, n: int) -> jnp.ndarray:
        """Spell out the engine's own prior here instead of inside it."""
        return jax.random.uniform(key, (n, box.shape[0]), minval=lo, maxval=hi)

    def run(
        sample: Callable[[jnp.ndarray, int], jnp.ndarray] | None,
    ) -> tuple[list[jnp.ndarray], jnp.ndarray, float, int]:
        served: list[jnp.ndarray] = []

        def evaluate_batch(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
            served.append(x)
            f_vals, grads = jax.vmap(pointwise)(x)
            return f_vals, grads, None

        res = n_path_schedule(
            evaluate_batch,
            bounds=box,
            sample=sample,
            budget_check=None,
            n_paths=2,
            max_iters=3,
            n_probe_evals=PROBE_EVALS,
            solver=StubSolver(),
            restart_rule=AlwaysRestartRule(),
            key=jax.random.PRNGKey(7),
        )
        return served, res.x, res.fun, res.n_points

    default_served, default_x, default_fun, default_points = run(None)
    ours_served, ours_x, ours_fun, ours_points = run(draw)
    assert default_served, "nothing was ever served"
    assert len(default_served) == len(ours_served)
    for a, b in zip(default_served, ours_served, strict=True):
        assert jnp.array_equal(a, b), (a, b)
    assert jnp.array_equal(default_x, ours_x), (default_x, ours_x)
    assert default_fun == ours_fun, (default_fun, ours_fun)
    assert default_points == ours_points, (default_points, ours_points)


def test_every_path_feeds_once_per_iteration() -> None:
    """Pin that every path feeds observe_continue once per iteration.

    Column 0 is non-finite in both init probe evals, so path 0 arrives
    stalled and never steps; the engine must still feed it one
    `observe_continue(0, inf)` per iteration, so its sample count advances in
    lockstep with its advancing cohort instead of leading it by one verdict
    (the off-by-one a top-of-iteration stall count would cause). A continue
    sample is one step, so three iterations take three iterations.
    """
    k = 2
    calls = [0]

    def evaluate_batch(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        calls[0] += 1
        f_vals = jnp.full((x.shape[0],), 1.0, dtype=x.dtype)
        if calls[0] <= PROBE_EVALS:  # init probe evals: column 0 has no evidence
            f_vals = f_vals.at[0].set(jnp.inf)
        return f_vals, jnp.zeros_like(x), None

    rule = RecordingRule()
    res = n_path_schedule(
        evaluate_batch,
        bounds=unit_bounds(),
        budget_check=lambda: True,
        n_paths=k,
        max_iters=PROBE_EVALS + 3,  # init probe + three iterations
        n_probe_evals=PROBE_EVALS,
        solver=StubSolver(),
        restart_rule=rule,
        key=jax.random.PRNGKey(6),
    )
    assert res.status == "max_iters", res
    counts: dict[int, int] = {}
    for i, _delta in rule.deltas:
        counts[i] = counts.get(i, 0) + 1
    assert counts == {0: 3, 1: 3}, counts
    assert [d for i, d in rule.deltas if i == 0] == [math.inf] * 3


def test_a_probes_iteration_is_not_read_back_as_a_zero_step() -> None:
    """A path that probed folds nothing that iteration: no Δ = 0 is reported.

    The probe's f becomes the Δ anchor and the probe's iteration is skipped, so
    the first `observe_continue` for a path carries the step that follows its
    probe. Reading the probe back as `f - f = 0` would hand the rule a fake
    "nothing changed" sample on every restart and on the initialization itself.
    """
    k = 2
    rule = RecordingRule()

    def evaluate_batch(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        # A ramp: distinct points have distinct f values, so a reported Δ of 0
        # can only come from comparing a point with itself.
        f_vals = 10.0 * x[:, 0]
        return f_vals, jnp.full_like(x, 10.0), None

    res = n_path_schedule(
        evaluate_batch,
        bounds=unit_bounds(),
        budget_check=lambda: True,
        n_paths=k,
        max_iters=PROBE_EVALS + 1,  # init probe + one step
        n_probe_evals=PROBE_EVALS,
        solver=StubSolver(),
        restart_rule=rule,
        key=jax.random.PRNGKey(9),
    )
    # StubSolver steps x -> x - 0.5·g = x - 5, which the box clips to 0, so every
    # step is a strict decrease and the probe's own f never reappears as a zero.
    deltas = [d for _i, d in rule.deltas]
    assert len(deltas) == k, rule.deltas  # one step each, after its probe
    assert all(d < 0.0 for d in deltas), deltas
    assert res.n_batches == PROBE_EVALS + 1, res


def test_one_sample_per_iteration_at_every_probe_size() -> None:
    """Each iteration folds one sample per path, at every probe size.

    The continue side's sample is one step, so it does not depend on
    n_probe_evals: after the init probe, M iterations of budget buy exactly M
    samples per path. This is what changed when the horizon clock went away - a
    sample used to span n_probe_evals iterations, so the same budget bought
    fewer of them. Sizes above 1 take the f-only seam, whose single extra batch
    for the winners is all the probe's depth costs in the batch count.
    """
    k = 2
    for probe_evals in (1, 2, 3, 4):
        evaluate_batch, _shapes = evaluator()
        rule = RecordingRule()
        evaluate_f = None if probe_evals == 1 else f_only(evaluate_batch)
        res = n_path_schedule(
            evaluate_batch,
            evaluate_f=evaluate_f,
            bounds=unit_bounds(),
            n_paths=k,
            max_iters=3,  # init probe + two steps
            n_probe_evals=probe_evals,
            solver=StubSolver(),
            restart_rule=rule,
            key=jax.random.PRNGKey(5),
        )
        counts: dict[int, int] = {}
        for i, _delta in rule.deltas:
            counts[i] = counts.get(i, 0) + 1
        assert counts == {0: 2, 1: 2}, (probe_evals, counts)
        assert res.status == "max_iters", (probe_evals, res)


def test_reports_points_and_batches_separately() -> None:
    """n_points and n_batches are two units, and both are always reported.

    A batch is the points one eval serves, so a run of k paths that serves the
    same batches reports k times the points of a k=1 run, and the same batch
    count. There is no billing mode to switch: the serial cost is `n_points`,
    the parallel wall-clock cost is `n_batches`.
    """

    def run(k: int) -> tuple[int, int]:
        evaluate_batch, _shapes = evaluator()
        res = n_path_schedule(
            evaluate_batch,
            bounds=unit_bounds(),
            budget_check=lambda: True,
            n_paths=k,
            max_iters=PROBE_EVALS + 1,  # init probe + one accepted trial
            n_probe_evals=PROBE_EVALS,
            solver=StubSolver(),
            restart_rule=RecordingRule(),
            key=jax.random.PRNGKey(8),
        )
        return res.n_points, res.n_batches

    # one probe of PROBE_EVALS(=1) batch, then one accepted trial batch
    batches = PROBE_EVALS + 1  # = 2, independent of n_paths
    points_3, batches_3 = run(3)
    points_1, batches_1 = run(1)
    assert batches_3 == batches_1 == batches
    assert points_3 == batches * 3  # 6 points
    assert points_1 == batches  # 2 points


def test_multi_draw_probe_is_refused_without_the_f_only_seam() -> None:
    """n_probe_evals > 1 needs evaluate_f: it would price g once per draw.

    The probe's cost asymmetry is in gradient counts, not batch counts - a step
    computes g once while an un-seamed multi-draw probe computes it once per
    draw - so the engine refuses the combination instead of silently overselling
    the probe side.
    """
    evaluate_batch, _shapes = evaluator()
    refused: str | None = None
    try:
        n_path_schedule(
            evaluate_batch,
            bounds=unit_bounds(),
            n_paths=2,
            n_probe_evals=2,
            max_iters=8,
            solver=StubSolver(),
            key=jax.random.PRNGKey(0),
        )
    except ValueError as e:
        refused = str(e)
    assert refused is not None, "n_probe_evals=2 was accepted without evaluate_f"
    assert "evaluate_f" in refused, refused


def test_f_only_probe_computes_one_gradient_per_probe() -> None:
    """An f-only probe: n_probe_evals f-only draws, then one fused batch.

    That single fused batch is what makes n_probe_evals > 1 cost-matched with a
    one-gradient step: every draw costs an f, the winners then pay the one
    gradient between them. Only that fused batch is billed. The group-min still
    picks per column across the draws, so the batch re-serves the same winners
    rather than making a new choice.
    """
    k, probe_evals = 2, 3
    f_calls: list[jnp.ndarray] = []
    fused_calls: list[jnp.ndarray] = []

    def evaluate_batch(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        fused_calls.append(x)
        f_vals, grads = jax.vmap(pointwise)(x)
        return f_vals, grads, None

    def evaluate_f(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        f_calls.append(x)
        f_vals, _grads = jax.vmap(pointwise)(x)
        return f_vals, None

    res = n_path_schedule(
        evaluate_batch,
        evaluate_f=evaluate_f,
        bounds=unit_bounds(),
        n_paths=k,
        n_probe_evals=probe_evals,
        max_iters=1,  # the probe alone is one iteration
        solver=StubSolver(),
        key=jax.random.PRNGKey(1),
    )
    assert len(f_calls) == probe_evals, f_calls
    assert len(fused_calls) == 1, fused_calls
    assert all(call.shape == (k, 1) for call in f_calls), f_calls
    # Column j's winner is the best of the draws' column j, so the fused batch
    # re-serves exactly those picks.
    f_vals = jnp.stack([jnp.sum((call - 0.25) ** 2, axis=-1) for call in f_calls])
    best = jnp.argmin(f_vals, axis=0)
    for j in range(k):
        assert jnp.allclose(fused_calls[0][j], f_calls[int(best[j])][j]), j
    # Only the fused pass is billed: the f-only draws are not budget events.
    assert res.n_batches == 1, res
    assert res.n_points == k, res
    assert res.status == "max_iters", res


def test_a_single_draw_probe_keeps_the_fused_path() -> None:
    """One draw plus evaluate_f: no f-only call, just the one fused batch.

    A probe with a single draw has no losing gradient to defer - the winner is
    the draw - so serving it f-only and re-serving the winner for its gradient
    would pay an extra f for nothing. Deferral is a property of the probe's
    depth, not of the seam being present.
    """
    f_calls: list[int] = []

    def evaluate_batch(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        f_vals, grads = jax.vmap(pointwise)(x)
        return f_vals, grads, None

    def evaluate_f(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        f_calls.append(int(x.shape[0]))
        f_vals, _grads = jax.vmap(pointwise)(x)
        return f_vals, None

    res = n_path_schedule(
        evaluate_batch,
        evaluate_f=evaluate_f,
        bounds=unit_bounds(),
        n_paths=2,
        n_probe_evals=1,
        max_iters=1,  # the probe alone is one iteration
        solver=StubSolver(),
        key=jax.random.PRNGKey(3),
    )
    assert f_calls == [], f_calls
    assert res.n_batches == 1, res
    assert res.n_points == 2, res


def test_a_non_finite_gradient_start_arrives_stalled() -> None:
    """A start is committed only when f *and* g are finite.

    A finite f with a non-finite gradient is not a usable start: `p.g` is what
    the method proposes from and folds on, so one NaN in it poisons that
    coordinate's magnitude EMAs for the rest of the run. Such a start arrives
    stalled exactly like a dead draw and waits for an EI verdict instead of
    stepping.
    """

    def evaluate_batch(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        f_vals, _grads = jax.vmap(pointwise)(x)
        return f_vals, jnp.full_like(x, jnp.nan), None

    solver = StubSolver()
    res = n_path_schedule(
        evaluate_batch,
        bounds=unit_bounds(),
        n_paths=2,
        n_probe_evals=1,
        max_iters=2,
        solver=solver,
        key=jax.random.PRNGKey(4),
    )
    # No path was ever live, so nothing stepped and nothing was judged.
    assert solver.verdicts == [], solver.verdicts
    assert solver.folds == [], solver.folds
    assert res.status == "max_iters", res


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"all {len(fns)} passed")
