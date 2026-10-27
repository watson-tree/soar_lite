"""Solver-seam self-checks.

The seam's contract is only proven by injection: a stub `LocalSolver` and a stub
`RestartRule` are plugged into the engine, which must drive them instead of its
defaults, and a foreign box (`bounds` as data) plus the caller's own x->θ map
composed into `eval_f_g` must be honoured too. These also pin the two
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

PROBE_DRAWS = 1  # the shipping probe depth: sizes above 1 need the f-only seam


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
    """Return (eval_f_g, recorded batch shapes)."""
    shapes: list[tuple[int, int]] = []

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        shapes.append((int(x.shape[0]), int(x.shape[1])))
        f_vals, grads = jax.vmap(pointwise)(x)
        return f_vals, grads, None

    return eval_f_g, shapes


def f_only(
    eval_f_g: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, None]],
) -> Callable[[jnp.ndarray], tuple[jnp.ndarray, None]]:
    """Derive the f-only sibling, so a test needs one objective, not two."""

    def eval_f(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        f_vals, _grads, feas = eval_f_g(x)
        return f_vals, feas

    return eval_f


def batches_below(shapes: list[tuple[int, int]], limit: int) -> Callable[[], bool]:
    """Return a batch gate that closes once `limit` batches have been served."""
    return lambda: len(shapes) < limit


class StubSolver:
    """Accepts every trial on its first verdict, or gives up instead.

    Records the calls the engine makes, so the test can prove the engine drove
    the injected method rather than its default `DiagonalPreconditioner`.
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

    def needs_verdict(self, i: int) -> bool:  # noqa: ARG002
        """Rule every trial: this stub's `give_up` policy needs a ruling to fire."""
        return True

    def verdict(self, i: int, p: PathPoint, x: jnp.ndarray, f: float) -> Verdict:  # noqa: ARG002
        """Record the verdict: accept, or give up on the first trial."""
        self.verdicts.append(i)
        return "give_up" if self.give_up else "accept"

    def fold(self, i: int, p: PathPoint, best: PathPoint) -> None:  # noqa: ARG002
        """Record the folded committed step; this stub folds nothing."""
        self.folds.append(i)


class UnjudgedSolver(StubSolver):
    """A method whose step commits its one proposal unread: no verdict needed."""

    def needs_verdict(self, i: int) -> bool:  # noqa: ARG002
        """Never rule: every step is taken as proposed."""
        return False


class PushTo(UnjudgedSolver):
    """Propose one fixed point, unruled, to place a chosen value under the gate."""

    def __init__(self, x_star: float) -> None:
        """Store the point every path proposes."""
        super().__init__()
        self.x_star = x_star

    def propose(self, i: int, p: PathPoint) -> jnp.ndarray:  # noqa: ARG002
        """Return the stored point, whatever the start."""
        return jnp.full_like(p.x, self.x_star)


class RecordingRule:
    """Records the engine's rule-side calls; restarts only the cold start."""

    def __init__(self) -> None:
        """Start with empty evidence, no restarts and no verdicts."""
        self.group_mins: list[float] = []
        self.deltas: list[tuple[int, float | None]] = []
        self.restarts: list[int] = []
        self.verdicts: list[int] = []

    def observe_continue(self, i: int, delta: float | None) -> None:
        """Record one f change (or a stall tick, `None`) for path i."""
        self.deltas.append((i, delta))

    def observe_restart(self, group_min: float) -> None:
        """Record one restart event's group-min (the pool carries no path index)."""
        self.group_mins.append(group_min)

    def should_restart(self, i: int, f_k: float, best_fun_raw: float) -> bool:  # noqa: ARG002
        """Record the verdict call; restart only while the pool is empty."""
        self.verdicts.append(i)
        return not self.group_mins  # the engine's initialization probe

    def restart(self, i: int) -> None:
        """Record path i's restart notification."""
        self.restarts.append(i)


class AlwaysRestartRule(RecordingRule):
    """Same recorder, but every verdict says restart."""

    def should_restart(self, i: int, f_k: float, best_fun_raw: float) -> bool:  # noqa: ARG002
        """Record the verdict call for path i and always restart."""
        self.verdicts.append(i)
        return True


def test_injected_solver_is_driven() -> None:
    """The engine calls the injected method, never its default."""
    k = 3
    eval_f_g, shapes = evaluator()
    solver = StubSolver()
    res = n_path_schedule(
        eval_f_g,
        bounds=unit_bounds(),
        budget_check=lambda: True,
        n_paths=k,
        max_iters=PROBE_DRAWS + 1,  # iteration 0 is the init probe
        n_draws=PROBE_DRAWS,
        solver=solver,
        restart_rule=RecordingRule(),
        key=jax.random.PRNGKey(0),
    )
    assert solver.inited == [0, 1, 2], solver.inited
    assert solver.verdicts == [0, 1, 2], solver.verdicts  # one trial, then accept
    assert res.n_points == sum(w for w, _d in shapes) == PROBE_DRAWS * k + k
    assert res.status == "max_iters"


def test_give_up_stalls_a_path() -> None:
    """A "give_up" verdict stalls that path at once: no further trial for it."""
    k = 3
    eval_f_g, shapes = evaluator()
    solver = StubSolver(give_up=True)
    res = n_path_schedule(
        eval_f_g,
        bounds=unit_bounds(),
        budget_check=lambda: True,
        n_paths=k,
        max_iters=PROBE_DRAWS + 1,  # iteration 0 is the init probe
        n_draws=PROBE_DRAWS,
        solver=solver,
        restart_rule=RecordingRule(),
        key=jax.random.PRNGKey(1),
    )
    assert solver.verdicts == [0, 1, 2]  # one verdict each, then stalled
    # init probe + the single stalled iteration; the rest serve nothing
    assert res.n_points == sum(w for w, _d in shapes) == PROBE_DRAWS * k + k
    assert res.status == "max_iters"


def test_injected_rule_is_driven() -> None:
    """The engine feeds its one injected rule instance all the bookkeeping."""
    k = 2
    eval_f_g, _shapes = evaluator()
    rule = RecordingRule()
    res = n_path_schedule(
        eval_f_g,
        bounds=unit_bounds(),
        budget_check=lambda: True,
        n_paths=k,
        max_iters=PROBE_DRAWS + PROBE_DRAWS,  # init probe + one step
        n_draws=PROBE_DRAWS,
        solver=StubSolver(),
        restart_rule=rule,
        key=jax.random.PRNGKey(2),
    )
    n_iterations = PROBE_DRAWS  # max_iters minus the init probe's iteration
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
    eval_f_g, shapes = evaluator()
    solver = StubSolver()
    rule = AlwaysRestartRule()
    res = n_path_schedule(
        eval_f_g,
        bounds=unit_bounds(),
        budget_check=lambda: True,
        n_paths=k,
        # Each iteration restarts every path, so each one buys a probe
        # (PROBE_DRAWS batches) and no step: a restarted path is skipped by that
        # iteration's step.
        max_iters=3,  # the init probe plus two restart iterations
        n_draws=PROBE_DRAWS,
        solver=solver,
        restart_rule=rule,
        key=jax.random.PRNGKey(3),
    )
    # init probe + two restart events, and nothing else
    assert res.n_points == sum(w for w, _d in shapes)
    assert res.n_points == PROBE_DRAWS * k + 2 * (PROBE_DRAWS * k)
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

        def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
            nonlocal billed
            billed += 1
            f_vals, grads = jax.vmap(pointwise)(x)
            return f_vals, grads, None

        def eval_f(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
            nonlocal drawn
            drawn += 1
            f_vals, _grads = jax.vmap(pointwise)(x)
            return f_vals, None

        res = n_path_schedule(
            eval_f_g,
            eval_f=eval_f,
            bounds=unit_bounds(),
            budget_check=lambda: billed < limit,
            n_paths=k,
            max_iters=10,  # not the binding stop: the gate closes first
            n_draws=probe_evals,
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
    eval_f_g, shapes = evaluator()
    res = n_path_schedule(
        eval_f_g,
        bounds=unit_bounds(),
        n_paths=k,
        max_iters=PROBE_DRAWS + 3,  # the init probe plus three steps
        n_draws=PROBE_DRAWS,
        solver=StubSolver(),
        restart_rule=RecordingRule(),
        key=jax.random.PRNGKey(4),
    )
    assert res.status == "max_iters", res
    assert res.n_points == sum(w for w, _d in shapes) == PROBE_DRAWS * k + 3 * k


def test_box_is_data_not_a_unit_cube_assumption() -> None:
    """A foreign box [1,2] plus the caller's own θ = x + 1 map: both are honoured.

    The engine must sample and clip inside the box it was handed, and must never
    see the caller's parameterization (that map is composed into
    `eval_f_g`). Every trial here lands outside the box - the landscape's
    minimizer x = 3 is not in it - so the clip is load-bearing, not incidental.
    """
    lo, hi = 1.0, 2.0
    served: list[float] = []

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        """Serve θ = x + 1, f = (θ - 4)²: the caller's map and landscape."""
        served.extend(float(v) for v in x[:, 0])
        theta = x + 1.0
        return jnp.sum((theta - 4.0) ** 2, axis=-1), 2.0 * (theta - 4.0), None

    bounds = jnp.stack([jnp.full(1, lo), jnp.full(1, hi)], axis=1)
    res = n_path_schedule(
        eval_f_g,
        bounds=bounds,
        budget_check=None,
        n_paths=2,
        max_iters=3,
        n_draws=PROBE_DRAWS,
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

        def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
            served.append(x)
            f_vals, grads = jax.vmap(pointwise)(x)
            return f_vals, grads, None

        res = n_path_schedule(
            eval_f_g,
            bounds=box,
            sample=sample,
            budget_check=None,
            n_paths=2,
            max_iters=3,
            n_draws=PROBE_DRAWS,
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
    `observe_continue(0, None)` per iteration, so its stage clock advances in
    lockstep with its advancing cohort instead of leading it by one verdict
    (the off-by-one a top-of-iteration stall count would cause). A continue
    sample is one step, so three iterations take three iterations.
    """
    k = 2
    calls = [0]

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        calls[0] += 1
        f_vals = jnp.full((x.shape[0],), 1.0, dtype=x.dtype)
        if calls[0] <= PROBE_DRAWS:  # init probe evals: column 0 has no evidence
            f_vals = f_vals.at[0].set(jnp.inf)
        return f_vals, jnp.zeros_like(x), None

    rule = RecordingRule()
    res = n_path_schedule(
        eval_f_g,
        bounds=unit_bounds(),
        budget_check=lambda: True,
        n_paths=k,
        max_iters=PROBE_DRAWS + 3,  # init probe + three iterations
        n_draws=PROBE_DRAWS,
        solver=StubSolver(),
        restart_rule=rule,
        key=jax.random.PRNGKey(6),
    )
    assert res.status == "max_iters", res
    counts: dict[int, int] = {}
    for i, _delta in rule.deltas:
        counts[i] = counts.get(i, 0) + 1
    assert counts == {0: 3, 1: 3}, counts
    assert [d for i, d in rule.deltas if i == 0] == [None] * 3


def test_a_probes_iteration_is_not_read_back_as_a_zero_step() -> None:
    """A path that probed folds nothing that iteration: no Δ = 0 is reported.

    The probe's f becomes the Δ anchor and the probe's iteration is skipped, so
    the first `observe_continue` for a path carries the step that follows its
    probe. Reading the probe back as `f - f = 0` would hand the rule a fake
    "nothing changed" sample on every restart and on the initialization itself.
    """
    k = 2
    rule = RecordingRule()

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        # A ramp: distinct points have distinct f values, so a reported Δ of 0
        # can only come from comparing a point with itself.
        f_vals = 10.0 * x[:, 0]
        return f_vals, jnp.full_like(x, 10.0), None

    res = n_path_schedule(
        eval_f_g,
        bounds=unit_bounds(),
        budget_check=lambda: True,
        n_paths=k,
        max_iters=PROBE_DRAWS + 1,  # init probe + one step
        n_draws=PROBE_DRAWS,
        solver=StubSolver(),
        restart_rule=rule,
        key=jax.random.PRNGKey(9),
    )
    # StubSolver steps x -> x - 0.5·g = x - 5, which the box clips to 0, so every
    # step is a strict decrease and the probe's own f never reappears as a zero.
    deltas = [d for _i, d in rule.deltas]
    assert len(deltas) == k, rule.deltas  # one step each, after its probe
    assert all(d is not None and d < 0.0 for d in deltas), deltas
    assert res.n_batches == PROBE_DRAWS + 1, res


def test_one_sample_per_iteration_at_every_probe_size() -> None:
    """Each iteration folds one sample per path, at every probe size.

    The continue side's sample is one step, so it does not depend on
    n_draws: after the init probe, M iterations of budget buy exactly M
    samples per path. This is what changed when the horizon clock went away - a
    sample used to span n_draws iterations, so the same budget bought
    fewer of them. Sizes above 1 take the f-only seam, whose single extra batch
    for the winners is all the probe's depth costs in the batch count.
    """
    k = 2
    for probe_evals in (1, 2, 3, 4):
        eval_f_g, _shapes = evaluator()
        rule = RecordingRule()
        eval_f = None if probe_evals == 1 else f_only(eval_f_g)
        res = n_path_schedule(
            eval_f_g,
            eval_f=eval_f,
            bounds=unit_bounds(),
            n_paths=k,
            max_iters=3,  # init probe + two steps
            n_draws=probe_evals,
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
        eval_f_g, _shapes = evaluator()
        res = n_path_schedule(
            eval_f_g,
            bounds=unit_bounds(),
            budget_check=lambda: True,
            n_paths=k,
            max_iters=PROBE_DRAWS + 1,  # init probe + one accepted trial
            n_draws=PROBE_DRAWS,
            solver=StubSolver(),
            restart_rule=RecordingRule(),
            key=jax.random.PRNGKey(8),
        )
        return res.n_points, res.n_batches

    # one probe of PROBE_DRAWS(=1) batch, then one accepted trial batch
    batches = PROBE_DRAWS + 1  # = 2, independent of n_paths
    points_3, batches_3 = run(3)
    points_1, batches_1 = run(1)
    assert batches_3 == batches_1 == batches
    assert points_3 == batches * 3  # 6 points
    assert points_1 == batches  # 2 points


def test_multi_draw_probe_is_refused_without_the_f_only_seam() -> None:
    """n_draws > 1 needs eval_f: it would price g once per draw.

    The probe's cost asymmetry is in gradient counts, not batch counts - a step
    computes g once while an un-seamed multi-draw probe computes it once per
    draw - so the engine refuses the combination instead of silently overselling
    the probe side.
    """
    eval_f_g, _shapes = evaluator()
    refused: str | None = None
    try:
        n_path_schedule(
            eval_f_g,
            bounds=unit_bounds(),
            n_paths=2,
            n_draws=2,
            max_iters=8,
            solver=StubSolver(),
            key=jax.random.PRNGKey(0),
        )
    except ValueError as e:
        refused = str(e)
    assert refused is not None, "n_draws=2 was accepted without eval_f"
    assert "eval_f" in refused, refused


def test_f_only_probe_computes_one_gradient_per_probe() -> None:
    """An f-only probe: n_draws f-only draws, then one fused batch.

    That single fused batch is what makes n_draws > 1 cost-matched with a
    one-gradient step: every draw costs an f, the winners then pay the one
    gradient between them. Only that fused batch is billed. The group-min still
    picks per column across the draws, so the batch re-serves the same winners
    rather than making a new choice.
    """
    k, probe_evals = 2, 3
    f_calls: list[jnp.ndarray] = []
    fused_calls: list[jnp.ndarray] = []

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        fused_calls.append(x)
        f_vals, grads = jax.vmap(pointwise)(x)
        return f_vals, grads, None

    def eval_f(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        f_calls.append(x)
        f_vals, _grads = jax.vmap(pointwise)(x)
        return f_vals, None

    res = n_path_schedule(
        eval_f_g,
        eval_f=eval_f,
        bounds=unit_bounds(),
        n_paths=k,
        n_draws=probe_evals,
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


def test_a_single_draw_probe_reads_its_start_at_the_gradient() -> None:
    """One draw plus eval_f: the start is never read on its own.

    Values are read early only to choose between candidates, and a single draw
    has no group to choose within - so the seam can leave the start unread. It
    travels to `settle` and comes back with f and g from the same pass, its
    value computed exactly once, where reading the draw first would have spent
    an f that the graded pass then repeated.
    """
    f_calls: list[int] = []
    fused_shapes: list[tuple[int, int]] = []

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        fused_shapes.append((int(x.shape[0]), int(x.shape[1])))
        f_vals, grads = jax.vmap(pointwise)(x)
        return f_vals, grads, None

    def eval_f(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        f_calls.append(int(x.shape[0]))
        f_vals, _grads = jax.vmap(pointwise)(x)
        return f_vals, None

    res = n_path_schedule(
        eval_f_g,
        eval_f=eval_f,
        bounds=unit_bounds(),
        n_paths=2,
        n_draws=1,
        max_iters=1,  # the probe alone is one iteration
        solver=StubSolver(),
        key=jax.random.PRNGKey(3),
    )
    assert f_calls == [], f_calls  # nothing to compare, so nothing read early
    assert fused_shapes == [(2, 1)], fused_shapes  # both halves, one pass
    assert res.n_batches == 1, res  # that pass is also the iteration's g batch
    assert res.n_points == 2, res


def test_a_rejected_trial_is_bought_no_gradient() -> None:
    """A retrying step pays one gradient pass in total, not one per trial.

    The ruled-on quantity is f, so each trial is served for its value alone and
    only the winner is graded. Two trial rounds therefore cost two f-only calls
    and one charged batch - the same single batch a no-line-search step costs,
    which is what makes a value-only seam cost-matched across the two sides.
    """
    k = 1
    f_calls: list[int] = []

    class OneRetry(StubSolver):
        """Retry the first trial, then accept - the shortest line search."""

        def __init__(self) -> None:
            """Start with no verdict served."""
            super().__init__()
            self.rounds = 0

        def verdict(self, i: int, p: PathPoint, x: jnp.ndarray, f: float) -> Verdict:  # noqa: ARG002
            """Reject the first trial, accept the second (no reason but the round)."""
            self.verdicts.append(i)
            self.rounds += 1
            return "retry" if self.rounds == 1 else "accept"

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        f_vals, grads = jax.vmap(pointwise)(x)
        return f_vals, grads, None

    def eval_f(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        f_calls.append(int(x.shape[0]))
        f_vals, _grads = jax.vmap(pointwise)(x)
        return f_vals, None

    solver = OneRetry()
    res = n_path_schedule(
        eval_f_g,
        eval_f=eval_f,
        bounds=unit_bounds(),
        n_paths=k,
        max_iters=2,  # the init probe, then this one step
        solver=solver,
        key=jax.random.PRNGKey(3),
    )
    # The step's two trial rounds, value-only. The probe's single draw now
    # travels to `settle` unread, so it spends no f of its own.
    assert f_calls == [k, k], f_calls
    assert solver.verdicts == [0, 0], solver.verdicts
    assert solver.folds == [0], solver.folds  # one winner, one fold
    # Two batches for the whole iteration pair: the starts', then the winner's.
    assert res.n_batches == 2, res
    assert res.n_points == 2 * k, res


def test_one_gradient_batch_per_iteration() -> None:
    """A restart's starts and a step's winner share one g batch.

    Deciding on values lets an iteration put everything it kept into a single
    gradient pass: the restarted paths' fresh starts and the stepped paths'
    winners travel together, so a mixed iteration costs one call instead of one
    per half. The shapes are what pins it - unmerged, iteration 1 would show two
    width-1 batches rather than one width-2 batch.
    """
    k = 2
    fused_shapes: list[tuple[int, int]] = []

    class RestartFirst(RecordingRule):
        """Cold start, then only path 0 keeps restarting."""

        def should_restart(self, i: int, f_k: float, best_fun_raw: float) -> bool:  # noqa: ARG002
            """Seed the pool cold, then keep restarting path 0 alone."""
            return not self.group_mins or i == 0

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        fused_shapes.append((int(x.shape[0]), int(x.shape[1])))
        f_vals, grads = jax.vmap(pointwise)(x)
        return f_vals, grads, None

    def eval_f(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        f_vals, _grads = jax.vmap(pointwise)(x)
        return f_vals, None

    res = n_path_schedule(
        eval_f_g,
        eval_f=eval_f,
        bounds=unit_bounds(),
        n_paths=k,
        max_iters=2,  # the cold start, then one mixed iteration
        solver=StubSolver(),
        restart_rule=RestartFirst(),
        key=jax.random.PRNGKey(3),
    )
    assert fused_shapes == [(2, 1), (2, 1)], fused_shapes  # k=2 both times
    assert res.n_batches == 2, res
    assert res.n_points == 2 * k, res


def test_a_winner_with_no_usable_gradient_is_not_committed() -> None:
    """The step is decided on f, so its winner's gradient is gated on arrival.

    Nothing gates a value-only trial (there is no gradient to gate), and the
    method is told nothing about gradients at all, so the engine's commit is the
    only place that can refuse one. A winner whose gradient is unusable does not
    become the next start - it stalls the path exactly like a dead probe start -
    and the fold never happens, so the method is not told a step landed.
    """
    fused = [0]

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        """Finite f always; the gradient is usable only for the probe's start."""
        fused[0] += 1
        f_vals, grads = jax.vmap(pointwise)(x)
        if fused[0] > 1:  # the second batch is the step winner's
            grads = jnp.full_like(grads, jnp.nan)
        return f_vals, grads, None

    def eval_f(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        f_vals, _grads = jax.vmap(pointwise)(x)
        return f_vals, None

    solver = StubSolver()
    rule = RecordingRule()
    res = n_path_schedule(
        eval_f_g,
        eval_f=eval_f,
        bounds=unit_bounds(),
        n_paths=1,
        max_iters=2,  # the init probe, then this one step
        solver=solver,
        restart_rule=rule,
        key=jax.random.PRNGKey(3),
    )
    assert solver.verdicts == [0], solver.verdicts  # the trial was judged
    assert solver.folds == [], solver.folds  # but never committed
    assert rule.deltas == [(0, None)], rule.deltas  # the stall, as absence
    assert res.status == "max_iters", res


def test_a_non_finite_gradient_start_arrives_stalled() -> None:
    """A start is committed only when f *and* g are finite.

    A finite f with a non-finite gradient is not a usable start: `p.g` is what
    the method proposes from and folds on, so one NaN in it poisons that
    coordinate's magnitude EMAs for the rest of the run. Such a start arrives
    stalled exactly like a dead draw and waits for an EI verdict instead of
    stepping.
    """

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        f_vals, _grads = jax.vmap(pointwise)(x)
        return f_vals, jnp.full_like(x, jnp.nan), None

    solver = StubSolver()
    res = n_path_schedule(
        eval_f_g,
        bounds=unit_bounds(),
        n_paths=2,
        n_draws=1,
        max_iters=2,
        solver=solver,
        key=jax.random.PRNGKey(4),
    )
    # No path was ever live, so nothing stepped and nothing was judged.
    assert solver.verdicts == [], solver.verdicts
    assert solver.folds == [], solver.folds
    assert res.status == "max_iters", res


def test_an_unjudged_step_never_buys_an_f_only_ruling() -> None:
    """A step that needs no verdict is served once, fused - as a probe start is.

    The engine leaves both halves of an unjudged proposal owed, so `settle` buys
    the value and its gradient together. A judged step pays for a value of its
    own: the ruling is f-only, and the winner's gradient is bought again, fused,
    in `settle`. Both lanes therefore bill the same fused batches; what the
    unjudged one never pays is the f-only pass, which is the whole reason to
    answer False.
    """
    k = 2

    def run(solver: StubSolver) -> tuple[int, int, int]:
        eval_f_g, _shapes = evaluator()
        f_only_widths: list[int] = []

        def eval_f(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
            f_only_widths.append(int(x.shape[0]))
            return jnp.sum((x - 0.25) ** 2, axis=-1), None

        res = n_path_schedule(
            eval_f_g,
            eval_f=eval_f,
            bounds=unit_bounds(),
            n_paths=k,
            max_iters=2,  # init probe + one step
            solver=solver,
            restart_rule=RecordingRule(),
            key=jax.random.PRNGKey(7),
        )
        return sum(f_only_widths), res.n_batches, res.n_points

    # The single-draw probe needs no ruling either, so iteration 0 is silent.
    assert run(StubSolver()) == (k, 2, 2 * k)
    assert run(UnjudgedSolver()) == (0, 2, 2 * k)


def test_an_unjudged_winner_must_be_finite_to_be_committed() -> None:
    """Unruled means the engine's own gate is the whole of the acceptance test.

    A judged winner was ruled finite before it was kept; an unjudged one is first
    seen in the batch that buys its gradient, so a non-finite half there must
    stall the path rather than fold a nan into the rule's next Δ and the pool.
    """

    def make_eval(
        bad: str | None,
    ) -> Callable[..., tuple[jnp.ndarray, jnp.ndarray, None]]:
        """Return an objective that turns `bad` non-finite past the init probe."""
        served: list[int] = []

        def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
            served.append(int(x.shape[0]))
            later = len(served) > 1  # the init probe's settle is the first batch
            n = int(x.shape[0])
            f = jnp.full(n, math.nan) if bad == "f" and later else jnp.zeros(n)
            g = jnp.zeros_like(x)
            if bad == "g" and later:
                g = jnp.full_like(x, math.nan)
            return f, g, None

        return eval_f_g

    def run(bad: str | None) -> StubSolver:
        solver = PushTo(0.5)
        n_path_schedule(
            make_eval(bad),
            bounds=unit_bounds(),
            n_paths=1,
            max_iters=3,
            solver=solver,
            restart_rule=RecordingRule(),
            key=jax.random.PRNGKey(3),
        )
        return solver

    # Control: a finite value and gradient commit, so the gate is what stalls below.
    assert run(None).folds == [0, 0]
    assert run("f").folds == []
    assert run("g").folds == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"all {len(fns)} passed")
