"""Plugin-seam self-checks.

The seams' contract is only proven by injection: a stub `LocalSolver`, a stub
`RestartRule` and a stub `GlobalModel` are plugged into the engine, which must
drive them instead of its defaults, and a foreign box (`bounds` as data) plus the
caller's own x->θ map composed into `eval_f_g` must be honoured too. These also
pin the documented zero-caller hook — a method returning `"give_up"` — and the
run with no extra gate (`budget_check=None` -> `status == "max_iters"`).

Run with:

    uv run python -m tests.test_plugin_seam
"""

import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from soar_lite.engine import (
    Bounds,
    NPathResult,
    PathPoint,
    Verdict,
    n_path_schedule,
)
from soar_lite.global_model import UniformGlobalModel

jax.config.update("jax_enable_x64", val=True)


def unit_bounds(d: int = 1) -> jnp.ndarray:
    """Return the engine's box in working coords: x ∈ [0,1]^d."""
    return jnp.stack([jnp.zeros(d), jnp.ones(d)], axis=1)


def flock(n_paths: int, d: int = 1) -> jnp.ndarray:
    """Return n_paths starts spread over the box's interior, one row each."""
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

    def needs_verdict(self, i: int, p: PathPoint) -> bool:  # noqa: ARG002
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

    def needs_verdict(self, i: int, p: PathPoint) -> bool:  # noqa: ARG002
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
    """Records the engine's rule-side calls; never fires.

    Both channels are recorded, so the tests can prove the engine asked once per
    path per iteration and reported once per path per iteration - and that a
    restarted path is reported to no one.
    """

    def __init__(self) -> None:
        """Start with no evidence and no verdicts."""
        self.deltas: list[tuple[int, float | None]] = []
        self.verdicts: list[int] = []

    def observe_step(self, i: int, delta: float | None) -> None:
        """Record one f change (or a stall tick, `None`) for path i."""
        self.deltas.append((i, delta))

    def should_restart(self, i: int, f_k: float, best_fun_raw: float) -> bool:  # noqa: ARG002
        """Record the verdict call for path i; never restart."""
        self.verdicts.append(i)
        return False


class AlwaysRestartRule(RecordingRule):
    """Same recorder, but every verdict says restart."""

    def should_restart(self, i: int, f_k: float, best_fun_raw: float) -> bool:  # noqa: ARG002
        """Record the verdict call for path i and always restart."""
        self.verdicts.append(i)
        return True


class RecordingModel:
    """Records what the engine asks and feeds it, and draws inside the box.

    A real `GlobalModel`, not a stub: it holds its own key (the engine draws for
    no model), so its answers are reproducible from the seed alone, and it
    records both channels - one ask per restarting path, one observe per served
    batch.
    """

    def __init__(self, box: jnp.ndarray, *, key: jnp.ndarray | None = None) -> None:
        """Store the box every draw lands in and the stream it is drawn from."""
        self.box = box
        self.d = int(box.shape[0])
        self.calls: list[int] = []  # one entry per ask, holding n_points
        self.observed: list[int] = []  # one entry per served batch, its width
        self._key = jax.random.PRNGKey(0) if key is None else key

    def observe(self, x: jnp.ndarray, f: list[float], g: jnp.ndarray | None) -> None:  # noqa: ARG002
        """Record the batch's width; this model fits nothing."""
        self.observed.append(int(x.shape[0]))

    def propose(self, n_points: int) -> jnp.ndarray:
        """Draw n_points uniform points over the stored box."""
        self.calls.append(n_points)
        self._key, sub = jax.random.split(self._key)
        return jax.random.uniform(
            sub,
            (n_points, self.d),
            minval=self.box[:, 0],
            maxval=self.box[:, 1],
        )


def test_injected_solver_is_driven() -> None:
    """The engine calls the injected method, never its default."""
    k = 3
    eval_f_g, shapes = evaluator()
    solver = StubSolver()
    res = n_path_schedule(
        eval_f_g,
        **flock_kwargs(flock(k), unit_bounds(), key=jax.random.PRNGKey(0)),
        bounds=unit_bounds(),
        budget_check=lambda: True,
        max_iters=2,  # iteration 0 places the flock, then one step
        local_solver=solver,
        restart_rule=RecordingRule(),
    )
    assert solver.inited == [0, 1, 2], solver.inited
    assert solver.verdicts == [0, 1, 2], solver.verdicts  # one trial, then accept
    # The flock's one fused batch, then the accepted winners' one: two batches,
    # k points each.
    assert res.n_points == sum(w for w, _d in shapes) == 2 * k
    assert res.status == "max_iters"


def test_give_up_stalls_a_path() -> None:
    """A "give_up" verdict stalls that path at once: no further trial for it."""
    k = 3
    eval_f_g, shapes = evaluator()
    solver = StubSolver(give_up=True)
    res = n_path_schedule(
        eval_f_g,
        **flock_kwargs(flock(k), unit_bounds(), key=jax.random.PRNGKey(1)),
        bounds=unit_bounds(),
        budget_check=lambda: True,
        max_iters=2,  # the flock, then the single stalled iteration
        local_solver=solver,
        restart_rule=RecordingRule(),
    )
    assert solver.verdicts == [0, 1, 2]  # one verdict each, then stalled
    # Nothing to commit: every batch this run serves is a trial, and a trial
    # without the f-only seam is fused, so the count is one flock plus one round.
    assert res.n_points == sum(w for w, _d in shapes) == 2 * k
    assert res.status == "max_iters"


def test_an_unusable_start_is_refused_before_it_seeds_the_method() -> None:
    """The place gate, seen from the method: a start it cannot use is not seed.

    The start batch serves a finite value with a NaN gradient, so the start is
    unusable: no cold-start seed reaches the method (`restart` is never called)
    and the path never steps. Seeding it would let the method propose from
    `-p.g` on a point no path stands at. The path also holds no point, so the
    engine owes it a fresh start - that re-draw is the run's second and last
    batch, and it is refused the same way.
    """

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        d = x - 0.5
        f = jnp.sum(d**2, axis=1)
        g = (2.0 * d).at[:, 0].set(jnp.nan)  # the value is fine, the gradient is not
        return f, g, None

    solver = StubSolver()
    res = n_path_schedule(
        eval_f_g,
        **flock_kwargs(flock(2), unit_bounds(), key=jax.random.PRNGKey(3)),
        bounds=unit_bounds(),
        budget_check=lambda: True,
        max_iters=2,  # the flock, then the iteration that finds nothing live
        local_solver=solver,
        restart_rule=RecordingRule(),
    )
    assert solver.inited == [], solver.inited  # an unusable start is never seed
    assert solver.folds == [], solver.folds  # and never becomes a start
    assert res.n_batches == 2, res.n_batches  # the flock and one re-draw
    assert res.status == "max_iters"


def test_injected_rule_is_driven() -> None:
    """The engine feeds its one injected rule instance both channels, once per path."""
    k = 2
    eval_f_g, _shapes = evaluator()
    rule = RecordingRule()
    res = n_path_schedule(
        eval_f_g,
        **flock_kwargs(flock(k), unit_bounds(), key=jax.random.PRNGKey(2)),
        bounds=unit_bounds(),
        budget_check=lambda: True,
        max_iters=2,  # the flock, then one step
        local_solver=StubSolver(),
        restart_rule=rule,
    )
    # Iteration 0 places the flock and asks nothing: there is no step to judge
    # and no Δ to report yet.
    assert rule.verdicts == [0, 1], rule.verdicts
    # One report per path on the one step, and it is a real decrease.
    assert [i for i, _delta in rule.deltas] == [0, 1], rule.deltas
    assert all(delta is not None and delta < 0.0 for _i, delta in rule.deltas)
    assert res.status == "max_iters"


def test_rule_can_force_restarts() -> None:
    """A rule that always says restart re-draws every iteration, never steps."""
    k = 2
    eval_f_g, shapes = evaluator()
    solver = StubSolver()
    rule = AlwaysRestartRule()
    model = RecordingModel(unit_bounds(), key=jax.random.PRNGKey(3))
    res = n_path_schedule(
        eval_f_g,
        n_paths=k,
        bounds=unit_bounds(),
        global_model=model,
        budget_check=lambda: True,
        # Each iteration restarts every path, so each one buys one batch of
        # starts and no step: a restarted path is skipped by that iteration.
        # The flock is no exception - iteration 0 is itself a restart.
        max_iters=3,  # the placement plus two restart iterations
        local_solver=solver,
        restart_rule=rule,
    )
    # One batch per iteration, the placement included - and nothing else.
    assert res.n_points == sum(w for w, _d in shapes) == 3 * k
    assert model.calls == [k, k, k], model.calls  # every iteration asks
    assert rule.verdicts == [0, 1] * 2, rule.verdicts  # one round per iteration
    assert rule.deltas == [], rule.deltas  # a restarted path reports nothing
    assert solver.verdicts == []  # restarted, not stepped
    # The flock seeds both paths once, then every restart seeds its path again.
    assert solver.inited == [0, 1] * 3, solver.inited
    assert res.status == "max_iters"


def test_the_gate_counts_g_batches_not_f_only_draws() -> None:
    """The gate bills the fused pass; a step's value-only trial is not billed.

    What a caller spends real time on is the value *and* gradient pass, so that
    is what `budget_check` counts. A judged step first serves its proposal
    value-only (the ruling is on f) and then buys the winner's gradient, fused
    with the value it already knows - so a step is one billed batch and one
    donated one, and the run never serves a fused batch past the gate.
    """
    k = 2

    def run(limit: int) -> tuple[int, int, int, str]:
        """Serve under one gate limit; return (billed, donated, n_points, status)."""
        billed = 0  # fused batches: the only billed calls
        donated = 0  # value-only trials: served, never billed

        def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
            nonlocal billed
            billed += 1
            f_vals, grads = jax.vmap(pointwise)(x)
            return f_vals, grads, None

        def eval_f(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
            nonlocal donated
            donated += 1
            f_vals, _grads = jax.vmap(pointwise)(x)
            return f_vals, None

        res = n_path_schedule(
            eval_f_g,
            eval_f=eval_f,
            **flock_kwargs(flock(k), unit_bounds(), key=jax.random.PRNGKey(7)),
            bounds=unit_bounds(),
            budget_check=lambda: billed < limit,
            max_iters=20,  # not the binding stop: the gate closes first
            local_solver=StubSolver(),
            restart_rule=RecordingRule(),
        )
        return billed, donated, res.n_points, res.status

    for limit in (1, 2, 4, 7):
        billed, donated, n_points, status = run(limit)
        # The gate closes on the billed count alone: never one fused batch past it.
        assert billed == limit, (limit, billed)
        # n_points is the charged half: an f-only trial serves work that is
        # never counted (and never billed).
        assert n_points == k * billed, (limit, n_points, donated)
        assert status == "budget_exhausted", (limit, status)
        # The value-only trials did happen, and stayed out of the billed count.
        # One donated trial per iteration after the flock's: the run's first
        # batch is the flock, so a gate of `limit` buys `limit - 1` trials.
        assert donated == limit - 1, (limit, donated)


def test_a_run_without_an_extra_gate_reports_max_iters() -> None:
    """No gate passed at all (the keyword defaults to None): the horizon only."""
    k = 2
    eval_f_g, shapes = evaluator()
    res = n_path_schedule(
        eval_f_g,
        **flock_kwargs(flock(k), unit_bounds(), key=jax.random.PRNGKey(4)),
        bounds=unit_bounds(),
        max_iters=4,  # the flock plus three steps
        local_solver=StubSolver(),
        restart_rule=RecordingRule(),
    )
    assert res.status == "max_iters", res
    # The flock's batch, then one fused batch per step (no f-only seam is
    # handed in, so a trial arrives already graded).
    assert res.n_points == sum(w for w, _d in shapes) == k + 3 * k


def test_box_is_data_not_a_unit_cube_assumption() -> None:
    """A foreign box [1,2] plus the caller's own θ = x + 1 map: both are honoured.

    The engine must clip inside the box it was handed, and must never see the
    caller's parameterization (that map is composed into `eval_f_g`). Every trial
    here lands outside the box - the landscape's minimizer x = 3 is not in it - so
    the clip is load-bearing, not incidental.
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
        **flock_kwargs(jnp.array([[1.0], [1.5]]), bounds, key=jax.random.PRNGKey(5)),
        bounds=bounds,
        budget_check=None,
        max_iters=3,
        local_solver=StubSolver(),  # step = x - 0.5·g = 3 exactly: outside the box
        restart_rule=RecordingRule(),
    )
    assert served, "nothing was ever served"
    assert min(served) >= lo, min(served)
    assert max(served) <= hi, max(served)
    # the clip really fired (the raw trial was 3.0) and the best point sits on it
    assert abs(max(served) - hi) < 1e-12, max(served)
    assert abs(float(res.x[0]) - hi) < 1e-12, res.x
    assert res.status == "max_iters", res


def test_the_default_model_is_exactly_a_uniform_global_model() -> None:
    """Pin that `global_model=None` is `UniformGlobalModel`, not merely uniform.

    The default is a shipped object a caller can hand over itself, so the two
    spellings have to be the same run to the bit: a finite-box number recorded
    with the default then stays comparable with one recorded after a model was
    handed over. Every verdict says restart, so both sites - the flock's
    placement and the re-draws - are exercised.
    """
    box = unit_bounds(2)

    def run(
        global_model: object | None,
    ) -> tuple[list[jnp.ndarray], jnp.ndarray, float, int]:
        served: list[jnp.ndarray] = []

        def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
            served.append(x)
            f_vals, grads = jax.vmap(pointwise)(x)
            return f_vals, grads, None

        res = n_path_schedule(
            eval_f_g,
            n_paths=2,
            bounds=box,
            global_model=global_model,  # pyright: ignore[reportArgumentType]
            budget_check=None,
            max_iters=3,
            local_solver=StubSolver(),
            restart_rule=AlwaysRestartRule(),
            key=jax.random.PRNGKey(7),
        )
        return served, res.x, res.fun, res.n_points

    default_served, default_x, default_fun, default_points = run(None)
    ours_served, ours_x, ours_fun, ours_points = run(
        UniformGlobalModel(box, key=jax.random.PRNGKey(7)),
    )
    assert default_served, "nothing was ever served"
    assert len(default_served) == len(ours_served)
    for a, b in zip(default_served, ours_served, strict=True):
        assert jnp.array_equal(a, b), (a, b)
    assert jnp.array_equal(default_x, ours_x), (default_x, ours_x)
    assert default_fun == ours_fun, (default_fun, ours_fun)
    assert default_points == ours_points, (default_points, ours_points)


def test_every_path_feeds_once_per_iteration() -> None:
    """Pin that a path feeds observe_step once per iteration it holds a point.

    Column 0 is non-finite in iteration 0, so path 0 arrives with no point at
    all - and a path with no point is restarted without being asked, so it is
    fed nothing that iteration. Its placement batch answers with a finite value,
    so from the next iteration it is asked exactly like its cohort: one call per
    iteration, and its stage clock starts one iteration late rather than being
    handed a stall it never had.
    """
    k = 2
    calls = [0]

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        calls[0] += 1
        f_vals = jnp.full((x.shape[0],), 1.0, dtype=x.dtype)
        if calls[0] == 1:  # the flock's batch: column 0 has no evidence
            f_vals = f_vals.at[0].set(jnp.inf)
        return f_vals, jnp.zeros_like(x), None

    rule = RecordingRule()
    res = n_path_schedule(
        eval_f_g,
        **flock_kwargs(flock(k), unit_bounds(), key=jax.random.PRNGKey(6)),
        bounds=unit_bounds(),
        budget_check=lambda: True,
        max_iters=4,  # the flock plus three iterations
        local_solver=StubSolver(),
        restart_rule=rule,
    )
    assert res.status == "max_iters", res
    counts: dict[int, int] = {}
    for i, _delta in rule.deltas:
        counts[i] = counts.get(i, 0) + 1
    # path 1 held its flock row and was asked every iteration; path 0 was
    # placed in iteration 1 and asked from there.
    assert counts == {0: 2, 1: 3}, counts
    assert [d for i, d in rule.deltas if i == 0] == [0.0, 0.0]


def test_a_flocks_start_is_not_read_back_as_a_zero_step() -> None:
    """A path's first Δ carries its first step: the flock is the anchor, not a Δ.

    The flock's f becomes the Δ anchor and iteration 0 reports nothing (the
    engine has no step to report yet), so the first Δ for a path is the step that
    follows its start. Reading the start back as `f - f = 0` would hand the rule a
    fake "nothing changed" sample on every run and on every restart.
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
        **flock_kwargs(flock(k), unit_bounds(), key=jax.random.PRNGKey(9)),
        bounds=unit_bounds(),
        budget_check=lambda: True,
        max_iters=2,  # the flock plus one step
        local_solver=StubSolver(),
        restart_rule=rule,
    )
    # StubSolver steps x -> x - 0.5·g = x - 5, which the box clips to 0, so every
    # step is a strict decrease and the start's own f never reappears as a zero.
    for path in range(k):
        seq = [d for i, d in rule.deltas if i == path]
        assert len(seq) == 1, (path, rule.deltas)  # the step, and nothing else
        assert seq[0] is not None, (path, seq)  # not read back as a no-step round
        assert seq[0] < 0.0, (path, seq)  # ... and it is a real decrease
    assert res.n_batches == 2, res


def test_reports_points_and_batches_separately() -> None:
    """n_points and n_batches are two units, and both are always reported.

    A batch is the points one eval serves, so a run of k paths that serves the
    same batches reports k times the points of a k=1 run, and the same batch
    count. There is no billing mode to switch: the serial cost is `n_points`, the
    parallel wall-clock cost is `n_batches`.
    """

    def run(k: int) -> tuple[int, int]:
        eval_f_g, _shapes = evaluator()
        res = n_path_schedule(
            eval_f_g,
            **flock_kwargs(flock(k), unit_bounds(), key=jax.random.PRNGKey(8)),
            bounds=unit_bounds(),
            budget_check=lambda: True,
            max_iters=2,  # the flock plus one accepted trial
            local_solver=StubSolver(),
            restart_rule=RecordingRule(),
        )
        return res.n_points, res.n_batches

    batches = 2  # the flock, then the winners; independent of n_paths
    points_3, batches_3 = run(3)
    points_1, batches_1 = run(1)
    assert batches_3 == batches_1 == batches
    assert points_3 == batches * 3  # 6 points
    assert points_1 == batches  # 2 points


def test_a_flocks_start_is_served_once_fused() -> None:
    """A start is never read on its own, even when eval_f is handed in.

    Values are read early only to choose between candidates, and the flock has
    no group to choose within - so the seam leaves the starts unread. They travel
    to `settle` and come back with f and g from the same pass, each value
    computed exactly once, where reading them first would have spent an f that
    the graded pass then repeated.
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
        **flock_kwargs(flock(2), unit_bounds(), key=jax.random.PRNGKey(3)),
        bounds=unit_bounds(),
        max_iters=1,  # the flock alone is iteration 0
        local_solver=StubSolver(),
    )
    assert f_calls == [], f_calls  # nothing to compare, so nothing read early
    assert fused_shapes == [(2, 1)], fused_shapes  # both halves, one pass
    assert res.n_batches == 1, res  # that pass is also the flock's batch
    assert res.n_points == 2, res


def test_a_rejected_trial_is_bought_no_gradient() -> None:
    """A retrying step pays one gradient pass in total, not one per trial.

    The ruled-on quantity is f, so each trial is served for its value alone and
    only the winner is graded. Two trial rounds therefore cost two f-only calls
    and one fused batch - the same single fused batch a no-line-search step
    costs, which is what makes a value-only seam cost-matched across the sides.
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
        **flock_kwargs(flock(k), unit_bounds(), key=jax.random.PRNGKey(3)),
        bounds=unit_bounds(),
        max_iters=2,  # the flock, then this one step
        local_solver=solver,
    )
    # The step's two trial rounds, value-only. The flock travels to `settle`
    # unread, so it spends no f of its own.
    assert f_calls == [k, k], f_calls
    assert solver.verdicts == [0, 0], solver.verdicts
    assert solver.folds == [0], solver.folds  # one winner, one fold
    # Two batches for the whole run: the flock's, then the winner's.
    assert res.n_batches == 2, res
    assert res.n_points == 2 * k, res


def test_one_gradient_batch_per_iteration() -> None:
    """A restart's starts and a step's winner share one fused batch.

    Deciding on values lets an iteration put everything it kept into a single
    gradient pass: the restarted paths' fresh starts and the stepped paths'
    winners travel together, so a mixed iteration costs one call instead of one
    per half. The shapes are what pins it - unmerged, iteration 1 would show two
    width-1 batches rather than one width-2 batch.
    """
    k = 2
    fused_shapes: list[tuple[int, int]] = []

    class RestartFirst(RecordingRule):
        """Restart path 0 every iteration; path 1 keeps stepping."""

        def should_restart(self, i: int, f_k: float, best_fun_raw: float) -> bool:  # noqa: ARG002
            """Record the verdict call and restart path 0 alone."""
            self.verdicts.append(i)
            return i == 0

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
        n_paths=k,
        bounds=unit_bounds(),
        global_model=RecordingModel(unit_bounds(), key=jax.random.PRNGKey(3)),
        max_iters=2,  # the flock, then one mixed iteration
        local_solver=StubSolver(),
        restart_rule=RestartFirst(),
    )
    assert fused_shapes == [(2, 1), (2, 1)], fused_shapes  # k=2 both times
    assert res.n_batches == 2, res
    assert res.n_points == 2 * k, res


def test_the_model_is_fed_at_both_bottoms_of_a_batch() -> None:
    """Pin where `observe` sits: inside the serve, after the call, before rulings.

    Both evaluation bottoms feed the model. A judged step first serves its
    proposals value-only - the ruling is on f - so the model hears about that
    batch with `g=None`; the winner's gradient arrives in the next fused batch
    and is observed again, graded. The duplicate is the model's business, not
    the engine's. A path that gives up on its trial commits nothing, and the
    batch is observed all the same: it was served, so a model that fits what it
    was shown is never missing a point the run paid for.
    """

    class Loud(RecordingModel):
        """A recording model that also logs the order of its own feeds."""

        def __init__(self, box: jnp.ndarray) -> None:
            """Start the recorder with an empty feed log."""
            super().__init__(box)
            self.log: list[str] = []

        def observe(
            self,
            x: jnp.ndarray,
            f: list[float],
            g: jnp.ndarray | None,
        ) -> None:
            """Log whether this batch came graded, then record it."""
            self.log.append("observe:f" if g is None else "observe:g")
            super().observe(x, f, g)

    def run(*, give_up: bool) -> tuple[Loud, NPathResult]:
        model = Loud(unit_bounds())

        def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
            model.log.append("eval:f_g")
            f_vals, grads = jax.vmap(pointwise)(x)
            return f_vals, grads, None

        def eval_f(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
            model.log.append("eval:f")
            f_vals, _grads = jax.vmap(pointwise)(x)
            return f_vals, None

        res = n_path_schedule(
            eval_f_g,
            eval_f=eval_f,
            n_paths=2,
            bounds=unit_bounds(),
            global_model=model,
            budget_check=None,
            max_iters=2,  # the flock, then one judged step
            local_solver=StubSolver(give_up=give_up),
            restart_rule=RecordingRule(),
        )
        return model, res

    stepped, _stepped_res = run(give_up=False)
    assert stepped.log == [
        "eval:f_g",
        "observe:g",  # iteration 0: the flock's placement
        "eval:f",
        "observe:f",  # iteration 1: the trials, value-only
        "eval:f_g",
        "observe:g",  # ...then the winners' single fused batch
    ], stepped.log

    gave_up, gave_up_res = run(give_up=True)
    assert gave_up.log == [
        "eval:f_g",
        "observe:g",  # the flock's placement
        "eval:f",
        "observe:f",  # the trial nobody commits: still observed
    ], gave_up.log
    assert gave_up_res.status == "max_iters", gave_up_res


def test_a_winner_with_no_usable_gradient_is_not_committed() -> None:
    """The step is decided on f, so its winner's gradient is gated on arrival.

    Nothing gates a value-only trial (there is no gradient to gate), and the
    method is told nothing about gradients at all, so the engine's commit is the
    only place that can refuse one. A winner whose gradient is unusable does not
    become the next start - it stalls the path exactly like a dead flock row -
    and the fold never happens, so the method is not told a step landed.
    """
    fused = [0]

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        """Finite f always; the gradient is usable only for the flock's start."""
        fused[0] += 1
        f_vals, grads = jax.vmap(pointwise)(x)
        if fused[0] > 1:  # the second fused batch is the step winner's
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
        **flock_kwargs(flock(1), unit_bounds(), key=jax.random.PRNGKey(3)),
        bounds=unit_bounds(),
        max_iters=2,  # the flock, then this one step
        local_solver=solver,
        restart_rule=rule,
    )
    assert solver.verdicts == [0], solver.verdicts  # the trial was judged
    assert solver.folds == [], solver.folds  # but never committed
    assert rule.deltas == [(0, None)], rule.deltas  # the one round was a stall
    assert res.status == "max_iters", res


def test_a_non_finite_gradient_start_arrives_stalled() -> None:
    """A start is committed only when f *and* g are finite.

    A finite f with a non-finite gradient is not a usable start: `p.g` is what
    the method proposes from and folds on, so one NaN in it poisons that
    coordinate's magnitude EMAs for the rest of the run. Such a start arrives
    stalled and waits for its rule to fire instead of stepping.
    """

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        f_vals, _grads = jax.vmap(pointwise)(x)
        return f_vals, jnp.full_like(x, jnp.nan), None

    solver = StubSolver()
    res = n_path_schedule(
        eval_f_g,
        **flock_kwargs(flock(2), unit_bounds(), key=jax.random.PRNGKey(4)),
        bounds=unit_bounds(),
        max_iters=2,
        local_solver=solver,
    )
    # No path was ever live, so nothing stepped and nothing was judged.
    assert solver.verdicts == [], solver.verdicts
    assert solver.folds == [], solver.folds
    assert res.status == "max_iters", res


def test_an_unjudged_step_never_buys_an_f_only_ruling() -> None:
    """A step that needs no verdict is served once, fused - as a flock row is.

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
            **flock_kwargs(flock(k), unit_bounds(), key=jax.random.PRNGKey(7)),
            bounds=unit_bounds(),
            max_iters=2,  # the flock plus one step
            local_solver=solver,
            restart_rule=RecordingRule(),
        )
        return sum(f_only_widths), res.n_batches, res.n_points

    # The flock needs no ruling either, so iteration 0 is silent.
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
        """Return an objective that turns `bad` non-finite past the flock."""
        served: list[int] = []

        def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
            served.append(int(x.shape[0]))
            later = len(served) > 1  # the flock's settle is the first batch
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
            **flock_kwargs(flock(1), unit_bounds(), key=jax.random.PRNGKey(3)),
            bounds=unit_bounds(),
            max_iters=3,
            local_solver=solver,
            restart_rule=RecordingRule(),
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
