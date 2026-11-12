"""Plugin-seam self-checks.

The seams' contract is only proven by injection: a stub `LocalSolver` and a stub
`GlobalModel` are plugged into the engine, which must drive them instead of its
defaults, and a foreign box (a `Domain`) plus the caller's own x->θ map
composed into `eval_f_g` must be honored too. These also pin the documented
zero-caller hook — a method naming no winner — and the run with no extra gate
(`budget_check=None` -> `status == "max_iters"`).

The two answers a method can give are the whole of its contract here: a point
winner, which `settle` grades in one batch, or `None`, which stops that path
where it stands. Whether it read its trial first is the method's business, and
this suite pins both lanes against the same counters - a read trial spends `f`
today and its gradient tomorrow, an unread one spends both at `settle`.

Run with:

    uv run python -m tests.test_plugin_seam
"""

import math
from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp

from soar_lite.domain import Domain
from soar_lite.engine import (
    NPathResult,
    PathPoint,
    Point,
    ServeTrials,
    ValuePoint,
    n_path_schedule,
)
from soar_lite.global_model import SampleGlobalModel
from tests.flock import flock, flock_kwargs, unit_domain

jax.config.update("jax_enable_x64", val=True)


def pointwise(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """F = (θ - 0.25)^2, grad = 2(θ - 0.25)."""
    return jnp.sum((theta - 0.25) ** 2, axis=-1), 2.0 * (theta - 0.25)


def recording_eval_f_g() -> tuple[
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
    """Reads each step's one trial and accepts it, or names no winner.

    The engine hands a method a seam and takes back a winner or `None`; what
    counts as a winner is the method's own ruling, so this stub spells the ruling
    out here and records every call the engine makes, so a test can prove the
    engine drove the injected method rather than its default
    `DiagonalPreconditioner`.

    The cohort is served as one batch and answered in one call: a method that
    looped per path would serialize the engine's only seam, which is the thing
    the batch counters below measure.
    """

    def __init__(self, *, give_up: bool = False) -> None:
        """Store the policy; give_up reads its trial and then names no winner."""
        self.give_up = give_up
        self.inited: list[int] = []
        self.trials: list[int] = []
        self.steps: list[int] = []
        self.points: dict[int, PathPoint] = {}

    def restart(self, points: Mapping[int, PathPoint]) -> None:
        """Record each path's cold start (the engine's only seed call)."""
        self.points.update(points)
        self.inited.extend(points)

    def step(self, points: Mapping[int, PathPoint]) -> None:
        """Record the committed steps and stand on them; nothing else is kept."""
        self.points.update(points)
        self.steps.extend(points)

    def _trial(self, paths: Sequence[int], domain: Domain) -> jnp.ndarray:
        """Return the cohort's halved steepest-descent steps, clipped."""
        held = [self.points[i] for i in paths]
        return jnp.clip(
            jnp.stack([p.x for p in held]) - 0.5 * jnp.stack([p.g for p in held]),
            domain.lo,
            domain.hi,
        )

    def propose(
        self, paths: Sequence[int], domain: Domain, serve_trials: ServeTrials
    ) -> dict[int, Point | None]:
        """Serve one trial per path, then name the winners (or none of them)."""
        self.trials.extend(paths)
        served = serve_trials(dict(zip(paths, self._trial(paths, domain), strict=True)))
        return {i: (None if self.give_up else served[i]) for i in paths}


class UnjudgedSolver(StubSolver):
    """A method whose step names its proposal unread: it serves no trial."""

    def propose(
        self, paths: Sequence[int], domain: Domain, serve_trials: ServeTrials
    ) -> dict[int, Point | None]:
        """Hand the steps back owed, so `settle` grades them in one batch."""
        del serve_trials
        self.trials.extend(paths)
        batch = self._trial(paths, domain)
        return {i: Point(batch[k]) for k, i in enumerate(paths)}


class RetryingSolver(StubSolver):
    """Reject a fixed number of trials per step before accepting one.

    A line search discovers trial k+1 only after ruling trial k, so its trials go
    out one at a time, one batch each - which is what puts the gate's per-serve
    ask under test. The engine has no trial cap of its own: how long a step
    enumerates is the method's business, so a method that retries stretches one
    iteration over many value-only batches.
    """

    def __init__(self, retries: int) -> None:
        """Store how many trials each path rejects before it accepts."""
        super().__init__()
        self.retries = retries

    def propose(
        self, paths: Sequence[int], domain: Domain, serve_trials: ServeTrials
    ) -> dict[int, Point | None]:
        """Serve trials one at a time and name the last one, read."""
        out: dict[int, Point | None] = {}
        for i in paths:
            row: ValuePoint | None = None
            for _attempt in range(self.retries + 1):
                self.trials.append(i)
                row = serve_trials({i: self._trial([i], domain)[0]})[i]
            out[i] = row
        return out


class PushTo(UnjudgedSolver):
    """Name one fixed point, unread, to place a chosen value under the gate."""

    def __init__(self, x_star: float) -> None:
        """Store the point every path proposes."""
        super().__init__()
        self.x_star = x_star

    def _trial(self, paths: Sequence[int], domain: Domain) -> jnp.ndarray:
        """Return the stored point, whatever the start."""
        del domain
        return jnp.full((len(paths), self.points[paths[0]].x.size), self.x_star)


class RecordingModel:
    """Records what the engine asks and feeds it, and draws inside the box.

    A real `GlobalModel`, not a stub: it holds its own key (the engine draws for
    no model), so its answers are reproducible from the seed alone, and it
    records both channels - one ask per restarting path, one observe per served
    batch. It carries no box: each ask hands it the run's `domain`.
    """

    def __init__(self, *, key: jnp.ndarray | None = None) -> None:
        """Store the stream every draw comes from."""
        self.calls: list[int] = []  # one entry per ask, holding len(paths)
        self.observed: list[int] = []  # one entry per served batch, its width
        self._key = jax.random.PRNGKey(0) if key is None else key

    def observe(self, served: Mapping[int, ValuePoint]) -> None:
        """Record the round's width; this model fits nothing."""
        self.observed.append(len(served))

    def propose(
        self, paths: Sequence[int], domain: Domain
    ) -> dict[int, jnp.ndarray | None]:
        """Draw one uniform point per path over the box this ask hands in."""
        self.calls.append(len(paths))
        self._key, sub = jax.random.split(self._key)
        rows = jax.random.uniform(
            sub,
            (len(paths), domain.d),
            minval=domain.lo,
            maxval=domain.hi,
        )
        return dict(zip(paths, rows, strict=True))


def test_injected_solver_is_driven() -> None:
    """The engine calls the injected method, never its default."""
    k = 3
    eval_f_g, shapes = recording_eval_f_g()
    solver = StubSolver()
    res = n_path_schedule(
        eval_f_g,
        **flock_kwargs(flock(k), unit_domain(), key=jax.random.PRNGKey(0)),
        budget_check=lambda _work: True,
        max_iters=2,  # iteration 0 places the flock, then one step
        local_solver=solver,
    )
    assert solver.inited == [0, 1, 2], solver.inited
    assert solver.trials == [0, 1, 2], solver.trials  # one trial, then accept
    # The flock's one fused batch, then the accepted winners' one: two batches,
    # k points each.
    assert res.n_f_g_points == sum(w for w, _d in shapes) == 2 * k
    assert res.status == "max_iters"


def test_give_up_stalls_a_path() -> None:
    """A "give_up" verdict stalls that path at once: no further trial for it."""
    k = 3
    eval_f_g, shapes = recording_eval_f_g()
    solver = StubSolver(give_up=True)
    res = n_path_schedule(
        eval_f_g,
        **flock_kwargs(flock(k), unit_domain(), key=jax.random.PRNGKey(1)),
        budget_check=lambda _work: True,
        max_iters=2,  # the flock, then the single stalled iteration
        local_solver=solver,
    )
    assert solver.trials == [0, 1, 2]  # one trial each, then named no winner
    # Nothing to commit: every batch this run serves is a trial, and a trial
    # without the value-only seam is fused, so the count is one flock plus one
    # iteration.
    assert res.n_f_g_points == sum(w for w, _d in shapes) == 2 * k
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
        **flock_kwargs(flock(2), unit_domain(), key=jax.random.PRNGKey(3)),
        budget_check=lambda _work: True,
        max_iters=2,  # the flock, then the iteration that finds nothing live
        local_solver=solver,
    )
    assert solver.inited == [], solver.inited  # an unusable start is never seed
    assert solver.steps == [], solver.steps  # and never becomes a start
    assert res.n_f_g_batches == 2, res.n_f_g_batches  # the flock and one re-draw
    assert res.status == "max_iters"


def test_the_gate_counts_g_batches_not_f_only_draws() -> None:
    """The gate bills the fused pass; a value-only trial is counted, not billed.

    What a caller spends real time on is the value *and* gradient pass, so that
    is what `budget_check` counts. A judged step first serves its proposal
    value-only (the ruling is on f) and then buys the winner's gradient, fused
    with the value it already knows - so a step is one billed batch and one
    counted-but-ungated one, and the run never serves a fused batch past the
    gate. The cheap half is reported on its own pair of counters instead of being
    lost; what it never does is move the fused pair or close the gate.
    """
    k = 2

    def run(limit: int) -> tuple[int, int, int, int, int, str]:
        """Serve under one limit; return (billed, donated, f_b, f_p, f_g_p, status)."""
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
            **flock_kwargs(flock(k), unit_domain(), key=jax.random.PRNGKey(7)),
            budget_check=lambda _work: billed < limit,
            max_iters=20,  # not the binding stop: the gate closes first
            local_solver=StubSolver(),
        )
        return (
            billed,
            donated,
            res.n_f_batches,
            res.n_f_points,
            res.n_f_g_points,
            res.status,
        )

    for limit in (1, 2, 4, 7):
        billed, donated, n_f_batches, n_f_points, n_f_g_points, status = run(limit)
        # The gate closes on the billed count alone: never one fused batch past it.
        assert billed == limit, (limit, billed)
        # The fused pair is the billable half, and it is exact.
        assert n_f_g_points == k * billed, (limit, n_f_g_points, donated)
        assert status == "budget_exhausted", (limit, status)
        # The value-only trials did happen, and stayed out of the billed count.
        # One donated trial per iteration after the flock's: the run's first
        # batch is the flock, so a gate of `limit` buys `limit - 1` trials.
        assert donated == limit - 1, (limit, donated)
        # ...and the f pair now records them: one counted batch per served trial,
        # one point per judged path in it. Neither number is what the gate closed
        # on, which is the whole point of splitting the counts.
        assert n_f_batches == donated, (limit, n_f_batches, donated)
        assert n_f_points == k * donated, (limit, n_f_points, donated)


def test_the_gate_is_asked_before_a_value_only_batch_too() -> None:
    """A gate counting value-only work is honored per batch, not per iteration.

    `budget_check` is the caller's own function and the engine cannot see what it
    is made of, so the gate is asked at every serve rather than only the fused
    one. This gate closes after two value-only calls: an un-gated f half would
    run the step's whole retry loop and only be caught by the fused pass that
    follows it, while asking per batch cuts the step at the second.
    """
    f_calls = 0
    eval_f_g, _shapes = recording_eval_f_g()

    def eval_f(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        nonlocal f_calls
        f_calls += 1
        return jnp.sum((x - 0.25) ** 2, axis=-1), None

    res = n_path_schedule(
        eval_f_g,
        eval_f=eval_f,
        **flock_kwargs(flock(2), unit_domain(), key=jax.random.PRNGKey(11)),
        budget_check=lambda _work: f_calls < 2,
        max_iters=10,  # never the binding stop: the caller's gate closes first
        local_solver=RetryingSolver(retries=5),
    )
    assert res.status == "budget_exhausted", res
    # Two value-only batches served; the third was refused before it ran, so the
    # step's remaining four retries were never bought.
    assert res.n_f_batches == 2, res
    # The only fused pass is the flock's: the cut step never reached `settle`.
    assert res.n_f_g_batches == 1, res


def test_a_run_without_an_extra_gate_reports_max_iters() -> None:
    """No gate passed at all (the keyword defaults to None): the horizon only."""
    k = 2
    eval_f_g, shapes = recording_eval_f_g()
    res = n_path_schedule(
        eval_f_g,
        **flock_kwargs(flock(k), unit_domain(), key=jax.random.PRNGKey(4)),
        max_iters=4,  # the flock plus three steps
        local_solver=StubSolver(),
    )
    assert res.status == "max_iters", res
    # The flock's batch, then one fused batch per step (no value-only seam is
    # handed in, so a trial arrives already graded).
    assert res.n_f_g_points == sum(w for w, _d in shapes) == k + 3 * k
    # No seam handed in, so the cheap pair never moves: a fused-only run reports
    # zero on the f axis rather than an absent field.
    assert (res.n_f_points, res.n_f_batches) == (0, 0), res


def test_box_is_data_not_a_unit_cube_assumption() -> None:
    """A foreign box [1,2] plus the caller's own θ = x + 1 map: both are honored.

    The engine must keep to the box it was handed, and must never see the
    caller's parameterization (that map is composed into `eval_f_g`). Every trial
    here would land outside the box unclipped - the landscape's minimizer x = 3
    is not in it - so this method clips its own step, and the engine's box test
    never has to refuse one.
    """
    lo, hi = 1.0, 2.0
    served: list[float] = []

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        """Serve θ = x + 1, f = (θ - 4)²: the caller's map and landscape."""
        served.extend(float(v) for v in x[:, 0])
        theta = x + 1.0
        return jnp.sum((theta - 4.0) ** 2, axis=-1), 2.0 * (theta - 4.0), None

    domain = Domain(jnp.stack([jnp.full(1, lo), jnp.full(1, hi)], axis=1))
    res = n_path_schedule(
        eval_f_g,
        **flock_kwargs(jnp.array([[1.0], [1.5]]), domain, key=jax.random.PRNGKey(5)),
        budget_check=None,
        max_iters=3,
        local_solver=StubSolver(),  # step = x - 0.5·g = 3 exactly: outside the box
    )
    assert served, "nothing was ever served"
    assert min(served) >= lo, min(served)
    assert max(served) <= hi, max(served)
    # the method clipped its own step (the raw trial was 3.0) and the best point
    # sits on the wall it clipped to
    assert abs(max(served) - hi) < 1e-12, max(served)
    assert abs(float(res.x[0]) - hi) < 1e-12, res.x
    assert res.status == "max_iters", res


def test_the_default_model_is_exactly_a_sample_global_model() -> None:
    """Pin that `global_model=None` is `SampleGlobalModel`, not merely a draw.

    The default is a shipped object a caller can hand over itself, so the two
    spellings have to be the same run to the bit: a finite-box number recorded
    with the default then stays comparable with one recorded after a model was
    handed over. Every path gives up on its first trial and is then re-drawn, so
    both sites - the flock's placement and the re-draw - are exercised.
    """
    box = unit_domain(2)

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
            domain=box,
            global_model=global_model,  # pyright: ignore[reportArgumentType]
            budget_check=None,
            max_iters=3,
            local_solver=StubSolver(give_up=True),
            restart_stalled=True,
            key=jax.random.PRNGKey(7),
        )
        return served, res.x, res.f, res.n_f_g_points

    default_served, default_x, default_f, default_points = run(None)
    ours_served, ours_x, ours_f, ours_points = run(
        SampleGlobalModel(key=jax.random.PRNGKey(7)),
    )
    assert default_served, "nothing was ever served"
    assert len(default_served) == len(ours_served)
    for a, b in zip(default_served, ours_served, strict=True):
        assert jnp.array_equal(a, b), (a, b)
    assert jnp.array_equal(default_x, ours_x), (default_x, ours_x)
    assert default_f == ours_f, (default_f, ours_f)
    assert default_points == ours_points, (default_points, ours_points)


def test_reports_four_counts_on_two_axes() -> None:
    """Points and batches are two units; f and f_g are two seams. All four report.

    A batch is the points one eval serves, so a run of k paths that serves the
    same batches reports k times the points of a k=1 run, and the same batch
    count. There is no billing mode to switch and no primary axis: `points` and
    `batches` are the serial cost and the parallel wall-clock cost, and which
    seam served them picks the pair. A run handed no `eval_f` seam reports its
    work on the fused pair and zero on the f pair.
    """

    def run(k: int) -> tuple[int, int, int, int]:
        eval_f_g, _shapes = recording_eval_f_g()
        res = n_path_schedule(
            eval_f_g,
            **flock_kwargs(flock(k), unit_domain(), key=jax.random.PRNGKey(8)),
            budget_check=lambda _work: True,
            max_iters=2,  # the flock plus one accepted trial
            local_solver=StubSolver(),
        )
        return res.n_f_points, res.n_f_batches, res.n_f_g_points, res.n_f_g_batches

    batches = 2  # the flock, then the winners; independent of n_paths
    f_points_3, f_batches_3, points_3, batches_3 = run(3)
    f_points_1, f_batches_1, points_1, batches_1 = run(1)
    assert (f_points_3, f_batches_3) == (0, 0)  # no seam, so no f work
    assert (f_points_1, f_batches_1) == (0, 0)
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
        **flock_kwargs(flock(2), unit_domain(), key=jax.random.PRNGKey(3)),
        max_iters=1,  # the flock alone is iteration 0
        local_solver=StubSolver(),
    )
    assert f_calls == [], f_calls  # nothing to compare, so nothing read early
    assert fused_shapes == [(2, 1)], fused_shapes  # both halves, one pass
    assert res.n_f_g_batches == 1, res  # that pass is also the flock's batch
    assert res.n_f_g_points == 2, res
    assert (res.n_f_points, res.n_f_batches) == (0, 0), res  # never read early


def test_a_rejected_trial_is_bought_no_gradient() -> None:
    """A retrying step pays one gradient pass in total, not one per trial.

    The read quantity is f, so each trial is served for its value alone and only
    the winner is graded. Two trials therefore cost two value-only calls and one
    fused batch - the same single fused batch a no-line-search step costs, which
    is what makes a value-only seam cost-matched across the sides.
    """
    k = 1
    f_calls: list[int] = []

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        f_vals, grads = jax.vmap(pointwise)(x)
        return f_vals, grads, None

    def eval_f(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        f_calls.append(int(x.shape[0]))
        f_vals, _grads = jax.vmap(pointwise)(x)
        return f_vals, None

    solver = RetryingSolver(retries=1)  # the shortest line search
    res = n_path_schedule(
        eval_f_g,
        eval_f=eval_f,
        **flock_kwargs(flock(k), unit_domain(), key=jax.random.PRNGKey(3)),
        max_iters=2,  # the flock, then this one step
        local_solver=solver,
    )
    # The step's two trials, value-only and one batch each. The flock travels to
    # `settle` unread, so it spends no f of its own.
    assert f_calls == [k, k], f_calls
    assert solver.trials == [0, 0], solver.trials
    assert solver.steps == [0], solver.steps  # one winner, one step
    # Two batches for the whole run: the flock's, then the winner's.
    assert res.n_f_g_batches == 2, res
    assert res.n_f_g_points == 2 * k, res


def test_one_gradient_batch_per_iteration() -> None:
    """A restart's starts and a step's winner share one fused batch.

    Deciding on values lets an iteration put everything it kept into a single
    gradient pass: the restarted paths' fresh starts and the stepped paths'
    winners travel together, so a mixed iteration costs one call instead of one
    per half. Path 0 arrives with no point at all (its flock value is non-finite)
    so the engine owes it a fresh start every iteration while path 1 steps; the
    shapes are what pins the merge - unmerged, iteration 1 would show two width-1
    batches rather than one width-2 batch.
    """
    k = 2
    calls = [0]
    fused_shapes: list[tuple[int, int]] = []

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        calls[0] += 1
        fused_shapes.append((int(x.shape[0]), int(x.shape[1])))
        f_vals, grads = jax.vmap(pointwise)(x)
        if calls[0] == 1:  # the flock's batch: column 0 has no evidence
            f_vals = f_vals.at[0].set(jnp.inf)
        return f_vals, grads, None

    def eval_f(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        f_vals, _grads = jax.vmap(pointwise)(x)
        return f_vals, None

    res = n_path_schedule(
        eval_f_g,
        eval_f=eval_f,
        **flock_kwargs(flock(k), unit_domain(), key=jax.random.PRNGKey(3)),
        max_iters=2,  # the flock, then one mixed iteration
        local_solver=StubSolver(),
    )
    assert fused_shapes == [(2, 1), (2, 1)], fused_shapes  # k=2 both times
    assert res.n_f_g_batches == 2, res
    assert res.n_f_g_points == 2 * k, res


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

        def __init__(self) -> None:
            """Start the recorder with an empty feed log."""
            super().__init__()
            self.log: list[str] = []

        def observe(self, served: Mapping[int, ValuePoint]) -> None:
            """Log whether this round came graded, then record it."""
            graded = isinstance(next(iter(served.values())), PathPoint)
            self.log.append("observe:g" if graded else "observe:f")
            super().observe(served)

    def run(*, give_up: bool) -> tuple[Loud, NPathResult]:
        domain = unit_domain()
        model = Loud()

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
            domain=domain,
            global_model=model,
            budget_check=None,
            max_iters=2,  # the flock, then one judged step
            local_solver=StubSolver(give_up=give_up),
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
    and `step` is never called, so the method is not told a step landed.
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
    res = n_path_schedule(
        eval_f_g,
        eval_f=eval_f,
        **flock_kwargs(flock(1), unit_domain(), key=jax.random.PRNGKey(3)),
        max_iters=2,  # the flock, then this one step
        local_solver=solver,
    )
    assert solver.trials == [0], solver.trials  # the trial was read
    assert solver.steps == [], solver.steps  # but never committed
    assert res.status == "max_iters", res


def test_a_non_finite_gradient_start_arrives_stalled() -> None:
    """A start is committed only when f *and* g are finite.

    A finite f with a non-finite gradient is not a usable start: `p.g` is what
    the method proposes from and blends in, so one NaN in it poisons that
    coordinate's magnitude EMAs for the rest of the run. Such a start arrives
    stalled and waits for its rule to fire instead of stepping.
    """

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        f_vals, _grads = jax.vmap(pointwise)(x)
        return f_vals, jnp.full_like(x, jnp.nan), None

    solver = StubSolver()
    res = n_path_schedule(
        eval_f_g,
        **flock_kwargs(flock(2), unit_domain(), key=jax.random.PRNGKey(4)),
        max_iters=2,
        local_solver=solver,
    )
    # No path was ever live, so nothing stepped and nothing was read.
    assert solver.trials == [], solver.trials
    assert solver.steps == [], solver.steps
    assert res.status == "max_iters", res


def test_an_unjudged_step_never_buys_an_f_only_ruling() -> None:
    """A step that needs no verdict is served once, fused - as a flock row is.

    The engine leaves both halves of an unjudged proposal owed, so `settle` buys
    the value and its gradient together. A judged step pays for a value of its
    own: the ruling is value-only, and the winner's gradient is bought again, fused,
    in `settle`. Both lanes therefore bill the same fused batches; what the
    unjudged one never pays is the value-only pass, which is the whole reason to
    answer False. The engine's own f counters agree with the seam's tally, so what
    the unjudged lane skips shows up as a zero rather than as an absence.
    """
    k = 2

    def run(solver: StubSolver) -> tuple[int, int, int, int, int]:
        eval_f_g, _shapes = recording_eval_f_g()
        f_only_widths: list[int] = []

        def eval_f(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
            f_only_widths.append(int(x.shape[0]))
            return jnp.sum((x - 0.25) ** 2, axis=-1), None

        res = n_path_schedule(
            eval_f_g,
            eval_f=eval_f,
            **flock_kwargs(flock(k), unit_domain(), key=jax.random.PRNGKey(7)),
            max_iters=2,  # the flock plus one step
            local_solver=solver,
        )
        return (
            sum(f_only_widths),
            res.n_f_points,
            res.n_f_batches,
            res.n_f_g_batches,
            res.n_f_g_points,
        )

    # The flock needs no ruling either, so iteration 0 is silent. A judged step is
    # one value-only batch of k points, then the winner's fused pass.
    assert run(StubSolver()) == (k, k, 1, 2, 2 * k)
    assert run(UnjudgedSolver()) == (0, 0, 0, 2, 2 * k)


def test_an_unjudged_winner_must_be_finite_to_be_committed() -> None:
    """Unruled means the engine's own gate is the whole of the acceptance test.

    A judged winner was ruled finite before it was kept; an unjudged one is first
    seen in the batch that buys its gradient, so a non-finite half there must
    stall the path rather than blend a nan into the rule's next Δ and the pool.
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
            **flock_kwargs(flock(1), unit_domain(), key=jax.random.PRNGKey(3)),
            max_iters=3,
            local_solver=solver,
        )
        return solver

    # Control: a finite value and gradient commit, so the gate is what stalls below.
    assert run(None).steps == [0, 0]
    assert run("f").steps == []
    assert run("g").steps == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"all {len(fns)} passed")
