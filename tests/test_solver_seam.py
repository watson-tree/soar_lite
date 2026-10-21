"""Solver-seam self-checks.

The seam's contract is only proven by injection: a stub `LocalSolver` and a stub
`RestartRule` are plugged into the engine, which must drive them instead of its
defaults, and a foreign box (`bounds` as data) plus the caller's own x->θ map
composed into `evaluate_batch` must be honoured too. These also pin the two
documented zero-caller hooks — a method
returning `"give_up"` and an unbounded run (`budget_check=None` ->
`status == "max_steps"`, the only stop left).

Run with:

    uv run python -m tests.test_solver_seam
"""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp

from soar_lite.engine import PathPoint, Verdict, n_path_schedule

jax.config.update("jax_enable_x64", val=True)

PROBE_ROUNDS = 2


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
        losses, grads = jax.vmap(pointwise)(x)
        return losses, grads, None

    return evaluate_batch, shapes


class StubSolver:
    """Accepts every trial in one round, or gives up on its first verdict.

    Records the calls the engine makes, so the test can prove the engine drove
    the injected method rather than its default `BBZhangHager`.
    """

    def __init__(self, *, give_up: bool = False) -> None:
        """Store the verdict policy; give_up stalls on the first verdict."""
        self.give_up = give_up
        self.inited: list[int] = []
        self.verdicts: list[int] = []
        self.folds: list[int] = []

    def init(self, i: int, p: PathPoint) -> None:  # noqa: ARG002
        """Record the cold start of path i."""
        self.inited.append(i)

    def restart(self, i: int, p: PathPoint) -> None:
        """Record path i's restart (a cold start again, like init)."""
        self.init(i, p)

    def propose(self, i: int, p: PathPoint) -> jnp.ndarray:  # noqa: ARG002
        """Return a halved steepest-descent step from p."""
        return p.x - 0.5 * p.g

    def verdict(self, i: int, p: PathPoint, trial: PathPoint) -> Verdict:  # noqa: ARG002
        """Record the verdict: accept, or give up on the first round."""
        self.verdicts.append(i)
        return "give_up" if self.give_up else "accept"

    def fold(self, i: int, p: PathPoint, best: PathPoint) -> None:  # noqa: ARG002
        """Record the folded committed step; this stub folds nothing."""
        self.folds.append(i)


class RecordingRule:
    """Records the engine's rule-side calls and never restarts."""

    def __init__(self) -> None:
        """Start with empty evidence, no restarts and no verdicts."""
        self.group_mins: list[float] = []
        self.deltas: list[tuple[int, float]] = []
        self.restarts: list[int] = []
        self.verdicts: list[int] = []

    def observe_continue(self, i: int, delta: float) -> None:
        """Record one loss change delta for path i."""
        self.deltas.append((i, delta))

    def observe_restart(self, r: float) -> None:
        """Record one restart event's payoff r (the pool carries no path index)."""
        self.group_mins.append(r)

    def should_restart(self, i: int, f_k: float, incumbent: float) -> bool:  # noqa: ARG002
        """Record the verdict call for path i and never restart."""
        self.verdicts.append(i)
        return False

    def samples_until_verdict(self) -> int:
        """Report an unknown cadence: the engine keeps its round-by-round rounds."""
        return 0

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
        max_steps=PROBE_ROUNDS + 1,  # the init probe is billed too
        n_probe_rounds=PROBE_ROUNDS,
        solver=solver,
        restart_rule=RecordingRule(),
        key=jax.random.PRNGKey(0),
    )
    assert solver.inited == [0, 1, 2], solver.inited
    assert solver.verdicts == [0, 1, 2], solver.verdicts  # one round, then accept
    assert res.nfev == sum(w for w, _d in shapes) == PROBE_ROUNDS * k + k
    assert res.status == "max_steps"


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
        max_steps=PROBE_ROUNDS + 1,  # the init probe is billed too
        n_probe_rounds=PROBE_ROUNDS,
        solver=solver,
        restart_rule=RecordingRule(),
        key=jax.random.PRNGKey(1),
    )
    assert solver.verdicts == [0, 1, 2]  # one verdict each, then stalled
    # init probe + the single stalled round; the remaining 4 steps serve nothing
    assert res.nfev == sum(w for w, _d in shapes) == PROBE_ROUNDS * k + k
    assert res.status == "max_steps"


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
        max_steps=PROBE_ROUNDS + PROBE_ROUNDS,  # init probe + one horizon
        n_probe_rounds=PROBE_ROUNDS,
        solver=StubSolver(),
        restart_rule=rule,
        key=jax.random.PRNGKey(2),
    )
    assert len(rule.group_mins) == k  # cold start: the n_paths init group-mins, once
    # One horizon is n_probe_rounds rounds, so this run has one feed per path -
    # the continue side's sample is priced like a probe, not like a step.
    assert rule.verdicts == [0, 1] * PROBE_ROUNDS  # asked once per path per round
    assert [i for i, _delta in rule.deltas] == [0, 1]  # Δf over one horizon each
    assert rule.restarts == []
    assert res.status == "max_steps"


def test_rule_can_force_restarts() -> None:
    """A rule that always says restart makes the engine re-probe every step."""
    k = 2
    evaluate_batch, shapes = evaluator()
    solver = StubSolver()
    rule = AlwaysRestartRule()
    res = n_path_schedule(
        evaluate_batch,
        bounds=unit_bounds(),
        budget_check=lambda: True,
        n_paths=k,
        # each round buys a restart probe (PROBE_ROUNDS batches) plus its step
        max_steps=PROBE_ROUNDS + 2 * (PROBE_ROUNDS + 1),
        n_probe_rounds=PROBE_ROUNDS,
        solver=solver,
        restart_rule=rule,
        key=jax.random.PRNGKey(3),
    )
    # init probe + both steps' (restart event + one accepted round)
    assert res.nfev == sum(w for w, _d in shapes)
    assert res.nfev == PROBE_ROUNDS * k + 2 * (PROBE_ROUNDS * k + k)
    assert rule.restarts == [0, 1] * 2  # every path restarted, every step
    assert len(rule.group_mins) == k + 2 * k  # n_paths init + payoff per restart
    assert solver.verdicts == [0, 1] * 2
    assert res.status == "max_steps"


def test_budget_is_never_exceeded_under_restart_storm() -> None:
    """Probes are billed, so a restart storm cannot buy evals outside the budget.

    The expensive event in the engine is the restart probe, and the deepest
    accounting bug this engine had was that the probes were never charged: a
    rule that restarts every round would then have served far more than
    `max_steps` batches. The budget is odd here so the last round cannot be
    bought whole, which is where an unbilled probe would show up.
    """
    k = 3
    for max_steps in (1, PROBE_ROUNDS, 2 * PROBE_ROUNDS + 1, 7):
        evaluate_batch, shapes = evaluator()
        res = n_path_schedule(
            evaluate_batch,
            bounds=unit_bounds(),
            n_paths=k,
            max_steps=max_steps,
            n_probe_rounds=PROBE_ROUNDS,
            solver=StubSolver(),
            restart_rule=AlwaysRestartRule(),
            key=jax.random.PRNGKey(7),
        )
        # served batches * points per batch, and never one batch more than asked
        assert res.nfev == sum(w for w, _d in shapes) <= max_steps * k
        assert len(shapes) <= max_steps, (max_steps, shapes)
        assert res.status == "max_steps", res


def test_no_budget_run_reports_max_steps() -> None:
    """No gate passed at all (the keyword defaults to None): max_steps only."""
    k = 2
    evaluate_batch, shapes = evaluator()
    res = n_path_schedule(
        evaluate_batch,
        bounds=unit_bounds(),
        n_paths=k,
        max_steps=PROBE_ROUNDS + 3,  # the init probe plus three round-batches
        n_probe_rounds=PROBE_ROUNDS,
        solver=StubSolver(),
        restart_rule=RecordingRule(),
        key=jax.random.PRNGKey(4),
    )
    assert res.status == "max_steps", res
    assert res.nfev == sum(w for w, _d in shapes) == PROBE_ROUNDS * k + 3 * k


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
        max_steps=3,
        n_probe_rounds=PROBE_ROUNDS,
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
    assert res.status == "max_steps", res


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
            losses, grads = jax.vmap(pointwise)(x)
            return losses, grads, None

        res = n_path_schedule(
            evaluate_batch,
            bounds=box,
            sample=sample,
            budget_check=None,
            n_paths=2,
            max_steps=3,
            n_probe_rounds=PROBE_ROUNDS,
            solver=StubSolver(),
            restart_rule=AlwaysRestartRule(),
            key=jax.random.PRNGKey(7),
        )
        return served, res.x, res.fun, res.nfev

    default_served, default_x, default_fun, default_nfev = run(None)
    ours_served, ours_x, ours_fun, ours_nfev = run(draw)
    assert default_served, "nothing was ever served"
    assert len(default_served) == len(ours_served)
    for a, b in zip(default_served, ours_served, strict=True):
        assert jnp.array_equal(a, b), (a, b)
    assert jnp.array_equal(default_x, ours_x), (default_x, ours_x)
    assert default_fun == ours_fun, (default_fun, ours_fun)
    assert default_nfev == ours_nfev, (default_nfev, ours_nfev)


def test_every_path_feeds_once_per_completed_horizon() -> None:
    """Pin that every path feeds observe_continue once per completed horizon.

    Column 0 is non-finite in both init probe rounds, so path 0 arrives
    stalled and never steps; the engine must still feed it one
    `observe_continue(0, inf)` per horizon, so its sample count advances in
    lockstep with its advancing cohort instead of leading it by one verdict
    (the off-by-one a top-of-round stall count would cause). One horizon is
    n_probe_rounds rounds - the continue side's sample is priced like a probe -
    so three horizons take 3 * n_probe_rounds rounds.
    """
    k = 2
    calls = [0]

    def evaluate_batch(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        calls[0] += 1
        losses = jnp.full((x.shape[0],), 1.0, dtype=x.dtype)
        if calls[0] <= PROBE_ROUNDS:  # init probe rounds: column 0 has no evidence
            losses = losses.at[0].set(jnp.inf)
        return losses, jnp.zeros_like(x), None

    rule = RecordingRule()
    res = n_path_schedule(
        evaluate_batch,
        bounds=unit_bounds(),
        budget_check=lambda: True,
        n_paths=k,
        max_steps=PROBE_ROUNDS + 3 * PROBE_ROUNDS,  # init probe + three horizons
        n_probe_rounds=PROBE_ROUNDS,
        solver=StubSolver(),
        restart_rule=rule,
        key=jax.random.PRNGKey(6),
    )
    assert res.status == "max_steps", res
    counts: dict[int, int] = {}
    for i, _delta in rule.deltas:
        counts[i] = counts.get(i, 0) + 1
    assert counts == {0: 3, 1: 3}, counts
    assert [d for i, d in rule.deltas if i == 0] == [math.inf] * 3


def test_a_horizon_costs_the_probe_it_is_priced_against() -> None:
    """A sample costs n_probe_rounds rounds at every M, so budget buys samples.

    The verdict prices one continue sample against one probe, and a probe is
    n_probe_rounds rounds - so at M = n_probe_rounds, 2M rounds of budget must
    yield exactly 2 samples per path at every M, never 2M of them. This is the
    assertion that fails if the cadence is ever counted in absolute rounds
    again (the pre-ADR-0004 bug: a one-round Δ against a two-round probe).
    """
    k = 2
    for probe_rounds in (1, 2, 3, 4):
        evaluate_batch, _shapes = evaluator()
        rule = RecordingRule()
        res = n_path_schedule(
            evaluate_batch,
            bounds=unit_bounds(),
            n_paths=k,
            max_steps=probe_rounds + 2 * probe_rounds,  # init probe + 2M rounds
            n_probe_rounds=probe_rounds,
            solver=StubSolver(),
            restart_rule=rule,
            key=jax.random.PRNGKey(5),
        )
        counts: dict[int, int] = {}
        for i, _delta in rule.deltas:
            counts[i] = counts.get(i, 0) + 1
        assert counts == {0: 2, 1: 2}, (probe_rounds, counts)
        assert res.status == "max_steps", (probe_rounds, res)


def test_per_round_evals_bills_a_parallel_round_as_one() -> None:
    """The two billing modes are different cost models, never the same number.

    Default per-point: a n-path round serves n_paths points, so it bills n_paths. Under
    `per_round_evals` the same round is one parallel batch and bills 1 - which
    is exactly why n_paths parallel paths can cost what one serial path costs. This
    pins that equivalence: n_paths=3 per-round and n_paths=1 per-point report the same
    nfev, while n_paths=3 per-point reports n_paths times that.
    """

    def run(*, per_round: bool, k: int) -> int:
        evaluate_batch, _shapes = evaluator()
        res = n_path_schedule(
            evaluate_batch,
            bounds=unit_bounds(),
            budget_check=lambda: True,
            n_paths=k,
            max_steps=PROBE_ROUNDS + 1,  # init probe + one accepted round
            n_probe_rounds=PROBE_ROUNDS,
            solver=StubSolver(),
            restart_rule=RecordingRule(),
            per_round_evals=per_round,
            key=jax.random.PRNGKey(8),
        )
        return res.nfev

    # one probe round is PROBE_ROUNDS(=2) batches, then one accepted step round
    rounds = PROBE_ROUNDS + 1  # = 3 batches, independent of n_paths
    assert run(per_round=False, k=3) == PROBE_ROUNDS * 3 + 3  # 9 points
    assert run(per_round=True, k=3) == rounds  # 3 batches
    assert run(per_round=False, k=1) == rounds  # 3 points
    assert run(per_round=True, k=1) == rounds  # 3 batches
    # the point of the parallel proxy: 3 paths cost one path's rounds
    assert run(per_round=True, k=3) == run(per_round=False, k=1)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"all {len(fns)} passed")
