"""The n-path engine: contracts, shared types, and the schedule.

This module is the seam (拍板) of soar_lite: it owns the two plugin contracts
(`LocalSolver` and `RestartRule`), the data the plugins exchange (`PathPoint`,
`Verdict`), and the shared plumbing (`NPathResult`, the box/prior seams, the
`_argmin_finite` helper, the two state holders `_Budget` and `_Horizon`, and the
`n_path_schedule` loop). Concretes import *from*
here, never the other way around at module level:

    soar_lite/
      engine.py   <- LocalSolver, RestartRule, PathPoint, Verdict, n_path_schedule
      solver.py   <- DiagonalPreconditioner (implements LocalSolver)
      restart.py  <- EIRestartRule (implements RestartRule)

Dependency direction: plugins depend on engine types; the engine binds its two
defaults lazily inside `n_path_schedule` (keeps the module graph acyclic and
`__init__` importable in any order). The engine is first-order (value-and-
gradient), coordinate-free, and owns the paths, the box, the shared budget,
the clip-to-bounds and the EI restart cadence; the local solver owns only
direction/step size/acceptance and the restart rule only the continue-vs-
restart decision.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PathPoint:
    """One path's point: coords x, value f, gradient g = ∂f/∂x."""

    x: jnp.ndarray
    f: float
    g: jnp.ndarray


Verdict: TypeAlias = Literal["accept", "retry", "give_up"]


@runtime_checkable
class LocalSolver(Protocol):
    """The search method the n-path engine drives (the engine's plugin point).

    The engine owns the schedule: row composition, projection, the shared
    budget, the n-path state machine (init/restart/stall), the incumbent B,
    and the line-search round count max_ls. The method owns the search: the
    direction, the step size, and the acceptance test. All state is per path,
    lives inside the instance, and is opaque to the engine (the engine never
    reads it) - implementers index it by path i.

    One engine step, for every non-stalled path i: propose(i, p) is called at
    most max_ls times. p is the path's current point, unchanged for the whole
    step (the engine commits a point only on "accept"), so p is also the
    step's starting point. The returned raw candidate is projected into the
    box by the engine before it is evaluated, so the actual step is
    the projected one: a method must judge the served point it is handed, not
    its own proposal. verdict(i, p, trial) is then called once per served
    candidate, with p = the step's starting point and trial = the served
    point. "accept" commits the lowest-f finite trial served this step - the
    engine's per-step argmin (for a method whose accepted trial is always its
    last served one, that is the same point) - and then calls the method's
    ``fold(i, p, best)`` with the committed winner, which is the method's only
    place to fold an accepted step into its model (the engine is the sole
    owner of "who won"; the method never tracks its own argmin, two copies
    would drift). "retry" spends another round, and "give_up" stalls the path
    at once - the hook for a method with its own convergence test (the
    delivery's method returns it once its per-coordinate stopping test is all
    true; a method that never returns it stalls on max_ls alone). Arbitration
    is split: ``verdict`` rules the enumeration and may only mutate
    enumeration-local state (a step length, a round counter); ``fold`` alone
    mutates the model, so a method folds exactly once per committed step.

    These methods are called by the engine, once per path per round, so they
    trust their arguments: `n_path_schedule` validates the object once (that
    it carries this contract) at the entry, and nothing here re-checks.
    """

    def init(self, i: int, p: PathPoint) -> None:
        """Seed path i's state from its start point (cold start)."""
        ...

    def restart(self, i: int, p: PathPoint) -> None:
        """Re-seed path i's state from its fresh start point after a restart."""
        ...

    def propose(self, i: int, p: PathPoint) -> jnp.ndarray:
        """Return path i's raw next candidate from its current point p."""
        ...

    def verdict(self, i: int, p: PathPoint, trial: PathPoint) -> Verdict:
        """Rule the step's enumeration from the served trial; never fold.

        The return is an enumeration decision, not a verdict on the trial's
        quality: "retry" asks for another candidate (the method typically
        first mutates enumeration-local state - shrinks a step length, bumps
        a round counter); "accept" ends the enumeration, committing the
        engine's per-step argmin (this trial, for a first-acceptant or
        single-sample method) which is then folded through ``fold``;
        "give_up" aborts the path's enumeration outright, the hook for a
        method's own convergence test.
        """
        ...

    def fold(self, i: int, p: PathPoint, best: PathPoint) -> None:
        """Fold the step the engine just committed (p -> best) into the model.

        The engine's only call site is the "accept" commit, with best = the
        per-step argmin it committed, so the folded secant is always the
        accepted step and can never disagree with it. The first call after
        init/restart seeds state outright; later calls blend.
        """
        ...


@runtime_checkable
class RestartRule(Protocol):
    """The engine's restart decision module (its second plugin point).

    One instance for the whole engine (like LocalSolver), so the per-path state
    lives inside it in dicts keyed by the path index, exactly as the method
    holds its `eta`/`reference`/`q`. The engine feeds the rule and asks for
    verdicts; the evidence lives inside the instance and the engine never reads
    it. The incumbent B arrives as a should_restart argument rather than as
    stored state, so the engine stays the only writer of B and every verdict
    reads the incumbent of its own iteration.

    The engine offers exactly two evidence channels, and the signatures say
    which is which: `observe_continue(i, delta)` is per-path (what continuing
    bought *that* path), while `observe_restart(r)` carries **no path index**
    because a restart payoff is a property of the sampler and the incumbent,
    not of any one path - every path pools the same history. Those two are the
    two alternatives the verdict compares, and they are the scalar facts the
    engine owns; a rule needing more (point geometry, a surrogate over the
    whole sample history) is an engine variant, not a RestartRule.

    These five methods are a **lower bound** on the engine's call surface - what
    any rule must accept to be pluggable - not a claim that two scalars are all
    a restart decision could ever use. A wider interface would not widen what a
    rule can see; only the engine's call sites could.

    These methods are called by the engine on its own schedule - `should_restart`
    once per path per round, `observe_continue` once per path per completed
    horizon - so they trust their arguments: `n_path_schedule` validates the
    object once (that it carries this contract) at the entry, and nothing here
    re-checks.
    """

    def observe_continue(self, i: int, delta: float) -> None:
        """Fold path i's value change over one horizon into its continuation evidence.

        The horizon is the engine's: `n_probe_rounds` rounds, which is one
        probe's cost, so a sample on this side and an `observe_restart` sample
        buy the same amount of work. A stall anywhere inside the horizon arrives
        as `inf` (the rule stores it and drops it at the read), never as a Δ that
        spans the stall.
        """
        ...

    def observe_restart(self, r: float) -> None:
        """Fold in the value r a fresh restart start achieved (restart payoff).

        The engine draws each restart start as the min of one probe group
        (`n_probe_rounds` i.i.d. points), so r is that group's min - a real
        sample of "what a restart buys". The history is pooled across paths
        (hence no path index); a rule that ignores payoff no-ops.
        """
        ...

    def should_restart(self, i: int, f_k: float, incumbent: float) -> bool:
        """Return True iff path i should restart from f_k with incumbent B."""
        ...

    def samples_until_verdict(self) -> int:
        """Report how many more samples until a verdict can be due (0 = unknown).

        A cadence hint, not evidence, and the engine asks it only once every
        path has stalled: it is what separates "waiting for a verdict" from
        "nothing can happen again". It counts *samples*, not rounds, because
        the horizon - and so what one sample costs in rounds - is the engine's
        `n_probe_rounds`; the engine prices it. A verdict unreachable before
        max_steps makes the remaining rounds pure spin - no eval, no state
        change - and the engine stops there instead of burning the budget. 0
        means "unknown cadence" and disables that shortcut, so a rule with no
        cadence of its own is never mistaken for a dead one.
        """
        ...

    def restart(self, i: int) -> None:
        """Drop path i's step-local evidence; the engine assumes nothing else."""
        ...


# --- engine: the n-path schedule ---

NPathStatus: TypeAlias = Literal["budget_exhausted", "max_steps", "all_stalled"]
# "all_stalled" is a third stop, distinct from the two budget ones: every path
# had stopped and no restart verdict could be reached before max_steps, so the
# engine ended the run there rather than spinning rounds that serve no eval.

# The one way the engine sees the objective: x rows (M,d) inside `bounds` in,
# (f (M,), grads_x (M,d), feas (M,)|None) out. Value and gradient come back
# together on purpose - the caller's fused value_and_grad is one AD pass, and
# every engine call site wants both (scipy's `jac=True` form, batched).
BatchEvaluator: TypeAlias = Callable[
    [jnp.ndarray], "tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]"
]

# The search box in scipy's spelling (which `_box_bounds` normalizes): one
# `(lb, ub)` pair per dim, a side of None or ±inf meaning unbounded. A `(d, 2)`
# array is the same data row-wise, for the callers that already hold one. No box
# at all is spelled `Bounds | None` at the call site (then d is the sampler's).
Bounds: TypeAlias = Sequence["tuple[float | None, float | None]"] | jnp.ndarray

# The one way the engine draws a start: (key, n) -> (n, d) points in the
# engine's own coordinates. It is the only authority on d, which is what makes
# an unbounded search well-posed - the engine has no x0 of its own to fall back
# on, so `sample=None` is only defined over a finite box.
Sampler: TypeAlias = Callable[[jnp.ndarray, int], jnp.ndarray]


@dataclass
class NPathResult:
    """End state of one n_path_schedule run.

    Field names follow `scipy.optimize.OptimizeResult` where the concepts
    correspond: `x` is the argmin in the coordinates the engine was handed
    (the caller's coords), `fun` is its objective value, `nfev` counts
    evaluated points (one eval per served point, batched or not), or
    served batches when the run was billed per round (`per_round_evals`) - the
    parallel-cost proxy - and `nit` counts engine rounds: one round may contain
    several linesearch rounds, so `nit` is not a linesearch count.

    Two deliberate differences from `OptimizeResult`: `success`/`message` are
    absent because the engine has no convergence criterion at all, and `fun` is
    the incumbent over *feasible* points only, so it is `inf` when nothing
    feasible was ever served - the
    trajectory still moves on the penalized objective, but B does not. scipy's `fun`
    has no such filter.
    """

    x: jnp.ndarray  # (d,) coords of the best point found
    fun: float  # min over feasible finite recorded evals; inf if none yet
    nfev: int  # points evaluated, or served batches under per_round_evals
    nit: int  # engine rounds attempted (not linesearch rounds)
    status: NPathStatus


def n_path_schedule(  # noqa: PLR0912, PLR0913, PLR0915, C901 (the run's own
    # orchestration: knobs + seams + per-round dispatch, with the two pieces of
    # shared state that could move out already moved - _Budget and _Horizon)
    evaluate_batch: BatchEvaluator,
    *,
    bounds: Bounds | None = None,  # scipy-style (lb, ub) pairs, a (d, 2) array, or None
    # The prior, injected: (key, n) -> (n, d) in the engine's own coordinates,
    # and the only authority on d. None -> i.i.d. uniform over a finite
    # `bounds` (the historical path); an unbounded or absent box then has no
    # start to draw and is rejected below rather than guessed.
    sample: Sampler | None = None,
    n_paths: int = 16,  # number of paths advanced each round, the engine's width
    n_probe_rounds: int = 2,  # draws/start; 1=init, inf=search; 2-3 empirical, pinned 2
    # The two batch budgets, adjacent: max_steps caps the run, max_ls caps one
    # step. A batch is one round of parallel evaluations - the wall-clock unit a
    # batched objective actually bills - so a probe (n_probe_rounds batches)
    # costs n_probe_rounds of max_steps, init included.
    max_steps: int = 1000,
    max_ls: int = 20,  # batches one step may serve before stalling a path
    # Keyword-only and defaulted, like scipy's `callback`: an extra gate for a
    # budget the engine cannot see (points, wall-clock, dollars), consulted once
    # per batch. None means "no extra gate" - the run then stops on max_steps
    # alone. The engine's own batch budget lives in max_steps, not here.
    budget_check: Callable[[], bool] | None = None,
    # The two plugins, adjacent: the search method and the restart decision.
    solver: LocalSolver | None = None,  # None -> DiagonalPreconditioner
    restart_rule: RestartRule | None = None,  # None -> EIRestartRule, the default rule
    # Billing mode. False (default) charges one eval per served point - the
    # serial-cost model (`nfev = batch size`).
    # True charges one eval per *batch* - a round of parallel evaluations, one
    # wall-clock unit when the objective is batched (jax.vmap or any parallel
    # evaluator). These are two different cost models, never the same number: a
    # round of n_paths paths costs n_paths points serially, 1 round in parallel.
    per_round_evals: bool = False,
    # The random source, last and optional (optax's `key` placement): None means
    # the fixed default PRNGKey(0), so an unseeded call still runs.
    key: jnp.ndarray | None = None,
) -> NPathResult:
    """Advance n_paths independent chains in x under one shared budget.

    The engine owns the schedule: it composes and clips the rows into
    `bounds`, serves one batched evaluate_batch call per round, keeps the
    n-path state machine, and owns the incumbent B. The search itself is
    injected: `solver` owns direction, step size and
    acceptance and keeps its per-path state in Python floats, and
    `restart_rule` is the engine's one rule instance (default
    `EIRestartRule()`, which owns its own knob defaults). `bounds` is data,
    not an injected object (scipy's spelling: a `(lb, ub)` pair per dim,
    `None`/±inf = unbounded) and `sample` is the injected prior: the engine
    clips into the box and draws its starts from the sampler, and knows
    nothing about any map from these coordinates to the caller's physical
    ones. With no sampler the prior is i.i.d. uniform over a finite `bounds`,
    which is then also the sole source of d; an unbounded box (or none) needs
    `sample`, because the engine has no x0 of its own. `key` seeds every draw
    (starts, restart probes); None means the fixed default PRNGKey(0), so an
    unseeded call still runs, deterministically. Billing is set by
    `per_round_evals`: with the default False every served point - probe,
    trial, restart probe - counts exactly one eval (nfev = batch size), the
    serial-cost model; with True each served batch
    counts one (nfev = number of batches), the parallel-cost proxy where a
    n-path round is one wall-clock unit. max_steps is the engine's own budget
    and it is always counted in batches, whatever the billing: every served
    batch is charged one, a probe costs n_probe_rounds of them, and the run
    stops when they are gone (status "max_steps"). budget_check() is the
    caller's additional batch-level predicate consulted once per
    batch (whole batch or None), for a budget the engine cannot see; a False
    verdict stops all paths globally (status "budget_exhausted"), and None
    means no extra gate; that external budget may over-run by <M points (the
    gate flips only between batches), while max_steps is exact in batches.
    fun is the true min over every feasible
    recorded eval with a finite value - every served point when the seam
    reports no aux;
    with aux, only points whose feas mask is True update it. The trajectory
    itself still moves on the penalized objective.

    Selection events (init, restart) never descend blind: each event with n
    starts to fill (n = n_paths at init, n = N restarting paths on a restart) draws
    n_probe_rounds value-only rounds of n points (pool = n_probe_rounds * n).
    Each round draws n i.i.d. points from the injected sampler (`sample`,
    default uniform over a finite `bounds`: the rule's payoff statistic below
    is defined against that sampler, so a different prior is injected rather
    than reparameterized into the box),
    so the n column-wise groups across rounds
    are already random groups, and each group's min becomes one start - n
    starts whose values are honest samples of the "min of n_probe_rounds random
    points" distribution (at the default n_probe_rounds = 2 a group is a pair;
    the 2-3 range is empirical - 1 -> plain init, inf -> random
    search - pinned at 2 by design), identical for init and restart
    regardless of how many paths restart
    (by design). Probe-batch grads are
    reused for the selected starts: starts cost zero extra evals. The rule's
    one shared payoff pool is cold-started with the n_paths init group-mins
    (sigma_r > 0 for n_paths >= 2); n_paths=1 yields a single sample -> sigma_r = 0 ->
    degenerate EI_restart = 0 -> the single path never restarts. Every restart
    event extends {r_i} with the event's group-mins for all paths at once (a
    payoff is a property of the sampler, not of one path). A path whose trials
    exhaust max_ls rounds stalls until an EI verdict revives it.

    The verdict compares equal-cost alternatives: a probe spends
    n_probe_rounds rounds, so the continue side's evidence is one Δf per
    completed horizon of n_probe_rounds rounds, non-overlapping, each path on
    its own clock (a restart opens a new horizon). A stall anywhere inside the
    horizon feeds inf - the read drops the value - rather than a Δ that spans
    the stall. One sample is therefore worth one probe, and a stage of M samples
    costs M * n_probe_rounds rounds.

    The caller's inputs are validated here, once per run: knob ranges and the
    box's own shape/ordering raise ValueError, and a seam object that does not
    carry its contract's methods raises TypeError. The seam methods
    themselves trust the engine - they run per path per round, and the engine
    is their only caller.

    Args:
        evaluate_batch: the objective as one batched pass. Takes rows ``x`` of
            shape ``(M, d)`` and returns ``(f: (M,), g: (M, d), feas: (M,) |
            None)`` - value, gradient and optional feasibility mask.
        bounds: the search box, scipy-style: one ``(lb, ub)`` pair per dim, a
            ``(d, 2)`` array, or ``None`` for no box. ``None``/±inf = unbounded.
        sample: the injected prior ``(key, n) -> (n, d)`` in the engine's own
            coordinates; ``None`` draws i.i.d. uniform over a finite ``bounds``.
        n_paths: number of paths advanced in lockstep each round.
        n_probe_rounds: probe group size - the draws per start, and the horizon
            one continue sample spans.
        max_steps: the engine's own batch budget (always counted in batches).
        max_ls: the per-step round cap after which a path stalls.
        budget_check: an extra whole-batch gate for an external budget the
            engine cannot see; ``None`` means none.
        solver: the local method (``None`` binds ``DiagonalPreconditioner``).
        restart_rule: the restart decision (``None`` binds ``EIRestartRule``).
        per_round_evals: bill batches instead of points (parallel-cost proxy).
        key: the PRNG seed; ``None`` uses the fixed default so unseeded runs
            are deterministic.

    Returns:
        An ``NPathResult`` with the run's end state:

        - ``x``: the argmin in the caller's coordinates
        - ``fun``: the incumbent (min finite feasible eval; ``inf`` if none)
        - ``nfev``: points evaluated, or batches under ``per_round_evals``
        - ``nit``: engine rounds attempted
        - ``status``: why the run stopped

    Example:
        Minimize ``f(x) = ‖x - 3‖²`` by handing the engine a batched objective:

        >>> import jax.numpy as jnp
        >>> from soar_lite import n_path_schedule

        >>> def evaluate_batch(x):
        ...     g = 2.0 * (x - 3.0)
        ...     return (jnp.sum((x - 3.0) ** 2, axis=1), g, None)

        >>> result = n_path_schedule(
        ...     evaluate_batch,
        ...     bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        ...     n_paths=4,
        ...     max_steps=30,
        ... )

        >>> result.x          # ≈ (3, 3)
        Array([3., 3.], dtype=float32)

        >>> result.fun        # its objective value
        0.0

    """
    # The caller's own inputs are checked here, once per run - the only place
    # a human hands this engine something. Nothing below re-checks them.
    for name, value, low in (
        ("n_paths", n_paths, 1),
        ("n_probe_rounds", n_probe_rounds, 1),
        ("max_ls", max_ls, 1),
        ("max_steps", max_steps, 0),
    ):
        if value < low:
            msg = f"{name} must be >= {low}, got {value}"
            raise ValueError(msg)
    _check_bounds(bounds)
    # The random source: an explicit key wins, otherwise the fixed default, so
    # an unseeded call still runs (deterministically).
    rng: jnp.ndarray = jax.random.PRNGKey(0) if key is None else key
    box = _box_bounds(bounds)
    d_bounds: int | None = None if box is None else box[0].shape[0]
    lo, hi = box if box is not None else (-math.inf, math.inf)
    if sample is None:
        # The sample=None contract: a finite box is the only prior the engine
        # can synthesize, so an unbounded side (or no box at all) is a wiring
        # error, not a run to guess our way out of.
        if box is None or not _finite_box(box):
            msg = (
                "an unbounded or absent box needs `sample`: the engine draws "
                "its starts from the sampler, and with sample=None it can only "
                "draw a finite box"
            )
            raise ValueError(msg)
        lo_f, hi_f = box

        def default_sample(sub: jnp.ndarray, n: int) -> jnp.ndarray:
            """Default prior: n i.i.d. uniform points over the finite box."""
            return jax.random.uniform(sub, (n, lo_f.shape[0]), minval=lo_f, maxval=hi_f)

        draw: Sampler = default_sample
    else:
        draw = sample
    budget = _Budget(max_steps, budget_check, per_round_evals=per_round_evals)
    incumbent = _Incumbent(d_bounds)

    def eval_x(
        x_batch: jnp.ndarray,
    ) -> tuple[list[float], jnp.ndarray] | None:
        """Charge one batch, serve it, fold it into B.

        The charge comes first, so a refused batch serves nothing - no partial
        prefix of a batch is ever evaluated. A refusal is global: the whole run
        stops, not just the path that asked. Returns (f_vals, g_batch), or None
        when the meter refused; the meter is the sole owner of both gates.
        """
        if not budget.charge(x_batch.shape[0]):
            return None
        f_batch, g_batch, feas = evaluate_batch(x_batch)
        # per-path bookkeeping (secant/restart) stays Python float,
        # so pull the batch values onto the host in one shot.
        f_vals = f_batch.tolist()
        incumbent.observe(f_vals, x_batch, feas)
        return f_vals, g_batch

    def probe_select(n: int) -> list[PathPoint] | None:
        """Draw n_probe_rounds rounds of n points, return n group-min starts.

        Each round is one (n,d) batch sized exactly n (value-only
        parallelism stays at the value+grad count, never larger). Every
        round draws from the same sampler, so the fixed column j across rounds
        is already a random group of n_probe_rounds points; each group's min
        becomes one start (x, f, grad), its value doubling as an {r_i}
        sample (a group with no finite member keeps its round-0 point and
        arrives stalled - it waits for an EI verdict instead of stepping).
        At n_probe_rounds = 2 (default) a group is a pair ("min of
        2 random points"); the 2-3 range is empirical (1 -> plain
        init, inf -> random search), pinned at 2 by design. Returns the n
        starts, or None when the meter refused a probe round.
        """
        nonlocal rng
        rounds: list[list[PathPoint]] = []
        for _ in range(n_probe_rounds):
            rng, sub = jax.random.split(rng)
            x_cand = draw(sub, n)
            # The sampler is the only authority on d, so a mismatch with the box
            # we were handed is the "two different Nones" wiring bug, caught
            # here rather than three calls later.
            if d_bounds is not None and x_cand.shape[1] != d_bounds:
                msg = (
                    f"the sampler drew d={x_cand.shape[1]} but bounds has d={d_bounds}"
                )
                raise ValueError(msg)
            out = eval_x(x_cand)
            if out is None:  # the meter refused; nothing more can be served
                return None
            f_vals, g_batch = out
            rounds.append(
                [PathPoint(x_cand[j], f_vals[j], g_batch[j]) for j in range(n)]
            )
        starts: list[PathPoint] = []
        for j in range(n):
            # Finite-only group-min: a failed (non-finite) probe ranked before
            # a real one would be selected as the start, poisoning that path's
            # local method state for the rest of the run.
            best = _argmin_finite([rounds[r][j].f for r in range(n_probe_rounds)])
            # Whole group non-finite: keep round 0 rather than invent a point.
            # Its f is non-finite, so init/restart stalls this path on arrival
            # and only an EI verdict revives it.
            starts.append(rounds[0 if best is None else best][j])
        return starts

    init_starts = probe_select(n_paths)
    if init_starts is None:
        # Nothing was served, so there is no x to report: the budget refused
        # even the init probe (max_steps < n_probe_rounds, or the caller's gate).
        return NPathResult(
            x=incumbent.x,
            fun=incumbent.fun,
            nfev=budget.nfev,
            nit=0,
            status=budget.stop_status(),
        )
    # A path's point is one object, not three parallel lists: it is what every
    # seam call takes, and a step commits one path at a time (so a list, whose
    # item assignment is the commit, beats a stacked array).
    points = init_starts
    # a start with no finite evidence never steps: it waits for EI
    stalled: list[bool] = [not math.isfinite(point.f) for point in points]
    # The continue side's sample is priced like the probe it is compared with,
    # so the clock that hands it out lives in _Horizon, not in the loop.
    horizon = _Horizon([point.f for point in points], n_probe_rounds)
    # The seam check: presence, not signatures. The seam's own members run per
    # path per round and stay unvalidated.
    _check_seam(solver, LocalSolver, "solver")
    _check_seam(restart_rule, RestartRule, "restart_rule")
    # Defaults are bound lazily, not at module import: solver and restart
    # import the engine's types, so importing them here would form a cycle.
    if solver is None:
        from soar_lite.solver import DiagonalPreconditioner  # noqa: PLC0415

        solver = DiagonalPreconditioner()
    if restart_rule is None:
        from soar_lite.restart import EIRestartRule  # noqa: PLC0415

        restart_rule = EIRestartRule()
    for i, point in enumerate(points):
        solver.init(i, point)
    for point in points:  # cold-start evidence: init group-mins, into the one pool
        restart_rule.observe_restart(point.f)

    def apply_restarts(idxs: list[int], step: int) -> None:
        """Restart the given paths from one fresh probe-select event.

        The event's group-min list extends the rule's one shared payoff pool;
        the method's and the rule's per-path state reset to cold start, while
        that pool is kept (same sampler, new trajectories). A path whose fresh
        group-min is non-finite arrives stalled and waits for the next EI
        verdict.
        """
        starts = probe_select(len(idxs))
        if starts is None:
            return  # the meter refused; nothing to restart from
        for point in starts:  # the event's group-mins extend the shared pool
            restart_rule.observe_restart(point.f)
        for i, point in zip(idxs, starts, strict=True):
            points[i] = point
            solver.restart(i, point)
            restart_rule.restart(i)  # the next Δ opens this path's next stage
            stalled[i] = not math.isfinite(point.f)
            horizon.restart_at(i, point.f, step)

    def decide_restarts(step: int) -> None:
        """Decide which paths to restart against B.

        B is passed to each verdict (the engine is its only writer), so every
        verdict reads the incumbent of its own iteration. Stalled paths' step
        counting now lives in the main loop's tail (complementary to the
        accepted paths'), so a stalled path's verdict cadence aligns with its
        advancing cohort instead of leading it by one round.
        """
        to_restart = [
            i
            for i in range(n_paths)
            if restart_rule.should_restart(i, points[i].f, incumbent.fun)
        ]
        if to_restart:
            apply_restarts(to_restart, step)

    def step_paths() -> None:  # noqa: C901 - per-path verdict dispatch
        """One batched engine step for every non-stalled path.

        The method proposes, the engine projects and serves, the method
        judges, the engine commits. Rounds run in lockstep over the
        still-rejecting subset, each served whole or not at all (a budget
        failure stops globally). A path's point is unchanged while it
        rejects, so the point handed to propose/verdict is also the step's
        start point. `stalled` is the engine's only path status: on return,
        a path is stalled exactly when it did not commit a step this round
        ("accept"; gave up, or max_ls exhausted on rejects),
        so the caller reads "accepted" as `not stalled[i]`.
        """
        active = [i for i in range(n_paths) if not stalled[i]]
        if not active:
            return  # everyone stalled; verdicts may revive paths next iteration
        rejecting = list(active)
        # Per-step argmin cache: "accept" commits the lowest-f finite trial
        # served this step (not the last served one), so a method can sweep a
        # step-length grid without re-evaluating its argmin. The engine is the
        # sole owner of "who won": the method learns the winner only through
        # the fold callback, never by tracking its own argmin (two copies
        # would drift).
        best: dict[int, PathPoint] = {}
        # max_ls stays engine-side as a failsafe: a solver that returned
        # "retry" forever would hang the engine without this cap.
        for _ in range(max_ls):
            if not rejecting or budget.refused:
                break
            # The method proposes raw candidates; the engine clips them into
            # `bounds`, so the caller's map never sees a point outside the
            # box the engine advertised. The clipped step is the actual one, so
            # verdict() is handed the served point.
            raw = jnp.stack([solver.propose(i, points[i]) for i in rejecting])
            x_trial = jnp.clip(raw, lo, hi)
            out = eval_x(x_trial)
            if out is None:
                break
            f_trial, g_trial = out
            still_rejecting: list[int] = []
            for pos, i in enumerate(rejecting):
                point = points[i]
                trial = PathPoint(x_trial[pos], f_trial[pos], g_trial[pos])
                finite = math.isfinite(trial.f) and bool(jnp.all(jnp.isfinite(trial.g)))
                if finite and (i not in best or trial.f < best[i].f):
                    best[i] = trial
                verdict = solver.verdict(i, point, trial)
                if verdict == "accept":
                    if i not in best:  # no finite sample served: nothing to win
                        stalled[i] = True
                    else:
                        points[i] = best[i]
                        solver.fold(i, point, best[i])
                elif verdict == "retry":
                    still_rejecting.append(i)
                else:  # "give_up": the method's own convergence test fired
                    stalled[i] = True
            rejecting = still_rejecting
        for i in rejecting:  # max_ls exhausted: stall; an EI verdict may revive
            stalled[i] = True

    nit = 0
    all_stalled = False
    for step in range(max_steps):
        # The budget meter: max_steps caps served batches (the work), and the
        # range caps attempted rounds at the same number, so a method never gets
        # more batches than another. A round serves at least one batch unless
        # every path is stalled, so the round cap only bites on a stalling run -
        # it is what stops a permanently-stalled run whose rule can state no
        # cadence, without spending anything.
        if budget.at_cap():
            break
        decide_restarts(step)
        if budget.refused:
            break
        # All paths are stuck and this round's restart votes (asked in
        # decide_restarts above) revived nothing. The next vote cannot be due
        # before the earliest path's stage fills: samples_until_verdict() more
        # samples, each priced n_probe_rounds rounds, the first of them no
        # sooner than that path's own next sample, plus the round the vote
        # is taken in. Both minima are taken separately, which can only
        # under-count the wait - the safe direction: an over-counted wait would
        # stop the engine for a vote that was still due. 0 samples means
        # "unknown cadence", which only happens before any observe, and the
        # arithmetic then goes negative, so the shortcut stays off (the round cap
        # above is then the only bound). The wait is in rounds and so is the
        # remaining round budget: a waiting round serves nothing here, so the
        # batch budget is not what bounds it. Once every path is stalled no eval
        # can fire later, so points / B / nfev are untouched.
        if all(stalled):
            needed = restart_rule.samples_until_verdict()
            stagger = min(horizon.rounds_until_sample(i, step) for i in range(n_paths))
            if stagger + (needed - 1) * n_probe_rounds + 1 >= max_steps - step:
                all_stalled = True
                break
        nit += 1  # a round that got past the restart phase is an attempted one
        step_paths()
        for i, delta in horizon.samples(step, [p.f for p in points], stalled):
            restart_rule.observe_continue(i, delta)
        if budget.refused:
            break

    if budget.refused:
        # Only the meter refuses, and it names the gate that did.
        status: NPathStatus = budget.stop_status()
    elif all_stalled:
        status = "all_stalled"
    else:
        status = "max_steps"
    return NPathResult(
        x=incumbent.x, fun=incumbent.fun, nfev=budget.nfev, nit=nit, status=status
    )


class _Incumbent:
    """B, the incumbent: the best feasible finite point served so far.

    Defined over feasible designs only - with no aux, every point is feasible -
    and over finite values only, so a NaN or inf trial neither becomes B nor
    displaces one. It is the objective every restart verdict is asked against,
    which is why it starts at inf: until something feasible arrives, no verdict
    has anything to compare against.
    """

    def __init__(self, d: int | None) -> None:
        self.fun = math.inf
        # d is the sampler's business: when the box does not supply it, this is
        # an empty placeholder until the first served point fills it in.
        self.x = jnp.full(0 if d is None else d, math.nan)

    def observe(
        self, f_vals: list[float], x_batch: jnp.ndarray, feas: jnp.ndarray | None
    ) -> None:
        """Fold one served batch's feasible finite values in."""
        # With no aux a batch has no feasibility mask and every point counts.
        ok_feas = [True] * len(f_vals) if feas is None else feas.tolist()
        candidates = [j for j, feasible in enumerate(ok_feas) if feasible]
        pos = _argmin_finite([f_vals[j] for j in candidates])
        if pos is None:  # nothing feasible served, or all of it non-finite
            return
        j = candidates[pos]
        if f_vals[j] < self.fun:
            self.fun = f_vals[j]
            self.x = x_batch[j]


class _Budget:
    """The run's batch meter: one unit per served batch, exactly.

    A batch is one round of parallel evaluations - the wall-clock unit a batched
    objective bills - so a probe (n_probe_rounds batches) costs n_probe_rounds
    of max_steps and a rejected trial batch costs one. The charge happens before
    the seam call, so a refused batch serves nothing, and a refusal is sticky:
    every path stops, not just the one that asked.

    Two counters, two units, never mixed: `spent` is the budget meter and is
    always batches; `nfev` is what the caller is told and is batches under
    `per_round_evals`, else the points in the batch.

    `budget_check` is the caller's extra gate for a budget this engine cannot
    see (points, wall-clock, dollars). It is consulted once per batch, so it can
    over-run by less than one batch and never more; None means there is none.
    """

    def __init__(
        self,
        max_steps: int,
        budget_check: Callable[[], bool] | None,
        *,
        per_round_evals: bool,
    ) -> None:
        self.max_steps = max_steps
        self.spent = 0
        self.nfev = 0
        self.refused = False
        self._gate = budget_check
        self._per_round = per_round_evals

    def charge(self, n_points: int) -> bool:
        """Reserve one batch of `n_points`; False once any gate says stop."""
        if (
            self.refused
            or self.spent >= self.max_steps
            or (self._gate is not None and not self._gate())
        ):
            self.refused = True
            return False
        self.spent += 1
        self.nfev += 1 if self._per_round else n_points
        return True

    def at_cap(self) -> bool:
        """Report whether the batch budget is used up (also the round cap)."""
        return self.spent >= self.max_steps

    def stop_status(self) -> NPathStatus:
        """Name the gate that refused: max_steps is exact in batches, so it wins."""
        return "max_steps" if self.at_cap() else "budget_exhausted"


class _Horizon:
    """The continue side's sample clock: one Δf per completed horizon.

    A continue sample is the alternative the verdict prices against a probe, so
    it must span what a probe costs (`n_probe_rounds` rounds). Each path runs its
    own clock and horizons never overlap, so a sample spans exactly that many
    rounds - right after a restart too.

    Two events poison a sample to inf instead of shortening it, because a Δ that
    hides either would read as progress:

    * a restart - the old anchor's Δ would span the jump to a fresh start, which
      is a probe's payoff, not a step's, so the restart resets the anchor and
      opens a new horizon;
    * a stall anywhere inside the horizon.

    A stalled path still closes its horizon on time (the inf is evidence, at the
    cost of the rounds it was measured in), so its cadence stays aligned with
    its advancing cohort rather than leading it by a round.
    """

    def __init__(self, f: list[float], n_probe_rounds: int) -> None:
        self._anchor = list(f)
        self._poisoned = [False] * len(f)
        self._done_at = [n_probe_rounds] * len(f)  # 1-based round it completes
        self._n_probe_rounds = n_probe_rounds

    def samples(
        self, step: int, f: list[float], stalled: list[bool]
    ) -> list[tuple[int, float]]:
        """Close each horizon ending with round `step`: return (path, Δf) pairs."""
        closed: list[tuple[int, float]] = []
        for i in range(len(f)):
            if stalled[i]:
                self._poisoned[i] = True
            if step + 1 >= self._done_at[i]:
                delta = math.inf if self._poisoned[i] else f[i] - self._anchor[i]
                closed.append((i, delta))
                self._anchor[i] = f[i]
                self._poisoned[i] = False
                self._done_at[i] += self._n_probe_rounds
        return closed

    def restart_at(self, i: int, f_i: float, step: int) -> None:
        """Open path i's next horizon at this round, after its restart."""
        self._anchor[i] = f_i
        self._poisoned[i] = False
        self._done_at[i] = step + self._n_probe_rounds

    def rounds_until_sample(self, i: int, step: int) -> int:
        """Return rounds until path i's open horizon completes (0 = this one)."""
        return self._done_at[i] - (step + 1)


def _check_seam(obj: object | None, proto: type, what: str) -> None:
    """Reject a seam object that does not carry the Protocol's methods.

    Presence only, never signatures - a wrong-signature method still fails at
    its own call site. Static typing already knows the type, so this is the
    runtime catch for a caller who hands in anything at all; `None` passes, as
    `n_path_schedule` defaults it right after the check. Deliberately in its own
    function: the isinstance is redundant to a type checker and needs saying once.
    """
    if obj is not None and not isinstance(obj, proto):
        msg = f"{what} does not implement {proto.__name__} (a method is missing)"
        raise TypeError(msg)


def _check_bounds(bounds: Bounds | None) -> None:
    """Reject a malformed box at the entry, not deep inside a clip.

    Both spellings `_box_bounds` accepts are checked: one `(lb, ub)` pair per
    dim, or a `(d, 2)` array. A NaN side is rejected too - it would poison
    every clip silently instead of failing; +-inf sides are legitimate
    (unbounded dim).
    """
    if bounds is None:
        return
    pairs: list[tuple[float | None, float | None]]
    if isinstance(bounds, jnp.ndarray):
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            msg = f"bounds as an array needs shape (d, 2), got {bounds.shape}"
            raise ValueError(msg)
        pairs = list(zip(bounds[:, 0].tolist(), bounds[:, 1].tolist(), strict=True))
    else:
        pairs = []
        for j, pair in enumerate(bounds):
            if len(pair) != 2:
                msg = f"bounds[{j}] must be a (lb, ub) pair, got {pair!r}"
                raise ValueError(msg)
            pairs.append((pair[0], pair[1]))
    for j, (low, high) in enumerate(pairs):
        lo_f = -math.inf if low is None else float(low)
        hi_f = math.inf if high is None else float(high)
        if math.isnan(lo_f) or math.isnan(hi_f):
            msg = f"bounds[{j}] has a NaN side: ({low!r}, {high!r})"
            raise ValueError(msg)
        if lo_f > hi_f:
            msg = f"bounds[{j}] has lo={lo_f} > hi={hi_f}"
            raise ValueError(msg)


def _box_bounds(bounds: Bounds | None) -> tuple[jnp.ndarray, jnp.ndarray] | None:
    """Normalize `bounds` to (lo, hi) arrays; None (no box at all) -> None.

    A `None` side becomes ∓inf, so every downstream consumer sees one spelling
    of "unbounded". A `(d, 2)` array carries the same pairs one row each.
    """
    if bounds is None:
        return None
    if isinstance(bounds, jnp.ndarray):
        return bounds[:, 0], bounds[:, 1]  # (d, 2) array: one pair per row
    lo = [-math.inf if pair[0] is None else pair[0] for pair in bounds]
    hi = [math.inf if pair[1] is None else pair[1] for pair in bounds]
    return jnp.asarray(lo), jnp.asarray(hi)


def _finite_box(box: tuple[jnp.ndarray, jnp.ndarray]) -> bool:
    """Report whether every side of every dim is finite (no-sampler precondition)."""
    lo, hi = box
    return bool(jnp.all(jnp.isfinite(lo)) and jnp.all(jnp.isfinite(hi)))


def _argmin_finite(values: list[float]) -> int | None:
    """Index of the smallest finite value, or None when none of them is finite.

    `min(..., key=...)` is not usable for a value list: NaN compares False
    against everything, so a NaN candidate ranked before the real minimum is
    kept, silently discarding the batch's best point along with its x.
    """
    best: int | None = None
    best_v = math.inf
    for j, v in enumerate(values):
        if math.isfinite(v) and v < best_v:
            best, best_v = j, v
    return best
