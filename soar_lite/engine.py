"""The n-path engine: contracts, shared types, and the schedule.

This module is the seam (拍板) of soar_lite: it owns the two plugin contracts
(`LocalSolver` and `RestartRule`), the data the plugins exchange (`PathPoint`,
`Verdict`), and the shared plumbing (`NPathResult`, the box/prior/evaluator
seams, the `_argmin_finite` helper, the state holder `_Budget`, and the
`n_path_schedule` loop). Concretes import *from*
here, never the other way around at module level:

    soar_lite/
      engine.py   <- LocalSolver, RestartRule, PathPoint, Verdict, n_path_schedule
      solver.py   <- DiagonalPreconditioner (implements LocalSolver)
      restart.py  <- EIRestartRule (implements RestartRule)

Dependency direction: plugins depend on engine types; the engine binds its two
defaults lazily inside `n_path_schedule` (keeps the module graph acyclic and
`__init__` importable in any order). The engine is first-order (value-and-
gradient), coordinate-free, and owns the paths, the box, the shared horizon,
the clip-to-bounds and the EI restart cadence; the local solver owns only
direction/step size/acceptance and the restart rule only the continue-vs-
restart decision.
"""

import math
import time
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
    horizon, the n-path state machine (restart/step/stall), the incumbent B,
    and the per-step trial cap max_ls. The method owns the search: the
    direction, the step size, and the acceptance test. All state is per path,
    lives inside the instance, and is opaque to the engine (the engine never
    reads it) - implementers index it by path i.

    One engine step, for every path i that steps this iteration: propose(i, p)
    is called at most max_ls times. p is the path's current point, unchanged
    for the whole step (the engine commits a point only on "accept"), so p is
    also the step's starting point. The returned raw candidate is projected
    into the box by the engine before it is evaluated, so the actual step is
    the projected one: a method must judge the served point it is handed, not
    its own proposal. verdict(i, p, trial) is then called once per served
    candidate, with p = the step's starting point and trial = the served
    point. "accept" commits the lowest-f finite trial served this step - the
    engine's per-step argmin (for a method whose accepted trial is always its
    last served one, that is the same point) - and then calls the method's
    ``fold(i, p, best)`` with the committed winner, which is the method's only
    place to fold an accepted step into its model (the engine is the sole
    owner of "who won"; the method never tracks its own argmin, two copies
    would drift). "retry" spends another trial, and "give_up" stalls the path
    at once - the hook for a method with its own convergence test (the
    delivery's method returns it once its per-coordinate stopping test is all
    true; a method that never returns it stalls on max_ls alone). Arbitration
    is split: ``verdict`` rules the enumeration and may only mutate
    enumeration-local state (a step length, a trial counter); ``fold`` alone
    mutates the model, so a method folds exactly once per committed step.

    These methods are called by the engine, once per path per iteration, so
    they trust their arguments: `n_path_schedule` validates the object once
    (that it carries this contract) at the entry, and nothing here re-checks.
    """

    def restart(self, i: int, p: PathPoint) -> None:
        """Seed path i's state from a fresh start point (its one cold start).

        Called at initialization and again after every restart: the engine has
        one way to place a path on a point, so a method need not tell "first"
        from "again" - both drop its per-path state to the cold one.
        """
        ...

    def propose(self, i: int, p: PathPoint) -> jnp.ndarray:
        """Return path i's raw next candidate from its current point p."""
        ...

    def verdict(self, i: int, p: PathPoint, trial: PathPoint) -> Verdict:
        """Rule the step's enumeration from the served trial; never fold.

        The return is an enumeration decision, not a verdict on the trial's
        quality: "retry" asks for another candidate (the method typically
        first mutates enumeration-local state - shrinks a step length, bumps
        a trial counter); "accept" ends the enumeration, committing the
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
        accepted step and can never disagree with it. The first call after a
        restart seeds state outright; later calls blend.
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
    once per path per iteration, `observe_continue` once per path per iteration,
    and `observe_restart`/`restart` once per probe event (the initialization
    probe included) - so they trust their arguments: `n_path_schedule`
    validates the object once (that it carries this contract) at the entry, and
    nothing here re-checks.
    """

    def observe_continue(self, i: int, delta: float) -> None:
        """Fold path i's one-step value change into its continuation evidence.

        The sample is the engine's: one iteration's step, so this side is
        priced in the step it just took. A stall arrives as `inf` (the rule
        stores it and drops it at the read), never as a Δ that spans the stall.
        """
        ...

    def observe_restart(self, r: float) -> None:
        """Fold in the value r a fresh restart start achieved (restart payoff).

        The engine draws each restart start as the min of one probe group
        (`n_probe_evals` i.i.d. points), so r is that group's min - a real
        sample of "what a restart buys". The history is pooled across paths
        (hence no path index); a rule that ignores payoff no-ops.
        """
        ...

    def should_restart(self, i: int, f_k: float, incumbent: float) -> bool:
        """Return True iff path i should restart from f_k with incumbent B.

        The engine asks this for every path on its first iteration, before any
        probe has run, and that ask is how the flock is seeded: a rule must
        return True while it holds no restart evidence at all (an empty payoff
        pool), which is what triggers the initialization probe. An empty pool
        is the true cold start; a pool that exists but is entirely non-finite
        is evidence of a dead start and must not be re-probed.
        """
        ...

    def samples_until_verdict(self) -> int:
        """Report how many more samples until a verdict can be due (0 = unknown).

        TODO(to be cut): the engine no longer calls this. It lost its only
        caller when the all-stalled shortcut went away - the iteration horizon
        bounds a stalled flock now, not a cadence query. Kept until a caller
        needs it; delete it (and `EIRestartRule`'s copy) once confirmed unused.

        A cadence hint, not evidence: samples, not iterations, though with one
        sample per iteration the two coincide. 0 means "unknown cadence".
        """
        ...

    def restart(self, i: int) -> None:
        """Drop path i's step-local evidence; the engine assumes nothing else.

        Called for every path at initialization (right after its probe), and
        for each path a restart re-probes, so a rule's per-path stage opens
        clean on the first iteration too.
        """
        ...


# --- engine: the n-path schedule ---

NPathStatus: TypeAlias = Literal["budget_exhausted", "max_iters", "time_limit"]
# Three names for the three gates that can end a run, in the order they are
# checked: the iteration horizon, the wall-clock deadline, and the caller's
# extra batch gate. A stalled flock has no fourth stop - it keeps folding inf
# samples until the horizon or the deadline ends the run.

# The one way the engine sees the objective: x rows (M,d) inside `bounds` in,
# (f (M,), grads_x (M,d), feas (M,)|None) out. Value and gradient come back
# together on purpose - the caller's fused value_and_grad is one AD pass, and
# the step's call sites want both at once (scipy's `jac=True` form, batched).
BatchEvaluator: TypeAlias = Callable[
    [jnp.ndarray], "tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]"
]

# The f-only sibling, for a caller that also holds a value function without the
# fused gradient. Same x rows in, (f (M,), feas (M,)|None) out - feasibility
# included because probe group-min and step selection must still tell "finite
# but infeasible" from viable, which f alone cannot. Opting in lets one probe
# compute `g` once (at its winners) instead of once per draw.
EvaluateF: TypeAlias = Callable[[jnp.ndarray], "tuple[jnp.ndarray, jnp.ndarray | None]"]

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
    (the caller's coords), `fun` is its objective value, and `n_iters` counts
    completed engine iterations: the initialization probe is iteration 0, and
    one iteration may serve several batches (restart probes plus up to the
    per-step trial cap), so `n_iters` is not a batch count. It is spelled
    `n_iters` rather than scipy's `nit` to sit beside the two work counts.
    Work is reported on two axes instead of scipy's single `nfev`: `n_points` is the
    g points served (the serial-cost number) and `n_batches` the g batches served
    (the parallel wall-clock number); they are different numbers and never mix.
    Only fused value+gradient passes are charged, so a probe's f-only draws
    appear in neither: the axis that dominates is exact, the cheap half is
    reported nowhere.

    Two deliberate differences from `OptimizeResult`: `success`/`message` are
    absent because the engine has no convergence criterion at all, and `fun` is
    the incumbent over *feasible* points only, so it is `inf` when nothing
    feasible was ever served - the
    trajectory still moves on the penalized objective, but B does not. scipy's `fun`
    has no such filter.
    """

    x: jnp.ndarray  # (d,) coords of the best point found
    fun: float  # min over feasible finite recorded evals; inf if none yet
    n_points: int  # points served (the serial-cost count)
    n_batches: int  # batches (evals) served (the parallel wall-clock count)
    n_iters: int  # completed iterations (the init probe is 0; not trials)
    status: NPathStatus


def n_path_schedule(  # noqa: PLR0912, PLR0913, PLR0915, C901 (the run's own
    # orchestration: knobs + seams + per-iteration dispatch, with the shared
    # state that could move out already moved - _Budget)
    evaluate_batch: BatchEvaluator,
    *,
    evaluate_f: EvaluateF | None = None,
    bounds: Bounds | None = None,  # scipy-style (lb, ub) pairs, a (d, 2) array, or None
    # The prior, injected: (key, n) -> (n, d) in the engine's own coordinates,
    # and the only authority on d. None -> i.i.d. uniform over a finite
    # `bounds` (the historical path); an unbounded or absent box then has no
    # start to draw and is rejected below rather than guessed.
    sample: Sampler | None = None,
    n_paths: int = 16,  # number of paths advanced each iteration, the engine's width
    n_probe_evals: int = 1,  # draws per start; >1 needs evaluate_f
    # The run's horizon, counted in iterations: the engine completes at most
    # max_iters iterations. An iteration is one move per path (a restart probe
    # or a step), not one eval, so this is a reproducible stop, not a cost or
    # wall-clock proxy - one iteration may serve from zero batches (a stalled
    # one) up to a full probe plus max_ls trial batches. None leaves the
    # horizon to `time_limit`.
    max_iters: int | None = None,
    # A wall-clock deadline in seconds, for the caller who budgets time rather
    # than iterations. None means none. A run needs a horizon: at least one of
    # max_iters / time_limit must be given.
    time_limit: float | None = None,
    max_ls: int = 20,  # trials one step may serve before stalling a path
    # Keyword-only and defaulted, like scipy's `callback`: an extra gate for a
    # budget the engine cannot see (points, dollars, an external clock),
    # consulted once per batch. None means "no extra gate". It is additional,
    # never a horizon: a gate that only a served batch advances cannot stop a
    # stalled flock, so a horizon is still required.
    budget_check: Callable[[], bool] | None = None,
    # The two plugins, adjacent: the search method and the restart decision.
    solver: LocalSolver | None = None,  # None -> DiagonalPreconditioner
    restart_rule: RestartRule | None = None,  # None -> EIRestartRule, the default rule
    # The random source, last and optional (optax's `key` placement): None means
    # the fixed default PRNGKey(0), so an unseeded call still runs.
    key: jnp.ndarray | None = None,
) -> NPathResult:
    """Advance n_paths independent chains in x under one shared horizon.

    The engine owns the schedule: it composes and clips the rows into
    `bounds`, serves one batched evaluate_batch call per eval, keeps the
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
    unseeded call still runs, deterministically. Work is reported on two axes,
    never mixed: `n_points` counts every served point (probe, trial, restart
    probe - one each), the serial-cost number; `n_batches` counts served
    batches (evals), the parallel wall-clock number where one n-path batch is
    one unit. They are reports, not caps. The run's own stop is the horizon: at
    most `max_iters` iterations (status "max_iters"), or, if `time_limit` is
    given, the wall-clock deadline (status "time_limit"); at least one of the
    two must be passed. Iterations are the unit on purpose - an iteration is one
    move per path, so the same number reproduces on any machine and does not
    change when the probe depth or the f-only seam changes the per-iteration
    batch count. It is not a cost bound: one iteration serves from zero batches
    (a fully stalled one) up to a full probe plus `max_ls` trial batches, so
    migrating a batch budget to `max_iters` can multiply the work done by that
    per-iteration spread. budget_check() is the caller's additional batch-level
    predicate consulted once per batch (whole batch or None), for a budget the
    engine cannot see; a False verdict stops all paths globally (status
    "budget_exhausted"), and None means no extra gate; that external budget may
    over-run by <M points (the gate flips only between batches).
    fun is the true min over every feasible
    recorded eval with a finite value - every served point when the seam
    reports no aux;
    with aux, only points whose feas mask is True update it. The trajectory
    itself still moves on the penalized objective.

    Selection events (init, restart) never descend blind: each event with n
    starts to fill (n = n_paths at init, n = N restarting paths on a restart) draws
    n_probe_evals value-only batches of n points (pool = n_probe_evals * n).
    Each eval draws n i.i.d. points from the injected sampler (`sample`,
    default uniform over a finite `bounds`: the rule's payoff statistic below
    is defined against that sampler, so a different prior is injected rather
    than reparameterized into the box),
    so the n column-wise groups across evals
    are already random groups, and each group's min becomes one start - n
    starts whose values are honest samples of the "min of n_probe_evals random
    points" distribution (at the default n_probe_evals = 1 a group is a single
    draw and that point is the start; more draws buy a fuller min-of-k and need
    the f-only seam), identical for init and restart regardless of how many
    paths restart (by design). Probe-batch grads are reused for the selected
    starts: starts cost zero extra evals. The rule's
    one shared payoff pool is cold-started with the n_paths init group-mins
    (sigma_r > 0 for n_paths >= 2); n_paths=1 yields a single sample -> sigma_r = 0 ->
    degenerate EI_restart = 0 -> the single path never restarts. Every restart
    event extends {r_i} with the event's group-mins for all paths at once (a
    payoff is a property of the sampler, not of one path). A path whose trials
    exhaust max_ls stalls until an EI verdict revives it.

    The verdict compares a probe against one continue sample. A probe spends
    n_probe_evals evals, while the continue side's evidence is one Δf per
    iteration - the step it just took - non-overlapping, each path on its own
    clock (a restart opens a new stage, and the first iteration after a restart
    is skipped so the probe is not read back as a step's Δ). A stalled iteration
    feeds inf - the read drops the value - rather than a Δ that spans the stall.
    So a stage of M samples costs M evals, but the batch count is not the cost
    that dominates: a probe computes g n_probe_evals times while a step computes
    it once, and g is the dearer half. So n_probe_evals > 1 requires evaluate_f;
    with it both sides compute g exactly once (the probe's f-only draws are not
    charged; only its winners' fused pass is), leaving only the cheap f-count to
    differ. That is why the default is 1.

    The caller's inputs are validated here, once per run: knob ranges and the
    box's own shape/ordering raise ValueError, and a seam object that does not
    carry its contract's methods raises TypeError. The seam methods
    themselves trust the engine - they run per path per iteration, and the
    engine is their only caller.

    Args:
        evaluate_batch: the objective as one batched pass. Takes rows ``x`` of
            shape ``(M, d)`` and returns ``(f: (M,), g: (M, d), feas: (M,) |
            None)`` - value, gradient and optional feasibility mask.
        evaluate_f: the objective's value half alone, same ``x`` rows, returning
            ``(f: (M,), feas: (M,) | None)``. Named for its output where
            ``evaluate_batch`` is named for its unit - the same seam at two
            granularities. ``None`` means no f-only path at
            all - the probe takes a gradient with each draw, which is why a
            depth above 1 is refused; supplying it lets a multi-draw probe
            compute ``g`` once, at its winners, instead of once per draw. The
            winners are served twice (f-only, then fused), but only the fused
            pass is charged: an f-only draw is not a budget event, so
            ``n_points`` counts each winner once.
        bounds: the search box, scipy-style: one ``(lb, ub)`` pair per dim, a
            ``(d, 2)`` array, or ``None`` for no box. ``None``/±inf = unbounded.
        sample: the injected prior ``(key, n) -> (n, d)`` in the engine's own
            coordinates; ``None`` draws i.i.d. uniform over a finite ``bounds``.
        n_paths: number of paths advanced in lockstep each iteration.
        n_probe_evals: probe group size - the draws per start. ``1`` (the
            default) is one draw. Above 1 requires ``evaluate_f``: without it a
            probe would compute ``g`` once per draw against a step's single
            ``g``, and that gradient count - not the batch count - is the
            expensive half of the asymmetry the verdict prices.
        max_iters: the run's horizon in iterations (``None`` leaves it to
            ``time_limit``); a run needs at least one of the two.
        time_limit: a wall-clock deadline in seconds (``None`` means none).
        max_ls: the per-step trial cap after which a path stalls.
        budget_check: an extra gate asked before every g batch, for an external
            budget the engine cannot see; additional, not a horizon, and never
            consulted inside a probe's f-only draws (``None`` means none).
        solver: the local method (``None`` binds ``DiagonalPreconditioner``).
        restart_rule: the restart decision (``None`` binds ``EIRestartRule``).
        key: the PRNG seed; ``None`` uses the fixed default so unseeded runs
            are deterministic.

    Returns:
        An ``NPathResult`` with the run's end state:

        - ``x``: the argmin in the caller's coordinates
        - ``fun``: the incumbent (min finite feasible eval; ``inf`` if none)
        - ``n_points``: g points served (the serial-cost count)
        - ``n_batches``: g batches served (the parallel wall-clock count)
        - ``n_iters``: completed iterations (the init probe is iteration 0)
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
            ...     max_iters=30,
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
        ("n_probe_evals", n_probe_evals, 1),
        ("max_ls", max_ls, 1),
    ):
        if value < low:
            msg = f"{name} must be >= {low}, got {value}"
            raise ValueError(msg)
    if max_iters is not None and max_iters < 0:
        msg = f"max_iters must be >= 0, got {max_iters}"
        raise ValueError(msg)
    if time_limit is not None and time_limit <= 0:
        msg = f"time_limit must be > 0, got {time_limit}"
        raise ValueError(msg)
    if max_iters is None and time_limit is None:
        msg = (
            "a run needs a horizon: pass max_iters (iterations) or "
            "time_limit (seconds); budget_check alone cannot stop a stalled "
            "flock, since a stalled iteration serves no batch"
        )
        raise ValueError(msg)
    _check_bounds(bounds)
    if evaluate_f is None and n_probe_evals != 1:
        # The verdict's expensive side is the gradient count, not the batch
        # count: a probe computes `g` once per draw, a continue step once. The
        # engine cannot see the solver's `n_alpha` (the seam checks presence,
        # not members), so the exact symmetry `n_alpha == n_probe_evals` is out
        # of reach. This is the conservative proxy it can express.
        msg = (
            f"n_probe_evals={n_probe_evals} without evaluate_f prices a probe "
            f"at {n_probe_evals} gradients against one step's single gradient; "
            "pass evaluate_f for multi-draw probes"
        )
        raise ValueError(msg)
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
    budget = _Budget(max_iters, time_limit, budget_check)
    incumbent = _Incumbent(d_bounds)
    # The sampler's d is a property of the callable, not of the draw, so a box
    # mismatch is a wiring bug validated once - on the first draw - rather than
    # re-checked on every probe draw.
    d_checked = False

    def _evaluate_batch(x_batch: jnp.ndarray) -> tuple[list[float], jnp.ndarray]:
        """Serve one g batch, or raise `_BudgetExhausted` if it is unaffordable.

        The gate is asked before the charge, so a batch is served whole or not
        at all - no partial prefix of a batch is ever evaluated - and the
        counters only ever hold g batches that were served. This is the run's
        only charged site: a probe's extra draws are f-only and cheaper, and
        are not budget events. The refusal is a signal, not a state: it unwinds
        straight to the run's one catch, so no intermediate frame has to check
        it. Returns (f_vals, g_batch).
        """
        if not budget.accept():
            raise _BudgetExhausted
        budget.charge(x_batch.shape[0])
        f_batch, g_batch, feas = evaluate_batch(x_batch)
        # per-path bookkeeping (secant/restart) stays Python float,
        # so pull the batch values onto the host in one shot.
        f_vals = f_batch.tolist()
        incumbent.observe(f_vals, x_batch, feas)
        return f_vals, g_batch

    def _evaluate_f(x_batch: jnp.ndarray) -> list[float]:
        """Serve one f-only batch: no gate asked, no charge, same bookkeeping.

        A probe's extra draws are the f half alone, which is the cheap half, so
        they are not budget events: the gate is not asked and the counters do
        not move. Its winners' fused pass still is one, and a probe is entered
        only after the loop head accepted the iteration, so the gate's blind
        spot stays under one probe. Only `probe_select`'s f-only branch calls
        this, which is why the `evaluate_f` assertion is here for the checker
        rather than as a check.
        """
        assert evaluate_f is not None  # the f-only branch's own precondition
        f_batch, feas = evaluate_f(x_batch)
        f_vals = f_batch.tolist()
        incumbent.observe(f_vals, x_batch, feas)
        return f_vals

    def draw_batch(n: int) -> jnp.ndarray:
        """Draw n candidate points; the sampler's d is validated on first draw."""
        nonlocal rng, d_checked
        rng, sub = jax.random.split(rng)
        x_cand = draw(sub, n)
        if not d_checked:
            d_checked = True
            # The sampler is the only authority on d, so a mismatch with the
            # box we were handed is the "two different Nones" wiring bug. Its
            # d is a property of the callable, constant for the run, so one
            # check on this first draw suffices.
            if d_bounds is not None and x_cand.shape[1] != d_bounds:
                msg = (
                    f"the sampler drew d={x_cand.shape[1]} but bounds has d={d_bounds}"
                )
                raise ValueError(msg)
        return x_cand

    def probe_select(n: int) -> list[PathPoint]:
        """Draw n_probe_evals groups of n points, return n group-min starts.

        Each draw is one (n,d) batch sized exactly n, and every draw comes
        from the same sampler, so the fixed column j across draws is already a
        random group of n_probe_evals points; each group's min becomes one
        start (x, f, grad), its value doubling as an {r_i} sample (a group with
        no finite member keeps its draw-0 point and arrives stalled - it waits
        for an EI verdict instead of stepping). Returns the n starts; raises
        `_BudgetExhausted` if a probe batch is unaffordable.

        With `evaluate_f` and more than one draw the draws are f-only and the
        winners' gradients come from one fused batch afterwards, so such a probe
        computes `g` once however many draws it made. One draw has nothing to
        defer - its single gradient is the one its start needs anyway - so a
        lone draw keeps the fused path even when `evaluate_f` is given, and so
        does a run without the seam. Both paths keep the batch width at exactly
        n: widening the f-only draw would multiply the peak activations, and a
        value built through a wide intermediate (a residual vector longer than
        d) can outgrow the fused batch's retained set, so the memory-neutral
        width is the one to keep.
        """
        # Folding a single draw's gradient into a winner batch would serve that
        # same point twice for nothing: with one draw the winner *is* the draw.
        defer_g = evaluate_f is not None and n_probe_evals > 1
        groups = [draw_batch(n) for _ in range(n_probe_evals)]
        f_by_draw: list[list[float]] = []
        g_by_draw: list[jnp.ndarray] = []
        for x_cand in groups:
            if not defer_g:
                f_vals, g_batch = _evaluate_batch(x_cand)
                f_by_draw.append(f_vals)
                g_by_draw.append(g_batch)
            else:
                f_by_draw.append(_evaluate_f(x_cand))
        # Finite-only group-min: a failed (non-finite) probe ranked before a
        # real one would be selected as the start, poisoning that path's local
        # method state for the rest of the run.
        picks: list[tuple[int, int]] = []
        for j in range(n):
            best = _argmin_finite([f_by_draw[r][j] for r in range(n_probe_evals)])
            # Whole group non-finite: keep draw 0 rather than invent a point.
            # Its f is non-finite, so init/restart stalls this path on arrival
            # and only an EI verdict revives it.
            picks.append((0 if best is None else best, j))
        if not defer_g:
            return [
                PathPoint(groups[r][j], f_by_draw[r][j], g_by_draw[r][j])
                for r, j in picks
            ]
        winners = jnp.stack([groups[r][j] for r, j in picks])
        f_win, g_win = _evaluate_batch(winners)
        return [PathPoint(winners[i], f_win[i], g_win[i]) for i in range(n)]

    # A path's point is one object, not three parallel lists: it is what every
    # seam call takes, and a step commits one path at a time (so a list, whose
    # item assignment is the commit, beats a stacked array). The placeholders
    # are replaced by the initialization probe on iteration 0, before any path
    # can step, so their empty coords and inf f are never served.
    points = [PathPoint(jnp.zeros(0), math.inf, jnp.zeros(0)) for _ in range(n_paths)]
    # Nothing has arrived yet, and a path that has not arrived is not stalled:
    # iteration 0 probes every path before the step filter reads this.
    stalled: list[bool] = [False] * n_paths
    # Paths whose probe ran this iteration. They moved by the probe, so they
    # do not also step: an iteration restarts a path or advances it, not both.
    just_restarted: set[int] = set()
    # The continue side's sample is one step, so the Δ it reports is
    # points[i].f - prev_f[i]; a restart resets the anchor to the probe's f.
    prev_f: list[float] = [math.inf] * n_paths
    # The seam check: presence, not signatures. The seam's own members run per
    # path per iteration and stay unvalidated.
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

    def apply_restarts(idxs: list[int]) -> None:
        """Restart the given paths from one fresh probe-select event.

        The event's group-min list extends the rule's one shared payoff pool;
        the method's and the rule's per-path state reset to cold start, while
        that pool is kept (same sampler, new trajectories). A path whose fresh
        group-min has a non-finite f *or* gradient arrives stalled and waits for
        the next EI verdict. The probe is this iteration's move for every
        restarted path, finite or not, so none of them also steps.
        """
        starts = probe_select(len(idxs))
        for point in starts:  # the event's group-mins extend the shared pool
            restart_rule.observe_restart(point.f)
        for i, point in zip(idxs, starts, strict=True):
            points[i] = point
            solver.restart(i, point)
            restart_rule.restart(i)  # the next Δ opens this path's next stage
            # Both halves gate the start, not just f: a finite f with a
            # non-finite gradient is proposed from (`-p.g`) and folded on
            # (`best.g - p.g`), so a single NaN in it poisons that coordinate's
            # magnitude EMAs for the rest of the run. Its g is left as served -
            # a stalled path never proposes, and a revival probes over it.
            stalled[i] = not (
                math.isfinite(point.f) and bool(jnp.all(jnp.isfinite(point.g)))
            )
            prev_f[i] = point.f
        just_restarted.update(idxs)

    def advance_paths() -> None:
        """Advance every path one move: restart it, or step it.

        The iteration's restart votes are folded in here, so a verdict reads the
        incumbent of its own iteration (the engine is B's only writer). A path
        that restarted moved by its probe and does not also step. A refused
        probe raises `_BudgetExhausted` out of here before the step runs.
        """
        to_restart = [
            i
            for i in range(n_paths)
            if restart_rule.should_restart(i, points[i].f, incumbent.fun)
        ]
        if to_restart:
            apply_restarts(to_restart)
        step_paths()

    def step_paths() -> None:
        """One batched engine step for every path that steps this iteration.

        The method proposes, the engine projects and serves, the method
        judges, the engine commits. Trial passes run in lockstep over the
        still-rejecting subset, each served whole or not at all (a budget
        failure stops globally). A path's point is unchanged while it
        rejects, so the point handed to propose/verdict is also the step's
        start point. A path that just restarted is not a stepper: its probe was
        this iteration's move. On return, a path is stalled exactly when it did
        not move at all this iteration (no probe, no commit: gave up, or max_ls
        exhausted on rejects).
        """
        active = [
            i for i in range(n_paths) if not stalled[i] and i not in just_restarted
        ]
        if not active:
            # Every path either waits for a verdict or already moved by its probe.
            return
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
            if not rejecting:
                break
            # The method proposes raw candidates; the engine clips them into
            # `bounds`, so the caller's map never sees a point outside the
            # box the engine advertised. The clipped step is the actual one, so
            # verdict() is handed the served point.
            raw = jnp.stack([solver.propose(i, points[i]) for i in rejecting])
            x_trial = jnp.clip(raw, lo, hi)
            f_trial, g_trial = _evaluate_batch(x_trial)
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

    # Iteration 0 is the initialization probe: the first pass sees no probe
    # evidence, so every rule's verdict is "restart" and the flock is seeded
    # through the same apply_restarts path a later restart uses.
    while True:
        # The horizon and the caller's gate are plain queries, asked before a new
        # iteration starts so the run never begins one it cannot finish. A
        # refusal mid-iteration arrives as _BudgetExhausted below instead.
        if not budget.accept():
            break
        just_restarted.clear()
        try:
            advance_paths()
        except _BudgetExhausted:
            # The iteration asked for a batch it could not afford and was cut
            # short: it did not finish, so it is not folded and not counted.
            break
        # Every path folds one sample per iteration, stalled or not: a stalled
        # path has no step to report, so its sample is inf ("no evidence"), and
        # its own clock keeps advancing in step with its live cohort. A fully
        # stalled flock therefore keeps folding inf until the horizon or the
        # deadline ends the run - it is bounded by those, never by this loop.
        for i in range(n_paths):
            if i in just_restarted:
                # The probe was this iteration's move, not a step: its first Δ
                # belongs to the next iteration, so leave the anchor at the
                # probe's f rather than reading the probe back as a zero-step.
                continue
            delta = math.inf if stalled[i] else points[i].f - prev_f[i]
            restart_rule.observe_continue(i, delta)
            prev_f[i] = points[i].f
        budget.tick()  # counted after the body: only finished iterations

    return NPathResult(
        x=incumbent.x,
        fun=incumbent.fun,
        n_points=budget.n_points,
        n_batches=budget.n_batches,
        n_iters=budget.n_iters,
        status=budget.stop_status(),
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


class _BudgetExhausted(Exception):  # noqa: N818 - a signal, not an error
    """A batch was requested past a closed gate.

    Raised only inside `n_path_schedule`, at the run's one charged site
    (`_evaluate_batch`), and caught by the loop. It carries
    nothing: the run's status is derived from the same queries, and an unserved
    batch is never charged, so they name the gate correctly.
    """


class _Budget:
    """The run's stop meter and work report.

    Three queries decide whether work may go on: the iteration horizon
    (`max_iters`), the wall-clock deadline (`time_limit`), and the caller's
    extra batch gate (`budget_check`). The first two are the run's own promises
    that it ends - at least one is required at entry - and they are the only
    things that bound a stalled flock, which serves no batch at all.
    `budget_check` is additional: a gate that only a served batch advances
    cannot stop a flock that serves nothing.

    `n_batches` and `n_points` are pure reports, not caps: the g batches served
    (one per fused value+gradient pass, the parallel wall-clock number) and the
    points they covered (the serial-cost number). A probe still costs its fused
    batch and a rejected trial still costs its batch, so the report stays the
    honest bill on the axis that dominates; a probe's f-only draws are cheaper
    work and are charged nowhere. The refusal is a signal (`_BudgetExhausted`),
    never a flag stored here, and a gate that only tightens (a counter, a clock)
    needs no latching: the run ends on the first thing it refuses.
    """

    def __init__(
        self,
        max_iters: int | None,
        time_limit: float | None,
        budget_check: Callable[[], bool] | None,
    ) -> None:
        self.max_iters = max_iters
        self.deadline = None if time_limit is None else time.monotonic() + time_limit
        self.n_points = 0
        self.n_batches = 0
        self.n_iters = 0
        self._gate = budget_check

    def charge(self, batch_size: int) -> None:
        """Record one served g batch - the run's only charged site."""
        self.n_batches += 1
        self.n_points += batch_size

    def tick(self) -> None:
        """Record one completed iteration. The caller ticks only finished ones."""
        self.n_iters += 1

    def full(self) -> bool:
        """Whether the iteration horizon is reached."""
        return self.max_iters is not None and self.n_iters >= self.max_iters

    def expired(self) -> bool:
        """Whether the wall-clock deadline has passed."""
        return self.deadline is not None and time.monotonic() >= self.deadline

    def gated(self) -> bool:
        """Whether the caller's extra gate has closed."""
        return self._gate is not None and not self._gate()

    def accept(self) -> bool:
        """Whether the next batch may be served (the run's one stop query)."""
        return not self.full() and not self.expired() and not self.gated()

    def stop_status(self) -> NPathStatus:
        """Name the gate that ended the run; the horizon outranks the clock."""
        if self.full():
            return "max_iters"
        return "time_limit" if self.expired() else "budget_exhausted"


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
