"""The n-path engine: contracts, shared types, and the schedule.

This module is the seam (拍板) of soar_lite: it owns the two plugin contracts
(`LocalSolver` and `RestartRule`), the data the plugins exchange (`PathPoint`,
`Verdict`), and the shared plumbing (`NPathResult`, the box/global-model and
evaluator seams, and the `n_path_schedule` loop). The run's state lives in five
private holders, one per state family: `_GlobalModel` (the restart-point
source), `_Evaluator` (the two evaluation seams and the one charged site),
`_Paths` (the n-path state machine), `_Best` (the run's best, `f⋆`), and
`_Budget` (the stop meter and work report). Concretes import *from* here, never
the other way around at module level:

    soar_lite/
      engine.py   <- LocalSolver, RestartRule, PathPoint, Verdict, n_path_schedule
      solver.py   <- DiagonalPreconditioner (implements LocalSolver)
      restart.py  <- NoRestartRule, StallRestartRule (implement RestartRule)

Dependency direction: plugins depend on engine types; the engine binds its two
defaults lazily inside `n_path_schedule` (keeps the module graph acyclic and
`__init__` importable in any order). The engine is first-order (value-and-
gradient), coordinate-free, and owns the flock, the box, the shared horizon, the
clip-to-bounds and the restart cadence; the local solver owns only
direction/step size/acceptance, the restart rule only the continue-vs-restart
verdict, and the global model only where a restart lands.
"""

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

import jax
import jax.numpy as jnp

Verdict: TypeAlias = Literal["retry", "accept", "give_up"]


@dataclass(frozen=True)
class PathPoint:
    """One path's point: coords x, value f, gradient g = ∂f/∂x."""

    x: jnp.ndarray
    f: float
    g: jnp.ndarray

    @property
    def usable(self) -> bool:
        """Return whether this point may be committed as a path's point.

        Both halves must be finite, because both are carried forward: a
        non-finite f poisons the rule's next Δ, and a non-finite g is proposed
        from (`-p.g`) and folded on (`best.g - p.g`), so one NaN there poisons
        that coordinate's magnitude EMAs for the rest of the run. The f half is
        asked first, so a dead value short-circuits before the device sync in
        `_all_finite`.
        """
        return math.isfinite(self.f) and _all_finite(self.g)


@dataclass(frozen=True)
class _Pending:
    """A produced point that `settle` has not promoted to a `PathPoint` yet.

    Producing a point is one decision and grading it is one batch, so a point
    travels between them as this: its value and its gradient are each either
    served or still owed. Anything produced goes through here, and `settle` is
    the one site that promotes it, which is what keeps "does it still need g"
    a question about the point rather than about which producer made it.

    The two halves never disagree in one direction: a fused pass serves both,
    and a point that was never read serves neither, so `g is not None` implies
    `f is not None`. Only `f` alone may be missing - a step's trial is served
    for its value alone while its gradient waits for `settle` to buy it.
    """

    x: jnp.ndarray
    f: float | None
    g: jnp.ndarray | None

    def value(self) -> float:
        """Return the served value, or raise if this point was never read."""
        if self.f is None:
            msg = "_Pending is missing its f"
            raise RuntimeError(msg)
        return self.f

    def graded(self) -> PathPoint:
        """Return this point as a `PathPoint`, or raise if either half is owed."""
        if self.f is None or self.g is None:
            missing = "f" if self.f is None else "g"
            msg = f"_Pending is missing its {missing}"
            raise RuntimeError(msg)
        return PathPoint(self.x, self.f, self.g)


@runtime_checkable
class LocalSolver(Protocol):
    """The search method the n-path engine drives (the engine's plugin point).

    The engine owns the schedule: row composition, projection, the shared
    horizon, the n-path state machine (restart/step/stall), and the best `f⋆`.
    The method owns the search: the direction, the step size, the acceptance
    test, **and how long the enumeration may run** - the engine loops until the
    method ends it, so ``verdict`` must reach "accept" or "give_up" on its own.
    An enumerator with no bound of its own is a hang, not a slow run.
    ``needs_verdict(i)`` is the other half of that ownership: a method that rules
    on each candidate says True, one whose step always commits its single
    proposal says False, and the engine then serves no value for a ruling it
    would only rubber-stamp. All state
    is per path, lives inside the instance, and is opaque to the engine (the
    engine never reads it) - implementers index it by path i. Attributes are
    public so an instance can be inspected and subclassed, not because the
    engine may write them: the engine only hands them to the methods below.

    One engine step, for every path i that steps this iteration: propose(i, p)
    is called repeatedly. p is the path's current point, unchanged for the whole
    step (the engine commits a point only on "accept"), so p is
    also the step's starting point. The returned raw candidate is projected
    into the box by the engine before it is evaluated, so the actual step is
    the projected one: a method must judge the served point it is handed, not
    its own proposal. verdict(i, p, x, f) is then called once per served
    candidate, with p = the step's starting point and (x, f) = the served
    point's coords and value - never its gradient: the engine may serve a
    candidate's value alone and buy the gradient only for the step's winner, so
    a ruling must rest on f and on p (whose gradient a start always has).
    "accept" commits the lowest-f finite trial served this step - the
    engine's per-step argmin (for a method whose accepted trial is always its
    last served one, that is the same point) - and then calls the method's
    ``fold(i, p, best)`` with the committed winner, which is the method's only
    place to fold an accepted step into its model (the engine is the sole
    owner of "who won"; the method never tracks its own argmin, two copies
    would drift). "retry" spends another trial, and "give_up" stalls the path
    at once - the hook for a method with its own convergence test (the
    delivery's method returns it once its per-coordinate stopping test is all
    true, or once its own trial budget runs out). Arbitration
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

    def needs_verdict(self, i: int) -> bool:
        """Report whether path i's next step is ruled on candidate by candidate.

        Asked once per path per iteration, before ``propose``, and the answer
        holds for that whole step. ``True`` is the enumerated kind: the engine
        serves each trial for its value alone and asks ``verdict`` what to do
        with it. ``False`` promises the step commits its one proposal unjudged -
        the engine leaves both halves of the point to `settle`, so its value is
        served once (fused, with the gradient) instead of once for a ruling and
        again for the gradient. What replaces the ruling is the engine's gate: a
        winner whose value or gradient is not finite stalls the path instead of
        being committed, which is the same gate a fresh restart start passes.

        The cost of ``True`` is the method's own, and it is the caller's to
        weigh: absent the ``eval_f`` seam every served trial is a fused pass, so
        a ruled step pays one gradient *per trial* where a committed one pays
        one per step. The engine cannot absorb that - how many trials a step
        runs is not its business - so a method that answers ``True`` for a
        multi-trial step is choosing that billing.
        """
        ...

    def propose(self, i: int, p: PathPoint) -> jnp.ndarray:
        """Return path i's raw next candidate from its current point p."""
        ...

    def verdict(self, i: int, p: PathPoint, x: jnp.ndarray, f: float) -> Verdict:
        """Rule the step's enumeration from the served trial's coords and value.

        `x` is the served (projected) point and `f` its value; the trial's own
        gradient is deliberately absent, because the engine may have served the
        value alone and left the gradient to be bought for the winner. A method
        whose ruling needs the trial's gradient cannot be driven by this engine;
        `p`'s gradient is always there, since a path only starts from a point
        with a usable one.

        Only asked for a path that answered True to ``needs_verdict``, so a step
        that commits its one proposal unjudged never reaches here.

        The return is an enumeration decision, not a verdict on the trial's
        quality: "retry" asks for another candidate (the method typically
        first mutates enumeration-local state - shrinks a step length, bumps
        a trial counter); "accept" ends the enumeration, committing the
        engine's per-step argmin (this trial, for a first-acceptant or
        single-sample method) which is then folded through ``fold``;
        "give_up" aborts the path's enumeration outright, the hook for a
        method's own convergence test. The engine keeps enumerating while this
        says "retry", so one of the two terminal answers is this method's
        responsibility - there is no engine-side trial cap to fall back on.
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

    The rule answers exactly one question - *when* does a path restart - and
    answers it from evidence the engine owns, never from a coordinate. Where a
    restart lands is the `global_model`'s business (the engine asks it for the
    points), so a rule never sees, draws or proposes a point: a restart is two
    seams, and this one is the verdict.

    One instance serves the whole run (like LocalSolver), so the per-path state
    lives inside it in dicts keyed by the path index, exactly as the method
    holds its own. The engine feeds the rule and asks for verdicts; the evidence
    lives inside the instance and the engine never reads it. The raw best `f⋆`
    arrives as a should_restart argument rather than as stored state, so the
    engine stays the only writer of `f⋆` and every verdict reads the best of its
    own iteration. That `f⋆` is the run's **raw** minimum - over every finite
    point served, feasible or not - because the continue side it is compared
    with (`f_k`) is raw too; the feasible minimum the run reports as `fun` is a
    different number whenever a batch carries a feasibility mask.

    Two methods are the whole contract. `observe_step` is the per-path continue
    side: what moving this round bought that path, or `None` when it moved
    nowhere. `should_restart` is the verdict. There is deliberately no channel
    for a restart itself: the round a path spends being placed on a fresh point
    is not a step, so the engine does not report it. A rule is therefore never
    told about its own restarts, which is what lets `StallRestartRule` read
    consecutive `None`s as consecutive stalls without mistaking its restarts for
    them.

    These methods are called by the engine on its own schedule -
    `should_restart` once per path per iteration, before that iteration's move,
    and `observe_step` once per path per iteration the path spent stepping or
    stalling - so they trust their arguments: `n_path_schedule` validates the
    object once (that it carries this contract) at the entry, and nothing here
    re-checks. The one exception is `observe_step`'s `delta`: a finite float or
    `None` (no step) are the only two things the engine reports, so a
    non-finite delta is a wiring error a rule may reject there rather than fold
    into its evidence.
    """

    def observe_step(self, i: int, delta: float | None) -> None:
        """Fold path i's step this round into the continue side, or report none.

        Called once per path per iteration that the path spent stepping or
        stalling. A path placed on a fresh point this round is not reported at
        all - that round is the restart's, not a step's (see the class
        docstring).

        `delta` is the finite one-step value change between the point the path
        held and the one it now holds, or `None` when the path took no step at
        all: its method gave up, or its fresh start or its winner was not a
        usable point. `None` is absence, not a value - the continue side held no
        new step this round.
        """
        ...

    def should_restart(self, i: int, f_k: float, best_fun_raw: float) -> bool:
        """Return True iff path i should be restarted from f_k with raw best `f⋆`.

        Asked for every path before the iteration's move, iteration 0 included:
        `x0` has been placed by then, so there is no cold start for a rule to
        special-case - a path that could not be placed is stalled, and its stall
        arrives on `observe_step`.

        `f⋆` is the raw minimum (see the class docstring) and `f_k` is path i's
        current value in the same unfiltered terms. A path with no current point
        - its initial point was refused as unusable - hands `nan`; a rule that
        restarts on absence reads that case, and a rule that weighs values sees
        only paths that have one.
        """
        ...


# --- engine: the n-path schedule ---

NPathStatus: TypeAlias = Literal[
    "max_iters", "time_limit", "budget_exhausted", "unbounded",
]
# Four names for four ways to end a run: the three gates, in the order they are
# checked (iteration horizon, wall-clock deadline, the caller's extra batch
# gate), plus the objective bottoming out at -inf. A stalled flock has no fifth
# stop - every report still advances a stage clock, so the verdicts keep coming
# until the horizon or the deadline ends the run.

# The one way the engine sees the objective: x rows (M,d) inside `bounds` in,
# (f (M,), grads_x (M,d), feas (M,)|None) out. Value and gradient come back
# together on purpose - the caller's fused value_and_grad is one AD pass, and
# the step's call sites want both at once (scipy's `jac=True` form, batched).
EvalFG: TypeAlias = Callable[
    [jnp.ndarray], "tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]",
]

# The f-only sibling, for a caller that also holds a value function without the
# fused gradient. Same x rows in, (f (M,), feas (M,)|None) out - feasibility
# included because step selection must still tell "finite but infeasible" from
# viable, which f alone cannot. Opting in lets a judged step serve its trials
# for their values alone and buy gradients only for its winners, in one batch.
EvalF: TypeAlias = Callable[[jnp.ndarray], "tuple[jnp.ndarray, jnp.ndarray | None]"]

# The search box in scipy's spelling (which `_box_bounds` normalizes): one
# `(lb, ub)` pair per dim, a side of None or ±inf meaning unbounded. A `(d, 2)`
# array is the same data row-wise, for the callers that already hold one. No box
# at all is spelled `Bounds | None` at the call site (then the flock's own d is
# the only dimension the engine knows).
Bounds: TypeAlias = Sequence["tuple[float | None, float | None]"] | jnp.ndarray

# The restart-point source - SOAR's *global model*: (key, n_points) ->
# (n_points, d) points in the engine's own coordinates. The engine asks it only
# when a restart fires, so it never seeds the run (`x0` does), and it is the
# only authority on where a restart lands. `None` means i.i.d. uniform over a
# finite `bounds`, the model-free default; an injected one may be anything that
# proposes points (a surrogate maximising expected improvement - the SOAR
# reading - a quasi-random grid, or a cached list).
GlobalModel: TypeAlias = Callable[[jnp.ndarray, int], jnp.ndarray]


@dataclass
class NPathResult:
    """End state of one n_path_schedule run.

    Field names follow `scipy.optimize.OptimizeResult` where the concepts
    correspond: `x` is the argmin in the coordinates the engine was handed
    (the caller's coords), `fun` is its objective value, and `n_iters` counts
    completed engine iterations: iteration 0 places the caller's initial points,
    and one iteration may serve several batches (a restart's fresh starts plus
    up to the method's own per-step trial count), so `n_iters` is not a batch
    count. It is spelled
    `n_iters` rather than scipy's `nit` to sit beside the two work counts.
    Work is reported on two axes instead of scipy's single `nfev`: `n_points` is the
    g points served (the serial-cost number) and `n_batches` the g batches served
    (the parallel wall-clock number); they are different numbers and never mix.
    Only fused value+gradient passes are charged, so a step's f-only trials
    appear in neither: the axis that dominates is exact, the cheap half is
    reported nowhere.

    Two deliberate differences from `OptimizeResult`: `success`/`message` are
    absent because the engine has no convergence criterion of its own - the one
    thing it watches for is an objective with no floor, which ends the run with
    `status="unbounded"` - and `fun` is the best over *feasible* points only, so
    it is `inf` when nothing feasible was ever served and `-inf` when a feasible
    point priced the objective at `-inf` (the trajectory still moves on the
    penalized objective, but `f⋆` does not). scipy's `fun` has no such filter.
    """

    x: jnp.ndarray  # (d,) coords of the best point found
    fun: float  # min over feasible recorded evals; inf if none, -inf if unbounded
    n_points: int  # points served (the serial-cost count)
    n_batches: int  # batches (evals) served (the parallel wall-clock count)
    n_iters: int  # completed iterations (iteration 0 places x0; not trials)
    status: NPathStatus


def n_path_schedule(  # noqa: PLR0913 (the caller's knobs, all keyword-only
    # after the one positional seam: one name per dial beats a config object here)
    eval_f_g: EvalFG,
    *,
    # The flock, and the only statement of the run's width and dimension: one
    # starting point per path, in the engine's own coordinates.
    x0: jnp.ndarray,
    eval_f: EvalF | None = None,
    bounds: Bounds | None = None,  # scipy-style (lb, ub) pairs, a (d, 2) array, or None
    # Where a restart lands: (key, n_points) -> (n_points, d) in the engine's
    # coordinates, asked only when a rule fires. None -> i.i.d. uniform over a
    # finite `bounds`; an unbounded or absent box then has nowhere to draw and
    # is rejected below rather than guessed.
    global_model: GlobalModel | None = None,
    # The run's horizon, counted in iterations: the engine completes at most
    # max_iters iterations. An iteration is one move per path (iteration 0
    # places `x0` instead), not one eval, so this is a reproducible stop, not a
    # cost or wall-clock proxy - one iteration may serve from zero batches (a
    # stalled one) up to a restart's starts plus however many trials the
    # method's own search serves. None leaves the horizon to `time_limit`.
    max_iters: int | None = None,
    # A wall-clock deadline in seconds, for the caller who budgets time rather
    # than iterations. None means none. A run needs a horizon: at least one of
    # max_iters / time_limit must be given.
    time_limit: float | None = None,
    # Keyword-only and defaulted, like scipy's `callback`: an extra gate for a
    # budget the engine cannot see (points, dollars, an external clock). It is
    # one of `_Budget.stop`'s three inputs, asked at the loop head every
    # iteration and again before each g batch (the f-only trials never ask it).
    # None means "no extra gate". It is additional, never a horizon: a gate
    # that advances only on served batches spins forever under a stalled flock,
    # and the engine cannot see which kind this is, so a horizon is required.
    budget_check: Callable[[], bool] | None = None,
    # The two plugins, adjacent: the search method and the restart decision.
    solver: LocalSolver | None = None,  # None -> DiagonalPreconditioner
    restart_rule: RestartRule | None = None,  # None -> NoRestartRule (frozen)
    # The random source of the restart points, last and optional (optax's `key`
    # placement): None means the fixed default PRNGKey(0), so an unseeded call
    # still runs.
    key: jnp.ndarray | None = None,
) -> NPathResult:
    """Advance a flock of chains in x from the caller's initial points.

    The engine owns the schedule: it places `x0`, composes and clips every
    proposal into `bounds`, serves each batch as one `eval_f_g` call, keeps the
    n-path state machine, and owns the best `f⋆`. Three things are injected.
    `solver` owns direction, step size and acceptance, and keeps its per-path
    state in Python floats. `restart_rule` decides **when** a path restarts,
    from the evidence the engine forwards (each path's step deltas and the raw
    best). `global_model` supplies **where** a restart lands: the engine asks it
    for one point per restarting path, clips them into the box and places them.
    Splitting a restart into those two seams is the point - a rule that never
    restarts (`NoRestartRule`, the default) needs no model at all, and a model
    that proposes points (a surrogate maximising expected improvement, an
    out-of-tree adapter) never touches the verdict.

    `x0` is the flock: shape ``(n_paths, d)``, one starting point per path, so
    `n_paths` and `d` are its shape rather than separate knobs. Iteration 0
    serves it as one fused batch and places every path on its row; a row whose
    served value or gradient comes back non-finite stalls that path on arrival
    instead of being placed. With `global_model=None` the engine draws restart
    points i.i.d. uniform over a finite `bounds`, which is then also the only
    source it has - an unbounded box (or none) needs an injected model, because
    the engine has no distribution of its own to draw from. `key` seeds those
    draws; None means the fixed default PRNGKey(0), so an unseeded call still
    runs, deterministically.

    `bounds` is data, not an injected object (scipy's spelling: a `(lb, ub)`
    pair per dim, `None`/±inf = unbounded); the engine clips into it and knows
    nothing about any map from these coordinates to the caller's physical ones.

    Work is billed on two axes, never mixed, and only the fused
    value+gradient pass is charged: f-only trials and rejected trial batches are
    billed nowhere. `n_points` is the g points served (the initial points, each
    restart start, and each trial winner - one each), the serial-cost number;
    `n_batches` is the g batches (one per fused pass), the parallel wall-clock
    number. They are reports, not caps. The run's own stop is the horizon: at
    most `max_iters` iterations (status "max_iters"), or, if `time_limit` is
    given, the wall-clock deadline (status "time_limit"); at least one of the
    two must be passed. Iterations are the unit on purpose - an iteration is one
    move per path (iteration 0 places `x0` instead), so the same number
    reproduces on any machine and does not change when the method's per-step
    trial count changes the per-iteration batch count. It is not a cost bound:
    one iteration serves from zero batches (a fully stalled one) up to a
    restart's starts plus however many trials the method's own search serves, so
    migrating a batch budget to `max_iters` can multiply the work done by that
    per-iteration spread. budget_check() is the caller's extra stop - one of the
    three inputs the run's one query (`_Budget.stop`) asks, at the loop head
    every iteration and again before each g batch - for a budget the engine
    cannot see; a False verdict stops all paths globally (status
    "budget_exhausted"), and None means no extra gate; that external budget may
    over-run by <M points (the gate flips only between its asks).
    fun is the true min over every feasible recorded eval with a finite value -
    every served point when the seam reports no aux; with aux, only points whose
    feas mask is True update it. The trajectory itself still moves on the
    penalized objective.

    The caller's inputs are validated here, once per run: knob ranges, the
    box's own shape/ordering, and `x0`'s shape and finiteness raise ValueError,
    and a seam object that does not carry its contract's methods raises
    TypeError. The seam methods themselves trust the engine - they run per path
    per iteration, and the engine is their only caller.

    Args:
        eval_f_g: the objective as one batched pass. Takes rows ``x`` of
            shape ``(M, d)`` and returns ``(f: (M,), g: (M, d), feas: (M,) |
            None)`` - value, gradient and optional feasibility mask. A mask
            holds 0/1 (or booleans) and nothing else: only 0 marks a point
            infeasible, so any other flag is refused rather than read as
            truthiness.
        x0: the flock's initial points, shape ``(n_paths, d)``, in the engine's
            own coordinates. Its shape is the run's width and dimension, so
            there is no separate ``n_paths`` knob; a non-finite entry is refused
            at the entry rather than left to stall a path.
        eval_f: the objective's value half alone, same ``x`` rows, returning
            ``(f: (M,), feas: (M,) | None)``. Both seams are named for what they
            return - ``eval_f_g`` hands back f and g together, ``eval_f`` f
            alone - so neither name leans on the unit it is called with.
            ``None`` means no f-only path at all: every served trial takes its
            gradient in the same fused pass. Supplying it lets a judged step
            serve its trials for their values alone and buy gradients only for
            its winners, in one batch.
        bounds: the search box, scipy-style: one ``(lb, ub)`` pair per dim, a
            ``(d, 2)`` array, or ``None`` for no box. ``None``/±inf = unbounded.
        global_model: the restart-point source ``(key, n_points) -> (n_points,
            d)`` in the engine's own coordinates, asked only when a restart
            fires; ``None`` draws i.i.d. uniform over a finite ``bounds``.
        max_iters: the run's horizon in iterations (``None`` leaves it to
            ``time_limit``); a run needs at least one of the two.
        time_limit: a wall-clock deadline in seconds (``None`` means none).
        budget_check: an extra gate for a budget the engine cannot see; one of
            the three inputs the run's one stop query asks, at the loop head
            every iteration and again before each g batch, never inside a step's
            f-only trials. Additional, not a horizon (``None`` means none).
        solver: the local method (``None`` binds ``DiagonalPreconditioner``).
        restart_rule: the restart decision (``None`` binds ``NoRestartRule``,
            the frozen flock that never re-seeds).
        key: the PRNG seed of the restart draws; ``None`` uses the fixed default
            so unseeded runs are deterministic.

    Returns:
        An ``NPathResult`` with the run's end state:

        - ``x``: the argmin in the caller's coordinates
        - ``fun``: the best value (min feasible eval; ``inf`` if none, ``-inf``
          when the objective has no floor, which also ends the run)
        - ``n_points``: g points served (the serial-cost count)
        - ``n_batches``: g batches served (the parallel wall-clock count)
        - ``n_iters``: completed iterations (iteration 0 places ``x0``)
        - ``status``: why the run stopped

    Example:
        Minimize ``f(x) = ‖x - 3‖²`` by handing the engine a batched objective:

            >>> import jax.numpy as jnp
            >>> from soar_lite import n_path_schedule

            >>> def eval_f_g(x):
            ...     g = 2.0 * (x - 3.0)
            ...     return (jnp.sum((x - 3.0) ** 2, axis=1), g, None)

            >>> result = n_path_schedule(
            ...     eval_f_g,
            ...     x0=jnp.array([[-1.0, -1.0], [1.0, 1.0], [-4.0, 4.0], [4.0, -4.0]]),
            ...     bounds=[(-5.0, 5.0), (-5.0, 5.0)],
            ...     max_iters=30,
            ... )

            >>> result.x          # ≈ (3, 3)
            Array([3., 3.], dtype=float32)

            >>> result.fun        # its objective value
            0.0

    """
    _validate_run(
        x0=x0,
        max_iters=max_iters,
        time_limit=time_limit,
        eval_f_g=eval_f_g,
        eval_f=eval_f,
        global_model=global_model,
        budget_check=budget_check,
        bounds=bounds,
    )
    # The seam check: presence, not signatures, and the two defaults, bound
    # lazily - solver and the restart rules import the engine's types, so
    # importing them at module scope would form a cycle. Kept with the knob
    # checks above and apart from the build below.
    _check_seam(solver, LocalSolver, "solver")
    _check_seam(restart_rule, RestartRule, "restart_rule")
    if solver is None:
        from soar_lite.solver import DiagonalPreconditioner  # noqa: PLC0415

        solver = DiagonalPreconditioner()
    if restart_rule is None:
        from soar_lite.restart import NoRestartRule  # noqa: PLC0415

        restart_rule = NoRestartRule()

    box = _box_bounds(bounds)
    clip = box if box is not None else (-math.inf, math.inf)
    # The flock is clipped once, here: the box is the engine's and every point a
    # path holds is inside it, so a caller's out-of-box start is brought in
    # rather than searched from outside. d is then x0's second axis - the run's
    # one statement of dimension - and `_Best`/`_GlobalModel` are told it.
    flock = jnp.clip(jnp.asarray(x0), *clip)
    d = flock.shape[1]
    if global_model is None:
        # The global_model=None contract: a finite box is the only distribution
        # the engine can synthesize, so an unbounded side (or no box at all) is
        # a wiring error, not a run to guess our way out of.
        if box is None or not _finite_box(box):
            msg = (
                "an unbounded or absent box needs `global_model`: the engine "
                "draws its restart points from the model, and with "
                "global_model=None it can only draw a finite box"
            )
            raise ValueError(msg)
        lo_f, hi_f = box

        def default_global_model(sub: jnp.ndarray, n_points: int) -> jnp.ndarray:
            """Default source: n_points i.i.d. uniform points over the finite box."""
            return jax.random.uniform(
                sub, (n_points, lo_f.shape[0]), minval=lo_f, maxval=hi_f,
            )

        propose: GlobalModel = default_global_model
    else:
        propose = global_model
    # The random source: an explicit key wins, otherwise the fixed default, so
    # an unseeded call still runs (deterministically). The model owns the draw's
    # own d check - its d is a property of the callable, not of the draw, so a
    # mismatch with the flock's d is validated once, on the first restart.
    model = _GlobalModel(propose, d, jax.random.PRNGKey(0) if key is None else key)
    budget = _Budget(max_iters, time_limit, budget_check)
    best = _Best(d)

    evaluator = _Evaluator(eval_f_g, eval_f, budget=budget, best=best)
    # Last, because it takes the resolved seams and their collaborators: the
    # state machine drives them rather than receiving them per call.
    paths = _Paths(flock, clip, evaluator, solver, restart_rule, model)

    status: NPathStatus | None = None
    while True:
        # The run's one stop query names its reason, asked before a new iteration
        # starts so the run never begins one it cannot finish. A refusal
        # mid-iteration arrives as _BudgetExhausted carrying the same reason.
        status = budget.stop()
        if status is not None:
            break
        try:
            # Iteration 0 places the flock; every later iteration reads the
            # best, decides which paths restart, decides every move on values,
            # then buys the gradients those decisions need in one batch. A
            # refused batch raises `_BudgetExhausted` here, before anything is
            # placed.
            paths.move(best.fun_raw)
        except _BudgetExhausted as exc:
            # The iteration asked for a batch it could not afford and was cut
            # short: it did not finish, so it is not folded and not counted.
            status = exc.reason
            break
        except _OptimumReached:
            # A served feasible point priced f at -inf. Nothing to unwind and
            # nothing left to decide: the batch that found it was paid for and
            # is counted, but the iteration did not finish, so it is not folded
            # or ticked either.
            status = "unbounded"
            break
        budget.tick()  # counted after the body: only finished iterations

    return NPathResult(
        x=best.x,
        fun=best.fun,
        n_points=budget.n_points,
        n_batches=budget.n_batches,
        n_iters=budget.n_iters,
        status=status,
    )


class _BudgetExhausted(Exception):  # noqa: N818 - a signal, not an error
    """A batch was requested past a closed gate, the reason carried along.

    Raised only inside `n_path_schedule`, at the run's one charged site
    (`_Evaluator._serve_f_g`), and caught by the loop. It carries the reason
    its `stop` answered at the batch altitude, so the loop names the run's
    status without re-asking the query (the caller's predicate is consulted
    once per ask, not re-derived by a second call).
    """

    reason: NPathStatus  # the gate `stop` named, carried to the loop's one catch

    def __init__(self, reason: NPathStatus) -> None:
        self.reason = reason
        super().__init__(reason)


class _OptimumReached(Exception):  # noqa: N818 - a signal, not an error
    """A served *feasible* point priced the objective at `-inf`.

    Raised inside `_Best.observe`, the single funnel both seams pass through,
    and caught by the loop. Unlike `_BudgetExhausted` it arrives after a batch
    was served and charged (the batch that found the point, which was paid for)
    and it leaves an answer behind instead of naming a gate: the run stops on
    that point, so `fun` is `-inf` and `status` is `"unbounded"`. Nothing is
    left to unwind - a batch raises before any point is placed or advanced.
    """


class _GlobalModel:
    """The run's restart-point source, and the draw's own d check.

    Holds the splitting key and the invariant that the model's d is a property
    of the callable, not of the draw, so a mismatch with the flock's d is a
    wiring bug caught once rather than on every restart.
    """

    def __init__(self, propose: GlobalModel, d: int, key: jnp.ndarray) -> None:
        self._propose = propose
        self._d = d
        self._key = key
        self._checked = False

    def points(self, n_points: int) -> jnp.ndarray:
        """Ask for n_points restart points; the model's d is validated once."""
        self._key, sub = jax.random.split(self._key)
        x_cand = self._propose(sub, n_points)
        if not self._checked:
            # d is a property of the callable, constant for the run: one check
            # after the first draw suffices.
            self._checked = True
            if x_cand.shape[1] != self._d:
                msg = (
                    f"the global model proposed d={x_cand.shape[1]} but the "
                    f"flock has d={self._d}"
                )
                raise ValueError(msg)
        return x_cand


class _Budget:
    """The run's stop meter and work report.

    Three predicates decide whether work may go on: the iteration horizon
    (`max_iters`), the wall-clock deadline (`time_limit`), and the caller's
    extra gate (`budget_check`). `stop` asks all three as one query - at the
    loop head every iteration, and again before each g batch - and names the
    reason it refuses. At least one of the first two is required at entry,
    because they are the only stops the engine can see advance on its own: a
    `budget_check` that flips only on served batches spins forever under a
    stalled flock, and the engine cannot tell it from one that flips alone.

    `n_batches` and `n_points` are pure reports, not caps: the g batches served
    (one per fused value+gradient pass, the parallel wall-clock number) and the
    points they covered (the serial-cost number). What is charged is the one
    gradient pass an iteration buys for the points it decided on, so a rejected
    trial is cheaper work that is charged nowhere and the report stays the
    honest bill on the axis that dominates. The refusal is a signal
    (`_BudgetExhausted`),
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

    def stop(self) -> NPathStatus | None:
        """Name the reason work may not go on, or None while it may.

        Asked at the loop head before every iteration, and again before each g
        batch, where a stop becomes the `_BudgetExhausted` signal. None while
        work may go on; otherwise the first closed gate, named - the horizon
        outranks the clock, which outranks the caller's gate.
        """
        if self.full():
            return "max_iters"
        if self.expired():
            return "time_limit"
        if self.gated():
            return "budget_exhausted"
        return None


class _Best:
    """The two bests the run keeps: what it reports, and what a verdict reads.

    `fun`/`x` are the answer: the minimum over feasible points served, which is
    what the run reports. Defined over feasible designs only - with no aux,
    every point is feasible. A NaN or +inf trial neither becomes `f⋆` nor
    displaces one: it carries no value to compare. A feasible -inf is not an
    improvement to weigh but the answer itself - the objective has no floor
    there, so no later trial can beat it and no step can improve on it, which
    is why it ends the run (`unbounded`) instead of entering a comparison. It
    starts at inf: until something feasible arrives there is no answer to
    report.
    `fun_raw` is the same minimum with the feasibility filter removed: every
    finite point served, feasible or not. This is `f⋆`, the value a restart
    verdict is asked against rather than `fun`, because the continue side is
    unfiltered too - `f_k` is the path's own served value wherever the path
    stands. A filtered `f⋆` read against that would put the two sides of one
    verdict on different scales.
    """

    def __init__(self, d: int) -> None:
        self.fun = math.inf
        self.fun_raw = math.inf
        # Set when a feasible point priced the objective at -inf: the run has an
        # answer that no later trial can beat, so it stops on it.
        self.unbounded = False
        # d is the flock's business (it is `x0`'s second axis), so the
        # placeholder is already the right shape.
        self.x = jnp.full(d, math.nan)

    def observe(
        self, f_vals: list[float], x_batch: jnp.ndarray, feas: jnp.ndarray | None,
    ) -> None:
        """Fold one served batch's values into both bests, stopping on -inf.

        Raises:
            ValueError: If a feasibility mask holds anything but 0/1. Nothing
                is folded in that case, so a batch's values are never added
                under a mask that cannot be read.

        """
        # The mask is read first, and read as flags: it is the caller's claim, so
        # a batch whose mask cannot be read is refused before either best moves.
        # With no aux a batch has no mask and every point counts.
        ok_feas = [True] * len(f_vals) if feas is None else _feas_flags(feas)
        j_raw = _argmin_finite(f_vals)
        if j_raw is not None and f_vals[j_raw] < self.fun_raw:
            self.fun_raw = f_vals[j_raw]
        candidates = [j for j, feasible in enumerate(ok_feas) if feasible]
        # Ahead of the finite read, which skips it: a feasible -inf is the
        # answer itself, not a value to compare, and the run ends on it here.
        j_floor = next((j for j in candidates if f_vals[j] == -math.inf), None)
        if j_floor is not None:
            self.fun = -math.inf
            self.x = x_batch[j_floor]
            self.unbounded = True
            raise _OptimumReached
        pos = _argmin_finite([f_vals[j] for j in candidates])
        if pos is None:  # nothing feasible served, or all of it non-finite
            return
        j = candidates[pos]
        if f_vals[j] < self.fun:
            self.fun = f_vals[j]
            self.x = x_batch[j]


class _Evaluator:
    """The run's evaluation seams, and what they charge and observe.

    Both halves share one rule, stated here once so no call site has to restate
    it: a fused `_serve_f_g` is the run's only charged site (the gate is asked
    before the charge, so a batch is served whole or not at all), and a
    `_serve_f` call is cheaper work that is not a budget event. Whichever half
    served a point updates the best. `seed_points` and `settle` live here too:
    iteration 0 serves the caller's `x0` through this seam, and `settle` is
    where an iteration's deferred gradients are bought together, since this
    object owns both the seam and the charge.
    """

    def __init__(
        self,
        eval_f_g: EvalFG,
        eval_f: EvalF | None,
        *,
        budget: _Budget,
        best: _Best,
    ) -> None:
        self._eval_f_g = eval_f_g
        self._eval_f = eval_f
        self._budget = budget
        self._best = best

    def seed_points(self, x0: jnp.ndarray) -> list[PathPoint]:
        """Serve the flock as one fused batch, graded on arrival.

        Iteration 0 has nothing deferred: the starts are the run's first points,
        so there is no earlier point to compare them with and no reason to
        serve them for values alone. The whole flock is one batch, so a wide
        flock costs one pass.
        """
        f_vals, g_batch = self._serve_f_g(x0)
        return [
            PathPoint(x0[pos], f_vals[pos], g_batch[pos]) for pos in range(len(f_vals))
        ]

    def pending_unread(self, x_batch: jnp.ndarray) -> list[_Pending]:
        """Return one unread point per row: both halves owed to `settle`.

        The twin of `serve_trials` for a batch nothing has read yet: a restart
        start has no values to compare, and an unjudged step winner is committed
        as proposed, so each defers even its cheap f and takes f and g from one
        fused pass. Owning this spelling here, beside `settle`'s promotion, is
        what keeps `_Pending` construction in this object instead of at the call
        sites.
        """
        return [_Pending(row, None, None) for row in x_batch]

    def serve_trials(self, x_batch: jnp.ndarray) -> list[_Pending]:
        """Serve one step's trial batch, as one produced candidate per row.

        A trial's gradient is only ever wanted for the step's winner, so with the
        f-only seam each trial is served for its value alone and the winners are
        graded once, together (`settle`). Without the seam a fused pass is the
        only way to a value, so every trial carries its gradient whether or not
        the step's argmin will keep it; `settle` then promotes the winner and
        buys nothing.
        """
        if self._eval_f is not None:
            return [
                _Pending(row, f, None)
                for row, f in zip(x_batch, self._serve_f(x_batch), strict=True)
            ]
        f_vals, g_batch = self._serve_f_g(x_batch)
        return [
            _Pending(x_batch[pos], f_vals[pos], g_batch[pos])
            for pos in range(len(f_vals))
        ]

    def settle(self, decided: list[_Pending]) -> list[PathPoint]:
        """Buy whatever gradient the iteration's decided points still owe, in one batch.

        Everything the f-only seam left pending goes out as a single batch, so an
        iteration costs one gradient pass however many moves it deferred - its
        restart starts plus its step winners - instead of one pass per trial. A
        point that already carries its gradient (every point on a run without the
        seam) is promoted as it stands, so a run that
        owes nothing buys nothing and the counters do not move. Returns the same
        points, in the same order, all graded.
        """
        pending_idxs = [k for k, point in enumerate(decided) if point.g is None]
        settled: dict[int, PathPoint] = {}
        if pending_idxs:
            f_vals, g_batch = self._serve_f_g(
                jnp.stack([decided[k].x for k in pending_idxs]),
            )
            settled = {
                k: PathPoint(decided[k].x, f_vals[pos], g_batch[pos])
                for pos, k in enumerate(pending_idxs)
            }
        return [
            settled[k] if point.g is None else point.graded()
            for k, point in enumerate(decided)
        ]

    def _serve_f_g(self, x_batch: jnp.ndarray) -> tuple[list[float], jnp.ndarray]:
        """Serve one g batch, or raise `_BudgetExhausted` if it is unaffordable.

        The stop query is asked at this batch altitude before the charge, so a
        batch is served whole or not at all - no partial prefix of a batch is
        ever evaluated - and the counters only ever hold g batches that were
        served. This is the run's only charged site: a step's rejected trials
        are f-only and cheaper, and are not budget events. The refusal is a
        signal, not a state: it carries its reason straight to the run's one
        catch, so no intermediate frame has to check it. Returns (f_vals,
        g_batch).
        """
        reason = self._budget.stop()
        if reason is not None:
            raise _BudgetExhausted(reason)
        self._budget.charge(x_batch.shape[0])
        f_batch, g_batch, feas = self._eval_f_g(x_batch)
        # per-path bookkeeping (secant/restart) stays Python float,
        # so pull the batch values onto the host in one shot.
        f_vals = f_batch.tolist()
        self._best.observe(f_vals, x_batch, feas)
        return f_vals, g_batch

    def _serve_f(self, x_batch: jnp.ndarray) -> list[float]:
        """Serve one f-only batch: no gate asked, no charge, same bookkeeping.

        A step's rejected trials are the f half alone, which is the cheap half,
        so they are not budget events: the stop query is not asked and the
        counters do not move. The winners' fused pass still is one, and the call
        site is entered only after the loop head's stop query cleared the
        iteration, so the f-only blind spot stays under one step. Only
        `serve_trials` calls this; the None-check below is for the type checker
        (it cannot see the caller is f-only) and is unreachable at runtime.
        """
        eval_f = self._eval_f
        if eval_f is None:
            msg = "_serve_f requires eval_f"
            raise RuntimeError(msg)
        f_batch, feas = eval_f(x_batch)
        f_vals = f_batch.tolist()
        self._best.observe(f_vals, x_batch, feas)
        return f_vals


class _Paths:
    """The n-path state machine: one point, one anchor and one stall flag per path.

    The invariants that the loop's four sites would otherwise carry implicitly,
    stated once here instead:

    - A path that restarted this iteration moved by its fresh start, so it does
      not also step, and it reports nothing to the rule at all: its first Δ
      belongs to the next iteration, so the restart is never read back as a
      zero-length step. `move` computes the restarted list once per iteration
      and threads it to `_restart_starts` and to the complement both halves
      share - one fact at the top instead of a stored flag that must be cleared
      each iteration.
    - An iteration decides on values and pays for gradients once: whatever the
      f-only seam left ungraded (restart starts and step winners alike) is
      bought
      in one batch before anything is placed, so a decision is never priced
      before it is made.
    - A path is stalled when it did not move at all this iteration - its method
      gave up on every trial, or its fresh start or its winner was not a usable
      point. A stalled path does not step and reports `None` to the rule (a
      stall tick, no evidence), so its own stage clock keeps pace with its live
      cohort.
    - A start or a winner is committed only when it is `usable` - finite in
      both halves - one gate defined once on the point type.
    """

    def __init__(  # noqa: PLR0913, PLR0917 - the run's collaborators, built once
        self,
        x0: jnp.ndarray,
        clip: tuple[jnp.ndarray | float, jnp.ndarray | float],
        evaluator: _Evaluator,
        solver: LocalSolver,
        restart_rule: RestartRule,
        global_model: _GlobalModel,
    ) -> None:
        self.n_paths = int(x0.shape[0])
        # The flock as handed in (the entry already clipped it): iteration 0
        # places it, and that is the only use it gets - a later restart asks the
        # global model instead, so the caller's starts are consumed once.
        self.x0 = x0
        self._seed_pending = True
        # Constant for the whole run, so it belongs beside the shape rather than
        # in `move`'s signature: every proposal below is clipped into this box.
        self.clip = clip
        # The four collaborators this state machine drives. Held rather than
        # handed in per call: every method below uses at least one of them, and
        # a step would otherwise thread them only to pass them straight back.
        self.evaluator = evaluator
        self.solver = solver
        self.restart_rule = restart_rule
        self.global_model = global_model
        # A path's point is one object, not three parallel lists: it is what
        # every seam call takes, and a step commits one path at a time (so a
        # list, whose item assignment is the commit, beats a stacked array). The
        # placeholder says "no point has arrived": a usable restart start
        # replaces it, an unusable one never becomes a point, and a stalled path
        # never serves or compares it.
        self.points = [
            PathPoint(jnp.zeros(0), math.nan, jnp.zeros(0)) for _ in range(self.n_paths)
        ]
        # Nothing has arrived yet, and a path that has not arrived is not
        # stalled: iteration 0 places the flock before anything reads the flag.
        self.stalled: list[bool] = [False] * self.n_paths
        # The continue side's sample is one step, so the Δ it reports is
        # points[i].f - prev_f[i]; a restart resets the anchor to the fresh
        # start's f. `nan` because no anchor has arrived yet, and only a placed
        # path reports a Δ - `_place` sets the anchor before that first Δ is read.
        self.prev_f: list[float] = [math.nan] * self.n_paths

    def move(self, best_fun_raw: float) -> None:
        """Take one iteration's move for every path, then settle what it deferred.

        Iteration 0 is the exception and returns early: the caller's flock is
        placed, and nothing has a step to take yet.

        The restart decision comes first, against the best the caller holds: a
        path whose rule fires is given a fresh start instead of stepping this
        iteration. Both halves then decide where a path goes from values alone
        when the f-only seam is present - the restarted paths by the global
        model's proposal, the cohort by its trials - and the gradients those
        decisions need are then bought together, in one batch (`settle`), so an
        iteration costs one gradient pass however many moves it deferred. What
        is not deferred is served fused and reaches `settle` already graded:
        every point of a run without the seam.

        The two halves touch disjoint paths - a restarted path is in neither -
        and every call into a collaborator here is per path, so deciding all
        moves before placing any of them is the same schedule as placing each
        move as it is decided.

        `best_fun_raw` is the barrier a restart has to clear: the value the rule
        judges against, not the best point's reported value, which is why it
        arrives already unwrapped.
        """
        if self._seed_pending:
            self._seed()
            return
        restarted = [
            i
            for i in range(self.n_paths)
            if self.restart_rule.should_restart(i, self.points[i].f, best_fun_raw)
        ]
        # Every path the rule did not fire on, stalled ones included: a stalled
        # path proposes nothing but still reports a stall this iteration, so its
        # own stage clock keeps pace with its live cohort.
        cohort = [i for i in range(self.n_paths) if i not in restarted]
        starts = self._restart_starts(len(restarted)) if restarted else []
        won = self._decide(cohort)
        # One batch: restart starts first, then the winners in path order. A
        # refusal (`_BudgetExhausted`) cuts the iteration short with nothing
        # placed - this is the only batch it serves. A point that arrived
        # already graded (a run without the seam) passes straight through
        # `settle` without being served again.
        order = sorted(won)
        settled = self.evaluator.settle([*starts, *(won[i] for i in order)])
        restarts = dict(zip(restarted, settled[: len(starts)], strict=True))
        wins = dict(zip(order, settled[len(starts) :], strict=True))
        self._place(restarts)
        self._advance(wins)
        self._report(cohort)

    def _seed(self) -> None:
        """Place every path on its row of the flock (iteration 0, one batch).

        The flock is the caller's answer to "where does the run start", so it is
        served once, fused, and never asked for again: a restart asks the global
        model instead. A row the engine cannot use is not placed - it stalls the
        path on the same `usable` gate `_place` and `_advance` apply - so
        `points[i]` keeps meaning "where the path stands".
        """
        self._seed_pending = False
        for i, point in enumerate(self.evaluator.seed_points(self.x0)):
            self.stalled[i] = not point.usable
            if self.stalled[i]:
                continue
            self.points[i] = point
            self.solver.restart(i, point)
            self.prev_f[i] = point.f

    def _restart_starts(self, n_points: int) -> list[_Pending]:
        """Ask the global model for n_points fresh starts, clipped into the box.

        Ask, clip, place: the model proposes in the engine's own coordinates but
        knows nothing of the box, so the clip is the engine's (the same one every
        step proposal passes). Both halves stay owed - a start has nothing to be
        judged on - so `settle` buys them in the same batch as the step winners,
        which is what keeps a restart from costing its own gradient pass.
        """
        x_cand = jnp.clip(self.global_model.points(n_points), *self.clip)
        return self.evaluator.pending_unread(x_cand)

    def _decide(self, cohort: list[int]) -> dict[int, _Pending]:
        """Run the cohort's trials: the method judges, the engine records winners.

        The cohort splits on ``needs_verdict``. A path that needs one has its
        step enumerated: the method proposes, the engine projects and serves,
        the method judges. Trial passes run in lockstep over the still-rejecting
        subset, each served whole or not at all (a budget failure stops
        globally). A path's point is unchanged while it rejects, so the point
        handed to propose and verdict is also the step's start point. A path that
        needs none commits its one proposal unread - the same deal a restart
        start gets - so the value it would have been judged on is served
        once, fused with its gradient, by the caller's `settle`.

        Returns the per-step argmin each accepting path won, keyed by path - still
        waiting for `settle` when the seam served values only (and always for the
        unjudged lane), which is what the caller's one settle batch then covers.
        Nothing is committed here: a path ends this iteration stalled exactly when
        its method found nothing to win (its convergence test fired, or it
        accepted with no finite sample). The cohort also holds paths that were
        already stalled: they run no trials and emit no winner.
        """
        if not cohort:
            # Every path restarted this iteration: the cohort is empty.
            return {}
        lo, hi = self.clip
        # A path already stalled only reports its sample this iteration, so it
        # stays out of both lanes; `stalled` also gates the exit observation
        # below without a second membership list.
        unjudged, rejecting = self._lanes(cohort)
        won = self._propose_unjudged(unjudged) if unjudged else {}
        # Per-step argmin cache: "accept" commits the lowest-f finite trial served
        # this step (not the last served one), so a method can sweep a step-length
        # grid without re-evaluating its argmin. The engine is the sole owner of
        # "who won": the method learns the winner only through the fold callback,
        # never by tracking its own argmin (two copies would drift). `won` holds
        # only the paths that actually won, so a path that gives up after a good
        # earlier trial commits nothing.
        best: dict[int, _Pending] = {}
        # No engine-side trial cap: how long a step's enumeration may run is the
        # method's own business (its convergence test, its own trial budget), and
        # `verdict` is where it says so. A cap here would be a second, invisible
        # stopping rule for a decision the method already owns.
        while rejecting:
            # The method proposes raw candidates; the engine clips them into
            # `bounds`, so the caller's map never sees a point outside the box the
            # engine advertised. The clipped step is the actual one, so the
            # verdict is handed the served point.
            raw = jnp.stack([self.solver.propose(i, self.points[i]) for i in rejecting])
            x_trial = jnp.clip(raw, lo, hi)
            served = self.evaluator.serve_trials(x_trial)
            still_rejecting: list[int] = []
            for pos, i in enumerate(rejecting):
                point = self.points[i]
                trial = served[pos]
                f_trial = trial.value()
                # A gradient the seam already served is gated here, on the same
                # rule as a start: nothing folds a, or proposes from, a point
                # whose gradient is not all finite. A value-only trial has none
                # to gate yet - it is gated by `settle` + `_advance` instead.
                finite = math.isfinite(f_trial) and (
                    trial.g is None or _all_finite(trial.g)
                )
                if finite and (i not in best or f_trial < best[i].value()):
                    best[i] = trial
                verdict = self.solver.verdict(i, point, trial.x, f_trial)
                if verdict == "retry":
                    still_rejecting.append(i)
                elif verdict == "accept":
                    if i not in best:  # no finite sample served: nothing to win
                        self.stalled[i] = True
                    else:
                        won[i] = best[i]
                else:  # "give_up": the method's own convergence test fired
                    self.stalled[i] = True
            rejecting = still_rejecting
        return won

    def _lanes(self, cohort: list[int]) -> tuple[list[int], list[int]]:
        """Split the cohort into its (unjudged, judged) lanes.

        A stalled path joins neither: it proposes nothing this iteration and
        emits no winner, it only reports its stall.
        """
        unjudged: list[int] = []
        judged: list[int] = []
        for i in cohort:
            if not self.stalled[i]:
                (judged if self.solver.needs_verdict(i) else unjudged).append(i)
        return unjudged, judged

    def _propose_unjudged(self, paths: list[int]) -> dict[int, _Pending]:
        """Take each unjudged path's one proposal as its step's winner, unread.

        Proposals are still clipped into the box: the engine owns that, and the
        served point is the clipped one. Both halves are left owed, which is what
        makes the iteration serve this point's value once rather than twice.
        """
        raw = jnp.stack([self.solver.propose(i, self.points[i]) for i in paths])
        clipped = jnp.clip(raw, *self.clip)
        return dict(zip(paths, self.evaluator.pending_unread(clipped), strict=True))

    def _place(self, placed: dict[int, PathPoint]) -> None:
        """Place the given paths on their fresh starts (one restart event).

        The method's per-path state resets to cold start. A start the engine
        cannot use is refused on the same `usable` gate `_advance` applies to a
        winner: it is not placed, the solver is not seeded from it, and the path
        waits, stalled, for its rule to fire again. Refusing rather than placing
        it keeps `points[i]` meaning "where the path stands": an unusable start
        never becomes one. The start is this iteration's move for every restarted
        path, usable or not, so none of them also steps.
        """
        for i, point in placed.items():
            self.stalled[i] = not point.usable
            if self.stalled[i]:
                continue  # unusable: no point, no seed, no anchor (see `usable`)
            self.points[i] = point
            self.solver.restart(i, point)
            self.prev_f[i] = point.f

    def _advance(self, won: dict[int, PathPoint]) -> None:
        """Commit each accepted winner as its path's next point, then fold it.

        The fold is the step's commit half: it is what tells the method the point
        landed, and it is the only place the secant is read, so a winner the
        engine refuses here is refused before the method has been told anything.
        """
        for i, best in won.items():
            if not best.usable:
                # A judged winner was ruled usable before it was kept, so this
                # gate only ever bites the unjudged lane, whose value is first
                # seen in the batch that bought its gradient. Same gate as
                # `_place`: an unusable point never becomes a path's point.
                self.stalled[i] = True
                continue
            point = self.points[i]  # the step's start, for the fold's secant
            self.points[i] = best
            self.solver.fold(i, point, best)

    def _report(self, cohort: list[int]) -> None:
        """Fold one report per path per round, on the one continue channel.

        Every non-restarting path reports exactly one observation per iteration,
        so the rule's stage clock counts one round however the path spent it. A
        cohort path reports the finite step Δ when it moved, or `None` when
        `stalled` says it did not - read back through the step's own exit state
        rather than an outside reader reconstructing it. A restarted path reports
        nothing at all: it took no step this round, and its anchor is already the
        fresh start's f (`_place` set `prev_f`), so the restart is never read back
        as a zero-length step. That silence is why the channel is
        one-directional - a rule is not told about its own restarts, so it cannot
        mistake one for a stall.
        """
        for i in cohort:
            if self.stalled[i]:
                # Absence, not a value: the rule counts the round, not a Δ.
                self.restart_rule.observe_step(i, None)
            else:
                delta = self.points[i].f - self.prev_f[i]
                self.restart_rule.observe_step(i, delta)
                self.prev_f[i] = self.points[i].f


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


def _validate_run(  # noqa: PLR0913 - the entry's own knobs, named once
    *,
    x0: jnp.ndarray,
    max_iters: int | None,
    time_limit: float | None,
    eval_f_g: EvalFG,
    eval_f: EvalF | None,
    global_model: GlobalModel | None,
    budget_check: Callable[[], bool] | None,
    bounds: Bounds | None,
) -> None:
    """Reject a run the engine cannot finish, cannot price, or cannot call.

    The caller's own inputs are checked here, once per run - the only place a
    human hands this engine something. Nothing below re-checks them.
    """
    if max_iters is not None and max_iters < 0:
        msg = f"max_iters must be >= 0, got {max_iters}"
        raise ValueError(msg)
    if time_limit is not None and time_limit <= 0:
        msg = f"time_limit must be > 0, got {time_limit}"
        raise ValueError(msg)
    if max_iters is None and time_limit is None:
        msg = (
            "a run needs a horizon: pass max_iters (iterations) or "
            "time_limit (seconds); budget_check is asked at the loop head and "
            "before each batch, but a gate that advances only on served batches "
            "spins forever under a stalled flock, and the engine cannot see "
            "which kind it is"
        )
        raise ValueError(msg)

    # Presence of the callables, not their signatures. Without this a mistyped
    # seam reaches the first batch and fails there as `'int' object is not
    # callable`, a symptom a frame away from the argument that caused it.
    if not callable(eval_f_g):
        msg = f"eval_f_g must be callable, got {type(eval_f_g).__name__}"
        raise TypeError(msg)
    for what, fn in (
        ("eval_f", eval_f),
        ("global_model", global_model),
        ("budget_check", budget_check),
    ):
        if fn is not None and not callable(fn):
            msg = f"{what} must be callable or None, got {type(fn).__name__}"
            raise TypeError(msg)
    _check_bounds(bounds)
    _check_flock(x0, bounds)


def _check_flock(x0: jnp.ndarray, bounds: Bounds | None) -> None:
    """Reject a flock the engine cannot place, or that disagrees with the box.

    The flock is the run's only statement of n_paths and d, so a malformed one
    would surface a frame later as a box or batch-shape mismatch. A non-finite
    row is refused rather than left to stall a path: a coordinate is not an
    objective that went non-finite, and no clip or proposal can repair it.
    """
    flock = jnp.asarray(x0)
    if flock.ndim != 2:
        msg = f"x0 must be a (n_paths, d) array, got shape {flock.shape}"
        raise ValueError(msg)
    if flock.shape[0] < 1 or flock.shape[1] < 1:
        msg = f"x0 needs at least one point of at least one dim, got {flock.shape}"
        raise ValueError(msg)
    if not _all_finite(flock):
        msg = "x0 must be finite: a non-finite starting point cannot be placed"
        raise ValueError(msg)
    # The two statements of d have to agree before either is used: x0's second
    # axis and the box's width.
    box = _box_bounds(bounds)
    if box is not None and box[0].shape[0] != flock.shape[1]:
        msg = f"x0 has d={flock.shape[1]} but bounds has d={box[0].shape[0]}"
        raise ValueError(msg)


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
    return _all_finite(lo) and _all_finite(hi)


def _all_finite(a: jnp.ndarray) -> bool:
    """Report whether every component of an array is finite (all-or-nothing)."""
    return bool(jnp.all(jnp.isfinite(a)))


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


def _feas_flags(feas: jnp.ndarray) -> list[bool]:
    """Read the caller's feasibility mask as flags, refusing anything but 0/1.

    Only 0 marks an infeasible point, so the mask is read as a truthiness test
    nowhere: a signed constraint margin arrives looking feasible on every point
    it rules out, and a NaN arrives looking feasible too, both of them silently
    promoting a point into the reported best. No cast repairs that (a NaN casts
    to True), so the batch is refused rather than guessed at.
    """
    if not bool(jnp.all((feas == 0) | (feas == 1))):
        msg = (
            "the feas mask must hold 0/1 (or booleans) - only 0 marks an "
            f"infeasible point, got {feas.tolist()}"
        )
        raise ValueError(msg)
    return [bool(v) for v in feas.tolist()]
