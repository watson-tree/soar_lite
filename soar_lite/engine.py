"""The n-path engine: contracts, shared types, and the schedule.

This module is the seam (拍板) of soar_lite: it owns the two plugin contracts
(`LocalSolver`, `GlobalModel`), the data the plugins exchange
(`PathPoint`, `Verdict`), and the shared plumbing (`NPathResult`, the evaluator
seam, and the `n_path_schedule` loop). The run's state lives in four private
holders, one per state family: `_Evaluator` (the two evaluation seams and what
each one charges), `_Paths` (the n-path state machine), `_Best` (the run's best,
`f⋆`), and `_Budget` (the stop meter and work report). Concretes import *from*
here, never the other way around at module level:

    soar_lite/
      engine.py        <- LocalSolver, GlobalModel, PathPoint, Verdict,
                          n_path_schedule
      domain.py        <- Domain, Bounds (the box, its scales and draws; a leaf)
      global_model/    <- the restart-model family
        sample.py          SampleGlobalModel (implements GlobalModel)
        spherical_linear.py  BLRGlobalModel + the Projections it reads
      local_solver.py  <- DiagonalPreconditioner (implements LocalSolver)

Dependency direction: plugins depend on engine types; the engine binds its two
concrete defaults lazily inside `n_path_schedule` (keeps the module graph acyclic
and `__init__` importable in any order). The engine is first-order (value-and-
gradient), coordinate-free, and owns the flock, the run's domain, the shared
horizon, the clip-to-domain and the restart cadence; the local solver owns only
direction/step size/acceptance, and the global model only where a restart lands.

The restart decision is the engine's own bit, not a third plugin. `_Paths`
marks a path stalled when its method gave up or its point was not usable, and
`restart_stalled` says whether such a path is re-drawn from the global model.
There is deliberately no restart-rule seam: *when* a path has stopped is the
local solver's business (its own stopping test), *where* it restarts is the
model's, and the engine already holds the one flag that joins them.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

import jax
import jax.numpy as jnp

from soar_lite.domain import Domain

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
        non-finite f is not a point the engine can report, re-start from or
        finish on, and a non-finite g is proposed from (`-p.g`) and folded on
        (`best.g - p.g`), so one NaN there poisons that coordinate's magnitude
        EMAs for the rest of the run. The f half is asked first, so a dead value
        short-circuits before the device sync in `all_finite`.
        """
        return math.isfinite(self.f) and all_finite(self.g)


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
class GlobalModel(Protocol):
    """The engine's restart-point source (its second plugin point).

    The model answers two questions the engine cannot: *where* a restart lands
    (`propose`) and *what the run has learned so far* (`observe`). The engine
    feeds it every batch it serves and asks for points only when a restart
    fires - it never seeds the run itself, since the first restart is the
    flock's placement. That split is what lets a caller cap how much evidence
    the model fits: a surrogate that wants more points than it has can answer
    `propose` with random search until its evidence arrives, and the engine has
    no opinion either way, since it only ever sees the points.

    A model carries no search space of its own. The run has one box and the
    engine owns it, so `propose` is handed that `Domain` at every ask and a
    model may not remember it: the box is what the answer must land in, not a
    property of the thing answering. That is what keeps a fitted model's
    geometry and its fallback draw measured in the same coordinate unit as the
    clip, without a second box to reconcile - there is only ever one.

    One instance serves the whole run (like `LocalSolver`), so whatever the
    model accumulates lives inside it; the engine never reads
    it. The methods are called by the engine on its own schedule - `observe`
    once per served batch, `propose` once per cohort of restarting paths - so
    they trust their arguments: `n_path_schedule` validates the object once at
    the entry (that it carries this contract), and nothing here re-checks.

    Implementing this pair is the whole extension point. A surrogate of another
    family - a Gaussian process, say - needs nothing else from the engine, and
    the engine needs nothing else from it; a caller who wants a shared feature
    map reads `global_model.spherical_linear.Projection` from inside its own
    `propose`, which is a utility, not part of this contract.
    """

    def observe(self, x: jnp.ndarray, f: list[float], g: jnp.ndarray | None) -> None:
        """Fold one served batch into the model's evidence.

        Called by the engine immediately after a batch is served, in the
        engine's own coordinates, before anything is decided on it: `x` is the
        batch's rows, `f` their values (Python floats - the same values the run
        compares) and `g` their gradients, or `None` when the batch was served
        for its values alone. A point may arrive twice: a step's trial is
        served for its value first and its gradient later if it wins the step,
        so a model keyed on coordinates sees the second call as the same design
        with more evidence.

        Nothing is filtered for feasibility here: the engine models the raw
        objective everywhere, and only the value it *reports* is filtered, so
        the model fits the same surface the paths move on. Called for batches
        the run never decides on too (a trial that loses its step, the batch
        that ended the run on an unbounded objective), because the work was
        served either way.
        """
        ...

    def propose(self, n_points: int, domain: Domain) -> jnp.ndarray:
        """Return ``n_points`` fresh restart starts, shape ``(n_points, d)``.

        Called with the number of paths that restart this iteration - on
        iteration 0 that is every path, and every path holds no point yet, so
        the first call places the flock - and with the run's own `Domain`,
        which is the box the answer must land in and the one statement of the
        dimension (`domain.d`). Asked in the same coordinates the objective
        takes. The engine still clips what comes back into that box, as the
        defensive seam for a model that oversteps the box it was handed, not as
        a way of reconciling two of them.

        The model owns its randomness (an instance holds its key), which is what
        makes a seeded model reproducible.
        """
        ...


@runtime_checkable
class LocalSolver(Protocol):
    """The search method the n-path engine drives (the engine's plugin point).

    The engine owns the schedule: row composition, projection, the shared
    horizon, the n-path state machine (restart/step/stall), and the best `f⋆`.
    The method owns the search: the direction, the step size, the acceptance
    test, **and how long the enumeration may run** - the engine loops until the
    method ends it, so ``verdict`` must reach "accept" or "give_up" on its own.
    An enumerator with no bound of its own is a hang, not a slow run.
    ``needs_verdict(i, p)`` is the other half of that ownership: a method that
    rules on each candidate says True, one whose step always commits its single
    proposal says False, and the engine then serves no value for a ruling it
    would only rubber-stamp. All state is per path, lives inside the instance,
    and is opaque to the engine (the engine never reads it) - implementers index
    it by path i. Attributes are public so an instance can be inspected and
    subclassed, not because the engine may write them: the engine only hands
    them to the methods below.

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

    def needs_verdict(self, i: int, p: PathPoint) -> bool:
        """Report whether path i's next step from p is ruled on candidate by candidate.

        Asked once per path per iteration, before ``propose``, so the only point
        it can see is the step's start p - whose gradient a path always has. A
        method with a stopping test reads it here: answering True is how it gets
        the one ``verdict`` call that carries "give_up", since a step it did not
        ask to rule is committed as proposed.

        ``True`` is the enumerated kind: the engine serves each trial for its
        value alone and asks ``verdict`` what to do with it. ``False`` promises
        the step commits its one proposal unjudged - the engine leaves both
        halves of the point to `settle`, so its value is served once (fused, with
        the gradient) instead of once for a ruling and again for the gradient.
        What replaces the ruling is the engine's gate: a winner whose value or
        gradient is not finite stalls the path instead of being committed, which
        is the same gate a fresh restart start passes.

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


# --- engine: the n-path schedule ---

# The one way the engine sees the objective: x rows (M,d) in the run's
# coordinates in, (f (M,), grads_x (M,d), feas (M,)|None) out. Value and gradient
# come back together on purpose - the caller's fused value_and_grad is one AD
# pass, and the step's call sites want both at once (scipy's `jac=True` form,
# batched).
EvalFG: TypeAlias = Callable[
    [jnp.ndarray],
    "tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]",
]

# The f-only sibling, for a caller that also holds a value function without the
# fused gradient. Same x rows in, (f (M,), feas (M,)|None) out - feasibility
# included because step selection must still tell "finite but infeasible" from
# viable, which f alone cannot. Opting in lets a judged step serve its trials
# for their values alone and buy gradients only for its winners, in one batch.
EvalF: TypeAlias = Callable[[jnp.ndarray], "tuple[jnp.ndarray, jnp.ndarray | None]"]

# The two halves of a `GlobalModel`, as the components that call them see them.
# The model itself never crosses into `_Evaluator` or `_Paths`: each takes the
# one method it uses, so neither holds a god object whose other half it ignores.
# `ObserveFn` is the command that folds a served batch into the evidence;
# `ProposeFn` is the query for fresh starts, handed the run's box because the
# box is the engine's and a model carries none.
ObserveFn: TypeAlias = Callable[[jnp.ndarray, list[float], jnp.ndarray | None], None]
ProposeFn: TypeAlias = Callable[[int, Domain], jnp.ndarray]

# Four names for four ways to end a run: the three gates, in the order they are
# checked (iteration horizon, wall-clock deadline, the caller's extra batch
# gate), plus the objective bottoming out at -inf. A stalled flock has no fifth
# stop - every report still advances a stage clock, so the verdicts keep coming
# until the horizon or the deadline ends the run.
NPathStatus: TypeAlias = Literal[
    "max_iters",
    "time_limit",
    "budget_exhausted",
    "unbounded",
]


@dataclass
class NPathResult:
    """End state of one n_path_schedule run.

    Field names follow `scipy.optimize.OptimizeResult` where the concepts
    correspond: `x` is the argmin in the coordinates the engine was handed
    (the caller's coords), `fun` is its objective value, and `n_iters` counts
    completed engine iterations: iteration 0 is the flock's placement (the run's
    first restart), and one iteration may serve several batches (a restart's
    fresh starts plus
    up to the method's own per-step trial count), so `n_iters` is not a batch
    count. It is spelled
    `n_iters` rather than scipy's `nit` to sit beside the four work counts.
    Work is reported as four counts instead of scipy's single `nfev`, on two
    independent axes that never mix. The seam that served the work decides the
    pair of fields: `n_f_*` is the value-only seam (`eval_f`), `n_f_g_*` the
    fused value+gradient one, the expensive half. Points and batches are the
    second axis: the serial cost and the
    parallel wall-clock cost. So `n_f_g_points` is the gradient points served and
    `n_f_g_batches` the gradient passes, while `n_f_points` / `n_f_batches` are
    the same two units for the cheap half. Which axis a comparison reads is the
    caller's business, not the engine's: a global-model comparison reads the f
    pair, a descent-cost comparison the f_g pair. Neither is secondary, so a run
    handed no `eval_f` seam simply reports zero on the f pair.

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
    n_f_points: int  # value-only points served (the f seam)
    n_f_batches: int  # value-only batches served (the f seam)
    n_f_g_points: int  # fused points served (the serial-cost count)
    n_f_g_batches: int  # fused batches served (the parallel wall-clock count)
    n_iters: int  # completed iterations (iteration 0 places the flock)
    status: NPathStatus


def n_path_schedule(  # noqa: PLR0913 (the caller's knobs, all keyword-only
    # after the one positional seam: one name per dial beats a config object here)
    eval_f_g: EvalFG,
    *,
    # The run's width: how many chains the engine drives.
    n_paths: int,
    # The run's box, a `Domain`: every proposal is clipped into it - the global
    # model's fresh starts and the local method's steps alike - and it is the
    # one statement of the dimension (`domain.d`) and of each coordinate's
    # prior `scale`. The engine owns it; a model is handed it per ask rather
    # than carrying one, so there is never a second box to reconcile.
    domain: Domain,
    eval_f: EvalF | None = None,
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
    # iteration and again before each served batch, fused or value-only.
    # None means "no extra gate". It is additional, never a horizon: a gate
    # that advances only on served batches spins forever under a stalled flock,
    # and the engine cannot see which kind this is, so a horizon is required.
    budget_check: Callable[[], bool] | None = None,
    # Whether a stalled path is re-drawn. The stall itself is the engine's own
    # bit (`_Paths.stalled`), set when a path's method gives up or its point is
    # not usable; this knob only decides whether the model is then asked for a
    # fresh point. False keeps the flock frozen (`_Paths` never re-seeds it),
    # True is the stall-restart cadence. The default is False, so the flock a
    # caller hands in is the whole run unless it opts in.
    restart_stalled: bool = False,
    # The two plugins, adjacent and in one order everywhere: where a restart
    # lands (the model is also the run's evidence sink, asked for points only
    # when a restart fires; None -> SampleGlobalModel, which draws whatever the
    # `domain` admits - a declared unbounded side is a prior the domain itself
    # already states, not a draw the engine has to invent), and the search
    # method.
    global_model: GlobalModel | None = None,
    local_solver: LocalSolver | None = None,  # None -> DiagonalPreconditioner
    # The random source of the restart points, last and optional (optax's `key`
    # placement): None means the fixed default PRNGKey(0), so an unseeded call
    # still runs.
    key: jnp.ndarray | None = None,
) -> NPathResult:
    """Advance a flock of chains in x from the caller's initial points.

    The engine owns the schedule: it composes and clips every proposal into
    `domain`, serves each batch as one `eval_f_g` call, keeps the n-path state
    machine, and owns the best `f⋆`. Two things are injected, and the engine
    spells them `global_model`, `local_solver` in that one order wherever it
    names them: `global_model` supplies **where** a restart lands (the engine
    asks it for one point per restarting path, clips them into the box and
    places them) and is also the run's evidence sink, and `local_solver` owns
    direction, step size and acceptance and keeps its per-path state in Python
    floats. `restart_stalled` is the third half of a restart, and it is a
    boolean: **when** a path restarts is the local solver's own stopping test
    (the engine reads the stall it produced), so a restart is two seams and one
    flag - a model that proposes points (a surrogate maximising expected
    improvement, an out-of-tree adapter) never touches the verdict, and a run
    that never restarts (`restart_stalled=False`, the default) needs no model
    of its own at all.

    `n_paths` is the run's width and nothing else; the dimension `d` is the
    model's (see below), so a run never spells it twice. Iteration 0 is the
    flock's placement and it is not a special case: no path holds a point yet,
    a path without a point restarts, so the run's first move asks the model for
    one start per path and serves that whole flock as one fused batch. A start
    whose served value or gradient comes back non-finite stalls its path on
    arrival instead of being placed, and - having no point - that path restarts
    again next iteration rather than idling forever.

    With `global_model=None` the engine builds a `SampleGlobalModel` over the
    run's `domain` and lets that draw: uniform inside a bounded dimension, its
    own ``N(0, scale_i^2)`` inside a free one, and that same normal truncated
    inside a half-bounded one. A declared unbounded side is thus a prior the
    domain already states - `scale` is the length the caller gave it, in
    coordinate units - rather than something the engine reads off a width or
    guesses. Only an absent domain is refused, because there is then nothing to
    draw from at all. `key` seeds that default model's draws; None means the
    fixed default PRNGKey(0), so an unseeded call still runs, deterministically.
    An injected model owns its randomness, so `key` is ignored for it.

    `domain` is the run's box and the coordinate space the engine works in, a
    `Domain` the caller built (its constructor refused a malformed one already);
    the engine clips into it and knows nothing about any map from these
    coordinates to the caller's physical ones. It is the model's box too, so an
    injected model - which carries it as `GlobalModel.domain` - makes the
    argument optional, and a *different* box is refused at the entry rather than
    clipped across: two boxes with equal bounds are still two `scale` and
    sampler stories, i.e. two search problems wearing one width.

    Work is reported as four counts, never mixed. The fused value+gradient seam
    is the expensive one, and `n_f_g_points` is the g points it served (the
    initial points, each restart start, and each trial winner - one each), the
    serial-cost number, with `n_f_g_batches` the fused passes, the parallel
    wall-clock number. The value-only seam (`eval_f`, when handed in) is counted
    the same way on its own pair, `n_f_points` / `n_f_batches`, so a run without
    the seam reports zero there. Both seams ask the gate before serving; neither
    pair of counts is what closes it. All four are reports, not caps.

    The run's own stop is the horizon: at most `max_iters` iterations (status
    "max_iters"), or, if `time_limit` is
    given, the wall-clock deadline (status "time_limit"); at least one of the
    two must be passed. Iterations are the unit on purpose - an iteration is one
    move per path (iteration 0 places the flock instead), so the same number
    reproduces on any machine and does not change when the method's per-step
    trial count changes the per-iteration batch count. It is not a cost bound:
    one iteration serves from zero batches (a fully stalled one) up to a
    restart's starts plus however many trials the method's own search serves, so
    migrating a batch budget to `max_iters` can multiply the work done by that
    per-iteration spread. budget_check() is the caller's extra stop - one of the
    three inputs the run's one query (`_Budget.stop`) asks, at the loop head
    every iteration and again before each served batch - for a budget the engine
    cannot see; a False verdict stops all paths globally (status
    "budget_exhausted"), and None means no extra gate; that external budget may
    over-run by <M points (the gate flips only between its asks).
    fun is the true min over every feasible recorded eval with a finite value -
    every served point when the seam reports no aux; with aux, only points whose
    feas mask is True update it. The trajectory itself still moves on the
    penalized objective.

    The caller's inputs are validated here, once per run: knob ranges and a
    model whose `d` disagrees with the run's `domain` raise ValueError, and a
    seam object that does not carry its contract's methods raises TypeError.
    The domain itself refused a malformed box when it was built, so no spelling
    of the box reaches the loop unchecked. The seam methods themselves trust the
    engine - they run per path per iteration, and the engine is their only
    caller.

    Args:
        eval_f_g: the objective as one batched pass. Takes rows ``x`` of
            shape ``(M, d)`` and returns ``(f: (M,), g: (M, d), feas: (M,) |
            None)`` - value, gradient and optional feasibility mask. A mask
            holds 0/1 (or booleans) and nothing else: only 0 marks a point
            infeasible, so any other flag is refused rather than read as
            truthiness.
        n_paths: how many chains to drive; one point per path per iteration. A
            value below 1 is refused at the entry.
        eval_f: the objective's value half alone, same ``x`` rows, returning
            ``(f: (M,), feas: (M,) | None)``. Both seams are named for what they
            return - ``eval_f_g`` hands back f and g together, ``eval_f`` f
            alone - so neither name leans on the unit it is called with.
            ``None`` means no f-only path at all: every served trial takes its
            gradient in the same fused pass. Supplying it lets a judged step
            serve its trials for their values alone and buy gradients only for
            its winners, in one batch.
        domain: the run's box, a `Domain`: every proposal is clipped into it,
            and with ``global_model=None`` it is also what the default model
            draws from. An injected model carries the box itself
            (``GlobalModel.domain``), so this is optional then - and given, it
            must be that same object: a rebuilt box is two `scale`/sampler
            stories behind equal-looking bounds. ``None`` with no model is
            refused, there being nothing to draw from.
        max_iters: the run's horizon in iterations (``None`` leaves it to
            ``time_limit``); a run needs at least one of the two.
        time_limit: a wall-clock deadline in seconds (``None`` means none).
        budget_check: an extra gate for a budget the engine cannot see; one of
            the three inputs the run's one stop query asks, at the loop head
            every iteration and again before each served batch. Additional, not
            a horizon (``None`` means none).
        global_model: the restart-point source and the run's evidence sink: a
            `GlobalModel` that is fed every served batch and asked for points
            only when a restart fires. It also carries the run's box
            (``GlobalModel.domain``), which is why ``domain`` is optional beside
            it and cannot be a different one. ``None`` builds a
            `SampleGlobalModel` over ``domain`` - bounded, free or half-bounded,
            wherever the domain's own draw reaches - which fits nothing.
            A model's ``d`` and its ``domain.d`` are the run's dimension, and the
            two are checked against each other.
        local_solver: the local method (``None`` binds ``DiagonalPreconditioner``).
        restart_stalled: whether a path the engine marks stalled (its method gave
            up, or its point was not usable) is re-drawn from the global model.
            ``False`` keeps the flock frozen.
        key: the PRNG seed of the default model's draws; ``None`` uses the
            fixed default, so unseeded runs are deterministic.

    Returns:
        An ``NPathResult`` with the run's end state:

        - ``x``: the argmin in the caller's coordinates
        - ``fun``: the best value (min feasible eval; ``inf`` if none, ``-inf``
          when the objective has no floor, which also ends the run)
        - ``n_f_points``: value-only points served (the f seam)
        - ``n_f_batches``: value-only batches served (the f seam)
        - ``n_f_g_points``: fused points served (the serial-cost count)
        - ``n_f_g_batches``: fused batches served (the parallel wall-clock count)
        - ``n_iters``: completed iterations (iteration 0 places the flock)
        - ``status``: why the run stopped

    Example:
        Minimize ``f(x) = ‖x - 3‖²`` by handing the engine a batched objective:

            >>> import jax
            >>> import jax.numpy as jnp
            >>> from soar_lite import Domain, SampleGlobalModel, n_path_schedule

            >>> def eval_f_g(x):
            ...     g = 2.0 * (x - 3.0)
            ...     return (jnp.sum((x - 3.0) ** 2, axis=1), g, None)

            >>> box = Domain([(-5.0, 5.0), (-5.0, 5.0)])
            >>> result = n_path_schedule(
            ...     eval_f_g,
            ...     n_paths=4,
            ...     domain=box,
            ...     # the flock rides the default model's first proposal
            ...     global_model=SampleGlobalModel(
            ...         key=jax.random.PRNGKey(0),
            ...         x0=jnp.array(
            ...             [[-1.0, -1.0], [1.0, 1.0], [-4.0, 4.0], [4.0, -4.0]],
            ...         ),
            ...     ),
            ...     max_iters=30,
            ... )

            >>> result.x          # ≈ (3, 3)
            Array([3., 3.], dtype=float32)

            >>> result.fun        # its objective value
            0.0

    """
    _validate_run(
        n_paths=n_paths,
        max_iters=max_iters,
        time_limit=time_limit,
        eval_f_g=eval_f_g,
        eval_f=eval_f,
        budget_check=budget_check,
    )
    # The seams: presence, not signatures, and the defaults, bound lazily -
    # the local solver imports the engine's types, so importing it at module
    # scope would form a cycle. Kept with the knob checks above and apart from
    # the build below.
    _check_seam(global_model, GlobalModel, "global_model")
    _check_seam(local_solver, LocalSolver, "local_solver")

    if global_model is None:
        from soar_lite.global_model import SampleGlobalModel  # noqa: PLC0415

        global_model = SampleGlobalModel(
            key=jax.random.PRNGKey(0) if key is None else key,
        )
    if local_solver is None:
        from soar_lite.local_solver import DiagonalPreconditioner  # noqa: PLC0415

        local_solver = DiagonalPreconditioner()

    # The run's one box, and the only place it is read for the clip; the
    # dimension comes with it.
    d = domain.d
    budget = _Budget(max_iters, time_limit, budget_check)
    best = _Best(d)
    # Each component gets the one half of the model it calls, not the object: an
    # evaluator folds evidence, a path stepper asks for starts, and neither has
    # any business holding the other's method.
    evaluator = _Evaluator(
        eval_f_g,
        eval_f,
        budget=budget,
        best=best,
        observe=global_model.observe,
    )
    # Last, because it takes the resolved seams and their collaborators: the
    # state machine drives them rather than receiving them per call.
    paths = _Paths(
        n_paths,
        domain,
        evaluator,
        global_model.propose,
        local_solver,
        restart_stalled=restart_stalled,
    )

    status: NPathStatus | None = None
    while True:
        # The run's one stop query names its reason, asked before a new iteration
        # starts so the run never begins one it cannot finish. A refusal
        # mid-iteration arrives as _BudgetExhausted carrying the same reason.
        status = budget.stop()
        if status is not None:
            break
        try:
            # Decide which paths restart (iteration 0: all of them, since no
            # path holds a point), decide every move on values, then buy the
            # gradients those decisions need in one batch. A refused batch
            # raises `_BudgetExhausted` here, before anything is placed.
            paths.move()
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
        n_f_points=budget.n_f_points,
        n_f_batches=budget.n_f_batches,
        n_f_g_points=budget.n_f_g_points,
        n_f_g_batches=budget.n_f_g_batches,
        n_iters=budget.n_iters,
        status=status,
    )


def pointwise(
    objective: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    feasible: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> tuple[EvalFG, EvalF]:
    """Build both eval seams from one per-point objective.

    `objective` is a scalar function of a single point, ``(d,) -> ()`` - the
    shape `jax.value_and_grad` differentiates - and `feasible` is the per-point
    predicate that marks a point viable, ``(d,) -> ()`` of a boolean. Both are
    vmapped here, so the caller writes neither `jax.vmap` nor
    `jax.value_and_grad`. The fused gradient is the per-row one,
    ``vmap(value_and_grad(objective))``, not `value_and_grad` of a summed
    batch, which would couple the rows through the sum.

    Feasibility is vmapped on its own because it cannot ride inside
    `value_and_grad`: that transformer differentiates a scalar output, so a
    function returning ``(f, feas)`` would be differentiated as a pair. Keeping
    `feasible` a separate point predicate leaves both halves scalar.

    The pair comes back fused-first, matching `n_path_schedule`'s argument
    order, and both halves are lazy closures: building them runs nothing, so a
    caller who wants only the fused seam discards the other with
    ``eval_f_g, _ = pointwise(objective)``.

    The f-only half is the fused objective's own forward pass,
    ``vmap(objective)``, not a second model of it. Returning it therefore costs
    nothing and buys the engine's cheap-trial path: a judged step serves its
    trials for values alone and buys gradients only for the winners. That does
    not make this the seam for a *value-only* objective: a function JAX cannot
    differentiate cannot be `objective` here at all, so a caller who holds
    values and no gradient needs an f-only adapter of its own.

    Args:
        objective: the per-point objective, ``(d,) -> ()``; a function
            `jax.value_and_grad` can differentiate, which is what makes the
            fused half exist.
        feasible: the per-point feasibility predicate, ``(d,) -> ()`` of a
            boolean; None (the default) marks every point viable, the same
            `None` both seams accept.

    Returns:
        ``(eval_f_g, eval_f)``: the two seam shapes `n_path_schedule` takes,
        a fused ``(f, g, feas)`` batch and its value-only ``(f, feas)`` sibling.

    """

    def feas_at(x: jnp.ndarray) -> jnp.ndarray | None:
        return None if feasible is None else jax.vmap(feasible)(x)

    def eval_f(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        return jax.vmap(objective)(x), feas_at(x)

    def eval_f_g(
        x: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
        f, g = jax.vmap(jax.value_and_grad(objective))(x)
        return f, g, feas_at(x)

    return eval_f_g, eval_f


class _BudgetExhausted(Exception):  # noqa: N818 - a signal, not an error
    """A batch was requested past a closed gate, the reason carried along.

    Raised only inside `n_path_schedule`, at a serve site (`_Evaluator._serve_f_g`
    or `_serve_f`), and caught by the loop. It carries the reason
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


class _Budget:
    """The run's stop meter and work report.

    Three predicates decide whether work may go on: the iteration horizon
    (`max_iters`), the wall-clock deadline (`time_limit`), and the caller's
    extra gate (`budget_check`). `stop` asks all three as one query - at the
    loop head every iteration, and again before each served batch - and names the
    reason it refuses. At least one of the first two is required at entry,
    because they are the only stops the engine can see advance on its own: a
    `budget_check` that flips only on served batches spins forever under a
    stalled flock, and the engine cannot tell it from one that flips alone.

    The four counters are pure reports, not caps, and not one of them gates: the
    fused pass counts on its pair (`n_f_g_points` / `n_f_g_batches`, one batch per
    value+gradient pass) and the value-only pass counts the same way on its own
    (`n_f_points` / `n_f_batches`). What the counts never do is decide: `stop`
    reads only the run's three predicates, so a value-only batch is counted here
    and can still be refused by the gate like any other serve. The refusal is a
    signal (`_BudgetExhausted`), never a flag stored here, and a gate that only
    tightens (a counter, a clock) needs no latching: the run ends on the first
    thing it refuses.
    """

    def __init__(
        self,
        max_iters: int | None,
        time_limit: float | None,
        budget_check: Callable[[], bool] | None,
    ) -> None:
        self.max_iters = max_iters
        self.deadline = None if time_limit is None else time.monotonic() + time_limit
        self.n_f_points = 0
        self.n_f_batches = 0
        self.n_f_g_points = 0
        self.n_f_g_batches = 0
        self.n_iters = 0
        self._gate = budget_check

    def charge_f(self, batch_size: int) -> None:
        """Record one served value-only batch. Counted on the f pair; never gates."""
        self.n_f_batches += 1
        self.n_f_points += batch_size

    def charge_f_g(self, batch_size: int) -> None:
        """Record one served fused batch. Counted on the f_g pair; never gates."""
        self.n_f_g_batches += 1
        self.n_f_g_points += batch_size

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

        Asked at the loop head before every iteration, and again before each
        served batch (fused or value-only), where a stop becomes the
        `_BudgetExhausted` signal. None while work may go on; otherwise the first
        closed gate, named - the horizon outranks the clock, which outranks the
        caller's gate.
        """
        if self.full():
            return "max_iters"
        if self.expired():
            return "time_limit"
        if self.gated():
            return "budget_exhausted"
        return None


class _Best:
    """The best the run keeps: the minimum over feasible points served.

    `fun`/`x` are the answer: the minimum over feasible points served, which is
    what the run reports. Defined over feasible designs only - with no aux,
    every point is feasible. A NaN or +inf trial neither becomes `f⋆` nor
    displaces one: it carries no value to compare. A feasible -inf is not an
    improvement to weigh but the answer itself - the objective has no floor
    there, so no later trial can beat it and no step can improve on it, which
    is why it ends the run (`unbounded`) instead of entering a comparison. It
    starts at inf: until something feasible arrives there is no answer to
    report.
    """

    def __init__(self, d: int) -> None:
        # d is the flock's business (it is `x0`'s second axis), so the
        # placeholder is already the right shape.
        self.x = jnp.full(d, math.nan)
        self.fun = math.inf
        # Set when a feasible point priced the objective at -inf: the run has an
        # answer that no later trial can beat, so it stops on it.
        self.unbounded = False

    def observe(
        self,
        x_batch: jnp.ndarray,
        f_vals: list[float],
        feas: jnp.ndarray | None,
    ) -> None:
        """Fold one served batch's values into the best, stopping on -inf.

        Raises:
            ValueError: If a feasibility mask holds anything but 0/1. Nothing
                is folded in that case, so a batch's values are never added
                under a mask that cannot be read.

        """
        # The mask is read first, and read as flags: it is the caller's claim, so
        # a batch whose mask cannot be read is refused before either best moves.
        # With no aux a batch has no mask and every point counts.
        ok_feas = [True] * len(f_vals) if feas is None else _feas_flags(feas)
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
    it: every batch asks the gate before it is served (so a batch is served
    whole or not at all), and the seam that served it decides which pair of
    counters moves - `_serve_f` the f pair, `_serve_f_g` the fused pair. No count
    is ever what closes the gate. Whichever half
    served a batch reports it to the two readers of the run's evidence: the best
    (`_Best`, which filters for feasibility) and the global model (which does
    not). That pair is deliberately the only place either is fed - a batch is
    reported once, by whichever seam served it, so no caller has to remember to
    feed either, and a batch nobody decides on is still reported. `settle` lives
    here too: it is where an iteration's deferred gradients are bought together,
    since this object owns both the seam and the charge.
    """

    def __init__(
        self,
        eval_f_g: EvalFG,
        eval_f: EvalF | None,
        *,
        budget: _Budget,
        best: _Best,
        observe: ObserveFn,
    ) -> None:
        self._eval_f_g = eval_f_g
        self._eval_f = eval_f
        self._budget = budget
        self._best = best
        self._observe = observe

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
        ever evaluated - and the fused counters only ever hold batches that were
        served. This is the expensive half: a step's rejected trials are f-only
        and cheaper, served by `_serve_f`, which asks the same gate and counts on
        the other pair. The refusal is a
        signal, not a state: it carries its reason straight to the run's one
        catch, so no intermediate frame has to check it. Returns (f_vals,
        g_batch).
        """
        reason = self._budget.stop()
        if reason is not None:
            raise _BudgetExhausted(reason)
        self._budget.charge_f_g(x_batch.shape[0])
        f_batch, g_batch, feas = self._eval_f_g(x_batch)
        # per-path bookkeeping (secant/restart) stays Python float,
        # so pull the batch values onto the host in one shot.
        f_vals = f_batch.tolist()
        # The model first: a feasible -inf below raises `_OptimumReached`, and
        # the batch was served and charged either way, so the model still sees
        # it even though the run keeps nothing else from this pass.
        self._observe(x_batch, f_vals, g_batch)
        self._best.observe(x_batch, f_vals, feas)
        return f_vals, g_batch

    def _serve_f(self, x_batch: jnp.ndarray) -> list[float]:
        """Serve one f-only batch: the same gate, the other pair of counters.

        The gate is asked here exactly as the fused half asks it, so a deadline
        that passes mid-step cuts the run at the next value-only batch rather
        than after a whole iteration's retries: the judged lane's `while
        rejecting` loop can turn many times without ever reaching the loop head
        or `settle`. What stays asymmetric is the charge - this batch counts on
        the f pair (`charge_f`), and no count closes the gate. Only
        `serve_trials` calls this; the None-check below is for the type checker
        (it cannot see the caller is f-only) and is unreachable at runtime.
        """
        eval_f = self._eval_f
        if eval_f is None:
            msg = "_serve_f requires eval_f"
            raise RuntimeError(msg)
        reason = self._budget.stop()
        if reason is not None:
            raise _BudgetExhausted(reason)
        self._budget.charge_f(x_batch.shape[0])
        f_batch, feas = eval_f(x_batch)
        f_vals = f_batch.tolist()
        # No gradient in hand: the f half alone. The same point is reported
        # again with `g` if it wins its step and `settle` buys the gradient.
        self._observe(x_batch, f_vals, None)
        self._best.observe(x_batch, f_vals, feas)
        return f_vals


class _Paths:
    """The n-path state machine: one point and one stall flag per path.

    The invariants that the loop's four sites would otherwise carry implicitly,
    stated once here instead:

    - A path that restarted this iteration moved by its fresh start, so it does
      not also step, and it runs no trials: its method sees no step, so the
      restart is never read back as a zero-length step. `move` computes the
      restarted list once per iteration and threads it to `_restart_starts` and
      to the complement both halves share - one fact at the top instead of a
      stored flag that must be cleared each iteration.
    - An iteration decides on values and pays for gradients once: whatever the
      f-only seam left ungraded (restart starts and step winners alike) is
      bought
      in one batch before anything is placed, so a decision is never priced
      before it is made.
    - A path is stalled when it did not move at all this iteration - its method
      gave up on every trial, or its fresh start or its winner was not a usable
      point. A stalled path does not step, so it serves nothing that iteration:
      its method's stage clock therefore keeps pace with its live cohort.
    - A start or a winner is committed only when it is `usable` - finite in
      both halves - one gate defined once on the point type.
    - A path holds a point or it does not. "No point" (an empty `x`, which only
      the placeholder has) is the machine's own state, not a stall: such a path
      restarts without consulting `restart_stalled`, which is how iteration 0
      places the flock and how a refused start is eventually re-drawn instead of
      idling forever. A held point with a non-finite value is the other case -
      a stall, so `restart_stalled` decides whether it is re-drawn.
    """

    def __init__(  # noqa: PLR0913 - the run's collaborators, built once
        self,
        n_paths: int,
        domain: Domain,
        evaluator: _Evaluator,
        propose: ProposeFn,
        local_solver: LocalSolver,
        *,
        restart_stalled: bool,
    ) -> None:
        self.n_paths = n_paths
        # Constant for the whole run, so it belongs beside the shape rather than
        # in `move`'s signature: every proposal below is clipped into this box,
        # and this is the only copy of it - the model is handed it per ask
        # rather than carrying one.
        self.domain = domain
        # The collaborators this state machine drives. Held rather than handed
        # in per call: every method below uses at least one of them, and a step
        # would otherwise thread them only to pass them straight back.
        self.evaluator = evaluator
        self.propose = propose
        self.local_solver = local_solver
        # A knob, not a collaborator: *when* a path restarts is read off
        # `stalled` below, so the only thing left to decide is whether the
        # engine acts on it at all.
        self.restart_stalled = restart_stalled
        # A path's point is one object, not three parallel lists: it is what
        # every seam call takes, and a step commits one path at a time (so a
        # list, whose item assignment is the commit, beats a stacked array). The
        # placeholder says "no point has arrived" and is the only empty `x` in
        # the run: a usable restart start replaces it, an unusable one never
        # becomes a point, and a stalled path never serves or compares it.
        self.points = [
            PathPoint(jnp.zeros(0), math.nan, jnp.zeros(0)) for _ in range(self.n_paths)
        ]
        # Nothing has arrived yet, and a path that has not arrived is not
        # stalled: iteration 0 restarts it rather than reading it as a stall.
        # This is the run's one restart bit, and `_place` is its only writer of
        # `False`: a path is stalled until it is placed on a usable point, so a
        # path that is re-drawn cannot stay stalled (which a second copy of the
        # bit, owned elsewhere, could not guarantee).
        self.stalled: list[bool] = [False] * self.n_paths

    def move(self) -> None:
        """Take one iteration's move for every path, then settle what it deferred.

        Iteration 0 is not a special case: no path holds a point, and a path
        with no point restarts (see the class docstring), so the run's first
        move asks the global model for one start per path - the flock - and
        places them in one batch.

        The restart decision comes first: a path that holds no point, or that is
        stalled and whose run opted into `restart_stalled`, is given a fresh
        start instead of stepping this iteration. Both halves then decide where a
        path goes from values alone when the f-only seam is present - the
        restarted paths by the global model's proposal, the cohort by its trials
        - and the gradients those
        decisions need are then bought together, in one batch (`settle`), so an
        iteration costs one gradient pass however many moves it deferred. What
        is not deferred is served fused and reaches `settle` already graded:
        every point of a run without the seam.

        The two halves touch disjoint paths - a restarted path is in neither -
        and every call into a collaborator here is per path, so deciding all
        moves before placing any of them is the same schedule as placing each
        move as it is decided.
        """
        # "No point" is the machine's own state, never a stall to weigh, so it
        # is read off the point's shape and not off its value: only the
        # placeholder has an empty `x`, and only a path that holds a point can
        # be read as stalled. A value test would conflate the two - a held point
        # whose f is not finite is a stall (`_place`/`_advance` refuse it and set
        # `stalled`), which `restart_stalled` owns, not an owed start.
        restarted = [
            i
            for i in range(self.n_paths)
            if self.points[i].x.size == 0 or (self.restart_stalled and self.stalled[i])
        ]
        # Every path that did not restart. A stalled path is in here only when
        # the run never restarts: it proposes nothing, but it still holds its
        # place in the cohort so its point stays readable.
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

    def _restart_starts(self, n_points: int) -> list[_Pending]:
        """Ask the model for n_points fresh starts, clipped into the run's box.

        Ask, check, clip, place: the model is handed the box it must land in and
        its answer is refused unless it is ``(n_points, domain.d)`` - the engine
        owns the width now, so a model that returns another one fails here,
        naming the shape, instead of deep in a broadcast during the clip. The
        clip itself stays as the defensive seam for a model that oversteps the
        box - the same one every step proposal passes. Both halves stay owed - a
        start has nothing to be judged on - so `settle` buys them in the same
        batch as the step winners, which is what keeps a restart from costing
        its own gradient pass.
        """
        d = self.domain.d
        proposed = self.propose(n_points, self.domain)
        if proposed.shape != (n_points, d):
            msg = (
                f"the global model must propose a ({n_points}, {d}) array, "
                f"got {proposed.shape}"
            )
            raise ValueError(msg)
        x_cand = jnp.clip(proposed, self.domain.lo, self.domain.hi)
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
        lo, hi = self.domain.lo, self.domain.hi
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
            # the run's `domain`, so the caller's map never sees a point outside
            # the box the engine advertised. The clipped step is the actual one,
            # so the verdict is handed the served point.
            raw = jnp.stack(
                [self.local_solver.propose(i, self.points[i]) for i in rejecting],
            )
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
                    trial.g is None or all_finite(trial.g)
                )
                if finite and (i not in best or f_trial < best[i].value()):
                    best[i] = trial
                verdict = self.local_solver.verdict(i, point, trial.x, f_trial)
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
            if self.stalled[i]:
                continue
            ruled = self.local_solver.needs_verdict(i, self.points[i])
            (judged if ruled else unjudged).append(i)
        return unjudged, judged

    def _propose_unjudged(self, paths: list[int]) -> dict[int, _Pending]:
        """Take each unjudged path's one proposal as its step's winner, unread.

        Proposals are still clipped into the box: the engine owns that, and the
        served point is the clipped one. Both halves are left owed, which is what
        makes the iteration serve this point's value once rather than twice.
        """
        raw = jnp.stack([self.local_solver.propose(i, self.points[i]) for i in paths])
        clipped = jnp.clip(raw, self.domain.lo, self.domain.hi)
        return dict(zip(paths, self.evaluator.pending_unread(clipped), strict=True))

    def _place(self, placed: dict[int, PathPoint]) -> None:
        """Place the given paths on their fresh starts (one restart event).

        The method's per-path state resets to cold start. A start the engine
        cannot use is refused on the same `usable` gate `_advance` applies to a
        winner: it is not placed, the local solver is not seeded from it, and
        the path stays stalled until it is re-drawn. Refusing rather than
        placing it keeps `points[i]` meaning "where the path stands": an
        unusable start never becomes one. The start is this iteration's move for
        every restarted path, usable or not, so none of them also steps.
        """
        for i, point in placed.items():
            self.stalled[i] = not point.usable
            if self.stalled[i]:
                continue  # unusable: no point, no seed (see `usable`)
            self.points[i] = point
            self.local_solver.restart(i, point)

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
            self.local_solver.fold(i, point, best)


def _check_seam(obj: object | None, proto: type, what: str) -> None:
    """Reject a seam object that does not carry the Protocol's methods.

    Presence only, never signatures - a wrong-signature method still fails at
    its own call site. Static typing already knows the type, so this is the
    runtime catch for a caller who hands in anything at all; `None` passes, as
    `n_path_schedule` defaults it right after the check. Deliberately in its own
    function: the isinstance is redundant to a type checker and needs saying once.
    """
    if obj is not None and not isinstance(obj, proto):
        msg = f"{what} does not implement {proto.__name__} (a member is missing)"
        raise TypeError(msg)


def _validate_run(  # noqa: PLR0913 - the entry's own knobs, named once
    *,
    n_paths: int,
    max_iters: int | None,
    time_limit: float | None,
    eval_f_g: EvalFG,
    eval_f: EvalF | None,
    budget_check: Callable[[], bool] | None,
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

    _check_positive_int(n_paths, "n_paths")

    # Presence of the callables, not their signatures. Without this a mistyped
    # seam reaches the first batch and fails there as `'int' object is not
    # callable`, a symptom a frame away from the argument that caused it.
    if not callable(eval_f_g):
        msg = f"eval_f_g must be callable, got {type(eval_f_g).__name__}"
        raise TypeError(msg)
    for what, fn in (("eval_f", eval_f), ("budget_check", budget_check)):
        if fn is not None and not callable(fn):
            msg = f"{what} must be callable or None, got {type(fn).__name__}"
            raise TypeError(msg)


def all_finite(a: jnp.ndarray) -> bool:
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


def _check_positive_int(value: object, name: str) -> int:
    """Return `value` if it is a positive int, refusing bools and the rest.

    Takes `object` on purpose: the two callers hand in what an injected object
    carries, so the check is the only thing that knows the answer.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{name} must be a positive int, got {value!r}"
        raise ValueError(msg)
    return value
