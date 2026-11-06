"""The n-path engine: contracts, shared types, and the schedule.

This module owns the two plugin contracts (`LocalSolver`, `GlobalModel`), the
data they exchange (`PathPoint`, `Verdict`), and the shared plumbing
(`NPathResult`, the evaluation seams, the `n_path_schedule` loop). A run's state
lives in four private holders, one per state family: `_Evaluator` (the two seams
and what each charges), `_Paths` (the n-path state machine), `_Best` (the run's
best, `f⋆`), and `_Budget` (the stop meter and the work report).

The dependency direction is one-way - plugins import these types, never the
reverse:

    engine.py         <- LocalSolver, GlobalModel, PathPoint, Verdict
    domain.py         <- Domain, Bounds (the box; a leaf)
    local_solver.py   <- DiagonalPreconditioner (implements LocalSolver)
    global_model/     <- SampleGlobalModel, BLRGlobalModel and its projections

The two concrete defaults are bound lazily inside `n_path_schedule`, which keeps
that graph acyclic. The engine is first-order (value and gradient),
coordinate-free, and owns the flock, the domain and its clip, the shared horizon
and the restart cadence; the local solver owns direction, step size and
acceptance, and the global model only where a restart lands.

Restarting is two seams and one flag, not a third plugin: `_Paths` marks a path
stalled when its method gave up or its point was not usable, `restart_stalled`
says whether such a path is re-drawn, and `GlobalModel.propose` says where.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import jax
import jax.numpy as jnp

from soar_lite.domain import Domain

Verdict: TypeAlias = Literal["retry", "accept", "give_up"]
"""What a judged step decides about one candidate.

``"retry"`` serves the candidate but commits nothing yet, ``"accept"`` commits
it as the path's point, ``"give_up"`` abandons the path's enumeration outright.
See `LocalSolver.verdict`, which returns it.
"""


@dataclass(frozen=True)
class PathPoint:
    """One path's point: coords x, value f, gradient g = ∂f/∂x."""

    x: jnp.ndarray
    """Coordinates, shape ``(d,)``, in the run's coordinates."""

    f: float
    """The objective's value there: finite, since an unusable point is never held."""

    g: jnp.ndarray
    """The gradient there, shape ``(d,)``; finite for the same reason."""

    @property
    def usable(self) -> bool:
        """Return whether this point may be committed as a path's point.

        Both halves must be finite, because both are carried forward: a
        non-finite f cannot be reported, re-started from or finished on, and a
        non-finite g is proposed from and folded on, so one NaN there poisons
        that coordinate's magnitude EMAs for the rest of the run. f is asked
        first, so a dead value short-circuits the device sync in `_all_finite`.
        """
        return math.isfinite(self.f) and _all_finite(self.g)


@dataclass(frozen=True)
class _Pending:
    """A produced point that `settle` has not promoted to a `PathPoint` yet.

    Producing a point is one decision and grading it is one batch, so a point
    travels between them as this: each half is either served or still owed.
    Everything produced goes through here, and `settle` is the one site that
    promotes it, which keeps "does it still need g" a question about the point
    rather than about which producer made it.

    `g is not None` implies `f is not None`: a fused pass serves both, and only
    a value-only trial arrives with `f` alone, its gradient waiting for
    `settle`.
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


class GlobalModel(Protocol):
    """The engine's restart-point source (its second plugin point).

    The model answers the two questions the engine cannot: *where* a restart
    lands (`propose`) and *what the run has learned* (`observe`). The engine
    feeds it every batch it serves and asks for points only when a restart
    fires, so it never seeds a run itself: a model that wants more evidence than
    it has can answer `propose` with a plain draw and the engine has no opinion.

    A model carries no search space of its own. The run has one box, the engine
    owns it, and `propose` is handed that `Domain` at every ask - so a model must
    not remember one. That is what keeps a fitted geometry and a fallback draw
    in the same coordinate unit as the clip, with no second box to reconcile.

    One instance serves the whole run (like `LocalSolver`), so whatever it
    accumulates lives inside it and the engine never reads it. The methods are
    called by the engine on its own schedule - `observe` once per served batch,
    `propose` once per cohort of restarting paths - so they trust their
    arguments: this Protocol is the whole contract.

    The flock is not this contract's: a caller's own starts go to
    `n_path_schedule` as `x0`, and the engine places them without asking any
    model. `propose` is asked only for restarts, and on iteration 0 only when no
    `x0` was handed in.

    Implementing this pair is the whole extension point; a projection is a
    utility a `propose` may read, not part of the contract.
    """

    def observe(self, x: jnp.ndarray, f: list[float], g: jnp.ndarray | None) -> None:
        """Fold one served batch into the model's evidence.

        Called right after a batch is served, in the engine's coordinates and
        before anything is decided on it: `x` is the batch's rows, `f` their
        values (Python floats - the same values the run compares) and `g` their
        gradients, or `None` for a batch served for its values alone. A point
        may arrive twice - a step's trial is served for its value first and for
        its gradient later if it wins - so a model keyed on coordinates sees the
        same design with more evidence.

        Nothing is filtered for feasibility: the engine models the raw objective
        everywhere and filters only the value it reports. Batches the run never
        decides on arrive too (a trial that lost its step, the batch that ended
        an unbounded run), because the work was served.
        """
        ...

    def propose(self, n_points: int, domain: Domain) -> jnp.ndarray:
        """Return ``n_points`` fresh restart starts, shape ``(n_points, d)``.

        `n_points` is how many paths restart this iteration - on iteration 0,
        every path when the caller handed no `x0` - and `domain` is the box the
        answer must land in and the one statement of the dimension. The answer
        is in the coordinates the objective takes. The engine still clips it into
        that box, as the defensive seam for a model that oversteps, not as a way
        of reconciling two boxes.

        The model owns its randomness (an instance holds its key), which is what
        makes a seeded model reproducible.
        """
        ...


class LocalSolver(Protocol):
    """The search method the n-path engine drives (its first plugin point).

    The engine owns the schedule - row composition, projection, the shared
    horizon, the n-path state machine (restart/step/stall), the best `f⋆` - and
    the method owns the search: direction, step size, acceptance, **and how long
    a step's enumeration may run**. The engine loops until the method ends it, so
    ``verdict`` must reach "accept" or "give_up" on its own; an enumerator with
    no bound of its own is a hang, not a slow run. ``needs_verdict(i, p)`` picks
    the step's shape: a method that rules on each candidate says True, one whose
    step always commits its single proposal says False, and the engine then
    serves no value for a ruling it would only rubber-stamp. All state is per
    path, lives inside the instance, and is opaque to the engine - implementers
    index it by path i.

    One step for a path i: propose(i, p) is called repeatedly, with p the path's
    current point, unchanged for the whole step (the engine commits a point only
    on "accept") and therefore also the step's start. The candidate the engine
    serves is the clipped one, so a method judges the point it is handed, not
    its own proposal. verdict(i, p, x, f) is then called once per served
    candidate, with (x, f) the served point's coords and value - never its
    gradient: the engine may serve a candidate's value alone and buy the
    gradient only for the step's winner, so a ruling rests on f and on p, whose
    gradient a start always has. "accept" commits the lowest-f finite trial
    served this step - the engine's per-step argmin - and calls ``fold(i, p,
    best)`` with the committed winner, which is the method's only place to fold
    an accepted step into its model (the engine is the sole owner of "who won",
    so the method never tracks its own argmin; two copies would drift). "retry"
    spends another trial, "give_up" stalls the path at once.

    Arbitration is split: ``verdict`` rules the enumeration and may only mutate
    enumeration-local state (a step length, a trial counter), while ``fold``
    alone mutates the model, so a method folds exactly once per committed step.

    These methods are called by the engine, once per path per iteration, so they
    trust their arguments: this Protocol is the whole contract.
    """

    def restart(self, i: int, p: PathPoint) -> None:
        """Seed path i's state from a fresh start point (its one cold start).

        Called at initialization and again after every restart: the engine has
        one way to place a path on a point, so a method need not tell "first"
        from "again" - both drop its per-path state to the cold one.
        """
        ...

    def needs_verdict(self, i: int, p: PathPoint) -> bool:
        """Report whether path i's next step from p is ruled candidate by candidate.

        Asked once per path per iteration, before ``propose``, so the only point
        it can see is the step's start p - whose gradient a path always has. It
        is also where a method with a stopping test says so: answering True is
        how it gets the one ``verdict`` call that can carry "give_up", since a
        step it did not ask to rule is committed as proposed.

        ``True`` is the enumerated kind: the engine serves each trial for its
        value alone and asks ``verdict`` what to do with it. ``False`` promises
        the step commits its one proposal unjudged: the engine serves that
        proposal's value fused with its gradient, in one batch, and gates the
        winner itself - a value or gradient that is not finite stalls the path
        instead of being committed. Nothing here grades the proposal; the gate
        is the whole of its acceptance.

        ``True`` costs a gradient *per trial* where ``False`` costs one per step,
        unless the caller handed an `eval_f` seam (without it every served trial
        is a fused pass). How many trials a step runs is the method's business,
        so that billing is its choice too.
        """
        ...

    def propose(self, i: int, p: PathPoint) -> jnp.ndarray:
        """Return path i's raw next candidate from its current point p."""
        ...

    def verdict(self, i: int, p: PathPoint, x: jnp.ndarray, f: float) -> Verdict:
        """Rule the step's enumeration from the served trial's coords and value.

        `x` is the served (clipped) point and `f` its value; the trial's own
        gradient is deliberately absent, because the engine may have served the
        value alone and left the gradient to be bought for the winner. `p` is
        the step's start, whose gradient a path always has. Only asked for a
        path that answered True to ``needs_verdict``, so a step that commits its
        one proposal unjudged never reaches here.

        The return is an enumeration decision, not a verdict on the trial's
        quality: "retry" asks for another candidate (the method typically
        shrinks a step length or bumps a trial counter first); "accept" ends the
        enumeration and commits the engine's per-step argmin (this trial, for a
        single-sample method), which is then folded through ``fold``;
        "give_up" aborts the path's enumeration outright. The engine keeps
        enumerating while this says "retry", so one of the two terminal answers
        is the method's responsibility - there is no engine-side trial cap.
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

EvalFG: TypeAlias = Callable[
    [jnp.ndarray],
    "tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]",
]
"""The objective as one fused pass: the one way the engine sees it.

Takes ``x`` rows ``(M, d)`` in the run's coordinates and returns
``(f: (M,), grad_x: (M, d), feas: (M,) | None)``. Value and gradient come back
together on purpose - the caller's fused ``value_and_grad`` is one AD pass, and
the step's call sites want both at once (scipy's ``jac=True`` form, batched).

``feas`` is optional: ``None`` means every point is feasible, otherwise only 0
marks one infeasible.
"""

EvalF: TypeAlias = Callable[[jnp.ndarray], "tuple[jnp.ndarray, jnp.ndarray | None]"]
"""The objective's value half alone, for a caller that also holds one.

Same ``x`` rows in, ``(f: (M,), feas: (M,) | None)`` out. Feasibility is
included because step selection must still tell "finite but infeasible" from
viable, which f alone cannot. Opting in lets a judged step serve its trials for
their values alone and buy gradients only for its winners, in one batch.
"""

# The two halves of a `GlobalModel`, as the components that call them see them.
# The model itself never crosses into `_Evaluator` or `_Paths`: each takes the
# one method it uses, so neither holds a god object whose other half it ignores.
# `ObserveFn` is the command that folds a served batch into the evidence;
# `ProposeFn` is the query for fresh starts, handed the run's box because the
# box is the engine's and a model carries none.
ObserveFn: TypeAlias = Callable[[jnp.ndarray, list[float], jnp.ndarray | None], None]
ProposeFn: TypeAlias = Callable[[int, Domain], jnp.ndarray]

NPathStatus: TypeAlias = Literal[
    "max_iters",
    "time_limit",
    "budget_exhausted",
    "unbounded",
    "all_stalled",
]
"""Why a run ended: five names for five ways to end one.

The three gates, in the order they are checked: ``"max_iters"`` (the iteration
horizon), ``"time_limit"`` (the wall-clock deadline), ``"budget_exhausted"``
(the caller's extra batch gate). Then the two early endings: ``"unbounded"``
(a feasible point priced the objective at ``-inf``) and ``"all_stalled"`` (no
path can move and ``restart_stalled`` is off).
"""


@dataclass
class NPathResult:
    """End state of one n_path_schedule run.

    Field names follow `scipy.optimize.OptimizeResult` where the concepts
    correspond. It is spelled ``n_iters`` rather than scipy's ``nit`` to sit
    beside the four work counts: one iteration may serve several batches (a
    restart's starts plus the method's trials), so an iteration is not a batch.

    Work is reported as four counts instead of scipy's single ``nfev``, on two
    axes that never mix. The seam that served the work decides the pair -
    ``n_f_*`` is the value-only seam, ``n_f_g_*`` the fused value+gradient one -
    and points versus batches are the serial cost and the parallel wall-clock
    cost. Which axis a comparison reads is the caller's business; a run handed no
    ``eval_f`` seam simply reports zero on the f pair.

    Two deliberate differences from `OptimizeResult`: ``success``/``message`` are
    absent, because the engine has no convergence criterion of its own - a run
    ends early only for an objective with no floor or for a flock every path of
    which stalled with restarts off - and ``fun`` is the best over *feasible*
    points only.
    """

    x: jnp.ndarray
    """The argmin in the coordinates the engine was handed, shape ``(d,)``.

    Not necessarily a row of `flock_x`: the best point may come from a path that
    was restarted afterwards.
    """

    fun: float
    """`x`'s objective value, the minimum over feasible recorded evals.

    ``nan`` when nothing feasible was ever served, ``-inf`` when a feasible
    point priced the objective there.
    """

    flock_x: jnp.ndarray
    """Each path's held coordinates as the run ended, shape ``(n_paths, d)``.

    Already served and neither filtered nor sorted.
    """

    flock_fun: jnp.ndarray
    """Each path's held value as the run ended, shape ``(n_paths,)``.

    ``nan`` for a path that never held a usable point.
    """

    n_f_points: int
    """Value-only points served (the ``eval_f`` seam)."""

    n_f_batches: int
    """Value-only batches served (the ``eval_f`` seam)."""

    n_f_g_points: int
    """Fused points served: the serial-cost count."""

    n_f_g_batches: int
    """Fused batches served: the parallel wall-clock count."""

    n_iters: int
    """Completed engine iterations; iteration 0 is the flock's placement."""

    status: NPathStatus
    """Why the run stopped; see `NPathStatus`."""


def n_path_schedule(  # noqa: PLR0913 (the caller's knobs, all keyword-only
    # after the one positional seam: one name per dial beats a config object here)
    eval_f_g: EvalFG,
    *,
    # The run's width: how many chains the engine drives.
    n_paths: int,
    # The run's box, a `Domain`: every proposal is clipped into it - the
    # global model's fresh starts and the local method's steps alike - and it
    # is the one statement of the dimension (`domain.d`) and of each
    # coordinate's prior `scale`.
    domain: Domain,
    # The caller's flock: one start per path, ``(n_paths, domain.d)``, placed
    # by iteration 0 exactly where the caller asked rather than drawn.
    x0: jnp.ndarray | None = None,
    eval_f: EvalF | None = None,
    # The run's horizon, counted in iterations: one move per path, not one eval.
    max_iters: int | None = None,
    # A wall-clock deadline in seconds. A run needs a horizon: at least one of
    # max_iters / time_limit must be given.
    time_limit: float | None = None,
    # Keyword-only and defaulted, like scipy's `callback`: an extra gate for a
    # budget the engine cannot see (points, dollars, an external clock).
    budget_check: Callable[[], bool] | None = None,
    # Whether a stalled path is re-drawn. The stall itself is the engine's own
    # bit (`_Paths.stalled`); this knob only decides whether the model is then
    # asked for a fresh point.
    restart_stalled: bool = True,
    # The two plugins, adjacent and in one order everywhere: where a restart
    # lands (the model is also the run's evidence sink), and the search method.
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
    machine, and owns the best `f⋆`. Two things are injected, spelled
    `global_model`, `local_solver` in that one order wherever they are named:
    `global_model` supplies **where** a restart lands (one point per restarting
    path, clipped into the box) and is also the run's evidence sink, while
    `local_solver` owns direction, step size and acceptance and keeps its
    per-path state inside itself. `restart_stalled` is the third half of a
    restart, and it is a boolean: **when** a path restarts is the local solver's
    own stopping test, so a restart is two seams and one flag. It is on by
    default, because a stalled path with restarts off is idle parallelism.

    `n_paths` is the run's width and nothing else; the dimension is `domain.d`,
    so a run never spells it twice. Iteration 0 is not a special case: no path
    holds a point yet, and a path without a point restarts, so the first move
    places the whole flock as one fused batch - `x0` when one was handed in,
    otherwise one start per path from the global model. A start whose served
    value or gradient comes back non-finite stalls its path on arrival, and that
    path restarts again next iteration.

    `domain` is the run's box and the coordinate space the engine works in: the
    engine clips into it and knows nothing about any map from these coordinates
    to the caller's physical ones. It is also what the default model draws from,
    and the only statement of the dimension. With `global_model=None` the engine
    builds a `SampleGlobalModel` and lets the domain draw (`Domain.draw`); `key`
    seeds those draws, and an injected model owns its randomness instead.

    The run's own stop is the horizon: at most `max_iters` iterations, or the
    wall-clock deadline when `time_limit` is given. Iterations are the unit on
    purpose - one iteration is one move per path, so the same number reproduces
    on any machine - but it is not a cost bound: one iteration serves one or
    more batches, so migrating a batch budget to `max_iters` can multiply the
    work done. `budget_check` is the caller's extra stop, and that external
    budget may over-run by less than one batch. `NPathStatus` names what each of
    these writes.

    `fun` is the true minimum over every feasible eval with a finite value -
    every served point when the seam reports no mask, only the points its mask
    passes otherwise - while the trajectory itself moves on the penalized
    objective. Work is reported as four counts, never mixed, and none of them
    caps the run (see `NPathResult`).

    The caller's inputs are validated here, once per run: the horizon bounds,
    `n_paths`, and `x0`'s shape, finiteness and containment raise ValueError.
    The domain refused a malformed box when it was built. The two plugin objects
    are taken on their types - a type checker refuses a wrong one, and nothing
    here re-checks it at run time.

    Args:
        eval_f_g: the objective as one batched pass. Takes rows ``x`` of shape
            ``(M, d)`` and returns ``(f: (M,), g: (M, d), feas: (M,) | None)`` -
            value, gradient and optional feasibility mask. A mask holds 0/1 (or
            booleans) and nothing else: only 0 marks a point infeasible, so any
            other flag is refused rather than read as truthiness.
        n_paths: how many chains to drive; one point per path per iteration.
        domain: the run's box, a `Domain`.
        x0: the caller's flock: one start per path, shape ``(n_paths, d)``,
            placed by iteration 0 exactly where given and never used again.
            ``None`` draws the flock from the global model like any other
            restart. Validated at the entry, because an authored start outside
            the declared box is the caller saying "search this box" and "begin
            there" at once, which a clip would silently relocate.
        eval_f: the objective's value half alone, same ``x`` rows, returning
            ``(f: (M,), feas: (M,) | None)``. ``None`` means every served trial
            takes its gradient in the same fused pass; supplying it lets a
            judged step serve its trials for their values alone and buy
            gradients only for its winners, in one batch.
        max_iters: the run's horizon in iterations (``None`` leaves it to
            ``time_limit``); a run needs at least one of the two.
        time_limit: a wall-clock deadline in seconds (``None`` means none).
        budget_check: an extra gate for a budget the engine cannot see, asked at
            the loop head every iteration and again before each served batch.
            Additional, not a horizon (``None`` means none).
        restart_stalled: whether a path the engine marks stalled (its method gave
            up, or its point was not usable) is re-drawn from the global model.
            ``False`` keeps the flock frozen; a fully stalled flock then ends the
            run with ``status="all_stalled"``.
        global_model: the restart-point source and the run's evidence sink.
            ``None`` builds a `SampleGlobalModel` over ``domain``, which fits
            nothing.
        local_solver: the local method (``None`` binds ``DiagonalPreconditioner``).
        key: the PRNG seed of the default model's draws; ``None`` uses the
            fixed default, so unseeded runs are deterministic.

    Returns:
        The run's end state and work report, an `NPathResult`.

    Example:
        Minimize ``f(x) = ‖x - 3‖²`` by handing the engine a batched objective:

            >>> import jax
            >>> import jax.numpy as jnp
            >>> from soar_lite import Domain, n_path_schedule

            >>> def eval_f_g(x):
            ...     g = 2.0 * (x - 3.0)
            ...     return (jnp.sum((x - 3.0) ** 2, axis=1), g, None)

            >>> box = Domain([(-5.0, 5.0), (-5.0, 5.0)])
            >>> result = n_path_schedule(
            ...     eval_f_g,
            ...     n_paths=4,
            ...     domain=box,
            ...     # the flock the caller hands in; iteration 0 places it
            ...     x0=jnp.array(
            ...         [[-1.0, -1.0], [1.0, 1.0], [-4.0, 4.0], [4.0, -4.0]],
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
    )
    # The caller's flock, checked before anything is built: the entry is the
    # only place a human hands it over.
    if x0 is not None:
        x0 = _check_x0(x0, n_paths, domain)

    # The defaults, bound lazily: the local solver imports the engine's types,
    # so importing it at module scope would form a cycle.
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
        x0=x0,
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
        except _AllStalled:
            # Every path is stalled and none restarts: nothing left to move, and
            # no batch was served this iteration, so it is not ticked.
            status = "all_stalled"
            break
        budget.tick()  # counted after the body: only finished iterations

    # Each path's point as the run ended, and its value. A path that never held
    # a usable point still carries an all-nan `x` - that is what the placeholder
    # is - so it reports `nan` on both halves with no branch here.
    flock_x = jnp.stack([p.x for p in paths.points])
    flock_fun = jnp.asarray([p.f for p in paths.points])

    return NPathResult(
        x=best.x,
        fun=best.fun,
        flock_x=flock_x,
        flock_fun=flock_fun,
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
    ``vmap(objective)``, not a second model of it, so returning it costs nothing
    and buys the engine's cheap-trial path. That does not make this the seam for
    a *value-only* objective: a function JAX cannot differentiate cannot be
    `objective` here at all, so a caller holding values and no gradient needs an
    f-only adapter of its own.

    Args:
        objective: the per-point objective, ``(d,) -> ()``; a function
            `jax.value_and_grad` can differentiate, which is what makes the
            fused half exist.
        feasible: the per-point feasibility predicate, ``(d,) -> ()`` of a
            boolean; None (the default) marks every point viable, the same
            `None` both seams accept.

    Returns:
        The two seam shapes `n_path_schedule` takes, as a pair.
            ``eval_f_g`` a fused ``(f, g, feas)`` batch, ``eval_f`` its
            value-only ``(f, feas)`` sibling.

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
    or `_serve_f`), and caught by the loop. It carries the reason `stop`
    answered at the batch altitude, so the loop names the run's status without
    re-asking the query.
    """

    reason: NPathStatus  # the gate `stop` named, carried to the loop's one catch

    def __init__(self, reason: NPathStatus) -> None:
        self.reason = reason
        super().__init__(reason)


class _OptimumReached(Exception):  # noqa: N818 - a signal, not an error
    """A served *feasible* point priced the objective at `-inf`.

    Raised inside `_Best.observe`, the single funnel both seams pass through,
    and caught by the loop. Unlike `_BudgetExhausted` it arrives after a batch
    was served and charged, and it leaves an answer behind instead of naming a
    gate: the run stops on that point, so `fun` is `-inf` and `status` is
    `"unbounded"`. Nothing is left to unwind - a batch raises before any point is
    placed or advanced.
    """


class _AllStalled(Exception):  # noqa: N818 - a signal, not an error
    """Every path has stalled and the run cannot re-draw any of them.

    Raised inside `_Paths.move` when no path restarts and every path is stalled
    - only reachable with `restart_stalled=False`, since a stalled path restarts
    otherwise. Nothing can un-stall a path but `_place`, which only ever sees the
    restarted paths, so the state is terminal: the iteration would serve
    nothing, and repeating it would idle out the horizon. The run stops on it,
    naming the state in `status`.
    """


class _Budget:
    """The run's stop meter and work report.

    Three predicates decide whether work may go on: the iteration horizon
    (`max_iters`), the wall-clock deadline (`time_limit`), and the caller's
    extra gate (`budget_check`). `stop` asks all three as one query - at the
    loop head every iteration, and again before each served batch - and names
    the reason it refuses. At least one of the first two is required at entry:
    they advance on their own and the engine can see them do it, while a
    `budget_check` is the caller's predicate, so the engine cannot tell a gate
    that closes from one that never does.

    The counters are pure reports, not caps, and not one of them gates: the
    fused pass counts on its pair (`n_f_g_points` / `n_f_g_batches`, one batch
    per value+gradient pass) and the value-only pass the same way on its own
    (`n_f_points` / `n_f_batches`). `stop` reads only the run's three
    predicates, so a counted batch can still be refused by the gate like any
    other serve. That refusal is a signal (`_BudgetExhausted`), never a flag
    stored here.
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

    `fun`/`x` are the answer the run reports, defined over feasible designs only
    - with no mask, every point is feasible. A NaN or +inf trial neither becomes
    `f⋆` nor displaces one: it carries no value to compare. A feasible -inf is
    not an improvement to weigh but the answer itself, so it ends the run
    (`unbounded`) instead of entering a comparison. It starts at `nan` and stays
    there until a finite feasible value arrives: `nan` is the "no answer" state,
    which leaves `inf` free to mean a value an objective actually took.
    """

    def __init__(self, d: int) -> None:
        # d is the flock's business (it is `x0`'s second axis), so the
        # placeholder is already the right shape.
        self.x = jnp.full(d, math.nan)
        self.fun = math.nan  # "no answer yet", not `inf`: a value may take it
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
        # nan means no answer yet, so the first finite feasible value lands
        # unconditionally; after that one must beat the running best.
        if math.isnan(self.fun) or f_vals[j] < self.fun:
            self.fun = f_vals[j]
            self.x = x_batch[j]


class _Evaluator:
    """The run's evaluation seams, and what they charge and observe.

    One rule, stated here so no call site restates it: every batch asks the gate
    before it is served (so a batch is served whole or not at all), and the seam
    that served it decides which pair of counters moves - `_serve_f` the f pair,
    `_serve_f_g` the fused pair. No count ever closes the gate.

    Whichever half served a batch reports it to the two readers of the run's
    evidence: the best (`_Best`, which filters for feasibility) and the global
    model (which does not). That is the only place either is fed, so a batch
    nobody decides on is still reported. `settle` lives here too: it is where an
    iteration's deferred gradients are bought together, since this object owns
    both the seam and the charge.
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
        fused pass. Keeping this spelling beside `settle`'s promotion keeps
        `_Pending` construction in this object instead of at the call sites.
        """
        return [_Pending(row, None, None) for row in x_batch]

    def serve_trials(self, x_batch: jnp.ndarray) -> list[_Pending]:
        """Serve one step's trial batch, as one produced candidate per row.

        A trial's gradient is only wanted for the step's winner, so with the
        f-only seam each trial is served for its value alone and the winners are
        graded once, together (`settle`). Without the seam a fused pass is the
        only way to a value, so every trial carries its gradient whether or not
        the step's argmin keeps it; `settle` then buys nothing.
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
        """Buy whatever gradient the iteration's points still owe, in one batch.

        Everything the f-only seam left pending goes out as a single batch, so an
        iteration costs one gradient pass however many moves it deferred - its
        restart starts plus its step winners - instead of one pass per trial. A
        point that already carries its gradient is promoted as it stands, so a
        run that owes nothing buys nothing and the counters do not move. Returns
        the same points, in the same order, all graded.
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
        """Serve one fused batch, or raise `_BudgetExhausted` if it is unaffordable.

        The stop query is asked at this batch altitude before the charge, so a
        batch is served whole or not at all and the fused counters only ever
        hold batches that were served. This is the expensive half; a judged
        step's rejected trials are served by `_serve_f`, which asks the same gate
        and counts on the other pair. Returns ``(f_vals, g_batch)``.
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
        than after a whole iteration's retries: the judged lane's loop can turn
        many times without reaching the loop head or `settle`. What stays
        asymmetric is the charge - this batch counts on the f pair, and no count
        closes the gate. Only `serve_trials` calls this; the None-check below is
        for the type checker and is unreachable at runtime.
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

    The invariants the loop's sites would otherwise carry implicitly, stated
    once here:

    - A path that restarted this iteration moved by its fresh start, so it does
      not also step and runs no trials: its method sees no step, so the restart
      is never read back as a zero-length step. `move` computes the restarted
      list once and threads it to `_restart_starts` and to the complement both
      halves share.
    - An iteration decides on values and pays for gradients once: whatever the
      f-only seam left ungraded (restart starts and step winners alike) is bought
      in one batch before anything is placed, so a decision is never priced
      before it is made.
    - A path is stalled when it did not move at all this iteration - its method
      gave up on every trial, or its fresh start or its winner was not usable. A
      stalled path does not step, so it serves nothing that iteration and the
      method's own state stands still.
    - A start or a winner is committed only when it is `usable` - finite in both
      halves - one gate defined once on the point type.
    - A path holds a point or it does not. "No point" (an all-nan `x`, which only
      the placeholder has) is the machine's own state, not a stall: such a path
      restarts without consulting `restart_stalled`, which is how iteration 0
      places the flock and how a refused start is eventually re-drawn. A refused
      point is the other case: the path keeps the point it holds and is marked
      stalled, so `restart_stalled` decides whether it is re-drawn.
    """

    def __init__(  # noqa: PLR0913 - the run's collaborators, built once
        self,
        n_paths: int,
        domain: Domain,
        evaluator: _Evaluator,
        propose: ProposeFn,
        local_solver: LocalSolver,
        *,
        x0: jnp.ndarray | None,
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
        # The caller's flock, handed over for the run's first move and consumed
        # there (`_restart_starts` clears it). Not a collaborator: it answers
        # exactly one ask, where a model answers every later one.
        self._x0 = x0
        # A knob, not a collaborator: *when* a path restarts is read off
        # `stalled` below, so the only thing left to decide is whether the
        # engine acts on it at all.
        self.restart_stalled = restart_stalled
        # A path's point is one object, not three parallel lists: it is what
        # every seam call takes, and a step commits one path at a time (so a
        # list, whose item assignment is the commit, beats a stacked array). The
        # placeholder says "no point has arrived" with an all-nan `x`: an `x` is
        # always `domain.d` wide, so "has no point" is a value on that shape, not
        # a second shape - the same placeholder `_Best` uses for "no answer". A
        # usable restart start replaces it, an unusable one never becomes a
        # point, and a stalled path never serves or compares it.
        self.points = [
            PathPoint(
                jnp.full(self.domain.d, jnp.nan),
                math.nan,
                jnp.full(self.domain.d, jnp.nan),
            )
            for _ in range(self.n_paths)
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
        with no point restarts, so the first move places the flock - the
        caller's `x0` if one was handed in, else one start per path from the
        global model - in one batch.

        The restart decision comes first: a path that holds no point, or that is
        stalled and whose run opted into `restart_stalled`, is given a fresh
        start instead of stepping. Both halves decide where a path goes from
        values alone when the f-only seam is present, and the gradients those
        decisions need are then bought together in one batch (`settle`), so an
        iteration costs one gradient pass however many moves it deferred. A run
        without the seam serves everything fused and reaches `settle` already
        graded.

        The two halves touch disjoint paths, and every call into a collaborator
        is per path, so deciding all moves before placing any is the same
        schedule as placing each as it is decided.

        A flock with nothing left to do ends the run instead of idling: when no
        path restarts and every path is stalled, `_AllStalled` is raised before
        any batch is asked for.
        """
        # "No point" is the machine's own state, never a stall to weigh, and it
        # is marked by an all-nan `x` rather than by a shape: `x` is always
        # `domain.d` wide, so the placeholder is a value on that shape (the same
        # one `_Best` uses for "no answer"). Only the placeholder has it, and
        # only a path that holds a point can be read as stalled: a held point is
        # committed through `usable`, so its `x` is clipped into the box and its
        # f is finite.
        restarted = [
            i
            for i in range(self.n_paths)
            if bool(jnp.all(jnp.isnan(self.points[i].x)))
            or (self.restart_stalled and self.stalled[i])
        ]
        # Nothing is going to move again: no path restarts, and every path is
        # stalled. `_place` is the only writer that clears a stall bit and it
        # only ever sees the restarted paths, so an empty `restarted` beside a
        # fully stalled flock is terminal - the iteration would serve nothing.
        # Only reachable with `restart_stalled=False`; the run ends here rather
        # than idling out the horizon.
        if not restarted and all(self.stalled):
            raise _AllStalled
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
        """Ask for n_points fresh starts, clipped into the run's box.

        The caller's flock, if it handed one, answers the run's first call and
        only that one: iteration 0 is the one move where every path holds no
        point, so `x0` is consumed here and every later restart asks the model.
        `x0` was validated at the entry, so it is placed as handed; the clip
        below is the roundoff sliver's snap onto a wall.

        Otherwise: ask, check, clip, place. The model is handed the box it must
        land in and its answer is refused unless it is ``(n_points, domain.d)``,
        so a model returning another width fails here, naming the shape, instead
        of deep in a broadcast during the clip. The clip stays as the defensive
        seam for a model that oversteps. Both halves stay owed - a start has
        nothing to be judged on - so `settle` buys them in the same batch as the
        step winners, and a restart costs no gradient pass of its own.
        """
        d = self.domain.d
        if self._x0 is not None:
            proposed, self._x0 = self._x0, None
        else:
            proposed = self.propose(n_points, self.domain)
        if proposed.shape != (n_points, d):
            msg = (
                f"a restart ask must propose a ({n_points}, {d}) array, "
                f"got {proposed.shape}"
            )
            raise ValueError(msg)
        x_cand = jnp.clip(proposed, self.domain.lo, self.domain.hi)
        return self.evaluator.pending_unread(x_cand)

    def _decide(self, cohort: list[int]) -> dict[int, _Pending]:
        """Run the cohort's trials: the method judges, the engine records winners.

        The cohort splits on ``needs_verdict``. A path that needs one has its
        step enumerated: the method proposes, the engine clips and serves, the
        method judges. Trial passes run in lockstep over the still-rejecting
        subset, each served whole or not at all. A path's point is unchanged
        while it rejects, so the point handed to propose and verdict is also the
        step's start. A path that needs none commits its one proposal unread -
        the same deal a restart start gets - so its value is served once, fused
        with its gradient, by the caller's `settle`.

        Returns the per-step argmin each accepting path won, keyed by path, still
        waiting for `settle` when the seam served values only. Nothing is
        committed here: a path ends this iteration stalled exactly when its
        method found nothing to win (its convergence test fired, or it accepted
        with no finite sample). The cohort also holds already-stalled paths: they
        run no trials and emit no winner.
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
                    trial.g is None or _all_finite(trial.g)
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
        the path stays stalled until it is re-drawn - which keeps `points[i]`
        meaning "where the path stands". The start is this iteration's move for
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


def _validate_run(
    *,
    n_paths: int,
    max_iters: int | None,
    time_limit: float | None,
) -> None:
    """Reject a run the engine cannot finish or cannot price.

    The caller's own inputs are checked here, once per run - the only place a
    human hands this engine something - and nothing below re-checks them. Only
    what the annotations cannot state is tested: a horizon, two ranges, a count.
    A seam's type is the annotations' job: a check for "callable" or "carries
    the Protocol" re-asserts what the type checker already refuses.
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
            "time_limit (seconds); budget_check is the caller's gate, and the "
            "engine cannot tell one that closes from one that never does"
        )
        raise ValueError(msg)

    _check_positive_int(n_paths, "n_paths")


def _check_x0(x0: jnp.ndarray, n_paths: int, domain: Domain) -> jnp.ndarray:
    """Reject a flock the run cannot place: bad shape, not finite, outside.

    Called once at the entry, where the width, the dimension and the box are all
    known. Three defects, each one no clip or proposal repairs: not a
    two-dimensional batch whose width and dimension match this run, a
    non-finite start, or a start outside the box.

    The third is the one not interchangeable with the engine's clip. An
    out-of-box start is the caller saying "search this box" and "begin there" at
    once; clipping it would slide the start onto a wall and hide the
    contradiction. The clip keeps its own job: proposals the model *generates*
    may overstep, and clamping those is the defensive seam.

    The box test carries a machine-precision tolerance: a start computed at a
    boundary rounds by a few representable steps, so a coordinate within
    ``16 * eps * max(1, |bound|)`` of a finite side counts as on it, where eps
    comes from the points' dtype (at most float32). Free dimensions never
    trigger it - an infinite side is compared against itself.
    """
    pts = jnp.asarray(x0)
    d = domain.d
    if pts.ndim != 2 or pts.shape[1] != d:
        msg = f"x0 must be a (n_paths, {d}) array, got shape {pts.shape}"
        raise ValueError(msg)
    if pts.shape[0] != n_paths:
        msg = f"x0 holds {pts.shape[0]} starts but n_paths is {n_paths}"
        raise ValueError(msg)
    if not _all_finite(pts):
        row, dim = (int(v) for v in jnp.argwhere(~jnp.isfinite(pts))[0])
        msg = (
            f"x0 row {row} is {float(pts[row, dim]):.6g} in dimension {dim}, not finite"
        )
        raise ValueError(msg)
    lo, hi = domain.lo, domain.hi
    eps = jnp.finfo(jnp.result_type(pts.dtype, lo.dtype, jnp.float32)).eps
    # The tolerance's own scale comes from the finite sides only: taking it from
    # an infinite side would carry it to inf and disarm the other, finite one.
    span = jnp.maximum(
        jnp.where(jnp.isfinite(lo), jnp.abs(lo), 0.0),
        jnp.where(jnp.isfinite(hi), jnp.abs(hi), 0.0),
    )
    tol = 16.0 * eps * jnp.maximum(1.0, span)
    outside = (pts < lo - tol) | (pts > hi + tol)
    if bool(jnp.any(outside)):
        row, dim = (int(v) for v in jnp.argwhere(outside)[0])
        msg = (
            f"x0 row {row} is {float(pts[row, dim]):.6g} in dimension {dim}, "
            f"outside the box [{float(lo[dim]):.6g}, {float(hi[dim]):.6g}]"
        )
        raise ValueError(msg)
    return pts


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

    Only 0 marks an infeasible point, so the mask is never read as a truthiness
    test: a signed constraint margin arrives looking feasible on every point it
    rules out, and a NaN arrives looking feasible too, both silently promoting a
    point into the reported best. No cast repairs that (a NaN casts to True), so
    the batch is refused rather than guessed at.
    """
    if not bool(jnp.all((feas == 0) | (feas == 1))):
        msg = (
            "the feas mask must hold 0/1 (or booleans) - only 0 marks an "
            f"infeasible point, got {feas.tolist()}"
        )
        raise ValueError(msg)
    return [bool(v) for v in feas.tolist()]


def _check_positive_int(value: int, name: str) -> int:
    """Return `value` if it is a positive int, refusing bools and the rest.

    The bool test is not redundant with the annotation: `bool` subclasses
    `int`, so a type checker accepts `True` where an `int` is asked for, and a
    flag is not a count.
    """
    if isinstance(value, bool) or value < 1:
        msg = f"{name} must be a positive int, got {value!r}"
        raise ValueError(msg)
    return value
