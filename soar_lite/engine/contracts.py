"""The plugin contracts, the data they exchange, and the evaluation seams.

`LocalSolver` and `GlobalModel` are the two seams a caller plugs into;
`Point`, `ValuePoint`, `PathPoint`, `Work` and the evaluation-seam aliases are
what crosses them; and `NPathResult` with `NPathStatus` is the other half of the
same exchange, what the caller reads back. `pointwise` builds both evaluation
seams from one per-point objective.

The machine depends on these types, never the reverse: nothing here imports
`soar_lite.engine.schedule`. The package facade re-exports both halves, so a
caller still names any of them from `soar_lite.engine`, and this module stays the
half a plugin author reads.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import jax
import jax.numpy as jnp

from soar_lite.domain import Domain


@dataclass(frozen=True)
class Point:
    """A point the run has located, with nothing of it read yet.

    Three kinds of point say how much of one the run has read, and each is the
    next one's base class: coordinates alone here, the value in `ValuePoint`,
    the gradient as well in `PathPoint`. What has been read is therefore the
    point's type rather than a field that may hold ``None`` - a half still owed
    cannot be asked for at all, and no caller has to check whether it may.

    Everything the run produces starts as one of these, because producing a
    point and reading it are two decisions: a restart start arrives from a model
    that read nothing, and a method may name a winner by its own rule instead of
    by its value.
    """

    x: jnp.ndarray
    """Coordinates, shape ``(d,)``, in the run's coordinates."""


@dataclass(frozen=True)
class ValuePoint(Point):
    """A point whose value has been read; its gradient is still owed.

    A value-only trial row is one of these, and stays one until `settle` buys
    the gradient and it becomes a `PathPoint`. It is also what a method holds
    for a weight it named unread and then had read: the read is the only thing
    this class says about a point, whoever asked for it.

    The interesting half of the definition is the missing one. There is no
    gradient-only point, and no `eval_g` seam that would serve one: a caller
    hands in a single objective whose value and gradient come back from one
    call, so a gradient is only ever bought as the pair, and the value that
    rides free with it is kept rather than thrown away.
    """

    f: float
    """The objective's value there."""


@dataclass(frozen=True)
class PathPoint(ValuePoint):
    """One path's point: coords x, value f, gradient g = ∂f/∂x.

    The engine builds these and the local method receives them; a caller
    normally reads the run's outcome through `NPathResult` instead of
    constructing one. A point is a candidate until `is_usable` passes, so a
    non-finite half is allowed on a point that has not been committed.
    """

    x: jnp.ndarray
    """Coordinates, shape ``(d,)``, in the run's coordinates."""

    f: float
    """The objective's value there. Finite for any point a path actually holds
    (``is_usable`` gates the commit); a point still being considered may carry a
    non-finite value."""

    g: jnp.ndarray
    """The gradient there, shape ``(d,)``; finite under the same ``is_usable`` gate
    as ``f`` - held points are finite, candidates need not be yet."""

    @property
    def is_usable(self) -> bool:
        """Return whether this point may be committed as a path's point.

        Both halves must be finite, because both are carried forward: a
        non-finite f cannot be reported, re-started from or finished on, and a
        non-finite g is proposed from and blended into the secant, so one NaN
        there poisons that coordinate's magnitude EMAs for the rest of the run.
        f is asked first, so a dead value skips g's finite scan.
        """
        return math.isfinite(self.f) and bool(jnp.all(jnp.isfinite(self.g)))


ServeTrials: TypeAlias = Callable[[Mapping[int, jnp.ndarray]], Mapping[int, ValuePoint]]
"""The trial hook a step is handed: one point per path in, one read point back.

Each call is one round over the paths it names - ``path -> (d,) coordinate`` in,
the same paths out with their values - and a method calls it as many times as
its enumeration needs: one round, ten rounds, or none. One point per path per
call rather than a batch of many is what keeps a round's memory at
``n_paths * d`` however long the enumeration runs; multiplicity lives in the
number of calls, so a line search that returns the best of its rounds holds one
round at a time and the run's peak never grows with how far a method looks.

What each answer carries follows the run's seams - the value alone when
``eval_f`` is connected, both halves when it is not - so an answer is a
`ValuePoint`, and a `PathPoint` on the fused seam. It is what
`LocalSolver.propose` receives in place of the caller's own function. Keys are
paths the step was asked about: the engine refuses a round naming any other,
and refuses a point outside the box, naming the path.
"""


class GlobalModel(Protocol):
    """The engine's restart-point source (its second plugin point).

    The model answers the two questions the engine cannot: *where* a restart
    lands (`propose`) and *what the run has learned* (`observe`). The engine
    feeds it every batch it serves and asks for points on a restart, or for
    iteration 0's placement when no `x0` was handed in, so `propose` is the only
    place it is ever asked: a model that wants more evidence than it has can
    answer with a plain draw and the engine has no opinion.

    A model carries no search space of its own. The run has one box, the engine
    owns it, and `propose` is handed that `Domain` at every ask - so a model must
    not remember one. That is what keeps a fitted geometry and a fallback draw
    in the same coordinate unit as the box test, with no second box to reconcile.

    One instance serves the whole run (like `LocalSolver`), so whatever it
    accumulates lives inside it and the engine never reads it. The methods are
    called by the engine on its own schedule - `observe` once per served batch,
    `propose` once per cohort of restarting paths (all the paths that restart in
    one iteration, in one call) - so they trust their arguments: this Protocol
    is the whole contract.

    The flock is not this contract's: a caller's own starts go to
    `n_path_schedule` as `x0`, and the engine places them without asking any
    model.

    Implementing this pair is the whole extension point; a projection is a
    utility a `propose` may read, not part of the contract.

    The smallest implementation is `SampleGlobalModel` below - `observe` keeps
    nothing, `propose` answers with the domain's own draw - so a model that fits
    something is that shape with a fit added: `BLRGlobalModel` keeps what it was
    handed and answers from its posterior once it has enough.
    """

    def observe(self, served: Mapping[int, ValuePoint]) -> None:
        """Fold one served round into the model's evidence.

        Called right after a batch is served, in the engine's coordinates and
        before anything is decided on it: `served` maps each path in the batch to
        the point it was served there, so the model knows whose point it holds
        rather than only how many rows arrived. The values are Python floats -
        the same values the run compares, and one host transfer per batch rather
        than a device sync per row - and a point served for its value alone
        carries no gradient: a fused batch arrives as `PathPoint`s, which are
        also `ValuePoint`s, so a model that wants gradients reads them off the
        type instead of a second nullable argument.

        The mapping's iteration order is the order to stack in, and the engine
        keeps it stable (ascending path order), so a model that folds rows keeps
        a seeded fit reproducible. Keys are unique within one call; a point may
        arrive twice across calls - a step's trial is served for its value first
        and for its gradient later if it wins - so a model keyed on coordinates
        sees the same design with more evidence.

        Nothing is filtered for feasibility: the model fits the raw objective
        everywhere, and only the value the run reports is filtered. Batches the
        run never decides on arrive too (a trial that lost its step, the batch
        that ended an unbounded run), because the work was served.
        """
        ...

    def propose(
        self, paths: Sequence[int], domain: Domain
    ) -> dict[int, jnp.ndarray | None]:
        """Answer one fresh start per path asked about, keyed by that path.

        `paths` is every path that restarts this iteration - on iteration 0,
        every path when the caller handed no `x0` - in ascending order, and
        `domain` is the box the answers must land in and the one statement of
        the dimension. Each answer is one point in the coordinates the objective
        takes; ``None`` says this path gets no start from this model, and the
        engine stalls the path where it stands (one holding no point is asked
        again next iteration). Declining is the model's own stopping rule, as
        ``None`` from a method's `propose` is the method's.

        The engine checks every answer it is given at the batch boundary - a
        ``(d,)`` point, finite and inside the box - and raises ``ValueError``
        naming the path and the dimension on a miss: an answer outside the box is
        a model disagreeing with the box it was handed, which the engine names
        rather than repairing. The box test allows the few ulps a start computed
        on a boundary can round by.

        The model owns its randomness (an instance holds its key), which is what
        makes a seeded model reproducible; ascending order is the other half of
        that, so a model that draws one row per path and zips them in the order
        asked reproduces a run from its seed alone.
        """
        ...


class LocalSolver(Protocol):
    """The search method the n-path engine drives (its first plugin point).

    The engine owns the schedule - batch composition, the box test, the shared
    horizon, the n-path state machine (restart/step/stall), the best
    `f⋆`, and the gradient purchase - and the method owns the search: direction,
    step size, acceptance, **and how many points a step looks at**. A step is one
    ``propose`` call, so a method that wants a trial grid or a line search runs
    it inside that call: the engine hands ``propose`` the run's trial hook
    (``serve_trials``) and serves whatever the method asks for. Runs end on the
    caller's horizon, not on the method. All state is per path, lives inside the
    instance, and is opaque to the engine - implementers index it by path ``i``.

    A path's point reaches the method in exactly two places: ``restart(points)``,
    which places it, and ``step(points)``, which moves it. Both are the engine's
    own commit points, so a method that keeps the point it was handed holds the
    one the engine holds. Both carry one row per path for every path that landed
    this iteration, keyed by path and in one call, so a method that keeps
    per-path state updates it in one place. A path that did not land - the
    method declined it, or its point arrived unusable - is in neither call: it
    stalls where it stands. A method that steps from a point must keep it,
    because no later call hands it back: ``propose`` reads the current point off
    the method's own state and answers from it. ``step`` is the only place a
    landed point is announced, so it is also the only place the method may
    advance its model - a secant is a step that was taken, and only the ``step``
    callback is told one was.

    The split between the two evaluation halves is the method's cost lever. A
    step's *value* is what its enumeration compares, so the method reads it:
    ``serve_trials`` serves a batch of candidate points for their values alone -
    or the fused pair, when the caller connected no ``eval_f`` - and may be
    called as often as the enumeration needs. A step's *gradient* is wanted only
    for the winner, so the engine buys it afterwards, in the iteration's one
    fused batch (``settle``): whatever is still owed, which is every read
    winner's gradient and both halves of every unread winner and restart start.
    ``step`` then hands the result back. A point therefore leaves ``propose`` as
    whatever it has read: a bare `Point` when the method never read it, a
    `ValuePoint` when it read it through ``serve_trials``, and a `PathPoint` when
    the caller supplied no ``eval_f`` seam and the hook served the pair.

    ``domain`` is handed to ``propose`` rather than stored, because the solver
    carries no search space of its own; a start is placed by ``restart``. A step
    and every trial row must already lie inside that box: the engine checks both
    and refuses what misses, so clipping is the method's own defence against the
    roundoff in its own arithmetic, not something the engine does for it.

    These methods are called by the engine on its own schedule - `restart` and
    `step` once per iteration with the paths that landed, `propose` once per
    iteration for the paths that step - so they trust their arguments: this
    Protocol is the whole contract.

    Example:
        A whole method - fixed-step gradient descent, one point per path, no
        trials read - which is enough to converge, because the engine pays for
        the gradient this method never asks for:

            >>> import jax.numpy as jnp
            >>> from soar_lite import Domain, Point, n_path_schedule

            >>> class FixedStep:
            ...     def __init__(self, rate=0.25):
            ...         self.rate, self._point = rate, {}
            ...     def restart(self, points):
            ...         self._point.update(points)
            ...     def step(self, points):
            ...         self._point.update(points)
            ...     def propose(self, paths, domain, serve_trials):
            ...         del serve_trials          # this method reads no trials
            ...         return {
            ...             i: Point(
            ...                 jnp.clip(
            ...                     self._point[i].x - self.rate * self._point[i].g,
            ...                     domain.lo,
            ...                     domain.hi,
            ...                 )
            ...             )
            ...             for i in paths
            ...         }

            >>> result = n_path_schedule(
            ...     lambda x: (jnp.sum((x - 3.0) ** 2, axis=1), 2.0 * (x - 3.0), None),
            ...     n_paths=2,
            ...     domain=Domain([(-5.0, 5.0), (-5.0, 5.0)]),
            ...     x0=jnp.array([[-1.0, -1.0], [1.0, 1.0]]),
            ...     max_iters=40,
            ...     local_solver=FixedStep(),
            ... )
            >>> bool(jnp.allclose(result.x, 3.0, atol=1e-2))
            True

        Nothing is read yet, so the winner is a bare `Point` and the gradient
        reaches `step` through the engine's `settle` batch. A method that reads
        its candidates through `serve_trials` hands each winner back as the
        `ValuePoint` it read instead, and the same batch settles only the
        gradient.

    """

    def restart(self, points: Mapping[int, PathPoint]) -> None:
        """Seed each path's state from a fresh start point (its one cold start).

        Called by the engine each time it places paths on points: once at
        initialization, and again on every redraw — of a stalled path with
        `restart_stalled` on, or of a path that holds no point, which is re-drawn
        either way. Every path that landed a start this iteration is in the one
        call; the engine has one way to place a path, so a method need not tell
        "first" from "again" - both drop that path's state to the cold one. A
        restarting path does not also step, so this call and `step` never name
        the same path in one iteration. One of the two calls that hand a point
        over, so a method that steps from a point keeps this one.
        """
        ...

    def step(self, points: Mapping[int, PathPoint]) -> None:
        """Take the steps the engine just committed: each path stands at its point.

        The engine's only call site is the commit of the usable winners, with
        each ``point`` the settled one it committed, so every recorded step was
        taken and can never disagree with the path. No path the engine refused
        (not `is_usable`) and no path whose stopping test held is in the call.
        Each ``point`` means what it means in ``restart``: the point that path now
        stands at. The point it stepped from is the method's own record, not an
        argument - the same rule that makes ``propose`` read its point instead of
        being handed it.
        """
        ...

    def propose(
        self, paths: Sequence[int], domain: Domain, serve_trials: ServeTrials
    ) -> dict[int, Point | None]:
        """Answer one step for each path in ``paths``: its winner, or ``None``.

        One call covers every path that steps this iteration, so a method that
        wants a trial grid proposes the cohort's samples in one round
        rather than once per path. The answer is keyed by the paths asked, and
        ``None`` means the method declines to name a winner: its stopping test
        holds, or the trials it read left it nothing usable to pick. Either way
        the engine stalls the path where it stands, serving nothing and
        committing nothing.

        A non-``None`` entry is the step's winner, held as the kind of point the
        method has read it to: a bare `Point` it never read, or a `ValuePoint`
        it read through ``serve_trials`` (a `PathPoint` on the fused seam). The
        engine buys whatever is still owed, then commits it as the path's next
        point and calls ``step`` - unless it is not `is_usable`, in which case
        the path stalls and ``step`` is never called.

        ``serve_trials`` is the run's trial hook. Every call
        serves one round and reports it to the run's best and to the global
        model, exactly as a fused batch is reported, so a method may call it as
        freely as its enumeration needs - a round is one point per path, so a
        long enumeration costs more rounds, never more memory. With no ``eval_f``
        seam the same call serves the fused pair, and the answers are
        `PathPoint`s. Every point it is handed must lie inside ``domain``: the
        engine checks the round and refuses one that misses, naming the path, so
        a method enumerates over the box it was handed rather than leaving the
        hook to repair its grid.

        The winner must already lie inside ``domain`` for the same reason: the
        method is handed the box, and the engine refuses a step that leaves it
        instead of sliding it back onto a wall.
        """
        ...


EvalFG: TypeAlias = Callable[
    [jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None],
]
"""The objective as one fused pass: the one seam a run must supply.

Takes ``x`` rows ``(M, d)`` in the run's coordinates and returns
``(f: (M,), grad_x: (M, d), feas: (M,) | None)``. Value and gradient come back
together on purpose - the caller's fused ``value_and_grad`` is one AD pass, and
the step's call sites want both at once: the batched equivalent of scipy's
``jac=True`` convention.

``feas`` is optional: ``None`` means every point is feasible, otherwise 0 marks
a point infeasible and 1 marks it feasible; any other value is refused rather
than read as truthiness.
"""

EvalF: TypeAlias = Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray | None]]
"""The objective's value half alone, for a caller that also holds one.

Same ``x`` rows in, ``(f: (M,), feas: (M,) | None)`` out. Feasibility is
included because it decides the value the run reports and whether a ``-inf``
ends the run, neither of which ``f`` alone can tell. Opting in lets a step that
enumerates serve its trials for their values alone and buy the winners'
gradients afterwards, together in one fused batch.
"""

# The two halves of a `GlobalModel`, as the components that call them see them.
# The model itself never crosses into `_EvalSeams` or `_Paths`: each takes the
# one method it uses, so neither holds a god object whose other half it ignores.
# `ObserveFn` is the command that folds a served round into the evidence: one
# point per path, values already on the host because the run's own bookkeeping
# compares and records them one row at a time - a batch then costs one host
# transfer instead of a device sync per row.
# `ProposeFn` is the query for fresh starts: one answer per path asked about,
# handed the run's box because the box is the engine's and a model carries none.
ObserveFn: TypeAlias = Callable[[Mapping[int, ValuePoint]], None]
ProposeFn: TypeAlias = Callable[[Sequence[int], Domain], dict[int, jnp.ndarray | None]]

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
(the caller's extra batch gate). The first one that holds is the one reported,
so a last iteration that reaches the horizon as the deadline passes reports
``"max_iters"``. Then the two early endings: ``"unbounded"`` (a feasible point
evaluated to ``-inf``) and ``"all_stalled"`` (no path can move and
``restart_stalled`` is off).
"""


@dataclass(frozen=True)
class Work:
    """What one run billed, on two axes that never mix.

    Four counts instead of scipy's single ``nfev``. Which evaluation seam served
    the work decides the pair: ``n_f_g_batches`` and ``n_f_g_points`` for the
    fused one, ``n_f_batches`` and ``n_f_points`` for the value-only seam.
    Batches are the parallel wall-clock cost and points the serial one. Which
    axis a comparison reads is the caller's business; a run handed no ``eval_f``
    seam reports zero on both f counts.

    These are reports, not caps: no count stops a run on its own, only the run's
    stop predicates decide. A caller's ``budget_check`` gate is handed one of
    these: it reads the ledger and cannot write it. Frozen, and replaced rather
    than mutated as the run charges it, so a reference to one is the snapshot of
    the moment it was handed over.

    At a serve site the gate is handed the ledger the pending batch *would*
    leave, which is why a points ceiling holds: a batch is charged only if the
    gate allows it, and the count a refusal was decided on never enters the
    ledger. What the report carries is therefore exactly the ledger the last
    allowed batch left - never one a gate refused. At the loop head, where no
    batch is pending, the gate is handed the ledger as it stands.

    The iteration count is not here: `max_iters` bounds it, and one iteration
    may serve several batches (a restart's starts plus the method's trials), so
    an iteration is not a batch.
    """

    n_f_g_batches: int
    """Fused batches served: the parallel wall-clock count."""

    n_f_g_points: int
    """Fused points served: the serial-cost count."""

    n_f_batches: int
    """Value-only batches served (the ``eval_f`` seam)."""

    n_f_points: int
    """Value-only points served (the ``eval_f`` seam)."""


@dataclass
class NPathResult:
    """End state of one n_path_schedule run.

    The answer is spelled the way a point is: ``x`` and ``f`` are the winning
    `PathPoint`'s coordinates and value, and the flock pair prefixes those same
    two names with ``flock_``. It is ``n_iters``, not scipy's ``nit``, to sit
    beside the four work counts: one iteration may serve several batches (a
    restart's starts plus the method's trials), so an iteration is not a batch.

    The work the run billed lives in `work` (`Work`), which is what those four
    counts are; they also read flat on the result, as its aliases.

    Two deliberate departures from `scipy.optimize.OptimizeResult`, whose
    fields these otherwise resemble: ``success``/``message`` are absent, because
    the engine has no convergence criterion of its own - apart from the caller's
    own gates (``max_iters``, ``time_limit``, ``budget_check``), it stops early
    on its own initiative only for an objective with no floor or for a flock in
    which every path stalled with restarts off - and ``f`` is the best over
    *feasible* points only.

    Example:
        Reading a finished run - the answer, then the flock, then why it
        stopped:

            >>> import jax.numpy as jnp
            >>> from soar_lite import Domain, n_path_schedule

            >>> result = n_path_schedule(
            ...     lambda x: (jnp.sum(x**2, axis=1), 2.0 * x, None),
            ...     n_paths=3,
            ...     domain=Domain([(-4.0, 4.0), (-4.0, 4.0)]),
            ...     x0=jnp.full((3, 2), 2.0),
            ...     max_iters=20,
            ... )

            >>> result.status              # why it stopped
            'max_iters'
            >>> float(result.f)            # the answer
            0.0
            >>> result.flock_x.shape, result.flock_f.shape
            ((3, 2), (3,))
            >>> int(result.n_iters)        # iterations, not batches
            20

        ``x`` and ``f`` are the best recorded feasible point, which is not
        necessarily a row of ``flock_x``: the path that found it may have been
        restarted afterwards. Read ``flock_*`` for "where did each path end up"
        and ``x``/``f`` for the answer. The cost side is `work`, which counts
        points and batches per seam; the four flat counts on the result are its
        aliases.

    """

    x: jnp.ndarray
    """The argmin in the coordinates the engine was handed, shape ``(d,)``.

    Not necessarily a row of `flock_x`: the best point may come from a path that
    was restarted afterwards. All ``nan`` when no feasible point with a *finite*
    value was ever served - the ``nan`` reading of ``f``. A feasible ``-inf``
    names its point instead: ``x`` is that point and ``f`` is ``-inf``. So
    ``x`` is ``nan`` only when no feasible point carried a finite value and none
    priced the objective at ``-inf``.
    """

    f: float
    """`x`'s objective value, the minimum over finite feasible recorded evals.

    ``nan`` when no feasible point carried a finite value and none evaluated to
    ``-inf``; ``-inf`` when a feasible point evaluated to it.
    """

    flock_x: jnp.ndarray
    """Each path's held coordinates as the run ended, shape ``(n_paths, d)``.

    The coordinates each path actually holds; rows are neither filtered nor
    sorted. A path that never held a usable point has an all-``nan`` row.
    """

    flock_f: jnp.ndarray
    """Each path's held value as the run ended, shape ``(n_paths,)``.

    ``nan`` for a path that never held a usable point.
    """

    work: Work
    """The work the run billed, per seam and per axis; see `Work`.

    The single home of the four counts: the `n_f_g_batches`, `n_f_g_points`,
    `n_f_batches` and `n_f_points` attributes below are read-only aliases
    for its fields.
    """

    n_iters: int
    """Completed engine iterations; iteration 0 is the flock's placement.

    A half-finished iteration is not counted: neither one a closed gate cut short
    at a batch, nor the one that ended the run (``unbounded``, ``all_stalled``).
    """

    status: NPathStatus
    """Why the run stopped; see `NPathStatus`."""

    @property
    def n_f_g_batches(self) -> int:
        """Fused batches served; `Work.n_f_g_batches`, read flat."""
        return self.work.n_f_g_batches

    @property
    def n_f_g_points(self) -> int:
        """Fused points served; `Work.n_f_g_points`, read flat."""
        return self.work.n_f_g_points

    @property
    def n_f_batches(self) -> int:
        """Value-only batches served; `Work.n_f_batches`, read flat."""
        return self.work.n_f_batches

    @property
    def n_f_points(self) -> int:
        """Value-only points served; `Work.n_f_points`, read flat."""
        return self.work.n_f_points


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
    batch, which would make each row's gradient pass through the sum and so
    depend on every other row.

    Feasibility is vmapped on its own because it cannot ride inside
    `value_and_grad`: that transformer differentiates a scalar output, so a
    function returning ``(f, feas)`` would be differentiated as a pair. Keeping
    `feasible` a separate point predicate leaves both halves scalar.

    The pair comes back fused-first, matching `n_path_schedule`'s two evaluation
    seams: ``eval_f_g=`` takes the first, ``eval_f=`` the second. Both are
    lazy closures, so building them runs nothing, and a caller who wants only
    the fused seam discards the other with ``eval_f_g, _ = pointwise(objective)``.

    The value-only half is the fused objective's own forward pass,
    ``vmap(objective)`` - the same forward computation, so returning it costs
    nothing and buys the engine's cheap-trial path. That does not make this the
    evaluation seam for a *value-only* objective: a function JAX cannot
    differentiate cannot be `objective` here at all, so a caller holding values
    and no gradient needs a value-only adapter of its own.

    Args:
        objective: the per-point objective, ``(d,) -> ()``; a function
            `jax.value_and_grad` can differentiate, which is what makes the
            fused half exist.
        feasible: the per-point feasibility predicate, ``(d,) -> ()`` of a
            boolean; None (the default) marks every point viable, the same
            `None` both evaluation seams accept.

    Returns:
        The two evaluation seam shapes `n_path_schedule` takes, as a pair.
            ``eval_f_g`` a fused ``(f, g, feas)`` batch, ``eval_f`` its
            value-only ``(f, feas)`` sibling.

    Example:
        Both seams for an objective restricted to the half-plane ``x[0] >= 0``:

            >>> import jax.numpy as jnp
            >>> from soar_lite import pointwise

            >>> eval_f_g, eval_f = pointwise(
            ...     lambda x: jnp.sum((x - 2.0) ** 2),
            ...     feasible=lambda x: x[0] >= 0.0,
            ... )

            >>> x = jnp.array([[1.0, 1.0], [-3.0, 1.0]])
            >>> f, g, mask = eval_f_g(x)

            >>> print(f, mask)          # the second row is infeasible
            [ 2. 26.] [ True False]
            >>> print(g[0])             # per row, 2*(x - 2)
            [-2. -2.]
            >>> print(eval_f(x)[1])     # the value-only mask agrees
            [ True False]

        The mask is a boolean because `feasible` returns one; 0/1 is the other
        spelling both seams accept. Nothing here drops the infeasible row - it
        was evaluated, and its value is what priced it - so the run reads the
        mask when it decides what to report.

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
