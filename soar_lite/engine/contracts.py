"""The plugin contracts, the data they exchange, and the evaluation seams.

`LocalSolver` and `GlobalModel` are the two seams a caller plugs into;
`PathPoint`, `Verdict` and the evaluation-seam aliases are what crosses them; and
`NPathResult` with `NPathStatus` is the other half of the same exchange, what the
caller reads back. `pointwise` builds both evaluation seams from one per-point
objective.

The machine depends on these types, never the reverse: nothing here imports
`soar_lite.engine.schedule`. The package facade re-exports both halves, so a
caller still names any of them from `soar_lite.engine`, and this module stays the
half a plugin author reads.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import jax
import jax.numpy as jnp

from soar_lite.domain import Domain

Verdict: TypeAlias = Literal["retry", "accept", "give_up"]
"""How a judged step rules one served candidate.

``"retry"`` spends another trial, ``"accept"`` ends the enumeration and commits
the step's argmin - the lowest-value finite trial among all the step has served
so far, the candidate just served for a single-sample step - and ``"give_up"``
abandons the path's enumeration outright, so the path stalls. ``"accept"``
commits nothing when the step served no finite trial or the winner fails the
engine's finite check: the path stalls instead. See `LocalSolver.verdict`, which
returns it.
"""


@dataclass(frozen=True)
class PathPoint:
    """One path's point: coords x, value f, gradient g = ∂f/∂x.

    The engine builds these and the local method receives them; a caller
    normally reads the run's outcome through `NPathResult` instead of
    constructing one. A point is a candidate until `usable` passes, so a
    non-finite half is allowed on a point that has not been committed.
    """

    x: jnp.ndarray
    """Coordinates, shape ``(d,)``, in the run's coordinates."""

    f: float
    """The objective's value there. Finite for any point a path actually holds
    (``usable`` gates the commit); a point still being considered may carry a
    non-finite value."""

    g: jnp.ndarray
    """The gradient there, shape ``(d,)``; finite under the same ``usable`` gate
    as ``f`` - held points are finite, candidates need not be yet."""

    @property
    def usable(self) -> bool:
        """Return whether this point may be committed as a path's point.

        Both halves must be finite, because both are carried forward: a
        non-finite f cannot be reported, re-started from or finished on, and a
        non-finite g is proposed from and folded on, so one NaN there poisons
        that coordinate's magnitude EMAs for the rest of the run. f is asked
        first, so a dead value skips g's finite scan.
        """
        return math.isfinite(self.f) and bool(jnp.all(jnp.isfinite(self.g)))


class GlobalModel(Protocol):
    """The engine's restart-point source (its second plugin point).

    The model answers the two questions the engine cannot: *where* a restart
    lands (`propose`) and *what the run has learned* (`observe`). The engine
    feeds it every batch it serves and asks for points only when a restart
    fires, so the model never has to seed a run itself: a model that wants more
    evidence than it has can answer `propose` with a plain draw and the engine
    has no opinion.

    A model carries no search space of its own. The run has one box, the engine
    owns it, and `propose` is handed that `Domain` at every ask - so a model must
    not remember one. That is what keeps a fitted geometry and a fallback draw
    in the same coordinate unit as the clip, with no second box to reconcile.

    One instance serves the whole run (like `LocalSolver`), so whatever it
    accumulates lives inside it and the engine never reads it. The methods are
    called by the engine on its own schedule - `observe` once per served batch,
    `propose` once per cohort of restarting paths (all the paths that restart in
    one iteration, in one call) - so they trust their arguments: this Protocol
    is the whole contract.

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

        Nothing is filtered for feasibility: the model fits the raw objective
        everywhere, and only the value the run reports is filtered. Batches the
        run never decides on arrive too (a trial that lost its step, the batch
        that ended an unbounded run), because the work was served.
        """
        ...

    def propose(self, n_points: int, domain: Domain) -> jnp.ndarray:
        """Return ``n_points`` fresh restart starts, shape ``(n_points, d)``.

        `n_points` is how many paths restart this iteration - on iteration 0,
        every path when the caller handed no `x0` - and `domain` is the box the
        answer must land in and the one statement of the dimension. The answer
        is in the coordinates the objective takes. The engine still clips it into
        that box, as a defensive clip for a model that oversteps, not as a way
        of reconciling two boxes. It checks the shape at the batch boundary and
        raises ``ValueError`` on a wrong one. A non-finite row is allowed: it is
        an unusable start, so the engine stalls that path rather than placing
        it.

        The model owns its randomness (an instance holds its key), which is what
        makes a seeded model reproducible.
        """
        ...


class LocalSolver(Protocol):
    """The search method the n-path engine drives (its first plugin point).

    The engine owns the schedule - batch composition, the clip into `domain`,
    the shared horizon, the n-path state machine (restart/step/stall), the best
    `f⋆` - and the method owns the search: direction, step size, acceptance,
    **and how long a step's enumeration may run**. The engine keeps asking a
    step for proposals while ``verdict`` answers "retry", and ``verdict`` must
    eventually reach "accept" or "give_up" on its own; an enumerator with no
    bound of its own is a hang, not a slow run. Runs end on the caller's horizon,
    not on the method.
    ``needs_verdict(i)`` picks the step's shape: a method that rules on each
    candidate says True, one whose step always commits its single proposal says
    False. All state is per path, lives inside the instance, and is opaque to the
    engine - implementers index it by path i.

    A path's point reaches the method in exactly two places: ``restart(i, p)``,
    which places it, and ``fold(i, p)``, which moves it. Both are the engine's
    own commit points, so a method that keeps the point it was handed holds the
    one the engine holds. A method that steps from a point must keep it, because
    no later call hands it back: ``propose(i)`` reads the current point off the
    method's own state and answers from it, repeatedly for one step - the engine
    commits a point only on "accept", so nothing moves under a step's
    enumeration. The candidate the engine serves is the clipped one, so a method
    judges the point it is handed, not its own proposal. verdict(i, x, f) is
    then called once per served candidate, with (x, f) the served point's coords
    and value - never its gradient: with `eval_f` connected the engine serves a
    candidate's value alone and buys the gradient only for the step's winner, so
    a ruling rests on f and on the method's own point, whose gradient a start
    always has. "accept" commits the lowest-f finite trial served this step - the
    engine's per-step argmin - and calls ``fold(i, p)`` with the committed
    winner, which is the method's
    only place to fold an accepted step into its model (the engine owns the
    argmin, so a method that tracked its own copy would drift from it). A step
    that served no finite trial, or whose winner fails the engine's finite
    check, stalls the path instead and ``fold`` is never called. "retry"
    spends another trial, "give_up" stalls the path at once.

    Arbitration is split: ``verdict`` rules the enumeration and may only mutate
    enumeration-local state (a step length, a trial counter), while ``fold``
    alone mutates the model, so a method folds exactly once per committed step.

    These methods are called by the engine on its own schedule - `restart` on
    every placement, `needs_verdict` once per path per iteration, `fold` once
    per committed step, `propose` and `verdict` once per trial - so they trust
    their arguments: this Protocol is the whole contract.
    """

    def restart(self, i: int, p: PathPoint) -> None:
        """Seed path i's state from a fresh start point (its one cold start).

        Called by the engine each time it places a path on a point: once at
        initialization, and again on every redraw of a stalled path, which
        needs `restart_stalled` on. The engine has one way to place a path, so
        a method need not tell "first" from "again" - both drop its per-path
        state to the cold one. One of the two calls that hand a point over, so
        a method that steps from a point keeps this one.
        """
        ...

    def needs_verdict(self, i: int) -> bool:
        """Report whether path i's next step is ruled candidate by candidate.

        Asked once per path per iteration, before ``propose``, so the point it
        can see is the one the method holds - its current point, whose gradient
        a path always has. A method with a stopping test also speaks here:
        answering True is how it gets the one ``verdict`` call that can carry
        "give_up", since a step it did not ask to rule is committed as
        proposed. That is why a converged path answers True even when its step
        has a single candidate and no grid to compare.

        ``True`` is the enumerated kind: the engine asks ``verdict`` what to do
        with each served trial - served for its value alone when `eval_f` is
        connected, fused with its gradient otherwise. ``False`` promises
        the step commits its one proposal unjudged: the engine serves that
        proposal's value together with its gradient, in one fused batch, and
        gates the winner itself - a value or gradient that is not finite stalls
        the path instead of being committed. Nothing here grades the proposal;
        the gate is the whole of its acceptance.

        ``True`` costs a gradient *per trial* where ``False`` costs one per step,
        unless the caller handed an `eval_f` seam, whose trials are served by
        value and whose winners' gradients ride one fused batch per iteration.
        How many trials a step runs is the method's business, so that billing is
        its choice too.
        """
        ...

    def propose(self, i: int) -> jnp.ndarray:
        """Return path i's raw next candidate, shape ``(d,)``, from its point.

        The point is the method's own: ``restart`` placed it and ``fold`` moved
        it, and no call re-passes it, so a method that steps from a point keeps
        it. Raw: the engine clips it into `domain` before serving, so a method
        needs no box of its own and never reconciles two of them.
        """
        ...

    def verdict(self, i: int, x: jnp.ndarray, f: float) -> Verdict:
        """Rule the step's enumeration from the served trial's coords and value.

        `x` is the served (clipped) point and `f` its value - a Python float,
        not a JAX scalar, because every served batch is pulled to the host once.
        The trial's own gradient is deliberately absent, because the engine may
        have served the value alone and left the gradient to be bought for the
        winner; the point the step starts from is the method's own, and it no
        longer has to be passed. Only asked for a path that answered True to
        ``needs_verdict``, so a step that commits its one proposal unjudged
        never reaches here.

        The return is an enumeration decision, not a verdict on the trial's
        quality: "retry" asks for another candidate (the method typically
        shrinks a step length or bumps a trial counter first); "accept" ends the
        enumeration and commits the engine's per-step argmin (this trial, for a
        single-sample method), which is then folded through ``fold`` - unless no
        trial of the step was finite, or the winner fails the engine's finite
        check, in which case the path stalls and nothing is folded;
        "give_up" aborts the path's enumeration outright. The engine keeps
        enumerating while this says "retry", so one of the two terminal answers
        is the method's responsibility - there is no engine-side trial cap.
        """
        ...

    def fold(self, i: int, p: PathPoint) -> None:
        """Fold the step the engine just committed into the model: the path is at p.

        The engine's only call site is the "accept" commit, with p = the
        per-step argmin it committed, so the folded secant is always the
        accepted step and can never disagree with it. It is never called for a
        step the engine did not commit. `p` means what it means in ``restart``:
        the point the path now stands at. The point it stepped from is the
        method's own record, not an argument - the same rule that makes
        ``propose`` read its point instead of being handed it.
        """
        ...


EvalFG: TypeAlias = Callable[
    [jnp.ndarray],
    "tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]",
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

EvalF: TypeAlias = Callable[[jnp.ndarray], "tuple[jnp.ndarray, jnp.ndarray | None]"]
"""The objective's value half alone, for a caller that also holds one.

Same ``x`` rows in, ``(f: (M,), feas: (M,) | None)`` out. Feasibility is
included because it decides the value the run reports and whether a ``-inf``
ends the run, neither of which ``f`` alone can tell. Opting in lets a judged
step serve its trials for their values alone and buy the winners' gradients
afterwards, together in one fused batch.
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
(a feasible point evaluated to ``-inf``) and ``"all_stalled"`` (no
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
    axes that never mix. Which evaluation seam served the work decides the pair:
    ``n_f_points`` and ``n_f_batches`` for the value-only seam, ``n_f_g_points``
    and ``n_f_g_batches`` for the fused one. Points are the serial cost and
    batches the parallel wall-clock cost. Which axis a comparison reads is the
    caller's business; a run handed no ``eval_f`` seam reports zero on both f
    counts (``n_f_points`` and ``n_f_batches``).

    Two deliberate differences from `OptimizeResult`: ``success``/``message`` are
    absent, because the engine has no convergence criterion of its own - apart
    from the caller's own gates (``max_iters``, ``time_limit``,
    ``budget_check``), it stops early on its own initiative only for an
    objective with no floor or for a flock in which every path stalled with
    restarts off - and ``fun`` is the best over *feasible* points only.
    """

    x: jnp.ndarray
    """The argmin in the coordinates the engine was handed, shape ``(d,)``.

    Not necessarily a row of `flock_x`: the best point may come from a path that
    was restarted afterwards. All ``nan`` when nothing feasible was ever served,
    the ``nan`` reading of ``fun``. A feasible ``-inf`` names its point instead:
    ``x`` is that point and ``fun`` is ``-inf``, so ``x`` is ``nan`` only when no
    feasible point was served at all.
    """

    fun: float
    """`x`'s objective value, the minimum over finite feasible recorded evals.

    ``nan`` when nothing feasible was ever served, ``-inf`` when a feasible
    point evaluated to it.
    """

    flock_x: jnp.ndarray
    """Each path's held coordinates as the run ended, shape ``(n_paths, d)``.

    The coordinates each path actually holds; rows are neither filtered nor
    sorted. A path that never held a usable point has an all-``nan`` row.
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
    """Completed engine iterations; iteration 0 is the flock's placement.

    A half-finished iteration is not counted: neither one a closed gate cut short
    at a batch, nor the one that ended the run (``unbounded``, ``all_stalled``).
    """

    status: NPathStatus
    """Why the run stopped; see `NPathStatus`."""


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
