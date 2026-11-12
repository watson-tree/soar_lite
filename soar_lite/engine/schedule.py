"""The schedule: the loop, its gates, and the state it drives.

`n_path_schedule` validates a run's arguments, drives the loop and assembles the
report. A run's state lives in four private holders, one per state family:
`_EvalSeams` (the two seams and what each charges), `_Paths` (the n-path state
machine), `_Best` (the run's best, `f⋆`), and `_Meter` (the stop meter and the
work report).

The types it drives come from `soar_lite.engine.contracts`, and the two concrete
defaults are bound lazily inside `n_path_schedule`, which keeps that graph
acyclic.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from soar_lite.domain import Domain
from soar_lite.engine.contracts import (
    EvalF,
    EvalFG,
    GlobalModel,
    LocalSolver,
    NPathResult,
    NPathStatus,
    ObserveFn,
    PathPoint,
    Point,
    ProposeFn,
    ValuePoint,
    Work,
)


def n_path_schedule(  # noqa: PLR0913 (the caller's knobs, all keyword-only
    # after the one positional argument: one name per dial beats a config object here)
    eval_f_g: EvalFG,
    *,
    # The run's width: how many paths the engine drives.
    n_paths: int,
    # The run's box, a `Domain`: every point served lies inside it - the engine
    # checks each seam's answer against it and refuses the ones that miss - and
    # it is the one statement of the dimension (`domain.d`) and of each
    # coordinate's prior `scale`.
    domain: Domain,
    # The caller's flock: one start per path, ``(n_paths, domain.d)``, placed
    # by iteration 0 exactly where the caller asked rather than drawn.
    x0: jnp.ndarray | None = None,
    eval_f: EvalF | None = None,
    # The run's horizon, in iterations: one pass of the loop, not one eval.
    max_iters: int | None = None,
    # A wall-clock deadline in seconds. A run needs a horizon: at least one of
    # max_iters / time_limit must be given.
    time_limit: float | None = None,
    # Keyword-only and defaulted, like scipy's `callback`: an extra gate for a
    # budget the engine cannot see (points, dollars, an external clock). A batch
    # is charged only if the gate allows it, so the count it gates on is the one
    # the report will carry once it says yes.
    budget_check: Callable[[Work], bool] | None = None,
    # Whether a stalled path is re-drawn. The stall itself is the engine's own
    # bit (`_Slot.is_stalled`); this knob only decides whether the model is then
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
    """Advance a flock of paths in x from the caller's initial points.

    The engine owns the schedule: it checks every point against `domain` and
    refuses the ones that miss, serves each batch as one call through the seam it
    came from - `eval_f_g` for a fused batch, `eval_f` for a value-only one -
    buys the iteration's deferred gradients in one batch, keeps the n-path state
    machine, and owns the best `f⋆` (`f` in the result). Two things are injected,
    named `global_model` and `local_solver`, in that order wherever they appear:
    `global_model` supplies **where** a restart lands (one point per restarting
    path, inside the box) and is also the run's evidence sink, while
    `local_solver` owns direction, step size, acceptance, and the enumeration
    inside a step, keeping its per-path state inside itself. `restart_stalled`
    is one flag on top of them:
    **when** a path stalls is the local solver's own stopping test or the finite
    commit gate refusing its point, **whether** it is then re-drawn is
    `restart_stalled`, and **where** the new start lands
    is `global_model.propose`, so the engine needs no restart rule of its own.
    The flag is on by default, because with restarts off, a stalled path just
    idles.

    `n_paths` is the run's width and nothing else; the dimension is `domain.d`,
    so a run never spells it twice. Iteration 0 needs no special case in the move
    loop: no path holds a point yet, and a path without a point restarts, so the
    first move places the whole flock as one fused batch - which is why the run's
    schedule sees iteration 0 as the flock's placement. It is `x0` where one was
    handed
    in, and otherwise one start per path from the global model. A start whose
    served value or gradient comes back non-finite is never placed: the path
    keeps the point it held - none at iteration 0 - and is marked stalled.
    `restart_stalled` governs a path that holds a point, so an unplaceable flock
    is asked for a fresh start every iteration whatever the flag says.

    `domain` is the run's box and the coordinate space the engine works in: the
    engine tests every point against it and knows nothing about any map from
    these coordinates to the caller's physical ones. It is also what the default
    model draws from, and the only statement of the dimension. With
    `global_model=None` the engine builds a `SampleGlobalModel` and lets the
    domain draw (`Domain.draw`); `key`
    seeds those draws, and an injected model owns its randomness instead.

    The run's own stop is the horizon: at most `max_iters` iterations, or the
    wall-clock deadline when `time_limit` is given. Iterations are the unit on
    purpose - one iteration is one pass of the loop, so the same iteration count
    reproduces the same run on any machine - but it is not a cost bound: one
    iteration serves one or more batches, so migrating a batch budget to
    `max_iters` can multiply the work done. `budget_check` is the caller's extra
    stop, asked of the engine's own ledger rather than kept beside it, and it is
    additional: a run is complete with a horizon alone. `NPathStatus` names what
    each of these writes.

    `f` is the true minimum over every feasible eval with a finite value -
    every served point when the objective reports no mask, only the points its
    mask passes otherwise. A feasible ``-inf`` is the answer rather than a
    contender: the run ends on the spot and reports that point as ``x`` with
    ``f = -inf``. The paths themselves move on every served value, mask or no
    mask. Work is reported, never budgeted: four counts on two
    axes that do not mix - points against batches, and f against f+g - and none
    of them stops the run (see `NPathResult`).

    The caller's inputs are validated here, once per run: the horizon bounds,
    `n_paths`, and `x0`'s shape, finiteness and containment raise ValueError.
    The domain refused a malformed box when it was built. The two plugin objects
    are taken on their types - a type checker refuses a wrong one, and nothing
    here re-checks it at run time.

    Args:
        eval_f_g: the one seam a run must supply: the objective as one batched
            pass; `pointwise` builds one from a per-point scalar. Takes rows
            ``x`` of shape ``(M, d)`` and returns
            ``(f: (M,), g: (M, d), feas: (M,) | None)`` - value, gradient and
            optional feasibility mask. A mask holds 0/1 (or
            booleans) and nothing else: 0 marks a point infeasible, 1 marks it
            feasible, and any other value is refused rather than read.
        n_paths: how many paths to drive, one current point each.
        domain: the run's box, a `Domain`.
        x0: the caller's flock: one start per path, shape ``(n_paths, d)``,
            placed by iteration 0 exactly where given and never used again.
            ``None`` draws the flock from the global model like any other
            restart; with nothing observed yet, both shipped models answer that
            ask with the domain's own draw, so iteration 0 is a draw from
            `domain` when `x0` is left out. Validated at the entry: a start
            outside the declared box raises rather than clips, so a start the
            caller authored is never silently relocated. It still passes the
            same finite gate as any start, so a non-finite value or gradient
            there is refused rather than placed.
        eval_f: the objective's value half alone, same ``x`` rows, returning
            ``(f: (M,), feas: (M,) | None)``. ``None`` means every served trial
            takes its gradient in the same fused pass; supplying it lets a
            method that enumerates its step serve trials for their values alone
            and buy the winners' gradients afterwards, together in one fused
            batch. That same final batch also buys the value and gradient of a
            winner the method never read, and of a redraw start: both are
            committed without anyone reading a value, so neither is served
            before it.
        max_iters: the run's horizon in iterations (``None`` leaves it to
            ``time_limit``); a run needs at least one of the two. ``0`` is
            legal: the loop's first stop query holds, so nothing is placed,
            ``n_iters`` is ``0`` and every ``flock_x`` row and every ``flock_f``
            entry is ``nan``.
        time_limit: a wall-clock deadline in seconds (``None`` means none),
            asked at the loop head and again before every batch, so a method
            that enumerates is cut at its next trial rather than at the
            iteration's end. The question comes before the batch is served, so
            a single long call can still take the run past the deadline by one
            batch.
        budget_check: an extra gate for a budget the engine cannot see, asked at
            the loop head every iteration and again before each served batch.
            Both are per batch, not per iteration: the value-only batches a step
            fires through its trial hook are asked like any other, so an
            enumeration that turns ten times is asked ten times, and a refusal
            there ends the run. It receives the run's `Work`, so it gates on the
            engine's own counts rather than on a parallel tally. Before a batch
            it is handed the ledger that batch would leave, and a batch it
            refuses is never charged, so ``<=`` is the idiom on any count:
            ``work.n_f_points <= ceiling`` stops within one batch width of the
            ceiling instead of crossing it. At the loop head, with no batch
            pending, it is handed the ledger as it stands. Additional, not a
            horizon (``None`` means none).
        restart_stalled: whether a path the engine marks stalled (its method gave
            up, or its point was not usable) is re-drawn from the global model.
            ``False`` keeps a stalled path that holds a point frozen; a path
            holding no point is still re-drawn. A fully stalled flock then ends
            the run with ``status="all_stalled"``.
        global_model: the restart-point source and the run's evidence sink.
            ``None`` builds a `SampleGlobalModel`, which fits nothing and draws
            from ``domain``.
        local_solver: the local method (``None`` binds ``DiagonalPreconditioner``).
        key: the PRNG seed of the default model's draws; ``None`` uses the
            fixed default, so unseeded runs are deterministic. Ignored when
            `global_model` is supplied (that model owns its key).

    Raises:
        ValueError: At the entry, once per run: if both `max_iters` and
            `time_limit` are missing, if `max_iters` is negative or `time_limit`
            is not positive, if `n_paths` is below one, or if `x0` is not a
            ``(n_paths, d)`` finite batch inside `domain` (a start the caller
            authored raises rather than being clipped). An injected piece can
            raise the same way later, at the batch that serves it: a
            feasibility mask holding anything but 0/1, or any point a seam hands
            the run - a restart ask of the wrong shape, a trial row, a step
            winner, a start - that is not finite or lands outside `domain`.

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

            >>> result.x          # ≈ [3., 3.]
            Array([3., 3.], dtype=float32)

            >>> result.f           # exact here: the explicit x0 reaches [3., 3.]
            0.0

        A per-point scalar can be wrapped into this shape with `pointwise`.

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

    # The run's one box, and the only place it is read for the box test; the
    # dimension comes with it.
    d = domain.d
    meter = _Meter(max_iters, time_limit, budget_check)
    best = _Best(d)
    # Each component gets the one half of the model it calls, not the object: the
    # eval seams fold evidence, a path stepper asks for starts, and neither has
    # any business holding the other's method.
    eval_seams = _EvalSeams(
        eval_f_g,
        eval_f,
        domain=domain,
        meter=meter,
        best=best,
        observe=global_model.observe,
    )
    # Last, because it takes the resolved seams and their collaborators: the
    # state machine drives them rather than receiving them per call.
    paths = _Paths(
        n_paths,
        domain,
        eval_seams,
        global_model.propose,
        local_solver,
        restart_stalled=restart_stalled,
        x0=x0,
    )

    status: NPathStatus | None = None
    while True:
        # The run's one stop query names its reason, asked before a new iteration
        # starts so the run never begins one it cannot finish. A refusal
        # mid-iteration arrives as _BatchRefused carrying the same reason.
        status = meter.stop()
        if status is not None:
            break
        try:
            # Decide which paths restart (iteration 0: all of them, since no
            # path holds a point), decide every move on values, then buy the
            # gradients those decisions need in one batch. A refused batch
            # raises `_BatchRefused` here, before anything is placed.
            paths.move()
        except _BatchRefused as exc:
            # The iteration asked for a batch it could not afford and was cut
            # short: it did not finish, so it is not ticked.
            status = exc.reason
            break
        except _OptimumReached:
            # A served feasible point priced f at -inf. Nothing to unwind and
            # nothing left to decide: the batch that found it was paid for and
            # is counted, but the iteration did not finish, so it is not ticked
            # either.
            status = "unbounded"
            break
        except _AllStalled:
            # Every path is stalled and none restarts: nothing left to move, and
            # no batch was served this iteration, so it is not ticked.
            status = "all_stalled"
            break
        meter.tick()  # counted after the body: only finished iterations

    # Each path's point as the run ended, and its value.
    flock_x, flock_f = paths.held()

    return NPathResult(
        x=best.x,
        f=best.f,
        flock_x=flock_x,
        flock_f=flock_f,
        work=meter.work,
        n_iters=meter.n_iters,
        status=status,
    )


class _BatchRefused(Exception):  # noqa: N818 - a signal, not an error
    """A serve was refused because a stop had closed, the reason carried along.

    Raised only inside `n_path_schedule`, by `_Meter.admit` on the way into a
    batch, and caught by the loop. It carries the reason `_refusal` answered at
    the batch altitude, so the loop names the run's status without re-asking the
    query.
    """

    reason: NPathStatus  # the gate `stop` named, carried to the loop's one catch

    def __init__(self, reason: NPathStatus) -> None:
        self.reason = reason
        super().__init__(reason)


class _OptimumReached(Exception):  # noqa: N818 - a signal, not an error
    """A served *feasible* point priced the objective at `-inf`.

    Raised inside `_Best.observe`, the single funnel both evaluation seams pass through,
    and caught by the loop. Unlike `_BatchRefused` it arrives after a batch
    was served and charged, and it leaves an answer behind instead of naming a
    gate: the run stops on that point, so `f` is `-inf` and `status` is
    `"unbounded"`. Nothing is left to unwind - a batch raises before any point is
    placed or advanced.
    """


class _AllStalled(Exception):  # noqa: N818 - a signal, not an error
    """Every path has stalled and the run cannot re-draw any of them.

    Raised inside `_Paths.move` when no path restarts and every path is stalled
    - only reachable with `restart_stalled=False`, since a stalled path restarts
    otherwise - or when every path that was going to restart was declined when
    nothing else was stepping. Nothing can un-stall a path but
    `_Paths._place(restart=True)`, which only ever sees the restarting paths, so
    the state is terminal: the iteration would serve nothing, and repeating it
    would idle out the horizon. The run stops on it, naming the state in
    `status`.
    """


class _Meter:
    """The run's stop meter and work report.

    Three predicates decide whether work may go on: the iteration horizon
    (`max_iters`), the wall-clock deadline (`time_limit`), and the caller's
    extra gate (`budget_check`), which reads the run's `Work`. Two questions put
    them to use: `stop` asks them of the run's ledger at the loop head, `admit`
    of the ledger its batch would leave. At least one of the first two is
    required at entry: they advance on their own and the engine can see them do
    it, while a `budget_check` is the caller's predicate, so the engine cannot
    tell a gate that closes from one that never does.

    The counters are pure reports, not caps, and not one of them gates: the
    fused pass counts on its pair (`n_f_g_batches` / `n_f_g_points`, one batch
    per value+gradient pass) and the value-only pass the same way on its own
    (`n_f_batches` / `n_f_points`). `_refusal` reads only the run's three
    predicates, so a counted batch can still be refused by the gate like any
    other serve. That refusal is a signal (`_BatchRefused`), never a flag
    stored here. The counts live in `work` (`Work`), frozen and replaced on each
    charge.

    `admit` charges a batch only after the gate allows it, and hands the gate
    the ledger that batch would produce, not the one before it: a ceiling on
    points is then one a run cannot cross, since a batch is served whole or not
    at all. What a refusal was decided on never enters the ledger, so the report
    carries exactly the counts the last allowed batch left.
    """

    def __init__(
        self,
        max_iters: int | None,
        time_limit: float | None,
        budget_check: Callable[[Work], bool] | None,
    ) -> None:
        self._max_iters = max_iters
        self._deadline = None if time_limit is None else time.monotonic() + time_limit
        self.work = Work(n_f_g_batches=0, n_f_g_points=0, n_f_batches=0, n_f_points=0)
        self.n_iters = 0
        self._gate = budget_check

    def stop(self) -> NPathStatus | None:
        """Name the reason work may not go on, or None while it may.

        The iteration-altitude question, asked at the loop head before every
        iteration against the ledger as it stands. `admit` asks the same three
        predicates of the ledger its batch would leave.

        Returns:
            The closed gate's reason, or None.

        """
        return self._refusal(self.work)

    def admit(self, batch_size: int, *, fused: bool) -> None:
        """Charge one batch the gate allows, or refuse it before it is served.

        The gate is asked for the ledger the batch would leave, before anything
        is charged: a batch is served whole or not at all, and the ledger it
        judged is the one committed rather than a second derivation of it. The
        projection never escapes this method, so no caller ever holds a charge
        a refusal threw away.

        Args:
            batch_size: Points in the batch about to be served.
            fused: Which pair of counters the calling seam moves.

        Raises:
            _BatchRefused: When a stop names a closed gate.

        """
        after = self._after(batch_size, fused=fused)
        reason = self._refusal(after)
        if reason is not None:
            raise _BatchRefused(reason)
        self._book(after)

    def tick(self) -> None:
        """Record one completed iteration. The caller ticks only finished ones."""
        self.n_iters += 1

    def _after(self, batch_size: int, *, fused: bool) -> Work:
        """Project the ledger past one batch, charging nothing.

        Args:
            batch_size: Points in the batch about to be served.
            fused: Which pair the batch would move, decided by the seam that
                would serve it.

        Returns:
            The projected `Work`.

        """
        work = self.work
        if fused:
            return replace(
                work,
                n_f_g_batches=work.n_f_g_batches + 1,
                n_f_g_points=work.n_f_g_points + batch_size,
            )
        return replace(
            work,
            n_f_batches=work.n_f_batches + 1,
            n_f_points=work.n_f_points + batch_size,
        )

    def _book(self, work: Work) -> None:
        """Charge a batch the gate allowed: the ledger becomes the one it judged."""
        self.work = work

    def _refusal(self, work: Work) -> NPathStatus | None:
        """Name the first closed gate for a ledger, or None while work may go on.

        The horizon outranks the clock, which outranks the caller's gate.
        """
        if self._is_full():
            return "max_iters"
        if self._is_expired():
            return "time_limit"
        if self._is_gated(work):
            return "budget_exhausted"
        return None

    def _is_full(self) -> bool:
        """Whether the iteration horizon is reached."""
        return self._max_iters is not None and self.n_iters >= self._max_iters

    def _is_expired(self) -> bool:
        """Whether the wall-clock deadline has passed."""
        return self._deadline is not None and time.monotonic() >= self._deadline

    def _is_gated(self, work: Work) -> bool:
        """Whether the caller's extra gate refuses the ledger it is handed."""
        return self._gate is not None and not self._gate(work)


class _Best:
    """The best the run keeps: the minimum over feasible points served.

    `f`/`x` are the answer the run reports, defined over feasible designs only
    - with no mask, every point is feasible. A NaN or +inf trial neither becomes
    `f⋆` nor displaces one: it carries no value to compare. A feasible -inf is
    not an improvement to weigh but the answer itself, so it ends the run
    instead of entering a comparison. It starts at `nan` and stays there until a
    finite feasible value arrives: `nan` is the "no answer" state, which leaves
    `inf` free to mean a value an objective actually took.
    """

    def __init__(self, d: int) -> None:
        # d is the flock's business (it is `x0`'s second axis), so the
        # placeholder is already the right shape.
        self.x = jnp.full(d, math.nan)
        self.f = math.nan  # "no answer yet", not `inf`: a value may take it

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
            self.f = -math.inf
            self.x = x_batch[j_floor]
            raise _OptimumReached
        pos = _argmin_finite([f_vals[j] for j in candidates])
        if pos is None:  # nothing feasible served, or all of it non-finite
            return
        j = candidates[pos]
        # nan means no answer yet, so the first finite feasible value lands
        # unconditionally; after that one must beat the running best.
        if math.isnan(self.f) or f_vals[j] < self.f:
            self.f = f_vals[j]
            self.x = x_batch[j]


class _EvalSeams:
    """The run's evaluation seams, and what they charge and observe.

    Every batch goes through two edges, each written once here: `_Meter.admit`
    charges it or refuses it before it is served (so a batch is served whole or
    not at all, and a refused one is neither charged nor paid for), and
    `_report` tells the run's two readers what it returned. The seam that served
    the batch decides which pair of counters moves - `_serve_f` the f pair,
    `_serve_f_g` the fused pair - and no count ever closes the gate.

    Whichever half served a batch reports it to the two readers of the run's
    evidence: the best (`_Best`, which filters for feasibility) and the global
    model (which does not). That is the only place either is fed, so a batch
    nobody decides on is still reported. `settle` lives here too: it is where an
    iteration's deferred gradients are bought together, since this object owns
    both the seam and the meter.
    """

    def __init__(  # noqa: PLR0913 - the seams' collaborators, built once
        self,
        eval_f_g: EvalFG,
        eval_f: EvalF | None,
        *,
        domain: Domain,
        meter: _Meter,
        best: _Best,
        observe: ObserveFn,
    ) -> None:
        self._eval_f_g = eval_f_g
        self._eval_f = eval_f
        self._domain = domain
        self._meter = meter
        self._best = best
        self._observe = observe

    def serve_trials(self, x_batch: jnp.ndarray) -> list[ValuePoint]:
        """Serve one step's trial batch, as one produced point per row.

        The batch is a method's own candidate grid, so it is checked like every
        other point the run holds: a row that is not finite or lies outside the
        box is refused, naming the row, before anything is served. A method that
        rules its trials over a box serves trials from inside it.

        A trial's gradient is only wanted for the step's winner, so with the
        value-only seam each trial is served for its value alone and the winners are
        settled once, together (`settle`). Without it a fused pass is the
        only way to a value, so every trial is served as a `PathPoint` whether
        or not the method's own enumeration keeps the gradient, and `settle`
        then buys nothing.
        """
        x_batch = jnp.asarray(x_batch)
        if x_batch.ndim != 2 or x_batch.shape[1] != self._domain.d:
            msg = (
                f"a trial batch must be a (n, {self._domain.d}) array, "
                f"got shape {x_batch.shape}"
            )
            raise ValueError(msg)
        x_batch = _check_inside(
            x_batch, what="local_solver's trial batch", domain=self._domain
        )
        if self._eval_f is not None:
            return [
                ValuePoint(row, f)
                for row, f in zip(x_batch, self._serve_f(x_batch), strict=True)
            ]
        f_vals, g_batch = self._serve_f_g(x_batch)
        return [
            PathPoint(x_batch[pos], f_vals[pos], g_batch[pos])
            for pos in range(len(f_vals))
        ]

    def settle(self, decided: dict[int, Point]) -> dict[int, PathPoint]:
        """Buy whatever gradient the iteration's points still owe, in one batch.

        Everything still short of a gradient goes out as a single batch, so an
        iteration costs one gradient pass however many moves it deferred - its
        restart starts plus its step winners - instead of one pass per trial. A
        point that already carries its gradient is passed on as it stands, so a
        run that owes nothing buys nothing and the counters do not move. Keyed by
        path on both sides, so a caller's halves come back keyed rather than
        re-paired by position.

        The gradient is only sold as the fused pair, so a `ValuePoint` is read
        twice - the value that rides along with the gradient is the one it
        already reported, from the same function at the same point.
        """
        owed = [i for i, point in decided.items() if not isinstance(point, PathPoint)]
        bought: dict[int, PathPoint] = {}
        if owed:
            f_vals, g_batch = self._serve_f_g(
                jnp.stack([decided[i].x for i in owed]),
            )
            bought = {
                i: PathPoint(decided[i].x, f_vals[pos], g_batch[pos])
                for pos, i in enumerate(owed)
            }
        return {
            i: point if isinstance(point, PathPoint) else bought[i]
            for i, point in decided.items()
        }

    def _serve_f_g(self, x_batch: jnp.ndarray) -> tuple[list[float], jnp.ndarray]:
        """Serve one fused batch: the gradient comes with the value.

        What is specific to this half is that the pass buys both, so its rows
        come back settled; the gate, the charge and the two reports are the same
        edges the value-only half goes through.

        Args:
            x_batch: The points to evaluate, shape ``(m, d)``.

        Returns:
            ``(f_vals, g_batch)``.

        Raises:
            _BatchRefused: When the gate refuses the batch.

        """
        self._meter.admit(x_batch.shape[0], fused=True)
        f_batch, g_batch, feas = self._eval_f_g(x_batch)
        # per-path bookkeeping (secant/restart) stays Python float,
        # so pull the batch values onto the host in one shot.
        f_vals = f_batch.tolist()
        self._report(x_batch, f_vals, g_batch, feas)
        return f_vals, g_batch

    def _serve_f(self, x_batch: jnp.ndarray) -> list[float]:
        """Serve one value-only batch, for a method's own enumeration.

        The gate rides along exactly as in the fused half, so a deadline that
        passes mid-step cuts the run at the next value-only batch rather than
        after a whole iteration's retries: a method's enumeration can turn many
        times without reaching the loop head or `settle`. What stays
        asymmetric is the pair it counts on - the f counters, which no count
        closes the gate - and that no gradient comes back. Only `serve_trials`
        calls this; the None-check below is for the type checker and is
        unreachable at runtime.

        Args:
            x_batch: The points to evaluate, shape ``(m, d)``.

        Returns:
            Their objective values, as host floats.

        Raises:
            _BatchRefused: When the gate refuses the batch.

        """
        eval_f = self._eval_f
        if eval_f is None:
            msg = "_serve_f requires eval_f"
            raise RuntimeError(msg)
        self._meter.admit(x_batch.shape[0], fused=False)
        f_batch, feas = eval_f(x_batch)
        f_vals = f_batch.tolist()
        # No gradient in hand: the f half alone. The same point is reported
        # again with `g` if it wins its step and `settle` buys the gradient.
        self._report(x_batch, f_vals, None, feas)
        return f_vals

    def _report(
        self,
        x_batch: jnp.ndarray,
        f_vals: list[float],
        g_batch: jnp.ndarray | None,
        feas: jnp.ndarray | None,
    ) -> None:
        """Report one served batch to the run's two readers.

        The model first: a feasible -inf below raises `_OptimumReached`, and the
        batch was served and charged either way, so the model still sees it even
        though the run keeps nothing else from that pass.

        Args:
            x_batch: The points the seam was handed.
            f_vals: Their objective values, as host floats.
            g_batch: Their gradients when the seam bought them, else None.
            feas: The feasibility mask the seam reported, or None for all-feasible.

        """
        self._observe(x_batch, f_vals, g_batch)
        self._best.observe(x_batch, f_vals, feas)


@dataclass
class _Slot:
    """One path's state: the point it holds and whether it stalled there.

    `point is None` is "no point yet" - a path the run has not placed. That is
    the machine's own state, not a stall: such a path is re-drawn whatever
    `restart_stalled` says, which is how iteration 0 places the flock and how a
    refused start is eventually re-drawn. `_place` is the one writer that clears
    `is_stalled`, so a re-drawn path cannot stay stalled.

    `x` is not gated on finiteness - `is_usable` asks f and g only - so a point
    whose coordinates are `nan` is still a point the path holds.
    """

    point: PathPoint | None = None
    is_stalled: bool = False


class _Paths:
    """The n-path state machine: one point and one stall flag per path.

    The invariants the loop's sites would otherwise carry implicitly, stated
    once here:

    - A path that restarted this iteration moved by its fresh start, so it does
      not also step and runs no trials: its method sees no step, so the restart
      is never read back as a zero-length step. `move` computes the restarting
      list once and threads it to `_restart_starts` and to the complement both
      halves share.
    - An iteration decides on values and pays for gradients once: whatever the
      value-only seam left owing a gradient - restart starts and step winners
      alike - is bought in one batch before anything is placed, so nothing is
      committed before it is settled.
    - A path is stalled when it did not move at all this iteration - its method
      gave up on every trial, or its fresh start or its winner was not usable. A
      stalled path does not step, so it serves nothing that iteration and the
      method's own state stands still.
    - A start or a winner is committed only when it is `is_usable` - finite in both
      halves - one gate defined once on the point type.
    - A path the global model declined gets no start, so it does not move this
      iteration: it keeps whatever point it holds and stays as stalled as it was.
    - A path holds a point or it does not. "No point" is a slot with no point
      on it, and a refused point is the other case: the path keeps the point it
      holds and is marked stalled, so `restart_stalled` decides whether it is
      re-drawn. Both live on one `_Slot` per path, so the two facts cannot drift
      apart.
    """

    def __init__(  # noqa: PLR0913 - the run's collaborators, built once
        self,
        n_paths: int,
        domain: Domain,
        eval_seams: _EvalSeams,
        propose: ProposeFn,
        local_solver: LocalSolver,
        *,
        x0: jnp.ndarray | None,
        restart_stalled: bool,
    ) -> None:
        self.n_paths = n_paths
        # Constant for the whole run, so it belongs beside the shape rather than
        # in `move`'s signature: every proposal below is checked against this box,
        # and this is the only copy of it - the model is handed it per ask
        # rather than carrying one.
        self.domain = domain
        # The collaborators this state machine drives. Held rather than handed
        # in per call: every method below uses at least one of them, and a step
        # would otherwise thread them only to pass them straight back.
        self.eval_seams = eval_seams
        self.propose = propose
        self.local_solver = local_solver
        # The caller's flock, handed over for the run's first move and consumed
        # there (`_restart_starts` clears it). Not a collaborator: it answers
        # exactly one ask, where a model answers every later one.
        self._x0 = x0
        # A knob, not a collaborator: *when* a path restarts is read off
        # `is_stalled` below, so the only thing left to decide is whether the
        # engine acts on it at all.
        self.restart_stalled = restart_stalled
        # One record per path, not a list per field: a path's point and its
        # stall bit are one path's state, read together by every restart
        # decision, and a step commits one path at a time (so a list, whose item
        # assignment is the commit, beats a stacked array).
        self.slots: list[_Slot] = [_Slot() for _ in range(self.n_paths)]

    def move(self) -> None:
        """Take one iteration's move for every path, then settle what it deferred.

        Iteration 0 is not a special case: no path holds a point, and a path
        with no point restarts, so the first move places the flock - the
        caller's `x0` if one was handed in, else one start per path from the
        global model - in one batch.

        The restart decision comes first: a path that holds no point, or that is
        stalled and whose run opted into `restart_stalled`, is given a fresh
        start instead of stepping. The paths that step are answered by one
        `propose` call, and the method makes the decision there - on values
        alone when the value-only seam is present, reading whatever trials it
        wants. The gradients those decisions need are then bought together in one
        batch (`settle`), so an iteration adds one gradient pass however many
        moves it deferred. A run without the seam serves everything fused and
        reaches `settle` already carrying both halves.

        The two halves touch disjoint paths, and no decision reads a point
        placed this iteration, so deciding all moves before placing any is the
        same schedule as placing each as it is decided.

        A flock with nothing left to do ends the run instead of idling: when no
        path restarts and every path is stalled, `_AllStalled` is raised before
        any batch is asked for. The same signal ends a run whose model declined
        every start it was asked about while nothing was stepping: the model
        that turns down every path is taken at its word, as a method answering
        `None` for every path would be.
        """
        restarting = [
            i
            for i, slot in enumerate(self.slots)
            if slot.point is None or (self.restart_stalled and slot.is_stalled)
        ]
        # Nothing is going to move again: no path restarts, and every path is
        # stalled. `_place` is the only writer that clears a stall bit, and no
        # stalled path is ever handed to it as a step, so an empty `restarting`
        # beside a fully stalled flock is terminal - the iteration would serve
        # nothing. Only reachable with `restart_stalled=False`; the run ends here
        # rather than idling out the horizon.
        if not restarting and all(slot.is_stalled for slot in self.slots):
            raise _AllStalled
        # Every path that did not restart and is not stalled. A stalled path is
        # left out rather than asked again: `restart_stalled` is its only way
        # back, and asking would let a method un-stall a path the engine has
        # already stopped - the two would disagree about whether it is running.
        stepping = [
            i
            for i, slot in enumerate(self.slots)
            if i not in restarting and not slot.is_stalled
        ]
        starts = self._restart_starts(restarting)
        # Every path that was going to restart was declined and nothing is
        # stepping: this iteration serves nothing, and the model that just
        # declined every path it was asked about is taken at its word - the same
        # way a method answering `None` for every path it was given stalls them
        # all. Ended here rather than idling out the horizon.
        if not starts and not stepping:
            raise _AllStalled
        winners = self._step_winners(stepping)
        # One batch over both halves, keyed by path so neither is re-paired by
        # position. A refusal (`_BatchRefused`) cuts the iteration short with
        # nothing placed - this is the only batch it serves. A point that arrived
        # already carrying both halves (a run without the seam) passes straight
        # through `settle` without being served again.
        settled = self.eval_seams.settle({**starts, **winners})
        self._place({i: settled[i] for i in starts}, restart=True)
        self._place({i: settled[i] for i in winners}, restart=False)

    def held(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Each path's held point as one row per path, `nan` where it holds none.

        Both halves read `nan` for a path the run never placed, so "no point"
        and "a point" stay one column each rather than needing a second channel
        to tell them apart. Rows are neither filtered nor sorted: row `i` is
        path `i`, whatever that path did.

        Returns:
            ``(flock_x, flock_f)``: coordinates ``(n_paths, d)`` and values
                ``(n_paths,)``.

        """
        points = [slot.point for slot in self.slots]
        flock_x = jnp.stack(
            [jnp.full(self.domain.d, math.nan) if p is None else p.x for p in points]
        )
        flock_f = jnp.asarray([math.nan if p is None else p.f for p in points])
        return flock_x, flock_f

    def _restart_starts(self, restarting: list[int]) -> dict[int, Point]:
        """Ask for one fresh start per restarting path, inside the run's box.

        The caller's flock, if it handed one, answers the run's first call and
        only that one: iteration 0 is the one move where every path holds no
        point, so `x0` is consumed here and every later restart asks the model.
        Both arrive already checked - `x0` at the entry, the model's answer just
        below - so both are placed as handed.

        Otherwise: ask, check, place. The model is handed the box its answers
        must land in and may decline a path, which leaves it out of the returned
        map: it starts nowhere this iteration, so it stays stalled where it
        stands (a path holding no point is asked again next iteration). An
        answer it does give is refused unless it is ``(domain.d,)`` and inside
        that box, so a model answering with another width, a NaN row, or a point
        past a wall fails here, naming the path and the dimension, instead of
        deep inside the iteration that would have served it. Nothing is read, so
        every start is a bare `Point` - a start has nothing to be compared - and
        `settle` buys its gradient in the same batch as the step winners, so a
        restart costs no gradient pass of its own.
        """
        if not restarting:
            return {}
        d = self.domain.d
        if self._x0 is not None:
            # The flock is one row per path in path order, so it pairs by
            # position; only the model answers per path.
            proposed, self._x0 = self._x0, None
            proposed = jnp.asarray(proposed)
            n_points = len(restarting)
            if proposed.shape != (n_points, d):
                msg = f"x0 must be a ({n_points}, {d}) array, got {proposed.shape}"
                raise ValueError(msg)
            inside = _check_inside(proposed, what="x0", domain=self.domain)
            return {i: Point(row) for i, row in zip(restarting, inside, strict=True)}
        asked = set(restarting)
        starts: dict[int, Point] = {}
        for i, answer in self.propose(restarting, self.domain).items():
            if i not in asked:
                msg = (
                    f"the global model answered for path {i}, which this "
                    f"iteration did not ask to restart (asked: {sorted(asked)})"
                )
                raise ValueError(msg)
            if answer is None:  # the model's own stopping rule: no start
                continue
            x = jnp.asarray(answer)
            if x.shape != (d,):
                msg = (
                    f"the global model's start for path {i} must be a ({d},) "
                    f"point, got shape {x.shape}"
                )
                raise ValueError(msg)
            starts[i] = Point(
                _check_inside(
                    x,
                    what=f"the global model's start for path {i}",
                    domain=self.domain,
                )
            )
        return starts

    def _step_winners(self, stepping: list[int]) -> dict[int, Point]:
        """Ask the method for its step on each stepping path, keep the winners.

        One call covers every stepping path, so a method that enumerates a step
        does so here, inside its own ``propose``. A ``None`` answer is the
        method's own stopping test firing: the path stalls where it stands, and
        nothing is served and no point is committed for it. A winner goes out as
        whatever the method read it to - a bare `Point` when it never read it, a
        `ValuePoint` when it read it through `serve_trials` - and `settle` buys
        the rest.

        A winner is checked where every other point is: one ``(d,)`` point,
        finite and inside the run's box, and the answer may name no path this
        iteration did not ask about. Those are the method's own coordinates, so
        a miss is a bug in the method and is refused here, with the path named.
        A *value* that comes back unusable is a different thing: it stalls the
        path, because a landscape nobody can evaluate is the run's own news.

        Nothing is committed here but the stall bit: a path ends this iteration
        stalled exactly when its method named no winner, or when `_place` refuses
        the winner as unusable. A stalled path is never in `stepping`: the engine
        asks nothing more of it until a restart re-draws it.
        """
        if not stepping:
            # Every path restarted this iteration: nothing steps.
            return {}
        answers = self.local_solver.propose(
            stepping, self.domain, self.eval_seams.serve_trials
        )
        d = self.domain.d
        asked = set(stepping)
        winners: dict[int, Point] = {}
        for i, answer in answers.items():
            if i not in asked:
                msg = (
                    f"local_solver answered for path {i}, which this iteration "
                    f"did not ask to step (asked: {sorted(asked)})"
                )
                raise ValueError(msg)
            if answer is None:
                self.slots[i].is_stalled = True
                continue
            x = jnp.asarray(answer.x)
            if x.shape != (d,):
                msg = (
                    f"local_solver's step on path {i} must be a ({d},) point, "
                    f"got shape {x.shape}"
                )
                raise ValueError(msg)
            x = _check_inside(
                jnp.asarray(answer.x),
                what=f"local_solver's step on path {i}",
                domain=self.domain,
            )
            # Re-seated at the checked coordinates, keeping how much of it the
            # method had read: `settle` buys what is owed by type.
            if isinstance(answer, PathPoint):
                winners[i] = PathPoint(x, answer.f, answer.g)
            elif isinstance(answer, ValuePoint):
                winners[i] = ValuePoint(x, answer.f)
            else:
                winners[i] = Point(x)
        return winners

    def _place(self, points: dict[int, PathPoint], *, restart: bool) -> None:
        """Commit each usable point as its path's next point, then tell the method.

        One gate for both channels, defined once on the point type: an unusable
        point - non-finite in either half - never becomes a path's point and the
        method is never told it landed. The path keeps the point it holds and is
        marked stalled, so `restart_stalled` alone decides whether it is re-drawn.
        `settle` bought the point and this is its first read, so a non-finite
        value or gradient is refused here rather than at the seam.

        The channels differ in one thing: which hook is called. A restart resets
        the method's per-path state to cold start and seeds it from the fresh
        start; it is this iteration's move for every restarting path, usable or
        not, so none of them also steps. A step tells the method the winner
        landed, and is where the secant is read: a winner refused here is refused
        before the method has been told anything. The point a step started from
        is not passed on - the method kept it when it was placed or last stepped
        from.

        The hook is called once, with every path that landed, so a method that
        keeps per-path state updates it in one call and never sees a path this
        iteration refused.

        Args:
            points: The points to commit, keyed by path.
            restart: Whether these are this iteration's fresh starts or its step
                winners.

        """
        landed: dict[int, PathPoint] = {}
        for i, point in points.items():
            slot = self.slots[i]
            if not point.is_usable:
                slot.is_stalled = True
                continue
            slot.is_stalled = False  # a placed path is moving again
            slot.point = point
            landed[i] = point
        if landed:
            notify = self.local_solver.restart if restart else self.local_solver.step
            notify(landed)


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
    An injected callable's type is the annotations' job: a check for "callable"
    or "carries the Protocol" re-asserts what the type checker already refuses.
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


def _check_inside(pts: jnp.ndarray, *, what: str, domain: Domain) -> jnp.ndarray:
    """Refuse a point, or a batch of them, the run cannot hold: not finite, outside.

    The one box test the run has, called at every seam that hands the flock a
    point - the caller's `x0`, the global model's restart ask, a method's trial
    batch and its step winners - so all of them answer to one rule and one
    tolerance. A defective point is refused, never repaired: `x0` says "search
    this box" and "begin there" at once, and a model or method that answers
    outside is reporting a bug of its own, so clipping it would hide the
    contradiction and serve a point its author never computed.

    ``what`` names the seam in the message ("the global model's restart ask row 2
    is 7.5 in dimension 1, outside the box [-5, 5]"), which is the whole point:
    the author of a wrong answer has to be able to find it. A batch also names
    the row; a single point is already identified by its seam.

    The box test carries a machine-precision tolerance: a point computed at a
    boundary rounds by a few representable steps, so a coordinate within
    ``16 * eps * max(1, |bound|)`` of a finite side counts as on it, where eps
    comes from the points' dtype (at most float32). Free dimensions never
    trigger it - an infinite side is compared against itself. Without it, an
    answer that missed a wall by one ulp would be reported as outside, its value
    printing as the bound itself.

    One host sync decides, and only a violation pays for the diagnosis: NaN
    compares False against both sides, so finiteness folds into the same mask
    rather than being tested first.
    """
    lo, hi = domain.lo, domain.hi
    eps = jnp.finfo(jnp.result_type(pts.dtype, lo.dtype, jnp.float32)).eps
    # The tolerance's own scale comes from the finite sides only: taking it from
    # an infinite side would carry it to inf and disarm the other, finite one.
    span = jnp.maximum(
        jnp.where(jnp.isfinite(lo), jnp.abs(lo), 0.0),
        jnp.where(jnp.isfinite(hi), jnp.abs(hi), 0.0),
    )
    tol = 16.0 * eps * jnp.maximum(1.0, span)
    bad = ~jnp.isfinite(pts) | (pts < lo - tol) | (pts > hi + tol)
    if not bool(jnp.any(bad)):
        return pts
    idx = tuple(int(v) for v in jnp.argwhere(bad)[0])
    dim = idx[-1]
    # A single point has no row to name; a batch does.
    where = f"{what} row {idx[0]}" if pts.ndim > 1 else what
    value = float(pts[idx])
    if not math.isfinite(value):
        msg = f"{where} is {value:.6g} in dimension {dim}, not finite"
        raise ValueError(msg)
    msg = (
        f"{where} is {value:.6g} in dimension {dim}, "
        f"outside the box [{float(lo[dim]):.6g}, {float(hi[dim]):.6g}]"
    )
    raise ValueError(msg)


def _check_x0(x0: jnp.ndarray, n_paths: int, domain: Domain) -> jnp.ndarray:
    """Reject a flock the run cannot place: bad shape, or a point off the box.

    Called once at the entry, where the width, the row count and the box are all
    known, and where the caller's row count can still be named against
    `n_paths`. The box half is `_check_inside`, the same test every other seam
    answers to, so a start the caller authored is refused exactly like a start a
    model drew.
    """
    pts = jnp.asarray(x0)
    d = domain.d
    if pts.ndim != 2 or pts.shape[1] != d:
        msg = f"x0 must be a (n_paths, {d}) array, got shape {pts.shape}"
        raise ValueError(msg)
    if pts.shape[0] != n_paths:
        msg = f"x0 holds {pts.shape[0]} starts but n_paths is {n_paths}"
        raise ValueError(msg)
    return _check_inside(pts, what="x0", domain=domain)


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

    Only 0 and 1 are read: 0 marks an infeasible point, 1 a feasible one, and
    the mask is never read as a truthiness test - a signed constraint margin
    arrives looking feasible on every point it rules out, and a NaN arrives
    looking feasible too, both silently promoting a point into the reported
    best. No cast repairs that (a NaN casts to True), so the batch is refused
    rather than guessed at.
    """
    if not bool(jnp.all((feas == 0) | (feas == 1))):
        msg = (
            "the feas mask must hold 0/1 (or booleans) - 0 marks infeasible and "
            f"1 feasible, got {feas.tolist()}"
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
