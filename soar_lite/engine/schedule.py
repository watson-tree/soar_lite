"""The schedule: the loop, its gates, and the state it drives.

`n_path_schedule` validates a run's arguments, drives the loop and assembles the
report. A run's state lives in four private holders, one per state family:
`_Evaluator` (the two seams and what each charges), `_Paths` (the n-path state
machine), `_Best` (the run's best, `f⋆`), and `_Budget` (the stop meter and the
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
    Pending,
    ProposeFn,
    Work,
)


def n_path_schedule(  # noqa: PLR0913 (the caller's knobs, all keyword-only
    # after the one positional argument: one name per dial beats a config object here)
    eval_f_g: EvalFG,
    *,
    # The run's width: how many paths the engine drives.
    n_paths: int,
    # The run's box, a `Domain`: every point served lies inside it - the engine
    # clips the starts it draws, and a method clips its own step - and it is the
    # one statement of the dimension (`domain.d`) and of each coordinate's prior
    # `scale`.
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
    # budget the engine cannot see (points, dollars, an external clock). It is
    # handed the run's `Work` so far, so the count it gates on is the one the
    # report will carry.
    budget_check: Callable[[Work], bool] | None = None,
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
    """Advance a flock of paths in x from the caller's initial points.

    The engine owns the schedule: it clips every restart start into `domain`,
    serves each batch as one call through the seam it came from - `eval_f_g` for
    a fused batch, `eval_f` for a value-only one - buys the iteration's deferred
    gradients in one batch, keeps the n-path state machine, and owns the best
    `f⋆` (`fun` in the result). Two things are injected, named
    `global_model` and `local_solver`, in that order wherever they appear:
    `global_model` supplies **where** a restart lands (one point per restarting
    path, clipped into the box) and is also the run's evidence sink, while
    `local_solver` owns direction, step size, acceptance, and the enumeration
    inside a step, keeping its per-path state inside itself. `restart_stalled`
    is one flag on top of them:
    **when** a path stalls is the local solver's own stopping test, **whether**
    it is then re-drawn is `restart_stalled`, and **where** the new start lands
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
    served value or gradient comes back non-finite stalls its path on arrival;
    with `restart_stalled` on (the default), that path restarts again next
    iteration.

    `domain` is the run's box and the coordinate space the engine works in: the
    engine clips into it and knows nothing about any map from these coordinates
    to the caller's physical ones. It is also what the default model draws from,
    and the only statement of the dimension. With `global_model=None` the engine
    builds a `SampleGlobalModel` and lets the domain draw (`Domain.draw`); `key`
    seeds those draws, and an injected model owns its randomness instead.

    The run's own stop is the horizon: at most `max_iters` iterations, or the
    wall-clock deadline when `time_limit` is given. Iterations are the unit on
    purpose - one iteration is one pass of the loop, so the same iteration count
    reproduces the same run on any machine - but it is not a cost bound: one
    iteration serves one or more batches, so migrating a batch budget to
    `max_iters` can multiply the work done. `budget_check` is the caller's extra
    stop; it is asked at the loop head every iteration and again before each
    served batch, and it is handed the run's `Work` so far - the same counts the
    report carries, so a gate never keeps a counter of its own. Asked per batch,
    a gate on points can be passed by less than one batch. `NPathStatus` names
    what each of these writes.

    `fun` is the true minimum over every feasible eval with a finite value -
    every served point when the objective reports no mask, only the points its
    mask passes otherwise. A feasible ``-inf`` is the answer rather than a
    contender: the run ends on the spot and reports that point as ``x`` with
    ``fun = -inf``. The paths themselves move on every served value, mask or no
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
            restart. Validated at the entry: a start outside the declared box
            raises rather than clips, so a start the caller authored is never
            silently relocated.
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
            ``n_iters`` is ``0`` and every ``flock_x`` row is ``nan``.
        time_limit: a wall-clock deadline in seconds (``None`` means none).
        budget_check: an extra gate for a budget the engine cannot see, asked at
            the loop head every iteration and again before each served batch.
            It receives the run's `Work` so far, so it gates on the engine's own
            counts rather than on a parallel tally. Additional, not a horizon
            (``None`` means none).
        restart_stalled: whether a path the engine marks stalled (its method gave
            up, or its point was not usable) is re-drawn from the global model.
            ``False`` keeps the flock frozen; a fully stalled flock then ends the
            run with ``status="all_stalled"``.
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
            feasibility mask holding anything but 0/1, or a restart ask of the
            wrong shape.

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
        budget.tick()  # counted after the body: only finished iterations

    # Each path's point as the run ended, and its value. A path that never held
    # a usable point has no point to report: it reads `nan` on both halves.
    flock_x = jnp.stack(
        [
            jnp.full(domain.d, math.nan) if slot.point is None else slot.point.x
            for slot in paths.flock
        ]
    )
    flock_fun = jnp.asarray(
        [math.nan if slot.point is None else slot.point.f for slot in paths.flock]
    )

    return NPathResult(
        x=best.x,
        fun=best.fun,
        flock_x=flock_x,
        flock_fun=flock_fun,
        work=budget.work,
        n_iters=budget.n_iters,
        status=status,
    )


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

    Raised inside `_Best.observe`, the single funnel both evaluation seams pass through,
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
    otherwise. Nothing can un-stall a path but `_restart`, which only ever sees
    the restarting paths, so the state is terminal: the iteration would serve
    nothing, and repeating it would idle out the horizon. The run stops on it,
    naming the state in `status`.
    """


class _Budget:
    """The run's stop meter and work report.

    Three predicates decide whether work may go on: the iteration horizon
    (`max_iters`), the wall-clock deadline (`time_limit`), and the caller's
    extra gate (`budget_check`), which reads the run's `Work`. `stop` asks all
    three as one query - at the
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
    stored here. The counts live in `work` (`Work`), frozen and replaced on each
    charge; the gate is handed that same object, so what a gate reads is what
    the report carries.
    """

    def __init__(
        self,
        max_iters: int | None,
        time_limit: float | None,
        budget_check: Callable[[Work], bool] | None,
    ) -> None:
        self.max_iters = max_iters
        self.deadline = None if time_limit is None else time.monotonic() + time_limit
        self.work = Work(n_f_points=0, n_f_batches=0, n_f_g_points=0, n_f_g_batches=0)
        self.n_iters = 0
        self._gate = budget_check

    def charge_f(self, batch_size: int) -> None:
        """Record one served value-only batch. Counted on the f pair; never gates."""
        work = self.work
        self.work = replace(
            work,
            n_f_batches=work.n_f_batches + 1,
            n_f_points=work.n_f_points + batch_size,
        )

    def charge_f_g(self, batch_size: int) -> None:
        """Record one served fused batch. Counted on the f_g pair; never gates."""
        work = self.work
        self.work = replace(
            work,
            n_f_g_batches=work.n_f_g_batches + 1,
            n_f_g_points=work.n_f_g_points + batch_size,
        )

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
        """Whether the caller's extra gate has closed, read over the work so far."""
        return self._gate is not None and not self._gate(self.work)

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
        # Set when a feasible point evaluated to -inf: the run has an
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

    def serve_trials(self, x_batch: jnp.ndarray) -> list[Pending]:
        """Serve one step's trial batch, as one produced candidate per row.

        A trial's gradient is only wanted for the step's winner, so with the
        value-only seam each trial is served for its value alone and the winners are
        graded once, together (`settle`). Without it a fused pass is the
        only way to a value, so every trial carries its gradient whether or not
        the method's own enumeration keeps it; `settle` then buys nothing.
        """
        if self._eval_f is not None:
            return [
                Pending(row, f, None)
                for row, f in zip(x_batch, self._serve_f(x_batch), strict=True)
            ]
        f_vals, g_batch = self._serve_f_g(x_batch)
        return [
            Pending(x_batch[pos], f_vals[pos], g_batch[pos])
            for pos in range(len(f_vals))
        ]

    def settle(self, decided: dict[int, Pending]) -> dict[int, PathPoint]:
        """Buy whatever gradient the iteration's points still owe, in one batch.

        Everything the value-only seam left pending goes out as a single batch, so an
        iteration costs one gradient pass however many moves it deferred - its
        restart starts plus its step winners - instead of one pass per trial. A
        point that already carries its gradient is promoted as it stands, so a
        run that owes nothing buys nothing and the counters do not move. Keyed by
        path on both sides, so a caller's halves come back keyed rather than
        re-paired by position.
        """
        owed = [i for i, point in decided.items() if point.g is None]
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
            i: bought[i] if point.g is None else point.graded()
            for i, point in decided.items()
        }

    def _serve_f_g(self, x_batch: jnp.ndarray) -> tuple[list[float], jnp.ndarray]:
        """Serve one fused batch, or raise `_BudgetExhausted` if it is unaffordable.

        The stop query is asked at this batch altitude before the charge, so a
        batch is served whole or not at all and the fused counters only ever
        hold batches that were served. This is the expensive half; an
        enumerating step's rejected trials are served by `_serve_f`, which asks
        the same gate and counts on the other pair. Returns ``(f_vals, g_batch)``.
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
        """Serve one value-only batch: the same gate, the other pair of counters.

        The gate is asked here exactly as the fused half asks it, so a deadline
        that passes mid-step cuts the run at the next value-only batch rather
        than after a whole iteration's retries: a method's enumeration can turn
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


@dataclass
class _Slot:
    """One path's state: the point it holds and whether it stalled there.

    `point is None` is "no point yet" - a path the run has not placed. That is
    the machine's own state, not a stall: such a path is re-drawn whatever
    `restart_stalled` says, which is how iteration 0 places the flock and how a
    refused start is eventually re-drawn. `_restart` is the one writer that
    clears `stalled`, so a re-drawn path cannot stay stalled.

    `x` is not gated on finiteness - `usable` asks f and g only - so a point
    whose coordinates are `nan` is still a point the path holds.
    """

    point: PathPoint | None = None
    stalled: bool = False


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
      value-only seam left ungraded - restart starts and step winners alike - is
      bought in one batch before anything is placed, so nothing is committed
      before it is graded.
    - A path is stalled when it did not move at all this iteration - its method
      gave up on every trial, or its fresh start or its winner was not usable. A
      stalled path does not step, so it serves nothing that iteration and the
      method's own state stands still.
    - A start or a winner is committed only when it is `usable` - finite in both
      halves - one gate defined once on the point type.
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
        # One record per path, not a list per field: a path's point and its
        # stall bit are one path's state, read together by every restart
        # decision, and a step commits one path at a time (so a list, whose item
        # assignment is the commit, beats a stacked array).
        self.flock: list[_Slot] = [_Slot() for _ in range(self.n_paths)]

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
        reaches `settle` already graded.

        The two halves touch disjoint paths, and every call into a collaborator
        is per path, so deciding all moves before placing any is the same
        schedule as placing each as it is decided.

        A flock with nothing left to do ends the run instead of idling: when no
        path restarts and every path is stalled, `_AllStalled` is raised before
        any batch is asked for.
        """
        restarting = [
            i
            for i, slot in enumerate(self.flock)
            if slot.point is None or (self.restart_stalled and slot.stalled)
        ]
        # Nothing is going to move again: no path restarts, and every path is
        # stalled. `_restart` is the only writer that clears a stall bit and it
        # only ever sees the restarting paths, so an empty `restarting` beside a
        # fully stalled flock is terminal - the iteration would serve nothing.
        # Only reachable with `restart_stalled=False`; the run ends here rather
        # than idling out the horizon.
        if not restarting and all(slot.stalled for slot in self.flock):
            raise _AllStalled
        # Every path that did not restart and is not stalled. A stalled path is
        # left out rather than asked again: `restart_stalled` is its only way
        # back, and asking would let a method un-stall a path the engine has
        # already stopped - the two would disagree about whether it is running.
        stepping = [
            i
            for i, slot in enumerate(self.flock)
            if i not in restarting and not slot.stalled
        ]
        starts = self._restart_starts(restarting)
        winners = self._step_winners(stepping)
        # One batch over both halves, keyed by path so neither is re-paired by
        # position. A refusal (`_BudgetExhausted`) cuts the iteration short with
        # nothing placed - this is the only batch it serves. A point that arrived
        # already graded (a run without the seam) passes straight through
        # `settle` without being served again.
        settled = self.evaluator.settle({**starts, **winners})
        self._restart({i: settled[i] for i in restarting})
        self._step({i: settled[i] for i in winners})

    def _restart_starts(self, restarting: list[int]) -> dict[int, Pending]:
        """Ask for one fresh start per restarting path, clipped into the run's box.

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
        nothing to be compared - so `settle` buys them in the same batch as the
        step winners, and a restart costs no gradient pass of its own.
        """
        if not restarting:
            return {}
        d = self.domain.d
        n_points = len(restarting)
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
        return dict(
            zip(restarting, (Pending(row, None, None) for row in x_cand), strict=True)
        )

    def _step_winners(self, stepping: list[int]) -> dict[int, Pending]:
        """Ask the method for its step on each stepping path, keep the winners.

        One call covers every stepping path, so a method that enumerates a step
        does so here, inside its own ``propose``. A ``None`` answer is the
        method's own stopping test firing: the path stalls where it stands, and
        nothing is served and no point is committed for it. A `Pending` winner
        goes out with whatever halves the method holds - both owed when it never
        read the point, the value alone when it read it through `serve_trials` -
        and `settle` buys the rest.

        Nothing is committed here but the stall bit: a path ends this iteration
        stalled exactly when its method named no winner, or when the winner turns
        out not to be usable (`_step` decides that). A stalled path is never in
        `stepping`: the engine asks nothing more of it until a restart re-draws
        it.
        """
        if not stepping:
            # Every path restarted this iteration: nothing steps.
            return {}
        answers = self.local_solver.propose(
            stepping, self.domain, self.evaluator.serve_trials
        )
        winners: dict[int, Pending] = {}
        for i, answer in answers.items():
            if answer is None:
                self.flock[i].stalled = True
                continue
            winners[i] = answer
        return winners

    def _restart(self, points: dict[int, PathPoint]) -> None:
        """Place the given paths on their fresh starts (one restart event).

        The method's per-path state resets to cold start. A start the engine
        cannot use is refused on the same `usable` gate `_step` applies to a
        winner: it is not placed, the local solver is not seeded from it, and
        the path stays stalled until it is re-drawn - which keeps `flock[i].point`
        meaning "where the path stands". The start is this iteration's move for
        every restarting path, usable or not, so none of them also steps.
        """
        for i, point in points.items():
            slot = self.flock[i]
            slot.stalled = not point.usable
            if slot.stalled:
                continue  # unusable: no point, no seed (see `usable`)
            slot.point = point
            self.local_solver.restart(i, point)

    def _step(self, points: dict[int, PathPoint]) -> None:
        """Commit each usable winner as its path's next point, then step it.

        `local_solver.step` is the step's commit half: it is what tells the
        method the point landed, and it is the only place the secant is read, so
        a winner the engine refuses here is refused before the method has been
        told anything. The point the step started from is not passed on: the
        method kept it when it was placed or last stepped from.
        """
        for i, point in points.items():
            slot = self.flock[i]
            if not point.usable:
                # `settle` graded the winner and this is the first read, so a
                # non-finite value or gradient is refused here: the path keeps
                # the point it held, and the method is never told the step
                # landed. Same gate as `_restart`: an unusable point never
                # becomes a path's point.
                slot.stalled = True
                continue
            slot.point = point
            self.local_solver.step(i, point)


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
    may overstep, and clamping those is the defensive clip.

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
