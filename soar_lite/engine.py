"""The K-path engine: contracts, shared types, and the schedule.

This module is the seam (拍板) of soar_lite: it owns the two plugin contracts
(`LocalSolver` and `RestartRule`), the data the plugins exchange (`PathPoint`,
`Verdict`), and the shared plumbing (`KPathResult`, the box/prior seams, the
`argmin_finite` helper, and the `kpath_schedule` loop). Concretes import *from*
here, never the other way around at module level:

    soar_lite/
      engine.py   <- LocalSolver, RestartRule, PathPoint, Verdict, kpath_schedule
      solver.py   <- DiagonalPreconditioner (implements LocalSolver)
      restart.py  <- EIRestartRule (implements RestartRule)

Dependency direction: plugins depend on engine types; the engine binds its two
defaults lazily inside `kpath_schedule` (keeps the module graph acyclic and
`__init__` importable in any order). The engine is first-order (value-and-
gradient), coordinate-free, and owns the paths, the box, the shared budget,
the clip-to-bounds and the EI restart cadence; the local solver owns only
direction/step size/acceptance and the restart rule only the continue-vs-
restart decision.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import jax
import jax.numpy as jnp


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
    """

    def observe_continue(self, i: int, delta: float) -> None:
        """Fold path i's one-step value change Δf into its continuation evidence."""
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

    def rounds_until_verdict(self) -> int:
        """Report the rounds until any path's verdict can be due again (0 = any round).

        A cadence hint, not evidence, and the engine asks it only once every
        path has stalled: it is what separates "waiting for a verdict" from
        "nothing can happen again". A verdict unreachable before max_steps
        makes the remaining rounds pure spin - no eval, no state change - and
        the engine stops there instead of burning the step budget. 0 means
        "may be due any round" and disables that shortcut, so a rule with no
        cadence of its own is never mistaken for a dead one.
        """
        ...

    def restart(self, i: int) -> None:
        """Drop path i's step-local evidence; the engine assumes nothing else."""
        ...

@dataclass(frozen=True)
class PathPoint:
    """One path's point: coords x, value f, gradient g = ∂f/∂x."""

    x: jnp.ndarray
    f: float
    g: jnp.ndarray


Verdict: TypeAlias = Literal["accept", "retry", "give_up"]



class LocalSolver(Protocol):
    """The search method the K-path engine drives (the engine's plugin point).

    The engine owns the schedule: row composition, projection, the shared
    budget, the K-path state machine (init/restart/stall), the incumbent B,
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
        method's own convergence test. The model itself is only ever mutated
        in ``fold``, never here.
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


# --- engine: the K-path schedule ---

KPathStatus: TypeAlias = Literal["budget_exhausted", "max_steps", "all_stalled"]
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

# The search box in scipy's spelling (which `box_bounds` normalizes): one
# `(lb, ub)` pair per dim, a side of None or ±inf meaning unbounded, a bare
# None meaning no box at all (then d is the sampler's). A `(d, 2)` array is the
# same data row-wise, for the callers that already hold one.
Bounds: TypeAlias = Sequence["tuple[float | None, float | None]"] | jnp.ndarray | None

# The one way the engine draws a start: (key, n) -> (n, d) points in the
# engine's own coordinates. It is the only authority on d, which is what makes
# an unbounded search well-posed - the engine has no x0 of its own to fall back
# on, so `sample=None` is only defined over a finite box.
Sampler: TypeAlias = Callable[[jnp.ndarray, int], jnp.ndarray]



def box_bounds(bounds: Bounds) -> tuple[jnp.ndarray, jnp.ndarray] | None:
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



def finite_box(box: tuple[jnp.ndarray, jnp.ndarray]) -> bool:
    """Report whether every side of every dim is finite (no-sampler precondition)."""
    lo, hi = box
    return bool(jnp.all(jnp.isfinite(lo)) and jnp.all(jnp.isfinite(hi)))



@dataclass
class KPathResult:
    """End state of one kpath_schedule run.

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
    status: KPathStatus



def argmin_finite(values: list[float]) -> int | None:
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



def kpath_schedule(  # noqa: PLR0912, PLR0913, PLR0915, C901 (seams+knobs; closures keep shared state - a module-level split would need 10+ plumbing params)
    evaluate_batch: BatchEvaluator,
    bounds: Bounds,  # scipy-style (lb, ub) pairs, a (d, 2) array, or None
    key: jnp.ndarray,
    *,
    # The prior, injected: (key, n) -> (n, d) in the engine's own coordinates,
    # and the only authority on d. None -> i.i.d. uniform over a finite
    # `bounds` (the historical path); an unbounded or absent box then has no
    # start to draw and is rejected below rather than guessed.
    sample: Sampler | None = None,
    # Keyword-only and defaulted, like scipy's `callback`: only the caller
    # knows where a budget lives, and None means unbounded - the run then
    # stops on max_steps alone. The gate is the shell's job, not the
    # engine's.
    budget_check: Callable[[], bool] | None = None,
    K: int = 16,  # noqa: N803 (K = number of paths, the engine's width)
    max_steps: int = 1000,
    solver: LocalSolver | None = None,  # None -> DiagonalPreconditioner
    max_ls: int = 20,  # engine rounds per step before stalling a path
    restart_rule: RestartRule | None = None,  # None -> EIRestartRule, the default rule
    n_probe_rounds: int = 2,  # draws/start; 1=init, inf=search; 2-3 empirical, pinned 2
    # Billing mode. False (default) charges one eval per served point - the
    # serial-cost model (`nfev = batch size`).
    # True charges one eval per *batch* - a round of parallel evaluations, one
    # wall-clock unit when the objective is batched (jax.vmap or any parallel
    # evaluator). These are two different cost models, never the same number: a
    # K-path round costs K points serially, 1 round in parallel.
    per_round_evals: bool = False,
) -> KPathResult:
    """Advance K independent chains in x under one shared budget.

    The engine owns the schedule: it composes and clips the rows into
    `bounds`, serves one batched evaluate_batch call per round, keeps the
    K-path state machine, and owns the incumbent B. The search itself is
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
    `sample`, because the engine has no x0 of its own. Billing is set by
    `per_round_evals`: with the default False every served point - probe,
    trial, restart probe - counts exactly one eval (nfev = batch size), the
    serial-cost model; with True each served batch
    counts one (nfev = number of batches), the parallel-cost proxy where a
    K-path round is one wall-clock unit. budget_check() is the
    shell's batch-level predicate consulted once per
    batch (whole batch or None), a False verdict stops all paths globally,
    and None means unbounded (no gate consulted, runs to max_steps); the
    terminal batch may over-run the budget by <M points (the gate
    flips only between batches). fun is the true min over every feasible
    recorded eval with a finite value - every served point when the seam
    reports no aux;
    with aux, only points whose feas mask is True update it. The trajectory
    itself still moves on the penalized objective.

    Selection events (init, restart) never descend blind: each event with n
    starts to fill (n = K at init, n = N restarting paths on a restart) draws
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
    regardless of K vs N
    (by design). Probe-batch grads are
    reused for the selected starts: starts cost zero extra evals. The rule's
    one shared payoff pool is cold-started with the K init group-mins
    (sigma_r > 0 for K >= 2); K=1 yields a single sample -> sigma_r = 0 ->
    degenerate EI_restart = 0 -> the single path never restarts. Every restart
    event extends {r_i} with the event's group-mins for all paths at once (a
    payoff is a property of the sampler, not of one path). A path whose trials
    exhaust max_ls rounds stalls (observe_continue(i, inf) counts the fold, the
    read drops the value) until an EI verdict revives it.
    """
    box = box_bounds(bounds)
    d_bounds: int | None = None if box is None else box[0].shape[0]
    lo, hi = box if box is not None else (-math.inf, math.inf)
    if sample is None:
        # The sample=None contract: a finite box is the only prior the engine
        # can synthesize, so an unbounded side (or no box at all) is a wiring
        # error, not a run to guess our way out of.
        if box is None or not finite_box(box):
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
    evals = 0
    best_fun = math.inf
    # d is the sampler's business: when the box does not supply it, this is an
    # empty placeholder until the first served point fills it in.
    best_x = jnp.full(0 if d_bounds is None else d_bounds, math.nan)
    alive = True

    def eval_x(
        x_batch: jnp.ndarray,
    ) -> tuple[list[float], jnp.ndarray] | None:
        """One budget gate check, one batched evaluate_batch call.

        budget_check() is the shell's batch-level predicate (it only flips
        after a batch is served, never between the rows of one batch), so we
        consult it once and serve the whole batch or None - no partial
        prefix. None means unbounded: no gate is consulted at all. Returns
        (f_vals, g_batch) or None when the gate is already
        False; alive goes False (global stop). A served batch may over-run a
        budget by <M points (the gate flips only between batches). Each point
        counts exactly one eval. Only feasible points' values fold into the
        incumbent B; the seam's third return is a per-point feas mask (None = no
        aux, every point counts).
        """
        nonlocal evals, best_fun, best_x, alive
        if budget_check is not None and not budget_check():
            alive = False
            return None
        f_batch, g_batch, feas = evaluate_batch(x_batch)
        # One batch is one round of parallel evaluations: charge it as one eval
        # under per_round_evals, else one per point (the serial model).
        evals += 1 if per_round_evals else x_batch.shape[0]
        # per-path bookkeeping (secant/restart) stays Python float,
        # so pull the batch values onto the host in one shot.
        f_vals = f_batch.tolist()
        # B (incumbent) is defined over feasible designs only; with no aux
        # every point is treated as feasible.
        feas_list = [True] * len(f_vals) if feas is None else feas.tolist()
        candidates = [j for j, ok in enumerate(feas_list) if ok]
        if candidates:
            pos = argmin_finite([f_vals[j] for j in candidates])
            if pos is not None:  # every feasible point was non-finite: no B update
                j_min = candidates[pos]
                if f_vals[j_min] < best_fun:
                    best_fun = f_vals[j_min]
                    best_x = x_batch[j_min]
        return f_vals, g_batch

    def probe_select(
        n: int,
    ) -> tuple[jnp.ndarray, list[float], jnp.ndarray] | None:
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
        init, inf -> random search), pinned at 2 by design. Returns (x, f, g)
        — (n,d) x-stack, value list, (n,d) grad-stack — or None when the
        budget gate is already False.
        """
        nonlocal key
        x_rounds: list[jnp.ndarray] = []
        f_rounds: list[list[float]] = []
        g_rounds: list[jnp.ndarray] = []
        for _ in range(n_probe_rounds):
            key, sub = jax.random.split(key)
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
            if out is None:  # eval_x is the only authority that sets alive = False
                return None
            p_f, p_g = out
            x_rounds.append(x_cand)
            f_rounds.append(p_f)
            g_rounds.append(p_g)
        x_list: list[jnp.ndarray] = []
        f_list: list[float] = []
        g_list: list[jnp.ndarray] = []
        for j in range(n):
            # Finite-only group-min: a failed (non-finite) probe ranked before
            # a real one would be selected as the start, poisoning that path's
            # local method state for the rest of the run.
            best = argmin_finite([f_rounds[i][j] for i in range(n_probe_rounds)])
            if best is None:
                # Whole group non-finite: keep round 0 rather than invent a
                # point. Its f is non-finite, so init/restart stalls this
                # path on arrival and only an EI verdict revives it.
                best = 0
            x_list.append(x_rounds[best][j])
            f_list.append(f_rounds[best][j])
            g_list.append(g_rounds[best][j])
        return jnp.stack(x_list), f_list, jnp.stack(g_list)

    init = probe_select(K)
    if init is None:
        return KPathResult(
            x=best_x, fun=best_fun, nfev=evals, nit=0, status="budget_exhausted"
        )
    x, f, g = init
    # a start with no finite evidence never steps: it waits for EI
    stalled: list[bool] = [not math.isfinite(v) for v in f]
    # Defaults are bound lazily, not at module import: solver and restart
    # import the engine's types, so importing them here would form a cycle.
    if solver is None:
        from soar_lite.solver import DiagonalPreconditioner  # noqa: PLC0415

        solver = DiagonalPreconditioner()
    if restart_rule is None:
        from soar_lite.restart import EIRestartRule  # noqa: PLC0415

        restart_rule = EIRestartRule()
    method: LocalSolver = solver
    rule: RestartRule = restart_rule
    for i in range(K):
        method.init(i, PathPoint(x[i], f[i], g[i]))
    for v in f:  # cold-start evidence: the K init group-mins, into the one pool
        rule.observe_restart(v)

    def restart_paths(idxs: list[int]) -> None:
        """Restart the given paths from one fresh probe-select event.

        The event's group-min list extends the rule's one shared payoff pool;
        the method's and the rule's per-path state reset to cold start, while
        that pool is kept (same sampler, new trajectories). A path whose fresh
        group-min is non-finite arrives stalled and waits for the next EI
        verdict.
        """
        nonlocal x, g
        out = probe_select(len(idxs))
        if out is None:
            return  # alive already False; caller stops
        r_x, r_f, r_g = out
        for v in r_f:  # the event's group-mins extend the shared payoff pool
            rule.observe_restart(v)
        for j, i in enumerate(idxs):
            x = x.at[i].set(r_x[j])
            g = g.at[i].set(r_g[j])
            f[i] = r_f[j]
            method.restart(i, PathPoint(r_x[j], r_f[j], r_g[j]))
            rule.restart(i)  # the next Δ opens this path's next stage
            stalled[i] = not math.isfinite(r_f[j])

    def restart_phase() -> None:
        """Decide which paths to restart against B.

        B is passed to each verdict (the engine is its only writer), so every
        verdict reads the incumbent of its own iteration. Stalled paths' step
        counting now lives in the main loop's tail (complementary to the
        accepted paths'), so a stalled path's verdict cadence aligns with its
        advancing cohort instead of leading it by one round.
        """
        to_restart = [i for i in range(K) if rule.should_restart(i, f[i], best_fun)]
        if to_restart:
            restart_paths(to_restart)

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
        nonlocal x, g
        active = [i for i in range(K) if not stalled[i]]
        if not active:
            return  # everyone stalled; verdicts may revive paths next iteration
        rejecting = list(active)
        # Per-step argmin cache: "accept" commits the lowest-f finite trial
        # served this step (not the last served one), so a method can sweep a
        # step-length grid without re-evaluating its argmin. The engine is the
        # sole owner of "who won": the method learns the winner only through
        # the fold callback, never by tracking its own argmin (two copies
        # would drift).
        best_f: dict[int, float] = {}
        best_x: dict[int, jnp.ndarray] = {}
        best_g: dict[int, jnp.ndarray] = {}
        # max_ls caps every retry verdict (the warm-up Armijo, which has no
        # solver-side round counter, plus the post-warmup grid whose own
        # ls_rounds is its real bound), and stays engine-side as a failsafe: a
        # solver that returned "retry" forever would hang the engine without
        # this cap.
        for _ls in range(max_ls):
            if not rejecting or not alive:
                break
            # The method proposes raw candidates; the engine clips them into
            # `bounds`, so the caller's map never sees a point outside the
            # box the engine advertised. The clipped step is the actual one, so
            # verdict() is handed the served point.
            raw = jnp.stack(
                [method.propose(i, PathPoint(x[i], f[i], g[i])) for i in rejecting]
            )
            x_trial = jnp.clip(raw, lo, hi)
            outcome = eval_x(x_trial)
            if outcome is None:
                break
            s_f, s_g = outcome
            still: list[int] = []
            for pos, i in enumerate(rejecting):
                point = PathPoint(x[i], f[i], g[i])
                trial = PathPoint(x_trial[pos], s_f[pos], s_g[pos])
                finite = math.isfinite(trial.f) and bool(jnp.all(jnp.isfinite(trial.g)))
                if finite and (i not in best_f or trial.f < best_f[i]):
                    best_f[i] = trial.f
                    best_x[i] = trial.x
                    best_g[i] = trial.g
                verdict = method.verdict(i, point, trial)
                if verdict == "accept":
                    if i not in best_f:  # no finite sample served: nothing to win
                        stalled[i] = True
                    else:
                        x = x.at[i].set(best_x[i])
                        g = g.at[i].set(best_g[i])
                        f[i] = best_f[i]
                        method.fold(
                            i, point, PathPoint(best_x[i], best_f[i], best_g[i])
                        )
                elif verdict == "retry":
                    still.append(i)
                else:  # "give_up": the method's own convergence test fired
                    stalled[i] = True
            rejecting = still
        for i in rejecting:  # max_ls exhausted: stall; an EI verdict may revive
            stalled[i] = True

    nit = 0
    all_stalled = False
    for step in range(max_steps):
        restart_phase()
        if not alive:
            break
        # All paths are stuck and this round's restart votes (asked in
        # restart_phase above) revived nothing. The next vote is d rounds away,
        # d = rounds_until_verdict(): it counts to the next round whose cadence
        # gate opens; 0 means "unknown cadence", only possible before any
        # observe, when max_steps - step >= 1 already rules the shortcut out.
        # Rounds are indexed range(max_steps), so a future vote is reachable
        # iff its round step + d <= max_steps - 1, i.e. d < max_steps - step;
        # exit on the complement d >= max_steps - step (>=, not >: a gate that
        # would first open at round max_steps never runs). Once every path is
        # stalled no eval can fire later, so x / best_fun / nfev are untouched.
        if all(stalled) and rule.rounds_until_verdict() >= max_steps - step:
            all_stalled = True
            break
        nit += 1  # a round that got past the restart phase is an attempted one
        f_prev = list(f)
        step_paths()
        # The restart rule's Δf feed is the engine's own bookkeeping. Every
        # path steps exactly once per round: `stalled` now encodes "did not
        # accept" (step_paths maintains it), so a stalled path feeds inf - the
        # read drops the value but the fold still counts, and a
        # stalled path's verdict cadence aligns with its advancing cohort
        # instead of leading it by a round (the off-by-one a top-of-round
        # stall count would cause). An advancing path feeds the real Δf.
        for i in range(K):
            rule.observe_continue(i, (f[i] - f_prev[i]) if not stalled[i] else math.inf)
        if not alive:
            break

    if not alive:
        status: KPathStatus = "budget_exhausted"
    elif all_stalled:
        status = "all_stalled"
    else:
        status = "max_steps"
    return KPathResult(x=best_x, fun=best_fun, nfev=evals, nit=nit, status=status)
