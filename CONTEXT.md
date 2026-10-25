# soar-lite

A first-order multi-path optimizer: value and gradient come back together from
the caller; `n_paths` independent descent paths advance in lockstep under one shared
budget, each restarted when the sample-measured expected improvement of
restarting beats that of continuing.

## Language

Verbs count, nouns name: **eval** is the verb (one `evaluate_batch` call),
**batch** is the noun (the points that call serves), and the two always
correspond one to one. Counts take the unit they count — `n_probe_evals` counts
evals, `max_iters` counts iterations — and names take the nouns (`batch`,
`point`, `path`).

Public knob history (pre-0.1.0 renames): `max_steps` → `max_evals` →
`max_iters`, `n_probe_rounds` → `n_probe_evals`, `per_round_evals` removed (both
counts are now always reported), `nfev` → `n_points` + `n_batches`. The old
names let "round"/"step"/"eval" each do double duty; they are dead in new code.

### Engine

**path**:
One independent descent chain: a current point plus that chain's own solver
state. The engine runs `n_paths` of them in lockstep.
_Avoid_: chain, trajectory, thread, particle

**n_paths**:
The engine's width — the number of paths the engine drives together.
_Avoid_: population size, swarm size

**iteration**:
One pass of the engine's main loop, counted in `n_iters`. Per iteration the engine
decides which paths restart and then steps the rest: a path either probes or
steps this iteration, never both. The parallel wall-clock unit; an iteration
may serve zero batches (a dead iteration where nothing moves) or several (the
restart probe's `n_probe_evals` batches plus the steppers' own trial budgets).
Iteration 0 is the initialization probe: the rule's cold-start
verdict says restart for every path, so starting the flock is the same
operation as any later restart. `n_iters` counts finished iterations only — one
cut short by a refused probe or batch is not counted, so a refused probe
reports `n_iters = 0` and a refused first step `n_iters = 1`.
_Avoid_: round (retired), epoch, iterations (as a name for the count)

**step**:
One move along a single path: propose → (trial)× → commit, or give_up. Each
path takes at most one move per iteration — a step, or a restart's probe; a
step consumes one or more trials.
_Avoid_: update, move

**trial**:
One served proposal within a step — one row of a batch that the method judges
via `verdict`. There is no engine-side cap: the method's own stopping test (or
its own trial budget) ends the enumeration, so `verdict` must return `accept`
or `give_up`.
_Avoid_: line-search round (retired)

**incumbent (B)**:
The best point seen so far, kept twice: `fun`/`x` is the best *feasible* point
(the run's reported answer), `fun_raw` the best over every finite point served.
Restart verdicts are asked against `fun_raw`, because both sides of a verdict
(`f_k`, the payoff pool) are unfiltered. The engine is the sole owner of both.
_Avoid_: best-so-far, current optimum

**stall / stalled**:
A path that moved nowhere this iteration — it neither restarted nor committed a
step: its local method signalled `give_up`, or its trials were exhausted.
Stalled is not dead: a stalled path can still be restarted by a later verdict.
_Avoid_: paused, idle, dead

### Evaluation

**eval** (verb, counted):
One `evaluate_batch` call — one serve. The unit of a probe's cost
(`n_probe_evals`) and of a `budget_check` gate that counts batches. The engine's
own stop does not count these: it counts iterations (`max_iters`). One eval
serves one batch.
_Avoid_: query, sample (when counting cost)

**batch** (noun):
The `(M, d)` points served by one eval. The engine composes one batch per
eval from the next candidates of the active paths.
_Avoid_: group (that is the probe column, below), lot, chunk

**point** (noun):
One candidate — one row of a batch. `n_points` counts the billed ones.
_Avoid_: query, sample, evaluation (retired)

**n_points / n_batches**:
The two reported work counts: g points and g batches (fused value+gradient
passes) billed. They are different numbers on different axes and must never
share a chart. Only fused passes are billed: a probe's f-only draws are the
cheap half and appear in neither count. The result reports both, replacing
scipy's single `nfev`.
_Avoid_: nfev (retired)

**objective** (目标 f):
The scalar function being minimized; its per-point values are the f values the
engine records. Distinct from the *cost model* (`n_points`/`n_batches`), which
counts work.
_Avoid_: loss, cost, fitness

**evaluate_batch** (批量求值函数):
The caller's per-batch callable (type `BatchEvaluator`): points in,
`(f, gradient, feas?)` out. The seam to the objective's values and
gradients; the engine is blind to it otherwise.
_Avoid_: oracle, evaluator, black box

**evaluate_f** (可选的值求值器, type `EvaluateF`):
The optional value-only sibling: points in, `(f, feas?)` out, no gradient.
It exists so a multi-draw probe can pay for `g` once, at its winners, instead
of once per draw. Without it the engine refuses a probe deeper than one draw.
_Avoid_: value oracle, cheap evaluator

**feasibility mask**:
The optional per-point boolean saying which served points may count toward the
reported incumbent (`fun`/`x`). `fun_raw` and the restart verdicts ignore it.
_Avoid_: validity mask

### Local method

**secant pair**:
The `(Δx, Δg)` between two consecutive points of one path — the only curvature
evidence the local method reads.
_Avoid_: gradient difference, step pair

**diagonal preconditioner**:
The local method itself: a per-coordinate step-size estimate read off the
secant pair, applied as `-d ⊙ g`. No matrix, no linear solve, no scalar
learning rate.
_Avoid_: diagonal Newton (it is a biased `1/H` estimate, not Newton)

**floor (λ)**:
The guard that keeps a coordinate's step-size denominator positive when its
secant denominator vanishes (a flat but unconverged coordinate).
_Avoid_: epsilon, addend, regularizer

**warmup**:
A path's first step, before any secant pair exists: an Armijo backtrack along
`-g₀`. The only step that carries an acceptance guard.
_Avoid_: initialization step

**give_up**:
A local method's own signal that its convergence test passed (or it has nothing
finite to try): the path stops spending budget this iteration.
_Avoid_: converge, stop, terminate

### Restart

**probe**:
Drawing fresh starts: `n_probe_evals` batches of candidate points, per-column
minimum → one start each. The same draw is one restart-payoff sample, and a
probe is one iteration (one billed g batch: either the draw's own fused pass,
or the winners' fused pass when a multi-draw probe defers its gradients). A
start is usable only if `f` **and** its gradient are finite; a finite `f` with
a non-finite gradient arrives stalled instead of stepping. Initialization and
restart use the same probe.
_Avoid_: selection event, random init, sampling

**payoff pool**:
The shared multiset of restart payoffs (each probe's group minimum), measured
rather than modelled. One pool serves every path.
_Avoid_: restart history, bandit arms

**restart verdict**:
The per-path comparison of continuing's expected improvement against
restarting's; `EI_restart > EI_continue` restarts the path. One sample each —
one step for continuing, one probe for restarting — so the verdict is a decision
about what to buy. The two sides are cost-matched in gradient count only when
an f-only evaluator is supplied; without one the engine refuses
`n_probe_evals > 1`, so the un-seamed default stays at 1.
_Avoid_: restart check, reset decision

**fold**:
The handoff by which the engine tells a method which of its served trials won,
so the method commits the matching state. The engine is the only owner of
"who won".
_Avoid_: accept, commit (as the name of this handoff)

### Cost model

**horizon (`max_iters`)**:
The engine's own stop, counted in iterations: every completed iteration costs
one, and iteration 0 is the init probe. A deterministic bound, not a cost
proxy — one iteration serves anywhere from no batch (a fully stalled flock) to a
probe plus however many trials the method's own search serves, so
`n_batches`/`n_points` report what a run
actually billed. At least one of `max_iters`/`time_limit` must be given: a
`budget_check` alone stops nothing, because a stalled iteration asks for no
batch.
_Avoid_: budget (in evals), cost estimate, time limit

**deadline (`time_limit`)**:
The same stop in seconds, for callers whose per-iteration cost is too uneven to
count iterations. Checked between iterations, never inside one, so a run can
over-run it by one iteration.
_Avoid_: budget, timeout (that is a per-call concern)

**budget gate** (`budget_check`):
The batch-level predicate consulted once per batch — serve the whole batch or
nothing, never a partial prefix. It exists for a budget the engine cannot see
(points, wall-clock, dollars). Such an external gate can over-run by less than
one batch. It is an extra gate, never a horizon.
_Avoid_: stopping budget

**box**:
The per-dimension `(lb, ub)` region the engine clips candidates into. Bounds
are data, never a unit-cube assumption.
_Avoid_: domain (as a name for the clipping region)

**sampler**:
The injected prior that draws starts; the sole authority on `d` when there is
no finite box.
_Avoid_: initializer, prior (as a name for the *callable*)