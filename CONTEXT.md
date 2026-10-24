# soar-lite

A first-order multi-path optimizer: value and gradient come back together from
the caller; `n_paths` independent descent paths advance in lockstep under one shared
budget, each restarted when the sample-measured expected improvement of
restarting beats that of continuing.

## Language

Verbs count, nouns name: **eval** is the verb (one `evaluate_batch` call),
**batch** is the noun (the points that call serves), and the two always
correspond one to one. Counts take the verb (`max_evals`, `n_probe_evals`);
names take the nouns (`batch`, `point`, `path`).

Public knob history (pre-0.1.0 renames): `max_steps` → `max_evals`,
`n_probe_rounds` → `n_probe_evals`, `per_round_evals` removed (both counts are
now always reported), `nfev` → `n_points` + `n_batches`. The old names let
"round"/"step"/"eval" each do double duty; they are dead in new code.

### Engine

**path**:
One independent descent chain: a current point plus that chain's own solver
state. The engine runs `n_paths` of them in lockstep.
_Avoid_: chain, trajectory, thread, particle

**n_paths**:
The engine's width — the number of paths the engine drives together.
_Avoid_: population size, swarm size

**iteration**:
One pass of the engine's main loop — one `nit`. Per iteration the engine
decides which paths restart and then steps the rest: a path either probes or
steps this iteration, never both. The parallel wall-clock unit; an iteration
may serve zero batches (a dead iteration where nothing moves) or several (the
restart probe's `n_probe_evals` batches plus the steppers' up to `max_ls`
trials each). An iteration cut short by a refused probe or batch did not
finish, so it is not counted in `nit`.
_Avoid_: round (retired), epoch

**step**:
One move along a single path: propose → (trial)× → commit, or give_up. Each
path takes at most one move per iteration — a step, or a restart's probe; a
step consumes one or more trials.
_Avoid_: update, move

**trial**:
One served proposal within a step — one row of a batch that the method judges
via `verdict`. `max_ls` caps how many trials a step may spend.
_Avoid_: line-search round (retired)

**incumbent (B)**:
The best feasible point seen so far. The engine is its sole owner.
_Avoid_: best-so-far, current optimum

**stall / stalled**:
A path that moved nowhere this iteration — it neither restarted nor committed a
step: its local method signalled `give_up`, or its trials were exhausted.
Stalled is not dead: a stalled path can still be restarted by a later verdict.
_Avoid_: paused, idle, dead

### Evaluation

**eval** (verb, counted):
One `evaluate_batch` call — one serve. The unit of the engine's budget
(`max_evals`) and of a probe's cost (`n_probe_evals`). One eval serves one
batch.
_Avoid_: query, sample (when counting cost)

**batch** (noun):
The `(M, d)` points served by one eval. The engine composes one batch per
eval from the next candidates of the active paths.
_Avoid_: group (that is the probe column, below), lot, chunk

**point** (noun):
One candidate — one row of a batch. `n_points` counts these.
_Avoid_: query, sample, evaluation (retired)

**n_points / n_batches**:
The two reported work counts: points served, and batches (evals) served. They
are different numbers on different axes and must never share a chart. The
result reports both, replacing scipy's single `nfev`.
_Avoid_: nfev (retired)

**objective** (目标 f):
The scalar function being minimized; its per-point values are the losses the
engine records. Distinct from the *cost model* (`n_points`/`n_batches`), which
counts work.
_Avoid_: cost, fitness

**evaluate_batch** (批量求值函数):
The caller's per-batch callable (type `BatchEvaluator`): points in,
`(loss, gradient, feas?)` out. The seam to the objective's values and
gradients; the engine is blind to it otherwise.
_Avoid_: oracle, evaluator, black box

**feasibility mask**:
The optional per-point boolean saying which served points may count toward the
incumbent.
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
minimum → one start each. The same draw is one restart-payoff sample, and it
costs `n_probe_evals` of the budget. Initialization and restart use the same
probe.
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
an f-only evaluator is supplied; without one, `n_probe_evals` stays at 1.
_Avoid_: restart check, reset decision

**fold**:
The handoff by which the engine tells a method which of its served trials won,
so the method commits the matching state. The engine is the only owner of
"who won".
_Avoid_: accept, commit (as the name of this handoff)

### Cost model

**budget (`max_evals`)**:
The engine's own allowance, counted in batches (evals): every served batch
costs one, so the init probe and each restart event cost `n_probe_evals` of
it. A stalled iteration serves nothing and costs nothing. `max_evals` also
caps the run's iterations at the same number — a second bound that only bites
on a stalling run.
_Avoid_: iterations, function-call budget (that is `n_points`, a different axis)

**budget gate** (`budget_check`):
The batch-level predicate consulted once per batch — serve the whole batch or
nothing, never a partial prefix. It exists for a budget the engine cannot see
(points, wall-clock, dollars); the engine's own batch budget is `max_evals`.
Such an external gate can over-run by less than one batch.
_Avoid_: stopping budget

**box**:
The per-dimension `(lb, ub)` region the engine clips candidates into. Bounds
are data, never a unit-cube assumption.
_Avoid_: domain (as a name for the clipping region)

**sampler**:
The injected prior that draws starts; the sole authority on `d` when there is
no finite box.
_Avoid_: initializer, prior (as a name for the *callable*)