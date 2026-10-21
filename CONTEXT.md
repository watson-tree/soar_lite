# soar-lite

A first-order multi-path optimizer: value and gradient come back together from
the caller; `n_paths` independent descent paths advance in lockstep under one shared
budget, each restarted when the sample-measured expected improvement of
restarting beats that of continuing.

## Language

### Engine

**path**:
One independent descent chain: a current point plus that chain's own solver
state. The engine runs `n_paths` of them in lockstep.
_Avoid_: chain, trajectory, thread, particle

**n_paths**:
The engine's width — the number of paths advanced each round.
_Avoid_: population size, swarm size

**round**:
One engine cycle: serve one batch, then let every active path take at most one
step. The parallel (wall-clock) unit, and one half of the billing split. A
probe is not a round: it costs `n_probe_rounds` rounds of the budget, so a run
that restarts serves fewer rounds than `max_steps`.
_Avoid_: iteration, epoch

**step**:
One move along a single path (propose, clip, serve, commit).
_Avoid_: update, iteration

**incumbent (B)**:
The best feasible point seen so far. The engine is its sole owner.
_Avoid_: best-so-far, current optimum

**stall / stalled**:
A path that took no step this round — its local method signalled `give_up` or
its retries were exhausted. Stalled is not dead: a stalled path can still be
restarted by a later verdict.
_Avoid_: paused, idle, dead

### Evaluation

**evaluation (eval)**:
One point's worth of one `evaluate_batch` call: its loss (and, through the
batch, its gradient). `nfev` counts evaluations or served batches, depending on
billing; the engine's running counter is `n_charged`, whose unit follows the
same billing mode.
_Avoid_: query, sample (when counting cost)

**batch**:
One `evaluate_batch` call serving several points at once — the next candidates
of every active path in one pass.
_Avoid_: lot, chunk

**objective** (目标 f):
The scalar function being minimized; its per-point values are the losses the
engine records. Distinct from the *cost model* (billing), which counts work.
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
finite to try): the path stops spending budget this round.
_Avoid_: converge, stop, terminate

### Restart

**probe**:
Drawing fresh starts: `n_probe_rounds` rounds of candidate points, per-column
minimum → one start each. The same draw is one restart-payoff sample, and it
costs `n_probe_rounds` rounds of the budget. Initialization and restart use the
same probe.
_Avoid_: selection event, random init, sampling

**horizon**:
One continue-side sample's span: a path's value change over `n_probe_rounds`
consecutive rounds, non-overlapping, poisoned to `inf` if any round inside it
stalled. Matched to one probe's cost, so both sides of a verdict buy the same
amount of work.
_Avoid_: window (that counts how many samples a stage holds), stride

**payoff pool**:
The shared multiset of restart payoffs (each probe's group minimum), measured
rather than modelled. One pool serves every path.
_Avoid_: restart history, bandit arms

**restart verdict**:
The per-path comparison of continuing's expected improvement against
restarting's; `EI_restart > EI_continue` restarts the path. Both sides are
fitted on equal-cost samples — one horizon for continuing, one probe for
restarting — so the verdict is a decision about what to buy, not a bias toward
either side.
_Avoid_: restart check, reset decision

**fold**:
The handoff by which the engine tells a method which of its served trials won,
so the method commits the matching state. The engine is the only owner of
"who won".
_Avoid_: accept, commit (as the name of this handoff)

### Cost model

**billing**:
How `nfev` counts work, in exactly two modes. Per-point: one eval per served
point (serial cost). Per-round: one eval per served batch (parallel cost).
They are different numbers and must never share a chart or table.
_Avoid_: budget accounting (billing counts work; it is not the objective)

**budget (`max_steps`)**:
The engine's own allowance, counted in batches: every batch the seam serves
costs one, so the init probe and each restart event cost `n_probe_rounds` of
it. A stalled round serves nothing and costs nothing. `max_steps` also caps the
run's rounds at the same number — a second bound that only bites on a stalling
run.
_Avoid_: iterations, function-call budget (that is `nfev`, a different axis)

**budget gate** (`budget_check`):
The batch-level predicate consulted once per batch — serve the whole batch or
nothing, never a partial prefix. It exists for a budget the engine cannot see
(points, wall-clock, dollars); the engine's own batch budget is `max_steps`.
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