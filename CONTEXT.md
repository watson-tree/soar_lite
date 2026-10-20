# soar-lite

A first-order multi-path optimizer: value and gradient come back together from
the caller; `K` independent descent paths advance in lockstep under one shared
budget, each restarted when the sample-measured expected improvement of
restarting beats that of continuing.

## Language

### Engine

**path**:
One independent descent chain: a current point plus that chain's own solver
state. The engine runs `K` of them in lockstep.
_Avoid_: chain, trajectory, thread, particle

**K**:
The engine's width — the number of paths advanced each round.
_Avoid_: population size, swarm size

**round**:
One engine cycle: serve one batch, then let every active path take at most one
step. The parallel (wall-clock) unit, and one half of the billing split.
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
billing.
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
Drawing fresh starts: several rounds of candidate points, per-column minimum →
one start each. The same draw is one restart-payoff sample. Initialization and
restart use the same probe.
_Avoid_: selection event, random init, sampling

**payoff pool**:
The shared multiset of restart payoffs (each probe's group minimum), measured
rather than modelled. One pool serves every path.
_Avoid_: restart history, bandit arms

**restart verdict**:
The per-path comparison of continuing's expected improvement against
restarting's; `EI_restart > EI_continue` restarts the path.
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

**budget gate**:
The batch-level predicate consulted once per batch — serve the whole batch or
nothing, never a partial prefix. The budget is the caller's; the engine only
consumes it.
_Avoid_: stopping budget

**box**:
The per-dimension `(lb, ub)` region the engine clips candidates into. Bounds
are data, never a unit-cube assumption.
_Avoid_: domain (as a name for the clipping region)

**sampler**:
The injected prior that draws starts; the sole authority on `d` when there is
no finite box.
_Avoid_: initializer, prior (as a name for the *callable*)