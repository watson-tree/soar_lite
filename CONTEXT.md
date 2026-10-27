# soar-lite

A first-order multi-path optimizer: value and gradient come back together from
the caller; `n_paths` independent descent paths advance in lockstep under one shared
budget, each restarted when the sample-measured expected improvement of
restarting beats that of continuing.

## Language

Public knob history (pre-0.1.0 renames): `max_steps` → `max_evals` →
`max_iters`, `n_probe_rounds` → `n_probe_evals` → `n_draws`, `per_round_evals`
removed (both counts are now always reported), `nfev` → `n_points` +
`n_batches`. The old names let "round"/"step"/"eval" each do double duty; they
are dead in new code.

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
decides which paths restart and then moves the rest: a path either probes or
steps this iteration, never both. The parallel wall-clock unit; an iteration
may serve zero batches (a dead iteration where nothing moves) or several (the
probe's draws, the steppers' trials, and the g batch that grades what they
kept).
Iteration 0 is the initialization probe: the rule's cold-start
verdict says restart for every path, so starting the flock is the same
operation as any later restart. `n_iters` counts finished iterations only — one
cut short by a refused probe or batch is not counted, so a refused probe
reports `n_iters = 0` and a refused first step `n_iters = 1`.
_Avoid_: round (retired), epoch, iterations (as a name for the count)

**step**:
One move along a single path: propose → (trial)× → commit, or give_up. Each
path takes at most one move per iteration — a step, or a restart's probe; a
step consumes zero or more trials.
_Avoid_: update, move

**trial**:
One served proposal within a step — one row of a batch that the method judges
via `verdict`. The method is handed its coords and value, never its gradient:
only the winner's gradient is bought (see *settle*), so a method that needs one
to rule is out of this engine's reach. There is no engine-side cap: the
method's own stopping test (or its own trial budget) ends the enumeration, so
`verdict` cannot keep answering `retry` — it must at some point return `accept`
or `give_up`. A step that needs no verdict (see *needs verdict*) serves none
at all.
_Avoid_: line-search round (retired)

**needs verdict**:
The method's declaration that its next step is decided candidate by candidate.
True is the enumerated step: each *trial*'s value decides the next move. False
is the committed step: the one proposal is taken as made, so the engine serves
no value for it and the winner arrives with both halves owed (see *settle*),
where the engine's own finiteness gate is the whole of its acceptance. Asked
once per path per iteration.
_Avoid_: max rounds, is non verdict

**best**:
The run's running best, kept twice: `fun`/`x` is the best *feasible* point — the
run's reported answer, and what `NPathResult` calls `fun`/`x` — while `fun_raw`
is the best over every finite point served. Restart verdicts are asked against
`fun_raw`, because both sides of a verdict (`f_k`, the group-min pool) are
unfiltered. The engine is the sole owner of both. A *feasible* `-inf` is not
weighed but taken: the objective has no floor there, so it becomes `fun`/`x` and
ends the run with `status` `unbounded`; a `nan`, a `+inf` or an infeasible `-inf`
changes neither best.
_Avoid_: incumbent (retired), best-so-far, current optimum

**stall / stalled**:
A path that moved nowhere this iteration — it neither restarted nor committed a
step: its local method signalled `give_up` (its own stopping test fired, or its
trial budget ran out), it accepted with no finite sample to win, or the sample it
did win had a non-finite gradient.
Stalled is not dead: a stalled path can still be restarted by a later verdict.
The engine reports it to the restart rule as `None` — absence, not a sample —
so the rule reads no continuation from it. The iteration still counts: the rule
keeps one stage clock per path, advanced once a round either by a report
(folded or stalled) or, for the path it is probing, by its own `restart` hook —
so no path, moving, stalled or restarted, expires behind its cohort. A verdict
ends the stage either way — a window is one stage, whatever the answer — so a
stalled path with nothing to win is re-judged once per window, not every round.
_Avoid_: paused, idle, dead

### Evaluation

**batch** (noun):
The `(M, d)` points served by one fused `eval_f_g` pass. The engine
composes one batch per pass from the next candidates of the active paths.
_Avoid_: group (that is the probe column, below), lot, chunk

**point** (noun):
One candidate — one row of a batch. `n_points` counts the billed ones.
_Avoid_: query, sample, evaluation (retired)

**n_points / n_batches**:
The two reported work counts: g points and g batches (fused value+gradient
passes) billed. They are different numbers on different axes and must never
share a chart. Only g passes are billed: value-only work — a probe's f-only
draws, a rejected trial — is the cheap half and appears in neither count. The
result reports both, replacing scipy's single `nfev`.
_Avoid_: nfev (retired)

**objective** (目标 f):
The scalar function being minimized; its per-point values are the f values the
engine records. Distinct from the *cost model* (`n_points`/`n_batches`), which
counts work.
_Avoid_: loss, cost, fitness

**eval_f_g** (批量求值函数):
The caller's per-batch callable (type `EvalFG`): points in,
`(f, gradient, feas?)` out. The seam to the objective's values and
gradients; the engine is blind to it otherwise.
_Avoid_: oracle, evaluator, black box

**eval_f** (可选的值求值器, type `EvalF`):
The optional value-only sibling: points in, `(f, feas?)` out, no gradient.
It exists so a multi-draw probe can pay for `g` once, at its winners, instead
of once per draw. Without it the engine refuses a probe deeper than one draw.
_Avoid_: value oracle, cheap evaluator

**feasibility mask**:
The optional per-point boolean saying which served points may count toward the
reported best (`fun`/`x`). `fun_raw` and the restart verdicts ignore it.
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
A path's first step, before any secant pair exists: one sample of the alpha grid
along `-g₀`, committed as proposed under the default `n_alpha = 1`. It carries
no acceptance guard of its own — the phase enters only the direction, never the
ruling (see *needs verdict*).
_Avoid_: initialization step

**give_up**:
A local method's own signal that its convergence test passed (or it has nothing
finite to try): the path stops spending budget this iteration.
_Avoid_: converge, stop, terminate

### Restart

**probe**:
Drawing fresh starts: `n_draws` batches of candidate points, per-column
minimum → one start each. Each column's minimum is one group-min sample, so a
probe both picks starts and feeds the group-min pool, and a probe is one
iteration (one billed g batch: either the draw's own fused pass, or the
winners' fused pass when a multi-draw probe defers its gradients). A start is
usable only if `f` **and** its gradient are finite; a finite `f` with a
non-finite gradient arrives stalled instead of stepping. Initialization and
restart use the same probe.
_Avoid_: selection event, random init, sampling

**group-min**:
The minimum `f` of one probe's draw group. It is one sample in the group-min
pool, and it is the value a restarted path starts from; it is not a payoff, and
not a reward.
_Avoid_: sample, payoff, reward, fitness

**group-min pool**:
The shared multiset of group-mins (one per draw group — n per probe event),
measured rather than modelled.
One pool serves every path. It is the sample the restart payoff is priced off,
not a pool of payoffs. A `+inf` group-min is a dead start, and a `nan` or `-inf`
is an event outside the finite set: all three contribute exactly 0 expected
improvement, so they enter the payoff read as zero terms that dilute it instead
of being dropped.
_Avoid_: payoff pool, restart history, bandit arms

**restart verdict**:
The per-path comparison of continuing's expected improvement against
restarting's; `EI_restart >= EI_continue` restarts the path. One sample each —
one step for continuing, one probe for restarting — so the verdict is a decision
about what to buy. The tie (`0 >= 0`) is a restart too: when continuing buys
nothing, the probe is the only action that buys evidence, and it cannot thrash
because the verdict's window is the pool's own size plus the round its stage
opened in, so each re-probe pushes the next ask a window further out. Each side
is modelled by a scott_mad KDE over its own samples (one Gaussian kernel per
sample), so neither side assumes a single Gaussian. A stalled path supplies no
continuation sample at all — absence, not a value — so its `EI_continue` is 0
and its verdict is one-sided; a moving path
is only compared once it holds as many steps as the restart side holds samples,
keeping the newest M. The two sides are cost-matched in gradient count only when
an f-only evaluator is supplied; without one the engine refuses
`n_draws > 1`, so the un-seamed default stays at 1.
_Avoid_: restart check, reset decision

**fold**:
The handoff by which the engine tells a method which of its served trials won,
so the method commits the matching state. The engine is the only owner of
"who won", and the fold is the only commit: a winner whose gradient turns out
unusable is refused before the method is told anything.
_Avoid_: accept, commit (as the name of this handoff)

**settle**:
The iteration's one g batch on a run with the f-only seam, and the only place a
produced point becomes a graded one. Everything left unread or ungraded — a
probe's starts, an unjudged step's proposal, a judged step's winners — is bought
here, after both halves have decided on values and before anything is placed. A
point that already arrived with its gradient (a probe's fused start) is promoted
as it stands: a pass that served both halves is never served twice. A judged
step without the seam therefore serves nothing extra here; an unjudged step is
pending in every run, and is served here exactly once.
_Avoid_: commit batch, gradient round

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
_Avoid_: eval budget, cost estimate, time limit

**deadline (`time_limit`)**:
The same stop in seconds, for callers whose per-iteration cost is too uneven to
count iterations. Checked before each g batch (and at the loop head, like the
horizon), so the over-run is under one g batch plus the one call in flight; only
an f-only prelude goes unchecked, so a seam run can over-run by that prelude.
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