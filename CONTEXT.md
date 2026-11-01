# soar-lite

A first-order multi-path optimizer: value and gradient come back together from
the caller; `n_paths` independent descent paths advance in lockstep under one shared
budget, each re-drawn from a global model once its own stopping test stalls it
(opt-in, behind `restart_stalled`).

## Language

Public knob history (pre-0.1.0 renames): `max_steps` → `max_evals` →
`max_iters`, `n_probe_rounds` → `n_probe_evals` → `n_draws`, then the probe
itself was removed (no `n_draws`, no `per_round_evals`), and the two-field
cost pair became four (`nfev` → the two seams × two rulers of §Evaluation).
The old names let "round"/"step"/"eval" each do double duty; they are dead in
new code.

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
splits the paths into a redraw set (no point yet, or stalled with
`restart_stalled` on) and a cohort that steps: a path either redraws or steps
this iteration, never both. The parallel wall-clock unit; an iteration may
serve zero batches (a fully stalled flock that is not redrawing) or several (the
redraw starts plus the steppers' trials, and the settle batch that grades what
they kept).
Iteration 0 is the flock's placement: an empty `x` redraws unconditionally,
so starting the flock is the same operation as any later redraw. `n_iters`
counts finished iterations only — one cut short by a refused batch is not
counted, so a refused placement reports `n_iters = 0` and a refused first
step `n_iters = 1`.
_Avoid_: round (retired), epoch, iterations (as a name for the count)

**step**:
One move along a single path: propose → (trial)× → commit, or give_up. Each
path takes at most one move per iteration — a step, or a redraw start; a
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
The run's running best, a single `fun`/`x`: the best *feasible* point — the
run's reported answer, and what `NPathResult` calls `fun`/`x`. The engine is the
sole owner. A *feasible* `-inf` is not weighed but taken: the objective has no
floor there, so it becomes `fun`/`x` and ends the run with `status` `unbounded`;
a `nan`, a `+inf` or an infeasible `-inf` changes nothing.
_Avoid_: incumbent (retired), best-so-far, current optimum

**stall / stalled**:
A path that moved nowhere this iteration — it neither redrew nor committed a
step: its local method signalled `give_up` (its own stopping test fired), it
accepted with no finite sample to win, or the sample it did win had a
non-finite value or gradient.
Stalled is not dead: when the run opted into `restart_stalled`, the path is
re-drawn by the global model on the next iteration; otherwise it stays parked —
still counted, still readable for `best` — but never moved again. A path with
no point at all (an empty `x`) is a redraw that ignores `restart_stalled`: it
is the machine's own initial state, not a stall.
_Avoid_: paused, idle, dead

### Evaluation

**batch** (noun):
The `(M, d)` points served by one pass — fused (`eval_f_g`) or value-only
(`eval_f`). The engine composes each batch from the next candidates of the
active paths and the redraw starts.
_Avoid_: group, lot, chunk

**point** (noun):
One candidate — one row of a batch. The four counters (below) count the billed
ones.
_Avoid_: query, sample, evaluation (retired)

**four counters** (`n_f_g_points` / `n_f_g_batches` / `n_f_points` / `n_f_batches`):
The run's reported work, on two axes and two seams. `points` counts billed
points (a batch bills its actual size), `batches` counts billed batches (a batch
bills 1); `f_g` is the fused value+gradient pair, `f` the value-only pair. All
four are counted — "billed" means "actually served out" — and none of them
decides the stop. Points and batches are different numbers on different axes
and must never share a chart, and the `f` pair must never be read as the `f_g`
pair. They replace scipy's single `nfev`.
_Avoid_: nfev (retired)

**objective** (目标 f):
The scalar function being minimized; its per-point values are the f values the
engine records. Distinct from the *cost model* (the four counters), which
counts work.
_Avoid_: loss, cost, fitness

**eval_f_g** (批量求值函数):
The caller's per-batch callable (type `EvalFG`): points in,
`(f, gradient, feas?)` out. The seam to the objective's values and
gradients; the engine is blind to it otherwise.
_Avoid_: oracle, evaluator, black box

**eval_f** (可选的值求值器, type `EvalF`):
The optional value-only sibling: points in, `(f, feas?)` out, no gradient.
It exists so a judged step's trials and the redraw starts can be read by value
first and their winners' gradients bought once at *settle*, instead of once per
trial. Without it every trial is itself a fused batch.
_Avoid_: value oracle, cheap evaluator

**feasibility mask**:
The optional per-point boolean saying which served points may count toward the
reported best (`fun`/`x`). It is
read as 0/1 and nothing else: only `0` (or `False`) marks a point infeasible, so
a mask holding anything else - a signed constraint margin, whose violated points
are all negative, or a `nan` - is refused with a `ValueError` instead of being
read as truthiness, where every one of those flags would have counted as
feasible and quietly promoted a point into the answer.
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

**global model** (type `GlobalModel`):
The engine's second seam, owner of *where* a redrawn path starts. It observes
every served batch (`observe(x, f, g?)`) and answers `propose(n, domain)` with
`n` fresh starts. The default `SampleGlobalModel` fits nothing: it draws starts
from the `domain` (with the caller's `x0` as the first answer). `BLRGlobalModel`
fits a Bayesian linear posterior and proposes the local minimum of `-log EI` —
EI there is an acquisition function deciding *where is worth trying*, not a
restart verdict.
_Avoid_: restart rule, sampler (the sampler is a per-dimension draw inside the domain)

**restart_stalled**:
The run's one restart bit, deciding *whether* a stall restarts: on, the stalled
path is re-drawn by the global model next iteration; off (the default), it stays
parked. It is orthogonal to *when* (the local method's own stopping test) and
*where* (the global model). A run that never restarts needs no model:
`restart_stalled=False` with the default `SampleGlobalModel` is pure descent
from the initial flock.
_Avoid_: restart check, reset decision

**fold**:
The handoff by which the engine tells a method which of its served trials won,
so the method commits the matching state. The engine is the only owner of
"who won", and the fold is the only commit: a winner whose gradient turns out
unusable is refused before the method is told anything.
_Avoid_: accept, commit (as the name of this handoff)

**settle**:
The iteration's one g batch on a run with the f-only seam, and the only place a
produced point becomes a graded one. Everything left unread or ungraded — the
redraw starts, an unjudged step's proposal, a judged step's winners — is bought
here, after both halves have decided on values and before anything is placed. A
point that already arrived with its gradient (a redraw start's fused pass) is
promoted as it stands: a pass that served both halves is never served twice. A
judged step without the seam therefore serves nothing extra here; an unjudged
step is pending in every run, and is served here exactly once.
_Avoid_: commit batch, gradient round

### Cost model

**horizon (`max_iters`)**:
The engine's own stop, counted in iterations: every completed iteration costs
one, and iteration 0 places the flock. A deterministic bound, not a cost
proxy — one iteration serves anywhere from no batch (a fully stalled flock that
is not redrawing) to the redraw starts plus however many trials the method's own
search serves, so the four counters report what a run
actually billed. At least one of `max_iters`/`time_limit` must be given: a
`budget_check` alone stops nothing, because a stalled iteration asks for no
batch.
_Avoid_: eval budget, cost estimate, time limit

**deadline (`time_limit`)**:
The same stop in seconds, for callers whose per-iteration cost is too uneven to
count iterations. Checked before every batch (fused or value-only) and at the
loop head, like the horizon, so the over-run is under one batch plus the one
call in flight.
_Avoid_: budget, timeout (that is a per-call concern)

**budget gate** (`budget_check`):
The batch-level predicate consulted once per batch — serve the whole batch or
nothing, never a partial prefix. It exists for a budget the engine cannot see
(points, wall-clock, dollars). Such an external gate can over-run by less than
one batch. It is an extra gate, never a horizon.
_Avoid_: stopping budget

**domain** (type `Domain`):
The object that carries everything the engine knows about the search space: the
per-dimension `(lb, ub)` sides (either may be unbounded), the per-dimension
`scale` the draws use, and the injected per-dimension samplers. It is the sole
authority on `d`; the engine clips every proposal into it. Bounds are data,
never a unit-cube assumption.
_Avoid_: box (the box is only the bounded-sides part of the domain)

**sampler**:
The injected per-dimension callable that draws starts; the default is uniform on
a bounded side and normal on a free side. The *domain* remains the sole
authority on `d`.
_Avoid_: initializer, prior (as a name for the *callable*)