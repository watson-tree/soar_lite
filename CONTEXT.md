# soar-lite

A first-order multi-path optimizer: value and gradient come back together from
the caller; `n_paths` independent descent paths advance in lockstep under one shared
budget, each re-drawn from a global model once its own stopping test stalls it
(the `restart_stalled` knob, on by default).

## Glossary

### Engine

**path**:
One independent descent: a current point plus that descent's own solver
state. The engine runs `n_paths` of them in lockstep. A restart replaces its
current point with a fresh draw, so the points one path holds across a run are
continuous in identity, not in space — which is why `best`, the reported answer,
need not be a point any path still holds.
_Avoid_: chain, trajectory, thread, particle

**n_paths**:
The engine's width — the number of paths the engine drives together.
_Avoid_: population size, swarm size

**flock**:
The paths' current points as one `(n_paths, d)` array: exactly one point per path,
so a flock is the run's state at one moment, not the set of paths. Every path
keeps its row whether or not it holds a usable point (`flock_f` is `nan` there),
so a flock always has `n_paths` of them. A *batch* is a set of points too and is
deliberately a different one: it is the work a single pass serves — a step's
trials, or the settle set of redraw starts plus winners — so it holds no point
for an idle path and is not the state. Iteration 0 places the flock from `x0`
where one was handed in, and otherwise from the global model; the engine's copy
is the flock as the run ended.
_Avoid_: the set of paths, population, ensemble, all the paths

**iteration**:
One pass of the engine's main loop, counted in `n_iters`. Per iteration the engine
splits the paths into a redraw set (no point yet, or stalled with
`restart_stalled` on) and a cohort that steps: a path either redraws or steps
this iteration, never both. The parallel wall-clock unit; an iteration serves
one or more batches (the steppers' trials as the method reads them, and the one
settle batch that buys whatever the iteration kept still owes). A flock in
which every path is parked — nothing left to do, since restarts are off — ends
the run on `all_stalled` instead of entering a batchless iteration.
Iteration 0 is the flock's placement: an empty `x` redraws unconditionally,
so starting the flock is the same operation as any later redraw. `n_iters`
counts finished iterations only — one cut short by a refused batch is not
counted, so a refused placement reports `n_iters = 0` and a refused first
step `n_iters = 1`.
_Avoid_: round (retired), epoch, iterations (as a name for the count)

**redraw set**:
The paths an iteration gives a fresh start: every path holding no point — always,
whatever `restart_stalled` says, which is how iteration 0 places the flock and
how a refused start eventually gets replaced — plus every stalled path when the
knob is on. Its size is what the engine asks the model for, one start per path,
so iteration 0 is the whole flock and a later one is only the paths that
stalled. Iteration 0 with an `x0` is the one redraw set whose starts the caller
supplied: `x0` is consumed there and no model is asked.
_Avoid_: restart list, restart batch

**cohort**:
The paths that step this iteration: neither in the redraw set nor stalled. One
`local_solver.propose` call is handed exactly this set, so a method never sees a
path the engine has stopped — not even a step that would answer `None`. It is
not the flock, and not just the redraw set's complement: a path parked by
`restart_stalled=False` belongs to neither, and sits out while the rest moves.
_Avoid_: the flock, active paths

**best**:
The run's running best, a single `f`/`x`: the best *feasible* point — the
run's reported answer, and what `NPathResult` calls `f`/`x`. The engine is the
sole owner. It starts at `nan` and stays there while no finite feasible value
has arrived, so `nan` means "no answer" and `inf` is never a sentinel - it is
left to mean a value an objective actually took. A *feasible* `-inf` is not
weighed but taken: the objective has no floor there, so it becomes `f`/`x` and
ends the run with `status` `unbounded`; a `nan`, a `+inf` or an infeasible
`-inf` changes nothing.
_Avoid_: incumbent (retired), best-so-far, current optimum

**stall / stalled**:
A path that moved nowhere this iteration — it neither redrew nor committed a
step: its local method answered no winner (`None` — its own stopping test
fired), or the point it was given was refused as not usable, whether the winner
the method named or the start the model drew: a non-finite value or gradient in
either.
Stalled is not dead: unless the run turned `restart_stalled` off, the path is
re-drawn by the global model on the next iteration; with the knob off it stays
parked — still counted, still readable for `best` — but never moved again. A
turn of the loop with every path parked that way is `all_stalled` — reachable
only when `restart_stalled` is off, since with it on a stalled path is re-drawn
rather than parked: no path can move and none is owed a start, so the run stops.
A path with no point at all is a redraw that ignores `restart_stalled`: the
machine's own initial state is not a stall. A start refused while the path held
nothing sets the bit anyway, but the no-point rule is asked first, so the flag
never sees that case.
_Avoid_: paused, idle, dead

**domain** (type `Domain`):
The object that carries everything the engine knows about the search space: the
per-dimension `(lb, ub)` sides (either may be unbounded), the per-dimension
`scale` the draws use, and the injected per-dimension samplers. It is the sole
authority on the dimension `d`, and the box every point the run serves must lie
inside: the engine checks each point a seam hands over and refuses one that
misses, instead of moving it onto a wall. Bounds are data, never a unit-cube
assumption.
_Avoid_: box (the box is only the bounded-sides part of the domain)

**sampler**:
The injected per-dimension callable that draws starts; the default is uniform on
a bounded side and normal on a free side. The *domain* remains the sole
authority on `d`.
_Avoid_: initializer, prior (as a name for the *callable*)

### Evaluation

**batch**:
The `(M, d)` points served by one pass — fused (`eval_f_g`) or value-only
(`eval_f`). The engine composes each batch from either one trial round — one
point per path a step is considering — or the settle set: the redraw starts and
the winners still owing a half.
_Avoid_: group, lot, chunk

**point**:
One candidate — one row of a batch. The four counters (below) count the billed
ones. Once settled it is a `PathPoint`: the `x`/`f`/`g` triple a path commits,
assembled only once both halves exist (see *settle*), so a point that arrived
fused travels on as one object rather than three arrays.
_Avoid_: query, sample, evaluation (retired)

**settle**:
The iteration's one fused batch that buys whatever a kept point still owes, and
the only place a produced `Point` becomes a settled `PathPoint`. A redraw start
and a winner the method never read owe both halves and take f and g from this
one pass; a winner the method read through `serve_trials` owes only its gradient
when the `eval_f` seam served it value-only — with no such seam the same call
served the fused pair, so the winner already carries both and passes through as
it stands, never served twice. Everything is bought before anything is placed,
so nothing commits before it is settled.
_Avoid_: commit batch, gradient round

**four counters** (`n_f_g_points` / `n_f_g_batches` / `n_f_points` / `n_f_batches`, all four in one `Work`):
The run's reported work, on two axes and two seams. `points` counts billed
points (a batch bills its actual size), `batches` counts billed batches (a batch
bills 1); `f_g` is the fused value+gradient pair, `f` the value-only pair. All
four are counted — "billed" means "actually served out" — and none of them
decides the stop. Points and batches are different numbers on different axes
and must never share a chart, and the `f` pair must never be read as the `f_g`
pair. They replace scipy's single `nfev`. The four live in one frozen `Work`,
reached as `result.work` and read flat on the result as well; that object is
what the budget gate is handed — as it stands at the loop head, and as a
pending batch would leave it at a serve site — so a gate reads counts on their
way into the report and cannot write them.
_Avoid_: nfev (retired)

**objective**:
The scalar function being minimized; its per-point values are the f values the
engine records. Distinct from the *cost model* (the four counters), which
counts work.
_Avoid_: loss, cost, fitness

**eval_f_g**:
The caller's per-batch callable (type `EvalFG`): points in,
`(f, gradient, feas?)` out. The seam to the objective's values and
gradients; the engine is blind to it otherwise.
_Avoid_: oracle, evaluator, black box

**eval_f**:
The optional value-only sibling (type `EvalF`): points in, `(f, feas?)` out,
no gradient.
It exists so a step's trials are read by value alone and only the winner's
gradient is bought once at *settle*, instead of once per trial. Without it
every trial is itself a fused batch.
_Avoid_: value oracle, cheap evaluator

**feasibility mask**:
The optional per-point boolean saying which served points may count toward the
reported best (`f`/`x`). It is
read as 0/1 and nothing else: only `0` (or `False`) marks a point infeasible, so
a mask holding anything else — a signed constraint margin, whose violated points
are all negative, or a `nan` — is refused with a `ValueError` instead of being
read as truthiness, where every one of those flags would have counted as
feasible and quietly promoted a point into the answer.
_Avoid_: validity mask

### Local method

**step**:
One move along a single path: propose → (trial)× → a winner, or none (`None`).
Each path takes at most one move per iteration — a step, or a redraw start; a
step consumes zero or more trials. The method lands it through `step(points)`: the
engine hands back the settled `PathPoint` `(x, f, g)` the path now stands at, so
the method moves the matching state forward. The **method** named the winner in
`propose`; the engine bought its owed halves in *settle* and commits here, so
`step` is the only commit and the only place a method may advance its model — a
secant pair is a move that was taken, and only this callback is told one was. It
is never called for a winner the engine refused (not usable) or for a path whose
stopping test held.
_Avoid_: update, move; accept, commit (as names for this handoff)

**round**:
One `serve_trials` call: one point per path of the cohort, served and reported as
one batch. It is the unit of both price and memory — a method that enumerates
reads its candidates as successive rounds, so however far it looks it holds one
point per path at a time and pays one batch per look. The method sets how many
rounds a step reads; the engine sets the width of each.
_Avoid_: trial (that is one point inside a round)

**trial**:
One served proposal within a step — this path's point on one round the method
reads through `serve_trials`. What a trial carries is the seam's business: with
`eval_f` connected it is the coords and value alone — the winner's gradient is
bought later (see *settle*), so a method that needs a gradient to rule needs the
fused seam — and without `eval_f` the same round serves the fused pair, gradient
included. There is no engine-side cap: the method's own stopping test (or its
own trial budget) ends the enumeration, one batch per round. A step that reads no
trial — a single full proposal, like the shipped method's — serves none at all.
_Avoid_: line-search round (retired)

**secant pair**:
The `(Δx, Δg)` between two consecutive points of one path — the only curvature
evidence the local method reads.
_Avoid_: gradient difference, step pair

**diagonal preconditioner**:
The local method itself: a per-coordinate inverse-curvature estimate read off
the secant pair, applied as `-precon ⊙ m` where `m` is the momentum (a blend of
the gradient and its history; the bare gradient `g` at `mu = 0`). No matrix, no
linear solve, no scalar learning rate.
_Avoid_: diagonal Newton (it is a biased `1/H` estimate, not Newton)

**floor (λ)**:
The guard that keeps a coordinate's denominator positive when the secant
vanishes (a flat but unconverged coordinate).
_Avoid_: epsilon, addend, regularizer

**warmup**:
A path's first step, before any secant pair exists: one full proposal along
`-g₀` at `alpha0`, committed as proposed. It carries no value guard of its own —
the phase enters only the direction; like the steady step, it is read by no
value and compared against no start.
_Avoid_: initialization step

### Restart

**global model** (type `GlobalModel`):
The engine's second seam, owner of *where* a redrawn path starts. It observes
every served batch as a mapping from path to the point served there
(`observe(served)`, one entry per path, so a model that only wants stacked rows
stacks them itself) and answers `propose(paths, domain)`
with one fresh start per path it was asked about, keyed by that path. `None` for
a path is the model declining it — its own stopping rule, as `None` from a
method's `propose` is the method's — and that path starts nowhere this
iteration. The default `SampleGlobalModel` fits nothing: it draws starts
from the `domain`. An `x0` the caller hands the entry states the first flock
directly and the model is not asked; with `x0=None` it draws that flock too, as
iteration 0's one ask. `BLRGlobalModel`
fits a Bayesian linear posterior and proposes the local minimum of `-log EI` —
EI there is an acquisition function deciding *where is worth trying*, not a
restart decision.
_Avoid_: restart rule, sampler (the sampler is a per-dimension draw inside the
domain)

**inner minimizer** (`minimize`, type `MinimizeFn`):
The global model's own seam, one level further in, owning *how* to descend the
acquisition the model builds (`-log EI` for `BLRGlobalModel`). The model holds
the acquisition and never searches it itself: `propose` hands the acquisition,
the ask's `domain`, the paths and a fresh subkey to `minimize`, and carries its
answer out as one start per path, `None` for a path it could not place. The
shipped default is `descent_minimize`, the engine run restart-free at the
cohort's width, so the acquisition is descended by the method it will later
drive; a caller that needs an inner knob the default does not expose — e.g. a
`time_limit`, another local solver, `budget_check`, restarts inside — passes its
own `MinimizeFn` instead.
_Avoid_: inner loop, acquisition optimizer (the model does not optimize the
acquisition, it asks for one)

**restart_stalled**:
The run's one restart bit, deciding *whether* a stall restarts: on (the
default), the stalled path is re-drawn by the global model next iteration; off,
it stays parked, and a flock that is all parked is a deadlock — the run ends on
`all_stalled` rather than idling to its horizon. It is orthogonal to *when* (the
local method's own stopping test) and *where* (the global model). A run that
never restarts needs no model of its own:
`restart_stalled=False` with the default `SampleGlobalModel` is pure descent
from the initial flock — the model runs once, drawing that flock, when `x0` is
left `None`, and never again.
_Avoid_: restart check, reset decision

### Cost model

**horizon (`max_iters`)**:
The engine's own stop, counted in iterations: every completed iteration costs
one, and iteration 0 places the flock. A deterministic bound, not a cost
proxy — one iteration serves however many trial rounds the method's own search
reads, plus the one settle batch that buys what it kept still owing, while a
flock with nothing left to do stops the run
(`all_stalled`) instead of billing a batchless iteration — so the four counters
report what a run actually billed. At least one of `max_iters`/`time_limit` must
be given, validated once at entry: `stop` asks the caller's gate like the other
two, but a gate is the caller's own predicate and the engine cannot see it
advance, so it cannot tell a gate that will close from one that never does — a
run may not rest on that alone.
_Avoid_: eval budget, cost estimate, time limit

**deadline (`time_limit`)**:
The same stop in seconds, for callers whose per-iteration cost is too uneven to
count iterations. Checked before every batch (fused or value-only) and at the
loop head, like the horizon, so the over-run is under one batch plus the one
call in flight.
_Avoid_: budget, timeout (that is a per-call concern)

**budget gate** (`budget_check`):
The caller's predicate, asked as one of the run's three stops at every loop head
and again before each batch — serve the whole batch or nothing, never a partial
prefix. It is handed the run's `Work`, the same four counts the result reports,
so it gates on any axis it likes (points, batches, both seams together) without
keeping a tally of its own. Before a batch it is handed the ledger that batch
would leave, and a batch it refuses is never charged: a ceiling on points is one
the run cannot cross, met from below by up to one batch width since a batch is
served whole or not at all. At the loop head it is handed the ledger as it
stands. It exists for a budget the engine cannot see (points, wall-clock,
dollars). It is an extra gate, never a horizon.
_Avoid_: stopping budget
