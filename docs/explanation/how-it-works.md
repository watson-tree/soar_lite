# How it works

`soar-lite` runs a flock of short local descents and restarts a path when it
stalls. This page explains the method: what each piece computes, why it is built
that way, and which trade-offs it accepts.

## The contract: one objective, value and gradient

The objective is called in batches: given a batch of points `x` of shape
`(n, d)` it returns `f` of shape `(n,)`, `∇f` of shape `(n, d)`, and optionally
a feasibility mask. The goal is to minimize `f` over the feasible points.

That is the whole entry contract. There is no Hessian and no curvature estimate:
a full Hessian is quadratic in the dimension, and with `n_paths` paths you would
need one per path — far too expensive in the medium-to-high dimension this method
targets — so the only second-order information is the secant implied by the last
pair of points. It is first-order in the sense that the caller supplies the
value and the gradient together — a zero-order objective does not fit, because
every point on a path must carry a finite gradient as well as a finite value.

Budget is counted in evaluations, and there are two ways to count them (see
Cost).

## The local step: a diagonal preconditioner

Every path descends by the same rule: a diagonal approximation of the inverse
Hessian, one entry per coordinate, estimated from secant pairs and applied
element-wise to the direction. The rule is one construction with a handful of
choices around it.

### The secant ratio

Quasi-Newton starts from the **secant equation** `B s = y`, with `B ≈ H`,
`s = x_k − x_{k−1}` and `y = g_k − g_{k−1}`. Keeping only its diagonal and
dropping the coupling (that is, assuming `yᵢ = Bᵢᵢ sᵢ`) gives directly

```text
Bᵢᵢ = yᵢ / sᵢ        ⇒        dᵢ := 1/Bᵢᵢ ≈ sᵢ / yᵢ
```

`dᵢ` is the observed displacement over the observed change in gradient on that
coordinate, with units `[x]²/[f]`, independent per coordinate. (The same object
falls out of the diagonal degeneration of the SR1 update; the secant equation is
simply the shorter road — it needs no full update formula and never stores `B`.)

A single pair `(s, y)` is noisy, so the trajectory smooths it with an EMA — **the
smoothing acts on the magnitudes of the secant pair, not on the ratio itself**:

`λ` in both denominators is the per-coordinate floor, defined in *The floor*
below.

| Variant | Numerator | Denominator | Inverse curvature | Step |
| --- | --- | --- | --- | --- |
| `"abs"` (mean magnitude) | `S₁ = EMA(√(s²))` | `Y₁ = EMA(√(y²))` | `d = S₁ / max(Y₁, λ)` | `Δx = −d ⊙ g` |
| `"rms"` (root mean square, default) | `S₂ = EMA(s²)` | `Y₂ = EMA(y²)` | `d = √(S₂ / max(Y₂, λ))` | `Δx = −d ⊙ g` |

On a single pair the two variants agree (`√(s²/y²) = |s/y|`); they differ only in
which moment is smoothed — `"abs"` averages the magnitude, `"rms"` averages its
square (the RMS), so the former is smoother on small secants and the latter more
sensitive to large ones. The EMA window is about `1/(1−β)` steps, and the default
`β = 0.63` is about `e` steps (2.7). Magnitudes rather than the signed `sᵢ` or
`yᵢ` are used to keep the estimate strictly positive: a signed value drifts and
can cross zero, which would flip the step's sign — and `d ≥ 0` is what keeps the
step descending.

The step then takes `Δx = −diag(B)⁻¹ g = −d ⊙ g`. What this does is change the
direction from a **gradient-like quantity** into a **displacement-like
quantity**: `−g` has units `[f]/[x]`, so using it directly as a displacement
makes the scale run backwards with the coordinate scale, and multiplying by `d`
supplies the missing `1/H`. It is one element-wise multiply-and-divide, with no
matrix, no linear solve, and no scalar learning rate.

Everything above steps along the bare gradient `g` — the quasi-Newton form just
reconstructed. The step can also carry a momentum, the gradient EMA
`m = (1−μ)·g + μ·m_{k−1}` with `μ ∈ [0,1)`; at `μ = 0` that is `m = g` and
nothing above changes. Because `d ≥ 0` by construction (absolute value or
square, then a positive floor), `(−d⊙g)·(−g) ≥ 0` holds coordinate-wise, so a
preconditioned gradient-like direction is a descent direction, and at `μ = 0`
the step needs no acceptance test and no line search to "verify the direction".
With `μ > 0` the direction is the momentum and that guarantee is gone — see
Momentum, where the loss is deliberate — but the closer `μ` is to zero, the more
nearly the direction still descends. The default `μ = 0.63` therefore does not
hold the guarantee.

### Unit invariance, and why there is no scalar learning rate

Rescaling a coordinate, `x̃ = c·x`, rescales the secant to `s̃ = c·s` and the
gradient to `g̃ = g/c` (hence `ỹ = y/c`), so `d̃ = s̃/ỹ = (c·s)/(y/c) = c²·d`,
and the step follows it:

```text
Δx̃ = −d̃ ⊙ g̃ = −(c²·d) ⊙ (g/c) = −c·d ⊙ g = c·Δx
```

Rescaling the value, `f̃ = c_f·f`, leaves `s` alone and scales `g̃ = c_f·g`, so
`d̃ = s/ỹ = s/(c_f·y) = d/c_f`, and the step is unchanged:

```text
Δx̃ = −(d/c_f) ⊙ (c_f·g) = −d ⊙ g = Δx
```

That is exactly the scaling a step carrying physical units should have, and it
is what lets one set of unitless knobs serve ℃, m and Pa at once: every knob is
a free coefficient with no units attached, and none of them carries `[x]`. Adam
differs at exactly one point: its `m/√v` is unitless, so the whole step scale
falls to the scalar learning rate `α`, which must therefore carry `[x]`, and with
mixed units no single `α` can be right for all coordinates at once.

### The floor: a guard on the denominator

`λ = θ·|gᵢ|` for `"abs"` (units `[f]/[x]`) or `λ = θ·gᵢ²` for `"rms"` (units
`[f]²/[x]²`), with `θ` unitless, and the denominator is `max(Y, λ)`. It is
anchored to the **current gradient** rather than to `y₀` or `|f_k|`: the floor
has to vanish with `g`, not with `y`. A flat but unconverged coordinate
(`y → 0` while `g ≠ 0`) still gets a finite step, and a genuinely converged
coordinate (`g → 0`) is caught by the stopping test before the division ever
repeats at `g ≈ 0`.

Anchoring to `g` is safe because the step then multiplies `d` and `g`
element-wise. When `Y` bottoms out, `d` is `1/|g|` up to constants — `S₁/(θ·|g|)`
in `"abs"`, `√(S₂/(θ·gᵢ²))` in `"rms"` — and the full step `−d ⊙ g` reduces to
`−(S₁/θ)·sign(g)` or `−(√S₂/√θ)·sign(g)`: a finite `[x]`-valued move in the
direction of `−sign(g)`. The `g` inside the floor cancels the `g` it is
multiplied by, which is why anchoring to `g` cannot make the step explode as
`g → 0`. The floor is a **guard**: it only touches `d` where `Y < λ`; everywhere
else `d = S/Y` stands, and how aggressive the step should be stays with the
separate knobs `alpha0` (whole-step damping), `backtrack` (grid spacing) and
`n_alpha` (grid size) — see *The warm-up step* below.

### The warm-up step

Warm-up and steady state share the **same step mechanism**: sample the `alpha`
grid `{alpha0, alpha0·backtrack, alpha0·backtrack², …}` along the current
direction, let the engine's step-wise argmin pick the winner, and let `fold`
record it. The two phases differ only in the **direction**: the first step has no
secant pair, so `d` cannot be estimated and the direction is `−g₀`; once the
first `(s, y)` exists it becomes `−d⊙g`. Everything except the direction
(`alpha0`, `backtrack`, `n_alpha`) is shared between them.

With the default `n_alpha = 1`, both phases take **one unguarded whole step**:
warm-up commits `x − α0·g₀` and steady state commits `x − α0·d⊙g`. Neither reads
a value, backtracks, or compares against its start. `needs_verdict` is therefore
**independent of the phase**: it answers True when `n_alpha > 1` and False
otherwise, because the commit is what was proposed.

One exception to unit invariance: the warm-up step is **not** unit-invariant.
`alpha0` is unitless in both phases; what changes is the direction it scales. In
the steady state `d⊙g` is already a displacement, so `alpha0` reads as a
fraction of the natural step. In the warm-up `−g₀` is a gradient-like quantity
(`x̃ = c·x` gives `g̃ = g/c`, so magnifying a coordinate shrinks its gradient),
so the same unitless `alpha0` has to be read as carrying `[x]²/[f]` — at the
default `alpha0 = 1` the warm-up step is literally `x − g₀`, a gradient quantity
where a displacement belongs. It stays one knob, not two: the grid is shared,
and only the direction's units differ.

### Stopping

A path stops when `‖g‖ ≤ τ_rel·‖g₀‖` (`τ_rel` unitless, about `1e-6`). It then
gives up, stops spending evaluations, and leaves the decision "start again or
not?" to the restart rule.

The norm is used rather than a per-coordinate `|gᵢ| ≤ τ_rel·|g₀,ᵢ|`: the
per-coordinate all-quantifier either never fires for a path whose coordinate
happens to start at zero (a phantom survivor), or fires early for a path whose
coordinate starts large. The norm reads only the aggregated gradient scale
relative to its start, so no single coordinate can hold the whole path alive or
sink it. This is also where a negative-curvature or saddle escape would attach;
this implementation does not connect one.

### Momentum

With `μ > 0` the direction stops using only this step's gradient and becomes the
**current momentum**, the EMA of the trajectory's gradients just updated with
`g_k`:

```text
m_k = (1−μ)·g_k + μ·m_{k−1},        m_k = g_k when μ = 0
direction = m_k
```

`m` has the units of `g`, so `d⊙m` is still displacement-like and the switch
costs nothing in units. It is the unprojected heavy-ball direction
(`m_k = g_k + μ(m_{k−1} − g_k)`), and it is not Nesterov's method: Nesterov
consumes the gradient difference `y` on its own, hiding the second-order units
inside a step length, and evaluates `g` at a look-ahead point, which the
propose-then-evaluate protocol cannot do.

**The default `μ = 0.63` shares its window with `β`.** Both answer the same
physical question — how fast does the geometry, or the direction, change — so
both take a window of about `e` steps. They stay two independent knobs: `β`
smooths the curvature estimate, `μ` averages past gradients, and the secant pair
keeps reading the raw `y = g_k − g_{k−1}` rather than the smoothed direction, so
moving one does not entangle the other.

**It is not projected, and it does not preserve a descent direction.** Projecting
`m` onto the `d`-metric orthogonal complement of `g` would buy the invariant
`⟨d⊙(g+m⊥), g⟩ > 0`, at the price of removing the component of `m` along `g` —
the half that damps zigzag, cancels overshoot and accumulates in flat regions.
The invariant also buys nothing where no line search runs: a descent direction
only promises that a small enough step decreases `f`, while this method takes one
unguarded whole step, read by no value and compared against no start. Convergence
rests on best-so-far accounting plus restarts and stalls, not on every step
descending; a caller who wants the guarantee can shrink `α` or run a line search
outside the engine.

**The cost is real.** Near an optimum `g → 0` while `m` decays on the `1/(1−μ)`
window, so `μ·m_{k−1}` can outweigh `g_k` and the direction is carried off by old
gradients. On a diagonal quadratic the experiment suite measures final accuracy
at `μ = 0` at machine level and `μ = 0.9` near `1e-8`: momentum buys progress
only where `d` is **structurally wrong** — the off-diagonal curvature a
per-coordinate ratio cannot express — and even there it is "use the old direction
once more", a mitigation rather than a repair. `μ = 0` is the switch that turns
it off, coordinate for coordinate back to the bare gradient.

## Restart

**For pure multi-start descent that never restarts, turn `restart_stalled` off
explicitly**: stopped paths stay where they are, spend no more budget, and
nothing wakes them; when the whole flock has stopped, the run ends with
`all_stalled` rather than idling to the horizon.

### The default is no surrogate

A restart has one job: give a stopped path a new start. That start is either
drawn straight from the `Domain` (`SampleGlobalModel`, the default) or pointed at
by a surrogate fitted on the samples already served (`BLRGlobalModel`). The
default is the former: drawing a point costs one evaluation and fits nothing, so
its cost is independent of the dimension. The sweet spot of a local method is the
landscape where `d` is right, and restarts there should be rare anyway; keeping a
surrogate ready for it trades global-side cost for a crutch most runs never lean
on.

### Three questions kept apart

A restart is assembled from three distinct decisions, and this implementation
keeps them in three different places, none answering for another:

- **When**: a path stopping is the local method's own judgment (the `τ_rel`
  stopping test, or a verdict with no finite sample to win). The engine adds no
  second stopping rule and does not guess "how far off we are".
- **Whether**: `restart_stalled`, a boolean knob, default `True`. While it is
  off, a stopped path stays stopped — `stalled` is still recorded, it still
  occupies its slot, and its point is still readable — it is simply not moved;
  while it is on, it is redrawn on the next round.
- **Where**: `global_model.propose(n, domain)`, the engine's second plugin point.

The split is deliberate: *when* is already stated by the method's stopping test,
*whether* is a policy bit, and only *where* needs a real implementation, so only
*where* becomes a plugin seam. A stopped path either stays put or gets a new point;
there is nothing in between to compare, and no third mechanism needs to weigh
the two alternatives.

### Two arms

**The sampling arm (default).** `SampleGlobalModel` fits nothing: `observe` is a
no-op, and `propose` hands back the point `Domain.draw` asked for — uniform on a
bounded side, `N(0, scale²)` on a free side, truncated normal on a half-bounded
side, and the caller's injected per-dimension sampler when one was supplied. It
does not own the flock: the first round's starts come from the entry's `x0`, and
it is asked only when a path actually restarts. So the marginal cost of a restart
is one point, one evaluation, independent of the dimension.

**The surrogate arm.** `BLRGlobalModel` fits a Bayesian linear posterior on the
samples already served (features given by an injected `Projection`), and
`propose` returns the local minimum of `−log EI` found inside the domain by
`MinimizeFn`. **EI's role here is an acquisition function** (Jones et al. 1998),
answering "where does the surrogate think is still worth trying" — it hands
back a start and makes no comparison. The descent on `−log EI` uses the
caller's injected `MinimizeFn` (usually this package's own n-path engine), so
this arm trades one inner search for a smarter start, at the cost of that inner
round's evaluations.

### A restart costs no extra gradient batch

`propose`'s answer is clipped into the domain, wrapped as a point that still
owes its `f` and `g`, and handed to `settle` together with the round's step
winners, so a restart start never buys a gradient batch of its own. With the
f-only seam connected the merge is even tighter: the restart start's value also
waits and is bought once in the same fused batch. An unusable start (`f` or `g`
not finite) does not land, and the path stays `stalled` to be redrawn next round
— it never becomes `points[i]`, so "where this path stands" cannot be polluted
by a degenerate point.

## The engine

Two vocabularies meet here. `propose` and `observe` from `GlobalModel`, and
`needs_verdict`, `verdict` and `fold` from `LocalSolver`, are the protocols a
caller implements; `settle`, `_place` and `_advance` are the engine's own
internals, named here because the order they run in is the subject and nothing
outside can call them.

### One iteration

A round first sorts the paths, then buys one batch:

1. **Redraw set**: paths with no point (the iteration-0 placeholder) or that are
   `stalled` with `restart_stalled` on.
2. **Queue**: the remaining paths. A path that is `stalled` and not being
   redrawn stays here — it proposes nothing this round and yields no winner, but
   its point is still readable (`f⋆` can be read from it).
3. **Starts**: the redraw set asks `global_model.propose(len(restarted), domain)`
   for a batch of starts. The engine checks the shape and clips into the domain;
   a wrong shape is a `ValueError` at once — the width belongs to the engine, and
   a model returning a different width fails at the batch boundary instead of
   deep inside a broadcast.
4. **Steps**: each path in the queue takes one step down one of two channels by
   `needs_verdict`, producing the winner of that step.
5. **One batch settled**: redraw starts first, then the step winners in path
   order; the **same** batch goes to `settle` to buy the gradients still owed.
6. **Landing**: a start goes through `_place` (unusable leaves it `stalled`,
   redrawn next round) and a winner through `_advance` (unusable is `stalled`, no
   `fold`).

The redraw set and the queue are disjoint, so "decide everything, then buy
together" and "decide one at a time, then buy" are the same schedule.

### Verdict and no-verdict channels

`needs_verdict(i, p)` asks "does this path need per-candidate verdicts for its
next step?". With the default `n_alpha = 1` the whole step answers False: the
method commits as proposed, the engine takes its single proposal as the winner of
this step **without reading a value first**, and the value and gradient are
bought once in `settle`'s fused batch. An `n_alpha > 1` step grid answers True:
the method proposes candidate by candidate, the engine clips and serves each, and
the method's `verdict` reads `retry`/`accept`/`give_up`.

A failure in the verdict channel has only two exits, both named by the method:
`give_up` (its convergence test fired) sets `stalled`, and `accept` with no finite
sample ever served also sets `stalled`. The engine sets no trial limit — how long
the enumeration runs is the method's business, and adding a limit would give an
already-decided judgment a second, invisible stopping rule; the price is one
contract: `verdict` must eventually return `accept` or `give_up`, so an
enumeration that never terminates hangs rather than runs slowly.

With the f-only seam connected, each trial in the verdict channel serves `f` only
(the verdict listens to values only) and the winner's gradient waits for
`settle`; without it, each trial is itself a fused batch, so `n_alpha`
candidates multiply directly into the fused batch count (see Cost).

For a start to be usable in the optimization, **both halves must be finite**: a
start with finite `f` but a non-finite gradient would be read by `propose` and
enter `fold`, and one NaN is enough to poison that coordinate's magnitude EMA
permanently. Such a start is treated as a dead start, is `stalled` on arrival, and
`restart_stalled` decides whether to redraw it.

### One argmin owner and one commit point

In a verdict step the engine commits **the lowest `f` among all served trials**,
not the last one. **The engine owns "who won" uniquely**: the method learns the
landing point only through `fold(i, step_start, winner)` and keeps no second
argmin of its own, so the bookkeeping of restarts and folds cannot drift from the
method's internal state.

A verdict receives only the **coordinate and the value**: `verdict(i, p, x, f)`,
where `p` is the step's start and `x` is the point actually served after clipping.
A trial's gradient is deliberately not in the contract — the engine buys `g` for
the winner only, and a method that needs another trial's gradient to judge is not
something this engine can drive. Correspondingly the winner's gradient is **first
seen at the commit point** (`f` decided first, `g` bought with it), and in a
no-verdict step even `f` is first seen here. So the commit gate checks **both
halves**: if either `f` or `g` is non-finite, nothing is committed and the path is
`stalled`. That makes `fold` the single commit point, and the method is never told
"here is the landing point" only to have it taken back.

### stalled is the only path state

`stalled` means the path took no step this round; redrawing a stalled path is
`restart_stalled`'s separate decision, one layer up. The flag has two writers:
the verdict channel, in either of its two exits — `give_up`, or `accept` with no
finite sample ever served — and the commit gate refusing a non-finite start or
winner. A refused start **does not** become `points[i]` — `points[i]` always
means "where this path stands" — so an unusable start stays `stalled` and is
redrawn next round instead of being clipped or forced into a degenerate point.

`stalled` and "has no point" are two different things, and only the latter is
outside `restart_stalled`'s control: the placeholder's all-nan `x` is the
machine's initial state (iteration 0 uses it to make every path go through one
redraw), so it is always redrawn; a refused point is a stall, and
`restart_stalled` decides whether that path is redrawn.

### All stalled: redraw, or end

When every path is `stalled`, the round has neither steps nor owed starts, and
only two outcomes remain, decided by one bit, `restart_stalled`:

- **On (default)**: stalled paths enter the redraw set on the next round, so every
  round has a batch to serve — the flock cycles between stalling and restarting,
  with the interval set by how quickly the local method stops.
- **Off**: stopped paths never revive themselves and the flock deadlocks; the
  engine recognizes the deadlock and ends the run with `status = all_stalled`.
  That round serves no batch, so — like a refused batch — it does not count in
  `n_iters`.

### The stop gates

`max_iters` is the engine's own horizon, counted in **iterations** (iteration 0 is
the round that places the flock). It is a deterministic stop, not a cost proxy:
how many batches a round serves is decided by the method and the evaluation
seam, from the batch that redraws starts to the method's trial batches.
`time_limit` is the wall-clock version of the same thing, for callers who cannot
say how expensive a round is. At least one of them must be given, or the entry
raises `ValueError`.
`budget_check` is the extra gate for budgets the engine cannot see (points, wall
clock, money, an external early stop).

The budget therefore carries two jobs: stopping metering and work reporting.
**Both pairs of counts are recorded** (fused and f-only; see Cost), but **the
counts never decide a stop**: `stop` reads only three predicates (horizon, wall
clock, the caller's gate). `charge` only records, `tick` records one iteration
after a completed round, and `stop` is the only stop query — three inputs
reporting one refusal reason, with the refusal not stored as state, or it would
drift from its own evidence.

The same `stop` is also the loop head's condition: a completed iteration may have
just reached the horizon, so it is asked once before a new round opens. A
mid-iteration refusal raises at the point of the request and ends the run, with
that round not counted in `n_iters` — so the middle layer never checks a return
value and never passes a flag up or down. **The gate is asked before every batch,
fused or f-only**: a verdict channel's `while` can spin many turns without
returning to the loop head, and without that question a step crossing the deadline
would run to completion.

`status` reports the first gate that holds, in the order `max_iters` →
`time_limit` → `budget_exhausted`. Two other stops are the engine's own
recognition:

- `unbounded`: some **feasible** point drives the objective to `-inf`. There is no
  lower bound there, no later trial can beat it and no step can move it, so that
  point becomes the answer (`fun = -inf`, `x` is it) and the run ends. It still
  reads the feasibility mask: an infeasible `-inf` changes nothing. The batch that
  found it is served and recorded, but the round did not finish, so — like a
  refused batch — it does not count in `n_iters`.
- `all_stalled`: with restarts off the flock deadlocks, with no path able to move
  and no path owing a start. It likewise does not count in `n_iters`.

Those two are the whole of the engine's own initiative; every remaining stop is
one of the caller's gates, so no plugin can make the loop run forever.

## Cost: two evaluation seams, two rulers

One evaluation serves a batch of `(M, d)` points, so the same run has two rulers,
answering different questions. They agree only while every batch carries exactly
one point — `n_paths = 1` with nothing to judge — and diverge the moment a batch
holds more:

| Ruler | Counts | Answers |
| --- | --- | --- |
| `points` | billed **points** (a batch counts its actual number of points) | serial efficiency: how much improvement one point buys |
| `batches` | billed **batches** (a batch counts as 1) | parallel efficiency: how much improvement one call buys |

The engine has two evaluation seams (`eval_f_g`, a fused batch; `eval_f`, values
only), each with a pair of counts, so a result carries four fields:

| Field | Counts |
| --- | --- |
| `n_f_g_points` / `n_f_g_batches` | points / batches of fused batches |
| `n_f_points` / `n_f_batches` | points / batches of f-only batches |

**Both pairs are recorded** — "billed" means "actually served": a caller may
bring a value-only `eval_f`, and that half's work is reported honestly, or the
serial ruler would erase `f`'s cost entirely in a run with the f-only seam
connected. The counts report and do not budget: `stop` never reads them.

**How many batches a round spends is decided by the evaluation seam**:

- **With `eval_f` connected**: each trial serves `f` only, and the round's one
  fused batch (`settle`) buys everything still owed — the accepted trial's
  gradient and every redraw start's value and gradient, since a start defers
  both halves. So an `n_alpha`-candidate step spends `n_alpha` f-only batches
  plus one fused batch.
- **Without `eval_f`**: the fused batch is the only way to read a value, so
  **each trial is itself a fused batch** and the winner needs no `settle` pass.
  An `n_alpha`-candidate step spends `n_alpha` fused batches, one per candidate
  as the method `retry`/`accept`s them in turn.

Both rulers must be reported: on the serial axis a flock of `n_paths` paths has
its gain divided by `n_paths`, so multi-path's book ceiling is lower; on the
parallel axis one evaluation is one parallel call's cost, and only then does
"many samples plus restart" show honestly. Neither equals wall clock — a faster
language removes fixed overhead but cannot change the structural "how many points
one evaluation must serve".

Stopping is recorded in iterations, and the four fields report rather than
budget: `batches` is a different axis from `max_iters`, so "how many batches a
full horizon spent" must be read from `n_*_batches`, never by using `max_iters` as
a cost number. **Never mix them**: one table or one plot uses one axis and must
say which, and fused-axis numbers must not be compared against f-only-axis
numbers.

## When this fits

- This is a **global** search assembled from one local method and one restart
  model: the local method descends on a par with Adam/RMSProp and pursues no
  Newton/quasi-Newton, and globality comes from the diversity of `n_paths` paths
  plus `restart_stalled` re-drawing fresh starts — from the domain, or from a
  fitted surrogate where one is wired in.
- The per-coordinate estimate ignores the coupling term `Σ_{j≠i} Hᵢⱼ sⱼ` — the
  off-diagonal Hessian entries — which lands inside `|yᵢ|` and makes `d` a biased
  estimate of `1/Hᵢᵢ` (the stronger the coupling, the bigger the bias, tending to
  systematic under-stepping: more stable, slower). That is "safe" traded for
  "potentially fast".
- The restart rate is decided jointly by `restart_stalled` and the local method's
  stopping test (`β`, `μ`, `θ`, `τ_rel`); this page gives the structure, not
  recommended values for a class of problems — those follow from the objective.
- **Not for low dimension.** With few coordinates several of the `n_paths`
  restarts collapse into the same basin, so the flock pays for paths that
  duplicate each other's work; a plain multistart local solve from a scrambled
  start set is the cheaper call, and the default per-dimension sampler —
  independent uniform/normal points, not a low-discrepancy sequence — is aimed
  at the regime above it.

## SOAR: attribution and distinction

The "soar" in the name is a nod to the **adaptive restart** framework of Mathesen
et al. 2021: it argues that the global side and the local method each have their
own stretch, joined by an **adaptive restart process** that decides where the next
start comes from — mainly by selecting a *sampling criterion*, an improvement
function such as expected improvement (EI) or probability of improvement (PI)
scored against
a global model fitted on what has been evaluated — rather than leaving global
diversity to the local search alone. `-lite`
means this package implements only the minimal version of that skeleton: restart
starts come from a `GlobalModel`. The rest is a different story:

- **Three parts become two.** SOAR strings a surrogate model, an acquisition
  rule (which start is worth buying next), and a local method into one framework.
  This package folds the surrogate and the acquisition into the one `GlobalModel`
  contract (`observe` plus `propose`), so the acquisition is half of a plugin
  seam, not a third part on its own — "where is still worth trying", never a
  restart verdict — and two plugin seams remain: the global model and the local
  solver. The fold is what permits the default arm: once surrogate and
  acquisition share one contract, an arm that fits nothing and simply draws
  satisfies it too, with no acquisition to speak of. The two plugin seams that
  survive the fold are the next two bullets, each chosen for the medium-to-high
  dimension this library targets.
- The **global model** ships two arms aimed at that regime: the default
  `SampleGlobalModel` — the model-free arm above — fits nothing and draws from
  the `Domain` with a plain per-dimension sampler, uniform on a bounded side and
  `N(0, scale²)` on a free one, independent points rather than a low-discrepancy
  sequence because the mid-to-high regime is the target; and the surrogate arm
  `BLRGlobalModel` fits a Bayesian linear posterior on inverse-stereographically
  projected coordinates (a fixed-hyperparameter port of Doumont et al. 2026),
  which cures the corner-seeking pathology a Euclidean RBF Gaussian-process
  posterior falls into in high dimension, and proposes the local minimum of
  `−log EI`. EI carries no acquisition hyperparameters of its own and is strong
  in practice, but choosing it as the default acquisition is a personal prior,
  not something the skeleton requires. SOAR's own default surrogate is that
  Gaussian process, the half of SOAR least at home in medium-to-high dimension.
- The default local method is the coordinate-wise secant-ratio preconditioner,
  first-order: the caller supplies the value and the gradient together, with no
  Hessian, no curvature model, and no direct solve of second-order information. A
  zero-order objective does not fit either — every point on a path must carry a
  finite gradient, and value-plus-gradient is the entry contract.

This package's **core agrees** with SOAR — the global side and the local method
each have their own stretch, joined by the adaptive restart process — while the
**details differ**: `-lite` is the minimal implementation of that skeleton,
not an implementation lifted from the SOAR literature. So it is **not** a pure
port of SOAR, and it implies no interface compatibility and no reproducible
results. The citation is for attribution and to locate the difference.

## References

- Conn, A. R., Gould, N. I. M., & Toint, P. L. (2000). *Trust-Region Methods*.
  SIAM/MPS. (The secant equation `B s = y` and the SR1 update; its diagonal
  degeneration is the ratio above.)
- Mathesen, L., Pedrielli, G., Ng, S. H., & Zabinsky, Z. B. (2021). Stochastic
  optimization with adaptive restart: a framework for integrated local and global
  learning. *Journal of Global Optimization*, 79, 87–110. (SOAR-style sample
  restarts.)
- Doumont, C., Fan, D., Maus, N., Gardner, J. R., Moss, H., & Pleiss, G. (2026).
  *We Still Don't Understand High-Dimensional Bayesian Optimization*.
  Proceedings of the Twenty-Ninth International Conference on Artificial
  Intelligence and Statistics (AISTATS), PMLR 300:2629–2637,
  [proceedings.mlr.press](https://proceedings.mlr.press/v300/doumont26a.html)
  — the inverse-stereographic projection the surrogate arm ports, from
  [github.com/colmont/linear-bo](https://github.com/colmont/linear-bo).
- Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization
  of expensive black-box functions. *Journal of Global Optimization*, 13,
  455–492. (The closed-form EI.)
