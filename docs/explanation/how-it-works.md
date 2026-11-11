# How it works

`soar-lite` runs a flock of short local descents and restarts a path when it
stalls. This page explains the method: what each piece computes, why it is built
that way, and which trade-offs it accepts.

A short vocabulary runs through everything below: a **path** is one descent —
one current point plus its solver state; the **flock** is the `n_paths`
current points taken together as the run's state at one moment (one point per
path); a
**batch** is the `(M, d)` points one call serves, and a **point** is one row of
it; a path's **warm-up** is its first step (no secant pair yet) and **steady
state** is every step after one exists.

Two interfaces supply the method names used below: `propose(n_points, domain)`
and `observe` belong to `GlobalModel`, while `restart`, `propose(idxs, domain,
serve_trials)` and `step` belong to `LocalSolver` — two different `propose`
methods, each called with its own arguments. One name is neither: `settle` is the
fused batch that closes an iteration, where every half the iteration's steps and
restarts still owe is bought together; `_place` is the engine's own landing
half, run once per channel, and is what calls the method's `restart(i, p)` or
`step(i, p)`. Those two are internal, explained where they run, and not callable
from outside.

## The contract: one objective, value and gradient

The objective is called in batches: given a batch of points `x` of shape
`(M, d)` — `M` points, `d` coordinates — it returns `f` of shape `(M,)`, `∇f`
of shape `(M, d)`, and optionally a feasibility mask of shape `(M,)`, `None`
meaning all feasible. The goal is to minimize `f` over the feasible points.
Infeasible points are still served, and
their gradients still drive the paths: the mask decides only what may become the
final answer.

That is the whole entry contract: one fused seam, `eval_f_g`, returning value
and gradient in one pass. A second, value-only seam (`eval_f`) can be connected
beside it for steps that read their trials one at a time; it is an optional half
of the same evaluation, never a gradient-free entry, and
[Cost](#cost-two-evaluation-seams-two-rulers) says what each buys. There is no
full Hessian and no full curvature model. A full Hessian is quadratic in the
dimension, and with `n_paths` paths you would need one per path — far too
expensive in the medium-to-high dimension this method targets. So the only
second-order information is the coordinate-wise secant ratio estimated from the
last pair of points. The entry contract is first-order: the caller supplies the
value and the gradient together, so a zero-order objective does not fit — every
point on a path must carry a finite gradient as well as a finite value.

Work is reported in two currencies, served points and batches, because one
evaluation serves a whole batch (see [Cost](#cost-two-evaluation-seams-two-rulers)).

## The local step: a diagonal preconditioner

Every path descends by the same rule: a diagonal approximation of the inverse
Hessian, one entry per coordinate, estimated from secant pairs and applied
element-wise to the direction.

### The secant ratio

Quasi-Newton starts from the **secant equation** `B s = y`, with `B ≈ H`,
`s = x_k − x_{k−1}` and `y = g_k − g_{k−1}`. Keeping only its diagonal and
dropping the coupling (that is, assuming `yᵢ = Bᵢᵢ sᵢ`) gives directly

```text
Bᵢᵢ = yᵢ / sᵢ        ⇒        preconᵢ := 1/Bᵢᵢ ≈ sᵢ / yᵢ
```

`preconᵢ` is the observed displacement over the observed change in gradient on
that coordinate, with units `[x]²/[f]`, independent per coordinate — the
per-coordinate inverse curvature. The ratio as
written can come out negative when `sᵢ` and `yᵢ` disagree in sign, which is
why the sign is discarded first: the magnitudes `|s|` and `|y|` are what get
smoothed below, never `|s/y|`. (The same quantity follows from the diagonal
version of the symmetric rank-one (SR1) update; the secant equation is simply
the shorter road — it needs no full update formula and never stores `B`.)

A single pair `(s, y)` is noisy, so the path smooths it with an EMA (an
exponential moving average) — **the smoothing acts on the magnitudes of the
secant pair, not on the ratio itself**:

| Variant | Numerator | Denominator | Inverse curvature | Step |
| --- | --- | --- | --- | --- |
| `"abs"` (mean magnitude) | `S₁ = EMA(|s|)` | `Y₁ = EMA(|y|)` | `precon = S₁ / max(Y₁, λ)` | `Δx = −precon ⊙ g` |
| `"rms"` (root mean square, default) | `S₂ = EMA(s²)` | `Y₂ = EMA(y²)` | `precon = √(S₂ / max(Y₂, λ))` | `Δx = −precon ⊙ g` |

`λ` in both denominators is the per-coordinate floor, defined in
[The floor](#the-floor-a-guard-on-the-denominator).

On a single pair the two variants agree (`|s|/|y| = |s/y|`); they differ only in
which moment is smoothed — `"abs"` averages the magnitude, so every pair counts in
proportion to its size, while `"rms"` averages the square, so large pairs
dominate. The EMA window is about `1/(1−β)` steps, where `β` is the EMA decay
rate, and the default `β = 0.63` is about 2.7 steps (`β = 0` keeps none, so one
pair decides). Magnitudes rather than the
signed `sᵢ` or `yᵢ` are used to keep the estimate strictly positive: a signed
value drifts, and a pair of opposite-sign differences can drive it through zero
— flipping `precon`'s sign or blowing it up — while smoothing `|s|` and `|y|`
keeps `precon ≥ 0`, which is what keeps the step descending. Why there is a
window at all — what it buys and what it costs — is [Smoothing](#smoothing).

The step then takes `Δx = −diag(B)⁻¹ g = −precon ⊙ g`. What this does is change
the direction from a **gradient-like quantity** into a **displacement-like
quantity**. `−g` has units `[f]/[x]`; using it directly as a displacement means
that when you magnify a coordinate (`x̃ = c·x`) the gradient shrinks (`g̃ = g/c`),
so the step shrinks instead of growing — the scale runs backwards. Multiplying
by `precon` supplies the missing `1/H`. It is one element-wise multiply by
`precon` (a ratio per coordinate), with no matrix, no linear solve, and no
scalar learning rate.

Everything above steps along the bare gradient `g` — the quasi-Newton form just
reconstructed. The step can also carry a momentum, the gradient EMA
`m = (1−μ)·g + μ·m_{k−1}` with `μ ∈ [0,1)`; at `μ = 0` that is `m = g` and
nothing above changes. Because `precon ≥ 0` by construction (absolute value or
square, then a positive floor), `(−precon⊙g)·(−g) ≥ 0` holds coordinate-wise,
so a preconditioned gradient-like direction is a descent direction, and at
`μ = 0` the direction is a guaranteed descent direction — no acceptance test is
needed, though the whole step may still fail to decrease `f`.
With `μ > 0` the direction is the momentum and that guarantee is gone — see
[Momentum](#momentum), where the loss of that guarantee is deliberate — but the
closer `μ` is to zero, the more nearly the direction still descends. The default
`μ = 0` therefore keeps the guarantee; it is `μ > 0` that gives it up, knowingly.

### Smoothing

`β` is the knob on the window: it sets how much history the curvature estimate
keeps. A fresh pair is already as current as the method knows — `y = g_k −
g_{k−1}` is the average of `H` along the step just taken, not a point sample —
so more history is never *more* accuracy; it is a trade of currentness for
stability, and the two sides are:

- **Stability.** One pair should report `1/Hᵢᵢ`, but the off-diagonal entries
  creep in: `sᵢ/yᵢ = 1/(Hᵢᵢ + Σ_{j≠i} Hᵢⱼ·sⱼ/sᵢ)`. The extra term is not a
  constant — it is set by the step's *direction*, through the ratios `sⱼ/sᵢ`.
  The same true `Hᵢᵢ` therefore comes back higher on one step and lower on the
  next, purely because the path turned; that step-to-step wobble carries no
  information about the curvature. An EMA spanning a few steps averages the
  wobble out and leaves `precon` steadier. On a rotated quadratic, where the
  direction turns hard every step, this is `β`'s whole effect.
- **Currentness.** Where the true curvature changes along the path — a
  non-quadratic landscape — every pair's reading is honest, and blending older
  pairs re-weights the estimate toward points the path has left behind. There
  the window is pure lag: it flattens the path of `precon`, but the shape it flattens
  is the real terrain, not a coupling artifact.

On a diagonal quadratic neither side applies — each pair reads `1/Hᵢᵢ` exactly
and nothing varies — so `β` is there neutral. `β = 0` is the currentness end of
the range `[0, 1)`: `precon` reads only the newest accepted secant. `β` defaults to
0.63 (a window of about 2.7 steps) — a plain default, with no deeper tuning
behind the number. (`μ` shapes the *direction* — see [Momentum](#momentum) — and
`β` the *scale*; the scale reads the raw gradient difference `g_k − g_{k−1}`,
never the smoothed `m`, so the two knobs never feed each other.)

### Unit invariance, and why there is no scalar learning rate

Rescaling a coordinate, `x̃ = c·x`, rescales the secant to `s̃ = c·s` and the
gradient to `g̃ = g/c` (hence `ỹ = y/c`), so `precon` scales by `c²`, and the
step follows it:

```text
Δx̃ = −(c²·precon) ⊙ (g/c) = −c·(precon ⊙ g) = c·Δx
```

Rescaling the value, `f̃ = c_f·f`, leaves `s` alone and scales `g̃ = c_f·g`, so
`precon` shrinks by `c_f`, and the step is unchanged:

```text
Δx̃ = −(precon/c_f) ⊙ (c_f·g) = −precon ⊙ g = Δx
```

That is exactly the scaling a step carrying physical units should have, and it
is what lets one set of unitless knobs serve variables in K, m and Pa — three
different units — at once: every knob is a free coefficient with no units
attached, and none of them carries `[x]`. Adam differs at exactly one point:
its `m/√v` is unitless, so the whole step scale falls to the scalar learning
rate `α`, which must therefore carry `[x]`, and with mixed units no single `α`
can be right for all coordinates at once. (The warm-up step is the one exception
— see [The warm-up step](#the-warm-up-step).)

### The floor: a guard on the denominator

`λ = θ·|gᵢ|` for `"abs"` (units `[f]/[x]`) or `λ = θ·gᵢ²` for `"rms"` (units
`[f]²/[x]²`), with `θ ≥ 0` unitless (default `1e-3`; `θ = 0` turns the floor
off, so `precon = S/Y` has nothing catching a vanishing `Y`), and the denominator is
`max(Y, λ)`. It is anchored to the **current gradient** rather than to the first
secant `y₀` or the current value `|f_k|`: the floor has to vanish with `g`, not
with `y`. A flat but unconverged coordinate (`y → 0` while `g ≠ 0`) still gets a
finite step, and a genuinely converged coordinate (`g → 0`) is caught by the
stopping test before the division ever repeats at `g ≈ 0`.

Anchoring to `g` is safe because the step then multiplies `precon` and `g`
element-wise. When `Y` bottoms out, `precon` is `1/|g|` up to constants —
`S₁/(θ·|g|)` in `"abs"`, `√(S₂/(θ·gᵢ²))` in `"rms"` — and the full step
`−precon ⊙ g` reduces to `−(S₁/θ)·sign(g)` or `−(√S₂/√θ)·sign(g)`: a finite
`[x]`-valued move in the direction of `−sign(g)`. The `g` inside the floor
cancels the `g` it is multiplied by, which is why anchoring to `g` cannot make
the step explode as `g → 0`. The floor is a **guard**: it only touches `precon`
where `Y < λ`; everywhere else `precon = S/Y` stands, and how aggressive the
step should be stays with the separate knob `alpha0` (whole-step damping,
default `1`) — a `DiagonalPreconditioner` parameter, not a schedule one — see
[The warm-up step](#the-warm-up-step). A line search is not a knob either: a
method that wants one enumerates trials itself and names the winner (see
[the step channel](#the-step-channel-trials-a-winner-or-no-winner)).

### The warm-up step

Warm-up and steady state share the **same step mechanism**: take the full
proposal along the current direction, and let `step` record the committed step.
The two phases differ only in the **direction**: warm-up has no secant pair yet,
so `precon` cannot be estimated and the direction is `−g₀`; once the first
`(s, y)` exists it becomes `−precon⊙m` — the momentum, equal to `g` at the
default `μ = 0` (see [Momentum](#momentum)). `alpha0` is the one knob either
phase reads.

`DiagonalPreconditioner` consults no value: both phases take **one whole step**
— warm-up commits `x − α0·g₀` and steady state commits `x − α0·precon⊙m` — so
neither runs a line search nor compares against its start. Value and gradient
still arrive together in the iteration's one fused `settle` batch, which is the
whole price of the step; the value is recorded, and no step decision reads it.
Its `propose` serves no trial and names its proposal as the winner.

A method that does want to compare takes the same seam further and pays for it
itself: it calls `serve_trials` for as many candidates as its own rule needs, per
step, and names the one that won (see
[the step channel](#the-step-channel-trials-a-winner-or-no-winner)). A step
length that shrinks across those candidates is the method's own state, re-armed
at `alpha0` when the step commits — so a step never inherits the previous one's
cheaper position.

One exception to unit invariance: the warm-up step is **not** unit-invariant.
`alpha0` is a bare number, but its effective units depend on the phase. In
the steady state `precon⊙m` is already a displacement, so `alpha0` reads as a
fraction of the natural step. In the warm-up `−g₀` is a gradient-like quantity
(`x̃ = c·x` gives `g̃ = g/c`, so magnifying a coordinate shrinks its gradient),
so here `alpha0` effectively carries `[x]²/[f]` — at the default `alpha0 = 1`
the warm-up step is literally `x − g₀`, a gradient quantity where a
displacement belongs. It stays one knob, not two: both phases read it, and only
the direction's units differ.

### Stopping

A path stops when `‖P g‖ ≤ τ_rel·‖g₀‖`, where `P` drops the components on a
bound whose descent direction points out of the box, `τ_rel` is unitless and
defaults to `1e-6` (a `DiagonalPreconditioner` knob, not a schedule one), and
`g₀` is the gradient at the path's current start, reset at each restart, and it
is **not** projected: the test is relative to the start's overall scale rather
than to each component's own, so the start's wall reaction stays in the
denominator. Dropping it would make the bar depend on which coordinates happened
to be stuck, and can push it below what those coordinates can resolve. It then
names no winner — `propose` answers `None` for it — stops spending evaluations,
and leaves the decision "start again or not?" to the restart rule.

The projection is what makes a wall a reachable stopping point. A constrained
minimum can sit on a wall carrying a perfectly large gradient — positive against
the lower wall, where the descent direction points below the box, negative
against the upper one — and the raw norm would never fall there, so the path
would spend the whole horizon proposing steps the clip pins back onto the same
point. Only components with a feasible descent direction left count, which is
the residual a projected-gradient method steps along. The drop needs both
halves: the component must sit on a bound and its descent direction must point
out of the box. A component still inside the box is never dropped, whatever its
sign; a step that merely overshoots a boundary is clipped, not pinned.

The test reads `g`, not the direction's momentum `m`. A stop is a claim about the
objective, so making it depend on the solver's history would let the same `(x,
g)` stop or run depending on how the path arrived: momentum that lags a gradient
which has already turned inward would declare a point done while a descent step
is still available. At the default `mu = 0` the two vectors coincide and the
distinction is invisible.

The norm is used rather than a per-coordinate `|gᵢ| ≤ τ_rel·|g₀,ᵢ|`: that test
asks every coordinate to clear its own threshold, so it either never fires for a
path whose coordinate happens to start at zero (a phantom survivor), or fires
early for a path whose coordinate starts large. The norm reads only the
aggregated gradient scale relative to its start, so no single coordinate can hold
the whole path alive or sink it. This is also where a negative-curvature or
saddle escape would attach; this implementation does not connect one.

### Momentum

With `μ > 0` the direction stops using only this step's gradient and becomes the
**current momentum**, the EMA of the path's gradients just updated with
`g_k`:

```text
m_k = (1−μ)·g_k + μ·m_{k−1},        m_k = g_k when μ = 0
direction = m_k
```

`m` has the units of `g`, so `precon⊙m` is still displacement-like and the switch
costs nothing in units. It is the unprojected heavy-ball direction
(`m_k = g_k + μ(m_{k−1} − g_k)`), and it is not Nesterov's method: Nesterov
consumes the gradient difference `y` on its own, hiding the second-order units
inside a step length, and evaluates `g` at a look-ahead point, which the
propose-then-evaluate protocol cannot do.

**The default is `μ = 0`, and the knob is the caller's prior.** What a gradient
memory buys is a property of two things the preconditioner cannot see:

- **The scaled spectrum.** Memory trades overshoot along steep directions
  against slow progress along flat ones, so the window that helps is set by how
  far the curvatures spread *after* `precon` has taken the per-coordinate scale out —
  the conditioning the iteration actually feels, not the one `f` has on its
  own. A wide spread wants a long window, a spectrum already near one scale
  wants none, and the ideal damping follows the *shape* of that spread as well
  as its widest ratio, so it is never one number: the answer differs from
  problem to problem, and more sharply from coordinate to coordinate (a
  per-coordinate `μ` is one of the things Adam's first moment buys, at the
  price of a knob per dimension).
- **Noise in the objective.** An EMA shrinks the variance of the direction at
  the cost of lag, so an objective whose gradient is noisy — a simulation, an
  fp32 evaluation, an average over samples — is the case a memory is *for*.
  A deterministic, well-resolved objective is not.

Zeroing both priors is the default: no memory. That is also the branch that
keeps a descending direction, so the default is the assumption-free one rather
than a tuned compromise. A caller who knows their spectrum, or their noise,
picks `μ` on purpose, from that knowledge rather than from a number here.

**`μ` does not repair the diagonal approximation.** It adds no off-diagonal
curvature, changes no `precon`, and cannot turn a wrong direction into a right one;
what it changes is the iteration's *response* along the direction `precon` already
gives. Where the off-diagonal part of `H` is dropped, it stays dropped —
recovering it is what the global side is for, and nothing on the local side
stands in for it.

**It is not projected, and it does not preserve a descent direction.** Projecting
`m` onto `g`'s orthogonal complement under the metric weighted by `precon` would give
the invariant `⟨precon⊙(g+m⊥), g⟩ > 0`, at the price of removing the component of `m`
along `g` —
and that component is exactly what damps zigzag, cancels overshoot, and
accumulates progress in flat regions.
The invariant also buys nothing where no line search runs: a descent direction
only promises that a small enough step decreases `f`, while this method takes
one whole step — value and gradient read together, compared against no start,
and clipped to the box as its only correction. Convergence rests on best-so-far
accounting plus restarts and stalls, not on every step descending; a caller who
wants the guarantee can leave `μ` at its default, shrink `alpha0`, or run a line
search outside the engine.

**The cost is real.** Near an optimum `g → 0` while `m` decays on the `1/(1−μ)`
window, so `μ·m_{k−1}` can outweigh `g_k` and the direction is carried off by old
gradients. `μ = 0` is the switch that turns it off, coordinate for coordinate
back to the bare gradient, and it is the default (above).

## Restart

**For pure multi-start descent that never restarts, turn `restart_stalled` off
explicitly**: stopped paths stay where they are, spend no more budget, and
nothing wakes them; when the whole flock has stopped, the run ends with
`all_stalled` rather than idling to the horizon.

### The default is sampling, not a surrogate

A restart has one job: give a stopped path a new start. That start is either
drawn straight from the `Domain` (`SampleGlobalModel`, the default) or pointed at
by a surrogate fitted on the samples already served (`BLRGlobalModel`, Bayesian
linear regression). The default is the former: drawing a point costs one row in
the next batch and fits nothing, so its model cost is independent of the
dimension.
The sweet spot of a local method is the landscape where `precon` (the step scale) is
right, and restarts there should be rare anyway; keeping a surrogate ready for
it trades global-side cost for a crutch most runs never lean on.

### Three questions kept apart

A restart is assembled from three distinct decisions, and this implementation
keeps them in three different places, none answering for another:

- **When**: a path stopping is the local method's own judgment (the `τ_rel`
  stopping test, or a step that read only non-finite trials and declined to name
  a winner), or the commit gate refusing a point it cannot use — a non-finite
  value or gradient, on the winner the method named or the start the model drew.
  The engine adds no second *convergence* rule and does not guess "how far off
  we are".
- **Whether**: `restart_stalled`, a boolean knob, default `True`. While it is
  off, a stopped path stays stopped — `stalled` is still recorded, it still
  occupies its slot, and its last usable point stays in the run's best — only
  the path itself stops moving; while it is on, it is re-drawn on the next
  iteration.
- **Where**: `global_model.propose(n_points, domain)`, the engine's second
  plugin point.

The split is deliberate: *when* is already stated by the method's stopping test,
*whether* is a policy bit, and only *where* needs a real implementation, so only
*where* becomes a plugin seam. A stopped path either stays put or gets a new point;
there is nothing in between to compare, and no third mechanism needs to weigh
the two alternatives.

### Two restart arms: sampling and surrogate

**The sampling arm (default).** `SampleGlobalModel` fits nothing: `observe` is a
no-op, and `propose` just returns draws from `Domain.draw` — uniform on a bounded
dimension, `N(0, scale²)` on a free one (both sides infinite), truncated normal
on a half-bounded one (exactly one finite side), and the caller's injected
per-dimension sampler when one was
supplied. It does not own the flock: iteration 0's starts are the caller's `x0`
when one was handed in (one initial point per path) — placed as handed, since
the entry validates and refuses an `x0` outside the box rather than clipping it
— and otherwise the model draws them. Every later restart asks the model. So the
marginal cost of a restart is one point — one row in the next batch, with no
model fit behind it. A higher dimension makes that row longer, not the batch
bigger; the surrogate arm below is what pays for a fit.

**The surrogate arm.** `BLRGlobalModel` fits a Bayesian linear posterior on the
samples already served (features given by an injected `Projection`), and
`propose` returns the local minimizers of `−log EI` (expected improvement) found
inside the domain by `MinimizeFn`. Until `observe` has given the posterior a
spread to standardize by, `propose` has nothing to fit and answers with the
domain's own draw, never calling `minimize` — see
[Surrogate restarts](../how-to/surrogate-restarts.md). **EI's role here is an
acquisition function** (Jones et al. 1998), answering "where does the surrogate
think is still worth trying" — it hands back a start and makes no comparison.
The descent on `−log EI` uses the caller's injected `MinimizeFn` (usually this
package's own n-path engine), so this arm trades one inner search for a smarter
start, at the cost of that inner search's evaluations.

### A restart costs no extra gradient batch

`propose`'s answer is clipped into the domain, wrapped as a point that still
owes its `f` and `g`, and handed to `settle` together with the iteration's step
winners, so a restart start never buys a gradient batch of its own. With the
value-only seam connected the merge is even tighter: the restart start's value
is deferred and bought once in the same fused batch. An unusable start (`f` or `g`
not finite) does not land, and the path stays `stalled` — re-drawn next
iteration
when `restart_stalled` is on, or regardless of that flag when the path holds no
point at all. It never becomes the point the path holds, so "where this path
stands" cannot be polluted by a degenerate point.

## The engine

### One iteration

An iteration first puts the paths in path-index order, then serves its batches in
that order — a fused `settle` batch when anything is still owed (step 5),
preceded by whatever trial batches its steps serve out of `serve_trials` (see
Cost):

1. **Redraw set**: paths that hold no point — iteration 0's whole flock, always,
   since the placement is one start per path whether it comes from `x0` or from
   the model — or that are `stalled` with `restart_stalled` on.
2. **Cohort**: the paths that are neither re-drawn nor `stalled`. A stopped path
   is asked nothing — not even a step that would answer `None` — so a method
   cannot un-stall a path the engine has already stopped; its last usable point
   stays in the run's best.
3. **Starts**: the redraw set asks `global_model.propose(len(restarted), domain)`
   for a batch of starts — except on iteration 0 with an `x0`, which the engine
   places as given and never asks a model for. The engine checks the shape and
   clips into the domain;
   a wrong shape is a `ValueError` at once — the width belongs to the engine, and
   a model returning a different width fails at the batch boundary instead of
   deep inside a broadcast.
4. **Steps**: the cohort asks `local_solver.propose(idxs, domain,
   serve_trials)` once, for the whole cohort, producing each path's winner or
   `None`.
5. **The `settle` batch** (issued only when something is still owed): redraw
   starts first, then the step winners in path order; the same batch buys the
   halves still owed — each read winner's gradient when the `eval_f` seam served
   it value-only, and both halves of anything named unread or drawn as a start.
   Nothing owed (no starts, and winners already carrying their gradient) means
   no batch is issued here and the counters do not move.
6. **Landing**: `_place` commits each usable point and calls that channel's
   hook — a start goes through `restart` (unusable leaves it `stalled`: re-drawn
   next iteration when `restart_stalled` is on, and regardless of that flag when
   it holds no point) and a winner through `step` (unusable is `stalled`, no
   `step`).

The redraw set and the cohort are disjoint, so "decide everything, then buy
together" and "decide one at a time, then buy" are the same schedule.

Every batch the run spends goes through `global_model.observe` on the way back
— the trial batches and the `settle` batch alike, in the order they are served —
because the model is the run's evidence sink. It sees a batch before the run
decides anything on it, and it sees the batches the run never decides on too (a
trial that lost its step, the batch that ended an unbounded run), because the
work was served. A point can arrive twice: a step's trial comes for its value,
and again for its gradient if it wins.

### The step channel: trials, a winner, or no winner

`propose` is one call for the whole cohort and answers one of two things per
path: a **winner**, or `None` to stop that path where it stands. That answer is
the whole of the protocol — the engine has no second question to ask, and
`propose` cannot answer one path at a time.

A winner is a `Pending` point, and what it carries says what the step read. A
**read** winner carries `x` and `f`: the method served it as a trial through
`serve_trials` and ruled on the value itself, so on the value-only seam its
gradient is still owed and `settle` buys it in the iteration's fused batch (on
the fused seam the trial brought the gradient too, and nothing is owed). An
**unread** winner carries `x` alone: the method named its proposal without
reading anything, so both halves are owed and both arrive in that same batch —
exactly what `DiagonalPreconditioner` does.

Which of the two a method sends is its own business. The engine stocks
`serve_trials`, calls `propose` once, and settles whatever comes back, so the
number of trials a step reads, the order it reads them in, and the rule that
picks among them are all inside the method: a line search, a grid, an early
exit, and the case of no comparison at all are one code path from where the
engine stands.

Two things set `stalled`: `propose` answering `None` (whatever the method's
reason — its stopping test, or a step that read only non-finite trials and
declined to name a winner), and the commit gate refusing a point whose `f` or
`g` is non-finite — an unusable start or winner. The engine sets no trial
limit — how long a step enumerates is the method's business, and a limit would
give an already-decided step a second, invisible stopping rule — so a method
whose enumeration never terminates blocks the process rather than running
slowly.

`serve_trials` takes the batch of candidate points and returns one `Pending` per
row. When `eval_f` is connected a trial costs its value alone; when it is not,
the same call serves the fused pair and the returned `Pending` carries both, so
the same step code costs `m` value-only batches on one seam and `m` fused ones on
the other (see [Cost](#cost-two-evaluation-seams-two-rulers)). A trial is served
for its value, not its fate: every call is reported to the run's best and to the
global model before the method decides anything, including the trials it goes on
to reject.

For a start to be usable in the optimization, **both halves must be finite**: a
start with finite `f` but a non-finite gradient would look usable on the value
alone, yet the gradient is what the next secant reads — one NaN is enough to
poison that coordinate's magnitude EMA permanently. Such a start is treated as
a dead start and is `stalled` on arrival; `restart_stalled` decides whether a
path that holds a point is re-drawn, and a path holding none is re-drawn either
way (see [Stalled is the only path state](#stalled-is-the-only-path-state)).

### The current point is state, not an argument

A method that steps from a point has to know where the path stands, and the
contract hands that point over in exactly two places: `restart(i, p)` places it
and `step(i, p)` moves it. Both are the engine's own commit points, so a method
that keeps the point it was handed holds the same one the engine holds.
`propose` receives the cohort's indices, the run's `domain` and the trial seam —
never the points; each method reads those out of its own state. The engine
commits only through `step`, so the point cannot move under a step's enumeration,
and the start a step works from is whichever point the last `restart` or `step`
left.

Keeping the point is the least a stepping method keeps, not the whole of what it
may keep: the abstract contract asks only that `propose` can be answered from the
indices it was handed, so a method whose step needs no history keeps no history.
`DiagonalPreconditioner` keeps more, and that extra is the concrete method's
requirement rather than the contract's — its secant pair `(s, y)` is read off the
point it stepped from together with the point it landed on, so it is the default,
not the interface, that needs the previous point as well as the current one.

### The method owns who won; `step` owns the commit

Which candidate a step keeps is decided inside the step: the method reads
whatever trials its own rule needs and names the winner, so a step that compares
six candidates and one that compares none are the same call from where the
engine stands. What the engine owns is the **price**: it batches whatever
gradients the iteration still owes into one fused pass, applies the finite gate,
and clips nothing that a method already placed. The method learns the landing
point only through `step(i, p)`, with `p` the point the engine committed, and
keeps no second copy of what the engine bought.

A trial carries its **coordinate, its value, and — on the fused seam, where the
value arrives with it — its gradient**, because a value is what a selection rule
reads; the topmost candidate rarely needs a gradient, so the value-only seam
does not buy one. What the commit point buys is whatever the winner still owes —
its gradient under the value-only seam — and for an unread winner, `f` is first
seen there too. So the commit gate checks **both halves**: if either `f` or `g`
is non-finite, nothing is committed and the path is `stalled`. That makes
`step` the single commit point: a landing point the method is told about is
never taken back.

### Stalled is the only path state

`stalled` means the path took no step this iteration; redrawing a stalled path is
`restart_stalled`'s separate decision, one layer up. The flag is written in two
places: inside the step — `propose` answering `None` — and by the commit gate,
which refuses a non-finite start or winner. A refused start **does not** become
the point the path holds, the slot that always means "where this path stands" —
so an unusable start stays `stalled` — re-drawn next iteration when
`restart_stalled` is on, or regardless of that flag when the path holds no
point — rather than being clipped or forced into a degenerate point.

`stalled` and "has no point" are two different things, and only the latter is
outside `restart_stalled`'s control. A path starts with no point at all, holding
nothing to be read (iteration 0 uses it to make every path go through one
redraw), so it is always re-drawn. A refused point is a stall, and
`restart_stalled` decides whether it is re-drawn — but a refused start leaves a
path that held nothing still holding nothing, so the no-point rule is asked
first and the flag never sees that case. In short: `restart_stalled` governs a
path that had a point to keep.

### All stalled: redraw, or end

When every path is `stalled`, the iteration has no cohort at all — nothing
steps. What is owed a start depends on one bit, `restart_stalled`, and so does
the outcome:

- **On (default)**: stalled paths enter the redraw set on the next iteration, so every
  iteration has a batch to serve — the flock cycles between stalling and restarting,
  with the interval set by how quickly the local method stops.
- **Off**: stopped paths never revive themselves and the flock deadlocks; the
  engine recognizes the deadlock and ends the run with `status = all_stalled`.
  That iteration serves no batch at all, so it is not counted in `n_iters` — the same
  rule that keeps a refused batch's half-iteration out of the count.

### The stop gates

`max_iters` is the engine's own horizon, counted in **iterations** (iteration 0 is
the iteration that places the flock). `max_iters=0` stops at the first loop-head
ask, before that placement, so nothing is placed and `n_iters` stays 0. It is a
deterministic stop, not a cost proxy:
how many batches an iteration serves is decided by the method and the evaluation
seam, from the batch that redraws starts to the method's trial batches.
`time_limit` is the wall-clock version of the same thing, for callers who cannot
say how expensive an iteration is. At least one of them must be given, or the entry
raises `ValueError`.
`budget_check` is the extra gate for budgets the engine cannot see (points,
money, an external early stop). It is handed the run's `Work`, the four counts
the result reports, so the ceiling is set on the engine's own ledger rather than
on a second tally the caller keeps in step by hand. Before a batch it is handed
the ledger **that batch would leave**, and the batch is charged only if the gate
allows it: a ceiling on points is one the run cannot cross, and the counts a
refusal was decided on never enter the result. Since a batch is served whole or
not at all, such a ceiling is met from below - the run stops within one batch
width of it. At the loop head, with no batch pending, the gate is handed the
ledger as it stands.

The budget object therefore holds two separate things: the three stop predicates
and the four work counts (one frozen `Work`, replaced on each charge).
**Both pairs of counts are recorded** (fused and value-only; see
[Cost](#cost-two-evaluation-seams-two-rulers)), but **the
counts never decide a stop**: the stop query reads only three predicates (horizon, wall
clock, the caller's gate). That query is asked at two heights: `stop` for the run's
ledger at the loop head, `admit` for the ledger a batch would leave - `admit`
projects the charge, asks, and commits it only when the gate allows it, so a
refusal leaves no trace in the counts. `tick` records one iteration after a
completed iteration. The refusal is **not**
stored as state: a cached flag could drift from the counts that justify it.
(`stop`, `admit` and `tick` are the budget object's whole surface; the projection
and the charge are its implementation.)

The loop head asks `stop` before each iteration opens: a completed iteration may
have just reached the horizon, so the run never begins one it cannot finish. A
mid-iteration refusal raises at the point of the request and ends the run, with
that iteration not counted in `n_iters` — so the engine's loop never checks a return
value and never passes a flag up or down. **The gate is asked before every batch,
fused or value-only**: a step's enumeration can spin many turns without
returning to the loop head, and without that question a step crossing the
deadline would run to completion.

`status` reports the first gate that holds, in the order `max_iters` →
`time_limit` → `budget_exhausted`. Two other stops are the engine's own
recognition:

- `unbounded`: some **feasible** point drives the objective to `-inf`. There is no
  lower bound there, no later trial can beat it and no step can move it, so that
  point becomes the answer (`f = -inf`, `x` is it) and the run ends. It still
  reads the feasibility mask: an infeasible `-inf` changes nothing. The batch that
  found it is served and recorded, but the iteration did not finish, so it is not
  counted in `n_iters` — the same rule as a refused batch.
- `all_stalled`: with restarts off the flock deadlocks, with no path able to move
  and no path owing a start. It likewise does not count in `n_iters`.

Those two are the whole of the engine's own initiative; every remaining stop is
one of the caller's gates. No plugin can bypass those gates and the schedule
itself is bounded — but a plugin method that never returns (a `propose` whose
enumeration never terminates, say) still blocks the process indefinitely.

## Cost: two evaluation seams, two rulers

One evaluation serves a batch of `(M, d)` points, so the same run has two rulers,
answering different questions. They agree only while every batch carries exactly
one point — `n_paths = 1` with nothing to read — and diverge the moment a batch
holds more:

| Ruler | Counts | Answers |
| --- | --- | --- |
| `points` | **points** served (a batch counts its actual number of points) | serial efficiency: how much improvement one point buys |
| `batches` | **batches** served (a batch counts as 1) | parallel efficiency: how much improvement one call buys |

The engine has two evaluation seams (`eval_f_g`, a fused batch; `eval_f`, values
only), each with a pair of counts, so `result.work` carries four:

| Field | Counts |
| --- | --- |
| `n_f_g_points` / `n_f_g_batches` | points / batches of fused batches |
| `n_f_points` / `n_f_batches` | points / batches of value-only work |

**Both pairs are recorded** — "billed" means "actually served". A caller may
bring a value-only `eval_f`, and that half's work is reported honestly;
otherwise the serial ruler would erase `f`'s cost entirely in a run with the
value-only seam connected. The counts report and do not budget: not one of them
stops a run on its own. The stop query reads three predicates (the horizon, the
clock, the caller's gate), and only that last one is the caller's own code —
the `budget_check` is handed the `Work` and may read any of the four counts.

**How many batches an iteration spends is decided by the step and by the
evaluation seam**:

- **With `eval_f` connected**: each trial a step reads serves `f` only, and the
  iteration's one fused batch (`settle`) buys everything still owed — each read
  winner's gradient, and both halves of every unread winner and redraw start. A
  step that reads `m` trials therefore spends `m` value-only batches plus the
  iteration's one fused batch, whatever `m` is: reading more candidates costs
  serial work, never another gradient pass.
- **Without `eval_f`**: the fused pass is the only way to read a value, so
  **each trial is itself a fused batch** — its value and gradient arrive
  together, and `settle` has nothing left to buy for it. An unread winner and a
  redraw start still go through `settle`, since both owe both halves. The same
  step that cost `m` value-only batches costs `m` fused batches here, and
  nothing is served for value alone: the `f` pair stays at zero.

Both rulers must be reported: on the serial axis a flock of `n_paths` paths
spreads the same point budget across the paths, so each gets about `1/n_paths`
of the steps and that ruler shows multi-path as worse per point; on the parallel
axis one evaluation is one parallel call's cost, and only then does
"many samples plus restart" show honestly. Neither equals wall clock — a faster
language removes fixed overhead but cannot change the structural "how many points
one evaluation must serve".

Stopping is recorded in iterations, and the four counts report rather than
budget: `batches` is a different quantity from `max_iters`, so "how many batches a
full horizon spent" must be read from `n_*_batches`, never by using `max_iters` as
a cost number. **Never mix them**: one table or one plot uses one axis and must
say which, and fused-axis numbers must not be compared against value-only
ones. The [Tutorial](../tutorial/getting-started.md#5-what-a-batch-and-a-point-cost)
reads the same counters off a worked result.

## When this fits

- This is a **global** search assembled from one local method and one restart
  model: the local method is designed to be competitive with Adam/RMSProp and
  builds no full Newton/quasi-Newton model, and globality comes from the
  diversity of `n_paths` paths plus `restart_stalled` re-drawing fresh starts —
  from the domain, or from a fitted surrogate where one is wired in.
- The per-coordinate estimate ignores the coupling term `Σ_{j≠i} Hᵢⱼ sⱼ`; since
  `yᵢ = Σ_j Hᵢⱼ sⱼ`, the off-diagonal entries stay inside `|yᵢ|` and make `precon` a
  biased estimate of `1/Hᵢᵢ` (the stronger the coupling, the bigger the bias —
  typically under-stepping, more stable and slower, though the terms can also
  cancel). That is "safe" traded for "potentially fast".
- The restart rate is decided jointly by `restart_stalled` and the local method's
  stopping test (`τ_rel`), together with its step, curvature and momentum
  parameters (`β`, `μ`, `θ`); this page gives the structure, not
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
own job, joined by an **adaptive restart process** that decides where the next
start comes from — mainly by choosing an *acquisition function*, an improvement
function such as expected improvement (EI) or probability of improvement (PI)
scored against a global model fitted on what has been evaluated — rather than
leaving global diversity to the local search alone. The framework's reference
implementation is
[SOAR-Algorithmic-Framework](https://github.com/Lmathesen/SOAR-Algorithmic-Framework);
its own citation request names a 2018 manuscript submitted for publication under
this title, with two authors — the journal article above is the same title in
its published form, with four. `-lite` means this package implements only the
minimal version of that skeleton: restart starts come from a `GlobalModel`. The
details differ:

- **Three parts become two.** SOAR strings a surrogate model, an acquisition
  rule (which start is worth buying next), and a local method into one framework.
  This package folds the surrogate and the acquisition into the one `GlobalModel`
  contract (`observe` plus `propose`), so the acquisition is half of a plugin
  seam, not a third part on its own — "where is still worth trying", never a
  restart decision — and two plugin seams remain: the global model and the local
  solver, both chosen for the medium-to-high dimension this library targets. The
  fold is what permits the default arm: once surrogate and acquisition share one
  contract, an arm that fits nothing and simply draws satisfies it too, with no
  acquisition to speak of.
- The **global model** ships two arms: the default
  `SampleGlobalModel` — the model-free arm above — fits nothing and draws from
  the `Domain` with a plain per-dimension sampler, uniform on a bounded side and
  `N(0, scale²)` on a free one, independent points rather than a low-discrepancy
  sequence because the mid-to-high regime is the target. The surrogate arm
  `BLRGlobalModel` fits a Bayesian linear posterior on projected coordinates —
  inverse-stereographic by default (`SphereProjection`), `IdentityProjection` the
  injectable alternate for a task whose optimum sits on a boundary — and that
  geometry cures the corner-seeking pathology a Euclidean RBF Gaussian-process
  posterior falls into in high dimension. What the bend adds is a prior that the
  optimum is interior: a linear head gets a bowl under it, an RBF a compactified
  metric weighted toward the center, and both are that one prior, so the cure is
  the prior's — on a task whose optimum sits on a corner the ranking inverts,
  which is what `IdentityProjection` is for (Tripp 2026). `propose` returns the
  local minimum of
  `−log EI`. It is a fixed-hyperparameter port of Doumont et al. 2026. EI carries
  no acquisition hyperparameters of its own and is strong in practice, but
  choosing it as the default acquisition is a personal prior,
  not something the skeleton requires. SOAR's own default surrogate is that
  Gaussian process, the part of its design least suited to medium-to-high
  dimension.
- The default local method is the coordinate-wise secant-ratio preconditioner,
  first-order: the caller supplies the value and the gradient together, with no
  full Hessian, no full curvature model, and no direct solve of second-order
  information. A zero-order objective does not fit either — every point on a
  path must carry a finite gradient, and value-plus-gradient is the entry
  contract. A derivative-free fallback (a finite-difference gradient, say) would
  reverse that contract rather than extend it, so it is a decision to argue for,
  not a default to add.

This package's **core agrees** with SOAR — the global side and the local method
each have their own job, joined by the adaptive restart process — while the
**details differ**: `-lite` is the minimal implementation of that skeleton,
not an implementation lifted from the SOAR literature. So it is **not** a pure
port of SOAR, and it implies no interface compatibility and no reproducible
results. The citation is for attribution and to locate the difference.

## References

- Conn, A. R., Gould, N. I. M., & Toint, P. L. (2000). *Trust region methods*.
  Society for Industrial and Applied Mathematics.
  [doi.org/10.1137/1.9780898719857](https://doi.org/10.1137/1.9780898719857).
  The textbook account of the secant equation `B s = y` and the SR1 update,
  whose diagonal degeneration is [the secant ratio](#the-secant-ratio).
- Mathesen, L., Pedrielli, G., Ng, S. H., & Zabinsky, Z. B. (2021). Stochastic
  optimization with adaptive restart: A framework for integrated local and
  global learning. *Journal of Global Optimization, 79*(1), 87–110.
  [doi.org/10.1007/s10898-020-00937-5](https://doi.org/10.1007/s10898-020-00937-5).
  The restart skeleton this package implements a minimal version of.
- Doumont, C., Fan, D., Maus, N., Gardner, J. R., Moss, H., & Pleiss, G. (2026).
  We still don't understand high-dimensional Bayesian optimization. In
  *Proceedings of the Twenty-Ninth International Conference on Artificial
  Intelligence and Statistics* (AISTATS), PMLR 300, 2629–2637.
  [proceedings.mlr.press](https://proceedings.mlr.press/v300/doumont26a.html).
  The inverse-stereographic projection the surrogate arm ports, from
  [colmont/linear-bo](https://github.com/colmont/linear-bo).
- Tripp, A. (2026). *Why does spherical-linear BO work?* [Report]. GitHub.
  [github.com/AustinT/2026-07-high-dim-bo](https://github.com/AustinT/2026-07-high-dim-bo/blob/8491b39e143c/REPORT.md),
  with a companion post at
  [austintripp.ca](https://www.austintripp.ca/blog/2026-07-25-spherical-linear-high-dim-bo/).
  The study that isolates what the bend contributes: the projection is a prior
  that the optimum is interior, and moving the optimum to a corner reverses the
  ranking of the methods compared.
- Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global
  optimization of expensive black-box functions. *Journal of Global
  Optimization, 13*(4), 455–492.
  [doi.org/10.1023/A:1008306431147](https://doi.org/10.1023/A:1008306431147).
  The closed-form expected improvement.
