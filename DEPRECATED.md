# Deprecated directions

Archived design directions that were explored and then abandoned or
superseded. Each entry records what was tried, why it failed, and the
evidence, so a future reader does not re-run the same line. Nothing in this
file is shipped: probe scripts cited here live in `.scratch/` and are not
committed, every number that matters is reproduced inline.

Entries are records: nothing is deleted, and nothing is rewritten to assert
something else. A `Status` line states the delivered code at the time of
writing; when the code changes, a new entry is appended below and the old claim
stays as written.

Two more rules:

- **Only abandoned or superseded directions belong here.** The direction that
  remains in the code stays in the `How it works` page / `DESIGN.md`; this
  file archives what was dropped and what replaced it, not the code's current
  shape.
- **Numbers are never re-edited.** A measured value stays as measured; a new
  measurement is a new entry, and the old numbers rest.

## Adaptive momentum window (residual-triggered `β`)

**Status: deprecated — rejected at the mechanism level, before any delivered
file changed.** `soar_lite/local_solver.py` still blends with the constant `β`
(`beta=0.95`, window 20) and the constant `μ` (`mu=0`, opt-in); the current
momentum `m_k = (1−μ)g_k + μ·m_{k−1}` stands.

### What it was

Make the momentum weight adaptive: replace the constant `μ` with a per-step
coefficient

    β_eff = β₀^dist,   dist = ‖m_{k−1} − g_k‖ / den

so the window shrinks when the old smoothed direction `m_{k−1}` disagrees with
the new gradient `g_k`, and holds when they agree. `β₀ ∈ (0, 1)` was the new
hyperparameter; `β_eff` inherited the distance-to-window reading from `μ`
(`window ≈ 1/(1−β_eff)`). Three denominators were tried:

- `den = ‖g_k‖` — the new gradient;
- `den = ‖g₀‖` — the path's cold-start gradient;
- `den = ‖m_{k−1}‖` — the momentum itself.

### Why it was plausible

On a landscape whose local complexity varies, a fixed window is always wrong
somewhere: too long it smears sharp turns, too short it drops the flat-direction
build-up. A distance trigger looked like a way to turn momentum off where it
hurts and keep it where it pays.

### Why it fails

1. **The signal is flat where complexity varies.** On Rosenbrock (d = 20) the
   gradient norm spans ~5.9e3× across path deciles (114 → 0.0195), yet the
   `‖g‖`-denominator `β_eff` spans only 0.10 (0.82–0.92). The distance carries
   no local-complexity information, so the trigger fires on noise, not terrain.
2. **Adaptive is never better than fixed at its own mean — where the mean is
   well-defined.** For the `‖g‖` and `‖m‖` denominators each arm collapses to
   "a smaller constant β plus noise", and the noise costs via Jensen's
   inequality: `E[g(β_eff)] ≥ g(E[β_eff])` with `g` locally convex. The penalty
   scales with the local steepness of the fixed-β outcome curve — ~5× on the
   rotated quadratic (steep), only ~8–15% on Rastrigin (gentle). See the
   control table below.
3. **The `g₀` denominator is degenerate** — `‖g₀‖` is the cold-start norm, so on a
   descending path `dist → 0` and `β_eff → β₀^0 = 1` for *any* `β₀`; the mean
   pins at ~0.99 and `β₀` only matters in the rare deep-tail steps. Its
   occasional hard cuts behave as "momentarily bare", which happens to help on
   Rastrigin (bare is the optimum there) but never approaches what `μ = 0` does
   directly, and is catastrophic on the rotated quadratic (worse than both the
   mean-matched constant and bare).
4. **Added failure mode.** On Rosenbrock all three `‖m‖` arms are bimodal: four
   of five seeds reach ~1e-5, one seed degrades to the bare median (3.9867).
   The constant `beta=0.95` has no such mode (worst seed 6.1e-07).

### The measurements (5 seeds × 3000 batches, n_paths = 1, median)

| arm | rot_quad (cond 1e6) | rastrigin (d 100) | Rosenbrock (d 20) |
| --- | --- | --- | --- |
| bare (`μ = 0`) | 57.83 | **461.66** | 3.9867 |
| fixed `β = 0.95` | **2.7689** | 1748.5 | **2.7957e-08** |
| g₀-0.37 | 47.97 | 1009.9 | 0.86721 |
| m-0.37 | 48.91 | 571.1 | 1.608e-04 |
| m-0.5 | 43.38 | 745.2 | 8.528e-05 |
| m-0.69 | 33.34 | 1160.1 | 2.491e-05 |

`‖g‖`-denominator medians (same budget, earlier probe): rot_quad adapt-0.69 =
17.65; rastrigin adapt-0.69 = 1386, adapt-0.5 = 1029, adapt-0.37 = 731.3.

The shortest `β₀` of each denominator trends monotonically *toward* `bare`
rather than crossing the per-terrain optimum — lowering `β₀` is just "leave less
memory", which the `μ` dial already does exactly.

### Mean-matched control (the decisive test)

Each adaptive arm is compared to a fixed `β` set to that arm's measured mean
`β_eff`. If the rule were doing anything beyond re-selecting a constant, it
would beat this control; it does not.

| problem | adaptive arm (mean β_eff) | adaptive median | fixed β at that mean | ratio |
| --- | --- | --- | --- | --- |
| rot_quad | ‖g‖-0.69 (0.87) | 17.65 | 5.714 | 3.1× worse |
| rastrigin | ‖g‖-0.69 (0.64) | 1386 | 1401 | ≈ tie |
| rastrigin | ‖g‖-0.5 (0.37) | 1029 | 1014 | 1.5% worse |
| rastrigin | ‖g‖-0.37 (0.21) | 731.3 | 625.8 | 17% worse |
| rot_quad | m-0.69 (0.85) | 33.34 | 6.524 | 5.1× worse |
| rastrigin | m-0.37 (0.14) | 571.1 | 526.3 | 8.5% worse |
| rastrigin | m-0.5 (0.23) | 745.2 | 691.5 | 7.8% worse |
| rastrigin | m-0.69 (0.37) | 1160.1 | 1013.9 | 14% worse |

The `g₀` denominator is excluded from this table — its mean is degenerate
(~0.99, see point 3), so "fixed β at its mean" is fixed `β ≈ 0.99`, which is
itself far from the optimum and not the comparison the adaptive rule needs to
win.

### Verdict

Rejected on mechanism, not on configuration. Three denominators, eight exact
mean-matched controls, one tracking probe — the distance trigger never adds
anything over `μ` and usually subtracts. The only data-supported conclusion is
that the two benchmark terrains want **opposite constant windows** (rotated
quadratic `β ≈ 0.95`, Rastrigin `μ = 0`), which is exactly what the existing
constant `β` / `μ` knobs already provide.

## Default windows `β = 0.95` / `μ = 0` (superseded by one ≈e-step window)

**Status: superseded — the resting point moved, the two knobs remain.** The
delivered defaults once read `beta=0.95` (secant-magnitude EMA, window ≈ 20
steps) and `mu=0.0` (momentum off, opt-in). They now both read `0.63`
(window ≈ `e` steps ≈ 2.7), so the diagonal's bandwidth and the momentum's are
the same length.

### What it was

`β` smoothed the secant magnitude over ≈ 20 steps; `μ` was opt-in, and only the
momentum arm (`mu=0.9`) turned it on.

### Why it changed

The two EMAs are the same operator (`new = ρ·old + (1−ρ)·x`) applied to the
ratio and to the direction, so the real knob is one window `W`, with
`β = μ = 1 − 1/W`. Asking how many steps a piece of geometry stays credible
gives one physical scale, `W ≈ e`, not 20 for one EMA and 0 for the other. The
default unifies on that scale; `β` and `μ` stay two independent controls
(`μ = 0` is still the off switch, `β = 0.95` is still reachable on its own) —
what changed is only where they rest, not how many of them there are.

## Default momentum `μ = 0.63` (reverted to `μ = 0`)

**Status: superseded — `DiagonalPreconditioner().mu` reads `0.0`; `β` keeps
its `0.63` default.**

### What it was

The default momentum window shared `β`'s: `β = μ = 0.63`, on the argument in
the entry above — that one ≈e-step physical scale answers both EMAs.

### Why the argument did not hold

"The two EMAs ask the same question" was a claim about two windows being of the
same *kind*, used as if it settled what each one is *for*. They buy different
things: `β` smooths the curvature *estimate* the step divides by, a property of
one object inside the method, while `μ` trades overshoot along steep directions
against slow progress along flat ones. That trade is set by the landscape's
scaled spectrum and by the objective's noise, and neither is visible from inside
the solver — so a default window is a prior about a caller's problem, not a
bandwidth the method can pick for itself.

It is also not a repair for the diagonal approximation. Momentum re-shapes the
iteration along the direction `d` already gives; the dropped off-diagonal
curvature stays dropped. "The momentum covers what the diagonal misses" is
false in the direction that matters: the missing term is a directional
correction, not a temporal one.

Two costs came with turning it on by default. It moved the delivered default off
the one branch that keeps a descending direction, and onto a value only a
caller's landscape should pick.

### What replaced it

`μ = 0` (no memory), `β = 0.63` unchanged. The `μ` knob stays, and
`experiments` keeps a `μ = 0.9` arm as the momentum comparison against Adam's
`b1 = 0.9`. Argument and wording: `docs/explanation/how-it-works.md`
§Momentum.

## Additive denominator floor (`+` instead of `max`)

**Status: rejected at the design level — `soar_lite/local_solver.py` reads
`den = max(Y, λ)`, and the additive spelling was never shipped.**

### What it was

A spelling of the gradient-anchored floor that adds instead of capping:

    den = Y + λ

instead of `den = max(Y, λ)`, with `λ = θ·|g_i|` (`"abs"`) or `θ·g_i²`
(`"rms"`).

### Why it was plausible

`Y + λ` is smooth at `Y = λ` and, read charitably, rolls the guard and a
shrinkage into one term.

### Why it fails

The two spellings agree only at the limits `Y → 0` and `Y → ∞`; in between, the
additive spelling always gives a **smaller** `d`:

    d_add = d_max · √(max(Y, λ) / (Y + λ))      ("rms"; the "abs" ratio is unsquared)

so λ deflates the step at *every* `Y`, not only where `Y` bottoms out. The
deflation peaks at `Y = λ`: `d_max/d_add = √((Y + λ)/max(Y, λ)) = √2`, or `2` for
the unsquared `"abs"` ratio, and tends to 1 at both ends. In effect an additive
floor folds λ into the job of tuning the step size, so `Y` stops answering
"is this coordinate curvature-dominated?".

### Verdict

Rejected analytically. The floor's only job is to keep `d` finite at `Y → 0`;
`max` keeps that job apart from step-size tuning — λ caps `d` below and,
everywhere `Y ≥ λ`, the estimate stays the faithful `S/Y`. Step aggressiveness
belongs to the separate knobs `alpha0`, `backtrack`, `n_alpha`, not to the
floor.

## EI continue-or-restart verdict

**Status: deprecated — removed from the code.** The single restart decision
once read `EI_restart >= EI_continue`; the shipped engine now splits it into
the local method's own stopping test (*when*), the `restart_stalled` boolean
(*whether*) and `GlobalModel.propose` (*where*). Expected improvement survives
only as `BLRGlobalModel`'s acquisition function, answering *where to draw*,
never *whether to stop*.

### What it was

One verdict owned the whole restart decision: at each stage it fitted a model
to each side's evidence — the path's own progress stream on the *continue*
side, a pool of group-min draws on the *restart* side — priced both sides as
expected improvement, and redrew the path when `EI_restart >= EI_continue`. A
stage clock and a window gated when the two EIs were compared.

### Why it was plausible

Expected improvement already sits on the surrogate arm, so "does this path
still beat a fresh draw?" looked like one natural acquisition question, and
folding *when*, *whether* and *where* into it meant one knob instead of three.

### Why it fails

The two sides have no common model that reads them symmetrically. Continuing
is priced from a path's own autocorrelated progress; restarting is priced from
a mixture pooled across every path's starts. Neither is the same quantity, so
no single expected-improvement scale compares them without smuggling in a
bias — toward *restart*, because a fresh draw always looks like it might
improve — and the run redraws instead of descending, starving the exploitation
its local steps already paid for.

### Verdict

Removed. The decision that was one statistical verdict is now the method's own
stopping test for *when*, a boolean for *whether*, and `GlobalModel.propose`
for *where*; a stopped path either stays put or gets a new point, with nothing
in between to compare. Expected improvement stays where it still has a
subject: the surrogate arm's acquisition function.

## Projected momentum direction (`m⊥` in place of `m`)

**Status: deprecated — rejected at the mechanism level, never shipped.**
`soar_lite/local_solver.py` still steps along the *unprojected* heavy-ball
momentum, `Δx = -d ⊙ m` with `m = (1−μ)·g + μ·m_prev`.

### What it was

Project the momentum onto the orthogonal complement of the current gradient
before stepping:

    m⊥ = m − ((m·g)/(g·g))·g,        Δx = -d ⊙ m⊥

`m⊥` has `m⊥·g = 0`, so `Δx·g = -(d ⊙ m⊥)·g ≤ 0` whenever `d ≥ 0` — the
descent-direction invariant the plain step has at `μ = 0` would then hold at
every `μ`, and a small enough step would be guaranteed to descend.

### Why it was plausible

With `μ = 0` the step `-d ⊙ g` provably descends (`d ≥ 0` by construction);
with `μ > 0` the step follows `m` and that guarantee is withdrawn. Projection
looked like the way to keep both the smoothing and the invariant.

### Why it fails

1. **The projected part is exactly what momentum is for.** The component of `m`
   along `g` carries heavy-ball's whole value: the zigzag damping on an
   ill-conditioned quadratic, and the build-up across a flat direction. What
   projection keeps is the transverse part, which only rotates the direction.
   It removes the useful component and preserves the cosmetic one.
2. **The invariant it bought was never cashed out.** It promises that a small
   enough step would descend, and this method takes one unruled full step (or an
   `alpha`-grid argmin that compares the grid's samples against each other, never
   against the step's start). Per-step descent was never enforced to begin with;
   convergence rests on the engine's best-so-far bookkeeping and restart.

### Verdict

Not adopted. A caller who wants the guarantee has it already: shrink `alpha0`,
set `n_alpha > 1`, or run a line search outside the seam. The unprojected
direction stays, and the docstring says plainly that it is unprojected and that
the guarantee is withdrawn.

## Zhang-Li / `λ_min` saddle escape

**Status: deprecated — not implemented, and no hook is kept for it.**
The stopping test in `soar_lite/local_solver.py` is a gradient-norm test,
`‖g‖ ≤ τ_rel·‖g_0‖`, and `give_up` ends the path's enumeration.

### What it was

On `give_up`, instead of letting the path stall and the engine restart it in
place, step along the eigenvector of the smallest curvature — Zhang-Li style,
with `λ_min` read off a Hessian approximation — to leave a saddle that the
gradient-norm test cannot tell from a minimum.

### Why it was plausible

A gradient norm cannot distinguish a saddle from a minimum: both have `g → 0`,
so a path resting on a saddle declares itself done, and the run spends a restart
on a point it could have escaped from where it stood. `λ_min < 0` would name the
escape direction exactly.

### Why it fails

The method deliberately carries no curvature to read it off. `d_i = s_i/y_i` is
a scalar ratio per coordinate; the off-diagonal coupling is precisely what it
drops, and near a saddle that coupling *is* the information that names the
negative-curvature direction. Adding `λ_min` means adding a Hessian
approximation, i.e. a second method — not a clause in this one. The engine also
already handles the outcome: a stalled path with `restart_stalled=True` is
re-drawn, which escapes a saddle for the price of one restart.

### Verdict

Left out. A solver that wants saddle escape implements `LocalSolver` with the
curvature it needs; today's diagonal preconditioner has none to spend, and the
stopping test stays a single dimensionless gradient-norm ratio.

## Integer-index `mask` spelling

**Status: deprecated — refused at validation.**
`SphereProjection`'s `mask` takes the boolean selector only.

### What it was

Accept a list of dimension indices (`mask=[0, 2]`) as an alternative to the
boolean vector (`mask=[True, False, True]`), matching numpy's `x[:, idx]` idiom.

### Why it was plausible

Naming three dimensions out of sixty is shorter as indices, and the gather
underneath wants indices anyway.

### Why it fails

The two spellings are indistinguishable at runtime. `bool` is a subclass of
`int`, so `jnp.asarray` names the dtype from the values: a `[1, 0, 1, 0]` meant
as booleans arrives as `int32` and, read as indices, projects a different set —
silently, with a different posterior behind it. Any acceptance rule that tries
to tell the two apart has to guess the caller's intent from the dtype, and
guessing wrong is the failure mode, not the fallback. Refusing the index
spelling removes the ambiguity instead of resolving it; the boolean vector is
also the form the feature gather consumes.

### Verdict

Refused, with the dtype named in the error. A caller holding indices writes
`mask = jnp.zeros(d, dtype=bool).at[idx].set(True)`, one line, no guessing.

## Nesterov acceleration in place of the gradient EMA

**Status: not adopted — recorded so it is not re-proposed as an oversight.**
The direction stays the unprojected gradient EMA `m`; no look-ahead point is
constructed.

### What it was

Replace `m_k = (1−μ)·g_k + μ·m_{k−1}` with Nesterov's accelerated gradient, and
justify it by the `O(1/k²)` rate on convex smooth objectives.

### Why it was plausible

Nesterov improves on plain momentum's method with a rate proof attached, and
Newton reaches the optimum in one step on a quadratic, so "why not the faster
one" is a fair question on a page that already divides a gradient difference by a
displacement.

### Why it fails

`O(1/k²)` is a convex, smooth bound and does not transfer: the target here is
non-convex and medium-to-high dimensional, where the rate says nothing about
which basin a path lands in. The second-order reading is already spent elsewhere
— `d = S/Y` divides the gradient difference `y` by the co-directional
displacement `s`, which is the same use of `y` as curvature information that
Nesterov's proof introduces; Nesterov consumes `y` on its own and hides the
second-order units inside the step length `1/L`. The two are one piece of
information counted differently, not two pieces to stack. And the mechanism does
not fit the seam: Nesterov evaluates `g` at a look-ahead point, while the engine
asks a path for one point, buys its value and gradient in one batch, and commits.
A look-ahead gradient needs a second evaluation between the proposal and the
step's ruling, which the propose-then-evaluate seam has no room for.

### Verdict

Not introduced. `μ`'s job is not to supply second-order information — `d`
already does — it is to reuse the old direction once more where the
per-coordinate estimate is structurally wrong, a mitigation rather than a
repair. The `How it works` page states that trade-off and its measured cost.

## The Ackley problem (`ackley_d100`)

**Status: deprecated — removed from the harness.** `experiments/problems.py` no
longer defines it and `--groups multimodal` returns rastrigin alone. Nothing in
`soar_lite/` ever referenced it.

### What it was

The funnel half of the multimodal panel: Ackley at d=100, box
`[-32.768, 32.768]`, `f* = 0`, run against the soar-lite ladder and Adam. It was
also the third side of the design's dividing line — gradient-free multimodal
relief, where neither method was expected to win.

### Why it was plausible

It is the canonical funnel benchmark: one broad basin, an exponential number of
local minima, and a global minimum a first-order method has to be steered into
rather than descend into. It paired with rastrigin (a lattice) to cover both
shapes of multimodal relief, and it was the only problem in the set where the
gradient itself carries no information — a claim worth drawing rather than only
stating.

### Why it fails

At d=100 the funnel is out of reach and the cosine lattice decides where a path
stops. A uniform start has root-mean-square ≈ 20, so the funnel term
`-20·exp(-0.2·rms)` is worth only ≈ -0.36, and reviving it would take an rms of
a few — all 100 coordinates within a few units of zero, which no single step
takes from a random start. The gradient that does exist steers the second term:
each coordinate is pulled to its nearest integer, where the cosine term
saturates at `-e`, and `f` lands at `20 + e - 0.39 - 2.72 ≈ 19.6`. The
decomposition, checked numerically:
uniform start 21.29 (funnel -0.36, cos -1.07), its rounded integer neighbour
19.61 (funnel -0.39, cos -2.72), the origin 0.00 (funnel -20.0, cos -2.72).

Measured (20 seeds x 3000 batches): every arm spent its whole budget — one
point per batch through batch 2999 — but the best-so-far stopped improving
almost immediately. Last strict improvement, median batch: 63 (max 81) for
`n_paths=1` at `mu` 0, 86 (max 126) at `mu` 0.63, 84 (max 126) for
`n_paths=16`, 329 (max 461) at `mu` 0.9, 343 (max 429) for Adam. Final medians:
Adam 19.57, `n_paths=1` at `mu` 0 / 0.63 / 0.9 → 19.71 / 19.71 / 19.79,
`n_paths=16` 19.55. Two consequences end the panel:

- **The spread is where the plunge landed, not the method.** The arms differ by
  at most ~0.24 at the end, and that is which lattice cell the descent fell
  into; the 0.1 edge `n_paths=16` held over Adam was best-of-16 starting points,
  and it did not survive the v2 re-measurement.
- **Nothing after the plunge reaches the recorded number.** The panel's
  statistic is best-so-far, and it was decided in the first few percent of the
  budget, so the difference between arms is start noise rather than a
  preconditioner or a momentum window at work.

One claim in `RESULTS.md` does not survive this reading: the v3 entry explains
the identical `mu` arms as "no step taken". The traces serve a point on every
batch, so steps were taken — the freeze is in the best-so-far, not in the
engine. The v6 entry carries the correction next to the retirement note.

### Verdict

Retired. Rastrigin carries the multimodal panel alone, and its cos oscillation —
an informative gradient that keeps changing sign — is the shape the momentum
question needs. The readings stay in `RESULTS.md` as the record of a retired
problem; they are not reproducible from this harness. The conclusion Ackley
demonstrated (no first-order method reaches the funnel from a regular start) is
stated in `DESIGN.md` rather than drawn, since a live arm for it spends a full
panel to show coincident flat lines.

If that demonstration is ever wanted back, a low `d` freezes the same way at a
fraction of the cost — the plateau is a property of the box, not of the
dimension.

## NAAdam (noise-augmented Adam)

**Status: deprecated — measured and dropped before any committed arm.**
`compare_baselines.py` has no NAAdam arm; the multimodal panel runs `Adam` with
its default moments.

### What it was

Adam with annealed Gaussian noise added to the iterate, `σ_t = σ0/√t`: jitter the
step so the method leaves the basin it would settle into instead of settling.
Cheap to add — one term in the arm's step — and it would have given the
multimodal panel a *stochastic* first-order reference to set beside a
memory-based method.

### Why it was plausible

The panel's question is whether accumulated direction finds something a plain
step misses. A method that jitters its way out of basins by construction is the
natural control for that: it separates "explores because of noise" from
"explores because of memory".

### Why it fails

Measured on the multimodal pair at d=100, before Ackley was retired and while
restarts were still on: the noise never separated from plain Adam. Larger `σ0`
read flat on rastrigin and worse on ackley — at no setting tried did the noise
produce a curve that plain Adam's own traversal did not already produce. The
mechanism behind that was not diagnosed, and the panel did not need it: a curve
that overlaps its baseline is not a control but a second copy of it, so the arm
would have spent a full panel drawing Adam twice.

Caveat on the record: these probes predate the retirement of Ackley and the
restart-off roster, and no traces were kept, so the readings cannot be replayed
from this harness.

### Verdict

Dropped, and not restored. The multimodal panel compares memory against no
memory — `soar-lite n_paths=1` at `mu` 0 and 0.9 — which is the contrast that was
wanted. That leaves the panel with no *stochastic* first-order reference; the gap
is deliberate and stated, not an oversight.

## Scalar step rescaling (`alpha0` damping, `n_alpha` grid)

**Status: deprecated as a tuning direction — settled against, and its sweep
driver is gone.** `compare_baselines.py` no longer sweeps either knob. The knobs
stay on `DiagonalPreconditioner`: `alpha0` arms every fresh step in
`_reset_step` and defaults to `1.0` (the full step), `n_alpha` stays `1`
(inert), both exposed so the relief is available if a harder landscape ever
shows the symptom.

### What it was

The hypothesis that the diagonal preconditioner's loss on a rotated problem is a
*step-length* problem, so a scalar knob could buy some of it back: damp the full
step (`alpha0 < 1`), or sample a backtracking grid of `alpha`s along the step's
own direction (`n_alpha >= 2` — named for what it counts, since its old name
`ls_rounds` wrongly suggested engine rounds) and let the engine commit the
grid's argmin.

### Why it was plausible

The rotated quadratic is where soar-lite loses by many orders of magnitude. A
step that overshoots along a wrong direction is a real failure mode, and both
knobs are cheap — one scalar, or a few extra objective samples per step. If the
loss were length, either knob would have shown up as a win.

### Why it fails

Both sweeps came back monotone the wrong way. Larger `n_alpha` is strictly worse
on the rotated quadratic (`n_paths=1`, `backtrack=0.5`: `1285 → 4220 → 7382 →
12535`); damping is monotone worse too (`alpha0` 1 → 0.5 → 0.25 → 0.125:
`1285 → 1976 → 3211 → 5356`), and `alpha0 > 1` is catastrophic (`8e5`). The full
step is already the right length. The reason is structural: the residual error is
in the **direction** `-d ⊙ g` — the diagonal secant ignores off-diagonal
curvature — and no scalar rescaling of a step fixes its direction. That loss is a
**capability boundary, not a tuning failure**.

### Scope

One landscape — the rotated and diagonal quadratics — so the measurement covers
**rotation** coupling only; the diagonal-error question is left open. `RESULTS.md`
under `pre-2026-10` states both limits in full, next to the readings.

### Verdict

Settled against, on architectural grounds: direction, not length. **Provenance:**
the sweeps ran on the engine as it stood before the 0.1.0 refactor and are not
reproducible — their `alpha0` sweep driver no longer exists. `RESULTS.md` under
`pre-2026-10` holds the readings as the evidence behind this verdict, not as
current benchmarks; a stale number does not retract an architectural conclusion.
The defaults are not a claim that the knobs are useless elsewhere — they are a
contract on the benchmark, which `README.md`'s reproducibility contract states in
full.

## Public knob renames (pre-0.1.0)

**Status: superseded names — dead as public keywords.** The current spellings
are the contract; no retired name is accepted as a Python keyword. Archived
traces are the one place an old spelling still lives: a record written under
`max_steps` is read by that key (`experiments/harness/record.py`) so it keeps
stating the budget it measured, and the next save writes the new name.

### What it was

An earlier keyword vocabulary, renamed in place during 0.1.0's build-up:

- the horizon: `max_steps` → `max_evals` → `max_iters`;
- the probe: `n_probe_rounds` → `n_probe_evals` → `n_draws`, then the probe
  itself was removed (no `n_draws`, no `per_round_evals`);
- the cost report: a two-field pair became four — `nfev` → the two seams × two
  rulers of `CONTEXT.md` §Evaluation.

### Why the names were retired

Each old name let one word do double duty: "round" named both a probe round and
an engine round, "step" both a solver step and a loop iteration, "eval" both one
objective point and one served batch. One concept, one name — on the axis it
actually counts on (iterations, points, batches, seams) — is what replaced the
lineage above; the retired spellings survive only in the `_Avoid_:` lines of
`CONTEXT.md`'s glossary, as names not to reintroduce.

## Engine-committed alpha grid (`n_alpha`, `backtrack`)

**Status: removed — `n_alpha`, `backtrack` and `alpha_sample` no longer exist.**
`DiagonalPreconditioner` reads six knobs (`moment`, `beta`, `mu`, `theta`,
`alpha0`, `tau_rel`) and its step is one full proposal at `alpha0`; nothing
samples an alpha grid. The entries above still name `backtrack` and `n_alpha`
among the separate aggressiveness knobs beside `alpha0`; that described the code
as it stood then, and `alpha0` is now the only step-aggressiveness knob left.

### What it was

Two keyword arguments and one state field:

- `n_alpha: int = 1` — alpha samples per step; `1` meant no line search;
- `backtrack: float = 0.5` — the grid's spacing (`alpha *= backtrack` per
  retry), one constant for both phases;
- `alpha_sample: int` on the per-path record — how many alphas of the current
  step had already been served, re-armed after every commit and every restart,
  so each `verdict` call knew whether the grid had a sample left to try
  (`"retry"`, `alpha *= backtrack`) or the step was done (`"accept"`).

With `n_alpha >= 2` a step served the grid `{alpha0, alpha0·backtrack,
alpha0·backtrack², …}` along its own direction, and the **engine** committed the
grid's `argmin` in `f`. At the default `n_alpha = 1` the grid collapsed to the
one full step, which is what shipped.

### Why it was removed, and what took its place

Two changes, in that order. The measurement in *Scalar step rescaling* above had
already closed the length question — larger grids came back monotone worse, and
the reason is structural: the error is in the step's direction, not its length.
Then the step channel was redesigned so the **solver** names its own winner
instead of the engine committing an `argmin` over whatever the method served.
With the argmin gone from the engine, "how many alphas the engine will consider"
had no owner left: it described a decision the engine no longer makes.

The capability is not lost, only relocated. A method that wants a grid now runs
it inside its own `propose`, serving candidates through `serve_trials` and
naming the one it wants; the engine clips the start, buys the winner's owed
halves in one fused batch (`settle`), and folds the winner back.
`DiagonalPreconditioner` takes the single full step the sweeps said was right,
and reads no trial at all.
