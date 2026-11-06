# Deprecated directions

Archived design directions that were explored and then abandoned or
superseded. Each entry records what was tried, why it failed, and the
evidence, so a future reader does not re-run the same line. Nothing in this
file is shipped: probe scripts cited here live in `.scratch/` and are not
committed, every number that matters is reproduced inline.

Three rules, the same discipline as `experiments/RESULTS.md`:

- **Entries are records: never re-edit them.** A `Status` line states the
  delivered code at the time of writing; when the code later changes, that
  historical claim stays exactly as written and a new entry is appended below.
  Rewriting an old `Status` erases the before/after the file exists to preserve.
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
