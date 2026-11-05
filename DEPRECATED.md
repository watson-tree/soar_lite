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