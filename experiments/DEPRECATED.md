# Deprecated experiments

The record of experiments this harness no longer runs. Two kinds end up here: a
problem retired from the benchmark set, and a probe sweep run under a knob set
the code no longer carries. Both are cited for the conclusion they reached, and
neither can be re-measured by the current script.

The library's retired *design* directions are a different record, in
`../DEPRECATED.md`: that file says why a knob or a rule was dropped, this one
says what was measured before it was. A direction that was tried and abandoned
belongs there; a run whose numbers can no longer be produced belongs here.

Same rules as that file. Entries are records: nothing is deleted, and nothing is
rewritten to assert something else. A new measurement is a new entry, and the
old numbers rest as measured. Both entries below were moved here whole from the
library's record, with relative paths adjusted and no other text touched.

## The Ackley problem (`ackley_d100`)

**Status: deprecated — removed from the harness.** `harness/problems.py` no
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

## pre-2026-10 — step-length calibration sweeps — NOT REPRODUCIBLE

Budget: mixed and ad hoc (the grid sweep was 300 batches, 3 seeds; the `alpha0`
damping sweep is not recorded at that detail). **The code path is gone**: the
`alpha0` sweep driver no longer exists in `compare_baselines.py`, so these
numbers cannot be re-measured by the current script, only cited.

These numbers were measured on the engine as it stood before the 0.1.0
refactor. They are the evidence behind an architectural verdict — the residual
error is in the *direction* `-d ⊙ g`, not its length — so a stale number does
not retract the verdict; treat any figure here as a floor on effort, not as a
reproducible result. The abandoned direction itself — why it was tried, and why
it was dropped — is recorded in `../DEPRECATED.md`.

- **`n_alpha` grid** (300 batches, 3 seeds, rotated `cond=1e6` d=100):
  `n_alpha` larger is strictly worse — `n_paths=1` at `backtrack=0.5`:
  `1285 → 4220 → 7382 → 12535`. `backtrack` then also mattered at `n_alpha=1`
  (`1285 → 1056` at 0.1) because it shrank the **warm-up Armijo** backtrack — a
  different first step (path shift), not a grid effect. That sweep predates the
  removal of the warm-up guard: with neither a guard nor a grid at `n_alpha=1`,
  `backtrack` is now inert there and only sets the grid's spacing.
- **`alpha0` damping** (the "small step along the wrong direction might pay"
  hypothesis, `alpha0 ∈ {0.125, 0.25, 0.5, 1.0, 2.0}`, `n_paths ∈ {1,16}`,
  rot+diag): damping is strictly worse on the rotated case
  (`1285 → 1976 → 3211 → 5356`) and `alpha0 > 1` is catastrophic (`8e5`); on
  diag, `1.0` is the one-shot win (`3.5e-65` in 20 points) and any damping only
  slows it.

Scope: these sweeps ran on **one landscape** (the rotated and diagonal
quadratics), which isolates the off-diagonal direction error. Measured for
rotation coupling only; that the headline's interaction coupling is equally
immune is an analogy from this, not a separate measurement. Silent on the
diagonal part of the error, where the per-coordinate secant is itself a biased
curvature estimate — unmeasured, and deliberately left open in `../DEPRECATED.md`.
