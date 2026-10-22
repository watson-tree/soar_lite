# Experiment design: `compare_baselines`

The canonical reproducible comparison artifact. One script, five problem
groups, two falsifiable propositions. This document is the lossless handoff:
read it before touching `compare_baselines.py` so no settled detail is
re-derived from memory.

## Goal

Prove two propositions by **drawing their win/lose boundaries** (never only
reporting wins — R5):

- **P② — local method.** Per-coordinate secant preconditioning (`soar-lite`,
  unit-aware, no scalar learning rate) vs Adam's adaptive single-`α` step.
  Tested on scale / conditioning / unit-mixing.
- **P① — restart.** Principled EI restart (payoff *measured* on sampled-starts
  group-min) vs brute-force decaying-noise escape (NAAdam). Tested on
  multimodality.

Design rule (from the start): problems must be **high-dimensional, d≈50–200** —
low dimension favors SOAR-style surrogate restart methods, which is someone
else's home turf.

## Problem set (plan A: "borrow the recipe")

Five groups, one role each:

1. **Headline — broadband device design (borrowed recipe).** A N-stage
   coupled-oscillator / vibration-isolation network. Per-stage parameters carry
   heterogeneous physical units: mass `m_k` [kg], stiffness `k_k` [N/m],
   damping `c_k` [N·s/m]. N=20 → d≈60.
   Loss = `Σ_j log₁₀(|H(ω_j)| / |H_target(ω_j)|)²` over a log-spaced grid, with
   `H` the (banded, linear) frequency response. Synthetic target design → known
   `f* = 0`; boxes are physical ranges. Serves **P② + unit-invariance**.
   Landscape: non-convex via resonance/interference — *not* a funnel (see Risk).
2. **Positive control / ceiling — axis-aligned ill-conditioned quadratic**
   `diag(10^i)`, d=100. Diagonal `H`, so soar-lite is Newton-exact in ~1 step
   (existing test already proves it). Checksum that the mechanism works in the
   ideal regime. Serves **P②**.
3. **Honest boundary — random-rotated quadratic**, d=100, condition ~1e6. Dense
   `H` breaks the diagonal assumption; shows the *cost* of diagonal approx, read
   against the L-BFGS reference. Serves **P② boundary** — the case we expect to
   lose on some axis, reported anyway.
4. **Multimodal — Ackley / Rastrigin**, d=100 (drop to 50 if cost bites). A
   lattice/funnel of local minima. Serves **P①**. NAAdam appears **only** here.
5. **Invariance panel — unit-rescaling consistency.** Run on the diag control
   (or the headline in its single-basin regime): re-measure in two unit
   systems (here `x' = 10x`, see `INVARIANCE_SCALE` in `compare_baselines.py`),
   **zero retune**. Claim is **outcome-level, not trajectory-level**:
   the best-so-far vs nfev curves overlap in both unit systems, while Adam's
   separate (one system needs a different `α`). Exact per-step covariance
   (`x̃=c·x ⇒ d̃=c²d, Δx̃=c·Δx`) is real but holds only for the secant-based
   steady state: the **Armijo warmup is by design not coordinate-invariant** (a
   scalar `α` applied uniformly across mixed-unit coords), so the first secant
   pair — and anything seeded by it — is not the transformed pair
   (`ALGORITHM.md` §1.4/§7). On the diagonal quadratic this self-corrects in
   one step (`d` reads `1/Hᵢ` exactly, independent of the seed pair's scale),
   so the diag control gives clean overlap; a general/multimodal problem would
   **not** guarantee same basin, so never use one for this panel. The exact
   claim is asserted at the step level (injected secant pair, as in
   `test_diag_step_scales_with_the_coordinate_unit`); the end-to-end claim is
   asserted statistically. A *property demo*, not a race.

## dfbench (Learn2Design-2026) relationship

- **Not in the main figures.** Independent-repo decision: dfbench is final
  real-world validation only.
- What we borrowed is the **recipe** (from its `Problems.md`), three separable
  ingredients: (1) heterogeneous-unit physical parameters; (2) log-scale
  frequency-response mismatch loss over a log-spaced grid; (3) physical
  constraints as squashed-ReLU penalties with a separate feas flag. Plus a
  synthetic reference target so `f*` stays known.
- If the optional final panel is added later: `dfbench` + `VoyagerProblem(
  n_frequencies=20)` as a dependency-group opt-in; wrap its `value_and_grad` +
  `is_feasible` into our `(f, ∇f, feas)` oracle seam. `dataset.h5` (72 MB) is
  only a sampling prior and is **not** needed.

## Methods × panels

Four curves per panel:

- **Panel "preconditioning"** (groups 2+3, **P②**): `soar-lite n_paths=1`, `Adam`,
  `L-BFGS-B` (ceiling *reference*, not a head-to-head baseline — full-memory +
  serial line search vs our per-coordinate + batched n_paths), plus `soar-lite n_paths=16`
  as a supplementary "what parallelism buys" curve.
- **Panel "escape"** (group 4, **P①**): `soar-lite n_paths=16`, `soar-lite n_paths=1`,
  `Adam`, `NAAdam`.
- **Invariance panel** (group 5): `soar-lite` (n_paths=1) vs `Adam`, two unit
  systems, overlaid after the coordinate transform.

NAAdam definition: Adam + `σ_t = σ0/√t` decaying Gaussian noise, `σ0=1e-2`
(tunable). A `σ0` sweep on ackley/rastrigin at d=100 shows the noise never
escapes — larger `σ0` is flat (rastrigin) to worse (ackley) — so the honest
escape baseline is effectively Adam; this *is* evidence for P①. NAAdam keeps
`σ0=1e-2` untuned. Gate: NAAdam runs **only** on non-random multimodal
objectives (R5 feedback — Adam beat NAAdam on the hidden topologies in both
mean and variance).

Adam/NAAdam learning rate is **per-problem grid-tuned**: the optimal lr spans
over two orders of magnitude (diag 1e-2 → rotated 2.0), and a wrong lr costs
~68 orders of magnitude on diag. The grid and picks live in `ADAM_LR` /
`--lr-sweep` in `compare_baselines.py`; soar-lite has no learning rate. Tuning
is itself budget — every trial is more objective points, and it is manual, not
adaptive.

### Step-length calibration (done: the knob is real, the win is not)

The verdict, measured on one landscape before any library default moved: **on
the rotated quadratic the full step is already the right length, and no scalar
step rescaling helps.**

- `n_alpha` (renamed from `ls_rounds` — a count, and "rounds" wrongly
  suggested engine rounds) is the post-warm-up `alpha`-grid size: at
  `n_alpha >= 2` it samples `alpha0 · backtrack^j` along `-d ⊙ g` and the
  engine's per-step argmin commits the winner. `n_alpha=1` is inert (default).
- **Grid sweep** (300 batches, 3 seeds, rotated `cond=1e6` d=100): `n_alpha`
  larger is strictly worse — `n_paths=1` at `backtrack=0.5`:
  `1285 → 4220 → 7382 → 12535`. `backtrack` only matters at `n_alpha=1`
  (`1285 → 1056` at 0.1), which is the warm-up Armijo start — a different
  first step (path shift), not a grid effect.
- **`alpha0` damping** (the "small step along the wrong direction might pay"
  hypothesis, `alpha0 ∈ {0.125, 0.25, 0.5, 1.0, 2.0}`, `n_paths ∈ {1,16}`,
  rot+diag): damping is strictly worse on the rotated case
  (`1285 → 1976 → 3211 → 5356`) and `alpha0 > 1` is catastrophic (`8e5`);
  on diag, `1.0` is the one-shot win (`3.5e-65` in 20 points) and any
  damping only slows it.

Why, once seen: the residual error is in the **direction** `-d ⊙ g` (the
diagonal secant ignores off-diagonal curvature), not in its length — no scalar
`alpha` grid or damping can fix a wrong direction. This is the diagonal
preconditioner's standing limitation, and the rotated quadratic is where it is
meant to lose: that loss is a **capability boundary, not a tuning failure**.

Off-diagonal coupling has two sources, and the headline chain has the second:
*rotation coupling* needs same-unit components of one vector (spatial velocity
or acceleration components, a preferred direction mixing them) — heterogeneous
units cannot rotate into each other, which is why the rotated quadratic is a
pure artificial stress test; *interaction coupling* needs no unit match — the
loss depends jointly on two parameters (a resonance tunes with mass and
stiffness together, `omega^2 ~ k/m`, and the stiffness profile couples within
the chain), and the oscillator's `kg`/`N/m`/`N*s/m` coordinates carry exactly
that, stated in its own spec ("approximately right, not exact"). So the
headline panel is *not* exempt from the direction error: whether soar-lite
still wins there while dropping real coupling is what that panel measures, not
something asserted up front.

Scope, to avoid over-claiming: the sweeps above ran on **one landscape** (the
rotated and diagonal quadratics). That landscape isolates the *off-diagonal,
direction* error, to which any scalar step knob is structurally immune —
measured for rotation coupling only; that the headline's interaction coupling
is equally immune is an **analogy from this, not a separate measurement**. It
is silent on a different failure — the **diagonal part of the error**, where the
per-coordinate secant is itself a biased curvature estimate under non-quadratic
terms or curvature that changes with position (complexity the two quadratics do
not have). Whether `alpha0` damping or a finer `n_alpha` grid relieves *that*
is **unmeasured**, and deliberately left open: `alpha0` names the one
hard-coded `=1` in `_reset_step`, `n_alpha` stays as `1`-inert, both exposed so
the relief is available if a harder landscape ever shows the symptom. The
defaults are not a claim that the knobs are useless elsewhere — they are a
contract on the *benchmark*: the formal runs contribute zero knobs (Adam pays a
per-problem lr grid each time), so "no tuning" is a fact about the benchmark,
not a universal theorem about the method.

## Budget, statistics, outputs

- Every method gets the same number of **batches**: `--max-steps 3000` (one
  batch = one round of parallel evaluations; a single chain spends one point per
  round). The engine charges every batch it serves against that budget, probes
  included, so the ceiling is exact — a run that restarts buys fewer rounds, not
  more work. Runs stop earlier on their own criterion (the engine's
  gradient-norm test, L-BFGS-B's tolerance), so 3000 is an upper bound, not a
  fixed cost. `L-BFGS-B` takes the same cap as `maxfun`/`maxiter` and is a serial
  reference, not the same cost axis.
- Two honest budgets, both read off the **same** runs, no re-runs:
  - **batches** figure — every method 3000 batches. This is the wall-clock axis
    of a batched setting, and the only axis where `n_paths=16` and a single chain
    compare at equal cost per step; it is the main chart.
  - **points** figure — each run truncated at 3000 objective points, the fair
    information unit of a serial setting. A chain spends exactly one point per
    batch, so `n_paths=16` shows only its first ~187 batches there.
- Seeds `N=20`; report **mean ± std** (shaded), log-y, log x.
- Outputs: `experiments/results_{group}_{batches,points}.png` +
  `results_{group}.json` per group, plus `results_invariance.png` and
  `results_lr_sweep.png`. The JSON holds the best-so-far mean/std arrays per
  problem×method for both axes, keyed by `max_steps`.

## Reproducibility contract

- Fixed seed(s); pinned versions in one install cell: `soar-lite==0.1.0`
  (git source pre-release → PyPI after publication), `optax==…`, `jax==…`,
  `scipy==…`, `matplotlib==…`.
- One canonical script in-repo; Colab is a runtime only (never a second
  codebase). Badge + `.ipynb` wrapper happen at open-source time, when the
  install line switches git → PyPI.
- CPU is sufficient.

## Build order

0. **Wait for explicit go-ahead before heavy coding.**
1. Add experiment deps via a dependency group (`optax`, `matplotlib`).
2. `experiments/problems.py` — the five groups + a thin oracle wrapper matching
   the soar-lite seam `x_batch → (loss, grad, feas)`.
3. `experiments/methods.py` — drivers: soar-lite n_paths=1/n_paths=16, Adam, NAAdam,
   L-BFGS-B.
4. `experiments/compare_baselines.py` — harness over `N=20` seeds, collects
   per-round best-so-far.
5. Plot + dump `results.json`. **Self-check first**: two curves on the diag
   quadratic (soar-lite n_paths=1 vs Adam) before running everything.
6. `experiments/README.md` — run instructions, written alongside the code.

## Open TODOs (release hygiene, outside this experiment)

- Fill LICENSE copyright holder; pyproject `authors` + `[project.urls]`.
- Switch notebook install source git → PyPI; add Open-in-Colab badge after
  publication.

## Risk to remember

The headline problem's "multimodality" is resonance/interference-type
non-convexity, **not** the funnel type that Ackley/Rastrigin provide. The two
complement each other; the headline must not be counted as covering the funnel
slot. Watch whether the diagonal approximation's "approximately right, not
exact" banded coupling actually lands the discriminator on soar-lite's side — a
two-curve self-check decides before the full run.