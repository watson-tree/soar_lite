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
   systems, **zero retune**. Claim is **outcome-level, not trajectory-level**:
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

## Budget, statistics, outputs

- `soar-lite` / `Adam` / `NAAdam`: `max_steps=300` rounds. `L-BFGS-B`: scipy
  default stopping, `maxiter` cap ~500 (it is a serial reference, not the same
  cost axis).
- Seeds `N=20`; report **mean ± std** (shaded), log-y, per-round x-axis as the
  main chart; per-point as secondary.
- Outputs: `experiments/results.png` + `experiments/results.json`
  (per-round best-so-far mean/std arrays, one block per problem×method).

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