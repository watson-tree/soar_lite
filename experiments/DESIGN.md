# Experiment design: `compare_baselines`

The canonical reproducible comparison artifact. One script, four problem
groups (`headline`, `smooth`, `multimodal`, `blr`) plus two appendix panels
(unit-rescaling invariance, Adam's lr sweep), three propositions. This
document is the lossless handoff: read it before touching
`compare_baselines.py` so no settled detail is re-derived from memory.

This document states no measured numbers. Every figure and table an experiment
produced lives in `RESULTS.md`, tagged with the lineup and budget that produced
it, so an old reading cannot age in place here; what follows is the design, the
budget contract and the conclusions those readings support.

## Goal

Prove three propositions by **drawing their win/lose boundaries** (never only
reporting wins — R5):

- **P② — direction repair (smooth + headline).** The per-coordinate secant's
  error is in the *direction* `-d ⊙ g`: it drops off-diagonal curvature. Two
  questions: does the preconditioner still beat Adam's adaptive step on scale /
  conditioning / unit-mixing, and does direction momentum (the default `mu`, an
  `≈e`-step gradient window) repair the dropped part — a residual folded back
  from the search history — where the rotation coupling (rotated quadratic) and
  interaction coupling (oscillator) live? `L-BFGS-B` is the ceiling reference
  (full-memory + exact line search),
  so the panel shows how much of any gap is "diagonal vs full second order"
  rather than a tuning failure.
- **P① — momentum drift / stall shape (multimodal).** Reframes the old "escape
  the saddle" question without answering it: these panels do not test whether
  `mu` escapes a saddle — that needs a landscape whose discriminator is
  saddle-vs-minimum, and escaping is just what momentum terms are usually
  hoped to buy in deep-network optimizers. What the panels do test is whether
  the momentum push changes *where a stalled chain ends up*. Rastrigin chains
  converge into a lattice basin and stall; Ackley chains freeze outside the
  funnel. The momentum-off arm (`mu=0`) is the contrast that reads whether the
  default momentum moves either stall, against serial, parallel and Adam.
- **Restart model — centre bias (rastrigin alone).** Where a restart lands: a
  BLR posterior centred on the observed best vs a uniform re-draw. Only
  rastrigin gives a restart a stage — chains stall in local basins, and the
  sample's re-draws provably never improve the best. No external baseline: a
  BLR restart serves no objective points, so on the points/batches axes its
  cost is invisible and a shared plot would sell a free lunch. Its figure is
  separate, all arms `n_paths=16`.

Design rule (from the start): problems must be **high-dimensional, d≈50–200** —
in low dimension the sampled paths entangle: with few coordinates to spread
them apart, several starts converge into the same basin, so the width `n_paths`
buys little and soar's multi-path advantage is dulled there, not sharpened.

Every arm leaves `restart_stalled` at the engine default (**on**). A probe
showed stalls fire only where a method converges into a basin — the diagonal
quadratic (~step 4) and rastrigin (~batch 700) — never on the rotated
quadratic, Ackley or the oscillator, so the restart bit is not a step any
panel takes. The two single-attribute steps are the width (`n_paths` 1 → 16)
and the momentum (`mu` 0 → the default); each curve moves exactly one.

## Problem set

Four run groups plus two appendix panels, one role each:

1. **Headline — broadband device design (borrowed recipe).** A N-stage
   coupled-oscillator / vibration-isolation network. Per-stage parameters carry
   heterogeneous physical units: mass `m_k` [kg], stiffness `k_k` [N/m],
   damping `c_k` [N·s/m]. N=20 → d≈60.
   Loss = `Σ_j log₁₀(|H(ω_j)| / |H_target(ω_j)|)²` over a log-spaced grid, with
   `H` the (banded, linear) frequency response. Synthetic target design → known
   `f* = 0`; boxes are physical ranges. Serves **P② + unit-invariance**.
   Landscape: non-convex via resonance/interference — *not* a funnel (see Risk).
2. **Smooth — the preconditioner's home and its boundary.**
   - **Diagonal control**: axis-aligned ill-conditioned quadratic `diag(10^i)`,
     d=100. Diagonal `H`, so soar-lite is Newton-exact in ~1 step (existing
     test already proves it). Checksum that the mechanism works in the ideal
     regime.
   - **Rotated honesty boundary**: random-rotated quadratic, d=100, condition
     ~1e6. Dense `H` breaks the diagonal assumption; shows the *cost* of the
     diagonal approximation. This is where direction momentum is asked whether
     it can buy any of the dropped coupling back.
   Serves **P②**.
3. **Multimodal — Ackley / Rastrigin**, d=100 (drop to 50 if cost bites). A
   funnel / lattice of local minima. Serves **P①**.
4. **BLR — Rastrigin alone.** The one problem where a restart has a stage:
   chains converge into local basins and stall, and Sample's re-draws provably
   never improve the best. Serves the **restart-model** proposition.

Two appendix panels, property demos rather than races:

5. **Invariance — unit-rescaling consistency.** Run on the diag control:
   re-measure in two unit systems (here `x' = 10x`, see `INVARIANCE_SCALE` in
   `compare_baselines.py`), **zero retune**. Claim is **outcome-level, not
   trajectory-level**: the best-so-far-vs-cost curves overlap in both unit
   systems, while Adam's separate (one system needs a different `α`). Exact
   per-step covariance (`x̃=c·x ⇒ d̃=c²d, Δx̃=c·Δx`) is real but holds only for
   the secant-based steady state: the **warmup step is by design not
   coordinate-invariant** (a scalar `α` applied uniformly across mixed-unit
   coords), so the first secant pair — and anything seeded by it — is not the
   transformed pair (`ALGORITHM.md` §1.4/§7). On the diagonal quadratic this
   self-corrects in one step (`d` reads `1/Hᵢ` exactly, independent of the seed
   pair's scale), so the diag control gives clean overlap; a general/multimodal
   problem would **not** guarantee same basin, so never use one for this panel.
   The exact claim is asserted at the step level (injected secant pair, as in
   `test_diag_step_scales_with_the_coordinate_unit`); the end-to-end claim is
   asserted statistically. A *property demo*, not a race.
6. **Adam lr sweep.** Log-scan of `lr` per problem, marking the panel pick
   (`ADAM_LR`) — the evidence that soar-lite has no learning rate where Adam
   pays a grid. Emitted by `--lr-sweep`.

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

Curves differ per panel; each curve has one role, and each single-attribute
step moves exactly one knob (`n_paths` or `mu`); restart stays at the engine
default (on) in every arm:

- **Smooth + headline** (**P②**): `soar-lite n_paths=1` (default `mu`),
  `soar-lite n_paths=16`, `soar-lite n_paths=1, mu=0` (momentum off), `Adam`
  (grid lr), `L-BFGS-B` (ceiling *reference*, not a head-to-head baseline —
  full-memory + serial line search vs our per-coordinate + batched n_paths).
- **Multimodal** (**P①**): the same three soar-lite arms, `Adam` (grid lr).
  The momentum-off arm is the P① contrast: with `mu=0.63` as the default, the
  ladder is off / on rather than a third hand-set strength.
- **BLR** (**restart model**, rastrigin alone, its own figure, all
  `n_paths=16`): `Sample draws`, `BLR identity`, `BLR sphere` — one method,
  three restart models, so nothing but where a restart lands differs.
- **Invariance**: `soar-lite` (n_paths=1) vs `Adam`, two unit systems, overlaid
  after the coordinate transform.

The width step (`n_paths` 1 → 16) and the momentum step (`mu` 0 → default) are
the two single-attribute knobs; a single `n_paths=16` curve that also flipped
restart would charge the whole gap to "parallelism". Restart is not a step the
panels take: a probe showed stalls fire only where a method has already settled
into a basin (the diagonal quadratic, rastrigin), never on the rotated
quadratic, Ackley or the oscillator.

The headline BLR panel is **cancelled**: a BLR restart needs a stalled stage to
act on, and the oscillator's non-convexity is resonance-type — not the funnel
that stalls a chain. The BLR proposition lives on rastrigin, where Sample's
re-draws provably never improve the best and a fitted restart is the question.

L-BFGS-B appears only on the P② panels. Its role there is the *ceiling
reference* — how close soar-lite's diagonal secant gets to a full-memory
second-order method on smooth, convex landscape. That role does not exist on
the P① panel: the question there is whether direction momentum changes a
stalled chain's resting place, not whether a full second-order method stalls.
L-BFGS-B is second-order and restart-free — on Ackley/Rastrigin it would dive
deterministically into the nearest local minimum and stay there at
second-order precision, so its curve could only show "a second-order method
also gets stuck", which discriminates nothing about drift or stall. It is also
serial (not the batched cost axis; see Budget) and its line search is fragile
on Rastrigin's cos oscillation.

NAAdam (Adam + `σ_t = σ0/√t` decaying Gaussian noise) was measured and
**dropped**: on ackley/rastrigin at d=100 the noise never separates from plain
Adam — larger `σ0` is flat (rastrigin) to worse (ackley) — so it would only be
a redundant, exactly-overlapping curve. The multimodal panel is therefore
`soar-lite` n_paths=1 (default `mu`) / 16 / (1, `mu=0`) vs `Adam`, which is the
honest contrast: momentum is read as a *drift* on these landscapes, not
asserted to be an escape mechanism in general.

Adam learning rate is **per-problem grid-tuned**: the picks span over two
orders of magnitude (diag and Ackley 1e-2 → rotated quadratic and oscillator
2.0), and on diag `lr=1e-4` costs 52 orders of magnitude against the pick
(`6.6e3` against `8.8e-49`). On diag and Ackley every grid point from 1e-2 up
sits on the same floor or plateau, so those two picks are one point of that tie
rather than a strict argmin — the figure's diamond marks the pick, not a
minimum. The grid and picks live in `ADAM_LR` / `--lr-sweep` in
`compare_baselines.py`, which prints the table it draws; soar-lite has no
learning rate. Tuning is itself budget — every trial is more objective points,
and it is manual, not adaptive.

### Step-length calibration (settled: the knob is real, the win is not)

**Provenance: the sweeps behind this verdict ran on the engine as it stood
before the 0.1.0 refactor, and they are not reproducible any more — their
`alpha0` sweep driver is gone.** They are recorded in `RESULTS.md` under
`pre-2026-10`: evidence behind the verdict, not current benchmarks. The verdict
is architectural (direction, not length), so a stale number does not retract
it.

The verdict, measured on one landscape before any library default moved: **on
the rotated quadratic the full step is already the right length, and no scalar
step rescaling helps.**

- `n_alpha` (renamed from `ls_rounds` — a count, and "rounds" wrongly
  suggested engine rounds) is the `alpha`-grid size, used by both phases: at
  `n_alpha >= 2` it samples `alpha0 · backtrack^j` along the step's own
  direction (`-g` in the warm-up, `-d ⊙ g` after) and the engine's per-step
  argmin commits the winner. `n_alpha=1` is inert (default): with neither a
  guard nor a grid there, `backtrack` only sets the spacing of a grid that is
  never sampled.
- Both sweeps — the `n_alpha` grid and the `alpha0` damping — are in
  `RESULTS.md`.

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

Scope, to avoid over-claiming: the sweeps in `RESULTS.md` ran on **one
landscape** (the rotated and diagonal quadratics). That landscape isolates the
*off-diagonal, direction* error, to which any scalar step knob is structurally
immune —
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
  batch = one pass of parallel evaluations; a single chain spends one point per
  batch). The engine charges every batch it serves against that budget, redraws
  included, so the ceiling is exact — a restart draws fresh starts inside the
  same budget, never beyond it. Runs stop earlier on their own criterion (the
  engine's
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
- Seeds `N=20`; report **median, shaded [Q1, Q3]** (interquartile band), log-y,
  log x — mean ± std would model a skewed best-so-far spread as symmetric and
  dip below the log axis.
- Outputs live under `experiments/results/`, one file per **(problem, arm)**:
  `results/<problem>/<arm-key>.json` holds that arm's raw per-seed trace (points,
  batches, best-so-far, final value), plus `results/<group>/{batches,points}.png`
  assembled from it, `results/invariance/invariance.png` and
  `results/lr_sweep/lr_sweep.png`. A panel is a roster over arm files, not a
  container that owns them, so re-measuring one arm rewrites one file and every
  panel listing it picks the new numbers up — rastrigin's `n_paths=16` arm is a
  single file that both the multimodal and the blr panel draw. The default run is
  `headline,smooth,multimodal,invariance`; `--groups blr` adds the restart-model
  panel (a single rastrigin subplot). `--plot-only` rebuilds figures from the
  arm files with no re-run, naming any arm that has no trace instead of drawing a
  gap. The invariance panel writes no trace — it is re-run (fast) rather than
  re-plotted.
- The arm key spells out the arm's whole configuration
  (`soar-sample-np1-mu0.63-restart-on`, `adam-lr0.01`, `lbfgs`) instead of
  leaning on a default. An engine default moving therefore changes the key and
  so the path: the old trace survives as a record of the old configuration, and
  a rerun cannot overwrite a measurement whose meaning has changed. The arms
  that take no knobs read `DiagonalPreconditioner`'s defaults off the class for
  the same reason.
- Re-measure one arm, leaving the rest of the panel untouched:
  `--problems <name> --arms <arm-key>` selects on the same identity the file
  path uses.

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
2. `experiments/problems.py` — the four groups (`headline`, `smooth`,
   `multimodal`, `blr`) + the invariance and `all` builders, each problem
   exposing the soar-lite seam `x_batch → (loss, grad, feas)`.
3. `experiments/methods.py` — drivers: soar-lite n_paths=1 / 16 / (1, mu=0),
   Adam, L-BFGS-B, and the three BLR arms (`surrogate_minimize` + the identity /
   sphere projections).
4. `experiments/compare_baselines.py` — harness over `N=20` seeds, one raw trace
   file per (problem, arm) under `results/`; panels: headline / smooth /
   multimodal / blr, plus invariance and the lr sweep.
5. Plot from the arm files. **Self-check first**: two curves on the diag
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

The funnel slot has a known blind spot, and it is deliberate: no
saddle/vanishing-gradient handling is built in. A Zhang-Li power iteration
triggered on a small gradient norm was considered and rejected — multi-path
sampling plus restart already cover that regime, so the extra complexity would
buy nothing. Ackley is where that shows; `RESULTS.md` holds the numbers.

## Readings live elsewhere

Every measured number this design has produced — the current lineup's medians,
the previous restart on/off lineup's Ackley and unit-rescaling readings, and the
step-length sweeps whose code path no longer exists — lives in `RESULTS.md`,
one entry per lineup, newest first. None of them are restated here, so none of
them can age in place.
