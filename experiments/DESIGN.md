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
  conditioning / unit-mixing, and does adding direction momentum change the
  *response* of that dropped part — the zigzag and the overshoot a wide residual
  spectrum produces — where the rotation coupling (rotated quadratic) and
  interaction coupling (oscillator) live? Momentum does not repair the dropped
  term (see `docs/explanation/how-it-works.md` §Momentum): it re-shapes the
  iteration along the direction `d` already gives, so the panel reads it as an
  axis of its own, never as a candidate fix. `L-BFGS-B` is the ceiling reference
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
  funnel. Momentum off is the default, so the contrast arm is `mu=0.9`: it reads
  whether a gradient memory moves either stall, against serial, parallel and
  Adam.
- **Restart model — center bias (rastrigin alone).** Where a restart lands: a
  BLR posterior centered on the observed best vs a uniform re-draw. Only
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
and the momentum (`mu` 0 → 0.9); each curve moves exactly one.

## Problem set

Four run groups plus two appendix panels, one role each:

1. **Headline — the two real inverse problems.** Both mix physical units and
   both have a non-convex response; one is read in the frequency domain, the
   other in time. Keeping two of them is what stops the headline from resting
   on one recipe, and they sit on the *same* side of the property that decides
   the outcome (see "What the problem set can support" below) — which is the
   honest limit of a sample of two.
   - **Vibration isolation** (borrowed recipe). A N-stage
     coupled-oscillator / vibration-isolation network. Per-stage parameters carry
     heterogeneous physical units: mass `m_k` [kg], stiffness `k_k` [N/m],
     damping `c_k` [N·s/m]. N=20 → d≈60.
     Loss = `Σ_j log₁₀(|H(ω_j)| / |H_target(ω_j)|)²` over a log-spaced grid, with
     `H` the frequency response of a tridiagonal (banded) system, solved
     densely - one `n×n` complex solve per frequency, O(n³) rather than O(n),
     which at N=20 is the cheap side of the set. Synthetic target design → known
     `f* = 0`; boxes are physical ranges. Serves **P② + unit-invariance**.
     The response is scale-invariant (`m`, `k`, `c` scaled together leave
     `|H|` unchanged), so `f* = 0` is a whole ray through the truth rather
     than a point: what is identified is the transmissibility, not the design,
     and the score is on `f`, never on the distance to the hidden `x*`. This
     is a **design** problem, not a **recovery** one, and not by accident:
     twenty response samples cannot fix sixty parameters (the Jacobian has at
     most twenty rows, and neighboring frequencies on a smooth spectrum are
     far from independent), so the panel asks whether a method drives the
     response onto the target, never whether it finds the design. `pk_chain`
     is where recovery is the question, and it was built so that its rates are
     identifiable.
     Landscape: non-convex via resonance/interference — *not* a funnel (see Risk).
   - **Compartment kinetics** (time domain). A bidirectional chain of `n`
     compartments exchanging first-order transfer rates `f_i` / `b_i` and
     eliminating at `gamma_i`, each in its own unit (1/h) and spanning four
     decades; n=21 → d=61. Loss =
     `Σ_{j,i} log₁₀(C_i(t_j) / C*_i(t_j))²` over a dyadic geometric time grid,
     on three observed compartments, with a bolus in each end at `t = 0`.
     Synthetic truth design → known `f* = 0`; boxes are truth × [¼, 4]. Serves
     **P②** on exactly the coupling shape the diagonal approximation is built
     for — nearest-neighbor in a compartment graph — without repeating the
     oscillator's physics. Two of its features were forced by measurement, not
     chosen: a single bolus leaves most of the rates unconstrained, and the
     window's tail falls below the assay limit, so those samples are dropped
     rather than scored. Both are argued and counted in `problems.pk_chain`
     and `RESULTS.md`.
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
   unit-invariant** (a scalar `α` applied uniformly across mixed-unit
   coords), so the first secant pair — and anything seeded by it — is not the
   transformed pair. On the diagonal quadratic this
   self-corrects in one step (`d` reads `1/Hᵢ` exactly, independent of the seed
   pair's scale), so the diag control gives clean overlap; a general/multimodal
   problem would **not** guarantee same basin, so never use one for this panel.
   The exact claim is asserted at the step level (injected secant pair, as in
   `test_diag_step_scales_with_the_coordinate_unit`); the end-to-end claim is
   asserted statistically. A *property demo*, not a race.
6. **Adam lr sweep.** Log-scan of `lr` per problem, marking the panel pick
   (`ADAM_LR`) — the evidence that soar-lite has no learning rate where Adam
   pays a grid. Emitted by `--lr-sweep`.

## What the problem set can support

A panel is a **matchup table, not a verdict.** Two real problems is a sample of
two, both on the same side of the one property that decides the outcome, so the
set can say *where* each method wins. It cannot say one wins in general, and no
figure here should be read as if it did.

The dividing line is **structure, not "scientific vs toy"**:

- **soar-lite's side** — smooth, near-separable, diagonal-trustworthy coupling
  (banded or sparse), with heterogeneous per-coordinate scale. This is the
  regime the per-coordinate secant reads as "approximately right, not exact".
- **Adam's side** — strongly ill-conditioned *and* densely rotated, where the
  off-diagonal terms dominate the Hessian and a diagonal reading of it is not
  approximately right at all. That gap is structural, not a missing knob: the
  quantity soar's preconditioner normalizes is a secant ratio, which rotation
  biases, while Adam's normalizes a squared gradient, which it does not.
- **neither** — gradient-free multimodal relief (Ackley), where the gradient
  says nothing about where to move next.

The `mu=0.9` column opens a second axis in the same panels, and its verdict is
not uniform: a longer momentum window is a trade, and on the two real problems
it lands on opposite sides of even. Neither real problem is on the rotated side
of that dividing line, so the panel does not measure it — that gap is recorded
in `RESULTS.md`, not papered over.

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

- **Smooth + headline** (**P②**): `soar-lite n_paths=1` (default `mu=0`),
  `soar-lite n_paths=1, mu=0.9`, `soar-lite n_paths=16`, `Adam` (grid lr),
  `L-BFGS-B` (ceiling *reference*, not a head-to-head baseline — full-memory +
  serial line search vs our per-coordinate + batched n_paths).
- **Multimodal** (**P①**): the same three soar-lite arms, `Adam` (grid lr).
  The momentum axis runs at `n_paths=1` — 0 and 0.9 — so the curve
  family differs from its neighbors by one knob, and the `mu=0.9` arm asks the
  momentum question at **Adam's own horizon** instead of ours: 0.9 is a 10-step
  window (~1/(1-0.9)), the same window Adam's first moment runs at `b1=0.9`.
  Adam keeps its defaults rather than being refitted to mirror soar — a baseline
  whose knobs are moved to match the method under test stops being a baseline,
  and `b1=b2=0.63` would not even be a sensible Adam (its second moment would
  get a ~2.7-step memory for a quantity that wants a long one). The arm also
  tests the *default* rather than the direction: if the momentum trade described
  below were an artefact of a short window, a ten-step window would remove the
  cost. It does not - the cost stays and only the landscape moves the sign - so
  the trade belongs to the mechanism, not to the horizon. `RESULTS.md` holds the
  numbers.
- **BLR** (**restart model**, rastrigin alone, its own figure, all
  `n_paths=16`): `Sample draws`, `BLR identity`, `BLR sphere` — one method,
  three restart models, so nothing but where a restart lands differs.

  This panel measures **a surrogate against no surrogate**, and only that. A
  fitted restart's proposal is found by descending the acquisition over the
  surrogate, and those evaluations are not charged: the batch budget counts the
  `eval_f_g` calls the engine serves, and a model read is not one of them.
  `Sample draws` spends its whole budget on the function; the two fitted arms
  spend the same function budget **plus** an unbilled search over their own
  model. So the comparison is about which landing place is better, not about
  what it costs - a cost-matched fitted arm would have to pay for its
  acquisition descent out of the same budget, and that variant is not measured.
  Read every `blr` gap with that asymmetry in hand.
- **Invariance**: `soar-lite` (n_paths=1) vs `Adam`, two unit systems, overlaid
  after the coordinate transform.

The width step (`n_paths` 1 → 16) and the momentum steps (`mu` default → 0.9 and
default → 0) are the single-attribute moves; a single `n_paths=16` curve that
also flipped restart would charge the whole gap to "parallelism". Restart is not a step the
panels take: a probe showed stalls fire only where a method has already settled
into a basin (the diagonal quadratic, rastrigin), never on the rotated
quadratic, Ackley or the oscillator.

The headline BLR panel is **canceled**: a BLR restart needs a stalled stage to
act on, and the oscillator's non-convexity is resonance-type — not the funnel
that stalls a chain. The BLR proposition lives on rastrigin, where Sample's
re-draws provably never improve the best and a fitted restart is the question.

**External evidence: the bent projection on real problems.** The restart-model
question is not only ours. Tripp and Fable, "Why Does Spherical-Linear Bayesian
Optimization Work?" (2026), an analysis of Doumont, Fan et al., "We Still Don't
Understand High-Dimensional Bayesian Optimization" (2025), run the same
projection against an unprojected linear head on three real zero-order problems
at 3 seeds x 150 evaluations each, all maximized: rover (60D) spherical-linear
-1.28 against standard-linear -1.35, lasso-DNA (180D) -0.309 against -0.333, SVM
(388D) -0.134 against -0.233. The bent head wins all three, which is the
direction this panel's center-bias reading predicts. The bent *RBF* on the same
pipeline beats both on all three, so the gain is the geometry and not the linear
function class; and the same study slides the optimum to a box corner and the
ranking inverts, so the projection is a prior that the optimum is interior, not
a free win.

Two limits keep that evidence external rather than a panel of ours. The three
objectives are black boxes with no gradient — a `scipy` spline path cost over
obstacles, a per-point `sklearn` SVR fit, a per-point `LassoBench` path solve —
so they cannot ride this engine's `eval_f_g` seam, and re-measuring them here
would mean a finite-difference gradient at 60-388 evaluations per step. And 150
evaluations is a zero-order BO count, not this harness's batch axis. The numbers
above are therefore quoted, not measured here.

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
`soar-lite` n_paths=1 (default `mu=0`, `mu=0.9`) / 16 vs `Adam`, which is the
honest contrast: momentum is read as a *drift* on these landscapes, not
asserted to be an escape mechanism in general.

Two rules keep the momentum arm from being over-read. A `mu > 0` is a
**trade, not a uniform win**: it costs where the choice of basin dominates and
pays where off-diagonal coupling does, and the lattice landscape that pays the
cost is not the headline. And on a landscape where every method converges, the
number to read is the **rate**, not the floor reached — each method stops on its
own criterion, so a floor at machine precision is a statement about the stopping
rule rather than about the algorithm.

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
contract on the *benchmark*: the formal runs contribute no **tuned** knobs (Adam
pays a per-problem lr grid each time), so "no tuning" is a fact about the
benchmark, not a universal theorem about the method. "No tuning" is also not
"no parameters": the class exposes eight, and `beta` and `mu` are live windows,
not inert placeholders — the `mu=0.9` arm exists precisely because the momentum
window is a knob with a measurable effect.

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
  - **points** figure — the same runs on the information unit of a serial
    setting. Unlike the batch axis, a point axis is *not* level across arms: a
    point count is what `n_paths` inflates, so an `n_paths=16` arm serves 16
    points per batch and would trail 16x past every single chain - 48000 against
    3000 on these runs. Every arm is therefore cut at the shortest extent present
    **on the points axis**, so all curves end together and the wide arm shows the
    first ~187 of its 3000 batches. That truncation is the reading, not a loss:
    the axis asks what each method bought from the same number of objective
    points served, and the batches figure is where the wide arm is shown whole.
    The rule is per-axis, so a panel whose arms share one width (the `blr` panel,
    all `n_paths=16`) is not truncated at all.
- Seeds `N=20`; report **median, shaded [Q1, Q3]** (interquartile band), log-y,
  log x — mean ± std would model a skewed best-so-far spread as symmetric and
  dip below the log axis.
- Outputs are split by kind. `experiments/results/` holds traces only, one file
  per **(problem, arm)**: `results/<problem>/<arm-key>.json` with that arm's raw
  per-seed trace (points, batches, best-so-far, final value).
  `experiments/figures/<panel>/` holds what was drawn from them —
  `{batches,points}.png`, plus `figures/invariance/invariance.png` and
  `figures/lr_sweep/lr_sweep.png` — so a figure is always re-derivable from the
  traces and a trace never from a figure. A panel is a roster over arm files, not a
  container that owns them, so re-measuring one arm rewrites one file and every
  panel listing it picks the new numbers up — rastrigin's `n_paths=16` arm is a
  single file that both the multimodal and the blr panel draw. The default run is
  `headline,smooth,multimodal,invariance`; `--groups blr` adds the restart-model
  panel (a single rastrigin subplot). `--plot-only` rebuilds figures from the
  arm files with no re-run, naming any arm that has no trace instead of drawing a
  gap. The invariance panel writes no trace — it is re-run (fast) rather than
  re-plotted.
- The `blr` panel runs at **its own budget**, 1000 batches rather than 3000,
  in its own invocation (`--groups blr --max-steps 1000`), because
  a restart model acts only once a chain stalls, rastrigin stalls it around
  batch 700, and the remaining 2000 batches buy that panel nothing. Two
  consequences are load-bearing. The cross-panel consistency check - the
  `Sample` arm's median matching the multimodal panel's `n_paths=16` median -
  holds only because both panels share a budget, so with different budgets the
  `blr` panel is read against its own arms alone and never subtracted from a
  3000-batch panel. And the shared rastrigin `n_paths=16` arm is one file that
  both panels *read*: only the multimodal invocation writes it, at 3000, and
  the budget guard refuses a second writer at another budget. Because a run is
  prefix-deterministic, the 3000-batch trace's first 1000 batches *are* a
  1000-batch run, so the `blr` figure truncates that arm to the shorter extent
  present instead of re-measuring it.
- The arm key spells out the arm's whole configuration
  (`soar-sample-np1-mu0-restart-on`, `adam-lr0.01`, `lbfgs`) instead of
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

The headline problems' "multimodality" is resonance/interference-type
non-convexity plus, in the compartment case, coupled near-flat directions where
several rates move together; **not** the funnel type that Ackley/Rastrigin
provide. The two complement each other; the headline must not be counted as
covering the funnel slot. The banded-coupling question — whether the diagonal
approximation's "approximately right, not exact" band lands the discriminator
on soar-lite's side — is now measured on both headline problems, and the answer
is yes at equal budget; `RESULTS.md` holds the numbers and the caveats,
including that Adam was still descending when the budget ran out. Note also
that L-BFGS-B is *not* a ceiling on the compartment problem: its median loses to
the serial arm and it fails outright on a good fraction of seeds, because a
full-memory serial line search with no per-coordinate scaling walks into a bad
basin. Expecting the smooth panel's ceiling to transfer here would be wrong.

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
