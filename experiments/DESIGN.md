# Experiment design: `compare_baselines`

The canonical reproducible comparison artifact. One script, three problem
groups (`headline`, `smooth`, `multimodal`) plus two appendix panels
(unit-rescaling invariance, Adam's lr sweep). This document is the lossless
handoff: read it before touching `compare_baselines.py` so no settled detail is
re-derived from memory.

This document states no measured numbers. Every figure and table an experiment
produced lives in `RESULTS.md`, tagged with the lineup and budget that produced
it, so an old reading cannot age in place here; what follows is the design, the
budget contract and the conclusions those readings support.

## Goal

Draw the boundary of where **soar-lite beats the baselines**, never only where
it wins. The question is comparative on every panel: under the same budget,
does the per-coordinate secant preconditioner reach a lower best-so-far than
Adam (a grid-tuned adaptive step), and how much of any remaining gap is
"diagonal vs full second order"? `L-BFGS-B` supplies that second number as the
full-memory ceiling, so a gap to it is not read as a tuning failure.

Two single-attribute steps explain *why* a curve sits where it does. Neither is
a claim on its own:

- **Width** (`n_paths` 1 → 16) spreads independent starts over the domain.
- **Momentum** (`mu` default → 0.9) reads the **descent process**, not an
  endpoint. The figures are read for rate as well as for final position, and a
  memory is what moves that rate: it trades overshoot along steep directions
  against slow progress along flat ones, on the window the *scaled spectrum*
  calls for (see `docs/explanation/how-it-works.md` §Momentum). No panel here
  has a problem whose discriminator is a stall — a stalled chain has already
  settled, so a memory cannot change where it stops, only how fast it got
  there. It is a trade, not a repair: momentum re-shapes the iteration along the
  direction `d` already gives, so it never buys back the dropped off-diagonal
  curvature — only the landscape moves the sign of the trade.

Design rule (from the start): problems must be **medium-to-high dimensional** —
the regime this library positions itself in. In low dimension the sampled paths
entangle: with few coordinates to spread them apart, several starts converge
into the same basin, so the width `n_paths` buys little and soar's multi-path
advantage is dulled there, not sharpened. There is no upper bound on `d` in
principle; the set runs at `d≈60–100`, and the `d≈200` end of the band is a
**budget** ceiling — above it every batch pays more for the same kind of
evidence, not for a new one.

Every arm runs with `restart_stalled` **off**. A restart would hand soar-lite an
advantage the baselines do not have: Adam and L-BFGS-B stop on their own
criterion, while a stalled soar path that keeps re-drawing keeps spending
budget. Off, a converged chain also just stops, so stopping is symmetric. A
probe showed stalls fire only where a method converges into a basin — the
diagonal quadratic (~step 4) and rastrigin (~batch 700) — never on the rotated
quadratic or the oscillator, and on rastrigin a uniform re-draw
provably never improves the best, so nothing is lost by turning it off. The
engine keeps the restart capability, and the harness keeps its public `model=`
factory (`BLRGlobalModel`), for a future problem where a fitted restart pays;
no panel arms it as a step.

## Problem set

Three run groups plus two appendix panels, one role each:

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
     `f* = 0`; boxes are physical ranges. Serves the headline race.
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
     Landscape: non-convex via resonance/interference — *not* a funnel (see Risks).
   - **Compartment kinetics** (time domain). A bidirectional chain of `n`
     compartments exchanging first-order transfer rates `f_i` / `b_i` and
     eliminating at `gamma_i`, each in its own unit (1/h) and spanning four
     decades; n=21 → d=61. Loss =
     `Σ_{j,i} log₁₀(C_i(t_j) / C*_i(t_j))²` over a dyadic geometric time grid,
     on three observed compartments, with a bolus in each end at `t = 0`.
     Synthetic truth design → known `f* = 0`; boxes are truth × [¼, 4]. Serves
     the headline race on exactly the coupling shape the diagonal approximation
     is built for — nearest-neighbor in a compartment graph — without repeating the
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
3. **Multimodal — Rastrigin**, d=100 (drop to 50 if cost bites). A regular
   lattice of local minima where progress is piecewise, since the cos
   oscillation keeps turning the gradient: the panel reads the descent rate
   there, not where a chain stops. One problem, not a sample of two — Ackley
   was retired from this group because no arm on it ever took a step (see
   `DEPRECATED.md`).

Two appendix panels, property demos rather than races:

4. **Invariance — unit-rescaling consistency.** Run on the diag control:
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
5. **Adam lr sweep.** Log-scan of `lr` per problem, marking the panel pick
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

Off-diagonal coupling has two sources, and the headline chain has the second:
*rotation coupling* needs same-unit components of one vector (spatial velocity
or acceleration components, a preferred direction mixing them) — heterogeneous
units cannot rotate into each other, which is why the rotated quadratic is a
pure artificial stress test; *interaction coupling* needs no unit match — the
loss depends jointly on two parameters (a resonance tunes with mass and
stiffness together, `omega^2 ~ k/m`, and the stiffness profile couples within
the chain), and the oscillator's `kg`/`N/m`/`N*s/m` coordinates carry exactly
that, stated in its own spec ("approximately right, not exact"). So the headline
panel is *not* exempt from the direction error: whether soar-lite still wins
there while dropping real coupling is what that panel measures, not something
asserted up front.

A third side used to close that line: gradient-free multimodal relief (Ackley),
where the gradient says nothing about where to move next. It is no longer a
panel — it discriminated nothing (see Risks, and `DEPRECATED.md` for the
readings). The conclusion it demonstrated is stated here rather than drawn: from
a regular start, no first-order method reaches the funnel.

The `mu=0.9` column opens a second axis in the same panels, and its verdict is
not uniform: a longer momentum window is a trade, and on the two real problems
it lands on opposite sides of even. Neither real problem is on the rotated side
of that dividing line, so the panel does not measure it — that gap is recorded
in `RESULTS.md`, not papered over.

## Methods × panels

Curves differ per panel; each curve has one role, and each single-attribute
step moves exactly one knob (`n_paths` or `mu`); restart is off in every arm:

- **Smooth + headline**: `soar-lite n_paths=1` (default `mu=0`),
  `soar-lite n_paths=1, mu=0.9`, `soar-lite n_paths=16`, `Adam` (grid lr),
  `L-BFGS-B` (ceiling *reference*, not a head-to-head baseline — full-memory +
  serial line search vs our per-coordinate + batched n_paths).
- **Multimodal**: the same three soar-lite arms, `Adam` (grid lr).
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
- **Invariance**: `soar-lite` (n_paths=1) vs `Adam`, two unit systems, overlaid
  after the coordinate transform.

The width step (`n_paths` 1 → 16) and the momentum step (`mu` default → 0.9)
are the single-attribute moves; a single `n_paths=16` curve that also flipped
restart would charge the whole gap to "parallelism".

L-BFGS-B appears only on the smooth and headline panels. Its role there is the *ceiling
reference* — how close soar-lite's diagonal secant gets to a full-memory
second-order method on smooth, convex landscape (on the compartment problem it is
not even that; see Risks). That role does not exist on
the multimodal panel: the question there is the descent *rate* where progress is
piecewise, and a full-memory method answers a different one. L-BFGS-B is
second-order and restart-free — on Rastrigin it would dive
deterministically into the nearest local minimum and set there at
second-order precision, so its curve could only show "a second-order method
also gets stuck", which says nothing about how a first-order memory shapes the
way down. It is also serial (not the batched cost axis; see Budget) and its
line search is fragile on Rastrigin's cos oscillation.

A `mu > 0` is read as a trade (Goal): it costs where the choice of basin
dominates and pays where off-diagonal coupling does. And on a landscape where
every method converges, the number to read is the **rate**, not the floor
reached — each method stops on its own criterion, so a floor at machine
precision is a statement about the stopping rule rather than about the
algorithm.

Adam learning rate is **per-problem grid-tuned**: the picks span over two
orders of magnitude (diag 1e-2 → rotated quadratic and oscillator
2.0), and on diag `lr=1e-4` costs 52 orders of magnitude against the pick
(`6.6e3` against `8.8e-49`). On diag every grid point from 1e-2 up
sits on the same floor or plateau, so that pick is one point of that tie
rather than a strict argmin — the figure's diamond marks the pick, not a
minimum. The grid and picks live in `ADAM_LR` / `--lr-sweep` in
`compare_baselines.py`, which prints the table it draws; soar-lite has no
learning rate. Tuning is itself budget — every trial is more objective points,
and it is manual, not adaptive. soar-lite contributes no tuned knob to any run —
its step knobs sit at their defaults because scalar step rescaling was swept and
settled against: the residual error is in the step's *direction*, not its length
(`DEPRECATED.md`). `README.md` states that contract in full.

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
- Two honest budgets, both read off the **same** runs, no re-runs. They differ in
  exactly one arm: every arm serves one point per batch, so its curve is the same
  on both axes, and only the wide `n_paths=16` arm moves between them.
  - **batches** figure — every method 3000 batches. This is the wall-clock axis
    of a batched setting, the axis this library is built for, and the only axis
    where `n_paths=16` and a single chain compare at equal cost per step. It is
    the main chart, and the one `README.md` shows, with the wide arm whole at its
    16 points per batch.
  - **points** figure — the same runs on the information unit of a serial
    setting, where `n_paths` pays for its width. A point count is what
    `n_paths` inflates, so an `n_paths=16` arm serves 16 points per batch and
    would trail 16x past every single chain - 48000 against 3000 on these runs.
    Every arm is therefore cut at the shortest extent present **on the points
    axis**, so all curves end together and the wide arm shows the first ~187 of
    its 3000 batches. That truncation is the reading, not a loss: on equal
    points, `n_paths=16` splits one long run into sixteen short ones - 1/16 the
    depth each - and reports their best, so the axis asks whether a breadth of
    short independent searches beats one deep one. Parallelism is a trade of
    per-chain depth for a tighter best-of-n spread, and the points axis is the
    only place that trade is charged instead of hidden behind a 16x point
    budget. It is the axis of the curve's *shape*, not of a final value: a
    converged median read off a truncated curve is an artefact, so final numbers
    come from the batches column in `RESULTS.md`.
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
  panel listing it picks the new numbers up. The default run is
  `headline,smooth,multimodal,invariance`. `--plot-only` rebuilds figures from the
  arm files with no re-run, naming any arm that has no trace instead of drawing a
  gap. The invariance panel writes no trace — it is re-run (fast) rather than
  re-plotted.
- The arm key spells out the arm's whole configuration
  (`soar-sample-np1-mu0-restart-off`, `adam-lr0.01`, `lbfgs`) instead of
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
2. `experiments/problems.py` — the three groups (`headline`, `smooth`,
   `multimodal`) + the invariance and `all` builders, each problem
   exposing the soar-lite seam `x_batch → (loss, grad, feas)`.
3. `experiments/methods.py` — one driver per arm (the rosters are in
   `Methods × panels`). The restart-model factory
   (`surrogate_minimize` + the identity / sphere projections) stays wired, but
   no panel arms it.
4. `experiments/compare_baselines.py` — harness over `N=20` seeds, one raw trace
   file per (problem, arm) under `results/`; panels: headline / smooth /
   multimodal, plus invariance and the lr sweep.
5. Plot from the arm files. **Self-check first**: two curves on the diag
   quadratic (soar-lite n_paths=1 vs Adam) before running everything.
6. `experiments/README.md` — run instructions, written alongside the code.

## Open TODOs (release hygiene, outside this experiment)

- Fill LICENSE copyright holder; pyproject `authors` + `[project.urls]`.
- Switch notebook install source git → PyPI; add Open-in-Colab badge after
  publication.

## Risks and known limitations

- **The headline is not the funnel.** The headline problems' "multimodality" is
  resonance/interference-type non-convexity plus, in the compartment case,
  coupled near-flat directions where several rates move together; **not** the
  funnel type that Rastrigin provides. The two complement each other; the
  headline must not be counted as covering the funnel slot.
- **L-BFGS-B is not a ceiling on the compartment problem.** Its median loses to
  the serial arm and it fails outright on a good fraction of seeds, because a
  full-memory serial line search with no per-coordinate scaling walks into a bad
  basin. Expecting the smooth panel's ceiling to transfer here would be wrong.
- **The funnel slot has a deliberate blind spot.** No saddle/vanishing-gradient
  handling is built in: a Zhang-Li power iteration triggered on a small gradient
  norm was considered and rejected, because multi-path sampling already covers
  that regime and the engine keeps a restart knob for it, so the extra
  complexity would buy nothing. That blind spot was demonstrated on Ackley,
  where every arm froze at its starting plateau — which is why it is retired
  (`DEPRECATED.md`); rastrigin is the multimodal problem that remains, and
  `RESULTS.md` holds its numbers.

The banded-coupling question — whether the diagonal approximation's
"approximately right, not exact" band lands the discriminator on soar-lite's
side — is now measured on both headline problems, and the answer is yes at equal
budget; `RESULTS.md` holds the numbers and the caveats, including that Adam was
still descending when the budget ran out. That is a finding, not a caveat, which
is why it sits outside the list.

## Appendix: dfbench (Learn2Design-2026) relationship

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

## Readings live elsewhere

Every measured number this design has produced — the current lineup's medians,
the previous restart on/off lineup's Ackley and unit-rescaling readings, and the
step-length sweeps whose code path no longer exists — lives in `RESULTS.md`,
one entry per lineup, newest first. None of them are restated here, so none of
them can age in place.
