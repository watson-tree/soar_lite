# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `experiments/DESIGN.md`: the restart-model panel now carries its external
  evidence. Tripp and Fable (2026) run the same inverse-stereographic projection
  against an unprojected linear head on rover (60D), lasso-DNA (180D) and SVM
  (388D) at 3 seeds x 150 evaluations, all maximized, and the bent head wins all
  three (-1.28/-1.35, -0.309/-0.333, -0.134/-0.233) - the direction this panel's
  centre-bias reading predicts. Their bent RBF beats both, so the gain is the
  geometry rather than the linear function class, and their optimum-to-corner
  sweep inverts the ranking, so the projection is a prior that the optimum is
  interior, not a free win. Those numbers are quoted, not measured here: the
  three objectives are gradient-free black boxes (a spline path cost, an SVR
  fit, a lasso path solve) and cannot ride this engine's `eval_f_g` seam.
- `tests/test_nonfinite_matrix.py`: the non-finite event matrix as one table.
  `{finite, +inf, -inf, nan} × {feasible, infeasible, no mask}` becomes nine
  rows against `_Best`; `_argmin_finite` gets its own value combinations (a NaN
  ranked first is never the minimum, `-inf` is not a smaller finite value, a tie
  keeps the first, nothing finite gives `None`); the feasibility mask gets its
  0/1 rows; and the seventh dimension - a finite value with an unusable gradient
  - gets two end-to-end rows (a half-dead start, a half-dead winner). The
  mechanisms were each covered already where they meet a specific test; this
  file states the whole table in one place, so a change to any one of the local
  `isfinite` decisions has to keep every other one's rows true.
- `NoRestartRule`: the explicit no-restart arm. It seeds the flock with the
  initialization probe and then never restarts, so a caller can ask for the
  pure multi-start descent (or the serial single chain) instead of depending on
  the EI rule's emergent behaviour on a one-sample pool.
- `eval_f`: an optional value-only evaluator seam, the f-only sibling of
  `eval_f_g` (`(f, feas?)` out, no gradient). Supplying it makes the whole
  search read values: every probe draw and every step trial is served for its
  value alone, the method rules on that value, and the single g batch the
  iteration serves grades what it kept - the probe's starts and the step's
  winners, together. A deeper probe therefore costs no extra gradient, a
  retrying line search costs one pass instead of one per trial, and a
  single-draw probe, having no group to compare, defers its start entirely: its
  value is first computed by that same batch.
  Without the seam every trial keeps its fused pass, and the engine refuses
  `n_draws > 1` with a `ValueError`.
- `pointwise`: build both eval seams from one per-point objective. It takes the
  scalar ``(d,) -> ()`` function a caller already wrote - the shape
  `jax.value_and_grad` differentiates - plus an optional per-point feasibility
  predicate, and returns `(eval_f_g, eval_f)` with the `vmap` done on this side.
  Feasibility is vmapped separately because it cannot ride inside
  `value_and_grad`, which differentiates a scalar output. Both halves are lazy
  closures, so building them runs nothing and a fused-only caller discards the
  second; the f-only half is the fused objective's own forward pass, which is
  what makes the engine's cheap-trial path reachable for free. It is not the
  seam for a value-only objective: a function JAX cannot differentiate cannot be
  `objective` at all.
- `DiagonalPreconditioner(mu=...)`: an opt-in momentum blend. With `mu > 0` the
  direction smooths a second history - an EMA `m` of the **past** steps' start
  gradients, the current step's joining it only once that step has committed -
  and blends it in as `g + m_perp`, where `m_perp = m - (<m,g>_d / <g,g>_d)*g`
  projects onto g's orthogonal complement in the **d-weighted** inner product
  `<a,b>_d = sum d_i a_i b_i`. The weighting is what keeps the invariant:
  `<d*(g+m_perp), g> = <g,g>_d > 0`, so the step still descends by construction
  and still needs no acceptance test - Euclidean orthogonality (`<m_perp,g> = 0`)
  is the wrong inner product and would void that silently while looking like a
  proof. History only is the other half of it: the projection is linear, so
  folding this step's `g` into `m` before projecting it would leave `mu*m_perp`,
  the along-`g` part being annihilated - `mu` would then scale the window and the
  transverse term at once. Default 0 (off) is not caution: near a minimiser
  `g -> 0` while `m` lags a whole window behind, so the direction stops vanishing
  and the `||g|| <= tau_rel*||g_0||` stopping test never fires. `mu` is
  independent of `beta` - that one smooths the secant pair, i.e. curvature, and
  `s`/`y` keep sharing it - and the secant pair keeps reading raw `g`
  differences, so the two knobs never entangle. The fold has one site, `fold`
  (the step's commit), so `propose` only ever reads `m` and an alpha-grid retry
  cannot fold the same gradient twice; `restart` disarms the secant and the next
  fold re-seeds `m` exactly as it re-seeds `s`/`y`, so no history crosses one;
  `mu = 0` stores no history at all.

### Changed

- `experiments/` stores one file per (problem, arm) instead of one file per
  panel: `results/<problem>/<arm-key>.json` holds that arm's raw per-seed trace,
  and a panel is a roster over those files drawn on demand. Re-measuring one arm
  used to mean re-running its whole panel, because the panel owned the data and
  wrote one dump covering every arm in it. The arm key spells out the arm's whole
  configuration (`soar-sample-np1-mu0.63-restart-on`, `adam-lr0.01`, `lbfgs`),
  so `--problems`/`--arms` filter on the same identity the path uses, and an
  engine default moving re-keys the arms that inherit it: the old trace survives
  as a record of the old configuration instead of being overwritten by a run
  whose meaning has changed. Two panels can list the same arm without storing it
  twice - rastrigin's `n_paths=16` arm is one file that both the multimodal and
  the blr panel draw, so their agreement is structural rather than a
  coincidence. `--plot-only` reads the files and names any arm with no trace
  instead of drawing a gap. Old flat `results_*.json` / `results_*.png`
  artifacts are not read by the new path.
- The panels drop the `mu=0.9` arm and read the `mu=0` arm against the default
  instead: with `0.63` as the shipped default, the momentum question is
  on-versus-off, and a third hand-set strength only spent a curve. The two
  proposition panels now share one soar-lite ladder (`n_paths` 1 / 16, then
  `mu` default / 0), so a single roster edit moves both. Nothing re-measured:
  the `mu=0.9` column in `experiments/RESULTS.md` stays as the record of what
  that arm did, annotated with the rule it ran under.
- The local solver's two smooths default to one window: `beta = 0.63`,
  `mu = 0.63` (`≈ 1/(1-β) = e` steps), instead of `beta = 0.95` / `mu = 0`.
  The secant-magnitude smooth and the direction's gradient smooth are two
  controls on one operator, and their defaults were two different windows - a
  20-step view of the curvature ratio under a bare gradient. They stay separate
  knobs: `mu = 0` is still the off switch and `beta = 0.95` still reachable. A
  caller that passed neither now runs with momentum it did not ask for, so this
  is a behaviour change, not a default tidy-up; the measurements in
  `experiments/RESULTS.md` were taken under the old defaults. Near a minimiser
  the momentum carries a stale gradient into the step, which on a quadratic - a
  problem the diagonal step otherwise lands in one shot - degrades 50-step error
  from ≤ `1e-9` to ~ `3e-6`; `beta = 0.63, mu = 0` holds `1e-9`.
- The run's starting flock is the entry's argument: `n_path_schedule(..., x0=...)`
  places it as handed, and `None` leaves the draw to the global model as before.
  `SampleGlobalModel` is now a pure draw - its `x0` is gone. The flock is where a
  control arm states a fixed start set, so it belongs beside `n_paths` and
  `domain`, not inside one of the two plugins: with it on the entry, any
  `GlobalModel` (the sampler, `BLRGlobalModel`, a caller's own) inherits it, and
  the model is asked only when a path actually restarts. The starts are validated
  once at the entry (shape against `n_paths` and `domain.d`, finiteness, and
  inside the closed box to machine precision) instead of on the model's first
  `propose`.
- `restart_stalled` defaults to `True`. The multistart restart is the cadence a
  parallel run wants: when a path's own stopping test fires, the global model
  hands it a fresh start and the flock keeps buying work until the horizon,
  instead of the converged paths parking while the live ones carry the run.
  `restart_stalled=False` keeps the frozen flock as an explicit opt-out - the
  pure-descent arm, and still the right answer for a single-basin problem,
  where a redraw is a restart into the same basin.
- A run with every path parked now ends as `all_stalled` instead of idling to
  its horizon. With restarts off, a fully parked flock is self-locking: no path
  can move and none is owed a start, so the iteration had nothing to ask for
  and each further turn only ticked `n_iters`. The engine names that state and
  returns; the turn that finds it serves no batch, so it is not counted in
  `n_iters` either (the same treatment as `unbounded` and a refused batch).
  Unreachable with restarts on, where a parked path is always owed a redraw.
- `n_path_schedule`'s `domain` is a required argument, and the run owns the
  box. `GlobalModel` no longer carries a `domain` or a `d`: `propose` is
  `propose(n_points, domain)`, handed the run's `Domain` at each ask, and the
  run's dimension is `domain.d`. Two boxes are therefore type-level
  unrepresentable rather than merely cross-checked, and the clip stays what it
  always was - the defensive seam for a model that oversteps the box it was
  handed, plus the load-bearing one for the local method's trial points.
  `BLRGlobalModel` follows the same seam: it takes a `projection_factory`
  (`Domain -> Projection`) instead of a projection instance, because the
  geometry (`glob_ls`) is a pure function of the domain and the model's
  hyperparameters, so shaping it per ask is more honest than storing one
  built from a box the model would otherwise have to carry.
- `SampleGlobalModel` refuses an `x0` that lies outside the box, rather than
  letting the clip slide it onto a wall. An authored start outside the declared
  box is the caller saying "search this box" and "begin there" at once, so it
  is named (row, dimension, both sides) instead of silently relocated to a
  degenerate point. The test carries a machine-precision tolerance scaled by
  the dtypes in play, so a boundary recomputed in another frame is not an
  error; free sides never trigger it. The clip keeps its own job - clamping
  proposals the model *generates* - because a generated proposal and an
  authored start are different claims.
- `BLRGlobalModel`'s `projection_factory` is optional. Left unset, it builds
  `SphereProjection` with its own defaults - the reference model's centre-biased
  bend - so the common case names no projection at all, and the factory drops
  back to being the override seam for a mask, another `kappa`, or no bend.
  `IdentityProjection` stays a first-class alternate prior rather than a
  stand-in for "no projection", and its docstring now says so. The two
  extension seams are documented where they are crossed: `GlobalModel`
  (`observe` + `propose`) is what a surrogate of another family implements, and
  `Projection.features` is the shared feature map such a surrogate may read
  without reusing any of this model's fit.
- With no `global_model`, an unbounded `domain` is accepted instead of
  refused. The engine's default `SampleGlobalModel` draws whatever the domain
  admits - uniform inside a bounded dimension, `N(0, scale_i^2)` inside a free
  one, that same normal truncated inside a half-bounded one - so a declared
  unbounded side is a prior the domain already states in coordinate units, and
  the old refusal (which claimed a uniform default had nothing to draw there)
  contradicted the model's own contract. Measured on a free dimension, the
  default flock's spread tracks `domain.scale` (std/scale 1.028 at 400 starts,
  mean 0), so nothing about the draw comes from a box width. An unbounded run
  spells its unbounded sides in the `Domain`; there is no `domain=None` left to
  refuse.
- The seams are checked for being callable, not only the two Protocol objects.
  `eval_f_g`, `eval_f`, `sample` and `budget_check` are refused with a
  `TypeError` naming the argument before any batch is served; before this a
  mistyped dial reached the first batch and failed there as `'int' object is
  not callable`, a symptom one frame away from the argument that caused it.
- A feasibility mask is validated instead of being read as truthiness. Only `0`
  (or `False`) marks a point infeasible, so a batch whose mask holds anything
  else - a signed constraint margin, whose violated points are all negative, or
  a `nan` - is now refused with a `ValueError` before either best absorbs it.
  Before this every one of those flags counted as feasible: a mask of `nan` or
  `-2.0` made a single served point at `f = 1.0` the reported answer even
  though the caller had marked it out. No cast repairs it (a NaN casts to
  `True`), and the mask is the one input that says which points the answer may
  come from, so it is refused rather than guessed at.
- The restart verdict no longer refuses to fire when no finite point has been
  served (`fun_raw` is `+inf`). It used to answer False until the horizon ended
  the run, which closed a loop the flock cannot leave on its own: recovering
  needs a probe, a probe needs a due verdict, and the verdict needed a finite
  best. One unlucky initialization group - every start dead - therefore froze
  the run: measured on a single path with a dead first draw, 20 rounds spent 1
  batch and returned `fun = inf` under `status = max_iters`. The verdict now
  treats "nothing to compare" like the cold start it resembles - the probe is
  the only action that buys the first finite sample - so it restarts at a due
  gate, and the dead samples still grow the pool, so the re-probe interval keeps
  growing (1, 2, 3, ...) as with every other probe. The same run now recovers
  after four probes and returns `fun = 0.0`; an all-dead landscape spends 6 of 40
  rounds probing instead of idling to the horizon.
- A restarted path now counts its probe round on its stage clock, so paths stay in
  phase. The engine reports every path every round on one continue channel: a
  round that took no step - a stall, or a re-probe's probe round - arrives as
  `None`. A restarted path's clock used to stay at 0 while its cohort ticked,
  and it opened its next stage one round late per restart - a cumulative drift,
  not a fixed offset, that left a restarting flock's paths expiring in ever more
  scattered rounds. The stage is now `M + 1` rounds: `M` folds plus the round it
  opened in, which no stage folds a sample in. The continue side reads the newest
  `M` of the `M + 1` folds a continuing path holds, so both sides of the verdict
  still read the same count. Runs whose paths restart at different rates now
  spend their horizon differently - a semantics change, not a refactor.
- A served *feasible* point whose value is `-inf` now ends the run: it becomes
  the answer (`fun = -inf`, `x` is that point) and `status` reports the new
  `"unbounded"`. No later trial can beat `-inf` and no step can improve on it,
  so the horizon was being spent after the answer was already in hand - and
  before this the point was dropped from `fun` entirely, leaving the run
  reporting a larger value than one it had served. An infeasible `-inf` still
  changes nothing: the mask is what separates "an answer no later trial can
  beat" from "a trial the caller cannot use".
- The restart side reads *every* group-min in the pool, not only the finite ones
  and the `+inf` dead starts: a `nan` or a `-inf` is now a zero term of the same
  mean and counts in the denominator, so `EI_restart` is the finite KDE scaled
  by `|finite| / M` with `M` the pool's length - the same `M` the continue side
  is read over. Before this only a `+inf` was a zero term, so a flock whose
  starts landed on a `nan` hole (or on `-inf`) had a window of 1 and re-probed
  every round, forever: the "each dead sample pushes the next ask a window out"
  spacing held for dead starts of one shape only. Where a run does serve `nan`s
  this also lowers `EI_restart` by the dead fraction, so measured results on
  such problems move - a semantics change, not a refactor.
- The restart verdict fires on a tie: `EI_restart >= EI_continue`, not `>`. When
  continuing promises nothing - a stalled path, a flat stage, a pool with no
  spread or of dead starts - both sides read 0, and a probe is then the only
  action that buys evidence. It cannot thrash: the verdict's window is the
  pool's length plus the round the stage opened in, so every tie-restart appends
  a sample and pushes the
  next ask a window further out (interval 1, 2, 3, ...). The `n_paths=1`
  "never restarts" property was a consequence of the strict `>` and is gone;
  `NoRestartRule` is now the explicit way to ask for it.
- `n_probe_evals` is now `n_draws`. The knob counts draws - batches of
  candidate points whose column minima become the starts - and one draw is the
  same unit whether the seam serves it fused or value-only, which "evals" (tied
  to one `eval_f_g` call) no longer was. The default and the
  `eval_f` requirement above one draw are unchanged.
- The run's running best is `_Best` internally: `fun`/`x` is the reported answer
  and `fun_raw` is the barrier a verdict reads. Same numbers, named after what
  `NPathResult` reports instead of the optimizer jargon "incumbent"; the float
  handed to a verdict is `best_fun_raw`.
- `LocalSolver.verdict` takes `(i, p, x, f)`, not the served trial point: a
  method rules on the step's start and the trial's coords and value. The trial's
  gradient is deliberately absent, because the engine buys a gradient only for
  the step's winner - a method that needs one to rule is out of this engine's
  reach. The winner's gradient is then first seen at the commit point; an
  unusable one stalls the path there and `fold` never runs, so the method is
  never told a step landed that did not. Everything a step needs to re-arm is
  now armed by the fold, not by the ruling.
- An iteration buys **one** gradient batch. Both halves decide on values first
  (the restarted paths by their probe, the cohort by its trials) and the
  gradients those decisions need go out as a single call, so a probe and a step
  no longer pay separate round trips. What is billed is unchanged in kind - the g
  pass - and a run without the seam bills exactly as before.
- `EIRestartRule` no longer fits a single Gaussian to either side. Both the
  continue side (the path's Δ stage) and the restart side (the shared payoff
  pool) are now a scott_mad KDE - one Gaussian kernel per sample, bandwidth
  `1.06 · MAD/Φ⁻¹(3/4) · n^(-1/5)` - so a wide or skewed sample is read as the mixture
  it is instead of having its tail inflated by a single-Gaussian fit. This is
  the regime behind the spurious restarts on single-basin problems (oscillator
  F20, `n_paths=16`), where the old fit priced restarts the payoff history did
  not support; scott_mad is chosen for its cost (O(n ln n), ten lines), not
  because it was shown to beat a direct-plug-in bandwidth end to end.
  Degenerate corners are unchanged: a stage or pool with no spread has
  `h = 0`, so the read is exactly `max(f⋆ - x, 0)`.
- A stalled path no longer fabricates a sample. `observe_step` takes
  `delta: float | None`, and the engine reports a stall as `None` - absence, not
  a value - so folding an `inf` that had to be filtered back out at the read is
  gone. The continue side is then exactly the evidence the path has: a stalled
  path's continuation is 0 by definition, so its verdict is one-sided
  (`EI_restart > 0`). One stage clock per path counts **iterations**, not folds,
  so both shapes of report advance it and a verdict comes due on the same round
  whatever the path lived through - a path that stalls mid-stage keeps the wait
  it had already served instead of starting a second full one. A verdict closes
  its stage whichever way it went (a window is one stage, the answer is not part
  of the window), so a path that is never restarted is re-judged once per window
  rather than re-asking the same question every round; `restart` therefore only
  reports the event, and a rule that resets at the verdict treats it as a no-op.
  A moving path still reads a full window of Δ, so the two sides are only ever
  compared on paths that actually folded one.
- A `+inf` group-min is read as a dead start instead of being dropped: it is a
  legal observation whose own improvement is exactly 0, so `EI_restart` is the
  finite KDE mean diluted by the fraction of starts that were alive
  (`|finite| / (|finite| + |+inf|)`), with the bandwidth taken from the finite
  subset alone. `nan` is still no information, and `-inf` is still not read.
- Initialization and restart are one operation: the engine's first iteration is
  the initialization probe, driven by the rule's cold-start verdict
  (`should_restart` must return `True` while the group-min pool is empty, so an
  empty `group_mins` is the test - a pool that exists but is entirely dead must
  not re-probe). `LocalSolver.init`
  is gone - `restart` is the single cold-start call - and `n_iters` counts the probe
  as iteration 0, so a run that only seeds the flock reports `n_iters = 1` instead
  of `0`, and a refused probe (`0`) is no longer confused with a refused first
  step (`1`).
- The continue side's sample is now one iteration's step, not one
  `n_draws`-iteration horizon: the `_Horizon` clock was removed, and a
  restarted path skips sampling for the iteration its probe served.
  `n_draws` therefore defaults to `1`; a larger probe is only
  cost-matched with the continue side through `eval_f` (see Added), and is
  refused without it.
- `pytest` is now a declared `dev` dependency and the canonical test runner
  (`uv run --no-sync python -m pytest -q`); CI installs from the lockfile with
  `--frozen`. Each test module stays runnable as a script.
- `publish.yml` runs the tests in an unprivileged job and publishes from a
  separate job holding only `id-token: write`, bound to a `pypi` environment.
  The build backend is capped (`hatchling>=1.27,<2`) because it is resolved from
  the network at build time and is not covered by `uv.lock`.
- The engine's own stop is an iteration horizon, `max_iters`, not a batch meter.
  One iteration costs one unit whatever it serves - from no batch, for a fully
  stalled flock, up to a probe plus however many trials the method's own line
  search serves - so the stop is a
  deterministic, reproducible bound rather than a cost estimate; read
  `n_batches`/`n_points` for what a run actually billed. `time_limit` adds the
  same stop in wall-clock seconds, checked between iterations, and at least one
  of the two must be given (else `ValueError`).
- Only the fused value+gradient pass is billed: `n_batches`/`n_points` count `g`
  batches and `g` points, and a probe's f-only draws are neither charged nor
  gated. A run with no f-only seam and `n_draws = 1` is unaffected (there
  are no such draws); a multi-draw probe no longer inflates the two counts, so
  they stay on the axis that dominates. The trade: the gate and the deadline
  have a blind spot under one probe.
- A single-draw probe defers its whole start, not just its gradient, as long as
  `eval_f` is supplied: a draw with no group to compare has no value worth
  reading early, so the start travels to the iteration's one g batch unread and
  both halves arrive from that pass. Deferral is keyed on the seam, not on
  `n_draws`, and the cheap half it spends nothing of is the `f` a
  value-read draw would have had recomputed anyway.
- `status` is now `max_iters`, `time_limit` or `budget_exhausted`.
- `budget_check` is unchanged as a per-batch gate but no longer a substitute for
  the horizon: a stalled iteration asks for no batch, so a gate alone cannot
  stop a spinning flock.
- The warm-up is no longer a guarded step. The first step samples the same
  `alpha` grid as every later one, along `-g` instead of `-d ⊙ g`, and under the
  default `n_alpha=1` it commits one full step exactly as proposed.
  `needs_verdict` is therefore phase-independent (`n_alpha > 1`), so a warm-up is
  ruled only when the grid actually compares samples; otherwise the engine's own
  finiteness gate is the whole of the acceptance. One mechanism now covers both
  phases, and the unit-rescaling first-step artifact (a coordinate-scaled warm-up
  needing `ceil(log2(s²))` extra halvings) disappears with the halvings. What is
  given up is the automatic localization of the first secant pair: an overshoot
  now stands until the next secant corrects it, `alpha0 < 1` damps it by hand,
  or `n_alpha >= 2` re-localizes it.
- `experiments/compare_baselines.py` plots `soar-lite n_paths=16` twice, once
  pinned to `NoRestartRule` and once to `EIRestartRule`, and names the rule in
  every curve label. The single `n_paths=16` curve moved width and restart rule
  together, so the step from `n_paths=1` could only be read as "what parallelism
  buys" whole - which is also how `DESIGN.md` had described it. Now
  `n_paths=1` -> `n_paths=16` at a fixed rule is the parallelism step and
  `NoRestartRule` -> `EIRestartRule` at a fixed width the restart step. The
  default rule is passed explicitly for the same reason: a curve that inherits
  it does not say what it measured.

### Removed

- `max_ls`: the engine no longer caps a step's enumeration. How long a step may
  run already belongs to the method (its convergence test, its own trial
  budget), and `verdict` is where it says so, so an engine-side cap was a
  second, invisible stopping rule for a decision the method owns. The engine now
  loops until every path accepts or gives up, which makes one contract
  load-bearing: `verdict` must eventually return `accept` or `give_up`. A custom
  method that retried forever used to stall on `max_ls` alone and now hangs, so
  give it its own bound. `DiagonalPreconditioner` already has one (`n_alpha` for
  the post-warm-up grid; the warm-up's geometric shrink bottoms out at a
  zero-length step, which passes Armijo).
- `max_evals`: replaced by `max_iters` (batch budget - iteration horizon). The
  migration is not a rename - an iteration can serve up to `n_draws + 1`
  batches plus the method's own trial count - so a straight `max_evals=N` →
  `max_iters=N` raises the work
  budget by up to that factor.
- The `all_stalled` status and its early exit: a fully stalled flock now idles to
  the horizon, ticking one stall report per path per iteration, instead of
  reading the rule's cadence to decide it can never be revived.
- `armijo_c`, and the warm-up Armijo backtrack it gated. The guard protected the
  objective's value, not the secant signal - on a quadratic `d = s/y` is exact at
  any step length, so the backtrack walked a length down for nothing - and it was
  the only reason one phase needed a mechanism the other did not. Passing
  `armijo_c=` to `DiagonalPreconditioner` is now a `TypeError`; `n_alpha >= 2` is
  the one way to localize a first step.

### Fixed

- A probe start is committed only when `f` **and** its gradient are finite. A
  finite `f` with a non-finite gradient was accepted as live and then proposed
  from (`-p.g`), so one NaN poisoned that coordinate's magnitude EMAs for the
  rest of the run. Such a start now arrives stalled, exactly like a dead draw,
  and waits for an EI verdict. The step side already gated trials on both.

## [0.1.0] - unreleased

### Added

- `n_path_schedule`: the n-path engine. Runs `n_paths` paths in lockstep, one
  batched evaluation per iteration, and owns the box, the shared budget, the restart
  verdicts and the stall state machine.
- `DiagonalPreconditioner`: the default local method. Per-coordinate diagonal
  secant step, EMA-smoothed, with a gradient-norm floor and an Armijo-guarded
  step grid.
- `EIRestartRule`: the default restart rule. Compares expected improvement of
  continuing a path against restarting it, from the path's own f changes and
  a pool of restart payoffs.
- `LocalSolver` and `RestartRule`: the two plugin protocols, so the method or
  the restart decision can be replaced without touching the engine.
- `experiments/compare_baselines.py`: the reproducible baseline comparison.
