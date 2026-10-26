# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `evaluate_f`: an optional value-only evaluator seam, the f-only sibling of
  `evaluate_batch` (`(f, feas?)` out, no gradient). Supplying it lets a
  multi-draw probe compute one gradient for all its draws - `n_probe_evals`
  value-only batches, then one fused batch at the per-column winners - which is
  what makes a probe deeper than one draw cost-matched with a step. Without it
  the engine refuses `n_probe_evals > 1` with a `ValueError`.

### Changed

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
  `h = 0`, so the read is exactly `max(B - x, 0)` and `n_paths=1` still never
  restarts.
- A stalled path no longer fabricates a sample. `observe_continue` takes
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
  `n_probe_evals`-iteration horizon: the `_Horizon` clock was removed, and a
  restarted path skips sampling for the iteration its probe served.
  `n_probe_evals` therefore defaults to `1`; a larger probe is only
  cost-matched with the continue side through `evaluate_f` (see Added), and is
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
  gated. A run with no f-only seam and `n_probe_evals = 1` is unaffected (there
  are no such draws); a multi-draw probe no longer inflates the two counts, so
  they stay on the axis that dominates. The trade: the gate and the deadline
  have a blind spot under one probe.
- A single-draw probe keeps the fused path even when `evaluate_f` is supplied:
  there is no losing gradient to defer (the winner is the draw), so an f-only
  draw plus a winner batch would serve the same point twice for nothing.
  Deferral is keyed on `n_probe_evals`, not on the seam being present.
- `status` is now `max_iters`, `time_limit` or `budget_exhausted`.
- `budget_check` is unchanged as a per-batch gate but no longer a substitute for
  the horizon: a stalled iteration asks for no batch, so a gate alone cannot
  stop a spinning flock.

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
  migration is not a rename - an iteration can serve up to `n_probe_evals + 1`
  batches plus the method's own trial count - so a straight `max_evals=N` →
  `max_iters=N` raises the work
  budget by up to that factor.
- The `all_stalled` status and its early exit: a fully stalled flock now idles to
  the horizon, ticking one stall report per path per iteration, instead of
  reading the rule's cadence to decide it can never be revived.

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
