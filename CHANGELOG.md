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

- Initialization and restart are one operation: the engine's first iteration is
  the initialization probe, driven by the rule's cold-start verdict
  (`should_restart` must return `True` while the payoff pool is empty, so an
  empty `group_mins` - not `pool_size == 0` - is the test). `LocalSolver.init`
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
  stalled flock, up to a probe plus `max_ls` trial batches - so the stop is a
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
- `status` is now `max_iters`, `time_limit` or `budget_exhausted`.
- `budget_check` is unchanged as a per-batch gate but no longer a substitute for
  the horizon: a stalled iteration asks for no batch, so a gate alone cannot
  stop a spinning flock.

### Removed

- `max_evals`: replaced by `max_iters` (batch budget - iteration horizon). The
  migration is not a rename - an iteration can serve up to `n_probe_evals + 1 +
  max_ls` batches - so a straight `max_evals=N` → `max_iters=N` raises the work
  budget by up to that factor.
- The `all_stalled` status and its early exit: a fully stalled flock now idles to
  the horizon, folding one `inf` sample per path per iteration, instead of
  reading the rule's cadence to decide it can never be revived.

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
