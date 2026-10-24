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
  is gone - `restart` is the single cold-start call - and `nit` counts the probe
  as iteration 0, so a run that only seeds the flock reports `nit = 1` instead
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
