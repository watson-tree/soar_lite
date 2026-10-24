# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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
  continuing a path against restarting it, from the path's own loss changes and
  a pool of restart payoffs.
- `LocalSolver` and `RestartRule`: the two plugin protocols, so the method or
  the restart decision can be replaced without touching the engine.
- `experiments/compare_baselines.py`: the reproducible baseline comparison.
