# Changelog

User-visible changes, newest first. This file starts at the first public
release: what happened before it is in `git log`, and the internal record
(a rename sweep, a default flipped three times before anyone could have
depended on it) is not a thing a user upgrading between versions needs to read.

## [Unreleased]

### Added

- `Work`: one frozen value holding the four counts a run billed -
  `n_f_points` / `n_f_batches` for the value-only seam, `n_f_g_points` /
  `n_f_g_batches` for the fused one - reachable as `result.work`. The same four
  numbers stay readable on `NPathResult` itself, so nothing that reads them has
  to move.

### Changed

- **Breaking:** `budget_check` is now handed that `Work`, instead of being
  called with no argument. A gate can read the engine's own ledger rather than
  keep a parallel tally that can disagree with it; migrate
  `lambda: n < N` to `lambda work: work.n_f_g_batches < N`. `DEPRECATED.md`
  records the `max_*` cap family this replaced, and why a family of caps was
  the wrong answer.
