# Changelog

User-visible changes, newest first. This file starts at the first public
release: what happened before it is in `git log`, and the internal record
(a rename sweep, a default flipped three times before anyone could have
depended on it) is not a thing a user upgrading between versions needs to read.

## 0.1.2

The install section still told a reader to build from `main` before the first
release was out. The first release is out, so the sentence now says only what it
still means: how to run `main` ahead of the latest tag. The metadata also gains a
`Changelog` link, so PyPI shows one beside the other project links. No API
change.

## 0.1.1

The PyPI page was showing this README's four figures as broken images and its
relative links as dead ones. PyPI stores the description verbatim and rewrites
nothing, so `docs/figures/headline-batches.png` was resolved against the
project's PyPI URL. The README now carries absolute destinations — the raw files
for the figures, the repository pages for the links — so one copy of it renders
on both GitHub and PyPI. No API change.

## 0.1.0

First public release. `README.md` and the site describe what this version is, so
this entry is the inventory rather than a diff against something earlier.

- `n_path_schedule` — the whole algorithm behind one call: `n_paths` descents
  advance in lockstep under one budget, each re-drawn from the global model once
  its own stopping test stalls it (`restart_stalled`, on by default), and the
  report names the best point evaluated as `x`/`f` beside the flock the run
  ended on.
- `DiagonalPreconditioner` — a per-coordinate inverse-curvature scale estimated
  from successive gradients, plus a momentum `mu` (0 by default). No matrix and
  no learning rate to tune, and once the first secant pair exists the step is
  invariant to rescaling of the objective.
- Two plugin seams, `LocalSolver` and `GlobalModel` — `SampleGlobalModel` draws
  restarts from the domain; `BLRGlobalModel` accumulates the run's evidence into
  a Bayesian surrogate whose acquisition descent is injected.
- `pointwise` and a batched `eval_f_g` — value and gradient in one call, with an
  optional feasibility mask, so one engine serves a scalar objective, a batched
  one, or a penalized transform of a constrained problem.

Below `1.0.0` a breaking change bumps the minor version: `0.2.0` may not run code
written for `0.1.0`, and only `1.0.0` starts the compatibility promise.
