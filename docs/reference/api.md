# API Reference

Generated from the docstrings in `soar_lite/`. The public surface is deliberate
and small, in four groups: the entry point, the data types it hands back and
through, the two plugin contracts, and the two concrete defaults.

## Entry point

::: soar_lite.n_path_schedule

## Data types

::: soar_lite.NPathResult
::: soar_lite.NPathStatus
::: soar_lite.PathPoint
::: soar_lite.Verdict

## Plugin contracts

Code against these to replace a piece; the engine calls them, never the reverse.

::: soar_lite.LocalSolver
::: soar_lite.RestartRule
::: soar_lite.EvalFG
::: soar_lite.EvalF
::: soar_lite.Sampler
::: soar_lite.Bounds

## Concrete defaults

Pass one by name to swap it, or implement the contract above to replace it.

::: soar_lite.DiagonalPreconditioner
::: soar_lite.EIRestartRule
