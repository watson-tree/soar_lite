# Spend less time in the objective

This page continues [the tutorial](../tutorial/getting-started.md)'s example:
`eval_f_g`, `eval_f` and `domain` below are the ones built there.

## The change

Wrap the fused seam in `jax.jit` before handing it over:

```python
import jax
from soar_lite import n_path_schedule

fast_f_g = jax.jit(eval_f_g)

result = n_path_schedule(
    fast_f_g,
    domain=domain,
    n_paths=4,
    max_iters=30,
    eval_f=eval_f,
)
```

Nothing about the contract moves: `pointwise` returns ordinary functions, and a
traced one is served like any other. The engine reads each batch on the host
after the seam has returned — values, feasibility flags, the box check — so it
never needs to see inside the trace.

## What it buys, and what it costs

What the call buys is the objective's own operations fused into one compiled
kernel per batch, instead of dispatched one at a time. What it costs is a
compile per argument shape, because jax caches a trace by shape. Which shape a
run serves, then, decides the payoff.

The batch an iteration serves holds the paths that owe work in it, so it narrows
when a path sits the iteration out. Two things make a path sit out: a point that
came back unusable — a non-finite value or gradient — and a path whose own
stopping test holds, which a converged line search reaches on a perfectly finite
landscape. A finite objective therefore makes a stall rarer, not impossible, and
does not on its own pin the width.

What decides whether the width comes back is `restart_stalled`. With the default
`True` the stalled path is drawn back in next iteration, so the width dips for
that iteration and returns to `(n_paths, d)`; paths that converge together stall
together, so such an iteration often serves no batch at all and takes every path
back at full width next time. With `False` a stalled path is parked for good: every
stall narrows the flock permanently, the width only ratchets down, and the tail
of such a run is where its distinct widths pile up. That run also reaches
`all_stalled` and stops before it can serve many batches, so there is little
work to amortize either way.

Any width compiles once, so a run that visits a few is still worth tracing,
while one whose width changes every iteration — the ratchet under
`restart_stalled=False` — pays for compilation more often than it saves
dispatch.

Neither case says the run gets faster: a cheap objective — a sum of squares is
three or four operations — is mostly dispatch already, so the fused kernel saves
little. The seam is worth tracing when the objective is the expensive part of
the batch. Measure it, don't assume it.

## Leave the value-only seam alone

`eval_f` is a different bet. `serve_trials` serves one round per batch, and the
round's width belongs to the method — a line search enumerates as many points as
it wants — so the width is not the run's to predict. The default solver reads no
trial at all, so a default run never calls this seam and there is nothing to
trace; trace it only if you wrote a method that enumerates trials, and only after
measuring.

## Check what it bought

The ledger says what was served: `result.work.n_f_g_batches` is the number of
fused batches and `n_f_g_points` the rows across them. Two runs of the same
problem with the same counters did the same amount of objective work, so
comparing wall-clock between them compares the seam, not the schedule.
