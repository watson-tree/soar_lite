# Restart from a surrogate

This page continues [the tutorial](../tutorial/getting-started.md)'s example:
`eval_f_g` and `domain` below are the ones built there.

`BLRGlobalModel` fits its posterior and builds its own `-log EI` acquisition,
but it never descends that acquisition itself: the descent recipe belongs to the
engine, so the model receives it as an injected `minimize`, a `MinimizeFn` with
signature `(acquisition, bounds, n_points, key) -> (n_points, d)`. The
acquisition is `(M, d) -> ((M,), (M, d))` (value and gradient, no feasibility
column) and `key` is a fresh subkey of the model's own key, split for this call
alone. The natural answer is to hand the engine back to itself: one restart-free
run whose width is the number of minimizers you want. Read the minimizers off
`flock_x`.

The projection is the other knob — it supplies the prior the model fits.
`SphereProjection` grows variance with distance from the origin;
`IdentityProjection`, also in `soar_lite`, passes the raw coordinates through.
Default to the first unless you have a reason not to.

```python
# eval_f_g and domain come from the tutorial's §1.2
import jax
from soar_lite import (
    BLRGlobalModel,
    Domain,
    SphereProjection,
    n_path_schedule,
)

INNER_ITERS = 40  # long enough to settle; shorter than the outer's horizon


def surrogate_minimize(acquisition, bounds, n_points, key):
    """Descend an acquisition with one restart-free n-path run."""
    # bounds arrives as the engine's Bounds: scipy-style pairs, or a (d, 2) array
    inner_domain = Domain(bounds)

    # the acquisition is two slots; the seam wants three, the last one None
    def fused_seam(x_batch):
        f, g = acquisition(x_batch)
        return f, g, None     # finite everywhere: no feasibility to mask

    result = n_path_schedule(
        fused_seam,
        domain=inner_domain,
        n_paths=n_points,     # one path per minimizer: one batch, not n runs
        max_iters=INNER_ITERS,
        restart_stalled=False,  # a pure descent, no restarting inside it
        key=key,
    )
    return result.flock_x    # one row per path; a refused path is a nan row


global_model = BLRGlobalModel(
    projection_factory=SphereProjection,  # the class, not an instance
    minimize=surrogate_minimize,
    key=jax.random.PRNGKey(0),
)

result = n_path_schedule(
    eval_f_g,
    domain=domain,
    n_paths=8,
    max_iters=200,
    global_model=global_model,
)
```

It is the engine inside the engine: the outer run asks the model for a restart
when a path stalls, the model asks `surrogate_minimize` to descend its
acquisition, and `surrogate_minimize` runs the engine again — restart-free this
time. Five things keep it honest:

- **`restart_stalled=False` inside.** The inner run has one job: reach the
  acquisition's local minimum from each start. Restarts would replace this job
  with random wandering, so they are off: a path that settles is parked for good
  and never moves again.
- **`n_paths = n_points`, not a loop.** The acquisition is batched, so one run
  of width `n_points` serves the whole cohort one row per path. Looping
  `n_paths=1` runs would buy the same points with `n_points` times the batches
  — the serial shape of a job the GPU wants wide.
- **The outer run's first restart draws, not descends.** Until `observe` has
  given the posterior a spread to standardize by, `propose` has nothing to fit
  and answers with the domain's own draw, never calling `minimize`.
- **A returned row can be all-`nan`.** A start at which the acquisition is not
  finite is refused, and so is every row of an inner run that does zero
  iterations (`max_iters=0`). An unplaced path carries an all-`nan` point; the
  outer run treats that row as an unusable start and stalls the path.
- **The acquisition has no feasibility mask (`feas=None`).** It is finite
  everywhere, so the fused seam's third slot is `None`.

`INNER_ITERS` is a horizon, not a convergence test: the inner run normally
reaches it with the acquisition already flat, which is why 40 is a reasonable
default. Raise it if the outer run keeps restarting paths that barely moved.

`MinimizeFn` and both projections are in the [API Reference](../reference/api.md),
with their exact signatures and defaults.
