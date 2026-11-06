# Restart from a surrogate

This page continues [the tutorial](../tutorial/getting-started.md)'s example:
`eval_f_g` and `domain` below are the ones built there.

`BLRGlobalModel` fits its posterior and builds its own `-log EI` acquisition,
but it never descends that acquisition: the descent recipe is the engine's, so
the model takes it as the injected `minimize` — a `MinimizeFn`,
`(acquisition, bounds, n_points, key) -> (n_points, d)`, where the acquisition
is `(m, d) -> ((m,), (m, d))` (value and gradient, no feasibility column). The
natural answer, and what the experiment drivers do, is to hand the engine back
to itself: one restart-free run whose width is the number of minimizers you
want, read off `flock_x`.

The prior is the other knob. `SphereProjection` grows variance with distance
from the origin, and `IdentityProjection` passes the raw coordinates through —
take the first unless you have a reason not to.

```python
import jax
from soar_lite import (
    BLRGlobalModel,
    DiagonalPreconditioner,
    Domain,
    SphereProjection,
    n_path_schedule,
)

INNER_ITERS = 40  # long enough to settle, short of the outer run's own


def surrogate_minimize(objective, bounds, n_points, key):
    """Descend an acquisition with one restart-free n-path run."""
    # bounds arrives as the engine's Bounds: scipy-style pairs, or a (d, 2) array
    domain = Domain(bounds)

    def eval_g(x_batch):      # an acquisition: (m, d) -> ((m,), (m, d))
        f, g = objective(x_batch)
        return f, g, None     # finite everywhere: no feasibility to mask

    result = n_path_schedule(
        eval_g,
        domain=domain,
        n_paths=n_points,     # one path per minimizer: one batch, not n runs
        max_iters=INNER_ITERS,
        local_solver=DiagonalPreconditioner(),
        restart_stalled=False,  # a pure descent, no restarting inside it
        key=key,
    )
    return result.flock_x    # one row per path: n_points local minimizers


global_model = BLRGlobalModel(
    projection_factory=SphereProjection,  # IdentityProjection is the other prior
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
acquisition, and `surrogate_minimize` runs the engine again — flat and
restart-free this time. Four things keep it honest:

- **`restart_stalled=False` inside.** The inner run has one job: reach the
  acquisition's local minimum from each start. Restarts would replace that with
  random wandering, so they are off: a path that settles is parked for good and
  never moves again.
- **`n_paths = n_points`, not a loop.** The acquisition is batched, so one run
  of width `n_points` serves the whole cohort one row per path. Looping
  `n_paths=1` runs would buy the same points with `n_points` times the batches
  — the serial shape of a job the GPU wants wide.
- **A fit comes before the inner run.** Until `observe` has given the posterior
  a spread to standardize by, `propose` answers with the domain's own draw and
  never calls `minimize`, so the deepest layer is not entered on the first
  restarts. If the inner horizon is short enough to end before a path is
  placed, that row of `flock_x` is `nan`, and the outer run serves it as an
  unusable start and stalls that path.
- **`feas=None`.** An acquisition is finite everywhere, so the fused seam's
  third slot is `None` — there is nothing to mask out.

`INNER_ITERS` is a horizon, not a convergence test: the inner run normally ends
on `max_iters` with the acquisition already flat, which is why 40 is enough.
Raise it when the outer run stalls on starts that never moved.

`MinimizeFn` and both projections are in the [API Reference](../reference/api.md),
which is the contract to code against.
