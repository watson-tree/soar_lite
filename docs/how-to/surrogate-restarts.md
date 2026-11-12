# Restart from a surrogate

This page continues [the tutorial](../tutorial/getting-started.md)'s example:
`eval_f_g` and `domain` below are the ones built there.

## The pieces

The job is one function: it takes the acquisition `BLRGlobalModel` builds — its
score for how promising a point looks, here `-log EI` — and returns the points to
restart from: the local minima of that score. The obvious way to descend it is to
hand the engine back to itself: one restart-free run (that is,
`restart_stalled=False`) whose width is the number of minimizers you want, read
off `flock_x`.

`BLRGlobalModel` fits its posterior and builds its own `-log EI` acquisition,
but it never descends that acquisition itself: how to descend it is the engine's
business, so the model receives it as an injected `minimize` — the parameter's
name, typed `MinimizeFn`, with signature
`(acquisition, domain, n_points, key) -> (n_points, d)`. The domain is the one
the outer run's `propose` call carried, not a box re-derived from it, so the
`scale` and samplers you declared reach the inner run's own draws. The acquisition is
`(M, d) -> ((M,), (M, d))` (value and gradient, no feasibility column) and `key`
is a fresh subkey of the model's own key, split for this call alone.

The outer engine is the model's only caller: it hands it every batch it serves
(`observe`) and asks it for points when a path restarts (`propose`), and the
model only ever answers. What it answers with, and when, is the rest of this page.

The projection is the other knob — it supplies the prior the model fits.
`SphereProjection` keeps the posterior variance bounded far from data;
`IdentityProjection`, also in `soar_lite`, passes the raw coordinates through,
so its variance grows with distance toward the box corners. Default to the first
unless you have a reason not to. `projection_factory` takes the class, not an
instance: the model calls it with the `domain`, once per `propose`.

```python
# eval_f_g and domain come from the tutorial's §1.2
import jax
import jax.numpy as jnp
from soar_lite import (
    BLRGlobalModel,
    SphereProjection,
    n_path_schedule,
)

INNER_ITERS = 40  # long enough to settle; shorter than the outer's horizon


def surrogate_minimize(acquisition, domain, n_points, key):
    """Descend an acquisition with one restart-free n-path run."""

    # the acquisition is two slots; the seam wants three, the last one None
    def fused_seam(x_batch):
        f, g = acquisition(x_batch)
        return f, g, None  # finite everywhere: no feasibility to mask

    result = n_path_schedule(
        fused_seam,
        domain=domain,
        n_paths=n_points,  # one path per minimizer: one batch, not n runs
        max_iters=INNER_ITERS,
        restart_stalled=False,  # a pure descent, no restarting inside it
        key=key,
    )
    # one row per path, and every row has to be a finite point inside the box: a
    # path the inner run could not place carries an all-nan point, which the outer
    # run refuses rather than stalls on, so it falls back to a fresh draw
    placed = result.flock_x
    ok = jnp.all(jnp.isfinite(placed), axis=1)[:, None]
    return jnp.where(ok, placed, domain.draw(key, n_points))


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

## Five things to get right

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
- **The outer run's first ask draws, not descends.** With no `x0`, iteration 0
  asks for the flock's placement and `propose` has nothing to fit, so it answers
  with the domain's own draw, never calling `minimize`. Any later ask also draws
  until `observe` has given the posterior a spread to standardize by.
- **Every returned row must be a finite point inside the box.** An inner path
  can end holding no point at all - the acquisition was not finite at its
  start, and every row of a run that does zero iterations (`max_iters=0`) is
  one - and an unplaced path carries an all-`nan` point. The outer run refuses
  such a row instead of stalling on it, so the driver fills it: here the
  domain's own draw does, in one `jnp.where`.
- **The acquisition has no feasibility mask (`feas=None`).** It is finite
  everywhere, so the fused seam's third slot is `None`.

## INNER_ITERS is a cap, not a test

`INNER_ITERS` is an upper bound, not a convergence test. A path that settles
stops there, so an inner run whose flock settles early ends on `all_stalled`
before reaching it; 40 is a cap with room to spare, not an iteration count to aim
at. Raise it if the outer run keeps restarting paths that barely moved.

The inner run bills its own `Work`, and none of it reaches the outer
`result.work`: the outer `budget_check` is handed the outer ledger alone, so a
budget that has to cover the surrogate descents must price them itself.

`MinimizeFn` and both projections are in the [API Reference](../reference/api.md),
with their exact signatures and defaults.
