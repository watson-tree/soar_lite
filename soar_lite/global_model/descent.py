"""The package's own `MinimizeFn`: descend an acquisition with the engine.

`BLRGlobalModel.propose` hands its acquisition - ``-log EI`` and its gradient -
to an injected `MinimizeFn` and never picks how to search itself. This module
holds the answer the package ships: `descent_minimize`, the engine used as its
own inner minimizer.
"""

from collections.abc import Sequence

import jax.numpy as jnp

from soar_lite.domain import Domain
from soar_lite.engine import Point, n_path_schedule
from soar_lite.global_model.blr import AcquisitionFn

__all__ = ["descent_minimize"]


def descent_minimize(
    acquisition: AcquisitionFn,
    domain: Domain,
    paths: Sequence[int],
    key: jnp.ndarray,
    *,
    inner_iters: int = 40,
) -> dict[int, Point | None]:
    """Descend an acquisition with one restart-free n-path run.

    The engine inside the engine, in one direction only: the outer model hands
    over its acquisition, this driver runs the library's own descent and answers
    one minimizer per named path.

    The inner run is a pure descent, not a restarting search:

    - `restart_stalled=False` - each path has one job, reach a local minimum
      from its draw; replacing it with a fresh draw is the outer run's business.
    - `n_paths=len(paths)` - the acquisition is batched, so one run of width
      `len(paths)` serves the whole cohort one row per path per batch, instead
      of `len(paths)` serial runs buying the same points with `len(paths)` times
      the batches.
    - no feasibility column - the acquisition is finite everywhere, so the fused
      seam carries `None` as its third slot.

    The inner run draws its starts from `domain` - the object the ask was
    handed, so its `scale` and samplers reach the inner draws - and bills its
    own `Work`, none of which is added to the outer run's ledger.

    A minimizer it could not place is answered `None`, never a coordinate: an
    inner path whose acquisition is not finite at its draw holds no point and
    its `flock_x` row is all-`nan`, so `None` is how the outer run is told to
    leave that path where it stands rather than restart it from a point the
    acquisition never ranked.

    Args:
        acquisition: the acquisition to descend, `(M, d) -> ((M,), (M, d))`:
            values and per-row gradients. Lower is better, the sign riding in
            the value.
        domain: the search box, as the ask declared it - the same object the
            outer run checks its answers against, `scale` and samplers
            included.
        paths: the paths to answer for, one local minimizer each.
        key: the PRNG key the inner run's starts are drawn from. The model hands
            a fresh subkey split for this ask alone.
        inner_iters: the inner run's horizon - an upper bound, not a convergence
            test; a path that settles stops earlier, so a run that settles early
            ends on `all_stalled` before reaching it. Default 40.

    Returns:
        One `Point` inside `domain` per named path, or `None` for a path the
        inner run could not place.

    Raises:
        ValueError: If `paths` is empty.

    """
    if not paths:
        msg = "descent_minimize needs at least one path, got none"
        raise ValueError(msg)

    def fused_seam(x_batch: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        f, g = acquisition(x_batch)
        return f, g, None

    result = n_path_schedule(
        fused_seam,
        n_paths=len(paths),
        domain=domain,
        max_iters=inner_iters,
        restart_stalled=False,
        key=key,
    )
    placed = result.flock_x
    usable = jnp.all(jnp.isfinite(placed), axis=1)
    return {
        i: (Point(row) if bool(ok) else None)
        for i, row, ok in zip(paths, placed, usable, strict=True)
    }
