"""The model-free restart sampler: draws from the box each ask hands it.

One of the engine's two plugin concretes, beside `local_solver.py`
(`DiagonalPreconditioner`): it implements the engine's `GlobalModel` contract,
which stays in `engine.py` because the engine owns its seams. It carries no box
of its own: the run's one `Domain` is handed to `propose` at every ask, so this
class holds nothing but the key and the `x0` it owes the first call. A restart
with no model to ask is a restart the engine has nowhere to take, and the model
is also where `x0` enters - the run's width is `n_paths`, but the points those
paths start from are the model's first proposal. Import direction is one-way -
this module reads the engine's types and the `Domain` it samples, and the engine
binds this class lazily inside `n_path_schedule` - so neither import is circular.
"""

import jax
import jax.numpy as jnp

from soar_lite.domain import Domain
from soar_lite.engine import GlobalModel, all_finite


class SampleGlobalModel(GlobalModel):
    """The model-free default: draws restart points straight from the `Domain`.

    Fits nothing - `observe` is a no-op - because with no surrogate the domain's
    own draw is the only distribution the run can synthesize. Whatever the
    domain admits, this draws: uniform inside a bounded side, ``N(0, scale^2)``
    inside a free one, that same normal truncated inside a half-bounded one, and
    an injected per-dimension sampler where the caller said so. That is what a
    plain finite-box uniform model could do and more - the extension is the
    domain's, and this class only hands it the key.

    It is also how a caller hands the engine a chosen flock: `x0`, when given,
    is what the first `propose` returns, so the run's first restart lands
    exactly where the caller asked and every later one is a draw.
    """

    def __init__(
        self,
        *,
        key: jnp.ndarray,
        x0: jnp.ndarray | None = None,
    ) -> None:
        """Bind the key and the flock this model owes the first ask.

        Args:
            key: the PRNG key the draws are split from.
            x0: the first ``propose``'s answer, shape ``(n, d)`` - n the run's
                number of paths. Ignored after that; ``None`` draws the first
                points like any other. It is checked against the run's domain
                on that first ask - the moment this model is first told both
                the dimension and the box.

        """
        self._key = key
        self._x0 = x0

    def observe(self, x: jnp.ndarray, f: list[float], g: jnp.ndarray | None) -> None:
        """Ignore the evidence: a model-free sampler fits nothing."""

    def propose(self, n_points: int, domain: Domain) -> jnp.ndarray:
        """Return x0 on the first ask, then n_points draws from `domain`.

        Raises:
            ValueError: If `x0`, on the first ask, is not a finite ``(n, d)``
                array whose d is the handed domain's, if it holds a different
                number of starts than this ask wants, or if a start lies
                outside the box.

        """
        if self._x0 is not None:
            x_first = _check_x0(self._x0, domain)
            self._x0 = None
            if x_first.shape[0] != n_points:
                msg = (
                    f"x0 holds {x_first.shape[0]} starts but the first restart "
                    f"asks for {n_points}"
                )
                raise ValueError(msg)
            return x_first
        self._key, sub = jax.random.split(self._key)
        return domain.draw(sub, n_points)


def _check_x0(x0: jnp.ndarray, domain: Domain) -> jnp.ndarray:
    """Reject a flock a model cannot hand back: bad shape, not finite, outside.

    Called on the first ask, the first moment this model is told the run's
    `Domain` - before that it knows neither the dimension nor the box. Three
    refusals, each a defect no clip or proposal repairs:

    - not a two-dimensional batch, or the wrong width;
    - no starts, or a start that is not all finite (a coordinate is not an
      objective that went non-finite);
    - a start outside the box.

    The third is the one that is not interchangeable with the engine's clip. An
    out-of-box start is the caller saying "search this box" and "begin there" at
    once; clipping it would silently slide the start onto a wall - a degenerate
    point - and hide the contradiction, so it is refused and named instead. The
    clip keeps its own job: proposals the model *generates* may overstep, and
    clamping those is the defensive seam. Generated proposals and an authored
    start are different claims, so they get different treatment.

    The box test carries a machine-precision tolerance: a start computed at a
    boundary rounds by a few representable steps, so a coordinate within
    ``16 * eps * max(1, |bound|)`` of a finite side counts as on it. Anything
    further out is a caller error, not roundoff. Free dimensions never trigger
    it - an infinite side is compared against itself.
    """
    pts = jnp.asarray(x0)
    d = domain.d
    if pts.ndim != 2 or pts.shape[1] != d:
        msg = f"x0 must be a (n_paths, {d}) array, got shape {pts.shape}"
        raise ValueError(msg)
    if pts.shape[0] < 1 or not all_finite(pts):
        msg = f"x0 needs at least one finite start, got shape {pts.shape}"
        raise ValueError(msg)
    lo, hi = domain.lo, domain.hi
    eps = jnp.finfo(jnp.result_type(pts.dtype, lo.dtype, jnp.float32)).eps
    # The tolerance's own scale comes from the finite sides only: taking it from
    # an infinite side would carry it to inf and disarm the other, finite one.
    span = jnp.maximum(
        jnp.where(jnp.isfinite(lo), jnp.abs(lo), 0.0),
        jnp.where(jnp.isfinite(hi), jnp.abs(hi), 0.0),
    )
    tol = 16.0 * eps * jnp.maximum(1.0, span)
    outside = (pts < lo - tol) | (pts > hi + tol)
    if bool(jnp.any(outside)):
        row, dim = (int(v) for v in jnp.argwhere(outside)[0])
        msg = (
            f"x0 row {row} is {float(pts[row, dim]):.6g} in dimension {dim}, "
            f"outside the box [{float(lo[dim]):.6g}, {float(hi[dim]):.6g}]"
        )
        raise ValueError(msg)
    return pts
