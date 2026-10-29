"""The shipped restart-point model: i.i.d. uniform over a finite box.

One of the engine's two plugin concretes, beside `local_solver.py`
(`DiagonalPreconditioner`): it implements the engine's `GlobalModel` contract,
which stays in `engine.py` because the engine owns its seams. A restart with no
model to ask is a restart the engine has nowhere to take, and the model is also
where `x0` enters - the run's width is `n_paths`, but the points those paths
start from are the model's first proposal. Import direction is one-way - this
module reads the engine's types, and the engine binds this class lazily inside
`n_path_schedule` - so neither import is circular.
"""

import jax
import jax.numpy as jnp

from soar_lite.engine import (
    Bounds,
    GlobalModel,
    all_finite,
    box_bounds,
    check_bounds,
    finite_box,
)


class UniformGlobalModel(GlobalModel):
    """The model-free default: i.i.d. uniform over a finite box.

    Fits nothing - `observe` is a no-op - because with no surrogate the box is
    the only distribution the engine can synthesize. It is also how a caller
    hands the engine a chosen flock: `x0`, when given, is what the first
    `propose` returns, so the run's first restart lands exactly where the caller
    asked and every later one is drawn uniformly.

    The box is the one thing this model validates harder than the engine does:
    an unbounded side has no width to draw from, and a NaN side would poison
    every draw silently, so both are refused at construction.
    """

    def __init__(
        self,
        bounds: Bounds,
        *,
        key: jnp.ndarray,
        x0: jnp.ndarray | None = None,
    ) -> None:
        """Bind the box, the key and the module-free draw.

        Args:
            bounds: the finite box the draws fill, scipy-style pairs or a
                ``(d, 2)`` array.
            key: the PRNG key the draws are split from.
            x0: the first ``propose``'s answer, shape ``(n, d)`` - n the run's
                number of paths. Ignored after that; ``None`` draws the first
                points like any other.

        Raises:
            ValueError: If the box is absent, has an unbounded or NaN side, or
                if `x0` is not a finite ``(n, d)`` array whose d matches the box.

        """
        check_bounds(bounds)
        box = box_bounds(bounds)
        if box is None or not finite_box(box):
            msg = (
                "UniformGlobalModel draws uniformly, so it needs a finite box: "
                f"got {bounds!r}"
            )
            raise ValueError(msg)
        self._lo, self._hi = box
        # The box is the projection here, so it is also the dimension: a (d, 2)
        # pair list cannot disagree with itself.
        self.d = int(self._lo.shape[0])
        self._key = key
        self._x0 = None if x0 is None else _check_x0(x0, self.d)

    def observe(self, x: jnp.ndarray, f: list[float], g: jnp.ndarray | None) -> None:
        """Ignore the evidence: a uniform model fits nothing."""

    def propose(self, n_points: int) -> jnp.ndarray:
        """Return x0 on the first ask, then n_points uniform draws."""
        if self._x0 is not None:
            x_first, self._x0 = self._x0, None
            if x_first.shape[0] != n_points:
                msg = (
                    f"x0 holds {x_first.shape[0]} starts but the first restart "
                    f"asks for {n_points}"
                )
                raise ValueError(msg)
            return x_first
        self._key, sub = jax.random.split(self._key)
        return jax.random.uniform(
            sub,
            (n_points, self.d),
            minval=self._lo,
            maxval=self._hi,
        )


def _check_x0(x0: jnp.ndarray, d: int) -> jnp.ndarray:
    """Reject a flock a model cannot hand back: wrong shape, or not finite.

    A coordinate is not an objective that went non-finite, and no clip or
    proposal repairs one, so a bad start is refused here rather than left to
    stall a path on arrival.
    """
    pts = jnp.asarray(x0)
    if pts.ndim != 2 or pts.shape[1] != d:
        msg = f"x0 must be a (n_paths, {d}) array, got shape {pts.shape}"
        raise ValueError(msg)
    if pts.shape[0] < 1 or not all_finite(pts):
        msg = f"x0 needs at least one finite start, got shape {pts.shape}"
        raise ValueError(msg)
    return pts
