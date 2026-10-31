"""The model-free restart sampler: draws from an injected search domain.

One of the engine's two plugin concretes, beside `local_solver.py`
(`DiagonalPreconditioner`): it implements the engine's `GlobalModel` contract,
which stays in `engine.py` because the engine owns its seams. A restart with no
model to ask is a restart the engine has nowhere to take, and the model is also
where `x0` enters - the run's width is `n_paths`, but the points those paths
start from are the model's first proposal. Import direction is one-way - this
module reads the engine's types and the `Domain` it samples, and the engine
binds this class lazily inside `n_path_schedule` - so neither import is circular.
"""

import jax
import jax.numpy as jnp

from soar_lite.domain import Domain
from soar_lite.engine import GlobalModel, all_finite


class SampleGlobalModel(GlobalModel):
    """The model-free default: draws restart points straight from a `Domain`.

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
        domain: Domain,
        *,
        key: jnp.ndarray,
        x0: jnp.ndarray | None = None,
    ) -> None:
        """Bind the domain, the key and the module-free draw.

        Args:
            domain: the search domain the draws come from - its box, its
                per-dimension scales and its samplers. It validated all three at
                its own construction, and its ``d`` is the run's dimension.
            key: the PRNG key the draws are split from.
            x0: the first ``propose``'s answer, shape ``(n, d)`` - n the run's
                number of paths. Ignored after that; ``None`` draws the first
                points like any other.

        Raises:
            ValueError: If `x0` is not a finite ``(n, d)`` array whose d matches
                the domain.

        """
        self._domain = domain
        self.d = domain.d
        self._key = key
        self._x0 = None if x0 is None else _check_x0(x0, self.d)

    def observe(self, x: jnp.ndarray, f: list[float], g: jnp.ndarray | None) -> None:
        """Ignore the evidence: a model-free sampler fits nothing."""

    def propose(self, n_points: int) -> jnp.ndarray:
        """Return x0 on the first ask, then n_points draws from the domain."""
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
        return self._domain.draw(sub, n_points)


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
