"""The model-free restart sampler: draws from the box each ask hands it.

One of the engine's two plugin concretes, beside `local_solver.py`
(`DiagonalPreconditioner`): it implements the engine's `GlobalModel` contract,
which stays in `engine.py` because the engine owns its seams. It carries no box
of its own: the run's one `Domain` is handed to `propose` at every ask, so this
class holds nothing but the key. A restart with no model to ask is a restart the
engine has nowhere to take; the run's flock is not this class's business - a
caller that chooses its starts hands them to `n_path_schedule` as `x0`, and the
engine places them without asking any model. Import direction is one-way - this
module reads the engine's types and the `Domain` it samples, and the engine
binds this class lazily inside `n_path_schedule` - so neither import is circular.
"""

import jax
import jax.numpy as jnp

from soar_lite.domain import Domain
from soar_lite.engine import GlobalModel


class SampleGlobalModel(GlobalModel):
    """The model-free default: draws restart points straight from the `Domain`.

    Fits nothing - `observe` is a no-op - because with no surrogate the domain's
    own draw is the only distribution the run can synthesize. Whatever the domain
    admits, this draws: uniform inside a bounded side, ``N(0, scale^2)`` inside a
    free one, that same normal truncated inside a half-bounded one, and an
    injected per-dimension sampler where the caller said so.
    """

    def __init__(self, *, key: jnp.ndarray) -> None:
        """Bind the key the draws are split from.

        Args:
            key: the PRNG key the draws are split from.

        """
        self._key = key

    def observe(self, x: jnp.ndarray, f: list[float], g: jnp.ndarray | None) -> None:
        """Ignore the evidence: a model-free sampler fits nothing."""

    def propose(self, n_points: int, domain: Domain) -> jnp.ndarray:
        """Return n_points draws from `domain`, one point per requested start."""
        self._key, sub = jax.random.split(self._key)
        return domain.draw(sub, n_points)
