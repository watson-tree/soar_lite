"""The model-free restart sampler: draws from the box each ask hands it.

One of the engine's two `GlobalModel` concretes, beside `BLRGlobalModel`, and
the one `n_path_schedule` builds when `global_model=None`: it implements the
engine's `GlobalModel` contract, which stays in `engine/contracts.py` because
the engine owns its evaluation seams. It carries no box of its own: the run's
one `Domain` is handed to `propose` at every ask, so this class holds nothing
but the key. A restart with no model to
ask is a restart the engine has nowhere to take. A caller's chosen flock is not
this class's business - it hands its starts to `n_path_schedule` as `x0`, and
the engine places them without asking any model. With `x0` left `None`, iteration
0's flock is this class's business: the engine asks it for that placement, like
any other restart.
Import direction is one-way - this module reads the engine's types and the
`Domain` it samples, and the engine binds this class lazily inside
`n_path_schedule` - so neither import is circular.
"""

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp

from soar_lite.domain import Domain
from soar_lite.engine import GlobalModel, ValuePoint


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

    def observe(self, served: Mapping[int, ValuePoint]) -> None:
        """Ignore the evidence: a model-free sampler fits nothing."""

    def propose(
        self, paths: Sequence[int], domain: Domain
    ) -> dict[int, jnp.ndarray | None]:
        """Draw one point per path, in the order the paths were asked."""
        if not paths:
            return {}
        self._key, sub = jax.random.split(self._key)
        return dict(zip(paths, domain.draw(sub, len(paths)), strict=True))
