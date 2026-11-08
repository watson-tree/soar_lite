"""The box, the flock, and the entry kwargs that plant it.

Every engine test states the same three things before it can run one iteration:
the width `n_paths`, the starts `x0`, and the one box the run searches. This
module spells them once so a suite that only wants "four paths, interior, in the
unit box" says so in one call instead of restating the shapes.
"""

from typing import Any

import jax
import jax.numpy as jnp

from soar_lite.domain import Domain


def unit_bounds(d: int = 1) -> jnp.ndarray:
    """Return the engine's box in working coords: x ∈ [0,1]^d."""
    return jnp.stack([jnp.zeros(d), jnp.ones(d)], axis=1)


def unit_domain(d: int = 1) -> Domain:
    """Return the engine's box in working coords as the `Domain` it is."""
    return Domain(unit_bounds(d))


def flock(n_paths: int, d: int = 1) -> jnp.ndarray:
    """Return `n_paths` deterministic starting points inside [0,1]^d.

    Spread evenly and off the walls rather than drawn: these are the engine's
    tests, so the starts must not depend on a PRNG the caller now owns, and an
    interior row keeps a steepest-descent step from being pinned at a face
    before the test has said anything.
    """
    row = jnp.linspace(0.0, 1.0, n_paths + 2)[1:-1]
    return jnp.repeat(row[:, None], d, axis=1)


def flock_kwargs(
    x0: jnp.ndarray,
    domain: Domain,
    *,
    key: jnp.ndarray | None = None,
) -> dict[str, Any]:
    """Return the entry's flock kwargs: width, starts and the box they live in.

    The engine takes its width as `n_paths`, every point a path starts from as
    the entry's `x0`, and the one box the run searches as `domain` - so a test
    that chooses its flock says all three here, and the box it names is the box
    the run clips into. `key` seeds the default model's restarts, so it is
    passed on as the entry's own knob rather than through a model.
    """
    return {
        "n_paths": int(x0.shape[0]),
        "domain": domain,
        "x0": x0,
        "key": jax.random.PRNGKey(0) if key is None else key,
    }
