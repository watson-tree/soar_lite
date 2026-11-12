"""The inner minimizer answers by path, not by row.

`BLRGlobalModel.propose` names the paths it wants starts for and carries a
``None`` answer out as its own decline, so the driver's inner run has to reply in
that shape: one point per named path, keyed by the path, and no coordinate
invented for a path it could not place.

Needs the ``exp`` dependency group: the harness imports matplotlib and optax.
"""

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("optax")

from experiments.harness.methods import surrogate_minimize
from soar_lite import Domain


def test_the_inner_minimizer_answers_one_point_per_named_path() -> None:
    """A run inside a run speaks the same shape as the model that asked."""
    box = Domain(jnp.asarray([[0.0, 1.0], [0.0, 1.0]]))

    def bowl(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        d = x - 0.25
        return jnp.sum(d**2, axis=1), 2.0 * d

    answers = surrogate_minimize(bowl, box, [7, 3], jax.random.PRNGKey(0))

    assert sorted(answers) == [3, 7], answers
    for point in answers.values():
        assert point is not None
        assert point.shape == (2,)
        assert bool(jnp.all(jnp.isfinite(point)))
