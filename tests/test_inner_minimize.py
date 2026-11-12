"""The inner minimizer answers by path, not by row.

`BLRGlobalModel.propose` names the paths it wants starts for and carries a
``None`` answer out as its own decline, so `descent_minimize` has to reply in
that shape: one `Point` per named path, keyed by the path, and no coordinate
invented for a path it could not place.
"""

import jax
import jax.numpy as jnp
import pytest

from soar_lite import BLRGlobalModel, Domain, Point, ValuePoint, descent_minimize


def _bowl(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Define a smooth quadratic whose minimum sits inside the unit box."""
    d = x - 0.25
    return jnp.sum(d**2, axis=1), 2.0 * d


def test_the_inner_minimizer_answers_one_point_per_named_path() -> None:
    """A run inside a run speaks the same shape as the model that asked."""
    box = Domain(jnp.asarray([[0.0, 1.0], [0.0, 1.0]]))

    answers = descent_minimize(_bowl, box, [7, 3], jax.random.PRNGKey(0))

    assert sorted(answers) == [3, 7], answers
    for point in answers.values():
        assert isinstance(point, Point)
        assert point.x.shape == (2,)
        assert bool(jnp.all(jnp.isfinite(point.x)))


def test_a_path_the_inner_run_cannot_place_is_declined() -> None:
    """Zero iterations places nothing, and that is a decline, not a coordinate."""
    box = Domain(jnp.asarray([[0.0, 1.0], [0.0, 1.0]]))

    answers = descent_minimize(
        _bowl,
        box,
        [0, 1],
        jax.random.PRNGKey(0),
        inner_iters=0,
    )

    assert answers == {0: None, 1: None}


def test_an_empty_ask_is_refused() -> None:
    """The ask is the paths; naming none is a caller error, not an empty answer."""
    box = Domain(jnp.asarray([[0.0, 1.0], [0.0, 1.0]]))

    with pytest.raises(ValueError, match="at least one path"):
        descent_minimize(_bowl, box, [], jax.random.PRNGKey(0))


def test_the_model_runs_on_the_shipped_minimizer_by_default() -> None:
    """No `minimize=` argument: the default binds and `propose` answers points."""
    box = Domain(jnp.asarray([[-2.0, 2.0]] * 2))
    rows = jnp.asarray([[0.3, -0.2], [0.4, -0.1], [0.2, -0.3]])
    model = BLRGlobalModel(key=jax.random.PRNGKey(0))
    model.observe(
        {i: ValuePoint(row, float(jnp.sum(row**2))) for i, row in enumerate(rows)}
    )

    answers = model.propose([0], box)

    assert sorted(answers) == [0]
    point = answers[0]
    assert isinstance(point, Point)
    assert point.x.shape == (2,)
