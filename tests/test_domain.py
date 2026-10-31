"""The search domain on its own: the box, the scales and the draws.

Nothing here needs a surrogate or a geometry. These live apart from
`test_spherical_linear.py` because a projection *reads* a domain rather than
being one, so the tests that pin the domain's own contract should not have to
build a projection to ask.
"""

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import pytest

from soar_lite import Domain


def test_a_domain_needs_a_declared_box() -> None:
    """Refuse a domain with nothing to read its dimension off."""
    with pytest.raises(ValueError, match="needs one"):
        Domain(None)  # pyright: ignore[reportArgumentType]


def test_a_malformed_box_is_refused_at_construction() -> None:
    """A box that is not a box fails here, not inside a clip.

    Both spellings are checked for the same three defects: an inverted dim, a
    NaN side (which would silently poison every clip instead of failing), and an
    array whose second axis is not the pair. The domain is built before a run
    sees it, so this is the one gate every spelling passes.
    """

    def rejects(bounds: object) -> str:
        try:
            Domain(bounds)  # pyright: ignore[reportArgumentType]
        except ValueError as e:
            return str(e)
        msg = f"accepted: {bounds!r}"
        raise AssertionError(msg)

    assert "lo=1.0 > hi=0.0" in rejects([(1.0, 0.0)])
    assert "NaN" in rejects([(0.0, float("nan"))])
    assert "(d, 2)" in rejects(jnp.zeros((3, 3)))


@pytest.mark.parametrize(
    "scale",
    [
        jnp.asarray(1.0),
        jnp.asarray([1.0]),
        jnp.ones((4, 1)),
        jnp.ones(3),
        jnp.asarray([1.0, 0.0, 1.0, 1.0]),
        jnp.asarray([1.0, -1.0, 1.0, 1.0]),
        jnp.asarray([1.0, jnp.inf, 1.0, 1.0]),
    ],
)
def test_a_scale_that_is_not_a_d_vector_is_refused(scale: jnp.ndarray) -> None:
    """Refuse a scalar, a short vector, a matrix and a non-positive entry.

    A scalar or a length-one vector would broadcast into a uniform vector, which
    is a real scale but never what a caller writing those shapes means, so it is
    refused rather than guessed at.
    """
    bounds = jnp.asarray([[-2.0, 2.0]] * 4)
    with pytest.raises(ValueError, match="scale"):
        Domain(bounds, scale=scale)


def test_the_default_scale_is_one_not_the_box_width() -> None:
    """Measure the prior in coordinate units, so no width is taken for a unit."""
    domain = Domain(jnp.asarray([[-2.0, 2.0], [-10.0, 10.0]]))
    assert jnp.array_equal(domain.scale, jnp.ones(2))
    assert jnp.array_equal(domain.center, jnp.zeros(2))
    assert jnp.array_equal(domain.width, jnp.asarray([4.0, 20.0]))


def test_a_mixed_box_draws_the_unbounded_dimension_at_its_own_scale() -> None:
    """Draw the unbounded column from ``N(0, scale^2)``, not from a zero width."""
    bounds = jnp.asarray([[0.0, 1.0], [-math.inf, math.inf]])
    domain = Domain(bounds, scale=jnp.asarray([1.0, 3.0]))
    draws = domain.draw(jax.random.PRNGKey(0), 256)
    assert draws.shape == (256, 2)
    assert bool(jnp.all(jnp.isfinite(draws)))
    assert bool(jnp.all(draws[:, 0] >= 0.0))
    assert bool(jnp.all(draws[:, 0] <= 1.0))
    assert float(jnp.std(draws[:, 1])) == pytest.approx(3.0, rel=0.3)


@pytest.mark.parametrize(
    "side",
    [[[0.0, math.inf]], [[-math.inf, 0.0]], [[5.0, math.inf]]],
)
def test_a_half_bounded_dimension_draws_inside_its_one_side(
    side: list[list[float]],
) -> None:
    """Truncate the default normal to the domain instead of stepping over it."""
    box = jnp.asarray(side)
    lower, upper = float(box[0, 0]), float(box[0, 1])
    draws = Domain(box).draw(jax.random.PRNGKey(0), 512)[:, 0]
    assert bool(jnp.all(jnp.isfinite(draws)))
    assert bool(jnp.all(draws >= lower))
    assert bool(jnp.all(draws <= upper))


def test_a_half_bounded_draw_is_the_half_normal_not_the_full_one() -> None:
    """Anchor the truncation at the declared side: mean ``scale*sqrt(2/pi)``."""
    draws = Domain(
        jnp.asarray([[0.0, math.inf]]),
        scale=jnp.asarray([2.0]),
    ).draw(jax.random.PRNGKey(0), 2048)[:, 0]
    assert float(jnp.mean(draws)) == pytest.approx(
        2.0 * math.sqrt(2.0 / math.pi),
        rel=0.1,
    )


def test_a_domain_truncated_past_float_resolution_lands_on_its_boundary() -> None:
    """Collapse onto the side rather than draw past it, or draw an infinity."""
    draws = Domain(
        jnp.asarray([[10.0, math.inf]]),
        scale=jnp.asarray([1.0]),
    ).draw(jax.random.PRNGKey(0), 64)[:, 0]
    assert bool(jnp.all(jnp.isfinite(draws)))
    assert bool(jnp.all(draws == 10.0))


def test_an_injected_sampler_replaces_that_dimensions_default() -> None:
    """Take one column's starts from the caller's own distribution."""
    bounds = jnp.asarray([[-2.0, 2.0], [-math.inf, math.inf]])

    def constant(key: jnp.ndarray, n_points: int) -> jnp.ndarray:
        del key
        return jnp.full((n_points,), 7.0)

    draws = Domain(bounds, samplers=[constant, None]).draw(
        jax.random.PRNGKey(0),
        5,
    )
    assert bool(jnp.all(draws[:, 0] == 7.0))
    assert bool(jnp.all(jnp.isfinite(draws[:, 1])))


@pytest.mark.parametrize("samplers", [[], [None], [None, None, None]])
def test_a_sampler_list_that_is_not_one_per_dimension_is_refused(
    samplers: Sequence[object],
) -> None:
    """Refuse a short or long list rather than leave the tail on its default."""
    bounds = jnp.asarray([[-2.0, 2.0]] * 2)
    with pytest.raises(ValueError, match="samplers"):
        # The point of the case is a list the annotation says is the wrong length.
        Domain(bounds, samplers=samplers)  # pyright: ignore[reportArgumentType]


def test_a_sampler_that_returns_the_wrong_shape_is_refused() -> None:
    """Refuse a column whose shape would silently widen the stacked draw."""
    bounds = jnp.asarray([[-2.0, 2.0]] * 2)

    def flat(key: jnp.ndarray, n_points: int) -> jnp.ndarray:
        del key
        return jnp.zeros(n_points * 2)

    domain = Domain(bounds, samplers=[flat, None])
    with pytest.raises(ValueError, match="drawn column"):
        domain.draw(jax.random.PRNGKey(0), 3)
