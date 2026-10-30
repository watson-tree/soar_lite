"""Tests for the spherical-linear surrogate restart model.

The projection and the improvement kernel are the model's own math, tested
where they live. Both are `_`-private because nothing outside the package
consumes them, which is the specification here rather than an oversight.
"""

# ruff: noqa: SLF001
# pyright: reportPrivateUsage=false

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from soar_lite import Bounds, SphericalLinearModel
from soar_lite.spherical_linear import _log_h

Objective = Callable[[jnp.ndarray], jnp.ndarray]


def _grid_minimize(
    objective: Objective,
    bounds: Bounds,
    key: jnp.ndarray,
    n_points: int,
) -> jnp.ndarray:
    """Stand in for the driver's inner minimizer with the best of a grid."""
    box = jnp.asarray(bounds)
    grid_key, _ = jax.random.split(key)
    grid = jax.random.uniform(
        grid_key,
        (512, box.shape[0]),
        minval=box[:, 0],
        maxval=box[:, 1],
    )
    values = jax.vmap(objective)(grid)
    ranked = jnp.argsort(jnp.where(jnp.isfinite(values), values, jnp.inf))
    return grid[ranked[:n_points]]


def _short_minimize(
    objective: Objective,
    bounds: Bounds,
    key: jnp.ndarray,
    n_points: int,
) -> jnp.ndarray:
    """Return fewer rows than asked, so the shape check has something to reject."""
    del objective, key
    return jnp.zeros((max(n_points - 1, 0), jnp.asarray(bounds).shape[0]))


def _model(d: int = 2, half: float = 2.0, noise: float = 0.01) -> SphericalLinearModel:
    """Build a model over the symmetric ``[-half, half]^d`` box."""
    bounds = jnp.asarray([[-half, half]] * d)
    return SphericalLinearModel(
        bounds,
        minimize=_grid_minimize,
        noise=noise,
        key=jax.random.PRNGKey(0),
    )


def _observe(model: SphericalLinearModel, n: int) -> None:
    """Feed `n` deterministic box points with a smooth, off-centre minimum."""
    axis = jnp.linspace(-1.9, 1.9, n)[:, None] * jnp.asarray([1.0, -1.0])[None, :]
    f = [float(jnp.sum((row - jnp.asarray([0.3, -0.2])) ** 2)) for row in axis]
    model.observe(axis, f, None)


def _reference_log_h(z: float) -> float:
    """float64 ``log(phi(z) + z*Phi(z))`` from erfc, with no library shortcut."""
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    tail = 0.5 * math.erfc(-z / math.sqrt(2.0))
    return math.log(phi + z * tail)


@pytest.mark.parametrize("z", [-8.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 2.0, 8.0])
def test_log_h_matches_its_definition_where_it_can_be_written_down(z: float) -> None:
    """Compare the kernel against a float64 erfc evaluation of its definition."""
    assert float(_log_h(jnp.asarray(z))) == pytest.approx(
        _reference_log_h(z),
        abs=2e-3,
    )


@pytest.mark.parametrize("z", [-20.0, -30.0])
def test_log_h_far_branch_tracks_the_asymptote_not_the_cancellation(z: float) -> None:
    """Follow the Mills asymptote where the direct difference cancels.

    The far branch drops the ``1 - 3/z^2`` correction, under one percent there.
    """
    assert float(_log_h(jnp.asarray(z))) == pytest.approx(
        _reference_log_h(z),
        abs=2e-2,
    )


def test_log_h_is_continuous_across_the_mills_seam() -> None:
    """Keep the segmented branches joined where the asymptote takes over."""
    near_side = float(_log_h(jnp.asarray(-9.999)))
    far_side = float(_log_h(jnp.asarray(-10.001)))
    assert near_side == pytest.approx(far_side, abs=5e-2)


def test_features_lie_on_the_unit_sphere() -> None:
    """Bind the posterior variance through unit-norm feature rows."""
    xs = jnp.asarray([[-1.9, 1.9], [0.0, 0.0], [1.3, -0.4]])
    norms = jnp.linalg.norm(_model()._features(xs), axis=-1)
    assert np.allclose(np.asarray(norms), 1.0, atol=1e-5)


def test_propose_draws_uniformly_until_a_fit_is_possible() -> None:
    """Fall back to the box draw while a standardization is impossible."""
    model = _model()
    for n_observed in (0, 1):
        if n_observed:
            model.observe(jnp.asarray([[0.5, -0.5]]), [1.0], None)
        draws = model.propose(4)
        assert draws.shape == (4, 2)
        assert bool(jnp.all(jnp.isfinite(draws)))
        assert bool(jnp.all(draws >= -2.0))
        assert bool(jnp.all(draws <= 2.0))


def test_integer_bounds_still_normalize_to_a_unit_box() -> None:
    """Keep the box usable when a caller spells its pairs in integers."""
    model = SphericalLinearModel(
        [(0, 1), (0, 1)],
        minimize=_grid_minimize,
        key=jax.random.PRNGKey(0),
    )
    draws = model.propose(3)
    assert bool(jnp.all(draws >= 0.0))
    assert bool(jnp.all(draws <= 1.0))
    model.observe(jnp.asarray([[0.2, 0.3], [0.8, 0.9]]), [1.0, 2.0], None)
    mean, sigma = model.predict(jnp.asarray([[0.5, 0.5]]))
    assert bool(jnp.all(jnp.isfinite(mean)))
    assert bool(jnp.all(jnp.isfinite(sigma)))


def test_predict_refuses_a_fit_it_cannot_standardize() -> None:
    """Refuse a posterior query that has no value spread to work from."""
    model = _model()
    model.observe(jnp.asarray([[0.5, -0.5]]), [1.0], None)
    with pytest.raises(ValueError, match="spread to standardize"):
        model.predict(jnp.asarray([[0.0, 0.0]]))


@pytest.mark.parametrize("bad", [(3, 2, 2), (3, 3)])
def test_observe_and_predict_demand_a_two_dimensional_batch(
    bad: tuple[int, ...],
) -> None:
    """Refuse a shape the fit would broadcast into nonsense."""
    model = _model()
    _observe(model, 4)
    with pytest.raises(ValueError, match="needs a"):
        model.observe(jnp.zeros(bad), [0.0] * bad[0], None)
    with pytest.raises(ValueError, match="needs a"):
        model.predict(jnp.zeros(bad))


def test_posterior_variance_stays_under_the_prior_far_from_data() -> None:
    """Bound the far-from-data variance, which is the whole point of the projection."""
    # The structural cure: sigma^2 = noise * (1 + p^T A^-1 p) with |p| = 1 and
    # A >= (noise/var_prior) I, so sigma^2 <= noise + var_prior everywhere.
    model = _model()
    _observe(model, 6)
    near, near_sigma = model.predict(jnp.asarray([[1.9, -1.9]]))
    far, far_sigma = model.predict(jnp.asarray([[1.9, 1.9]]))
    assert bool(jnp.all(jnp.isfinite(near)))
    assert bool(jnp.all(jnp.isfinite(far)))
    assert float(far_sigma[0]) <= math.sqrt(1.01) + 1e-3  # noise + var_prior
    assert float(far_sigma[0]) > float(near_sigma[0])


def test_propose_descends_the_acquisition_into_the_box() -> None:
    """Return the minimizer's starts, finite and inside the box."""
    model = _model()
    _observe(model, 12)
    starts = model.propose(3)
    assert starts.shape == (3, 2)
    assert bool(jnp.all(jnp.isfinite(starts)))
    assert bool(jnp.all(starts >= -2.0))
    assert bool(jnp.all(starts <= 2.0))
    mean, sigma = model.predict(starts)
    assert bool(jnp.all(jnp.isfinite(mean)))
    assert bool(jnp.all(jnp.isfinite(sigma)))


def test_observe_drops_non_finite_values_and_checks_its_rows() -> None:
    """Drop a value the fit cannot use and refuse a ragged batch."""
    model = _model()
    model.observe(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]), [1.0, -math.inf], None)
    assert len(model._fs) == 1
    with pytest.raises(ValueError, match="rows but"):
        model.observe(jnp.asarray([[0.0, 0.0]]), [1.0, 2.0], None)


def test_minimize_output_shape_is_validated() -> None:
    """Refuse an injected minimizer that answers the wrong shape."""
    bounds = jnp.asarray([[-2.0, 2.0]] * 2)
    model = SphericalLinearModel(
        bounds,
        minimize=_short_minimize,
        key=jax.random.PRNGKey(1),
    )
    _observe(model, 4)
    with pytest.raises(ValueError, match=r"\(3, 2\) array"):
        model.propose(3)


def test_a_bad_box_or_knob_or_minimizer_is_refused() -> None:
    """Refuse an unbounded box, a non-callable minimizer and a zero noise."""
    unbounded = jnp.asarray([[0.0, math.inf]])
    with pytest.raises(ValueError, match="finite box"):
        SphericalLinearModel(
            unbounded,
            minimize=_grid_minimize,
            key=jax.random.PRNGKey(0),
        )
    finite = jnp.asarray([[-1.0, 1.0]])
    with pytest.raises(TypeError, match="callable"):
        SphericalLinearModel(
            finite,
            minimize=None,  # pyright: ignore[reportArgumentType]
            key=jax.random.PRNGKey(0),
        )
    with pytest.raises(ValueError, match="noise"):
        _model(noise=0.0)
