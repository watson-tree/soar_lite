"""Tests for the spherical-linear surrogate restart model.

The projection and the improvement kernel are the model's own math, tested
where they live. Both are `_`-private because nothing outside the package
consumes them, which is the specification here rather than an oversight.
"""

# ruff: noqa: SLF001
# pyright: reportPrivateUsage=false

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from soar_lite import AcquisitionFn, Bounds, SphericalLinearModel
from soar_lite.spherical_linear import SphereProjection, _log_h


def _grid_minimize(
    objective: AcquisitionFn,
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
    values, _ = objective(grid)
    ranked = jnp.argsort(jnp.where(jnp.isfinite(values), values, jnp.inf))
    return grid[ranked[:n_points]]


def _short_minimize(
    objective: AcquisitionFn,
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
    norms = jnp.linalg.norm(_model()._projection.features(xs), axis=-1)
    assert np.allclose(np.asarray(norms), 1.0, atol=1e-5)


def test_features_stay_d_plus_one_wide_whatever_the_mask_bends() -> None:
    """Keep the ridge width fixed: a bent dimension donates one column."""
    bounds = jnp.asarray([[-2.0, 2.0]] * 4)
    xs = jnp.linspace(-1.9, 1.9, 6)[:, None] * jnp.ones(4)[None, :]
    for mask in (None, [0], [0, 2], [True, False, True, False]):
        projection = SphereProjection(bounds, mask=mask)
        assert projection.features(xs).shape == (6, 5)


def test_dimensions_left_out_keep_their_box_coordinate() -> None:
    """Read unmasked dimensions as the plain unit coordinate, with no bowl."""
    bounds = jnp.asarray([[-2.0, 2.0], [0.0, 10.0]])
    projection = SphereProjection(bounds, mask=[0])
    xs = jnp.asarray([[-1.0, 2.5], [1.0, 7.5]])
    unit = (xs - projection.lo) / projection.width
    assert jnp.array_equal(projection.features(xs)[:, 2:], unit[:, 1:])


def test_the_bent_block_alone_still_lies_on_the_unit_sphere() -> None:
    """Keep the variance bound on the bent subset, not the whole row."""
    bounds = jnp.asarray([[-2.0, 2.0]] * 4)
    xs = jnp.linspace(-1.9, 1.9, 6)[:, None] * jnp.ones(4)[None, :]
    bent = SphereProjection(bounds, mask=[0, 1]).features(xs)[:, :3]
    assert np.allclose(np.asarray(jnp.linalg.norm(bent, axis=-1)), 1.0, atol=1e-6)


@pytest.mark.parametrize(
    "mask",
    [[], [4], [-1], [0.0, 1.0], [True, False], [False, False, False, False]],
)
def test_a_mask_that_selects_nothing_usable_is_refused(mask: Sequence[object]) -> None:
    """Refuse an empty, out-of-range or wrongly-typed selector."""
    bounds = jnp.asarray([[-2.0, 2.0]] * 4)
    with pytest.raises(ValueError, match="mask"):
        # The point of the case is a selector the type says is not a selector.
        SphereProjection(bounds, mask=mask)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    "ell",
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
def test_a_lengthscale_that_is_not_a_d_vector_is_refused(ell: jnp.ndarray) -> None:
    """Refuse a scalar, a short vector, a matrix and a non-positive entry.

    A scalar or a length-one vector would broadcast into a uniform vector, which
    is a real projection but never what a caller writing those shapes means, so
    it is refused rather than guessed at.
    """
    bounds = jnp.asarray([[-2.0, 2.0]] * 4)
    with pytest.raises(ValueError, match="ell"):
        SphereProjection(bounds, ell=ell)


def test_none_lengthscale_bends_every_dimension_alike() -> None:
    """Read `None` as the uniform default, bit-identical to an explicit unit vector."""
    bounds = jnp.asarray([[-2.0, 2.0]] * 3)
    xs = jnp.asarray([[-1.9, 2.7, 0.05], [0.0, 0.0, 0.5]])
    assert jnp.array_equal(
        SphereProjection(bounds).features(xs),
        SphereProjection(bounds, ell=jnp.ones(3)).features(xs),
    )


def test_a_non_uniform_lengthscale_moves_the_features() -> None:
    """Carry anisotropy through the ratios between the vector's entries."""
    bounds = jnp.asarray([[-2.0, 2.0]] * 3)
    xs = jnp.asarray([[-1.9, 2.7, 0.05], [1.3, -2.1, 0.8]])
    uniform = SphereProjection(bounds).features(xs)
    bent = SphereProjection(bounds, ell=jnp.asarray([3.0, 0.5, 2.0])).features(xs)
    assert not bool(jnp.allclose(uniform, bent))


def test_a_masked_model_proposes_inside_the_box() -> None:
    """Keep the acquisition path working when only some dimensions bend."""
    bounds = jnp.asarray([[-2.0, 2.0]] * 3)
    model = SphericalLinearModel(
        bounds,
        minimize=_grid_minimize,
        mask=[0, 2],
        key=jax.random.PRNGKey(0),
    )
    xs = jnp.linspace(-1.9, 1.9, 6)[:, None] * jnp.ones(3)[None, :]
    center = jnp.asarray([0.3, -0.2, 0.1])
    model.observe(xs, [float(jnp.sum((row - center) ** 2)) for row in xs], None)
    starts = model.propose(2)
    assert starts.shape == (2, 3)
    mean, sigma = model.predict(starts)
    assert bool(jnp.all(jnp.isfinite(starts)))
    assert bool(jnp.all(jnp.isfinite(mean)))
    assert bool(jnp.all(jnp.isfinite(sigma)))


def test_the_acquisition_gradient_is_the_jacobian_diagonal() -> None:
    """Read off the per-row gradient the batch Jacobian would only keep on its diagonal.

    The acquisition's rows are independent, so its ``m x m`` Jacobian is
    diagonal. This checks the `vmap`-of-`value_and_grad` gradient of
    `_acq_value_and_grad` against forward AD of the batch form written out
    here - an independent pass down the same math - so a row or shape mixup
    in the reverse-mode path shows up as a mismatch.
    """
    model = _model()
    _observe(model, 8)
    xs = jnp.asarray([[-1.4, 0.7], [0.3, -1.1], [1.6, 0.9]])
    weights, cov = model._posterior()
    best = model._standardize(jnp.asarray(model._fs)).min()

    def values(x: jnp.ndarray) -> jnp.ndarray:
        mean, sigma = model._posterior_at(x, weights, cov)
        z = (best - mean) / sigma
        return -(jnp.log(sigma) + _log_h(z))

    value, grad = model._acq_value_and_grad(xs, weights, cov, best)
    jac = jax.jacfwd(values)(xs)
    diagonal = jnp.stack([jac[i, i] for i in range(xs.shape[0])])

    assert value.shape == (3,)
    assert grad.shape == (3, 2)
    assert np.allclose(np.asarray(value), np.asarray(values(xs)), atol=2e-4)
    assert np.allclose(np.asarray(grad), np.asarray(diagonal), atol=2e-4)


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


def test_a_flat_batch_has_no_spread_to_fit() -> None:
    """Fall back like a lone observation: equal values leave nothing to scale."""
    model = _model()
    model.observe(jnp.asarray([[0.5, -0.5], [-0.5, 0.5]]), [3.0, 3.0], None)
    assert not model._can_fit()
    draws = model.propose(4)
    assert draws.shape == (4, 2)
    assert bool(jnp.all(draws >= -2.0))
    assert bool(jnp.all(draws <= 2.0))
    with pytest.raises(ValueError, match="spread to standardize"):
        model.predict(jnp.asarray([[0.0, 0.0]]))


def test_posterior_deviation_never_dips_below_the_observation_noise() -> None:
    """Keep the root defined by construction: noise plus a non-negative form."""
    model = _model()
    _observe(model, 6)
    _, sigma = model.predict(jnp.stack(model._xs))
    assert float(jnp.min(sigma)) >= math.sqrt(model._noise) - 1e-6


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


def test_an_unbounded_box_or_a_bad_knob_is_refused() -> None:
    """Refuse an unbounded box and a zero noise, the two that reach the fit."""
    unbounded = jnp.asarray([[0.0, math.inf]])
    with pytest.raises(ValueError, match="finite box"):
        SphericalLinearModel(
            unbounded,
            minimize=_grid_minimize,
            key=jax.random.PRNGKey(0),
        )
    with pytest.raises(ValueError, match="noise"):
        _model(noise=0.0)
