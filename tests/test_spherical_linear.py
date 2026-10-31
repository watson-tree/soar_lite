"""Tests for the Bayesian linear-regression restart model.

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

from soar_lite import (
    AcquisitionFn,
    BLRGlobalModel,
    Bounds,
    IdentityProjection,
    Projection,
    SphereProjection,
)
from soar_lite.spherical_linear import _log_h


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


def _model(d: int = 2, half: float = 2.0, noise: float = 0.01) -> BLRGlobalModel:
    """Build a model over the symmetric ``[-half, half]^d`` box."""
    bounds = jnp.asarray([[-half, half]] * d)
    return BLRGlobalModel(
        SphereProjection(bounds),
        minimize=_grid_minimize,
        noise=noise,
        key=jax.random.PRNGKey(0),
    )


def _observe(model: BLRGlobalModel, n: int) -> None:
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
    for mask in (None, [True, False, False, False], [True, False, True, False]):
        projection = SphereProjection(bounds, mask=mask)
        assert projection.features(xs).shape == (6, 5)


def test_dimensions_left_out_keep_their_centred_coordinate() -> None:
    """Read unmasked dimensions as a plain trend in their own coordinate unit."""
    bounds = jnp.asarray([[-2.0, 2.0], [0.0, 10.0]])
    projection = SphereProjection(bounds, mask=[True, False])
    xs = jnp.asarray([[-1.0, 2.5], [1.0, 7.5]])
    trend = (xs - projection.center) / projection.scale
    assert jnp.array_equal(projection.features(xs)[:, 2:], trend[:, 1:])


def test_the_bent_block_alone_still_lies_on_the_unit_sphere() -> None:
    """Keep the variance bound on the bent subset, not the whole row."""
    bounds = jnp.asarray([[-2.0, 2.0]] * 4)
    xs = jnp.linspace(-1.9, 1.9, 6)[:, None] * jnp.ones(4)[None, :]
    bent = SphereProjection(bounds, mask=[True, True, False, False]).features(xs)[:, :3]
    assert np.allclose(np.asarray(jnp.linalg.norm(bent, axis=-1)), 1.0, atol=1e-6)


@pytest.mark.parametrize(
    "mask",
    [[], [0, 2], [0.0, 1.0], [True, False], [False, False, False, False]],
)
def test_a_mask_that_selects_nothing_usable_is_refused(mask: Sequence[object]) -> None:
    """Refuse an empty, non-boolean, short or all-false selector."""
    bounds = jnp.asarray([[-2.0, 2.0]] * 4)
    with pytest.raises(ValueError, match="mask"):
        # The point of the case is a selector the type says is not a selector.
        SphereProjection(bounds, mask=mask)  # pyright: ignore[reportArgumentType]


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
        SphereProjection(bounds, scale=scale)


def test_none_scale_is_the_uniform_default() -> None:
    """Read `None` as one everywhere, bit-identical to an explicit unit vector."""
    bounds = jnp.asarray([[-2.0, 2.0]] * 3)
    xs = jnp.asarray([[-1.9, 2.7, 0.05], [0.0, 0.0, 0.5]])
    assert jnp.array_equal(
        SphereProjection(bounds).features(xs),
        SphereProjection(bounds, scale=jnp.ones(3)).features(xs),
    )


def test_a_non_uniform_scale_moves_the_features() -> None:
    """Carry anisotropy through the ratios between the vector's entries."""
    bounds = jnp.asarray([[-2.0, 2.0]] * 3)
    xs = jnp.asarray([[-1.9, 2.7, 0.05], [1.3, -2.1, 0.8]])
    uniform = SphereProjection(bounds).features(xs)
    bent = SphereProjection(bounds, scale=jnp.asarray([3.0, 0.5, 2.0])).features(xs)
    assert not bool(jnp.allclose(uniform, bent))


def test_a_masked_model_proposes_inside_the_box() -> None:
    """Keep the acquisition path working when only some dimensions bend."""
    bounds = jnp.asarray([[-2.0, 2.0]] * 3)
    model = BLRGlobalModel(
        SphereProjection(bounds, mask=[True, False, True]),
        minimize=_grid_minimize,
        key=jax.random.PRNGKey(0),
    )
    xs = jnp.linspace(-1.9, 1.9, 6)[:, None] * jnp.ones(3)[None, :]
    center = jnp.asarray([0.3, -0.2, 0.1])
    model.observe(xs, [float(jnp.sum((row - center) ** 2)) for row in xs], None)
    starts = model.propose(2)
    assert starts.shape == (2, 3)
    mean, sigma = model._predict(starts)
    assert bool(jnp.all(jnp.isfinite(starts)))
    assert bool(jnp.all(jnp.isfinite(mean)))
    assert bool(jnp.all(jnp.isfinite(sigma)))


def test_the_injected_projection_is_the_geometry_authority() -> None:
    """Read the run's dimension and box off the projection, not a bounds param."""
    projection = SphereProjection(
        jnp.asarray([[-1.0, 1.0]] * 4),
        mask=[False, True, False, True],
    )
    model = BLRGlobalModel(
        projection,
        minimize=_grid_minimize,
        key=jax.random.PRNGKey(0),
    )
    assert model.d == 4
    assert model._projection is projection


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
    model = BLRGlobalModel(
        SphereProjection([(0, 1), (0, 1)]),
        minimize=_grid_minimize,
        key=jax.random.PRNGKey(0),
    )
    draws = model.propose(3)
    assert bool(jnp.all(draws >= 0.0))
    assert bool(jnp.all(draws <= 1.0))
    model.observe(jnp.asarray([[0.2, 0.3], [0.8, 0.9]]), [1.0, 2.0], None)
    mean, sigma = model._predict(jnp.asarray([[0.5, 0.5]]))
    assert bool(jnp.all(jnp.isfinite(mean)))
    assert bool(jnp.all(jnp.isfinite(sigma)))


def test_predict_refuses_a_fit_it_cannot_standardize() -> None:
    """Refuse a posterior query that has no value spread to work from."""
    model = _model()
    model.observe(jnp.asarray([[0.5, -0.5]]), [1.0], None)
    with pytest.raises(ValueError, match="spread to standardize"):
        model._predict(jnp.asarray([[0.0, 0.0]]))


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
        model._predict(jnp.asarray([[0.0, 0.0]]))


def test_posterior_deviation_never_dips_below_the_observation_noise() -> None:
    """Keep the root defined by construction: noise plus a non-negative form."""
    model = _model()
    _observe(model, 6)
    _, sigma = model._predict(jnp.stack(model._xs))
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
        model._predict(jnp.zeros(bad))


def test_posterior_variance_stays_under_the_prior_far_from_data() -> None:
    """Bound the far-from-data variance, which is the whole point of the projection."""
    # The structural cure: sigma^2 = noise * (1 + p^T A^-1 p) with |p| = 1 and
    # A >= (noise/var_prior) I, so sigma^2 <= noise + var_prior everywhere.
    model = _model()
    _observe(model, 6)
    near, near_sigma = model._predict(jnp.asarray([[1.9, -1.9]]))
    far, far_sigma = model._predict(jnp.asarray([[1.9, 1.9]]))
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
    mean, sigma = model._predict(starts)
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
    model = BLRGlobalModel(
        SphereProjection(bounds),
        minimize=_short_minimize,
        key=jax.random.PRNGKey(1),
    )
    _observe(model, 4)
    with pytest.raises(ValueError, match=r"\(3, 2\) array"):
        model.propose(3)


def test_a_projection_needs_a_declared_box_and_the_model_a_positive_noise() -> None:
    """Split the boundary: the box is the projection's, the noise the fit's."""
    with pytest.raises(ValueError, match="needs one"):
        SphereProjection(None)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="noise"):
        _model(noise=0.0)


def test_identity_features_are_the_coordinates_themselves() -> None:
    """Read every coordinate back unchanged, with no added width."""
    bounds = jnp.asarray([[-2.0, 2.0]] * 3)
    xs = jnp.asarray([[1.0, -0.5, 2.0], [-2.0, 0.0, 0.4]])
    feats = IdentityProjection(bounds).features(xs)
    assert bool(jnp.array_equal(feats, xs))
    assert feats.shape == (2, 3)


def test_an_identity_projection_is_a_projection_and_the_sphere_is_too() -> None:
    """Pin the abstraction: both shipped projections inherit `Projection`."""
    bounds = jnp.asarray([[-2.0, 2.0]] * 2)
    assert isinstance(IdentityProjection(bounds), Projection)
    assert isinstance(SphereProjection(bounds), Projection)


def test_the_identity_model_fits_a_width_d_posterior() -> None:
    """The feature width is read off `features`, not hardcoded ``d + 1``."""
    bounds = jnp.asarray([[-2.0, 2.0]] * 2)
    model = BLRGlobalModel(
        IdentityProjection(bounds),
        minimize=_grid_minimize,
        key=jax.random.PRNGKey(0),
    )
    _observe(model, 4)
    mean, sigma = model._predict(jnp.asarray([[0.5, -0.5]]))
    assert bool(jnp.all(jnp.isfinite(mean)))
    assert bool(jnp.all(jnp.isfinite(sigma)))
    draws = model.propose(3)
    assert draws.shape == (3, 2)
    assert bool(jnp.all(jnp.isfinite(draws)))
    assert bool(jnp.all(draws >= -2.0))
    assert bool(jnp.all(draws <= 2.0))


def test_a_projection_accepts_an_unbounded_side() -> None:
    """Let stereography bend a line: its domain is all of ``R``."""
    projection = SphereProjection(jnp.asarray([[0.0, math.inf]]))
    feats = projection.features(jnp.asarray([[0.0], [1.0], [1e6]]))
    assert bool(jnp.all(jnp.isfinite(feats)))
    assert np.allclose(np.asarray(jnp.linalg.norm(feats, axis=-1)), 1.0, atol=1e-5)


def test_an_identity_projection_accepts_an_unbounded_side_too() -> None:
    """The box check no longer demands a finite box, so both subclasses pass."""
    bounds = jnp.asarray([[0.0, math.inf]])
    xs = jnp.asarray([[3.0]])
    assert bool(jnp.array_equal(IdentityProjection(bounds).features(xs), xs))


def test_the_default_scale_is_one_not_the_box_width() -> None:
    """Measure the prior in coordinate units, so no width is taken for a unit."""
    projection = SphereProjection(jnp.asarray([[-2.0, 2.0], [-10.0, 10.0]]))
    assert jnp.array_equal(projection.scale, jnp.ones(2))
    assert jnp.array_equal(projection.center, jnp.zeros(2))
    assert jnp.array_equal(projection.width, jnp.asarray([4.0, 20.0]))


def test_a_uniform_scale_cancels_out_of_the_projection() -> None:
    """Keep the ratio-only semantics: `glob_ls` divides a uniform scale back out."""
    bounds = jnp.asarray([[-2.0, 2.0]] * 3)
    xs = jnp.asarray([[-1.9, 0.4, 1.1]])
    assert jnp.allclose(
        SphereProjection(bounds).features(xs),
        SphereProjection(bounds, scale=jnp.full(3, 7.0)).features(xs),
        atol=1e-6,
    )


@pytest.mark.parametrize("d", [2, 8])
def test_the_global_lengthscale_holds_the_far_corner_at_one_over_kappa(d: int) -> None:
    """Tie the far corner to ``|z|^2 = 1/kappa`` whatever the dimension."""
    projection = SphereProjection(jnp.asarray([[0.0, 1.0]] * d), kappa=0.25)
    far = (jnp.ones(d) - projection.center) / projection.scale / projection.glob_ls
    assert float(jnp.sum(far * far)) == pytest.approx(1.0 / 0.25, rel=1e-6)


def test_a_mixed_box_draws_the_unbounded_dimension_at_its_own_scale() -> None:
    """Draw the unbounded column from ``N(0, scale^2)``, not from a zero width."""
    bounds = jnp.asarray([[0.0, 1.0], [-math.inf, math.inf]])
    projection = SphereProjection(bounds, scale=jnp.asarray([1.0, 3.0]))
    draws = projection.draw(jax.random.PRNGKey(0), 256)
    assert draws.shape == (256, 2)
    assert bool(jnp.all(jnp.isfinite(draws)))
    assert bool(jnp.all(draws[:, 0] >= 0.0))
    assert bool(jnp.all(draws[:, 0] <= 1.0))
    assert float(jnp.std(draws[:, 1])) == pytest.approx(3.0, rel=0.3)


def test_an_unbounded_box_proposes_and_predicts_without_a_width() -> None:
    """Keep the whole path finite where no side has a width to divide by."""
    bounds = jnp.asarray([[-math.inf, math.inf]] * 2)
    model = BLRGlobalModel(
        SphereProjection(bounds),
        minimize=_grid_minimize,
        key=jax.random.PRNGKey(0),
    )
    draws = model.propose(4)  # no fit yet: the projection's own draw
    assert draws.shape == (4, 2)
    assert bool(jnp.all(jnp.isfinite(draws)))
    _observe(model, 6)
    _, sigma = model._predict(jnp.asarray([[0.0, 0.0], [1e3, -1e3]]))
    assert bool(jnp.all(jnp.isfinite(sigma)))
    # Far out the bend returns the variance to the prior instead of growing.
    assert float(sigma[1]) <= math.sqrt(1.01) + 1e-3


def test_an_injected_sampler_replaces_that_dimensions_default() -> None:
    """Take one column's starts from the caller's own distribution."""
    bounds = jnp.asarray([[-2.0, 2.0], [-math.inf, math.inf]])

    def constant(key: jnp.ndarray, n_points: int) -> jnp.ndarray:
        del key
        return jnp.full((n_points,), 7.0)

    draws = SphereProjection(bounds, samplers=[constant, None]).draw(
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
        SphereProjection(bounds, samplers=samplers)  # pyright: ignore[reportArgumentType]


def test_a_sampler_that_returns_the_wrong_shape_is_refused() -> None:
    """Refuse a column whose shape would silently widen the stacked draw."""
    bounds = jnp.asarray([[-2.0, 2.0]] * 2)

    def flat(key: jnp.ndarray, n_points: int) -> jnp.ndarray:
        del key
        return jnp.zeros(n_points * 2)

    projection = SphereProjection(bounds, samplers=[flat, None])
    with pytest.raises(ValueError, match="drawn column"):
        projection.draw(jax.random.PRNGKey(0), 3)
