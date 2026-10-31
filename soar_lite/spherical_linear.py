"""A spherical-linear Bayesian surrogate as the engine's restart model.

The second `GlobalModel` concrete, beside `global_model.py`'s uniform draw. It
fits a weight posterior over an inverse-stereographic projection of the
normalized box and proposes restart points by descending the *log* expected
improvement of that posterior - the sample-based restart the uniform draw only
approximates.

The structure is a fixed-hyperparameter port of Doumont, Fan, et al., "We Still
Don't Understand High-Dimensional Bayesian Optimization" (2025),
``github.com/colmont/linear-bo``: a linear kernel read after projecting onto the
unit sphere, a box-diagonal lengthscale normalization, and the resulting
posterior variance bounded away from the prior away from data. The projection is
`SphereProjection`, a geometry of its own: it can bend only a chosen subset of
the dimensions, leaving the rest linear. Two things are deliberately dropped.
That implementation fits its lengthscale, coefficient and
noise hyperparameters by marginal likelihood (it is gpytorch; this package is
jax-only), so here the same knobs are constructor values instead. And it
descends its acquisition with a shipped multi-start L-BFGS-B; here the
minimizer is injected instead, so a driver can descend with the library's own
n-path engine rather than a second recipe - see `MinimizeFn`.

What is kept is the part that cures the pathology this model exists for: under
a Euclidean RBF posterior the variance returns to the prior far from data, so
expected improvement becomes variance-dominated and points at the box corners;
the projection bounds it, so improvement is driven by the fitted values instead.
"""

import math
from collections.abc import Callable, Sequence
from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr

from soar_lite.engine import (
    Bounds,
    GlobalModel,
    box_bounds,
    check_bounds,
    finite_box,
)

# The acquisition `propose` hands the injected minimizer: x rows (m, d) in,
# (values (m,), gradients (m, d)) out - the engine's `EvalFG` shape with the
# feasibility column dropped, since an acquisition is finite everywhere and
# nothing is infeasible. Batched, so a zero-order minimizer reads the values off
# a whole grid in one call and a first-order one gets the gradient it needs
# alongside.
AcquisitionFn: TypeAlias = Callable[
    [jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray],
]

# The inner minimizer `propose` hands its own acquisition to, injected by the
# composition root (the experiment driver) instead of imported, so this module
# never reaches back into the engine's scheduler:
# `minimize(acquisition, bounds, key, n_points)` returns up to `n_points` local
# minimizers of `acquisition` over `bounds`, one per row, shape `(n_points, d)`.
# The engine already owns the descent recipe; a second one here would be a
# second answer to the same question.
MinimizeFn: TypeAlias = Callable[
    [AcquisitionFn, Bounds, jnp.ndarray, int],
    jnp.ndarray,
]

_HALF_LOG_2PI = 0.5 * jnp.log(2.0 * jnp.pi)


class SphereProjection:
    """Inverse-stereographic projection of a chosen subset of the box's dims.

    The restart models read the box in whatever coordinates the objective takes.
    This maps them onto the unit sphere, ``[2z; |z|^2 - 1] / (|z|^2 + 1)``, where
    ``z`` is the box's coordinates centered, divided by the per-dimension
    lengthscales and by one global lengthscale. Two properties matter
    downstream.

    Every projected row has norm one, so a linear head read on these features
    has a constant self inner product: its posterior variance stays bounded far
    from data instead of climbing back to the prior, which is the corner-seeking
    pathology the projection exists to cure. And the last coordinate,
    ``(|z|^2 - 1)/(|z|^2 + 1)``, is a radially symmetric, saturating feature of
    the input - the one extra degree of freedom a linear head gets, an isotropic
    bowl whose center the weights place.

    `mask` keeps that bowl off dimensions whose natural prior is uniform - an
    angle, say, wants no attraction to the box center. A left-out dimension
    keeps its box coordinate and enters a linear head as a plain trend, so the
    feature vector is ``d + 1`` wide however many dimensions the projection
    bends.
    """

    def __init__(
        self,
        bounds: Bounds,
        *,
        ell: jnp.ndarray | None = None,
        kappa: float = 0.5,
        mask: jnp.ndarray | Sequence[bool] | Sequence[int] | None = None,
    ) -> None:
        """Bind the box, the lengthscales and the projected subset.

        Args:
            bounds: the finite box the projection normalizes, scipy-style pairs
                or a ``(d, 2)`` array. Every side must be finite.
            ell: per-dimension lengthscale in unit-box units, a ``(d,)`` vector
                or ``None`` for the default, which bends every dimension
                alike. Only the ratios between the projected dimensions' entries
                move the features: `glob_ls` divides one global lengthscale back
                out, so a uniform vector and ``None`` give the same projection.
            kappa: fraction of the box diagonal the projection radius spans.
                Must lie in ``(0, 1)``; without that tie a high-dimensional box
                pushes every corner past the pole and the posterior collapses
                there.
            mask: which dimensions the projection bends, a ``(d,)`` boolean
                selector or the indices of the projected dimensions; ``None``
                bends all of them.

        Raises:
            ValueError: If the box is absent, unbounded on any side, or has a
                NaN side; if `ell` is not a ``(d,)`` vector of finite positive
                values, or `kappa` is out of range; or if `mask` selects no
                dimension or names one outside the box.

        """
        check_bounds(bounds)
        box = box_bounds(bounds)
        if box is None or not finite_box(box):
            msg = (
                "SphereProjection normalizes the search box, so it needs a "
                f"finite box: got {bounds!r}"
            )
            raise ValueError(msg)
        self.lo, self.hi = box
        # The dimension is what `bounds` declares: read it off the argument,
        # not off `lo`/`hi`, which `box_bounds` just derived from it.
        self.d = len(bounds)
        self.width = self.hi - self.lo
        self.bounds = bounds
        self.mask = _check_mask(mask, self.d)
        self.ell = _check_lengthscale(ell, self.d)
        kappa_v = _check_finite_positive("kappa", kappa)
        if kappa_v >= 1.0:
            msg = (
                "kappa ties the projection radius to the box diagonal, so it "
                f"must lie in (0, 1); got {kappa!r}"
            )
            raise ValueError(msg)
        # The global lengthscale is the piece that makes the projection scale-
        # free: it maps the box's farthest corner to radius sqrt(2*kappa) for any
        # d, so a wider box does not push every corner to the pole. Summed over
        # the projected dimensions only - a dimension the projection leaves out
        # has no radius to tie.
        self.glob_ls = jnp.sqrt(kappa_v * jnp.sum((0.5 / self.ell[self.mask]) ** 2))

    def features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return the ``(m, d + 1)`` feature rows for ``(m, d)`` coordinates.

        The bent dimensions become their stereographic image on the unit sphere;
        the rest keep their box coordinate, uncentered and unscaled, so a linear
        head reads them as a plain linear trend.
        """
        unit = (x - self.lo) / self.width
        z = (unit[:, self.mask] - 0.5) / self.ell[self.mask] / self.glob_ls
        sq = jnp.sum(z * z, axis=-1, keepdims=True)
        sphere = jnp.concatenate([2.0 * z, sq - 1.0], axis=-1) / (sq + 1.0)
        return jnp.concatenate([sphere, unit[:, ~self.mask]], axis=-1)


class SphericalLinearModel(GlobalModel):
    """Restart points from a spherical-linear posterior's log improvement.

    `observe` accumulates the run's served batches (values only - the gradient
    is not fitted), and `propose` fits the closed-form weight posterior over
    those points and returns the minimizers of ``-log EI`` the injected
    `minimize` finds. Below two observations - and with every value equal -
    there is no spread to standardize by, so it answers with the uniform draw
    the default model would give.

    Every knob is a fixed hyperparameter, not a fitted one; the defaults are the
    reference implementation's starting points rather than its marginal-
    likelihood optimum. The search box and the geometry knobs (`ell`, `kappa`,
    `mask`) live in the injected `SphereProjection`; what this class adds on top
    is the value fit and the acquisition that descends it.
    """

    def __init__(
        self,
        projection: SphereProjection,
        *,
        minimize: MinimizeFn,
        var_prior: float = 1.0,
        noise: float = 0.01,
        key: jnp.ndarray,
    ) -> None:
        """Bind the projection, the injected minimizer and the value fits.

        Args:
            projection: the `SphereProjection` turning a coordinate into the
                posterior's feature row. It is the authority for the run's
                dimension and search box - `d` and `bounds` are read off it,
                and its own construction already validated the box, `ell`,
                `kappa` and `mask`.
            minimize: the acquisition's inner optimizer, shape
                ``MinimizeFn``. Owned by the caller: the descent recipe is the
                engine's, not this model's.
            var_prior: weight-prior variance (the ridge scale opposite `noise`).
            noise: observation-noise variance added to the posterior mean's
                self-covariance.
            key: the PRNG key `propose` splits every ask from.

        Raises:
            ValueError: If `var_prior` or `noise` is not a finite positive
                value.

        """
        # The projection is the geometry's authority: it validated the box and
        # the geometry knobs at its own construction, and it is the single
        # source of the run's dimension. `d` is lifted here because the
        # `GlobalModel` seam asks the model for it.
        self._projection = projection
        self.d = projection.d
        self._minimize = minimize
        self._var_prior = _check_finite_positive("var_prior", var_prior)
        self._noise = _check_finite_positive("noise", noise)
        self._key = key
        self._xs: list[jnp.ndarray] = []
        self._fs: list[float] = []

    def observe(
        self,
        x: jnp.ndarray,
        f: list[float],
        g: jnp.ndarray | None,
    ) -> None:
        """Fold one served batch into the model's value-only evidence.

        A value-only batch and a fused one contribute the same, since the
        surrogate fits values over coordinates. A non-finite value is dropped,
        not fitted: the engine observes the objective bottoming out at ``-inf``
        before it stops, and one such row would drag the standardization with
        it. A point served twice is kept twice - under the tiny observation
        noise that is a slight double weight, which is what a second look at the
        same coordinate is worth.

        Raises:
            ValueError: If `x` is not an ``(m, d)`` batch, or if the batch has a
                different number of rows than values.

        """
        del g

        rows = jnp.asarray(x)
        if rows.ndim != 2 or rows.shape[1] != self.d:
            msg = f"observe needs a (m, {self.d}) batch, got {rows.shape}"
            raise ValueError(msg)
        if rows.shape[0] != len(f):
            msg = f"observe got {rows.shape[0]} rows but {len(f)} values"
            raise ValueError(msg)
        for row, value in zip(rows, f, strict=True):
            if math.isfinite(value):
                self._xs.append(row)
                self._fs.append(value)

    def propose(self, n_points: int) -> jnp.ndarray:
        """Return `n_points` restart points: a uniform draw until a fit is possible.

        Until the fit has a spread to standardize, the answer is the uniform
        draw the default model would give - the engine only ever sees the
        points.
        Afterwards the answer is the injected `minimize`'s local minimizers of
        ``-log EI`` over the box; the log form keeps the acquisition's gradient
        alive in the region where the raw improvement has decayed to zero.

        Raises:
            ValueError: If `n_points` is below one, or if `minimize` returns
                anything but a ``(n_points, d)`` array.

        """
        if n_points <= 0:
            msg = f"propose needs at least one point, got {n_points}"
            raise ValueError(msg)
        self._key, sub = jax.random.split(self._key)
        if not self._can_fit():
            return jax.random.uniform(
                sub,
                (n_points, self.d),
                minval=self._projection.lo,
                maxval=self._projection.hi,
            )
        weights, cov = self._posterior()
        best = self._standardize(jnp.asarray(self._fs)).min()

        # shortcut: the minimizers are independent, so a flat acquisition can
        # return a cluster on one face. Upgrade to greedy conditioning on the
        # points already picked when a cohort asks for many at once.
        def acquisition(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            return self._acq_value_and_grad(x, weights, cov, best)

        starts = jnp.asarray(
            self._minimize(acquisition, self._projection.bounds, sub, n_points),
        )
        if starts.shape != (n_points, self.d):
            msg = (
                "minimize must return a "
                f"({n_points}, {self.d}) array, got shape {starts.shape}"
            )
            raise ValueError(msg)
        return starts

    def _predict(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return the posterior mean and standard deviation at `x` rows.

        A read-only diagnostic kept `_`-private for the tests: the engine never
        calls it, and `propose` reaches the posterior directly rather than
        through this method, so it is not part of the `GlobalModel` contract a
        caller implements. It also answers only in the *standardized* value
        scale - the fit centers and scales the observed values - which is one
        more reason it stays off the public face.

        Raises:
            ValueError: If the fit has no spread to standardize by yet, or if
                `x` is not an ``(m, d)`` batch.

        """
        if not self._can_fit():
            msg = "a posterior read needs a spread to standardize by"
            raise ValueError(msg)
        batch = jnp.asarray(x)
        if batch.ndim != 2 or batch.shape[1] != self.d:
            msg = f"a posterior read needs a (m, {self.d}) batch, got {batch.shape}"
            raise ValueError(msg)
        weights, cov = self._posterior()
        return self._posterior_at(batch, weights, cov)

    def _can_fit(self) -> bool:
        """Return whether the values have a spread to standardize by.

        One observation has no spread of its own, and neither has a batch of
        equal values: both leave nothing to divide by and no improvement to
        measure against.
        """
        return len(self._xs) > 1 and bool(jnp.asarray(self._fs).std() > 0)

    def _standardize(self, f: jnp.ndarray) -> jnp.ndarray:
        """Return `f` centered and scaled to unit variance.

        Only reached once `_can_fit` has found a non-zero spread, so the divisor
        never is zero and no floor is needed to keep it finite.
        """
        return (f - f.mean()) / f.std()

    def _posterior(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return the weight posterior's mean and covariance.

        Written as the precision it is: ``P^T P / noise + I / var_prior`` is
        likelihood plus prior, and one inversion gives the covariance matrix
        itself - not an inverse scaled afterwards - with
        ``weights = cov @ P^T y / noise`` its mean.
        """
        x_obs = jnp.stack(self._xs)
        f_obs = jnp.asarray(self._fs, dtype=x_obs.dtype)
        values = self._standardize(f_obs)
        features = self._projection.features(x_obs)
        precision = (
            features.T @ features / self._noise
            + jnp.eye(self.d + 1, dtype=features.dtype) / self._var_prior
        )
        cov = jnp.linalg.inv(precision)
        weights = cov @ features.T @ values / self._noise
        return weights, cov

    def _posterior_at(
        self,
        x: jnp.ndarray,
        weights: jnp.ndarray,
        cov: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Posterior mean and deviation of the standardized value at ``(m, d)`` rows.

        The variance is the observation noise plus the covariance's quadratic
        form, so it stays above `noise` and its root needs no floor.
        """
        features = self._projection.features(x)
        mean = features @ weights
        var = self._noise + jnp.einsum("ij,jk,ik->i", features, cov, features)
        return mean, jnp.sqrt(var)

    def _acq_value_and_grad(
        self,
        x: jnp.ndarray,
        weights: jnp.ndarray,
        cov: jnp.ndarray,
        best: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``-log EI`` and its per-row gradient at ``(m, d)`` rows.

        `jax.value_and_grad` differentiates a scalar output only, and the rows
        of the acquisition are independent of one another, so the gradient is
        taken one row at a time under `vmap`. The batch Jacobian is diagonal,
        and this reads only its diagonal where `jax.jacobian` would build the
        whole ``m x m`` block and throw all but that away.
        """

        def scalar(row: jnp.ndarray) -> jnp.ndarray:
            mean, sigma = self._posterior_at(row[None, :], weights, cov)
            z = (best - mean[0]) / sigma[0]
            return -(jnp.log(sigma[0]) + _log_h(z))

        return jax.vmap(jax.value_and_grad(scalar))(x)


def _log_h(z: jnp.ndarray) -> jnp.ndarray:
    """Return ``log(phi(z) + z*Phi(z))``, stable over the whole real line.

    ``h`` is the improvement kernel inside expected improvement: ``EI =
    sigma * h(z)`` with ``z = (best - mean) / sigma``. It is positive everywhere,
    but for ``z < 0`` it is the difference of two nearly equal terms (``phi``
    above ``|z|*Phi``), and a plain ``log(max(h, 1e-12))`` floor erases its
    gradient long before a descent needs to walk away from that region. Three
    segments keep it meaningful: ``z >= 0`` sums two positives in log space;
    ``-10 < z < 0`` folds the difference with ``log1mexp``; below that the
    cancellation is wider than a float32 mantissa, so the leading Mills-ratio
    asymptotics ``log phi(z) - 2 log|z|`` take over - its ``1 - 3/z^2``
    correction is under 3 percent at the seam, under 1 percent from ``z = -20``,
    and the improvement is already ``~1e-24`` there.
    """
    log_phi = -0.5 * z * z - _HALF_LOG_2PI
    log_z_phi = jnp.log(jnp.abs(z)) + log_ndtr(z)  # log|z * Phi(z)|
    log_h_sum = jnp.logaddexp(log_phi, log_z_phi)
    gap = jnp.minimum(log_z_phi - log_phi, -1e-7)
    log_h_diff = log_phi + jnp.log(-jnp.expm1(gap))
    log_h_far = log_phi - 2.0 * jnp.log(-z)
    return jnp.where(
        z >= 0.0,
        log_h_sum,
        jnp.where(z > -10.0, log_h_diff, log_h_far),
    )


def _check_finite_positive(name: str, value: float) -> float:
    """Return `value` unchanged, refusing a non-finite or non-positive knob."""
    if not math.isfinite(value) or value <= 0.0:
        msg = f"{name} must be a finite positive number, got {value!r}"
        raise ValueError(msg)
    return value


def _check_mask(
    mask: jnp.ndarray | Sequence[bool] | Sequence[int] | None,
    d: int,
) -> jnp.ndarray:
    """Normalize `mask` to a ``(d,)`` boolean selector, refusing an empty one.

    Two spellings are accepted - a boolean selector, or the indices of the
    projected dimensions - because the index form is how a caller names a few
    dimensions while the boolean form is what the feature gather needs. A
    floating-point selector is refused rather than rounded into indices: a
    caller who writes ``[0.0, 1.0]`` means something this cannot guess.
    """
    if mask is None:
        return jnp.ones(d, dtype=bool)
    given = jnp.asarray(mask)
    if given.ndim != 1:
        msg = f"mask must be a 1-D selector, got shape {given.shape}"
        raise ValueError(msg)
    if given.shape[0] == 0:
        msg = "mask must project at least one dimension"
        raise ValueError(msg)
    if given.dtype == jnp.bool_:
        if given.shape[0] != d:
            msg = f"mask needs {d} boolean entries, got {given.shape[0]}"
            raise ValueError(msg)
        selected = given
    elif given.dtype.kind in "iu":
        indices = given.astype(jnp.int32)
        if bool(jnp.any(indices < 0)) or bool(jnp.any(indices >= d)):
            msg = f"mask indices must lie in [0, {d}), got {mask!r}"
            raise ValueError(msg)
        selected = jnp.zeros(d, dtype=bool).at[indices].set(True)
    else:
        msg = f"mask must be a boolean selector or integer indices, got {mask!r}"
        raise ValueError(msg)
    if not bool(jnp.any(selected)):
        msg = "mask must project at least one dimension"
        raise ValueError(msg)
    return selected


def _check_lengthscale(ell: jnp.ndarray | None, d: int) -> jnp.ndarray:
    """Normalize the per-dimension lengthscale to ``(d,)``, refusing a bad one.

    ``None`` is every dimension at one, the projection that bends them alike.
    Anything but a ``(d,)`` vector is refused rather than broadcast into one: a
    scalar or a length-one vector would silently become a uniform vector, which
    is a real projection but never what a caller reaching for those shapes
    means.
    """
    if ell is None:
        return jnp.ones(d)
    lengthscale = jnp.asarray(ell)
    if lengthscale.ndim != 1 or lengthscale.shape[0] != d:
        msg = f"ell must be a ({d},) vector or None, got shape {lengthscale.shape}"
        raise ValueError(msg)
    if not bool(jnp.all(jnp.isfinite(lengthscale))) or not bool(
        jnp.all(lengthscale > 0.0),
    ):
        msg = f"ell must hold finite positive values, got {ell!r}"
        raise ValueError(msg)
    return lengthscale
