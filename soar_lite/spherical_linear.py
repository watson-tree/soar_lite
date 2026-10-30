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
posterior variance bounded away from the prior away from data. Two things are
deliberately dropped. That implementation fits its lengthscale, coefficient and
noise hyperparameters by marginal likelihood (it is gpytorch; this package is
jax-only), so here the same knobs are constructor values instead. And it
minimizes its acquisition with a multi-start L-BFGS-B, which is why the
minimizer is injected rather than shipped - see `MinimizeFn`.

What is kept is the part that cures the pathology this model exists for: under
a Euclidean RBF posterior the variance returns to the prior far from data, so
expected improvement becomes variance-dominated and points at the box corners;
the projection bounds it, so improvement is driven by the fitted values instead.
"""

import math
from collections.abc import Callable
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

# The inner minimizer `propose` hands its own acquisition function to, injected
# by the composition root (the experiment driver) instead of imported, so this
# module never reaches back into the engine's scheduler:
# `minimize(objective, bounds, key, n_points)` returns up to `n_points` local
# minimizers of the scalar `objective` over `bounds`, one per row, shape
# `(n_points, d)`. The engine already owns the descent recipe; a second one here
# would be a second answer to the same question.
MinimizeFn: TypeAlias = Callable[
    [Callable[[jnp.ndarray], jnp.ndarray], Bounds, jnp.ndarray, int],
    jnp.ndarray,
]

_HALF_LOG_2PI = 0.5 * jnp.log(2.0 * jnp.pi)


class SphericalLinearModel(GlobalModel):
    """Restart points from a spherical-linear posterior's log improvement.

    `observe` accumulates the run's served batches (values only - the gradient
    is not fitted), and `propose` fits the closed-form weight posterior over
    those points and returns the minimizers of ``-log EI`` the injected
    `minimize` finds. Below two observations there is no spread to standardize
    by, so it answers with the uniform draw the default model would give.

    Every knob is a fixed hyperparameter, not a fitted one; the defaults are the
    reference implementation's starting points rather than its marginal-
    likelihood optimum. `ell` is in unit-box units, so ``1.0`` is "the whole
    box", matching how the reference reads its inputs.
    """

    def __init__(  # noqa: PLR0913 - each knob is a distinct, independently-swept facet
        self,
        bounds: Bounds,
        *,
        minimize: MinimizeFn,
        ell: float | jnp.ndarray = 1.0,
        kappa: float = 0.5,
        var_prior: float = 1.0,
        noise: float = 0.01,
        key: jnp.ndarray,
    ) -> None:
        """Bind the box, the injected minimizer and the fixed hyperparameters.

        Args:
            bounds: the finite box the surrogate fits over, scipy-style pairs or
                a ``(d, 2)`` array. It is normalized to the unit box, so every
                side must be finite.
            minimize: the acquisition's inner optimizer, shape
                ``MinimizeFn``. Owned by the caller: the descent recipe is the
                engine's, not this model's.
            ell: per-dimension lengthscale in unit-box units, a scalar or a
                ``(d,)`` array.
            kappa: fraction of the box diagonal the projection radius spans.
                Must lie in ``(0, 1)``; without that tie a high-dimensional box
                pushes every corner past the pole and the posterior collapses
                there.
            var_prior: weight-prior variance (the ridge scale opposite `noise`).
            noise: observation-noise variance added to the posterior mean's
                self-covariance.
            key: the PRNG key `propose` splits every ask from.

        Raises:
            ValueError: If the box is absent, unbounded on any side, or has a
                NaN side; if `n_points` or any hyperparameter is out of range.
            TypeError: If `minimize` is not callable.

        """
        check_bounds(bounds)
        box = box_bounds(bounds)
        if box is None or not finite_box(box):
            msg = (
                "SphericalLinearModel normalizes the search box, so it needs a "
                f"finite box: got {bounds!r}"
            )
            raise ValueError(msg)
        if not callable(minimize):
            msg = f"minimize must be callable, got {minimize!r}"
            raise TypeError(msg)
        lo, hi = box
        dtype = jnp.promote_types(jnp.result_type(lo, hi), jnp.float32)
        self._lo = jnp.asarray(lo, dtype=dtype)
        self._hi = jnp.asarray(hi, dtype=dtype)
        self.d = int(self._lo.shape[0])
        self._bounds = bounds
        self._minimize = minimize
        self._width = self._hi - self._lo
        self._var_prior = _check_positive("var_prior", var_prior)
        self._noise = _check_positive("noise", noise)
        kappa_v = _check_positive("kappa", kappa)
        if kappa_v >= 1.0:
            msg = (
                "kappa ties the projection radius to the box diagonal, so it "
                f"must lie in (0, 1); got {kappa!r}"
            )
            raise ValueError(msg)
        self._ell = _check_lengthscale(ell, self.d, dtype)
        # The global lengthscale is the piece that makes the projection scale-
        # free: it maps the box's farthest corner to radius sqrt(2*kappa) for any
        # d, so a wider box does not push every corner to the pole.
        self._glob_ls = jnp.sqrt(kappa_v * jnp.sum((0.5 / self._ell) ** 2))
        self._key = key
        self._xs: list[jnp.ndarray] = []
        self._fs: list[float] = []

    def observe(
        self,
        x: jnp.ndarray,
        f: list[float],
        g: jnp.ndarray | None,  # noqa: ARG002 - the seam carries it; the fit is f-only
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
            ValueError: If the batch has a different number of rows than values.

        """
        rows = jnp.atleast_2d(jnp.asarray(x))
        values = [float(v) for v in f]
        if rows.shape[0] != len(values):
            msg = f"observe got {rows.shape[0]} rows but {len(values)} values"
            raise ValueError(msg)
        for row, value in zip(rows, values, strict=True):
            if math.isfinite(value):
                self._xs.append(row)
                self._fs.append(value)

    def propose(self, n_points: int) -> jnp.ndarray:
        """Return `n_points` restart points: a uniform draw until a fit is possible.

        Fewer than two observations cannot be standardized (one point has no
        spread), so until that evidence arrives the answer is the uniform draw
        the default model would give - the engine only ever sees the points.
        Afterwards the answer is the injected `minimize`'s local minimizers of
        ``-log EI`` over the box; the log form keeps the acquisition's gradient
        alive in the region where the raw improvement has decayed to zero.

        Raises:
            ValueError: If `n_points` is below one, or if `minimize` returns
                anything but a ``(n_points, d)`` array.

        """
        if n_points < 1:
            msg = f"propose needs at least one point, got {n_points}"
            raise ValueError(msg)
        self._key, sub = jax.random.split(self._key)
        if len(self._xs) < 2:
            return jax.random.uniform(
                sub,
                (n_points, self.d),
                minval=self._lo,
                maxval=self._hi,
            )
        beta, inv_gram, best = self._posterior()

        # shortcut: the minimizers are independent, so a flat acquisition can
        # return a cluster on one face. Upgrade to greedy conditioning on the
        # points already picked when a cohort asks for many at once.
        def acquisition(point: jnp.ndarray) -> jnp.ndarray:
            mean, sigma = self._posterior_at(point[None, :], beta, inv_gram)
            z = (best - mean[0]) / sigma[0]
            return -(jnp.log(sigma[0]) + _log_h(z))

        starts = jnp.asarray(self._minimize(acquisition, self._bounds, sub, n_points))
        if starts.shape != (n_points, self.d):
            msg = (
                "minimize must return a "
                f"({n_points}, {self.d}) array, got shape {starts.shape}"
            )
            raise ValueError(msg)
        return starts

    def predict(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return the posterior mean and standard deviation at `x` rows.

        The surrogate's own query point: `propose` builds its acquisition from
        it, so a caller inspecting the fit reads the same posterior. Answers in
        the engine's coordinates but the *standardized* value scale (the fit
        centers and scales the observed values), which is why the acquisition is
        the caller that consumes it.

        Raises:
            ValueError: If fewer than two points have been observed - the fit
                has no spread to standardize by.

        """
        if len(self._xs) < 2:
            msg = "predict needs at least two observed points"
            raise ValueError(msg)
        beta, inv_gram, _ = self._posterior()
        return self._posterior_at(jnp.atleast_2d(jnp.asarray(x)), beta, inv_gram)

    def _features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Inverse-stereographic features of the normalized, lengthscaled rows.

        Maps ``x`` into the unit sphere: ``[2z; |z|^2 - 1] / (|z|^2 + 1)``. Every
        row has norm one, which is what bounds the linear kernel's inner product
        and with it the posterior variance far from data.
        """
        unit = (x - self._lo) / self._width
        z = (unit - 0.5) / self._ell / self._glob_ls
        sq = jnp.sum(z * z, axis=-1, keepdims=True)
        return jnp.concatenate([2.0 * z, sq - 1.0], axis=-1) / (sq + 1.0)

    def _posterior(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Closed-form weight posterior over standardized values, and its best.

        Returns (beta, inv_gram, best) for the ridge form
        ``(Phi^T Phi + noise/var_prior I) beta = Phi^T y``.
        """
        x_obs = jnp.stack(self._xs)
        f_obs = jnp.asarray(self._fs, dtype=x_obs.dtype)
        values = (f_obs - f_obs.mean()) / (f_obs.std() + 1e-8)
        features = self._features(x_obs)
        gram = features.T @ features + (self._noise / self._var_prior) * jnp.eye(
            self.d + 1,
            dtype=features.dtype,
        )
        inv_gram = jnp.linalg.inv(gram)
        beta = inv_gram @ features.T @ values
        return beta, inv_gram, values.min()

    def _posterior_at(
        self,
        x: jnp.ndarray,
        beta: jnp.ndarray,
        inv_gram: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Posterior mean and deviation of the standardized value at ``(m, d)`` rows."""
        features = self._features(jnp.atleast_2d(x))
        mean = features @ beta
        var = self._noise * (
            1.0 + jnp.einsum("ij,jk,ik->i", features, inv_gram, features)
        )
        return mean, jnp.sqrt(jnp.maximum(var, 1e-12))


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


def _check_positive(name: str, value: float) -> float:
    """Return `value` as a float, refusing a non-finite or non-positive knob."""
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        msg = f"{name} must be a finite positive number, got {value!r}"
        raise ValueError(msg)
    return v


def _check_lengthscale(
    ell: float | jnp.ndarray,
    d: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Broadcast the per-dimension lengthscale to ``(d,)``, refusing a bad one."""
    lengthscale = jnp.broadcast_to(jnp.asarray(ell, dtype=dtype), (d,))
    if not bool(jnp.all(jnp.isfinite(lengthscale))) or not bool(
        jnp.all(lengthscale > 0.0),
    ):
        msg = f"ell must be a finite positive scalar or (d,) array, got {ell!r}"
        raise ValueError(msg)
    return lengthscale
