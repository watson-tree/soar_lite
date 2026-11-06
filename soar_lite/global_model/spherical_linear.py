"""A Bayesian linear-regression surrogate as the engine's restart model.

The second `GlobalModel` concrete, beside `SampleGlobalModel`'s model-free draw.
It fits a closed-form weight posterior over the feature rows a `Projection`
hands it and proposes restart points by descending the *log* expected
improvement of that posterior.

The geometry lives entirely in the injected `Projection`, so the model itself
is projection-agnostic: `SphereProjection` bends a chosen subset of the
coordinates through inverse stereography, and `IdentityProjection` reads the
coordinates as their own features. The two are two priors over where the optimum
sits - the bend pulls the corners toward the pole and centers the bowl, the
identity puts the center nowhere special - and the caller picks the one its
belief deserves. A dimension may be unbounded: inverse stereography takes all of
``R``, so the same bend applies to a line, and `scale` states each dimension's
prior width in coordinate units rather than reading a box width as a unit of
measure.

The structure of the sphere is a fixed-hyperparameter port of Doumont et al.
(2026), "We Still Don't Understand High-Dimensional Bayesian Optimization",
AISTATS, PMLR 300:2629-2637 (reference implementation
``github.com/colmont/linear-bo``):
a linear kernel read after projecting onto the unit sphere, a box-diagonal
lengthscale normalization, and a posterior variance bounded away from the prior
away from data. Two things differ. That implementation fits its lengthscale,
coefficient and noise hyperparameters by marginal likelihood (it is gpytorch;
this package is jax-only), so here the same knobs are constructor values. And it
descends its acquisition with a shipped multi-start L-BFGS-B, where here the
minimizer is injected (`MinimizeFn`) so a driver can descend with this library's
own n-path engine.

What the bend keeps is the part that cures the corner-seeking pathology under a
Euclidean RBF posterior: the variance returns to the prior far from data and
expected improvement becomes variance-dominated, pointing at the box corners.
Bounding the feature rows makes improvement follow the fitted values instead -
a center-biased prior, which is why `IdentityProjection` exists for tasks whose
optimum lives on a boundary.
"""

import abc
import math
from collections.abc import Callable, Sequence
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import log_ndtr

from soar_lite.domain import Bounds, Domain
from soar_lite.engine import GlobalModel

AcquisitionFn: TypeAlias = Callable[
    [jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray],
]
"""What `propose` minimizes: ``-log EI`` and its per-row gradient.

Takes ``(M, d)`` rows and returns ``(value: (M,), grad: (M, d))``. Lower is
better, the sign already carried in the value, so a minimizer needs no special
case. Batched, so a zero-order minimizer reads the values off a whole grid in
one call and a first-order one gets the gradient it needs alongside.
"""

MinimizeFn: TypeAlias = Callable[
    [AcquisitionFn, Bounds, int, jnp.ndarray],
    jnp.ndarray,
]
"""The inner minimizer `propose` hands its acquisition to.

``minimize(acquisition, bounds, n_points, key)`` returns up to ``n_points``
local minimizers of the acquisition over ``bounds``, one per row, shape
``(n_points, d)``. The count comes before the seed, as at every entry here: what
to ask for is the caller's question, the key only supplies the randomness.

Injected by the composition root rather than imported, so this module never
reaches back into the engine's scheduler - the engine already owns the descent
recipe, and a second one here would be a second answer to the same question.
"""

ProjectionFactory: TypeAlias = Callable[[Domain], "Projection"]
"""Build the `Projection` a model reads one domain through.

Geometry is a pure function of the domain - a center, a width, one lengthscale -
so the model carries the recipe rather than one projection bound to one box:
`partial(SphereProjection, kappa=..., mask=...)` for the bent arm,
`IdentityProjection` itself for the straight one. Called once per ask with the
box that ask was handed, so the geometry is always the caller's box and never a
projection bound to an older one.
"""

_HALF_LOG_2PI = 0.5 * jnp.log(2.0 * jnp.pi)


class Projection(abc.ABC):
    """The feature map a restart model reads domain coordinates through.

    A projection wraps a `Domain` and adds one map: `features` turns every
    ``(m, d)`` batch of coordinates into an ``(m, w)`` batch of posterior
    feature rows - `w` is this projection's width, read off `features` by the
    model rather than assumed ``d + 1``.

    The domain stays the domain's: a projection *reads* its box, scales and
    draws; it is not one. That keeps the two jobs apart at the type level - a
    caller who only samples builds a `Domain` and never meets a projection, and
    a projection held where a `Domain` is expected is a wiring error instead of
    a silently accepted one - and keeps `scale` and `samplers` out of this
    constructor, where every fitted model would otherwise have restated them.

    `features` is the one contract: a pure query of the coordinates, the same
    ``(m, d)`` input mapping to the same ``(m, w)`` rows, with `w` stable across
    calls for one projection. It is also the seam a surrogate of another family
    reads - nothing in a projection is specific to `BLRGlobalModel`'s posterior,
    since it carries geometry, not fit state.
    """

    def __init__(self, domain: Domain) -> None:
        """Bind the domain the geometry reads.

        Args:
            domain: the box, per-dimension `scale` and draws the projection
                reads its geometry off - center, width and which sides are
                unbounded. Holding one domain rather than being one keeps the
                geometry and the fallback draw in the same coordinate unit with
                no second knob to keep in step. Validated by `Domain`'s own
                constructor.

        """
        self.domain = domain

    @abc.abstractmethod
    def features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return the ``(m, w)`` feature rows for ``(m, d)`` coordinates.

        `w` is this projection's feature width: the model reads it off the
        returned shape, never off `d`, so a projection is free to widen or
        narrow the representation however its geometry needs.
        """
        ...


class SphereProjection(Projection):
    """Inverse-stereographic projection of a chosen subset of the dimensions.

    The restart models read the domain in whatever coordinates the objective
    takes. This maps them onto the unit sphere, ``[2z; |z|^2 - 1] / (|z|^2 + 1)``,
    where ``z`` is the coordinates centered, divided by the per-dimension
    `scale` and by one global lengthscale. Two properties matter downstream.

    Every projected row has norm one, so a linear head read on these features has
    a constant self inner product: its posterior variance stays bounded far from
    data instead of climbing back to the prior. That climb is the corner-seeking
    pathology the projection exists to cure - variance growing with distance
    reads as unexplored room, and an acquisition spends its asks on the box
    corners. The last coordinate, ``(|z|^2 - 1)/(|z|^2 + 1)``, is then a
    radially symmetric, saturating feature of the input - the one extra degree of
    freedom a linear head gets, an isotropic bowl whose center the weights place.

    `mask` keeps that bowl off dimensions whose natural prior is uniform - an
    angle, say, wants no attraction to the center. A left-out dimension keeps its
    coordinate and enters a linear head as a plain trend, so the feature vector is
    ``d + 1`` wide however many dimensions the projection bends.

    A dimension may be unbounded: inverse stereography takes all of ``R``, so the
    same map bends a line, and its far end returns to the pole - the variance
    goes back to the prior instead of growing with ``|x|``, which is what an
    unbent coordinate does.
    """

    def __init__(
        self,
        domain: Domain,
        *,
        kappa: float = 0.5,
        mask: jnp.ndarray | Sequence[bool] | None = None,
    ) -> None:
        """Bind the domain and the projected subset.

        Args:
            domain: the `Domain` the projection normalizes; its `scale` is each
                dimension's prior width in coordinate units. Only the ratios
                between the bent dimensions' entries move the features - the
                projection derives one global lengthscale from `scale` and
                `kappa` and divides it back out - but the common magnitude is
                what the fallback draw and an unbounded dimension's radius are
                measured in.
            kappa: the tie between the projection's radius and the domain - a
                bent dimension's far end lands at ``|z|^2 = 1/kappa`` whatever
                ``d`` is. Must lie in ``(0, 1)``; without such a tie a
                high-dimensional box pushes every corner past the pole and the
                posterior collapses there.
            mask: which dimensions the projection bends, a ``(d,)`` boolean
                selector; ``None`` bends all of them.

        Raises:
            ValueError: If `kappa` is out of range, or `mask` is not a ``(d,)``
                boolean selector selecting at least one dimension. The domain is
                validated by its own constructor.

        """
        # Validate before building: the geometry knobs are this constructor's,
        # the domain was validated when it was made.
        self.mask = _check_mask(mask, domain.d)
        kappa_v = _check_finite_positive("kappa", kappa)
        if kappa_v >= 1.0:
            msg = (
                "kappa ties the projection's radius to the domain, so it must "
                f"lie in (0, 1); got {kappa!r}"
            )
            raise ValueError(msg)
        super().__init__(domain)
        # The global lengthscale is what makes the projection dimension-free: a
        # bent dimension's reference radius, divided by `glob_ls`, is
        # `1/sqrt(kappa)` in total, so `d` raises the features' typical radius
        # without ever pushing the farthest point past the pole. A bounded
        # dimension's radius is half its width in `scale` units; an unbounded
        # one has no width, so its radius is one `scale`. Summed over the bent
        # dimensions only - a dimension the projection leaves out has no radius
        # to tie.
        radius = jnp.where(
            domain.bounded,
            domain.width / (2.0 * domain.scale),
            1.0,
        )
        tied = jnp.sum(jnp.where(self.mask, radius * radius, 0.0))
        self.glob_ls = jnp.sqrt(kappa_v * tied)

    def features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return the ``(m, d + 1)`` feature rows for ``(m, d)`` coordinates.

        Every dimension is first read in its own coordinate unit,
        ``(x - center) / scale``, so no column inherits a box width as its unit.
        The bent dimensions become their stereographic image on the unit sphere,
        divided by the global lengthscale; the rest keep that centered, scaled
        coordinate, so a linear head reads them as a plain linear trend.
        """
        z = (x - self.domain.center) / self.domain.scale
        bent = z[:, self.mask] / self.glob_ls
        sq = jnp.sum(bent * bent, axis=-1, keepdims=True)
        sphere = jnp.concatenate([2.0 * bent, sq - 1.0], axis=-1) / (sq + 1.0)
        return jnp.concatenate([sphere, z[:, ~self.mask]], axis=-1)


class IdentityProjection(Projection):
    """The no-geometry projection: coordinates are their own features.

    `features` returns the coordinates unchanged - width ``d``, no center bowl,
    no normalization to the unit box. Where `SphereProjection` bends the corners
    toward the pole and seats the center at the south pole (a prior that the
    optimum lives near the middle), this one keeps every coordinate's own scale
    and range and puts the center nowhere special: a prior that the optimum may
    well sit on a boundary or a corner.

    The cost of no geometry is no bound. The posterior variance is a quadratic
    form in these unbent features, so it grows toward the corners and expected
    improvement on this arm is extrapolation-dominated, pointing at corners -
    exactly the behavior wanted when the truth lives there, and the one
    `SphereProjection` exists to cure when it does not.

    `features` ignores the domain's `scale` outright: the coordinates arrive as
    they are, and only the domain's `draw` reads it, to place the fallback
    starts.
    """

    def features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return the coordinates unchanged: the identity feature map."""
        return x


class BLRGlobalModel(GlobalModel):
    """Restart points from a Bayesian linear posterior's log improvement.

    `observe` accumulates the run's served batches (values only - the gradient is
    not fitted), and `propose` fits the closed-form weight posterior over those
    points and returns the minimizers of ``-log EI`` the injected `minimize`
    finds. Below two observations, and with every value equal, there is no spread
    to standardize by, so it answers with the domain's own `draw`.

    Every knob is a fixed hyperparameter, not a fitted one. The geometry lives in
    the `Projection` this model builds for each ask - `SphereProjection` for the
    center-biased bend, `IdentityProjection` for no prior on where the optimum
    sits - and it is built from the domain that ask hands in. So the model
    carries the recipe and not a box: one instance asked inside a different box
    has no stale geometry and no second normalization drifting behind it. What
    this class adds on top is the value fit and the acquisition that descends it;
    the feature width is read off `features`, so any projection width fits.

    The projection is optional. Left unset, the bent arm is the default, because
    that is the reference model - the one whose geometry cures the corner-seeking
    pathology. `IdentityProjection` stays the explicit alternate arm, for a task
    whose optimum lives on a boundary; it is a different prior, not a stand-in
    for "no projection".
    """

    def __init__(
        self,
        projection_factory: ProjectionFactory | None = None,
        *,
        minimize: MinimizeFn,
        var_prior: float = 1.0,
        noise: float = 0.01,
        key: jnp.ndarray,
    ) -> None:
        """Bind the geometry recipe, the injected minimizer and the value fits.

        Args:
            projection_factory: builds the `Projection` a given `Domain` is read
                through - ``partial(SphereProjection, mask=..., kappa=...)`` for
                a custom bend, `IdentityProjection` itself for the unbent arm.
                ``None`` (the default) is `SphereProjection` with its own
                defaults, the reference model's center-biased bend. Called once
                per `propose`, so the geometry is always the box that ask was
                handed, and `Projection`'s own constructor validates the domain.
            minimize: the acquisition's inner optimizer, shape ``MinimizeFn``.
                Supplied by the caller: this model holds the recipe and calls
                it, and never picks one itself.
            var_prior: weight-prior variance (the ridge scale opposite `noise`).
            noise: observation-noise variance added to the posterior mean's
                self-covariance.
            key: the PRNG key `propose` splits every ask from.

        Raises:
            ValueError: If `var_prior` or `noise` is not a finite positive
                value.

        """
        self._projection_factory = projection_factory
        self._minimize = minimize
        self._var_prior = _check_finite_positive("var_prior", var_prior)
        self._noise = _check_finite_positive("noise", noise)
        self._key = key
        # Folded per batch, not per row: every ask walks this list once, and one
        # element per served batch instead of one per point keeps that walk cheap
        # on a long run.
        self._xs: list[jnp.ndarray] = []
        self._fs: list[jnp.ndarray] = []
        # The value moments as scalars, merged per batch. Held as numbers rather
        # than as the arrays they summarize so that standardizing costs one
        # fixed-shape read: an array whose length grows with the run recompiles
        # the fit on every ask instead of reusing it. Three of the four are what
        # they say over the kept rows - their sum, their squared deviation about
        # that sum's mean, the smallest - and the count is what each is read
        # against.
        self._n = 0
        self._sum = 0.0
        self._m2 = 0.0
        self._min = math.inf

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
        before it stops, and one such row would drag the standardization with it.
        A point served twice is kept twice - under the tiny observation noise
        that is a slight double weight, which is what a second look at the same
        coordinate is worth.

        Raises:
            ValueError: If `x` is not a two-dimensional batch, has a different
                number of rows than values, or is wider or narrower than the
                first batch folded in.

        """
        del g

        rows = jnp.asarray(x)
        if rows.ndim != 2:
            msg = f"observe needs an (m, d) batch, got shape {rows.shape}"
            raise ValueError(msg)
        if rows.shape[0] != len(f):
            msg = f"observe got {rows.shape[0]} rows but {len(f)} values"
            raise ValueError(msg)
        # The model holds no `d` to check against: the first batch is what fixes
        # the fit's width, and a later batch of another width would otherwise
        # only fail at `jnp.stack` inside the fit.
        if self._xs and rows.shape[1] != self._xs[0].shape[-1]:
            msg = (
                "observe's first batch fixed the width at "
                f"{self._xs[0].shape[-1]}, got {rows.shape[1]}"
            )
            raise ValueError(msg)
        values = jnp.asarray(f, dtype=rows.dtype)
        keep = jnp.isfinite(values)
        if bool(keep.any()):
            self._xs.append(rows[keep])
            self._fs.append(values[keep])
            self._fold_moments(values[keep])

    def propose(self, n_points: int, domain: Domain) -> jnp.ndarray:
        """Return `n_points` restart points: a draw until a fit is possible.

        The box is the caller's, handed in per ask: the geometry is rebuilt from
        it here, and nothing about it is remembered between calls. Until the fit
        has a spread to standardize, the answer is the domain's own `draw` -
        uniform inside a bounded dimension, the domain's prior breadth inside an
        unbounded one. Afterwards it is the injected `minimize`'s local
        minimizers of ``-log EI`` over the box; the log form keeps the
        acquisition's gradient alive where the raw improvement has decayed to
        zero.

        Raises:
            ValueError: If `n_points` is below one, or if `minimize` returns
                anything but a ``(n_points, d)`` array.

        """
        if n_points <= 0:
            msg = f"propose needs at least one point, got {n_points}"
            raise ValueError(msg)
        self._key, sub = jax.random.split(self._key)
        projection = self._projection_for(domain)
        if not self._can_fit():
            return domain.draw(sub, n_points)
        weights, cov = self._posterior(projection)
        best = self._standardize(jnp.asarray(self._min))

        # shortcut: the minimizers are independent, so a flat acquisition can
        # return a cluster on one face. Upgrade to greedy conditioning on the
        # points already picked when a cohort asks for many at once.
        def acquisition(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            return self._acq_value_and_grad(projection, x, weights, cov, best)

        starts = jnp.asarray(
            self._minimize(acquisition, domain.bounds, n_points, sub),
        )
        if starts.shape != (n_points, domain.d):
            msg = (
                "minimize must return a "
                f"({n_points}, {domain.d}) array, got shape {starts.shape}"
            )
            raise ValueError(msg)
        return starts

    def _predict(
        self,
        x: jnp.ndarray,
        domain: Domain,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return the posterior mean and standard deviation at `x` rows.

        A read-only diagnostic kept `_`-private for the tests: the engine never
        calls it, and `propose` reaches the posterior directly rather than
        through this method, so it is not part of the `GlobalModel` contract a
        caller implements. The domain is asked for on the spot, exactly as
        `propose` asks for it, since the model holds no box. It also answers only
        in the *standardized* value scale - the fit centers and scales the
        observed values - which is one more reason it stays off the public face.

        Raises:
            ValueError: If the fit has no spread to standardize by yet, or if
                `x` is not a batch of the fit's width.

        """
        if not self._can_fit():
            msg = "a posterior read needs a spread to standardize by"
            raise ValueError(msg)
        batch = jnp.asarray(x)
        width = self._xs[0].shape[-1]
        if batch.ndim != 2 or batch.shape[1] != width:
            msg = f"a posterior read needs a (m, {width}) batch, got {batch.shape}"
            raise ValueError(msg)
        projection = self._projection_for(domain)
        weights, cov = self._posterior(projection)
        return self._posterior_at(projection, batch, weights, cov)

    def _projection_for(self, domain: Domain) -> Projection:
        """Build this ask's geometry: the caller's recipe, or the bent default."""
        if self._projection_factory is None:
            return SphereProjection(domain)
        return self._projection_factory(domain)

    def _can_fit(self) -> bool:
        """Return whether the values have a spread to standardize by.

        One observation has no spread of its own, and neither has a batch of
        equal values: both leave nothing to divide by and no improvement to
        measure against.
        """
        return self._n > 1 and self._m2 > 0

    def _values(self) -> jnp.ndarray:
        """Return every kept value, in the order the batches were folded in."""
        return jnp.concatenate(self._fs)

    def _fold_moments(self, kept: jnp.ndarray) -> None:
        """Merge one batch's count, sum, spread and minimum into the fit's.

        Chan's parallel merge, so each batch's spread is measured against its own
        mean before the two are combined. The one-pass ``mean of squares minus
        square of mean`` loses the spread exactly where a converged run spends
        its batches - values that agree to most of their digits - and can come
        out negative there. The value sum needs no such care: it is carried raw,
        and the center is that sum over the count.
        """
        batch = np.asarray(kept, dtype=np.float64)
        count = batch.size
        batch_mean = float(batch.mean())
        batch_m2 = float(((batch - batch_mean) ** 2).sum())
        total = self._n + count
        delta = batch_mean - (self._sum / self._n if self._n else 0.0)
        self._sum += float(batch.sum())
        self._m2 = self._m2 + batch_m2 + delta**2 * self._n * count / total
        self._n = total
        self._min = min(self._min, float(batch.min()))

    def _moments(self) -> tuple[float, float]:
        """Return the fit's value center and spread, both as Python floats.

        Both are read off the same pair of sums - center is the kept values'
        mean, spread their population standard deviation - so neither field
        carries a normalization the other does not.
        """
        return self._sum / self._n, math.sqrt(self._m2 / self._n)

    def _standardize(self, f: jnp.ndarray) -> jnp.ndarray:
        """Return `f` on the fit's standardized scale.

        The center and spread are the accumulated moments', not `f`'s own: the fit
        has one value scale, and everything read off it answers in that one. Only
        reached once `_can_fit` found a non-zero spread, so the divisor is never
        zero and needs no floor.
        """
        center, spread = self._moments()
        return (f - center) / spread

    def _posterior(self, projection: Projection) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return the weight posterior's mean and covariance.

        Written as the precision it is: ``P^T P / noise + I / var_prior`` is
        likelihood plus prior, and one inversion gives the covariance matrix
        itself, with ``weights = cov @ P^T y / noise`` its mean. `projection` is
        the geometry built for this ask's domain; nothing about it is remembered.

        The rows are walked one served batch at a time and three sums kept,
        rather than concatenated into one array and multiplied once: the products
        are the same, but their shapes stop growing with the run, and a shape that
        changes is a fresh compilation of the whole fit. Standardizing cannot ride
        along in the same accumulation, since the center and spread are the fit's
        rather than the batch's, so the values cross is carried raw and shifted
        here: the spread divides outside the sum, the center scales the column
        sums.
        """
        gram, cross, colsum = self._fold_gram(projection)
        precision = (
            gram / self._noise
            + jnp.eye(gram.shape[0], dtype=gram.dtype) / self._var_prior
        )
        cov = jnp.linalg.inv(precision)
        center, spread = self._moments()
        weights = cov @ ((cross - center * colsum) / spread) / self._noise
        return weights, cov

    def _fold_gram(
        self,
        projection: Projection,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return ``P^T P``, ``P^T y`` and ``P^T 1`` over the whole evidence."""
        blocks = zip(self._xs, self._fs, strict=True)
        x_0, f_0 = next(blocks)
        first = projection.features(x_0)
        gram = first.T @ first
        cross = first.T @ f_0.astype(first.dtype)
        colsum = first.sum(axis=0)
        for x_b, f_b in blocks:
            features = projection.features(x_b)
            gram = gram + features.T @ features
            cross = cross + features.T @ f_b.astype(features.dtype)
            colsum = colsum + features.sum(axis=0)
        return gram, cross, colsum

    def _posterior_at(
        self,
        projection: Projection,
        x: jnp.ndarray,
        weights: jnp.ndarray,
        cov: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Posterior mean and deviation of the standardized value at ``(m, d)`` rows.

        The variance is the observation noise plus the covariance's quadratic
        form, so it stays above `noise` and its root needs no floor.
        """
        features = projection.features(x)
        mean = features @ weights
        var = self._noise + jnp.einsum("ij,jk,ik->i", features, cov, features)
        return mean, jnp.sqrt(var)

    def _acq_value_and_grad(
        self,
        projection: Projection,
        x: jnp.ndarray,
        weights: jnp.ndarray,
        cov: jnp.ndarray,
        best: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``-log EI`` and its per-row gradient at ``(m, d)`` rows.

        `jax.value_and_grad` differentiates a scalar output only, and the rows of
        the acquisition are independent, so the gradient is taken one row at a
        time under `vmap`. The batch Jacobian is diagonal, and this reads only its
        diagonal where `jax.jacobian` would build the whole ``m x m`` block and
        throw all but that away.
        """

        def scalar(row: jnp.ndarray) -> jnp.ndarray:
            mean, sigma = self._posterior_at(projection, row[None, :], weights, cov)
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
    and the improvement is negligible there either way.
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
    mask: jnp.ndarray | Sequence[bool] | None,
    d: int,
) -> jnp.ndarray:
    """Normalize `mask` to a ``(d,)`` boolean selector, refusing a bad one.

    Only the boolean spelling is accepted; the integer-index spelling is
    indistinguishable from it at runtime, so it is not offered at all.
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
    if given.dtype != jnp.bool_:
        msg = f"mask must be a boolean selector, got dtype {given.dtype}"
        raise ValueError(msg)
    if given.shape[0] != d:
        msg = f"mask needs {d} boolean entries, got {given.shape[0]}"
        raise ValueError(msg)
    if not bool(jnp.any(given)):
        msg = "mask must project at least one dimension"
        raise ValueError(msg)
    return given
