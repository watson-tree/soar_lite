"""A Bayesian linear-regression surrogate as the engine's restart model.

The second `GlobalModel` concrete, beside `global_model.py`'s uniform draw. It
fits a closed-form weight posterior over the feature rows a `Projection` hands
it and proposes restart points by descending the *log* expected improvement of
that posterior - the sample-based restart the uniform draw only approximates.

The geometry lives entirely in the injected `Projection`, so the model itself
is projection-agnostic: `SphereProjection` bends a chosen subset of the box's
dimensions through inverse stereography, and `IdentityProjection` reads the
coordinates as their own features. The two are two priors over where the
optimum sits - the bend pulls the corners toward the pole and centres the bowl,
the identity keeps every coordinate's own scale and puts the centre nowhere
special - and the caller picks the one the task's belief deserves. A dimension
may be unbounded: inverse stereography takes all of ``R``, so the same bend
applies to a line, and `scale` states each dimension's prior width in
coordinate units rather than reading a box width as if it were a unit of
measure.

The structure of the sphere is a fixed-hyperparameter port of Doumont, Fan, et
al., "We Still Don't Understand High-Dimensional Bayesian Optimization" (2025),
``github.com/colmont/linear-bo``: a linear kernel read after projecting onto the
unit sphere, a box-diagonal lengthscale normalization, and the resulting
posterior variance bounded away from the prior away from data. Two things are
deliberately dropped. That implementation fits its lengthscale, coefficient and
noise hyperparameters by marginal likelihood (it is gpytorch; this package is
jax-only), so here the same knobs are constructor values instead. And it
descends its acquisition with a shipped multi-start L-BFGS-B; here the
minimizer is injected instead, so a driver can descend with the library's own
n-path engine rather than a second recipe - see `MinimizeFn`.

What the bend keeps is the part that cures the corner-seeking pathology under a
Euclidean RBF posterior: the variance returns to the prior far from data and
expected improvement becomes variance-dominated, pointing at the box corners.
Bounding the feature rows makes improvement follow the fitted values instead -
a centre-biased prior, which is why `IdentityProjection` exists for tasks whose
optimum lives on a boundary.
"""

import abc
import math
from collections.abc import Callable, Sequence
from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, ndtr, ndtri

from soar_lite.engine import Bounds, GlobalModel, box_bounds, check_bounds

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

# One dimension's fallback start distribution, injected per dimension so a
# restart model can draw where a finite box does not reach: `sampler(key, n)`
# returns `n` coordinates for that one dimension. `None` leaves the projection's
# own default in place - uniform inside a bounded side, ``N(0, scale_i^2)``
# where the domain is unbounded.
DrawFn: TypeAlias = Callable[[jnp.ndarray, int], jnp.ndarray]

_HALF_LOG_2PI = 0.5 * jnp.log(2.0 * jnp.pi)


class Projection(abc.ABC):
    """The map a restart model reads box coordinates through.

    A projection turns every ``(m, d)`` batch of coordinates into an ``(m, w)``
    batch of posterior feature rows - `w` is this projection's width, read off
    `features` by the model rather than assumed ``d + 1``. The box is shared:
    `features` centres and scales against it and `draw` samples from it, so the
    projection carries the box and every subclass gets the same validation. A
    subclass decides only what the feature rows look like - the geometry, its
    centre bias or none - not the box bookkeeping.

    A side may be unbounded. The box then declares only the dimension and which
    sides are unbounded; `scale` is what carries a length there, because no
    width exists to read one off. Not even a bounded side's width is treated as
    a natural unit: `scale` is in coordinate units, so a caller changing a
    variable from metres to millimetres changes `scale`, never the box.

    `features` is the one contract. It is a pure query of the coordinates: the
    same ``(m, d)`` input maps to the same ``(m, w)`` rows, with `w` stable
    across calls for one projection.
    """

    def __init__(
        self,
        bounds: Bounds,
        *,
        scale: jnp.ndarray | Sequence[float] | None = None,
        samplers: Sequence[DrawFn | None] | None = None,
    ) -> None:
        """Validate and carry the declared box, the scales and the samplers.

        Args:
            bounds: the box, scipy-style pairs or a ``(d, 2)`` array. A side may
                be ``None`` or ``+-inf`` for an unbounded domain; a NaN side, or
                a lower side above its upper, is refused. ``None`` as the whole
                argument is refused too: the box declares the dimension and
                which sides are unbounded, so a projection has nothing to read
                without one.
            scale: per-dimension prior width in coordinate units, a ``(d,)``
                vector or ``None`` for every dimension at one. Subclasses that
                normalize compare the entries to each other, but the common
                magnitude is not free either: `scale` is the length the features
                and the default draw are measured in, so a dimension whose
                coordinates span far more or far less than one wants its own
                entry rather than the default.
            samplers: one optional draw per dimension, ``d`` entries; a ``None``
                entry keeps the default its domain admits - uniform inside a
                bounded dimension, ``N(0, scale_i^2)`` inside a free one, and
                that same normal restricted to the domain inside a
                half-bounded one, where drawing the unrestricted normal would
                step over the one side that is declared. Dimensions are drawn
                independently, so a prior that couples them needs its own
                `GlobalModel`, not a projection.

        Raises:
            ValueError: If the box is absent or malformed, or `scale` is not a
                ``(d,)`` vector of finite positive values, or `samplers` is not
                ``d`` entries long.

        """
        check_bounds(bounds)
        box = box_bounds(bounds)
        if box is None:
            msg = (
                "a projection reads its dimension and its unbounded sides off "
                f"the box it is given, so it needs one: got {bounds!r}"
            )
            raise ValueError(msg)
        self.lo, self.hi = box
        # The dimension is what `bounds` declares: read it off the argument,
        # not off `lo`/`hi`, which `box_bounds` just derived from it.
        self.d = len(bounds)
        self.bounds = bounds
        self.scale = _check_scale(scale, self.d)
        self._samplers = _check_samplers(samplers, self.d)
        # The domain's shape per dimension. An unbounded side has no width, so
        # `width` is zero there and the centre falls back to the origin - the
        # only centre an unbounded dimension has, and `scale` the only length
        # a caller can state about it.
        self.bounded = jnp.isfinite(self.lo) & jnp.isfinite(self.hi)
        # Neither side finite. The half-bounded dimension - one side finite,
        # one not - is the remaining case, and the only one whose default draw
        # has a boundary to respect.
        self.free = ~jnp.isfinite(self.lo) & ~jnp.isfinite(self.hi)
        self.width = jnp.where(self.bounded, self.hi - self.lo, 0.0)
        self.center = jnp.where(self.bounded, self.lo + self.width / 2.0, 0.0)

    @abc.abstractmethod
    def features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return the ``(m, w)`` feature rows for ``(m, d)`` coordinates.

        `w` is this projection's feature width: the model reads it off the
        returned shape, never off `d`, so a projection is free to widen or
        narrow the representation however its geometry needs.
        """
        ...

    def draw(self, key: jnp.ndarray, n_points: int) -> jnp.ndarray:
        """Return ``(n_points, d)`` restart starts, one dimension at a time.

        A dimension with an injected `DrawFn` takes its column from that, else
        from the default its domain admits: uniform inside a bounded side,
        ``N(0, scale_i^2)`` inside a free one, and the same normal *truncated*
        to the domain inside a half-bounded one - a sampler free to step over
        the side the caller declared is not sampling from the declared box at
        all, and truncating is what the normal the default promises looks like
        once it may not cross. That default is also why an injected sampler is
        the caller's own statement about the dimension's spread - nothing checks
        the drawn values against `scale`, since a black-box distribution has no
        readable spread to check them against.

        Raises:
            ValueError: If `n_points` is below one, or a column does not come
                back ``(n_points,)`` long.

        """
        if n_points <= 0:
            msg = f"a draw needs at least one point, got {n_points}"
            raise ValueError(msg)
        columns: list[jnp.ndarray] = []
        for j, sub in enumerate(jax.random.split(key, self.d)):
            sampler = self._samplers[j]
            if sampler is not None:
                column = sampler(sub, n_points)
            elif bool(self.bounded[j]):
                column = jax.random.uniform(
                    sub,
                    (n_points,),
                    minval=self.lo[j],
                    maxval=self.hi[j],
                )
            elif bool(self.free[j]):
                column = self.scale[j] * jax.random.normal(sub, (n_points,))
            else:
                column = _truncated_normal(
                    sub,
                    n_points,
                    lower=float(self.lo[j]),
                    upper=float(self.hi[j]),
                    scale=float(self.scale[j]),
                )
            if jnp.shape(column) != (n_points,):
                msg = (
                    f"each drawn column must be ({n_points},) long, got "
                    f"{jnp.shape(column)} for dimension {j}"
                )
                raise ValueError(msg)
            columns.append(column)
        return jnp.stack(columns, axis=-1)


class SphereProjection(Projection):
    """Inverse-stereographic projection of a chosen subset of the box's dims.

    The restart models read the box in whatever coordinates the objective takes.
    This maps them onto the unit sphere, ``[2z; |z|^2 - 1] / (|z|^2 + 1)``, where
    ``z`` is the box's coordinates centered, divided by the per-dimension
    `scale` and by one global lengthscale. Two properties matter downstream.

    Every projected row has norm one, so a linear head read on these features
    has a constant self inner product: its posterior variance stays bounded far
    from data instead of climbing back to the prior, which is the corner-seeking
    pathology the projection exists to cure. And the last coordinate,
    ``(|z|^2 - 1)/(|z|^2 + 1)``, is a radially symmetric, saturating feature of
    the input - the one extra degree of freedom a linear head gets, an isotropic
    bowl whose center the weights place.

    `mask` keeps that bowl off dimensions whose natural prior is uniform - an
    angle, say, wants no attraction to the box center. A left-out dimension
    keeps its coordinate and enters a linear head as a plain trend, so the
    feature vector is ``d + 1`` wide however many dimensions the projection
    bends.

    A dimension may be unbounded here too: inverse stereography takes all of
    ``R``, so the same map bends a line, and the far end of it returns to the
    pole - the variance goes back to the prior instead of growing with ``|x|``,
    which is what an unbent coordinate does.
    """

    def __init__(
        self,
        bounds: Bounds,
        *,
        scale: jnp.ndarray | Sequence[float] | None = None,
        kappa: float = 0.5,
        mask: jnp.ndarray | Sequence[bool] | None = None,
        samplers: Sequence[DrawFn | None] | None = None,
    ) -> None:
        """Bind the box, the scales and the projected subset.

        Args:
            bounds: the box the projection normalizes, scipy-style pairs or a
                ``(d, 2)`` array; a side may be unbounded.
            scale: per-dimension prior width in coordinate units, a ``(d,)``
                vector or ``None`` for every dimension at one. Only the ratios
                between the bent dimensions' entries move the features, since
                `glob_ls` divides one global lengthscale back out; the common
                magnitude is what the fallback draw and an unbounded
                dimension's radius are measured in.
            kappa: the tie between the projection's radius and the domain - a
                bent dimension's far end lands at ``|z|^2 = 1/kappa`` whatever
                ``d`` is. Must lie in ``(0, 1)``; without such a tie a
                high-dimensional box pushes every corner past the pole and the
                posterior collapses there.
            mask: which dimensions the projection bends, a ``(d,)`` boolean
                selector; ``None`` bends all of them.
            samplers: one optional draw per dimension; see `Projection`.

        Raises:
            ValueError: If `scale` is not a ``(d,)`` vector of finite positive
                values, or `kappa` is out of range, or `mask` is not a ``(d,)``
                boolean selector selecting at least one dimension. The box is
                validated by `Projection`'s constructor.

        """
        super().__init__(bounds, scale=scale, samplers=samplers)
        self.mask = _check_mask(mask, self.d)
        kappa_v = _check_finite_positive("kappa", kappa)
        if kappa_v >= 1.0:
            msg = (
                "kappa ties the projection's radius to the domain, so it must "
                f"lie in (0, 1); got {kappa!r}"
            )
            raise ValueError(msg)
        # The global lengthscale is what makes the projection dimension-free: a
        # bent dimension's reference radius, divided by `glob_ls`, is
        # `1/sqrt(kappa)` in total, so `d` raises the features' typical radius
        # without ever pushing the farthest point past the pole. A bounded
        # dimension's radius is half its width in `scale` units; an unbounded
        # one has no width, so its radius is one `scale`. Summed over the bent
        # dimensions only - a dimension the projection leaves out has no radius
        # to tie.
        radius = jnp.where(self.bounded, self.width / (2.0 * self.scale), 1.0)
        tied = jnp.sum(jnp.where(self.mask, radius * radius, 0.0))
        self.glob_ls = jnp.sqrt(kappa_v * tied)

    def features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return the ``(m, d + 1)`` feature rows for ``(m, d)`` coordinates.

        Every dimension is first read in its own coordinate unit,
        ``(x - center) / scale``, so no column inherits a box width as its unit.
        The bent dimensions become their stereographic image on the unit sphere,
        divided by the global lengthscale; the rest keep that centred, scaled
        coordinate, so a linear head reads them as a plain linear trend.
        """
        z = (x - self.center) / self.scale
        bent = z[:, self.mask] / self.glob_ls
        sq = jnp.sum(bent * bent, axis=-1, keepdims=True)
        sphere = jnp.concatenate([2.0 * bent, sq - 1.0], axis=-1) / (sq + 1.0)
        return jnp.concatenate([sphere, z[:, ~self.mask]], axis=-1)


class IdentityProjection(Projection):
    """The no-geometry projection: coordinates are their own features.

    `features` returns the coordinates unchanged - width ``d``, no centre bowl,
    no normalization to the unit box. Where `SphereProjection` bends the corners
    toward the pole and seats the box centre at the south pole (a prior that the
    optimum lives near the middle), this one keeps every coordinate's own scale
    and range and puts the centre nowhere special: a prior that the optimum may
    well sit on a boundary or a corner.

    The cost of no geometry is no bound. The posterior variance is a quadratic
    form in these unbent features, so it grows toward the box corners and
    expected improvement on this arm is extrapolation-dominated, pointing at
    corners - exactly the behaviour wanted when the truth lives there, and the
    one `SphereProjection` exists to cure when it does not.

    `features` ignores the base class's `scale` and `samplers` outright: the
    coordinates arrive as they are, and only `draw` reads them, to place the
    fallback starts.
    """

    def features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return the coordinates unchanged: the identity feature map."""
        return x


class BLRGlobalModel(GlobalModel):
    """Restart points from a Bayesian linear posterior's log improvement.

    `observe` accumulates the run's served batches (values only - the gradient
    is not fitted), and `propose` fits the closed-form weight posterior over
    those points and returns the minimizers of ``-log EI`` the injected
    `minimize` finds. Below two observations - and with every value equal -
    there is no spread to standardize by, so it answers with the uniform draw
    the default model would give.

    Every knob is a fixed hyperparameter, not a fitted one; the defaults are the
    reference implementation's starting points rather than its marginal-
    likelihood optimum. The geometry lives in the injected `Projection` -
    `SphereProjection` for the centre-biased bend of the reference model,
    `IdentityProjection` for no prior on where the optimum sits - and the box
    comes with it. What this class adds on top is the value fit and the
    acquisition that descends it; the feature width is read off `features`, so
    any projection width fits.
    """

    def __init__(
        self,
        projection: Projection,
        *,
        minimize: MinimizeFn,
        var_prior: float = 1.0,
        noise: float = 0.01,
        key: jnp.ndarray,
    ) -> None:
        """Bind the projection, the injected minimizer and the value fits.

        Args:
            projection: the `Projection` turning a coordinate into the
                posterior's feature row. It is the authority for the run's
                dimension and search box - `d`, `bounds` and the fallback
                draw's corners are read off it, and `Projection`'s constructor
                already validated the box; a `SphereProjection` further
                validates its own `scale`, `kappa` and `mask`.
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
        """Return `n_points` restart points: a draw until a fit is possible.

        Until the fit has a spread to standardize, the answer is the
        projection's own `draw` - the points the engine's uniform model would
        give inside a finite box, and the prior's own breadth outside one.
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
            return self._projection.draw(sub, n_points)
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
            + jnp.eye(features.shape[1], dtype=features.dtype) / self._var_prior
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
    mask: jnp.ndarray | Sequence[bool] | None,
    d: int,
) -> jnp.ndarray:
    """Normalize `mask` to a ``(d,)`` boolean selector, refusing a bad one.

    Only the boolean spelling is accepted. The integer-index spelling was
    refused because it was indistinguishable from a boolean vector at runtime:
    ``bool`` is a subclass of ``int``, so ``asarray`` names the dtype from the
    values, and a ``[1, 0, 1, 0]`` meant as booleans would be read as indices
    and project a different set silently. A caller naming a few dimensions
    writes the boolean vector instead, which is also the form the feature
    gather consumes.
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


def _check_scale(scale: jnp.ndarray | Sequence[float] | None, d: int) -> jnp.ndarray:
    """Normalize the per-dimension prior width to ``(d,)``, refusing a bad one.

    ``None`` is every dimension at one, which puts the coordinate unit itself in
    charge: the caller has either made each variable's spread O(1) or is about
    to say what it is. Anything but a ``(d,)`` vector is refused rather than
    broadcast into one: a scalar or a length-one vector would silently become a
    uniform vector, which is a real scale but never what a caller reaching for
    those shapes means.
    """
    if scale is None:
        return jnp.ones(d)
    given = jnp.asarray(scale)
    if given.ndim != 1 or given.shape[0] != d:
        msg = f"scale must be a ({d},) vector or None, got shape {given.shape}"
        raise ValueError(msg)
    if not bool(jnp.all(jnp.isfinite(given))) or not bool(jnp.all(given > 0.0)):
        msg = f"scale must hold finite positive values, got {scale!r}"
        raise ValueError(msg)
    return given


def _check_samplers(
    samplers: Sequence[DrawFn | None] | None,
    d: int,
) -> tuple[DrawFn | None, ...]:
    """Normalize the injected draws to one entry per dimension.

    ``None`` is every dimension on its default draw. The length is checked
    because zipping a short sequence would leave the tail dimensions silently on
    the default - a wiring error that would look like a working run.
    """
    if samplers is None:
        return (None,) * d
    given = tuple(samplers)
    if len(given) != d:
        msg = f"samplers needs {d} entries, got {len(given)}"
        raise ValueError(msg)
    return given


_TRUNCATED_TAIL = 9.0
"""Clip the standard-normal draw to this many scales either way.

Only the domain's own endpoints can push the CDF inverse past it: `ndtri` is
undefined at 0 and 1, and a truncation far into the tail rounds both bounds of
the uniform interval onto the same endpoint. The clip keeps such a draw finite,
and it costs nothing else - beyond nine scales the normal's mass is 1e-19, so
no reachable draw is touched, while a domain truncated that far out collapses
onto its own boundary, which is the right limit anyway.
"""


def _truncated_normal(
    key: jnp.ndarray,
    n_points: int,
    *,
    lower: float,
    upper: float,
    scale: float,
) -> jnp.ndarray:
    """Draw ``N(0, scale^2)`` restricted to ``[lower, upper]``.

    Both bounds take an infinity, so this one routine serves either orientation
    of a half-bounded domain. Inverse-CDF because the alternative - draw and
    reject - costs a variable number of rounds that JAX cannot express without
    a fixed cap, which would bias whatever the cap cuts off.

    Args:
        key: the PRNG key to draw from.
        n_points: how many values to draw.
        lower: the domain's lower bound, possibly ``-inf``.
        upper: the domain's upper bound, possibly ``+inf``.
        scale: the prior's standard deviation, in coordinate units.

    Returns:
        ``(n_points,)`` finite values, all inside ``[lower, upper]``.

    """
    v = jax.random.uniform(key, (n_points,))
    # Sample the *tail* probability, not the CDF value. A half-line is one tail,
    # and a truncation far out makes that tail small enough that `1 - Phi(a)` and
    # `Phi(a)` agree to the last float32 - the CDF value loses its precision
    # exactly where the truncated mass still lives. Multiplying the small tail by
    # `v` keeps the relative precision, so the draw tracks the true truncated
    # normal instead of piling its far tail onto one clipped value.
    if lower == -math.inf:
        p = ndtr(jnp.asarray(upper) / scale)
        z = ndtri(p * v)
    else:
        p = ndtr(-jnp.asarray(lower) / scale)
        z = -ndtri(p * (1.0 - v))
    z = jnp.clip(z, -_TRUNCATED_TAIL, _TRUNCATED_TAIL)
    return jnp.clip(scale * z, lower, upper)
