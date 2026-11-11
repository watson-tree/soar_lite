"""The geometries a restart model reads domain coordinates through.

`Projection` is the feature map itself: one pure query, `features`, turning
``(M, d)`` coordinates into ``(M, w)`` rows a model fits its own posterior over.
Two concretes ship as two priors over where the optimum sits.

`SphereProjection` bends a chosen subset of the coordinates through inverse
stereography - the bend pulls the corners toward the pole and centers the bowl,
a prior that the optimum lives near the middle. `IdentityProjection` reads the
coordinates as their own features and puts the center nowhere special - a prior
that the optimum may well sit on a boundary or a corner. The caller picks the
one its belief deserves.

A dimension may be unbounded: inverse stereography takes all of ``R``, so the
same bend applies to a line, and `scale` states each dimension's prior width in
coordinate units rather than reading a box width as a unit of measure.

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

import jax.numpy as jnp

from soar_lite.domain import Domain

ProjectionFactory: TypeAlias = Callable[[Domain], "Projection"]
"""Build the `Projection` a model reads one domain through.

Geometry is a pure function of the domain - a center, a width, one lengthscale -
so the model carries the recipe rather than one projection bound to one box:
`partial(SphereProjection, kappa=..., mask=...)` for the bent arm,
`IdentityProjection` itself for the straight one. Called once per ask with the
box that ask was handed, so the geometry is always the caller's box and never a
projection bound to an older one.
"""


class Projection(abc.ABC):
    """The feature map a restart model reads domain coordinates through.

    A projection wraps a `Domain` and adds one map: `features` turns every
    ``(M, d)`` batch of coordinates into an ``(M, w)`` batch of feature rows,
    which a model fits its own posterior over - `w` is this projection's width,
    read off `features` by the model rather than assumed ``d + 1``.

    The domain stays the domain's: a projection *reads* its box, scales and
    draws; it is not one. That keeps the two jobs apart at the type level - a
    caller who only samples builds a `Domain` and never meets a projection, and
    a projection held where a `Domain` is expected is a wiring error instead of
    a silently accepted one - and keeps `scale` and `samplers` out of this
    constructor, where every fitted model would otherwise have restated them.

    `features` is the one contract: a pure query of the coordinates, the same
    ``(M, d)`` input mapping to the same ``(M, w)`` rows, with `w` stable across
    calls for one projection. It is also the one method a surrogate of another
    family reads - nothing in a projection is specific to `BLRGlobalModel`'s
    posterior, since it carries geometry, not fit state.
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
        """Return the ``(M, w)`` feature rows for ``(M, d)`` coordinates.

        `w` is this projection's feature width: the model reads it off the
        returned shape, never off `d`, so a projection is free to widen or
        narrow the representation however its geometry needs.
        """
        ...


class SphereProjection(Projection):
    """Inverse-stereographic projection of a chosen subset of the dimensions.

    The restart models read the domain in whatever coordinates the objective
    takes. This maps them onto the unit sphere, ``[2z; |z|^2 - 1] / (|z|^2 + 1)``
    with ``|·|`` the Euclidean norm, where ``z`` holds only the projected
    dimensions' coordinates, centered, divided by the per-dimension `scale` and
    by one global lengthscale. Two properties matter downstream.

    Every projected row has its bent part at norm one - the stereographic
    coordinates plus the closing bowl coordinate - so a linear head read on
    those features has a constant self inner product: its posterior variance
    stays bounded far from data instead of climbing back to the prior. That
    climb is the corner-seeking pathology the projection exists to cure -
    variance growing with distance reads as unexplored room, and an acquisition
    spends its asks on the box corners. The last coordinate,
    ``(|z|^2 - 1)/(|z|^2 + 1)``, is then a radially symmetric, saturating feature
    of the input - the one extra degree of freedom a linear head gets, an
    isotropic bowl whose center the weights place.

    `mask` keeps that bowl off dimensions whose natural prior is uniform - an
    angle, say, wants no attraction to the center. A left-out dimension keeps
    its own centered, scaled coordinate and enters a linear head as a plain
    trend, so the feature vector is ``d + 1`` wide however many dimensions the
    projection bends - and only its bent part is unit-norm.

    A dimension may be unbounded: inverse stereography takes all of ``R``, so the
    same map bends a line, and its far end returns to the pole - the variance
    goes back to the prior instead of growing with ``|x|``, which is what an
    unbent coordinate does.

    References:
        - Doumont et al. (2026), AISTATS — the spherical-linear model this
          projection comes from.
        - Tripp (2026),
          [github.com/AustinT/2026-07-high-dim-bo](https://github.com/AustinT/2026-07-high-dim-bo)
          — the study that isolates what the bend contributes: a prior that the
          optimum is interior, strong enough that sliding the optimum to a corner
          reverses the ranking. Use this projection where that prior holds and
          `IdentityProjection` where it does not.

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
        kappa_validated = _check_kappa(kappa)
        super().__init__(domain)
        # The global lengthscale is what makes the projection dimension-free: a
        # bent dimension's reference radius, divided by `global_lengthscale`, is
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
        self.global_lengthscale = jnp.sqrt(kappa_validated * tied)

    def features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return the ``(M, d + 1)`` feature rows for ``(M, d)`` coordinates.

        Every dimension is first read in its own coordinate unit,
        ``(x - center) / scale``, so no column inherits a box width as its unit.
        The bent dimensions become their stereographic image on the unit sphere,
        divided by the global lengthscale; the rest keep that centered, scaled
        coordinate, so a linear head reads them as a plain linear trend.
        """
        z = (x - self.domain.center) / self.domain.scale
        bent = z[:, self.mask] / self.global_lengthscale
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


def _check_kappa(kappa: float) -> float:
    """Return `kappa` unchanged, refusing a value outside ``(0, 1)``."""
    if not math.isfinite(kappa) or not 0.0 < kappa < 1.0:
        msg = (
            "kappa ties the projection's radius to the domain, so it must "
            f"lie in (0, 1); got {kappa!r}"
        )
        raise ValueError(msg)
    return kappa


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
