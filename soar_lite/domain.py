"""The search domain: a declared box, its per-dimension scales and its draws.

This module owns the coordinate space a restart model reads and the distribution
it proposes restart points from - the box (any side possibly unbounded), each
dimension's prior width in coordinate units, and the per-dimension draw, either
the default the domain admits or one the caller injected. It is the object a
`Projection` is built on: the geometry adds a `features` map, this owns
everything about *where* the model may look and *how* a start is proposed, so a
model-free restart sampler needs nothing but this and a key.

The box's own spelling lives here too - `Bounds`, and the validation and
normalization behind it. This module imports nothing from the package, so it is
the leaf that both the engine's `n_path_schedule` and the model modules read a
box from; the clip that keeps every proposal inside it stays the engine's.

A side may be unbounded, and there are three cases, not two: a bounded dimension
draws uniformly, a free one (both sides infinite) draws ``N(0, scale_i^2)``, and
a half-bounded one (exactly one finite side) draws that same normal *truncated*
to its declared side - a sampler free to step over the side the caller declared
is not sampling from the declared domain at all.
"""

import math
from collections.abc import Callable, Sequence
from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri

# The search box in scipy's spelling (which `_box_bounds` normalizes): one
# `(lb, ub)` pair per dim, a side of None or +-inf meaning unbounded. A `(d, 2)`
# array is the same data row-wise, for the callers that already hold one.
Bounds: TypeAlias = Sequence["tuple[float | None, float | None]"] | jnp.ndarray

# One dimension's draw, injected per dimension so a restart model can sample
# where a bounded box does not reach: `sampler(key, n)` returns `n` coordinates
# for that one dimension. `None` leaves the domain's own default in place -
# uniform inside a bounded side, ``N(0, scale_i^2)`` where the domain is free,
# and that normal restricted to the declared side where it is half-bounded.
DrawFn: TypeAlias = Callable[[jnp.ndarray, int], jnp.ndarray]


class Domain:
    """The coordinates a restart model may read, and how a start is drawn.

    A domain carries the box, the per-dimension scales and the per-dimension
    draws, and validates all three once at construction: a NaN side, a lower
    side above its upper, a scale that is not a ``(d,)`` vector of finite
    positive values, or a sampler count that is not ``d`` are refused here
    rather than deep inside a clip or a draw. It is concrete - a caller sampling
    with no surrogate needs nothing more - and `Projection` wraps one to add the
    feature map a fitted model reads.

    A side may be unbounded. The box then declares only the dimension and which
    sides are unbounded; `scale` is what carries a length there, because no
    width exists to read one off. Not even a bounded side's width is treated as
    a natural unit: `scale` is in coordinate units, so a caller changing a
    variable from metres to millimetres changes `scale`, never the box.

    `draw` is the one contract: ``(key, n) -> (n, d)`` starts drawn one
    dimension at a time.
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
                which sides are unbounded, so a domain has nothing to read
                without one.
            scale: per-dimension prior width in coordinate units, a ``(d,)``
                vector or ``None`` for every dimension at one. A projection that
                normalizes compares the entries to each other, but the common
                magnitude is not free either: `scale` is the length the default
                draw is measured in, so a dimension whose coordinates span far
                more or far less than one wants its own entry rather than the
                default.
            samplers: one optional draw per dimension, ``d`` entries; a ``None``
                entry keeps the default its domain admits - uniform inside a
                bounded dimension, ``N(0, scale_i^2)`` inside a free one, and
                that same normal restricted to the domain inside a half-bounded
                one, where drawing the unrestricted normal would step over the
                one side that is declared. Dimensions are drawn independently,
                so a prior that couples them needs its own `GlobalModel`, not a
                domain.

        Raises:
            ValueError: If the box is absent or malformed, or `scale` is not a
                ``(d,)`` vector of finite positive values, or `samplers` is not
                ``d`` entries long.

        """
        _check_bounds(bounds)
        box = _box_bounds(bounds)
        if box is None:
            msg = (
                "a domain reads its dimension and its unbounded sides off the "
                f"box it is given, so it needs one: got {bounds!r}"
            )
            raise ValueError(msg)
        self.lo, self.hi = box
        # The dimension is what `bounds` declares: read it off the argument,
        # not off `lo`/`hi`, which `_box_bounds` just derived from it.
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


def _check_bounds(bounds: Bounds | None) -> None:
    """Reject a malformed box at construction, not deep inside a clip.

    Both spellings are checked: one `(lb, ub)` pair per dim, or a `(d, 2)`
    array. A NaN side is rejected too - it would poison every clip silently
    instead of failing; +-inf sides are legitimate (unbounded dim).
    """
    if bounds is None:
        return
    pairs: list[tuple[float | None, float | None]]
    if isinstance(bounds, jnp.ndarray):
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            msg = f"bounds as an array needs shape (d, 2), got {bounds.shape}"
            raise ValueError(msg)
        pairs = list(zip(bounds[:, 0].tolist(), bounds[:, 1].tolist(), strict=True))
    else:
        pairs = []
        for j, pair in enumerate(bounds):
            if len(pair) != 2:
                msg = f"bounds[{j}] must be a (lb, ub) pair, got {pair!r}"
                raise ValueError(msg)
            pairs.append((pair[0], pair[1]))
    for j, (low, high) in enumerate(pairs):
        lo_f = -math.inf if low is None else low
        hi_f = math.inf if high is None else high
        if math.isnan(lo_f) or math.isnan(hi_f):
            msg = f"bounds[{j}] has a NaN side: ({low!r}, {high!r})"
            raise ValueError(msg)
        if lo_f > hi_f:
            msg = f"bounds[{j}] has lo={lo_f} > hi={hi_f}"
            raise ValueError(msg)


def _box_bounds(bounds: Bounds | None) -> tuple[jnp.ndarray, jnp.ndarray] | None:
    """Normalize a box to (lo, hi) arrays; None (no box at all) -> None.

    A `None` side becomes ∓inf, so every downstream consumer sees one spelling
    of "unbounded". A `(d, 2)` array carries the same pairs one row each.
    """
    if bounds is None:
        return None
    if isinstance(bounds, jnp.ndarray):
        return bounds[:, 0], bounds[:, 1]  # (d, 2) array: one pair per row
    lo = [-math.inf if pair[0] is None else pair[0] for pair in bounds]
    hi = [math.inf if pair[1] is None else pair[1] for pair in bounds]
    return jnp.asarray(lo), jnp.asarray(hi)


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
