"""Tests for `pointwise`: one per-point objective, both batched eval seams.

The adapter exists so a caller writes a scalar ``(d,) -> ()`` function and no
`vmap`/`value_and_grad`; these tests pin each half against the spelling it
replaces, and check the pair actually drives the engine.
"""

import jax
import jax.numpy as jnp

from soar_lite import DiagonalPreconditioner, Domain, n_path_schedule, pointwise


def _objective(row: jnp.ndarray) -> jnp.ndarray:
    """Return a smooth off-center quadratic, pointwise: ``(d,) -> ()``."""
    return jnp.sum((row - jnp.asarray([0.3, -0.2])) ** 2)


def _batch(n: int = 5) -> jnp.ndarray:
    """Return an ``(n, 2)`` batch of distinct rows inside a ``[-2, 2]^2`` box."""
    return jnp.linspace(-1.5, 1.5, n)[:, None] * jnp.asarray([1.0, -1.0])[None, :]


def _feasible(row: jnp.ndarray) -> jnp.ndarray:
    """Return a per-point predicate, true on the inner disc."""
    return jnp.sum(row**2) < 2.0


def test_the_fused_half_is_the_per_row_value_and_grad() -> None:
    """`vmap(value_and_grad)` is what it replaces, so it must match bitwise."""
    x = _batch()
    eval_f_g, _ = pointwise(_objective)

    f, g, feas = eval_f_g(x)
    want_f, want_g = jax.vmap(jax.value_and_grad(_objective))(x)

    assert f.shape == (x.shape[0],), f.shape
    assert g.shape == x.shape, g.shape
    assert feas is None
    assert jnp.array_equal(f, want_f), (f, want_f)
    assert jnp.array_equal(g, want_g), (g, want_g)


def test_feasibility_is_a_separate_per_point_vmap() -> None:
    """The mask is vmapped on its own, since it cannot ride inside the grad."""
    x = _batch()
    eval_f_g, eval_f = pointwise(_objective, feasible=_feasible)

    _, _, feas = eval_f_g(x)
    assert feas is not None
    assert feas.shape == (x.shape[0],), feas.shape
    assert feas.dtype == jnp.bool_, feas.dtype
    assert jnp.array_equal(feas, jax.vmap(_feasible)(x)), feas
    # The value-only half must report the same mask, not its own.
    _, feas_f = eval_f(x)
    assert feas_f is not None
    assert jnp.array_equal(feas_f, feas)


def test_the_f_only_half_is_the_forward_pass() -> None:
    """Its values are `vmap(objective)` - the fused half's own forward pass."""
    x = _batch()
    _, eval_f = pointwise(_objective)

    f, feas = eval_f(x)
    assert f.shape == (x.shape[0],), f.shape
    assert feas is None
    assert jnp.array_equal(f, jax.vmap(_objective)(x)), f


def test_building_the_pair_runs_nothing() -> None:
    """Both halves are lazy closures: constructing them touches no point."""
    calls: list[jnp.ndarray] = []

    def never(row: jnp.ndarray) -> jnp.ndarray:
        calls.append(row)
        return jnp.sum(row)

    eval_f_g, eval_f = pointwise(never)

    assert calls == []
    assert callable(eval_f_g)
    assert callable(eval_f)


def test_the_pair_drives_a_run() -> None:
    """The seams it returns are the ones `n_path_schedule` takes."""
    domain = Domain(jnp.asarray([[-2.0, 2.0], [-2.0, 2.0]]))
    eval_f_g, eval_f = pointwise(_objective)

    result = n_path_schedule(
        eval_f_g,
        n_paths=4,
        domain=domain,
        eval_f=eval_f,
        max_iters=5,
        # n_alpha >= 2 puts every path in the judged lane, so the f-only seam is
        # exercised from the first iteration instead of whenever a gradient floor
        # happens to be hit - with the default n_alpha=1 it took a lucky path, and
        # a different direction rule moved it out of reach.
        local_solver=DiagonalPreconditioner(n_alpha=2),
    )

    assert result.status == "max_iters", result.status
    assert bool(jnp.all(jnp.isfinite(result.x))), result.x.tolist()
    assert result.fun < 1.0, result.fun
    # With the f-only seam in play, trials are served for values alone, so the
    # run must have used both counters rather than the fused half only.
    assert result.n_f_points > 0, result.n_f_points
