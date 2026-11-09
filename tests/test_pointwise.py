"""Tests for `pointwise`: one per-point objective, both batched eval seams.

The adapter exists so a caller writes a scalar ``(d,) -> ()`` function and no
`vmap`/`value_and_grad`; these tests pin each half against the spelling it
replaces, and check the pair actually drives the engine - including the value-only
half, which only a method that reads its own trials ever reaches.
"""

import jax
import jax.numpy as jnp

from soar_lite import (
    DiagonalPreconditioner,
    Domain,
    LocalSolver,
    PathPoint,
    Pending,
    ServeTrials,
    n_path_schedule,
    pointwise,
)


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


class _ValueReader(LocalSolver):
    """A method that reads its candidate's value: the value-only seam's client.

    One fixed step along -g, read before it is handed back, so the winner arrives
    with its value served and owes only its gradient. That is the minimal method
    that makes both counters move, which is what the seam pair is for.
    """

    def __init__(self) -> None:
        """Start with no point: `restart` places the first one."""
        self.p: PathPoint | None = None

    def restart(self, i: int, p: PathPoint) -> None:
        """Take the point the engine placed."""
        del i
        self.p = p

    def fold(self, i: int, p: PathPoint) -> None:
        """Take the point the engine committed."""
        del i
        self.p = p

    def propose(
        self, idxs: list[int], domain: Domain, serve_trials: ServeTrials
    ) -> dict[int, Pending | None]:
        """Return the read candidate for every path asked about."""
        assert self.p is not None
        x = jnp.clip(self.p.x - 0.1 * self.p.g, domain.lo, domain.hi)
        out: dict[int, Pending | None] = {}
        for i in idxs:
            (read,) = serve_trials(x[None, :])
            out[i] = read
        return out


def test_the_pair_drives_a_run() -> None:
    """The seams it returns are the ones `n_path_schedule` takes."""
    domain = Domain(jnp.asarray([[-2.0, 2.0], [-2.0, 2.0]]))
    eval_f_g, eval_f = pointwise(_objective)

    result = n_path_schedule(
        eval_f_g,
        n_paths=4,
        domain=domain,
        eval_f=eval_f,
        max_iters=20,
        local_solver=DiagonalPreconditioner(),
    )

    assert result.status == "max_iters", result.status
    assert bool(jnp.all(jnp.isfinite(result.x))), result.x.tolist()
    assert result.fun < 1.0, result.fun
    # The default method takes one full step and reads nothing, so its only seam
    # is the fused one: `eval_f` being supplied costs it nothing.
    assert result.n_f_points == 0, result.n_f_points


def test_the_value_only_seam_carries_the_trials_a_method_reads() -> None:
    """A method that reads its trials is served by `eval_f`, and only by it."""
    domain = Domain(jnp.asarray([[-2.0, 2.0], [-2.0, 2.0]]))
    eval_f_g, eval_f = pointwise(_objective)

    result = n_path_schedule(
        eval_f_g,
        n_paths=1,
        domain=domain,
        x0=jnp.asarray([[1.5, 1.5]]),
        eval_f=eval_f,
        max_iters=20,
        local_solver=_ValueReader(),
    )

    assert result.fun < 1e-3, result.fun
    # Every iteration served one trial for its value and then bought that
    # winner's gradient on its own: the two counters are the two seams.
    assert result.n_f_points > 0, result.n_f_points
    assert result.n_f_g_points > 0, result.n_f_g_points
