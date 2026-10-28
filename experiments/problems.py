"""Benchmark problems for the soar-lite comparison.

Every problem exposes the engine's seam directly:

    eval_f_g(x: (M, d)) -> (f: (M,), grad: (M, d), feas: None)

``feas`` is always ``None``: the box is the only constraint and the engine
clips into it, so every served point counts toward the best. Each problem
is one batched JAX pass, differentiable, and has a known global value
``f_star``.

The headline problem carries both properties the comparison is about: mixed
physical units (kg, N/m, N*s/m) and a non-convex response landscape. The
remaining problems isolate one property each.

Import sets ``jax_enable_x64``: the oscillator chain solves a complex linear
system and the ill-conditioned quadratics have condition numbers up to 1e6,
both of which need float64 to be a fair test rather than a rounding study.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", val=True)

EvalFG: TypeAlias = Callable[[jnp.ndarray], "tuple[jnp.ndarray, jnp.ndarray, None]"]


@dataclass(frozen=True)
class Problem:
    """One benchmark: its seam, its box, and the global value it aims at."""

    name: str
    d: int
    bounds: jnp.ndarray  # (d, 2): one (lb, ub) row per coordinate
    eval_f_g: EvalFG
    f_star: float
    description: str
    x_star: jnp.ndarray | None = None  # a known global minimiser, for checks

    def eval_point(self, x: jnp.ndarray) -> tuple[float, jnp.ndarray]:
        """Value and gradient at a single point (a batch of one, squeezed)."""
        f, g, _ = self.eval_f_g(x[None, :])
        return float(f[0]), g[0]


def _box(d: int, lo: float, hi: float) -> jnp.ndarray:
    """Return a (d, 2) box with the same ``[lo, hi]`` on every coordinate."""
    return jnp.stack([jnp.full((d,), lo), jnp.full((d,), hi)], axis=1)


def _jit_pointwise(fn: Callable[[jnp.ndarray], jnp.ndarray]) -> EvalFG:
    """Lift a pointwise value fn to the batched seam via value_and_grad + vmap."""

    @jax.jit  # pyright: ignore[reportUnknownMemberType] - jax stubs are partial
    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        f, g = jax.vmap(jax.value_and_grad(fn))(x)
        return f, g, None

    return eval_f_g


# --- the smooth panel: preconditioning on scale and conditioning --------------


def diag_quadratic(d: int = 100, cond: float = 1e4) -> Problem:
    """Axis-aligned ill-conditioned quadratic: the diagonal sweet spot.

    Hessian ``diag(a)`` with ``a`` log-spaced from 1 to ``cond``, minimiser at
    the origin, ``f* = 0``. A per-coordinate secant reads ``1/H_ii`` exactly, so
    this is the positive control: any failure here is a harness bug, not a
    method limit.
    """
    a = jnp.logspace(0.0, math.log10(cond), d)

    def fn(x: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * jnp.sum(a * x**2)

    return Problem(
        name=f"diag_quad_d{d}_cond{cond:g}",
        d=d,
        bounds=_box(d, -1.0, 1.0),
        eval_f_g=_jit_pointwise(fn),
        f_star=0.0,
        description=f"axis-aligned quadratic, diagonal Hessian, cond={cond:g}",
        x_star=jnp.zeros(d),
    )


def rotated_quadratic(d: int = 100, cond: float = 1e6, seed: int = 1) -> Problem:
    """Randomly rotated ill-conditioned quadratic: the honest boundary.

    Dense ``H = Q diag(a) Q^T`` with ``a`` log-spaced 1..``cond``; the diagonal
    approximation is now wrong, so this is where the cost of dropping the
    coupling should show. Same ``f* = 0`` at the origin as the diagonal case.
    """
    q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(seed), (d, d)))
    a = jnp.logspace(0.0, math.log10(cond), d)
    hess = (q * a) @ q.T

    def fn(x: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * x @ hess @ x

    return Problem(
        name=f"rot_quad_d{d}_cond{cond:g}",
        d=d,
        bounds=_box(d, -1.0, 1.0),
        eval_f_g=_jit_pointwise(fn),
        f_star=0.0,
        description=f"rotated quadratic, dense Hessian, cond={cond:g}",
        x_star=jnp.zeros(d),
    )


# --- the multimodal panel: escape on a funnel / a lattice ---------------------


def ackley(d: int = 100) -> Problem:
    """Ackley: one broad funnel plus an exponential number of local minima."""
    a, b, c = 20.0, 0.2, 2.0 * math.pi
    e = math.e

    def fn(x: jnp.ndarray) -> jnp.ndarray:
        # The 1e-16 keeps sqrt differentiable at the exact origin, where the
        # plain form is 0/0 and reports a NaN gradient right on the minimum.
        # It shifts f(0) by ~4e-8, far below the comparison's resolution.
        s2 = jnp.sum(x**2) / d + 1e-16
        sc = jnp.sum(jnp.cos(c * x)) / d
        return -a * jnp.exp(-b * jnp.sqrt(s2)) - jnp.exp(sc) + e + a

    return Problem(
        name=f"ackley_d{d}",
        d=d,
        bounds=_box(d, -32.768, 32.768),
        eval_f_g=_jit_pointwise(fn),
        f_star=0.0,
        description="Ackley funnel with many local minima",
        x_star=jnp.zeros(d),
    )


def rastrigin(d: int = 100) -> Problem:
    """Rastrigin: a regular lattice of local minima around one global basin."""

    def fn(x: jnp.ndarray) -> jnp.ndarray:
        return 10.0 * d + jnp.sum(x**2 - 10.0 * jnp.cos(2.0 * math.pi * x))

    return Problem(
        name=f"rastrigin_d{d}",
        d=d,
        bounds=_box(d, -5.12, 5.12),
        eval_f_g=_jit_pointwise(fn),
        f_star=0.0,
        description="Rastrigin lattice of local minima",
        x_star=jnp.zeros(d),
    )


# --- the headline: mixed-unit device design (borrowed recipe) ----------------


def oscillator_chain(
    n_masses: int = 20,
    n_freqs: int = 20,
    seed: int = 0,
    omega_lo: float = 0.5,
    omega_hi: float = 50.0,
) -> Problem:
    """Design a lumped vibration-isolation chain to match a target transmissibility.

    A chain of ``n_masses`` masses joined by springs and dashpots, base-excited
    at ``n_freqs`` log-spaced frequencies. The unknowns are the physical
    parameters in their own units - masses ``m_i`` [kg], stiffnesses ``k_i``
    [N/m], dampings ``c_i`` [N*s/m] - so the search happens in genuinely
    heterogeneous coordinates and ``d = 3*n_masses``.

    Loss is the log-scale mismatch of the payload transmissibility
    ``|H(omega)|`` against a synthetic target produced by a hidden physical
    design, ``sum_j (log10 |H_j| - log10 |H*_j|)^2``, so ``f* = 0`` sits
    exactly on that design. The response is a batched complex solve of

        A(omega) X = (k_0 + i*omega*c_0) e_0,
        A = -omega^2 M + i*omega C + K,

    which is cheap (one tridiagonal N x N solve per frequency) and differentiable.
    The landscape is non-convex through resonance and anti-resonance, and the
    off-diagonal coupling between parameters is what the diagonal approximation
    drops - the intended "approximately right, not exact" middle ground.

    The box is the hidden design scaled by ``[1/4, 4]`` per parameter: wide
    enough that the start is not already solved, narrow enough that the target
    resonances stay inside the band. Boxes make positivity structural.
    """
    key_m, key_k, key_c = jax.random.split(jax.random.PRNGKey(seed), 3)
    n = n_masses
    m_star = jnp.exp(
        jax.random.uniform(key_m, (n,), minval=jnp.log(0.5), maxval=jnp.log(5.0)),
    )
    k_star = jnp.exp(
        jax.random.uniform(key_k, (n,), minval=jnp.log(1e2), maxval=jnp.log(1e5)),
    )
    c_star = jnp.exp(
        jax.random.uniform(key_c, (n,), minval=jnp.log(1.0), maxval=jnp.log(1e2)),
    )
    theta_star = jnp.concatenate([m_star, k_star, c_star])
    omega = jnp.logspace(math.log10(omega_lo), math.log10(omega_hi), n_freqs)

    def transmissibility(theta: jnp.ndarray) -> jnp.ndarray:
        """|H(omega)| for one design theta (3n,) over the frequency grid (F,)."""
        m, k, c = theta[:n], theta[n : 2 * n], theta[2 * n :]
        k_up = jnp.concatenate([k[1:], jnp.zeros(1)])  # spring k_{i+1}
        c_up = jnp.concatenate([c[1:], jnp.zeros(1)])

        def one_freq(w: jnp.ndarray) -> jnp.ndarray:
            """Payload-to-base ratio at one angular frequency w."""
            diag = -(w**2) * m + 1j * w * (c + c_up) + (k + k_up)
            mat = jnp.diag(diag)
            # off-diagonal A_{i,i+1} = -k_{i+1} - i*w*c_{i+1}
            upper = -k_up - 1j * w * c_up
            mat = mat + jnp.diag(upper[:-1], 1) + jnp.diag(upper[:-1], -1)
            rhs = jnp.zeros(n, dtype=jnp.complex128).at[0].set(k[0] + 1j * w * c[0])
            sol = jnp.linalg.solve(mat, rhs)
            return jnp.abs(sol[-1]) + 1e-12  # floor keeps log finite at nulls

        return jax.vmap(one_freq)(omega)

    target = transmissibility(theta_star)

    def fn(theta: jnp.ndarray) -> jnp.ndarray:
        ratio = transmissibility(theta) / target
        return jnp.sum(jnp.log10(ratio) ** 2)

    lo = theta_star / 4.0
    hi = theta_star * 4.0
    bounds = jnp.stack([lo, hi], axis=1)
    return Problem(
        name=f"oscillator_n{n}_F{n_freqs}",
        d=3 * n,
        bounds=bounds,
        eval_f_g=_jit_pointwise(fn),
        f_star=0.0,
        description=f"vibration-isolation chain, {n} masses, d={3 * n}, mixed units",
        x_star=theta_star,
    )


def build(group: str) -> list[Problem]:
    """Return the problems of one design group.

    group: "smooth" (preconditioning), "multimodal" (escape), "headline"
    (mixed units + non-convex response), or "all".
    """
    groups = {
        "headline": lambda: [oscillator_chain()],
        "smooth": lambda: [diag_quadratic(), rotated_quadratic()],
        "multimodal": lambda: [ackley(), rastrigin()],
    }
    if group == "all":
        return [p for g in groups.values() for p in g()]
    return groups[group]()
