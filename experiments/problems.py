"""Benchmark problems for the soar-lite comparison.

Every problem exposes the engine's seam directly:

    eval_f_g(x: (M, d)) -> (f: (M,), grad: (M, d), feas: None)

``feas`` is always ``None``: the box is the only constraint and the engine
clips into it, so every served point counts toward the best. Each problem
is one batched JAX pass, differentiable, and has a known global value
``f_star``.

The headline problems carry both properties the comparison is about: mixed
physical units and a non-convex response landscape. They are the two real
inverse problems - a frequency-domain one (vibration isolation) and a
time-domain one (compartment kinetics). The remaining problems isolate one
property each.

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
import jax.scipy

jax.config.update("jax_enable_x64", val=True)

EvalFG: TypeAlias = Callable[[jnp.ndarray], "tuple[jnp.ndarray, jnp.ndarray, None]"]

# Assay detection limit of the compartment problem, relative to the bolus dose:
# a sample below it is not a measurement and does not enter the loss. Double
# precision reaches ~1e-300, so this is never active as a numerical floor.
PK_DETECT = 1e-8
PK_FLOOR = 1e-30


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


# --- the multimodal panel: momentum drift on a funnel / a lattice -------------


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
    exactly on that design. Both sides carry a ``1e-12`` floor, which is a
    ``-inf`` guard for an exact null rather than a shaping term: responses
    drawn from this box stay several orders of magnitude above it, so the
    slope the optimiser sees is the log mismatch, not the floor. The response
    is a batched complex solve of

        A(omega) X = (k_0 + i*omega*c_0) e_0,
        A = -omega^2 M + i*omega C + K,

    and ``A`` is tridiagonal, but it is assembled and solved **densely**: one
    ``(n, n)`` complex solve per frequency, O(n^3) rather than the O(n) a banded
    solve would give. At ``n = 20`` that is the cheap side of this problem set -
    XLA batches the solves and the reverse pass is an adjoint solve of the same
    size - and a banded solve would only pay if ``n_masses`` grew substantially,
    which nothing here asks for. The landscape is non-convex through resonance
    and anti-resonance, and the off-diagonal coupling between parameters is what
    the diagonal approximation drops - the intended "approximately right, not
    exact" middle ground.

    The box is the hidden design scaled by ``[1/4, 4]`` per parameter: wide
    enough that the start is not already solved, narrow enough that the target
    resonances stay inside the band. Boxes make positivity structural, and this
    one is not merely a safeguard: ``m``, ``k`` and ``c`` enter ``A``
    homogeneously, so scaling every parameter by a common factor leaves ``|H|``
    unchanged - the response sees only ``omega_n = sqrt(k/m)`` and
    ``zeta = c / (2 sqrt(k m))``. Every point of the ray ``lambda * theta_star``
    inside the box is therefore an exact solution: ``f* = 0`` is a set rather
    than a point, and what this problem identifies is the transmissibility, not
    the design. A dimension count says the same thing more bluntly: the loss
    reads twenty real numbers, one ``|H|`` per frequency, against ``d = 3*n``
    unknowns, so the response's Jacobian has at most twenty rows and cannot be
    full rank for ``n > 6``; at the truth far fewer directions than that are
    independent, because neighbouring frequencies of a smooth spectrum measure
    much the same thing. Parameter recovery is not a question this problem can
    answer, and that division of labour is deliberate: this is a **design**
    problem - does an optimiser drive the response onto the target - while
    ``pk_chain`` is the recovery problem, built so that its rates are
    identifiable. ``x_star`` is one point on the solution set, the point the
    target was drawn from; the benchmark scores ``f`` and never the distance to
    it.
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
            return jnp.abs(sol[-1]) + 1e-12  # -inf guard at a null, not a shaper

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


def pk_chain(
    n_comp: int = 21,
    seed: int = 0,
    n_times: int = 20,
) -> Problem:
    """Recover a compartment model's micro rate constants from concentrations.

    The textbook pharmacokinetic inverse problem. ``n_comp`` compartments form a
    bidirectional chain, exchanging first-order transfer rates ``f_i`` (i ->
    i+1), ``b_i`` (i+1 -> i) and eliminating at ``gamma_i`` from every one of
    them, so the unknowns are ``3n - 2`` rates in their own unit (1/h) spanning
    four decades. A bolus sits in each end at ``t = 0`` and the first three
    compartments are measured; the loss is the log-scale mismatch against the
    hidden true design, so ``f* = 0`` sits exactly on it.

    This is the time-domain counterpart of the oscillator and the second real
    problem in the panel. Its coupling is local in a compartment graph, not a
    dense rotation, which is the structure the diagonal approximation reads as
    "approximately right": the system matrix is a compartmental one, every row
    summing to minus its elimination rate.

    Two details are load-bearing, and both were measured rather than assumed.
    A bolus in one end only leaves most of the rates invisible - the far
    compartments never carry mass, so the loss can fall to ~1e-7 while most of
    the 61 rates stay wherever they started, a flat subspace a benchmark should
    not hide. The sampling times are a dyadic geometric grid,
    ``t_j = first_t * 2**j``, because that is the natural grid for a spectrum
    spanning decades and because
    ``expm(A t_j) = expm(A first_t)**(2**j)`` then costs one matrix exponential
    plus repeated squaring instead of one exponential per sample - which is
    what makes a 20-seed, 3000-batch run of this problem affordable.

    The window runs long enough that its tail falls under ``PK_DETECT``, and
    those samples are dropped rather than scored: a concentration ten orders of
    magnitude below the peak is what a real assay cannot see, and scoring it
    would put a numerical floor where a measurement should be.

    The loss is non-convex in the box, and not only mildly. Coupling makes the
    flat directions *joint*: each coordinate on its own moves the loss (moving
    one to a box edge is visible for 17/20 transfer and 19/20 backward rates,
    median change ~0.3), yet several can move together and stay at a low loss.
    Measured: L-BFGS-B started exactly at the truth stops after one evaluation
    at ``f = 7e-33``, but started at ``truth * 1.05`` it walks to a box edge
    and rests at ``7.6e-4`` with coordinates up to 3x off, and random starts
    end anywhere from ``1e-3`` to ``3``. That is the real difficulty of
    compartment fitting - the observed curve does not pin every rate - and it
    is why this problem sits in the non-convex headline group rather than
    among the quadratics.

    ``first_t`` is fixed from the true design's fastest mode, so no
    eigendecomposition enters the differentiated path.
    """
    key_f, key_b, key_g = jax.random.split(jax.random.PRNGKey(seed), 3)

    def log_uniform(
        key: jnp.ndarray,
        lo: float,
        hi: float,
        shape: tuple[int, ...],
    ) -> jnp.ndarray:
        """Draw rates spanning decades: what makes the per-coordinate scale matter."""
        return jnp.exp(
            jax.random.uniform(
                key,
                shape,
                minval=math.log(lo),
                maxval=math.log(hi),
            ),
        )

    theta_star = jnp.concatenate(
        [
            log_uniform(key_f, 1e-2, 1e2, (n_comp - 1,)),
            log_uniform(key_b, 1e-2, 1e2, (n_comp - 1,)),
            log_uniform(key_g, 1e-3, 1.0, (n_comp,)),
        ],
    )

    def system_matrix(theta: jnp.ndarray) -> jnp.ndarray:
        """Return the compartmental matrix ``A`` with ``dC/dt = A C``."""
        n = n_comp
        trans_f = theta[: n - 1]
        trans_b = theta[n - 1 : 2 * (n - 1)]
        elim = theta[2 * (n - 1) :]
        idx = jnp.arange(n)
        out_f = jnp.concatenate([trans_f, jnp.zeros(1)])
        out_b = jnp.concatenate([jnp.zeros(1), trans_b])
        mat = jnp.zeros((n, n))
        mat = mat.at[idx, idx].set(-(out_f + out_b + elim))
        mat = mat.at[idx[:-1] + 1, idx[:-1]].set(trans_f)
        return mat.at[idx[:-1], idx[:-1] + 1].set(trans_b)

    def trajectory(theta: jnp.ndarray, first_t: jnp.ndarray) -> jnp.ndarray:
        """Concentrations at ``first_t * 2**j`` for ``j < n_times``, shape (T, n)."""
        load = jnp.zeros((n_comp,)).at[0].set(1.0).at[n_comp - 1].set(1.0)

        def square(carry: jnp.ndarray, _: None) -> tuple[jnp.ndarray, jnp.ndarray]:
            return carry @ carry, carry @ load

        start = jax.scipy.linalg.expm(system_matrix(theta) * first_t)
        _, conc = jax.lax.scan(square, start, None, length=n_times)
        return conc

    speeds = jnp.abs(jnp.linalg.eigvals(system_matrix(theta_star)))
    first_t = 0.1 / jnp.max(speeds[speeds > 1e-9])
    observed = jnp.arange(3)
    target = trajectory(theta_star, first_t)[:, observed]
    measured = target > PK_DETECT

    def fn(theta: jnp.ndarray) -> jnp.ndarray:
        conc = trajectory(theta, first_t)[:, observed]
        ratio = jnp.log10((conc + PK_FLOOR) / (target + PK_FLOOR)) ** 2
        return jnp.sum(jnp.where(measured, ratio, 0.0))

    d = 3 * n_comp - 2
    return Problem(
        name=f"pk_chain_n{n_comp}_d{d}",
        d=d,
        bounds=jnp.stack([theta_star / 4.0, theta_star * 4.0], axis=1),
        eval_f_g=_jit_pointwise(fn),
        f_star=0.0,
        description=(
            f"compartment kinetics, {n_comp} compartments, d={d}, rates over 4 decades"
        ),
        x_star=theta_star,
    )


def build(group: str) -> list[Problem]:
    """Return the problems of one design group.

    group: "smooth" (preconditioning), "multimodal" (momentum drift), "headline"
    (the two real inverse problems: mixed units and a non-convex response),
    "blr" (restart model: rastrigin, the one problem where a restart has a
    stage), or "all" (the union of the four, one entry per problem even though
    the groups overlap).
    """
    groups = {
        "headline": lambda: [oscillator_chain(), pk_chain()],
        "smooth": lambda: [diag_quadratic(), rotated_quadratic()],
        "multimodal": lambda: [ackley(), rastrigin()],
        "blr": lambda: [rastrigin()],
    }
    if group == "all":
        # The groups overlap on rastrigin (multimodal and blr both list it), so
        # dedupe by name: a caller that draws one panel per entry would
        # otherwise draw the same problem twice.
        unique: dict[str, Problem] = {}
        for g in groups.values():
            for p in g():
                unique.setdefault(p.name, p)
        return list(unique.values())
    return groups[group]()
