"""Self-checks for the diagonal preconditioner, this package's local method.

Pins the secant ratio, the gradient-norm floor (a floor, not an addend), the
decay, the momentum blend, the gradient-norm stopping test, the unit invariance
of the steady-state step, and the clip the method owns on its own proposal. No
engine wiring beyond the two round-trips at the end. Run with:

    uv run python -m tests.test_local_solver
"""

# This suite is the spec for the solver's state transitions, so reading them is
# the intended read: `_state` is the interface being pinned, not an internal.
# ruff: noqa: SLF001
# pyright: reportPrivateUsage=false

import math
from collections.abc import Callable
from dataclasses import replace

import jax
import jax.numpy as jnp

from soar_lite.domain import Domain
from soar_lite.engine import (
    PathPoint,
    Pending,
    n_path_schedule,
)
from soar_lite.local_solver import DiagonalPreconditioner
from tests.flock import flock_kwargs

jax.config.update("jax_enable_x64", val=True)


def close(
    a: jnp.ndarray | float,
    b: jnp.ndarray | float,
    tol: float = 1e-5,
) -> bool:
    """Return True iff all elements of a and b agree within tol."""
    return bool(jnp.max(jnp.abs(a - b)) < tol)


def wide(d: int) -> Domain:
    """Return a box no number in this suite comes near, in `d` dimensions."""
    return Domain(jnp.tile(jnp.asarray([[-1e12, 1e12]]), (d, 1)))


def _unread(x_batch: jnp.ndarray) -> list[Pending]:
    """Return nothing: this method takes one full step and reads no trials."""
    msg = f"propose served {x_batch.shape[0]} trials; this method reads none"
    raise AssertionError(msg)


def proposed(solver: DiagonalPreconditioner, i: int = 0) -> Pending:
    """Return the `Pending` path `i` proposes, failing if the method stalls.

    One path per call, in a box wide enough that the clip never bites: the tests
    that care about the clip pass their own box to `propose`.
    """
    out = solver.propose([i], wide(int(solver._state[i].point.x.shape[0])), _unread)
    pending = out[i]
    assert pending is not None, "the method named no winner"
    return pending


# --- DiagonalPreconditioner ---


def test_diag_warmup_is_one_gradient_step() -> None:
    """The warm-up is the steady state's step: one candidate along -g, unread."""
    solver = DiagonalPreconditioner()
    p = PathPoint(x=jnp.zeros(3), f=2.0, g=jnp.array([3.0, 0.0, 4.0]))
    solver.restart(0, p)
    assert solver._state[0].is_warmup
    assert solver._state[0].alpha == 1.0
    pending = proposed(solver)
    assert close(pending.x, p.x + 1.0 * (-p.g), 1e-12)
    # Both halves are owed: the method read nothing, so the engine's `settle`
    # buys this candidate's f and g in the same fused batch as everything else
    # the iteration produced.
    assert pending.f is None
    assert pending.g is None
    # Committed exactly as proposed: the step need not beat the start's value.
    above = PathPoint(x=p.x - p.g, f=9.0, g=p.g)
    solver.step(0, above)
    assert not solver._state[0].is_warmup
    assert close(solver._state[0].s_ema, jnp.square(-p.g), 1e-12)


def test_a_stationary_path_proposes_nothing() -> None:
    """A converged path answers ``None``; the engine stalls it where it stands.

    ``None`` is the contract's single stopping channel, and the engine acts on it
    by stalling the path - so a converged path has to say so here, or it is
    stepped off its stationary point forever and the stall a restart rule reads
    never happens.
    """
    solver = DiagonalPreconditioner()
    start = PathPoint(x=jnp.zeros(2), f=1.0, g=jnp.array([1.0, 1.0]))
    solver.restart(0, start)
    # The floor is where the path ends up, so it is a committed point: handed to
    # `step`, not to the ask - what the stopping test reads is the path itself.
    floor = PathPoint(x=start.x - start.g, f=0.5, g=jnp.array([1e-9, -1e-9]))
    solver.step(0, floor)
    assert solver.propose([0], wide(2), _unread) == {0: None}


def test_a_path_pinned_by_an_outward_gradient_is_stationary() -> None:
    """A wall is a reachable stop: the gradient points out, so no step is feasible."""
    box = Domain(jnp.stack([jnp.zeros(2), jnp.ones(2)], axis=1))
    solver = DiagonalPreconditioner()
    start = PathPoint(x=jnp.full(2, 0.5), f=1.0, g=jnp.full(2, 3.0))
    solver.restart(0, start)
    solver.restart(1, start)
    # Path 0 stands on the lower wall with g > 0, path 1 on the upper wall with
    # g < 0: in both the descent -g points out of the box, so the clip would pin
    # every step to the point the path already stands on.
    solver.step(0, PathPoint(x=jnp.zeros(2), f=1.0, g=jnp.full(2, 2.0)))
    solver.step(1, PathPoint(x=jnp.ones(2), f=1.0, g=jnp.full(2, -2.0)))

    assert solver.propose([0, 1], box, _unread) == {0: None, 1: None}


def test_the_wall_decides_what_pins_a_point_not_the_gradient() -> None:
    """One (x, g) is a stop under the upper wall and a live descent under the lower."""
    start = PathPoint(x=jnp.array([-0.5]), f=1.0, g=jnp.array([-3.0]))
    point = PathPoint(x=jnp.zeros(1), f=1.0, g=jnp.array([-2.0]))
    lower_wall = Domain(jnp.asarray([[0.0, 1e6]]))
    upper_wall = Domain(jnp.asarray([[-1e6, 0.0]]))

    def verdict(box: Domain) -> Pending | None:
        solver = DiagonalPreconditioner()
        solver.restart(0, start)
        solver.step(0, point)
        return solver.propose([0], box, _unread)[0]

    assert verdict(upper_wall) is None  # g < 0 against the upper wall: pinned
    assert verdict(lower_wall) is not None  # the same g against the lower: inward


def test_momentum_pointing_out_does_not_pin_a_point() -> None:
    """The stop reads the gradient, so lagging momentum cannot declare a path done."""
    solver = DiagonalPreconditioner(mu=0.9)
    start = PathPoint(x=jnp.array([0.5]), f=1.0, g=jnp.array([3.0]))
    solver.restart(0, start)
    solver.step(0, PathPoint(x=jnp.zeros(1), f=1.0, g=jnp.array([-2.0])))
    # The blend leaves m positive, so the direction -m still points out of the
    # box while -g points back in. Read m here and this path is stalled away from
    # a descent it can still take.
    assert float(solver._state[0].m[0]) > 0.0

    lower_wall = Domain(jnp.asarray([[0.0, 1e6]]))

    assert solver.propose([0], lower_wall, _unread)[0] is not None


def test_propose_answers_the_whole_cohort_in_one_call() -> None:
    """One call covers every path asked about, each entry independent."""
    solver = DiagonalPreconditioner()
    live = PathPoint(x=jnp.zeros(2), f=1.0, g=jnp.array([1.0, 1.0]))
    solver.restart(0, live)
    solver.restart(1, live)
    solver.step(1, PathPoint(x=live.x, f=0.5, g=jnp.array([1e-9, 1e-9])))

    out = solver.propose([0, 1], wide(2), _unread)

    assert set(out) == {0, 1}
    assert out[1] is None  # path 1's stopping test holds
    assert out[0] is not None  # path 0 still steps
    assert close(out[0].x, live.x - live.g, 1e-12)


def test_propose_clips_its_own_step_into_the_domain() -> None:
    """The box is handed to `propose`, and the method clips: the engine will not."""
    box = Domain(jnp.stack([jnp.zeros(2), jnp.ones(2)], axis=1))
    solver = DiagonalPreconditioner()
    start = PathPoint(x=jnp.array([0.5, 0.5]), f=1.0, g=jnp.array([1.0, 0.0]))
    solver.restart(0, start)

    pending = solver.propose([0], box, _unread)[0]

    assert pending is not None
    # Only the out-of-bound coordinate is pinned to the wall: the clip is
    # per-coordinate, so the other one takes its step as usual.
    assert close(pending.x, jnp.array([0.0, 0.5]), 1e-12)


def _warmup(
    solver: DiagonalPreconditioner,
    quad: Callable[[jnp.ndarray], tuple[float, jnp.ndarray]],
) -> PathPoint:
    """Drive the warm-up by hand for path 0 the way the engine does.

    One unread proposal, graded and stepped: the warm-up always commits what it
    proposed, so there is no ruling to wait for. The start is whatever `restart`
    was last given - the method holds it, so nothing here passes it back in.
    """
    x = proposed(solver).x
    f, g = quad(x)
    trial = PathPoint(x=x, f=f, g=g)
    solver.step(0, trial)
    return trial


def test_diag_l2_step_is_newton_exact_on_a_quadratic() -> None:
    """Quadratic f = ½Σhᵢxᵢ²: the warm-up secant gives d = 1/h, so the step hits 0."""
    h = jnp.array([2.0, 4.0, 8.0])
    x0 = jnp.ones(3)

    def quad(x: jnp.ndarray) -> tuple[float, jnp.ndarray]:
        """Return (f, g) = (½Σhᵢxᵢ², h⊙x) at x."""
        return 0.5 * float(jnp.sum(h * x**2)), h * x

    f0, g0 = quad(x0)
    p = PathPoint(x=x0, f=f0, g=g0)
    solver = DiagonalPreconditioner(theta=0.0, mu=0.0)
    solver.restart(0, p)
    # alpha = 1 overshoots by a factor 25 (f = 215) and is committed anyway:
    # the secant is exact at any length on a quadratic, so the overshoot costs
    # one point rather than a backtrack walk.
    _warmup(solver, quad)
    # The warm-up ends re-armed at a full step: the next proposal is x - d·g,
    # and alpha only ever multiplies it.
    assert solver._state[0].alpha == 1.0
    assert not solver._state[0].is_warmup
    # y = h·s ⇒ √(S/Y) = 1/h ⇒ Δx = -(1/h)·h·x₁ = -x₁, the Newton step. The step
    # is taken from the point `step` left the path on, which is what `_warmup`
    # committed.
    assert close(proposed(solver).x, jnp.zeros(3), 1e-12)


def test_diag_abs_step_is_newton_exact_on_a_quadratic() -> None:
    """The abs spelling |s|/|y| gives the same 1/h on the same quadratic."""
    h = jnp.array([2.0, 4.0, 8.0])
    x0 = jnp.ones(3)

    def quad(x: jnp.ndarray) -> tuple[float, jnp.ndarray]:
        """Return (f, g) = (½Σhᵢxᵢ², h⊙x) at x."""
        return 0.5 * float(jnp.sum(h * x**2)), h * x

    f0, g0 = quad(x0)
    p = PathPoint(x=x0, f=f0, g=g0)
    solver = DiagonalPreconditioner(theta=0.0, moment="abs", mu=0.0)
    solver.restart(0, p)
    _warmup(solver, quad)
    assert close(proposed(solver).x, jnp.zeros(3), 1e-12)


def test_diag_step_scales_with_the_coordinate_unit() -> None:
    """Invariant: x̃ = c·x ⇒ d̃ = c²·d and Δx̃ = c·Δx, on one scaled secant.

    Pinned on `propose` with an injected secant pair, not through a warm-up: the
    warm-up step is deliberately *not* coordinate-invariant (its direction is
    -g, a gradient-like quantity that scales the wrong way), so driving this
    through a warm-up would test the opposite of the invariant. Here the pair
    (c·s, y/c) is the same physical secant read in coordinates scaled by c, and
    the step must scale with the coordinate.
    """
    s = jnp.array([1.0, -2.0])
    y = jnp.array([3.0, 4.0])
    g = jnp.array([0.5, -1.5])

    def step_under(c: jnp.ndarray) -> jnp.ndarray:
        """Return Δx for the secant (c·s, y/c) at gradient g/c."""
        p = PathPoint(x=jnp.zeros(2), f=0.0, g=g / c)
        best = PathPoint(x=c * s, f=-1.0, g=g / c + y / c)
        solver = DiagonalPreconditioner(theta=0.0)
        solver.restart(0, p)
        solver.step(0, best)
        return proposed(solver).x - best.x

    c = jnp.array([1e3, 1e-3])
    assert close(step_under(c), c * step_under(jnp.ones(2)), 1e-9)


def test_diag_gradient_floor_carries_a_flat_secant() -> None:
    """A flat secant (y = 0, constant slope) still yields a finite descending step."""
    g = jnp.array([2.0, -1.0])
    p = PathPoint(x=jnp.zeros(2), f=1.0, g=g)
    solver = DiagonalPreconditioner(theta=1e-4)
    solver.restart(0, p)
    x1 = p.x - g  # the full step, committed as proposed
    trial = PathPoint(x=x1, f=0.5, g=g)  # g unchanged ⇒ y = 0
    solver.step(0, trial)
    # d_i = |s_i|/(√θ·|g_i|) = |g_i|/(√θ·|g_i|) = 1/√θ = 100 for both dims.
    assert close(proposed(solver).x, x1 - 100.0 * g, 1e-9)
    step = proposed(solver).x - x1
    assert float(jnp.vdot(step, g)) < 0.0  # d > 0 ⇒ always a descent direction


def test_diag_floor_is_a_floor_not_an_addend() -> None:
    """Set den = max(Y, λ), not Y + λ: λ never deflates a curvature-dominated d.

    At the crossover Y = λ the two spellings differ by exactly √2, so a secant
    engineered onto that point makes the assert pick the max form out
    unambiguously (the additive form would propose √2 shorter per coordinate).
    """
    theta = 1e-4
    solver = DiagonalPreconditioner(theta=theta, mu=0.0)
    g0 = jnp.array([1.0, 0.5])
    p = PathPoint(x=jnp.zeros(2), f=1.0, g=g0)
    solver.restart(0, p)
    # g1 = g0/(1+√θ) puts the warm-up secant y = g0-g1 = √θ·g1 exactly on the
    # crossover Y = θ·g1² = λ (elementwise, in the mean-square spelling).
    g1 = g0 / (1.0 + math.sqrt(theta))
    trial = PathPoint(x=p.x - g0, f=0.9, g=g1)  # committed unread; f is not read
    solver.step(0, trial)
    lam = theta * jnp.square(g1)
    assert close(solver._state[0].y_ema, lam, 1e-12)  # secant sits on the crossover
    # max floor: d = √(S/max(Y, λ)) = √(g0²/θg1²) = (1+√θ)/√θ per coordinate.
    step = proposed(solver).x - trial.x
    scale = (1.0 + math.sqrt(theta)) / math.sqrt(theta)
    assert close(step, -scale * g1, 1e-9)
    # The additive form would propose 1/√2 of this; assert it does not.
    assert not close(step, -scale / math.sqrt(2.0) * g1, 1e-9)


def test_diag_decay_blends_later_secants() -> None:
    """The warm-up pair seeds the EMA; later pairs blend at β."""
    solver = DiagonalPreconditioner(beta=0.5, theta=0.0)
    g0 = jnp.array([1.0, 2.0])
    p = PathPoint(x=jnp.zeros(2), f=1.0, g=g0)
    solver.restart(0, p)
    x1 = p.x - g0  # the full step, committed unread
    g1 = jnp.array([0.5, 1.0])
    first = PathPoint(x=x1, f=0.9, g=g1)
    solver.step(0, first)
    s1, y1 = x1 - p.x, g1 - g0
    assert close(solver._state[0].s_ema, jnp.square(s1), 1e-12)
    assert close(solver._state[0].y_ema, jnp.square(y1), 1e-12)
    s2, y2 = jnp.array([0.25, -0.5]), jnp.array([0.1, 0.2])
    second = PathPoint(x=x1 + s2, f=0.8, g=g1 + y2)
    solver.step(0, second)
    s_expect = 0.5 * jnp.square(s1) + 0.5 * jnp.square(s2)
    y_expect = 0.5 * jnp.square(y1) + 0.5 * jnp.square(y2)
    assert close(solver._state[0].s_ema, s_expect, 1e-12)
    assert close(solver._state[0].y_ema, y_expect, 1e-12)


def test_diag_stops_only_when_the_gradient_norm_converged() -> None:
    """‖g‖ ≤ τ_rel·‖g_0‖ stops the path; one loud coordinate keeps the norm up."""
    solver = DiagonalPreconditioner(tau_rel=1e-3)
    p = PathPoint(x=jnp.zeros(2), f=1.0, g=jnp.array([1.0, 2.0]))
    solver.restart(0, p)
    # The start never trips the test: ‖g0‖ <= τ‖g0‖ is false for τ < 1.
    first = PathPoint(x=p.x - p.g, f=0.5, g=jnp.array([0.5, 1.0]))
    solver.step(0, first)  # the warm-up step, committed as proposed
    # Each point below is where a step landed, so handing it to `step` is also how
    # the path gets there: the stopping test reads the path, not a point handed to it.
    # dim 1's magnitude dominates the norm: not converged.
    partial = PathPoint(x=first.x, f=0.5, g=jnp.array([1e-5, 1.0]))
    solver.step(0, partial)
    assert proposed(solver)  # still stepping
    converged = PathPoint(x=partial.x, f=0.5, g=jnp.array([9e-4, 3e-4]))
    solver.step(0, converged)
    assert solver.propose([0], wide(2), _unread) == {0: None}


def test_diag_restart_rearms_the_warmup_and_reseeds_g0() -> None:
    """A restart drops the secant history, the warm flag and the old anchor."""
    solver = DiagonalPreconditioner()
    p = PathPoint(x=jnp.zeros(2), f=1.0, g=jnp.array([1.0, 2.0]))
    solver.restart(0, p)
    first = PathPoint(x=p.x - p.g, f=0.5, g=jnp.array([0.5, 1.0]))
    solver.step(0, first)
    assert not solver._state[0].is_warmup
    fresh = PathPoint(x=jnp.array([0.5, 0.5]), f=3.0, g=jnp.array([4.0, 8.0]))
    solver.restart(0, fresh)
    assert solver._state[0].is_warmup
    assert close(solver._state[0].g0, fresh.g, 1e-12)
    assert close(solver._state[0].s_ema, jnp.zeros(2), 1e-12)
    assert solver._state[0].alpha == 1.0  # the warm-up length is re-armed too
    assert close(proposed(solver).x, fresh.x + 1.0 * (-fresh.g), 1e-12)


# --- momentum blend (opt-in, mu > 0) ---

S_MOM = jnp.array([2.0, -1.0])
Y_MOM = jnp.array([1.0, 3.0])


def _solver_with_secant(
    mu: float,
    g: jnp.ndarray,
    *,
    s: jnp.ndarray = S_MOM,
    y: jnp.ndarray = Y_MOM,
) -> tuple[DiagonalPreconditioner, PathPoint]:
    """Return a solver holding one committed secant (s, y) and the point it sits on.

    The step start's gradient is ``g - y``, so the committed point's gradient is
    exactly ``g``: the same secant the engine would have committed after a step of
    ``s``. ``restart`` seeds the momentum chain with the step start's gradient and
    that first `step` blends the committed ``g`` into it, so the record is left
    holding ``mu·(g - y) + (1 - mu)·g`` - history only; the next ``propose``
    reads it as is.
    """
    solver = DiagonalPreconditioner(mu=mu, theta=0.0)
    start = PathPoint(x=jnp.zeros_like(g), f=1.0, g=g - y)
    best = PathPoint(x=start.x + s, f=0.5, g=g)
    solver.restart(0, start)
    solver.step(0, best)
    return solver, best


def _direction(solver: DiagonalPreconditioner) -> jnp.ndarray:
    """Return the proposed direction d⊙m, with the alpha multiplier divided out."""
    state = solver._state[0]
    return (state.point.x - proposed(solver).x) / state.alpha


def test_diag_momentum_is_the_committed_gradient_blend() -> None:
    """The direction is -d⊙m with m = (1-mu)·g + mu·m_prev, unprojected.

    No projection means the descent gain on g is *not* pinned to the plain
    step's: the blend adds ``mu·m_prev`` wherever the history helps or opposes
    g. What the step must reproduce exactly is the weighted average of the
    committed point's gradient and the history the commit blended it into.
    """
    g = jnp.array([0.5, -1.5])
    mu = 0.5
    plain, _ = _solver_with_secant(0.0, g)
    mom, _ = _solver_with_secant(mu, g)
    d_plain = _direction(plain)  # d⊙g, alpha divided out
    d_mom = _direction(mom)
    d = d_plain / g  # per-coordinate d; both solvers blend the same secant
    m_prev = g - Y_MOM  # restart seeded m with the step start's gradient
    assert close(d_mom, d * ((1.0 - mu) * g + mu * m_prev), 1e-12)
    assert not close(d_mom, d_plain, 1e-9)  # the momentum really was blended in
    # Unprojected: the gain is free to move off the plain step's value.
    assert not close(float(jnp.vdot(d_mom, g)), float(jnp.vdot(d_plain, g)), 1e-9)


def test_diag_propose_never_writes_the_momentum() -> None:
    """``propose`` reads the history and never writes it; only ``step`` does.

    Asked twice between commits it answers the same step both times: the model
    cannot advance on a point the engine has not committed, which is what keeps
    a secant a step that was actually taken.
    """
    g = jnp.array([0.5, -1.5])
    solver, at = _solver_with_secant(0.5, g)
    m_before = solver._state[0].m
    # `step` already blended the committed g into the seeded (g - y).
    assert close(m_before, 0.5 * (at.g - Y_MOM) + 0.5 * at.g, 1e-12)
    first = proposed(solver).x
    assert jnp.array_equal(solver._state[0].m, m_before)  # propose wrote nothing
    second = proposed(solver).x
    assert jnp.array_equal(solver._state[0].m, m_before)  # still nothing
    assert jnp.array_equal(first, second)  # nothing moved between the calls


def test_diag_restart_drops_the_momentum_history() -> None:
    """A restart re-seeds the momentum chain from the new start's gradient.

    The seed is the restart's own write, not a later step's: the `step` that
    follows only blends one gradient into it, so what the old path had
    accumulated cannot survive at any weight - the same reason S and Y are
    re-seeded outright rather than blended against.
    """
    g = jnp.array([0.5, -1.5])
    solver, at = _solver_with_secant(0.5, g)
    assert close(solver._state[0].m, 0.5 * (at.g - Y_MOM) + 0.5 * at.g, 1e-12)
    fresh = PathPoint(x=jnp.array([0.5, 0.5]), f=3.0, g=jnp.array([4.0, 8.0]))
    solver.restart(0, fresh)
    assert close(solver._state[0].m, fresh.g, 1e-12)  # nothing crossed the restart
    landed = PathPoint(x=fresh.x + S_MOM, f=0.5, g=fresh.g + Y_MOM)
    solver.step(0, landed)
    assert close(solver._state[0].m, 0.5 * fresh.g + 0.5 * landed.g, 1e-12)


def test_diag_mu_zero_leaves_no_history_behind() -> None:
    """The default keeps no memory in the direction: a weight of 0 stores none.

    The record still records a history at every ``mu`` - it costs one vector and
    is there for the moment the knob is turned up - so the property worth
    pinning is that `step` at ``mu = 0`` writes the arriving gradient outright,
    whatever the history held.
    """
    g = jnp.array([0.5, -1.5])
    solver, at = _solver_with_secant(0.0, g)
    solver._state[0] = replace(solver._state[0], m=jnp.array([1e6, -1e6]))
    landed = PathPoint(x=at.x + S_MOM, f=0.25, g=jnp.array([1.0, 2.0]))
    solver.step(0, landed)
    assert jnp.array_equal(solver._state[0].m, landed.g)


def test_diag_momentum_moves_only_where_d_has_support() -> None:
    """The step is zero where d = 0, however much history that coordinate holds.

    There is no projection and no ``<g,g>_d == 0`` special case under the
    current-momentum blend: the step is ``-alpha·d⊙m``, so a coordinate with
    d = 0 contributes nothing even when the history is live there. The history
    does reach a coordinate with d > 0 but g = 0 - heavy-ball's flat-direction
    build-up - which is why the blend is not a pure rotation of the plain step.
    """
    start = PathPoint(x=jnp.zeros(2), f=1.0, g=jnp.array([1.0, 0.5]))
    solver = DiagonalPreconditioner(mu=0.5, theta=0.0)
    solver.restart(0, start)
    # y = [-1, 0] against s = [1, 0] leaves d = [1, 0]; g lives on coordinate 1.
    at = PathPoint(x=start.x + jnp.array([1.0, 0.0]), f=0.5, g=jnp.array([0.0, 0.5]))
    solver.step(0, at)  # m = ½·start.g + ½·at.g, live on coordinate 0
    assert close(solver._state[0].m, 0.5 * start.g + 0.5 * at.g, 1e-12)
    step = at.x - proposed(solver).x
    assert close(step[1], 0.0, 1e-12)  # d = 0 there: no step, history or not
    assert not close(step[0], 0.0, 1e-12)  # d > 0 and history live: it moves


def test_diag_drives_a_quadratic_to_its_minimum_through_the_engine() -> None:
    """Engine round-trip: warm-up plus one diagonal step lands on the argmin."""
    h = jnp.array([2.0, 4.0])
    center = jnp.array([0.3, 0.4])

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        """Return value and gradient of f = ½Σhᵢ(xᵢ-cᵢ)², grad = h⊙(x-c)."""
        return jnp.sum(0.5 * h * (x - center) ** 2, axis=1), h * (x - center), None

    result = n_path_schedule(
        eval_f_g,
        **flock_kwargs(
            jnp.array([[0.5, 0.5]]),
            Domain(jnp.stack([jnp.zeros(2), jnp.ones(2)], axis=1)),
            key=jax.random.PRNGKey(0),
        ),
        max_iters=50,
        local_solver=DiagonalPreconditioner(theta=0.0, mu=0.0),
    )
    # mu = 0 pins the diagonal mechanics: with the momentum default on, the tail
    # is a detour rather than the exact Newton step, and the 1e-9 bound below
    # would be measuring the momentum instead.
    # θ = 0 makes the post-warm-up step exactly Newton, so g = 0 after it and the
    # next `propose` answers `None` (the stopping test); the point is still the
    # best, which is what `f`/`x` report.
    assert close(result.x, center, 1e-9)
    assert result.f < 1e-12


def test_diag_momentum_runs_through_the_engine() -> None:
    """Engine round-trip with momentum on: it runs, descends, stays finite.

    Not a convergence claim - near the argmin g → 0 while m does not, so the
    gradient-norm stopping test loses its anchor. That is the cost the default
    ``mu`` pays, and the reason the exactness tests above pin ``mu = 0``.
    What this pins is the engine's own restart→propose→step gluing: ``restart``
    must have seeded the path's momentum, or the steady propose raises.
    """
    h = jnp.array([2.0, 4.0])
    center = jnp.array([0.3, 0.4])
    x0 = jnp.array([[0.5, 0.5]])

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        """Return value and gradient of f = ½Σhᵢ(xᵢ-cᵢ)², grad = h⊙(x-c)."""
        return jnp.sum(0.5 * h * (x - center) ** 2, axis=1), h * (x - center), None

    result = n_path_schedule(
        eval_f_g,
        **flock_kwargs(
            x0,
            Domain(jnp.stack([jnp.zeros(2), jnp.ones(2)], axis=1)),
            key=jax.random.PRNGKey(0),
        ),
        max_iters=60,
        local_solver=DiagonalPreconditioner(theta=0.0, mu=0.9),
    )
    assert math.isfinite(result.f)
    assert result.f < 0.5 * float(jnp.sum(h * (x0[0] - center) ** 2))


def test_a_winner_that_grades_to_nan_stalls_the_path() -> None:
    """The engine, not the method, is what refuses a step it cannot commit.

    The start is served finite and off the minimizer, so its gradient is not
    stationary and the method takes its step; from the second batch on every
    service answers NaN, so the fused batch that grades the winner refuses it.
    The method proposed a point, not a ruling, so the refusal is not its to make:
    the engine's `is_usable` gate stalls the path, the frozen flock leaves it
    stalled, and the run ends still holding the start it was given.
    """
    n_batches = [0]

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        """Return f = Σ(x - ½)² and its gradient; NaN on every step batch."""
        n_batches[0] += 1
        d = x - 0.5
        f = jnp.sum(d**2, axis=1)
        if n_batches[0] > 1:  # the step's own batch, from here on
            f = jnp.full_like(f, jnp.nan)
        return f, 2.0 * d, None

    res = n_path_schedule(
        eval_f_g,
        **flock_kwargs(
            jnp.array([[0.2, 0.2]]),
            Domain(jnp.stack([jnp.zeros(2), jnp.ones(2)], axis=1)),
            key=jax.random.PRNGKey(0),
        ),
        max_iters=10,
        restart_stalled=False,
        local_solver=DiagonalPreconditioner(theta=0.0),
    )

    assert res.n_f_g_batches == 2, res.n_f_g_batches  # the flock + the winner
    assert res.status == "all_stalled"
    assert math.isfinite(float(res.f))  # the start it was left holding


def test_beta_zero_is_legal_and_reads_only_the_newest_secant() -> None:
    """`beta = 0` is the currentness end of the range: one pair, exactly.

    The smooth trades currentness for stability: it damps the step-to-step wobble
    the dropped coupling puts into the ratio. A diagonal problem has no coupling
    and nothing wobbling, so a caller may ask for the raw secant; the knob's range
    is [0, 1) and 0 blends each pair on top of the last instead of refusing to
    exist.
    """
    solver = DiagonalPreconditioner(beta=0.0, theta=0.0)
    p = PathPoint(x=jnp.zeros(2), f=1.0, g=jnp.array([1.0, 1.0]))
    solver.restart(0, p)
    solver.step(0, PathPoint(x=p.x - p.g, f=0.0, g=2.0 * p.g))
    assert close(solver._state[0].s_ema, jnp.square(p.g), 1e-12)
    # A second, different pair replaces the first rather than blending into it.
    # Its start is the point the path now stands on, not one passed in again.
    q = solver._state[0].point
    solver.step(0, PathPoint(x=q.x - q.g, f=0.0, g=2.0 * q.g))
    assert close(solver._state[0].s_ema, jnp.square(q.g), 1e-12)
    assert close(solver._state[0].y_ema, jnp.square(q.g), 1e-12)


def test_diag_knobs_out_of_range_are_rejected() -> None:
    """A knob out of its own range fails at construction, naming the knob.

    Each of these would still run and silently produce something other than the
    method's contract (beta = 1 never forgets, alpha0 = 0 never steps), so the
    object refuses to exist instead of degrading quietly.
    """

    def rejects(**knobs: float | str) -> str:
        try:
            # the knob under test is deliberately out of contract
            DiagonalPreconditioner(**knobs)  # pyright: ignore[reportArgumentType]
        except ValueError as e:
            return str(e)
        msg = f"accepted: {knobs}"
        raise AssertionError(msg)

    for bad in (
        {"beta": 1.0},
        {"theta": -1.0},
        {"tau_rel": 0.0},
        {"alpha0": 0.0},
    ):
        assert next(iter(bad)) in rejects(**bad), bad


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all solver tests passed")
