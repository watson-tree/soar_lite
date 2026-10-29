"""Self-checks for the diagonal preconditioner, this package's local method.

Pins the secant ratio, the gradient-norm floor (a floor, not an addend), the two
entry knobs, the decay, the gradient-norm stopping test, and the unit
invariance of the steady-state step. No engine wiring. Run with:

    uv run python -m tests.test_local_solver
"""

import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from soar_lite.engine import (
    Bounds,
    PathPoint,
    n_path_schedule,
)
from soar_lite.global_model import UniformGlobalModel
from soar_lite.local_solver import DiagonalPreconditioner

jax.config.update("jax_enable_x64", val=True)


def close(
    a: jnp.ndarray | float,
    b: jnp.ndarray | float,
    tol: float = 1e-5,
) -> bool:
    """Return True iff all elements of a and b agree within tol."""
    return bool(jnp.max(jnp.abs(a - b)) < tol)


def flock_kwargs(
    x0: jnp.ndarray,
    bounds: Bounds,
    *,
    key: jnp.ndarray | None = None,
) -> dict[str, Any]:
    """Return the entry's two flock kwargs: the run's width and its starts.

    The engine takes its width as `n_paths` and every point a path starts from as
    the global model's first proposal, so a test that chooses its flock says both
    here. The model is the shipped default one: it hands `x0` back on the first
    `propose` and draws any later re-draw from `key`.
    """
    return {
        "n_paths": int(x0.shape[0]),
        "global_model": UniformGlobalModel(
            bounds,
            key=jax.random.PRNGKey(0) if key is None else key,
            x0=x0,
        ),
    }


# --- DiagonalPreconditioner ---


def test_diag_warmup_is_one_unruled_gradient_step() -> None:
    """The warm-up is the steady state's step: one sample along -g, no ruling."""
    solver = DiagonalPreconditioner()
    p = PathPoint(x=jnp.zeros(3), f=2.0, g=jnp.array([3.0, 0.0, 4.0]))
    solver.restart(0, p)
    assert not solver.has_secant[0]
    # n_alpha=1 and a gradient far above its floor: nothing to compare
    assert not solver.needs_verdict(0, p)
    assert solver.alpha[0] == 1.0
    assert close(solver.propose(0, p), p.x + 1.0 * (-p.g), 1e-12)
    # Committed exactly as proposed: the engine never asks for a ruling, and the
    # step's value need not beat the start's.
    above = PathPoint(x=p.x - p.g, f=9.0, g=p.g)
    solver.fold(0, p, above)
    assert solver.has_secant[0]
    assert close(solver.s_ema[0], jnp.square(-p.g), 1e-12)


def test_diag_warmup_is_ruled_when_the_grid_compares_samples() -> None:
    """n_alpha >= 2 rules the warm-up like any step: the grid's argmin decides."""
    solver = DiagonalPreconditioner(n_alpha=2)
    p = PathPoint(x=jnp.zeros(2), f=1.0, g=jnp.array([1.0, 1.0]))
    solver.restart(0, p)
    assert solver.needs_verdict(0, p)
    # No descent test any more: a sample worse than the start still advances the
    # grid, because only the sibling comparison will decide the winner.
    assert solver.verdict(0, p, p.x - p.g, 7.0) == "retry"
    assert solver.alpha[0] == solver.backtrack
    assert solver.verdict(0, p, p.x - solver.backtrack * p.g, 8.0) == "accept"


def test_a_stationary_path_asks_for_the_ruling_that_ends_it() -> None:
    """n_alpha=1 rules nothing else, but the stopping test needs its own verdict.

    `verdict` is the contract's only channel for "give_up", and the engine only
    asks a path that answered True to `needs_verdict` - so a converged path has
    to ask, or it is committed unjudged and steps off its stationary point
    forever, never reporting the stall a restart rule reads.
    """
    solver = DiagonalPreconditioner()
    start = PathPoint(x=jnp.zeros(2), f=1.0, g=jnp.array([1.0, 1.0]))
    solver.restart(0, start)
    floor = PathPoint(x=start.x, f=1.0, g=jnp.array([1e-9, -1e-9]))
    assert solver.needs_verdict(0, floor)
    assert solver.verdict(0, floor, floor.x, floor.f) == "give_up"


def test_diag_warmup_grid_drops_a_nonfinite_sample() -> None:
    """A non-finite warm-up sample is a retry; an all-non-finite grid gives up."""
    for bad in (math.nan, math.inf, -math.inf):
        solver = DiagonalPreconditioner(n_alpha=2)
        p = PathPoint(x=jnp.zeros(2), f=1.0, g=jnp.array([1.0, 1.0]))
        solver.restart(0, p)
        assert solver.verdict(0, p, p.x - p.g, bad) == "retry"
        assert solver.verdict(0, p, p.x, bad) == "give_up"


def _warm_up(
    solver: DiagonalPreconditioner,
    p: PathPoint,
    quad: Callable[[jnp.ndarray], tuple[float, jnp.ndarray]],
) -> PathPoint:
    """Drive the warm-up by hand for path 0 the way the engine does.

    An unruled step is committed as proposed; a ruled one walks its alpha grid
    until the method accepts.
    """
    for _ in range(20):  # the loop bound is this helper's, not the engine's
        x = solver.propose(0, p)
        f, g = quad(x)
        trial = PathPoint(x=x, f=f, g=g)
        ruled = solver.needs_verdict(0, p)
        if not ruled or solver.verdict(0, p, trial.x, trial.f) == "accept":
            solver.fold(0, p, trial)
            return trial
    raise AssertionError  # 20 trials without an accepted warm-up step


def test_diag_l2_step_is_newton_exact_on_a_quadratic() -> None:
    """Quadratic f = ½Σhᵢxᵢ²: the warm-up secant gives d = 1/h, so the step hits 0."""
    h = jnp.array([2.0, 4.0, 8.0])
    x0 = jnp.ones(3)

    def quad(x: jnp.ndarray) -> tuple[float, jnp.ndarray]:
        """Return (f, g) = (½Σhᵢxᵢ², h⊙x) at x."""
        return 0.5 * float(jnp.sum(h * x**2)), h * x

    f0, g0 = quad(x0)
    p = PathPoint(x=x0, f=f0, g=g0)
    solver = DiagonalPreconditioner(theta=0.0)
    solver.restart(0, p)
    # alpha = 1 overshoots by a factor 25 (f = 215) and is committed anyway:
    # the secant is exact at any length on a quadratic, so the overshoot costs
    # one point rather than a backtrack walk.
    trial = _warm_up(solver, p, quad)
    # The warm-up ends re-armed at a full step: the next proposal is x - d·g
    # (alpha 1.0), and alpha only ever acts as the alpha-grid's multiplier.
    assert solver.alpha[0] == 1.0
    assert solver.has_secant[0]
    # y = h·s ⇒ √(S/Y) = 1/h ⇒ Δx = -(1/h)·h·x₁ = -x₁, the Newton step.
    assert close(solver.propose(0, trial), jnp.zeros(3), 1e-12)


def test_diag_abs_step_is_newton_exact_on_a_quadratic() -> None:
    """The abs spelling |s|/|y| gives the same 1/h on the same quadratic."""
    h = jnp.array([2.0, 4.0, 8.0])
    x0 = jnp.ones(3)

    def quad(x: jnp.ndarray) -> tuple[float, jnp.ndarray]:
        """Return (f, g) = (½Σhᵢxᵢ², h⊙x) at x."""
        return 0.5 * float(jnp.sum(h * x**2)), h * x

    f0, g0 = quad(x0)
    p = PathPoint(x=x0, f=f0, g=g0)
    solver = DiagonalPreconditioner(theta=0.0, moment="abs")
    solver.restart(0, p)
    trial = _warm_up(solver, p, quad)
    assert close(solver.propose(0, trial), jnp.zeros(3), 1e-12)


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
        solver.fold(0, p, best)
        return solver.propose(0, best) - best.x

    c = jnp.array([1e3, 1e-3])
    assert close(step_under(c), c * step_under(jnp.ones(2)), 1e-9)


def test_diag_gradient_floor_carries_a_flat_secant() -> None:
    """A flat secant (y = 0, constant slope) still yields a finite descending step."""
    g = jnp.array([2.0, -1.0])
    p = PathPoint(x=jnp.zeros(2), f=1.0, g=g)
    solver = DiagonalPreconditioner(theta=1e-4)
    solver.restart(0, p)
    x1 = p.x - g  # alpha = 1, accepted: f = 0.5 <= 1 - 1e-4·5
    trial = PathPoint(x=x1, f=0.5, g=g)  # g unchanged ⇒ y = 0
    assert solver.verdict(0, p, trial.x, trial.f) == "accept"
    solver.fold(0, p, trial)
    # d_i = |s_i|/(√θ·|g_i|) = |g_i|/(√θ·|g_i|) = 1/√θ = 100 for both dims.
    assert close(solver.propose(0, trial), x1 - 100.0 * g, 1e-9)
    step = solver.propose(0, trial) - x1
    assert float(jnp.vdot(step, g)) < 0.0  # d > 0 ⇒ always a descent direction


def test_diag_floor_is_a_floor_not_an_addend() -> None:
    """Set den = max(Y, λ), not Y + λ: λ never deflates a curvature-dominated d.

    At the crossover Y = λ the two spellings differ by exactly √2, so a secant
    engineered onto that point makes the assert pick the max form out
    unambiguously (the additive form would propose √2 shorter per coordinate).
    """
    theta = 1e-4
    solver = DiagonalPreconditioner(theta=theta)
    g0 = jnp.array([1.0, 0.5])
    p = PathPoint(x=jnp.zeros(2), f=1.0, g=g0)
    solver.restart(0, p)
    # g1 = g0/(1+√θ) puts the warm-up secant y = g0-g1 = √θ·g1 exactly on the
    # crossover Y = θ·g1² = λ (elementwise, in the mean-square spelling).
    g1 = g0 / (1.0 + math.sqrt(theta))
    trial = PathPoint(x=p.x - g0, f=0.9, g=g1)  # committed unruled; f is not read
    assert solver.verdict(0, p, trial.x, trial.f) == "accept"
    solver.fold(0, p, trial)
    lam = theta * jnp.square(g1)
    assert close(solver.y_ema[0], lam, 1e-12)  # secant sits on the crossover
    # max floor: d = √(S/max(Y, λ)) = √(g0²/θg1²) = (1+√θ)/√θ per coordinate.
    step = solver.propose(0, trial) - trial.x
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
    x1 = p.x - g0  # alpha = 1, the full step, committed unruled
    g1 = jnp.array([0.5, 1.0])
    first = PathPoint(x=x1, f=0.9, g=g1)
    assert solver.verdict(0, p, first.x, first.f) == "accept"
    solver.fold(0, p, first)
    s1, y1 = x1 - p.x, g1 - g0
    assert close(solver.s_ema[0], jnp.square(s1), 1e-12)
    assert close(solver.y_ema[0], jnp.square(y1), 1e-12)
    s2, y2 = jnp.array([0.25, -0.5]), jnp.array([0.1, 0.2])
    second = PathPoint(x=x1 + s2, f=0.8, g=g1 + y2)
    assert solver.verdict(0, first, second.x, second.f) == "accept"
    solver.fold(0, first, second)
    assert close(solver.s_ema[0], 0.5 * jnp.square(s1) + 0.5 * jnp.square(s2), 1e-12)
    assert close(solver.y_ema[0], 0.5 * jnp.square(y1) + 0.5 * jnp.square(y2), 1e-12)


def test_diag_gives_up_only_when_the_gradient_norm_converged() -> None:
    """‖g‖ ≤ τ_rel·‖g_0‖ gives up; one loud coordinate keeps the norm up."""
    solver = DiagonalPreconditioner(tau_rel=1e-3)
    p = PathPoint(x=jnp.zeros(2), f=1.0, g=jnp.array([1.0, 2.0]))
    solver.restart(0, p)
    # The start never trips the test: ‖g0‖ <= τ‖g0‖ is false for τ < 1.
    first = PathPoint(x=p.x - p.g, f=0.5, g=jnp.array([0.5, 1.0]))
    assert solver.verdict(0, p, first.x, first.f) == "accept"  # warm-up
    solver.fold(0, p, first)
    # dim 1's magnitude dominates the norm: not converged.
    partial = PathPoint(x=first.x, f=0.5, g=jnp.array([1e-5, 1.0]))
    assert solver.verdict(0, partial, partial.x, partial.f) == "accept"
    converged = PathPoint(x=first.x, f=0.5, g=jnp.array([9e-4, 3e-4]))
    assert solver.verdict(0, converged, converged.x, converged.f) == "give_up"


def test_diag_gives_up_on_a_nonfinite_trial() -> None:
    """Past the warm-up a non-finite trial stalls rather than committing or looping."""
    for bad in (math.nan, math.inf, -math.inf):
        solver = DiagonalPreconditioner()
        p = PathPoint(x=jnp.zeros(2), f=1.0, g=jnp.array([1.0, 1.0]))
        solver.restart(0, p)
        warm = PathPoint(x=p.x - p.g, f=0.5, g=jnp.array([0.5, 0.5]))
        assert solver.verdict(0, p, warm.x, warm.f) == "accept"
        solver.fold(0, p, warm)
        trial = PathPoint(x=warm.x - 0.1 * warm.g, f=bad, g=warm.g)
        assert solver.verdict(0, warm, trial.x, trial.f) == "give_up"


def test_diag_judges_a_trial_on_its_value_alone() -> None:
    """A trial is handed over as x and f: its gradient cannot decide anything.

    The engine buys a trial's gradient only for the step's winner and gates it
    there, so the method's ruling is a function of f (and the step's start). A
    non-finite *value* is still a ruling: past the warm-up the only sample gives
    up, and nothing about a gradient can change that.
    """
    solver = DiagonalPreconditioner()
    p = PathPoint(x=jnp.zeros(2), f=1.0, g=jnp.array([1.0, 1.0]))
    solver.restart(0, p)
    warm = PathPoint(x=p.x - p.g, f=0.5, g=jnp.array([0.5, 0.5]))
    assert solver.verdict(0, p, warm.x, warm.f) == "accept"
    solver.fold(0, p, warm)
    assert solver.verdict(0, warm, warm.x, 0.4) == "accept"


def test_diag_restart_rearms_the_warmup_and_reseeds_g0() -> None:
    """A restart drops the secant history, the warm flag and the old anchor."""
    solver = DiagonalPreconditioner()
    p = PathPoint(x=jnp.zeros(2), f=1.0, g=jnp.array([1.0, 2.0]))
    solver.restart(0, p)
    first = PathPoint(x=p.x - p.g, f=0.5, g=jnp.array([0.5, 1.0]))
    assert solver.verdict(0, p, first.x, first.f) == "accept"
    solver.fold(0, p, first)
    assert solver.has_secant[0]
    fresh = PathPoint(x=jnp.array([0.5, 0.5]), f=3.0, g=jnp.array([4.0, 8.0]))
    solver.restart(0, fresh)
    assert not solver.has_secant[0]
    assert close(solver.g0[0], fresh.g, 1e-12)
    assert close(solver.s_ema[0], jnp.zeros(2), 1e-12)
    assert solver.alpha[0] == 1.0  # the warm-up length is re-armed too
    assert close(solver.propose(0, fresh), fresh.x + 1.0 * (-fresh.g), 1e-12)


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
            jnp.stack([jnp.zeros(2), jnp.ones(2)], axis=1),
            key=jax.random.PRNGKey(0),
        ),
        bounds=jnp.stack([jnp.zeros(2), jnp.ones(2)], axis=1),
        max_iters=50,
        local_solver=DiagonalPreconditioner(theta=0.0),
    )
    # θ = 0 makes the post-warm-up step exactly Newton, so g = 0 after it and
    # the next verdict gives up (the stopping test); the point is still the
    # best, which is what `fun`/`x` report.
    assert close(result.x, center, 1e-9)
    assert result.fun < 1e-12


def test_diag_line_search_walks_the_grid_and_folds_the_argmin() -> None:
    """n_alpha=3: retry, retry, then accept; fold blends the argmin."""
    h = jnp.array([2.0, 4.0, 8.0])
    x0 = jnp.ones(3)

    def quad(x: jnp.ndarray) -> tuple[float, jnp.ndarray]:
        """Return (f, g) = (½Σhᵢxᵢ², h⊙x) at x."""
        return 0.5 * float(jnp.sum(h * x**2)), h * x

    f0, g0 = quad(x0)
    p = PathPoint(x=x0, f=f0, g=g0)
    solver = DiagonalPreconditioner(theta=0.0, n_alpha=3)
    solver.restart(0, p)
    p = _warm_up(solver, p, quad)  # secant in hand; alpha re-armed to 1.0
    trials: list[PathPoint] = []
    verdicts: list[str] = []
    for _ in range(3):  # n_alpha=3: exactly three grid samples
        x = solver.propose(0, p)
        f, g = quad(x)
        trial = PathPoint(x=x, f=f, g=g)
        verdicts.append(solver.verdict(0, p, trial.x, trial.f))
        trials.append(trial)
    assert verdicts == ["retry", "retry", "accept"]
    # The grid shrank by the backtrack constant: 1.0, bt, bt².
    bt = solver.backtrack
    assert close(trials[1].x - p.x, bt * (trials[0].x - p.x), 1e-12)
    assert close(trials[2].x - p.x, bt * bt * (trials[0].x - p.x), 1e-12)
    # The engine commits the argmin (here the full step, Newton-exact) and folds
    # only that secant, blended into the warm-up pair at beta.
    argmin = min(trials, key=lambda t: t.f)
    s_old, y_old = solver.s_ema[0], solver.y_ema[0]
    solver.fold(0, p, argmin)
    s, y = argmin.x - p.x, argmin.g - p.g
    b = solver.beta  # the EMA decay the fold blends at
    assert close(solver.s_ema[0], b * s_old + (1.0 - b) * jnp.square(s), 1e-12)
    assert close(solver.y_ema[0], b * y_old + (1.0 - b) * jnp.square(y), 1e-12)
    assert solver.alpha[0] == solver.alpha0  # next step re-arms at the full step


def test_diag_line_search_gives_up_when_no_sample_is_finite() -> None:
    """A grid with every sample non-finite gives up rather than committing."""
    h = jnp.array([2.0, 4.0])
    x0 = jnp.ones(2)

    def quad(x: jnp.ndarray) -> tuple[float, jnp.ndarray]:
        """Return (f, g) = (½Σhᵢxᵢ², h⊙x) at x."""
        return 0.5 * float(jnp.sum(h * x**2)), h * x

    f0, g0 = quad(x0)
    p = PathPoint(x=x0, f=f0, g=g0)
    solver = DiagonalPreconditioner(theta=0.0, n_alpha=3)
    solver.restart(0, p)
    p = _warm_up(solver, p, quad)
    for rnd in range(3):  # two shrinks, then the last sample finds no finite win
        x = solver.propose(0, p)
        trial = PathPoint(x=x, f=math.nan, g=jnp.full(2, math.nan))
        v = solver.verdict(0, p, trial.x, trial.f)
        assert v == ("give_up" if rnd == 2 else "retry")


def test_diag_line_search_drives_a_quadratic_through_the_engine() -> None:
    """The engine's accept + fold round-trip still lands on the argmin."""
    h = jnp.array([2.0, 4.0])
    center = jnp.array([0.3, 0.4])

    def eval_f_g(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        """Return value and gradient of f = ½Σhᵢ(xᵢ-cᵢ)², grad = h⊙(x-c)."""
        return jnp.sum(0.5 * h * (x - center) ** 2, axis=1), h * (x - center), None

    result = n_path_schedule(
        eval_f_g,
        **flock_kwargs(
            jnp.array([[0.5, 0.5]]),
            jnp.stack([jnp.zeros(2), jnp.ones(2)], axis=1),
            key=jax.random.PRNGKey(0),
        ),
        bounds=jnp.stack([jnp.zeros(2), jnp.ones(2)], axis=1),
        max_iters=50,
        local_solver=DiagonalPreconditioner(theta=0.0, n_alpha=3),
    )
    assert close(result.x, center, 1e-9)
    assert result.fun < 1e-12


def test_diag_knobs_out_of_range_are_rejected() -> None:
    """A knob out of its own range fails at construction, naming the knob.

    Each of these would still run and silently produce something other than the
    method's contract (beta = 1 never forgets, n_alpha = 0 serves no
        sample, alpha0 = 0 never steps),
    so the object refuses to exist instead of degrading quietly.
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
        {"moment": "l2"},
        {"n_alpha": 0},
        {"alpha0": 0.0},
        {"backtrack": 1.0},
    ):
        assert next(iter(bad)) in rejects(**bad), bad


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all solver tests passed")
