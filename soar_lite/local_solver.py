"""The default local solver: diagonal secant preconditioning (first order).

`DiagonalPreconditioner` descends along ``-d ⊙ m``, where the per-coordinate
scale ``d`` is the magnitude ratio of one secant pair (|Δx|/|Δy| in ``abs``
mode, √(Δx²/Δy²) in ``rms`` mode), smoothed over the path's history. With
``mu > 0`` the direction ``m`` is the current momentum, an unprojected blend of
the gradient and its own history; at ``mu = 0`` it is the bare gradient ``g``.
It implements the engine's `LocalSolver` contract and owns only direction, step
size and acceptance - no full curvature model and no Hessian.
"""

from dataclasses import dataclass, replace
from typing import Literal

import jax.numpy as jnp

from soar_lite.domain import Domain
from soar_lite.engine import LocalSolver, PathPoint, Pending, ServeTrials


@dataclass(frozen=True, eq=False)
class _PathState:
    """One path's point, secant estimate, momentum history, and current step.

    Three lifetimes share the record, which is why the fields are grouped by
    them:

    - the point the path stands at is written only when a point arrives, and the
      contract hands one over in exactly two places: `restart` places it and
      `step` moves it. `propose` is not handed it again, so a method that steps
      from a point keeps it - this field is that copy;
    - the curvature estimate, the direction's history, and the stopping test's
      anchor live for the whole run: the first two are blended into by every
      committed step, and a restart re-seeds all three at the new start;
    - the step's ``alpha`` progress is re-armed after every commit and every
      restart.

    Frozen so that both resets are one named construction (`restarted`,
    `armed`) rather than an assignment order a later edit can break: a field
    only one of them names cannot be cleared by the other's accident.
    ``eq=False`` keeps a stray ``==`` off a field-wise array comparison.
    """

    # where the path stands: `restart` places it and `step` moves it, nothing
    # else writes it - the two calls that pair with the engine's own two writes
    # to its copy, so the two can never disagree
    p: PathPoint
    # no pair in hand yet: set by `restart`, cleared by the first committed step,
    # so `propose` descends along -g and that first `step` seeds outright
    warm_up: bool
    # the curvature estimate: seeded by `restart`, blended once per committed step
    s_ema: jnp.ndarray
    y_ema: jnp.ndarray
    # the direction's history: the gradient at each committed point, EMA'd at
    # `mu` (0 only leaves it out of the direction), re-seeded at every restart to
    # that start's own gradient, so the history never crosses one
    m: jnp.ndarray
    # the stopping test's anchor: the gradient at the path's cold start
    g0: jnp.ndarray
    # this step's progress: re-armed after every commit and every restart
    alpha: float

    @classmethod
    def restarted(cls, p: PathPoint, alpha0: float) -> "_PathState":
        """Return the state a path cold-starts at from p: no secant, nothing in it."""
        return cls(
            p=p,
            warm_up=True,
            s_ema=jnp.zeros_like(p.g),
            y_ema=jnp.zeros_like(p.g),
            m=p.g,
            g0=p.g,
            alpha=alpha0,
        )

    def armed(self, alpha0: float) -> "_PathState":
        """Return this path's state armed for a fresh step, estimate and history kept.

        ``alpha0`` is a scalar multiplier, not a step length in x: the warm-up
        direction ``-g`` carries the coordinate scale itself.
        """
        return replace(self, alpha=alpha0)


class DiagonalPreconditioner(LocalSolver):
    """Per-coordinate secant magnitude ratio with a gradient-norm floor.

    Each coordinate carries its own effective inverse curvature d_i, of
    dimension [x]²/[f], estimated from the magnitude ratio of one secant
    (s = x_k - x_{k-1}, y = g_k - g_{k-1}), and the step is Δx_i = -d_i·g_i.
    Two equivalent spellings, picked by ``moment`` (default "rms"):

        "abs": S₁ = EMA(|s|), Y₁ = EMA(|y|), d = S₁/max(Y₁, λ)
        "rms": S₂ = EMA(s²), Y₂ = EMA(y²), d = √(S₂/max(Y₂, λ))

    Both are unit-invariant: scaling x by c scales d by c² and the step by c,
    while scaling f by c_f scales d by 1/c_f and leaves the step alone.

    The floor λ = θ·|g_i| ("abs") or θ·g_i² ("rms"), θ dimensionless, is
    anchored to the *current* gradient, so a flat-but-not-converged coordinate
    (y → 0 while g ≠ 0) still gets a finite step; a converged coordinate is
    caught by the stopping test instead. It joins via ``max``, so λ only guards
    Y → 0 and never deflates a curvature-dominated coordinate.
    d ≥ 0 by construction, so at ``mu = 0`` -d⊙g descends and the plain step
    needs no acceptance test. It is a *diagonal* preconditioner, not a diagonal
    Newton: coupling Σ_{j≠i} Hᵢⱼ·sⱼ lands inside |y_i| and makes the secant
    ratio depend on the step's direction; the EMA spans a few steps so ``d``
    tracks the underlying curvature instead of that direction. Where a path
    keeps one direction the smooth only lags it; on a diagonal problem no
    coupling enters, each pair already reads 1/Hᵢᵢ exactly, and ``beta = 0``
    reads the newest secant.

    With ``mu > 0`` the direction is the *current* momentum,
    ``m = (1 - mu)·g + mu·m_prev``, where ``m_prev`` is an EMA of the past
    steps' start gradients (window ≈ ``1/(1-mu)``; the current step's gradient
    enters only once that step has committed, blended in by ``step``). At ``mu = 0``
    it degenerates to ``g`` exactly. This is the unprojected heavy-ball direction;
    being unprojected, it withdraws the descent-direction guarantee, so
    convergence rests on the engine's best-so-far bookkeeping and restart.

    ``beta`` defaults to a window of about 2.7 steps (0.63); ``mu`` defaults to
    0, no memory. The window that helps a direction belongs to how far the
    curvatures spread once ``d`` has taken the per-coordinate scale out (and to
    the shape of that spread, not only its widest ratio), and to how noisy the
    objective is - neither is knowable from inside the solver, so the default is
    the branch that assumes neither, and it is also the branch that keeps the
    descent guarantee stated above. A caller who knows their spectrum, or whose
    objective is noisy (fp32 evaluation, a simulation, a sampled average), sets
    ``mu`` accordingly; there is no window to copy from here. It is not a repair
    for the diagonal approximation either: momentum re-shapes the iteration
    along the direction ``d`` already gives, and the dropped off-diagonal
    curvature stays dropped. The two smooths stay two knobs - ``beta`` smooths
    the ratio that sets the step's scale, ``mu`` smooths the direction - because
    the two histories are different quantities.

    Two phases, one step mechanic. Every step takes the full proposal at
    ``alpha0`` along its own direction, and ``step`` records the committed step.
    The phases differ only in the direction: the warm-up (the first step) has no
    secant pair yet, so it descends along -g from ``alpha0``, and afterwards the
    direction is -d⊙m.

    A path whose stopping test holds describes no step: ``propose`` answers
    ``None`` for it and the engine stalls it, so a converged path is not stepped
    off its stationary point.

    The stopping test is a projected-gradient test, ‖P g‖ ≤ τ_rel·‖g_0‖: at a
    wall the components whose descent direction points out of the box are
    dropped (a constrained minimum sits on a wall with a nonzero gradient), and
    the rest is
    tested. That is how a path reports that it is done. The norm rather than a
    per-coordinate test keeps one flat coordinate from pinning a path alive (or
    one huge-start coordinate from declaring it dead), and keeps τ_rel
    dimensionless.

    The derivation behind both spellings - the secant equation read one
    coordinate at a time, and why no matrix is stored - is the explanation
    page's local-step section ([how it works](../explanation/how-it-works.md)).
    """

    def __init__(  # noqa: PLR0913 - each knob is a distinct, independently-swept facet
        self,
        *,
        moment: Literal["abs", "rms"] = "rms",  # mean ratio / mean-square ratio
        beta: float = 0.63,  # EMA decay; the window is ≈ 1/(1-β) ≈ e steps
        mu: float = 0.0,  # gradient-EMA decay for the direction; 0 = bare g, default
        theta: float = 1e-3,  # gradient-norm floor: λ = θ·|g_i| or θ·g_i²
        alpha0: float = 1.0,  # direction multiplier; a step fraction in steady state
        tau_rel: float = 1e-6,  # gradient-norm stopping test vs ‖g_0‖
    ) -> None:
        """Store the knobs; per-path state is seeded by restart.

        ``alpha0`` multiplies the step's direction, and the two phases give that
        multiplier different units. In the steady state ``-d⊙m`` is already a
        displacement, so ``alpha0`` reads as a fraction of the natural step. In
        the warm-up the direction is ``-g``, a gradient-like ``[f]/[x]``
        quantity, so ``alpha0`` carries ``[x]²/[f]`` there and no unit
        invariance holds - at the default ``alpha0 = 1`` the warm-up step is
        literally ``x - g``. One knob covers both phases. A numeric knob outside
        its range raises ValueError here, once per object; nothing downstream
        re-checks it. ``moment`` is the one knob
        whose whole range is its annotation's (a ``Literal``), so it is left to
        the type checker: at run time any value but ``"abs"`` takes the
        ``"rms"`` branch.

        Args:
            moment: which magnitude ratio to smooth, ``"abs"`` (mean of |s|,
                |y|) or ``"rms"`` (mean of s², y²).
            beta: EMA decay for the secant-magnitude smooth (window ≈ 1/(1-β)
                steps), in [0, 1); 0 keeps no history, so ``d`` reads the
                newest accepted secant.
            mu: EMA decay of the gradient history blended into the direction,
                ``(1 - mu)·g + mu·m_prev``; 0 (the default) leaves the bare
                gradient, in [0, 1).
            theta: gradient-norm floor coefficient (``λ = θ·|g_i|`` for "abs",
                ``θ·g_i²`` for "rms"), >= 0.
            alpha0: direction multiplier every path is armed at, > 0; the two
                phases read it in different units (see above).
            tau_rel: relative gradient-norm stopping threshold, tested against
                ``‖g_0‖``, > 0.

        """
        for name, value, ok, want in (
            ("beta", beta, 0.0 <= beta < 1.0, "in [0, 1)"),
            ("mu", mu, 0.0 <= mu < 1.0, "in [0, 1)"),
            ("theta", theta, theta >= 0.0, ">= 0"),
            ("alpha0", alpha0, alpha0 > 0.0, "> 0"),
            ("tau_rel", tau_rel, tau_rel > 0.0, "> 0"),
        ):
            if not ok:
                msg = f"{name} must be {want}, got {value!r}"
                raise ValueError(msg)
        self.moment = moment
        self.beta = beta
        self.mu = mu
        self.theta = theta

        self.alpha0 = alpha0

        self.tau_rel = tau_rel

        # One record per path, seeded by `restart`.
        self._state: dict[int, _PathState] = {}

    def restart(self, i: int, p: PathPoint) -> None:
        """Seed path i at p: warm-up armed at alpha0, g0 kept, no estimate yet.

        The engine's only cold start, so this runs at initialization and after
        every restart - the secant history never carries across one. The
        momentum history is re-seeded at this start's own gradient for the same
        reason: a restart is where a path's history begins again. The point
        itself is kept, because no later call hands it back: it is the first of
        the contract's two arrivals.
        """
        # alpha0 is a scalar multiplier; the warm-up direction -g carries the
        # coordinate scale itself
        self._state[i] = _PathState.restarted(p, self.alpha0)

    def step(self, i: int, p: PathPoint) -> None:
        """Take the step the engine just committed for path i.

        ``p`` means what it means in `restart`: the point the path now stands
        at, handed over once. The point it stepped from is the record's own, so
        the pair the step observed is read entirely off (record, p).

        Only the winner is taken in, never a trial the method's own enumeration
        rejected: a secant from a point the path never stood on would smear the
        magnitude EMAs. The engine calls this with the exact point it just
        committed, so both histories see what the step observed: the pair
        (Δx, Δg) for the curvature, and the gradient at the point arrived at for
        the momentum. The first call after a restart is the warm-up pair and
        seeds the secant EMAs outright; later calls blend at beta - the record's
        ``warm_up`` is that same first-step marker, cleared here. This is also
        where a step is re-armed: a winner the engine refuses never reaches
        here, so the method is left untouched rather than half-reset.
        """
        state = self._state[i]
        s_ema, y_ema = self._blended_secant(state, p.x - state.p.x, p.g - state.p.g)
        # Read only now that the step has committed, and read from the point it
        # committed to: the gradient this step started from was already spent on
        # this step's direction, so the arriving one is the news. `restart`
        # seeds the descent with the start's own gradient, which is why every
        # step after it blends rather than seeding; at mu = 0 the blend keeps
        # only the arriving gradient, which is the default's bare direction.
        m = self.mu * state.m + (1.0 - self.mu) * p.g
        self._state[i] = replace(
            state,
            p=p,
            warm_up=False,
            s_ema=s_ema,
            y_ema=y_ema,
            m=m,
        ).armed(self.alpha0)

    def propose(
        self, idxs: list[int], domain: Domain, serve_trials: ServeTrials
    ) -> dict[int, Pending | None]:
        """Propose one step for every path in ``idxs``, or ``None`` if it is done.

        A `Pending` here carries the clipped proposal alone, both halves still
        owed, so the engine's `settle` can buy its f and g in one fused batch
        together with everything else the iteration produced. ``None`` means the
        path's stopping test holds: the engine stalls it where it stands and
        takes no step.

        The proposal is clipped into ``domain`` here rather than by the engine,
        because a step is the method's own and the engine only clips restart
        starts up front.

        ``serve_trials`` is the value-only evaluation hook, for a method that
        rules its own trial grid; this one takes a single full step, so it leaves
        the hook unused and lets `settle` buy the gradient. That is also why
        every proposal comes back with both halves owed: nothing was read.

        Warm-up: x + alpha·(-g) at the current alpha. Afterwards: x + alpha·(-d⊙m).
        Steps from the point the record holds - `restart` placed it and `step`
        moved it - so the step's start is not passed in a second time. ``m`` is
        the history ``step`` left behind: the EMA over the gradients at
        committed points, which includes the gradient of the point this step
        starts from. At ``mu = 0`` the last step stored that gradient outright,
        so the direction is the bare gradient.
        """
        del serve_trials  # single full step: nothing to rule, so nothing to read
        return {
            i: Pending(jnp.clip(self._propose(i), domain.lo, domain.hi), None, None)
            if not self._stationary(i, domain)
            else None
            for i in idxs
        }

    def _propose(self, i: int) -> jnp.ndarray:
        state = self._state[i]
        if state.warm_up:
            return state.p.x + state.alpha * (-state.p.g)
        diag = self._diag(i, state.p.g)
        return state.p.x + state.alpha * (-diag * state.m)

    def _stationary(self, i: int, domain: Domain) -> bool:
        """Return whether path i's projected gradient has hit its floor.

        The test is ``‖P g‖ ≤ τ_rel·‖g_0‖``: a component on a bound whose
        descent direction points out of the box is dropped from ``g`` before the
        norm, because a constrained minimum can sit on a wall with a large
        gradient - positive against the lower wall, negative against the upper
        one - and the raw norm would never fall there. What is left is the
        residual a projected-gradient method steps along, so the test reads the
        descent the path can still take rather than the one the box forbids.

        The projection reads ``g`` and not the direction's momentum ``m``: a stop
        is a claim about the objective, so it must not depend on the path's
        history. With the default ``mu = 0`` the two are the same vector, and the
        choice only shows where momentum lags a gradient that has turned.

        The anchor is ``g_0``, the gradient at the path's cold start, so the test
        at the start itself reads ‖g_0‖ ≤ τ·‖g_0‖ and holds only for a
        zero-gradient one. Read off the record's own point - the step's start,
        the point ``propose`` would step from - so the test and the step can
        never read different points.

        That anchor is deliberately not projected: it is the start's full
        gradient, wall reaction included. The test is relative to the start's
        overall scale, not to each component's own, so dropping the pinned part
        would re-import the per-coordinate judgment this norm exists to reject,
        make the bar depend on which coordinates happened to be stuck, and push
        it below what those coordinates can resolve. The wall reaction is real
        forcing in the caller's units; it belongs in the scale.
        """
        state = self._state[i]
        g, x = state.p.g, state.p.x
        # A component is pinned where its descent direction -g_i points out of
        # the box: g_i > 0 against the lower wall, g_i < 0 against the upper one.
        # Zeroing exactly those is the projection; every other component still has
        # a feasible descent direction.
        g_free = jnp.where(
            ((x == domain.lo) & (g > 0.0)) | ((x == domain.hi) & (g < 0.0)),
            0.0,
            g,
        )
        return bool(jnp.linalg.norm(g_free) <= self.tau_rel * jnp.linalg.norm(state.g0))

    def _diag(self, i: int, g: jnp.ndarray) -> jnp.ndarray:
        """Return the per-coordinate inverse curvature d_i for path i at gradient g."""
        # λ anchors to the current gradient's own magnitude, so a coordinate
        # with no usable secant (y_i = 0) still steps instead of 0/0. It is a
        # floor (max), not an addend: where Y ≥ λ the estimate is the faithful
        # S/Y ratio, and λ only caps d below that.
        floor = self.theta * (jnp.abs(g) if self.moment == "abs" else jnp.square(g))
        state = self._state[i]
        den = jnp.maximum(state.y_ema, floor)
        # den_i == 0 means y_i = 0 *and* g_i = 0: no curvature information and
        # already stationary in i, so step nowhere there rather than NaN. One
        # guard is enough - `where` is a total select and d is never
        # differentiated, so an inf/nan in the unselected lane cannot leak
        # anywhere; a derivative of this line would see it, so revisit then.
        d = jnp.where(den > 0.0, state.s_ema / den, 0.0)
        return jnp.sqrt(d) if self.moment == "rms" else d

    def _blended_secant(
        self, state: _PathState, s: jnp.ndarray, y: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return the magnitude EMAs of the secant pair (s, y), blended in.

        A warm-up record (no pair in hand yet) seeds the EMA outright (weight 1,
        no β-mixing against a zero); later pairs blend at β. Signed quantities
        are reduced to magnitudes first (|·| for "abs", squaring for "rms"),
        which is what makes the ratio positive and cancels the sign cancellation
        `E[s]/E[y]` suffers.
        """
        if self.moment == "abs":
            s_mag, y_mag = jnp.abs(s), jnp.abs(y)
        else:
            s_mag, y_mag = jnp.square(s), jnp.square(y)
        if state.warm_up:
            return s_mag, y_mag
        return (
            self.beta * state.s_ema + (1.0 - self.beta) * s_mag,
            self.beta * state.y_ema + (1.0 - self.beta) * y_mag,
        )
