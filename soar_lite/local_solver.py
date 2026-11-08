"""The default local solver: diagonal secant preconditioning (first order).

`DiagonalPreconditioner` descends along ``-d ⊙ m``, where the per-coordinate
scale ``d`` is the magnitude ratio of one secant pair (|Δx|/|Δy| in ``abs``
mode, √(Δx²/Δy²) in ``rms`` mode), smoothed over the path's history. With
``mu > 0`` the direction ``m`` is the current momentum, an unprojected blend of
the gradient and its own history; at ``mu = 0`` it is the bare gradient ``g``.
It implements the engine's `LocalSolver` contract and owns only direction, step
size and acceptance - no curvature model, no Hessian.
"""

import math
from dataclasses import dataclass, replace
from typing import Literal

import jax.numpy as jnp

from soar_lite.engine import LocalSolver, PathPoint, Verdict


@dataclass(frozen=True, eq=False)
class _PathState:
    """One path's secant estimate, momentum history, and current step.

    Three lifetimes share the record, which is why the fields are grouped by
    them:

    - the curvature estimate and the stopping test's anchor are re-seeded by
      every restart and otherwise persist for the whole run;
    - ``m``, the direction's history, is seeded by the first committed step and
      is the one field a restart keeps;
    - the step's ``alpha`` progress is re-armed after every commit and every
      restart.

    Frozen so that both resets are one named construction (`restarted`,
    `armed`) rather than an assignment order a later edit can break: a field
    only one of them names cannot be cleared by the other's accident.
    ``eq=False`` keeps a stray ``==`` off a field-wise array comparison.
    """

    # the curvature estimate: seeded by `restart`, blended once per committed step
    has_secant: bool
    s_ema: jnp.ndarray
    y_ema: jnp.ndarray
    # the stopping test's anchor: the gradient at the path's cold start
    g0: jnp.ndarray
    # the direction's history: past step-start gradients, re-seeded at every
    # restart to that start's own gradient, so the history never crosses one
    m: jnp.ndarray
    # this step's progress: re-armed after every commit and every restart
    alpha: float
    alpha_sample: int
    saw_finite: bool

    @classmethod
    def restarted(cls, g0: jnp.ndarray, alpha0: float) -> "_PathState":
        """Return the state a path cold-starts at: no secant, nothing folded yet."""
        return cls(
            has_secant=False,
            s_ema=jnp.zeros_like(g0),
            y_ema=jnp.zeros_like(g0),
            g0=g0,
            m=g0,
            alpha=alpha0,
            alpha_sample=0,
            saw_finite=False,
        )

    def armed(self, alpha0: float) -> "_PathState":
        """Return this path's state armed for a fresh step, estimate and history kept.

        ``alpha0`` is a scalar multiplier, not a step length in x: the warm-up
        direction ``-g`` carries the coordinate scale itself.
        """
        return replace(self, alpha=alpha0, alpha_sample=0, saw_finite=False)


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
    enters only once that step has committed, folded by ``fold``). At ``mu = 0``
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

    Two phases, one step mechanic. Every step samples the alpha grid
    {alpha0, alpha0·backtrack, alpha0·backtrack², …} along its own direction and
    hands the win to the engine's per-step argmin, which ``fold`` records. The
    phases differ only in the direction: the warm-up (the first step) has no
    secant pair yet, so it descends along -g from ``alpha0``, and afterwards the
    direction is -d⊙m. The default ``n_alpha=1`` commits the full step
    unconditionally; a larger grid rules its samples against each other, never
    against the step's start, so nothing here is monotone.

    ``needs_verdict`` is therefore phase-independent: it answers True when the
    grid has more than one sample, and also whenever the stopping test already
    holds - ``verdict`` is the contract's only channel for "give_up", so a
    converged path must ask for its own last ruling rather than be committed
    unjudged and step off a stationary point.

    The stopping test is a gradient-norm test, ‖g‖ ≤ τ_rel·‖g_0‖, which is how a
    path reports that it is done. The norm rather than a per-coordinate test
    keeps one flat coordinate from pinning a path alive (or one huge-start
    coordinate from declaring it dead), and keeps τ_rel dimensionless.

    The derivation behind both spellings - the secant equation read one
    coordinate at a time, and why no matrix is stored - is the explanation
    page's local-step section ([how it works](../explanation/how-it-works.md)).
    """

    def __init__(  # noqa: PLR0913 - each knob is a distinct, independently-swept facet
        self,
        *,
        moment: Literal["abs", "rms"] = "rms",  # mean ratio / mean-square ratio
        beta: float = 0.63,  # EMA decay; the window is ≈ 1/(1-β) ≈ e steps
        theta: float = 1e-3,  # gradient-norm floor: λ = θ·|g_i| or θ·g_i²
        mu: float = 0.0,  # gradient-EMA decay for the direction; 0 = bare g, default
        alpha0: float = 1.0,  # the full step's length, in units of the natural step
        n_alpha: int = 1,  # alpha samples per step; 1 = no line search
        backtrack: float = 0.5,  # alpha *= backtrack per retry; grid spacing too
        tau_rel: float = 1e-6,  # gradient-norm stopping test vs ‖g_0‖
    ) -> None:
        """Store the knobs; per-path state is seeded by restart.

        ``alpha0`` is the step length every path is armed at, in units of the
        natural step: -g in the warm-up, -d⊙m after, and the full step of both
        phases under the default ``n_alpha=1``. It never grows
        back on its own: the alpha grid only samples down from it.
        ``backtrack`` is that grid's spacing (``n_alpha >= 2``), one constant for
        both phases. A numeric knob outside its range raises ValueError here, once
        per object; nothing downstream re-checks it. ``moment`` is the one knob
        whose whole range is its annotation's (a ``Literal``), so it is left to
        the type checker: at run time any value but ``"abs"`` takes the
        ``"rms"`` branch.

        Args:
            moment: which magnitude ratio to smooth, ``"abs"`` (mean of |s|,
                |y|) or ``"rms"`` (mean of s², y²).
            beta: EMA decay for the secant-magnitude smooth (window ≈ 1/(1-β)
                steps), in [0, 1); 0 keeps no history, so ``d`` reads the
                newest accepted secant.
            theta: gradient-norm floor coefficient (``λ = θ·|g_i|`` for "abs",
                ``θ·g_i²`` for "rms"), >= 0.
            mu: EMA decay of the gradient history blended into the direction,
                ``(1 - mu)·g + mu·m_prev``; 0 (the default) leaves the bare
                gradient, in [0, 1).
            alpha0: step length every path is armed at; the full step's
                length, > 0.
            n_alpha: alpha samples per step along the step's direction; 1 means
                no line search, >= 1.
            backtrack: the alpha-grid spacing, in (0, 1).
            tau_rel: relative gradient-norm stopping threshold, tested against
                ``‖g_0‖``, > 0.

        """
        for name, value, ok, want in (
            ("beta", beta, 0.0 <= beta < 1.0, "in [0, 1)"),
            ("theta", theta, theta >= 0.0, ">= 0"),
            ("mu", mu, 0.0 <= mu < 1.0, "in [0, 1)"),
            ("alpha0", alpha0, alpha0 > 0.0, "> 0"),
            ("n_alpha", n_alpha, n_alpha >= 1, ">= 1"),
            ("backtrack", backtrack, 0.0 < backtrack < 1.0, "in (0, 1)"),
            ("tau_rel", tau_rel, tau_rel > 0.0, "> 0"),
        ):
            if not ok:
                msg = f"{name} must be {want}, got {value!r}"
                raise ValueError(msg)
        self.moment = moment
        self.beta = beta
        self.theta = theta
        self.mu = mu

        self.alpha0 = alpha0
        self.n_alpha = n_alpha
        self.backtrack = backtrack

        self.tau_rel = tau_rel

        # One record per path, seeded by `restart`.
        self._state: dict[int, _PathState] = {}

    def restart(self, i: int, p: PathPoint) -> None:
        """Seed path i at p: warm-up armed at alpha0, g0 kept, no estimate yet.

        The engine's only cold start, so this runs at initialization and after
        every restart - the secant history never carries across one. The
        momentum history is re-seeded at this start's own gradient for the same
        reason: a restart is where a path's history begins again.
        """
        # alpha0 is a scalar multiplier; the warm-up direction -g carries the
        # coordinate scale itself
        self._state[i] = _PathState.restarted(p.g, self.alpha0)

    def needs_verdict(self, i: int, p: PathPoint) -> bool:
        """Answer True when the grid rules per candidate, or the stopping test holds.

        ``n_alpha > 1`` is the grid case: the step serves that many samples,
        shrinking alpha by the grid's spacing between them. The phase does not
        enter - an ``n_alpha >= 2`` warm-up is ruled on its grid like any
        steady step.

        The second case is the stopping test: a path already at its
        gradient-norm floor has nothing left to rule on, but ``verdict`` is the
        contract's only channel for "give_up", so it asks for that one ruling.
        Anywhere else a single-sample step is committed as proposed, and the
        engine's gate on the winner's value and gradient is the whole of its
        acceptance.
        """
        return self.n_alpha > 1 or self._stationary(i, p)

    def propose(self, i: int, p: PathPoint) -> jnp.ndarray:
        """Warm-up: x + alpha·(-g) at the current alpha. Afterwards: x - alpha·d⊙m.

        `m` is the bare gradient `g` at ``mu = 0``, and the current momentum
        ``(1 - mu)·g + mu·m_prev`` otherwise.
        """
        state = self._state[i]
        if not state.has_secant:
            return p.x + state.alpha * (-p.g)
        diag = self._diag(i, p.g)
        m = p.g if self.mu == 0.0 else self._blend_momentum(i, p.g)
        return p.x - state.alpha * (diag * m)

    def verdict(
        self,
        i: int,
        p: PathPoint,
        x: jnp.ndarray,
        f: float,
    ) -> Verdict:
        """Give up at a stationary point; otherwise rule one alpha-grid sample.

        Only consulted for a path that answered True to ``needs_verdict``, and
        the phase does not enter: ``n_alpha > 1`` rules a warm-up grid and a
        steady grid alike, while a single-sample step is committed without a
        ruling. `x` is in the protocol's signature for methods that judge
        geometry, so it is dropped here.

        The grid's rule is a count, not a comparison: every sample but the last
        answers "retry", and the last closes the step - "accept" if any sample
        this step served was finite, "give_up" if none was. So `f` is read for
        that one bit, its finiteness; which sample won is the engine's per-step
        argmin, not a choice made here. The stopping test reads ``p``, the step's start,
        whose gradient a path always has - a trial's own gradient is not read,
        since the engine may buy it only for the step's winner.
        """
        del x

        # Checked on the *current* point, so a path that has just landed on a
        # stationary one gives up on the next iteration (costing that one trial's
        # eval, which the engine's best still counts) rather than stepping
        # off it. g0 is the anchor: the same test at the start is ‖g0‖ ≤ τ‖g0‖,
        # true only for a zero-gradient start.
        if self._stationary(i, p):
            return "give_up"
        return self._step_verdict(i, f)

    def fold(self, i: int, p: PathPoint, best: PathPoint) -> None:
        """Fold the engine's committed argmin for path i (the step that won).

        Only the winner is folded, never an intermediate grid sample: an
        intermediate secant is a step the path never took, and folding it would
        smear the magnitude EMAs. The engine calls this with the exact point it
        just committed, so the folded secant is the accepted step. The first
        call after a restart is the warm-up pair and seeds the EMAs outright;
        later calls blend at beta - `has_secant` doubles as the first-fold marker,
        so it is set here rather than in ``verdict``. This is also where a step
        is re-armed: a ruling the engine did not commit (a winner whose gradient
        it refused) leaves the method untouched rather than half-reset.
        """
        state = self._state[i]
        first = not state.has_secant
        s_ema, y_ema = self._fold_ema(state, best.x - p.x, best.g - p.g, first=first)
        momentum = state.m
        if self.mu > 0.0:
            # The step's start gradient, folded only now that the step has
            # committed, so the momentum the next propose reads is history and
            # never the gradient that propose is about to use. The first fold
            # seeds the EMA outright, exactly as it seeds S and Y.
            momentum = p.g if first else self.mu * state.m + (1.0 - self.mu) * p.g
        self._state[i] = replace(
            state,
            has_secant=True,
            s_ema=s_ema,
            y_ema=y_ema,
            m=momentum,
        ).armed(self.alpha0)

    def _reset_step(self, i: int) -> None:
        """Arm path i for a fresh step: full step, no alpha-grid sample served yet."""
        self._state[i] = self._state[i].armed(self.alpha0)

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

    def _blend_momentum(self, i: int, g: jnp.ndarray) -> jnp.ndarray:
        """Return path i's current momentum: (1 - mu)·g + mu·m_prev.

        ``m_prev`` (``state.m``) is history only - the past step-start
        gradients, folded by `fold` once each step has committed - so this
        step's own gradient enters fresh, at weight ``1 - mu``. With ``mu = 0``
        this is ``g`` exactly, which is why `propose` skips the call then.
        """
        return (1.0 - self.mu) * g + self.mu * self._state[i].m

    def _stationary(self, i: int, p: PathPoint) -> bool:
        """Return whether path i's gradient has hit its floor: ‖g‖ ≤ τ_rel·‖g_0‖.

        The anchor is ``g_0``, the gradient at the path's cold start, so the test
        at the start itself reads ‖g_0‖ ≤ τ·‖g_0‖ and holds only for a
        zero-gradient one. Read off the step's starting point ``p`` - the point
        ``verdict`` also reads - so the ask in ``needs_verdict`` and the ruling
        in ``verdict`` can never disagree.
        """
        return bool(
            jnp.linalg.norm(p.g) <= self.tau_rel * jnp.linalg.norm(self._state[i].g0),
        )

    def _step_verdict(self, i: int, f: float) -> Verdict:
        """Judge one alpha-grid sample on its value; never fold here.

        Reached by the engine only with ``n_alpha >= 2``, in either phase: a
        single-sample step answers False to ``needs_verdict`` and is committed
        unruled, and a path at its stopping test gives up in ``verdict`` before
        reaching here. One grid of ``n_alpha`` samples along the current step
        direction, shrinking alpha by the grid's spacing each retry; the win is
        handed to the engine on "accept", which commits its per-step argmin and
        folds the winner through ``fold`` once that winner passes the engine's
        finite gate.

        Nothing non-finite may reach ``fold``: one inf folded into Y leaves
        den = inf forever, and that coordinate never steps again. A non-finite
        value drops out of the candidate set - the sweep keeps shrinking alpha,
        a real line search probing past a NaN - and a grid with no finite sample
        at all gives up.
        """
        state = self._state[i]
        saw_finite = state.saw_finite or math.isfinite(f)
        n_served = state.alpha_sample + 1
        if n_served < self.n_alpha:
            self._state[i] = replace(
                state,
                saw_finite=saw_finite,
                alpha_sample=n_served,
                alpha=state.alpha * self.backtrack,
            )
            return "retry"
        self._state[i] = replace(state, saw_finite=saw_finite, alpha_sample=n_served)
        if not saw_finite:  # every sample this step was non-finite
            return "give_up"
        return "accept"

    def _fold_ema(
        self,
        state: _PathState,
        s: jnp.ndarray,
        y: jnp.ndarray,
        *,
        first: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Fold one accepted secant (s, y) into the path's magnitude EMAs.

        The warm-up pair seeds the EMA outright (weight 1, no β-mixing against a
        zero); later pairs blend at β. Signed quantities are reduced to
        magnitudes first (|·| for "abs", squaring for "rms"), which is what makes the
        ratio positive and cancels the sign cancellation `E[s]/E[y]` suffers.
        """
        if self.moment == "abs":
            s_mag, y_mag = jnp.abs(s), jnp.abs(y)
        else:
            s_mag, y_mag = jnp.square(s), jnp.square(y)
        if first:
            return s_mag, y_mag
        return (
            self.beta * state.s_ema + (1.0 - self.beta) * s_mag,
            self.beta * state.y_ema + (1.0 - self.beta) * y_mag,
        )
