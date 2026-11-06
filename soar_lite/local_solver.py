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
from typing import Literal

import jax.numpy as jnp

from soar_lite.engine import LocalSolver, PathPoint, Verdict


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
    Newton: coupling Σ_{j≠i} Hᵢⱼ·sⱼ lands inside |y_i| and biases d.

    With ``mu > 0`` the direction is the *current* momentum,
    ``m = (1 - mu)·g + mu·m_prev``, where ``m_prev`` is an EMA of the past
    steps' start gradients (window ≈ ``1/(1-mu)``; the current step's gradient
    enters only once that step has committed, folded by ``fold``). At ``mu = 0``
    it degenerates to ``g`` exactly. This is the unprojected heavy-ball direction;
    being unprojected, it withdraws the descent-direction guarantee, so
    convergence rests on the engine's best-so-far bookkeeping and restart.

    Both smooths default to a window of about ``e`` steps, ``beta = mu = 0.63``.
    They stay two knobs - ``beta`` smooths the ratio that sets the step's scale,
    ``mu`` smooths the direction - because the two histories are different
    quantities.

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
        mu: float = 0.63,  # gradient-EMA decay for the direction; 0 = bare g
        alpha0: float = 1.0,  # the full step's length, in units of the natural step
        n_alpha: int = 1,  # alpha samples per step; 1 = no line search
        backtrack: float = 0.5,  # alpha *= backtrack per retry; grid spacing too
        tau_rel: float = 1e-6,  # gradient-norm stopping test vs ‖g_0‖
    ) -> None:
        """Store the knobs; per-path state is seeded by restart.

        ``alpha0`` is the step length every path is armed at (``_reset_step``),
        in units of the natural step: -g in the warm-up, -d⊙m after, and the
        full step of both phases under the default ``n_alpha=1``. It never grows
        back on its own: the alpha grid only samples down from it.
        ``backtrack`` is that grid's spacing (``n_alpha >= 2``), one constant for
        both phases. A knob outside its range raises ValueError here, once per
        object; nothing downstream re-checks it. ``moment`` is the one knob whose
        whole range is its annotation's (a ``Literal``), so it is not re-tested.

        Args:
            moment: which magnitude ratio to smooth, ``"abs"`` (mean of |s|,
                |y|) or ``"rms"`` (mean of s², y²).
            beta: EMA decay for the secant-magnitude smooth (window ≈ 1/(1-β)
                steps), in (0, 1).
            theta: gradient-norm floor coefficient (``λ = θ·|g_i|`` for "abs",
                ``θ·g_i²`` for "rms"), >= 0.
            mu: EMA decay of the gradient history blended into the direction,
                ``(1 - mu)·g + mu·m_prev``; 0 leaves the bare gradient, in
                [0, 1).
            alpha0: step length every path is armed at; the full step's
                length, > 0.
            n_alpha: alpha samples per step along the step's direction; 1 means
                no line search, >= 1.
            backtrack: the alpha-grid spacing, in (0, 1).
            tau_rel: relative gradient-norm stopping threshold, tested against
                ``‖g_0‖``, > 0.

        """
        for name, value, ok, want in (
            ("beta", beta, 0.0 < beta < 1.0, "in (0, 1)"),
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

        self.has_secant: dict[int, bool] = {}  # per path: first pair in hand?
        self.s_ema: dict[int, jnp.ndarray] = {}  # per path: S₁ ("abs") or S₂ ("rms")
        self.y_ema: dict[int, jnp.ndarray] = {}  # per path: Y₁ ("abs") or Y₂ ("rms")
        self.m: dict[int, jnp.ndarray] = {}  # per path: past-gradient EMA (mu > 0 only)

        self.alpha: dict[int, float] = {}  # per path: current step multiplier
        self.alpha_sample: dict[int, int] = {}  # per path: samples served this step
        self.saw_finite: dict[int, bool] = {}  # per path: any finite sample this step?

        self.g0: dict[int, jnp.ndarray] = {}  # per path: the stopping test's anchor

    def restart(self, i: int, p: PathPoint) -> None:
        """Seed path i at p: warm-up armed at alpha0, g0 kept, no estimate yet.

        The engine's only cold start, so this runs at initialization and after
        every restart - the secant history never carries across one.
        """
        # alpha0 is a scalar multiplier; the warm-up direction -g carries the
        # coordinate scale itself
        self.g0[i] = p.g
        self.has_secant[i] = False
        self.s_ema[i] = jnp.zeros_like(p.g)
        self.y_ema[i] = jnp.zeros_like(p.g)
        # m is left alone: `has_secant` is False again, so the next fold seeds
        # it from this start's gradient - the history never crosses a restart
        self._reset_step(i)

    def needs_verdict(self, i: int, p: PathPoint) -> bool:
        """Rule per candidate when the grid compares samples, or p stops the path.

        ``n_alpha > 1`` is the grid case: the step serves that many samples and
        each served value decides whether to shrink and try again. The phase does
        not enter - an ``n_alpha >= 2`` warm-up is ruled on its grid like any
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

        m is g, or the current-momentum blend when mu > 0 (see
        `_blend_momentum`).
        """
        if not self.has_secant[i]:
            return p.x + self.alpha[i] * (-p.g)
        d = self._diag(i, p.g)
        m = p.g if self.mu == 0.0 else self._blend_momentum(i, p.g)
        return p.x - self.alpha[i] * (d * m)

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
        ruling. Only the served trial's value `f` is read: `x` is in the
        protocol's signature for methods that judge geometry, so it is dropped
        here. The engine buys a
        trial's gradient only for the step's winner, so nothing here may depend
        on one; the stopping test reads ``p``, the step's start, whose gradient a
        path always has.
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
        first = not self.has_secant[i]
        self._fold(i, best.x - p.x, best.g - p.g, first=first)
        self.has_secant[i] = True
        if self.mu > 0.0:
            # The step's start gradient, folded only now that the step has
            # committed, so the momentum the next propose reads is history and
            # never the gradient that propose is about to use. `first` seeds the
            # EMA outright, exactly as it seeds S and Y.
            self.m[i] = p.g if first else self.mu * self.m[i] + (1.0 - self.mu) * p.g
        self._reset_step(i)

    def _reset_step(self, i: int) -> None:
        """Arm path i for a fresh step: full step, no alpha-grid sample served yet."""
        self.alpha[i] = self.alpha0
        self.alpha_sample[i] = 0
        self.saw_finite[i] = False

    def _diag(self, i: int, g: jnp.ndarray) -> jnp.ndarray:
        """Return the per-coordinate inverse curvature d_i for path i at gradient g."""
        # λ anchors to the current gradient's own magnitude, so a coordinate
        # with no usable secant (y_i = 0) still steps instead of 0/0. It is a
        # floor (max), not an addend: where Y ≥ λ the estimate is the faithful
        # S/Y ratio, and λ only caps d below that.
        floor = self.theta * (jnp.abs(g) if self.moment == "abs" else jnp.square(g))
        den = jnp.maximum(self.y_ema[i], floor)
        # den_i == 0 means y_i = 0 *and* g_i = 0: no curvature information and
        # already stationary in i, so step nowhere there rather than NaN. One
        # guard is enough - `where` is a total select and d is never
        # differentiated, so an inf/nan in the unselected lane cannot leak
        # anywhere; a derivative of this line would see it, so revisit then.
        d = jnp.where(den > 0.0, self.s_ema[i] / den, 0.0)
        return jnp.sqrt(d) if self.moment == "rms" else d

    def _blend_momentum(self, i: int, g: jnp.ndarray) -> jnp.ndarray:
        """Return path i's current momentum: (1 - mu)·g + mu·m_prev.

        ``m_prev`` (``self.m[i]``) is history only - the past step-start
        gradients, folded by `fold` once each step has committed - so this
        step's own gradient enters fresh, at weight ``1 - mu``. With ``mu = 0``
        this is ``g`` exactly, which is why `propose` skips the call then.
        """
        return (1.0 - self.mu) * g + self.mu * self.m[i]

    def _stationary(self, i: int, p: PathPoint) -> bool:
        """Return whether path i's gradient has hit its floor: ‖g‖ ≤ τ_rel·‖g_0‖.

        The anchor is ``g_0``, the gradient at the path's cold start, so the test
        at the start itself reads ‖g_0‖ ≤ τ·‖g_0‖ and holds only for a
        zero-gradient one. Read off the step's starting point ``p`` - the point
        ``verdict`` also reads - so the ask in ``needs_verdict`` and the ruling
        in ``verdict`` can never disagree.
        """
        return bool(jnp.linalg.norm(p.g) <= self.tau_rel * jnp.linalg.norm(self.g0[i]))

    def _step_verdict(self, i: int, f: float) -> Verdict:
        """Judge one alpha-grid sample on its value; never fold here.

        Reached by the engine only with ``n_alpha >= 2``, in either phase: a
        single-sample step answers False to ``needs_verdict`` and is committed
        unruled, and a path at its stopping test gives up in ``verdict`` before
        reaching here. One grid of ``n_alpha`` samples along the current step
        direction, shrinking alpha by the grid's spacing each retry; the win is
        handed to the engine on "accept", which commits its per-step argmin and
        folds the winner through ``fold``.

        Nothing non-finite may reach ``fold``: one inf folded into Y leaves
        den = inf forever, and that coordinate never steps again. A non-finite
        value drops out of the candidate set - the sweep keeps shrinking alpha,
        a real line search probing past a NaN - and a grid with no finite sample
        at all gives up.
        """
        if math.isfinite(f):
            self.saw_finite[i] = True
        self.alpha_sample[i] += 1
        if self.alpha_sample[i] < self.n_alpha:
            self.alpha[i] *= self.backtrack
            return "retry"
        if not self.saw_finite[i]:  # every sample this step was non-finite
            return "give_up"
        return "accept"

    def _fold(self, i: int, s: jnp.ndarray, y: jnp.ndarray, *, first: bool) -> None:
        """Fold one accepted secant (s, y) into path i's per-coordinate magnitude EMAs.

        The warm-up pair seeds the EMA outright (weight 1, no β-mixing against a
        zero); later pairs blend at β. Signed quantities are reduced to
        magnitudes first (|·| for "abs", squaring for "rms"), which is what makes the
        ratio positive and cancels the sign cancellation `E[s]/E[y]` suffers.
        """
        if self.moment == "abs":
            s_ema, y_ema = jnp.abs(s), jnp.abs(y)
        else:
            s_ema, y_ema = jnp.square(s), jnp.square(y)
        if first:
            self.s_ema[i], self.y_ema[i] = s_ema, y_ema
        else:
            self.s_ema[i] = self.beta * self.s_ema[i] + (1.0 - self.beta) * s_ema
            self.y_ema[i] = self.beta * self.y_ema[i] + (1.0 - self.beta) * y_ema
