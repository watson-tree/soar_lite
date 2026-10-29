"""The default local solver: diagonal secant preconditioning (first order).

`DiagonalPreconditioner` descends along ``-d ⊙ g``, where the per-coordinate
scale ``d`` is the magnitude ratio of one secant pair (|Δx|/|Δy| in ``abs``
mode, √(Δx²/Δy²) in ``rms`` mode), smoothed over the path's history. It
implements the engine's `LocalSolver` contract and owns only direction, step
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
    while scaling f by c_f scales d by 1/c_f and leaves the step alone. The
    floor λ = θ·|g_i| ("abs") or θ·g_i² ("rms"), θ dimensionless, is anchored to the
    *current* gradient, so a flat-but-not-converged coordinate (y → 0 while
    g ≠ 0) still gets a finite step; a truly converged coordinate is caught by
    the stopping test below instead. The floor joins via ``max``, not ``+``:
    λ is a pure guard on Y → 0, never a deflation of a curvature-dominated
    coordinate (the additive spelling shrinks every d by √(Y/(Y+λ)), so `max`
    is the arm that leaves a curvature-dominated coordinate undeformed).
    d ≥ 0 by construction, so -d⊙g is always a
    descent direction and needs no acceptance test. Named "diagonal
    preconditioner", not "diagonal Newton": coupling Σ_{j≠i} Hᵢⱼ·sⱼ lands inside
    |y_i| and makes d a biased estimator of 1/Hᵢᵢ.

    Two phases per path, one step mechanic. Every step samples the alpha grid
    {alpha0, alpha0·backtrack, alpha0·backtrack², …} along its own direction and
    hands the win to the engine's per-step argmin, which ``fold`` records. The
    phases differ only in the direction: the warm-up (the first step) has no
    secant pair yet, so it descends along -g from the start length ``alpha0``
    (default 1) - a magnitude ratio needs one secant pair before it can estimate
    anything. The shrink factor (``backtrack``) makes the grid coordinate-free
    (alpha multiplies the direction, and the constant is dimensionless), but the
    direction scales the wrong way: a step must add a length-like direction,
    while -g scales like a gradient (with 1/x under x̃=cx), not a length, so the
    warm-up step still carries the coordinate scale. Nothing scalar can repair
    that; a length-like direction could (physical spans, or a box-width
    scaling), which matters only if the working box ever stops being [0,1]ᵈ -
    where the engine's clip absorbs it today. Once that pair is in hand the
    direction becomes -d⊙g, which needs no acceptance test at all (d ≥ 0 ⇒
    descent), and the default (n_alpha=1) commits the full step
    unconditionally: no backoff, an overshoot is left to the next secant to
    correct (Zhang-Li is deliberately dropped). Nothing nonmonotone is used in
    either phase, because neither phase tests monotonicity: a grid sample is
    ruled against its siblings, never against the step's start.

    ``needs_verdict`` is therefore phase-independent: it answers True exactly
    when the grid has more than one sample to compare (n_alpha > 1). A
    single-sample step - warm-up or steady - is committed as proposed, so the
    engine never asks for a ruling and its own gate on the winner's value and
    gradient is the whole of its acceptance.

    The stopping test is a gradient-norm test: ‖g‖ ≤ τ_rel·‖g_0‖ gives up, the
    engine's only way to report a path that is done. The norm (not the
    per-coordinate test |g_i| ≤ τ_rel·|g_0,i|) keeps one flat coordinate from
    pinning a path "alive" (a flat-start coordinate that never satisfies the
    test) or "dead" (a huge-start coordinate that passes immediately): only the
    gradient's aggregated size relative to its starting size counts, and τ_rel
    stays dimensionless. That is also the hook a Zhang-Li / λ_min saddle escape
    would attach to; it is not implemented.

    Derivation: the secant equation ``B·s = y`` read one coordinate at a time
    (``y_i = B_ii·s_i``, couplings dropped) gives ``d_i = 1/B_ii = s_i/y_i`` -
    the ratio of the observed displacement to the observed gradient change in
    that coordinate. The step is then ``dx = -diag(B)^{-1}·g = -d⊙g``:
    multiplying by ``d`` is what turns a gradient-like direction (whose scale
    runs opposite to the coordinate's) into a displacement-like one. The same
    object falls out of the diagonal degeneration of the SR1 update; the secant
    equation is the shorter route to it. Nothing here is a matrix method: no B
    is stored, and the smoothing acts on the secant pair, not on the ratio.
    """

    def __init__(  # noqa: PLR0913 - each knob is a distinct, independently-swept facet
        self,
        *,
        beta: float = 0.95,  # EMA decay; the window is ≈ 1/(1-β) steps
        theta: float = 1e-3,  # gradient-norm floor: λ = θ·|g_i| or θ·g_i²
        tau_rel: float = 1e-6,  # gradient-norm stopping test vs ‖g_0‖
        moment: Literal["abs", "rms"] = "rms",  # mean ratio / mean-square ratio
        alpha0: float = 1.0,  # the full step's length, in units of the natural step
        n_alpha: int = 1,  # alpha samples per step; 1 = no line search
        backtrack: float = 0.5,  # alpha *= backtrack per retry; grid spacing too
    ) -> None:
        """Store the knobs; per-path state is seeded by restart.

        ``alpha0`` is the step length every path is armed at (``_reset_step``),
        in units of the natural step: -g in the warm-up, -d⊙g after, and the
        full step of both phases under the default ``n_alpha=1``. It never grows
        back on its own: the alpha grid only samples down from it.
        ``backtrack`` is that grid's spacing (``n_alpha >= 2``), one constant
        for both phases. A knob outside its own range raises ValueError here,
        once per object; nothing downstream re-checks it.

        Args:
            beta: EMA decay for the secant-magnitude smooth (window ≈ 1/(1-β)
                steps), in (0, 1).
            theta: gradient-norm floor coefficient (``λ = θ·|g_i|`` for "abs",
                ``θ·g_i²`` for "rms"), >= 0.
            tau_rel: relative gradient-norm stopping threshold, tested against
                ``‖g_0‖``, > 0.
            moment: which magnitude ratio to smooth, ``"abs"`` (mean of |s|,
                |y|) or ``"rms"`` (mean of s², y²).
            alpha0: step length every path is armed at; the full step's
                length, > 0.
            n_alpha: alpha samples per step along the step's direction; 1 means
                no line search, >= 1.
            backtrack: the alpha-grid spacing, in (0, 1).

        """
        for name, value, ok, want in (
            ("beta", beta, 0.0 < beta < 1.0, "in (0, 1)"),
            ("theta", theta, theta >= 0.0, ">= 0"),
            ("tau_rel", tau_rel, tau_rel > 0.0, "> 0"),
            ("moment", moment, moment in ("abs", "rms"), '"abs" or "rms"'),
            ("alpha0", alpha0, alpha0 > 0.0, "> 0"),
            ("n_alpha", n_alpha, n_alpha >= 1, ">= 1"),
            ("backtrack", backtrack, 0.0 < backtrack < 1.0, "in (0, 1)"),
        ):
            if not ok:
                msg = f"{name} must be {want}, got {value!r}"
                raise ValueError(msg)
        self.beta = beta
        self.theta = theta
        self.tau_rel = tau_rel
        self.moment = moment
        self.alpha0 = alpha0
        self.n_alpha = n_alpha
        self.backtrack = backtrack
        self.alpha: dict[int, float] = {}  # per path: current step multiplier
        self.g0: dict[int, jnp.ndarray] = {}  # per path: the stopping test's anchor
        self.has_secant: dict[int, bool] = {}  # per path: first pair in hand?
        self.s_ema: dict[int, jnp.ndarray] = {}  # per path: S₁ ("abs") or S₂ ("rms")
        self.y_ema: dict[int, jnp.ndarray] = {}  # per path: Y₁ ("abs") or Y₂ ("rms")
        self.alpha_sample: dict[int, int] = {}  # per path: samples served this step
        self.saw_finite: dict[int, bool] = {}  # per path: any finite sample this step?

    def restart(self, i: int, p: PathPoint) -> None:
        """Seed path i at p: warm-up armed at alpha0, g0 kept, no estimate yet.

        The engine's only cold start, so this runs at initialization and after
        every restart - the secant history never carries across one.
        """
        # alpha0 is a scalar multiplier; see the class docstring for what it
        # does not fix (the direction -g carries the coordinate scale)
        self.g0[i] = p.g
        self.has_secant[i] = False
        self.s_ema[i] = jnp.zeros_like(p.g)
        self.y_ema[i] = jnp.zeros_like(p.g)
        self._reset_step(i)

    def needs_verdict(self, i: int) -> bool:  # noqa: ARG002 - the seam's per-path shape
        """Rule per candidate only when the alpha grid compares samples.

        True exactly when ``n_alpha > 1``: the grid serves that many samples and
        each served value decides whether to shrink and try again, so the engine
        must ask for a ruling. The phase does not enter - an ``n_alpha >= 2``
        warm-up is ruled on its grid like any steady step. The default
        single-sample step is not ruled: it is committed as proposed, and the
        engine's own gate on the winner's value and gradient is the whole of its
        acceptance (a non-finite one stalls the path instead of folding a nan
        into every later step).
        """
        return self.n_alpha > 1

    def propose(self, i: int, p: PathPoint) -> jnp.ndarray:
        """Warm-up: x + alpha·(-g) at the current alpha. Afterwards: x - d⊙g."""
        if not self.has_secant[i]:
            return p.x + self.alpha[i] * (-p.g)
        # λ anchors to the current gradient's own magnitude, so a coordinate
        # with no usable secant (y_i = 0) still steps instead of 0/0. It is a
        # floor (max), not an addend: where Y ≥ λ the estimate is the faithful
        # S/Y ratio, and λ only caps d below that.
        floor = self.theta * (jnp.abs(p.g) if self.moment == "abs" else jnp.square(p.g))
        den = jnp.maximum(self.y_ema[i], floor)
        # den_i == 0 means y_i = 0 *and* g_i = 0: no curvature information and
        # already stationary in i, so step nowhere there rather than NaN. One
        # guard is enough - `where` is a total select and d is never
        # differentiated, so an inf/nan in the unselected lane cannot leak
        # anywhere; a derivative of this line would see it, so revisit then.
        d = jnp.where(den > 0.0, self.s_ema[i] / den, 0.0)
        if self.moment == "rms":
            d = jnp.sqrt(d)
        return p.x - self.alpha[i] * (d * p.g)

    def verdict(
        self,
        i: int,
        p: PathPoint,
        x: jnp.ndarray,  # noqa: ARG002 - stays in the seam's trial shape
        f: float,
    ) -> Verdict:
        """Give up at a stationary point; otherwise rule one alpha-grid sample.

        Only consulted for a path that answered True to ``needs_verdict``, and
        the phase does not enter: ``n_alpha > 1`` rules a warm-up grid and a
        steady grid alike, while a single-sample step is committed without a
        ruling. Only the served trial's value `f` is read - `x` stays in the
        seam for methods that judge geometry, and this one does not. The engine
        buys a trial's gradient only for the step's winner, so nothing here may
        depend on one, and the winner's gradient is gated by the engine before
        it folds. The stopping test reads ``p`` - the step's start, whose
        gradient a path always has.
        """
        # Checked on the *current* point, so a path that has just landed on a
        # stationary one gives up on the next iteration (costing that one trial's
        # eval, which the engine's best still counts) rather than stepping
        # off it. g0 is the anchor: the same test at the start is ‖g0‖ ≤ τ‖g0‖,
        # true only for a zero-gradient start.
        if bool(jnp.linalg.norm(p.g) <= self.tau_rel * jnp.linalg.norm(self.g0[i])):
            return "give_up"
        return self._step_verdict(i, f)

    def fold(self, i: int, p: PathPoint, best: PathPoint) -> None:
        """Fold the engine's committed argmin for path i (the step that won).

        Only the winner is folded, never an intermediate grid sample: an
        intermediate secant is a step the path never took, and folding it would
        smear the magnitude EMAs. The engine calls this with the exact point it
        just committed on "accept", so the folded secant is the accepted step
        and the two can never disagree. The first call after init/restart is
        the warm-up pair and seeds the EMAs outright (weight 1); later calls
        blend at beta - `has_secant` doubles as the first-fold marker, so it is
        set here rather than in ``verdict``. This is also where a step is
        re-armed: the fold is the step's commit, so a ruling that the engine did
        not commit (a winner whose gradient it refused) leaves the method
        untouched rather than half-reset.
        """
        first = not self.has_secant[i]
        self._fold(i, best.x - p.x, best.g - p.g, first=first)
        self.has_secant[i] = True
        self._reset_step(i)

    def _reset_step(self, i: int) -> None:
        """Arm path i for a fresh step: full step, no alpha-grid sample served yet."""
        self.alpha[i] = self.alpha0
        self.alpha_sample[i] = 0
        self.saw_finite[i] = False

    def _step_verdict(self, i: int, f: float) -> Verdict:
        """Judge one alpha-grid sample on its value; never fold here.

        Reached by the engine only with ``n_alpha >= 2``, in either phase: a
        single-sample step answers False to ``needs_verdict`` and is committed
        unruled, with the engine's own finiteness gate standing in for the
        give-up below - the grid code is the same one sample, so a direct call
        with ``n_alpha == 1`` still rules that way. One grid
        of ``n_alpha`` samples along the current step direction, shrinking alpha
        by the grid's spacing each retry; the win is handed to the engine on
        "accept", which commits its per-step argmin and folds the winner through
        ``fold``.

        The ruling reads `f` alone: no sample's gradient is in hand, and the
        engine gates the winner's gradient before it folds, so a non-finite
        gradient stalls the path there instead of poisoning a secant here.
        Nothing non-finite may reach ``fold``: one inf folded into Y leaves
        den = inf forever, and that coordinate never steps again. A non-finite
        value drops out of the candidate set - the sweep keeps shrinking alpha,
        a real line search probing past a NaN - and a grid with no finite sample
        at all gives up, leaving the path to a restart rule.
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
