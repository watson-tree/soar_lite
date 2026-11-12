"""The default local solver: diagonal secant preconditioning (first order).

`DiagonalPreconditioner` descends along ``-precon ⊙ m``, where the per-coordinate
scale ``precon`` is the magnitude ratio of one secant pair, smoothed over the path's
history - ``abs`` averages |Δx| and |Δy|, ``rms`` averages Δx² and Δy² and takes
the root of their ratio. With
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
    point: PathPoint
    # no pair in hand yet: set by `restart`, cleared by the first committed step,
    # so `propose` descends along -g and that first `step` seeds outright
    is_warmup: bool
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
    def restarted(cls, point: PathPoint, alpha0: float) -> "_PathState":
        """Return the cold-start state at ``point``: no secant, nothing in it."""
        return cls(
            point=point,
            is_warmup=True,
            s_ema=jnp.zeros_like(point.g),
            y_ema=jnp.zeros_like(point.g),
            m=point.g,
            g0=point.g,
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

    Each coordinate carries its own effective inverse curvature precon_i, of
    dimension [x]²/[f], estimated from the magnitude ratio of one secant
    (s = x_k - x_{k-1}, y = g_k - g_{k-1}), and the step is Δx_i = -precon_i·g_i.
    Every step takes the full proposal at ``alpha0`` along its own direction,
    and ``step`` records the committed step. The two phases differ only in the
    direction: the warm-up - the first step after a restart, with no secant pair
    yet - descends along -g, and every later step along -precon⊙m.

    A path whose stopping test holds describes no step: ``propose`` answers
    ``None`` for it and the engine stalls it, so a converged path is not stepped
    off its stationary point. The test is on the projected gradient,
    ‖P g‖ ≤ τ_rel·‖g_0‖, with the components whose descent direction points out
    of the box dropped at a wall - which is how a constrained minimum sitting on
    a wall reports done rather than stationary.

    Why the ratio is a per-coordinate magnitude rather than a matrix, why the
    two moment spellings agree, what the floor is anchored to, and what ``beta``
    and ``mu`` buy: the explanation page's local-step section
    ([how it works](../explanation/how-it-works.md)).

    References:
        - Conn et al. (2000), *Trust region methods* (SIAM) — the textbook
          account of the secant equation and the SR1 update, whose diagonal
          degeneration is this ratio.

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
        multiplier different units. In the steady state ``-precon⊙m`` is already a
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
                steps), in [0, 1); 0 keeps no history, so ``precon`` reads the
                newest accepted secant.
            mu: EMA decay of the gradient history blended into the direction,
                ``(1 - mu)·g + mu·m_prev``; 0 (the default) leaves the bare
                gradient, in [0, 1).
            theta: gradient-norm floor coefficient (``λ = θ·|g_i|`` for "abs",
                ``θ·g_i²`` for "rms"), >= 0.
            alpha0: direction multiplier every path is armed at, > 0; one knob,
                two units - a fraction of the natural step in steady state, and
                ``[x]²/[f]`` in the warm-up, where the direction is ``-g``.
            tau_rel: relative gradient-norm stopping threshold, tested against
                ``‖g_0‖``, in (0, 1).

        """
        for name, value, ok, want in (
            ("beta", beta, 0.0 <= beta < 1.0, "in [0, 1)"),
            ("mu", mu, 0.0 <= mu < 1.0, "in [0, 1)"),
            ("theta", theta, theta >= 0.0, ">= 0"),
            ("alpha0", alpha0, alpha0 > 0.0, "> 0"),
            ("tau_rel", tau_rel, 0.0 < tau_rel < 1.0, "in (0, 1)"),
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

    def restart(self, i: int, point: PathPoint) -> None:
        """Seed path i at point: warm-up armed at alpha0, g0 kept, no estimate yet.

        The engine's only cold start, so this runs at initialization and after
        every restart - the secant history never carries across one. The
        momentum history is re-seeded at this start's own gradient for the same
        reason: a restart is where a path's history begins again. The point
        itself is kept, because no later call hands it back: it is the first of
        the contract's two arrivals.
        """
        # alpha0 is a scalar multiplier; the warm-up direction -g carries the
        # coordinate scale itself
        self._state[i] = _PathState.restarted(point, self.alpha0)

    def step(self, i: int, point: PathPoint) -> None:
        """Take the step the engine just committed for path i.

        ``point`` means what it means in `restart`: the point the path now stands
        at, handed over once. The point it stepped from is the record's own, so
        the pair the step observed is read entirely off (record, point).

        Only the winner is taken in, never a trial the method's own enumeration
        rejected: a secant from a point the path never stood on would smear the
        magnitude EMAs. The engine calls this with the exact point it just
        committed, so both histories see what the step observed: the pair
        (Δx, Δg) for the curvature, and the gradient at the point arrived at for
        the momentum. The first call after a restart is the warm-up pair and
        seeds the secant EMAs outright; later calls blend at beta - the record's
        ``is_warmup`` is that same first-step marker, cleared here. This is also
        where a step is re-armed: a winner the engine refuses never reaches
        here, so the method is left untouched rather than half-reset.
        """
        state = self._state[i]
        s_ema, y_ema = self._blended_secant(
            state, point.x - state.point.x, point.g - state.point.g
        )
        # Read only now that the step has committed, and read from the point it
        # committed to: the gradient this step started from was already spent on
        # this step's direction, so the arriving one is the news. `restart`
        # seeds the descent with the start's own gradient, which is why every
        # step after it blends rather than seeding; at mu = 0 the blend keeps
        # only the arriving gradient, which is the default's bare direction.
        m = self.mu * state.m + (1.0 - self.mu) * point.g
        self._state[i] = replace(
            state,
            point=point,
            is_warmup=False,
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

        The proposal is clipped into ``domain`` here because the engine refuses a
        step that lands outside the box rather than clipping it: a step is the
        method's own, so holding it inside the box is the method's own job too.

        ``serve_trials`` is the value-only evaluation hook, for a method that
        rules its own trial grid; this one takes a single full step, so it leaves
        the hook unused and lets `settle` buy the gradient. That is also why
        every proposal comes back with both halves owed: nothing was read.

        Warm-up: x + alpha·(-g) at the current alpha. Afterwards: x + alpha·(-precon⊙m).
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
            if not self._is_stationary(i, domain)
            else None
            for i in idxs
        }

    def _propose(self, i: int) -> jnp.ndarray:
        state = self._state[i]
        if state.is_warmup:
            return state.point.x + state.alpha * (-state.point.g)
        precon = self._precon(i, state.point.g)
        return state.point.x + state.alpha * (-precon * state.m)

    def _is_stationary(self, i: int, domain: Domain) -> bool:
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
        zero-gradient one - which is why the constructor keeps τ below one. Read
        off the record's own point - the step's start, the point ``propose``
        would step from - so the test and the step can never read different
        points.

        That anchor is deliberately not projected: it is the start's full
        gradient, wall reaction included. The test is relative to the start's
        overall scale, not to each component's own, so dropping the pinned part
        would re-import the per-coordinate judgment this norm exists to reject,
        make the bar depend on which coordinates happened to be stuck, and push
        it below what those coordinates can resolve. The wall reaction is real
        forcing in the caller's units; it belongs in the scale.
        """
        state = self._state[i]
        g, x = state.point.g, state.point.x
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

    def _precon(self, i: int, g: jnp.ndarray) -> jnp.ndarray:
        """Return path i's per-coordinate inverse curvature at gradient g."""
        # λ anchors to the current gradient's own magnitude, so a coordinate
        # with no usable secant (y_i = 0) still steps instead of 0/0. It is a
        # floor (max), not an addend: where Y ≥ λ the estimate is the faithful
        # S/Y ratio, and λ only caps the estimate below that.
        floor = self.theta * (jnp.abs(g) if self.moment == "abs" else jnp.square(g))
        state = self._state[i]
        den = jnp.maximum(state.y_ema, floor)
        # den_i == 0 means y_i = 0 *and* g_i = 0: no curvature information and
        # already stationary in i, so step nowhere there rather than NaN. One
        # guard is enough - `where` is a total select and precon is never
        # differentiated, so an inf/nan in the unselected lane cannot leak
        # anywhere; a derivative of this line would see it, so revisit then.
        precon = jnp.where(den > 0.0, state.s_ema / den, 0.0)
        return jnp.sqrt(precon) if self.moment == "rms" else precon

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
        if state.is_warmup:
            return s_mag, y_mag
        return (
            self.beta * state.s_ema + (1.0 - self.beta) * s_mag,
            self.beta * state.y_ema + (1.0 - self.beta) * y_mag,
        )
