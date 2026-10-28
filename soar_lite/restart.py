"""Restart rules: the two examples that ship with the engine.

A rule owns *when* a path restarts, and nothing else. Where the restart lands is
the engine's global-model seam, so a rule here never sees, draws or proposes a
coordinate - it reads the evidence the engine forwards (`observe_step`) and
answers a verdict (`should_restart`). Neither rule fits a surrogate; the
engine's flock is the sample.

`NoRestartRule` is the default: the flock the caller handed in is the flock the
run keeps. `StallRestartRule` is the fallback that fires on a path that has
stopped buying anything - cheap, but it reads a path's own progress, so it
measures conditioning rather than the scarcity of better basins elsewhere.
"""

import math

from soar_lite.engine import RestartRule


class NoRestartRule(RestartRule):
    """Never restart: the frozen flock.

    The default, and the honest baseline: every path descends once from the
    caller's initial point and a finished path stays where it finished. With
    `restart_rule=None` this is what `n_path_schedule` binds, so a run costs one
    fused batch per iteration and never draws another point.

    It carries no state at all, which is the point of the seam split: with `x0`
    placed by the engine before any verdict is asked, "never restart" needs no
    cold-start exception, and a rule that does restart needs no seeding probe
    either.
    """

    def observe_step(self, i: int, delta: float | None) -> None:
        """No-op: this rule never weighs continuing against restarting."""

    def should_restart(self, i: int, f_k: float, best_fun_raw: float) -> bool:  # noqa: ARG002
        """Never: the paths keep the points they were given."""
        return False


class StallRestartRule(RestartRule):
    """Restart a path whose last `window` rounds bought nothing.

    Per path the rule keeps the last `window` deltas the engine reported and
    fires when none of them is a step that moved the path by at least `rel_tol`
    relatively - every round in the window was a stall (`delta is None`) or a
    step too small to count. A path that takes one real step keeps its window
    and its place; a path that keeps micro-stepping the same basin is restarted
    even though its method never gave up.

    The tolerance is relative to the value the path stands at (`|f_k|`) rather
    than absolute, because the two failure modes the rule is meant to catch are
    both scale-relative: a converged basin and a nearly flat one. A path with no
    usable point (`f_k` is `nan`) never counts as progress, so it also fires
    after `window` stalled rounds - a start that landed on non-finite values is
    given another draw instead of holding the path forever.

    Known limit: this is a single-path signal, so it cannot tell "this basin is
    exhausted" from "this basin is badly conditioned" - on an ill-conditioned
    unimodal problem it restarts paths that were still converging. Restarting
    without a model that can propose a better basin elsewhere mostly pays for the
    lost descent. It ships as the cheap fallback, not as the recommended rule.

    Args:
        window: how many consecutive no-progress rounds a path needs before it
            is restarted. Larger is more conservative.
        rel_tol: how much of ``|f_k|`` a step has to gain to count as progress.

    Raises:
        ValueError: If `window` < 1, or `rel_tol` is negative or not finite.

    Example:
        The engine binds it like any other seam:

            >>> from soar_lite import StallRestartRule, n_path_schedule
            >>> result = n_path_schedule(
            ...     eval_f_g, x0=x0, bounds=[(-5.0, 5.0)], max_iters=50,
            ...     restart_rule=StallRestartRule(window=10),
            ... )  # doctest: +SKIP

    """

    def __init__(self, *, window: int = 10, rel_tol: float = 1e-3) -> None:
        """Validate the two knobs and start with no path's history."""
        if window < 1:
            msg = f"window must be >= 1, got {window}"
            raise ValueError(msg)
        if not math.isfinite(rel_tol) or rel_tol < 0:
            msg = f"rel_tol must be finite and >= 0, got {rel_tol}"
            raise ValueError(msg)
        self.window = window
        self.rel_tol = rel_tol
        # Per path, the last `window` deltas: `None` for a round that took no
        # step. Bounded by the flock's width, and dropped when the rule fires.
        self._deltas: dict[int, list[float | None]] = {}

    def observe_step(self, i: int, delta: float | None) -> None:
        """Append this round's delta to path i's window of last rounds.

        Raises:
            ValueError: If `delta` is neither `None` nor finite. The engine
                spells "took no step" as `None`, so a non-finite float here is a
                wiring fault, not evidence.

        """
        if delta is not None and not math.isfinite(delta):
            msg = f"path {i}: delta must be None or finite, got {delta!r}"
            raise ValueError(msg)
        history = self._deltas.setdefault(i, [])
        history.append(delta)
        del history[: -self.window]

    def should_restart(self, i: int, f_k: float, best_fun_raw: float) -> bool:  # noqa: ARG002
        """Return True once path i's whole window bought nothing.

        Returning True also drops that path's window, so the next verdict judges
        the fresh descent instead of re-reading the history that fired. This is
        the one place a rule acts in a query, and the contract is why: the engine
        never reports a path's own restart (that round took no step), so the
        verdict is the only call that can carry it.
        """
        history = self._deltas.get(i)
        if history is None or len(history) < self.window:
            return False
        gain = -self.rel_tol * abs(f_k)
        if any(delta is not None and delta < gain for delta in history):
            return False
        self._deltas.pop(i, None)
        return True
