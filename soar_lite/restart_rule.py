"""Restart rules: the two examples that ship with the engine.

A rule owns *when* a path restarts, and nothing else. Where the restart lands is
the engine's global-model seam, so a rule here never sees, draws or proposes a
coordinate - it reads the evidence the engine forwards (`observe_step`) and
answers a verdict (`should_restart`). Neither rule fits a surrogate; the
engine's flock is the sample.

`NoRestartRule` is the default: the flock the caller handed in is the flock the
run keeps. `StallRestartRule` is the simplest rule that ever fires: it restarts
a path the moment the engine reports that the path stopped (a stall round,
`delta is None`). It carries no knob, because the engine's stall is sticky - a
stalled path never moves again until it is restarted - so a longer window would
only delay the re-draw without buying new evidence. It is still a single-path
signal: what a restart can buy is exactly what the global model's `propose` can
offer as an alternative basin.
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
    """Restart a path the moment the engine reports it stopped.

    A path stops when the engine takes no step for it: its method gave up
    (converged, or found no finite sample), or the point it was handed was not
    usable. The engine spells that round `delta is None` on `observe_step`, and
    that `None` is the rule's whole evidence - a finite delta means the path
    moved, whatever the step was worth, so it is not stalled.

    The rule is a one-bit latch per path, and that is all it needs to be: the
    engine's stall is sticky (a stalled path proposes nothing, so every later
    round reports `None` until a restart places a usable point), and a restart
    round is silent (the engine does not report it, so it cannot arrive here as
    evidence). One `None` is therefore the whole signal - a window over more
    rounds would read the same `None` again and only delay the re-draw.

    A consequence worth naming: the rule never judges whether a step was an
    improvement, only whether the path moved. An uphill or micro step is still a
    step; whether it was worth taking is the local solver's verdict, not this
    rule's.

    What a restart is worth is the global model's question, not this rule's:
    with `UniformGlobalModel` every restart is a fresh uniform draw, so the rule
    is a random multistart dressed as a restart policy. It pays off only when
    the model can propose a better basin elsewhere.

    Example:
        The engine binds it like any other seam:

            >>> from soar_lite import StallRestartRule, n_path_schedule
            >>> result = n_path_schedule(
            ...     eval_f_g, n_paths=4, bounds=[(-5.0, 5.0)], max_iters=50,
            ...     restart_rule=StallRestartRule(),
            ... )  # doctest: +SKIP

    """

    def __init__(self) -> None:
        """Start with every path unknown, which reads as "did not stall"."""
        self._stalled: set[int] = set()

    def observe_step(self, i: int, delta: float | None) -> None:
        """Latch whether path i moved this round (`None` means it stalled).

        Raises:
            ValueError: If `delta` is neither `None` nor finite. The engine
                spells "took no step" as `None`, so a non-finite float here is a
                wiring fault, not evidence.

        """
        if delta is not None and not math.isfinite(delta):
            msg = f"path {i}: delta must be None or finite, got {delta!r}"
            raise ValueError(msg)
        if delta is None:
            self._stalled.add(i)
        else:
            self._stalled.discard(i)

    def should_restart(self, i: int, f_k: float, best_fun_raw: float) -> bool:  # noqa: ARG002
        """Return True iff path i's last reported round was a stall.

        A pure query: firing does not clear the latch. The engine's next report
        clears it - a step discards the mark, another stall keeps it, which is
        what re-draws a path whose fresh start was unusable.
        """
        return i in self._stalled
