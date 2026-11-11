"""Panels: one table of names, and the run that fills one.

A panel is a *roster over the record*, so re-measuring one arm rewrites exactly
one file and every panel listing that arm picks the new numbers up. Everything a
name on the command line means is in `PANELS`: the problems it measures, the
arms it measures them with, and - for the panel that compares as it draws and so
stores no per-arm trace - the run that does it. `selection` flattens a name into
the pairs it measures; `run_panel` measures exactly those pairs and writes one
trace file each.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np

from experiments.harness import problems
from experiments.harness.arms import DEFAULT_MU, LBFGS_ARM, Arm, adam_arm, soar_arm
from experiments.harness.record import HERE, Roots, save_arm

LABEL_SERIAL = f"soar-lite n_paths=1, mu={DEFAULT_MU:g} (default)"
LABEL_PARALLEL = f"soar-lite n_paths=16, mu={DEFAULT_MU:g}"
LABEL_MOMENTUM_09 = "soar-lite n_paths=1, mu=0.9 (Adam's window)"

# Style grammar: linestyle encodes `n_paths` (one solid line per width: 1 ->
# solid, 16 -> dashed) and color encodes the momentum level. Without a rule
# like this two single-path arms read as different kinds of object because one
# of them happens to switch momentum off.
STYLE_SERIAL: dict[str, Any] = {"color": "C0", "lw": 2.0}
STYLE_PARALLEL: dict[str, Any] = {"color": "C0", "lw": 2.0, "ls": "--"}
STYLE_MOMENTUM_09: dict[str, Any] = {"color": "C1", "lw": 1.5}

Build: TypeAlias = Callable[[], problems.Problem]
Roster: TypeAlias = Callable[[problems.Problem], list[Arm]]
Measure: TypeAlias = Callable[[Roots, int, int], None]


@dataclass(frozen=True)
class Panel:
    """One name on the command line: its problems, and how they are measured.

    `arms` and `run` are the two ways a panel measures, and a name carries one
    of them. A roster panel has `arms` and stores one trace per pair; a panel
    that compares two problems as it draws builds its own, so `problems` stays
    empty and `run` is what measures it. A panel with no `arms` selects
    nothing.
    """

    problems: tuple[Build, ...] = ()
    arms: Roster | None = None
    run: Measure | None = None


def soar_ladder() -> list[Arm]:
    """Build the soar-lite roster both proposition panels read: one knob at a time.

    A partial grid, not a chain: the momentum axis runs at `n_paths=1` (`mu` 0
    and 0.9) and one width step (`n_paths` 1 -> 16) sits at the default `mu`.
    The default is `mu=0`, so "momentum off" *is* the default arm - one curve,
    not two names reading one file. It is also what shows the diagonal sweet
    spot landing in a single step, which a `mu > 0` gives up near a minimizer -
    a cost the panel should show, not state.

    The `mu=0.9` arm reads the momentum *horizon*, and it is a comparison arm,
    not a tuned one: its window is ~1/(1-0.9) = 10 steps, the same window Adam's
    first moment runs at `b1=0.9`, so the panel can ask whether the secant beats
    Adam's RMS at Adam's momentum reach as well as at ours. Adam itself stays at
    its own defaults - a baseline whose knobs are moved to mirror the method
    under test stops being the baseline. This arm is the current, unprojected
    momentum; the `mu=0.9` column retired from the v2 tables was the *projected*
    direction, a different rule whose numbers are not these.
    """
    return [
        soar_arm(n_paths=1, mu=DEFAULT_MU, label=LABEL_SERIAL, style=STYLE_SERIAL),
        soar_arm(
            n_paths=1,
            mu=0.9,
            label=LABEL_MOMENTUM_09,
            style=STYLE_MOMENTUM_09,
        ),
        soar_arm(
            n_paths=16,
            mu=DEFAULT_MU,
            label=LABEL_PARALLEL,
            style=STYLE_PARALLEL,
        ),
    ]


def _preconditioning_arms(prob: problems.Problem) -> list[Arm]:
    """Build the smooth/headline roster: the soar-lite ladder, Adam, L-BFGS-B."""
    return [*soar_ladder(), adam_arm(prob), LBFGS_ARM]


def _drift_arms(prob: problems.Problem) -> list[Arm]:
    """Build the multimodal roster: the same ladder, to read the descent rate."""
    return [*soar_ladder(), adam_arm(prob)]


def _appendix_invariance(roots: Roots, n_seeds: int, max_batches: int) -> None:
    """Measure the unit-rescaling panel.

    Resolved at call time because `appendix` imports this module for the
    ladder's labels, so the two modules can only meet in one direction at
    import time.
    """
    from experiments.harness import appendix  # noqa: PLC0415 - the cycle, above

    appendix.run_invariance(roots, n_seeds, max_batches)


PANELS: dict[str, Panel] = {
    "headline": Panel(
        problems=(problems.oscillator_chain, problems.pk_chain),
        arms=_preconditioning_arms,
    ),
    "smooth": Panel(
        problems=(problems.diag_quadratic, problems.rotated_quadratic),
        arms=_preconditioning_arms,
    ),
    "multimodal": Panel(problems=(problems.rastrigin,), arms=_drift_arms),
    "invariance": Panel(run=_appendix_invariance),
}
GROUPS: tuple[str, ...] = tuple(PANELS)


def every_problem() -> list[problems.Problem]:
    """Every problem any panel measures, one per name, in panel order.

    What the lr sweep runs over. No panel draws the whole set, and the sweep
    needs it without a panel that would exist only to hold it.
    """
    built: dict[str, problems.Problem] = {}
    for panel in PANELS.values():
        for build in panel.problems:
            prob = build()
            built.setdefault(prob.name, prob)
    return list(built.values())


def run_panel(  # noqa: PLR0913 - roots, panel, budget and two filters
    roots: Roots,
    panel: str,
    n_seeds: int,
    max_batches: int,
    *,
    problems_filter: frozenset[str] | None = None,
    arms_filter: frozenset[str] | None = None,
) -> None:
    """Run the selected arms of ``panel``, one trace file each.

    The filters are how one arm is re-measured without its panel: they select
    on the arm key (and on the problem), so the run touches only the files it
    names and every other trace stays as it was. ``main`` refuses a selection
    that comes back empty, and refuses a budget that would replace a trace
    measured at another one, both before this is reached.
    """
    for prob, arm in selection(panel, problems_filter, arms_filter):
        traces = [
            arm.runner(prob, max_batches=max_batches, seed=s) for s in range(n_seeds)
        ]
        path = save_arm(
            roots,
            prob,
            arm,
            traces,
            n_seeds=n_seeds,
            max_batches=max_batches,
        )
        fins = [t.f for t in traces]
        print(
            f"  {prob.name:26s} {arm.key:44s} "
            f"final mean best = {np.mean(fins):.4e}  -> {path.relative_to(HERE)}",
        )


def selection(
    panel: str,
    problems_filter: frozenset[str] | None = None,
    arms_filter: frozenset[str] | None = None,
) -> list[tuple[problems.Problem, Arm]]:
    """Select the ``(problem, arm)`` pairs of ``panel`` that both filters leave.

    A filter matches by name and nothing more: a name the panel does not carry
    selects nothing here rather than raising, so the caller compares the result
    with the filters it passed and can refuse the run when it comes back empty.
    Neither does a panel with no roster select anything - it stores no pairs.
    """
    arms = PANELS[panel].arms
    if arms is None:
        return []
    probs = [build() for build in PANELS[panel].problems]
    return [
        (prob, arm)
        for prob in probs
        if not problems_filter or prob.name in problems_filter
        for arm in arms(prob)
        if not arms_filter or arm.key in arms_filter
    ]
