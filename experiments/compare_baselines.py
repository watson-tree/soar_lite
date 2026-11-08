"""Measure the arms on each panel, write per-arm traces, draw figures.

The harness is one module per thing it owns: `record` (the file per problem and
arm, and the budget it was measured at), `arms` (what to run and how to draw
it), `panels` (rosters over arms, and the run that fills one), `figures`
(drawing from the record), `appendix` (the two panels that measure as they
draw), `cli` (flags, refusals, and the run). This module is the entry point and
re-exports the names the tests and the notebook read.

Panels: `DESIGN.md`. Commands and storage layout: `README.md`. What the last
full run measured: `RESULTS.md`.
"""

from experiments.arms import ADAM_LR
from experiments.cli import main
from experiments.panels import selection, soar_ladder
from experiments.record import Roots, refuse_budget_change

__all__ = [
    "ADAM_LR",
    "Roots",
    "main",
    "refuse_budget_change",
    "selection",
    "soar_ladder",
]

if __name__ == "__main__":
    main()
