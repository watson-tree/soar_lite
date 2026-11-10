"""Measure the arms on each panel, write per-arm traces, draw figures.

The harness lives in `harness/`, one module per thing it owns: `record` (the
file per problem and arm, and the budget it was measured at), `arms` (what to
run and how to draw it), `panels` (rosters over arms, and the run that fills
one), `figures` (drawing from the record), `appendix` (the two panels that
measure as they draw), `cli` (flags, refusals, and the run), and under those
`methods` (the drivers that spend the budget), `problems` (the benchmark set)
and `tuning` (the recorded picks). This module stays
at the top of the directory because it is the command - `python -m
experiments.compare_baselines` - and because it re-exports the names the tests
and the notebook read.

Panels: `DESIGN.md`. Commands and storage layout: `README.md`. What the last
full run measured: `RESULTS.md`.
"""

from experiments.harness.cli import main
from experiments.harness.panels import selection, soar_ladder
from experiments.harness.record import Roots, refuse_budget_change

__all__ = [
    "Roots",
    "main",
    "refuse_budget_change",
    "selection",
    "soar_ladder",
]

if __name__ == "__main__":
    main()
