# Figures: a snapshot of one run

The panels of `headline`, `smooth` and `multimodal`, both axes of each, plus
`invariance` and the Adam lr sweep — copied whole out of `experiments/figures/`
after the 2026-11-11 run.

`experiments/figures/` is gitignored: a run writes there and the next run
overwrites it. These copies are the only pictures that outlive the traces, and
they are a snapshot of **one** run. The lineup, the budget, the medians behind
every curve and the caveats that qualify them are in `experiments/RESULTS.md`
under that date.

The set moves at once, in the commit that records the run, and only after
every panel has been looked at. That is what makes the check meaningful: all
eight files are compared against one record entry. Copied one at a time, each
file is compared against nothing in particular, and the page slowly stops
describing the run it names.

| file | panel | axis | budget |
| --- | --- | --- | --- |
| `headline-batches.png` | `headline` — oscillator chain, compartment chain | batches | 20 seeds x 3000 |
| `headline-points.png` | `headline` | points | 20 seeds x 3000 |
| `smooth-batches.png` | `smooth` — diagonal quadratic, rotated quadratic | batches | 20 seeds x 3000 |
| `smooth-points.png` | `smooth` | points | 20 seeds x 3000 |
| `multimodal-batches.png` | `multimodal` — Rastrigin | batches | 20 seeds x 3000 |
| `multimodal-points.png` | `multimodal` | points | 20 seeds x 3000 |
| `invariance-units.png` | `invariance` — one physics, two unit systems | — | 20 seeds x 3000 |
| `lr-sweep.png` | Adam learning-rate grid | — | 2 seeds x 3000 per point |

The sweep is the one panel whose y axis is a **mean**: it is the statistic the
pick takes its argmin of, so its figure keeps mean rather than the panels'
median. See the sweep's entry in `experiments/RESULTS.md`.

## The two axes ask different questions

**Batches** counts parallel passes. One batch is one time step of the whole
flock, which is what a GPU bills. An `n_paths=16` arm costs the same per batch
as `n_paths=1` and gets sixteen independent starts for that price.

**Points** counts objective evaluations. Every path of the wide arm pays for its
own point, so at equal points it has had sixteen times fewer iterations. This is
the axis that charges width, and on the two real problems it reverses the wide
arm's lead: ahead of both single chains at the last batch the three cohorts
share, behind both at the last point count they share. That reversal is the
reason the axis is here, and it is not a property the method owns either way —
on the rotated quadratic the wide arm is not the best arm even on batches. What
a batch or a point is charged for decides the order, not the method.
`experiments/RESULTS.md` has the numbers.

Two shapes on the points figures are alignment, not missing data:

- **The wide arm's curve begins at 16, not at 1.** Its first batch *is* sixteen
  points, so a log x axis starting at 1 leaves an empty stretch to the left of
  it: the sixteen single-chain points it never spent.
- **A curve ends where the first seed of its cohort stopped** — a chain that
  converged or stalled — while its neighbours keep running to the budget. The
  median is taken over the whole cohort, so it cannot be drawn past that point.

## Redrawing

The canonical run writes every panel above:

```bash
uv run --group exp python -m experiments.compare_baselines \
    --seeds 20 --groups headline,smooth,multimodal,invariance --max-batches 3000
```

then copy the set across from `experiments/figures/`:

```bash
cd experiments/figures
for p in headline smooth multimodal; do
    cp $p/batches.png ../../docs/figures/$p-batches.png
    cp $p/points.png   ../../docs/figures/$p-points.png
done
cp invariance/invariance.png ../../docs/figures/invariance-units.png
cp lr_sweep/lr_sweep.png     ../../docs/figures/lr-sweep.png
```

## Two rules for this page

- **Numbers stay in `RESULTS.md`.** A caption on the root README names the panel
  and the budget; it never repeats a median, because a median without the line
  that measured it reads as a property of the library.
- **A figure is evidence of one run, not something a reader regenerates in a
  minute.** The traces behind them are 83 MB and gitignored. Hosting them as a
  release asset is still undecided; until then the commands above are the only
  route back to the pixels.
