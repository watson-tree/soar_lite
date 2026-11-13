# Reproducing the comparison experiments

The panels in `DESIGN.md` come from one canonical entry point,
`experiments/compare_baselines.py` — no notebooks, no hidden code path. It is a
thin shell over the harness in `harness/`: `record` (the per-arm files), `arms`
(what to run), `panels` (rosters over arms, and the run that fills one),
`figures` (drawing from the record), `appendix` (the panels that measure as they
draw) and `cli` (flags and the run). Three more sit under those: `methods` (the
drivers that spend the budget), `problems` (the benchmark set) and `tuning` (the
recorded Adam picks). Nothing outside `harness/` measures anything.

That is the whole directory: code in `harness/`, its self-checks in `tests/`,
and beside them the four documents plus the two output trees `results/` and
`figures/`. This file is the run-book; `RESULTS.md` is the ledger of what past
runs measured, and `DEPRECATED.md` is the record of the experiments retired
from it.

## Where this sits

A benchmark sub-library, not a dependency. It imports `soar_lite` and cites
`docs/site/explanation/how-it-works.md`; nothing in `soar_lite/`, `docs/` or the root
`README.md` links back in, and neither the wheel nor the sdist ships this
directory. Its one entry point is `compare_baselines.py`; its record stays in
the files next to this one.

One consequence is worth knowing before changing anything here: the repo's
pytest gate runs these tests too, since `testpaths` covers both trees. A harness
change breaks the same command a library change does, and the tests travel with
the directory if it is ever extracted.

## Prerequisites

One pinned environment, reproduced by `uv.lock`:

```bash
uv sync --all-groups
```

`uv.lock` pins every transitive dependency (jax 0.10.2, optax 0.2.8,
matplotlib 3.11.2, scipy 1.17.1, numpy 2.4.6 on Python 3.11). If you prefer
pip, the same pins are:

```bash
pip install 'jax==0.10.2' 'optax==0.2.8' 'matplotlib==3.11.2' 'scipy==1.17.1'
pip install -e .
```

The default sync pins the CPU JAX build, and a CPU is a legitimate place to
measure this: neither axis counts seconds — a batch is a batch on either backend
— so the ordering in `RESULTS.md` says nothing about GPU hardware. The GPU-shaped
thing is the **library**: on a GPU box a batch's wall clock barely grows as it
widens, which is the whole reason the schedule is wide. To run it there, install
the CUDA build matching your driver line alongside the CPU wheel —
`uv pip install 'jax[cuda12]'` (or `'jax[cuda13]'`) — and note that a later
`uv sync` restores the CPU wheel from the lock. JAX picks its backend from what is
installed: to pin one inside the process, call
`jax.config.update("jax_platform_name", "cpu")` — or `"cuda"` — before any JAX
work.

## Smoke check

```bash
uv run --group exp python -m experiments.compare_baselines --quick
```

The quick gate runs only the `smooth` panel — the diag control and the rotated
quadratic — with 2 seeds and `--max-batches 300` per method, so it writes just
that panel's figures (seconds, not minutes). It writes them under
`results/smoke/` and `figures/smoke/`, not into the record: a smoke check
produces exactly the file names a full run does, and those files are the
measurement. Use it to confirm the environment
before committing to the full schedule.

## Canonical run

```bash
uv run --group exp python -m experiments.compare_baselines \
    --seeds 20 \
    --groups headline,smooth,multimodal,invariance \
    --max-batches 3000
```

`--max-batches` is **batches
per method**; the per-point chart re-reads the same runs on the "objective
points served" axis, so both billing notions come from one execution.

Run one panel to save time, e.g. `--groups smooth`; any subset works, and each
panel writes its files as it finishes.

## The Adam learning-rate grid

Adam pays a per-problem lr chosen by an offline sweep. The pick is recorded
per (method, problem) in `experiments/tuning/adam.json`, written by
`--lr-sweep` and read by `experiments/harness/tuning.py`. It is that sweep's
argmin, ties going to the smallest rate - a number derived from the curve, not
typed next to it. The figure below draws that sweep.
Reproduce the sweep figure with:

```bash
uv run --group exp python -m experiments.compare_baselines --lr-sweep --seeds 2 --max-batches 3000
```

The sweep is eight learning rates x five problems, so it keeps its own 2 seeds.
`--seeds` reaches it like every other mode, but a different count is refused
rather than measured: that file is a committed input, so a cheap sweep would
replace the picks the panels use. Changing the budget on purpose means deleting
it (see below).

The C0 diamond marks the lr the panels actually use, which is the argmin of
every curve here (`lr=1e-4` costs more than fifty orders of magnitude on the
diagonal quadratic against its pick). What that argmin is worth varies. On the
diagonal quadratic every rate from `1e-2` through `1.0` has already reached the
quadratic's rounding floor (`1e-49..1e-52`), so the winner there is decided by
rounding, not by a rate difference. On the rotated quadratic and the oscillator
the curve is still falling at the right edge — `lr=2.0` — so a wider grid could
move those picks. On rastrigin the floor is flat over `1e-3..2e-1` and the
pick sits ~5% below it: real, but a short gap. The compartment chain
(`pk_chain_n21_d61`) is the one problem that goes the other way: its curve rises
steeply on both sides of the diamond, so there the pick is a strict interior
optimum rather than an edge or a tie.

## Outputs

Two trees, both gitignored — regenerated, not source. `experiments/results/`
holds traces, `experiments/figures/` holds what was drawn from them, so a
figure is always re-derivable and a trace never is. The unit is one
**(problem, arm)** pair, so a rerun touches one file:

- `results/<problem>/<arm-key>.json` — that arm's raw per-seed trace on that
  problem: points, batches, best-so-far per batch, final value, and a `meta`
  block naming the problem, the arm key, the seed count and the budget. Raw, not
  resampled: the grid is a plotting decision that depends on which other arms
  are present.
- `figures/<group>/batches.png` — budget axis: batches served.
- `figures/<group>/points.png` — budget axis: objective points served (the
  serial contest). Every arm is cut at the shortest extent present **on that
  axis**, taken over the arms that served their batch budget: an `n_paths=16`
  arm serves 16 points per batch, so it shows only the first ~187 of its 3000
  batches, while an arm that stopped early is drawn to its own last point
  instead of clipping the panel. With every arm completing, all curves end
  together. That is what the axis is for, not a clipped result.
- `figures/invariance/invariance.png` — panel 4, one physics in two unit systems,
  zero re-tune.
- `figures/lr_sweep/lr_sweep.png` — the Adam grid sweep. Its y axis is the
  sweep's own statistic, the **mean** over its two seeds: the same number the
  pick takes its argmin of, so this figure keeps mean rather than the panels'
  median.
- `results/smoke/`, `figures/smoke/` — where `--quick` writes, one smoke run
  per panel. Same names as the record on purpose (it exercises the same code)
  in roots of its own, so checking the environment cannot overwrite it.
- `results/<name>/`, `figures/<name>/` — where `--out <name>` writes, for any
  run that must not touch the record.
- Every curve plots the **median** best-so-far with the **[Q1, Q3] interquartile
  band** (not mean ± std — a skewed spread would read as symmetric and dip
  below the log axis), and only while every seed of that cohort is still
  running, so a curve ends at its first seed's stop.

The arm key is the file name and spells out the arm's whole configuration
(`soar_lite-sample-n_paths1-mu0-restart-off`, `adam-lr0.1`, `lbfgs`), so a changed
default or a changed lr is a new key and a new file: the old trace stays as a
record of the old configuration instead of being silently overwritten.

Every knob is in the key, and the budget is not one: `--seeds` and
`--max-batches` say how the arm was *measured*, not which arm it is. That left one
way to replace a record, and it was the easiest to type by accident — so the
harness refuses to write over a file whose `meta` names a different budget, and
it refuses before the first batch. The message names the file, both budgets and
the two legitimate exits:

```bash
# refused: holds 20 seeds x 3000 batches, this run asks for 2 x 300
uv run --group exp python -m experiments.compare_baselines \
    --groups smooth --seeds 2 --max-batches 300

# same numbers as the record: this is the documented rerun
uv run --group exp python -m experiments.compare_baselines \
    --groups smooth --seeds 20 --max-batches 3000

# changing the budget on purpose starts by deleting the file
rm experiments/results/<problem>/<arm-key>.json
```

Deleting is the deliberate act; a flag to approve the overwrite would be a
second way to say yes and would get typed without reading. `--quick` needs no
exemption because it writes to its own roots: the smoke check cannot overwrite
the record it is checking. `--out <name>` is the general form — it puts both
roots under `results/<name>` and `figures/<name>`, so a demo or a check at
another budget writes without touching the record.

Re-measuring one arm is a filter on that same identity:

```bash
# rewrite exactly one file, leave the rest of the panel alone
uv run --group exp python -m experiments.compare_baselines \
    --groups smooth --problems diag_quad_d100_cond10000 \
    --arms soar_lite-sample-n_paths1-mu0-restart-off --seeds 20 --max-batches 3000

# rebuild every figure from the files, no re-run
uv run --group exp python -m experiments.compare_baselines --plot-only
```

`--plot-only` names any arm that has no trace file instead of drawing a gap, so
a half-filled panel is visible rather than plausible.

A filter is a name match, so it is checked before the first batch: a misspelled
problem, an arm key that belongs to another panel, or two names that are each
real but never meet in one arm stops the run and prints the names it does have.
Without that check the command would exit clean having measured nothing, and the
figure it redrew would be the old one — a re-measurement that silently did not
happen. `--plot-only` and `--lr-sweep` measure no arm at all, so they refuse
`--problems` and `--arms` instead of ignoring them.

## Reproducibility contract

- **Fixed seeds**: `range(n_seeds)` from 0; every runner takes the seed, and JAX
  uses it for the start point and every stochastic step. Same lock + same seed
  ⇒ same number.
- **No *tuned* knobs on soar-lite** — which is not the same as no knobs.
  `DiagonalPreconditioner` exposes six (`moment`, `beta`, `mu`, `theta`,
  `alpha0`, `tau_rel`), and two are live smoothing windows: `beta` sets the
  preconditioner's memory (see `docs/site/explanation/how-it-works.md` §Smoothing),
  `mu` the direction's (0 by default, so a run takes direction memory only if it
  asks — §Momentum on the same page). What the formal runs do is take them at
  their factory defaults end to end, so soar-lite pays no per-problem search;
  Adam pays the per-problem lr grid each time. That asymmetry is about
  *tuning*, not about having parameters, and it is the point.
- **No seam on soar-lite either**: the comparison runs the factory
  `value_and_grad` path — no `eval_f` (value-only) seam — so every batch the
  axis counts is a full value+gradient batch. The only attributes stepped are
  `n_paths` (1 → 16) and `mu` (0 → 0.9); `restart_stalled`
  is **off** in every arm, so a converged chain stops like the baselines do
  instead of re-drawing. The restart model stays reachable through the public
  `model=` factory (`BLRGlobalModel`, the default `SampleGlobalModel` when
  unset), but no panel arms it as a step.
- **Canvas only in-repo**: `experiments/DESIGN.md` is the design and the
  story; figures are drawn from the trace files, never hand-tuned.

## The notebook

`notebooks/run_panels.ipynb` is what the README's "Open in Colab" badge opens,
and it is the cockpit rather than a second implementation: one section per
panel, each calling `main` from this harness and then showing the figure that
run wrote. Nothing is measured outside the command line above.

It carries a `MODE` switch. `"record"` runs the canonical budget into the
record's own roots — the committed images came from that run. `"demo"` runs
4 seeds x 600 batches under `--out demo`, so a cheap pass cannot overwrite the
record. The lr sweep is the one section the switch does not reach: its budget
belongs to the record it writes, `tuning/adam.json`, and a demo-sized sweep of
it would be refused rather than measured.