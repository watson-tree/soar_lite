# Reproducing the comparison experiments

The panels in `DESIGN.md` come from one canonical script,
`experiments/compare_baselines.py` — no notebooks, no hidden code path. This
file is the run-book; `RESULTS.md` is the ledger of what past runs measured.

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

## Smoke check

```bash
uv run --group exp python -m experiments.compare_baselines --quick
```

The quick gate runs only the `smooth` panel — the diag control and the rotated
quadratic — with 2 seeds and `--max-steps 300` per method, so it writes just
that panel's figures (seconds, not minutes). Use it to confirm the environment
before committing to the full schedule.

## Canonical run

```bash
uv run --group exp python -m experiments.compare_baselines \
    --seeds 20 \
    --groups headline,smooth,multimodal,invariance \
    --max-steps 3000
```

`--max-steps` is **batches
per method**; the per-point chart re-reads the same runs on the "objective
points served" axis, so both billing notions come from one execution.

Run one panel to save time, e.g. `--groups smooth`; any subset works, and each
panel writes its files as it finishes.

The restart-model panel is run separately and on purpose: `--groups blr` draws
rastrigin alone (all arms `n_paths=16`) — `Sample` vs `BLR identity` vs `BLR
sphere` — as its own single-subplot figure. It is left out of the default list
because a fitted BLR restart walks an acquisition descent on every redraw, so
it is the most expensive panel per batch.

## The Adam learning-rate grid

Adam pays a per-problem lr chosen by an offline sweep
(`ADAM_LR` in `compare_baselines.py`); the figure below draws that sweep.
Reproduce the sweep figure with:

```bash
uv run --group exp python -m experiments.compare_baselines --lr-sweep --max-steps 3000
```

The C0 diamond marks the lr the panels actually use (`ADAM_LR`). It coincides
with the sweep's lowest point on the rotated quadratic, the oscillator and
rastrigin, but it is not a strict argmin everywhere: on the diagonal quadratic
and Ackley every grid point from `1e-2` up sits on the same floor or plateau,
so the pick is one point of that tie (`lr=1e-4` costs 52 orders of magnitude on
the diagonal quadratic against it). On the rotated quadratic and the oscillator
the curve is still falling at the right edge — `lr=2.0` — so there the diamond
is the best *sampled* point, not an interior minimum; on rastrigin the floor is
flat over `1e-3..2e-1` and the pick sits ~5% below it. The compartment chain
(`pk_chain_n21_d61`) is the one problem that goes the other way: its curve rises
steeply on both sides of the diamond, so there the pick is a strict interior
optimum rather than an edge or a tie.

## Outputs

Two trees, both gitignored — regenerated, not source. `experiments/results/`
holds traces, `experiments/figures/` holds what was drawn from them, so a
figure is always re-derivable and a trace never is. The unit is one
**(problem, arm)** pair, so a rerun touches one file:

- `results/<problem>/<arm-key>.json` — that arm's raw per-seed trace on that
  problem: points, batches, best-so-far per step, final value, and a `meta`
  block naming the problem, the arm key, the seed count and the budget. Raw, not
  resampled: the grid is a plotting decision that depends on which other arms
  are present.
- `figures/<group>/batches.png` — budget axis: batches served.
- `figures/<group>/points.png` — budget axis: objective points served (the
  serial contest, pinned in ADR-0004). Every arm is cut at the shortest extent
  present **on that axis**, so all curves end together: an `n_paths=16` arm
  serves 16 points per batch, so here it shows only the first ~187 of its 3000
  batches. That is what the axis is for, not a clipped result.
- `figures/invariance/invariance.png` — panel 5, one physics in two unit systems,
  zero re-tune.
- `figures/lr_sweep/lr_sweep.png` — the Adam grid sweep.
- Every curve plots the **median** best-so-far with the **[Q1, Q3] interquartile
  band** (not mean ± std — a skewed spread would read as symmetric and dip
  below the log axis).

The arm key is the file name and spells out the arm's whole configuration
(`soar-sample-np1-mu0.63-restart-on`, `adam-lr0.01`, `lbfgs`), so a changed
default or a changed lr is a new key and a new file: the old trace stays as a
record of the old configuration instead of being silently overwritten.
Re-measuring one arm is a filter on that same identity:

```bash
# rewrite exactly one file, leave the rest of the panel alone
uv run --group exp python -m experiments.compare_baselines \
    --groups smooth --problems diag_quad_d100_cond10000 \
    --arms soar-sample-np1-mu0-restart-on --seeds 20 --max-steps 3000

# rebuild every figure from the files, no re-run
uv run --group exp python -m experiments.compare_baselines --plot-only
```

`--plot-only` names any arm that has no trace file instead of drawing a gap, so
a half-filled panel is visible rather than plausible.

## Reproducibility contract

- **Fixed seeds**: `range(n_seeds)` from 0; every runner takes the seed, and JAX
  uses it for the start point and every stochastic step. Same lock + same seed
  ⇒ same number.
- **No *tuned* knobs on soar-lite** — which is not the same as no knobs.
  `DiagonalPreconditioner` exposes eight (`moment`, `beta`, `theta`, `mu`,
  `alpha0`, `n_alpha`, `backtrack`, `tau_rel`), and two of them are live
  smoothing windows: `beta` sets the preconditioner's memory and `mu` the
  direction's, so neither is inert. What the formal runs do is take them at
  their factory defaults end to end, so soar-lite pays no per-problem search;
  Adam pays the per-problem lr grid each time. That asymmetry is about
  *tuning*, not about having parameters, and it is the point.
- **No seam on soar-lite either**: the comparison runs the factory
  `value_and_grad` path — no `eval_f` (value-only) seam — so every batch the
  axis counts is a full value+gradient batch. The only attributes stepped are
  `n_paths` (1 → 16) and `mu` (0 → 0.9 → the class default); `restart_stalled`
  stays at its engine default (on) in every arm, and the blr panel injects its
  restart model through the public `model=` factory (`BLRGlobalModel`; the other
  arms use the default `SampleGlobalModel`).
- **Canvas only in-repo**: `experiments/DESIGN.md` is the design and the
  story; figures are drawn from the trace files, never hand-tuned.

## Open-source-time follow-ups (blocked on a public repo URL)

- A Colab wrapper that installs this repo and runs the canonical command —
  runtime only, never a second codebase. Blocked: needs the published URL to
  install from.
- The README "Open in Colab" badge. Blocked on the same URL.