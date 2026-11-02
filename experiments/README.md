# Reproducing the comparison experiments

The panels in `DESIGN.md` come from one canonical script,
`experiments/compare_baselines.py` — no notebooks, no hidden code path. This
file is the run-book.

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
flat over `1e-3..2e-1` and the pick sits ~5% below it.

## Outputs

All in `experiments/` (gitignored — regenerated, not source):

- `results_{group}_batches.png` — budget axis: batches served.
- `results_{group}_points.png` — budget axis: objective points served (the
  serial contest, pinned in ADR-0004).
- `results_invariance.png` — panel 5, one physics in two unit systems, zero re-tune.
- `results_lr_sweep.png` — the Adam grid sweep.
- `results_{group}.json` — the per-seed best-so-far curves (one grid plus one
  curve per seed) behind each figure; redraw any figure from it with
  `--plot-only`, no re-run.
- Every curve plots the **median** best-so-far with the **[Q1, Q3] interquartile
  band** (not mean ± std — a skewed spread would read as symmetric and dip
  below the log axis).

## Reproducibility contract

- **Fixed seeds**: `range(n_seeds)` from 0; every runner takes the seed, and JAX
  uses it for the start point and every stochastic step. Same lock + same seed
  ⇒ same number.
- **Zero knobs on soar-lite**: its two start lengths / grid sizes ship inert
  (`n_alpha=1`, `alpha0=1`); the formal runs use factory defaults end to end.
  Adam pays the per-problem lr grid each time — that asymmetry is the point.
- **No seam on soar-lite either**: the comparison runs the factory
  `value_and_grad` path — no `eval_f` (value-only) seam — so every batch the
  axis counts is a full value+gradient batch. The only attributes stepped are
  `n_paths` (1 → 16) and `mu` (0 → 0.9); `restart_stalled` stays at its engine
  default (on) in every arm, and the blr panel injects its restart model
  through the public `model=` factory (`BLRGlobalModel`; the other arms use the
  default `SampleGlobalModel`).
- **Canvas only in-repo**: `experiments/DESIGN.md` is the design and the
  story; figures are drawn from the JSON, never hand-tuned.

## Open-source-time follow-ups (blocked on a public repo URL)

- A Colab wrapper that installs this repo and runs the canonical command —
  runtime only, never a second codebase. Blocked: needs the published URL to
  install from.
- The README "Open in Colab" badge. Blocked on the same URL.