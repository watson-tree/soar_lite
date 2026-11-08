# Figures shown in the root README

Curated copies, not build output. `experiments/figures/` is where a run writes
its panels and is gitignored, so a file lands here only when someone picks it.
The record behind each one — budget, lineup, medians — is
[`experiments/RESULTS.md`](../../experiments/RESULTS.md); this page only says
which command redraws it.

| file | panel | budget |
| --- | --- | --- |
| `headline-batches.png` | `headline`, batches axis | 20 seeds × 3000 batches |
| `invariance-units.png` | `invariance` | 20 seeds × 3000 batches |

The canonical run draws both (and `smooth`, `multimodal`):

```bash
uv run --group exp python -m experiments.compare_baselines \
    --seeds 20 --groups headline,smooth,multimodal,invariance --max-batches 3000
```

## Updating one

Re-run the command, look at `experiments/figures/<panel>/batches.png`, copy it
over the file here only if it is the panel the caption claims, and adjust the
row above when the budget moved. Nothing syncs these automatically: an automatic
copy would let the image drift every time somebody re-ran something, which is
exactly how a README figure ends up showing a run it does not describe.

Two rules for this page:

- **Numbers stay in `RESULTS.md`.** A caption in the root README names the
  lineup and the budget; it never repeats a median, because a median without the
  line that measured it reads as a property of the library.
- **A figure is evidence of one run, not a thing the reader regenerates in a
  minute.** The traces behind them are 83 MB and gitignored. Hosting them as a
  release asset is still undecided; until then the commands above are the only
  route back to the pixels.
