# Plan

Handoff notes for the next working session: where the repository stands, what
comes next, and the conventions a fresh session must not relearn by trial.
Terminology lives in `CONTEXT.md`; decisions in `docs/adr/`; the experiment
design in `experiments/DESIGN.md`.

## Status

One linear history (through `7de2378`, the Adam lr retune); working tree clean.
The optimizer is a three-module first-order library — `soar_lite/engine.py`,
`soar_lite/solver.py`, `soar_lite/restart.py` — with assert-style tests
(pytest-collectable, 66 pass). Comparison experiments Phase A–D (headline,
smooth, multimodal, invariance, Adam lr sweep) are designed in
`experiments/DESIGN.md`, reproducible per `experiments/README.md`, and archived
in git history. Adam's per-problem lr was retuned to the sweep grid argmin.

## Next

1. **Folder rename** — `~/projects/light_multi_paths` → `~/projects/soar_lite`.
   Nothing in the source references the folder name; only the venv's console
   scripts and `activate*` shebangs do. After the move rebuild the venv:
   `rm -rf .venv && uv sync --all-groups`.
2. **Phase E — release hygiene** (blocked: no git remote configured yet).
   Before the first PyPI upload:
   - `LICENSE`: copyright holder + year.
   - `pyproject.toml`: `authors` and a `[project.urls]` block (Repository,
     Homepage).
   - `README.md`: CI/status badge + a Colab-runnable link to the experiments.
   - Build (`hatchling`) and upload to PyPI.

## Conventions

- One linear history — no branches, no merges.
- `experiments/results_*.json` and `*.png` are gitignored. Rebuild figures from
  JSON with `--plot-only`; style/color/rename changes never trigger a re-run.
- Toolchain: `python3`, not `python`. Lint/type: `uv run --no-sync ruff check`
  and `uv run --no-sync pyright`. Tests:
  `uv run --with pytest --no-project python -m pytest -q`. Experiments:
  `uv run --no-sync python -m experiments.compare_baselines …`; install extra
  deps with `--group dev` and `--group exp`.