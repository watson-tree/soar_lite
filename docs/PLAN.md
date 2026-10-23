# Plan

Handoff notes for the next working session: where the repository stands, what
comes next, and the conventions a fresh session must not relearn by trial.
Terminology lives in `CONTEXT.md`; decisions in `docs/adr/`; the experiment
design in `experiments/DESIGN.md`.

## Status

One linear history; remote `origin` (`git@github.com:watson-tree/soar_lite.git`,
public) is configured and pushed. The optimizer is a three-module first-order
library — `soar_lite/engine.py`, `soar_lite/solver.py`, `soar_lite/restart.py`
— with assert-style tests (pytest-collectable, 66 pass). Comparison experiments
Phase A–D (headline, smooth, multimodal, invariance, Adam lr sweep) are designed
in `experiments/DESIGN.md`, reproducible per `experiments/README.md`, and
archived in git history. Adam's per-problem lr was retuned to the sweep grid
argmin.

The folder was renamed `light_multi_paths` → `soar_lite` and the venv rebuilt
(`uv sync --all-groups`, CPython 3.11). Release hygiene is in place:

- `LICENSE`: Watson Tree, 2026.
- `pyproject.toml`: `authors` + `[project.urls]` (Repository, Homepage); PEP 639
  `license = "MIT"` with the old `License ::` classifier dropped.
- `README.md`: CI badge + Colab badge; `notebooks/quickstart.ipynb` runs the
  `--quick` smoke check.
- `.github/workflows/ci.yml`: on push/PR, Python 3.11 + 3.12, ruff → pyright →
  pytest. `.github/workflows/publish.yml`: on `v*` tags, `uv build` then trusted
  publishing. `.github/workflows/docs.yml`: builds the site strictly on every PR
  and deploys it to GitHub Pages from `main`.
- User docs: MkDocs + `mkdocstrings`, sources in `docs/` (see ADR-0005). The
  internal notes (`PLAN.md`, `adr/`) sit beside them and are excluded from the
  build. `CHANGELOG.md` starts at 0.1.0.
- Docstrings: `n_path_schedule` carries `Args` + a runnable `Example`;
  `DiagonalPreconditioner.__init__` documents every knob.

## Next

1. **First PyPI upload is a tag away** (not published yet; version stays 0.1.0).
   - One-time PyPI setup: add the project as an **OIDC trusted publisher** for
     `github.com/watson-tree/soar_lite` → `.github/workflows/publish.yml` (no
     token; `publish.yml` already requests `id-token: write`). Then
     `git tag v0.1.0 && git push origin v0.1.0`.
   - `soar-lite` was available on PyPI at last check; rebuild `dist/` from a
     clean tree first (the committed `dist/` predates this release work).
2. **Docs site one-time setup**: set the repository's Pages source to
   **GitHub Actions**; `docs.yml` then publishes
   `https://watson-tree.github.io/soar_lite/` (linked from `[project.urls]`).

## Conventions

- One linear history — no branches, no merges.
- `experiments/results_*.json` and `*.png` are gitignored. Rebuild figures from
  JSON with `--plot-only`; style/color/rename changes never trigger a re-run.
- Toolchain: `python3`, not `python`. Lint/type: `uv run --no-sync ruff check`
  and `uv run --no-sync pyright`. Tests:
  `uv run --no-sync python -m pytest -q` (`pytest` sits in `dev`). Experiments:
  `uv run --no-sync python -m experiments.compare_baselines …`; install extra
  deps with `--group dev` and `--group exp`. CI (`ci.yml`) runs the same four
  commands; publishing is `uv build` in `publish.yml`, never a manual wheel.
- Docs: `uv run --no-sync mkdocs build --strict`; sync the `docs` group first
  (`uv sync --all-groups` keeps dev/exp/docs together).