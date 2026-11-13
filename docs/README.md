# `docs/`

Two things live here, and only one of them is built.

- `docs/site/` — the source of the published site. `mkdocs.yml` sets
  `docs_dir: docs/site`, so the tutorial, the how-tos, the API reference and the
  explanation all build from here, with `docs/site/hooks.py` as the build hook.
- `docs/figures/` — the panel PNGs the root README shows, plus the maintenance
  notes in `docs/figures/README.md` on how they were rendered. Nothing under
  `docs/site/` links them, and nothing here is built: the directory sits outside
  `docs_dir`, so keeping it out of the site needs no exclude rule.

The generated site is `site/` at the repository root — a different directory
from its source, `docs/site/`.
