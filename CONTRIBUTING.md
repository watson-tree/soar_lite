# Contributing

Welcome. Whether you have a question, a bug report or a change in mind, feel
free to open an issue or a pull request — the sections below say what a change
has to clear.

## Proposing a change

Open a pull request; GitHub runs the checks below on it. An issue is the right
place for a question, a bug report, or an idea that has no implementation yet.

## The checks a change has to pass

```bash
uv sync --all-groups          # jax, the dev tools, and the docs build
uv run python -m pytest -q    # all tests in one go
uv run --no-sync python -m tests.test_local_solver   # or one module at a time,
uv run --no-sync python -m tests.test_global_model   # each module also runs
uv run --no-sync python -m tests.test_scheduler      # as a plain script
uv run --no-sync python -m tests.test_plugin_seam
uv run ruff check && uv run ruff format --check && uv run pyright
uv run mkdocs build --strict  # the docs site; a warning fails the build
```

The tests are assert-style self-checks: plain `assert` in `test_*` functions,
pytest-collectable, each module also runnable as a script — no fixtures, no
recorded golden numbers that would encode this repository's history.

## Front doors

`docs/index.md` is the canonical home page and the root `README.md` mirrors it
for the sections both carry. A fact in the opening, or in any section either one
carries from the other — `Install`, `Minimal run`, `What's inside`, `Citing` — is
corrected in `docs/index.md` and copied across;
`tests/test_front_doors_agree.py` fails if the two drift apart, or if a link in
either one stops resolving. Link destinations are the sole exception — the site
addresses its neighbours from inside `docs/`, the README from the root.

## Names

Names have fixed meanings: `CONTEXT.md` is the glossary, including the terms to
avoid. Consult it before introducing one.

## Records are append-only

`DEPRECATED.md`, `experiments/DEPRECATED.md`, `experiments/RESULTS.md` and
`CHANGELOG.md` are records, not living documents: nothing in them is deleted or
rewritten to say something else, and a measured number is never re-edited — a
new measurement or a superseded direction is a new entry, and the old one stays
as written. `experiments/DESIGN.md` is the opposite: a current-state document,
so a passage that is no longer true there is rewritten in place.

## The benchmark suite

`experiments/` holds the comparison harness behind the figures in the README. It
is a sub-library, not a dependency: it imports `soar_lite`, nothing in the
package or the docs imports it, and neither the wheel nor the sdist ships it. Run
instructions are in `experiments/README.md`, what it measures and the rules it
follows in `experiments/DESIGN.md`, and the numbers in `experiments/RESULTS.md`.

## Open directions

The two plugin seams (`LocalSolver`, `GlobalModel`) are the natural extension
points. Writing your own is currently documented only as the contracts in
`docs/reference/api.md` and the `Protocol`s in `soar_lite/engine/contracts.py`,
not yet as a how-to guide. `BLRGlobalModel`'s acquisition has a known upgrade
(see the `shortcut:` note in `soar_lite/global_model/blr.py`), and more global
models or local solvers (e.g. line search) are welcome — `experiments/RESULTS.md`
records where the tuned baselines stand.
