# Contributing

## The checks a change has to pass

```bash
uv sync                       # jax
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

## The benchmark suite

`experiments/` holds the comparison harness behind the figures in the README. It
is a sub-library, not a dependency: it imports `soar_lite`, nothing in the
package or the docs imports it, and neither the wheel nor the sdist ships it. Run
instructions are in `experiments/README.md`, what it measures and the rules it
follows in `experiments/DESIGN.md`, and the numbers in `experiments/RESULTS.md`.
