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
  pytest. `.github/workflows/publish.yml`: on `v*` tags, tests → `uv build` in
  an unprivileged job, then a separate `pypi`-environment job holding only
  `id-token: write` does the trusted publishing.
  `.github/workflows/docs.yml`: builds the site strictly on every PR
  and deploys it to GitHub Pages from `main`.
- User docs: MkDocs + `mkdocstrings`, sources in `docs/` (see ADR-0005). The
  internal notes (`PLAN.md`, `adr/`) sit beside them and are excluded from the
  build. `CHANGELOG.md` starts at 0.1.0.
- Docstrings: `n_path_schedule` carries `Args` + a runnable `Example`;
  `DiagonalPreconditioner.__init__` documents every knob.

## Restart-cost symmetry — f-only (landed); horizon=1 (landed)

Not an ADR yet; the durable record of a decision + its in-progress refactor.
The verdict's evidence (`observe_continue` Δ, `observe_restart` payoff, the
incumbent) is f-only; `g` is needed only to propose the step direction (the
start's `g`, free) and to fold the committed point's secant. Gradient-sweeping
the losing points is pure waste. The moves:

- **f-only evaluation (opt-in, landed).** A keyword-only
  `evaluate_f: EvaluateF | None = None` seam: points in, `(f, feas?)` out. The
  mask is required, not a nicety: probe group-min and step-selection must still
  tell "finite but infeasible" from viable, and f alone cannot. `None` means no
  f-only path at all - the probe takes a gradient with each draw - which is why
  a depth above one is refused (next bullet). `evaluate_batch` stays the one
  positional argument (sealed).
- **Multi-draw probes are f-only-only (gate + default).** The engine cannot see
  the solver's `n_alpha` (the seam checks presence, not members), so it cannot
  enforce the true symmetry `n_alpha == n_probe_evals`; its conservative proxy
  is the **g-count**: without `evaluate_f`, a probe computes `g` `n_probe_evals`
  times while one continue step computes `g` once (default `n_alpha=1`), so the
  expensive term diverges `n_probe_evals:1` — the f-count barely matters (g ≈
  2–5× f). With `evaluate_f`, both sides compute `g` exactly once (the rest is
  f-only), collapsing that asymmetry and leaving only the cheap f-count to
  match. So `evaluate_f is None and n_probe_evals != 1`
  raises `ValueError` at entry, and the default becomes `n_probe_evals = 1`.
  Deferral is keyed on depth, not on the seam: with a single draw there is no
  losing gradient to defer (the winner *is* the draw), so `n_probe_evals == 1`
  takes the fused path even when `evaluate_f` is supplied — otherwise the point
  would be served twice for nothing.
  Consequences: under that default a group-min is a single fresh draw, not
  a min-of-n group-min — the "pinned 2 (empirical)" advice in the docstring
  moves to "2 iff you also pass `evaluate_f`", and the Phase A–D numbers were
  measured at 2. With `evaluate_f`,
  `n_alpha != n_probe_evals` is the cheap, bounded (f-only) asymmetry that is
  accepted, not banned.
- **A start needs both halves finite (landed).** A probe start is committed
  only when `f` *and* `g` are finite: a finite `f` with a non-finite gradient
  is proposed from (`-p.g`) and folded on (`best.g - p.g`), so one NaN in it
  poisons that coordinate's magnitude EMAs for the rest of the run. Such a
  start arrives stalled, like a dead draw, and waits for an EI verdict. The
  step side already gated trials on `f` and `g` together; this closes the same
  hole on the probe side.
- **Iteration clock stays; a work clock never lands.** A cost-denominated
  horizon would denominate in runtime state (how many paths restarted or
  stalled), which is not JIT-static — a restarted flock's size differs from
  `n_paths` and forces re-traces (dfbench evidence). Iteration count is the one
  static-countable quantity, so the horizon stays an iteration clock.

**horizon=1 is the symmetric point, not a rollback.** Reverse-mode `g` ≈ 2–5×
`f`, an algorithmic constant on CPU and GPU alike (the "batch wall-clock"
argument is about *widening*, not *f vs f+g*: the backward pass over the batch
is real work on GPU too). Under f-only, one argmin step = `n_alpha` f-only +
1 g, one probe = `n_probe_evals` f-only + 1 g; equal iff
`n_alpha == n_probe_evals`, and then one continue sample (one step, horizon=1)
costs exactly one probe. Default `n_alpha=1` keeps a documented, bounded
asymmetry (continue = one gradient point vs probe = min-of-n). A genuine Armijo
line search is also f-only-compatible (its test reads the *start*'s g + the
trial's f); only Wolfe/curvature conditions read the trial's own g and go
asymmetric — bounded by the method's own enumeration, document, don't redesign. Do **not** force
`n_alpha = n_probe_evals`: they are independent knobs (line-search resolution
vs exploration breadth).

### `_Horizon` deletion (landed, uncommitted)

With horizon collapsed to 1, `_Horizon`'s `_span`/`_done_at`/`_poisoned`/
`iterations_until_sample` all vanish: every path closes every iteration, and a
restart is back in phase instantly (no out-of-phase clock). Replace with:

- loop-local `prev_f: list[float] = [point.f for point in points]`;
- in `apply_restarts`: `prev_f[i] = point.f` (instead of `horizon.restart_at`);
- the fold, skipping just-restarted paths (their probe was this iteration's
  move, not a step, so their first fold is the next iteration's one-step Δ):
  `for i in range(n_paths): if i in just_restarted: continue; delta = inf if
  stalled[i] else points[i].f - prev_f[i]; observe_continue(i, delta);
  prev_f[i] = points[i].f`
  (a later increment replaced the `inf` with `None` - see
  `docs/adr/0006-restart-rule-reads-stall-as-absence.md`)
- the all-stalled shortcut collapses to
  `if restart_rule.samples_until_verdict() >= max_evals - it: all_stalled =
  True; break` (one sample = 1 iteration; the `+1` vote term merges into
  `needed`; `needed=0` still disables the shortcut as today).
  (This shortcut has since been removed together with the batch budget - see
  *Budget: an iteration horizon* below.)

Ripple (swept): `engine.py` (module doc, `n_path_schedule` doc, loop
comments); `restart.py` (`observe_continue` / `samples_until_verdict` / class
docstrings — the "priced n_probe_evals evals" wording is gone);
`tests/test_solver_seam.py` (the horizon-count runs restated as one sample per
iteration, `test_one_sample_per_iteration_at_every_probe_size`, and
`test_every_path_feeds_once_per_iteration`); `tests/test_scheduler.py`
(stalled-flock test: `n_iters` is now 1, not 2); `ALGORITHM.md`, `CONTEXT.md` (the
`horizon` glossary entry is gone), `docs/adr/0004-*` (amended, part 1
superseded), `docs/tutorial/getting-started.md`.

**Landed in the working tree (uncommitted):** engine, restart, tests, and every
listed doc are swept; 68 tests green, ruff and pyright clean. The default
`n_probe_evals` was flipped 2 → 1 in the `_Horizon` deletion (`0ffe135`); the
`evaluate_f` seam and its gate follow in the next commit. Shape the seam took:
`n_path_schedule(..., evaluate_f=...)` is keyword-only and optional, a probe
serves `n_probe_evals` value-only draws then one fused batch at the per-column
winners (so a probe serves `n_probe_evals` uncharged f-only draws plus one
billed fused batch, and one `g`), the draws
stay exactly `n` wide rather than widening to `n_probe_evals × n`, and
`evaluate_f is None and n_probe_evals != 1` raises `ValueError` at entry.

Caveat to record, not erase: after horizon→1 the verdict compares one probe
against one step, and the load-bearing cost is g-count — without `evaluate_f`
the non-f-only world is g-symmetric only at `n_probe_evals = n_alpha` (default
1 = 1, which the gate enforces); with `evaluate_f` both sides compute g once
and the only residual is the cheap f-count `n_alpha != n_probe_evals`, plus a
genuine Wolfe/curvature solver that reads the trial's own g (bounded by its own
enumeration). Don't claim the verdict is exactly unbiased in those residual cases.

**Init folded into restart (landed):** the dedicated initialization path (a
probe before the loop, then a cold-start `solver.init` / `observe_restart`
block) was redundant - `apply_restarts` already probes, extends the group-min pool
and re-seeds the method. The loop's iteration 0 is now the initialization
probe, unlocked by one rule contract: `should_restart` returns `True` while the
group-min pool is empty. `LocalSolver.init` is gone (`restart` is the only
cold-start call). Two consequences worth keeping: `n_iters` counts iteration 0, so
the two stops that both used to report `0` are now distinct (a refused probe is
`0`, a refused first step is `1`); and the cold start opens each path's stage,
so `samples_until_verdict()` reports a real window instead of `0` before the
first fold.

### Budget: an iteration horizon (landed)

`max_evals` billed served batches, which made the engine's own stop a cost
meter. Two things broke it: a step serves as many trials as its method asks for
and a
restart serves a probe, so an iteration's batch count is runtime state the
caller cannot predict; and a stalled iteration serves nothing, so the meter
under-counts exactly where the loop spins - which is why an all-stalled
shortcut had to exist at all. Replaced by:

- `max_iters`: a deterministic horizon, iteration 0 = the init probe. Not a cost
  proxy - one iteration serves from zero batches (a fully stalled flock) up to a
  probe plus however many trials the method's line search serves.
- `time_limit`: the same stop in wall-clock seconds, checked between iterations
  and never inside one.
- At least one of the two, else `ValueError`. `budget_check` stays an extra
  per-batch gate and the escape hatch for any other stop condition; it cannot
  replace the horizon, because a stalled iteration asks for no batch.
- `n_batches`/`n_points` are reports, not caps, and they count the only billed
  unit: the fused value+gradient pass. A probe's f-only draws are cheaper work
  and appear in neither (nor are they gated), so the counts stay on the axis
  that dominates; the default shape has no f-only draw at all. `status` is
  `max_iters`, `time_limit` or `budget_exhausted`.
- The all-stalled shortcut, the `all_stalled` status and the loop's `it` counter
  are gone: a stalled flock idles to the horizon, folding one `inf` sample per
  path per iteration. `samples_until_verdict()` loses its only engine caller and
  is marked to be cut.

Migration is not a rename: an iteration can burn about `1 + (n_probe_evals + 1)`
times the batches of a one-batch iteration before the method's own trials are
counted, so `max_evals=N` →
`max_iters=N` silently raises the work budget. Re-tune from `n_batches`.

Still open, deferred until the finite-value refactor: a flock that stalls as a
whole waits out its verdict cadence (the cheap fix is to restart every path once
the flock stalls). The case that really needs it is a dead pool - a first probe
that comes back all-non-finite, where `EI_restart = 0` and no verdict revives
the paths.

Deferred, and why: **f-only line search.** The trial's `g` is pure waste when the
trial loses - warm-up backtracks and every non-argmin sample of an `n_alpha` grid
pay a gradient the run never folds. Deferring it the way a probe does (serve `f`,
let the winner be chosen, then pay one fused batch) would need the engine to know
whether *this* step does more than one trial, and it cannot: `n_alpha` lives in
the solver, and the seam deliberately reads presence, not member values. Keying
the deferral on `n_alpha == 1` is therefore not expressible engine-side, and at
the default `n_alpha = 1` there is nothing to defer anyway (the single trial is
the winner, and f-only would serve it twice). The clean home for this is a
solver-side protocol ("this batch needs values only"), which is not worth
extending for a cost the default configuration never pays.

Landed: **`max_ls` is retired.** It was an engine-side cap on a loop the solver
owns - the same quantity as `n_alpha`, on the wrong side of the seam. The step
loop is now `while rejecting`, with termination a stated contract (`verdict` must
eventually return `accept` or `give_up`) instead of an engine failsafe;
`DiagonalPreconditioner` terminates on its own (the post-warm-up grid is capped
by `n_alpha`, and the warm-up's geometric shrink bottoms out at a zero-length
step, which passes Armijo because `gtd = 0`). The migration bullet
above therefore no longer quotes an engine-side bound for one iteration's
batches.

## Next

1. **First PyPI upload is a tag away** (not published yet; version stays 0.1.0).
   - One-time PyPI setup: add the project as an **OIDC trusted publisher** for
     `github.com/watson-tree/soar_lite` → `.github/workflows/publish.yml` with
     environment `pypi` (no token; `publish.yml` already requests
     `id-token: write`). The four fields — owner, repository, workflow filename
     and environment — must match the YAML exactly or publishing fails.
   - Add a `pypi` GitHub environment with required reviewers if you want a manual
     gate before upload; the workflow already names it.
   - Then `git tag v0.1.0 && git push origin v0.1.0`.
   - `soar-lite` was available on PyPI at last check; rebuild `dist/` from a
     clean tree first (the committed `dist/` predates this release work).
2. **Docs site one-time setup**: set the repository's Pages source to
   **GitHub Actions**; `docs.yml` then publishes
   `https://watson-tree.github.io/soar_lite/` (linked from `[project.urls]`).
3. **f-only evaluator plus the probe gate** (`## Restart-cost symmetry`
   above): both landed in the working tree, uncommitted. Once committed,
   re-run the multi-path escape experiment to decide whether the default probe
   depth should return to 2 — and measure whether an f-only draw's kernel-launch
   saving survives the extra winner batch on a GPU.

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