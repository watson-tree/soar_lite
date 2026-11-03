# Measurement record: `compare_baselines`

Every number these panels have produced lives here, in reverse chronological
order, each entry tagged with the lineup and the budget that produced it.
`DESIGN.md` says what the panels *are* and what they are meant to show; it
states no measured numbers, so nothing there can age in place.

Three rules:

- Every entry names its **lineup**, its **budget** (seeds x batches) and its
  **status** — `CURRENT`, `SUPERSEDED by <entry>`, or `NOT REPRODUCIBLE`.
- **Numbers are never re-edited.** A new measurement is a new entry plus a
  status change on the old one, and the old numbers stay exactly as measured.
  Quietly updating a number is how a document ends up quoting a run it no
  longer describes.
- Only measurements belong here. Propositions, rules and the budget contract
  live in `DESIGN.md`, release history in `CHANGELOG.md`, frozen decisions in
  `docs/adr/`.

Unless a table says otherwise it holds **medians**, the statistic the figures
plot. The harness's stdout line is `final mean best` — mean, not median, and
not comparable with the tables below.

## 2026-11-02 — lineup v2 (momentum arm, BLR restart panel) — CURRENT

Budget: 20 seeds x 3000 batches, all five groups (`headline`, `smooth`,
`multimodal`, `invariance`, `blr`). Code: `experiments/` at `18c146b`.

Wall clock on the machine this ran on: 5h06m total, 3h12m of it the `blr`
panel. That is machine-specific, and it is a record rather than a run-book
budget — do not quote it as an expectation.

### P② and P① medians, batches axis

| problem | n_paths=1 | n_paths=16 | mu=0.9 | Adam | L-BFGS-B |
| --- | --- | --- | --- | --- | --- |
| diag_quad_d100_cond10000 | 1.1e-60 | 7.6e-61 | 5.8e-9 | 2.0e-63 | 1.5e-7 |
| rot_quad_d100_cond1e+06 | 61.0 | 33.9 | 44.1 | 0.547 | 1.4e-4 |
| oscillator_n20_F20 | 7.7e-6 | 2.2e-7 | 4.7e-6 | 7.5e-5 | 7.5e-3 |
| ackley_d100 | 19.78 | 19.59 | 19.82 | 19.57 | — |
| rastrigin_d100 | 444.0 | 376.2 | 1437 | 807.9 | — |

**Annotation, 2026-11-03 — the `mu=0.9` column belongs to a retired rule.** It
was measured with the projected momentum direction (`w = g + m⊥`, which kept
the step a descent direction). That rule was dropped: the direction is now the
unprojected current momentum `m = (1-mu)·g + mu·m_prev`. Re-measured on the
diagonal quadratic alone, 2 seeds x 3000 batches, `n_paths=1`: `mu=0` 1.3e-60,
`mu=0.9` 6.6e-9 - the accuracy cost survives the rule change. The other
`mu=0.9` cells (rotated quadratic 44.1, oscillator 4.7e-6, ackley 19.82,
rastrigin 1437) have not been re-measured; read them as directional, not as the
current arm's numbers.

The points axis is not a second opinion on this table: a chain spends exactly
one point per batch, so `n_paths=16` covers only its first ~187 batches there
and its medians are truncation artefacts (rotated quadratic 1100 on points
against 33.9 on batches). Read the batches column.

### Reading 1 — momentum repairs direction, it does not substitute for width

Against serial, `mu=0.9` helps exactly where off-diagonal coupling exists:
rotated quadratic -28% (61.0 → 44.1), oscillator -39% (7.7e-6 → 4.7e-6). It
loses exactly where coupling is absent or where the choice of basin dominates:
the diagonal quadratic (5.8e-9 against 1.1e-60 — momentum smears the
exact-Newton tail) and rastrigin (+224%, 444 → 1437 — chains are pushed into
worse basins). Ackley is a tie at 19.8, where nothing moves at all.

Width beats momentum on all five problems: rotated quadratic 33.9 < 44.1,
oscillator 2.2e-7 < 4.7e-6, rastrigin 376 < 1437, diagonal 7.6e-61 < 5.8e-9.
The two knobs are not interchangeable, and `mu=0.9` is not a cheap substitute
for `n_paths=16`.

### Reading 2 — the full-memory ceiling only pays where coupling dominates

L-BFGS-B wins only on the rotated quadratic (1.4e-4, four orders below our best
33.9). Everywhere else it loses, and not marginally: on the diagonal quadratic
soar-lite's 7.6e-61 against its 1.5e-7 is 53 orders, and on the oscillator its
7.5e-3 against 2.2e-7 is 4.5 orders. A second-order method pays for the
off-diagonal curvature it carries, so it collects exactly where the diagonal
secant's error *is* that curvature; where the error lies elsewhere its line
search dives deterministically into a local minimum and stays there.

### Reading 3 — a fitted restart beats a uniform re-draw on every seed

Rastrigin, all arms `n_paths=16`, medians: Sample draws 376.2, BLR identity
186.6, BLR sphere 279.1.

Identity beats Sample on 20/20 seeds, and the two distributions do not overlap
(identity 11-264, Sample 338-410). Sphere also beats Sample on 20/20, but loses
to identity on 20/20 (251-311): the gain is the centre bias *plus* the right
projection, not fitting on its own.

Consistency check: the Sample arm's median equals the multimodal panel's
`n_paths=16` median to the last digit (376.161 both). Same runner, same seeds -
which is what makes the two panels comparable in the first place.

### What this supersedes

The 2026-10-28 entry below, whose restart on/off lineup the momentum and BLR
panels replaced. Its Ackley and unit-rescaling readings are carried here
unchanged, because they describe behaviour the v2 panels do not touch.

## 2026-10-28 — lineup v1 (restart on/off) — SUPERSEDED by the v2 entry

Budget: 20 seeds x 3000 batches. This entry records the restart on/off lineup:
the width and the restart were the two single-attribute steps (`n_paths=1` →
`n_paths=16` at restart off is parallelism; restart off → on at fixed width is
the restart).

The figures were re-run on the refactored engine (20 seeds, 3000 batches). Adam
and L-BFGS-B came back bit-identical — the harness did not move — so every
shift recorded here is engine-side.

The restart arm is `restart_stalled=True` with a `SampleGlobalModel`: a stalled
path is re-drawn to a fresh start from the domain — no EI, no kernel density
estimate, no shared pool. Adam and L-BFGS-B come back bit-identical again, the
single-path arm moves by at most 4.5% (on Ackley not at all), and the restart
arm moves the most. A reading of that restart arm is **not written here**: the
EI-era diagnosis was deleted together with the rule, and a replacement must be
re-derived from the plots, not asserted. The two readings below travel with
them.

### Ackley is a failure for every curve, not a loss for the preconditioner

Ackley d=100 at 3000 batches: `n_paths=16` 19.47, Adam 19.57, `n_paths=1` 19.78,
against `f* = 0`. All three miss the funnel; 19.5-19.9 is what Ackley looks like
outside it, and the 1% spread only says where each froze. `n_paths=1` is
bit-flat from batch 781 on: the diagonal secant has nothing to act on where the
gradient is a ~1e-8 radial term buried in cos-frequency noise. That run predates
the removal of the warm-up guard — it froze because Armijo kept shrinking
`alpha` to a no-op step, and the same dead signal without the guard commits
steps that go nowhere; the freeze is not re-measured here. Either way it is the
designed boundary of a first-order diagonal preconditioner, not a tuning
failure. `n_paths=16`'s 0.1 edge over Adam was brute-force best-of-16 starts,
which is why a working restart-and-sample story was expected to show up rather
than suspicious.

Measured again under v2 (3000 batches, 20 seeds): `n_paths=16` 19.59, Adam
19.57, `n_paths=1` 19.78 — the edge over Adam is gone, the ordering otherwise
unchanged.

### The unit-rescaling residual was exactly the warm-up guard

Diag quadratic d=100, `x' = 10x`, zero retune, measured while the warm-up still
carried an Armijo backtrack: soar-lite lands at `3.3e-65` in 19 points (base)
and `1.6e-65` in 12 (rescaled); Adam goes `2.0e-63` vs `7.2e-5`, 58 orders
apart. The panel claims outcome-level overlap, and it holds — 2x in f at a floor
of machine zero, against 58 orders for a single-lr method. The 1.6x in points
was fully explained and not noise: the warm-up step is `alpha0 * (-g)`,
unpreconditioned, so under `x' = s x` its physical length scales as `1/s^2`, and
the base system needed `ceil(log2(s^2)) = 7` more Armijo halvings. Measured
then: base was rejected 12 times, the rescaled system 5 — exactly 7 apart — and
after the first acceptance both needed exactly 6 points to reach their floor.
The visible horizontal shift was therefore a first-step artefact, not
path-dependent optimization.

That artefact is structural rather than tuned away, so it is gone by
construction now that the warm-up is a single unruled step: there is nothing to
halt, and both systems take exactly one first step. What remains — and is not
addressed — is the same scale dependence in the warm-up *direction*: `-g` is a
gradient-type quantity, so the first step's physical length still scales as
`1/s^2`; it simply stops being measured against the objective. Removing that
needs a scale-free warm-up, which the secant cannot provide before it has its
first pair.

Not re-measured under v2: the invariance panel writes no JSON and prints no
numbers, so these floors remain a reading of that run, not a value anyone can
re-derive from a shipped artifact.

## pre-2026-10 — step-length calibration sweeps — NOT REPRODUCIBLE

Budget: mixed and ad hoc (the grid sweep was 300 batches, 3 seeds; the `alpha0`
damping sweep is not recorded at that detail). **The code path is gone**: the
`alpha0` sweep driver no longer exists in `compare_baselines.py`, so these
numbers cannot be re-measured by the current script, only cited.

These numbers were measured on the engine as it stood before the 0.1.0
refactor. They are the evidence behind an architectural verdict — the residual
error is in the *direction* `-d ⊙ g`, not its length — so a stale number does
not retract the verdict; treat any figure here as a floor on effort, not as a
reproducible result.

- **`n_alpha` grid** (300 batches, 3 seeds, rotated `cond=1e6` d=100):
  `n_alpha` larger is strictly worse — `n_paths=1` at `backtrack=0.5`:
  `1285 → 4220 → 7382 → 12535`. `backtrack` then also mattered at `n_alpha=1`
  (`1285 → 1056` at 0.1) because it shrank the **warm-up Armijo** backtrack — a
  different first step (path shift), not a grid effect. That sweep predates the
  removal of the warm-up guard: with neither a guard nor a grid at `n_alpha=1`,
  `backtrack` is now inert there and only sets the grid's spacing.
- **`alpha0` damping** (the "small step along the wrong direction might pay"
  hypothesis, `alpha0 ∈ {0.125, 0.25, 0.5, 1.0, 2.0}`, `n_paths ∈ {1,16}`,
  rot+diag): damping is strictly worse on the rotated case
  (`1285 → 1976 → 3211 → 5356`) and `alpha0 > 1` is catastrophic (`8e5`); on
  diag, `1.0` is the one-shot win (`3.5e-65` in 20 points) and any damping only
  slows it.

Scope: these sweeps ran on **one landscape** (the rotated and diagonal
quadratics), which isolates the off-diagonal direction error. Measured for
rotation coupling only; that the headline's interaction coupling is equally
immune is an analogy from this, not a separate measurement. Silent on the
diagonal part of the error, where the per-coordinate secant is itself a biased
curvature estimate — unmeasured, and deliberately left open in `DESIGN.md`.
