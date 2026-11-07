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
  live in `DESIGN.md`.

Unless a table says otherwise it holds **medians**, the statistic the figures
plot. The harness's stdout line is `final mean best` — mean, not median, and
not comparable with the tables below.

## 2026-11-07 — lineup v5 (`mu` default back to 0) — CURRENT, NOT YET MEASURED

`DiagonalPreconditioner`'s default `mu` moved from 0.63 back to 0. Every arm
that takes the default re-keys with it (`...-mu0.63-...` → `...-mu0-...`), and
the momentum-off arm is no longer a curve of its own: it *is* the default arm.

**Nothing has been re-measured against this lineup.** No table here describes
what the harness now draws. The v3 traces stay on disk and stay valid as the
record of the 0.63 default; figures drawn from them say so. The roster is now
`n_paths=1` (default `mu=0`), `n_paths=1, mu=0.9`, `n_paths=16`; the missing
measurement is `np16-mu0` on every problem plus the blr panel's three arms at
`mu=0`.

Why the window moves back, in one line: memory damps the residual spectrum and
shrinks direction noise, and both are landscape properties the preconditioner
cannot see — so the default is the branch that assumes neither, and it is also
the one that keeps a descending direction. The argument is in
`docs/explanation/how-it-works.md` §Momentum.

## 2026-11-04 — lineup v4 supplement: the compartment-kinetics problem

Budget 20 seeds x 3000 batches, `headline` panel, at the code of this commit.
This is a **row added to the lineup**, not a new lineup: `pk_chain_n21_d61` is a
sixth problem, so it has no previous measurement to supersede and no other
panel was re-measured for it. Every trace file listed below is new.

Why a sixth problem: the panel had one real inverse problem (the oscillator, a
frequency-domain one) and a synthetic stand-in for dense rotational coupling.
The compartment chain is a real *time-domain* one, and it sits on exactly the
structure the diagonal approximation is built for — nearest-neighbor coupling
in a compartment graph, rates spanning four decades — without repeating the
oscillator's physics. It is the second real problem and the first one whose
loss has coupled near-flat directions rather than resonance non-convexity.

| arm | median | Q1 | Q3 | worst seed | seeds above 1e-2 |
| --- | --- | --- | --- | --- | --- |
| soar `n_paths=16`, mu=0.63 | 4.51e-7 | 2.93e-7 | 7.52e-7 | 1.54e-6 | 0/20 |
| soar `n_paths=1`, mu=0.63 | 3.66e-5 | 6.11e-6 | 5.70e-5 | 1.69e-4 | 0/20 |
| soar `n_paths=1`, mu=0 | 2.06e-5 | 2.21e-6 | 6.52e-5 | 1.30e-3 | 0/20 |
| soar `n_paths=1`, mu=0.9 | 3.14e-4 | 3.99e-5 | 3.47e-4 | 2.13 | 1/20 |
| Adam `lr=1e-1` | 2.51e-4 | 1.11e-4 | 7.15e-4 | 2.28e-3 | 0/20 |
| L-BFGS-B | 5.06e-3 | 3.68e-5 | 4.40e-1 | 6.17 | 9/20 |

Median best-so-far along the budget (batches 100 / 300 / 1000 / 3000):

| arm | 100 | 300 | 1000 | 3000 |
| --- | --- | --- | --- | --- |
| soar `n_paths=16` | 2.55e-5 | 6.22e-7 | 4.51e-7 | 4.51e-7 |
| soar `n_paths=1`, mu=0.63 | 3.12e-4 | 7.69e-5 | 5.63e-5 | 3.66e-5 |
| soar `n_paths=1`, mu=0 | 5.30e-3 | 2.18e-5 | 2.08e-5 | 2.06e-5 |
| soar `n_paths=1`, mu=0.9 | 2.74e-1 | 9.14e-4 | 5.42e-4 | 3.14e-4 |
| Adam `lr=1e-1` | 2.47 | 9.00e-1 | 2.30e-2 | 2.51e-4 |

### Annotations

**The win is real and it is a rate gap, not a ceiling.** At equal budget the
serial arm ends ~7x under Adam and the wide arm ~560x under it. But Adam's row
is still falling steeply when the budget runs out (2.30e-2 at 1000 batches to
2.51e-4 at 3000), so unlike the rotated quadratic this is *not* a problem where
the first-order method cannot reach the serial arm's accuracy — it is one where
it is far slower and the budget here is what stops it. Saying "Adam cannot"
would be a claim about a run that was never continued.

**The wide arm carries most of the margin, and it stops early.** `n_paths=16`
reaches its final median by batch 300 and is flat afterwards — 187 batches is
all the batches figure shows it, so on that axis it is nearly a horizontal line
from the start. The points figure is where its cost is paid.

**`mu=0.9` hurts here, and one seed in twenty diverges.** Median 3.14e-4 sits
above Adam's 2.51e-4, and one seed ends at 2.13 — a factor ~10 above the problem's
starting loss. This is the mirror image of the oscillator, where the same arm is
the best serial column: a 10-step momentum window is a trade, and four decades
of rate spread is the regime that punishes it, because averaging gradients over
a window mixes coordinates whose scales differ by 1e4.

**L-BFGS-B is the least reliable arm here, not the ceiling.** Its Q1 (3.68e-5)
is better than Adam's median, its median (5.06e-3) is worse than Adam's, and 9
of 20 seeds end above 1e-2. The distribution is bimodal, not noisy. A
throwaway probe against `problems.pk_chain()` (not part of the lineup) shows
why: started exactly at the truth it stops after one evaluation at `f = 7e-33`,
but started at `truth * 1.05` it walks to a box edge and rests at `7.6e-4` with
coordinates up to 3x off. The loss is non-convex in the *coupled* sense —
individual coordinates are visible (moving one to a box edge changes the loss
by a median ~0.3) but several can move together and stay near the floor — so a
full-memory serial line search with no per-coordinate scaling lands in a bad
basin about half the time. The smooth panel's "exact curvature is the ceiling"
reading does not transfer to this problem.

**Adam's lr is the sweep's only strict interior optimum.** The grid gives 5.03
at `1e-2`, 1.10e-3 at `1e-1`, 2.75e-2 at `2e-1` (2 seeds, 3000 batches, the
`--lr-sweep` run that produced the appendix figure): both neighbors are far
worse, so this pick is a real optimum rather than an edge point or one of a tie.
It is also the only problem in the table where that is true.

**Identifiability, measured on the shipped problem** (construction check, not a
lineup measurement): moving one coordinate at a time from the truth to a box
edge makes the loss visible (>1e-6) for 17/20 transfer, 19/20 backward and
20/21 elimination rates in the ×4 direction, and 17/20, 19/20, 15/21 in the ×¼
direction. The weak block is elimination: a far compartment's clearance rate is
weakly constrained, which is the real ill-conditioning of compartment fitting
rather than a defect. The design that makes the transfer and backward rates
visible at all is the second bolus — with one end dosed the far rates barely
move the loss, which is why the problem doses both ends.

## 2026-11-04 — figure rule: each axis truncates at its own shortest extent

No trace was re-measured; every number in this file stands. What changed is where
the curves stop.

**The points axis used to cut in batches.** That fixed an earlier unit error (the
cut value was `max_steps`, a batch count, applied as a point count) but left the
symptom it was meant to fix: an `n_paths=16` arm serves 16 points per batch, so
on the points axis it ran to 48000 while every single chain ended at 3000, and
its curve trailed alone past the last x where any other arm had data. Each axis
now cuts at the shortest extent present **on that axis**: 3000 points on the
points figure, 3000 batches on the batches figure. All curves end together.

The wide arm therefore shows only its first ~187 of 3000 batches on that
figure. That is the axis doing its job - equal objective points served - and the
batches figure still shows the run whole. The `blr` panel is unaffected: all of
its arms are `n_paths=16`, so it is never truncated on points (14368 against
1000 batches, the identity arm having stopped at batch 898 on every seed).

**Arm styles now encode one attribute each**: linestyle is `n_paths` (solid 1,
dashed 16) and color is the momentum level. `mu=0` was drawn dash-dot on the
strength of being a special arm, but it is also `n_paths=1`, and only one of
those two facts should decide a linestyle.

## 2026-11-04 — lineup v3 supplement: the `mu=0.9` momentum arm

Budget and code unchanged from the entry below (20 seeds x 3000 batches, the
`headline` / `smooth` / `multimodal` panels at `878fda9`). This arm was measured
after that entry as a **column added to it**, not a new lineup: nothing it
measures is superseded.

Why this arm exists: its window is about 10 steps, the same window Adam's first
moment runs at `b1=0.9`, so the panels can ask the momentum question at Adam's
reach as well as at soar's own. Adam was not moved to meet it - a baseline whose
knobs mirror the method under test stops being one.

| problem | mu=0.9 | mu=0.63 (default) | mu=0 | n_paths=16 | Adam |
| --- | --- | --- | --- | --- | --- |
| diag_quad_d100_cond10000 | 7.2e-9 | 1.9e-9 | 6.2e-61 | 1.5e-9 | 2.0e-63 |
| rot_quad_d100_cond1e+06 | 25.1 | 42.0 | 90.2 | 24.7 | 0.499 |
| oscillator_n20_F20 | 1.6e-5 | 9.2e-6 | 3.1e-5 | 4.5e-8 | 7.5e-5 |
| ackley_d100 | 19.8 | 19.71 | 19.71 | 19.55 | 19.57 |
| rastrigin_d100 | 1700 | 1615 | 722 | 1520 | 807.9 |

Every column but `mu=0.9` restates the entry below; those numbers are that
measurement, not a second one. L-BFGS-B is left out: it takes no `mu`, so the
column would be the same numbers again.

### Annotations

**The arm answers a gap the v2 entry had to leave open.** That entry carries a
`mu=0.9` column measured under the then-current *projected* momentum rule
(rotated quadratic 44.1, oscillator 4.7e-6, rastrigin 1437) and says those cells
"have not been re-measured" under the unprojected rule. They are measured now:
25.1, 1.6e-5, 1700. The rotated case improved by ~40% when the projection was
dropped, but this is **not** the controlled comparison it looks like - the
engine changed in several other ways between the two lineups (the restart
default, the fold refactor, the momentum read), so the change is not attributable
to the projection alone.

**Momentum's effect on rastrigin does not scale with the window.** Default to
0.9 costs 5% (1615 -> 1700), while off to default costs 124% (722 -> 1615):
the damage is done by having momentum history at all on that landscape, not by
how far back it reaches. On rastrigin no momentum level recovers the `mu=0`
floor.

**The rotated quadratic is the one landscape that pays**, and it pays a lot:
90.2 (`mu=0`) -> 42.0 (default) -> 25.1 (`mu=0.9`), all but reaching what the
16-wide arm gets (24.7). Read that as confirming where momentum helps rather
than as a reason to raise the default - the same `mu=0.9` costs 3.8x on the
diagonal quadratic, where `mu=0` is exact, and the oscillator is non-monotone in
`mu` (1.6e-5 at 0.9 against 9.2e-6 at the default).

## 2026-11-04 — lineup v3 (`mu=0.63` default, BLR panel at its own budget) — SUPERSEDED by the v5 entry, which no run has measured yet

Budget: 20 seeds x 3000 batches for `headline`, `smooth`, `multimodal` and
`invariance`; **20 seeds x 1000 batches for `blr`**, which has its own budget
(see `DESIGN.md`). The traces were measured at `878fda9`; the figures were
redrawn after the truncation fix noted below.

Wall clock, as a record rather than an expectation: 4h38m for `blr` and 1h38m
for the other four groups. The `blr` panel did **not** scale down with its
budget — its cost is the restart proposal's acquisition descent, not the batch
count, and the `blr-sphere` arm alone took about three hours.

### P② and P① medians, batches axis

The `n_paths=1` column is the arm that takes the engine default, so in this
entry it carries `mu=0.63`; the momentum-off arm is spelled out beside it.

| problem | n_paths=1 (mu=0.63) | n_paths=16 | n_paths=1, mu=0 | Adam | L-BFGS-B |
| --- | --- | --- | --- | --- | --- |
| diag_quad_d100_cond10000 | 1.9e-9 | 1.5e-9 | 6.2e-61 | 2.0e-63 | 9.4e-8 |
| rot_quad_d100_cond1e+06 | 42.0 | 24.7 | 90.2 | 0.499 | 1.4e-4 |
| oscillator_n20_F20 | 9.2e-6 | 4.5e-8 | 3.1e-5 | 7.5e-5 | 7.3e-3 |
| ackley_d100 | 19.71 | 19.55 | 19.71 | 19.57 | — |
| rastrigin_d100 | 1615 | 1520 | 722 | 807.9 | — |

### BLR panel medians, 1000 batches (rastrigin, all arms `n_paths=16`)

| arm | median best |
| --- | --- |
| Sample draws | 1551 |
| BLR identity | 275.5 |
| BLR sphere | 738.5 |

### Annotations

**The `Sample draws` row is read off a shared arm file, and nothing was lost.**
The multimodal panel re-measured that same arm at 3000 batches in the same
session, so the file holds the longer trace. The run is prefix-deterministic —
same seeds, same steps — so the first 1000 batches of that trace *are* the
1000-batch run: its median at batch 1000 (1551) and its mean (1.5489e+03) both
reproduce the `blr` phase's own log line. The `blr` figure truncates the arm at
the panel's budget instead of re-measuring it.

**v3 is not a re-measurement of v2, and the two cannot be subtracted.** Between
the lineups `DiagonalPreconditioner`'s default `mu` moved from 0 to 0.63 (one
window, two knobs: `beta=mu=0.63`), and the `mu=0.9` arm was retired with the
projected momentum direction. Every arm above therefore ran at a setting no v2
arm had. v3 carries its own momentum contrast (`n_paths=1, mu=0`); cross-entry
differences would confound the default change with the measurement.

**The points axis now truncates in batches, at the shortest budget present.**
It used to cut every arm at `max_steps` *points* — a unit error that stopped an
`n_paths=16` arm after ~187 batches and flattened all three `blr` arms onto
their initial value, so the figure showed one line inside a narrow y band. The
cut is now at the panel's own budget in batches, and a panel whose arms
disagree prints which budget it used. This is a figure change only: no trace
was re-measured, and the same-budget panels are unaffected.

**Traces and figures now live in separate trees.** `experiments/results/` holds
only per-(problem, arm) JSON; the panels were moved verbatim to
`experiments/figures/<panel>/`.

## 2026-11-02 — lineup v2 (momentum arm, BLR restart panel) — SUPERSEDED by the v3 entry

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
to identity on 20/20 (251-311): the gain is the center bias *plus* the right
projection, not fitting on its own.

Consistency check: the Sample arm's median equals the multimodal panel's
`n_paths=16` median to the last digit (376.161 both). Same runner, same seeds -
which is what makes the two panels comparable in the first place.

### What this supersedes

The 2026-10-28 entry below, whose restart on/off lineup the momentum and BLR
panels replaced. Its Ackley and unit-rescaling readings are carried here
unchanged, because they describe behavior the v2 panels do not touch.

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

Measured again under v3 (3000 batches, 20 seeds, medians): `n_paths=16` 19.55,
Adam 19.57, `n_paths=1` at `mu` 0 / 0.63 / 0.9 → 19.71 / 19.71 / 19.79. The two
single-chain `mu` arms are identical to the digit, which is what a frozen path
looks like: with no step taken there is no secant to fold, so **no `beta`
question can be asked on this problem** either. No cell of the multimodal panel
discriminates on ackley; the rastrigin cell is what carries that panel.

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
