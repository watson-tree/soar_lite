# Measurement record: `compare_baselines`

Every number these panels have produced lives here, in reverse chronological
order, each entry tagged with the lineup and the budget that produced it. The
one exception is the pre-2026-10 probe sweeps, which measured a knob set the
code no longer carries; those are not measurements of a lineup and live with the
other retired experiments in `DEPRECATED.md`.
`DESIGN.md` says what the panels *are* and what they are meant to show; it
states no measured numbers, so nothing there can age in place.

Three rules:

- Every entry names its **lineup**, its **budget** (seeds x batches) and its
  **status** — `CURRENT`, `SUPERSEDED by <entry>`, or `NOT REPRODUCIBLE`.
- **Numbers are never re-edited.** A new measurement is a new entry plus a
  status change on the old one, and the old numbers stay exactly as measured.
  Quietly updating a number is how a document ends up quoting a run it no
  longer describes. A claim that has gone false gets an added note, not a
  rewrite.
- Only measurements belong here. Propositions, rules and the budget contract
  live in `DESIGN.md`.

Unless a table says otherwise it holds **medians**, the statistic the figures
plot. The harness's stdout line is `final mean best` — mean, not median, and
not comparable with the tables below.

Two entries (v2 and v3) also record wall clock. That is a property of the
machine that ran them, not an expectation: no comparison here is made on time,
and a re-run should reproduce the medians, not the hours.

## 2026-11-11 — what the rastrigin panel's ordering is made of — CURRENT

A one-off probe, not a lineup. The problem object, the seed list and the start
call are the harness's own — a re-derived `x0` reproduced each arm's stored
first evaluation on 20/20 seeds — but the driver is not kept, because
`experiments/` holds panels and not probes. So the procedure below is the
record, not a command, and reproducing it means re-writing those few lines
against `harness/problems.rastrigin` rather than running a script. Nothing in it
is stale: it measures the arms and the problem this entry's lineup still has,
which is why it lives here and not in `DEPRECATED.md`.

Lineup read: the v6 rastrigin arms (`n_paths=1, mu=0`; `n_paths=16, mu=0`;
Adam `lr=0.5`) at 20 seeds x 3000 batches. The baselines carry their own
budgets, named per row.

| baseline | median |
| --- | --- |
| uniform random, best of 3000 evaluations, 200 independent trials | 1501.4 |
| `round(x0)` — the nearest lattice well of the start, 20 seeds | 876.0 |
| soar-lite `n_paths=1, mu=0` final | 896.0 |
| Adam `lr=0.5` final | 807.9 |
| wide `n_paths=16, mu=0` final (its best-of-16 `round(x0)` floor: 746.0) | 745.2 |
| coordinate descent, f values only, from `round(x0)` | 0.0000 at 1201 evaluations |

Random's interquartile range over those 200 trials is 1482.6–1521.1, so it does
not reach the slowest measured arm: uniform points barely roll into a well at
all, and "a big step is a random walk" is not what the panel is showing.

**Soar's descent walks to a free rounding of its own start.** Per seed, the gap
from `f(x0)` down to `f(round(x0))` is closed to a median of 98.3% (range
89.5%–114.7%); per-seed `final − round(x0)²` has median +17.2 and max
|Δ| 139.7, with 7/20 seeds below that floor and 13/20 above it. So the descent
reaches the nearest well mostly, occasionally crosses one ridge further, and
otherwise stops a little short of the well. The nearest-well reading is an
approximation in distribution, not an identity per seed.

**Adam is below that floor on every seed.** `Adam − round(x0)` is negative on
20/20, median −87.1 — the panel's one systematic direction, and what the
mechanism split predicts: the secant damps at a sign-reversing gradient and the
RMS step does not, so Adam's coordinates can cross a ridge that stops
soar-lite.

**The wide arm's lead is its sixteen starts.** The floor `min` over the 16
starts of `round(x0)²` is 746.0 median and the arm's final is 745.2: per-seed
median +13.2, 7/20 below the floor. Width buys draw count here, not a better
resolution of the well a single chain lands in.

**A coordinate descent solves the panel within 40% of the budget.** From
`round(x0)`, probing each coordinate at `xᵢ ± 1` and keeping the lower value,
repeated until a pass moves nothing, reaches exactly 0.0000 on 3/3 seeds in 6
passes and 1201 evaluations of the 3000 available. The landscape is separable,
so axis-at-a-time search alone resolves it — and every arm's score sits within
reach of rounding plus, for Adam, a few one-coordinate crossings. That is what
makes this a boundary panel (`DESIGN.md`, panel 3): the ordering tracks start
count and ridge crossing, not the per-coordinate scaling of coupled curvature
that is the library's subject.

Probe design, since it is not kept: `x0` from
`jax.random.uniform(split(PRNGKey(seed))[1], (n_paths, d), ±5.12)`, seeds
0..19, validated against the stored first evaluations; the random baseline is
3000 uniform draws per trial over the same box, best value, 200 trials; the
round baseline is the value at `round(x0)` (a grid point, so `Σkᵢ²`); the
descent starts at `round(x0)` and counts every probe as one evaluation.

## 2026-11-11 — invariance panel: the rotated quadratic replaces the diag control — CURRENT

20 seeds x 3000 batches, base vs `x' = 10x`, zero retune, one panel re-run at
`harness/appendix.py` with the control problem changed from the diagonal
quadratic to the rotated one. The panel now prints the gap it draws: the maximum
over the run of |log10 base - log10 x10| on the shared-grid cohort median is
**0.044 decades for soar-lite** (`n_paths=1`, `mu=0`) and **1.863 decades for
Adam**, which keeps one `lr` across both unit systems. Figure:
`experiments/figures/invariance/invariance.png`, copied to
`docs/figures/invariance-units.png`.

**Why the control moved.** On the diagonal quadratic the two unit systems take a
different number of steps to the floor: the serial arm ends base in 3 points and
the rescaled system in 4, so at point 3 base reads `5.05e-28` while the rescaled
system is still at `0.681` — 27 orders apart, in the only region where there is
anything to see. The cause is the warm-up pair, which is asymmetric between the
unit systems: the unit-stiffness factor `1 - aᵢ` sends the soft coordinates to
zero while its scaled counterpart `1 - aᵢ/100` leaves them nearly in place, and
both clip the stiff ones. The curves then meet only at the floor, which is the
trivial half of the claim, and the figure argued against its own title.

The rotated quadratic descends across the whole budget and its warm-up step
overshoots an ill-conditioned rotated landscape enough to be worse than the
start, so best-so-far never records it (no seed in either unit system improved
on its start at the warm-up point, 20/20). What the figure draws is therefore
steady-state covariance alone: soar-lite's two curves hug, Adam's split.

Read against the 2026-10-28 entry's "unit-rescaling residual" section with
care: that is a real measurement of the *retired* diag control, at the warm-up
that still carried an Armijo backtrack, and it no longer describes what this
panel draws. The end-to-end claim is still asserted statistically, and this
panel writes no trace JSON: the two gaps above are printed by the run, not
re-derivable from a shipped artifact.


No trace was re-measured; every number in this file stands. What changed is how
far each curve is drawn.

**The tilt.** A seed's own best-so-far never rises - checked over all 480 traces,
zero rising steps. The curve did: `_resample` leaves a seed's column empty after
its run stops, and `nanmedian` took the median over whoever was left. So a seed
that stopped early *left the sample*, and when the seeds that stall first are
also the good ones, the median over the survivors steps up. On this measurement
that is exactly the case: `pk_chain`'s serial arm stops on 7 of 20 seeds between
batches 367 and 527, and those seeds end at 1.05e-7 and 5.01e-7 where the
full-budget seeds end at 1.30e-3. Twenty of the 48 drawn curves carried a rising
tail - both axes of 10 arms - worst 2.54x on `pk_chain`'s L-BFGS-B at batch 434,
1.32x on that soar arm at batch 373, 1.30x on the diagonal quadratic's L-BFGS-B,
1.17x on the oscillator's `mu=0.9` at batch 2649, and 1.01x-1.05x on rastrigin.

**The rule.** A column is drawn only where every seed of the cohort is still
running, so each curve now ends at its cohort's first stop and the band is the
same twenty seeds throughout. Re-drawn, all 48 curves are monotone
non-increasing. The price is the tail itself, and it is where the runs had
already gone flat: `pk_chain` serial stops drawing at batch 367 (its median moves
2.2e-5 to 2.13e-5 over the rest), `mu=0.9` at 760, the oscillator's `mu=0.9` at
1973, rastrigin at 54 / 69 / 253, the diagonal quadratic at 3 / 264 (458 for
L-BFGS-B), `pk_chain`'s L-BFGS-B at 76 and the oscillator's at 124. The rotated
quadratic serves its whole budget on every arm and is untouched, as is every
curve on a panel whose seeds all stop together or not at all.

**What did not change.** The axis still runs to the panel's budget (`_cut`), so a
short curve is visible next to the arms that spent theirs, and this file's final
values stay the median of each seed's own endpoint - a different and stronger
quantity than any common-batch median, which is why the tables never carried
this artefact.

Covered by `experiments/tests/test_figure_cut.py`, whose
`test_the_median_is_taken_over_the_whole_cohort` pins it: a cohort of three
seeds whose good one stops first draws to that stop and never rises.

## 2026-11-11 — lineup v6, measured — CURRENT

Budget 20 seeds x 3000 batches, panels `headline,smooth,multimodal,invariance`,
at the code of this commit, on CPU. The 2026-11-08 entry below says what the
lineup is; this one holds what it measured. Every file is new: a v5 key reads
`...-mu0.63-restart-on`, a v6 key `...-mu0-restart-off`, so nothing below was
carried over from an earlier entry.

Wall clock ~56 min for the three roster panels plus the invariance re-run — this
machine and this roster, which no longer carries the BLR panel; not an
expectation.

### The two headline problems

| arm | median | Q1 | Q3 | worst seed | stopped early (seeds / at batches) |
| --- | --- | --- | --- | --- | --- |
| soar `n_paths=16`, mu=0 | 3.13e-7 | 5.56e-8 | 8.92e-7 | 2.44e-6 | all 3000 |
| soar `n_paths=1`, mu=0.9 | 1.61e-5 | 4.04e-6 | 6.83e-5 | 5.36e-3 | 3/20 at 1973–2649 |
| soar `n_paths=1`, mu=0 | 3.12e-5 | 8.61e-6 | 1.95e-4 | 1.49e-3 | all 3000 |
| Adam `lr=2` | 7.52e-5 | 2.09e-5 | 7.06e-4 | 1.03e-2 | all 3000 |
| L-BFGS-B | 7.32e-3 | 1.24e-3 | 2.34e-2 | 7.00e-2 | 20/20 at 124–1088 |

`oscillator_n20_F20`, `headline`.

| arm | median | Q1 | Q3 | worst seed | stopped early (seeds / at batches) |
| --- | --- | --- | --- | --- | --- |
| soar `n_paths=16`, mu=0 | 3.56e-7 | 2.31e-7 | 4.81e-7 | 1.58e-6 | all 3000 |
| soar `n_paths=1`, mu=0 | 2.13e-5 | 3.79e-6 | 7.22e-5 | 1.30e-3 | 7/20 at 367–527 |
| Adam `lr=0.1` | 2.51e-4 | 1.10e-4 | 7.34e-4 | 2.28e-3 | all 3000 |
| soar `n_paths=1`, mu=0.9 | 3.14e-4 | 3.45e-5 | 3.48e-4 | 2.13 | 3/20 at 760–978 |
| L-BFGS-B | 5.06e-3 | 3.46e-5 | 5.74e-1 | 6.17 | 20/20 at 76–1193 |

`pk_chain_n21_d61`, `headline`.

Median best-so-far along the budget (batches 100 / 300 / 1000 / 3000):

| problem | arm | 100 | 300 | 1000 | 3000 |
| --- | --- | --- | --- | --- | --- |
| oscillator | soar `n_paths=16` | 3.67e-4 | 1.95e-5 | 7.55e-7 | 3.13e-7 |
| oscillator | soar `n_paths=1`, mu=0 | 7.66e-3 | 7.97e-4 | 6.21e-5 | 3.12e-5 |
| oscillator | Adam `lr=2` | 4.01e-2 | 1.84e-2 | 1.04e-3 | 7.52e-5 |
| pk_chain | soar `n_paths=16` | 2.12e-4 | 5.65e-7 | 3.56e-7 | 3.56e-7 |
| pk_chain | soar `n_paths=1`, mu=0 | 5.46e-3 | 2.18e-5 | 2.15e-5 | 2.13e-5 |
| pk_chain | Adam `lr=0.1` | 2.47 | 9.02e-1 | 2.30e-2 | 2.51e-4 |

### The smooth panel

| arm | median | Q1 | Q3 | worst seed | stopped early (seeds / at batches) |
| --- | --- | --- | --- | --- | --- |
| Adam `lr=0.1` | 1.05e-51 | 1.34e-52 | 2.41e-51 | 5.30e-51 | all 3000 |
| soar `n_paths=16`, mu=0 | 2.24e-28 | 1.80e-28 | 2.53e-28 | 3.36e-28 | 20/20 at 3 |
| soar `n_paths=1`, mu=0 | 4.36e-28 | 3.29e-28 | 5.75e-28 | 9.75e-28 | 20/20 at 3 |
| soar `n_paths=1`, mu=0.9 | 1.51e-8 | 1.23e-8 | 1.94e-8 | 2.43e-8 | 20/20 at 264–273 |
| L-BFGS-B | 9.41e-8 | 7.57e-8 | 1.11e-7 | 3.86e-7 | 20/20 at 458–508 |

`diag_quad_d100_cond10000`, `smooth`.

| arm | median | Q1 | Q3 | worst seed | stopped early (seeds / at batches) |
| --- | --- | --- | --- | --- | --- |
| L-BFGS-B | 1.38e-4 | 9.49e-5 | 1.92e-4 | 6.00e-4 | all 3000 |
| Adam `lr=2` | 4.99e-1 | 1.87e-1 | 1.24 | 4.44 | all 3000 |
| soar `n_paths=1`, mu=0.9 | 25.1 | 21.3 | 29.9 | 36.5 | all 3000 |
| soar `n_paths=16`, mu=0 | 50.2 | 47.7 | 56.9 | 67.3 | all 3000 |
| soar `n_paths=1`, mu=0 | 90.3 | 67.6 | 110.7 | 151.2 | all 3000 |

`rot_quad_d100_cond1e+06`, `smooth`.

### The multimodal panel

| arm | median | Q1 | Q3 | worst seed | stopped early (seeds / at batches) |
| --- | --- | --- | --- | --- | --- |
| soar `n_paths=16`, mu=0 | 745.2 | 706.9 | 784.8 | 839.7 | 20/20 at 69–89 |
| Adam `lr=0.5` | 807.9 | 748.0 | 838.7 | 937.2 | all 3000 |
| soar `n_paths=1`, mu=0 | 896.0 | 857.9 | 955.7 | 1060.0 | 20/20 at 54–81 |
| soar `n_paths=1`, mu=0.9 | 1849.0 | 1815.0 | 1932.0 | 2021.0 | 20/20 at 253–259 |

`rastrigin_d100`, `multimodal`, against `f* = 0`.

### Annotations

**The stop rule is visible in every trace, and it settles v6's open question the
other way.** With `restart_stalled` off a fully stalled flock ends the run, so
`stopped early` is where the knob costs budget: every seed of the diagonal
quadratic (batch 3) and of rastrigin (batches 54–259), seven of the compartment
chain's twenty serial seeds (from batch 367) and three of the oscillator's
twenty `mu=0.9` seeds, and never the rotated quadratic or any Adam run. v6 left
`pk_chain` unchecked on this point; it is checked now, and a stall does fire
there. The seeds it takes are the *good* ones - they end at 1.05e-7, 5.01e-7
- where the seeds that spend the whole budget end at 1.30e-3 - so symmetric
stopping costs those seeds the rest of their run even though it leaves the arm's
median where it is.

**v6's rastrigin expectation does not hold.** That entry predicts a stall at
~batch 700; measured, it fires on all 20 seeds between 54 and 81 (serial), 69
and 89 (wide) and 253 and 259 (`mu=0.9`), two orders of magnitude earlier. The
diagonal quadratic's first-batches stop is as v6 describes, and the rotated
quadratic is the one measured problem where nothing stalls.

**The multimodal ordering is set inside the first hundred batches.** The
starting median is 1849 (1680 for the wide arm, whose sixteen starts include a
better one). Adam is at 970 by batch 10, 827 by 30, 809 by 60 and 807.9 from
batch 100 to 3000; the wide arm falls 865.6 -> 745.2 between batches 30 and 60
and is flat afterwards; the serial arm reaches 896 by batch 60 and stops there.
So the panel's spread is a *level* set early, not a rate sustained along the
budget, and half of it is a stopping rule: Adam spends 2900 flat batches after
its best is in, because it has no convergence test here.

**`mu=0.9` never improves on the starting point on rastrigin.** On all 20 seeds
the final value equals that seed's own start, exactly, after 254 batches of
work: median final 1849.0 is the median start, Q1 1815, Q3 1932. On the same
panel the wide arm's best-of-16 start is why it leads (745.2), and its 74
batches buy it 745.2 against the serial arm's 896.0, not a longer descent.

**The momentum trade carries both signs, on every panel that can read it.**
Against the default `mu=0` at `n_paths=1`, the 0.9 arm is better on the
oscillator (1.61e-5 vs 3.12e-5) and on the rotated quadratic (25.1 vs 90.3, 3.6x,
the coupling it buys back), worse on the compartment chain (3.14e-4 vs 2.13e-5,
15x) and worse on rastrigin. This is the same sign pattern the v3/v4 entries
recorded at `mu=0.63`; what the v6 lineup adds is the default arm, so the trade
is now read against momentum off rather than against another momentum level.

**On the headline panel the gap is a rate gap, and Adam is still falling when
the budget ends.** The serial arm ends 2.4x under Adam on the oscillator
(3.12e-5 vs 7.52e-5) and 12x under it on the compartment chain (2.13e-5 vs
2.51e-4); the wide arm ends 240x and 706x under Adam (3.13e-7, 3.56e-7), and
100x / 60x under the serial arm. Adam is still descending steeply on the
compartment chain at the end (2.30e-2 at batch 1000 -> 2.51e-4 at 3000), so this
is a budget-limited rate gap, not a floor Adam cannot reach. The wide arm's
headline curves are flat from batch 1000 on.

**The diagonal quadratic is the panel where equal batches favour the baseline.**
Soar-lite stops at batch 3 holding 4.36e-28 (serial) and 2.24e-28 (wide), while
Adam spends all 3000 and ends at 1.05e-51 — and is at 1.90e-10 by batch 300 and
2.73e-42 by batch 1000, so it crosses the chain's final value somewhere between
the two, against three batches. That is the stopping rule and the figure's
floor, not the algorithm: the curves are flat from batch 3 (soar) and 475
(L-BFGS-B) and the only arm still moving at 3000 is Adam.

**L-BFGS-B keeps its two roles.** It is the smooth panel's ceiling on both
problems (1.38e-4 on the rotated quadratic, 9.41e-8 on the diagonal, best in
each column) and it is not a ceiling on either real problem: it is the worst arm
on the oscillator (7.32e-3, 9 of 20 seeds above 1e-2) and bimodal on the
compartment chain (Q1 3.46e-5, better than Adam's median; median 5.06e-3; Q3
0.574; 9 of 20 seeds above 1e-2, worst 6.17). The v4 reading — a full-memory
serial line search with no per-coordinate scaling lands in a bad basin about
half the time — reproduces on the same problem.

## 2026-11-08 — lineup v6 (`restart_stalled` off, restart-model panel retired) — CURRENT, measured in the 2026-11-11 entry

Two changes to what the harness runs. Nothing has been re-measured against
either.

**Every arm now runs with `restart_stalled` off.** Soar-lite's arm keys gain
`-restart-off` where they read `-restart-on`, and a converged chain stops the
way Adam and L-BFGS-B stop instead of re-drawing inside the same budget. Where
a stall fires — the diagonal quadratic (~step 4) and rastrigin (~batch 700) —
a v1–v5 trace is a `restart-on` record and no longer describes what the harness
does. Where it never fired (the rotated quadratic and the oscillator) the
run is the same either way; `pk_chain` has not been checked on that point.

**The restart-model panel is retired**: `blr-identity`, `blr-sphere` and the
`n_paths=16` rastrigin arm they were read against are out of the harness,
`--groups blr` is refused, and the figure left the README. That follows from
the line above rather than from the numbers: a restart model's only mechanism
is a restart, and with restarts off for symmetry the panel has no stage left.
It also never showed the payoff the design anticipated — the bent (sphere)
projection, the one the external evidence supports, was the worse of the two
fitted models in both entries below, losing to a plain identity linear head.
Against that it ran at a budget of its own on a contrived bowl, and its
acquisition descent made it the most expensive panel per batch (the v2 entry's
3h12m of a 5h06m run). The library support stays (`BLRGlobalModel`, the
projections, the public `model=` factory), and the harness keeps that factory
wired for a problem where a fitted restart pays.

The blr readings below stay exactly as measured, as the record of a retired
panel. They are not reproducible from this harness — their arms no longer
exist, so redrawing them takes the code at their own commit.

**Ackley is retired** from the multimodal group, which is now rastrigin alone:
no cell discriminated. Every arm spent its whole budget — one point per batch
through batch 2999 — yet the best-so-far stopped improving early: the last
strict improvement lands at median batch 63 (max 81) for `n_paths=1, mu=0`,
median 343 (max 429) for Adam, and the final medians are 19.57 (Adam),
19.71 / 19.71 / 19.79 (`n_paths=1` at `mu` 0 / 0.63 / 0.9), 19.55
(`n_paths=16`), against `f* = 0`. **This corrects the v3 entry below**, which
reads the identical `mu` arms as "no step taken": the traces serve a point on
every batch, so steps were taken, and the freeze is in the best-so-far — set
where the plunge landed, never beaten afterwards. `problems.ackley` is gone,
`--groups multimodal` returns one problem, and the ackley readings below stay
as the record of a retired problem. The reasoning is in `DEPRECATED.md`.

**Note added 2026-11-11:** both changes have now been measured, and the numbers
are in the entry above. Two statements here do not survive that run: the
rastrigin stall fires at median batch 61 / 74 / 254, not at ~batch 700, and the
`pk_chain` point is settled — its chains serve the whole 3000 batches, so the
restart never fired there either.

## 2026-11-07 — lineup v5 (`mu` default back to 0) — SUPERSEDED by the v6 entry

`DiagonalPreconditioner`'s default `mu` moved from 0.63 back to 0. Every arm
that takes the default re-keys with it (`...-mu0.63-...` → `...-mu0-...`), and
the momentum-off arm is no longer a curve of its own: it *is* the default arm.

**Nothing has been re-measured against this lineup.** No table here describes
what the harness now draws. The v3 traces stay on disk and stay valid as the
record of the 0.63 default; figures drawn from them say so. The roster is now
`n_paths=1` (default `mu=0`), `n_paths=1, mu=0.9`, `n_paths=16`; the missing
measurement is `np16-mu0` on every problem plus the blr panel's three arms at
`mu=0`.

**Note added in v6:** the blr panel is retired, so the second half of that
missing measurement no longer exists. The paragraph's own reading of the
`mu=0.63` traces still stands.

Why the window moves back, in one line: memory damps the scaled spectrum and
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
cut value was `max_steps` — the flag is now `--max-batches`, same quantity — a
batch count, applied as a point count) but left the
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

**Annotation, 2026-11-10 — the cut now spares an arm that stopped early.** An
arm that stops before its batch budget (a chain that converged, on the diagonal
quadratic within a handful of batches) is drawn to its own last point and no
longer sets the shared cut; only arms that served their budget do. "All curves
end together" therefore holds when every arm completes, which is every panel
but the diagonal quadratic and rastrigin. Two consequences, both in the figure
only: an early arm's last step is now sampled into the shared grid (without it
the step that carried the run's answer was the one never drawn), and the cut it
dictated is gone. No trace was re-measured and no number above changes.

**Annotation, 2026-11-11 - the median is taken over the whole cohort.** The rule
above decides where a curve *stops*; it left the median averaging whoever was
still running, which let a curve tilt up as seeds stopped. A column is now drawn
only where every seed of the cohort is alive, so each curve ends at its first
stop. See the 2026-11-11 figure entry. Figure only: no number above changes.

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
It used to cut every arm at `max_steps` *points* (that flag is now
`--max-batches`, same quantity) — a unit error that stopped an
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

## pre-2026-10 — step-length calibration sweeps — moved

That entry measured a knob set the current code no longer has (`alpha0` damping,
the `n_alpha` grid), so it was never a measurement of a lineup. It now sits with
the other retired experiments in `DEPRECATED.md`, where the sweeps and the
Ackley problem are the two records of things this harness no longer runs.
