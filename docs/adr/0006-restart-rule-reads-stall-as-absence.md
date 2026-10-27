# ADR-0006: the restart rule reads "stalled" as absence, not as a non-finite value

- Status: **Proposed** (plan delivered for review; no code written)
- Deciders: user (direction ratified), assistant (plan)
- Date: 2026-02-11

## Context

The restart verdict compares two scott_mad KDEs: the path's recent one-step
Δs (continue side) against the shared group-min pool (restart side). The class
docstring already states the load-bearing invariant: *"what makes the
comparison readable is that the two are worth about the same number of
samples."*

That invariant is violated today, and the violation has one root cause:
**a float stream is used for three different things.**

1. A finite Δ is a *value* (evidence: what one continuing step bought).
2. `inf` folded for a stalled path is *absence* (no step happened), but it is
   spelled with the same `inf` that a genuinely diverging objective returns.
3. A `+inf`/`nan`/`-inf` group-min is a *value with a defined or undefined
   EI contribution*, but it is conflated with absence too.

The consequence is the machinery that recovers the distinctions afterwards:
`pool_size` (finite-only count), `_stage_deltas` (a filter), the
inf-folds-as-clock-ticks, and the cold-start/dead-pool split riding on
`group_mins` emptiness. And the actual bug: a stalled path reaches the verdict
at the M-fold cadence while reading **fewer than M** continue samples (its
finite Δs minus its stalls), so `_continue_payoff` and `_restart_payoff` are
not built on matched counts for exactly the paths whose verdict matters.

## Decision

### The three ratified directions

1. **Stalled paths leave the EI cadence.** A stalled path's continue payoff is
   0 *by definition* (it is not moving, so continuing buys nothing). Its
   verdict is one-sided: `EI_restart > 0`. A moving path keeps the full
   matched-M comparison. Deferral to a unified/phase-aligned restart is a
   *scheduling* concern, kept, but no longer smuggled through false samples.

2. **`+inf` group-mins become 0-buy events.** The restart side is read as two
   events: the finite subset (a scott_mad KDE, bandwidth from the finite
   subset only) and the `+inf` subset (each contributing exactly 0 — a dead
   start buys nothing). The total is the finite KDE's mean multiplied by
   `|finite| / (|finite| + |+inf|)`, a constant in `(0, 1]` whenever dead
   starts exist. This matches the ratified spec: "finite KDE × a constant < 1".

3. **The continue seam keeps a conservative guard.** `observe_step` raises
   `ValueError` on a non-finite float (the only legal non-`None` continuation
   sample is finite); the restart seam accepts non-finite group-mins (they are
   legal values).

### `nan` and `-inf` — audited, not changed in this increment

- `nan` (objective `f` only, per the reminder): **already adequate.** A `nan`
  draw never wins a probe pick (`_argmin_finite` skips it); a `nan` trial never
  commits (`isfinite` gate) and routes to stall; a `nan` group-min is stored,
  excluded from the window and from the read, and does not trigger cold start
  (`group_mins` non-empty). The plan pins this with one explicit test; no
  behavior change.
- `-inf`: **deferred to a follow-up.** A `-inf` group-min reaches the pool only
  when an entire probe group is non-finite with `-inf` as draw 0. Two distinct
  semantics are wanted and neither is settled:
  - *feasible* `-inf` should become the reported optimum — but the incumbent
    (`_Incumbent.observe`, `_argmin_finite`) and the commit gate are all
    finite-only, so a `-inf` point is invisible today;
  - *infeasible* `-inf` is a raw-domain signal: finding the feasible optimum
    then needs restarting/exploring, which argues for `EI_restart = +inf` on a
    `-inf` pool entry (restart buys the bottom for free).
  Both touch the incumbent's finite-only rule and the verdict's finite
  comparison, are unreachable in the current experiment problems, and deserve
  their own decision — they are **out of scope** here.

### Stall via a typed channel, not a fold value

The engine folds `None` for a stalled path (absence), a finite float for a
moved path (evidence). The rule keeps two per-path structures: `deltas` (finite
Δ only) and `stall_rounds` (how many iterations the path has been stalled).
`None` clears the stage and ticks `stall_rounds`; a finite Δ appends and zeros
`stall_rounds`; `restart` clears both.

The gate becomes:

- cold start (`not group_mins`) → restart (unchanged);
- `stall_rounds[i] >= _stage_window()` → due, one-sided `EI_restart > 0`
  (continue side is empty by construction, so `_continue_payoff`'s 0.0 branch
  is exactly this);
- `len(deltas[i]) >= _stage_window()` → due, full `EI_restart > EI_continue`,
  close the stage (unchanged).

This is *behavior-preserving* on an end-to-end fully-stalled flock
(`test_stalled_paths_wait_for_ei_then_restart`, `test_nonfinite_start_waits_...`)
because today's inf-tick used the same window, and `T = _stage_window()`
reproduces that timing. The one deviation is a *mid-stage* stall (a path moved
`< M` folds, then stalled): today its `M - k` remaining inf-ticks reuse its
finite folds; under the new rule the stall starts a fresh clock. That is
arguably more correct — a stall means `continue = 0` regardless of earlier
folds — but it is a timing change and depends on the deferral knob below.

## Consequences

- The "matched counts" lie is gone: a stalled path never reads a short continue
  side; it reads the honest degenerate 0.
- `pool_size`, `_stage_deltas`, and the inf-as-tick all disappear.
- `+inf` pool entries now *dilute* `EI_restart` (correct dead-start evidence)
  instead of being silently dropped. This is the `× (|finite|/N)` factor.
- Changing stall timing and the `+inf` factor changes restart counts and
  trajectories: the 20-seed canonical panel and the oscillator/ackley
  diagnostics are **stale** and must be re-run once the code lands.

## Implementation plan

### `soar_lite/engine.py`

- `RestartRule.observe_step`: signature `(self, i: int, delta: float | None)`;
  docstring: `None` = "the path stalled this iteration, no Δ to fold"; a finite
  float is the one-step Δ; the rule ticks its stall clock on `None`, never spans
  a stall with a value.
- `_Paths.step` exit loop: fold `None` when `self.stalled[i]`, else the finite
  `points[i].f - prev_f[i]` (and update `prev_f` only on the move).
- Docstring/comment sweep: `_Paths` class "stalled" bullet, `continuers`,
  `n_path_schedule`'s "A stalled iteration feeds inf…" paragraph, the
  `NPathStatus` "keeps folding inf samples" comment — all reworded to
  "ticks the path's stall clock".

### `soar_lite/restart.py`

- Fields: drop `pool_size`; add `stall_rounds: dict[int, int]`.
- `observe_step(i, delta: float | None)`:
  - `None` → `self.deltas[i] = []`; `stall_rounds[i] += 1`.
  - finite → append; `stall_rounds[i] = 0`.
  - other non-finite float → `raise ValueError` (guard, decision 3).
- `restart(i)` → clear `deltas[i]` and `stall_rounds[i]`.
- `should_restart(i, f_k, incumbent)` → the three-branch gate above; both due
  branches call `self._stage_window()` (so `ForcedWindow` keeps forcing both).
- `_stage_window()` → `max(1, |finite ∪ {+inf}|)` over `group_mins` (scan; the
  pool is small), excluding `nan` and `-inf`.
- `_restart_payoff(incumbent)` →
  `finite = [v finite]`, `n_inf = count(+inf)`, `n = len(finite) + n_inf`;
  `n == 0 → 0.0`; `h = _scott_mad_bandwidth(finite)` (or 0 when finite empty);
  `sum(_gaussian_ei(h, B, m) for m in finite) / n` — the `+inf` entries are the
  implicit 0 terms.
- Delete `_stage_deltas`; `_continue_payoff` reads `self.deltas.get(i, [])`
  directly (finite-only by construction) and keeps its empty→0.0 branch.
- Class/module docstrings: rewrite the "NaN/inf Δ kept but never read, still
  count towards the window" text to the three-way scheme.

### `tests/test_solver_seam.py`

- Fake rule `observe_step(self, i, delta: float | None)`.

### Test flips

*Flip (semantics changed):*
- `test_restart.py::test_ei_restart_empty_single_and_nonfinite_skipped` — the
  `+inf` cases now dilute instead of skip: `[inf, 0.0, 2.0]` reads
  `KDE_LITERAL × 2/3`; `[inf, inf, 2.0]` and `[inf, -inf]` stay `0.0`. Rename
  to reflect "`+inf` is a 0-buy event, `nan`/-inf skipped".
- `test_restart.py::test_window_is_the_pool_size` — `pool_size` is gone; assert
  via `_stage_window()`, and the `+inf` sample now counts (window becomes
  finite+inf).
- `test_restart.py::test_an_empty_pool_is_the_cold_start_and_a_dead_pool_is_not`
  — drop the `pool_size == 0` line; the cold/dead distinction still rides on
  `not group_mins`.
- `test_restart.py::test_nonfinite_delta_counts_towards_the_window_but_is_not_read`
  — replace with a `None`-based test: `observe_step(0, None)` leaves
  `deltas` empty, ticks `stall_rounds`, gate fires at `stall_rounds >= window`,
  `_continue_payoff == 0.0`.

*Stay green (timing reproduced by `T = _stage_window()`):*
- `test_scheduler.py::test_stalled_paths_wait_for_ei_then_restart` (docstring
  wording only: "one stall tick" instead of "one inf sample").
- `test_scheduler.py::test_nonfinite_start_waits_for_ei_instead_of_stepping`.
- `test_scheduler.py::test_probe_inf_group_min_does_not_silence_restarts`.
- The KDE-literal, one-ruler, and wiring tests that touch neither non-finite
  samples nor `pool_size`.

*New:*
- one test pinning `nan` group-min = no information (excluded from window and
  read, does not cold-start, `_argmin_finite` never picks it).
- one test pinning the `+inf` dilution factor exactly.

### Commit order

One coherent unit (not a structural refactor — it changes verdict semantics, so
the "zero behavior change" rule does not apply):

    refactor: price a stall as absence and +inf group-mins as dead starts

Keyword notes for the message: stalled paths get an honest zero continue side
(no more `inf`-as-tick), `+inf` pool entries dilute `EI_restart` by the
dead-start fraction instead of being dropped, `nan` stays no-information,
`-inf` is deferred.

## Out of scope / follow-ups

- The stall-deferral knob `T` is hardwired to `_stage_window()` (reproduces the
  current cadence). `T = 1` (immediate restart) is the user's "一旦 stalled 就
  可以重启" taken literally and is a one-line change, but it raises probe
  frequency and shifts restart timing — a later tuning decision.
- `-inf` semantics (feasible `-inf` → incumbent; infeasible `-inf` → raw-domain
  restart incentive): its own decision/ADR.
- 20-seed canonical panel + oscillator/ackley diagnostics re-run after the code
  lands (the current `results_*.png` predate this change and the KDE change).

## Risks

- Mid-stage stall timing shifts (see Decision), changing some traces vs. today.
- The `+inf` dilution lowers `EI_restart` on any run with dead starts; on the
  current problems (no `+inf`) it is a no-op there, so the 20-seed re-run only
  reveals the stall-clock effect, not the dilution, unless a `+inf` fixture is
  exercised.
## Amendment: one stage clock, not two

The decision above kept **two** per-path counters — `deltas` (folds) and
`stall_rounds` (stalled iterations) — and gave each its own due test. That was
wrong on one point: it made the two counters independent, so a *mid-stage*
stall started a fresh wait. A path that moved `k < M` steps and then stalled
waited another full `M` rounds, falling behind its live cohort by exactly the
iterations it had already served.

That contradicts the property the whole design rests on: the engine reports
exactly one sample per path per iteration, so the verdict's cadence is a
**lockstep**, not a per-shape schedule. A stall is one of those reports, so it
must advance the same clock a Δ would.

`stall_rounds` is therefore replaced by `rounds` — iterations since the path's
last verdict — and only that counter gates the verdict:

- every `observe_step` advances `rounds[i]` by one, whether it carried a
  finite Δ or `None`;
- a stalled stage still drops `deltas[i]` (continue = 0 by definition) and adds
  no sample, so the read is one-sided exactly as above;
- `rounds[i] >= _stage_window()` is the only due test; `deltas[i]` non-empty
  selects the full comparison, empty selects the one-sided one;
- the full comparison closes the stage and zeroes the clock (a one-sided read
  closes nothing — a stalled path has no stage and must stay eligible);
- `restart(i)` zeroes both.

Consequences:

- **Mid-stage stall no longer shifts.** It comes due on the same round its
  cohort does; only its evidence changes (one-sided instead of a comparison).
  This removes the "one deviation" the Decision section declared, and the
  `-inf` / `+inf` / `nan` triage is untouched.
- The continue side of a mid-stage stall is still 0: the drops above mean the
  folds collected before the stall are discarded, so the verdict never averages
  a trend across the stall. The `inf`-tick rule this ADR replaces would have
  reused those folds (filtered); the absence rule does not, by design.
- A fully stalled flock keeps asking the same question each round (the clock is
  only zeroed by a restart), which is the idle-to-horizon case, not a probe
  loop — unchanged.

Field and wording changes: `stall_rounds` → `rounds`; "stall clock" → "stage
clock" wherever it names the per-path counter (the reset point, the lockstep
argument). Nothing else in the decision moved.

## Amendment: a verdict closes its stage, so `restart` reports and nothing more

The amendment above still let the two shapes of stage end differently: a full
comparison zeroed the clock, a one-sided read did not ("a stalled path has no
stage and must stay eligible"). That split was wrong on its own terms. If a
False verdict leaves `rounds` in the due band, the `rounds < _stage_window()`
gate is permanently open for that path: it enters the due branch every round,
re-runs the same comparison, and `rounds` only ever grows. The counter stops
being "iterations since the last verdict" - the only reading a due test can be
built on - and becomes "iterations since the last restart". A window is one
stage; the stage ends when it is read, and the answer is not part of the window.

So the two shapes collapse into one tail: due means compare, then close. The
one-sided case was never a separate branch, either - `_continue_payoff` returns
`0.0` on the empty stage a stall leaves, so `EI_restart > 0` is the general
`EI_restart > EI_continue` at `EI_continue = 0`, and a single expression now
serves both.

`restart(i)` consequently has nothing left to drop: the stage it would have
closed was already closed at its verdict, on every path, True or False. It stays
on the protocol as the engine's restart *event* - a rule that resets per-path
state at the event rather than at the verdict does it there - and this rule's
implementation is a no-op with the reason written down.

Consequences:

- The "fully stalled flock keeps asking the same question each round" bullet
  above is superseded: a dead landscape is now re-judged once per window. It
  still idles to the horizon rather than probing, and the verdict is unchanged
  while the pool does not move - only the re-ask cadence is one window, not one
  round.
- A stalled path's wake-up can now lag a pool change by up to one window instead
  of one round. That is the same cadence a continuing path has after a False
  verdict (it also waits a fresh window), so it is lockstep, not a regression -
  the old one-round re-ask was the asymmetry, and it was bought by leaving the
  clock uncleared.
- `test_verdict_is_due_once_per_window_and_closes_the_stage` and
  `test_an_empty_pool_is_the_cold_start_and_a_dead_pool_does_not_restart` pin
  the new cadence; `test_restart_keeps_the_shared_group_min_pool` pins that the
  event touches no state.

## Amendment: the probe round reports like any round, so `restart` loses the clock

The amendment above declared `restart` an event-only no-op. Phase-locking then
added the probe round back as a *counted* round, and it landed the tick in
`restart` (`rounds[i] = 1`): the engine reported a round only to the paths that
folded or stalled, a re-probed path did neither, so the rule counted the round
itself, and the stage became `M + 1` rounds (the newest-M slice absorbs the extra
fold). That gave the one invariant two channels - `observe_step` for folds
and stalls, `restart` for the probe round - and it left the docs lying: the
amendment above says `restart` is a no-op, but a rule that honoured it fell one
round late per restart, cumulatively.

This change reunites the round on the one report channel. The engine now reports
every path every iteration through `observe_step`: a fold arrives as its
finite Δ, and a round that took no step arrives as `None` - a stall, a
re-probed path's probe round, or the initialization itself. `restart` returns to
the event-only no-op the amendment above described, but now the round is counted
in `observe_step`, not assumed away. The probe's f stays the Δ anchor
(`_place` sets `prev_f`), so the first Δ after a probe is the step that follows
it - never `f - f = 0`.

Consequences:

- `observe_step(i, None)` now means "no step this round" (a stall, or a
  re-probe's probe round); the `RestartRule` docstring and
  `EIRestartRule.observe_step` say so. `EIRestartRule.restart` is an
  event-only no-op again.
- `_stage_window()`'s `+ 1` (the probe or verdict round a stage opens in) and
  the newest-M slice of `_continue_payoff` are unchanged: they were never the
  patch, only the window derivation.
- `test_a_probes_iteration_is_not_read_back_as_a_zero_step` now pins that the
  probe round reports `None` (not a zero Δ); the phase-locking test hand-enacts
  `observe_step(i, None)`; `test_restart_keeps_the_shared_group_min_pool`
  pins that the event leaves `rounds` untouched.
