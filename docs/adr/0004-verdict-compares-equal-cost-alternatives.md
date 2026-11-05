# The restart verdict compares equal-cost alternatives, and every probe is billed

**Status: amended, three times.** Part 1 (the `n_probe_evals`-iteration horizon)
is superseded: the horizon clock `_Horizon` was removed and one continue sample
is now one step. Part 2's batch budget is superseded: the engine stops on
`max_iters`/`time_limit`, and batches are reported, not budgeted. Part 3 stands.
The third amendment is naming only. See "Amendment" at the end of each.

`EI_restart` is fitted on probe group-mins, and one probe costs `n_probe_evals`
served batches (one batch each). `EI_continue` was fitted on one-step Δ samples,
and one step costs one batch. The verdict was therefore comparing a purchase of
`n_probe_evals` batches against a purchase of one batch, and the bias ran one
way: the restart side's payoff had been aggregated over more exploration, so
`EI_restart` won verdicts the evidence did not support. In the same direction,
every probe event spent `n_probe_evals` batches that the engine never charged
to `max_evals` — the init probe was billed to nobody at all. Both errors
flattered restarts, which is exactly the effect the multi-path escape claim
rests on, so this is a correctness bug, not a tuning question.

The decision, in three parts:

1. **One continue sample spans `n_probe_evals` iterations.** The engine feeds
   `f(t) - f(t - n_probe_evals)` once per horizon, non-overlapping, and feeds
   `inf` for that sample if any iteration inside it stalled. Both sides of the
   verdict are then "what `n_probe_evals` batches buy".
2. **The engine bills served batches.** Every batch the seam serves costs one
   unit of `max_evals`, so the init probe and each restart event cost
   `n_probe_evals` units and the budget means batches, exactly. A run also
   attempts no more iterations than `max_evals`: an iteration serves at least
   one batch unless every path is stalled, so that cap only bites on a stalling
   run - a stalled iteration is free, and the cap is what stops a permanently
   stalled run whose rule can state no cadence. `nit` counts iterations the
   main loop attempted; the served quantities are `n_points` and `n_batches`,
   both kept in their own counters inside the engine's `_Budget`, with the
   meter charging `n_batches`.
3. **The horizon lives in the engine, not the rule.** `n_probe_evals` is a
   fact of the schedule the engine owns, so the engine feeds equal-cost samples
   and `EIRestartRule` stays knob-free: it keeps reading a plain mean and
   population sd over its samples. This also makes the existing
   `_stage_window() = M` derivation exact rather than approximate — that window
   was chosen as "the most samples a verdict can have in as many iterations as
   the restart side already has", and equal-cost samples make the two counts
   the same purchase.

Rejected: shrinking the probe to one draw (`n_probe_evals = 1`) so that it
matches a one-step continue side. That ties the init policy — a "min of two
draws" start — to the restart accounting, and throws away the better start in
order to fix a comparison.

Consequences:

- The verdict cadence slows by `n_probe_evals` (default 2): a stage of M
  samples needs M·`n_probe_evals` iterations to fill. Fewer restarts, honestly
  earned. Every multi-path-escape number measured before this change is void
  and has to be re-measured; nothing was committed as a figure, so no cleanup
  beyond rerunning.
- One accounting authority: once the engine bills probes, a caller no longer
  needs a batch gate of its own. `experiments/methods.py` drops its own
  `budget_check` gate and lets `max_evals` govern.
  `budget_check` survives only for a budget the engine cannot see (points,
  wall-clock, dollars), and such a gate can over-run by <M points because it
  flips only between batches; `max_evals` itself is exact in batches.
- Do not add a `horizon` knob to the rule "for experiments". A horizon that is
  not the probe's own cost reintroduces the mismatch in the other direction.

Known gap, still open: the equal-cost argument holds only in the steady state,
where one iteration serves one batch. An iteration that retries a line search
serves `max_ls` batches, so a horizon priced in `n_probe_evals` iterations buys
fewer batches than a probe once steps start retrying, and the verdict's two
sides drift apart again. A restarting iteration drifts it the other way: it
serves a probe instead of a step, and how many paths restart is a runtime
number, so no iteration clock can be priced in batches. `docs/PLAN.md` tracks
the residual asymmetry; the fix is not part of this decision.

Implemented with this decision: the per-path horizon as the engine's `_Horizon`
clock, per-batch billing as its `_Budget` meter, the scheduler and seam tests
restated in batches, and three new invariants (samples produced per completed
horizon, the budget never exceeded under restart storms, a stall inside a
horizon poisons that sample). The `How it works` page carries the resulting
wording.

## Amendment: the horizon clock is gone

Part 1 is superseded. `_Horizon` was removed: a continue sample is now one
iteration's step, and a restarted path skips sampling for the iteration its
probe served, so the probe is neither read back as a step's Δ nor as a
zero-step. Parts 2 and 3 stand (billing in batches; the rule stays knob-free and
reads samples, not evals).

The reason is that the dominant cost is the gradient count, not the eval count.
A probe computes `g` `n_probe_evals` times; one step computes it once (default
`n_alpha = 1`). With `g` ≈ 2–5× `f`, that is the asymmetry worth closing, and it
closed with an opt-in f-only evaluator seam: a probe draws `n_probe_evals`
value-only batches and then serves one fused batch at the per-column winners, so
it computes `g` once too. Without the seam the engine refuses `n_probe_evals > 1`
with a `ValueError`, so no configuration compares a `n_probe_evals`-gradient probe
against a one-gradient step; the default is `n_probe_evals = 1`. The gate is a
conservative proxy: exact symmetry is `n_probe_evals == n_alpha`, and `n_alpha`
lives in the solver, which the engine sees as a member that exists, not as a
value. What is left is the f-count - cheap, and bounded by `max_ls` on the step
side.

## Amendment: the stop is an iteration horizon, and batches are only reported

Part 2's meter is retired with it: `max_evals` no longer exists. The engine stops
on a deterministic horizon, `max_iters` (iteration 0 is the init probe), or on an
optional wall-clock `time_limit`; at least one of the two must be given.

Reporting follows the axis that dominates: only the fused value+gradient pass is
charged, so `n_batches`/`n_points` count `g` batches and `g` points. A probe's
f-only draws are billed nowhere and are not gated - they are the cheap half, and
counting them would mix two different costs into one number, which is what made
the old `n_batches` unfit as a comparison unit once the f-only seam existed. The
default shape (no `evaluate_f`, `n_probe_evals = 1`) has no f-only batch at all,
so there the two definitions coincide. The price of dropping them: the gate's
and the deadline's blind spot inside a multi-draw probe is under one probe.

Why the meter had to go rather than be fixed: a batch budget can only be exact
in an iteration that serves one batch, and this ADR's own "known gap" records
that a retrying step serves `max_ls` and a restarting one serves a probe, so the
number of batches an iteration costs is the method's business, at runtime. An
engine that bills batches therefore cannot say how many iterations a caller has
bought, and a run that stalls outright asks for no batch at all - the meter
under-counts exactly where the loop is spinning. The horizon is the quantity the
engine actually owns: one completed iteration per unit, whatever it served.

What survives from Part 2: the probes are still charged somewhere, only now the
thing charged is an iteration, and the init probe is iteration 0 rather than an
unbilled prelude. The field this paragraph calls `nit` is `n_iters` now: it sits
beside `n_points`/`n_batches`, and the scipy spelling stops at `x`/`fun`.
`budget_check` keeps Part 2's role unchanged - a batch-level
gate for a budget the engine cannot see - and gains one: it is the escape hatch
for any stop condition the engine has no knob for. It cannot replace the
horizon, because a fully stalled flock never asks for a batch, so a
`budget_check` alone would never fire.

Consequences:

- `status` becomes `max_iters`, `time_limit` or `budget_exhausted`. There is no
  `all_stalled` stop any more: a stalled flock idles to the horizon instead of
  ending early, which is the price of a stop that does not read the rule's
  cadence.
- The migration is not a rename. `max_evals = N` bounded batches; `max_iters = N`
  bounds iterations, each of which may serve up to one probe plus `max_ls` trial
  batches. Moving a run from one to the other without re-tuning is a silent
  budget increase of up to ~`n_probe_evals + 1 + max_ls` times.
- `samples_until_verdict()` loses its only engine caller, since the shortcut that
  read it is gone; it stays on the protocol for now, marked to be cut.

## Amendment: the per-step trial cap moves into the method

`max_ls` is gone. Both amendments above quote it as the bound on an iteration's
batches, and both stay as written - they record what the engine looked like
then. What changed is whose decision the enumeration length is.

It was never the engine's. A step's enumeration ends because the method's own
test fired (its convergence criterion, its `n_alpha` grid, its trial budget);
`max_ls` was a second, invisible stopping rule layered over a decision the
method already owned, and it could only fire in one of two ways: as a failsafe
for a method that never terminates, or as a stall for a method that was still
making progress. The first is the method's bug, and the second silently converts
a converging step into a stall that only a restart can undo. So the step loop is
`while rejecting` now, like the run loop, and termination is a stated contract
instead of an engine knob: `verdict` must eventually return `accept` or
`give_up`. A method that keeps saying "retry" hangs the engine - loudly, at the
caller's own fault, rather than by being quietly truncated.

Neither migration bullet above survives unchanged in spirit, so read them as
history: an iteration's batches are bounded by the method's own search, and the
engine cannot quote a number for them. `DiagonalPreconditioner` needs no new
bound - its post-warm-up grid is capped by `n_alpha`, and its warm-up's
geometric shrink bottoms out at a zero-length step, which passes Armijo because
the directional derivative of that zero step is zero.

## Amendment: neither side is a single Gaussian

The "plain mean and population sd" in point 3 above is history. Both sides of
the verdict are now a scott_mad KDE over their own samples (one Gaussian kernel
per sample, `h = 1.06 · MAD/Φ⁻¹(3/4) · n^(-1/5)`): the restart side's pool is a mixture
of per-start draws, and the continue side's Δ stream is not guaranteed Gaussian
either. What this amendment keeps is the decision the point argued for - one
estimator on both sides, reading `_stage_window() = M` samples with the continue
side's count matched to the pool's - not the particular estimator.

## Amendment: two terms renamed, no decision changed

This record is left as written; two of its names are retired. Read them as
follows, here and in any other pre-0.1.0 text.

- `n_probe_evals` is now **`n_draws`**. The knob counts draws: a draw is one
  batch of candidate points whose column minima become the starts, and a probe
  serves `n_draws` of them. The old name counted "evals", which was defined as
  one `evaluate_batch` call - no longer the only way to serve a probe, since an
  f-only evaluator serves the draws value-only. The count itself, the
  `evaluate_f` requirement above one draw, and the pricing argument in this
  record are unchanged; only the unit's name moved off the batch.
- **`incumbent`** is now **`best`**. It is the same object kept twice: `fun`/`x`
  is the reported answer (best feasible point) and `fun_raw` the raw minimum the
  verdict reads. The float handed to `should_restart` is `best_fun_raw`, and the
  engine's holder is `_Best`. "Incumbent" was optimizer jargon for something the
  public result already names `fun`/`x`, so a reader of both had to translate.
