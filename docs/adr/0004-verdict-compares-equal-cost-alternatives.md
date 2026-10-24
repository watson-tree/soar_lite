# The restart verdict compares equal-cost alternatives, and every probe is billed

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
number, so no iteration clock can be priced in batches. `CONTEXT.md` records
the open question under `horizon`; the fix is not part of this decision.

Implemented with this decision: the per-path horizon as the engine's `_Horizon`
clock, per-batch billing as its `_Budget` meter, the scheduler and seam tests
restated in batches, and three new invariants (samples produced per completed
horizon, the budget never exceeded under restart storms, a stall inside a
horizon poisons that sample). `ALGORITHM.md` carries the resulting wording.
