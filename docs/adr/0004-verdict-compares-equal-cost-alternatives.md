# The restart verdict compares equal-cost alternatives, and every probe is billed

`EI_restart` is fitted on probe group-mins, and one probe costs `n_probe_rounds`
served batches (one round each). `EI_continue` was fitted on one-step Δ samples,
and one step costs one batch. The verdict was therefore comparing a purchase of
`n_probe_rounds` rounds against a purchase of one round, and the bias ran one
way: the restart side's payoff had been aggregated over more exploration, so
`EI_restart` won verdicts the evidence did not support. In the same direction,
every probe event spent `n_probe_rounds` batches that the engine never charged
to `max_steps` — the init probe was billed to nobody at all. Both errors
flattered restarts, which is exactly the effect the multi-path escape claim
rests on, so this is a correctness bug, not a tuning question.

The decision, in three parts:

1. **One continue sample spans `n_probe_rounds` rounds.** The engine feeds
   `f(t) - f(t - n_probe_rounds)` once per horizon, non-overlapping, and feeds
   `inf` for that sample if any round inside it stalled. Both sides of the
   verdict are then "what `n_probe_rounds` rounds buy".
2. **The engine bills served batches.** Every batch the seam serves costs one
   unit of `max_steps`, so the init probe and each restart event cost
   `n_probe_rounds` units and the budget means batches, exactly. A run also
   attempts no more rounds than `max_steps`: a round serves at least one batch
   unless every path is stalled, so that cap only bites on a stalling run - a
   stalled round is free, and the cap is what stops a permanently stalled run
   whose rule can state no cadence. `nit` stays "rounds the main loop
   attempted"; the billed quantity is `nfev`, whose counter is `n_charged`
   because its unit follows the billing mode.
3. **The horizon lives in the engine, not the rule.** `n_probe_rounds` is a
   fact of the schedule the engine owns, so the engine feeds equal-cost samples
   and `EIRestartRule` stays knob-free: it keeps reading a plain mean and
   population sd over its samples. This also makes the existing
   `_stage_window() = M` derivation exact rather than approximate — that window
   was chosen as "the most samples a verdict can have in as many rounds as the
   restart side already has", and equal-cost samples make the two counts the
   same purchase.

Rejected: shrinking the probe to one draw (`n_probe_rounds = 1`) so that it
matches a one-step continue side. That ties the init policy — a "min of two
draws" start — to the restart accounting, and throws away the better start in
order to fix a comparison.

Consequences:

- The verdict cadence slows by `n_probe_rounds` (default 2): a stage of M
  samples needs M·`n_probe_rounds` rounds to fill. Fewer restarts, honestly
  earned. Every multi-path-escape number measured before this change is void
  and has to be re-measured; nothing was committed as a figure, so no cleanup
  beyond rerunning.
- One accounting authority: once the engine bills probes, a caller no longer
  needs a batch gate of its own. `experiments/methods.py` drops
  `budget_check=lambda: n_rounds < max_steps` and lets `max_steps` govern.
  `budget_check` survives only for a budget the engine cannot see (points,
  wall-clock, dollars), and such a gate can over-run by <M points because it
  flips only between batches; `max_steps` itself is exact in batches.
- Do not add a `horizon` knob to the rule "for experiments". A horizon that is
  not the probe's own cost reintroduces the mismatch in the other direction.

Implemented with this decision: the per-path horizon and its stall flag in the
engine, per-batch billing with the `spent` meter, the scheduler and seam tests
restated in batches, and three new invariants (samples produced per completed
horizon, the budget never exceeded under restart storms, a stall inside a
horizon poisons that sample). `ALGORITHM.md` carries the resulting wording.
