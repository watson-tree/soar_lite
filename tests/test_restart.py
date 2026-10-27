"""Assert-style self-checks for the empirical EI restart policy.

The rule takes no cadence knob: a path is read over a window that
`_stage_window()` derives from the group-min pool's sample count, with a
scott_mad KDE over that window's Δ on both sides. These tests therefore choose a
window by growing a pool, and read the stage's evidence back off the rule.

No jax: the policy is pure-Python float bookkeeping. Run with:

    uv run python -m tests.test_restart
"""

# ruff: noqa: SLF001
# pyright: reportPrivateUsage=false
# This suite is the spec for the rule's `_`-private reads and closed forms;
# importing them here is the intended read, not a boundary crossing.

import math

import pytest

from soar_lite.restart import (
    EIRestartRule,
    NoRestartRule,
    _gaussian_ei,
    _normal_cdf,
    _normal_pdf,
    _scott_mad_bandwidth,
)

# The window these tests grow the pool to. 8 is arbitrary - the tests are about
# the read, not about a size - and even, so the pool below has a defined spread.
WINDOW = 8

# Rounds a stage spans: the WINDOW folds its read keeps, plus the round it opened
# in - a probe for a restarted path, the closing verdict for a continuing one.
STAGE = WINDOW + 1

# The scott_mad KDE read of the pool {0, 2} against f⋆ = 1 (h = 1.36712).
KDE_LITERAL = 0.6851344670507414


def close(a: float, b: float, tol: float = 1e-5) -> bool:
    """Return True iff |a - b| < tol."""
    return abs(a - b) < tol


def grow_pool(policy: EIRestartRule, to: int = WINDOW) -> None:
    """Grow the pool to `to` samples: {0, 2} repeated has MAD = 1."""
    for _ in range(to // 2):
        policy.observe_restart(0.0)
        policy.observe_restart(2.0)


def continue_payoff(f_k: float, best_fun_raw: float, deltas: list[float]) -> float:
    """Read path 0's continue payoff after folding `deltas` into its stage.

    The pool is grown to the stage's own length first: the read keeps the newest
    M samples, so a shorter pool would slice the stage. This helper is the
    closed form, not the cadence - the slice has its own test.
    """
    policy = EIRestartRule()
    for _ in deltas:
        policy.observe_restart(0.0)
    for delta in deltas:
        policy.observe_continue(0, delta)
    return policy._continue_payoff(0, f_k, best_fun_raw)


def restart_payoff(best_fun_raw: float, group_mins: list[float]) -> float:
    """Read the restart payoff with `group_mins` as the shared pool."""
    policy = EIRestartRule()
    for group_min in group_mins:
        policy.observe_restart(group_min)
    return policy._restart_payoff(best_fun_raw)


def test_normal_functions_match_known_values() -> None:
    """Φ/φ are the standard normal tables (independent literals)."""
    assert close(_normal_cdf(0.0), 0.5, 1e-12)
    assert close(_normal_cdf(1.0), 0.8413447460685429, 1e-9)
    assert close(_normal_cdf(-1.0), 0.15865525393145707, 1e-9)
    assert close(_normal_pdf(0.0), 0.3989422804014327, 1e-9)  # 1/√(2π)
    assert close(_normal_pdf(1.0), 0.24197072451914337, 1e-9)  # exp(-1/2)/√(2π)


def test_ei_continue_degenerate_at_zero_bandwidth() -> None:
    """One fold (or a constant stage) gives h = 0: EI = max(f⋆ - (f_k + Δ), 0)."""
    assert close(continue_payoff(1.0, 0.0, [-0.5]), 0.0, 1e-15)  # 0 - 0.5 < 0 → 0
    assert close(continue_payoff(-1.0, 0.0, [0.0]), 1.0, 1e-12)  # 0 - (-1) = 1
    assert close(continue_payoff(1.0, 0.5, [-2.0]), 1.5, 1e-12)  # 0.5 - (-1) = 1.5
    assert close(continue_payoff(0.0, 0.0, [0.5, 0.5, 0.5]), 0.0, 1e-15)  # MAD = 0


def test_ei_continue_is_a_mean_of_per_sample_ei() -> None:
    """The stage's read is the KDE mean of per-sample EI (independent literal).

    Δ = {-0.5, 0.5}, f_k = 0.5, f⋆ = 0 give h ≈ 0.68356 and EI ≈ 0.14724 - the
    mean of _gaussian_ei at the two next-value centres {0.0, 1.0}.
    """
    assert close(continue_payoff(0.5, 0.0, [-0.5, 0.5]), 0.14724429770396302, 1e-9)


def test_ei_restart_matches_kde_literal() -> None:
    """A scott_mad KDE over {r_i}: [0.0, 2.0] → h ≈ 1.36712 → EI ≈ 0.68513."""
    assert close(restart_payoff(1.0, [0.0, 2.0]), 0.6851344670507414, 1e-9)


def test_ei_restart_is_a_mean_of_per_sample_ei() -> None:
    """The pool is a mixture, not one Gaussian: EI is the mean of per-sample EI.

    h = _scott_mad_bandwidth({0, 2}) and the read is the plain mean of
    _gaussian_ei(h, f⋆, r_i) - a single-Gaussian fit (0.3989 here) is the bug
    this replaced.
    """
    h = _scott_mad_bandwidth([0.0, 2.0])
    expected = (_gaussian_ei(h, 1.0, 0.0) + _gaussian_ei(h, 1.0, 2.0)) / 2.0
    assert close(restart_payoff(1.0, [0.0, 2.0]), expected, 1e-15)
    assert not close(restart_payoff(1.0, [0.0, 2.0]), 0.3989422804014327, 1e-3)


def test_ei_restart_tail_below_all_observations() -> None:
    """All r_i ≥ f⋆, yet the KDE tail still gives EI_restart > 0 (regression).

    An empirical positive-part is always 0 here; the kernel tail is not:
    [1.5, 2.0] → h ≈ 0.34178 → EI ≈ 5.531594e-3.
    """
    assert close(restart_payoff(1.0, [1.5, 2.0]), 0.005531593511968263, 1e-12)


def test_ei_restart_prices_every_nonfinite_start_as_a_zero_term() -> None:
    """M=0 → 0; M=1 → h=0 degenerate; +inf, NaN and -inf all buy 0."""
    assert restart_payoff(1.0, []) == 0.0
    assert restart_payoff(3.0, [2.0]) == 1.0  # max(3 - 2, 0)
    assert restart_payoff(1.0, [2.0]) == 0.0  # max(1 - 2, 0)
    # A non-finite sample says nothing about how much a restart buys, so it is the
    # zero term of the same mean: the finite KDE diluted by the finite fraction
    # 2/3, whichever shape it is - and it must not widen the bandwidth either,
    # which is why h stays the finite one.
    for zero_term in (float("inf"), float("nan"), float("-inf")):
        assert close(
            restart_payoff(1.0, [0.0, 2.0, zero_term]),
            KDE_LITERAL * 2.0 / 3.0,
            1e-9,
        )
    # No finite start at all: the finite subset is empty, so the read is 0.
    assert restart_payoff(1.0, [float("inf"), float("inf"), 2.0]) == 0.0
    assert restart_payoff(1.0, [float("inf"), float("-inf")]) == 0.0
    assert restart_payoff(1.0, [float("nan"), float("nan")]) == 0.0


def test_window_is_the_pool_plus_its_opening_round() -> None:
    """The cadence follows the evidence: window = M + 1 rounds, M = the pool.

    The two sides are worth about the same number of samples only when their
    counts match, so the stage is as long as the pool is: M folds plus the round
    it opened in, which no stage folds in (the restarted path probes, the
    continuing one is being judged). Every path that opens a stage in round T is
    therefore due in round T + M, and reads M samples. Both counts are read off
    the pool rather than injected, so no caller can ask for a window its own pool
    cannot back. Every sample counts, whatever it was worth: the read is one mean
    over the pool with the non-finite samples as its zero terms, so a `nan` or a
    `-inf` spaces re-probes exactly like a `+inf` dead start. Nothing pads M to
    2: at M = 1 both reads collapse to their documented degenerate form (no
    spread on either side), so a longer window would only delay a verdict that
    already cannot restart.
    """
    policy = EIRestartRule()
    assert policy._sample_count() == 0
    assert policy._stage_window() == 1  # empty pool: cold start, never read
    for m in range(1, 13):
        policy.observe_restart(0.0)
        assert policy._sample_count() == m  # M itself
        assert policy._stage_window() == m + 1  # plus the round it opens in
    policy.observe_restart(math.inf)  # a dead start is a sample like any other
    assert policy._stage_window() == 14
    policy.observe_restart(math.nan)  # no information, still a zero term
    assert policy._stage_window() == 15
    policy.observe_restart(-math.inf)
    assert policy._stage_window() == 16


def test_stage_read_is_a_kde_over_its_deltas() -> None:
    """One read per stage: the scott_mad KDE over its Δ, equal weight per fold.

    No fading weights, so the read is worth exactly the folds it holds; it is
    the same KDE the restart side reads (equal weights on the *same* samples
    must give the same numbers), with `f_k + Δ` as the next value's centre. A
    stall never lands here - it is reported as `None`, not as a value.
    """
    policy = EIRestartRule()
    grow_pool(policy)  # window = 8
    deltas = [0.0, 2.0]
    for delta in deltas:
        policy.observe_continue(0, delta)
    one_ruler = continue_payoff(-1.0, 0.0, deltas)
    assert close(policy._continue_payoff(0, -1.0, 0.0), one_ruler, 1e-15)
    assert policy.deltas.get(0, []) == deltas


def test_the_continue_side_reads_only_the_newest_m_folds() -> None:
    """A continuing path holds M + 1 folds when judged and reads the newest M.

    The round a stage opens in folds a sample into the stage for a continuing
    path, while the same round only probes for a restarted one - so reading
    everything would leave the continuing side one sample heavier than the pool
    it is priced against. The slice drops the oldest fold, which is the one
    furthest from where the path is now.
    """
    policy = EIRestartRule()
    grow_pool(policy)  # M = WINDOW
    deltas = [0.1 * (k + 1) for k in range(WINDOW + 1)]  # one more than the pool
    for delta in deltas:
        policy.observe_continue(0, delta)
    assert policy.deltas[0] == deltas  # all of them are held ...
    assert close(
        policy._continue_payoff(0, 0.0, 0.0),  # ... and the newest M are read
        continue_payoff(0.0, 0.0, deltas[1:]),
        1e-15,
    )


def test_the_two_sides_read_the_same_samples_the_same_way() -> None:
    """Identical samples on both sides agree exactly when f_k + Δ = r.

    Both sides are the same scott_mad KDE, so with the same values and the same
    reference value both must return the same EI - this is the "one ruler"
    property, and it fails for any scale factor between them.
    """
    policy = EIRestartRule()
    grow_pool(policy, 2)  # M = 2 = the stage's own length below
    for delta in (0.0, 2.0):
        policy.observe_continue(0, delta)
    assert close(
        policy._continue_payoff(0, 0.0, 1.0),  # f_k + Δ = {0, 2} = the pool
        policy._restart_payoff(1.0),  # pool {0, 2}: same samples, same KDE
        tol=1e-15,
    )


def test_a_stall_supplies_no_delta_and_counts_the_iteration() -> None:
    """A stall is absence: the stage drops, the round still counts.

    The engine reports a stalled path with `None`, not a value, so the continue
    side holds no sample to read (payoff 0.0 - continuing bought nothing). The
    stage clock is one clock for both shapes of evidence, so a stalled path comes
    due on the same round a moving one would, not on a fabricated fold count
    whose reads could never be matched against the pool.
    """
    policy = EIRestartRule()
    grow_pool(policy)  # window = STAGE, h well defined
    for _ in range(STAGE - 1):
        policy.observe_continue(0, None)
    assert policy.deltas.get(0, []) == []
    assert policy.rounds[0] == STAGE - 1
    assert policy.should_restart(0, 0.0, 0.0) is False  # one round short of due
    policy.observe_continue(0, None)
    assert policy.should_restart(0, 0.0, 0.0) is True  # due: continue 0 < restart > 0
    assert policy._continue_payoff(0, 0.0, 0.0) == 0.0
    assert policy.rounds[0] == 0  # the verdict closed the stage, restart won or not


def test_a_stall_clears_a_partial_stage_and_keeps_its_wait() -> None:
    """A stall drops the folds but not the rounds already served for them.

    The read is one sided once the stage is gone (continue = 0), and the clock is
    what decides when to ask - so a mid-stage stall comes due when its cohort
    does, never a full window later.
    """
    policy = EIRestartRule()
    grow_pool(policy)  # window = STAGE
    for _ in range(WINDOW - 3):
        policy.observe_continue(0, 0.1)
    policy.observe_continue(0, None)
    assert policy.deltas.get(0, []) == []  # no continuation to average
    assert policy.rounds[0] == WINDOW - 2  # the stall served its round
    assert policy.should_restart(0, 0.0, 0.0) is False
    for _ in range(2):  # the stall plus these two: rounds reach M
        policy.observe_continue(0, None)
    assert policy.rounds[0] == WINDOW
    assert policy.should_restart(0, 0.0, 0.0) is False  # one short, as a mover would be
    policy.observe_continue(0, None)
    assert policy.should_restart(0, 0.0, 0.0) is True  # due on its cohort's round


def test_a_nonfinite_delta_is_rejected_at_the_seam() -> None:
    """`inf`/`nan` are not stall spellings: the engine says `None`, so reject."""
    policy = EIRestartRule()
    for bad in (math.inf, math.nan, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            policy.observe_continue(0, bad)


def test_an_empty_pool_seeds_and_a_dead_pool_re_probes_once_per_window() -> None:
    """An empty pool seeds the flock; a dead pool ties and re-probes by window.

    The cold-start ask happens before any probe has run, and that ask is what
    triggers the initialization probe - so a rule must answer True while it
    holds no restart evidence at all. A pool that exists but is entirely `+inf`
    (or `nan`) is not cold start: the probe ran and every start was dead, so
    `EI_restart` prices 0. That 0 ties a stalled continue side, so the verdict
    fires and spends one probe - the only action that can still buy evidence. It
    cannot spin: the dead samples it appends grow the window, so the next ask on
    the path comes a window later.
    """
    policy = EIRestartRule()
    assert policy.should_restart(0, math.inf, math.inf) is True  # no probe yet
    for _ in range(WINDOW):
        policy.observe_restart(math.inf)  # the probe ran; every start was dead
    assert policy._stage_window() == STAGE  # dead starts are samples too
    for _ in range(STAGE):  # the flock stalls out on the dead landscape
        policy.observe_continue(0, None)
    assert policy.should_restart(0, 0.0, 0.0) is True  # 0 >= 0: buy a sample
    # The verdict closes the stage whether it fired or not: one window, one
    # verdict - so the clock is not left in the due band re-asking every round.
    assert policy.rounds[0] == 0
    assert policy.deltas.get(0, []) == []
    policy.observe_continue(0, None)
    assert policy.should_restart(0, 0.0, 0.0) is False  # not due again until a window

    # The same holds for a pool of `nan`: the window is the pool's length, not
    # its finite count, so a pool that never reads a number still spaces its
    # re-probes (a window of the finite count would be 1 here, re-probing every
    # round).
    nan_pool = EIRestartRule()
    for _ in range(WINDOW):
        nan_pool.observe_restart(math.nan)
    for _ in range(STAGE):
        nan_pool.observe_continue(0, None)
    assert nan_pool.should_restart(0, 0.0, 0.0) is True
    nan_pool.observe_restart(math.nan)  # the probe's own dead group-min
    nan_pool.observe_continue(0, None)
    assert nan_pool._stage_window() == STAGE + 1
    assert nan_pool.should_restart(0, 0.0, 0.0) is False


def test_verdict_is_due_once_per_window_and_closes_the_stage() -> None:
    """The gate opens a verdict and the verdict takes it: due once per window.

    Re-asking a full stage every round would turn one verdict's evidence into a
    restart vote per round; the stage closes whatever the verdict said, and the
    next fold opens the next one.
    """
    policy = EIRestartRule()
    grow_pool(policy)  # window = STAGE
    for _ in range(2):  # the cadence must not drift over two full verdicts
        for step in range(1, STAGE + 1):
            policy.observe_continue(0, 0.1)
            assert policy.should_restart(0, 0.0, 0.0) is (step == STAGE)
        assert policy.deltas[0] == []  # the stage's one read was taken
        assert policy.rounds[0] == 0  # ... and its clock starts over with it
    # A stage that only closed is not restarted by the rule itself: the engine
    # restarts paths, not the rule, and `restart` only counts the probe round.
    assert policy.group_mins == [0.0, 2.0] * (WINDOW // 2)


def test_worsening_flow_restarts_improving_flow_continues() -> None:
    """Worsening Δ collapses EI_continue; the KDE restart tail still fires.

    A constant Δ has MAD 0, so its read is the degenerate max(f⋆ - f_k - Δ, 0):
    a path that keeps losing ground is priced at 0 against the pool's 0.6851,
    while a path that keeps gaining is priced above it.
    """
    policy = EIRestartRule()
    grow_pool(policy)
    f = 0.0
    for step in range(1, STAGE + 1):
        policy.observe_continue(0, 0.1)  # every step worsens by +0.1
        f += 0.1
        assert policy.should_restart(0, f, 0.0) is (step == STAGE)

    policy = EIRestartRule()
    grow_pool(policy)
    f = 0.0
    for _ in range(STAGE):
        policy.observe_continue(0, -0.2)  # every step improves by 0.2
        f -= 0.2
        assert policy.should_restart(0, f, 0.0) is False


def test_a_zero_payoff_tie_restarts_to_buy_a_second_sample() -> None:
    """Both sides read 0: the tie fires and the probe is the evidence bought.

    A pool of copies has MAD 0 (`h = 0`) and its value is the best, so
    `EI_restart = max(f⋆ - 0, 0) = 0`; a constant worsening stage also has MAD 0,
    so `EI_continue = max(f⋆ - (f_k + Δ), 0) = 0` for `f_k + Δ >= f⋆`. Neither side
    can improve a verdict, but the probe's group-min is a real i.i.d. sample, so
    the tie spends it instead of idling.
    """
    policy = EIRestartRule()
    for _ in range(WINDOW):  # pool = 8 copies of 0.0: MAD = 0, so h = 0
        policy.observe_restart(0.0)
    assert policy._stage_window() == STAGE
    for _ in range(STAGE):
        policy.observe_continue(0, 0.5)
    assert policy._restart_payoff(0.0) == 0.0
    assert policy._continue_payoff(0, 0.0, 0.0) == 0.0
    assert policy.should_restart(0, 0.0, 0.0) is True  # 0 >= 0 is True


def test_no_restart_rule_seeds_once_then_never() -> None:
    """`NoRestartRule` answers the seeding ask, then refuses every later one.

    The first ask happens before any probe, so every path must hear True or the
    flock never seeds. The flip lives in `observe_restart`, which the engine
    calls after the probe: flipping it inside `should_restart` would seed only
    the path asked first and leave the rest on their `inf` placeholders.
    """
    rule = NoRestartRule()
    assert [rule.should_restart(i, math.inf, math.inf) for i in range(4)] == [True] * 4
    for _ in range(4):  # the initialization probe reports one group-min per path
        rule.observe_restart(0.5)
    assert not any(rule.should_restart(i, 0.0, 0.0) for i in range(4))
    rule.observe_continue(0, None)  # neither channel re-opens it
    rule.observe_continue(0, 0.1)
    rule.restart(0)
    assert rule.should_restart(0, 0.0, 0.0) is False


def test_no_best_still_restarts_at_a_due_gate() -> None:
    """Cold start (best = inf) restarts, and the stage still closes.

    A verdict with nothing to weigh is a probe: with no finite point served yet,
    re-drawing is the only action that can produce one. Answering False here
    would leave the one path that could re-draw blocked from asking, so a single
    all-dead initialization group would freeze the run - the loop the flock
    cannot leave on its own.
    """
    policy = EIRestartRule()
    grow_pool(policy)
    for _ in range(STAGE):  # a full stage: the gate is due
        policy.observe_continue(0, 0.5)
    assert policy.should_restart(0, 0.0, math.inf) is True
    assert policy.deltas.get(0, []) == []  # the stage closed
    policy.observe_continue(0, 0.5)
    assert policy.deltas.get(0, []) == [0.5]  # a fresh stage is running


def test_no_best_waits_for_the_gate_before_reprobing() -> None:
    """No best does not mean every round: the stage clock still rules.

    The window is the pool's sample count plus one, so the re-probe interval
    grows as the pool does - the same anti-thrash argument the tie rests on.
    """
    policy = EIRestartRule()
    grow_pool(policy)
    for _ in range(STAGE - 1):
        policy.observe_continue(0, 0.5)
    assert policy.should_restart(0, 0.0, math.inf) is False  # one round early


def test_group_min_pool_is_shared_across_paths() -> None:
    """A group-min is a sampler property: every path reads one pooled pool.

    Path 1 never observed a restart itself, yet the event's group-mins are
    visible to its verdict - that is what makes EI_restart one global quantity
    instead of n_paths noisy per-path estimates of the same thing.
    """
    policy = EIRestartRule()
    for group_min in (0.0, 2.0):  # one restart event's group-mins
        policy.observe_restart(group_min)
    policy.observe_continue(1, 0.5)  # path 1 has its own continuation evidence
    assert policy._restart_payoff(0.0) == restart_payoff(0.0, [0.0, 2.0])
    # ... and the pool is read the same way whatever the best is: no best is a
    # probe (see test_no_best_still_restarts_at_a_due_gate), not a per-path gate.
    assert policy.should_restart(1, 0.0, math.inf) is False  # not due yet
    assert policy.should_restart(0, 0.0, math.inf) is False


def test_per_path_stage_is_independent() -> None:
    """Folds and closes are per path; only the group-min pool is shared."""
    policy = EIRestartRule()
    grow_pool(policy)  # window = STAGE
    policy.observe_continue(0, 0.4)
    policy.observe_continue(0, 0.6)
    policy.observe_continue(1, -0.5)
    assert policy.deltas.get(0, []) == [0.4, 0.6]
    assert policy.deltas.get(1, []) == [-0.5]  # its own stage, its own fold
    for _ in range(WINDOW - 1):
        policy.observe_continue(0, 0.1)
    policy.observe_continue(1, -0.5)  # path 1 is two rounds in, not due
    assert policy.rounds[0] == STAGE
    assert policy.rounds[1] == 2
    policy.should_restart(0, 0.0, 0.0)  # path 0 is due: its stage closes
    assert policy.deltas.get(0, []) == []
    assert policy.rounds[0] == 0
    assert policy.deltas.get(1, []) == [-0.5, -0.5]  # path 1's stage is untouched
    assert policy.rounds[1] == 2
    assert policy.group_mins == [0.0, 2.0] * (WINDOW // 2)  # the pool is shared


def test_a_path_that_restarts_at_every_verdict_keeps_its_cohorts_rounds() -> None:
    """Phase-locking: the probe round is a round, so no path drifts out of step.

    The engine reports only the paths that folded or stalled (`_report`), and a
    restarted path did neither - so the rule counts that round itself in
    `restart`. Without it a restarted path opens its next stage a round late and
    falls a further round behind at each restart, so a flock whose paths keep
    restarting drifts out of phase instead of coming due together.

    Two paths against one shared pool: path 0 always continues (its f keeps
    improving, so its read never loses to the pool's), path 1 always restarts
    (its f keeps worsening until the probe resets it). Every verdict must fall on
    the same round for both, and both must read the pool's M when it does.
    """
    policy = EIRestartRule()
    grow_pool(policy)  # the pool every verdict is priced against, grown by each probe
    f = {0: 0.0, 1: 0.0}
    due: dict[int, list[int]] = {0: [], 1: []}
    samples: dict[int, dict[int, int]] = {0: {}, 1: {}}
    pool_at: dict[int, int] = {}
    for round_no in range(1, 200):
        m = policy._sample_count()
        window = policy._stage_window()
        pool_at[round_no] = m
        fired: dict[int, bool] = {}
        for i in (0, 1):
            if policy.rounds.get(i, 0) >= window:
                due[i].append(round_no)
                # What the path brings to the verdict: never more than the pool.
                samples[i][round_no] = min(len(policy.deltas.get(i, [])), m)
            fired[i] = policy.should_restart(i, f[i], 0.0)
        for i in (0, 1):
            if fired[i]:
                policy.restart(i)  # the engine probes: the rule counts the round
                policy.observe_restart(0.0)  # ... and the probe's group-min lands
                f[i] = 0.0
            else:
                delta = -0.2 if i == 0 else 0.1
                policy.observe_continue(i, delta)
                f[i] += delta
    assert len(due[1]) >= 3  # the restarting path really did restart, repeatedly
    assert due[0] == due[1]  # same rounds: no path falls behind
    assert samples[0] == samples[1]  # ... and at each, the same sample count
    assert all(count == pool_at[round_no] for round_no, count in samples[0].items())


def test_restart_keeps_the_shared_group_min_pool() -> None:
    """The restart event never discards pooled group-mins (it touches no state)."""
    policy = EIRestartRule()
    policy.observe_restart(0.0)
    policy.observe_restart(2.0)
    for _ in range(3):
        policy.observe_continue(0, 0.1)
    policy.restart(0)
    assert policy.group_mins == [0.0, 2.0]
    assert policy.deltas.get(0, []) == [0.1] * 3  # resets at the verdict, not here
    assert policy.rounds[0] == 1  # the probe round opens the next stage
    assert policy._restart_payoff(1.0) == restart_payoff(1.0, [0.0, 2.0])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all restart policy tests passed")
