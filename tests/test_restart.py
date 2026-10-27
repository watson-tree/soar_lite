"""Assert-style self-checks for the empirical EI restart policy.

The rule takes no cadence knob: a path is read over a window that
`_stage_window()` derives from the group-min pool's defined sample count, with a
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

# The scott_mad KDE read of the pool {0, 2} against B = 1 (h = 1.36712).
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
    """Read path 0's continue payoff after folding `deltas` into its stage."""
    policy = EIRestartRule()
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
    """One fold (or a constant stage) gives h = 0: EI = max(B - (f_k + Δ), 0)."""
    assert close(continue_payoff(1.0, 0.0, [-0.5]), 0.0, 1e-15)  # 0 - 0.5 < 0 → 0
    assert close(continue_payoff(-1.0, 0.0, [0.0]), 1.0, 1e-12)  # 0 - (-1) = 1
    assert close(continue_payoff(1.0, 0.5, [-2.0]), 1.5, 1e-12)  # 0.5 - (-1) = 1.5
    assert close(continue_payoff(0.0, 0.0, [0.5, 0.5, 0.5]), 0.0, 1e-15)  # MAD = 0


def test_ei_continue_is_a_mean_of_per_sample_ei() -> None:
    """The stage's read is the KDE mean of per-sample EI (independent literal).

    Δ = {-0.5, 0.5}, f_k = 0.5, B = 0 give h ≈ 0.68356 and EI ≈ 0.14724 - the
    mean of _gaussian_ei at the two next-value centres {0.0, 1.0}.
    """
    assert close(continue_payoff(0.5, 0.0, [-0.5, 0.5]), 0.14724429770396302, 1e-9)


def test_ei_restart_matches_kde_literal() -> None:
    """A scott_mad KDE over {r_i}: [0.0, 2.0] → h ≈ 1.36712 → EI ≈ 0.68513."""
    assert close(restart_payoff(1.0, [0.0, 2.0]), 0.6851344670507414, 1e-9)


def test_ei_restart_is_a_mean_of_per_sample_ei() -> None:
    """The pool is a mixture, not one Gaussian: EI is the mean of per-sample EI.

    h = _scott_mad_bandwidth({0, 2}) and the read is the plain mean of
    _gaussian_ei(h, B, r_i) - a single-Gaussian fit (0.3989 here) is the bug
    this replaced.
    """
    h = _scott_mad_bandwidth([0.0, 2.0])
    expected = (_gaussian_ei(h, 1.0, 0.0) + _gaussian_ei(h, 1.0, 2.0)) / 2.0
    assert close(restart_payoff(1.0, [0.0, 2.0]), expected, 1e-15)
    assert not close(restart_payoff(1.0, [0.0, 2.0]), 0.3989422804014327, 1e-3)


def test_ei_restart_tail_below_all_observations() -> None:
    """All r_i ≥ B, yet the KDE tail still gives EI_restart > 0 (regression).

    An empirical positive-part is always 0 here; the kernel tail is not:
    [1.5, 2.0] → h ≈ 0.34178 → EI ≈ 5.531594e-3.
    """
    assert close(restart_payoff(1.0, [1.5, 2.0]), 0.005531593511968263, 1e-12)


def test_ei_restart_prices_inf_as_a_dead_start_and_skips_nan_and_neginf() -> None:
    """M=0 → 0; M=1 → h=0 degenerate; +inf is a 0-buy event; NaN/-inf not read."""
    assert restart_payoff(1.0, []) == 0.0
    assert restart_payoff(3.0, [2.0]) == 1.0  # max(3 - 2, 0)
    assert restart_payoff(1.0, [2.0]) == 0.0  # max(1 - 2, 0)
    # NaN is "no information": neither counted nor fitted.
    assert close(restart_payoff(1.0, [0.0, 2.0, float("nan")]), KDE_LITERAL, 1e-9)
    # +inf is a dead start: a legal 0-improvement event, so it dilutes the finite
    # KDE by the alive fraction 2/3 instead of vanishing from the read (and it
    # must not widen the bandwidth either, which is why h stays the finite one).
    assert close(
        restart_payoff(1.0, [float("inf"), 0.0, 2.0]),
        KDE_LITERAL * 2.0 / 3.0,
        1e-9,
    )
    # No finite start at all: the finite subset is empty, so the read is 0.
    assert restart_payoff(1.0, [float("inf"), float("inf"), 2.0]) == 0.0
    assert restart_payoff(1.0, [float("inf"), float("-inf")]) == 0.0


def test_window_is_the_defined_sample_count() -> None:
    """The cadence follows the evidence: window = finite plus +inf, floor 1.

    The two sides are worth about the same number of samples only when their
    counts match, so reading the path over exactly the pool's M is what stops
    the noisier side's own noise from deciding the verdict - and it is derived
    rather than injected, so no caller can ask for a window the pool cannot
    back. A dead start is still a defined sample (it is the zero term of the
    same mean), so it counts; `nan` is no information and does not. Nothing pads
    M to 2: at M = 1 both reads collapse to their documented degenerate form (no
    spread on either side), so a longer window would only delay a verdict that
    already cannot restart.
    """
    policy = EIRestartRule()
    assert policy._stage_window() == 1  # empty pool: nothing to read, not a prior
    policy.observe_restart(0.0)
    assert policy._stage_window() == 1  # M = 1: the degenerate read itself
    for m in range(2, 12):
        policy.observe_restart(0.0)
        assert policy._stage_window() == m  # M itself
    policy.observe_restart(math.inf)  # a dead start is a defined sample: counted
    assert len(policy.group_mins) == 12
    assert policy._stage_window() == 12
    policy.observe_restart(math.nan)  # no information: not counted
    assert len(policy.group_mins) == 13
    assert policy._stage_window() == 12


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
    grow_pool(policy)  # window = 8, h well defined
    for _ in range(WINDOW - 1):
        policy.observe_continue(0, None)
    assert policy.deltas.get(0, []) == []
    assert policy.rounds[0] == WINDOW - 1
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
    grow_pool(policy)  # window = 8
    for _ in range(WINDOW - 3):
        policy.observe_continue(0, 0.1)
    policy.observe_continue(0, None)
    assert policy.deltas.get(0, []) == []  # no continuation to average
    assert policy.rounds[0] == WINDOW - 2  # the stall served its round
    assert policy.should_restart(0, 0.0, 0.0) is False
    policy.observe_continue(0, None)
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
    assert policy._stage_window() == WINDOW  # dead starts are still defined samples
    for _ in range(WINDOW):  # the flock stalls out on the dead landscape
        policy.observe_continue(0, None)
    assert policy.should_restart(0, 0.0, 0.0) is True  # 0 >= 0: buy a sample
    # The verdict closes the stage whether it fired or not: one window, one
    # verdict - so the clock is not left in the due band re-asking every round.
    assert policy.rounds[0] == 0
    assert policy.deltas.get(0, []) == []
    policy.observe_continue(0, None)
    assert policy.should_restart(0, 0.0, 0.0) is False  # not due again until a window


def test_verdict_is_due_once_per_window_and_closes_the_stage() -> None:
    """The gate opens a verdict and the verdict takes it: due once per window.

    Re-asking a full stage every round would turn one verdict's evidence into a
    restart vote per round; the stage closes whatever the verdict said, and the
    next fold opens the next one.
    """
    policy = EIRestartRule()
    grow_pool(policy)  # window = 8
    for _ in range(2):  # the cadence must not drift over two full verdicts
        for step in range(1, WINDOW + 1):
            policy.observe_continue(0, 0.1)
            assert policy.should_restart(0, 0.0, 0.0) is (step == WINDOW)
        assert policy.deltas[0] == []  # the stage's one read was taken
        assert policy.rounds[0] == 0  # ... and its clock starts over with it
    # A stage that only closed is not restarted by the rule itself: the engine
    # restarts paths, not the rule, and `restart` only drops evidence.
    assert policy.group_mins == [0.0, 2.0] * (WINDOW // 2)


def test_worsening_flow_restarts_improving_flow_continues() -> None:
    """Worsening Δ collapses EI_continue; the KDE restart tail still fires.

    A constant Δ has MAD 0, so its read is the degenerate max(B - f_k - Δ, 0):
    a path that keeps losing ground is priced at 0 against the pool's 0.6851,
    while a path that keeps gaining is priced above it.
    """
    policy = EIRestartRule()
    grow_pool(policy)
    f = 0.0
    for step in range(1, WINDOW + 1):
        policy.observe_continue(0, 0.1)  # every step worsens by +0.1
        f += 0.1
        assert policy.should_restart(0, f, 0.0) is (step == WINDOW)

    policy = EIRestartRule()
    grow_pool(policy)
    f = 0.0
    for _ in range(WINDOW):
        policy.observe_continue(0, -0.2)  # every step improves by 0.2
        f -= 0.2
        assert policy.should_restart(0, f, 0.0) is False


def test_a_zero_payoff_tie_restarts_to_buy_a_second_sample() -> None:
    """Both sides read 0: the tie fires and the probe is the evidence bought.

    A pool of copies has MAD 0 (`h = 0`) and its value is the best, so
    `EI_restart = max(B - 0, 0) = 0`; a constant worsening stage also has MAD 0,
    so `EI_continue = max(B - (f_k + Δ), 0) = 0` for `f_k + Δ >= B`. Neither side
    can improve a verdict, but the probe's group-min is a real i.i.d. sample, so
    the tie spends it instead of idling.
    """
    policy = EIRestartRule()
    for _ in range(WINDOW):  # pool = 8 copies of 0.0: MAD = 0, so h = 0
        policy.observe_restart(0.0)
    assert policy._stage_window() == WINDOW
    for _ in range(WINDOW):
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


def test_no_best_never_restarts_but_does_not_leave_the_gate_open() -> None:
    """Cold start (best = inf) never restarts, even at a due gate.

    The stage still closes: B is the engine's to write, and holding a full stage
    open would stall the path's cadence behind a verdict that cannot be read.
    """
    policy = EIRestartRule()
    grow_pool(policy)
    for _ in range(WINDOW):
        policy.observe_continue(0, 0.5)
    assert policy.should_restart(0, 0.0, math.inf) is False
    assert policy.deltas.get(0, []) == []  # the read was skipped, the gate was not
    policy.observe_continue(0, 0.5)
    assert policy.deltas.get(0, []) == [0.5]  # a fresh stage is running


def test_group_min_pool_is_shared_across_paths() -> None:
    """A group-min is a sampler property: every path reads one pooled pool.

    Path 1 never observed a restart itself, yet the event's group-mins are
    visible to its verdict - that is what makes EI_restart one global quantity
    instead of n_paths noisy per-path estimates of the same thing (the
    pre-refactor defect).
    """
    policy = EIRestartRule()
    for group_min in (0.0, 2.0):  # one restart event's group-mins
        policy.observe_restart(group_min)
    policy.observe_continue(1, 0.5)  # path 1 has its own continuation evidence
    assert policy._restart_payoff(0.0) == restart_payoff(0.0, [0.0, 2.0])
    # ... and B = inf gates both paths off regardless of pooled evidence
    assert policy.should_restart(1, 0.0, math.inf) is False


def test_per_path_stage_is_independent() -> None:
    """Folds and closes are per path; only the group-min pool is shared."""
    policy = EIRestartRule()
    grow_pool(policy)  # window = 8
    policy.observe_continue(0, 0.4)
    policy.observe_continue(0, 0.6)
    policy.observe_continue(1, -0.5)
    assert policy.deltas.get(0, []) == [0.4, 0.6]
    assert policy.deltas.get(1, []) == [-0.5]  # its own stage, its own fold
    for _ in range(WINDOW - 2):
        policy.observe_continue(0, 0.1)
    policy.observe_continue(1, -0.5)  # path 1 is two rounds in, not due
    assert policy.rounds[0] == WINDOW
    assert policy.rounds[1] == 2
    policy.should_restart(0, 0.0, 0.0)  # path 0 is due: its stage closes
    assert policy.deltas.get(0, []) == []
    assert policy.rounds[0] == 0
    assert policy.deltas.get(1, []) == [-0.5, -0.5]  # path 1's stage is untouched
    assert policy.rounds[1] == 2
    assert policy.group_mins == [0.0, 2.0] * (WINDOW // 2)  # the pool is shared


def test_restart_keeps_the_shared_group_min_pool() -> None:
    """The restart event never discards pooled group-mins (it touches no state)."""
    policy = EIRestartRule()
    policy.observe_restart(0.0)
    policy.observe_restart(2.0)
    policy.observe_continue(0, 0.1)
    policy.restart(0)
    assert policy.group_mins == [0.0, 2.0]
    assert policy.deltas.get(0, []) == [0.1]  # resets at the verdict, not here
    assert policy.rounds[0] == 1
    assert policy._restart_payoff(1.0) == restart_payoff(1.0, [0.0, 2.0])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all restart policy tests passed")
