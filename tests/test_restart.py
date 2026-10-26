"""Assert-style self-checks for the empirical EI restart policy.

The rule takes no cadence knob: a path is read over a window that
`_stage_window()` derives from the group-min pool's size, with a scott_mad KDE over
that window's finite Δ on both sides. These tests therefore choose a
window by growing a pool, and read the stage's evidence back off the rule.

No jax: the policy is pure-Python float bookkeeping. Run with:

    uv run python -m tests.test_restart
"""

# ruff: noqa: SLF001
# pyright: reportPrivateUsage=false
# This suite is the spec for the rule's `_`-private reads and closed forms;
# importing them here is the intended read, not a boundary crossing.

import math

from soar_lite.restart import (
    EIRestartRule,
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


def continue_payoff(f_k: float, incumbent: float, deltas: list[float]) -> float:
    """Read path 0's continue payoff after folding `deltas` into its stage."""
    policy = EIRestartRule()
    for delta in deltas:
        policy.observe_continue(0, delta)
    return policy._continue_payoff(0, f_k, incumbent)


def restart_payoff(incumbent: float, group_mins: list[float]) -> float:
    """Read the restart payoff with `group_mins` as the shared pool."""
    policy = EIRestartRule()
    for group_min in group_mins:
        policy.observe_restart(group_min)
    return policy._restart_payoff(incumbent)


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


def test_ei_restart_empty_single_and_nonfinite_skipped() -> None:
    """M=0 → 0; M=1 → h=0 degenerate max(B-r,0); non-finite skipped."""
    assert restart_payoff(1.0, []) == 0.0
    assert restart_payoff(3.0, [2.0]) == 1.0  # max(3 - 2, 0)
    assert restart_payoff(1.0, [2.0]) == 0.0  # max(1 - 2, 0)
    assert close(restart_payoff(1.0, [0.0, 2.0, float("nan")]), KDE_LITERAL, 1e-9)
    # +inf must be skipped too, not only NaN: keeping it blows the scott_mad
    # bandwidth to inf (or NaN), so the read stops being evidence and the
    # verdict fires unconditionally (or is silently disabled by NaN).
    assert close(restart_payoff(1.0, [float("inf"), 0.0, 2.0]), KDE_LITERAL, 1e-9)
    assert restart_payoff(1.0, [float("inf"), float("inf"), 2.0]) == 0.0
    assert restart_payoff(1.0, [float("inf"), float("-inf")]) == 0.0


def test_window_is_the_pool_size() -> None:
    """The cadence follows the evidence: window = M, with 1 standing for "empty".

    The two sides are worth about the same number of samples only when their
    counts match, so reading the path over exactly the pool's M is what stops
    the noisier side's own noise from deciding the verdict - and it is derived
    rather than injected, so no caller can ask for a window the pool cannot
    back. Nothing pads M to 2:
    at M = 1 both reads collapse to their documented degenerate form (no spread
    on either side), so a longer window would only delay a verdict that already
    cannot restart.
    """
    policy = EIRestartRule()
    assert policy._stage_window() == 1  # empty pool: nothing to read, not a prior
    policy.observe_restart(0.0)
    assert policy._stage_window() == 1  # M = 1: the degenerate read itself
    for m in range(2, 12):
        policy.observe_restart(0.0)
        assert policy._stage_window() == m  # M itself
    policy.observe_restart(math.inf)  # stays in the pool, is not fitted
    assert len(policy.group_mins) == 12
    assert policy.pool_size == 11
    assert policy._stage_window() == 11  # matched to what the KDE is built on


def test_stage_read_is_a_kde_over_its_finite_deltas() -> None:
    """One read per stage: the scott_mad KDE over its Δ, equal weight per fold.

    No fading weights, so the read is worth exactly the folds it holds; it is
    the same KDE the restart side reads (equal weights on the *same* samples
    must give the same numbers), with `f_k + Δ` as the next value's centre.
    """
    policy = EIRestartRule()
    grow_pool(policy)  # window = 8
    deltas = [0.0, 2.0]
    for delta in deltas:
        policy.observe_continue(0, delta)
    one_ruler = continue_payoff(-1.0, 0.0, deltas)
    assert close(policy._continue_payoff(0, -1.0, 0.0), one_ruler, 1e-15)
    assert policy._stage_deltas(0) == deltas


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


def test_nonfinite_delta_counts_towards_the_window_but_is_not_read() -> None:
    """A stall folds inf every round: the gate must still come due, the read not.

    The engine reports a stalled path by feeding inf, and that path's verdict has
    to stay on the same clock as its moving cohort - so the guard sits at the
    read, where an empty stage returns 0.0 (the degenerate "continuing buys
    nothing" read, the same one a zero-bandwidth stage gives). That is what
    revives a stalled flock instead of letting one +inf poison the KDE into NaN
    (`NaN > EI_continue` is False, so no path would ever restart again).
    """
    policy = EIRestartRule()
    grow_pool(policy)  # window = 8, h well defined
    for _ in range(WINDOW - 1):
        policy.observe_continue(0, math.inf)
    assert policy._stage_deltas(0) == []
    assert policy.should_restart(0, 0.0, 0.0) is False  # one fold short of due
    policy.observe_continue(0, math.nan)
    assert policy.should_restart(0, 0.0, 0.0) is True  # due, and EI_restart > 0
    assert policy._continue_payoff(0, 0.0, 0.0) == 0.0


def test_an_empty_pool_is_the_cold_start_and_a_dead_pool_is_not() -> None:
    """An empty group-min pool seeds the flock; a pool of dead starts must not.

    The engine asks every path on its first iteration, before any probe has
    run, and that ask is what triggers the initialization probe - so a rule must
    answer True while it holds no restart evidence at all. `group_mins` is the
    test, not `pool_size`: a pool that exists but is entirely non-finite means
    the probe ran and every start was dead, which should stall the flock and
    wait for a later vote rather than re-probe every iteration.
    """
    policy = EIRestartRule()
    assert policy.should_restart(0, math.inf, math.inf) is True  # no probe yet
    for _ in range(WINDOW):
        policy.observe_restart(math.inf)  # the probe ran; every start was dead
    assert policy.pool_size == 0  # nothing usable, but the pool is not empty
    assert policy.should_restart(0, math.inf, math.inf) is False


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
        assert policy._stage_deltas(0) == []
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


def test_no_restart_evidence_never_restarts() -> None:
    """A pool with no spread means EI_restart = 0: no verdict can restart."""
    policy = EIRestartRule()
    for _ in range(WINDOW):  # pool = 8 copies of 0.0: MAD = 0, so h = 0
        policy.observe_restart(0.0)
    assert policy.pool_size == WINDOW
    for _ in range(WINDOW):
        policy.observe_continue(0, 0.5)
    assert policy.should_restart(0, 0.0, 0.0) is False  # 0 > 0 is False


def test_no_incumbent_never_restarts_but_does_not_stall_the_clock() -> None:
    """Cold start (incumbent = inf) never restarts, even at a due gate.

    The stage still closes: B is the engine's to write, and holding a full stage
    open would stall the path's cadence behind a verdict that cannot be read.
    """
    policy = EIRestartRule()
    grow_pool(policy)
    for _ in range(WINDOW):
        policy.observe_continue(0, 0.5)
    assert policy.should_restart(0, 0.0, math.inf) is False
    assert policy._stage_deltas(0) == []  # the read was skipped, the gate was not
    policy.observe_continue(0, 0.5)
    assert policy._stage_deltas(0) == [0.5]  # a fresh stage is running


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
    """Folds and restarts are per path; only the group-min pool is shared."""
    policy = EIRestartRule()
    policy.observe_continue(0, 0.4)
    policy.observe_continue(0, 0.6)
    assert policy._stage_deltas(0) == [0.4, 0.6]
    assert policy._stage_deltas(1) == []  # path 1 has folded nothing
    policy.observe_continue(1, -0.5)
    assert policy._stage_deltas(1) == [-0.5]
    policy.restart(0)
    assert policy._stage_deltas(0) == []  # path 0's stage is dropped
    assert policy._stage_deltas(1) == [-0.5]  # path 1's is untouched
    assert policy.group_mins == []  # restart keeps the (empty) shared pool


def test_restart_keeps_the_shared_group_min_pool() -> None:
    """A restart drops the path's stage but never discards pooled group-mins."""
    policy = EIRestartRule()
    policy.observe_restart(0.0)
    policy.observe_restart(2.0)
    policy.observe_continue(0, 0.1)
    policy.restart(0)
    assert policy.group_mins == [0.0, 2.0]
    assert policy._restart_payoff(1.0) == restart_payoff(1.0, [0.0, 2.0])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all restart policy tests passed")
