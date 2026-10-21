"""Assert-style self-checks for the empirical EI restart policy.

The rule takes no cadence knob: a path is read over a window that
`_stage_window()` derives from the payoff pool's size, with the plain sample mean
and population sd over that window's finite Δ. These tests therefore choose a
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
    _ei_continue,
    _ei_restart,
    _normal_cdf,
    _normal_pdf,
)

# The window these tests grow the pool to. 8 is arbitrary - the tests are about
# the read, not about a size - and even, so the pool below can keep mu_r = 1.
WINDOW = 8


def close(a: float, b: float, tol: float = 1e-5) -> bool:
    """Return True iff |a - b| < tol."""
    return abs(a - b) < tol


def grow_pool(policy: EIRestartRule, to: int = WINDOW) -> None:
    """Grow the pool to `to` samples: {0, 2} repeated fits mu_r = 1, sd_r = 1."""
    for _ in range(to // 2):
        policy.observe_restart(0.0)
        policy.observe_restart(2.0)


def test_normal_functions_match_known_values() -> None:
    """Φ/φ are the standard normal tables (independent literals)."""
    assert close(_normal_cdf(0.0), 0.5, 1e-12)
    assert close(_normal_cdf(1.0), 0.8413447460685429, 1e-9)
    assert close(_normal_cdf(-1.0), 0.15865525393145707, 1e-9)
    assert close(_normal_pdf(0.0), 0.3989422804014327, 1e-9)  # 1/√(2π)
    assert close(_normal_pdf(1.0), 0.24197072451914337, 1e-9)  # exp(-1/2)/√(2π)


def test_ei_continue_exact_degenerate_at_sigma_zero() -> None:
    """Degenerate branch: sigma_p = 0 gives EI = max(B - (f_k + μ_p), 0) exactly."""
    assert close(_ei_continue(0.0, 0.0, 1.0, -0.5), 0.0, 1e-15)  # 0 - 0.5 < 0 → 0
    assert close(_ei_continue(0.0, 0.0, -1.0, 0.0), 1.0, 1e-12)  # 0 - (-1) = 1
    assert close(_ei_continue(0.0, 0.5, 1.0, -2.0), 1.5, 1e-12)  # 0.5 - (-1) = 1.5


def test_ei_continue_matches_gaussian_ei_literal() -> None:
    """Gaussian EI at z = -1 takes a hand-computed value (independent literal).

    sigma_p = 0.6, B = 0, f_k = 0.3, μ_p = 0.3 give z = -1 and EI ≈ 0.04998928.
    """
    assert close(_ei_continue(0.6, 0.0, 0.3, 0.3), 0.04998928, 1e-6)


def test_ei_restart_matches_gaussian_fit_literal() -> None:
    """Gaussian fit over {r_i}: [0.0, 2.0] → mu_r=1, sigma_r=1, z=0 → EI = φ(0)."""
    assert close(_ei_restart(1.0, [0.0, 2.0]), 0.3989422804014327, 1e-9)


def test_ei_restart_fits_a_population_std() -> None:
    """The pool's spread is the population sd (÷n), the same one the path side uses.

    [0.0, 2.0] has population sd 1 and sample sd √2 ≈ 1.4142; the fit is the
    former, which is what makes the two sides' reads commensurable.
    """
    assert close(_ei_restart(1.0, [0.0, 2.0]), _ei_restart(1.0, [0.0, 2.0]))
    sample_sd = 1.0 / math.sqrt(2.0) * 0.39894228  # what the ÷(n-1) sd would give
    assert not close(_ei_restart(1.0, [0.0, 2.0]), sample_sd, 1e-3)


def test_ei_restart_gaussian_tail_below_all_observations() -> None:
    """All r_i ≥ B, yet the Gaussian tail gives EI_restart > 0 (regression).

    The former empirical positive-part was always 0 here; [1.5, 2.0] →
    mu_r=1.75, sigma_r=0.25, z=-3 → EI ≈ 9.553858e-5.
    """
    assert close(_ei_restart(1.0, [1.5, 2.0]), 9.553857926192428e-5, 1e-9)


def test_ei_restart_empty_single_and_nonfinite_skipped() -> None:
    """M=0 → 0; M=1 → sigma_r=0 degenerate max(B-mu_r,0); non-finite skipped."""
    assert _ei_restart(1.0, []) == 0.0
    assert _ei_restart(3.0, [2.0]) == 1.0  # max(3 - 2, 0)
    assert _ei_restart(1.0, [2.0]) == 0.0  # max(1 - 2, 0)
    assert close(_ei_restart(1.0, [0.0, 2.0, float("nan")]), 0.3989422804014327, 1e-9)
    # +inf must be skipped too, not only NaN: keeping it makes mu_r = inf and
    # variance inf - inf = NaN, so EI_restart comes back NaN and every restart
    # is silently disabled (`NaN > EI_continue` is False).
    assert close(_ei_restart(1.0, [float("inf"), 0.0, 2.0]), 0.3989422804014327, 1e-9)
    assert _ei_restart(1.0, [float("inf"), float("inf"), 2.0]) == 0.0
    assert _ei_restart(1.0, [float("inf"), float("-inf")]) == 0.0


def test_window_is_the_pool_size() -> None:
    """The cadence follows the evidence: window = M, with 1 standing for "empty".

    Both sides' relative sd is ~1/sqrt(2(n - 1)) in their own sample counts, so
    reading the path over exactly the pool's M is what stops the noisier side's
    own noise from deciding the verdict - and it is derived rather than injected,
    so no caller can ask for a window the pool cannot back. Nothing pads M to 2:
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
    assert policy._stage_window() == 11  # matched to what sigma_r is fitted on


def test_stage_read_is_mean_and_population_sd_of_its_finite_deltas() -> None:
    """One read per stage: the plain sample mean and population sd of its Δ.

    No fading weights, so the estimate is worth exactly the folds it holds; the
    read is the module's own _ei_continue over those two moments (equal weights
    on the *same* samples must give the same numbers), with `f_k + mu` as the
    next value's mean.
    """
    policy = EIRestartRule()
    grow_pool(policy)  # window = 8
    deltas = [0.0, 2.0]
    for delta in deltas:
        policy.observe_continue(0, delta)
    mu = 1.0  # mean of {0, 2}
    sd = 1.0  # population sd of {0, 2}
    one_ruler = _ei_continue(sd, 0.0, -1.0, mu)
    assert close(policy._continue_payoff(0, -1.0, 0.0), one_ruler, 1e-15)
    assert policy._stage_deltas(0) == deltas


def test_the_two_sides_read_the_same_samples_the_same_way() -> None:
    """Identical samples on both sides agree exactly when f_k + mu = mu_r.

    The continue side's mean/population sd of Δ and the restart side's fit of the
    pool are the same estimator, so with the same values and the same reference
    value both must return the same EI - this is the "one ruler" property, and
    it fails for any scale factor between them.
    """
    policy = EIRestartRule()
    grow_pool(policy, 2)  # M = 2 = the stage's own length below
    for delta in (0.0, 2.0):
        policy.observe_continue(0, delta)
    assert close(
        policy._continue_payoff(0, 0.0, 1.0),  # f_k = 0, mu = 1 -> mean 1 = mu_r
        policy._restart_payoff(1.0),  # pool {0, 2}: mu_r = 1, sigma_r = 1
        tol=1e-15,
    )


def test_nonfinite_delta_counts_towards_the_window_but_is_not_read() -> None:
    """A stall folds inf every round: the gate must still come due, the read not.

    The engine reports a stalled path by feeding inf, and that path's verdict has
    to stay on the same clock as its moving cohort - so the guard sits at the
    read, where an empty stage returns 0.0 (the degenerate "continuing buys
    nothing" read, the same one a zero-spread stage gives). That is what revives
    a stalled flock instead of letting one +inf poison the mean and spread into
    NaN (`NaN > EI_continue` is False, so no path would ever restart again).
    """
    policy = EIRestartRule()
    grow_pool(policy)  # window = 8, mu_r = 1, sigma_r = 1
    for _ in range(WINDOW - 1):
        policy.observe_continue(0, math.inf)
    assert policy._stage_deltas(0) == []
    assert policy.should_restart(0, 0.0, 0.0) is False  # one fold short of due
    policy.observe_continue(0, math.nan)
    assert policy.should_restart(0, 0.0, 0.0) is True  # due, and EI_restart > 0
    assert policy._continue_payoff(0, 0.0, 0.0) == 0.0


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
    """Worsening Δ collapses EI_continue; the Gaussian restart tail still fires.

    A constant Δ has population sd 0, so its read is the degenerate
    max(B - f_k - mu, 0): a path that keeps losing ground is priced at 0 against
    the pool's 0.0833, while a path that keeps gaining is priced above it.
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
    for _ in range(WINDOW):  # pool = 8 copies of 0.0: mu_r = sigma_r = 0
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
    assert policy.samples_until_verdict() == WINDOW - 1  # a fresh stage is running


def test_payoff_pool_is_shared_across_paths() -> None:
    """A restart payoff is a sampler property: every path reads one pooled pool.

    Path 1 never observed a restart itself, yet the event's r values are visible
    to its verdict - that is what makes EI_restart one global quantity instead of
    n_paths noisy per-path estimates of the same thing (the pre-refactor defect).
    """
    policy = EIRestartRule()
    for v in (0.0, 2.0):  # one restart event's group-mins
        policy.observe_restart(v)
    policy.observe_continue(1, 0.5)  # path 1 has its own continuation evidence
    assert policy._restart_payoff(0.0) == _ei_restart(0.0, [0.0, 2.0])
    # ... and B = inf gates both paths off regardless of pooled evidence
    assert policy.should_restart(1, 0.0, math.inf) is False


def test_per_path_stage_is_independent() -> None:
    """Folds and restarts are per path; only the payoff pool is shared."""
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


def test_restart_keeps_the_shared_payoff_pool() -> None:
    """A restart drops the path's stage but never discards pooled payoffs."""
    policy = EIRestartRule()
    policy.observe_restart(0.0)
    policy.observe_restart(2.0)
    policy.observe_continue(0, 0.1)
    policy.restart(0)
    assert policy.group_mins == [0.0, 2.0]
    assert policy._restart_payoff(1.0) == _ei_restart(1.0, [0.0, 2.0])


def test_samples_until_verdict_counts_to_the_next_full_stage() -> None:
    """The shortcut's cadence oracle: window - folds, per the earliest path.

    0 with nothing observed (no cadence to bound -> the engine keeps its ordinary
    rounds). The window is read fresh, so a stage that runs while the pool grows
    is counted down against the grown window - and since the pool only ever
    grows, this can never put a verdict further off than it is (an overestimated
    cadence would be the engine stopping for a vote that was still due).
    """
    policy = EIRestartRule()
    assert policy.samples_until_verdict() == 0  # no path evidence at all
    grow_pool(policy)  # window = 8
    for step in range(1, WINDOW + 1):  # two paths folded in lockstep
        policy.observe_continue(0, 0.0)
        policy.observe_continue(1, 0.0)
        assert policy.samples_until_verdict() == max(WINDOW - step, 0)
    policy.restart(0)  # a restarted path counts from 0: a full window away
    assert policy.samples_until_verdict() == 0  # path 1's full stage is due now
    policy.restart(1)  # both stages are fresh, so both count a full window
    assert policy.samples_until_verdict() == WINDOW
    policy.observe_continue(0, 0.0)
    assert policy.samples_until_verdict() == WINDOW - 1  # path 0 is the earliest
    grow_pool(policy, 20)  # the pool grows to 28 while path 0's stage runs
    assert policy.samples_until_verdict() == 27  # read over the current window


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all restart policy tests passed")
