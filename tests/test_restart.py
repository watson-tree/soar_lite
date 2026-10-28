"""Self-checks for the two shipped restart rules.

Both rules are pure-Python bookkeeping over the evidence the engine forwards, so
nothing here needs jax: the engine's own behaviour (when a verdict is asked, when
a start is placed) is `test_scheduler.py`'s business.

Run with:

    uv run python -m tests.test_restart
"""

import math

import pytest

from soar_lite.restart import NoRestartRule, StallRestartRule


def test_no_restart_never_fires() -> None:
    """The frozen flock: no ask, however it is reached, ever restarts."""
    rule = NoRestartRule()
    assert rule.should_restart(0, 1.0, 1.0) is False
    # Walking a path through stalls and steps changes nothing.
    for delta in (None, -1.0, 0.0, 1.0):
        rule.observe_step(0, delta)
        assert rule.should_restart(0, 5.0, 1.0) is False


def test_no_restart_ignores_path_identity() -> None:
    """There is no first-ask exception left to leak across paths."""
    rule = NoRestartRule()
    assert [rule.should_restart(i, 0.0, 0.0) for i in range(4)] == [False] * 4


@pytest.mark.parametrize("window", [0, -3])
def test_stall_rule_rejects_a_bad_window(window: int) -> None:
    """A window of zero rounds cannot be read: there is no history to judge."""
    with pytest.raises(ValueError, match="window"):
        StallRestartRule(window=window)


@pytest.mark.parametrize("rel_tol", [-1e-3, math.nan, math.inf])
def test_stall_rule_rejects_a_bad_tolerance(rel_tol: float) -> None:
    """A negative or non-finite tolerance is refused where it is handed over."""
    with pytest.raises(ValueError, match="rel_tol"):
        StallRestartRule(rel_tol=rel_tol)


def test_stall_rule_is_quiet_until_the_window_is_full() -> None:
    """One stall is not a verdict: the window has to fill up first."""
    rule = StallRestartRule(window=3)
    for _ in range(2):
        rule.observe_step(0, None)
        assert rule.should_restart(0, 1.0, 1.0) is False


def test_stall_rule_fires_on_a_window_of_stalls() -> None:
    """No step at all, three rounds running: restart."""
    rule = StallRestartRule(window=3)
    for _ in range(3):
        rule.observe_step(0, None)
    assert rule.should_restart(0, 1.0, 1.0) is True


def test_stall_rule_counts_a_micro_step_as_no_progress() -> None:
    """A step smaller than rel_tol of the value is not progress, but a stall."""
    rule = StallRestartRule(window=3, rel_tol=1e-3)
    for _ in range(2):
        rule.observe_step(0, -0.05)  # 0.05 < 1e-3 * 100
        assert rule.should_restart(0, 100.0, 0.0) is False
    rule.observe_step(0, -0.05)
    assert rule.should_restart(0, 100.0, 0.0) is True


def test_stall_rule_ignores_the_best_so_far() -> None:
    """The verdict is a path's own progress; the raw best is not consulted."""
    rule = StallRestartRule(window=2)
    rule.observe_step(0, None)
    rule.observe_step(0, None)
    assert rule.should_restart(0, math.nan, -math.inf) is True


def test_one_real_gain_keeps_the_path() -> None:
    """A window with one step that moved the path is not a stall."""
    rule = StallRestartRule(window=3, rel_tol=1e-3)
    for delta in (-0.5, -0.01, -0.01):
        rule.observe_step(0, delta)
    assert rule.should_restart(0, 100.0, 0.0) is False


def test_an_uphill_step_is_not_progress() -> None:
    """A step that made the path worse bought nothing."""
    rule = StallRestartRule(window=2, rel_tol=0.0)
    rule.observe_step(0, 1.0)
    rule.observe_step(0, 1.0)
    assert rule.should_restart(0, 5.0, 1.0) is True


def test_a_gain_ages_out_of_the_window() -> None:
    """The window is the last `window` rounds, not the whole history."""
    rule = StallRestartRule(window=3, rel_tol=1e-3)
    rule.observe_step(0, -0.5)  # the one real gain
    rule.observe_step(0, -0.01)
    rule.observe_step(0, -0.01)
    assert rule.should_restart(0, 100.0, 0.0) is False
    rule.observe_step(0, -0.01)  # the gain falls out of a 3-round window
    assert rule.should_restart(0, 100.0, 0.0) is True


def test_a_path_with_no_usable_point_fires_after_a_full_window() -> None:
    """`nan` never counts as progress, so a dead start is re-drawn."""
    rule = StallRestartRule(window=2)
    rule.observe_step(0, None)
    rule.observe_step(0, None)
    assert rule.should_restart(0, math.nan, math.inf) is True


def test_the_verdict_resets_that_path_only() -> None:
    """Firing drops the firing path's window, so it judges a fresh descent."""
    rule = StallRestartRule(window=2)
    rule.observe_step(0, None)
    rule.observe_step(0, None)
    rule.observe_step(1, None)
    rule.observe_step(1, None)
    assert rule.should_restart(0, 1.0, 1.0) is True
    # Path 0's history is gone: one more stall is not a window.
    rule.observe_step(0, None)
    assert rule.should_restart(0, 1.0, 1.0) is False
    # Path 1 was not asked, so its window is untouched.
    assert rule.should_restart(1, 1.0, 1.0) is True


def test_paths_are_judged_independently() -> None:
    """One path's progress says nothing about another's."""
    rule = StallRestartRule(window=2, rel_tol=1e-3)
    rule.observe_step(0, None)
    rule.observe_step(0, None)
    rule.observe_step(1, -9.0)
    rule.observe_step(1, -9.0)
    assert rule.should_restart(0, 1.0, 1.0) is True
    assert rule.should_restart(1, 100.0, 1.0) is False


def test_a_nonfinite_delta_is_rejected_at_the_seam() -> None:
    """`None` is the engine's spelling for "no step"; inf/nan is a fault."""
    rule = StallRestartRule(window=2)
    for bad in (math.inf, math.nan, -math.inf):
        with pytest.raises(ValueError, match="delta"):
            rule.observe_step(0, bad)
