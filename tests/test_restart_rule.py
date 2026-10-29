"""Self-checks for the two shipped restart rules.

Both rules are pure-Python bookkeeping over the evidence the engine forwards, so
nothing here needs jax: the engine's own behaviour (when a verdict is asked, when
a start is placed) is `test_scheduler.py`'s business.

Run with:

    uv run python -m tests.test_restart_rule
"""

import math

import pytest

from soar_lite.restart_rule import NoRestartRule, StallRestartRule


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


def test_stall_rule_is_quiet_before_any_evidence() -> None:
    """No round reported yet: the path is unknown, which reads as "moving"."""
    rule = StallRestartRule()
    assert rule.should_restart(0, 1.0, 1.0) is False


def test_stall_rule_fires_on_one_stall() -> None:
    """One stall round is the whole signal: restart on the next verdict."""
    rule = StallRestartRule()
    rule.observe_step(0, None)
    assert rule.should_restart(0, 1.0, 1.0) is True


def test_a_step_clears_the_latch() -> None:
    """Any finite step - uphill or micro - ends the stall."""
    rule = StallRestartRule()
    rule.observe_step(0, None)
    assert rule.should_restart(0, 1.0, 1.0) is True
    rule.observe_step(0, 0.0)
    assert rule.should_restart(0, 1.0, 1.0) is False
    rule.observe_step(0, -1e-12)
    assert rule.should_restart(0, 1.0, 1.0) is False
    rule.observe_step(0, 3.0)  # uphill is still a step, not a stall
    assert rule.should_restart(0, 1.0, 1.0) is False


def test_the_verdict_consumes_the_stall() -> None:
    """One stall buys one restart; the second ask says no until re-armed.

    The engine asks before it places anything and silences a path for the round
    it re-places, so a latch left set after firing would re-place that path every
    iteration - evaluated each time, never given a round to step in - and nothing
    would ever be reported to clear it.
    """
    rule = StallRestartRule()
    rule.observe_step(0, None)
    assert rule.should_restart(0, 1.0, 1.0) is True
    assert rule.should_restart(0, 1.0, 1.0) is False
    rule.observe_step(0, None)  # the fresh start stalled too
    assert rule.should_restart(0, 1.0, 1.0) is True
    assert rule.should_restart(0, 1.0, 1.0) is False


def test_stall_rule_ignores_the_best_so_far() -> None:
    """The verdict reads a path's own round; the raw best is not consulted."""
    rule = StallRestartRule()
    rule.observe_step(0, None)
    assert rule.should_restart(0, math.nan, -math.inf) is True


def test_a_path_with_no_usable_point_is_redrawn() -> None:
    """A dead start reports `None`, so it fires and is given another draw."""
    rule = StallRestartRule()
    rule.observe_step(0, None)
    assert rule.should_restart(0, math.nan, math.inf) is True


def test_paths_are_judged_independently() -> None:
    """One path's stall says nothing about another's progress."""
    rule = StallRestartRule()
    rule.observe_step(0, None)
    rule.observe_step(1, -9.0)
    assert rule.should_restart(0, 1.0, 1.0) is True
    assert rule.should_restart(1, 100.0, 1.0) is False


def test_a_nonfinite_delta_is_rejected_at_the_seam() -> None:
    """`None` is the engine's spelling for "no step"; inf/nan is a fault."""
    rule = StallRestartRule()
    for bad in (math.inf, math.nan, -math.inf):
        with pytest.raises(ValueError, match="delta"):
            rule.observe_step(0, bad)
