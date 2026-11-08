"""A ``--problems``/``--arms`` filter that matches nothing must be refused.

The filters exist so one arm can be re-measured without its panel. They match by
name, which means a misspelled name - or two names that are each real but belong
to different arms - used to finish as a clean exit with no arm run at all. That
is the one failure a re-measurement cannot show: the figure it redraws is the
old one, at the old budget, and nothing says so.

Needs the ``exp`` dependency group (the harness imports matplotlib and optax), so
a plain ``dev`` test run skips this file instead of failing it.

`selection` is what a filter is asked of: `main` measures, so asking it what a
filter picks would run an arm to learn a set of names.
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
pytest.importorskip("matplotlib")
pytest.importorskip("optax")

from experiments import compare_baselines as harness
from experiments.harness import record


@pytest.fixture(autouse=True)
def _isolated_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Point both output trees at a temp dir so a refusal writes nothing anywhere."""
    monkeypatch.setattr(record, "RECORD_RESULTS", tmp_path / "results")
    monkeypatch.setattr(record, "RECORD_FIGURES", tmp_path / "figures")


def _run(*argv: str) -> str:
    """Run ``main`` with these arguments and return the message it exits with."""
    with pytest.raises(SystemExit) as caught:
        harness.main([*argv])
    return str(caught.value)


def test_a_misspelled_problem_name_is_refused_before_anything_runs(
    tmp_path: Path,
) -> None:
    """The typo is named, and so is the name it probably meant."""
    message = _run("--problems", "rastrigin_100")

    assert "no such problem: rastrigin_100" in message
    assert "rastrigin_d100" in message
    assert not (tmp_path / "results").exists()


def test_names_that_never_meet_in_one_arm_are_refused() -> None:
    """Both names are real; the Adam arm of the diagonal quadratic is `adam-lr0.1`."""
    message = _run(
        "--groups",
        "smooth",
        "--problems",
        "diag_quad_d100_cond10000",
        "--arms",
        "adam-lr2",
    )

    assert "no single arm carries both filters" in message


def test_a_legal_filter_selects_exactly_the_pair_it_names() -> None:
    """Naming the arm the problem actually has leaves one pair, and only that one."""
    selected = harness.selection(
        "smooth",
        frozenset({"diag_quad_d100_cond10000"}),
        frozenset({"adam-lr0.1"}),
    )

    assert [(prob.name, arm.key) for prob, arm in selected] == [
        ("diag_quad_d100_cond10000", "adam-lr0.1"),
    ]


@pytest.mark.parametrize("group", ["bogus", "all"])
def test_an_undrawable_group_is_named_before_anything_runs(group: str) -> None:
    """`all` is a real problem group, but it is a union to iterate, not a panel."""
    message = _run("--groups", group)

    assert f"no group {group!r}" in message
    assert "headline" in message


@pytest.mark.parametrize("flag", ["--plot-only", "--lr-sweep"])
def test_a_filter_on_a_flag_that_measures_nothing_is_refused(flag: str) -> None:
    """Neither flag runs an arm, so a filter would be dropped without a word."""
    message = _run(flag, "--problems", "rastrigin_d100")

    assert flag in message
