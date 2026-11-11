"""The L-BFGS-B arm serves exactly its budget, not one call more.

scipy stops only after it has served the evaluation that went over ``maxfun``,
so asking for N gets N + 1 calls. The arm is the smooth panel's ceiling
reference, and a ceiling that spends one more evaluation than the arm it is
compared against flatters itself - visibly so at a small budget, where one call
is a few percent. The run cuts the paid-but-over-budget call out of the trace,
which is also the behaviour that stays right if scipy ever stops at N exactly.

Needs the ``exp`` dependency group: the harness imports matplotlib and optax.
"""

import os

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
pytest.importorskip("matplotlib")
pytest.importorskip("optax")

from experiments.harness import problems
from experiments.harness.methods import run_lbfgs

# Ill-conditioned enough that it hits the limit instead of converging first, so
# the over-budget call really happens on the way to this assertion.
BUDGET = 20


def test_the_lbfgs_arm_stops_at_its_budget() -> None:
    """Whatever scipy serves, the record holds at most the budget."""
    prob = problems.diag_quadratic(d=100, cond=1e4)

    trace = run_lbfgs(prob, max_batches=BUDGET, seed=0)

    assert len(trace.points) == len(trace.batches) == len(trace.best) == BUDGET
    assert trace.points[-1] == BUDGET
    assert trace.batches[-1] == BUDGET


def test_the_reported_value_comes_from_the_kept_curve() -> None:
    """Truncating must not leave `f` describing the call that was cut."""
    trace = run_lbfgs(
        problems.diag_quadratic(d=100, cond=1e4), max_batches=BUDGET, seed=0
    )

    assert trace.f == trace.best[-1]
