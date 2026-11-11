"""The non-finite event matrix: one table for what every odd value may become.

There is no single owner of the "is this number usable" decision - `_Best`,
`_argmin_finite`, the paths' start/trial/winner gates and the solver's step
gate each decide locally, on purpose, because each decides a different
question. What they must agree on is the table, and the table is short:

    value       feasible      becomes f/x     ends the run
    finite      yes           yes             no
    finite      no            no              no
    +inf        any           no              no
    nan         any           no              no
    -inf        yes           -inf            yes
    -inf        no            no              no

"No answer" is `nan` in both fields: `f` starts there and `x` is all-nan.
`inf` is left to mean a value an objective actually took, never a sentinel.

A seventh dimension is not in that table because it is not a value: with the
value finite, either half of the pair can still be broken. A gradient that is
not all finite is "half dead" - the value is real and reports, the point cannot
be proposed from or stepped on - so the run stalls the path instead of poisoning
the secant. `test_local_solver.py` owns the step gate; the engine-level rows are here.

`test_scheduler.py` covers the same events where they meet a specific
mechanism (a probe group with a dead column, an all-NaN pool, the
payoff of a zero term). This file is the table itself, so a change to any one
site has to keep every other site's rows true.
"""

# pyright: reportPrivateUsage=false
# The table is the spec for `_Best` and `_argmin_finite`, so reading them here
# is the intended read, not a boundary crossing.

import math

import jax
import jax.numpy as jnp
import pytest

from soar_lite.engine import n_path_schedule
from soar_lite.engine.schedule import _argmin_finite, _Best, _OptimumReached
from tests.flock import flock_kwargs, unit_domain

jax.config.update("jax_enable_x64", val=True)


@pytest.mark.parametrize(
    ("f", "feasible", "best_f", "ends_run"),
    [
        (1.0, True, 1.0, False),  # a plain answer
        (1.0, None, 1.0, False),  # no mask: every point counts as feasible
        (1.0, False, math.nan, False),  # usable evidence, not an answer
        (math.inf, True, math.nan, False),  # a dead start
        (math.nan, True, math.nan, False),
        (math.nan, False, math.nan, False),
        (math.nan, None, math.nan, False),
        # Below every finite value but unreachable: no answer to report, and the
        # run goes on (the point was served, the caller just cannot use it).
        (-math.inf, False, math.nan, False),
        # A feasible floor is the answer itself: it ends the run on the spot.
        (-math.inf, True, -math.inf, True),
    ],
)
def test_best_counts_a_value_only_when_it_is_feasible_and_finite(
    f: float,
    best_f: float,
    *,
    feasible: bool | None,
    ends_run: bool,
) -> None:
    """One served point, every row of the table, against one fresh `_Best`."""
    best = _Best(1)
    x_batch = jnp.array([[0.25]])
    feas = None if feasible is None else jnp.array([feasible])

    if ends_run:
        with pytest.raises(_OptimumReached):
            best.observe(x_batch, [f], feas)
    else:
        best.observe(x_batch, [f], feas)

    if math.isnan(best_f):
        assert math.isnan(best.f)
    else:
        assert best.f == best_f
    assert best.unbounded is ends_run
    if math.isnan(best.f):
        assert bool(jnp.all(jnp.isnan(best.x)))  # no answer, no x to point at


def test_a_mask_of_zero_and_one_flags_is_accepted() -> None:
    """The one accepted shape: 0/1 numbers, or the booleans they stand for."""
    for mask in (jnp.array([0.0, 1.0]), jnp.array([False, True])):
        best = _Best(1)
        best.observe(jnp.array([[0.1], [0.9]]), [1.0, 2.0], mask)
        # Row 0 is the cheaper value and row 1 is the flagged one, so the read has
        # to follow the mask row by row rather than dropping the flagged points.
        assert best.f == 2.0


@pytest.mark.parametrize("flag", [math.nan, -2.0, 0.5, 2.0, 1e-12])
def test_a_mask_flag_that_is_not_zero_or_one_is_refused(flag: float) -> None:
    """Every truthy-but-not-1 flag is refused, and refused before any step."""
    best = _Best(1)
    with pytest.raises(ValueError, match="feas mask"):
        best.observe(jnp.array([[0.25]]), [1.0], jnp.array([flag]))
    assert math.isnan(best.f)  # the batch was refused before any step


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([math.nan, 1.0, 0.5], 2),  # NaN ranks first and compares False to all
        ([1.0, math.nan, 0.5], 2),
        ([math.inf, 0.5, math.nan], 1),
        ([math.inf, math.nan], None),  # nothing finite to compare
        ([math.nan, math.nan], None),
        ([-math.inf, 1.0], 1),  # -inf is outside the finite set, not below it
        ([-math.inf], None),
        ([2.0, 2.0], 0),  # a tie keeps the first, as `min` does
        ([], None),
    ],
)
def test_argmin_finite_never_returns_a_non_finite_candidate(
    values: list[float],
    expected: int | None,
) -> None:
    """The finite-only argmin, which both bests and the feasible read sit on."""
    assert _argmin_finite(values) == expected


def test_a_half_dead_start_stalls_the_path_but_its_value_still_reports() -> None:
    """The seventh dimension at the start gate: value good, gradient unusable.

    Every start has a finite value and a NaN in the first gradient component.
    Nothing may be proposed from such a point, so the path never steps - and
    since it never holds a point, the engine owes it a fresh start every
    iteration, which is refused the same way. No restart is involved: a path
    with no point is re-drawn by the engine's own "no point" rule. Yet the value
    is a value the run served, so it is the answer. The two halves are gated on
    different questions and are allowed to disagree: the report is about the
    value, the re-draw is about what can be done next.
    """
    batches: list[jnp.ndarray] = []

    def eval_f_g(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        d = theta - 0.5
        f = jnp.sum(d**2, axis=1)
        g = (2.0 * d).at[:, 0].set(jnp.nan)  # the value is fine, the gradient is not
        batches.append(f)
        return f, g, None

    res = n_path_schedule(
        eval_f_g,
        **flock_kwargs(jnp.array([[0.5]]), unit_domain(), key=jax.random.PRNGKey(0)),
        max_iters=10,
    )

    # One batch per iteration: a point-less path is re-drawn, never stepped.
    assert res.n_f_g_batches == 10, res.n_f_g_batches
    assert res.f == float(batches[0].min())  # and its value is still the answer
    assert math.isfinite(float(res.f))
    assert bool(jnp.all(jnp.isfinite(res.x)))
    assert res.status == "max_iters"


def test_a_half_dead_winner_is_refused_before_it_becomes_a_start() -> None:
    """The same dimension one lane over: the value wins, the gradient refuses it.

    The flock hands back usable gradients; every later batch has a
    finite value and a NaN first component. The value is the best served so far
    and is allowed to enter the report, but the point is never committed as the
    path's next start - the batch is the run's last. With the gate missing, the
    path would step from an unusable point and serve at least one more batch.
    The arm is the frozen flock (`restart_stalled=False`): the path stalls and
    nothing can un-stall it, so the run ends on `all_stalled` rather than idling
    out the horizon.

    The start sits off the minimizer on purpose: one placed on it is stationary,
    and the method stops the path itself without serving a step - a different row
    of the table, owned by the method's stopping test rather than by this gate.
    """
    n_batches = [0]

    def eval_f_g(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        n_batches[0] += 1
        d = theta - 0.5
        f = jnp.sum(d**2, axis=1)
        g = 2.0 * d
        if n_batches[0] > 1:  # the step lane, from here on
            g = g.at[:, 0].set(jnp.nan)
        return f, g, None

    res = n_path_schedule(
        eval_f_g,
        **flock_kwargs(jnp.array([[0.2]]), unit_domain(), key=jax.random.PRNGKey(0)),
        max_iters=10,
        restart_stalled=False,
    )

    assert res.n_f_g_batches == 2, res.n_f_g_batches  # the flock + the one refused step
    assert res.status == "all_stalled"  # and nothing can un-stall the frozen path
    assert math.isfinite(float(res.f))
