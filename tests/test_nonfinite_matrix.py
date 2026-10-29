"""The non-finite event matrix: one table for what every odd value may become.

There is no single owner of the "is this number usable" decision - `_Best`,
`_argmin_finite`, the paths' start/trial/winner gates, the solver's fold gate
and the restart rule each decide locally, on purpose, because each decides a
different question. What they must agree on is the table, and the table is
short:

    value       feasible      becomes fun/x   becomes fun_raw   ends the run
    finite      yes           yes             yes               no
    finite      no            no              yes               no
    +inf        any           no              no                no
    nan         any           no              no                no
    -inf        yes           -inf            no                yes
    -inf        no            no              no                no

A seventh dimension is not in that table because it is not a value: with the
value finite, either half of the pair can still be broken. A gradient that is
not all finite is "half dead" - the value is real and reports, the point cannot
be proposed from or folded on - so the run stalls the path instead of poisoning
the secant. `test_local_solver.py` owns the fold gate; the engine-level rows are here.

`test_scheduler.py` and `test_restart_rule.py` cover the same events where they meet
a specific mechanism (a probe group with a dead column, an all-NaN pool, the
payoff of a zero term). This file is the table itself, so a change to any one
site has to keep every other site's rows true.
"""

# pyright: reportPrivateUsage=false
# The table is the spec for `_Best` and `_argmin_finite`, so reading them here
# is the intended read, not a boundary crossing.

import math
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from soar_lite.engine import (
    Bounds,
    _argmin_finite,
    _Best,
    _OptimumReached,
    n_path_schedule,
)
from soar_lite.global_model import UniformGlobalModel
from soar_lite.restart_rule import NoRestartRule

jax.config.update("jax_enable_x64", val=True)


def unit_bounds(d: int = 1) -> jnp.ndarray:
    """Return the engine's box in working coords: x ∈ [0,1]^d."""
    return jnp.stack([jnp.zeros(d), jnp.ones(d)], axis=1)


def flock_kwargs(
    x0: jnp.ndarray,
    bounds: Bounds,
    *,
    key: jnp.ndarray | None = None,
) -> dict[str, Any]:
    """Return the entry's two flock kwargs: the run's width and its starts.

    The engine takes its width as `n_paths` and every point a path starts from as
    the global model's first proposal, so a test that chooses its flock says both
    here. The model is the shipped default one: it hands `x0` back on the first
    `propose` and draws any later re-draw from `key`.
    """
    return {
        "n_paths": int(x0.shape[0]),
        "global_model": UniformGlobalModel(
            bounds,
            key=jax.random.PRNGKey(0) if key is None else key,
            x0=x0,
        ),
    }


@pytest.mark.parametrize(
    ("f", "feasible", "fun", "fun_raw", "ends_run"),
    [
        (1.0, True, 1.0, 1.0, False),  # a plain answer
        (1.0, None, 1.0, 1.0, False),  # no mask: every point counts as feasible
        (1.0, False, math.inf, 1.0, False),  # usable evidence, not an answer
        (math.inf, True, math.inf, math.inf, False),  # a dead start
        (math.nan, True, math.inf, math.inf, False),
        (math.nan, False, math.inf, math.inf, False),
        (math.nan, None, math.inf, math.inf, False),
        # Below every finite value but unreachable: no answer to report, and the
        # run goes on (the point was served, the caller just cannot use it).
        (-math.inf, False, math.inf, math.inf, False),
        # A feasible floor is the answer itself: it ends the run on the spot and
        # is not a `fun_raw` sample, which stays over finite points only.
        (-math.inf, True, -math.inf, math.inf, True),
    ],
)
def test_best_counts_a_value_only_when_it_is_feasible_and_finite(
    f: float,
    fun: float,
    fun_raw: float,
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
            best.observe([f], x_batch, feas)
    else:
        best.observe([f], x_batch, feas)

    assert best.fun == fun
    assert best.fun_raw == fun_raw
    assert best.unbounded is ends_run
    if best.fun == math.inf:
        assert bool(jnp.all(jnp.isnan(best.x)))  # no answer, no x to point at


def test_a_mask_of_zero_and_one_flags_is_accepted() -> None:
    """The one accepted shape: 0/1 numbers, or the booleans they stand for."""
    for mask in (jnp.array([0.0, 1.0]), jnp.array([False, True])):
        best = _Best(1)
        best.observe([1.0, 2.0], jnp.array([[0.1], [0.9]]), mask)
        # Row 0 is the cheaper value and row 1 is the flagged one, so the read has
        # to follow the mask row by row rather than dropping the flagged points.
        assert best.fun == 2.0
        assert best.fun_raw == 1.0


@pytest.mark.parametrize("flag", [math.nan, -2.0, 0.5, 2.0, 1e-12])
def test_a_mask_flag_that_is_not_zero_or_one_is_refused(flag: float) -> None:
    """Every truthy-but-not-1 flag is refused, and refused before any fold."""
    best = _Best(1)
    with pytest.raises(ValueError, match="feas mask"):
        best.observe([1.0], jnp.array([[0.25]]), jnp.array([flag]))
    assert best.fun == math.inf  # neither best absorbed the batch
    assert best.fun_raw == math.inf


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
    iteration, which is refused the same way. `NoRestartRule` is never asked: a
    path with no point is restarted without a verdict. Yet the value is a value
    the run served, so it is the answer. The two halves are gated on different
    questions and are allowed to disagree: the report is about the value, the
    re-draw is about what can be done next.
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
        **flock_kwargs(jnp.array([[0.5]]), unit_bounds(), key=jax.random.PRNGKey(0)),
        bounds=unit_bounds(),
        max_iters=10,
        restart_rule=NoRestartRule(),
    )

    # One batch per iteration: a point-less path is re-drawn, never stepped.
    assert res.n_batches == 10, res.n_batches
    assert res.fun == float(batches[0].min())  # and its value is still the answer
    assert math.isfinite(float(res.fun))
    assert bool(jnp.all(jnp.isfinite(res.x)))
    assert res.status == "max_iters"


def test_a_half_dead_winner_is_refused_before_it_becomes_a_start() -> None:
    """The same dimension one lane over: the value wins, the gradient refuses it.

    The flock hands back usable gradients; every later batch has a
    finite value and a NaN first component. The value is the best served so far
    and is allowed to enter the report, but the point is never committed as the
    path's next start - the batch is the run's last. With the gate missing, the
    path would step from an unusable point and serve at least one more batch.
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
        **flock_kwargs(jnp.array([[0.5]]), unit_bounds(), key=jax.random.PRNGKey(0)),
        bounds=unit_bounds(),
        max_iters=10,
        restart_rule=NoRestartRule(),
    )

    assert res.n_batches == 2, res.n_batches  # the flock + the one refused step
    assert res.status == "max_iters"
    assert math.isfinite(float(res.fun))
