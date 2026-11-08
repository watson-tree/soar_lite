"""`_Paths.move` one ruling at a time.

The end-to-end suites reach `move` through `n_path_schedule`, so what they can
see is a whole run's arithmetic: the horizon, the three stop gates and the
assembled report all sit between the caller and a single iteration's decisions.
This suite drives `move` directly - no horizon, no clock, no report - so each
verdict the method can rule becomes one statement about where the path stands
and which counters moved.

That needs no seam: `_Paths` takes all four of its collaborators injected, and
`_Budget`, `_Best` and `_Evaluator` are four lines each to build. Every run here
is one path in the unit box unless it says otherwise, and every step is stated as
the script it is - the candidates `propose` hands back and the rulings `verdict`
gives, in call order.

The two lifecycles this pins, which no whole-run test can see cheaply:

- a path that stalls keeps the point it holds and buys no gradient, while a path
  that wins buys exactly one, in one batch, however many trials it served;
- every coordinate the engine serves is clipped into the box first, so the
  method judges the point that was actually served rather than the one it asked
  for.
"""

# pyright: reportPrivateUsage=false
# This suite's subject is the schedule's `_`-private state machine; there is no
# public entry that stops after one iteration, so reading it directly is the
# intended use rather than a boundary crossing.

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

from soar_lite.domain import Domain
from soar_lite.engine import PathPoint, Verdict, pointwise
from soar_lite.engine.schedule import (
    _AllStalled,
    _Best,
    _Budget,
    _Evaluator,
    _Paths,
)
from soar_lite.local_solver import LocalSolver
from tests.flock import flock, unit_domain

jax.config.update("jax_enable_x64", val=True)

# The box the runs below search, and the objective they minimise: a bowl centred
# at 0.25, so a trial at 0.2 is better than one at 0.4 and the step's argmin is
# not simply the last point served.
BOWL_MIN = 0.25


def bowl(x: jnp.ndarray) -> jnp.ndarray:
    """Return the per-point objective: ``(d,) -> ()``."""
    return jnp.sum(jnp.square(x - BOWL_MIN))


class Rigged:
    """An objective that serves its value until the test kills it.

    A run whose very first batch is non-finite never places a flock, so reaching
    "accepted, but the step served no finite sample" needs a value that is live
    when the flock lands and dead afterwards.
    """

    def __init__(self) -> None:
        """Start live: the flock this objective is handed lands."""
        self.dead = False

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return the bowl's value, or NaN once `dead` is set."""
        return jnp.asarray(jnp.nan) if self.dead else bowl(x)


class Script:
    """A `LocalSolver` that walks a caller's script, one call at a time.

    `moves` is consumed by `propose` and `rules` by `verdict`, both in call
    order, so a test states a step as the two lists it already is. The points the
    engine hands over (`restart`, `fold`) are kept in `got`, and the points the
    method was asked to judge in `judged`: those are the two facts a test needs
    about the commit, and neither requires reading the path's own state.
    """

    def __init__(
        self,
        moves: list[float],
        rules: list[Verdict],
        *,
        ruled: bool = True,
    ) -> None:
        """Keep the script, and start with nothing handed over.

        Args:
            moves: one raw candidate per `propose` call, in call order.
            rules: one verdict per judged trial, in call order.
            ruled: whether this path's step is judged at all.

        """
        self._moves = list(moves)
        self._rules = list(rules)
        self._ruled = ruled
        self.got: list[PathPoint] = []
        self.judged: list[tuple[jnp.ndarray, float]] = []

    def restart(self, i: int, p: PathPoint) -> None:
        """Keep the point the engine seeded path `i` with."""
        del i  # one path per run below: the index carries nothing extra
        self.got.append(p)

    def fold(self, i: int, p: PathPoint) -> None:
        """Keep the point the engine committed for path `i`."""
        del i
        self.got.append(p)

    def needs_verdict(self, i: int) -> bool:
        """Return whether this run's step is judged."""
        del i
        return self._ruled

    def propose(self, i: int) -> jnp.ndarray:
        """Return the script's next raw candidate."""
        del i
        return jnp.asarray([self._moves.pop(0)])

    def verdict(self, i: int, x: jnp.ndarray, f: float) -> Verdict:
        """Return the script's next ruling, keeping the served point."""
        del i
        self.judged.append((x, f))
        return self._rules.pop(0)


def model_at(x: jnp.ndarray) -> Callable[[int, Domain], jnp.ndarray]:
    """Return a restart source that proposes one fixed row, `n_points` times."""

    def propose(n_points: int, domain: Domain) -> jnp.ndarray:
        del domain  # the engine clips; this stub only repeats its row
        return jnp.repeat(x[None, :], n_points, axis=0)

    return propose


def paths(
    solver: LocalSolver,
    *,
    objective: Callable[[jnp.ndarray], jnp.ndarray] = bowl,
    x0: jnp.ndarray | None = None,
    restart_at: jnp.ndarray | None = None,
    value_only: bool = True,
) -> tuple[_Paths, _Budget]:
    """Return a state machine around `solver`, plus the meter it charges.

    No horizon and no clock: `_Budget`'s three gates are the entry's business and
    this suite never wants one closing, so the run's length is however many
    `move` calls the test makes. `value_only` picks which seam serves a trial -
    the value-only one when present, which is what lets a step's rejected trials
    be paid for without their gradients.

    Args:
        solver: the method to drive, ordinarily a `Script`.
        objective: the per-point objective behind both evaluation seams.
        x0: the caller's flock, consumed by the first `move` when handed in.
        restart_at: the row a restart proposes, zeros when None.
        value_only: whether the value-only seam is present.

    Returns:
        The state machine and the meter it charges, so a test can name a
        counter's pair.

    """
    eval_f_g, eval_f = pointwise(objective)
    budget = _Budget(None, None, None)
    domain = unit_domain()
    n_paths = 1 if x0 is None else int(x0.shape[0])
    evaluator = _Evaluator(
        eval_f_g,
        eval_f if value_only else None,
        budget=budget,
        best=_Best(domain.d),
        observe=lambda _x, _f, _g: None,
    )
    restart_at = jnp.zeros(domain.d) if restart_at is None else restart_at
    started = _Paths(
        n_paths,
        domain,
        evaluator,
        model_at(restart_at),
        solver,
        x0=x0,
        restart_stalled=False,
    )
    return started, budget


def placed(solver: LocalSolver) -> tuple[_Paths, _Budget, float]:
    """Return a run one `move` in: the flock planted at 0.6, nothing served yet."""
    start = jnp.asarray([[0.6]])
    run, budget = paths(solver, x0=start)
    run.move()
    return run, budget, float(start[0, 0])


def test_a_retry_then_an_accept_commits_the_steps_argmin() -> None:
    """The commit is the lowest finite trial served, not the last one served."""
    solver = Script([0.2, 0.4], ["retry", "accept"])
    run, budget, start = placed(solver)
    assert float(run.points[0].x[0]) == pytest.approx(start)

    run.move()

    # 0.4 was served last and is worse than 0.2, so a commit of the last served
    # point would land on the wrong side of the bowl's centre.
    assert float(run.points[0].x[0]) == pytest.approx(0.2)
    assert not run.stalled[0]
    assert solver.got[-1] is run.points[0]
    # Two value-only turns, one served point each, then one fused batch buying
    # the winner's gradient: an iteration costs one gradient pass, not one per
    # trial.
    assert (budget.n_f_batches, budget.n_f_points) == (2, 2)
    assert (budget.n_f_g_batches, budget.n_f_g_points) == (2, 2)


@pytest.mark.parametrize(
    ("kill_before_step", "rules"),
    [
        (False, ["give_up"]),  # the method's own stopping test fired
        (True, ["accept"]),  # accepted, but the step served no finite sample
    ],
)
def test_a_step_with_nothing_to_win_stalls_without_buying_a_gradient(
    *,
    kill_before_step: bool,
    rules: list[Verdict],
) -> None:
    """Both routes into a stall keep the point the path holds and charge no g."""
    start = 0.6
    objective = Rigged()
    solver = Script([0.2], rules)
    run, budget = paths(solver, objective=objective, x0=jnp.asarray([[start]]))
    run.move()  # plants the flock: one fused batch, nothing served for value
    assert (budget.n_f_g_batches, budget.n_f_g_points) == (1, 1)
    objective.dead = kill_before_step

    run.move()  # the step, which finds nothing to win

    assert run.stalled[0]
    # Where the path stood when its step failed is still where it stands: a
    # stalled path is not moved, and it is not re-drawn either.
    assert float(run.points[0].x[0]) == pytest.approx(start)
    assert solver.got == [run.points[0]]  # seeded once, never folded
    # The stall bought nothing: the only fused batch was the one that placed the
    # flock, and the trial's value came from the value-only seam.
    assert (budget.n_f_g_batches, budget.n_f_g_points) == (1, 1)
    assert (budget.n_f_batches, budget.n_f_points) == (1, 1)


def test_the_unjudged_lane_commits_its_proposal_unread() -> None:
    """A path whose method needs no verdict buys its proposal's f with its g."""
    solver = Script([0.2], [], ruled=False)
    run, budget, _ = placed(solver)

    run.move()

    assert float(run.points[0].x[0]) == pytest.approx(0.2)
    assert not run.stalled[0]
    assert solver.got[-1] is run.points[0]
    # Nothing was served for its value alone: the value came with the gradient.
    assert (budget.n_f_batches, budget.n_f_points) == (0, 0)
    assert (budget.n_f_g_batches, budget.n_f_g_points) == (2, 2)


def test_the_engine_clips_every_served_coordinate() -> None:
    """The method judges the point the engine served, not the one it asked for."""
    solver = Script([-5.0], ["accept"])
    run, _, _ = placed(solver)

    run.move()

    served_x, _ = solver.judged[0]
    assert float(served_x[0]) == pytest.approx(0.0)  # the box's lower wall
    assert float(run.points[0].x[0]) == pytest.approx(0.0)


def test_an_unusable_start_is_refused_rather_than_placed() -> None:
    """A start the engine cannot use leaves the path with no point and no seed."""
    solver = Script([], [])
    run, budget = paths(solver, restart_at=jnp.asarray([math.nan]))

    run.move()

    assert run.stalled[0]
    assert all(math.isnan(float(v)) for v in run.points[0].x)
    # Refused before the method was told anything: it holds no point either.
    assert solver.got == []
    # The batch was still served and charged - a refusal is about the point, not
    # about the work the run already paid for.
    assert (budget.n_f_g_batches, budget.n_f_g_points) == (1, 1)


def test_a_flock_with_nothing_left_to_do_ends_the_run() -> None:
    """With no re-draws, an entirely stalled flock is terminal, not idle."""
    solver = Script([0.2], ["give_up"])
    run, _, _ = placed(solver)
    run.move()
    assert run.stalled[0]

    with pytest.raises(_AllStalled):
        run.move()


def test_the_caller_flock_is_placed_in_one_batch() -> None:
    """Iteration 0 hands `x0` over whole: no model, one fused batch."""
    starts = flock(3)
    solver = Script([], [])
    run, budget = paths(solver, x0=starts)

    run.move()

    for i in range(3):
        assert float(run.points[i].x[0]) == pytest.approx(float(starts[i][0]))
    assert run.stalled == [False, False, False]
    assert [point.x[0] for point in solver.got] == pytest.approx(list(starts[:, 0]))
    assert (budget.n_f_g_batches, budget.n_f_g_points) == (1, 3)
