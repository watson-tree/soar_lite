"""`_Paths.move` one step at a time.

The end-to-end suites reach `move` through `n_path_schedule`, so what they can
see is a whole run's arithmetic: the horizon, the three stop gates and the
assembled report all sit between the caller and a single iteration's decisions.
This suite drives `move` directly - no horizon, no clock, no report - so each
statement about where a path stands, which counters moved and what the method was
told becomes one test.

That needs no seam: `_Paths` takes all five of its collaborators injected, and
`_Meter`, `_Best` and `_EvalSeams` are a few lines each to build. Every run here
is one path in the unit box unless it says otherwise, and every step is stated as
the script it is - the candidates `propose` hands back, in call order.

The three promises this pins, which no whole-run test can see cheaply:

- a winner the method never read is graded by one fused batch, and a winner the
  method read through `serve_trials` only still owes its gradient;
- an iteration buys its gradients once, covering restart starts and step winners
  in the same batch;
- no seam is clipped: a start or a step that lands off the box is refused by the
  engine, which names the offending row instead of moving it.
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
from soar_lite.engine import PathPoint, Point, ServeTrials, pointwise
from soar_lite.engine.schedule import (
    _AllStalled,
    _Best,
    _EvalSeams,
    _Meter,
    _Paths,
)
from soar_lite.local_solver import LocalSolver
from tests.flock import flock, unit_domain

jax.config.update("jax_enable_x64", val=True)

# The box the runs below search, and the objective they minimise: a bowl centred
# at 0.25, so a step from 0.6 towards 0.2 is an improvement and a step towards
# 0.4 is not.
BOWL_MIN = 0.25


def bowl(x: jnp.ndarray) -> jnp.ndarray:
    """Return the per-point objective: ``(d,) -> ()``."""
    return jnp.sum(jnp.square(x - BOWL_MIN))


def nan_at_bowl_floor(x: jnp.ndarray) -> jnp.ndarray:
    """Return the bowl's value above 0.5 and NaN below it.

    A run whose very first batch is non-finite never places a flock, so reaching
    "a winner the engine cannot use" needs a value that is live where the flock
    lands and dead where the step goes.
    """
    return jnp.where(x[0] > 0.5, bowl(x), jnp.asarray(jnp.nan))


class Script:
    """A `LocalSolver` that walks a caller's script, one call at a time.

    `candidates` is consumed by `propose` in call order, one entry per path, so a test
    states a step as the list it already is: a number is a raw candidate for that
    path, and ``None`` is its method's stopping test firing.

    `serve` picks what the returned point already holds. False hands the
    candidate back unread, the shape a method that takes one full step returns.
    True reads the candidate's value through `serve_trials` first, the shape a
    method that enumerates its step returns - so its winner still owes only its
    gradient.

    The points the engine hands over (`restart`, `step`) are kept in `got`, and
    the values the method served in `served`: those are the two facts a test
    needs about the commit, and neither requires reading the path's own state.
    """

    def __init__(self, candidates: list[float | None], *, serve: bool = False) -> None:
        """Keep the script, and start with nothing handed over.

        Args:
            candidates: one raw candidate per path per `propose` call, in call order.
            serve: whether a candidate is read for its value before it is
                returned.

        """
        self._candidates = list(candidates)
        self._serve = serve
        self.got: list[PathPoint] = []
        self.served: list[float] = []

    def restart(self, points: dict[int, PathPoint]) -> None:
        """Keep the starts the engine placed, one per path of the batch."""
        self.got.extend(points.values())

    def step(self, points: dict[int, PathPoint]) -> None:
        """Keep the points the engine committed, one per path of the batch."""
        self.got.extend(points.values())

    def propose(
        self, paths: list[int], domain: Domain, serve_trials: ServeTrials
    ) -> dict[int, Point | None]:
        """Return the script's next entry for each path asked about."""
        del domain  # the script states a candidate verbatim: the box is not its job
        out: dict[int, Point | None] = {}
        for i in paths:
            candidate = self._candidates.pop(0)
            if candidate is None:
                out[i] = None
                continue
            x = jnp.asarray([candidate])
            if self._serve:
                (read,) = serve_trials(x[None, :])
                self.served.append(read.f)
                out[i] = read
            else:
                out[i] = Point(x)
        return out


class _Stray(Script):
    """A method that answers for a path this iteration never offered it."""

    def propose(
        self, paths: list[int], domain: Domain, serve_trials: ServeTrials
    ) -> dict[int, Point | None]:
        """Answer one path past every path asked about."""
        del domain, serve_trials  # one fixed answer, and no trial to serve
        return {i + 1: Point(jnp.asarray([0.5])) for i in paths}


class _Misfit(Script):
    """A method that answers with a point of the wrong width."""

    def propose(
        self, paths: list[int], domain: Domain, serve_trials: ServeTrials
    ) -> dict[int, Point | None]:
        """Answer every path with a two-dimensional point."""
        del domain, serve_trials
        return {i: Point(jnp.zeros(2)) for i in paths}


def model_at(x: jnp.ndarray) -> Callable[[int, Domain], jnp.ndarray]:
    """Return a restart source that proposes one fixed row, `n_points` times."""

    def propose(n_points: int, domain: Domain) -> jnp.ndarray:
        del domain  # this stub only repeats its row; the box is the engine's to check
        return jnp.repeat(x[None, :], n_points, axis=0)

    return propose


def paths(  # noqa: PLR0913 - each knob names one axis a test varies
    solver: LocalSolver,
    *,
    objective: Callable[[jnp.ndarray], jnp.ndarray] = bowl,
    x0: jnp.ndarray | None = None,
    restart_at: jnp.ndarray | None = None,
    value_only: bool = True,
    restart_stalled: bool = False,
) -> tuple[_Paths, _Meter]:
    """Return a state machine around `solver`, plus the meter it charges.

    No horizon and no clock: `_Meter`'s three gates are the entry's business and
    this suite never wants one closing, so the run's length is however many
    `move` calls the test makes. `value_only` picks which seam serves a trial -
    the value-only one when present, which is what lets a method's read trials be
    paid for without their gradients.

    Args:
        solver: the method to drive, ordinarily a `Script`.
        objective: the per-point objective behind both evaluation seams.
        x0: the caller's flock, consumed by the first `move` when handed in.
        restart_at: the row a restart proposes, zeros when None.
        value_only: whether the value-only seam is present.
        restart_stalled: whether a stalled path is re-drawn.

    Returns:
        The state machine and the meter it charges, so a test can name a
        counter's pair.

    """
    eval_f_g, eval_f = pointwise(objective)
    meter = _Meter(None, None, None)
    domain = unit_domain()
    n_paths = 1 if x0 is None else int(x0.shape[0])
    eval_seams = _EvalSeams(
        eval_f_g,
        eval_f if value_only else None,
        domain=domain,
        meter=meter,
        best=_Best(domain.d),
        observe=lambda _x, _f, _g: None,
    )
    restart_at = jnp.zeros(domain.d) if restart_at is None else restart_at
    started = _Paths(
        n_paths,
        domain,
        eval_seams,
        model_at(restart_at),
        solver,
        x0=x0,
        restart_stalled=restart_stalled,
    )
    return started, meter


def held(run: _Paths, i: int = 0) -> PathPoint:
    """Return path `i`'s point, failing here if it holds none."""
    point = run.slots[i].point
    assert point is not None
    return point


def placed(solver: LocalSolver) -> tuple[_Paths, _Meter, float]:
    """Return a run one `move` in: the flock planted at 0.6, nothing served yet."""
    start = jnp.asarray([[0.6]])
    run, meter = paths(solver, x0=start)
    run.move()
    return run, meter, float(start[0, 0])


def test_a_winner_never_read_is_graded_by_one_fused_batch() -> None:
    """The method may not read its own winner: `settle` buys both halves once."""
    solver = Script([0.2])
    run, meter, start = placed(solver)
    assert float(held(run).x[0]) == pytest.approx(start)

    run.move()

    assert float(held(run).x[0]) == pytest.approx(0.2)
    assert not run.slots[0].is_stalled
    assert solver.got[-1] is held(run)  # stepped onto the very point the engine holds
    assert solver.served == []  # nothing was read on the way in
    # Its value was never served alone: the one batch that graded the winner
    # carried f and g together.
    assert (meter.work.n_f_batches, meter.work.n_f_points) == (0, 0)
    assert (meter.work.n_f_g_batches, meter.work.n_f_g_points) == (2, 2)


def test_a_winner_read_for_its_value_still_owes_its_gradient() -> None:
    """A read trial is paid for once, and the gradient is bought afterwards."""
    solver = Script([0.2], serve=True)
    run, meter, _ = placed(solver)

    run.move()

    assert float(held(run).x[0]) == pytest.approx(0.2)
    assert solver.served == pytest.approx([bowl(jnp.asarray([0.2]))])
    # One value-only turn for the candidate, then one fused batch buying the
    # winner's gradient alone - and the path is graded, not just valued.
    assert (meter.work.n_f_batches, meter.work.n_f_points) == (1, 1)
    assert (meter.work.n_f_g_batches, meter.work.n_f_g_points) == (2, 2)
    assert math.isfinite(float(held(run).f))
    assert jnp.all(jnp.isfinite(held(run).g))


def test_a_named_stall_keeps_the_point_and_buys_nothing() -> None:
    """``None`` is the method's own stopping test: the path stays put, uncharged."""
    start = 0.6
    solver = Script([None])
    run, meter = paths(solver, x0=jnp.asarray([[start]]))
    run.move()  # plants the flock: one fused batch, nothing served for value

    run.move()  # the step, which the method declines to take

    assert run.slots[0].is_stalled
    # Where the path stood when its step failed is still where it stands: a
    # stalled path is not moved, and it is not re-drawn either.
    assert float(held(run).x[0]) == pytest.approx(start)
    assert solver.got == [held(run)]  # seeded once, never stepped
    # The stall bought nothing: the only fused batch was the one that placed the
    # flock.
    assert (meter.work.n_f_g_batches, meter.work.n_f_g_points) == (1, 1)
    assert (meter.work.n_f_batches, meter.work.n_f_points) == (0, 0)


def test_an_unusable_winner_stalls_without_stepping() -> None:
    """A winner the engine cannot grade leaves the path where it stood."""
    start = 0.6
    solver = Script([0.2])
    run, meter = paths(solver, objective=nan_at_bowl_floor, x0=jnp.asarray([[start]]))
    run.move()  # plants the flock; 0.6 is above the NaN floor

    run.move()  # the step, whose winner grades to NaN

    assert run.slots[0].is_stalled
    assert float(held(run).x[0]) == pytest.approx(start)
    assert solver.got == [held(run)]  # the refused winner was never stepped
    # The winner was still served and charged: a refusal is about the point, not
    # about the work the run already paid for.
    assert (meter.work.n_f_g_batches, meter.work.n_f_g_points) == (2, 2)


def test_a_start_off_the_box_is_refused_by_the_engine() -> None:
    """A model's start outside the box is named, not slid onto the wall."""
    solver = Script([])
    run, _ = paths(solver, restart_at=jnp.asarray([5.0]))

    with pytest.raises(
        ValueError,
        match=r"the global model's restart ask row 0 is 5 in dimension 0, outside",
    ):
        run.move()

    assert run.slots[0].point is None
    assert solver.got == []  # nothing was placed, so nothing was told


def test_a_step_off_the_box_is_refused_rather_than_slid() -> None:
    """A step is the method's own, and the engine checks it instead of clipping."""
    solver = Script([-5.0])
    run, _, _ = placed(solver)

    with pytest.raises(
        ValueError, match=r"local_solver's step on path 0 is -5 in dimension 0, outside"
    ):
        run.move()

    # The path keeps the point it held, and the method was told nothing about
    # the winner it never landed on.
    assert float(held(run).x[0]) == pytest.approx(0.6)
    assert solver.got == [held(run)]


def test_a_trial_row_off_the_box_is_refused_before_it_is_served() -> None:
    """A method's own trial grid is checked like any other point the run holds."""
    solver = Script([5.0], serve=True)
    run, _, _ = placed(solver)

    with pytest.raises(
        ValueError, match=r"local_solver's trial batch row 0 is 5 in dimension 0"
    ):
        run.move()

    assert solver.served == []  # refused before the batch was paid for


def test_an_answer_for_a_path_nobody_asked_about_is_refused() -> None:
    """The winner map is keyed by path: a stray key is a bug, not an extra move."""
    run, _, _ = placed(_Stray([]))

    with pytest.raises(ValueError, match=r"answered for path 1, which this iteration"):
        run.move()


def test_a_winner_of_the_wrong_width_is_refused() -> None:
    """A step is one ``(d,)`` point, so a wider answer is named at the seam."""
    run, _, _ = placed(_Misfit([]))

    with pytest.raises(ValueError, match=r"must be a \(1,\) point, got shape \(2,\)"):
        run.move()


def test_a_non_finite_start_is_refused_by_the_engine() -> None:
    """A NaN start is named rather than served and stalled."""
    solver = Script([])
    run, meter = paths(solver, restart_at=jnp.asarray([math.nan]))

    with pytest.raises(ValueError, match=r"row 0 is nan in dimension 0, not finite"):
        run.move()

    assert run.slots[0].point is None
    assert solver.got == []
    # Refused before anything was served: the box test is the run's own.
    assert (meter.work.n_f_g_batches, meter.work.n_f_g_points) == (0, 0)


def test_a_flock_with_nothing_left_to_do_ends_the_run() -> None:
    """With no re-draws, an entirely stalled flock is terminal, not idle."""
    solver = Script([None])
    run, _, _ = placed(solver)
    run.move()
    assert run.slots[0].is_stalled

    with pytest.raises(_AllStalled):
        run.move()


def test_the_caller_flock_is_placed_in_one_batch() -> None:
    """Iteration 0 hands `x0` over whole: no model, one fused batch."""
    starts = flock(3)
    solver = Script([])
    run, meter = paths(solver, x0=starts)

    run.move()

    for i in range(3):
        assert float(held(run, i).x[0]) == pytest.approx(float(starts[i][0]))
    assert [slot.is_stalled for slot in run.slots] == [False] * 3
    assert [float(point.x[0]) for point in solver.got] == pytest.approx(
        [float(start) for start in starts[:, 0]]
    )
    assert (meter.work.n_f_g_batches, meter.work.n_f_g_points) == (1, 3)


def test_one_batch_grades_the_starts_and_the_winners_together() -> None:
    """A restart and a step in the same iteration share one gradient pass."""
    starts = jnp.asarray([[0.6], [0.6]])
    solver = Script([None, 0.2, 0.4])
    run, meter = paths(solver, x0=starts, restart_stalled=True)
    run.move()  # plants both paths: one fused batch of two
    assert (meter.work.n_f_g_batches, meter.work.n_f_g_points) == (1, 2)

    run.move()  # path 0 declines, path 1 wins
    assert run.slots[0].is_stalled
    assert (meter.work.n_f_g_batches, meter.work.n_f_g_points) == (2, 3)

    run.move()  # path 0 is re-drawn, path 1 steps again

    # One batch, not two: the re-drawn start and the step winner were both owed
    # their gradient, so `settle` bought them together.
    assert (meter.work.n_f_g_batches, meter.work.n_f_g_points) == (3, 5)
    assert not run.slots[0].is_stalled
    assert float(held(run, 0).x[0]) == pytest.approx(0.0)  # the restart row
    assert float(held(run, 1).x[0]) == pytest.approx(0.4)
