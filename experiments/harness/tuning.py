"""Committed tuning records: one JSON per method, written by that method's sweep.

A pick is the grid argmin of the curve the sweep measured - a pure function of
the measurement, ties going to the smallest rate. Nothing here is hand-edited: a
number a reader cannot re-derive is a number no one can check.

The record sits beside the code because it is an input - `results/` is
regenerated, this tree is committed and read - and it carries the curve next to
the pick, so a reader can tell a tie from a grid edge without re-running the
sweep.

Refresh: ``--lr-sweep``.
"""

import json
from pathlib import Path
from typing import Any

TUNING = Path(__file__).resolve().parents[1] / "tuning"

# The grid the Adam sweep scans, ascending. The sweep reads it from here and
# records it, so widening the grid is one edit here plus a re-sweep.
ADAM_GRID: list[float] = [1e-4, 1e-3, 1e-2, 1e-1, 2e-1, 5e-1, 1.0, 2.0]


def grid(method: str) -> list[float]:
    """Read back the grid the record's curves were measured on, ascending."""
    return [float(lr) for lr in _record(method)["grid"]]


def picks(method: str) -> dict[str, float]:
    """Return every problem's pick for ``method``, keyed by problem name."""
    table = _record(method)["problems"]
    return {name: float(entry["lr"]) for name, entry in table.items()}


def pick(method: str, problem_name: str) -> float:
    """Look up the tuned value ``method`` uses on ``problem_name``.

    Raises:
        KeyError: when the record has no entry for that problem, which is what
            a renamed or newly added problem looks like from here.

    """
    table = picks(method)
    if problem_name not in table:
        msg = (
            f"{_path(method)} has no {method} pick for {problem_name!r}; "
            "run --lr-sweep to record one"
        )
        raise KeyError(msg)
    return table[problem_name]


def save(
    method: str,
    *,
    grid: list[float],
    seeds: int,
    max_batches: int,
    curves: dict[str, list[float]],
) -> Path:
    """Write one method's record: the curves, and the pick each one argmins to.

    Args:
        method: The method whose tuning this is; it names the file.
        grid: The rates swept, ascending; every curve has one entry per rate.
        seeds: Seeds each rate was measured over.
        max_batches: Batches each arm ran.
        curves: Measured best-so-far per problem, aligned with ``grid``.

    Returns:
        The file written.

    """
    record = {
        "method": method,
        "grid": grid,
        "seeds": seeds,
        "max_batches": max_batches,
        "problems": {
            name: {"lr": _argmin(grid, fins), "fin": fins}
            for name, fins in curves.items()
        },
    }
    path = _path(method)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def refuse_other_budget(method: str, *, seeds: int, max_batches: int) -> None:
    """Refuse to rerecord over a file measured at another budget.

    The record is a committed input, so a quick sweep would replace real picks
    with 1-seed ones. Same rule as a trace - `refuse_recorded_budget` - and the
    same budget reader; only the file is this module's own.
    """
    # Deferred: `record` imports `arms`, which imports this module for `pick`, so
    # the two can only meet one way at import time.
    from experiments.harness.record import budget_of, refuse_recorded_budget  # noqa: PLC0415

    refuse_recorded_budget(
        _path(method),
        read=budget_of,
        n_seeds=seeds,
        max_batches=max_batches,
    )


def _argmin(grid: list[float], fins: list[float]) -> float:
    """Return the lowest measured rate; ``index`` takes the first, so ties go small."""
    return grid[fins.index(min(fins))]


def _path(method: str) -> Path:
    """Where one method's record lives."""
    return TUNING / f"{method}.json"


def _record(method: str) -> dict[str, Any]:
    """Read one method's record, parsed.

    Raises:
        FileNotFoundError: when the method has no record at all.
        ValueError: when the file exists but is not readable JSON.

    """
    path = _path(method)
    if not path.is_file():
        msg = f"no tuning record for {method!r}: {path} does not exist"
        raise FileNotFoundError(msg)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        msg = f"unreadable tuning record: {path} ({exc!r})"
        raise ValueError(msg) from exc
