"""The record: one file per (problem, arm), and the budget it was measured at.

`experiments/results/<problem>/<arm-key>.json` holds that arm's raw per-seed
trace, not a resampled grid - the grid is a plotting decision that depends on
which other arms are present.

The budget is part of the record (`meta.seeds`, `meta.max_steps`): a run that
asks for a different budget is refused rather than allowed to overwrite.

The two roots are module globals read at call time, so a test can point one run
at its own directory (`roots_for` is the only place that names them).
"""

import json
from dataclasses import dataclass
from pathlib import Path

from experiments import problems
from experiments.arms import Arm
from experiments.methods import Trace

HERE = Path(__file__).parent
RECORD_RESULTS = HERE / "results"  # traces only: one file per (problem, arm)
RECORD_FIGURES = HERE / "figures"  # one directory per panel


@dataclass(frozen=True)
class Loaded:
    """One arm's traces as read back from disk, with the budget they ran at."""

    traces: list[Trace]
    seeds: int
    max_steps: int


@dataclass(frozen=True)
class Roots:
    """Where one run reads and writes: the record, or a namespace inside it.

    The arm key and the file name encode no budget, so a 2-seed smoke trace
    carries exactly the names a full run does and would replace it. A namespace
    is how a smoke check or a notebook demo gets roots of its own; `roots_for`
    builds one, and nothing outside `main` decides where a run writes.
    """

    results: Path  # one file per (problem, arm)
    figures: Path  # one directory per panel

    def trace(self, problem_name: str, arm_key: str) -> Path:
        """Where one arm's trace lives: per problem, one file per arm."""
        return self.results / problem_name / f"{arm_key}.json"

    def panel_dir(self, panel: str) -> Path:
        """Where one panel's figures go."""
        return self.figures / panel


def save_arm(  # noqa: PLR0913 - one parameter per fact the file records
    roots: Roots,
    prob: problems.Problem,
    arm: Arm,
    traces: list[Trace],
    *,
    n_seeds: int,
    max_steps: int,
) -> Path:
    """Write one arm's raw per-seed traces (no resampling: that is plotting's)."""
    path = roots.trace(prob.name, arm.key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "problem": prob.name,
            "arm": arm.key,
            "label": arm.label,
            "description": prob.description,
            "d": prob.d,
            "f_star": prob.f_star,
            "seeds": n_seeds,
            "max_steps": max_steps,
        },
        "seeds": [
            {
                "points": t.points,
                "batches": t.batches,
                "best": t.best,
                "fun": t.fun,
            }
            for t in traces
        ],
    }
    path.write_text(json.dumps(payload, indent=1))
    return path


def load_arm(roots: Roots, problem_name: str, arm_key: str) -> Loaded | None:
    """Read an arm's traces, or None when nothing has measured it at this key."""
    try:
        payload = json.loads(roots.trace(problem_name, arm_key).read_text())
    except (OSError, ValueError):
        return None
    meta = payload["meta"]
    return Loaded(
        traces=[
            Trace(
                points=s["points"],
                batches=s["batches"],
                best=s["best"],
                fun=s["fun"],
            )
            for s in payload["seeds"]
        ],
        seeds=int(meta["seeds"]),
        max_steps=int(meta["max_steps"]),
    )


def refuse_budget_change(
    roots: Roots,
    problem_name: str,
    arm_key: str,
    *,
    n_seeds: int,
    max_steps: int,
) -> None:
    """Stop before the compute if this file already holds another budget.

    A trace file is named ``(problem, arm)``, which says nothing about the
    budget it was measured at, so a run at another ``--seeds`` /
    ``--max-steps`` would replace a record with numbers that no longer belong
    to the entry it came from. The file *is* the record, so replacing it has to
    be a deliberate delete rather than a side effect of typing the wrong flag.

    Args:
        roots: The namespace to read the existing trace from.
        problem_name: The problem half of the file's identity.
        arm_key: The arm half of the file's identity.
        n_seeds: Seeds this run will measure.
        max_steps: Batches this run will serve per method.

    Raises:
        SystemExit: the file on disk holds a different ``(seeds, max_steps)``.
            Nothing has been computed when this fires.

    """
    path = roots.trace(problem_name, arm_key)
    try:
        meta = json.loads(path.read_text())["meta"]
        on_disk = (int(meta["seeds"]), int(meta["max_steps"]))
    except (OSError, ValueError, KeyError, TypeError):
        return
    if on_disk == (n_seeds, max_steps):
        return
    shown = path.relative_to(HERE) if path.is_relative_to(HERE) else path
    msg = (
        f"budget conflict: {shown} holds "
        f"{on_disk[0]} seeds x {on_disk[1]} batches, this run asks for "
        f"{n_seeds} x {max_steps}. Nothing was written.\n"
        f"  same budget: pass the same --seeds and --max-steps.\n"
        f"  replacing the record on purpose: delete that file and rerun."
    )
    raise SystemExit(msg)


def roots_for(namespace: str) -> Roots:
    """Return the two roots ``namespace`` names; ``""`` is the record itself.

    Neither the arm key nor the file name encodes the budget, so a 2-seed,
    300-batch trace carries exactly the names a full run does and would replace
    it. A namespace is how a smoke check or a notebook demo gets roots of its
    own.

    Both roots are derived from `_RECORD_*` rather than from wherever a previous
    run pointed, so a second call in one process - a notebook cell re-run -
    lands where the first did, not inside it.

    Args:
        namespace: One path component, used as a subdirectory of both roots.

    Returns:
        The roots this run reads and writes.

    Raises:
        SystemExit: ``namespace`` is not a single component, or is dot-named.
            Nothing has been written when this fires.

    """
    if not namespace:
        return Roots(RECORD_RESULTS, RECORD_FIGURES)
    if Path(namespace).name != namespace or namespace in {".", ".."}:
        msg = (
            f"--out takes one directory name, not a path: got {namespace!r}. "
            f"Both roots get it as a subdirectory (results/{namespace}, "
            f"figures/{namespace})."
        )
        raise SystemExit(msg)
    return Roots(RECORD_RESULTS / namespace, RECORD_FIGURES / namespace)
