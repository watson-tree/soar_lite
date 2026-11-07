"""Measure the arms on each panel, write per-arm traces, draw figures.

Storage is one file per (problem, arm):
`experiments/results/<problem>/<arm-key>.json` holds that arm's raw per-seed
trace, not a resampled grid - the grid is a plotting decision that depends on
which other arms are present. A panel is a *roster over those files*, so
re-measuring one arm rewrites exactly one file and every panel listing that arm
picks the new numbers up.

The arm key spells out the arm's whole configuration
(`soar-sample-np1-mu0-restart-on`), never a path through a default it could
have inherited. An engine default moving therefore changes the key and the file
path: the old trace stays as a record of the old configuration.

The budget is part of the record (`meta.seeds`, `meta.max_steps`): a run that
asks for a different budget is refused rather than allowed to overwrite.

Panels and their propositions: `DESIGN.md`. Commands and storage layout:
`README.md`. What the last full run measured: `RESULTS.md`.
"""

import argparse
import json
import textwrap
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import jax.numpy as jnp
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from experiments import problems
from experiments.methods import (
    Trace,
    run_adam,
    run_lbfgs,
    run_soar_lite,
    surrogate_minimize,
)
from soar_lite import (
    BLRGlobalModel,
    DiagonalPreconditioner,
    GlobalModel,
    IdentityProjection,
    SphereProjection,
)

HERE = Path(__file__).parent
RESULTS = HERE / "results"  # traces only: one file per (problem, arm)
FIGURES = HERE / "figures"  # one directory per panel
Runner: TypeAlias = Callable[..., Trace]


# --- arms: (identity, runner, presentation) ----------------------------------


@dataclass(frozen=True)
class Arm:
    """One curve: what to run, and how to draw it.

    ``key`` is the arm's identity on disk, so it names every attribute the
    runner pins - including the ones the runner could have left to a default.
    ``label`` and ``style`` are presentation only: two panels may draw the same
    key under different labels, and do.
    """

    key: str
    label: str
    runner: Runner
    style: dict[str, Any]


# The preconditioner's own momentum default. Arms that take no knobs read it
# from the class instead of restating it, so a default move re-keys them (a new
# file) rather than silently changing what an existing file means.
DEFAULT_MU = DiagonalPreconditioner().mu

LABEL_SERIAL = f"soar-lite n_paths=1, mu={DEFAULT_MU:g} (default)"
LABEL_PARALLEL = f"soar-lite n_paths=16, mu={DEFAULT_MU:g}"
LABEL_MOMENTUM_09 = "soar-lite n_paths=1, mu=0.9 (Adam's window)"
# The blr panel's arms: one method, three restart models, so nothing but where
# a restart lands differs.
LABEL_BLR_SAMPLE = "n_paths=16, Sample draws"
LABEL_BLR_IDENTITY = "n_paths=16, BLR identity"
LABEL_BLR_SPHERE = "n_paths=16, BLR sphere"

# Style grammar: linestyle encodes `n_paths` (one solid line per width: 1 ->
# solid, 16 -> dashed) and color encodes the momentum level. Without a rule
# like this two single-path arms read as different kinds of object because one
# of them happens to switch momentum off.
STYLE_SERIAL: dict[str, Any] = {"color": "C0", "lw": 2.0}
STYLE_PARALLEL: dict[str, Any] = {"color": "C0", "lw": 2.0, "ls": "--"}
STYLE_MOMENTUM_09: dict[str, Any] = {"color": "C1", "lw": 1.5}


def _model_for(model: str) -> Callable[[jnp.ndarray], GlobalModel] | None:
    """Return the restart-model factory a key's model part names.

    ``sample`` is the engine's own uniform draw, so it answers None.
    """
    if model == "sample":
        return None
    projections = {
        "blr-identity": IdentityProjection,
        "blr-sphere": SphereProjection,
    }
    try:
        projection = projections[model]
    except KeyError:
        msg = f"unknown restart model {model!r}"
        raise SystemExit(msg) from None
    return lambda key: BLRGlobalModel(
        projection_factory=projection,
        minimize=surrogate_minimize,
        key=key,
    )


def soar_arm(  # noqa: PLR0913 - one keyword per attribute the key must name
    *,
    n_paths: int,
    mu: float,
    label: str,
    style: dict[str, Any],
    model: str = "sample",
    restart_stalled: bool = True,
) -> Arm:
    """Build one soar-lite arm, its key derived from exactly these attributes."""
    key = (
        f"soar-{model}-np{n_paths}-mu{mu:g}"
        f"-restart-{'on' if restart_stalled else 'off'}"
    )

    def runner(prob: problems.Problem, *, max_steps: int, seed: int) -> Trace:
        return run_soar_lite(
            prob,
            n_paths=n_paths,
            max_steps=max_steps,
            seed=seed,
            restart_stalled=restart_stalled,
            mu=mu,
            model=_model_for(model),
        )

    return Arm(key=key, label=label, runner=runner, style=style)


# Per-problem Adam learning rate, the pick the panels use (see --lr-sweep).
# The picks span over two orders of magnitude across the panels, and lr=1e-4
# costs 52 orders of magnitude on the diagonal quadratic against its pick.
# Tuning is itself budget the y axis never shows -- every trial is more
# objective points served, and it is manual, not adaptive; soar-lite has no
# learning rate at all. Two optima sit at the sweep grid's right edge (the
# rotated quadratic and the oscillator keep improving to lr=2.0). On the
# diagonal quadratic and Ackley every grid point from 1e-2 up is a floor or a
# plateau, so those two picks are one point of a tie, not a strict argmin; on
# rastrigin the floor is flat over 1e-3..2e-1 and 0.5 sits ~5% below it. The
# compartment chain is the one strict interior argmin: its curve falls from
# 5.0 at lr=1e-2 to 1.1e-3 at 1e-1 and back up to 2.8e-2 at 2e-1, so there the
# pick is a real optimum rather than an edge or a tie.
ADAM_LR: dict[str, float] = {
    "oscillator_n20_F20": 2.0,
    "pk_chain_n21_d61": 1e-1,
    "diag_quad_d100_cond10000": 1e-2,
    "rot_quad_d100_cond1e+06": 2.0,
    "ackley_d100": 1e-2,
    "rastrigin_d100": 0.5,
}


def _adam_lr(prob: problems.Problem) -> float:
    """Return the grid-tuned Adam lr for this problem.

    There is no fallback: `_refuse_untuned_problems` keeps the table a claim
    about the whole roster, so a miss here is a bug in that guard, not a case
    to paper over with a default - the arm key and the sweep's C0 diamond both
    read this number, and neither would say it had been guessed.
    """
    return ADAM_LR[prob.name]


def adam_arm(prob: problems.Problem) -> Arm:
    """Adam at this problem's grid-tuned lr; the lr is part of the identity."""
    lr = _adam_lr(prob)
    key = f"adam-lr{lr:g}"

    def runner(p: problems.Problem, *, max_steps: int, seed: int) -> Trace:
        return run_adam(p, lr=lr, max_steps=max_steps, seed=seed)

    return Arm(
        key=key,
        label="Adam (grid lr)",
        runner=runner,
        style={"color": "C3", "lw": 1.5},
    )


LBFGS_ARM = Arm(
    key="lbfgs",
    label="L-BFGS-B",
    runner=run_lbfgs,
    style={"color": "C7", "lw": 1.2, "ls": ":"},
)


# --- panels: rosters over arms ----------------------------------------------


def _soar_ladder() -> list[Arm]:
    """Build the soar-lite roster both proposition panels read: one knob at a time.

    A partial grid, not a chain: the momentum axis runs at `n_paths=1` (`mu` 0
    and 0.9) and one width step (`n_paths` 1 -> 16) sits at the default `mu`.
    The default is `mu=0`, so "momentum off" *is* the default arm - one curve,
    not two names reading one file. It is also what shows the diagonal sweet
    spot landing in a single step, which a `mu > 0` gives up near a minimizer -
    a cost the panel should show, not state.

    The `mu=0.9` arm reads the momentum *horizon*, and it is a comparison arm,
    not a tuned one: its window is ~1/(1-0.9) = 10 steps, the same window Adam's
    first moment runs at `b1=0.9`, so the panel can ask whether the secant beats
    Adam's RMS at Adam's momentum reach as well as at ours. Adam itself stays at
    its own defaults - a baseline whose knobs are moved to mirror the method
    under test stops being the baseline. This arm is the current, unprojected
    momentum; the `mu=0.9` column retired from the v2 tables was the *projected*
    direction, a different rule whose numbers are not these.
    """
    return [
        soar_arm(n_paths=1, mu=DEFAULT_MU, label=LABEL_SERIAL, style=STYLE_SERIAL),
        soar_arm(
            n_paths=1,
            mu=0.9,
            label=LABEL_MOMENTUM_09,
            style=STYLE_MOMENTUM_09,
        ),
        soar_arm(
            n_paths=16,
            mu=DEFAULT_MU,
            label=LABEL_PARALLEL,
            style=STYLE_PARALLEL,
        ),
    ]


def _preconditioning_arms(prob: problems.Problem) -> list[Arm]:
    """Build the P② roster: the soar-lite ladder, Adam, the L-BFGS-B ceiling."""
    return [*_soar_ladder(), adam_arm(prob), LBFGS_ARM]


def _drift_arms(prob: problems.Problem) -> list[Arm]:
    """Build the P① roster: the same ladder, to read where a stalled chain rests."""
    return [*_soar_ladder(), adam_arm(prob)]


def _blr_arms(prob: problems.Problem) -> list[Arm]:
    """Build the restart-model roster: every arm n_paths=16, only the model differs.

    The Sample arm is the same key as the parallel arm, so the two panels share
    one trace file - the consistency check is structural, not a coincidence.
    """
    del prob  # same roster on every problem; only rastrigin lists the panel
    return [
        soar_arm(
            n_paths=16,
            mu=DEFAULT_MU,
            label=LABEL_BLR_SAMPLE,
            style=STYLE_PARALLEL,
        ),
        soar_arm(
            model="blr-identity",
            n_paths=16,
            mu=DEFAULT_MU,
            label=LABEL_BLR_IDENTITY,
            style={"color": "C4", "lw": 2.0, "ls": "--"},
        ),
        soar_arm(
            model="blr-sphere",
            n_paths=16,
            mu=DEFAULT_MU,
            label=LABEL_BLR_SPHERE,
            style={"color": "C5", "lw": 2.0, "ls": "-."},
        ),
    ]


PANELS: dict[str, Callable[[problems.Problem], list[Arm]]] = {
    "headline": _preconditioning_arms,
    "smooth": _preconditioning_arms,
    "multimodal": _drift_arms,
    "blr": _blr_arms,
}

INVARIANCE_SCALE = 10.0  # the second unit system re-measures x as x' = 10 x


# --- storage: one file per (problem, arm) -----------------------------------


@dataclass(frozen=True)
class Loaded:
    """One arm's traces as read back from disk, with the budget they ran at."""

    traces: list[Trace]
    seeds: int
    max_steps: int


def trace_path(problem_name: str, arm_key: str) -> Path:
    """Where one arm's trace lives: per problem, one file per arm."""
    return RESULTS / problem_name / f"{arm_key}.json"


def save_arm(
    prob: problems.Problem,
    arm: Arm,
    traces: list[Trace],
    *,
    n_seeds: int,
    max_steps: int,
) -> Path:
    """Write one arm's raw per-seed traces (no resampling: that is plotting's)."""
    path = trace_path(prob.name, arm.key)
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


def load_arm(problem_name: str, arm_key: str) -> Loaded | None:
    """Read an arm's traces, or None when nothing has measured it at this key."""
    try:
        payload = json.loads(trace_path(problem_name, arm_key).read_text())
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
        problem_name: The problem half of the file's identity.
        arm_key: The arm half of the file's identity.
        n_seeds: Seeds this run will measure.
        max_steps: Batches this run will serve per method.

    Raises:
        SystemExit: the file on disk holds a different ``(seeds, max_steps)``.
            Nothing has been computed when this fires.

    """
    path = trace_path(problem_name, arm_key)
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


def _selection(
    panel: str,
    problems_filter: frozenset[str] | None = None,
    arms_filter: frozenset[str] | None = None,
) -> list[tuple[problems.Problem, Arm]]:
    """Select the ``(problem, arm)`` pairs of ``panel`` that both filters leave.

    A filter matches by name and nothing more: a name the panel does not carry
    selects nothing here rather than raising, so the caller compares the result
    with the filters it passed and can refuse the run when it comes back empty.
    """
    arms = PANELS[panel]
    probs = problems.build(panel)
    return [
        (prob, arm)
        for prob in probs
        if not problems_filter or prob.name in problems_filter
        for arm in arms(prob)
        if not arms_filter or arm.key in arms_filter
    ]


def run_panel(
    panel: str,
    n_seeds: int,
    max_steps: int,
    *,
    problems_filter: frozenset[str] | None = None,
    arms_filter: frozenset[str] | None = None,
) -> None:
    """Run the selected arms of ``panel``, one trace file each.

    The filters are how one arm is re-measured without its panel: they select
    on the arm key (and on the problem), so the run touches only the files it
    names and every other trace stays as it was. ``main`` refuses a selection
    that comes back empty before this is reached.
    """
    for prob, arm in _selection(panel, problems_filter, arms_filter):
        refuse_budget_change(
            prob.name,
            arm.key,
            n_seeds=n_seeds,
            max_steps=max_steps,
        )
        traces = [arm.runner(prob, max_steps=max_steps, seed=s) for s in range(n_seeds)]
        path = save_arm(prob, arm, traces, n_seeds=n_seeds, max_steps=max_steps)
        fins = [t.fun for t in traces]
        print(
            f"  {prob.name:26s} {arm.key:44s} "
            f"final mean best = {np.mean(fins):.4e}  -> {path.relative_to(HERE)}",
        )


# --- plotting ---------------------------------------------------------------

Axis: TypeAlias = Literal["points", "batches"]
"""The two cost units a panel can be drawn against.

Named as a type so a caller inside the repo gets the pair from the checker; the
comparisons below still read it at run time, since `--groups` never reaches here.
"""


def plot_panel(panel: str, out_dir: Path) -> None:
    """Draw a panel from whatever arm files exist; a missing arm is skipped."""
    probs = problems.build(panel)
    out_dir.mkdir(parents=True, exist_ok=True)
    for axis in ("points", "batches"):
        fig, axes = plt.subplots(
            1,
            len(probs),
            figsize=(6.4 * len(probs), 4.6),
            squeeze=False,
        )
        for ax, prob in zip(axes[0], probs, strict=True):
            drawn = _draw_problem(ax, panel, prob, axis)
            title = prob.description if drawn else "no trace files yet"
            ax.set_title(f"{prob.name}\n{title}", fontsize=9)
        fig.suptitle(
            f"soar-lite vs first-order baselines - {panel} panel ({axis} axis)",
        )
        fig.tight_layout()
        out = out_dir / f"{axis}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out.relative_to(HERE)}")


def _draw_problem(
    ax: Axes,
    panel: str,
    prob: problems.Problem,
    axis: Axis,
) -> bool:
    """Draw one subplot; False when the arm files it needs are all absent."""
    entries = [(arm, load_arm(prob.name, arm.key)) for arm in PANELS[panel](prob)]
    present = [(arm, loaded) for arm, loaded in entries if loaded is not None]
    missing = [arm.key for arm, loaded in entries if loaded is None]
    if missing:
        print(f"  no trace: {prob.name}: {', '.join(missing)}")
    if not present:
        return False
    # Cut every arm at the shortest extent **of the axis being drawn**: one arm
    # can be shared with a panel that runs longer, and a curve that outlives the
    # others on x reads as a comparison it is not. The unit is the axis's own -
    # batches are level across arms, points are not, since a point count is
    # exactly what n_paths inflates and an n_paths=16 arm would otherwise trail
    # 16x past every single chain. A panel whose arms are all one width is
    # untouched by this, so this cannot starve a wide arm of its own run.
    xcol = 0 if axis == "points" else 1
    xmax = min(
        max(
            float(t.as_arrays()[xcol].max()) for t in loaded.traces if len(t.points) > 0
        )
        for _, loaded in present
    )
    if len({loaded.max_steps for _, loaded in present}) > 1:
        print(f"  mixed budgets: {prob.name}: cutting all arms at {xmax:g} {axis}")
    shown = {
        arm.key: [_up_to(t, axis, xmax) for t in loaded.traces]
        for arm, loaded in present
    }
    # Log-spaced: the descent these panels are about happens in the first
    # decade of served cost, and a linear grid would compress it into the left
    # edge of the axis.
    grid = np.logspace(0.0, np.log10(xmax), 120)
    grid[-1] = xmax  # logspace rounds the last sample past its own end
    for arm, _ in present:
        med, q1, q3 = _summary(_resample(shown[arm.key], grid, axis))
        ax.plot(grid, np.maximum(med, 1e-16), label=arm.label, **arm.style)
        ax.fill_between(
            grid,
            np.maximum(q1, 1e-16),
            np.maximum(q3, 1e-16),
            color=arm.style["color"],
            alpha=0.15,
            lw=0,
        )
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("objective points served" if axis == "points" else "batches served")
    ax.set_ylabel("best-so-far (median [Q1, Q3] over seeds)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    return True


def _resample(traces: list[Trace], grid: np.ndarray, axis: Axis) -> np.ndarray:
    """Resample every seed's best-so-far onto ``grid``.

    Returns a ``(n_seeds, len(grid))`` array with NaN where a seed has no data
    (before its first batch and after it stops). Callers aggregate over the
    seed axis so each aggregation (median, quartiles) lives in one place.

    ``grid`` is the shared x-point set: each seed's step curve lives on its own
    x coordinates (different stop points, and n_paths shifts the batch axis), so
    aligning them onto one grid is what makes a per-column aggregate over seeds
    possible. ``ax.set_xscale("log")`` only changes how ticks are drawn — it
    cannot align ragged per-seed x coordinates.
    """
    stacked = []
    for trace in traces:
        points, batches, best = trace.as_arrays()
        x = points if axis == "points" else batches
        idx = np.clip(np.searchsorted(x, grid, side="right") - 1, 0, len(best) - 1)
        # The step curve exists only between a run's first and last cost unit.
        curve = np.where((grid < x[0]) | (grid > x[-1]), np.nan, best[idx])
        stacked.append(curve)
    return np.vstack(stacked)


def _summary(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median and [Q1, Q3] over seeds, NaN-ignoring. Skew-safe on a log axis."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(arr, axis=0)
        q1 = np.nanpercentile(arr, 25, axis=0)
        q3 = np.nanpercentile(arr, 75, axis=0)
    return med, q1, q3


def _up_to(trace: Trace, axis: Axis, max_x: float) -> Trace:
    """Return the part of a run that served at most ``max_x`` units of ``axis``."""
    x = trace.points if axis == "points" else trace.batches
    stop = sum(1 for v in x if v <= max_x)
    return Trace(
        points=trace.points[:stop],
        batches=trace.batches[:stop],
        best=trace.best[:stop],
        fun=trace.fun,
    )


# --- appendix panels --------------------------------------------------------


def run_lr_sweep(n_seeds: int, max_steps: int) -> None:
    """Plot best-so-far vs Adam lr, one curve per problem (appendix panel).

    Shows the optimal lr drifting across problems and a wrong lr costing many
    orders of magnitude. The C0 diamond marks the lr actually used in the
    panels (``ADAM_LR``). It agrees with the sweep's argmin on the rotated
    quadratic, the oscillator, the compartment chain and rastrigin; on the
    diagonal quadratic and Ackley every grid point sits at the same floor or
    plateau, so the pick is one point of that tie rather than a strict argmin.
    On the rotated quadratic and the oscillator the curve is still falling at
    the grid edge (lr=2.0 is the best *sampled* point, not an interior
    minimum); the compartment chain is the mirror case, an interior optimum
    that is sharp on both sides; on rastrigin the floor is flat over
    1e-4..2e-1, so its argmin is a noise-level nudge.
    """
    lr_grid = [1e-4, 1e-3, 1e-2, 1e-1, 2e-1, 5e-1, 1.0, 2.0]
    probs = problems.build("all")
    fig, axes = plt.subplots(
        1,
        len(probs),
        figsize=(6.4 * len(probs), 4.6),
        squeeze=False,
    )
    for ax, prob in zip(axes[0], probs, strict=True):
        fins = [
            float(
                np.mean(
                    [
                        run_adam(prob, max_steps=max_steps, seed=s, lr=lr).fun
                        for s in range(n_seeds)
                    ],
                ),
            )
            for lr in lr_grid
        ]
        print(
            f"{prob.name:26s} pick={_adam_lr(prob)}",
            " ".join(f"{lr:g}:{f:.6g}" for lr, f in zip(lr_grid, fins, strict=True)),
        )
        ax.plot(lr_grid, np.maximum(fins, 1e-16), marker="o", lw=1.5, color="C3")
        best_lr = _adam_lr(prob)
        best_idx = lr_grid.index(best_lr) if best_lr in lr_grid else None
        if best_idx is not None:
            ax.plot(
                [lr_grid[best_idx]],
                [max(fins[best_idx], 1e-16)],
                marker="D",
                color="C0",
                ms=9,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Adam learning rate")
        ax.set_ylabel("best-so-far (mean over seeds)")
        ax.set_title(prob.name, fontsize=9)
        ax.grid(alpha=0.25, which="both")
    fig.suptitle(
        "Adam lr sensitivity - C0 diamond = the lr the panels use; on diag and "
        "Ackley it is one point of a floor/plateau tie, not a strict argmin; "
        "rot & oscillator still falling at lr=2.0 (edge); pk_chain is a sharp "
        "interior optimum; rastrigin flat < 2e-1",
    )
    fig.tight_layout()
    out = FIGURES / "lr_sweep" / "lr_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.relative_to(HERE)}")


def run_invariance(n_seeds: int, max_steps: int) -> None:
    """Draw panel 5: one physics in two unit systems, zero retune.

    soar-lite's per-coordinate secant should overlap (outcome-level), Adam's
    single lr should separate because the effective lr rescales with the unit.
    """
    base = problems.diag_quadratic()
    scaled = _rescale(base, INVARIANCE_SCALE)
    lr = _adam_lr(base)
    serial = soar_arm(
        n_paths=1,
        mu=DEFAULT_MU,
        label=LABEL_SERIAL,
        style=STYLE_SERIAL,
    )
    methods = [
        (serial.label, "C0", serial.runner),
        (
            "Adam (same lr)",
            "C3",
            lambda p, **kw: run_adam(p, lr=lr, **kw),
        ),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for label, color, runner in methods:
        for prob, ls, tag in ((base, "-", "base"), (scaled, "--", "x10")):
            traces = [runner(prob, max_steps=max_steps, seed=s) for s in range(n_seeds)]
            # Seeds stop at different point counts (diag converges in tens of
            # points), so raw trace arrays ragged -- resample onto one
            # log-spaced grid, exactly as the other panels do.
            xmax = max(
                float(t.as_arrays()[0].max()) for t in traces if len(t.points) > 0
            )
            grid = np.logspace(0.0, np.log10(xmax), 120)
            grid[-1] = xmax
            med, q1, q3 = _summary(_resample(traces, grid, "points"))
            ax.plot(
                grid,
                np.maximum(med, 1e-16),
                color=color,
                ls=ls,
                lw=1.5,
                label=f"{label} [{tag}]",
            )
            ax.fill_between(
                grid,
                np.maximum(q1, 1e-16),
                np.maximum(q3, 1e-16),
                color=color,
                alpha=0.12,
                lw=0,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("objective points served")
    ax.set_ylabel("best-so-far (median [Q1, Q3] over seeds)")
    ax.set_title(
        f"unit rescaling x' = {INVARIANCE_SCALE:g}x, no retune (Adam lr={lr:g})",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    fig.suptitle("panel 5 - unit-rescaling consistency")
    fig.tight_layout()
    out = FIGURES / "invariance" / "invariance.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.relative_to(HERE)}")


def _rescale(prob: problems.Problem, scale: float) -> problems.Problem:
    """Return the same physics re-measured in coordinates ``x' = scale * x``.

    The box is ``scale`` wider and the gradient ``1/scale`` smaller (chain rule
    with ``x = x'/scale``); a runner on this problem is the "second unit
    system", re-tuned by exactly zero.
    """
    bounds = prob.bounds * scale

    def eval_f_g(xp: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        f, g, _ = prob.eval_f_g(xp / scale)
        return f, g / scale, None

    x_star = prob.x_star * scale if prob.x_star is not None else None
    return problems.Problem(
        name=f"{prob.name}@x{scale:g}",
        d=prob.d,
        bounds=bounds,
        eval_f_g=eval_f_g,
        f_star=prob.f_star,
        description=f"{prob.description} (x' = {scale:g} x)",
        x_star=x_star,
    )


def _comma_set(text: str) -> frozenset[str] | None:
    """Parse a comma list into a filter set, or None when it selects everything."""
    items = frozenset(item.strip() for item in text.split(",") if item.strip())
    return items or None


def _refuse_untuned_problems() -> None:
    """Refuse a problem with no entry in ``ADAM_LR``, before anything runs.

    The table is keyed by problem name, which `problems.build` and this table
    agree on by hand. A rename therefore reads as a brand-new problem: the arm
    becomes ``adam-lr0.01`` and a figure is drawn at a lr nobody tuned, with the
    number in the file name as the only trace of it.
    """
    missing = [prob.name for prob in problems.build("all") if prob.name not in ADAM_LR]
    if missing:
        msg = (
            f"no tuned Adam lr for {', '.join(missing)}; add an entry to ADAM_LR "
            f"(the lr sweep picks one)"
        )
        raise SystemExit(msg)


def _refuse_undrawable_groups(groups: list[str]) -> None:
    """Refuse a ``--groups`` name that no panel draws, before anything runs.

    ``problems.build`` knows one group this does not: ``all``, which is the union
    a sweep iterates, not a panel anyone can draw.
    """
    choices = ", ".join([*PANELS, "invariance"])
    for group in groups:
        if group != "invariance" and group not in PANELS:
            msg = f"no group {group!r}; pick one of {choices}"
            raise SystemExit(msg)


def _refuse_empty_selection(
    groups: list[str],
    problems_filter: frozenset[str] | None,
    arms_filter: frozenset[str] | None,
) -> None:
    """Refuse a ``--problems``/``--arms`` selection that matches no arm at all.

    A filter matches by name, so a typo used to finish as a clean exit with no
    arm run at all - the one failure a re-measurement cannot show. The check
    costs no batches: it reads the panels, never the objective. An unknown group
    name surfaces here too, from `problems.build`.
    """
    if not (problems_filter or arms_filter):
        return
    chosen = [group for group in groups if group != "invariance"]
    if any(_selection(group, problems_filter, arms_filter) for group in chosen):
        return
    raise SystemExit(_empty_selection_msg(groups, problems_filter, arms_filter))


def _empty_selection_msg(
    groups: list[str],
    problems_filter: frozenset[str] | None,
    arms_filter: frozenset[str] | None,
) -> str:
    """Explain a filter that would measure nothing, and name what it could use.

    The names it offers are read off the panels themselves, so the list cannot
    drift from the arms the harness has.
    """
    pairs = [
        pair for group in groups if group != "invariance" for pair in _selection(group)
    ]
    if not pairs:
        return (
            f"{'/'.join(groups)} writes no per-arm traces, so --problems and "
            f"--arms select nothing. Drop them, or name a panel group."
        )
    known = {
        "problem": sorted({prob.name for prob, _ in pairs}),
        "arm": sorted({arm.key for _, arm in pairs}),
    }
    lines = ["the filters select no arm, so nothing would be measured."]
    named = False
    for label, given in (("problem", problems_filter), ("arm", arms_filter)):
        if not given:
            continue
        unknown = sorted(given - set(known[label]))
        if not unknown:
            continue
        named = True
        lines.append(f"  no such {label}: {', '.join(unknown)}")
        lines.append(
            textwrap.fill(
                f"  known {label}s: {', '.join(known[label])}",
                width=80,
                subsequent_indent="    ",
                # A name is meant to be copied out of this message, so break only
                # between names: the keys are full of hyphens.
                break_on_hyphens=False,
                break_long_words=False,
            ),
        )
    if not named:
        lines.append(
            "  every name is one this harness has, but no single arm carries both "
            "filters.",
        )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    """Build the command line, declared apart from `main` so a test can read it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="2-seed smoke check")
    parser.add_argument(
        "--seeds",
        type=int,
        default=20,
        help="seeds per arm",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3000,
        help="batches per method",
    )
    parser.add_argument(
        "--groups",
        default="headline,smooth,multimodal,invariance",
        help=f"panels to draw: {', '.join([*PANELS, 'invariance'])}",
    )
    parser.add_argument(
        "--problems",
        default="",
        help="comma list of problem names: re-measure only these",
    )
    parser.add_argument(
        "--arms",
        default="",
        help="comma list of arm keys: re-measure only these",
    )
    parser.add_argument(
        "--lr-sweep",
        action="store_true",
        help="emit the Adam lr sensitivity figure",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="rebuild figures from the trace files (no re-run)",
    )
    return parser


def _budget(args: argparse.Namespace) -> tuple[int, int]:
    """Return this run's ``(seeds, max_steps)``; ``--quick`` means 2 x 300."""
    return (2, 300) if args.quick else (args.seeds, args.max_steps)


def _use_smoke_roots() -> None:
    """Point both output roots under ``smoke/``, before anything writes.

    The arm key does not encode the budget, so a 2-seed, 300-batch trace carries
    exactly the file name a full run's does and would replace it.
    """
    global RESULTS, FIGURES  # noqa: PLW0603 - the output roots are script state
    RESULTS = RESULTS / "smoke"  # pyright: ignore[reportConstantRedefinition]
    FIGURES = FIGURES / "smoke"  # pyright: ignore[reportConstantRedefinition]


def main(argv: list[str] | None = None) -> None:
    """Parse args, run the selected arms, write a trace file each, then draw."""
    args = _parser().parse_args(argv)

    groups = ["smooth"] if args.quick else args.groups.split(",")
    n_seeds, max_steps = _budget(args)
    problems_filter = _comma_set(args.problems)
    arms_filter = _comma_set(args.arms)
    _refuse_undrawable_groups(groups)
    _refuse_untuned_problems()

    if (problems_filter or arms_filter) and (args.plot_only or args.lr_sweep):
        flag = "--plot-only" if args.plot_only else "--lr-sweep"
        msg = (
            f"--problems and --arms name what to measure, and {flag} measures "
            f"nothing, so they would be dropped without a word. Drop them, or "
            f"run the measurement they name."
        )
        raise SystemExit(msg)

    # Below this line every branch writes, so a smoke check rebinds the roots here.
    if args.quick:
        _use_smoke_roots()

    if args.lr_sweep:
        run_lr_sweep(n_seeds=2, max_steps=max_steps)
        return

    if args.plot_only:
        for group in groups:
            if group == "invariance":
                print(f"{group} writes no trace files; rerun --groups {group} (fast)")
                continue
            plot_panel(group, FIGURES / group)
        return

    _refuse_empty_selection(groups, problems_filter, arms_filter)

    for group in groups:
        if group == "invariance":
            run_invariance(n_seeds, max_steps)
            continue
        print(f"[{group}] seeds={n_seeds} max_steps={max_steps}")
        run_panel(
            group,
            n_seeds,
            max_steps,
            problems_filter=problems_filter,
            arms_filter=arms_filter,
        )
        plot_panel(group, FIGURES / group)


if __name__ == "__main__":
    main()
