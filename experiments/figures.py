"""Figures: draw a panel from whatever the record already holds.

Nothing here serves an objective point - a figure is read back from the trace
files - so `--plot-only` and the tail of a full run draw the same picture.
"""

import warnings
from typing import Any, Literal, TypeAlias

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from experiments import problems
from experiments.methods import Trace
from experiments.panels import PANELS
from experiments.record import HERE, Roots, load_arm

Axis: TypeAlias = Literal["points", "batches"]
"""The two cost units a panel can be drawn against.

Named as a type so a caller inside the repo gets the pair from the checker; the
comparisons below still read it at run time, since `--groups` never reaches here.
"""


def plot_panel(roots: Roots, panel: str) -> None:
    """Draw a panel from whatever arm files exist; a missing arm is skipped."""
    probs = problems.build(panel)
    out_dir = roots.panel_dir(panel)
    out_dir.mkdir(parents=True, exist_ok=True)
    for axis in ("points", "batches"):
        fig, axes = plt.subplots(
            1,
            len(probs),
            figsize=(6.4 * len(probs), 4.6),
            squeeze=False,
        )
        for ax, prob in zip(axes[0], probs, strict=True):
            drawn = _draw_problem(ax, roots, panel, prob, axis)
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
    roots: Roots,
    panel: str,
    prob: problems.Problem,
    axis: Axis,
) -> bool:
    """Draw one subplot; False when the arm files it needs are all absent."""
    entries = [
        (arm, load_arm(roots, prob.name, arm.key)) for arm in PANELS[panel](prob)
    ]
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
    for arm, _ in present:
        traces = shown[arm.key]
        # A run can stop before the cut this axis shares - the diag control
        # converges within a handful of batches - and then it served nothing at
        # or below `xmax` here. Named, not drawn: an empty cohort has no extent.
        if not any(len(t.points) > 0 for t in traces):
            print(f"  cut before first point: {prob.name}: {arm.key}")
            continue
        plot_cohort(ax, traces, axis, label=arm.label, style=arm.style, xmax=xmax)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("objective points served" if axis == "points" else "batches served")
    ax.set_ylabel("best-so-far (median [Q1, Q3] over seeds)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    return True


def plot_cohort(  # noqa: PLR0913 - one parameter per fact the call site decides
    ax: Axes,
    traces: list[Trace],
    axis: Axis,
    *,
    label: str,
    style: dict[str, Any],
    xmax: float | None = None,
) -> None:
    """Draw one cohort's median best-so-far with its [Q1, Q3] band.

    At least one trace must carry a point.

    Args:
        ax: The subplot to draw into.
        traces: One trace per seed, all on the same run.
        axis: The cost unit this subplot's x-axis carries.
        label: Legend text for this cohort.
        style: Passed to `Axes.plot`, and must carry ``color`` for the band.
        xmax: The x extent to resample over; ``None`` takes the cohort's own,
            which is what a caller with one cohort wants. `_draw_problem`
            passes the cut it shares across arms instead, so every curve on a
            subplot is sampled at the same positions.

    """
    if xmax is None:
        live = [t for t in traces if len(t.points) > 0]
        xmax = max(
            float(t.as_arrays()[0 if axis == "points" else 1].max()) for t in live
        )
    # Log-spaced: the descent these panels are about happens in the first decade
    # of served cost, and a linear grid would compress it into the left edge.
    grid = np.logspace(0.0, np.log10(xmax), 120)
    grid[-1] = xmax  # logspace rounds the last sample past its own end
    med, q1, q3 = _summary(_resample(traces, grid, axis))
    ax.plot(grid, np.maximum(med, 1e-16), label=label, **style)
    ax.fill_between(
        grid,
        np.maximum(q1, 1e-16),
        np.maximum(q3, 1e-16),
        color=style["color"],
        alpha=0.15,
        lw=0,
    )


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
