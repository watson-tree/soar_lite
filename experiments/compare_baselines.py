"""Run the soar-lite / baseline comparison and emit figures plus results.json.

Two panels per method family, following the design in `DESIGN.md`:

* smooth + headline (preconditioning proposition): soar-lite n_paths=1,
  soar-lite n_paths=16, Adam, L-BFGS-B (ceiling reference);
* multimodal (escape proposition): soar-lite n_paths=1, soar-lite n_paths=16,
  Adam.

Every method gets the same number of **batches** (one batch = one eval of
parallel points, ``--max-steps``) - the wall-clock axis of a batched
setting, and the only axis on which an n_paths=16 run and a single chain compare
at equal cost per step. The second figure replots the same runs against
objective points served, the fair information unit for a serial setting: there a
chain serves one point per batch, so the point axis is truncated at
``--max-steps`` points and a batched run shows only the batches it could afford -
no figure re-runs anything. Curves are the median best-so-far over seeds,
shaded the [Q1, Q3] interquartile band, log y (mean +/- std would model a
skewed spread as symmetric and dip below the axis).

Adam carries a per-problem learning rate, grid-tuned per problem
(reproduce the sweep with --lr-sweep): the best lr spans over two orders of
magnitude across smooth problems and a wrong lr costs ~68 orders of magnitude
on the diagonal quadratic. That tuning is budget the axis never shows -- every
trial is more objective points served, and it is manual rather than adaptive.
soar-lite has no learning rate to tune.

Usage:
    python -m experiments.compare_baselines --quick     # 2-seed smoke check
    python -m experiments.compare_baselines             # the full run
    python -m experiments.compare_baselines --lr-sweep  # Adam lr sensitivity panel
"""

import argparse
import json
import math
import warnings
from collections.abc import Callable
from pathlib import Path

import jax.numpy as jnp
import matplotlib as mpl

mpl.use("Agg")

from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np

from experiments import problems
from experiments.methods import (
    Trace,
    run_adam,
    run_lbfgs,
    run_soar_lite,
)
from soar_lite import NoRestartRule

HERE = Path(__file__).parent
Runner: TypeAlias = Callable[..., Trace]


def _serial(prob: problems.Problem, **kw: int) -> Trace:
    """soar-lite n_paths=1 with no restart: the pure serial descent arm.

    The serial arm is pinned to `NoRestartRule` instead of relying on a
    one-sample pool never tying: a probe is a batched action, and this arm is
    the one-batch-per-point reference the batched arms are compared against.
    """
    return run_soar_lite(prob, n_paths=1, restart_rule=NoRestartRule(), **kw)


CurveEntry: TypeAlias = dict[str, list[float] | list[list[float | None]]]

# Per-problem Adam learning rate, the grid-tuned optimum (see --lr-sweep).
# The optimal lr spans over two orders of magnitude across the smooth problems,
# and a wrong lr costs ~68 orders of magnitude on the diagonal quadratic.
# Tuning is itself budget the y axis never shows -- every trial is more
# objective points served, and it is manual, not adaptive; soar-lite has no
# learning rate at all. Two optima sit at the sweep grid's right edge (the
# rotated quadratic and the oscillator keep improving to lr=2.0), and rastrigin
# is flat over 1e-4..5e-1 so its argmin (0.5) is a ~5% noise-level nudge.
ADAM_LR: dict[str, float] = {
    "oscillator_n20_F20": 2.0,
    "diag_quad_d100_cond10000": 1e-2,
    "rot_quad_d100_cond1e+06": 2.0,
    "ackley_d100": 1e-2,
    "rastrigin_d100": 0.5,
}


def _adam_lr(prob: problems.Problem) -> float:
    """Return the grid-tuned Adam lr for this problem (fallback 1e-2)."""
    return ADAM_LR.get(prob.name, 1e-2)


INVARIANCE_SCALE = 10.0  # the second unit system re-measures x as x' = 10 x

# Method roster per problem family: (label, runner, style kwargs).
SMOOTH_METHODS: list[tuple[str, Runner, dict[str, object]]] = [
    (
        "soar-lite n_paths=1",
        _serial,
        {"color": "C0", "lw": 2.0},
    ),
    (
        "soar-lite n_paths=16",
        lambda prob, **kw: run_soar_lite(prob, n_paths=16, **kw),
        {"color": "C1", "lw": 1.2, "ls": "--"},
    ),
    (
        "Adam (grid lr)",
        lambda prob, **kw: run_adam(prob, lr=_adam_lr(prob), **kw),
        {"color": "C3", "lw": 1.5},
    ),
    (
        "L-BFGS-B",
        run_lbfgs,
        {"color": "C7", "lw": 1.2, "ls": ":"},
    ),
]

# NAAdam (Adam + sigma_t = sigma0/sqrt(t) noise) was measured and dropped: the
# decaying noise never separates from plain Adam on ackley/rastrigin, so it is
# a redundant curve. The honest escape panel is soar-lite n_paths=16 vs
# n_paths=1 vs Adam.
MULTIMODAL_METHODS: list[tuple[str, Runner, dict[str, object]]] = [
    (
        "soar-lite n_paths=16",
        lambda prob, **kw: run_soar_lite(prob, n_paths=16, **kw),
        {"color": "C1", "lw": 2.0},
    ),
    (
        "soar-lite n_paths=1",
        _serial,
        {"color": "C0", "lw": 1.2, "ls": "--"},
    ),
    (
        "Adam (grid lr)",
        lambda prob, **kw: run_adam(prob, lr=_adam_lr(prob), **kw),
        {"color": "C3", "lw": 1.5},
    ),
]

PANELS: dict[str, list[tuple[str, Runner, dict[str, object]]]] = {
    "headline": SMOOTH_METHODS,
    "smooth": SMOOTH_METHODS,
    "multimodal": MULTIMODAL_METHODS,
}


def run_lr_sweep(out: Path, seeds: int, max_steps: int) -> None:
    """Plot best-so-far vs Adam lr, one curve per problem (appendix panel).

    Shows the optimal lr drifting across problems and a wrong lr costing many
    orders of magnitude. The C0 diamond marks the lr actually used in the
    panels (``ADAM_LR``), the grid argmin of this sweep. On the rotated
    quadratic and the oscillator the curve is still falling at the grid edge
    (lr=2.0 is the best *sampled* point, not an interior minimum); on rastrigin
    the floor is flat over 1e-4..1e-1, so its argmin is a noise-level nudge.
    """
    lr_grid = [1e-4, 1e-3, 1e-2, 1e-1, 2e-1, 5e-1, 1.0, 2.0]
    probs = problems.build("all")
    fig, axes = plt.subplots(
        1, len(probs), figsize=(6.4 * len(probs), 4.6), squeeze=False
    )
    for ax, prob in zip(axes[0], probs, strict=True):
        fins = [
            float(
                np.mean(
                    [
                        run_adam(prob, max_steps=max_steps, seed=s, lr=lr).fun
                        for s in range(seeds)
                    ]
                )
            )
            for lr in lr_grid
        ]
        ax.plot(lr_grid, np.maximum(fins, 1e-16), marker="o", lw=1.5, color="C3")
        best_lr = ADAM_LR.get(prob.name)
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
        "Adam lr sensitivity - C0 diamond = panel pick (grid argmin); "
        "rot & oscillator still falling at lr=2.0 (edge); rastrigin flat < 1e-1"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_group_from_json(group: str, out_dir: Path) -> None:
    """Redraw one group's figures straight from its results JSON (no re-run)."""
    path = out_dir / f"results_{group}.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        msg = f"cannot read {path}: {e}"
        raise SystemExit(msg) from e
    labels = payload["meta"].get("labels", {})
    for axis in ("points", "batches"):
        plot(
            payload["curves"],
            labels,
            group,
            axis,
            out_dir / f"results_{group}_{axis}.png",
        )


def run_invariance(out_dir: Path, seeds: int, max_steps: int) -> None:
    """Draw panel 5: one physics in two unit systems, zero retune.

    soar-lite's per-coordinate secant should overlap (outcome-level), Adam's
    single lr should separate because the effective lr rescales with the unit.
    """
    base = problems.diag_quadratic()
    scaled = _rescale(base, INVARIANCE_SCALE)
    lr = _adam_lr(base)
    methods = [
        (
            "soar-lite n_paths=1",
            "C0",
            _serial,
        ),
        ("Adam (same lr)", "C3", lambda p, **kw: run_adam(p, lr=lr, **kw)),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for label, color, runner in methods:
        for prob, ls, tag in ((base, "-", "base"), (scaled, "--", "x10")):
            traces = [runner(prob, max_steps=max_steps, seed=s) for s in range(seeds)]
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
        f"unit rescaling x' = {INVARIANCE_SCALE:g}x, no retune (Adam lr={lr:g})"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    fig.suptitle("panel 5 - unit-rescaling consistency")
    fig.tight_layout()
    fig.savefig(out_dir / "results_invariance.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out_dir / 'results_invariance.png'}")


def run_panel(
    group: str, seeds: int, max_steps: int
) -> tuple[dict[str, dict[str, CurveEntry]], dict[str, str]]:
    """Run every problem in ``group`` across every method and seed."""
    methods = PANELS[group]
    curves: dict[str, dict[str, CurveEntry]] = {}
    labels: dict[str, str] = {}
    for prob in problems.build(group):
        labels[prob.name] = prob.description
        curves[prob.name] = {}
        traces = {
            label: [runner(prob, max_steps=max_steps, seed=s) for s in range(seeds)]
            for label, runner, _ in methods
        }
        for label, _, _ in methods:
            fins = [t.fun for t in traces[label]]
            tail = f"final mean best = {np.mean(fins):.4e}"
            print(f"  {prob.name:26s} {label:16s} {tail}")
        for axis in ("points", "batches"):
            col = 0 if axis == "points" else 1
            # Equal batches is the batched budget, equal points the serial one,
            # and they are the same runs seen twice. Truncating the point axis
            # at max_steps is what a chain's own run costs, so an n-path run
            # shows exactly the batches it could afford there.
            shown = {
                label: [
                    _up_to(t, max_steps) if axis == "points" else t
                    for t in traces[label]
                ]
                for label, _, _ in methods
            }
            xmax = max(
                float(t.as_arrays()[col].max())
                for ts in shown.values()
                for t in ts
                if len(t.points) > 0
            )
            # Log-spaced: an n-path run finishes its batches far earlier
            # than a single chain finishes its points, and a linear grid
            # would squash the batched curve into the left edge of the axis.
            grid = np.logspace(0.0, np.log10(xmax), 120)
            grid[-1] = xmax  # logspace rounds the last sample past its own end
            for label, _, _ in methods:
                arr = _resample(shown[label], grid, axis)
                entry = curves[prob.name].setdefault(label, {})
                entry[f"{axis}_grid"] = grid.tolist()
                entry[f"{axis}_curves"] = _json_curves(arr)
    return curves, labels


def plot(
    curves: dict[str, dict[str, CurveEntry]],
    labels: dict[str, str],
    group: str,
    axis: str,
    out: Path,
) -> None:
    """Draw one figure: a subplot per problem, best-so-far vs the chosen axis."""
    methods = PANELS[group]
    names = list(curves)
    fig, axes = plt.subplots(
        1, len(names), figsize=(6.4 * len(names), 4.6), squeeze=False
    )
    for ax, name in zip(axes[0], names, strict=True):
        for label, _, style in methods:
            data = curves[name][label]
            grid = np.asarray(data[f"{axis}_grid"], dtype=float)
            arr = np.asarray(data[f"{axis}_curves"], dtype=float)  # None -> NaN
            med, q1, q3 = _summary(arr)
            ax.plot(grid, np.maximum(med, 1e-16), label=label, **style)
            ax.fill_between(
                grid,
                np.maximum(q1, 1e-16),
                np.maximum(q3, 1e-16),
                color=style["color"],
                alpha=0.15,
                lw=0,
            )
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel(
            "objective points served" if axis == "points" else "batches served"
        )
        ax.set_ylabel("best-so-far (median [Q1, Q3] over seeds)")
        ax.set_title(f"{name}\n{labels[name]}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    fig.suptitle(f"soar-lite vs first-order baselines - {group} panel ({axis} axis)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def _resample(traces: list[Trace], grid: np.ndarray, axis: str) -> np.ndarray:
    """Resample every seed's best-so-far onto ``grid``.

    Returns a ``(seeds, len(grid))`` array with NaN where a seed has no data
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


def _json_curves(arr: np.ndarray) -> list[list[float | None]]:
    """NaN -> None for a JSON-safe per-seed matrix (None reloads as NaN)."""
    return [[None if math.isnan(v) else float(v) for v in row] for row in arr.tolist()]


def _up_to(trace: Trace, max_points: int) -> Trace:
    """Return the first ``max_points`` objective points of a run, nothing else."""
    stop = sum(1 for p in trace.points if p <= max_points)
    return Trace(
        points=trace.points[:stop],
        batches=trace.batches[:stop],
        best=trace.best[:stop],
        fun=trace.fun,
    )


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


def main() -> None:
    """Parse args, run the panels, write figures and the JSON dump."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="2-seed smoke check")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument(
        "--max-steps", type=int, default=3000, help="batches per method"
    )
    parser.add_argument("--groups", default="headline,smooth,multimodal,invariance")
    parser.add_argument(
        "--lr-sweep", action="store_true", help="emit the Adam lr sensitivity figure"
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="rebuild figures from results_*.json (no re-run)",
    )
    args = parser.parse_args()

    if args.lr_sweep:
        run_lr_sweep(HERE / "results_lr_sweep.png", seeds=2, max_steps=args.max_steps)
        return

    if args.plot_only:
        for group in args.groups.split(","):
            if group == "invariance":
                print(
                    "invariance panel writes no JSON; rerun --groups invariance (fast)"
                )
                continue
            plot_group_from_json(group, HERE)
        return

    groups = ["smooth"] if args.quick else args.groups.split(",")
    seeds = 2 if args.quick else args.seeds
    max_steps = 300 if args.quick else args.max_steps

    for group in groups:
        if group == "invariance":
            run_invariance(HERE, seeds, max_steps)
            continue
        print(f"[{group}] seeds={seeds} max_steps={max_steps}")
        curves, labels = run_panel(group, seeds, max_steps)
        star = {p.name: p.f_star for p in problems.build(group)}
        for axis in ("points", "batches"):
            plot(
                curves,
                labels,
                group,
                axis,
                out=HERE / f"results_{group}_{axis}.png",
            )
        with (HERE / f"results_{group}.json").open("w") as fh:
            json.dump(
                {
                    "meta": {
                        "group": group,
                        "seeds": seeds,
                        "max_steps": max_steps,
                        "f_star": star,
                        "labels": labels,
                    },
                    "curves": curves,
                },
                fh,
                indent=1,
            )
        print(f"wrote {HERE / f'results_{group}.json'}")


if __name__ == "__main__":
    main()
