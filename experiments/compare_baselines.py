"""Run the soar-lite / baseline comparison and emit figures plus results.json.

Two panels per method family, following the design in `DESIGN.md`:

* smooth + headline (preconditioning proposition): soar-lite n_paths=1,
  soar-lite n_paths=16, Adam, L-BFGS-B (ceiling reference);
* multimodal (escape proposition): soar-lite n_paths=1, soar-lite n_paths=16,
  Adam, NAAdam.

Every method gets the same number of **batches** (one batch = one round of
parallel evaluations, ``--max-steps``) - the wall-clock axis of a batched
setting, and the only axis on which an n_paths=16 run and a single chain compare
at equal cost per step. The second figure replots the same runs against
objective points served, the fair information unit for a serial setting: there a
chain serves one point per batch, so the point axis is truncated at
``--max-steps`` points and a batched run shows only the batches it could afford -
no figure re-runs anything. Curves are mean of best-so-far over seeds, shaded
+-1 standard deviation, log y.

Adam and NAAdam carry a per-problem learning rate, picked on a coarse 1-seed
grid sweep (reproduce it with --lr-sweep): the best lr spans over two orders of
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
    run_naadam,
    run_soar_lite,
)

HERE = Path(__file__).parent
Runner: TypeAlias = Callable[..., Trace]

# Per-problem Adam/NAAdam learning rate, chosen on a coarse 1-seed grid sweep
# (see --lr-sweep): the optimal lr spans over two orders of magnitude across the
# smooth problems, and a wrong lr costs ~68 orders of magnitude on the diagonal
# quadratic. Tuning is itself budget the y axis never shows -- every trial is
# more objective points served, and it is manual, not adaptive. soar-lite has
# no learning rate at all. The rotated optimum only appears above lr=1 (peak
# 2.0, ~11x better than 1.0), so it is the hardest dial to find by hand.
ADAM_LR: dict[str, float] = {
    "oscillator_n20_F20": 0.5,
    "diag_quad_d100_cond10000": 1e-2,
    "rot_quad_d100_cond1e+06": 2.0,
    "ackley_d100": 1e-2,
    "rastrigin_d100": 1e-2,
}


def _adam_lr(prob: problems.Problem) -> float:
    """Return the grid-tuned Adam/NAAdam lr for this problem (fallback 1e-2)."""
    return ADAM_LR.get(prob.name, 1e-2)


INVARIANCE_SCALE = 10.0  # the second unit system re-measures x as x' = 10 x

# Method roster per problem family: (label, runner, style kwargs).
SMOOTH_METHODS: list[tuple[str, Runner, dict[str, object]]] = [
    (
        "soar-lite n_paths=1",
        lambda prob, **kw: run_soar_lite(prob, n_paths=1, **kw),
        {"color": "C0", "lw": 2.0},
    ),
    (
        "soar-lite n_paths=16",
        lambda prob, **kw: run_soar_lite(prob, n_paths=16, **kw),
        {"color": "C0", "lw": 1.2, "ls": "--"},
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

# NAAdam keeps sigma0=1e-2 (its default, untuned): a sigma0 sweep on ackley and
# rastrigin at d=100 shows the decaying noise never escapes -- larger sigma0 is
# flat (rastrigin) to worse (ackley), so the honest escape baseline is plain
# Adam. That the noise buys nothing is part of P1's case for soar-lite's
# sample-measured restarts.
MULTIMODAL_METHODS: list[tuple[str, Runner, dict[str, object]]] = [
    (
        "soar-lite n_paths=16",
        lambda prob, **kw: run_soar_lite(prob, n_paths=16, **kw),
        {"color": "C0", "lw": 2.0},
    ),
    (
        "soar-lite n_paths=1",
        lambda prob, **kw: run_soar_lite(prob, n_paths=1, **kw),
        {"color": "C0", "lw": 1.2, "ls": "--"},
    ),
    (
        "Adam (grid lr)",
        lambda prob, **kw: run_adam(prob, lr=_adam_lr(prob), **kw),
        {"color": "C3", "lw": 1.5},
    ),
    (
        "NAAdam (grid lr)",
        lambda prob, **kw: run_naadam(prob, lr=_adam_lr(prob), **kw),
        {"color": "C2", "lw": 1.5, "ls": "-."},
    ),
]

PANELS: dict[str, list[tuple[str, Runner, dict[str, object]]]] = {
    "headline": SMOOTH_METHODS,
    "smooth": SMOOTH_METHODS,
    "multimodal": MULTIMODAL_METHODS,
}


def resample(
    traces: list[Trace], grid: np.ndarray, axis: str
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate seeds onto ``grid``: mean and std of the best-so-far step curve."""
    stacked = []
    for trace in traces:
        points, batches, best = trace.as_arrays()
        x = points if axis == "points" else batches
        idx = np.searchsorted(x, grid, side="right") - 1
        idx = np.clip(idx, 0, len(best) - 1)
        # The step curve exists only between a run's first and last cost unit:
        # before its first batch and after it stops, there is no data.
        curve = np.where((grid < x[0]) | (grid > x[-1]), np.nan, best[idx])
        stacked.append(curve)
    arr = np.vstack(stacked)
    # An n-path run has no data before its first batch (n_paths points
    # in), so the leading grid points are all-NaN columns by construction.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def _up_to(trace: Trace, max_points: int) -> Trace:
    """Return the first ``max_points`` objective points of a run, nothing else."""
    stop = sum(1 for p in trace.points if p <= max_points)
    return Trace(
        points=trace.points[:stop],
        batches=trace.batches[:stop],
        best=trace.best[:stop],
        fun=trace.fun,
    )


def run_panel(
    group: str, seeds: int, max_steps: int
) -> tuple[dict[str, dict[str, dict[str, list[float]]]], dict[str, str]]:
    """Run every problem in ``group`` across every method and seed."""
    methods = PANELS[group]
    curves: dict[str, dict[str, dict[str, list[float]]]] = {}
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
                mean, std = resample(shown[label], grid, axis)
                entry = curves[prob.name].setdefault(label, {})
                entry[f"{axis}_grid"] = grid.tolist()
                entry[f"{axis}_mean"] = mean.tolist()
                entry[f"{axis}_std"] = std.tolist()
    return curves, labels


def plot(
    curves: dict[str, dict[str, dict[str, list[float]]]],
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
            grid = np.asarray(data[f"{axis}_grid"])
            mean = np.maximum(np.asarray(data[f"{axis}_mean"]), 1e-16)
            std = np.asarray(data[f"{axis}_std"])
            ax.plot(grid, mean, label=label, **style)
            ax.fill_between(
                grid,
                np.maximum(mean - std, 1e-16),
                np.maximum(mean + std, 1e-16),
                color=style["color"],
                alpha=0.15,
                lw=0,
            )
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel(
            "objective points served" if axis == "points" else "batches served"
        )
        ax.set_ylabel("best-so-far (mean over seeds)")
        ax.set_title(f"{name}\n{labels[name]}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    fig.suptitle(f"soar-lite vs first-order baselines - {group} panel ({axis} axis)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def run_lr_sweep(out: Path, seeds: int, max_steps: int) -> None:
    """Plot best-so-far vs Adam lr, one curve per problem (appendix panel).

    Shows the optimal lr drifting across problems and a wrong lr costing many
    orders of magnitude; the point picked per problem (``ADAM_LR``) is marked
    with a C0 diamond.
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
    fig.suptitle("Adam lr sensitivity - C0 diamond marks the per-problem pick")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


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
            lambda p, **kw: run_soar_lite(p, n_paths=1, **kw),
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
                float(t.as_arrays()[0].max())
                for t in traces
                if len(t.points) > 0
            )
            grid = np.logspace(0.0, np.log10(xmax), 120)
            grid[-1] = xmax
            mean, std = resample(traces, grid, "points")
            mean = np.maximum(mean, 1e-16)
            ax.plot(grid, mean, color=color, ls=ls, lw=1.5, label=f"{label} [{tag}]")
            ax.fill_between(
                grid,
                np.maximum(mean - std, 1e-16),
                np.maximum(mean + std, 1e-16),
                color=color,
                alpha=0.12,
                lw=0,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("objective points served")
    ax.set_ylabel("best-so-far (mean ± std over seeds)")
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


def _rescale(prob: problems.Problem, scale: float) -> problems.Problem:
    """Return the same physics re-measured in coordinates ``x' = scale * x``.

    The box is ``scale`` wider and the gradient ``1/scale`` smaller (chain rule
    with ``x = x'/scale``); a runner on this problem is the "second unit
    system", re-tuned by exactly zero.
    """
    bounds = prob.bounds * scale

    def evaluate_batch(xp: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, None]:
        f, g, _ = prob.evaluate_batch(xp / scale)
        return f, g / scale, None

    x_star = prob.x_star * scale if prob.x_star is not None else None
    return problems.Problem(
        name=f"{prob.name}@x{scale:g}",
        d=prob.d,
        bounds=bounds,
        evaluate_batch=evaluate_batch,
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
    args = parser.parse_args()

    if args.lr_sweep:
        run_lr_sweep(
            HERE / "results_lr_sweep.png", seeds=2, max_steps=args.max_steps
        )
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
                    },
                    "curves": curves,
                },
                fh,
                indent=1,
            )
        print(f"wrote {HERE / f'results_{group}.json'}")


if __name__ == "__main__":
    main()
