"""Run the soar-lite / baseline comparison and emit figures plus results.json.

Two panels per method family, following the design in `DESIGN.md`:

* smooth + headline (preconditioning proposition): soar-lite K=1, soar-lite
  K=16, Adam, L-BFGS-B (ceiling reference);
* multimodal (escape proposition): soar-lite K=1, soar-lite K=16, Adam, NAAdam.

Cost axis is objective evaluations (the fair information unit); a second figure
replots the same runs against sequential batched rounds, the wall-clock axis
for a batched setting. Curves are mean of best-so-far over seeds, shaded +-1
standard deviation, log y.

Usage:
    python -m experiments.compare_baselines --quick     # 2-seed smoke check
    python -m experiments.compare_baselines             # the full run
"""

import argparse
import json
import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

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
Runner = Callable[..., Trace]

# Method roster per problem family: (label, runner, style kwargs).
SMOOTH_METHODS: list[tuple[str, Runner, dict[str, object]]] = [
    (
        "soar-lite K=1",
        lambda prob, **kw: run_soar_lite(prob, K=1, **kw),
        {"color": "C0", "lw": 2.0},
    ),
    (
        "soar-lite K=16",
        lambda prob, **kw: run_soar_lite(prob, K=16, **kw),
        {"color": "C0", "lw": 1.2, "ls": "--"},
    ),
    (
        "Adam",
        lambda prob, **kw: run_adam(prob, lr=0.05, **kw),
        {"color": "C3", "lw": 1.5},
    ),
    (
        "L-BFGS-B",
        run_lbfgs,
        {"color": "C7", "lw": 1.2, "ls": ":"},
    ),
]

MULTIMODAL_METHODS: list[tuple[str, Runner, dict[str, object]]] = [
    (
        "soar-lite K=16",
        lambda prob, **kw: run_soar_lite(prob, K=16, **kw),
        {"color": "C0", "lw": 2.0},
    ),
    (
        "soar-lite K=1",
        lambda prob, **kw: run_soar_lite(prob, K=1, **kw),
        {"color": "C0", "lw": 1.2, "ls": "--"},
    ),
    (
        "Adam",
        lambda prob, **kw: run_adam(prob, lr=0.05, **kw),
        {"color": "C3", "lw": 1.5},
    ),
    (
        "NAAdam",
        lambda prob, **kw: run_naadam(prob, lr=0.05, **kw),
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
        evals, rounds, best = trace.as_arrays()
        x = evals if axis == "evals" else rounds
        idx = np.searchsorted(x, grid, side="right") - 1
        idx = np.clip(idx, 0, len(best) - 1)
        # The step curve exists only between a run's first and last cost unit:
        # before its first batch and after it stops, there is no data.
        curve = np.where((grid < x[0]) | (grid > x[-1]), np.nan, best[idx])
        stacked.append(curve)
    arr = np.vstack(stacked)
    # K-path runs have no data before their first batch (K evals in), so the
    # leading grid points are all-NaN columns by construction, not an error.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def run_panel(
    group: str, seeds: int, max_evals: int
) -> tuple[dict[str, dict[str, dict[str, list[float]]]], dict[str, str]]:
    """Run every problem in ``group`` across every method and seed."""
    methods = PANELS[group]
    curves: dict[str, dict[str, dict[str, list[float]]]] = {}
    labels: dict[str, str] = {}
    for prob in problems.build(group):
        labels[prob.name] = prob.description
        curves[prob.name] = {}
        traces = {
            label: [runner(prob, max_evals=max_evals, seed=s) for s in range(seeds)]
            for label, runner, _ in methods
        }
        for label, _, _ in methods:
            fins = [t.fun for t in traces[label]]
            tail = f"final mean best = {np.mean(fins):.4e}"
            print(f"  {prob.name:26s} {label:16s} {tail}")
        for axis in ("evals", "rounds"):
            col = 0 if axis == "evals" else 1
            xmax = max(
                float(t.as_arrays()[col].max())
                for ts in traces.values()
                for t in ts
                if len(t.evals) > 0
            )
            # Log-spaced: a K-path run finishes its rounds far earlier than a
            # single chain finishes its evals, and a linear grid would squash
            # the batched curve into the first few percent of the axis.
            grid = np.logspace(0.0, np.log10(xmax), 120)
            for label, _, _ in methods:
                mean, std = resample(traces[label], grid, axis)
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
            "objective evaluations" if axis == "evals" else "batched rounds"
        )
        ax.set_ylabel("best-so-far (median over seeds)")
        ax.set_title(f"{name}\n{labels[name]}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    fig.suptitle(f"soar-lite vs first-order baselines - {group} panel ({axis} axis)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    """Parse args, run the panels, write figures and the JSON dump."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="2-seed smoke check")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--max-evals", type=int, default=3000)
    parser.add_argument("--groups", default="headline,smooth,multimodal")
    args = parser.parse_args()

    groups = ["smooth"] if args.quick else args.groups.split(",")
    seeds = 2 if args.quick else args.seeds
    max_evals = 600 if args.quick else args.max_evals

    for group in groups:
        print(f"[{group}] seeds={seeds} max_evals={max_evals}")
        curves, labels = run_panel(group, seeds, max_evals)
        star = {p.name: p.f_star for p in problems.build(group)}
        for axis in ("evals", "rounds"):
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
                        "max_evals": max_evals,
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
