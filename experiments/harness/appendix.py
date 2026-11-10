"""Appendix panels: the two figures that measure as they draw.

Unlike `figures`, these serve objective points: the lr sweep re-runs Adam at
every grid point, and the invariance panel re-runs both methods in a second unit
system.
"""

import jax.numpy as jnp
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.harness import problems, tuning
from experiments.harness.arms import DEFAULT_MU, soar_arm
from experiments.harness.figures import plot_cohort
from experiments.harness.methods import run_adam
from experiments.harness.panels import LABEL_SERIAL, STYLE_SERIAL, every_problem
from experiments.harness.record import HERE, Roots

INVARIANCE_SCALE = 10.0  # the second unit system re-measures x as x' = 10 x


def run_lr_sweep(roots: Roots, n_seeds: int, max_batches: int) -> None:
    """Plot best-so-far vs Adam lr, one curve per problem (appendix panel).

    Shows the optimal lr drifting across problems and a wrong lr costing many
    orders of magnitude. The C0 diamond marks the pick the panels use, which is
    this sweep's argmin (ties to the smallest rate): different problems land on
    different points of the grid, and on the diagonal quadratic a rate far from
    the pick costs dozens of orders of magnitude. Tuning is itself
    budget the y axis never shows - every trial is more objective points served,
    and it is manual, not adaptive.

    What that argmin is worth varies, and the figure is how a reader sees it.
    On the diagonal quadratic every rate from the sweep's second point through
    1.0 has already reached the quadratic's rounding floor, so the winner there
    is decided by rounding, not by a rate difference. On the rotated quadratic
    and the oscillator the curve is still falling at the grid edge, so a wider
    grid could move the pick; the compartment chain is the mirror case, an
    interior minimum sharp on both sides. On rastrigin the floor is flat across
    four of the grid's points and the pick sits a few percent below it: real,
    but a short gap. The curves behind those statements are in `RESULTS.md`.

    The record is written before the figure, so the picks are re-derivable from
    the curve that produced them.
    """
    tuning.refuse_other_budget("adam", seeds=n_seeds, max_batches=max_batches)
    lr_grid = tuning.ADAM_GRID
    probs = every_problem()
    curves = {
        prob.name: [
            float(
                np.mean(
                    [
                        run_adam(prob, max_batches=max_batches, seed=s, lr=lr).fun
                        for s in range(n_seeds)
                    ],
                ),
            )
            for lr in lr_grid
        ]
        for prob in probs
    }
    record = tuning.save(
        "adam",
        grid=lr_grid,
        seeds=n_seeds,
        max_batches=max_batches,
        curves=curves,
    )
    print(f"wrote {record.relative_to(HERE)}")
    fig, axes = plt.subplots(
        1,
        len(probs),
        figsize=(6.4 * len(probs), 4.6),
        squeeze=False,
    )
    for ax, prob in zip(axes[0], probs, strict=True):
        fins = curves[prob.name]
        best_idx = fins.index(min(fins))
        print(
            f"{prob.name:26s} pick={lr_grid[best_idx]:g}",
            " ".join(f"{lr:g}:{f:.6g}" for lr, f in zip(lr_grid, fins, strict=True)),
        )
        ax.plot(lr_grid, np.maximum(fins, 1e-16), marker="o", lw=1.5, color="C3")
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
        "Adam lr sensitivity - C0 diamond = the pick the panels use (this "
        "sweep's argmin, ties to the smallest rate); on diag every rate from "
        "1e-2 up is at the rounding floor; rot & oscillator pick at the grid "
        "edge; pk_chain is a sharp interior minimum; rastrigin's floor is "
        "within 5%",
    )
    fig.tight_layout()
    out = roots.panel_dir("lr_sweep") / "lr_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.relative_to(HERE)}")


def run_invariance(roots: Roots, n_seeds: int, max_batches: int) -> None:
    """Draw panel 4: one physics in two unit systems, zero retune.

    soar-lite's per-coordinate secant should overlap (outcome-level), Adam's
    single lr should separate because the effective lr rescales with the unit.
    """
    base = problems.diag_quadratic()
    scaled = _rescale(base, INVARIANCE_SCALE)
    lr = tuning.pick("adam", base.name)
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
            traces = [
                runner(prob, max_batches=max_batches, seed=s) for s in range(n_seeds)
            ]
            # Seeds stop at different point counts (diag converges in tens of
            # points), so the raw trace arrays are ragged: plot_cohort resamples
            # them onto one log-spaced grid, exactly as the other panels do.
            plot_cohort(
                ax,
                traces,
                "points",
                label=f"{label} [{tag}]",
                style={"color": color, "ls": ls, "lw": 1.5},
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
    fig.suptitle("panel 4 - unit-rescaling consistency")
    fig.tight_layout()
    out = roots.panel_dir("invariance") / "invariance.png"
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
