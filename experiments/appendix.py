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

from experiments import problems
from experiments.arms import DEFAULT_MU, adam_lr, soar_arm
from experiments.figures import plot_cohort
from experiments.methods import run_adam
from experiments.panels import LABEL_SERIAL, STYLE_SERIAL
from experiments.record import HERE, Roots

INVARIANCE_SCALE = 10.0  # the second unit system re-measures x as x' = 10 x


def run_lr_sweep(roots: Roots, n_seeds: int, max_steps: int) -> None:
    """Plot best-so-far vs Adam lr, one curve per problem (appendix panel).

    Shows the optimal lr drifting across problems and a wrong lr costing many
    orders of magnitude. The C0 diamond marks the lr actually used in the
    panels (``ADAM_LR``). It agrees with the sweep's argmin on the rotated
    quadratic, the oscillator, the compartment chain and rastrigin; on the
    diagonal quadratic every grid point sits at the same floor or
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
            f"{prob.name:26s} pick={adam_lr(prob)}",
            " ".join(f"{lr:g}:{f:.6g}" for lr, f in zip(lr_grid, fins, strict=True)),
        )
        ax.plot(lr_grid, np.maximum(fins, 1e-16), marker="o", lw=1.5, color="C3")
        best_lr = adam_lr(prob)
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
        "Adam lr sensitivity - C0 diamond = the lr the panels use; on diag "
        "it is one point of a floor/plateau tie, not a strict argmin; "
        "rot & oscillator still falling at lr=2.0 (edge); pk_chain is a sharp "
        "interior optimum; rastrigin flat < 2e-1",
    )
    fig.tight_layout()
    out = roots.panel_dir("lr_sweep") / "lr_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.relative_to(HERE)}")


def run_invariance(roots: Roots, n_seeds: int, max_steps: int) -> None:
    """Draw panel 5: one physics in two unit systems, zero retune.

    soar-lite's per-coordinate secant should overlap (outcome-level), Adam's
    single lr should separate because the effective lr rescales with the unit.
    """
    base = problems.diag_quadratic()
    scaled = _rescale(base, INVARIANCE_SCALE)
    lr = adam_lr(base)
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
    fig.suptitle("panel 5 - unit-rescaling consistency")
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
