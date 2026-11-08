"""The command line: flags, refusals, and one run.

One run flows as ``argv`` -> `_budget` plus the ``--problems``/``--arms``
filters -> `selection` flattens the groups into (problem, arm) pairs ->
`run_panel` runs each pair and writes one trace -> `plot_panel` reads the traces
back and draws the group.
"""

import argparse
import textwrap

from experiments.harness import appendix, figures, panels, record
from experiments.harness.panels import GROUPS, TRACELESS, selection
from experiments.harness.record import Roots, refuse_budget_change


def _comma_set(text: str) -> frozenset[str] | None:
    """Parse a comma list into a filter set, or None when it selects everything."""
    items = frozenset(item.strip() for item in text.split(",") if item.strip())
    return items or None


def _refuse_undrawable_groups(groups: list[str]) -> None:
    """Refuse a ``--groups`` name that no panel draws, before anything runs.

    ``problems.build`` knows one group this does not: ``all``, which is the union
    a sweep iterates, not a panel anyone can draw.
    """
    choices = ", ".join(GROUPS)
    for group in groups:
        if group not in GROUPS:
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

    A group that stores no per-arm traces (`TRACELESS`) selects nothing whatever
    the filter says, so naming only those is refused on that ground rather than
    as a mismatch.
    """
    if not (problems_filter or arms_filter):
        return
    chosen = [group for group in groups if group not in TRACELESS]
    if not chosen:
        msg = (
            f"{'/'.join(groups)} writes no per-arm traces, so --problems and "
            f"--arms select nothing. Drop them, or name a panel group."
        )
        raise SystemExit(msg)
    if any(selection(group, problems_filter, arms_filter) for group in chosen):
        return
    raise SystemExit(_filter_mismatch_msg(chosen, problems_filter, arms_filter))


def _filter_mismatch_msg(
    groups: list[str],
    problems_filter: frozenset[str] | None,
    arms_filter: frozenset[str] | None,
) -> str:
    """Explain a filter that would measure nothing, and name what it could use.

    The names it offers are read off the panels themselves, so the list cannot
    drift from the arms the harness has.
    """
    pairs = [pair for group in groups for pair in selection(group)]
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
        "--max-batches",
        type=int,
        default=3000,
        help="batches per method",
    )
    parser.add_argument(
        "--groups",
        default=",".join(GROUPS),
        help=f"panels to draw: {', '.join(GROUPS)}",
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
    parser.add_argument(
        "--out",
        default="",
        metavar="NAME",
        help=(
            "write to results/NAME and figures/NAME instead of the record's "
            "roots; --quick already writes to 'smoke'"
        ),
    )
    return parser


def _budget(args: argparse.Namespace) -> tuple[int, int]:
    """Return this run's ``(seeds, max_batches)``; ``--quick`` means 2 x 300."""
    return (2, 300) if args.quick else (args.seeds, args.max_batches)


def _validate(
    args: argparse.Namespace,
    groups: list[str],
    problems_filter: frozenset[str] | None,
    arms_filter: frozenset[str] | None,
) -> None:
    """Refuse the command lines that would measure or write the wrong thing.

    Called before any branch writes, so that no refusal can leave half a run
    behind: a smoke check or a demo has to say where it writes, and two flags
    naming two namespaces are refused rather than resolved in some order.

    Args:
        args: The parsed command line.
        groups: The panel groups this run names.
        problems_filter: ``--problems`` as a set, or None when not given.
        arms_filter: ``--arms`` as a set, or None when not given.

    Raises:
        SystemExit: the first refusal that applies, with its reason.

    """
    _refuse_undrawable_groups(groups)
    if args.quick and args.out:
        msg = "--quick writes to the smoke roots and --out names another: pick one."
        raise SystemExit(msg)
    if (problems_filter or arms_filter) and (args.plot_only or args.lr_sweep):
        flag = "--plot-only" if args.plot_only else "--lr-sweep"
        msg = (
            f"--problems and --arms name what to measure, and {flag} measures "
            f"nothing, so they would be dropped without a word. Drop them, or "
            f"run the measurement they name."
        )
        raise SystemExit(msg)


def main(argv: list[str] | None = None) -> Roots:
    """Parse args, run the selected arms, write a trace file each, then draw.

    Returns:
        The roots this run wrote into. A caller in the same process (the
        notebook) displays a figure from them rather than re-deriving the
        namespace the flags named.

    """
    args = _parser().parse_args(argv)

    groups = ["smooth"] if args.quick else args.groups.split(",")
    n_seeds, max_batches = _budget(args)
    problems_filter = _comma_set(args.problems)
    arms_filter = _comma_set(args.arms)
    _validate(args, groups, problems_filter, arms_filter)

    # The namespace is decided once, before the first write.
    roots = record.roots_for("smoke" if args.quick else args.out)

    if args.lr_sweep:
        appendix.run_lr_sweep(roots, n_seeds, max_batches)
        return roots

    if args.plot_only:
        for group in groups:
            if group in TRACELESS:
                print(
                    f"{group} writes no trace files: "
                    f"run --groups {group} without --plot-only",
                )
                continue
            figures.plot_panel(roots, group)
        return roots

    _refuse_empty_selection(groups, problems_filter, arms_filter)
    # A trace that holds another budget has to stop the run before the first arm
    # is measured, not on whichever arm happens to sit after a re-measured one.
    for group in groups:
        if group in TRACELESS:
            continue
        for prob, arm in selection(group, problems_filter, arms_filter):
            refuse_budget_change(
                roots,
                prob.name,
                arm.key,
                n_seeds=n_seeds,
                max_batches=max_batches,
            )

    for group in groups:
        if group == "invariance":
            appendix.run_invariance(roots, n_seeds, max_batches)
            continue
        print(f"[{group}] seeds={n_seeds} max_batches={max_batches}")
        panels.run_panel(
            roots,
            group,
            n_seeds,
            max_batches,
            problems_filter=problems_filter,
            arms_filter=arms_filter,
        )
        figures.plot_panel(roots, group)
    return roots
