"""An arm key names one configuration, so a roster may not name one file twice.

The key is the trace file, and a figure draws whatever the key resolves to, so
two arms that differ only in spelling draw one curve under two labels. Moving the
default `mu` back to 0 is exactly the move that makes that collision reachable:
the ladder's default arm and its explicit `mu=0` arm became the same key.

Needs the ``exp`` dependency group (the harness imports matplotlib and optax), so
a plain ``dev`` test run skips this file instead of failing it.
"""

# ruff: noqa: SLF001
# pyright: reportPrivateUsage=false

import os

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
pytest.importorskip("matplotlib")
pytest.importorskip("optax")

from experiments import compare_baselines as harness


def test_the_ladder_names_distinct_arm_files() -> None:
    """Two arms sharing a key would draw one trace under two labels."""
    keys = [arm.key for arm in harness._soar_ladder()]
    assert len(set(keys)) == len(keys), keys


def test_the_default_arm_is_the_momentum_off_arm() -> None:
    """`mu=0` in the default arm's key, so the class default is what it claims."""
    assert harness._soar_ladder()[0].key == "soar-sample-np1-mu0-restart-off"
