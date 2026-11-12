"""The model-free sampler, which draws from the box each ask hands it.

`SampleGlobalModel` fits nothing, so what it can do is exactly what the domain
it is handed admits. These pin the thing a plain uniform model could not: an
unbounded side is drawn from the domain's own default.
"""

import jax
import jax.numpy as jnp

from soar_lite import Domain, GlobalModel, SampleGlobalModel


def ask(model: GlobalModel, n_points: int, domain: Domain) -> jnp.ndarray:
    """Ask for `n_points` starts as paths ``0..n_points-1``, stacked into rows."""
    answers = model.propose(list(range(n_points)), domain)
    rows = []
    for i in range(n_points):
        point = answers[i]
        assert point is not None, f"path {i} was declined"
        rows.append(point.x)
    return jnp.stack(rows)


def test_a_model_free_domain_draws_its_unbounded_dimension() -> None:
    """One half-bounded side and one box side: both inside the declared domain."""
    box = Domain(jnp.asarray([[0.0, jnp.inf], [-5.0, 5.0]]))
    model = SampleGlobalModel(key=jax.random.PRNGKey(0))

    points = ask(model, 512, box)

    assert points.shape == (512, 2)
    assert bool(jnp.all(jnp.isfinite(points)))
    assert bool(jnp.all(points[:, 0] >= 0.0)), float(jnp.min(points[:, 0]))
    assert bool(jnp.all(points[:, 1] >= -5.0)), float(jnp.min(points[:, 1]))
    assert bool(jnp.all(points[:, 1] <= 5.0)), float(jnp.max(points[:, 1]))


def test_the_draws_are_the_handed_domains_own() -> None:
    """The box is a per-call argument: one instance serves two different boxes."""
    model = SampleGlobalModel(key=jax.random.PRNGKey(0))

    wide = ask(model, 256, Domain([(0.0, 1.0)] * 3))
    assert wide.shape == (256, 3), "the width came from the second ask's domain"

    narrow = ask(model, 64, Domain([(10.0, 11.0)] * 2))
    assert narrow.shape == (64, 2)
    assert bool(jnp.all(narrow >= 10.0)), float(jnp.min(narrow))
    assert bool(jnp.all(narrow <= 11.0)), float(jnp.max(narrow))
