"""The model-free sampler, which draws from the box each ask hands it.

`SampleGlobalModel` fits nothing, so what it can do is exactly what the domain
it is handed admits. These pin the two things a plain uniform model could not:
an unbounded side is drawn from the domain's own default, and the flock `x0` is
the first proposal and only the first.
"""

import jax
import jax.numpy as jnp
import pytest

from soar_lite import Domain, SampleGlobalModel


def test_a_model_free_domain_draws_its_unbounded_dimension() -> None:
    """One half-bounded side and one box side: both inside the declared domain."""
    box = Domain(jnp.asarray([[0.0, jnp.inf], [-5.0, 5.0]]))
    model = SampleGlobalModel(key=jax.random.PRNGKey(0))

    points = model.propose(512, box)

    assert points.shape == (512, 2)
    assert bool(jnp.all(jnp.isfinite(points)))
    assert bool(jnp.all(points[:, 0] >= 0.0)), float(jnp.min(points[:, 0]))
    assert bool(jnp.all(points[:, 1] >= -5.0)), float(jnp.min(points[:, 1]))
    assert bool(jnp.all(points[:, 1] <= 5.0)), float(jnp.max(points[:, 1]))


def test_the_draws_are_the_handed_domains_own() -> None:
    """The box is a per-call argument: one instance serves two different boxes."""
    model = SampleGlobalModel(key=jax.random.PRNGKey(0))

    wide = model.propose(256, Domain([(0.0, 1.0)] * 3))
    assert wide.shape == (256, 3), "the width came from the second ask's domain"

    narrow = model.propose(64, Domain([(10.0, 11.0)] * 2))
    assert narrow.shape == (64, 2)
    assert bool(jnp.all(narrow >= 10.0)), float(jnp.min(narrow))
    assert bool(jnp.all(narrow <= 11.0)), float(jnp.max(narrow))


def test_x0_is_the_first_proposal_and_only_the_first() -> None:
    """The flock rides the first ask; every later ask is a domain draw."""
    box = Domain([(-5.0, 5.0)] * 2)
    x0 = jnp.asarray([[-1.0, 2.0], [3.0, -4.0]])
    model = SampleGlobalModel(key=jax.random.PRNGKey(0), x0=x0)

    assert jnp.array_equal(model.propose(2, box), x0)
    later = model.propose(2, box)
    assert later.shape == (2, 2)
    assert not jnp.array_equal(later, x0)


def test_x0_must_match_the_first_asks_domain_dimension() -> None:
    """A flock of the wrong width is a wiring error, refused on the first ask."""
    model = SampleGlobalModel(key=jax.random.PRNGKey(0), x0=jnp.zeros((2, 2)))

    with pytest.raises(ValueError, match=r"x0 must be a \(n_paths, 3\) array"):
        model.propose(2, Domain([(0.0, 1.0)] * 3))
