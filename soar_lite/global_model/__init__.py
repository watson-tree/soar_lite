"""The restart-model family: the engine's two `GlobalModel` concretes.

`sample.py` is the model-free draw (`SampleGlobalModel`), `spherical_linear.py`
is the Bayesian linear-regression surrogate (`BLRGlobalModel`) together with the
`Projection` geometries its features are read through. Both implement the
engine's `GlobalModel` protocol, which stays in `engine.py` because the engine
owns its seams.

`soar_lite/domain.py` - the box these models draw from and carry as their
`domain` - deliberately stays outside this package: the engine reads that same
object for its clip, so the domain is a shared leaf, not a member of the model
family.
"""

from soar_lite.global_model.sample import SampleGlobalModel
from soar_lite.global_model.spherical_linear import (
    AcquisitionFn,
    BLRGlobalModel,
    IdentityProjection,
    MinimizeFn,
    Projection,
    SphereProjection,
)

__all__ = [
    "AcquisitionFn",
    "BLRGlobalModel",
    "IdentityProjection",
    "MinimizeFn",
    "Projection",
    "SampleGlobalModel",
    "SphereProjection",
]
