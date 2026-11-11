"""The restart-model family: the engine's two `GlobalModel` concretes.

`sample.py` is the model-free draw (`SampleGlobalModel`), `blr.py` is the
Bayesian linear-regression surrogate (`BLRGlobalModel`), and `projection.py`
holds the `Projection` geometries its features are read through. Both implement
the engine's `GlobalModel` protocol, which stays in `engine/contracts.py`
because the engine owns its seams.

`soar_lite/domain.py` - the box these models draw from on every ask -
deliberately stays outside this package: the engine reads that same
object for its clip, so the domain is a shared leaf, not a member of the model
family.
"""

from soar_lite.global_model.blr import (
    AcquisitionFn,
    BLRGlobalModel,
    MinimizeFn,
)
from soar_lite.global_model.projection import (
    IdentityProjection,
    Projection,
    ProjectionFactory,
    SphereProjection,
)
from soar_lite.global_model.sample import SampleGlobalModel

__all__ = [
    "AcquisitionFn",
    "BLRGlobalModel",
    "IdentityProjection",
    "MinimizeFn",
    "Projection",
    "ProjectionFactory",
    "SampleGlobalModel",
    "SphereProjection",
]
