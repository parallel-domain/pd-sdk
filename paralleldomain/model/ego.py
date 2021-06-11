from paralleldomain.model.transformation import Transformation
from paralleldomain.utilities.lazy_load_cache import LAZY_LOAD_CACHE

try:
    from typing import Protocol, Callable
except ImportError:
    from typing_extensions import Protocol  # type: ignore


class EgoPose(Transformation):
    ...


class EgoFrame:
    """
    This Objects contains informations of a frame of the Ego vehicle/drone/person the sensor rig was attached to.
    """

    def __init__(self,  unique_cache_key: str, pose_loader: Callable[[], EgoPose]):
        self._unique_cache_key = unique_cache_key
        self._pose_loader = pose_loader

    @property
    def pose(self) -> EgoPose:
        return LAZY_LOAD_CACHE.get_item(key=self._unique_cache_key + "pose", loader=self._pose_loader)
