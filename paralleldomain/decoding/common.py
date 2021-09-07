from typing import Optional, TypeVar

from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.lazy_load_cache import LAZY_LOAD_CACHE, LazyLoadCache, cache_max_ram_usage_factor

T = TypeVar("T")


def create_cache_key(
    dataset_name: str,
    scene_name: Optional[SceneName] = None,
    frame_id: Optional[FrameId] = None,
    sensor_name: Optional[SensorName] = None,
    extra: Optional[str] = None,
) -> str:
    return "-".join([v for v in [dataset_name, scene_name, frame_id, sensor_name, extra] if v is not None])


class LazyLoadPropertyMixin:
    @property
    def lazy_load_cache(self) -> LazyLoadCache:
        return LAZY_LOAD_CACHE
