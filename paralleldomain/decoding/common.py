from typing import Optional, TypeVar

from paralleldomain.model.type_aliases import FrameId, SensorFrameSetName, SensorName

T = TypeVar("T")


def create_cache_key(
    dataset_name: str,
    set_name: Optional[SensorFrameSetName] = None,
    frame_id: Optional[FrameId] = None,
    sensor_name: Optional[SensorName] = None,
    extra: Optional[str] = None,
) -> str:
    cache_key = f"{dataset_name}"
    if set_name is not None:
        cache_key += f"-{set_name}"
    if frame_id is not None:
        cache_key += f"-{frame_id}"
    if sensor_name is not None:
        cache_key += f"-{sensor_name}"
    if extra is not None:
        cache_key += f"-{extra}"
    return cache_key
