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
    return "-".join([v for v in [dataset_name, set_name, frame_id, sensor_name, extra] if v is not None])
