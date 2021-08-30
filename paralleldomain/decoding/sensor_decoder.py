import abc
from typing import Optional, Set, TypeVar

from paralleldomain.decoding.common import create_cache_key
from paralleldomain.decoding.sensor_frame_decoder import SensorFrameDecoder
from paralleldomain.model.sensor import SensorFrame
from paralleldomain.model.type_aliases import FrameId, SensorFrameSetName, SensorName
from paralleldomain.utilities.lazy_load_cache import LazyLoadCache

T = TypeVar("T")


class SensorDecoder:
    def __init__(self, dataset_name: str, set_name: SensorFrameSetName, lazy_load_cache: LazyLoadCache):
        self.set_name = set_name
        self.lazy_load_cache = lazy_load_cache
        self.dataset_name = dataset_name

    def get_unique_sensor_id(
        self, sensor_name: SensorName, frame_id: Optional[FrameId] = None, extra: Optional[str] = None
    ) -> str:
        return create_cache_key(
            dataset_name=self.dataset_name,
            set_name=self.set_name,
            sensor_name=sensor_name,
            frame_id=frame_id,
            extra=extra,
        )

    def get_frame_ids(self, sensor_name: SensorName) -> Set[FrameId]:
        _unique_cache_key = self.get_unique_sensor_id(sensor_name=sensor_name, extra="frame_ids")
        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: self._decode_frame_id_set(sensor_name=sensor_name),
        )

    @abc.abstractmethod
    def _decode_frame_id_set(self, sensor_name: SensorName) -> Set[FrameId]:
        pass

    @abc.abstractmethod
    def _decode_sensor_frame(
        self, decoder: SensorFrameDecoder, frame_id: FrameId, sensor_name: SensorName
    ) -> SensorFrame:
        pass

    @abc.abstractmethod
    def _create_sensor_frame_decoder(self) -> SensorFrameDecoder:
        pass

    def get_sensor_frame(self, frame_id: FrameId, sensor_name: SensorName) -> SensorFrame:
        unique_sensor_id = self.get_unique_sensor_id(frame_id=frame_id, sensor_name=sensor_name, extra="SensorFrame")
        return self.lazy_load_cache.get_item(
            key=unique_sensor_id,
            loader=lambda: self._decode_sensor_frame(
                decoder=self._create_sensor_frame_decoder(),
                frame_id=frame_id,
                sensor_name=sensor_name,
            ),
        )
