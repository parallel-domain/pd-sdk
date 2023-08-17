import abc
from datetime import datetime
from functools import lru_cache
from typing import Set

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.data_stream.data_accessor import DataStreamDataAccessor
from paralleldomain.decoding.data_stream.sensor_frame_decoder import DataStreamCameraSensorFrameDecoder
from paralleldomain.decoding.sensor_decoder import SensorDecoder, CameraSensorDecoder, TDateTime
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.model.type_aliases import SensorName, FrameId, SceneName


class DataStreamSensorDecoder(SensorDecoder[datetime], metaclass=abc.ABCMeta):
    def __init__(
        self, dataset_name: str, scene_name: SceneName, settings: DecoderSettings, data_accessor: DataStreamDataAccessor
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self._data_accessor = data_accessor

    def _decode_frame_id_set(self, sensor_name: SensorName) -> Set[FrameId]:
        return self._data_accessor.get_frame_ids()


class DataStreamCameraSensorDecoder(DataStreamSensorDecoder, CameraSensorDecoder[datetime]):
    def __init__(
        self, dataset_name: str, scene_name: SceneName, settings: DecoderSettings, data_accessor: DataStreamDataAccessor
    ):
        DataStreamSensorDecoder.__init__(self=self, dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        CameraSensorDecoder.__init__(self=self, dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self._create_camera_sensor_frame_decoder = lru_cache(maxsize=1)(self._create_camera_sensor_frame_decoder)
        self._data_accessor = data_accessor

    def _decode_camera_sensor_frame(
        self, decoder: CameraSensorFrameDecoder[datetime], frame_id: FrameId, camera_name: SensorName
    ) -> CameraSensorFrame[datetime]:
        return CameraSensorFrame[datetime](sensor_name=camera_name, frame_id=frame_id, decoder=decoder)

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[TDateTime]:
        return DataStreamCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            settings=self.settings,
            data_accessor=self._data_accessor,
        )
