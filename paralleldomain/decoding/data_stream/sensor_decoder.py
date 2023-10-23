import abc
from datetime import datetime
from typing import Set

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.data_stream.data_accessor import DataStreamDataAccessor
from paralleldomain.decoding.data_stream.sensor_frame_decoder import DataStreamCameraSensorFrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, SensorDecoder, TDateTime
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName


class DataStreamSensorDecoder(SensorDecoder[datetime], metaclass=abc.ABCMeta):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        settings: DecoderSettings,
        data_accessor: DataStreamDataAccessor,
        is_unordered_scene: bool,
        scene_decoder,
    ):
        super().__init__(
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        self._data_accessor = data_accessor

    def _decode_frame_id_set(self) -> Set[FrameId]:
        return {fid for fid, sensors in self._data_accessor.sensors.items() if self.sensor_name in sensors}


class DataStreamCameraSensorDecoder(DataStreamSensorDecoder, CameraSensorDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        settings: DecoderSettings,
        data_accessor: DataStreamDataAccessor,
        is_unordered_scene: bool,
        scene_decoder,
    ):
        DataStreamSensorDecoder.__init__(
            self=self,
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
            data_accessor=data_accessor,
        )
        CameraSensorDecoder.__init__(
            self=self,
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        self._data_accessor = data_accessor

    def _create_camera_sensor_frame_decoder(self, frame_id: FrameId) -> CameraSensorFrameDecoder[TDateTime]:
        return DataStreamCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=self.sensor_name,
            frame_id=frame_id,
            settings=self.settings,
            data_accessor=self._data_accessor,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )
