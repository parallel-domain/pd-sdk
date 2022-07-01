from datetime import datetime
from functools import lru_cache
from typing import List, Optional, Set

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.decoding.waymo.common import get_record_iterator
from paralleldomain.decoding.waymo.sensor_frame_decoder import WaymoOpenDatasetCameraSensorFrameDecoder
from paralleldomain.model.class_mapping import ClassDetail
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class WaymoOpenDatasetCameraSensorDecoder(CameraSensorDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self._dataset_path = dataset_path

    def _decode_frame_id_set(self, sensor_name: SensorName) -> Set[FrameId]:
        record = self._dataset_path / self.scene_name
        frame_ids = list()
        for _, frame_id in get_record_iterator(record_path=record, read_frame=False):
            frame_ids.append(frame_id)
        return frame_ids

    def _decode_camera_sensor_frame(
        self, decoder: CameraSensorFrameDecoder[datetime], frame_id: FrameId, camera_name: SensorName
    ) -> CameraSensorFrame[None]:
        return CameraSensorFrame[None](sensor_name=camera_name, frame_id=frame_id, decoder=decoder)

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[None]:
        return WaymoOpenDatasetCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self._dataset_path,
            settings=self.settings,
        )
