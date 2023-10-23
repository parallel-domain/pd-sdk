from datetime import datetime
from functools import lru_cache
from typing import Set

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.flying_chairs.sensor_frame_decoder import FlyingChairsCameraSensorFrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class FlyingChairsCameraSensorDecoder(CameraSensorDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        image_folder: str,
        optical_flow_folder: str,
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
        self._dataset_path = dataset_path
        self._image_folder = image_folder
        self._optical_flow_folder = optical_flow_folder

    def _decode_frame_id_set(self) -> Set[FrameId]:
        frame_ids = {self.scene_name + "_img1.ppm", self.scene_name + "_img2.ppm"}
        return frame_ids

    def _create_camera_sensor_frame_decoder(self, frame_id: FrameId) -> CameraSensorFrameDecoder[datetime]:
        return FlyingChairsCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=self.sensor_name,
            frame_id=frame_id,
            dataset_path=self._dataset_path,
            settings=self.settings,
            image_folder=self._image_folder,
            optical_flow_folder=self._optical_flow_folder,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )
