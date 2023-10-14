from datetime import datetime
from typing import Any, Dict, List

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.flying_chairs.common import frame_id_to_timestamp
from paralleldomain.decoding.flying_chairs.sensor_frame_decoder import FlyingChairsCameraSensorFrameDecoder
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_frame_decoder import (
    CameraSensorFrameDecoder,
    LidarSensorFrameDecoder,
    RadarSensorFrameDecoder,
)
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class FlyingChairsFrameDecoder(FrameDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        frame_id: FrameId,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        image_folder: str,
        optical_flow_folder: str,
        camera_name: str,
        is_unordered_scene: bool,
        scene_decoder,
    ):
        super().__init__(
            dataset_name=dataset_name,
            scene_name=scene_name,
            frame_id=frame_id,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        self.dataset_path = dataset_path
        self._image_folder = image_folder
        self._optical_flow_folder = optical_flow_folder
        self.camera_name = camera_name

    def _decode_ego_pose(self) -> EgoPose:
        raise ValueError("FlyingChairs does not support ego pose!")

    def _decode_available_sensor_names(self) -> List[SensorName]:
        return [self.camera_name]

    def _decode_available_camera_names(self) -> List[SensorName]:
        return [self.camera_name]

    def _decode_available_lidar_names(self) -> List[SensorName]:
        raise ValueError("FlyingChairs does not support lidar data!")

    def _decode_datetime(self) -> datetime:
        return frame_id_to_timestamp(frame_id=self.frame_id)

    def _create_camera_sensor_frame_decoder(self, sensor_name: SensorName) -> CameraSensorFrameDecoder[datetime]:
        return FlyingChairsCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            frame_id=self.frame_id,
            sensor_name=sensor_name,
            dataset_path=self.dataset_path,
            settings=self.settings,
            image_folder=self._image_folder,
            optical_flow_folder=self._optical_flow_folder,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )

    def _create_lidar_sensor_frame_decoder(self, sensor_name: SensorName) -> LidarSensorFrameDecoder[datetime]:
        raise ValueError("FlyingChairs does not support lidar data!")

    def _decode_available_radar_names(self) -> List[SensorName]:
        raise ValueError("FlyingChairs does not support radar data!")

    def _create_radar_sensor_frame_decoder(self, sensor_name: SensorName) -> RadarSensorFrameDecoder[datetime]:
        raise ValueError("FlyingChairs does not support radar data!")

    def _decode_metadata(self) -> Dict[str, Any]:
        return dict()
