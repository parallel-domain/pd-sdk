from datetime import datetime
from typing import Any, Dict, List

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.kitti_flow.common import frame_id_to_timestamp
from paralleldomain.decoding.kitti_flow.sensor_frame_decoder import KITTIFlowCameraSensorFrameDecoder
from paralleldomain.decoding.sensor_frame_decoder import (
    CameraSensorFrameDecoder,
    LidarSensorFrameDecoder,
    RadarSensorFrameDecoder,
)
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class KITTIFlowFrameDecoder(FrameDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        image_folder: str,
        occ_optical_flow_folder: str,
        noc_optical_flow_folder: str,
        use_non_occluded: bool,
        camera_name: str,
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self.dataset_path = dataset_path
        self._image_folder = image_folder
        self._occ_optical_flow_folder = occ_optical_flow_folder
        self._noc_optical_flow_folder = noc_optical_flow_folder
        self._use_non_occluded = use_non_occluded
        self.camera_name = camera_name

    def _decode_ego_pose(self, frame_id: FrameId) -> EgoPose:
        raise ValueError("KITTI-flow does not support ego pose!")

    def _decode_available_sensor_names(self, frame_id: FrameId) -> List[SensorName]:
        return [self.camera_name]

    def _decode_available_camera_names(self, frame_id: FrameId) -> List[SensorName]:
        return [self.camera_name]

    def _decode_available_lidar_names(self, frame_id: FrameId) -> List[SensorName]:
        raise ValueError("KITTI-flow does not support lidar data!")

    def _decode_datetime(self, frame_id: FrameId) -> datetime:
        return frame_id_to_timestamp(frame_id=frame_id)

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[datetime]:
        return KITTIFlowCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            settings=self.settings,
            image_folder=self._image_folder,
            occ_optical_flow_folder=self._occ_optical_flow_folder,
            noc_optical_flow_folder=self._noc_optical_flow_folder,
            use_non_occluded=self._use_non_occluded,
        )

    def _create_lidar_sensor_frame_decoder(self) -> LidarSensorFrameDecoder[datetime]:
        raise ValueError("KITTI-flow does not support lidar data!")

    def _decode_available_radar_names(self, frame_id: FrameId) -> List[SensorName]:
        raise ValueError("KITTI-flow does not support radar data!")

    def _create_radar_sensor_frame_decoder(self) -> RadarSensorFrameDecoder[datetime]:
        raise ValueError("KITTI-flow does not support radar data!")

    def _decode_metadata(self, frame_id: FrameId) -> Dict[str, Any]:
        return dict()
