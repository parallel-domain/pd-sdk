import abc
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Set

from paralleldomain.common.dgp.v0.dtos import SceneDataDTO, SceneSampleDTO
from paralleldomain.decoding.dgp.sensor_frame_decoder import DGPCameraSensorFrameDecoder, DGPLidarSensorFrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder, SensorDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder, LidarSensorFrameDecoder
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame
from paralleldomain.model.transformation import Transformation
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class DGPSensorDecoder(SensorDecoder[datetime], metaclass=abc.ABCMeta):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        scene_samples: Dict[FrameId, SceneSampleDTO],
        scene_data: List[SceneDataDTO],
        custom_reference_to_box_bottom: Transformation,
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name)
        self.scene_data = scene_data
        self.custom_reference_to_box_bottom = custom_reference_to_box_bottom
        self.scene_samples = scene_samples
        self.dataset_path = dataset_path

    def _decode_frame_id_set(self, sensor_name: SensorName) -> Set[FrameId]:
        return set()  # Todo


class DGPCameraSensorDecoder(DGPSensorDecoder, CameraSensorDecoder[datetime]):
    def _decode_camera_sensor_frame(
        self, decoder: CameraSensorFrameDecoder[datetime], frame_id: FrameId, camera_name: SensorName
    ) -> CameraSensorFrame[datetime]:
        return CameraSensorFrame[datetime](sensor_name=camera_name, frame_id=frame_id, decoder=decoder)

    @lru_cache(maxsize=1)
    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[datetime]:
        return DGPCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            scene_samples=self.scene_samples,
            scene_data=self.scene_data,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
        )


class DGPLidarSensorDecoder(DGPSensorDecoder, LidarSensorDecoder[datetime]):
    @lru_cache(maxsize=1)
    def _create_lidar_sensor_frame_decoder(self) -> DGPLidarSensorFrameDecoder:
        return DGPLidarSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            scene_samples=self.scene_samples,
            scene_data=self.scene_data,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
        )

    def _decode_lidar_sensor_frame(
        self, decoder: LidarSensorFrameDecoder[datetime], frame_id: FrameId, lidar_name: SensorName
    ) -> LidarSensorFrame[datetime]:
        return LidarSensorFrame[datetime](sensor_name=lidar_name, frame_id=frame_id, decoder=decoder)
