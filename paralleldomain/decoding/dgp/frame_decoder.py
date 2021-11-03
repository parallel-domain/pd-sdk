import abc
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List

from paralleldomain.common.dgp.v0.dtos import SceneDataDTO, SceneSampleDTO, scene_sample_to_date_time
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.dgp.sensor_frame_decoder import DGPCameraSensorFrameDecoder, DGPLidarSensorFrameDecoder
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder, LidarSensorFrameDecoder
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.transformation import Transformation


class DGPFrameDecoder(FrameDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        scene_samples: Dict[FrameId, SceneSampleDTO],
        scene_data: List[SceneDataDTO],
        custom_reference_to_box_bottom: Transformation,
        settings: DecoderSettings,
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self.scene_data = scene_data
        self.custom_reference_to_box_bottom = custom_reference_to_box_bottom
        self.scene_samples = scene_samples
        self.dataset_path = dataset_path

    def _decode_ego_pose(self, frame_id: FrameId) -> EgoPose:
        sensor_name = next(iter(self._decode_available_camera_names(frame_id=frame_id)), None)
        if sensor_name is None:
            sensor_name = next(iter(self._decode_available_lidar_names(frame_id=frame_id)))
            sensor_frame = self._decode_lidar_sensor_frame(
                frame_id=frame_id, sensor_name=sensor_name, decoder=self._create_lidar_sensor_frame_decoder()
            )
        else:
            sensor_frame = self._decode_camera_sensor_frame(
                frame_id=frame_id, sensor_name=sensor_name, decoder=self._create_camera_sensor_frame_decoder()
            )

        vehicle_pose = sensor_frame.pose @ sensor_frame.extrinsic.inverse
        return EgoPose(quaternion=vehicle_pose.quaternion, translation=vehicle_pose.translation)

    def _decode_datetime(self, frame_id: FrameId) -> datetime:
        sample = self.scene_samples[frame_id]
        return scene_sample_to_date_time(sample=sample)

    @lru_cache(maxsize=1)
    def _data_by_key(self) -> Dict[str, SceneDataDTO]:
        return {d.key: d for d in self.scene_data}

    def _decode_available_sensor_names(self, frame_id: FrameId) -> List[SensorName]:
        sample = self.scene_samples[frame_id]
        sensor_data = self._data_by_key()
        return [sensor_data[key].id.name for key in sample.datum_keys]

    def _decode_available_camera_names(self, frame_id: FrameId) -> List[SensorName]:
        sample = self.scene_samples[frame_id]
        sensor_data = self._data_by_key()
        return [sensor_data[key].id.name for key in sample.datum_keys if sensor_data[key].datum.image]

    def _decode_available_lidar_names(self, frame_id: FrameId) -> List[SensorName]:
        sample = self.scene_samples[frame_id]
        sensor_data = self._data_by_key()
        return [sensor_data[key].id.name for key in sample.datum_keys if sensor_data[key].datum.point_cloud]

    def _decode_metadata(self, frame_id: FrameId) -> Dict[str, Any]:
        sample = self.scene_samples[frame_id]
        return sample.metadata

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[datetime]:
        return DGPCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            scene_samples=self.scene_samples,
            scene_data=self.scene_data,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
        )

    def _decode_camera_sensor_frame(
        self, decoder: CameraSensorFrameDecoder[datetime], frame_id: FrameId, sensor_name: SensorName
    ) -> CameraSensorFrame[datetime]:
        return CameraSensorFrame[datetime](sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)

    def _create_lidar_sensor_frame_decoder(self) -> LidarSensorFrameDecoder[datetime]:
        return DGPLidarSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            scene_samples=self.scene_samples,
            scene_data=self.scene_data,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
        )

    def _decode_lidar_sensor_frame(
        self, decoder: LidarSensorFrameDecoder[datetime], frame_id: FrameId, sensor_name: SensorName
    ) -> LidarSensorFrame[datetime]:
        return LidarSensorFrame[datetime](sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)
