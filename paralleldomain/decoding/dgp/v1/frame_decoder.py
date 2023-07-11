from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List

from paralleldomain.common.dgp.v1 import sample_pb2
from paralleldomain.common.dgp.v1.utils import timestamp_to_datetime
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.dgp.v1.common import map_container_to_dict
from paralleldomain.decoding.dgp.v1.sensor_frame_decoder import (
    DGPCameraSensorFrameDecoder,
    DGPLidarSensorFrameDecoder,
    DGPRadarSensorFrameDecoder,
)
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_frame_decoder import (
    CameraSensorFrameDecoder,
    LidarSensorFrameDecoder,
    RadarSensorFrameDecoder,
)
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, RadarSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.transformation import Transformation


class DGPFrameDecoder(FrameDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        scene_samples: Dict[FrameId, sample_pb2.Sample],
        scene_data: List[sample_pb2.Datum],
        ontologies: Dict[str, str],
        custom_reference_to_box_bottom: Transformation,
        point_cache_folder_exists: bool,
        settings: DecoderSettings,
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self.scene_data = scene_data
        self.custom_reference_to_box_bottom = custom_reference_to_box_bottom
        self.scene_samples = scene_samples
        self.dataset_path = dataset_path
        self._ontologies = ontologies
        self._data_by_key = lru_cache(maxsize=1)(self._data_by_key)
        self._point_cache_folder_exists = point_cache_folder_exists

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
        return timestamp_to_datetime(sample.id.timestamp)

    def _data_by_key(self) -> Dict[str, sample_pb2.Datum]:
        return {d.key: d for d in self.scene_data}

    def _decode_available_sensor_names(self, frame_id: FrameId) -> List[SensorName]:
        sample = self.scene_samples[frame_id]
        sensor_data = self._data_by_key()
        return [sensor_data[key].id.name for key in sample.datum_keys]

    def _decode_available_camera_names(self, frame_id: FrameId) -> List[SensorName]:
        sample = self.scene_samples[frame_id]
        sensor_data = self._data_by_key()
        return [sensor_data[key].id.name for key in sample.datum_keys if sensor_data[key].datum.HasField("image")]

    def _decode_available_lidar_names(self, frame_id: FrameId) -> List[SensorName]:
        sample = self.scene_samples[frame_id]
        sensor_data = self._data_by_key()
        return [sensor_data[key].id.name for key in sample.datum_keys if sensor_data[key].datum.HasField("point_cloud")]

    def _decode_available_radar_names(self, frame_id: FrameId) -> List[SensorName]:
        sample = self.scene_samples[frame_id]
        sensor_data = self._data_by_key()
        return [
            sensor_data[key].id.name
            for key in sample.datum_keys
            if sensor_data[key].datum.HasField("radar_point_cloud")
        ]

    def _decode_metadata(self, frame_id: FrameId) -> Dict[str, Any]:
        sample = self.scene_samples[frame_id]
        return map_container_to_dict(attributes=sample.metadata)

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[datetime]:
        return DGPCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            scene_samples=self.scene_samples,
            scene_data=self.scene_data,
            ontologies=self._ontologies,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
            point_cache_folder_exists=self._point_cache_folder_exists,
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
            ontologies=self._ontologies,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
            point_cache_folder_exists=self._point_cache_folder_exists,
        )

    def _decode_lidar_sensor_frame(
        self, decoder: LidarSensorFrameDecoder[datetime], frame_id: FrameId, sensor_name: SensorName
    ) -> LidarSensorFrame[datetime]:
        return LidarSensorFrame[datetime](sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)

    def _create_radar_sensor_frame_decoder(self) -> RadarSensorFrameDecoder[datetime]:
        return DGPRadarSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            scene_samples=self.scene_samples,
            scene_data=self.scene_data,
            ontologies=self._ontologies,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
            point_cache_folder_exists=self._point_cache_folder_exists,
        )

    def _decode_radar_sensor_frame(
        self, decoder: RadarSensorFrameDecoder[datetime], frame_id: FrameId, sensor_name: SensorName
    ) -> RadarSensorFrame[datetime]:
        return RadarSensorFrame[datetime](sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)
