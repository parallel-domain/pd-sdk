from datetime import datetime
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
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.transformation import Transformation


class DGPFrameDecoder(FrameDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        frame_id: FrameId,
        scene_samples: Dict[FrameId, sample_pb2.Sample],
        scene_data: List[sample_pb2.Datum],
        ontologies: Dict[str, str],
        custom_reference_to_box_bottom: Transformation,
        point_cache_folder_exists: bool,
        settings: DecoderSettings,
        is_unordered_scene: bool,
        scene_decoder,
    ):
        super().__init__(
            dataset_name=dataset_name,
            scene_name=scene_name,
            frame_id=frame_id,
            settings=settings,
            is_unordered_scene=is_unordered_scene,
            scene_decoder=scene_decoder,
        )
        self.frame_sample = scene_samples[frame_id]
        self.sensor_name_to_frame_data = {d.id.name: d for d in scene_data if d.key in self.frame_sample.datum_keys}
        self.custom_reference_to_box_bottom = custom_reference_to_box_bottom
        self.dataset_path = dataset_path
        self._ontologies = ontologies
        self._point_cache_folder_exists = point_cache_folder_exists

    def _decode_ego_pose(self) -> EgoPose:
        sensor_name = next(iter(self._decode_available_camera_names()), None)
        if sensor_name is None:
            sensor_name = next(iter(self._decode_available_lidar_names()))
            sensor_frame = self._decode_lidar_sensor_frame(
                decoder=self._create_lidar_sensor_frame_decoder(sensor_name=sensor_name)
            )
        else:
            sensor_frame = self._decode_camera_sensor_frame(
                decoder=self._create_camera_sensor_frame_decoder(sensor_name=sensor_name)
            )

        vehicle_pose = sensor_frame.pose @ sensor_frame.extrinsic.inverse
        return EgoPose(quaternion=vehicle_pose.quaternion, translation=vehicle_pose.translation)

    def _decode_datetime(self) -> datetime:
        return timestamp_to_datetime(self.frame_sample.id.timestamp)

    def _decode_available_sensor_names(self) -> List[SensorName]:
        return [n for n in self.sensor_name_to_frame_data.keys()]

    def _decode_available_camera_names(self) -> List[SensorName]:
        return [n for n, d in self.sensor_name_to_frame_data.items() if d.datum.HasField("image")]

    def _decode_available_lidar_names(self) -> List[SensorName]:
        return [n for n, d in self.sensor_name_to_frame_data.items() if d.datum.HasField("point_cloud")]

    def _decode_available_radar_names(self) -> List[SensorName]:
        return [n for n, d in self.sensor_name_to_frame_data.items() if d.datum.HasField("radar_point_cloud")]

    def _decode_metadata(self) -> Dict[str, Any]:
        return map_container_to_dict(attributes=self.frame_sample.metadata)

    def _create_camera_sensor_frame_decoder(self, sensor_name: SensorName) -> CameraSensorFrameDecoder[datetime]:
        sensor_frame_data = self.sensor_name_to_frame_data[sensor_name]
        return DGPCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            frame_id=self.frame_id,
            sensor_name=sensor_name,
            dataset_path=self.dataset_path,
            sensor_frame_data=sensor_frame_data,
            frame_sample=self.frame_sample,
            ontologies=self._ontologies,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
            point_cache_folder_exists=self._point_cache_folder_exists,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )

    def _create_lidar_sensor_frame_decoder(self, sensor_name: SensorName) -> LidarSensorFrameDecoder[datetime]:
        sensor_frame_data = self.sensor_name_to_frame_data[sensor_name]
        return DGPLidarSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            frame_id=self.frame_id,
            sensor_name=sensor_name,
            dataset_path=self.dataset_path,
            sensor_frame_data=sensor_frame_data,
            frame_sample=self.frame_sample,
            ontologies=self._ontologies,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
            point_cache_folder_exists=self._point_cache_folder_exists,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )

    def _create_radar_sensor_frame_decoder(self, sensor_name: SensorName) -> RadarSensorFrameDecoder[datetime]:
        sensor_frame_data = self.sensor_name_to_frame_data[sensor_name]
        return DGPRadarSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            frame_id=self.frame_id,
            sensor_name=sensor_name,
            dataset_path=self.dataset_path,
            sensor_frame_data=sensor_frame_data,
            frame_sample=self.frame_sample,
            ontologies=self._ontologies,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
            settings=self.settings,
            point_cache_folder_exists=self._point_cache_folder_exists,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )
