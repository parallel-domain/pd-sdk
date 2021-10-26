from datetime import datetime
from typing import Any, Dict, Generator, List, Union

import numpy as np
from iso8601 import iso8601
from pyquaternion import Quaternion

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.frame_decoder import FrameDecoder, TDateTime
from paralleldomain.decoding.nuscenes.common import NUSCENES_IMU_TO_INTERNAL_CS, NuScenesDataAccessMixin
from paralleldomain.decoding.nuscenes.sensor_frame_decoder import (
    NuScenesCameraSensorFrameDecoder,
    NuScenesLidarSensorFrameDecoder,
)
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder, LidarSensorFrameDecoder
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class NuScenesFrameDecoder(FrameDecoder[datetime], NuScenesDataAccessMixin):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        split_name: str,
        scene_name: SceneName,
        settings: DecoderSettings,
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        FrameDecoder.__init__(self=self, dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        NuScenesDataAccessMixin.__init__(
            self=self, dataset_name=dataset_name, split_name=split_name, dataset_path=self._dataset_path
        )

    def _decode_ego_pose(self, frame_id: FrameId) -> EgoPose:
        trans = self.get_ego_pose(log_token=self.scene_name, frame_id=frame_id)
        return EgoPose.from_transformation_matrix(mat=trans)

    def _decode_available_sensor_names(self, frame_id: FrameId) -> List[SensorName]:
        sensor_names = set()
        for data in self.get_sample_data_with_frame_id(scene_token=self.scene_name, frame_id=frame_id):
            calib_sensor_token = data["calibrated_sensor_token"]
            calib_sensor = self.nu_calibrated_sensors[calib_sensor_token]
            sensor = self.get_nu_sensor(sensor_token=calib_sensor["sensor_token"])
            sensor_names.add(sensor["channel"])
        return list(sensor_names)

    def _decode_available_camera_names(self, frame_id: FrameId) -> List[SensorName]:
        camera_names = set()
        for data in self.get_sample_data_with_frame_id(scene_token=self.scene_name, frame_id=frame_id):
            calib_sensor_token = data["calibrated_sensor_token"]
            calib_sensor = self.nu_calibrated_sensors[calib_sensor_token]
            sensor = self.get_nu_sensor(sensor_token=calib_sensor["sensor_token"])
            if sensor["modality"] == "camera":
                camera_names.add(sensor["channel"])
        return list(camera_names)

    def _decode_available_lidar_names(self, frame_id: FrameId) -> List[SensorName]:
        lidar_names = set()
        for data in self.get_sample_data_with_frame_id(scene_token=self.scene_name, frame_id=frame_id):
            calib_sensor_token = data["calibrated_sensor_token"]
            calib_sensor = self.nu_calibrated_sensors[calib_sensor_token]
            sensor = self.get_nu_sensor(sensor_token=calib_sensor["sensor_token"])
            if sensor["modality"] == "lidar":
                lidar_names.add(sensor["channel"])
        return list(lidar_names)

    def _decode_datetime(self, frame_id: FrameId) -> datetime:
        sample_timestamp = self.get_sample_with_frame_id(scene_token=self.scene_name, frame_id=frame_id)["timestamp"]
        return datetime.fromtimestamp(sample_timestamp / 1000000)

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[TDateTime]:
        return NuScenesCameraSensorFrameDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            split_name=self.split_name,
            settings=self.settings,
        )

    def _decode_camera_sensor_frame(
        self, decoder: CameraSensorFrameDecoder[TDateTime], frame_id: FrameId, sensor_name: SensorName
    ) -> CameraSensorFrame[TDateTime]:
        return CameraSensorFrame[datetime](sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)

    def _create_lidar_sensor_frame_decoder(self) -> LidarSensorFrameDecoder[TDateTime]:
        return NuScenesLidarSensorFrameDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            split_name=self.split_name,
            settings=self.settings,
        )

    def _decode_lidar_sensor_frame(
        self, decoder: LidarSensorFrameDecoder[TDateTime], frame_id: FrameId, sensor_name: SensorName
    ) -> LidarSensorFrame[TDateTime]:
        return LidarSensorFrame[datetime](sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)
