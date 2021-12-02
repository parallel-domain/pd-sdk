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
        self.scene_token = self.nu_scene_name_to_scene_token[scene_name]

    def _decode_ego_pose(self, frame_id: FrameId) -> EgoPose:
        trans = self.get_ego_pose(scene_token=self.scene_name, frame_id=frame_id)
        return EgoPose.from_transformation_matrix(mat=trans)

    def _decode_available_sensor_names_by_modality(
        self, frame_id: FrameId, modality: List[str] = None
    ) -> List[SensorName]:
        sensor_names = set()
        if modality is None:
            modality = ["camera", "lidar"]
        data_list = self.nu_samples_data[frame_id]
        sensors = self.nu_calibrated_sensors
        for data in data_list:
            calib_sensor_token = data["calibrated_sensor_token"]
            calib_sensor = sensors[calib_sensor_token]
            sensor = self.get_nu_sensor(sensor_token=calib_sensor["sensor_token"])
            if sensor["modality"] in modality:
                sensor_names.add(sensor["channel"])
        return list(sensor_names)

    def _decode_available_sensor_names(self, frame_id: FrameId) -> List[SensorName]:
        return self._decode_available_sensor_names_by_modality(frame_id=frame_id, modality=["camera", "lidar"])

    def _decode_available_camera_names(self, frame_id: FrameId) -> List[SensorName]:
        return self._decode_available_sensor_names_by_modality(frame_id=frame_id, modality=["camera"])

    def _decode_available_lidar_names(self, frame_id: FrameId) -> List[SensorName]:
        return self._decode_available_sensor_names_by_modality(frame_id=frame_id, modality=["lidar"])

    def _decode_metadata(self, frame_id: FrameId) -> Dict[str, Any]:
        return dict()

    def _decode_datetime(self, frame_id: FrameId) -> datetime:
        return self.get_datetime_with_frame_id(self.scene_token, frame_id)

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
