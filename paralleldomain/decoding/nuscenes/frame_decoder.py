from datetime import datetime
from typing import Any, Dict, List, Union

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.frame_decoder import FrameDecoder, TDateTime
from paralleldomain.decoding.nuscenes.common import NuScenesDataAccessMixin
from paralleldomain.decoding.nuscenes.sensor_frame_decoder import (
    NuScenesCameraSensorFrameDecoder,
    NuScenesLidarSensorFrameDecoder,
)
from paralleldomain.decoding.sensor_frame_decoder import (
    CameraSensorFrameDecoder,
    LidarSensorFrameDecoder,
    RadarSensorFrameDecoder,
)
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class NuScenesFrameDecoder(FrameDecoder[datetime], NuScenesDataAccessMixin):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        split_name: str,
        scene_name: SceneName,
        frame_id: FrameId,
        settings: DecoderSettings,
        is_unordered_scene: bool,
        scene_decoder,
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        FrameDecoder.__init__(
            self=self,
            dataset_name=dataset_name,
            scene_name=scene_name,
            frame_id=frame_id,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        NuScenesDataAccessMixin.__init__(
            self=self, dataset_name=dataset_name, split_name=split_name, dataset_path=self._dataset_path
        )
        self.scene_token = self.nu_scene_name_to_scene_token[scene_name]

    def _decode_ego_pose(self) -> EgoPose:
        trans = self.get_ego_pose(scene_token=self.scene_token, frame_id=self.frame_id)
        return EgoPose.from_transformation_matrix(mat=trans)

    def _decode_available_sensor_names_by_modality(self, modality: List[str] = None) -> List[SensorName]:
        sensor_names = set()
        if modality is None:
            modality = ["camera", "lidar"]
        data_list = self.nu_samples_data[self.frame_id]
        sensors = self.nu_calibrated_sensors
        for data in data_list:
            calib_sensor_token = data["calibrated_sensor_token"]
            calib_sensor = sensors[calib_sensor_token]
            sensor = self.get_nu_sensor(sensor_token=calib_sensor["sensor_token"])
            if sensor["modality"] in modality:
                sensor_names.add(sensor["channel"])
        return list(sensor_names)

    def _decode_available_sensor_names(self) -> List[SensorName]:
        return self._decode_available_sensor_names_by_modality(modality=["camera", "lidar"])

    def _decode_available_camera_names(self) -> List[SensorName]:
        return self._decode_available_sensor_names_by_modality(modality=["camera"])

    def _decode_available_lidar_names(self) -> List[SensorName]:
        return self._decode_available_sensor_names_by_modality(modality=["lidar"])

    def _decode_metadata(self) -> Dict[str, Any]:
        return dict()

    def _decode_datetime(self) -> datetime:
        return self.get_datetime_with_frame_id(scene_token=self.scene_token, frame_id=self.frame_id)

    def _create_camera_sensor_frame_decoder(self, sensor_name: SensorName) -> CameraSensorFrameDecoder[TDateTime]:
        return NuScenesCameraSensorFrameDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=sensor_name,
            frame_id=self.frame_id,
            split_name=self.split_name,
            settings=self.settings,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )

    def _create_lidar_sensor_frame_decoder(self, sensor_name: SensorName) -> LidarSensorFrameDecoder[TDateTime]:
        return NuScenesLidarSensorFrameDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=sensor_name,
            frame_id=self.frame_id,
            split_name=self.split_name,
            settings=self.settings,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )

    def _decode_available_radar_names(self) -> List[SensorName]:
        """Not supported yet"""
        return list()

    def _create_radar_sensor_frame_decoder(self, sensor_name: SensorName) -> RadarSensorFrameDecoder[TDateTime]:
        raise ValueError("Currently do not support radar data!")
