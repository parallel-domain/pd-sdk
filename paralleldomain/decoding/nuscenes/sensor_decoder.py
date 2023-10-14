from datetime import datetime
from typing import Set, Union

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.nuscenes.common import NuScenesDataAccessMixin
from paralleldomain.decoding.nuscenes.sensor_frame_decoder import (
    NuScenesCameraSensorFrameDecoder,
    NuScenesLidarSensorFrameDecoder,
)
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder, LidarSensorFrameDecoder
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class NuScenesLidarSensorDecoder(LidarSensorDecoder[datetime], NuScenesDataAccessMixin):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        split_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        settings: DecoderSettings,
        is_unordered_scene: bool,
        scene_decoder,
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        LidarSensorDecoder.__init__(
            self=self,
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        NuScenesDataAccessMixin.__init__(
            self=self, dataset_name=dataset_name, split_name=split_name, dataset_path=self._dataset_path
        )
        self.scene_token = self.nu_scene_name_to_scene_token[scene_name]

    def _create_lidar_sensor_frame_decoder(self, frame_id: FrameId) -> LidarSensorFrameDecoder[datetime]:
        return NuScenesLidarSensorFrameDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            frame_id=frame_id,
            sensor_name=self.sensor_name,
            split_name=self.split_name,
            settings=self.settings,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )

    def _decode_frame_id_set(self) -> Set[FrameId]:
        samples = self.nu_samples[self.scene_token]
        sample_tokens = [sample["token"] for sample in samples]
        frame_ids = set()

        data_dict = self.nu_samples_data
        for sample_token in sample_tokens:
            data_list = data_dict[sample_token]
            for data in data_list:
                calib_sensor_token = data["calibrated_sensor_token"]
                calib_sensor = self.nu_calibrated_sensors[calib_sensor_token]
                sensor = self.get_nu_sensor(sensor_token=calib_sensor["sensor_token"])
                if sensor["channel"] == self.sensor_name:
                    frame_ids.add(sample_token)
        return frame_ids


class NuScenesCameraSensorDecoder(CameraSensorDecoder[datetime], NuScenesDataAccessMixin):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        split_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        settings: DecoderSettings,
        is_unordered_scene: bool,
        scene_decoder,
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        CameraSensorDecoder.__init__(
            self=self,
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        NuScenesDataAccessMixin.__init__(
            self=self, dataset_name=dataset_name, split_name=split_name, dataset_path=self._dataset_path
        )
        self.scene_token = self.nu_scene_name_to_scene_token[scene_name]

    def _create_camera_sensor_frame_decoder(self, frame_id: FrameId) -> CameraSensorFrameDecoder[datetime]:
        return NuScenesCameraSensorFrameDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            frame_id=frame_id,
            sensor_name=self.sensor_name,
            split_name=self.split_name,
            settings=self.settings,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )

    def _decode_frame_id_set(self) -> Set[FrameId]:
        samples = self.nu_samples[self.scene_token]
        sample_tokens = [sample["token"] for sample in samples]
        frame_ids = set()

        data_dict = self.nu_samples_data
        for sample_token in sample_tokens:
            data_list = data_dict[sample_token]
            for data in data_list:
                calib_sensor_token = data["calibrated_sensor_token"]
                calib_sensor = self.nu_calibrated_sensors[calib_sensor_token]
                sensor = self.get_nu_sensor(sensor_token=calib_sensor["sensor_token"])
                if sensor["channel"] == self.sensor_name:
                    frame_ids.add(sample_token)
        return frame_ids
