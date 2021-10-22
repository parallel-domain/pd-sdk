from datetime import datetime
from typing import Any, Dict, List, Set, Union

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.nuscenes.common import NuScenesDataAccessMixin
from paralleldomain.decoding.nuscenes.sensor_frame_decoder import NuScenesCameraSensorFrameDecoder, NuScenesLidarSensorFrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder, LidarSensorFrameDecoder
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath

### MHS: Add NuScenesLidarSensorDecoder
class NuScenesLidarSensorDecoder(LidarSensorDecoder[datetime], NuScenesDataAccessMixin):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        split_name: str,
        scene_name: SceneName,
        settings: DecoderSettings,
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        LidarSensorDecoder.__init__(self=self, dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        NuScenesDataAccessMixin.__init__(
            self=self, dataset_name=dataset_name, split_name=split_name, dataset_path=self._dataset_path
        )

    def _decode_lidar_sensor_frame(
        self, decoder: LidarSensorFrameDecoder[datetime], frame_id: FrameId, lidar_name: SensorName
    ) -> LidarSensorFrame[datetime]:
        return LidarSensorFrame[datetime](sensor_name=lidar_name, frame_id=frame_id, decoder=decoder)

    def _create_lidar_sensor_frame_decoder(self) -> LidarSensorFrameDecoder[datetime]:
        return NuScenesLidarSensorFrameDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            split_name=self.split_name,
            settings=self.settings,
        )

    def _decode_frame_id_set(self, sensor_name: SensorName) -> Set[FrameId]:
        samples = self.nu_samples[self.scene_name]
        sample_tokens = [sample["token"] for sample in samples]
        frame_ids = set()

        data_dict = self.nu_samples_data
        for sample_token in sample_tokens:
            data_list = data_dict[sample_token]
            for data in data_list:
                calib_sensor_token = data["calibrated_sensor_token"]
                calib_sensor = self.nu_calibrated_sensors[calib_sensor_token]
                sensor = self.get_nu_sensor(sensor_token=calib_sensor["sensor_token"])
                if sensor["channel"] == sensor_name:
                    frame_ids.add(sample_token)
        return frame_ids

class NuScenesCameraSensorDecoder(CameraSensorDecoder[datetime], NuScenesDataAccessMixin):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        split_name: str,
        scene_name: SceneName,
        settings: DecoderSettings,
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        CameraSensorDecoder.__init__(self=self, dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        NuScenesDataAccessMixin.__init__(
            self=self, dataset_name=dataset_name, split_name=split_name, dataset_path=self._dataset_path
        )

    def _decode_camera_sensor_frame(
        self, decoder: CameraSensorFrameDecoder[datetime], frame_id: FrameId, camera_name: SensorName
    ) -> CameraSensorFrame[datetime]:
        return CameraSensorFrame[datetime](sensor_name=camera_name, frame_id=frame_id, decoder=decoder)

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[datetime]:
        return NuScenesCameraSensorFrameDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            split_name=self.split_name,
            settings=self.settings,
        )

    def _decode_frame_id_set(self, sensor_name: SensorName) -> Set[FrameId]:
        samples = self.nu_samples[self.scene_name]
        sample_tokens = [sample["token"] for sample in samples]
        frame_ids = set()

        data_dict = self.nu_samples_data
        for sample_token in sample_tokens:
            data_list = data_dict[sample_token]
            for data in data_list:
                calib_sensor_token = data["calibrated_sensor_token"]
                calib_sensor = self.nu_calibrated_sensors[calib_sensor_token]
                sensor = self.get_nu_sensor(sensor_token=calib_sensor["sensor_token"])
                if sensor["channel"] == sensor_name:
                    frame_ids.add(sample_token)
        return frame_ids

