from datetime import datetime
from typing import Set, Union

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.nuimages.common import NuImagesDataAccessMixin
from paralleldomain.decoding.nuimages.sensor_frame_decoder import NuImagesCameraSensorFrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class NuImagesCameraSensorDecoder(CameraSensorDecoder[datetime], NuImagesDataAccessMixin):
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
        NuImagesDataAccessMixin.__init__(
            self=self, dataset_name=dataset_name, split_name=split_name, dataset_path=self._dataset_path
        )

    def _create_camera_sensor_frame_decoder(self, frame_id: FrameId) -> CameraSensorFrameDecoder[datetime]:
        return NuImagesCameraSensorFrameDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=self.sensor_name,
            frame_id=frame_id,
            split_name=self.split_name,
            settings=self.settings,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )

    def _decode_frame_id_set(self) -> Set[FrameId]:
        samples = self.nu_samples[self.scene_name]
        key_camera_tokens = [sample["key_camera_token"] for sample in samples]
        frame_ids = set()

        data_dict = self.nu_samples_data
        for key_camera_token in key_camera_tokens:
            data = data_dict[key_camera_token]
            calib_sensor_token = data["calibrated_sensor_token"]
            calib_sensor = self.nu_calibrated_sensors[calib_sensor_token]
            sensor = self.get_nu_sensor(sensor_token=calib_sensor["sensor_token"])
            if sensor["channel"] == self.sensor_name:
                frame_ids.add(str(data["timestamp"]))
        return frame_ids
