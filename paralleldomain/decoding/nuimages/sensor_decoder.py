from datetime import datetime
from typing import Set, Union

from paralleldomain.decoding.nuimages.common import NuImagesDataAccessMixin
from paralleldomain.decoding.nuimages.sensor_frame_decoder import NuImagesCameraSensorFrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class NuImagesCameraSensorDecoder(CameraSensorDecoder[datetime], NuImagesDataAccessMixin):
    def __init__(self, dataset_path: Union[str, AnyPath], dataset_name: str, split_name: str, scene_name: SceneName):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        CameraSensorDecoder.__init__(self=self, dataset_name=dataset_name, scene_name=scene_name)
        NuImagesDataAccessMixin.__init__(
            self=self, dataset_name=dataset_name, split_name=split_name, dataset_path=self._dataset_path
        )

    def _decode_camera_sensor_frame(
        self, decoder: CameraSensorFrameDecoder[datetime], frame_id: FrameId, camera_name: SensorName
    ) -> CameraSensorFrame[datetime]:
        return CameraSensorFrame[datetime](sensor_name=camera_name, frame_id=frame_id, decoder=decoder)

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[datetime]:
        return NuImagesCameraSensorFrameDecoder(
            dataset_path=self._dataset_path,
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            split_name=self.split_name,
        )

    def _decode_frame_id_set(self, sensor_name: SensorName) -> Set[FrameId]:
        samples = self.get_nu_samples(log_token=self.scene_name)

        sample_tokens = [sample["token"] for sample in samples]
        frame_ids = set()
        calib_sensors = {calib_sensor["token"]: calib_sensor for calib_sensor in self.nu_calibrated_sensors}
        for data in self.nu_samples_data:
            if data["sample_token"] in sample_tokens:
                calib_sensor_token = data["calibrated_sensor_token"]
                calib_sensor = calib_sensors[calib_sensor_token]
                sensor = self.get_nu_sensor(sensor_token=calib_sensor["sensor_token"])
                if sensor["channel"] == sensor_name:
                    frame_ids.add(str(data["timestamp"]))
        return frame_ids
