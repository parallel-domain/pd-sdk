from datetime import datetime
from typing import Any, Dict, Tuple, TypeVar, Union

import numpy as np

from paralleldomain.decoding.nuimages.common import NuImagesDataAccessMixin
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder, TDateTime
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.sensor import SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import AnnotationIdentifier, FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath

T = TypeVar("T")


class NuImagesCameraSensorFrameDecoder(CameraSensorFrameDecoder[datetime], NuImagesDataAccessMixin):
    def __init__(self, dataset_path: Union[str, AnyPath], dataset_name: str, split_name: str, scene_name: SceneName):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        CameraSensorFrameDecoder.__init__(self=self, dataset_name=dataset_name, scene_name=scene_name)
        NuImagesDataAccessMixin.__init__(
            self=self, dataset_name=dataset_name, split_name=split_name, dataset_path=self._dataset_path
        )

    def _get_nu_calibrated_sensor(self, sensor_name: SensorName, frame_id: FrameId) -> Dict[str, Any]:
        samples = self.get_nu_samples(log_token=self.scene_name)

        sample_tokens = [sample["token"] for sample in samples if str(sample["timestamp"]) == frame_id]
        calib_sensors = {calib_sensor["token"]: calib_sensor for calib_sensor in self.nu_calibrated_sensors}
        for data in self.nu_samples_data:
            if data["sample_token"] in sample_tokens:
                calib_sensor_token = data["calibrated_sensor_token"]
                calib_sensor = calib_sensors[calib_sensor_token]

                sensor = self.get_nu_sensor(sensor_token=calib_sensor["sensor_token"])
                if sensor["channel"] == sensor_name:
                    return calib_sensor
        return dict()

    def _decode_intrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorIntrinsic:
        # calibrated_sensor = self._get_nu_calibrated_sensor(sensor_name=sensor_name, frame_id=frame_id)
        return SensorIntrinsic()

    def _decode_image_dimensions(self, sensor_name: SensorName, frame_id: FrameId) -> Tuple[int, int, int]:
        pass

    def _decode_image_rgba(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        pass

    def _decode_available_annotation_types(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> Dict[AnnotationType, AnnotationIdentifier]:
        pass

    def _decode_date_time(self, sensor_name: SensorName, frame_id: FrameId) -> TDateTime:
        pass

    def _decode_extrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorExtrinsic:
        pass

    def _decode_sensor_pose(self, sensor_name: SensorName, frame_id: FrameId) -> SensorPose:
        pass

    def _decode_annotations(
        self, sensor_name: SensorName, frame_id: FrameId, identifier: AnnotationIdentifier, annotation_type: T
    ) -> T:
        pass
