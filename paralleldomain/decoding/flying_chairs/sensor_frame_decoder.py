from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import imagesize
import numpy as np

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.flying_chairs.common import frame_id_to_timestamp
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder, F
from paralleldomain.model.annotation import AnnotationType, AnnotationTypes, OpticalFlow
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.image import Image
from paralleldomain.model.sensor import SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import AnnotationIdentifier, FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image, read_json

T = TypeVar("T")


class FlyingChairsCameraSensorFrameDecoder(CameraSensorFrameDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        image_folder: str,
        optical_flow_folder: str,
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self._dataset_path = dataset_path
        self._image_folder = image_folder
        self._optical_flow_folder = optical_flow_folder

    def _decode_intrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorIntrinsic:
        return SensorIntrinsic()

    def _decode_image_dimensions(self, sensor_name: SensorName, frame_id: FrameId) -> Tuple[int, int, int]:
        img_path = self._dataset_path / self._image_folder / f"{frame_id}"
        with img_path.open("rb") as fh:
            width, height = imagesize.get(BytesIO(fh.read()))
            return height, width, 3

    def _decode_image_rgba(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        img_path = self._dataset_path / self._image_folder / f"{frame_id}"
        image_data = read_image(path=img_path, convert_to_rgb=True)

        ones = np.ones((*image_data.shape[:2], 1), dtype=image_data.dtype)
        concatenated = np.concatenate([image_data, ones], axis=-1)
        return concatenated

    def _decode_class_maps(self) -> Dict[AnnotationType, ClassMap]:
        return dict()

    def _decode_available_annotation_types(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> Dict[AnnotationType, AnnotationIdentifier]:
        optical_flow_file_name = f"{frame_id}"

        return {
            AnnotationTypes.OpticalFlow: optical_flow_file_name,
        }

    def _decode_metadata(self, sensor_name: SensorName, frame_id: FrameId) -> Dict[str, Any]:
        return dict()

    def _decode_date_time(self, sensor_name: SensorName, frame_id: FrameId) -> datetime:
        return frame_id_to_timestamp(frame_id=frame_id)

    def _decode_extrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorExtrinsic:
        return SensorExtrinsic.from_transformation_matrix(np.eye(4))

    def _decode_sensor_pose(self, sensor_name: SensorName, frame_id: FrameId) -> SensorPose:
        return SensorPose.from_transformation_matrix(np.eye(4))

    def _decode_annotations(
        self, sensor_name: SensorName, frame_id: FrameId, identifier: AnnotationIdentifier, annotation_type: T
    ) -> T:
        if issubclass(annotation_type, OpticalFlow):
            flow_vectors = self._decode_optical_flow(
                scene_name=self.scene_name, annotation_identifier=identifier, frame_id=frame_id
            )
            return OpticalFlow(vectors=flow_vectors)
        else:
            raise NotImplementedError(f"{annotation_type} is not supported!")

    def _decode_optical_flow(self, scene_name: str, frame_id: FrameId, annotation_identifier: str) -> np.ndarray:
        """
        Reads optical flow files in the .flo format. Notably used for Flying Chairs:
        https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs"""

        if frame_id[9] == "2":
            return None
        flow_filename = frame_id[:5] + "_flow.flo"
        annotation_path = self._dataset_path / self._optical_flow_folder / f"{flow_filename}"
        with annotation_path.open(mode="rb") as fp:
            header = fp.read(4)
            if header.decode("utf-8") != "PIEH":
                raise Exception("Flow file header does not contain PIEH")
            hw_raw = np.frombuffer(fp.read(8), dtype=np.int32)
            width = hw_raw[0]
            height = hw_raw[1]
            raw = np.frombuffer(fp.read(), dtype=np.float32)
            flow = raw[: (width * height) * 2].reshape((height, width, 2))
        return flow

    def _decode_file_path(self, sensor_name: SensorName, frame_id: FrameId, data_type: Type[F]) -> Optional[AnyPath]:
        annotation_identifiers = self.get_available_annotation_types(sensor_name=sensor_name, frame_id=frame_id)
        if issubclass(data_type, OpticalFlow):
            if data_type in annotation_identifiers:
                annotation_path = self._dataset_path / self._optical_flow_folder / f"{frame_id}"
                return annotation_path
        elif issubclass(data_type, Image):
            img_path = self._dataset_path / self._image_folder / f"{frame_id}"
            return img_path
        return None