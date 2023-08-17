from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import imagesize
import numpy as np

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.kitti_flow.common import frame_id_to_timestamp
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.model.annotation import AnnotationTypes, OpticalFlow, AnnotationIdentifier
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.image import Image
from paralleldomain.model.sensor import SensorExtrinsic, SensorIntrinsic, SensorPose, SensorDataCopyTypes
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image

T = TypeVar("T")


class KITTIFlowCameraSensorFrameDecoder(CameraSensorFrameDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        image_folder: str,
        occ_optical_flow_folder: str,
        noc_optical_flow_folder: str,
        use_non_occluded: bool,
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self._dataset_path = dataset_path
        self._image_folder = image_folder
        self._occ_optical_flow_folder = occ_optical_flow_folder
        self._noc_optical_flow_folder = noc_optical_flow_folder
        self._use_non_occluded = use_non_occluded

    def _decode_intrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorIntrinsic:
        return SensorIntrinsic(fx=721.5377, fy=721.5377, cx=609.5593, cy=172.854)

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

    def _decode_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        return dict()

    def _decode_available_annotation_identifiers(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> List[AnnotationIdentifier]:
        if frame_id[-6:] == "10.png":
            return [AnnotationIdentifier(annotation_type=AnnotationTypes.OpticalFlow)]
        else:
            return list()

    def _decode_metadata(self, sensor_name: SensorName, frame_id: FrameId) -> Dict[str, Any]:
        return dict()

    def _decode_date_time(self, sensor_name: SensorName, frame_id: FrameId) -> datetime:
        return frame_id_to_timestamp(frame_id=frame_id)

    def _decode_extrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorExtrinsic:
        return SensorExtrinsic.from_transformation_matrix(np.eye(4))

    def _decode_sensor_pose(self, sensor_name: SensorName, frame_id: FrameId) -> SensorPose:
        return SensorPose.from_transformation_matrix(np.eye(4))

    def _decode_annotations(self, sensor_name: SensorName, frame_id: FrameId, identifier: AnnotationIdentifier[T]) -> T:
        if issubclass(identifier.annotation_type, OpticalFlow):
            flow_vectors, valid_mask = self._decode_optical_flow(frame_id=frame_id)
            return OpticalFlow(vectors=flow_vectors, valid_mask=valid_mask)
        else:
            raise NotImplementedError(f"{identifier} is not supported!")

    def _decode_optical_flow(
        self,
        frame_id: FrameId,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._use_non_occluded:
            annotation_path = self._dataset_path / self._noc_optical_flow_folder / f"{frame_id}"
        else:
            annotation_path = self._dataset_path / self._occ_optical_flow_folder / f"{frame_id}"

        image_data = read_image(path=annotation_path, convert_to_rgb=True, is_indexed=False).astype(np.float32)
        vectors = (image_data[:, :, :2] - 2**15) / 64.0
        valid_mask = image_data[:, :, -1]

        return vectors, valid_mask

    def _decode_file_path(
        self, sensor_name: SensorName, frame_id: FrameId, data_type: SensorDataCopyTypes
    ) -> Optional[AnyPath]:
        annotation_identifiers = self.get_available_annotation_identifiers(sensor_name=sensor_name, frame_id=frame_id)
        if isinstance(data_type, AnnotationIdentifier) and issubclass(data_type.annotation_type, OpticalFlow):
            if data_type in annotation_identifiers:
                if self._use_non_occluded:
                    annotation_path = self._dataset_path / self._noc_optical_flow_folder / f"{frame_id}"
                else:
                    annotation_path = self._dataset_path / self._occ_optical_flow_folder / f"{frame_id}"
                return annotation_path
        elif issubclass(data_type, Image):
            img_path = self._dataset_path / self._image_folder / f"{frame_id}"
            return img_path
        return None
