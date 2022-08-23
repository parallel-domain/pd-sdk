from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import imagesize
import numpy as np

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.kitti_flow.common import frame_id_to_timestamp
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder, F
from paralleldomain.model.annotation import AnnotationType, AnnotationTypes, OpticalFlow
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.image import Image
from paralleldomain.model.sensor import SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import AnnotationIdentifier, FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image, read_json

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
        if self._metadata_folder is None:
            return dict()
        metadata_path = self._dataset_path / self._metadata_folder / f"{AnyPath(frame_id).stem + '.json'}"
        return read_json(metadata_path)

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
            flow_vectors, valid_mask = self._decode_optical_flow(
                scene_name=self.scene_name, annotation_identifier=identifier, frame_id=frame_id
            )
            return OpticalFlow(vectors=flow_vectors, valid_mask=valid_mask)
        else:
            raise NotImplementedError(f"{annotation_type} is not supported!")

    def _decode_optical_flow(self, scene_name: str, frame_id: FrameId, annotation_identifier: str) -> np.ndarray:
        if frame_id[-7:] == "_11.png":
            return None
        if self._use_non_occluded:
            annotation_path = self._dataset_path / self._noc_optical_flow_folder / f"{frame_id}"
        else:
            annotation_path = self._dataset_path / self._occ_optical_flow_folder / f"{frame_id}"
        image_data = read_image(path=annotation_path, convert_to_rgb=True, is_indexed=False)
        vectors = (image_data[:, :, :2] - 2**15) / 64.0
        valid_mask = image_data[:, :, -1].astype(np.float32)

        return vectors, valid_mask

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
