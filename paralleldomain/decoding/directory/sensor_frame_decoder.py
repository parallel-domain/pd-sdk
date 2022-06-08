from typing import Any, Dict, List, Tuple, TypeVar

import numpy as np
import imagesize

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.model.annotation import (
    AnnotationType,
    AnnotationTypes,
    SemanticSegmentation2D,
)
from paralleldomain.model.sensor import SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import AnnotationIdentifier, FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image, read_json

T = TypeVar("T")


class DirectoryCameraSensorFrameDecoder(CameraSensorFrameDecoder[None]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        image_folder: str,
        semantic_segmentation_folder: str,
        metadata_folder: str,
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self._dataset_path = dataset_path
        self._image_folder = image_folder
        self._semantic_segmentation_folder = semantic_segmentation_folder
        self._metadata_folder = metadata_folder

    def _decode_intrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorIntrinsic:
        return None

    def _decode_image_dimensions(self, sensor_name: SensorName, frame_id: FrameId) -> Tuple[int, int, int]:
        width, height = imagesize.get(f"{frame_id}")
        return width, height, 3

    def _decode_image_rgba(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        scene_images_folder = self._dataset_path / self._image_folder
        img_path = scene_images_folder / f"{frame_id}"
        image_data = read_image(path=img_path, convert_to_rgb=True)

        ones = np.ones((*image_data.shape[:2], 1), dtype=image_data.dtype)
        concatenated = np.concatenate([image_data, ones], axis=-1)
        return concatenated

    def _decode_available_annotation_types(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> Dict[AnnotationType, AnnotationIdentifier]:
        semseg_file_name = f"{frame_id}"

        return {
            AnnotationTypes.SemanticSegmentation2D: semseg_file_name,
        }

    def _decode_metadata(self, sensor_name: SensorName, frame_id: FrameId) -> Dict[str, Any]:
        metadata_path = self._dataset_path / self._metadata_folder / f"{AnyPath(frame_id).stem + '.json'}"
        return read_json(metadata_path)

    def _decode_date_time(self, sensor_name: SensorName, frame_id: FrameId) -> None:
        return None

    def _decode_extrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorExtrinsic:
        return SensorExtrinsic.from_transformation_matrix(np.eye(4))

    def _decode_sensor_pose(self, sensor_name: SensorName, frame_id: FrameId) -> SensorPose:
        return SensorPose.from_transformation_matrix(np.eye(4))

    def _decode_annotations(
        self, sensor_name: SensorName, frame_id: FrameId, identifier: AnnotationIdentifier, annotation_type: T
    ) -> T:
        if issubclass(annotation_type, SemanticSegmentation2D):
            class_ids = self._decode_semantic_segmentation_2d(
                scene_name=self.scene_name, annotation_identifier=identifier, frame_id=frame_id
            )
            return SemanticSegmentation2D(class_ids=class_ids)
        else:
            raise NotImplementedError(f"{annotation_type} is not supported!")

    def _decode_semantic_segmentation_2d(
        self, scene_name: str, frame_id: FrameId, annotation_identifier: str
    ) -> np.ndarray:
        annotation_path = self._dataset_path / self._semantic_segmentation_folder / f"{frame_id}"
        class_ids = read_image(path=annotation_path, convert_to_rgb=False)
        class_ids = class_ids.astype(int)
        return np.expand_dims(class_ids, axis=-1)
