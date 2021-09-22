from typing import Dict, List, Tuple, TypeVar

import cv2
import numpy as np

from paralleldomain.decoding.cityscapes.common import get_scene_labels_path, get_scene_path
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.model.annotation import (
    AnnotationType,
    AnnotationTypes,
    InstanceSegmentation2D,
    SemanticSegmentation2D,
)
from paralleldomain.model.sensor import SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import AnnotationIdentifier, FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath

T = TypeVar("T")


class CityscapesCameraSensorFrameDecoder(CameraSensorFrameDecoder[None]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name)
        self._dataset_path = dataset_path

    def _decode_intrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorIntrinsic:
        return SensorIntrinsic(fx=2262.52, fy=2265.3017905988554, cx=1096.98, cy=513.137)

    def _decode_image_dimensions(self, sensor_name: SensorName, frame_id: FrameId) -> Tuple[int, int, int]:
        return 1024, 2048, 3

    def _decode_image_rgba(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        scene_images_folder = get_scene_path(
            dataset_path=self._dataset_path, scene_name=self.scene_name, camera_name=sensor_name
        )
        img_path = scene_images_folder / frame_id
        image_data = read_png(path=img_path)[..., ::-1]

        ones = np.ones((*image_data.shape[:2], 1), dtype=image_data.dtype)
        concatenated = np.concatenate([image_data, ones], axis=-1)
        return concatenated

    def _decode_available_annotation_types(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> Dict[AnnotationType, AnnotationIdentifier]:

        img_suffix = "_leftImg8bit"
        seg_map_suffix = "_gtFine_labelIds"
        # seg_map_suffix = "_gtFine_labelTrainIds"
        inst_map_suffix = "_gtFine_instanceIds"
        semseg_file_name = frame_id.replace(img_suffix, seg_map_suffix)
        instseg_file_name = frame_id.replace(img_suffix, inst_map_suffix)

        return {
            AnnotationTypes.SemanticSegmentation2D: semseg_file_name,
            AnnotationTypes.InstanceSegmentation2D: instseg_file_name,
        }

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
                scene_name=self.scene_name, annotation_identifier=identifier
            )
            return SemanticSegmentation2D(class_ids=class_ids)
        if issubclass(annotation_type, InstanceSegmentation2D):
            instance_ids = self._decode_instance_segmentation_2d(
                scene_name=self.scene_name, annotation_identifier=identifier
            )
            return InstanceSegmentation2D(instance_ids=instance_ids)
        else:
            raise NotImplementedError(f"{annotation_type} is not supported!")

    def _decode_semantic_segmentation_2d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        scene_labels_folder = get_scene_labels_path(dataset_path=self._dataset_path, scene_name=scene_name)
        annotation_path = scene_labels_folder / annotation_identifier
        class_ids = read_png(path=annotation_path)
        class_ids = class_ids.astype(int)
        return np.expand_dims(class_ids, axis=-1)

    def _decode_instance_segmentation_2d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        scene_labels_folder = get_scene_labels_path(dataset_path=self._dataset_path, scene_name=scene_name)
        annotation_path = scene_labels_folder / annotation_identifier
        instance_ids = read_png(path=annotation_path)
        instance_ids = instance_ids.astype(int)
        return np.expand_dims(instance_ids, axis=-1)


def read_png(path: AnyPath) -> np.ndarray:
    with path.open(mode="rb") as fp:
        image_data = cv2.imdecode(
            buf=np.frombuffer(fp.read(), np.uint8),
            flags=cv2.IMREAD_UNCHANGED,
        )
    return image_data
