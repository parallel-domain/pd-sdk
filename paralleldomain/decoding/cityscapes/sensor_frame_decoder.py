from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import cv2
import numpy as np

from paralleldomain.decoding.cityscapes.common import decode_class_maps, get_scene_labels_path, get_scene_path
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.model.annotation import (
    Annotation,
    AnnotationType,
    AnnotationTypes,
    InstanceSegmentation2D,
    SemanticSegmentation2D,
)
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.image import Image
from paralleldomain.model.point_cloud import PointCloud
from paralleldomain.model.sensor import SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import AnnotationIdentifier, FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image

T = TypeVar("T")
F = TypeVar("F", Image, PointCloud, Annotation)


class CityscapesCameraSensorFrameDecoder(CameraSensorFrameDecoder[None]):
    def __init__(self, dataset_name: str, scene_name: SceneName, dataset_path: AnyPath, settings: DecoderSettings):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self._dataset_path = dataset_path

    def _decode_intrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorIntrinsic:
        return SensorIntrinsic(fx=2262.52, fy=2265.3017905988554, cx=1096.98, cy=513.137)

    def _decode_image_dimensions(self, sensor_name: SensorName, frame_id: FrameId) -> Tuple[int, int, int]:
        return 1024, 2048, 3

    def _decode_class_maps(self) -> Dict[AnnotationType, ClassMap]:
        return decode_class_maps()

    def _decode_image_rgba(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        scene_images_folder = get_scene_path(
            dataset_path=self._dataset_path, scene_name=self.scene_name, camera_name=sensor_name
        )
        img_path = scene_images_folder / frame_id
        image_data = read_image(path=img_path, convert_to_rgb=True)

        ones = np.ones((*image_data.shape[:2], 1), dtype=image_data.dtype)
        concatenated = np.concatenate([image_data, ones], axis=-1)
        return concatenated

    def _decode_file_path(self, sensor_name: SensorName, frame_id: FrameId, data_type: Type[F]) -> Optional[AnyPath]:
        annotation_identifiers = self.get_available_annotation_types(sensor_name=sensor_name, frame_id=frame_id)
        if issubclass(data_type, SemanticSegmentation2D) or issubclass(data_type, InstanceSegmentation2D):
            if data_type in annotation_identifiers:
                annotation_identifier = annotation_identifiers[data_type]
                scene_labels_folder = get_scene_labels_path(dataset_path=self._dataset_path, scene_name=self.scene_name)
                return scene_labels_folder / annotation_identifier
        elif issubclass(data_type, Image):
            scene_images_folder = get_scene_path(
                dataset_path=self._dataset_path, scene_name=self.scene_name, camera_name=sensor_name
            )
            img_path = scene_images_folder / frame_id
            return img_path
        return None

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

    def _decode_metadata(self, sensor_name: SensorName, frame_id: FrameId) -> Dict[str, Any]:
        return {}

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
        class_ids = read_image(path=annotation_path, convert_to_rgb=False)
        class_ids = class_ids.astype(int)
        return np.expand_dims(class_ids, axis=-1)

    def _decode_instance_segmentation_2d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        scene_labels_folder = get_scene_labels_path(dataset_path=self._dataset_path, scene_name=scene_name)
        annotation_path = scene_labels_folder / annotation_identifier
        instance_ids = read_image(path=annotation_path, convert_to_rgb=False)
        instance_ids = instance_ids.astype(int)
        return np.expand_dims(instance_ids, axis=-1)
