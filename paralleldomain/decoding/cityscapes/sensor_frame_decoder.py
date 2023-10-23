from typing import Any, Dict, List, Optional, Tuple, TypeVar

import numpy as np

from paralleldomain.decoding.cityscapes.common import decode_class_maps, get_scene_labels_path, get_scene_path
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.model.annotation import (
    AnnotationIdentifier,
    AnnotationTypes,
    InstanceSegmentation2D,
    SemanticSegmentation2D,
)
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.image import Image
from paralleldomain.model.sensor import SensorDataCopyTypes, SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image

T = TypeVar("T")


class CityscapesCameraSensorFrameDecoder(CameraSensorFrameDecoder[None]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        frame_id: FrameId,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        is_unordered_scene: bool,
        scene_decoder,
    ):
        super().__init__(
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            frame_id=frame_id,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        self.dataset_path = dataset_path

    def _decode_intrinsic(self) -> SensorIntrinsic:
        return SensorIntrinsic(fx=2262.52, fy=2265.3017905988554, cx=1096.98, cy=513.137)

    def _decode_image_dimensions(self) -> Tuple[int, int, int]:
        return 1024, 2048, 3

    def _decode_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        return decode_class_maps()

    def _decode_image_rgba(self) -> np.ndarray:
        scene_images_folder = get_scene_path(
            dataset_path=self.dataset_path, scene_name=self.scene_name, camera_name=self.sensor_name
        )
        img_path = scene_images_folder / self.frame_id
        image_data = read_image(path=img_path, convert_to_rgb=True)

        ones = np.ones((*image_data.shape[:2], 1), dtype=image_data.dtype)
        concatenated = np.concatenate([image_data, ones], axis=-1)
        return concatenated

    def _decode_file_path(self, data_type: SensorDataCopyTypes) -> Optional[AnyPath]:
        if isinstance(data_type, AnnotationIdentifier) and data_type.annotation_type in [
            SemanticSegmentation2D,
            InstanceSegmentation2D,
        ]:
            img_suffix = "_leftImg8bit"
            if data_type.annotation_type is SemanticSegmentation2D:
                relative_path = self.frame_id.replace(img_suffix, "_gtFine_labelIds")
            elif data_type.annotation_type is InstanceSegmentation2D:
                relative_path = self.frame_id.replace(img_suffix, "_gtFine_instanceIds")
            scene_labels_folder = get_scene_labels_path(dataset_path=self.dataset_path, scene_name=self.scene_name)
            return scene_labels_folder / relative_path
        elif issubclass(data_type, Image):
            scene_images_folder = get_scene_path(
                dataset_path=self.dataset_path, scene_name=self.scene_name, camera_name=self.sensor_name
            )
            img_path = scene_images_folder / self.frame_id
            return img_path
        return None

    def _decode_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        return [
            AnnotationIdentifier(annotation_type=AnnotationTypes.SemanticSegmentation2D),
            AnnotationIdentifier(annotation_type=AnnotationTypes.InstanceSegmentation2D),
        ]

    def _decode_metadata(self) -> Dict[str, Any]:
        return {}

    def _decode_date_time(self) -> None:
        return None

    def _decode_extrinsic(self) -> SensorExtrinsic:
        return SensorExtrinsic.from_transformation_matrix(np.eye(4))

    def _decode_sensor_pose(self) -> SensorPose:
        return SensorPose.from_transformation_matrix(np.eye(4))

    def _decode_annotations(self, identifier: AnnotationIdentifier[T]) -> T:
        file_path = self.get_file_path(data_type=identifier)
        if identifier.annotation_type is SemanticSegmentation2D:
            class_ids = self._decode_semantic_segmentation_2d(file_path=file_path)
            return SemanticSegmentation2D(class_ids=class_ids)
        if identifier.annotation_type is InstanceSegmentation2D:
            instance_ids = self._decode_instance_segmentation_2d(file_path=file_path)
            return InstanceSegmentation2D(instance_ids=instance_ids)
        else:
            raise NotImplementedError(f"{identifier.annotation_type} is not supported!")

    def _decode_semantic_segmentation_2d(self, file_path: AnyPath) -> np.ndarray:
        class_ids = read_image(path=file_path, convert_to_rgb=False)
        class_ids = class_ids.astype(int)
        return np.expand_dims(class_ids, axis=-1)

    def _decode_instance_segmentation_2d(self, file_path: AnyPath) -> np.ndarray:
        instance_ids = read_image(path=file_path, convert_to_rgb=False)
        instance_ids = instance_ids.astype(int)
        return np.expand_dims(instance_ids, axis=-1)
