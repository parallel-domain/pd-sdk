import abc
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import imagesize
import numpy as np

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.directory.common import decode_class_maps, resolve_scene_folder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder, SensorFrameDecoder
from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationTypes, SemanticSegmentation2D
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.image import Image
from paralleldomain.model.sensor import SensorDataCopyTypes, SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image, read_json

T = TypeVar("T")


class DirectoryBaseSensorFrameDecoder(SensorFrameDecoder[None], metaclass=abc.ABCMeta):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        frame_id: FrameId,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        folder_to_data_type: Dict[str, SensorDataCopyTypes],
        class_map: List[ClassDetail],
        metadata_folder: Optional[str],
        is_unordered_scene: bool,
        scene_decoder,
        img_file_extension: Optional[str] = "png",
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
        self._dataset_path = dataset_path
        self._data_type_to_folder_name = {v: k for k, v in folder_to_data_type.items()}
        self._class_map = class_map
        self._metadata_folder = metadata_folder
        self._scene_path = resolve_scene_folder(dataset_path=self._dataset_path, scene_name=self.scene_name)
        self._img_file_extension = img_file_extension

    def _decode_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        annotation_types = [
            annotation_type
            for annotation_type in self._data_type_to_folder_name.keys()
            if isinstance(annotation_type, AnnotationIdentifier)
        ]
        return decode_class_maps(class_map=self._class_map, annotation_types=annotation_types)

    def _decode_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        return [
            annotation_type
            for annotation_type in self._data_type_to_folder_name.keys()
            if isinstance(annotation_type, AnnotationIdentifier)
        ]

    def _decode_metadata(self) -> Dict[str, Any]:
        if self._metadata_folder is None:
            return dict()
        metadata_path = self._scene_path / self._metadata_folder / f"{AnyPath(self.frame_id).stem + '.json'}"
        return read_json(metadata_path)

    def _decode_date_time(self) -> None:
        return None

    def _decode_extrinsic(self) -> SensorExtrinsic:
        return SensorExtrinsic.from_transformation_matrix(np.eye(4))

    def _decode_sensor_pose(self) -> SensorPose:
        return SensorPose.from_transformation_matrix(np.eye(4))

    def _decode_file_path(self, data_type: SensorDataCopyTypes) -> Optional[AnyPath]:
        if isinstance(data_type, AnnotationIdentifier) and data_type.annotation_type is SemanticSegmentation2D:
            annotation_path = (
                self._dataset_path
                / self._data_type_to_folder_name[SemanticSegmentation2D]
                / f"{self.frame_id}.{self._img_file_extension}"
            )
            return annotation_path

        elif data_type is SemanticSegmentation2D:
            # Note: We also support Type[Annotation] for data_type for backwards compatibility
            annotation_path = (
                self._dataset_path
                / self._data_type_to_folder_name[SemanticSegmentation2D]
                / f"{self.frame_id}.{self._img_file_extension}"
            )
            return annotation_path

        elif issubclass(data_type, Image):
            img_path = (
                self._dataset_path
                / self._data_type_to_folder_name[Image]
                / f"{self.frame_id}.{self._img_file_extension}"
            )
            return img_path
        return None


class DirectoryCameraSensorFrameDecoder(DirectoryBaseSensorFrameDecoder, CameraSensorFrameDecoder[None]):
    def _decode_annotations(self, identifier: AnnotationIdentifier[T]) -> T:
        if identifier.annotation_type is SemanticSegmentation2D:
            class_ids = self._decode_semantic_segmentation_2d()
            return SemanticSegmentation2D(class_ids=class_ids)
        else:
            raise NotImplementedError(f"{identifier.annotation_type} is not supported!")

    def _decode_semantic_segmentation_2d(self) -> np.ndarray:
        annotation_path = (
            self._dataset_path
            / self._data_type_to_folder_name[SemanticSegmentation2D]
            / f"{self.frame_id}.{self._img_file_extension}"
        )
        class_ids = read_image(path=annotation_path, convert_to_rgb=False)
        class_ids = class_ids.astype(int)
        return np.expand_dims(class_ids, axis=-1)

    def _decode_intrinsic(self) -> SensorIntrinsic:
        return SensorIntrinsic()

    def _decode_image_dimensions(self) -> Tuple[int, int, int]:
        img_path = (
            self._scene_path / self._data_type_to_folder_name[Image] / f"{self.frame_id}.{self._img_file_extension}"
        )
        with img_path.open("rb") as fh:
            width, height = imagesize.get(BytesIO(fh.read()))
            return height, width, 3

    def _decode_image_rgba(self) -> np.ndarray:
        scene_images_folder = self._scene_path / self._data_type_to_folder_name[Image]
        img_path = scene_images_folder / f"{self.frame_id}.{self._img_file_extension}"
        image_data = read_image(path=img_path, convert_to_rgb=True)

        ones = np.ones((*image_data.shape[:2], 1), dtype=image_data.dtype)
        concatenated = np.concatenate([image_data, ones], axis=-1)
        return concatenated
