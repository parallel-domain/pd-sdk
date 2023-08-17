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
from paralleldomain.model.sensor import SensorExtrinsic, SensorIntrinsic, SensorPose, SensorDataCopyTypes
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image, read_json

T = TypeVar("T")


class DirectoryBaseSensorFrameDecoder(SensorFrameDecoder[None]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        folder_to_data_type: Dict[str, SensorDataCopyTypes],
        class_map: List[ClassDetail],
        metadata_folder: Optional[str],
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self._dataset_path = dataset_path
        self._data_type_to_folder_name = {v: k for k, v in folder_to_data_type.items()}
        self._class_map = class_map
        self._metadata_folder = metadata_folder
        self._scene_path = resolve_scene_folder(dataset_path=self._dataset_path, scene_name=self.scene_name)

    def _decode_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        annotation_types = [
            annotation_type
            for annotation_type in self._data_type_to_folder_name.keys()
            if isinstance(annotation_type, AnnotationIdentifier)
        ]
        return decode_class_maps(class_map=self._class_map, annotation_types=annotation_types)

    def _decode_available_annotation_identifiers(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> List[AnnotationIdentifier]:
        return [
            annotation_type
            for annotation_type in self._data_type_to_folder_name.keys()
            if isinstance(annotation_type, AnnotationIdentifier)
        ]
        # return [
        #     AnnotationIdentifier(annotation_type=AnnotationTypes.SemanticSegmentation2D),
        #     AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes3D),
        # ]

    def _decode_metadata(self, sensor_name: SensorName, frame_id: FrameId) -> Dict[str, Any]:
        if self._metadata_folder is None:
            return dict()
        metadata_path = self._scene_path / self._metadata_folder / f"{AnyPath(frame_id).stem + '.json'}"
        return read_json(metadata_path)

    def _decode_date_time(self, sensor_name: SensorName, frame_id: FrameId) -> None:
        return None

    def _decode_extrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorExtrinsic:
        return SensorExtrinsic.from_transformation_matrix(np.eye(4))

    def _decode_sensor_pose(self, sensor_name: SensorName, frame_id: FrameId) -> SensorPose:
        return SensorPose.from_transformation_matrix(np.eye(4))

    def _decode_file_path(
        self, sensor_name: SensorName, frame_id: FrameId, data_type: SensorDataCopyTypes
    ) -> Optional[AnyPath]:
        if isinstance(data_type, AnnotationIdentifier) and data_type.annotation_type is SemanticSegmentation2D:
            annotation_path = (
                self._dataset_path / self._data_type_to_folder_name[SemanticSegmentation2D] / f"{frame_id}"
            )
            return annotation_path
        elif issubclass(data_type, Image):
            img_path = self._dataset_path / self._image_folder / f"{frame_id}"
            return img_path
        return None


class DirectoryCameraSensorFrameDecoder(DirectoryBaseSensorFrameDecoder, CameraSensorFrameDecoder[None]):
    def _decode_annotations(self, sensor_name: SensorName, frame_id: FrameId, identifier: AnnotationIdentifier[T]) -> T:
        if identifier.annotation_type is SemanticSegmentation2D:
            class_ids = self._decode_semantic_segmentation_2d(scene_name=self.scene_name, frame_id=frame_id)
            return SemanticSegmentation2D(class_ids=class_ids)
        else:
            raise NotImplementedError(f"{identifier.annotation_type} is not supported!")

    def _decode_semantic_segmentation_2d(self, scene_name: str, frame_id: FrameId) -> np.ndarray:
        annotation_path = self._dataset_path / self._data_type_to_folder_name[SemanticSegmentation2D] / f"{frame_id}"
        class_ids = read_image(path=annotation_path, convert_to_rgb=False)
        class_ids = class_ids.astype(int)
        return np.expand_dims(class_ids, axis=-1)

    def _decode_intrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorIntrinsic:
        return SensorIntrinsic()

    def _decode_image_dimensions(self, sensor_name: SensorName, frame_id: FrameId) -> Tuple[int, int, int]:
        img_path = self._scene_path / self._data_type_to_folder_name[Image] / f"{frame_id}.png"
        with img_path.open("rb") as fh:
            width, height = imagesize.get(BytesIO(fh.read()))
            return height, width, 3

    def _decode_image_rgba(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        scene_images_folder = self._scene_path / self._data_type_to_folder_name[Image]
        img_path = scene_images_folder / f"{frame_id}.png"
        image_data = read_image(path=img_path, convert_to_rgb=True)

        ones = np.ones((*image_data.shape[:2], 1), dtype=image_data.dtype)
        concatenated = np.concatenate([image_data, ones], axis=-1)
        return concatenated
