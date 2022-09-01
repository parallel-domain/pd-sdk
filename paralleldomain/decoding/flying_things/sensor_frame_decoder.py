from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import imagesize
import numpy as np

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.flying_chairs.common import frame_id_to_timestamp
from paralleldomain.decoding.flying_things.common import (
    OPTICAL_FLOW_BACKWARD_DIRECTION_NAME,
    OPTICAL_FLOW_FORWARD_DIRECTION_NAME,
    get_image_folder_name,
    get_scene_flow_folder,
    get_scene_folder,
    read_flow,
)
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder, F
from paralleldomain.model.annotation import AnnotationType, AnnotationTypes, OpticalFlow
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.image import Image
from paralleldomain.model.sensor import SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import AnnotationIdentifier, FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image, read_json

T = TypeVar("T")


class FlyingThingsCameraSensorFrameDecoder(CameraSensorFrameDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        split_name: str,
        split_list: List[int],
        is_full_dataset_format: bool = False,
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self._is_full_dataset_format = is_full_dataset_format
        self._split_list = split_list
        self._dataset_path = dataset_path
        self._split_name = split_name

    def _decode_intrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorIntrinsic:
        return SensorIntrinsic()

    def _get_image_file_path(self, sensor_name: SensorName, frame_id: FrameId) -> AnyPath:
        base_folder = get_scene_folder(
            dataset_path=self._dataset_path, scene_name=self.scene_name, split_name=self._split_name
        )
        if not self._is_full_dataset_format:
            base_folder = base_folder.parent

        return base_folder / get_image_folder_name(scene_name=self.scene_name) / sensor_name / f"{frame_id}.png"

    def _get_flow_file_path(self, sensor_name: SensorName, frame_id: FrameId, forward: bool) -> AnyPath:
        direction = OPTICAL_FLOW_FORWARD_DIRECTION_NAME if forward else OPTICAL_FLOW_BACKWARD_DIRECTION_NAME

        # if not forward:
        #     frame_id = str(int(frame_id) + 1).zfill(7)
        suffix = "pfm"
        if self._is_full_dataset_format:
            base_folder = get_scene_flow_folder(
                dataset_path=self._dataset_path, scene_name=self.scene_name, split_name=self._split_name
            )
            annotation_path = base_folder / direction / sensor_name / f"{frame_id}.{suffix}"
        else:
            suffix = "flo"
            base_folder = get_scene_flow_folder(
                dataset_path=self._dataset_path, scene_name=self.scene_name, split_name=self._split_name
            ).parent
            annotation_path = base_folder / sensor_name / direction / f"{frame_id}.{suffix}"

        return annotation_path

    def _decode_image_dimensions(self, sensor_name: SensorName, frame_id: FrameId) -> Tuple[int, int, int]:
        img_path = self._get_image_file_path(sensor_name=sensor_name, frame_id=frame_id)

        with img_path.open("rb") as fh:
            width, height = imagesize.get(BytesIO(fh.read()))
            return height, width, 3

    def _decode_image_rgba(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        img_path = self._get_image_file_path(sensor_name=sensor_name, frame_id=frame_id)
        image_data = read_image(path=img_path, convert_to_rgb=True)

        ones = np.ones((*image_data.shape[:2], 1), dtype=image_data.dtype)
        concatenated = np.concatenate([image_data, ones], axis=-1)
        return concatenated

    def _decode_class_maps(self) -> Dict[AnnotationType, ClassMap]:
        return dict()

    def _decode_available_annotation_types(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> Dict[AnnotationType, AnnotationIdentifier]:
        optical_flow_file_name = f"{sensor_name}/{frame_id}"

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
            flow_vectors, backward_flow_vectors = self._decode_optical_flow(
                sensor_name=sensor_name, annotation_identifier=identifier, frame_id=frame_id
            )
            return OpticalFlow(vectors=flow_vectors, backward_vectors=backward_flow_vectors)
        else:
            raise NotImplementedError(f"{annotation_type} is not supported!")

    def _decode_optical_flow(
        self, sensor_name: SensorName, frame_id: FrameId, annotation_identifier: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        forward_flow_file_path = self._get_flow_file_path(sensor_name=sensor_name, frame_id=frame_id, forward=True)
        backward_flow_file_path = self._get_flow_file_path(sensor_name=sensor_name, frame_id=frame_id, forward=False)

        back_flow = read_flow(file_path=backward_flow_file_path) if backward_flow_file_path.exists() else None
        flow = read_flow(file_path=forward_flow_file_path) if forward_flow_file_path.exists() else None
        return flow, back_flow

    def _decode_file_path(self, sensor_name: SensorName, frame_id: FrameId, data_type: Type[F]) -> Optional[AnyPath]:
        if issubclass(data_type, OpticalFlow):
            annotation_identifiers = self.get_available_annotation_types(sensor_name=sensor_name, frame_id=frame_id)
            if data_type in annotation_identifiers:
                annotation_path = (
                    get_scene_flow_folder(
                        dataset_path=self._dataset_path, scene_name=self.scene_name, split_name=self._split_name
                    )
                    / annotation_identifiers[data_type]
                )
                return annotation_path
        elif issubclass(data_type, Image):
            img_path = self._get_image_file_path(sensor_name=sensor_name, frame_id=frame_id)
            return img_path
        return None
