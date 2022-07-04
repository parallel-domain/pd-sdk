from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import cv2
import imagesize
import numpy as np

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder, F
from paralleldomain.decoding.waymo.common import WAYMO_CAMERA_NAME_TO_INDEX, WaymoFileAccessMixin, decode_class_maps
from paralleldomain.model.annotation import (
    AnnotationType,
    AnnotationTypes,
    InstanceSegmentation2D,
    SemanticSegmentation2D,
)
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.image import Image
from paralleldomain.model.sensor import SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import AnnotationIdentifier, FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image_bytes, read_json

T = TypeVar("T")


class WaymoOpenDatasetCameraSensorFrameDecoder(CameraSensorFrameDecoder[datetime], WaymoFileAccessMixin):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
    ):
        CameraSensorFrameDecoder.__init__(
            self=self, dataset_name=dataset_name, scene_name=scene_name, settings=settings
        )
        WaymoFileAccessMixin.__init__(self=self, record_path=dataset_path / scene_name)
        self._dataset_path = dataset_path

    def _decode_intrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorIntrinsic:
        return SensorIntrinsic()

    def _decode_image_dimensions(self, sensor_name: SensorName, frame_id: FrameId) -> Tuple[int, int, int]:
        img = self.get_image_rgba(sensor_name=sensor_name, frame_id=frame_id)
        return img.shape[0], img.shape[1], 3

    def _decode_image_rgba(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        record = self.get_record_at(frame_id=frame_id)

        cam_index = WAYMO_CAMERA_NAME_TO_INDEX[sensor_name] - 1
        cam_data = record.images[cam_index]

        image_data = read_image_bytes(images_bytes=cam_data.image, convert_to_rgb=True)

        ones = np.ones((*image_data.shape[:2], 1), dtype=image_data.dtype)
        concatenated = np.concatenate([image_data, ones], axis=-1)
        return concatenated

    def _decode_class_maps(self) -> Dict[AnnotationType, ClassMap]:
        return decode_class_maps()

    def _decode_available_annotation_types(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> Dict[AnnotationType, AnnotationIdentifier]:
        record = self.get_record_at(frame_id=frame_id)

        cam_index = WAYMO_CAMERA_NAME_TO_INDEX[sensor_name] - 1
        cam_data = record.images[cam_index]
        if cam_data.camera_segmentation_label.panoptic_label:
            return {
                AnnotationTypes.SemanticSegmentation2D: f"{frame_id}",
                AnnotationTypes.InstanceSegmentation2D: f"{frame_id}",
            }
        return dict()

    def _decode_metadata(self, sensor_name: SensorName, frame_id: FrameId) -> Dict[str, Any]:
        record = self.get_record_at(frame_id=frame_id)
        return dict(stats=record.context.stats, name=record.context.name)

    def _decode_date_time(self, sensor_name: SensorName, frame_id: FrameId) -> datetime:
        record = self.get_record_at(frame_id=frame_id)

        cam_index = WAYMO_CAMERA_NAME_TO_INDEX[sensor_name] - 1
        cam_data = record.images[cam_index]
        return datetime.fromtimestamp(cam_data.camera_trigger_time)

    def _decode_extrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorExtrinsic:
        return SensorExtrinsic.from_transformation_matrix(np.eye(4))

    def _decode_sensor_pose(self, sensor_name: SensorName, frame_id: FrameId) -> SensorPose:
        return SensorPose.from_transformation_matrix(np.eye(4))

    def _decode_annotations(
        self, sensor_name: SensorName, frame_id: FrameId, identifier: AnnotationIdentifier, annotation_type: T
    ) -> T:
        if issubclass(annotation_type, SemanticSegmentation2D):
            class_ids = self._decode_semantic_segmentation_2d(sensor_name=sensor_name, frame_id=frame_id)
            return SemanticSegmentation2D(class_ids=class_ids)
        if issubclass(annotation_type, InstanceSegmentation2D):
            instance_ids = self._decode_isntance_segmentation_2d(sensor_name=sensor_name, frame_id=frame_id)
            return InstanceSegmentation2D(instance_ids=instance_ids)
        else:
            raise NotImplementedError(f"{annotation_type} is not supported!")

    def _decode_semantic_segmentation_2d(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        record = self.get_record_at(frame_id=frame_id)

        cam_index = WAYMO_CAMERA_NAME_TO_INDEX[sensor_name] - 1
        cam_data = record.images[cam_index]
        panoptic_label = cam_data.camera_segmentation_label.panoptic_label

        panoptic_label = cv2.imdecode(
            buf=np.frombuffer(panoptic_label, np.uint8),
            flags=cv2.IMREAD_UNCHANGED,
        )
        panoptic_label_divisor = cam_data.camera_segmentation_label.panoptic_label_divisor

        segmentation_label, _ = np.divmod(panoptic_label, panoptic_label_divisor)
        return np.expand_dims(segmentation_label, -1).astype(int)

    def _decode_isntance_segmentation_2d(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        record = self.get_record_at(frame_id=frame_id)

        cam_index = WAYMO_CAMERA_NAME_TO_INDEX[sensor_name] - 1
        cam_data = record.images[cam_index]
        panoptic_label = cam_data.camera_segmentation_label.panoptic_label

        panoptic_label = cv2.imdecode(
            buf=np.frombuffer(panoptic_label, np.uint8),
            flags=cv2.IMREAD_UNCHANGED,
        )
        panoptic_label_divisor = cam_data.camera_segmentation_label.panoptic_label_divisor

        _, instance_label = np.divmod(panoptic_label, panoptic_label_divisor)
        return np.expand_dims(instance_label, -1).astype(int)

    def _decode_file_path(self, sensor_name: SensorName, frame_id: FrameId, data_type: Type[F]) -> Optional[AnyPath]:
        return None
