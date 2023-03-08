from datetime import datetime
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import cv2
import imagesize
import numpy as np

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.sensor_frame_decoder import (
    CameraSensorFrameDecoder,
    F,
    LidarSensorFrameDecoder,
    SensorFrameDecoder,
)
from paralleldomain.decoding.waymo_open_dataset.common import (
    WAYMO_CAMERA_NAME_TO_INDEX,
    WaymoFileAccessMixin,
    decode_class_maps,
    get_cached_pre_calculated_scene_to_has_segmentation,
)
from paralleldomain.decoding.waymo_open_dataset.frame_utils import (
    convert_range_image_to_point_cloud,
    parse_range_image_and_camera_projection,
)
from paralleldomain.model.annotation import (
    AnnotationType,
    AnnotationTypes,
    BoundingBox3D,
    BoundingBoxes3D,
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


class WaymoOpenDatasetSensorFrameDecoder(SensorFrameDecoder[datetime], WaymoFileAccessMixin):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        use_precalculated_maps: bool,
        split_name: str,
    ):
        CameraSensorFrameDecoder.__init__(
            self=self, dataset_name=dataset_name, scene_name=scene_name, settings=settings
        )
        WaymoFileAccessMixin.__init__(self=self, record_path=dataset_path / scene_name)
        self._dataset_path = dataset_path
        self.split_name = split_name
        self.use_precalculated_maps = use_precalculated_maps

    def _decode_class_maps(self) -> Dict[AnnotationType, ClassMap]:
        return decode_class_maps()

    def _decode_available_annotation_types(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> Dict[AnnotationType, AnnotationIdentifier]:
        if self.use_precalculated_maps and self.split_name == "training":
            has_segmentation = get_cached_pre_calculated_scene_to_has_segmentation(
                lazy_load_cache=self.lazy_load_cache,
                dataset_name=self.dataset_name,
                scene_name=self.scene_name,
                sensor_name=sensor_name,
                frame_id=frame_id,
                split_name=self.split_name,
            )
            if has_segmentation:
                return {
                    AnnotationTypes.SemanticSegmentation2D: f"{frame_id}",
                    AnnotationTypes.InstanceSegmentation2D: f"{frame_id}",
                }
            else:
                return dict()

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
        return dict(stats=record.context.stats, name=record.context.name, timestamp_micros=record.timestamp_micros)

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
        if issubclass(annotation_type, BoundingBoxes3D):
            boxes = self._decode_bounding_boxes_3d(sensor_name=sensor_name, frame_id=frame_id)
            return BoundingBoxes3D(boxes=boxes)
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

    # TODO: Fill in
    def _decode_bounding_boxes_3d(self, sensor_name: SensorName, frame_id: FrameId) -> List[BoundingBox3D]:
        record = self.get_record_at(frame_id=frame_id)
        # TODO: Should we separate LiDAR scanners? The bboxes are not split out by scanner but the point cloud is.

    def _decode_file_path(self, sensor_name: SensorName, frame_id: FrameId, data_type: Type[F]) -> Optional[AnyPath]:
        return None


class WaymoOpenDatasetLidarSensorFrameDecoder(LidarSensorFrameDecoder[datetime], WaymoOpenDatasetSensorFrameDecoder):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        use_precalculated_maps: bool,
        split_name: str,
    ):
        self._dataset_path = dataset_path
        LidarSensorFrameDecoder.__init__(self=self, dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        WaymoOpenDatasetSensorFrameDecoder.__init__(
            self=self,
            dataset_name=dataset_name,
            scene_name=scene_name,
            dataset_path=self._dataset_path,
            settings=settings,
            use_precalculated_maps=use_precalculated_maps,
            split_name=split_name,
        )
        self._decode_point_cloud_data = lru_cache(maxsize=1)(self._decode_point_cloud_data)

    # TODO: Add include_second_returns boolean
    # TODO: Add intensity
    # TODO: Add ring index
    def _decode_point_cloud_data(self, frame_id: FrameId, sensor_name: SensorName = "lidar") -> Optional[np.ndarray]:
        """
        Waymo record.lasers schema is [range, intensity, and elongation, x, y, z]
        """
        record = self.get_record_at(frame_id=frame_id)
        # Range image to point cloud processing
        (range_images, _, range_image_top_pose) = parse_range_image_and_camera_projection(record)

        # Point Cloud Conversion and Viz
        points = convert_range_image_to_point_cloud(record, range_images, range_image_top_pose)
        points_ri2 = convert_range_image_to_point_cloud(record, range_images, range_image_top_pose, ri_index=1)
        # 3d points in vehicle frame.
        points_all = np.concatenate(points, axis=0)
        points_all_ri2 = np.concatenate(points_ri2, axis=0)

        include_second_returns = True  # Pull this flag out
        if include_second_returns:
            return np.concatenate([points_all, points_all_ri2], axis=0)
        else:
            return points_all

    def _decode_point_cloud_size(self, sensor_name: SensorName, frame_id: FrameId) -> int:
        data = self._decode_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id)
        return len(data)

    def _decode_point_cloud_xyz(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        data = self._decode_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id)
        return data[:, :3]

    def _decode_point_cloud_rgb(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        """
        Waymo Open Dataset point cloud does not have RGB values, so returns np.ndarray of zeros .
        """
        cloud_size = self._decode_point_cloud_size(sensor_name=sensor_name, frame_id=frame_id)
        return np.zeros([cloud_size, 3])

    def _decode_point_cloud_intensity(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        # data = self._decode_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id)
        # return data[:, 3].reshape(-1, 1)
        raise NotImplementedError("Currently not decoding intensity for Waymo Open Dataset.")

    def _decode_point_cloud_timestamp(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        # return -1 * np.ones(self._decode_point_cloud_size(sensor_name=sensor_name, frame_id=frame_id))
        raise NotImplementedError("Currently not decoding timestamps for Waymo Open Dataset.")

    # TODO: Replace with decode_point_cloud_laser_index
    def _decode_point_cloud_ring_index(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        raise NotImplementedError("No ring data for Waymo Open Dataset.")

    def _decode_point_cloud_ray_type(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        return None


class WaymoOpenDatasetCameraSensorFrameDecoder(CameraSensorFrameDecoder[datetime], WaymoOpenDatasetSensorFrameDecoder):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        use_precalculated_maps: bool,
        split_name: str,
    ):
        CameraSensorFrameDecoder.__init__(
            self=self, dataset_name=dataset_name, scene_name=scene_name, settings=settings
        )
        WaymoOpenDatasetSensorFrameDecoder.__init__(
            self=self,
            dataset_name=dataset_name,
            scene_name=scene_name,
            dataset_path=dataset_path,
            settings=settings,
            use_precalculated_maps=use_precalculated_maps,
            split_name=split_name,
        )
        self._dataset_path = dataset_path
        self.split_name = split_name
        self.use_precalculated_maps = use_precalculated_maps

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
