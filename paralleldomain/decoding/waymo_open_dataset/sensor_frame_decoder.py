import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import cv2
import numpy as np

from paralleldomain.decoding.common import DecoderSettings, create_cache_key
from paralleldomain.decoding.sensor_frame_decoder import (
    CameraSensorFrameDecoder,
    LidarSensorFrameDecoder,
    SensorFrameDecoder,
)
from paralleldomain.decoding.waymo_open_dataset.common import (
    WAYMO_CAMERA_NAME_TO_INDEX,
    WaymoFileAccessMixin,
    decode_class_maps,
    get_cached_pre_calculated_scene_to_has_annotation,
)
from paralleldomain.decoding.waymo_open_dataset.frame_utils import (
    convert_range_image_to_point_cloud,
    parse_range_image_and_camera_projection,
)
from paralleldomain.model.annotation import (
    AnnotationIdentifier,
    AnnotationTypes,
    BoundingBox2D,
    BoundingBox3D,
    BoundingBoxes2D,
    BoundingBoxes3D,
    InstanceSegmentation2D,
    SemanticSegmentation2D,
)
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.sensor import SensorDataCopyTypes, SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image_bytes
from paralleldomain.utilities.transformation import Transformation

T = TypeVar("T")

"""
Waymo point data is in FLU. Parsing the extrinsic is not implemented right now
"""


class WaymoOpenDatasetSensorFrameDecoder(SensorFrameDecoder[datetime], WaymoFileAccessMixin):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        frame_id: FrameId,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        use_precalculated_maps: bool,
        split_name: str,
        is_unordered_scene: bool,
        index_folder: Optional[AnyPath],
        scene_decoder,
    ):
        CameraSensorFrameDecoder.__init__(
            self=self,
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            frame_id=frame_id,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        WaymoFileAccessMixin.__init__(self=self, record_path=dataset_path / scene_name)
        self._dataset_path = dataset_path
        self.split_name = split_name
        self.index_folder = index_folder
        self.use_precalculated_maps = use_precalculated_maps
        if use_precalculated_maps is True and index_folder is None:
            raise ValueError("Index folder is required to use precalculated maps!")

    def _decode_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        return decode_class_maps()

    def _decode_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        available_annotations = [AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes3D)]

        if self.use_precalculated_maps:
            has_segmentation = get_cached_pre_calculated_scene_to_has_annotation(
                lazy_load_cache=self.lazy_load_cache,
                dataset_name=self.dataset_name,
                scene_name=self.scene_name,
                sensor_name=self.sensor_name,
                frame_id=self.frame_id,
                split_name=self.split_name,
                annotation_type=AnnotationTypes.SemanticSegmentation2D,
                index_folder=self.index_folder,
            )
            if has_segmentation:
                available_annotations.append(
                    AnnotationIdentifier(annotation_type=AnnotationTypes.SemanticSegmentation2D)
                )
                available_annotations.append(
                    AnnotationIdentifier(annotation_type=AnnotationTypes.InstanceSegmentation2D)
                )

            has_bounding_box_2d = get_cached_pre_calculated_scene_to_has_annotation(
                lazy_load_cache=self.lazy_load_cache,
                dataset_name=self.dataset_name,
                scene_name=self.scene_name,
                sensor_name=self.sensor_name,
                frame_id=self.frame_id,
                split_name=self.split_name,
                annotation_type=AnnotationTypes.BoundingBoxes2D,
                index_folder=self.index_folder,
            )

            if has_bounding_box_2d:
                available_annotations.append(AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D))

        else:
            record = self.get_record_at(frame_id=self.frame_id)
            if isinstance(self, WaymoOpenDatasetCameraSensorFrameDecoder):
                cam_index = WAYMO_CAMERA_NAME_TO_INDEX[self.sensor_name] - 1
                cam_data = record.images[cam_index]
                if cam_data.camera_segmentation_label.panoptic_label:
                    available_annotations.append(
                        AnnotationIdentifier(annotation_type=AnnotationTypes.SemanticSegmentation2D)
                    )
                    available_annotations.append(
                        AnnotationIdentifier(annotation_type=AnnotationTypes.InstanceSegmentation2D)
                    )

                cam_labels = record.camera_labels[cam_index].labels
                if cam_labels:
                    available_annotations.append(AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes2D))

        return available_annotations

    def _decode_metadata(self) -> Dict[str, Any]:
        record = self.get_record_at(frame_id=self.frame_id)
        return dict(stats=record.context.stats, name=record.context.name, timestamp_micros=record.timestamp_micros)

    def _decode_extrinsic(self) -> SensorExtrinsic:
        return SensorExtrinsic.from_transformation_matrix(np.eye(4))

    def _decode_sensor_pose(self) -> SensorPose:
        return SensorPose.from_transformation_matrix(np.eye(4))

    def _decode_annotations(self, identifier: AnnotationIdentifier[T]) -> T:
        if issubclass(identifier.annotation_type, SemanticSegmentation2D):
            class_ids = self._decode_semantic_segmentation_2d()
            return SemanticSegmentation2D(class_ids=class_ids)
        if issubclass(identifier.annotation_type, InstanceSegmentation2D):
            instance_ids = self._decode_instance_segmentation_2d()
            return InstanceSegmentation2D(instance_ids=instance_ids)
        if issubclass(identifier.annotation_type, BoundingBoxes3D):
            boxes = self._decode_bounding_boxes_3d()
            return BoundingBoxes3D(boxes=boxes)
        if issubclass(identifier.annotation_type, BoundingBoxes2D):
            boxes = self._decode_bounding_boxes_2d()
            return BoundingBoxes2D(boxes=boxes)
        else:
            raise NotImplementedError(f"{identifier} is not supported!")

    def _decode_semantic_segmentation_2d(self) -> np.ndarray:
        record = self.get_record_at(frame_id=self.frame_id)

        cam_index = WAYMO_CAMERA_NAME_TO_INDEX[self.sensor_name] - 1
        cam_data = record.images[cam_index]
        panoptic_label = cam_data.camera_segmentation_label.panoptic_label

        panoptic_label = cv2.imdecode(buf=np.frombuffer(panoptic_label, np.uint8), flags=cv2.IMREAD_UNCHANGED)
        panoptic_label_divisor = cam_data.camera_segmentation_label.panoptic_label_divisor

        segmentation_label, _ = np.divmod(panoptic_label, panoptic_label_divisor)
        return np.expand_dims(segmentation_label, -1).astype(int)

    def _decode_instance_segmentation_2d(self) -> np.ndarray:
        record = self.get_record_at(frame_id=self.frame_id)

        cam_index = WAYMO_CAMERA_NAME_TO_INDEX[self.sensor_name] - 1
        cam_data = record.images[cam_index]
        panoptic_label = cam_data.camera_segmentation_label.panoptic_label

        panoptic_label = cv2.imdecode(buf=np.frombuffer(panoptic_label, np.uint8), flags=cv2.IMREAD_UNCHANGED)
        panoptic_label_divisor = cam_data.camera_segmentation_label.panoptic_label_divisor

        _, instance_label = np.divmod(panoptic_label, panoptic_label_divisor)
        return np.expand_dims(instance_label, -1).astype(int)

    def _decode_bounding_boxes_3d(self) -> List[BoundingBox3D]:
        boxes = list()
        record = self.get_record_at(frame_id=self.frame_id)
        """
        From Waymo Open Dataset utils.keypoint_data:
        heading: The heading of the bounding box (in radians). It is a float tensor
            with shape [B]. Boxes are axis aligned in 2D, so it is None for camera
            bounding boxes. For 3D boxes the heading is the angle required to rotate
            +x to the surface normal of the box front face. It is normalized to [-pi,
            pi).
        """
        for raw_box in record.laser_labels:
            pose = Transformation.from_euler_angles(
                angles=[0, 0, raw_box.box.heading],
                order="xyz",
                degrees=False,
                translation=[raw_box.box.center_x, raw_box.box.center_y, raw_box.box.center_z],
            )

            boxes.append(
                BoundingBox3D(
                    pose=pose,
                    length=raw_box.box.length,  # x-axis
                    width=raw_box.box.width,  # y-axis
                    height=raw_box.box.height,  # z-axis
                    class_id=raw_box.type,
                    num_points=raw_box.num_lidar_points_in_box,
                    # convert the hash-IDs from waymo into an 8 digit integer-ID
                    instance_id=int(hashlib.sha256(raw_box.id.encode("utf-8")).hexdigest(), 16) % 10**8,
                    attributes={"ori_instance_id": raw_box.id},  # additional metadata can be added here
                )
            )
        return boxes

    def _decode_bounding_boxes_2d(self) -> List[BoundingBox2D]:
        boxes = list()
        record = self.get_record_at(frame_id=self.frame_id)
        cam_index = WAYMO_CAMERA_NAME_TO_INDEX[self.sensor_name] - 1
        cam_labels = record.camera_labels[cam_index].labels

        for raw_box in cam_labels:
            boxes.append(
                BoundingBox2D(
                    # The length/width naming in waymo is a bit weird, probably because they
                    # wanted to some consistency between box dimensions between 3D and 2D.
                    x=round(raw_box.box.center_x - 0.5 * raw_box.box.length),
                    y=round(raw_box.box.center_y - 0.5 * raw_box.box.width),
                    width=round(raw_box.box.length),
                    height=round(raw_box.box.width),
                    class_id=raw_box.type,
                    # convert the hash-IDs from waymo into an 8 digit integer-ID
                    instance_id=int(hashlib.sha256(raw_box.id.encode("utf-8")).hexdigest(), 16) % 10**8,
                    attributes={"ori_instance_id": raw_box.id},  # additional metadata can be added here
                )
            )
        return boxes

    def _decode_file_path(self, data_type: SensorDataCopyTypes) -> Optional[AnyPath]:
        # Record path is the same for all sensors.
        return self.record_path

    def _decode_date_time(self) -> datetime:
        record = self.get_record_at(frame_id=self.frame_id)
        return datetime.fromtimestamp(record.timestamp_micros / 1000000)


class WaymoOpenDatasetLidarSensorFrameDecoder(LidarSensorFrameDecoder[datetime], WaymoOpenDatasetSensorFrameDecoder):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        sensor_name: SensorName,
        frame_id: FrameId,
        settings: DecoderSettings,
        use_precalculated_maps: bool,
        split_name: str,
        include_second_returns: bool,
        is_unordered_scene: bool,
        index_folder: Optional[AnyPath],
        scene_decoder,
    ):
        self._dataset_path = dataset_path
        self.include_second_returns = include_second_returns
        LidarSensorFrameDecoder.__init__(
            self=self,
            dataset_name=dataset_name,
            scene_name=scene_name,
            settings=settings,
            sensor_name=sensor_name,
            frame_id=frame_id,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        WaymoOpenDatasetSensorFrameDecoder.__init__(
            self=self,
            dataset_name=dataset_name,
            scene_name=scene_name,
            dataset_path=self._dataset_path,
            sensor_name=sensor_name,
            frame_id=frame_id,
            settings=settings,
            use_precalculated_maps=use_precalculated_maps,
            split_name=split_name,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
            index_folder=index_folder,
        )

    def _decode_date_time(self) -> datetime:
        # Lidar timestamp is timestamp of the first top LiDAR scan within this frame.
        # Will not match exactly the datetime for camera sensor frames or the ego pose,
        # Both of which should align roughly in the middle of the lidar sensor frame.
        record = self.get_record_at(frame_id=self.frame_id)
        return datetime.fromtimestamp(record.timestamp_micros / 1e6)

    # TODO: Add include_second_returns boolean
    def _decode_point_cloud_data(self) -> Optional[np.ndarray]:
        """
        Waymo record.lasers schema is [x, y, z, intensity, elongation]
        Elongation :
        Lidar elongation refers to the elongation of the pulse beyond its nominal width.
        Returns with long pulse elongation, for example, indicate that the laser reflection
        is potentially smeared or refracted, such that the return pulse is elongated in time.
        (Source: https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/waymo_dataset.html)
        """
        _unique_cache_key = create_cache_key(
            dataset_name="Waymo Open Dataset",
            scene_name=self.record_path.name,
            frame_id=self.frame_id,
            extra="_decode_point_cloud_data",
            sensor_name=self.sensor_name,
        )

        def _decode_point_cloud_data() -> Optional[np.ndarray]:
            record = self.get_record_at(frame_id=self.frame_id)
            # Range image to point cloud processing
            (range_images, _, range_image_top_pose) = parse_range_image_and_camera_projection(
                record=record, sensor_name=self.sensor_name
            )

            # Point Cloud Conversion and Viz
            points = convert_range_image_to_point_cloud(record, range_images, range_image_top_pose)
            points_ri2 = convert_range_image_to_point_cloud(record, range_images, range_image_top_pose, ri_index=1)

            include_second_returns = True  # Pull this flag out
            if include_second_returns:
                return np.concatenate([*points, *points_ri2], axis=0)
            else:
                return np.concatenate([*points], axis=0)

        return self.lazy_load_cache.get_item(key=_unique_cache_key, loader=lambda: _decode_point_cloud_data())

    def _decode_point_cloud_size(self) -> int:
        data = self._decode_point_cloud_data()
        return len(data)

    def _decode_point_cloud_xyz(self) -> Optional[np.ndarray]:
        data = self._decode_point_cloud_data()
        return data[:, :3]

    def _decode_point_cloud_rgb(self) -> Optional[np.ndarray]:
        """
        Waymo Open Dataset point cloud does not have RGB values, so returns np.ndarray of zeros .
        """
        cloud_size = self._decode_point_cloud_size()
        return np.zeros([cloud_size, 3])

    def _decode_point_cloud_intensity(self) -> Optional[np.ndarray]:
        data = self._decode_point_cloud_data()
        return data[:, 3].reshape(-1, 1)

    def _decode_point_cloud_elongation(self) -> Optional[np.ndarray]:
        data = self._decode_point_cloud_data()
        return data[:, 4].reshape(-1, 1)

    def _decode_point_cloud_timestamp(self) -> Optional[np.ndarray]:
        return -1 * np.ones((self._decode_point_cloud_size(), 1))

    def _decode_point_cloud_ring_index(self) -> Optional[np.ndarray]:
        return -1 * np.ones((self._decode_point_cloud_size(), 1))

    def _decode_point_cloud_ray_type(self) -> Optional[np.ndarray]:
        return None


class WaymoOpenDatasetCameraSensorFrameDecoder(CameraSensorFrameDecoder[datetime], WaymoOpenDatasetSensorFrameDecoder):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        sensor_name: SensorName,
        frame_id: FrameId,
        settings: DecoderSettings,
        use_precalculated_maps: bool,
        split_name: str,
        is_unordered_scene: bool,
        index_folder: Optional[AnyPath],
        scene_decoder,
    ):
        CameraSensorFrameDecoder.__init__(
            self=self,
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            frame_id=frame_id,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        WaymoOpenDatasetSensorFrameDecoder.__init__(
            self=self,
            dataset_name=dataset_name,
            scene_name=scene_name,
            dataset_path=dataset_path,
            sensor_name=sensor_name,
            frame_id=frame_id,
            settings=settings,
            use_precalculated_maps=use_precalculated_maps,
            split_name=split_name,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
            index_folder=index_folder,
        )
        self._dataset_path = dataset_path
        self.split_name = split_name
        self.use_precalculated_maps = use_precalculated_maps
        self._dimensions: Optional[Tuple[int, int, int]] = None

    def _decode_date_time(self) -> datetime:
        # Camera timestamp is camera shutter time, which will not match exactly the datetime
        # for the lidar sensor frame, since lidar scanning is not instantaneous.
        record = self.get_record_at(frame_id=self.frame_id)

        cam_index = WAYMO_CAMERA_NAME_TO_INDEX[self.sensor_name] - 1
        cam_data = record.images[cam_index]
        return datetime.fromtimestamp(cam_data.camera_trigger_time)

    def _decode_intrinsic(self) -> SensorIntrinsic:
        return SensorIntrinsic()

    def _decode_image_dimensions(self) -> Tuple[int, int, int]:
        if self._dimensions is None:
            record = self.get_record_at(frame_id=self.frame_id)
            cam_index = WAYMO_CAMERA_NAME_TO_INDEX[self.sensor_name] - 1
            cam_data = record.images[cam_index]
            image_data = read_image_bytes(images_bytes=cam_data.image, convert_to_rgb=True)
            self._dimensions = (image_data.shape[0], image_data.shape[1], 3)
        return self._dimensions

    def _decode_image_rgba(self) -> np.ndarray:
        record = self.get_record_at(frame_id=self.frame_id)

        cam_index = WAYMO_CAMERA_NAME_TO_INDEX[self.sensor_name] - 1
        cam_data = record.images[cam_index]

        image_data = read_image_bytes(images_bytes=cam_data.image, convert_to_rgb=True)

        ones = np.ones((*image_data.shape[:2], 1), dtype=image_data.dtype)
        if image_data.dtype == np.uint8:
            ones *= 255
        concatenated = np.concatenate([image_data, ones], axis=-1)
        self._dimensions = (image_data.shape[0], image_data.shape[1], 3)
        return concatenated
