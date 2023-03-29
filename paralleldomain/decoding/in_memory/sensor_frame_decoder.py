from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Generic, Optional, Tuple, Type, TypeVar, Union

import numpy as np

from paralleldomain.decoding.sensor_frame_decoder import F, T
from paralleldomain.model.annotation import Annotation, AnnotationType
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.sensor import (
    CameraSensorFrame,
    LidarSensorFrame,
    RadarSensorFrame,
    SensorExtrinsic,
    SensorIntrinsic,
    SensorPose,
)
from paralleldomain.model.type_aliases import AnnotationIdentifier, FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.projection import DistortionLookup

SensorFrameTypes = Union[CameraSensorFrame, RadarSensorFrame, LidarSensorFrame]
TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


@dataclass
class InMemorySensorFrameDecoder(Generic[TDateTime]):
    dataset_name: str
    scene_name: SceneName
    extrinsic: SensorExtrinsic
    sensor_pose: SensorPose
    annotations: Dict[str, Annotation]
    class_maps: Dict[str, ClassMap]

    date_time: TDateTime
    metadata: Dict[str, Any]

    def get_extrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> "SensorExtrinsic":
        return self.extrinsic

    def get_sensor_pose(self, sensor_name: SensorName, frame_id: FrameId) -> "SensorPose":
        return self.sensor_pose

    def get_annotations(
        self, sensor_name: SensorName, frame_id: FrameId, identifier: AnnotationIdentifier, annotation_type: T
    ) -> T:
        return self.annotations[annotation_type.__name__]

    def get_file_path(self, sensor_name: SensorName, frame_id: FrameId, data_type: Type[F]) -> Optional[AnyPath]:
        return None

    def get_available_annotation_types(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> Dict[AnnotationType, AnnotationIdentifier]:
        return {type(ann): rep for rep, ann in self.annotations.items()}

    def get_metadata(self, sensor_name: SensorName, frame_id: FrameId) -> Dict[str, Any]:
        return self.metadata

    def get_date_time(self, sensor_name: SensorName, frame_id: FrameId) -> TDateTime:
        return self.date_time

    def get_class_maps(self) -> Dict[AnnotationType, ClassMap]:
        return {type(self.annotations[k]): v for k, v in self.class_maps.items()}


@dataclass
class InMemoryCameraFrameDecoder(InMemorySensorFrameDecoder[TDateTime]):
    intrinsic: SensorIntrinsic
    rgba: np.ndarray
    image_dimensions: Tuple[int, int, int]
    distortion_lookup: Optional[DistortionLookup]

    def get_image_dimensions(self, sensor_name: SensorName, frame_id: FrameId) -> Tuple[int, int, int]:
        return self.image_dimensions

    def get_image_rgba(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        return self.rgba

    def get_intrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> "SensorIntrinsic":
        return self.intrinsic

    def get_distortion_lookup(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[DistortionLookup]:
        return self.distortion_lookup

    @staticmethod
    def from_camera_frame(camera_frame: CameraSensorFrame[TDateTime]) -> "InMemoryCameraFrameDecoder":
        annotations = dict()
        class_maps = dict()
        for anno_type in camera_frame.available_annotation_types:
            annotations[anno_type.__name__] = camera_frame.get_annotations(annotation_type=anno_type)
            class_maps[anno_type.__name__] = camera_frame.class_maps[anno_type]

        return InMemoryCameraFrameDecoder(
            dataset_name=camera_frame.dataset_name,
            scene_name=camera_frame.scene_name,
            extrinsic=camera_frame.extrinsic,
            sensor_pose=camera_frame.pose,
            annotations=annotations,
            class_maps=class_maps,
            date_time=camera_frame.date_time,
            metadata=camera_frame.metadata,
            distortion_lookup=camera_frame.distortion_lookup,
            intrinsic=camera_frame.intrinsic,
            rgba=camera_frame.image.rgba,
            image_dimensions=(camera_frame.image.height, camera_frame.image.width, camera_frame.image.channels),
        )


@dataclass
class InMemoryLidarFrameDecoder(InMemorySensorFrameDecoder[TDateTime]):
    point_cloud_size: int
    cloud_xyz: Optional[np.ndarray]
    cloud_rgb: Optional[np.ndarray]
    cloud_intensity: Optional[np.ndarray]
    cloud_timestamp: Optional[np.ndarray]
    cloud_ring_index: Optional[np.ndarray]
    cloud_ray_type: Optional[np.ndarray]

    def get_point_cloud_size(self, sensor_name: SensorName, frame_id: FrameId) -> int:
        return self.point_cloud_size

    def get_point_cloud_xyz(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        return self.cloud_xyz

    def get_point_cloud_rgb(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        return self.cloud_rgb

    def get_point_cloud_intensity(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        return self.cloud_intensity

    def get_point_cloud_timestamp(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        return self.cloud_timestamp

    def get_point_cloud_ring_index(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        return self.cloud_ring_index

    def get_point_cloud_ray_type(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        return self.cloud_ray_type

    @staticmethod
    def from_lidar_frame(lidar_frame: LidarSensorFrame[TDateTime]) -> "InMemoryLidarFrameDecoder":
        annotations = dict()
        class_maps = dict()
        for anno_type in lidar_frame.available_annotation_types:
            annotations[anno_type.__name__] = lidar_frame.get_annotations(annotation_type=anno_type)
            class_maps[anno_type.__name__] = lidar_frame.class_maps[anno_type]

        return InMemoryLidarFrameDecoder(
            dataset_name=lidar_frame.dataset_name,
            scene_name=lidar_frame.scene_name,
            extrinsic=lidar_frame.extrinsic,
            sensor_pose=lidar_frame.pose,
            annotations=annotations,
            class_maps=class_maps,
            date_time=lidar_frame.date_time,
            metadata=lidar_frame.metadata,
            point_cloud_size=lidar_frame.point_cloud.length,
            cloud_xyz=lidar_frame.point_cloud.xyz,
            cloud_rgb=lidar_frame.point_cloud.rgb,
            cloud_intensity=lidar_frame.point_cloud.intensity,
            cloud_timestamp=lidar_frame.point_cloud.ts,
            cloud_ring_index=lidar_frame.point_cloud.ring,
            cloud_ray_type=lidar_frame.point_cloud.ray_type,
        )


@dataclass
class InMemoryRadarFrameDecoder(InMemorySensorFrameDecoder[TDateTime]):
    point_cloud_size: int
    cloud_xyz: Optional[np.ndarray]
    cloud_rgb: Optional[np.ndarray]
    cloud_doppler: Optional[np.ndarray]
    cloud_power: Optional[np.ndarray]
    cloud_range: Optional[np.ndarray]
    cloud_azimuth: Optional[np.ndarray]
    cloud_elevation: Optional[np.ndarray]
    cloud_timestamp: Optional[np.ndarray]

    def get_radar_point_cloud_size(self, sensor_name: SensorName, frame_id: FrameId) -> int:
        return self.point_cloud_size

    def get_radar_point_cloud_xyz(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        return self.cloud_xyz

    def get_radar_point_cloud_doppler(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        return self.cloud_doppler

    def get_radar_point_cloud_rgb(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        return self.cloud_rgb

    def get_radar_point_cloud_power(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        return self.cloud_power

    def get_radar_point_cloud_range(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        return self.cloud_range

    def get_radar_point_cloud_azimuth(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        return self.cloud_azimuth

    def get_radar_point_cloud_elevation(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        return self.cloud_elevation

    def get_radar_point_cloud_timestamp(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        return self.cloud_timestamp

    @staticmethod
    def from_radar_frame(radar_frame: RadarSensorFrame[TDateTime]) -> "InMemoryRadarFrameDecoder":
        annotations = dict()
        class_maps = dict()
        for anno_type in radar_frame.available_annotation_types:
            annotations[anno_type.__name__] = radar_frame.get_annotations(annotation_type=anno_type)
            class_maps[anno_type.__name__] = radar_frame.class_maps[anno_type]

        return InMemoryRadarFrameDecoder(
            dataset_name=radar_frame.dataset_name,
            scene_name=radar_frame.scene_name,
            extrinsic=radar_frame.extrinsic,
            sensor_pose=radar_frame.pose,
            annotations=annotations,
            class_maps=class_maps,
            date_time=radar_frame.date_time,
            metadata=radar_frame.metadata,
            point_cloud_size=radar_frame.radar_point_cloud.length,
            cloud_xyz=radar_frame.radar_point_cloud.xyz,
            cloud_rgb=radar_frame.radar_point_cloud.rgb,
            cloud_timestamp=radar_frame.radar_point_cloud.ts,
            cloud_doppler=radar_frame.radar_point_cloud.doppler,
            cloud_power=radar_frame.radar_point_cloud.power,
            cloud_range=radar_frame.radar_point_cloud.range,
            cloud_azimuth=radar_frame.radar_point_cloud.azimuth,
            cloud_elevation=radar_frame.radar_point_cloud.elevation,
        )
