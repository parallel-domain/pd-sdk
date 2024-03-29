from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np

from paralleldomain.decoding.sensor_frame_decoder import T
from paralleldomain.model.annotation import Annotation, AnnotationIdentifier
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.sensor import (
    CameraSensorFrame,
    LidarSensorFrame,
    RadarSensorFrame,
    SensorDataCopyTypes,
    SensorExtrinsic,
    SensorIntrinsic,
    SensorPose,
)
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.projection import DistortionLookup

SensorFrameTypes = Union[CameraSensorFrame, RadarSensorFrame, LidarSensorFrame]
TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


@dataclass
class InMemorySensorFrameDecoder(Generic[TDateTime]):
    dataset_name: str
    scene_name: SceneName
    frame_id: FrameId
    sensor_name: SensorName
    scene_name: SceneName
    extrinsic: SensorExtrinsic
    sensor_pose: SensorPose
    annotations: Dict[AnnotationIdentifier, Annotation]
    class_maps: Dict[AnnotationIdentifier, ClassMap]

    date_time: TDateTime
    metadata: Dict[str, Any]

    def get_extrinsic(self) -> "SensorExtrinsic":
        return self.extrinsic

    def get_sensor_pose(self) -> "SensorPose":
        return self.sensor_pose

    def get_annotations(self, identifier: AnnotationIdentifier[T]) -> T:
        return self.annotations[identifier]

    def get_file_path(self, data_type: SensorDataCopyTypes) -> Optional[AnyPath]:
        # Note: We also support Type[Annotation] for data_type for backwards compatibility
        return None

    def get_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        return list(self.annotations.keys())

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata

    def get_date_time(self) -> TDateTime:
        return self.date_time

    def get_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        return self.class_maps


@dataclass
class InMemoryCameraFrameDecoder(InMemorySensorFrameDecoder[TDateTime]):
    intrinsic: SensorIntrinsic
    rgba: np.ndarray
    image_dimensions: Tuple[int, int, int]
    distortion_lookup: Optional[DistortionLookup]

    def get_image_dimensions(self) -> Tuple[int, int, int]:
        return self.image_dimensions

    def get_image_rgba(self) -> np.ndarray:
        return self.rgba

    def get_intrinsic(self) -> "SensorIntrinsic":
        return self.intrinsic

    def get_distortion_lookup(self) -> Optional[DistortionLookup]:
        return self.distortion_lookup

    @staticmethod
    def from_camera_frame(camera_frame: CameraSensorFrame[TDateTime]) -> "InMemoryCameraFrameDecoder":
        annotations = dict()
        class_maps = dict()
        for anno_identifier in camera_frame.available_annotation_identifiers:
            annotations[anno_identifier] = camera_frame.get_annotations(annotation_identifier=anno_identifier)
            if anno_identifier in camera_frame.class_maps:
                class_maps[anno_identifier] = camera_frame.class_maps[anno_identifier]

        return InMemoryCameraFrameDecoder(
            dataset_name=camera_frame.dataset_name,
            scene_name=camera_frame.scene_name,
            frame_id=camera_frame.frame_id,
            sensor_name=camera_frame.sensor_name,
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
    cloud_elongation: Optional[np.ndarray]
    cloud_timestamp: Optional[np.ndarray]
    cloud_ring_index: Optional[np.ndarray]
    cloud_ray_type: Optional[np.ndarray]

    def get_point_cloud_size(self) -> int:
        return self.point_cloud_size

    def get_point_cloud_xyz(self) -> Optional[np.ndarray]:
        return self.cloud_xyz

    def get_point_cloud_rgb(self) -> Optional[np.ndarray]:
        return self.cloud_rgb

    def get_point_cloud_intensity(self) -> Optional[np.ndarray]:
        return self.cloud_intensity

    def get_point_cloud_elongation(self) -> Optional[np.ndarray]:
        return self.cloud_elongation

    def get_point_cloud_timestamp(self) -> Optional[np.ndarray]:
        return self.cloud_timestamp

    def get_point_cloud_ring_index(self) -> Optional[np.ndarray]:
        return self.cloud_ring_index

    def get_point_cloud_ray_type(self) -> Optional[np.ndarray]:
        return self.cloud_ray_type

    @staticmethod
    def from_lidar_frame(lidar_frame: LidarSensorFrame[TDateTime]) -> "InMemoryLidarFrameDecoder":
        annotations = dict()
        class_maps = dict()
        for anno_identifier in lidar_frame.available_annotation_identifiers:
            annotations[anno_identifier] = lidar_frame.get_annotations(annotation_identifier=anno_identifier)
            class_maps[anno_identifier] = lidar_frame.class_maps[anno_identifier]

        return InMemoryLidarFrameDecoder(
            dataset_name=lidar_frame.dataset_name,
            scene_name=lidar_frame.scene_name,
            frame_id=lidar_frame.frame_id,
            sensor_name=lidar_frame.sensor_name,
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
            cloud_elongation=lidar_frame.point_cloud.elongation,
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

    def get_radar_point_cloud_size(self) -> int:
        return self.point_cloud_size

    def get_radar_point_cloud_xyz(self) -> Optional[np.ndarray]:
        return self.cloud_xyz

    def get_radar_point_cloud_doppler(self) -> Optional[np.ndarray]:
        return self.cloud_doppler

    def get_radar_point_cloud_rgb(self) -> Optional[np.ndarray]:
        return self.cloud_rgb

    def get_radar_point_cloud_power(self) -> Optional[np.ndarray]:
        return self.cloud_power

    def get_radar_point_cloud_range(self) -> Optional[np.ndarray]:
        return self.cloud_range

    def get_radar_point_cloud_azimuth(self) -> Optional[np.ndarray]:
        return self.cloud_azimuth

    def get_radar_point_cloud_elevation(self) -> Optional[np.ndarray]:
        return self.cloud_elevation

    def get_radar_point_cloud_timestamp(self) -> Optional[np.ndarray]:
        return self.cloud_timestamp

    @staticmethod
    def from_radar_frame(radar_frame: RadarSensorFrame[TDateTime]) -> "InMemoryRadarFrameDecoder":
        annotations = dict()
        class_maps = dict()
        for anno_identifier in radar_frame.available_annotation_identifiers:
            annotations[anno_identifier] = radar_frame.get_annotations(annotation_identifier=anno_identifier)
            class_maps[anno_identifier] = radar_frame.class_maps[anno_identifier]

        return InMemoryRadarFrameDecoder(
            dataset_name=radar_frame.dataset_name,
            scene_name=radar_frame.scene_name,
            frame_id=radar_frame.frame_id,
            sensor_name=radar_frame.sensor_name,
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
