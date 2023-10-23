import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from paralleldomain.decoding.directory.common import resolve_scene_folder
from paralleldomain.decoding.directory.sensor_frame_decoder import DirectoryBaseSensorFrameDecoder
from paralleldomain.decoding.kitti.common import _cached_point_cloud, _cached_sensor_frame_calibrations
from paralleldomain.decoding.sensor_frame_decoder import LidarSensorFrameDecoder, T
from paralleldomain.model.annotation import (
    AnnotationIdentifier,
    AnnotationType,
    AnnotationTypes,
    BoundingBox3D,
    BoundingBoxes3D,
)
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.point_cloud import PointCloud
from paralleldomain.model.sensor import SensorExtrinsic, SensorPose
from paralleldomain.model.type_aliases import FrameId, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.coordinate_system import CoordinateSystem
from paralleldomain.utilities.transformation import Transformation

logger = logging.getLogger(__name__)
RDF_TO_FLU = CoordinateSystem("RDF") > CoordinateSystem("FLU")


class KittiLidarSensorFrameDecoder(DirectoryBaseSensorFrameDecoder, LidarSensorFrameDecoder[None]):
    """
    Kitti labels are organised as follow:
    (see https://docs.nvidia.com/tao/tao-toolkit/text/data_annotation_format.html)
    Note: camera (RDF) - velodyne (FLU)

        #Values    Name      Description
        ----------------------------------------------------------------------------
           1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                             'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                             'Misc' or 'DontCare'
           1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                             truncated refers to the object leaving image boundaries
           1    occluded     Integer (0,1,2,3) indicating occlusion state:
                             0 = fully visible, 1 = partly occluded
                             2 = largely occluded, 3 = unknown
           1    alpha        Observation angle of object, ranging [-pi..pi]
           4    bbox         2D bounding box of object in the image (0-based index):
                             contains left, top, right, bottom pixel coordinates
           3    dimensions   3D object dimensions: height, width, length (in meters)
           3    location     3D object location x,y,z in camera coordinates (in meters)
           1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
           1    score        Only for results: Float, indicating confidence in
                             detection, needed for p/r curves, higher is better.
    """

    def __init__(self, point_cloud_dim: int, sensor_name: SensorName, frame_id: FrameId, **kwargs):
        self.point_cloud_dim = point_cloud_dim
        super().__init__(sensor_name=sensor_name, frame_id=frame_id, **kwargs)

    def _decode_calibration(self, scene_folder_path: AnyPath) -> Dict:
        return _cached_sensor_frame_calibrations(scene_folder_path=scene_folder_path, frame_id=self.frame_id)

    def _decode_extrinsic(self) -> SensorExtrinsic:
        scene_folder = resolve_scene_folder(dataset_path=self._dataset_path, scene_name=self.scene_name)
        calibration_dict = self._decode_calibration(scene_folder_path=scene_folder)

        return SensorExtrinsic.from_transformation_matrix(
            calibration_dict["Tr_imu_to_velo"], approximate_orthogonal=True
        ).inverse

    def _decode_sensor_pose(self) -> SensorPose:
        scene_folder = resolve_scene_folder(dataset_path=self._dataset_path, scene_name=self.scene_name)
        calibration_dict = self._decode_calibration(scene_folder_path=scene_folder)

        return SensorPose.from_transformation_matrix(
            calibration_dict["Tr_imu_to_velo"], approximate_orthogonal=True
        ).inverse

    def _get_height_width_length_from_kitti_line(self, split_line: List[str]) -> Tuple[float, float, float]:
        dimensions = [float(element) for element in split_line[8:11]]
        return dimensions[0], dimensions[1], dimensions[2]  # height, width, length

    def _get_rdf_position(self, split_line: List[str], height: Optional[float] = 0) -> Tuple[float, float, float]:
        location = [float(element) for element in split_line[11:14]]
        # Kitti box 3D origin is on the bottom plane, PD-sdk box 3D origin is the true centre of the box
        # we need to shift up the location by half the height (note: locations are in RDF coordinate)
        return location[0], location[1] - 0.5 * height, location[2]  # RDF coordinate

    def _get_class_name(self, split_line: List[str]) -> str:
        return split_line[0]

    def _get_yaw(self, split_line: List[str]) -> float:
        return float(split_line[14]) + np.pi / 2

    def _get_euler_angles(self, split_line: List[str]) -> List[float]:
        return [0.0, self._get_yaw(split_line=split_line), 0.0]

    def _decode_annotations(self, identifier: AnnotationIdentifier[T]) -> T:
        scene_folder = resolve_scene_folder(dataset_path=self._dataset_path, scene_name=self.scene_name)
        calibration_dict = self._decode_calibration(scene_folder_path=scene_folder)

        # note that 3D annotations are in camera coordinate system (RDF),
        # PD-SDK annotations need to be in sensor coordinate system (FLU)
        velo_to_cam = Transformation.from_transformation_matrix(
            calibration_dict["Tr_velo_to_cam"], approximate_orthogonal=True
        )

        if identifier.annotation_type is AnnotationTypes.BoundingBoxes3D:
            annotation_file = (
                scene_folder / self._data_type_to_folder_name[AnnotationTypes.BoundingBoxes3D] / f"{self.frame_id}.txt"
            )
            class_map = self.get_class_maps()[identifier.annotation_type]

            boxes = list()
            with annotation_file.open("r") as f:
                for i, line in enumerate(f):
                    kitti_label_split_line = line.split(" ")
                    class_type = self._get_class_name(split_line=kitti_label_split_line)

                    height, width, length = self._get_height_width_length_from_kitti_line(
                        split_line=kitti_label_split_line
                    )
                    classdetail = class_map.get_class_detail_from_name(class_name=class_type)
                    if classdetail is None:
                        logger.warning(f"class name {class_type} is not in ClassDetails, ignored!")
                        pass
                    else:
                        pose = Transformation.from_euler_angles(
                            angles=self._get_euler_angles(split_line=kitti_label_split_line),
                            translation=self._get_rdf_position(split_line=kitti_label_split_line, height=height),
                            order="xyz",
                            degrees=False,
                        )
                        # re-centre the box pose into FLU sensor coordinate system
                        pose = velo_to_cam.inverse @ pose @ RDF_TO_FLU.inverse
                        box = BoundingBox3D(
                            pose=pose,
                            class_id=classdetail.id,
                            instance_id=i,
                            num_points=-1,
                            height=height,
                            width=width,
                            length=length,
                            attributes={},
                        )
                        boxes.append(box)

            return BoundingBoxes3D(boxes=boxes)
        else:
            raise NotImplementedError(f"{identifier.annotation_type} is not supported!")

    def _read_from_cache_point_cloud(self, pointcloud_file: AnyPath):
        return _cached_point_cloud(pointcloud_file=str(pointcloud_file), point_cloud_dim=self.point_cloud_dim)

    def _decode_point_cloud_size(self) -> int:
        pass

    def _decode_point_cloud_xyz(self) -> Optional[np.ndarray]:
        scene_folder = resolve_scene_folder(dataset_path=self._dataset_path, scene_name=self.scene_name)

        pointcloud_file = scene_folder / self._data_type_to_folder_name[PointCloud] / f"{self.frame_id}.bin"

        point_cloud_data = self._read_from_cache_point_cloud(pointcloud_file=pointcloud_file)
        xyz = point_cloud_data[:, :3]
        return xyz

    def _decode_point_cloud_rgb(self) -> Optional[np.ndarray]:
        pass

    def _decode_point_cloud_intensity(self) -> Optional[np.ndarray]:
        scene_folder = resolve_scene_folder(dataset_path=self._dataset_path, scene_name=self.scene_name)

        pointcloud_file = scene_folder / self._data_type_to_folder_name[PointCloud] / f"{self.frame_id}.bin"

        point_cloud_data = self._read_from_cache_point_cloud(pointcloud_file=pointcloud_file)
        intensity = point_cloud_data[:, -1].astype(np.uint8)  # reflectance
        return intensity.reshape(-1, 1)

    def _decode_point_cloud_elongation(self) -> Optional[np.ndarray]:
        pass

    def _decode_point_cloud_timestamp(self) -> Optional[np.ndarray]:
        pass

    def _decode_point_cloud_ring_index(self) -> Optional[np.ndarray]:
        pass

    def _decode_point_cloud_ray_type(self) -> Optional[np.ndarray]:
        pass
