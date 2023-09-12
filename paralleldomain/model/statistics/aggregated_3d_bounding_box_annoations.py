from collections import defaultdict
import pickle

import logging
from typing import Union

import numpy as np

from paralleldomain.model.radar_point_cloud import RadarPointCloud
from paralleldomain.model.scene import Scene
from paralleldomain.model.sensor import LidarSensorFrame, CameraSensorFrame, RadarSensorFrame
from paralleldomain.model.annotation import BoundingBoxes3D, AnnotationIdentifier
from paralleldomain.model.statistics.base import Statistic
from paralleldomain.model.statistics.constants import STATISTICS_REGISTRY
from paralleldomain.utilities.any_path import AnyPath

from paralleldomain.model.image import Image
from paralleldomain.model.point_cloud import PointCloud


@STATISTICS_REGISTRY.register_module()
class Aggregated3DBoundingBoxAnnotations(Statistic):
    """
    Class that handles parsing and loading/saving differents stats from 3D bounding box annotations:
        - x, y, z (int): box location (sensor coordinate for DGP data format) (meters)
        - height, width, length (float): box dimensions (meters)
        - yaw, pitch, roll (float): box orientation (radian)
        - volume (float): box volume (meters cube)
        - num_pts (int): number of Lidar/Radar points in box
        - range_to_ego (int): box forward location to ego (meters)
        - bev_polar_angle (float): bev location of the box compare to ego (radian)
    """

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def _reset(self):
        self._recorder = defaultdict(list)

    def _record_empty_frames(self, scene: Scene, sensor_frame: Union[LidarSensorFrame, CameraSensorFrame]):
        self._recorder["skipped_frames"].append(self.parse_sensor_frame_properties(scene, sensor_frame))

    @staticmethod
    def parse_3d_bounding_box_annotations(sensor_frame: Union[LidarSensorFrame, CameraSensorFrame]):
        bbox_3d_annotations = {}

        bbox_3d: BoundingBoxes3D = sensor_frame.get_annotations(annotation_type=BoundingBoxes3D)
        class_map = sensor_frame.class_maps[BoundingBoxes3D]

        boxes = []
        for box in bbox_3d.boxes:
            class_detail = class_map[box.class_id]
            yaw, pitch, roll = (sensor_frame.sensor_to_ego @ box.pose).quaternion.yaw_pitch_roll
            range_to_ego = np.linalg.norm((sensor_frame.sensor_to_ego @ box.pose).translation)
            # get polar angle to compute BEV orientation to ego
            forward_box = (sensor_frame.sensor_to_ego @ box.pose).translation / range_to_ego
            forward_sensor = np.array([1.0, 0.0, 0.0])
            polar_angle = np.arccos(np.clip(np.dot(forward_box, forward_sensor), -1.0, 1.0))
            if forward_box[1] < 0:
                polar_angle = 2 * np.pi - polar_angle
            box_dict = {
                "class_name": class_detail.name,
                "class_id": box.class_id,
                "instance_id": box.instance_id,
                "x": box.pose.translation[0],
                "y": box.pose.translation[1],
                "z": box.pose.translation[2],
                "height": box.height,
                "width": box.width,
                "length": box.length,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "volume": box.volume,
                "num_pts": box.num_points,
                "range_to_ego": range_to_ego,
                "bev_polar_angle": polar_angle,
                "attributes": box.attributes,
            }
            boxes.append(box_dict)

        if isinstance(sensor_frame, CameraSensorFrame):
            bbox_3d_annotations["img_height"] = sensor_frame.image.height
            bbox_3d_annotations["img_width"] = sensor_frame.image.width
            bbox_3d_annotations["img_filepath"] = sensor_frame.get_file_path(Image)
        elif isinstance(sensor_frame, LidarSensorFrame):
            bbox_3d_annotations["pointcloud_filepath"] = sensor_frame.get_file_path(PointCloud)

        bbox_3d_annotations["bboxes_3d"] = boxes

        return bbox_3d_annotations

    def _update(self, scene: Scene, sensor_frame: Union[LidarSensorFrame, CameraSensorFrame]):
        if BoundingBoxes3D in sensor_frame.available_annotation_types:
            bbox_3d_annotations = self.parse_3d_bounding_box_annotations(sensor_frame=sensor_frame)
            # add the properties_to_log to the annotations we want to save.
            bbox_3d_annotations.update(self.parse_sensor_frame_properties(scene, sensor_frame))
            self._recorder["bbox_3d_annotations"].append(bbox_3d_annotations)
        else:
            logging.warning("No 3D bounding box annotations available for current frame... Logging and skipping")
            self._record_empty_frames(scene=scene, sensor_frame=sensor_frame)

    def _load(self, file_path: Union[str, AnyPath]):
        file_path = AnyPath(file_path)
        with file_path.open("rb") as f:
            self._recorder = pickle.load(f)

    def _save(self, file_path: Union[str, AnyPath]):
        file_path = AnyPath(file_path)
        with file_path.open("wb") as f:
            pickle.dump(self._recorder, f, protocol=pickle.HIGHEST_PROTOCOL)
