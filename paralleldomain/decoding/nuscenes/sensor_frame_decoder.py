from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.nuscenes.common import NUSCENES_IMU_TO_INTERNAL_CS, NuScenesDataAccessMixin
from paralleldomain.decoding.sensor_frame_decoder import (
    CameraSensorFrameDecoder,
    LidarSensorFrameDecoder,
    SensorFrameDecoder,
)
from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationTypes, BoundingBox3D, BoundingBoxes3D
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.image import Image
from paralleldomain.model.point_cloud import PointCloud
from paralleldomain.model.sensor import SensorDataCopyTypes, SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image
from paralleldomain.utilities.transformation import Transformation

T = TypeVar("T")

"""
Nuscenes point data is in RFU, images in RDF. The extrinsic calibration transforms into FLU.
Please note that transforming 3d boxes into camera sensor frame is currently not implemented.
"""


class NuScenesSensorFrameDecoder(SensorFrameDecoder[datetime], NuScenesDataAccessMixin):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        split_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        frame_id: FrameId,
        settings: DecoderSettings,
        is_unordered_scene: bool,
        scene_decoder,
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        SensorFrameDecoder.__init__(
            self=self,
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            frame_id=frame_id,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        NuScenesDataAccessMixin.__init__(
            self=self, dataset_name=dataset_name, split_name=split_name, dataset_path=self._dataset_path
        )
        self.scene_token = self.nu_scene_name_to_scene_token[scene_name]

    def _decode_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        anno_identifiers = list()
        if self.split_name != "v1.0-test":
            if self.frame_id in self.nu_frame_id_to_available_anno_types:
                has_obj, has_surface = self.nu_frame_id_to_available_anno_types[self.frame_id]
                if has_obj:
                    anno_identifiers.append(AnnotationIdentifier(annotation_type=AnnotationTypes.BoundingBoxes3D))
                # if has_surface:
                #     anno_identifiers[AnnotationTypes.SemanticSegmentation2D] = "SemanticSegmentation2D"
        return anno_identifiers

    def _decode_metadata(self) -> Dict[str, Any]:
        return {}

    def _decode_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        return self.nu_class_maps

    def _decode_date_time(self) -> datetime:
        return self.get_datetime_with_frame_id(scene_token=self.scene_token, frame_id=self.frame_id)

    def _decode_extrinsic(self) -> SensorExtrinsic:
        sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
            scene_token=self.scene_token, frame_id=self.frame_id, sensor_name=self.sensor_name
        )
        data = self.nu_samples_data_by_token[sample_data_id]
        calib_sensor_token = data["calibrated_sensor_token"]
        calib_sensor = self.nu_calibrated_sensors[calib_sensor_token]
        trans = Transformation(quaternion=calib_sensor["rotation"], translation=calib_sensor["translation"])
        trans = NUSCENES_IMU_TO_INTERNAL_CS @ trans

        return trans

    def _decode_sensor_pose(self) -> SensorPose:
        sensor_to_ego = self.get_extrinsic()
        ego_to_world_mat = self.get_ego_pose(scene_token=self.scene_token, frame_id=self.frame_id)
        ego_to_world = EgoPose.from_transformation_matrix(ego_to_world_mat)
        sensor_to_world = ego_to_world @ sensor_to_ego
        return sensor_to_world

    def _decode_file_path(self, data_type: SensorDataCopyTypes) -> Optional[AnyPath]:
        if issubclass(data_type, Image):
            sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
                scene_token=self.scene_token, frame_id=self.frame_id, sensor_name=self.sensor_name
            )
            if sample_data_id is not None:
                data = self.nu_samples_data_by_token[sample_data_id]

                img_path = AnyPath(self._dataset_path) / data["filename"]
                return img_path
        elif issubclass(data_type, PointCloud):
            sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
                scene_token=self.scene_token, frame_id=self.frame_id, sensor_name=self.sensor_name
            )
            if sample_data_id is not None:
                lidar_filename = self.nu_samples_data_by_token[sample_data_id]["filename"]
                return self._dataset_path / lidar_filename

        return None

    def _decode_annotations(self, identifier: AnnotationIdentifier[T]) -> T:
        if issubclass(identifier.annotation_type, BoundingBoxes3D):
            boxes = self._decode_bounding_boxes_3d()
            return BoundingBoxes3D(boxes=boxes)
        # elif issubclass(annotation_type, InstanceSegmentation3D):
        #     instance_ids = self._decode_instance_segmentation_3d(sensor_name=sensor_name, frame_id=frame_id)
        #     return InstanceSegmentation3D(instance_ids=instance_ids)
        # elif issubclass(annotation_type, SemanticSegmentation3D):
        #     class_ids = self._decode_semantic_segmentation_3d(sensor_name=sensor_name, frame_id=frame_id)
        #     return SemanticSegmentation3D(class_ids=class_ids)
        else:
            raise NotImplementedError(f"{identifier} is not supported!")

    def _decode_bounding_boxes_3d(self) -> List[BoundingBox3D]:
        boxes = list()
        sensor_to_world = self._decode_sensor_pose()
        for ann in self.nu_sample_annotation[self.frame_id]:
            instance_token = ann["instance_token"]
            category_token = self.nu_instance[instance_token]["category_token"]
            attribute_tokens = ann["attribute_tokens"]
            category_name = self.nu_category[category_token]["name"]
            attributes = {self.nu_attribute[tk]["name"]: self.nu_attribute[tk] for tk in attribute_tokens}
            class_id = self.nu_name_to_index[category_name]
            instance_id = self.nu_instance_to_instance_id_map[instance_token]
            # nuScenes annotations are in global coordinate system
            box_to_world = Transformation(quaternion=ann["rotation"], translation=ann["translation"])
            box_to_sensor = (sensor_to_world.inverse) @ box_to_world

            boxes.append(
                BoundingBox3D(
                    pose=box_to_sensor,
                    length=ann["size"][1],  # x-axis
                    width=ann["size"][0],  # y-axis
                    height=ann["size"][2],  # z-axis
                    class_id=class_id,
                    instance_id=instance_id,
                    num_points=ann["num_lidar_pts"],
                    attributes=attributes,
                )
            )
        return boxes


# The LidarSensorFrameDecoder api exposes individual access to point fields, but they are stored in one single file.
# We don't want to download the file multiple times directly after each other, so we cache it.
# The LidarSensorFrameDecoder itself is cached, too, so we can't tie this cache to that instance.
# Also note that we don't set  the cache size to one so that the cache works with threaded downloads
@lru_cache(maxsize=16)
def load_point_cloud(path: str) -> np.ndarray:
    with AnyPath(path).open(mode="rb") as fp:
        raw_lidar = np.frombuffer(fp.read(), dtype=np.float32)
    return raw_lidar.reshape((-1, 5))


class NuScenesLidarSensorFrameDecoder(LidarSensorFrameDecoder[datetime], NuScenesSensorFrameDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        split_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        frame_id: FrameId,
        settings: DecoderSettings,
        is_unordered_scene: bool,
        scene_decoder,
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        LidarSensorFrameDecoder.__init__(
            self=self,
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            frame_id=frame_id,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        NuScenesSensorFrameDecoder.__init__(
            self=self,
            dataset_path=self._dataset_path,
            dataset_name=dataset_name,
            split_name=split_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            frame_id=frame_id,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )

    def _decode_point_cloud_data(self) -> Optional[np.ndarray]:
        """
        NuScenes .pcd.bin schema is [x,y,z,intensity,ring_index]
        """
        sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
            scene_token=self.scene_token, frame_id=self.frame_id, sensor_name=self.sensor_name
        )
        lidar_filename = self.nu_samples_data_by_token[sample_data_id]["filename"]
        full_anypath = self._dataset_path / lidar_filename
        return load_point_cloud(str(full_anypath))

    def _decode_point_cloud_size(self) -> int:
        data = self._decode_point_cloud_data()
        return len(data)

    def _decode_point_cloud_xyz(self) -> Optional[np.ndarray]:
        data = self._decode_point_cloud_data()
        return data[:, :3]

    def _decode_point_cloud_rgb(self) -> Optional[np.ndarray]:
        """
        NuScenes point cloud does not have RGB values, so returns np.ndarray of zeros .
        """
        cloud_size = self._decode_point_cloud_size()
        return np.zeros([cloud_size, 3])

    def _decode_point_cloud_intensity(self) -> Optional[np.ndarray]:
        data = self._decode_point_cloud_data()
        return data[:, 3].reshape(-1, 1)

    def _decode_point_cloud_elongation(self) -> Optional[np.ndarray]:
        return None

    def _decode_point_cloud_timestamp(self) -> Optional[np.ndarray]:
        return -1 * np.ones(self._decode_point_cloud_size())

    def _decode_point_cloud_ring_index(self) -> Optional[np.ndarray]:
        data = self._decode_point_cloud_data()
        return data[:, 4]

    def _decode_point_cloud_ray_type(self) -> Optional[np.ndarray]:
        return None


class NuScenesCameraSensorFrameDecoder(CameraSensorFrameDecoder[datetime], NuScenesSensorFrameDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        split_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        frame_id: FrameId,
        settings: DecoderSettings,
        is_unordered_scene: bool,
        scene_decoder,
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
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
        NuScenesSensorFrameDecoder.__init__(
            self=self,
            dataset_path=self._dataset_path,
            dataset_name=dataset_name,
            split_name=split_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            frame_id=frame_id,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )

    def _decode_intrinsic(self) -> SensorIntrinsic:
        sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
            scene_token=self.scene_token, frame_id=self.frame_id, sensor_name=self.sensor_name
        )
        data = self.nu_samples_data_by_token[sample_data_id]
        calib_sensor_token = data["calibrated_sensor_token"]
        calib_sensor = self.nu_calibrated_sensors[calib_sensor_token]
        return SensorIntrinsic(
            fx=calib_sensor["camera_intrinsic"][0][0],
            fy=calib_sensor["camera_intrinsic"][1][1],
            cx=calib_sensor["camera_intrinsic"][0][2],
            cy=calib_sensor["camera_intrinsic"][1][2],
            k1=0,
            k2=0,
            p1=0,
            p2=0,
            k3=0,
            k4=0,
            k5=0,
            k6=0,
        )

    def _decode_image_dimensions(self) -> Tuple[int, int, int]:
        sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
            scene_token=self.scene_token, frame_id=self.frame_id, sensor_name=self.sensor_name
        )
        data = self.nu_samples_data_by_token[sample_data_id]
        return int(data["height"]), int(data["width"]), 3

    def _decode_image_rgba(self) -> np.ndarray:
        sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
            scene_token=self.scene_token, frame_id=self.frame_id, sensor_name=self.sensor_name
        )
        data = self.nu_samples_data_by_token[sample_data_id]

        img_path = AnyPath(self._dataset_path) / data["filename"]
        image_data = read_image(path=img_path, convert_to_rgb=True)

        ones = np.ones((*image_data.shape[:2], 1), dtype=image_data.dtype)
        concatenated = np.concatenate([image_data, ones], axis=-1)
        return concatenated
