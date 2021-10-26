import base64
from datetime import datetime
from functools import lru_cache
from typing import Any, ByteString, Dict, List, Optional, Tuple, TypeVar, Union

import cv2
import numpy as np
from pyquaternion import Quaternion

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.nuscenes.common import NUSCENES_IMU_TO_INTERNAL_CS, NuScenesDataAccessMixin
from paralleldomain.decoding.sensor_frame_decoder import (
    SensorFrameDecoder,
    CameraSensorFrameDecoder,
    LidarSensorFrameDecoder,
    TDateTime,
)
from paralleldomain.model.annotation import (
    AnnotationPose,
    AnnotationType,
    AnnotationTypes,
    BoundingBox3D,
    BoundingBoxes3D,
    # InstanceSegmentation3D,
    # SemanticSegmentation3D,
)
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.point_cloud import PointCloud
from paralleldomain.model.sensor import SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import AnnotationIdentifier, FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image
from paralleldomain.utilities.transformation import Transformation

T = TypeVar("T")

### TO DO: Pull common functionality into this class, have camera and lidar inherit from it.
class NuScenesSensorFrameDecoder(SensorFrameDecoder[datetime], NuScenesDataAccessMixin):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        split_name: str,
        scene_name: SceneName,
        settings: DecoderSettings,
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        SensorFrameDecoder.__init__(self=self, dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        NuScenesDataAccessMixin.__init__(
            self=self, dataset_name=dataset_name, split_name=split_name, dataset_path=self._dataset_path
        )

    ### MHS: Can extend this function for lidar-semseg
    def _decode_available_annotation_types(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> Dict[AnnotationType, AnnotationIdentifier]:
        anno_types = dict()
        if self.split_name != "v1.0-test":
            if frame_id in self.nu_frame_id_to_available_anno_types:
                has_obj, has_surface = self.nu_frame_id_to_available_anno_types[frame_id]
                if has_obj:
                    anno_types[AnnotationTypes.BoundingBoxes3D] = "BoundingBoxes3D"
                # if has_surface:
                #     anno_types[AnnotationTypes.SemanticSegmentation2D] = "SemanticSegmentation2D"
        return anno_types

    def _decode_date_time(self, sensor_name: SensorName, frame_id: FrameId) -> datetime:
        sample = self.get_sample_with_frame_id(self.scene_name, frame_id)
        return datetime.fromtimestamp(int(sample["timestamp"]) / 1000000)

    def _decode_extrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorExtrinsic:

        sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
            scene_token=self.scene_name, frame_id=frame_id, sensor_name=sensor_name
        )
        data = self.nu_samples_data_by_token[sample_data_id]
        calib_sensor_token = data["calibrated_sensor_token"]
        calib_sensor = self.nu_calibrated_sensors[calib_sensor_token]
        trans = np.eye(4)
        trans[:3, :3] = Quaternion(calib_sensor["rotation"]).rotation_matrix
        trans[:3, 3] = np.array(calib_sensor["translation"])
        trans = NUSCENES_IMU_TO_INTERNAL_CS @ trans

        return SensorExtrinsic.from_transformation_matrix(trans)

    def _decode_sensor_pose(self, sensor_name: SensorName, frame_id: FrameId) -> SensorPose:
        sensor_to_ego = self.get_extrinsic(sensor_name=sensor_name, frame_id=frame_id)
        ego_to_world_mat = self.get_ego_pose(scene_token=self.scene_name, frame_id=frame_id)
        ego_to_world = EgoPose.from_transformation_matrix(ego_to_world_mat)
        sensor_to_world = ego_to_world @ sensor_to_ego
        return sensor_to_world

    def _decode_annotations(
        self, sensor_name: SensorName, frame_id: FrameId, identifier: AnnotationIdentifier, annotation_type: T
    ) -> T:
        if issubclass(annotation_type, BoundingBoxes3D):
            boxes = self._decode_bounding_boxes_3d(sensor_name=sensor_name, frame_id=frame_id)
            return BoundingBoxes3D(boxes=boxes)
        # elif issubclass(annotation_type, InstanceSegmentation3D):
        #     instance_ids = self._decode_instance_segmentation_3d(sensor_name=sensor_name, frame_id=frame_id)
        #     return InstanceSegmentation3D(instance_ids=instance_ids)
        # elif issubclass(annotation_type, SemanticSegmentation3D):
        #     class_ids = self._decode_semantic_segmentation_3d(sensor_name=sensor_name, frame_id=frame_id)
        #     return SemanticSegmentation3D(class_ids=class_ids)
        else:
            raise NotImplementedError(f"{annotation_type} is not supported!")

    def _decode_bounding_boxes_3d(self, sensor_name: SensorName, frame_id: FrameId) -> List[BoundingBox3D]:
        boxes = list()
        for i, ann in enumerate(self.nu_sample_annotation[frame_id], start=1):
            instance_token = ann["instance_token"]
            category_token = self.nu_instance[instance_token]["category_token"]
            attribute_tokens = ann["attribute_tokens"]
            category_name = self.nu_category[category_token]["name"]
            attributes = {self.nu_attribute[tk]["name"]: self.nu_attribute[tk] for tk in attribute_tokens}
            class_id = self.nu_name_to_index[category_name]
            # nuScenes annotations are in global coordinate system
            sensor_to_world = self._decode_sensor_pose(sensor_name=sensor_name, frame_id=frame_id)
            box_to_world = AnnotationPose(quaternion=Quaternion(ann["rotation"]), translation=ann["translation"])
            box_to_sensor = (sensor_to_world.inverse) @ box_to_world

            boxes.append(
                BoundingBox3D(
                    pose=box_to_sensor,
                    length=ann["size"][0],  # x-axis
                    width=ann["size"][1],  # y-axis
                    height=ann["size"][2],  # z-axis
                    class_id=class_id,
                    instance_id=i,
                    num_points=ann["num_lidar_pts"],
                    attributes=attributes,
                )
            )
        return boxes


class NuScenesLidarSensorFrameDecoder(LidarSensorFrameDecoder[datetime], NuScenesSensorFrameDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        split_name: str,
        scene_name: SceneName,
        settings: DecoderSettings,
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        LidarSensorFrameDecoder.__init__(self=self, dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        NuScenesSensorFrameDecoder.__init__(
            self=self,
            dataset_path=self._dataset_path,
            dataset_name=dataset_name,
            split_name=split_name,
            scene_name=scene_name,
            settings=DecoderSettings,
        )

    @lru_cache(maxsize=1)
    def _decode_point_cloud_data(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        """
        NuScenes .pcd.bin schema is [x,y,z,intensity,ring_index]
        """
        sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
            scene_token=self.scene_name, frame_id=frame_id, sensor_name=sensor_name
        )
        lidar_filename = self.nu_samples_data_by_token[sample_data_id]["filename"]
        raw_lidar = np.fromfile(str(self._dataset_path / lidar_filename), dtype=np.float32)
        return raw_lidar.reshape((-1, 5))

    def _decode_point_cloud_size(self, sensor_name: SensorName, frame_id: FrameId) -> int:
        data = self._decode_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id)
        return len(data)

    def _decode_point_cloud_xyz(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        data = self._decode_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id)
        return data[:, :3]

    def _decode_point_cloud_rgb(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        """
        NuScenes point cloud does not have RGB values, so returns np.ndarray of zeros .
        """
        cloud_size = self._decode_point_cloud_size(sensor_name=sensor_name, frame_id=frame_id)
        return np.zeros([cloud_size, 3])

    def _decode_point_cloud_intensity(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        data = self._decode_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id)
        return data[:, 3].reshape(-1, 1)

    def _decode_point_cloud_timestamp(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        pass

    def _decode_point_cloud_ring_index(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        data = self._decode_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id)
        return data[:, 4]


class NuScenesCameraSensorFrameDecoder(CameraSensorFrameDecoder[datetime], NuScenesSensorFrameDecoder):
    def __init__(
        self,
        dataset_path: Union[str, AnyPath],
        dataset_name: str,
        split_name: str,
        scene_name: SceneName,
        settings: DecoderSettings,
    ):
        self._dataset_path: AnyPath = AnyPath(dataset_path)
        CameraSensorFrameDecoder.__init__(
            self=self, dataset_name=dataset_name, scene_name=scene_name, settings=settings
        )
        NuScenesSensorFrameDecoder.__init__(
            self=self,
            dataset_path=self._dataset_path,
            dataset_name=dataset_name,
            split_name=split_name,
            scene_name=scene_name,
            settings=DecoderSettings,
        )

    ### MHS: nuscenes does not have camera_distortion, so will need to update this function.
    def _decode_intrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorIntrinsic:
        sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
            scene_token=self.scene_name, frame_id=frame_id, sensor_name=sensor_name
        )
        data = self.nu_samples_data_by_token[sample_data_id]
        calib_sensor_token = data["calibrated_sensor_token"]
        calib_sensor = self.nu_calibrated_sensors[calib_sensor_token]
        return SensorIntrinsic(
            fx=calib_sensor["camera_intrinsic"][0][0],
            fy=calib_sensor["camera_intrinsic"][1][1],
            cx=calib_sensor["camera_intrinsic"][0][2],
            cy=calib_sensor["camera_intrinsic"][1][2],
        )

    def _decode_image_dimensions(self, sensor_name: SensorName, frame_id: FrameId) -> Tuple[int, int, int]:
        sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
            scene_token=self.scene_name, frame_id=frame_id, sensor_name=sensor_name
        )
        data = self.nu_samples_data_by_token[sample_data_id]
        return int(data["height"]), int(data["width"]), 3

    def _decode_image_rgba(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        sample_data_id = self.get_sample_data_id_frame_id_and_sensor_name(
            scene_token=self.scene_name, frame_id=frame_id, sensor_name=sensor_name
        )
        data = self.nu_samples_data_by_token[sample_data_id]

        img_path = AnyPath(self._dataset_path) / data["filename"]
        image_data = read_image(path=img_path, convert_to_rgb=True)

        ones = np.ones((*image_data.shape[:2], 1), dtype=image_data.dtype)
        concatenated = np.concatenate([image_data, ones], axis=-1)
        return concatenated


### MHS: these helper functions may or may not be useful in nuScenes.
def mask_decode(mask: dict) -> np.ndarray:
    """
    Decode the mask from base64 string to binary string, then decoded to get a mask.
    :param mask: The mask dictionary with fields `size` and `counts`.
    :return: A numpy array representing the binary mask for this class.
    """
    new_mask = mask.copy()
    new_mask["counts"] = base64.b64decode(mask["counts"])
    mask_rle = _rle_from_leb_string(leb_bytes=new_mask["counts"])
    return _rle_to_mask(mask_rle=mask_rle, shape=new_mask["size"])


def _rle_to_mask(mask_rle: List[int], shape: Tuple[int, int]) -> np.ndarray:
    flat_size = shape[0] * shape[1]
    mask = np.zeros(flat_size, dtype=int)
    rle_np = np.array([a for a in mask_rle], dtype=int)
    rle_np_cum = np.cumsum(rle_np)
    one_slice = np.array(list(zip(rle_np_cum[::2], rle_np_cum[1::2])))
    for start, stop in one_slice:
        mask[start:stop] = 1
    return mask.reshape(shape, order="F")


def _rle_from_leb_string(leb_bytes: ByteString) -> List[int]:
    """
    Coco custom LEB128 decoding. See https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c#L205
    """
    m = 0
    p = 0
    cnts = list()
    max_st_len = len(leb_bytes)
    while p < max_st_len:
        x = 0
        k = 0
        more = True
        while more:
            c = leb_bytes[p] - 48
            x |= (c & 0x1F) << 5 * k
            more = c & 0x20 == 32
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << 5 * k
        if m > 2:
            x += cnts[m - 2]
        cnts.append(x)
        m += 1
    return cnts
