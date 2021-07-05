import json
from json import JSONDecodeError
from typing import BinaryIO, Dict, Optional, Type, TypeVar, cast

import imageio
import numpy as np
from pyquaternion import Quaternion

from paralleldomain.decoding.dgp.constants import ANNOTATION_TYPE_MAP, DGP_TO_INTERNAL_CS, TransformType
from paralleldomain.decoding.dgp.dtos import (
    AnnotationsBoundingBox2DDTO,
    AnnotationsBoundingBox3DDTO,
    CalibrationDTO,
    CalibrationExtrinsicDTO,
    CalibrationIntrinsicDTO,
    PoseDTO,
    SceneDataDatum,
)
from paralleldomain.model.annotation import (
    AnnotationPose,
    AnnotationType,
    BoundingBox2D,
    BoundingBox3D,
    BoundingBoxes2D,
    BoundingBoxes3D,
    InstanceSegmentation2D,
    InstanceSegmentation3D,
    OpticalFlow,
    SemanticSegmentation2D,
    SemanticSegmentation3D,
)
from paralleldomain.model.class_mapping import ClassIdMap, ClassMap
from paralleldomain.model.sensor import ImageData, PointCloudData, SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.transformation import Transformation
from paralleldomain.model.type_aliases import AnnotationIdentifier, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath

T = TypeVar("T")


class DGPFrameLazyLoader:
    def __init__(
        self,
        unique_cache_key_prefix: str,
        dataset_path: AnyPath,
        scene_name: SceneName,
        sensor_name: SensorName,
        class_map: ClassMap,
        calibration_key: str,
        datum: SceneDataDatum,
        custom_reference_to_box_bottom: Transformation,
        custom_id_map: Optional[ClassIdMap] = None,
    ):
        self.custom_id_map = custom_id_map
        self._dataset_path = dataset_path
        self.class_map = class_map
        self.datum = datum
        self._unique_cache_key_prefix = unique_cache_key_prefix
        self.sensor_name = sensor_name
        self.scene_name = scene_name
        self.calibration_key = calibration_key
        self._custom_reference_to_box_bottom = custom_reference_to_box_bottom

    def load_intrinsic(self) -> SensorIntrinsic:
        dto = self._decode_intrinsic_calibration(
            scene_name=self.scene_name,
            calibration_key=self.calibration_key,
            sensor_name=self.sensor_name,
        )

        if dto.fisheye is True or dto.fisheye == 1:
            camera_model = "fisheye"
        elif dto.fisheye is False or dto.fisheye == 0:
            camera_model = "brown_conrady"
        elif dto.fisheye > 1:
            camera_model = f"custom_{dto.fisheye}"

        return SensorIntrinsic(
            cx=dto.cx,
            cy=dto.cy,
            fx=dto.fx,
            fy=dto.fy,
            k1=dto.k1,
            k2=dto.k2,
            p1=dto.p1,
            p2=dto.p2,
            k3=dto.k3,
            k4=dto.k4,
            k5=dto.k5,
            k6=dto.k6,
            skew=dto.skew,
            fov=dto.fov,
            camera_model=camera_model,
        )

    def load_extrinsic(self) -> SensorExtrinsic:
        dto = self._decode_extrinsic_calibration(
            scene_name=self.scene_name,
            calibration_key=self.calibration_key,
            sensor_name=self.sensor_name,
        )
        sensor_to_box_bottom = _pose_dto_to_transformation(dto=dto, transformation_type=SensorExtrinsic)
        sensor_to_custom_reference = (
            self._custom_reference_to_box_bottom.inverse @ sensor_to_box_bottom
        )  # from center-bottom to center rear-axle
        return sensor_to_custom_reference

    def load_point_cloud(self) -> Optional[PointCloudData]:
        if self.datum.point_cloud:
            unique_cache_key = f"{self._unique_cache_key_prefix}-point_cloud"
            return PointCloudData(
                unique_cache_key=unique_cache_key,
                point_format=self.datum.point_cloud.point_format,
                load_data=lambda: self._decode_point_cloud(
                    scene_name=self.scene_name, cloud_identifier=self.datum.point_cloud.filename
                ),
            )
        return None

    def load_image(self) -> Optional[ImageData]:
        if self.datum.image:
            unique_cache_key = f"{self._unique_cache_key_prefix}-image"
            return ImageData(
                load_data_rgba=lambda: self._decode_image_rgb(
                    scene_name=self.scene_name,
                    cloud_identifier=self.datum.image.filename,
                ),
                unique_cache_key=unique_cache_key,
            )

    def load_sensor_pose(self) -> SensorPose:
        if self.datum.image:
            return _pose_dto_to_transformation(dto=self.datum.image.pose, transformation_type=SensorPose)
        else:
            return _pose_dto_to_transformation(dto=self.datum.point_cloud.pose, transformation_type=SensorPose)

    def load_annotations(self, identifier: AnnotationIdentifier, annotation_type: Type[T]) -> T:
        if issubclass(annotation_type, BoundingBoxes3D):
            dto = self._decode_bounding_boxes_3d(scene_name=self.scene_name, annotation_identifier=identifier)

            box_list = []
            for box_dto in dto.annotations:
                pose = _pose_dto_to_transformation(dto=box_dto.box.pose, transformation_type=AnnotationPose)

                # Add Truncation, Occlusion
                attr_parsed = {"occlusion": box_dto.box.occlusion, "truncation": box_dto.box.truncation}
                # Read + parse other attributes
                for k, v in box_dto.attributes.items():
                    try:
                        attr_parsed[k] = json.loads(v)
                    except JSONDecodeError:
                        attr_parsed[k] = v
                class_id = box_dto.class_id
                if self.custom_id_map is not None:
                    class_id = self.custom_id_map[class_id]

                box = BoundingBox3D(
                    pose=pose,
                    width=box_dto.box.width,
                    length=box_dto.box.length,
                    height=box_dto.box.height,
                    class_id=class_id,
                    instance_id=box_dto.instance_id,
                    num_points=box_dto.num_points,
                    attributes=attr_parsed,
                )
                box_list.append(box)

            return BoundingBoxes3D(boxes=box_list, class_map=self.class_map)
        elif issubclass(annotation_type, BoundingBoxes2D):
            dto = self._decode_bounding_boxes_2d(scene_name=self.scene_name, annotation_identifier=identifier)

            box_list = []
            for box_dto in dto.annotations:

                attr_parsed = {"iscrowd": box_dto.iscrowd}
                for k, v in box_dto.attributes.items():
                    try:
                        attr_parsed[k] = json.loads(v)
                    except JSONDecodeError:
                        attr_parsed[k] = v

                class_id = box_dto.class_id
                if self.custom_id_map is not None:
                    class_id = self.custom_id_map[class_id]

                box = BoundingBox2D(
                    x=box_dto.box.x,
                    y=box_dto.box.y,
                    width=box_dto.box.w,
                    height=box_dto.box.h,
                    class_id=class_id,
                    instance_id=box_dto.instance_id,
                    attributes=attr_parsed,
                )
                box_list.append(box)

            return BoundingBoxes2D(boxes=box_list, class_map=self.class_map)
        elif issubclass(annotation_type, SemanticSegmentation3D):
            segmentation_mask = self._decode_semantic_segmentation_3d(
                scene_name=self.scene_name, annotation_identifier=identifier
            )
            if self.custom_id_map is not None:
                segmentation_mask = self.custom_id_map[segmentation_mask]
            return SemanticSegmentation3D(mask=segmentation_mask, class_map=self.class_map)
        elif issubclass(annotation_type, InstanceSegmentation3D):
            instance_mask = self._decode_instance_segmentation_3d(
                scene_name=self.scene_name, annotation_identifier=identifier
            )
            return InstanceSegmentation3D(mask=instance_mask)
        elif issubclass(annotation_type, SemanticSegmentation2D):
            class_ids = self._decode_semantic_segmentation_2d(
                scene_name=self.scene_name, annotation_identifier=identifier
            )
            if self.custom_id_map is not None:
                class_ids = self.custom_id_map[class_ids]
            return SemanticSegmentation2D(class_ids=class_ids, class_map=self.class_map)
        elif issubclass(annotation_type, OpticalFlow):
            flow_vectors = self._decode_optical_flow(scene_name=self.scene_name, annotation_identifier=identifier)
            return OpticalFlow(vectors=flow_vectors)
        elif issubclass(annotation_type, InstanceSegmentation2D):
            instance_ids = self._decode_instance_segmentation_2d(
                scene_name=self.scene_name, annotation_identifier=identifier
            )
            return InstanceSegmentation2D(instance_ids=instance_ids)

    def load_available_annotation_types(
        self,
    ) -> Dict[AnnotationType, AnnotationIdentifier]:
        if self.datum.image:
            type_to_path = self.datum.image.annotations
        else:
            type_to_path = self.datum.point_cloud.annotations
        return {ANNOTATION_TYPE_MAP[k]: v for k, v in type_to_path.items()}

    def _decode_point_cloud(self, scene_name: str, cloud_identifier: str) -> np.ndarray:
        cloud_path = self._dataset_path / scene_name / cloud_identifier
        with cloud_path.open(mode="rb") as cloud_binary:
            npz_data = np.load(cast(BinaryIO, cloud_binary))
            pc_data = npz_data.f.data
            return np.column_stack([pc_data[c] for c in pc_data.dtype.names])

    def _decode_calibration(self, scene_name: str, calibration_key: str) -> CalibrationDTO:
        calibration_path = self._dataset_path / scene_name / "calibration" / f"{calibration_key}.json"
        with calibration_path.open("r") as f:
            cal_dict = json.load(f)
            return CalibrationDTO.from_dict(cal_dict)

    def _decode_extrinsic_calibration(
        self, scene_name: str, calibration_key: str, sensor_name: SensorName
    ) -> CalibrationExtrinsicDTO:
        calibration_dto = self._decode_calibration(scene_name=scene_name, calibration_key=calibration_key)
        index = calibration_dto.names.index(sensor_name)
        return calibration_dto.extrinsics[index]

    def _decode_intrinsic_calibration(
        self, scene_name: str, calibration_key: str, sensor_name: SensorName
    ) -> CalibrationIntrinsicDTO:
        calibration_dto = self._decode_calibration(scene_name=scene_name, calibration_key=calibration_key)
        index = calibration_dto.names.index(sensor_name)
        return calibration_dto.intrinsics[index]

    def _decode_image_rgb(self, scene_name: str, cloud_identifier: str) -> np.ndarray:
        cloud_path = self._dataset_path / scene_name / cloud_identifier
        with cloud_path.open(mode="rb") as cloud_binary:
            image_data = np.asarray(imageio.imread(cast(BinaryIO, cloud_binary), format="png"))
            return image_data

    def _decode_bounding_boxes_3d(self, scene_name: str, annotation_identifier: str) -> AnnotationsBoundingBox3DDTO:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        with annotation_path.open("r") as f:
            return AnnotationsBoundingBox3DDTO.from_dict(json.load(f))

    def _decode_bounding_boxes_2d(self, scene_name: str, annotation_identifier: str) -> AnnotationsBoundingBox2DDTO:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        with annotation_path.open("r") as f:
            return AnnotationsBoundingBox2DDTO.from_dict(json.load(f))

    def _decode_semantic_segmentation_3d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        with annotation_path.open(mode="rb") as cloud_binary:
            npz_data = np.load(cast(BinaryIO, cloud_binary))
            return npz_data.f.segmentation

    def _decode_instance_segmentation_3d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        with annotation_path.open(mode="rb") as cloud_binary:
            npz_data = np.load(cast(BinaryIO, cloud_binary))
            return npz_data.f.instance

    def _decode_semantic_segmentation_2d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        with annotation_path.open(mode="rb") as cloud_binary:
            image_data = np.asarray(imageio.imread(cast(BinaryIO, cloud_binary), format="png")).astype(np.int)

            class_ids = (image_data[..., 2:3] << 16) + (image_data[..., 1:2] << 8) + image_data[..., 0:1]
            return class_ids

    def _decode_optical_flow(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        with annotation_path.open(mode="rb") as cloud_binary:
            image_data = np.asarray(imageio.imread(cast(BinaryIO, cloud_binary), format="png")).astype(np.int)
            vectors = (image_data[..., [0, 2]] << 8) + image_data[..., [1, 3]]

            return vectors

    def _decode_instance_segmentation_2d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        with annotation_path.open(mode="rb") as cloud_binary:
            image_data = np.asarray(imageio.imread(cast(BinaryIO, cloud_binary), format="png")).astype(np.int)

            instance_ids = (image_data[..., 2:3] << 16) + (image_data[..., 1:2] << 8) + image_data[..., 0:1]
            return instance_ids


def _pose_dto_to_transformation(dto: PoseDTO, transformation_type: Type[TransformType]) -> TransformType:
    transform = transformation_type(
        quaternion=Quaternion(dto.rotation.qw, dto.rotation.qx, dto.rotation.qy, dto.rotation.qz),
        translation=np.array([dto.translation.x, dto.translation.y, dto.translation.z]),
    )
    return transformation_type.from_transformation_matrix(DGP_TO_INTERNAL_CS @ transform.transformation_matrix)
