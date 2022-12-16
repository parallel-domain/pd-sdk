import abc
from datetime import datetime
from enum import Enum
from functools import lru_cache
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import numpy as np
import ujson
from pyquaternion import Quaternion

from paralleldomain.common.dgp.v0.constants import ANNOTATION_TYPE_MAP, DGP_TO_INTERNAL_CS, TransformType
from paralleldomain.common.dgp.v0.dtos import (
    AnnotationsBoundingBox2DDTO,
    AnnotationsBoundingBox3DDTO,
    CalibrationDTO,
    CalibrationExtrinsicDTO,
    CalibrationIntrinsicDTO,
    OntologyFileDTO,
    PoseDTO,
    SceneDataDatum,
    SceneDataDatumPointCloud,
    SceneDataDTO,
    SceneDTO,
    SceneSampleDTO,
    scene_data_to_date_time,
)
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.dgp.common import decode_class_maps
from paralleldomain.decoding.sensor_frame_decoder import (
    CameraSensorFrameDecoder,
    LidarSensorFrameDecoder,
    SensorFrameDecoder,
)
from paralleldomain.model.annotation import (
    Albedo2D,
    Annotation,
    AnnotationType,
    BoundingBox2D,
    BoundingBox3D,
    BoundingBoxes2D,
    BoundingBoxes3D,
    Depth,
    InstanceSegmentation2D,
    InstanceSegmentation3D,
    MaterialProperties2D,
    OpticalFlow,
    PointCache,
    PointCacheComponent,
    PointCaches,
    SceneFlow,
    SemanticSegmentation2D,
    SemanticSegmentation3D,
    SurfaceNormals2D,
    SurfaceNormals3D,
)
from paralleldomain.model.annotation.material_properties_3d import MaterialProperties3D
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.image import Image
from paralleldomain.model.point_cloud import PointCloud
from paralleldomain.model.sensor import CameraModel, SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import AnnotationIdentifier, FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image, read_json, read_json_str, read_npz, read_png
from paralleldomain.utilities.transformation import Transformation

T = TypeVar("T")
F = TypeVar("F", Image, PointCloud, Annotation)


class DGPSensorFrameDecoder(SensorFrameDecoder[datetime], metaclass=abc.ABCMeta):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        scene_samples: Dict[FrameId, SceneSampleDTO],
        scene_data: List[SceneDataDTO],
        ontologies: Dict[str, str],
        custom_reference_to_box_bottom: Transformation,
        settings: DecoderSettings,
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self._dataset_path = dataset_path
        self.scene_samples = scene_samples
        self.scene_data = scene_data
        self._ontologies = ontologies
        self._custom_reference_to_box_bottom = custom_reference_to_box_bottom
        self._data_by_sensor_name = lru_cache(maxsize=1)(self._data_by_sensor_name)
        self._get_sensor_frame_data = lru_cache(maxsize=1)(self._get_sensor_frame_data)
        self._get_3d_boxes_for_point_cache = lru_cache(maxsize=5)(self._get_3d_boxes_for_point_cache)

    def _data_by_sensor_name(self, sensor_name: SensorName) -> Dict[str, SceneDataDTO]:
        return {d.key: d for d in self.scene_data if d.id.name == sensor_name}

    def _get_current_frame_sample(self, frame_id: FrameId) -> SceneSampleDTO:
        return self.scene_samples[frame_id]

    def _get_sensor_frame_data(self, frame_id: FrameId, sensor_name: SensorName) -> SceneDataDTO:
        sample = self._get_current_frame_sample(frame_id=frame_id)
        # all sensor data of the sensor
        sensor_data = self._data_by_sensor_name(sensor_name=sensor_name)
        # read ontology -> Dict[str, ClassMap]
        # datum ley of sample that references the given sensor name
        datum_key = next(iter([key for key in sample.datum_keys if key in sensor_data]))
        scene_data = sensor_data[datum_key]
        return scene_data

    def _get_sensor_frame_data_datum(self, frame_id: FrameId, sensor_name: SensorName) -> SceneDataDatum:
        scene_data = self._get_sensor_frame_data(frame_id=frame_id, sensor_name=sensor_name)
        return scene_data.datum

    def _decode_class_maps(self) -> Dict[AnnotationType, ClassMap]:
        return decode_class_maps(
            ontologies=self._ontologies, dataset_path=self._dataset_path, scene_name=self.scene_name
        )

    def _decode_date_time(self, sensor_name: SensorName, frame_id: FrameId) -> datetime:
        data = self._get_sensor_frame_data(frame_id=frame_id, sensor_name=sensor_name)
        return scene_data_to_date_time(data=data)

    def _decode_extrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorExtrinsic:
        sample = self._get_current_frame_sample(frame_id=frame_id)
        dto = self._decode_extrinsic_calibration(
            scene_name=self.scene_name,
            calibration_key=sample.calibration_key,
            sensor_name=sensor_name,
        )
        sensor_to_box_bottom = _pose_dto_to_transformation(dto=dto, transformation_type=SensorExtrinsic)
        sensor_to_custom_reference = (
            self._custom_reference_to_box_bottom.inverse @ sensor_to_box_bottom
        )  # from center-bottom to center rear-axle
        return sensor_to_custom_reference

    def _decode_sensor_pose(self, sensor_name: SensorName, frame_id: FrameId) -> SensorPose:
        datum = self._get_sensor_frame_data_datum(frame_id=frame_id, sensor_name=sensor_name)
        if datum.image:
            return _pose_dto_to_transformation(dto=datum.image.pose, transformation_type=SensorPose)
        else:
            return _pose_dto_to_transformation(dto=datum.point_cloud.pose, transformation_type=SensorPose)

    def _decode_annotations(
        self, sensor_name: SensorName, frame_id: FrameId, identifier: AnnotationIdentifier, annotation_type: T
    ) -> T:
        if issubclass(annotation_type, BoundingBoxes3D):
            dto = self._decode_bounding_boxes_3d(scene_name=self.scene_name, annotation_identifier=identifier)

            box_list = []
            for box_dto in dto.annotations:
                pose = _pose_dto_to_transformation(dto=box_dto.box.pose, transformation_type=Transformation)

                # Add Truncation, Occlusion
                attr_parsed = {"occlusion": box_dto.box.occlusion, "truncation": box_dto.box.truncation}
                # Read + parse other attributes
                for k, v in box_dto.attributes.items():
                    try:
                        attr_parsed[k] = ujson.loads(v)
                    except (ValueError, JSONDecodeError):
                        attr_parsed[k] = v
                class_id = box_dto.class_id

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

            return BoundingBoxes3D(boxes=box_list)
        elif issubclass(annotation_type, BoundingBoxes2D):
            dto = self._decode_bounding_boxes_2d(scene_name=self.scene_name, annotation_identifier=identifier)

            box_list = []
            for box_dto in dto.annotations:

                attr_parsed = {"iscrowd": box_dto.iscrowd}
                for k, v in box_dto.attributes.items():
                    try:
                        attr_parsed[k] = ujson.loads(v)
                    except (ValueError, JSONDecodeError, TypeError):
                        attr_parsed[k] = v

                class_id = box_dto.class_id

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

            return BoundingBoxes2D(boxes=box_list)
        elif issubclass(annotation_type, SemanticSegmentation3D):
            segmentation_mask = self._decode_semantic_segmentation_3d(
                scene_name=self.scene_name, annotation_identifier=identifier
            )
            return SemanticSegmentation3D(class_ids=segmentation_mask)
        elif issubclass(annotation_type, InstanceSegmentation3D):
            instance_mask = self._decode_instance_segmentation_3d(
                scene_name=self.scene_name, annotation_identifier=identifier
            )
            return InstanceSegmentation3D(instance_ids=instance_mask)
        elif issubclass(annotation_type, SemanticSegmentation2D):
            class_ids = self._decode_semantic_segmentation_2d(
                scene_name=self.scene_name, annotation_identifier=identifier
            )
            return SemanticSegmentation2D(class_ids=class_ids)
        elif issubclass(annotation_type, InstanceSegmentation2D):
            instance_ids = self._decode_instance_segmentation_2d(
                scene_name=self.scene_name, annotation_identifier=identifier
            )
            return InstanceSegmentation2D(instance_ids=instance_ids)
        elif issubclass(annotation_type, OpticalFlow):
            forward_vectors, backward_vectors = self._decode_optical_flow(
                scene_name=self.scene_name, annotation_identifier=identifier
            )
            return OpticalFlow(vectors=forward_vectors, backward_vectors=backward_vectors)
        elif issubclass(annotation_type, Depth):
            depth_mask = self._decode_depth(scene_name=self.scene_name, annotation_identifier=identifier)
            return Depth(depth=depth_mask)
        elif issubclass(annotation_type, SceneFlow):
            forward_vectors, backward_vectors = self._decode_scene_flow(
                scene_name=self.scene_name, annotation_identifier=identifier
            )
            return SceneFlow(vectors=forward_vectors, backward_vectors=backward_vectors)
        elif issubclass(annotation_type, SurfaceNormals3D):
            normals = self._decode_surface_normals_3d(scene_name=self.scene_name, annotation_identifier=identifier)
            return SurfaceNormals3D(normals=normals)
        elif issubclass(annotation_type, SurfaceNormals2D):
            normals = self._decode_surface_normals_2d(scene_name=self.scene_name, annotation_identifier=identifier)
            return SurfaceNormals2D(normals=normals)
        elif issubclass(annotation_type, PointCaches):
            caches = self._decode_point_caches(scene_name=self.scene_name, annotation_identifier=identifier)
            return PointCaches(caches=caches)
        elif issubclass(annotation_type, MaterialProperties3D):
            material_ids, roughness, metallic, specular, emissive, opacity, flags = self._decode_material_properties_3d(
                scene_name=self.scene_name, annotation_identifier=identifier
            )
            return MaterialProperties3D(
                material_ids=material_ids,
                roughness=roughness,
                metallic=metallic,
                specular=specular,
                emissive=emissive,
                opacity=opacity,
                flags=flags,
            )
        elif issubclass(annotation_type, Albedo2D):
            color = self._decode_albedo_2d(scene_name=self.scene_name, annotation_identifier=identifier)
            return Albedo2D(color=color)
        elif issubclass(annotation_type, MaterialProperties2D):
            roughness = self._decode_material_properties_2d(
                scene_name=self.scene_name, annotation_identifier=identifier
            )
            return MaterialProperties2D(roughness=roughness)
        else:
            raise NotImplementedError(f"{annotation_type} is not implemented yet in this decoder!")

    def _decode_available_annotation_types(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> Dict[AnnotationType, AnnotationIdentifier]:
        datum = self._get_sensor_frame_data_datum(frame_id=frame_id, sensor_name=sensor_name)
        if datum.image:
            type_to_path = datum.image.annotations
        else:
            type_to_path = datum.point_cloud.annotations

        available_annotation_types = {ANNOTATION_TYPE_MAP[k]: v for k, v in type_to_path.items()}

        point_cache_folder = self._dataset_path / self.scene_name / "point_cache"
        if BoundingBoxes3D in available_annotation_types and point_cache_folder.exists():
            available_annotation_types[PointCaches] = "$".join(
                [available_annotation_types[BoundingBoxes3D], sensor_name, frame_id]
            )

        return available_annotation_types

    def _decode_metadata(self, sensor_name: SensorName, frame_id: FrameId) -> Dict[str, Any]:
        datum = self._get_sensor_frame_data_datum(frame_id=frame_id, sensor_name=sensor_name)
        if datum.image:
            return datum.image.metadata
        else:
            return datum.point_cloud.metadata

    # ---------------------------------

    def _decode_calibration(self, scene_name: str, calibration_key: str) -> CalibrationDTO:
        calibration_path = self._dataset_path / scene_name / "calibration" / f"{calibration_key}.json"
        cal_dict = read_json(path=calibration_path)
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

    def _get_3d_boxes_for_point_cache(
        self, bbox_annotation_identifier: str, sensor_name: SensorName, frame_id: FrameId
    ):
        boxes = self.get_annotations(
            sensor_name=sensor_name,
            frame_id=frame_id,
            identifier=bbox_annotation_identifier,
            annotation_type=BoundingBoxes3D,
        )
        return boxes

    def _decode_point_caches(self, scene_name: str, annotation_identifier: str) -> List[PointCache]:
        bbox_annotation_identifier, sensor_name, frame_id = annotation_identifier.split("$")
        point_cache_folder = self._dataset_path / scene_name / "point_cache"
        boxes = self._get_3d_boxes_for_point_cache(
            sensor_name=sensor_name,
            frame_id=frame_id,
            bbox_annotation_identifier=bbox_annotation_identifier,
        )
        caches = []

        for box in boxes.boxes:
            if "point_cache" in box.attributes:
                component_dicts = box.attributes["point_cache"]
                components = []
                for component_dict in component_dicts:
                    pose = _pose_dto_to_transformation(
                        dto=PoseDTO.from_dict(component_dict["pose"]), transformation_type=Transformation
                    )
                    component = PointCacheComponent(
                        component_name=component_dict["component"],
                        points_decoder=DGPPointCachePointsDecoder(
                            sha=component_dict["sha"],
                            size=component_dict["size"],
                            cache_folder=point_cache_folder,
                            pose=pose,
                            parent_pose=box.pose,
                        ),
                    )
                    components.append(component)

                cache = PointCache(instance_id=box.instance_id, components=components)
                caches.append(cache)

        return caches

    def _decode_bounding_boxes_3d(self, scene_name: str, annotation_identifier: str) -> AnnotationsBoundingBox3DDTO:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        bb_dict = read_json(path=annotation_path)
        return AnnotationsBoundingBox3DDTO.from_dict(bb_dict)

    def _decode_bounding_boxes_2d(self, scene_name: str, annotation_identifier: str) -> AnnotationsBoundingBox2DDTO:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        bb_dict = read_json(path=annotation_path)
        return AnnotationsBoundingBox2DDTO.from_dict(bb_dict)

    def _decode_semantic_segmentation_3d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        segmentation_data = read_npz(path=annotation_path, files="segmentation")

        return segmentation_data

    def _decode_instance_segmentation_3d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        instance_data = read_npz(path=annotation_path, files="instance")

        return instance_data

    def _decode_semantic_segmentation_2d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        image_data = read_image(path=annotation_path)
        image_data = image_data.astype(int)
        class_ids = (image_data[..., 2:3] << 16) + (image_data[..., 1:2] << 8) + image_data[..., 0:1]

        return class_ids

    def _decode_optical_flow(
        self, scene_name: str, annotation_identifier: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if "back_motion_vectors_2d" in annotation_identifier:
            forward_annotation_path = (
                self._dataset_path
                / scene_name
                / annotation_identifier.replace("back_motion_vectors_2d", "motion_vectors_2d")
            )
            back_annotation_path = self._dataset_path / scene_name / annotation_identifier
        else:
            forward_annotation_path = self._dataset_path / scene_name / annotation_identifier
            back_annotation_path = (
                self._dataset_path
                / scene_name
                / annotation_identifier.replace("motion_vectors_2d", "back_motion_vectors_2d")
            )

        def read_flow_file(path: AnyPath) -> np.ndarray:
            image_data = read_image(path=path)
            image_data = image_data.astype(int)
            height, width = image_data.shape[0:2]
            vectors = image_data[..., [0, 2]] + (image_data[..., [1, 3]] << 8)
            vectors = (vectors / 65535.0 - 0.5) * [width, height] * 2
            return vectors

        forward_vectors = read_flow_file(path=forward_annotation_path) if forward_annotation_path.exists() else None
        backward_vectors = read_flow_file(path=back_annotation_path) if back_annotation_path.exists() else None

        return forward_vectors, backward_vectors

    def _decode_albedo_2d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        color = read_image(path=annotation_path)[..., :3]
        return color

    def _decode_material_properties_2d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        roughness = read_png(path=annotation_path)[..., :3]
        return roughness

    def _decode_depth(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        depth_data = read_npz(path=annotation_path, files="data")

        return np.expand_dims(depth_data, axis=-1)

    def _decode_instance_segmentation_2d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        image_data = read_image(path=annotation_path)
        image_data = image_data.astype(int)
        instance_ids = (image_data[..., 2:3] << 16) + (image_data[..., 1:2] << 8) + image_data[..., 0:1]

        return instance_ids

    def _decode_scene_flow(
        self, scene_name: str, annotation_identifier: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if "back_motion_vectors_3d" in annotation_identifier:
            forward_annotation_path = (
                self._dataset_path
                / scene_name
                / annotation_identifier.replace("back_motion_vectors_3d", "motion_vectors_3d")
            )
            back_annotation_path = self._dataset_path / scene_name / annotation_identifier
        else:
            forward_annotation_path = self._dataset_path / scene_name / annotation_identifier
            back_annotation_path = (
                self._dataset_path
                / scene_name
                / annotation_identifier.replace("motion_vectors_3d", "back_motion_vectors_3d")
            )

        forward_vectors = (
            read_npz(path=forward_annotation_path, files="motion_vectors") if forward_annotation_path.exists() else None
        )
        backward_vectors = (
            read_npz(path=back_annotation_path, files="motion_vectors") if back_annotation_path.exists() else None
        )

        return forward_vectors, backward_vectors

    def _decode_surface_normals_3d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        vectors = read_npz(path=annotation_path, files="surface_normals")
        return vectors

    def _decode_surface_normals_2d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier

        encoded_norms = read_png(path=annotation_path)[..., :3]
        encoded_norms_f = encoded_norms.astype(np.float)
        decoded_norms = ((encoded_norms_f / 255) - 0.5) * 2
        decoded_norms = decoded_norms / np.linalg.norm(decoded_norms, axis=-1, keepdims=True)

        return decoded_norms

    def _decode_material_properties_3d(
        self, scene_name: str, annotation_identifier: str
    ) -> (np.ndarray, Dict[str, np.ndarray]):
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        try:
            material_data = read_npz(path=annotation_path, files="material_properties")
        except KeyError:  # Temporary solution
            material_data = read_npz(
                path=annotation_path, files="surface_properties"
            )  # TODO: Replace with `material_properties`

        material_ids = np.round(material_data[:, 6].reshape(-1, 1) * 255).astype(int)
        roughness = material_data[:, 0].reshape(-1, 1)
        metallic = material_data[:, 1].reshape(-1, 1)
        specular = material_data[:, 2].reshape(-1, 1)
        emissive = material_data[:, 3].reshape(-1, 1)
        opacity = material_data[:, 4].reshape(-1, 1)
        flags = material_data[:, 5].reshape(-1, 1)

        return material_ids, roughness, metallic, specular, emissive, opacity, flags

    def _decode_file_path(self, sensor_name: SensorName, frame_id: FrameId, data_type: Type[F]) -> Optional[AnyPath]:
        annotation_identifiers = self.get_available_annotation_types(sensor_name=sensor_name, frame_id=frame_id)
        if data_type in annotation_identifiers:
            annotation_identifier = annotation_identifiers[data_type]
            return self._dataset_path / self.scene_name / annotation_identifier
        elif issubclass(data_type, Image):
            datum = self._get_sensor_frame_data_datum(frame_id=frame_id, sensor_name=sensor_name)
            return self._dataset_path / self.scene_name / datum.image.filename
        elif issubclass(data_type, PointCloud):
            datum = self._get_sensor_frame_data_datum(frame_id=frame_id, sensor_name=sensor_name)
            return self._dataset_path / self.scene_name / datum.point_cloud.filename

        return None


class DGPCameraSensorFrameDecoder(DGPSensorFrameDecoder, CameraSensorFrameDecoder[datetime]):
    def _decode_image_dimensions(self, sensor_name: SensorName, frame_id: FrameId) -> Tuple[int, int, int]:
        datum = self._get_sensor_frame_data_datum(frame_id=frame_id, sensor_name=sensor_name)
        return datum.image.height, datum.image.width, datum.image.channels

    def _decode_image_rgba(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        datum = self._get_sensor_frame_data_datum(frame_id=frame_id, sensor_name=sensor_name)
        cloud_path = self._dataset_path / self.scene_name / datum.image.filename
        image_data = read_image(path=cloud_path)

        return image_data

    def _decode_intrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> SensorIntrinsic:
        sample = self._get_current_frame_sample(frame_id=frame_id)
        dto = self._decode_intrinsic_calibration(
            scene_name=self.scene_name,
            calibration_key=sample.calibration_key,
            sensor_name=sensor_name,
        )

        if dto.fisheye is True or dto.fisheye == 1:
            camera_model = CameraModel.OPENCV_FISHEYE
        elif dto.fisheye is False or dto.fisheye == 0:
            camera_model = CameraModel.OPENCV_PINHOLE
        elif dto.fisheye == 3:
            camera_model = CameraModel.PD_FISHEYE
        elif dto.fisheye == 6:
            camera_model = CameraModel.PD_ORTHOGRAPHIC
        else:
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


class PointInfo(Enum):
    X = "X"
    Y = "Y"
    Z = "Z"
    I = "INTENSITY"  # noqa: E741
    R = "R"
    G = "G"
    B = "B"
    RING = "RING"
    TS = "TIMESTAMP"


class DGPLidarSensorFrameDecoder(DGPSensorFrameDecoder, LidarSensorFrameDecoder[datetime]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._decode_point_cloud_format = lru_cache(maxsize=1)(self._decode_point_cloud_format)
        self._decode_point_cloud_data = lru_cache(maxsize=1)(self._decode_point_cloud_data)

    def _get_index(self, p_info: PointInfo, sensor_name: SensorName, frame_id: FrameId):
        point_format = self._decode_point_cloud_format(sensor_name=sensor_name, frame_id=frame_id)
        point_cloud_info = {PointInfo(val): idx for idx, val in enumerate(point_format)}
        return point_cloud_info[p_info]

    def _decode_point_cloud_format(self, sensor_name: SensorName, frame_id: FrameId) -> List[str]:
        datum = self._get_sensor_frame_data_datum(frame_id=frame_id, sensor_name=sensor_name)
        return datum.point_cloud.point_format

    def _decode_point_cloud_data(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        datum = self._get_sensor_frame_data_datum(frame_id=frame_id, sensor_name=sensor_name)
        cloud_path = self._dataset_path / self.scene_name / datum.point_cloud.filename
        pc_data = read_npz(path=cloud_path, files="data")
        return np.column_stack([pc_data[c] for c in pc_data.dtype.names])

    def _has_point_cloud_data(self, sensor_name: SensorName, frame_id: FrameId) -> bool:
        datum = self._get_sensor_frame_data_datum(frame_id=frame_id, sensor_name=sensor_name)
        return isinstance(datum, SceneDataDatumPointCloud)

    def _decode_point_cloud_size(self, sensor_name: SensorName, frame_id: FrameId) -> int:
        data = self._decode_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id)
        return len(data)

    def _decode_point_cloud_xyz(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        xyz_index = [
            self._get_index(p_info=PointInfo.X, sensor_name=sensor_name, frame_id=frame_id),
            self._get_index(p_info=PointInfo.Y, sensor_name=sensor_name, frame_id=frame_id),
            self._get_index(p_info=PointInfo.Z, sensor_name=sensor_name, frame_id=frame_id),
        ]
        data = self._decode_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id)
        return data[:, xyz_index]

    def _decode_point_cloud_rgb(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        rgb_index = [
            self._get_index(p_info=PointInfo.R, sensor_name=sensor_name, frame_id=frame_id),
            self._get_index(p_info=PointInfo.G, sensor_name=sensor_name, frame_id=frame_id),
            self._get_index(p_info=PointInfo.B, sensor_name=sensor_name, frame_id=frame_id),
        ]
        data = self._decode_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id)
        return data[:, rgb_index]

    def _decode_point_cloud_intensity(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        intensity_index = [
            self._get_index(p_info=PointInfo.I, sensor_name=sensor_name, frame_id=frame_id),
        ]
        data = self._decode_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id)
        return data[:, intensity_index]

    def _decode_point_cloud_timestamp(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        ts_index = [
            self._get_index(p_info=PointInfo.TS, sensor_name=sensor_name, frame_id=frame_id),
        ]
        data = self._decode_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id)
        return data[:, ts_index]

    def _decode_point_cloud_ring_index(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        ring_index = [
            self._get_index(p_info=PointInfo.RING, sensor_name=sensor_name, frame_id=frame_id),
        ]
        data = self._decode_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id)
        return data[:, ring_index]

    def _decode_point_cloud_ray_type(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        return None


class DGPPointCachePointsDecoder:
    def __init__(self, sha: str, size: float, cache_folder: AnyPath, pose: Transformation, parent_pose: Transformation):
        self._size = size
        self._cache_folder = cache_folder
        self._sha = sha
        self._pose = pose
        self._parent_pose = parent_pose
        self._file_path = cache_folder / (sha + ".npz")

    @lru_cache(maxsize=1)
    def get_point_data(self) -> np.ndarray:
        with self._file_path.open("rb") as f:
            cache_points = np.load(f)["data"]
        return cache_points

    def get_points_xyz(self) -> Optional[np.ndarray]:
        cache_points = self.get_point_data()
        point_data = np.column_stack([cache_points["X"], cache_points["Y"], cache_points["Z"]]) * self._size

        points = np.column_stack([point_data, np.ones(point_data.shape[0])])
        return (self._parent_pose @ (self._pose @ points.T)).T[:, :3]

    def get_points_normals(self) -> Optional[np.ndarray]:
        cache_points = self.get_point_data()
        normals = np.column_stack([cache_points["NX"], cache_points["NY"], cache_points["NZ"]]) * self._size
        return (self._parent_pose.rotation @ (self._pose.rotation @ normals.T)).T


def _pose_dto_to_transformation(dto: PoseDTO, transformation_type: Type[TransformType]) -> TransformType:
    transform = transformation_type(
        quaternion=Quaternion(dto.rotation.qw, dto.rotation.qx, dto.rotation.qy, dto.rotation.qz),
        translation=np.array([dto.translation.x, dto.translation.y, dto.translation.z]),
    )
    return transformation_type.from_transformation_matrix(DGP_TO_INTERNAL_CS @ transform.transformation_matrix)
