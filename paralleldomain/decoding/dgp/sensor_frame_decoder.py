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
    PoseDTO,
    SceneDataDatum,
    SceneDataDatumPointCloud,
    SceneDataDTO,
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
    AnnotationIdentifier,
    BackwardOpticalFlow,
    BackwardSceneFlow,
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
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.image import Image
from paralleldomain.model.point_cloud import PointCloud
from paralleldomain.model.sensor import CameraModel, SensorDataCopyTypes, SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image, read_json, read_npz, read_png
from paralleldomain.utilities.projection import DistortionLookup, DistortionLookupTable
from paralleldomain.utilities.transformation import Transformation

T = TypeVar("T")

"""
All our sensor data is in RDF coordinate system. Applying the extrinsic/sensor_to_ego transformation transforms it
into FLU.
Please note that transforming 3d boxes into camera sensor frame is currently not implemented.
"""


class DGPSensorFrameDecoder(SensorFrameDecoder[datetime], metaclass=abc.ABCMeta):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        frame_id: FrameId,
        dataset_path: AnyPath,
        frame_sample: SceneSampleDTO,
        sensor_frame_data: SceneDataDTO,
        ontologies: Dict[str, str],
        custom_reference_to_box_bottom: Transformation,
        settings: DecoderSettings,
        is_unordered_scene: bool,
        point_cache_folder_exists: bool,
        scene_decoder,
    ):
        super().__init__(
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            frame_id=frame_id,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        self._dataset_path = dataset_path
        self.current_frame_sample = frame_sample
        # self.current_frame_sample = scene_samples[frame_id]
        self._ontologies = ontologies
        self._custom_reference_to_box_bottom = custom_reference_to_box_bottom
        # self._data_by_sensor_name = {d.key: d for d in scene_data if d.id.name == self.sensor_name}
        #
        # datum_key = next(
        #     iter([key for key in frame_sample.datum_keys if key in self._data_by_sensor_name])
        # )
        # self.sensor_frame_data = self._data_by_sensor_name[datum_key]
        self.sensor_frame_data = sensor_frame_data
        self._point_cache_folder_exists = point_cache_folder_exists

    def _decode_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        return decode_class_maps(
            ontologies=self._ontologies, dataset_path=self._dataset_path, scene_name=self.scene_name
        )

    def _decode_date_time(self) -> datetime:
        return scene_data_to_date_time(data=self.sensor_frame_data)

    def _decode_extrinsic(self) -> SensorExtrinsic:
        dto = self._decode_extrinsic_calibration()
        sensor_to_box_bottom = _pose_dto_to_transformation(dto=dto, transformation_type=SensorExtrinsic)
        sensor_to_custom_reference = (
            self._custom_reference_to_box_bottom.inverse @ sensor_to_box_bottom
        )  # from center-bottom to center rear-axle
        return sensor_to_custom_reference

    def _decode_sensor_pose(self) -> SensorPose:
        datum = self.sensor_frame_data.datum
        if datum.image:
            return _pose_dto_to_transformation(dto=datum.image.pose, transformation_type=SensorPose)
        else:
            return _pose_dto_to_transformation(dto=datum.point_cloud.pose, transformation_type=SensorPose)

    def _decode_annotations(self, identifier: AnnotationIdentifier[T]) -> T:
        annotation_type = identifier.annotation_type
        relative_path = self._get_annotation_relative_path(identifier=identifier)

        if issubclass(annotation_type, BoundingBoxes3D):
            dto = self._decode_bounding_boxes_3d(relative_path=relative_path)

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
            dto = self._decode_bounding_boxes_2d(relative_path=relative_path)

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
            segmentation_mask = self._decode_semantic_segmentation_3d(relative_path=relative_path)
            return SemanticSegmentation3D(class_ids=segmentation_mask)
        elif issubclass(annotation_type, InstanceSegmentation3D):
            instance_mask = self._decode_instance_segmentation_3d(relative_path=relative_path)
            return InstanceSegmentation3D(instance_ids=instance_mask)
        elif issubclass(annotation_type, SemanticSegmentation2D):
            class_ids = self._decode_semantic_segmentation_2d(relative_path=relative_path)
            return SemanticSegmentation2D(class_ids=class_ids)
        elif issubclass(annotation_type, InstanceSegmentation2D):
            instance_ids = self._decode_instance_segmentation_2d(relative_path=relative_path)
            return InstanceSegmentation2D(instance_ids=instance_ids)
        elif issubclass(annotation_type, OpticalFlow):
            vectors = self._decode_optical_flow(relative_path=relative_path)
            return OpticalFlow(vectors=vectors)
        elif issubclass(annotation_type, BackwardOpticalFlow):
            vectors = self._decode_optical_flow(relative_path=relative_path)
            return BackwardOpticalFlow(vectors=vectors)
        elif issubclass(annotation_type, Depth):
            depth_mask = self._decode_depth(relative_path=relative_path)
            return Depth(depth=depth_mask)
        elif issubclass(annotation_type, SceneFlow):
            vectors = self._decode_scene_flow(relative_path=relative_path)
            return SceneFlow(vectors=vectors)
        elif issubclass(annotation_type, BackwardSceneFlow):
            vectors = self._decode_scene_flow(relative_path=relative_path)
            return BackwardSceneFlow(vectors=vectors)
        elif issubclass(annotation_type, SurfaceNormals3D):
            normals = self._decode_surface_normals_3d(relative_path=relative_path)
            return SurfaceNormals3D(normals=normals)
        elif issubclass(annotation_type, SurfaceNormals2D):
            normals = self._decode_surface_normals_2d(relative_path=relative_path)
            return SurfaceNormals2D(normals=normals)
        elif issubclass(annotation_type, PointCaches):
            caches = self._decode_point_caches(relative_path=relative_path)
            return PointCaches(caches=caches)
        elif issubclass(annotation_type, MaterialProperties3D):
            material_ids, roughness, metallic, specular, emissive, opacity, flags = self._decode_material_properties_3d(
                relative_path=relative_path
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
            color = self._decode_albedo_2d(relative_path=relative_path)
            return Albedo2D(color=color)
        elif issubclass(annotation_type, MaterialProperties2D):
            roughness = self._decode_material_properties_2d(relative_path=relative_path)
            return MaterialProperties2D(roughness=roughness)
        else:
            raise NotImplementedError(f"{annotation_type} is not implemented yet in this decoder!")

    def _get_annotation_relative_path(self, identifier: AnnotationIdentifier[T]) -> str:
        datum = self.sensor_frame_data.datum
        if datum.image:
            type_to_path = datum.image.annotations
        else:
            type_to_path = datum.point_cloud.annotations

        available_annotation_types = {ANNOTATION_TYPE_MAP[k]: v for k, v in type_to_path.items()}

        if BoundingBoxes3D in available_annotation_types and self._point_cache_folder_exists:
            available_annotation_types[PointCaches] = "point_cache"

        return available_annotation_types[identifier.annotation_type]

    def _decode_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        datum = self.sensor_frame_data.datum
        if datum.image:
            type_to_path = datum.image.annotations
        else:
            type_to_path = datum.point_cloud.annotations

        available_annotation_types = [
            AnnotationIdentifier(annotation_type=ANNOTATION_TYPE_MAP[k]) for k in type_to_path.keys()
        ]

        if BoundingBoxes3D in available_annotation_types and self._point_cache_folder_exists:
            available_annotation_types.append(AnnotationIdentifier(annotation_type=PointCaches))

        return available_annotation_types

    def _decode_metadata(self) -> Dict[str, Any]:
        datum = self.sensor_frame_data.datum
        if datum.image:
            return datum.image.metadata
        else:
            return datum.point_cloud.metadata

    # ---------------------------------

    def _decode_calibration(self) -> CalibrationDTO:
        calibration_key = self.current_frame_sample.calibration_key
        calibration_path = self._dataset_path / self.scene_name / "calibration" / f"{calibration_key}.json"
        cal_dict = read_json(path=calibration_path)
        return CalibrationDTO.from_dict(cal_dict)

    def _decode_extrinsic_calibration(self) -> CalibrationExtrinsicDTO:
        calibration_dto = self._decode_calibration()
        index = calibration_dto.names.index(self.sensor_name)
        return calibration_dto.extrinsics[index]

    def _decode_intrinsic_calibration(self) -> CalibrationIntrinsicDTO:
        calibration_dto = self._decode_calibration()
        index = calibration_dto.names.index(self.sensor_name)
        return calibration_dto.intrinsics[index]

    def _get_3d_boxes_for_point_cache(self):
        boxes = self.get_annotations(
            identifier=AnnotationIdentifier(annotation_type=BoundingBoxes3D),
        )
        return boxes

    def _decode_point_caches(self, relative_path: str) -> List[PointCache]:
        point_cache_folder = self._dataset_path / self.scene_name / relative_path
        boxes = self._get_3d_boxes_for_point_cache()
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

    def _decode_bounding_boxes_3d(self, relative_path: str) -> AnnotationsBoundingBox3DDTO:
        annotation_path = self._dataset_path / self.scene_name / relative_path
        bb_dict = read_json(path=annotation_path)
        return AnnotationsBoundingBox3DDTO.from_dict(bb_dict)

    def _decode_bounding_boxes_2d(self, relative_path: str) -> AnnotationsBoundingBox2DDTO:
        annotation_path = self._dataset_path / self.scene_name / relative_path
        bb_dict = read_json(path=annotation_path)
        return AnnotationsBoundingBox2DDTO.from_dict(bb_dict)

    def _decode_semantic_segmentation_3d(self, relative_path: str) -> np.ndarray:
        annotation_path = self._dataset_path / self.scene_name / relative_path
        segmentation_data = read_npz(path=annotation_path, files="segmentation")

        return segmentation_data

    def _decode_instance_segmentation_3d(self, relative_path: str) -> np.ndarray:
        annotation_path = self._dataset_path / self.scene_name / relative_path
        instance_data = read_npz(path=annotation_path, files="instance")

        return instance_data

    def _decode_semantic_segmentation_2d(self, relative_path: str) -> np.ndarray:
        annotation_path = self._dataset_path / self.scene_name / relative_path
        image_data = read_image(path=annotation_path)
        image_data = image_data.astype(int)
        class_ids = (image_data[..., 2:3] << 16) + (image_data[..., 1:2] << 8) + image_data[..., 0:1]

        return class_ids

    def _decode_optical_flow(self, relative_path: str) -> np.ndarray:
        annotation_path = self._dataset_path / self.scene_name / relative_path

        image_data = read_image(path=annotation_path)
        image_data = image_data.astype(int)
        height, width = image_data.shape[0:2]
        vectors = image_data[..., [0, 2]] + (image_data[..., [1, 3]] << 8)
        vectors = (vectors / 65535.0 - 0.5) * [width, height] * 2

        return vectors

    def _decode_albedo_2d(self, relative_path: str) -> np.ndarray:
        annotation_path = self._dataset_path / self.scene_name / relative_path
        color = read_image(path=annotation_path)[..., :3]
        return color

    def _decode_material_properties_2d(self, relative_path: str) -> np.ndarray:
        annotation_path = self._dataset_path / self.scene_name / relative_path
        roughness = read_png(path=annotation_path)[..., :3]
        return roughness

    def _decode_depth(self, relative_path: str) -> np.ndarray:
        annotation_path = self._dataset_path / self.scene_name / relative_path
        depth_data = read_npz(path=annotation_path, files="data")

        return np.expand_dims(depth_data, axis=-1)

    def _decode_instance_segmentation_2d(self, relative_path: str) -> np.ndarray:
        annotation_path = self._dataset_path / self.scene_name / relative_path
        image_data = read_image(path=annotation_path)
        image_data = image_data.astype(int)
        instance_ids = (image_data[..., 2:3] << 16) + (image_data[..., 1:2] << 8) + image_data[..., 0:1]

        return instance_ids

    def _decode_scene_flow(self, relative_path: str, files: str = "motion_vectors") -> np.ndarray:
        annotation_path = self._dataset_path / self.scene_name / relative_path

        vectors = read_npz(path=annotation_path, files=files)

        return vectors

    def _decode_surface_normals_3d(self, relative_path: str) -> np.ndarray:
        annotation_path = self._dataset_path / self.scene_name / relative_path
        vectors = read_npz(path=annotation_path, files="surface_normals")
        return vectors

    def _decode_surface_normals_2d(self, relative_path: str) -> np.ndarray:
        annotation_path = self._dataset_path / self.scene_name / relative_path

        encoded_norms = read_png(path=annotation_path)[..., :3]
        encoded_norms_f = encoded_norms.astype(float)
        decoded_norms = ((encoded_norms_f / 255) - 0.5) * 2
        decoded_norms = decoded_norms / np.linalg.norm(decoded_norms, axis=-1, keepdims=True)

        return decoded_norms

    def _decode_material_properties_3d(self, relative_path: str) -> (np.ndarray, Dict[str, np.ndarray]):
        annotation_path = self._dataset_path / self.scene_name / relative_path
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

    def _decode_file_path(self, data_type: SensorDataCopyTypes) -> Optional[AnyPath]:
        annotation_identifiers = self.get_available_annotation_identifiers()
        annotation_identifiers = {a.annotation_type: a for a in annotation_identifiers}
        if isinstance(data_type, AnnotationIdentifier) and data_type.annotation_type in annotation_identifiers:
            relative_path = self._get_annotation_relative_path(identifier=annotation_identifiers[data_type])
            return self._dataset_path / self.scene_name / relative_path
        elif data_type in annotation_identifiers:
            # Note: We also support Type[Annotation] for data_type for backwards compatibility
            relative_path = self._get_annotation_relative_path(identifier=annotation_identifiers[data_type])
            return self._dataset_path / self.scene_name / relative_path
        elif issubclass(data_type, Image):
            datum = self.sensor_frame_data.datum
            return self._dataset_path / self.scene_name / datum.image.filename
        elif issubclass(data_type, PointCloud):
            datum = self.sensor_frame_data.datum
            return self._dataset_path / self.scene_name / datum.point_cloud.filename

        return None


class DGPCameraSensorFrameDecoder(DGPSensorFrameDecoder, CameraSensorFrameDecoder[datetime]):
    def _decode_image_dimensions(self) -> Tuple[int, int, int]:
        datum = self.sensor_frame_data.datum
        return datum.image.height, datum.image.width, datum.image.channels

    def _decode_image_rgba(self) -> np.ndarray:
        datum = self.sensor_frame_data.datum
        cloud_path = self._dataset_path / self.scene_name / datum.image.filename
        image_data = read_image(path=cloud_path)

        return image_data

    def _decode_intrinsic(self) -> SensorIntrinsic:
        dto = self._decode_intrinsic_calibration()

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

    def _decode_distortion_lookup(self) -> Optional[DistortionLookup]:
        lookup = super()._decode_distortion_lookup()
        if lookup is None:
            lut_csv_path = self._dataset_path / self.scene_name / "calibration" / f"{self.sensor_name}.csv"
            if lut_csv_path.exists():
                with lut_csv_path.open() as f:
                    lut = np.loadtxt(f, delimiter=",", dtype="float")
                lookup = DistortionLookupTable.from_ndarray(lut)
        return lookup


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
    AZIMUTH = "AZIMUTH"
    ELEVATION = "ELEVATION"


# The LidarSensorFrameDecoder api exposes individual access to point fields, but they are stored in one single file.
# We don't want to download the file multiple times directly after each other, so we cache it.
# The LidarSensorFrameDecoder itself is cached, too, so we can't tie this cache to that instance.
# Also note that we don't set  the cache size to one so that the cache works with threaded downloads
@lru_cache(maxsize=16)
def load_point_cloud(path: str) -> np.ndarray:
    pc_data = read_npz(path=AnyPath(path=path), files="data")
    return np.column_stack([pc_data[c] for c in pc_data.dtype.names])


class DGPLidarSensorFrameDecoder(DGPSensorFrameDecoder, LidarSensorFrameDecoder[datetime]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.point_format = self.sensor_frame_data.datum.point_cloud.point_format
        self.point_format = ['X', 'Y', 'Z', 'INTENSITY', 'R', 'G', 'B', 'RING', 'TIMESTAMP', "AZIMUTH", "ELEVATION"]

    def _get_index(self, p_info: PointInfo):
        point_cloud_info = {PointInfo(val): idx for idx, val in enumerate(self.point_format)}
        return point_cloud_info[p_info]

    def _decode_point_cloud_data(self) -> Optional[np.ndarray]:
        datum = self.sensor_frame_data.datum
        cloud_path = self._dataset_path / self.scene_name / datum.point_cloud.filename
        return load_point_cloud(str(cloud_path))

    def _has_point_cloud_data(self) -> bool:
        datum = self.sensor_frame_data.datum
        return isinstance(datum, SceneDataDatumPointCloud)

    def _decode_point_cloud_size(self) -> int:
        data = self._decode_point_cloud_data()
        return len(data)

    def _decode_point_cloud_xyz(self) -> Optional[np.ndarray]:
        xyz_index = [
            self._get_index(p_info=PointInfo.X),
            self._get_index(p_info=PointInfo.Y),
            self._get_index(p_info=PointInfo.Z),
        ]
        data = self._decode_point_cloud_data()
        return data[:, xyz_index]

    def _decode_point_cloud_rgb(self) -> Optional[np.ndarray]:
        rgb_index = [
            self._get_index(p_info=PointInfo.R),
            self._get_index(p_info=PointInfo.G),
            self._get_index(p_info=PointInfo.B),
        ]
        data = self._decode_point_cloud_data()
        return data[:, rgb_index]

    def _decode_point_cloud_intensity(self) -> Optional[np.ndarray]:
        intensity_index = [
            self._get_index(p_info=PointInfo.I),
        ]
        data = self._decode_point_cloud_data()
        return data[:, intensity_index]

    def _decode_point_cloud_elongation(self) -> Optional[np.ndarray]:
        return None

    def _decode_point_cloud_timestamp(self) -> Optional[np.ndarray]:
        ts_index = [
            self._get_index(p_info=PointInfo.TS),
        ]
        data = self._decode_point_cloud_data()
        return data[:, ts_index]

    def _decode_point_cloud_ring_index(self) -> Optional[np.ndarray]:
        ring_index = [
            self._get_index(p_info=PointInfo.RING),
        ]
        data = self._decode_point_cloud_data()
        return data[:, ring_index]

    def _decode_point_cloud_ray_type(self) -> Optional[np.ndarray]:
        return None

    def _decode_point_cloud_azimuth(self) -> Optional[np.ndarray]:
        azimuth_index = [
            self._get_index(p_info=PointInfo.AZIMUTH),
        ]
        data = self._decode_point_cloud_data()
        return data[:, azimuth_index]

    def _decode_point_cloud_elevation(self) -> Optional[np.ndarray]:
        elevation_index = [
            self._get_index(p_info=PointInfo.ELEVATION),
        ]
        data = self._decode_point_cloud_data()
        return data[:, elevation_index]


@lru_cache(maxsize=16)
def get_point_cache_data(path: AnyPath) -> np.ndarray:
    with path.open("rb") as f:
        cache_points = np.load(f)["data"]
    return cache_points


class DGPPointCachePointsDecoder:
    def __init__(self, sha: str, size: float, cache_folder: AnyPath, pose: Transformation, parent_pose: Transformation):
        self._size = size
        self._cache_folder = cache_folder
        self._sha = sha
        self._pose = pose
        self._parent_pose = parent_pose
        self._file_path = cache_folder / (sha + ".npz")

    def get_point_data(self) -> np.ndarray:
        return get_point_cache_data(path=self._file_path)

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
