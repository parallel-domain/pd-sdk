import abc
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import ujson
from pyquaternion import Quaternion

from paralleldomain.common.dgp.v1 import annotations_pb2, geometry_pb2, point_cloud_pb2, sample_pb2
from paralleldomain.common.dgp.v1.constants import ANNOTATION_TYPE_MAP, DGP_TO_INTERNAL_CS, TransformType
from paralleldomain.common.dgp.v1.utils import timestamp_to_datetime
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.sensor_frame_decoder import (
    CameraSensorFrameDecoder,
    LidarSensorFrameDecoder,
    SensorFrameDecoder,
)
from paralleldomain.model.annotation import (
    AnnotationPose,
    AnnotationType,
    BoundingBox2D,
    BoundingBox3D,
    BoundingBoxes2D,
    BoundingBoxes3D,
    Depth,
    InstanceSegmentation2D,
    InstanceSegmentation3D,
    Line2D,
    OpticalFlow,
    Point2D,
    Points2D,
    Polygon2D,
    Polygons2D,
    Polyline2D,
    Polylines2D,
    SceneFlow,
    SemanticSegmentation2D,
    SemanticSegmentation3D,
    SurfaceNormals2D,
    SurfaceNormals3D,
)
from paralleldomain.model.sensor import CameraModel, SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import AnnotationIdentifier, FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_image, read_json_message, read_npz, read_png
from paralleldomain.utilities.transformation import Transformation

T = TypeVar("T")


def _decode_attributes(attributes: Dict[str, str]) -> Dict[str, Any]:
    attributes_decoded = {}
    for k, v in attributes.items():
        try:
            attributes_decoded[k] = ujson.loads(v)
        except (ValueError, JSONDecodeError):
            attributes_decoded[k] = v

    return attributes_decoded


class DGPSensorFrameDecoder(SensorFrameDecoder[datetime], metaclass=abc.ABCMeta):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        scene_samples: Dict[FrameId, sample_pb2.Sample],
        scene_data: List[sample_pb2.Datum],
        custom_reference_to_box_bottom: Transformation,
        settings: DecoderSettings,
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self._dataset_path = dataset_path
        self.scene_samples = scene_samples
        self.scene_data = scene_data
        self._custom_reference_to_box_bottom = custom_reference_to_box_bottom

    @lru_cache(maxsize=1)
    def _data_by_sensor_name(self, sensor_name: SensorName) -> Dict[str, sample_pb2.Datum]:
        return {d.key: d for d in self.scene_data if d.id.name == sensor_name}

    def _get_current_frame_sample(self, frame_id: FrameId) -> sample_pb2.Sample:
        return self.scene_samples[frame_id]

    @lru_cache(maxsize=1)
    def _get_sensor_frame_data(self, frame_id: FrameId, sensor_name: SensorName) -> sample_pb2.Datum:
        sample = self._get_current_frame_sample(frame_id=frame_id)
        # all sensor data of the sensor
        sensor_data = self._data_by_sensor_name(sensor_name=sensor_name)
        # read ontology -> Dict[str, ClassMap]
        # datum ley of sample that references the given sensor name
        datum_key = next(iter([key for key in sample.datum_keys if key in sensor_data]))
        scene_data = sensor_data[datum_key]
        return scene_data

    def _get_sensor_frame_data_datum(self, frame_id: FrameId, sensor_name: SensorName) -> sample_pb2.DatumValue:
        scene_data = self._get_sensor_frame_data(frame_id=frame_id, sensor_name=sensor_name)
        return scene_data.datum

    def _decode_date_time(self, sensor_name: SensorName, frame_id: FrameId) -> datetime:
        data = self._get_sensor_frame_data(frame_id=frame_id, sensor_name=sensor_name)
        return timestamp_to_datetime(ts=data.id.timestamp)

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
                pose = _pose_dto_to_transformation(dto=box_dto.box.pose, transformation_type=AnnotationPose)

                # Decode generic attributes from and handle json encoded values
                attr_decoded = _decode_attributes(attributes=box_dto.attributes)
                # Read truncation, occlusion and move them to attribute section in model
                attr_decoded.update(
                    {
                        "occlusion": box_dto.box.occlusion,
                        "truncation": box_dto.box.truncation,
                    }
                )

                class_id = box_dto.class_id

                box = BoundingBox3D(
                    pose=pose,
                    width=box_dto.box.width,
                    length=box_dto.box.length,
                    height=box_dto.box.height,
                    class_id=class_id,
                    instance_id=box_dto.instance_id,
                    num_points=box_dto.num_points,
                    attributes=attr_decoded,
                )
                box_list.append(box)

            return BoundingBoxes3D(boxes=box_list)
        elif issubclass(annotation_type, BoundingBoxes2D):
            dto = self._decode_bounding_boxes_2d(scene_name=self.scene_name, annotation_identifier=identifier)

            box_list = []
            for box_dto in dto.annotations:

                # Decode generic attributes from and handle json encoded values
                attr_decoded = _decode_attributes(attributes=box_dto.attributes)
                # Read "iscrowd" and move them to attribute section in model
                attr_decoded.update(
                    {
                        "iscrowd": box_dto.iscrowd,
                    }
                )

                class_id = box_dto.class_id

                box = BoundingBox2D(
                    x=box_dto.box.x,
                    y=box_dto.box.y,
                    width=box_dto.box.w,
                    height=box_dto.box.h,
                    class_id=class_id,
                    instance_id=box_dto.instance_id,
                    attributes=attr_decoded,
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
            flow_vectors = self._decode_optical_flow(scene_name=self.scene_name, annotation_identifier=identifier)
            return OpticalFlow(vectors=flow_vectors)
        elif issubclass(annotation_type, Depth):
            depth_mask = self._decode_depth(scene_name=self.scene_name, annotation_identifier=identifier)
            return Depth(depth=depth_mask)
        elif issubclass(annotation_type, SceneFlow):
            flow = self._decode_scene_flow(scene_name=self.scene_name, annotation_identifier=identifier)
            return SceneFlow(vectors=flow)
        elif issubclass(annotation_type, SurfaceNormals3D):
            normals = self._decode_surface_normals_3d(scene_name=self.scene_name, annotation_identifier=identifier)
            return SurfaceNormals3D(normals=normals)
        elif issubclass(annotation_type, SurfaceNormals2D):
            normals = self._decode_surface_normals_2d(scene_name=self.scene_name, annotation_identifier=identifier)
            return SurfaceNormals2D(normals=normals)
        elif issubclass(annotation_type, Polylines2D):
            polylines = self._decode_polylines_2d(scene_name=self.scene_name, annotation_identifier=identifier)
            return Polylines2D(polylines=polylines)
        elif issubclass(annotation_type, Polygons2D):
            polygons = self._decode_polygons_2d(scene_name=self.scene_name, annotation_identifier=identifier)
            return Polygons2D(polygons=polygons)
        elif issubclass(annotation_type, Points2D):
            points = self._decode_points_2d(scene_name=self.scene_name, annotation_identifier=identifier)
            return Points2D(points=points)
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
        return {ANNOTATION_TYPE_MAP[k]: v for k, v in type_to_path.items()}

    # ---------------------------------

    def _decode_calibration(self, scene_name: str, calibration_key: str) -> sample_pb2.SampleCalibration:
        calibration_path = self._dataset_path / scene_name / "calibration" / f"{calibration_key}.json"
        return read_json_message(obj=sample_pb2.SampleCalibration(), path=calibration_path)

    def _decode_extrinsic_calibration(
        self, scene_name: str, calibration_key: str, sensor_name: SensorName
    ) -> geometry_pb2.Pose:
        calibration_dto = self._decode_calibration(scene_name=scene_name, calibration_key=calibration_key)
        index = next(i for i, v in enumerate(calibration_dto.names) if v == sensor_name)
        return calibration_dto.extrinsics[index]

    def _decode_intrinsic_calibration(
        self, scene_name: str, calibration_key: str, sensor_name: SensorName
    ) -> geometry_pb2.CameraIntrinsics:
        calibration_dto = self._decode_calibration(scene_name=scene_name, calibration_key=calibration_key)
        index = next(i for i, v in enumerate(calibration_dto.names) if v == sensor_name)
        return calibration_dto.intrinsics[index]

    def _decode_bounding_boxes_3d(
        self, scene_name: str, annotation_identifier: str
    ) -> annotations_pb2.BoundingBox3DAnnotations:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        return read_json_message(obj=annotations_pb2.BoundingBox3DAnnotations(), path=annotation_path)

    def _decode_bounding_boxes_2d(
        self, scene_name: str, annotation_identifier: str
    ) -> annotations_pb2.BoundingBox2DAnnotations:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        return read_json_message(obj=annotations_pb2.BoundingBox2DAnnotations(), path=annotation_path)

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

    def _decode_optical_flow(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        image_data = read_image(path=annotation_path)
        image_data = image_data.astype(int)
        vectors = (image_data[..., [0, 2]] << 8) + image_data[..., [1, 3]]

        return vectors

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

    def _decode_scene_flow(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        vectors = read_npz(path=annotation_path, files="motion_vectors")
        return vectors

    def _decode_surface_normals_3d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        vectors = read_npz(path=annotation_path, files="surface_normals")
        return vectors

    def _decode_surface_normals_2d(self, scene_name: str, annotation_identifier: str) -> np.ndarray:
        annotation_path = self._dataset_path / scene_name / annotation_identifier

        encoded_norms = read_png(path=annotation_path)[..., :3]
        encoded_norms_f = encoded_norms.astype(np.float)
        decoded_norms = ((encoded_norms_f / 255) - 0.5) * 2
        # Deactivated for now so original RGB-encoded can be recovered in encoding
        # decoded_norms = decoded_norms / np.linalg.norm(decoded_norms, axis=-1, keepdims=True)

        return decoded_norms

    def _lines_from_pb_annotation(
        self,
        annotation: Union[annotations_pb2.Polygon2DAnnotation, annotations_pb2.KeyLine2DAnnotation],
        class_id: int,
        instance_id: int,
    ) -> List[Line2D]:
        lines = list()
        points = list()
        for vertex in annotation.vertices:
            point = Point2D(x=vertex.x, y=vertex.y, class_id=class_id, instance_id=instance_id)
            points.append(point)

        for start, end in zip(points, points[1:]):
            line = Line2D(start=start, end=end, class_id=class_id, directed=False, instance_id=instance_id)
            lines.append(line)

        return lines

    def _decode_polygons_2d(self, scene_name: str, annotation_identifier: str) -> List[Polygon2D]:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        poly_annotations = read_json_message(obj=annotations_pb2.Polygon2DAnnotations(), path=annotation_path)
        polygons = list()
        for annotation in poly_annotations.annotations:
            attributes = _decode_attributes(attributes=annotation.attributes)
            instance_id = attributes.pop("instance_id", -1)
            class_id = annotation.class_id
            lines = self._lines_from_pb_annotation(annotation=annotation, class_id=class_id, instance_id=instance_id)

            polygon = Polygon2D(lines=lines, class_id=class_id, instance_id=instance_id, attributes=attributes)
            polygons.append(polygon)
        return polygons

    def _decode_polylines_2d(self, scene_name: str, annotation_identifier: str) -> List[Polyline2D]:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        poly_annotations = read_json_message(obj=annotations_pb2.KeyLine2DAnnotations(), path=annotation_path)
        polylines = list()
        for annotation in poly_annotations.annotations:
            attributes = _decode_attributes(attributes=annotation.attributes)
            instance_id = attributes.pop("instance_id", -1)
            class_id = annotation.class_id
            key = annotation.key
            attributes["key"] = key

            lines = self._lines_from_pb_annotation(annotation=annotation, class_id=class_id, instance_id=instance_id)

            polyline = Polyline2D(lines=lines, class_id=class_id, instance_id=instance_id, attributes=attributes)
            polylines.append(polyline)
        return polylines

    def _decode_points_2d(self, scene_name: str, annotation_identifier: str) -> List[Point2D]:
        annotation_path = self._dataset_path / scene_name / annotation_identifier
        poly_annotations = read_json_message(obj=annotations_pb2.KeyPoint2DAnnotations(), path=annotation_path)
        points = list()
        for annotation in poly_annotations.annotations:
            attributes = _decode_attributes(attributes=annotation.attributes)
            instance_id = attributes.pop("instance_id", -1)
            class_id = annotation.class_id
            key = annotation.key
            attributes["key"] = key

            point = Point2D(
                x=annotation.point.x,
                y=annotation.point.y,
                class_id=class_id,
                instance_id=instance_id,
                attributes=attributes,
            )

            points.append(point)
        return points


class DGPCameraSensorFrameDecoder(DGPSensorFrameDecoder, CameraSensorFrameDecoder[datetime]):
    def _decode_image_dimensions(self, sensor_name: SensorName, frame_id: FrameId) -> Tuple[int, int, int]:
        datum = self._get_sensor_frame_data_datum(frame_id=frame_id, sensor_name=sensor_name)
        return (datum.image.height, datum.image.width, datum.image.channels)

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


class PointInfo:
    X: str = "X"
    Y: str = "Y"
    Z: str = "Z"
    I: str = "INTENSITY"  # noqa: E741
    R: str = "R"
    G: str = "G"
    B: str = "B"
    RING: str = "RING"
    TS: str = "TIMESTAMP"


class DGPLidarSensorFrameDecoder(DGPSensorFrameDecoder, LidarSensorFrameDecoder[datetime]):
    def _get_index(self, p_info: str, sensor_name: SensorName, frame_id: FrameId):
        point_format = self._decode_point_cloud_format(sensor_name=sensor_name, frame_id=frame_id)
        point_cloud_info = {point_cloud_pb2.PointCloud.ChannelType.Name(val): val for val in point_format}
        return point_cloud_info[p_info]

    @lru_cache(maxsize=1)
    def _decode_point_cloud_format(self, sensor_name: SensorName, frame_id: FrameId) -> List[str]:
        datum = self._get_sensor_frame_data_datum(frame_id=frame_id, sensor_name=sensor_name)
        return datum.point_cloud.point_format

    @lru_cache(maxsize=1)
    def _decode_point_cloud_data(self, sensor_name: SensorName, frame_id: FrameId) -> Optional[np.ndarray]:
        datum = self._get_sensor_frame_data_datum(frame_id=frame_id, sensor_name=sensor_name)
        cloud_path = self._dataset_path / self.scene_name / datum.point_cloud.filename
        pc_data = read_npz(path=cloud_path, files="data")
        return np.column_stack([pc_data[c] for c in pc_data.dtype.names])

    def _has_point_cloud_data(self, sensor_name: SensorName, frame_id: FrameId) -> bool:
        datum = self._get_sensor_frame_data_datum(frame_id=frame_id, sensor_name=sensor_name)
        return datum.HasField("point_cloud")

    def _decode_point_cloud_size(self, sensor_name: SensorName, frame_id: FrameId) -> int:
        data = self._decode_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id)
        return len(data)

    def _decode_point_cloud_xyz(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        xyz_index = [
            self._get_index(p_info=PointInfo.X, sensor_name=sensor_name, frame_id=frame_id),
            self._get_index(p_info=PointInfo.Y, sensor_name=sensor_name, frame_id=frame_id),
            self._get_index(p_info=PointInfo.Z, sensor_name=sensor_name, frame_id=frame_id),
        ]
        data = self._decode_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id)
        return data[:, xyz_index]

    def _decode_point_cloud_rgb(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        rgb_index = [
            self._get_index(p_info=PointInfo.R, sensor_name=sensor_name, frame_id=frame_id),
            self._get_index(p_info=PointInfo.G, sensor_name=sensor_name, frame_id=frame_id),
            self._get_index(p_info=PointInfo.B, sensor_name=sensor_name, frame_id=frame_id),
        ]
        data = self._decode_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id)
        return data[:, rgb_index]

    def _decode_point_cloud_intensity(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        intensity_index = [
            self._get_index(p_info=PointInfo.I, sensor_name=sensor_name, frame_id=frame_id),
        ]
        data = self._decode_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id)
        return data[:, intensity_index]

    def _decode_point_cloud_timestamp(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        ts_index = [
            self._get_index(p_info=PointInfo.TS, sensor_name=sensor_name, frame_id=frame_id),
        ]
        data = self._decode_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id)
        return data[:, ts_index]

    def _decode_point_cloud_ring_index(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        ring_index = [
            self._get_index(p_info=PointInfo.RING, sensor_name=sensor_name, frame_id=frame_id),
        ]
        data = self._decode_point_cloud_data(sensor_name=sensor_name, frame_id=frame_id)
        return data[:, ring_index]


def _pose_dto_to_transformation(dto: geometry_pb2.Pose, transformation_type: Type[TransformType]) -> TransformType:
    transform = transformation_type(
        quaternion=Quaternion(dto.rotation.qw, dto.rotation.qx, dto.rotation.qy, dto.rotation.qz),
        translation=np.array([dto.translation.x, dto.translation.y, dto.translation.z]),
    )
    return transformation_type.from_transformation_matrix(DGP_TO_INTERNAL_CS @ transform.transformation_matrix)
