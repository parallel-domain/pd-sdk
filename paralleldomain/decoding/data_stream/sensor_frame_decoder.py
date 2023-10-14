import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from google.protobuf import json_format
from pd.internal.proto.label_engine.generated.python.bounding_box_2d_pb2 import VisibilitySampleMetadata
from pd.internal.proto.label_engine.generated.python.bounding_box_3d_pb2 import Cuboid3dMetadata
from pd.internal.proto.label_engine.generated.python.instance_point_pb2 import (
    InstancePoint2DMetadata,
    InstancePoint3DMetadata,
)
from pd.label_engine import LabelData
from pd.state import CameraSensor, Pose6D
from pyquaternion import Quaternion

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.data_stream import INSTANCE_POINT_COLOR_MAP
from paralleldomain.decoding.data_stream.common import TYPE_TO_FILE_FORMAT
from paralleldomain.decoding.data_stream.data_accessor import DataStreamDataAccessor
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder, SensorFrameDecoder, T
from paralleldomain.model.annotation import (
    Albedo2D,
    AnnotationIdentifier,
    AnnotationTypes,
    BackwardOpticalFlow,
    BoundingBox2D,
    BoundingBox3D,
    BoundingBoxes2D,
    BoundingBoxes3D,
    Depth,
    InstanceSegmentation2D,
    MaterialProperties2D,
    OpticalFlow,
    Point2D,
    Point3D,
    Points2D,
    Points3D,
    SemanticSegmentation2D,
    SurfaceNormals2D,
)
from paralleldomain.model.class_mapping import ClassDetail, ClassMap
from paralleldomain.model.image import Image
from paralleldomain.model.sensor import CameraModel, SensorDataCopyTypes, SensorExtrinsic, SensorIntrinsic, SensorPose
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.coordinate_system import CoordinateSystem
from paralleldomain.utilities.projection import DistortionLookup, fov_to_focal_length
from paralleldomain.utilities.transformation import Transformation


class DataStreamSensorFrameDecoder(SensorFrameDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        frame_id: FrameId,
        settings: DecoderSettings,
        data_accessor: DataStreamDataAccessor,
        is_unordered_scene: bool,
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
        self._data_accessor = data_accessor

    def _decode_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        label_data = self._data_accessor.get_ontology_data(frame_id=self.frame_id)
        ontology = label_data.data_as_semantic_label_map
        semantic_label_map = ontology.semantic_label_map
        pd_class_details = []
        for semantic_id in semantic_label_map:
            c = semantic_label_map[semantic_id]
            class_detail = ClassDetail(
                name=c.label,
                id=int(c.id),
                meta=dict(supercategory="", color={"r": c.color.red, "g": c.color.green, "b": c.color.blue}),
            )
            pd_class_details.append(class_detail)

        class_maps = {
            identifier: ClassMap(classes=pd_class_details) for identifier in self.get_available_annotation_identifiers()
        }
        return class_maps

    def _decode_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        return self._data_accessor.available_annotation_identifiers

    def _decode_metadata(self) -> Dict[str, Any]:
        return dict()

    def _decode_date_time(self) -> datetime:
        return self._data_accessor.get_frame_id_to_date_time_map()[self.frame_id]

    def _decode_extrinsic(self) -> SensorExtrinsic:
        sensor = self._data_accessor.get_sensor(sensor_name=self.sensor_name, frame_id=self.frame_id)
        if isinstance(sensor.pose, Pose6D):
            sensor_to_ego_RFU = SensorExtrinsic(
                quaternion=sensor.pose.rotation, translation=list(sensor.pose.translation)
            )
        else:
            pose = sensor.pose
            if sensor.pose[3, :3].sum() != 0.0:
                # stored poses are transposed for some reason
                pose = pose.T
            sensor_to_ego_RFU = SensorExtrinsic.from_transformation_matrix(mat=pose, approximate_orthogonal=True)

        RDF_to_RFU = CoordinateSystem("RDF") > CoordinateSystem("RFU")
        RFU_to_FLU = CoordinateSystem("RFU") > CoordinateSystem("FLU")

        result = RFU_to_FLU @ sensor_to_ego_RFU @ RDF_to_RFU
        return SensorExtrinsic(quaternion=result.quaternion, translation=result.translation)

    def _decode_sensor_pose(self) -> SensorPose:
        sensor_to_ego = self._decode_extrinsic()
        ego_to_world = self._data_accessor.get_ego_pose(frame_id=self.frame_id)

        result = ego_to_world @ sensor_to_ego
        return SensorPose(quaternion=result.quaternion, translation=result.translation)

    def _decode_annotations(self, identifier: AnnotationIdentifier[T]) -> T:
        if identifier.name is None:
            raise ValueError("AnnotationIdentifiers without name are not supported!")
        annotation_type = identifier.annotation_type

        if issubclass(annotation_type, BoundingBoxes2D):
            label_data = self._data_accessor.get_label_data(
                stream_name=identifier.name,
                sensor_name=self.sensor_name,
                frame_id=self.frame_id,
                file_ending=TYPE_TO_FILE_FORMAT[BoundingBoxes2D],
            )
            return self._decode_bounding_boxes_2d(label_data=label_data)
        elif issubclass(annotation_type, BoundingBoxes3D):
            label_data = self._data_accessor.get_label_data(
                stream_name=identifier.name,
                sensor_name=None,
                frame_id=self.frame_id,
                file_ending=TYPE_TO_FILE_FORMAT[BoundingBoxes3D],
            )
            return self._decode_bounding_boxes_3d(
                label_data=label_data,
                # ego_to_sensor=ego_to_world.inverse,
                world_to_sensor=self.get_sensor_pose().inverse,
            )
        elif issubclass(annotation_type, SemanticSegmentation2D):
            label_data = self._data_accessor.get_label_data(
                stream_name=identifier.name,
                sensor_name=self.sensor_name,
                frame_id=self.frame_id,
                file_ending=TYPE_TO_FILE_FORMAT[SemanticSegmentation2D],
            )
            class_ids = self._decode_segmentation_mask_2d(label_data=label_data)
            return SemanticSegmentation2D(class_ids=class_ids)
        elif issubclass(annotation_type, InstanceSegmentation2D):
            label_data = self._data_accessor.get_label_data(
                stream_name=identifier.name,
                sensor_name=self.sensor_name,
                frame_id=self.frame_id,
                file_ending=TYPE_TO_FILE_FORMAT[InstanceSegmentation2D],
            )
            instance_ids = self._decode_instance_mask_2d(label_data=label_data)
            return InstanceSegmentation2D(instance_ids=instance_ids)
        elif issubclass(annotation_type, OpticalFlow):
            label_data = self._data_accessor.get_label_data(
                stream_name=identifier.name,
                sensor_name=self.sensor_name,
                frame_id=self.frame_id,
                file_ending=TYPE_TO_FILE_FORMAT[OpticalFlow],
            )
            vectors = self._decode_optical_flow(label_data=label_data)
            return OpticalFlow(vectors=vectors)
        elif issubclass(annotation_type, BackwardOpticalFlow):
            label_data = self._data_accessor.get_label_data(
                stream_name=identifier.name,
                sensor_name=self.sensor_name,
                frame_id=self.frame_id,
                file_ending=TYPE_TO_FILE_FORMAT[BackwardOpticalFlow],
            )
            vectors = self._decode_optical_flow(label_data=label_data)
            return BackwardOpticalFlow(vectors=vectors)
        elif issubclass(annotation_type, Depth):
            label_data = self._data_accessor.get_label_data(
                stream_name=identifier.name,
                sensor_name=self.sensor_name,
                frame_id=self.frame_id,
                file_ending=TYPE_TO_FILE_FORMAT[Depth],
            )
            depth_mask = self._decode_depth(label_data=label_data)
            return Depth(depth=depth_mask)
        elif issubclass(annotation_type, SurfaceNormals2D):
            label_data = self._data_accessor.get_label_data(
                stream_name=identifier.name,
                sensor_name=self.sensor_name,
                frame_id=self.frame_id,
                file_ending=TYPE_TO_FILE_FORMAT[SurfaceNormals2D],
            )
            normals = self._decode_surface_normals_2d(label_data=label_data)
            return SurfaceNormals2D(normals=normals)
        elif issubclass(annotation_type, Points2D):
            label_data = self._data_accessor.get_label_data(
                stream_name=identifier.name,
                sensor_name=self.sensor_name,
                frame_id=self.frame_id,
                file_ending=TYPE_TO_FILE_FORMAT[Points2D],
            )
            return self._decode_points_2d(label_data=label_data)
        elif issubclass(annotation_type, Points3D):
            label_data = self._data_accessor.get_label_data(
                stream_name=identifier.name,
                frame_id=self.frame_id,
                sensor_name=None,
                file_ending=TYPE_TO_FILE_FORMAT[Points3D],
            )
            return self._decode_points_3d(label_data=label_data)
        elif issubclass(annotation_type, Albedo2D):
            label_data = self._data_accessor.get_label_data(
                stream_name=identifier.name,
                frame_id=self.frame_id,
                sensor_name=self.sensor_name,
                file_ending=TYPE_TO_FILE_FORMAT[Points3D],
            )
            return Albedo2D(color=label_data.data_as_rgb)
        elif issubclass(annotation_type, MaterialProperties2D):
            label_data = self._data_accessor.get_label_data(
                stream_name=identifier.name,
                frame_id=self.frame_id,
                sensor_name=self.sensor_name,
                file_ending=TYPE_TO_FILE_FORMAT[Points3D],
            )
            return MaterialProperties2D(roughness=label_data.data_as_rgb)

    def _decode_file_path(self, data_type: SensorDataCopyTypes) -> Optional[AnyPath]:
        return self._data_accessor.get_file_path(
            sensor_name=self.sensor_name, frame_id=self.frame_id, data_type=data_type
        )

    def _decode_bounding_boxes_2d(self, label_data: LabelData) -> BoundingBoxes2D:
        annotations = label_data.data_as_annotation
        boxes = [a for a in annotations.geometry.primitives if a.WhichOneof("geometry_oneof") == "box_2d"]
        result = list()
        for box in boxes:
            metadata = VisibilitySampleMetadata()
            success = box.metadata.Unpack(metadata)
            if success is False:
                raise ValueError(
                    f"Expected that box_2d has metadata of type VisibilitySampleMetadata. Got {box.metadata}"
                )
            metadata_dict = json_format.MessageToDict(
                message=metadata, including_default_value_fields=True, preserving_proto_field_name=True
            )
            if metadata_dict["type"] == "eInstanceSample":
                box = BoundingBox2D(
                    x=int(box.box_2d.min.x),
                    y=int(box.box_2d.min.y),
                    width=int(box.box_2d.max.x - box.box_2d.min.x),
                    height=int(box.box_2d.max.y - box.box_2d.min.y),
                    class_id=metadata.semantic_id,
                    instance_id=metadata.instance_id,
                    attributes={k: v for k, v in metadata_dict.items() if k not in {"instance_id", "semantic_id"}},
                )
                if box.width > 0 and box.height > 0:  # TODO: why is this needed?
                    result.append(box)
        return BoundingBoxes2D(boxes=result)

    def _decode_points_2d(self, label_data: LabelData) -> Points2D:
        annotations = label_data.data_as_annotation
        points = [a for a in annotations.geometry.primitives if a.WhichOneof("geometry_oneof") == "point_2d"]
        result = list()
        for point in points:
            metadata = InstancePoint2DMetadata()
            success = point.metadata.Unpack(metadata)
            if success is False:
                raise ValueError(
                    f"Expected that point_2d has metadata of type InstancePoint2DMetadata. Got {point.metadata}"
                )
            metadata_dict = json_format.MessageToDict(
                message=metadata, including_default_value_fields=True, preserving_proto_field_name=True
            )

            point = Point2D(
                x=point.point_2d.x,
                y=point.point_2d.y,
                instance_id=metadata.instance_id,
                class_id=metadata.semantic_id,
                attributes={k: v for k, v in metadata_dict.items() if k != "instance_id"},
            )
            result.append(point)
        return Points2D(points=result)

    def _decode_bounding_boxes_3d(self, label_data: LabelData, world_to_sensor: Transformation) -> BoundingBoxes3D:
        annotations = label_data.data_as_annotation
        boxes = [a for a in annotations.geometry.primitives if a.WhichOneof("geometry_oneof") == "cuboid_3d"]
        result = list()
        RFU_to_FLU = CoordinateSystem("RFU") > CoordinateSystem("FLU")
        world_to_sensor = world_to_sensor @ RFU_to_FLU
        for box in boxes:
            metadata = Cuboid3dMetadata()
            success = box.metadata.Unpack(metadata)
            if success is False:
                raise ValueError(f"Expected that box_3d has metadata of type Cuboid3dMetadata. Got {box.metadata}")

            geometry = box.cuboid_3d
            pose = world_to_sensor @ Transformation(
                quaternion=Quaternion(
                    x=geometry.rotation.x, y=geometry.rotation.y, z=geometry.rotation.z, w=geometry.rotation.w
                ),
                translation=[geometry.translation.x, geometry.translation.y, geometry.translation.z],
            )
            box = BoundingBox3D(
                pose=pose,
                length=geometry.scale.x,  # TODO: this assumes FLU
                width=geometry.scale.y,
                height=geometry.scale.z,
                class_id=metadata.semantic_id,
                instance_id=metadata.instance_id,
                num_points=-1,  # TODO: not available right now
                attributes=dict(),
            )
            result.append(box)
        return BoundingBoxes3D(boxes=result)

    def _decode_points_3d(self, label_data: LabelData) -> Points3D:
        annotations = label_data.data_as_annotation
        points = [a for a in annotations.geometry.primitives if a.WhichOneof("geometry_oneof") == "point_3d"]
        result = list()
        for point in points:
            metadata = InstancePoint3DMetadata()
            success = point.metadata.Unpack(metadata)
            if success is False:
                raise ValueError(
                    f"Expected that point_3d has metadata of type InstancePoint3DMetadata. Got {point.metadata}"
                )
            metadata_dict = json_format.MessageToDict(
                message=metadata, including_default_value_fields=True, preserving_proto_field_name=True
            )
            point = Point3D(
                x=point.point_3d.x,
                y=point.point_3d.y,
                z=point.point_3d.z,
                instance_id=metadata.instance_id,
                class_id=metadata.semantic_id,
                attributes={k: v for k, v in metadata_dict.items() if k != "instance_id"},
            )
            result.append(point)
        return Points3D(points=result)

    def _decode_depth(self, label_data: LabelData) -> np.ndarray:
        return label_data.data_as_depth[:, :, np.newaxis].astype(float)

    def _decode_segmentation_mask_2d(self, label_data: LabelData) -> np.ndarray:
        return label_data.data_as_segmentation_ids[:, :, np.newaxis].astype(int)

    def _decode_instance_mask_2d(self, label_data: LabelData) -> np.ndarray:
        return label_data.data_as_instance_ids[:, :, np.newaxis].astype(int)

    def _decode_optical_flow(self, label_data: LabelData) -> np.ndarray:
        raise NotImplementedError()

    def _decode_surface_normals_2d(self, label_data: LabelData) -> np.ndarray:
        return label_data.data_as_surface_normals


class DataStreamCameraSensorFrameDecoder(DataStreamSensorFrameDecoder, CameraSensorFrameDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        frame_id: FrameId,
        settings: DecoderSettings,
        data_accessor: DataStreamDataAccessor,
        is_unordered_scene: bool,
        scene_decoder,
    ):
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
        DataStreamSensorFrameDecoder.__init__(
            self=self,
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            frame_id=frame_id,
            settings=settings,
            data_accessor=data_accessor,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )

    def _decode_intrinsic(self) -> SensorIntrinsic:
        sensor = self._data_accessor.get_sensor(sensor_name=self.sensor_name, frame_id=self.frame_id)
        if not isinstance(sensor, CameraSensor):
            raise ValueError(f"{self.sensor_name} is not a CameraSensor!")

        if sensor.distortion_params is not None:
            distortion = sensor.distortion_params
            if sensor.fisheye_model == 1:
                camera_model = CameraModel.OPENCV_FISHEYE
            elif sensor.fisheye_model == 3:
                camera_model = CameraModel.PD_FISHEYE
            elif sensor.fisheye_model == 6:
                camera_model = CameraModel.PD_ORTHOGRAPHIC
            elif distortion.is_fisheye is False or sensor.fisheye_model == 0:
                camera_model = CameraModel.OPENCV_PINHOLE
            else:
                camera_model = f"custom_{sensor.fisheye_model}"

            if distortion.fx == 0.0:
                fx = fy = fov_to_focal_length(fov=math.radians(sensor.field_of_view_degrees), sensor_size=sensor.width)
                cx, cy = sensor.width / 2.0, sensor.height / 2.0
            else:
                fx = distortion.fx
                fy = distortion.fy
                cx = distortion.cx
                cy = distortion.cy

            return SensorIntrinsic(
                cx=cx,
                cy=cy,
                fx=fx,
                fy=fy,
                k1=distortion.k1,
                k2=distortion.k2,
                p1=distortion.p1,
                p2=distortion.p2,
                k3=distortion.k3,
                k4=distortion.k4,
                k5=distortion.k5,
                k6=distortion.k6,
                skew=distortion.skew,
                fov=sensor.field_of_view_degrees,
                camera_model=camera_model,
            )
        else:
            return SensorIntrinsic.from_field_of_view(
                field_of_view_degrees=sensor.field_of_view_degrees, width=sensor.width, height=sensor.height
            )

    def _decode_distortion_lookup(self) -> Optional[DistortionLookup]:
        sensor_name, _ = self.sensor_name.rsplit("-", maxsplit=1)
        if self.sensor_name in self.settings.distortion_lookups:
            return self.settings.distortion_lookups[self.sensor_name]
        elif sensor_name in self.settings.distortion_lookups:
            return self.settings.distortion_lookups[sensor_name]
        else:
            return None

    def _decode_image_dimensions(self) -> Tuple[int, int, int]:
        sensor = self._data_accessor.get_sensor(sensor_name=self.sensor_name, frame_id=self.frame_id)
        if not isinstance(sensor, CameraSensor):
            raise ValueError(f"{self.sensor_name} is not a CameraSensor!")

        return sensor.height, sensor.width, 3

    def _decode_image_path(self) -> Optional[AnyPath]:
        return self.get_file_path(data_type=Image)

    def _decode_image_rgba(self) -> np.ndarray:
        label_data = self._data_accessor.get_label_data(
            frame_id=self.frame_id,
            sensor_name=self.sensor_name,
            file_ending=TYPE_TO_FILE_FORMAT[Image],
            stream_name=self._data_accessor.camera_image_stream_name,
        )

        image_data = label_data.data_as_rgb
        ones = np.ones((*image_data.shape[:2], 1), dtype=image_data.dtype) * 255
        concatenated = np.concatenate([image_data, ones], axis=-1)
        return concatenated
