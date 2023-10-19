from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, Optional, Union

import numpy as np
from google.protobuf import json_format
from pd.internal.proto.keystone.generated.python import pd_sensor_pb2
from pd.internal.proto.label_engine.generated.python import (  # telemetry_pb2,
    annotation_pb2,
    bounding_box_2d_pb2,
    bounding_box_3d_pb2,
    data_pb2,
    geometry_pb2,
    instance_point_pb2,
    mesh_map_pb2,
    options_pb2,
    sensor_le_pb2,
    transform_map_pb2,
)
from pyquaternion import Quaternion

from paralleldomain.encoding.dgp.v1.format.common import encode_flow_vectors
from paralleldomain.encoding.pipeline_encoder import DataType as DType
from paralleldomain.encoding.pipeline_encoder import EncodingFormat, ScenePipelineItem, UnorderedScenePipelineItem
from paralleldomain.model.annotation import (
    AnnotationIdentifier,
    AnnotationTypes,
    BoundingBox2D,
    BoundingBox3D,
    Point2D,
    Polygon2D,
    Polyline2D,
)
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.image import Image
from paralleldomain.model.sensor import CameraModel, CameraSensorFrame, LidarSensorFrame, RadarSensorFrame, SensorFrame
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.coordinate_system import CoordinateSystem
from paralleldomain.utilities.fsio import (
    copy_file,
    read_message,
    write_json_message,
    write_message,
    write_npz,
    write_png,
)
from paralleldomain.utilities.transformation import Transformation

_DEFAULT_ONTOLOGY_NAME = "ontology"
_KEYPOINT_ONTOLOGY_NAME = "keypoint_ontology"


class DataStreamEncodingFormat(EncodingFormat[Union[ScenePipelineItem, UnorderedScenePipelineItem]]):
    _fisheye_camera_model_map: Dict[str, int] = defaultdict(
        lambda: 2,
        {
            CameraModel.OPENCV_PINHOLE: 0,
            CameraModel.OPENCV_FISHEYE: 1,
            CameraModel.PD_FISHEYE: 3,
            CameraModel.PD_ORTHOGRAPHIC: 6,
        },
    )

    def __init__(
        self,
        output_path: Union[str, AnyPath],
        camera_image_stream_name: str = "rgb",
        encode_binary: bool = False,
    ):
        self.message_suffix = "json" if not encode_binary else "bin"
        self.camera_image_stream_name = camera_image_stream_name
        self.output_path = AnyPath(output_path)

    def _ensure_agent_id_in_sensor_name(self, source_dataset_format: str, sensor_name: str) -> str:
        if source_dataset_format == self.get_format():
            return sensor_name
        elif source_dataset_format == "step":
            ego_agent_id = sensor_name.split("-")[-1]
            if ego_agent_id.replace(".", "", 1).isdigit():
                return sensor_name
            else:
                return sensor_name + "-0"
        else:
            return sensor_name + "-0"

    def _box_3d_to_proto(self, box: BoundingBox3D, sensor_to_world: Transformation):
        meta = bounding_box_3d_pb2.Cuboid3dMetadata(instance_id=box.instance_id, semantic_id=box.class_id)

        pose = box.pose
        FLU_to_RFU = CoordinateSystem("FLU") > CoordinateSystem("RFU")
        pose = FLU_to_RFU @ sensor_to_world @ pose

        annotation = annotation_pb2.GeometryAnnotation(
            cuboid_3d=geometry_pb2.Cuboid3D(
                translation=self._np_to_vector3(vec=pose.translation),
                rotation=self._convert_quaternion(quaternion=pose.quaternion),
                scale=self._np_to_vector3(vec=np.array([box.length, box.width, box.height])),  # assumes FLU
            )
        )
        annotation.metadata.Pack(meta)
        return annotation

    @staticmethod
    def _polyline_2d_to_proto(polyline: Polyline2D):
        keys = [
            "instance_point_type",
            "instance_point_name",
            "distance_from_ego",
            "visibility",
        ]

        attr = deepcopy(polyline.attributes)
        usr_data = attr.pop("user_data", dict())
        attr.update(usr_data)
        meta = {k: v for k, v in attr.items() if k in keys}
        meta = instance_point_pb2.InstancePoint2DMetadata(instance_id=polyline.instance_id, **meta)

        vertices = [
            geometry_pb2.Vector2(
                x=point.x,
                y=point.y,
            )
            for line in polyline.lines
            for point in [line.start]
        ]
        vertices.append(
            geometry_pb2.Vector2(
                x=polyline.lines[-1].end.x,
                y=polyline.lines[-1].end.y,
            )
        )
        annotation = annotation_pb2.GeometryAnnotation(poly_line_2d=geometry_pb2.PolyLine2D(vertices=vertices))
        annotation.metadata.Pack(meta)
        return annotation

    @staticmethod
    def _polygon_2d_to_proto(polygon: Polygon2D):
        keys = [
            "instance_point_type",
            "instance_point_name",
            "distance_from_ego",
            "visibility",
        ]

        attr = deepcopy(polygon.attributes)
        usr_data = attr.pop("user_data", dict())
        attr.update(usr_data)
        meta = {k: v for k, v in attr.items() if k in keys}
        meta = instance_point_pb2.InstancePoint2DMetadata(instance_id=polygon.instance_id, **meta)

        vertices = [
            geometry_pb2.Vector2(
                x=point.x,
                y=point.y,
            )
            for line in polygon.lines
            for point in [line.start]
        ]
        vertices.append(
            geometry_pb2.Vector2(
                x=polygon.lines[-1].end.x,
                y=polygon.lines[-1].end.y,
            )
        )
        annotation = annotation_pb2.GeometryAnnotation(polygon_2d=geometry_pb2.Polygon2D(vertices=vertices))
        annotation.metadata.Pack(meta)
        return annotation

    @staticmethod
    def _point_2d_to_proto(point: Point2D):
        keys = [
            "instance_point_type",
            "instance_point_name",
            "distance_from_ego",
            "visibility",
        ]

        attr = deepcopy(point.attributes)
        usr_data = attr.pop("user_data", dict())
        attr.update(usr_data)
        meta = {k: v for k, v in attr.items() if k in keys}
        meta = instance_point_pb2.InstancePoint2DMetadata(instance_id=point.instance_id, **meta)
        annotation = annotation_pb2.GeometryAnnotation(
            point_2d=geometry_pb2.Vector2(
                x=point.x,
                y=point.y,
            )
        )
        annotation.metadata.Pack(meta)
        return annotation

    @staticmethod
    def _box_2d_to_proto(box: BoundingBox2D):
        keys = [
            "type",
            "label",
            "mesh_id",
            "submesh_index",
            "material_index",
            "visibility",
            "truncation",
            "num_total_points",
            "num_visible_points",
            "num_on_screen_points",
        ]

        attr = deepcopy(box.attributes)
        usr_data = attr.pop("user_data", dict())
        attr.update(usr_data)
        meta = {k: v for k, v in attr.items() if k in keys}
        meta = bounding_box_2d_pb2.VisibilitySampleMetadata(
            instance_id=box.instance_id, semantic_id=box.class_id, **meta
        )
        annotation = annotation_pb2.GeometryAnnotation(
            box_2d=geometry_pb2.Box2D(
                min=geometry_pb2.Vector2(
                    x=box.x_min,
                    y=box.y_min,
                ),
                max=geometry_pb2.Vector2(
                    x=box.x_max,
                    y=box.y_max,
                ),
            )
        )
        annotation.metadata.Pack(meta)
        return annotation

    def save_data(
        self, pipeline_item: Union[ScenePipelineItem, UnorderedScenePipelineItem], data_type: DType, data: Any
    ):
        sensor_name = self._ensure_agent_id_in_sensor_name(
            sensor_name=pipeline_item.sensor_name, source_dataset_format=pipeline_item.dataset_format
        )
        # TODO: InstanceSegmentation3D, SceneFlow, SemanticSegmentation3D, SurfaceNormals3D
        # TODO: Point3D, Polygon3D, Polyline3D, PointCloud, RadarPointCloud

        if isinstance(data_type, AnnotationIdentifier):
            ontology_name_map = self._get_ontology_name_map(class_maps=pipeline_item.sensor_frame.class_maps)
            ontology_name = ontology_name_map.get(data_type, "")
            if data_type.annotation_type == AnnotationTypes.BoundingBoxes2D:
                path = self._get_storage_path(
                    scene_name=pipeline_item.scene_name,
                    frame_id=pipeline_item.frame_id,
                    stream_name=data_type.name if data_type.name is not None else "bounding_box_2d_xyz",
                    sensor_name=sensor_name,
                    file_ending=f"pb.{self.message_suffix}",
                    create_folders=True,
                    data_type=options_pb2.DataType.eAnnotation,
                    ontology_name=ontology_name,
                )
                if isinstance(data, AnnotationTypes.BoundingBoxes2D):
                    write_message(
                        obj=annotation_pb2.Annotation(
                            geometry=annotation_pb2.GeometryCollection(
                                primitives=[self._box_2d_to_proto(box=box) for box in data.boxes]
                            )
                        ),
                        path=path,
                    )
            elif data_type.annotation_type == AnnotationTypes.BoundingBoxes3D:
                path = self._get_storage_path(
                    scene_name=pipeline_item.scene_name,
                    frame_id=pipeline_item.frame_id,
                    stream_name=data_type.name if data_type.name is not None else "bounding_box_3d_xyz",
                    sensor_name=sensor_name,
                    file_ending=f"pb.{self.message_suffix}",
                    create_folders=True,
                    data_type=options_pb2.DataType.eAnnotation,
                    aggregation_folder=True,
                    aggregation_folder_name="aggregate_bbox_3d",
                    ontology_name=ontology_name,
                )
                if isinstance(data, AnnotationTypes.BoundingBoxes3D):
                    sensor_to_world = pipeline_item.sensor_frame.sensor_to_world
                    primitives = [self._box_3d_to_proto(box=box, sensor_to_world=sensor_to_world) for box in data.boxes]
                    obj = annotation_pb2.Annotation(geometry=annotation_pb2.GeometryCollection(primitives=primitives))
                    write_message(
                        obj=obj,
                        path=path,
                    )
            elif data_type.annotation_type == AnnotationTypes.Depth:
                path = self._get_storage_path(
                    scene_name=pipeline_item.scene_name,
                    frame_id=pipeline_item.frame_id,
                    stream_name=data_type.name if data_type.name is not None else "depth",
                    sensor_name=sensor_name,
                    file_ending="npz",
                    create_folders=True,
                    data_type=options_pb2.DataType.eImage,
                )
                if isinstance(data, AnnotationTypes.Depth):
                    write_npz(obj=dict(data=data.depth[..., 0].astype(np.float32)), path=path)
            elif data_type.annotation_type == AnnotationTypes.Albedo2D:
                path = self._get_storage_path(
                    scene_name=pipeline_item.scene_name,
                    frame_id=pipeline_item.frame_id,
                    stream_name=data_type.name if data_type.name is not None else "base",
                    sensor_name=sensor_name,
                    file_ending="png",
                    create_folders=True,
                    data_type=options_pb2.DataType.eImage,
                )
                if isinstance(data, AnnotationTypes.Albedo2D):
                    write_png(obj=data.color, path=path)
            elif data_type.annotation_type == AnnotationTypes.MaterialProperties2D:
                path = self._get_storage_path(
                    scene_name=pipeline_item.scene_name,
                    frame_id=pipeline_item.frame_id,
                    stream_name=data_type.name if data_type.name is not None else "material",
                    sensor_name=sensor_name,
                    file_ending="png",
                    create_folders=True,
                    data_type=options_pb2.DataType.eImage,
                )
                if isinstance(data, AnnotationTypes.MaterialProperties2D):
                    write_png(obj=data.roughness, path=path)
            elif data_type.annotation_type == AnnotationTypes.InstanceSegmentation2D:
                path = self._get_storage_path(
                    scene_name=pipeline_item.scene_name,
                    frame_id=pipeline_item.frame_id,
                    stream_name=data_type.name if data_type.name is not None else "instance_mask",
                    sensor_name=sensor_name,
                    file_ending="png",
                    create_folders=True,
                    data_type=options_pb2.DataType.eImage,
                    ontology_name=ontology_name,
                )
                if isinstance(data, AnnotationTypes.InstanceSegmentation2D):
                    class_ids = data.instance_ids.astype(np.uint16).view(np.uint8)
                    pad_channel = np.ones_like(class_ids)
                    pad_channel[..., 0] *= 0  # blue channel to 0
                    pad_channel[..., 1] *= 255  # alpha channel to 1 so that image can be previewed on disk
                    stacked = np.concatenate([class_ids, pad_channel], axis=-1)
                    write_png(obj=stacked, path=path)
            elif data_type.annotation_type == AnnotationTypes.OpticalFlow:
                path = self._get_storage_path(
                    scene_name=pipeline_item.scene_name,
                    frame_id=pipeline_item.frame_id,
                    stream_name=data_type.name if data_type.name is not None else "motion_vectors",
                    sensor_name=sensor_name,
                    file_ending="png",
                    create_folders=True,
                    data_type=options_pb2.DataType.eImage,
                )
                # TODO: How to encode valid masks?
                if isinstance(data, AnnotationTypes.OpticalFlow):
                    write_png(obj=encode_flow_vectors(data.vectors), path=path)
            elif data_type.annotation_type == AnnotationTypes.BackwardOpticalFlow:
                path = self._get_storage_path(
                    scene_name=pipeline_item.scene_name,
                    frame_id=pipeline_item.frame_id,
                    stream_name=data_type.name if data_type.name is not None else "backward_motion_vectors",
                    sensor_name=sensor_name,
                    file_ending="png",
                    create_folders=True,
                    data_type=options_pb2.DataType.eImage,
                )
                # TODO: How to encode valid masks?
                if isinstance(data, AnnotationTypes.BackwardOpticalFlow):
                    write_png(obj=encode_flow_vectors(data.vectors), path=path)
            elif data_type.annotation_type == AnnotationTypes.Points2D:
                path = self._get_storage_path(
                    scene_name=pipeline_item.scene_name,
                    frame_id=pipeline_item.frame_id,
                    stream_name=data_type.name if data_type.name is not None else "instance_points_2d",
                    sensor_name=sensor_name,
                    file_ending=f"pb.{self.message_suffix}",
                    create_folders=True,
                    data_type=options_pb2.DataType.eAnnotation,
                    ontology_name=ontology_name,
                )
                if isinstance(data, AnnotationTypes.Points2D):
                    write_message(
                        obj=annotation_pb2.Annotation(
                            geometry=annotation_pb2.GeometryCollection(
                                primitives=[self._point_2d_to_proto(point=point) for point in data.points]
                            )
                        ),
                        path=path,
                    )
            elif data_type.annotation_type == AnnotationTypes.Polygons2D:
                path = self._get_storage_path(
                    scene_name=pipeline_item.scene_name,
                    frame_id=pipeline_item.frame_id,
                    stream_name=data_type.name if data_type.name is not None else "polygons_2d",
                    sensor_name=sensor_name,
                    file_ending=f"pb.{self.message_suffix}",
                    create_folders=True,
                    data_type=options_pb2.DataType.eAnnotation,
                )
                if isinstance(data, AnnotationTypes.Polygons2D):
                    write_message(
                        obj=annotation_pb2.Annotation(
                            geometry=annotation_pb2.GeometryCollection(
                                primitives=[self._polygon_2d_to_proto(polygon=polygon) for polygon in data.polygons]
                            )
                        ),
                        path=path,
                    )
            elif data_type.annotation_type == AnnotationTypes.Polylines2D:
                path = self._get_storage_path(
                    scene_name=pipeline_item.scene_name,
                    frame_id=pipeline_item.frame_id,
                    stream_name=data_type.name if data_type.name is not None else "polylines_2d",
                    sensor_name=sensor_name,
                    file_ending=f"pb.{self.message_suffix}",
                    create_folders=True,
                    data_type=options_pb2.DataType.eAnnotation,
                )
                if isinstance(data, AnnotationTypes.Polylines2D):
                    write_message(
                        obj=annotation_pb2.Annotation(
                            geometry=annotation_pb2.GeometryCollection(
                                primitives=[
                                    self._polyline_2d_to_proto(polyline=polyline) for polyline in data.polylines
                                ]
                            )
                        ),
                        path=path,
                    )
            elif data_type.annotation_type == AnnotationTypes.SemanticSegmentation2D:
                path = self._get_storage_path(
                    scene_name=pipeline_item.scene_name,
                    frame_id=pipeline_item.frame_id,
                    stream_name=data_type.name if data_type.name is not None else "semantic_mask",
                    sensor_name=sensor_name,
                    file_ending="png",
                    create_folders=True,
                    data_type=options_pb2.DataType.eImage,
                    ontology_name=ontology_name,
                )
                if isinstance(data, AnnotationTypes.SemanticSegmentation2D):
                    class_ids = data.class_ids.astype(np.uint16).view(np.uint8)
                    pad_channel = np.ones_like(class_ids)
                    pad_channel[..., 0] *= 0  # blue channel to 0
                    pad_channel[..., 1] *= 255  # alpha channel to 1 so that image can be previewed on disk
                    stacked = np.concatenate([class_ids, pad_channel], axis=-1)
                    write_png(obj=stacked, path=path)
        elif data_type == Image:
            path = self._get_storage_path(
                scene_name=pipeline_item.scene_name,
                frame_id=pipeline_item.frame_id,
                stream_name=self.camera_image_stream_name,
                sensor_name=sensor_name,
                file_ending="png",
                create_folders=True,
                data_type=options_pb2.DataType.eImage,
            )
            if isinstance(data, np.ndarray):
                write_png(obj=data, path=path)

        if isinstance(data, AnyPath):
            copy_file(source=data, target=path)

    def _get_aggregation_folder_path(self, scene_name: str, aggregation_folder_name: str):
        return self.output_path / scene_name / aggregation_folder_name

    def _get_storage_path(
        self,
        scene_name: str,
        frame_id: str,
        stream_name: str,
        sensor_name: Optional[str],
        file_ending: str,
        data_type: options_pb2.DataType,
        create_folders: bool = False,
        aggregation_folder: bool = False,
        aggregation_folder_name: str = "aggregation",
        ontology_name: str = "",
    ) -> AnyPath:
        frame_id = f"{int(frame_id):09d}"

        file_name = f"{frame_id}.{file_ending}"
        scene_folder = self.output_path / scene_name
        if aggregation_folder:
            scene_folder = self._get_aggregation_folder_path(
                scene_name=scene_name, aggregation_folder_name=aggregation_folder_name
            )

        if sensor_name is not None:
            path = scene_folder / stream_name / sensor_name / file_name
        else:
            path = scene_folder / stream_name / file_name

        if create_folders is True:
            path.parent.mkdir(parents=True, exist_ok=True)
            if sensor_name is not None:
                self._write_type_file(folder_path=scene_folder / stream_name, data_type=options_pb2.DataType.eNone)
                self._write_type_file(
                    folder_path=scene_folder / stream_name / sensor_name,
                    data_type=data_type,
                    ontology_name=ontology_name,
                )
            else:
                self._write_type_file(
                    folder_path=scene_folder / stream_name,
                    data_type=data_type,
                    ontology_name=ontology_name,
                )

        return path

    @staticmethod
    def _write_type_file(data_type: options_pb2.DataType, folder_path: AnyPath, ontology_name: str = ""):
        type_file = folder_path / ".type"
        if not type_file.exists():
            type_file.parent.mkdir(parents=True, exist_ok=True)
            write_json_message(obj=data_pb2.DataTypeRecord(type=data_type, ontology=ontology_name), path=type_file)

    def supports_copy(
        self,
        pipeline_item: Union[ScenePipelineItem, UnorderedScenePipelineItem],
        data_type: options_pb2.DataType,
        data_path: AnyPath,
    ):
        if pipeline_item.dataset_format == "data-stream":
            return True
        return False

    def save_sensor_frame(self, pipeline_item: Union[ScenePipelineItem, UnorderedScenePipelineItem], data: Any = None):
        self._write_sensor_rig_config(pipeline_item=pipeline_item)

    def save_frame(self, pipeline_item: Union[ScenePipelineItem, UnorderedScenePipelineItem], data: Any = None):
        self._write_type_file(
            folder_path=self.output_path / pipeline_item.scene_name, data_type=options_pb2.DataType.eNone
        )
        self._write_telemetry(pipeline_item=pipeline_item)
        self._aggregate_3d_boxes(pipeline_item=pipeline_item)
        self._write_ontologies(pipeline_item=pipeline_item)

    def _aggregate_3d_boxes(self, pipeline_item: Union[ScenePipelineItem, UnorderedScenePipelineItem]):
        aggregation_folder = self._get_aggregation_folder_path(
            scene_name=pipeline_item.scene_name, aggregation_folder_name="aggregate_bbox_3d"
        )
        ontology_name = ""
        if aggregation_folder.exists():
            for bbox_3d_stream in aggregation_folder.iterdir():
                if bbox_3d_stream.is_dir():
                    primitives = list()
                    existing_primitives_instance_ids = set()
                    for sensor_stream in bbox_3d_stream.iterdir():
                        if sensor_stream.is_dir():
                            frame_file = sensor_stream / f"{int(pipeline_item.frame_id):09d}.pb.{self.message_suffix}"
                            obj = read_message(obj=annotation_pb2.Annotation(), path=frame_file)
                            for box in obj.geometry.primitives:
                                metadata = bounding_box_3d_pb2.Cuboid3dMetadata()
                                box.metadata.Unpack(metadata)
                                if metadata.instance_id not in existing_primitives_instance_ids:
                                    existing_primitives_instance_ids.add(metadata.instance_id)
                                    primitives.append(box)
                            frame_file.rm(missing_ok=True)
                            # We assume ontology is the same across sensors
                            if ontology_name == "":
                                ontology_name = get_type_file_ontology(folder=sensor_stream)

                    path = self._get_storage_path(
                        scene_name=pipeline_item.scene_name,
                        frame_id=pipeline_item.frame_id,
                        stream_name=bbox_3d_stream.name,
                        sensor_name=None,
                        file_ending=f"pb.{self.message_suffix}",
                        create_folders=True,
                        data_type=options_pb2.DataType.eAnnotation,
                        ontology_name=ontology_name,
                    )
                    obj = annotation_pb2.Annotation(geometry=annotation_pb2.GeometryCollection(primitives=primitives))
                    write_message(
                        obj=obj,
                        path=path,
                    )

    def _write_sensor_rig_config(self, pipeline_item: Union[ScenePipelineItem, UnorderedScenePipelineItem]):
        sensor_frame = pipeline_item.sensor_frame
        name_and_id = self._ensure_agent_id_in_sensor_name(
            sensor_name=sensor_frame.sensor_name, source_dataset_format=pipeline_item.dataset_format
        )
        sensor_name, agent_id = name_and_id.rsplit(sep="-", maxsplit=1)
        path = self._get_storage_path(
            scene_name=pipeline_item.scene_name,
            frame_id=pipeline_item.frame_id,
            stream_name="sensor",
            sensor_name=agent_id,
            file_ending="pb.json",
            create_folders=True,
            data_type=options_pb2.DataType.eSensor,
        )
        write_message(
            obj=sensor_le_pb2.SensorRigConfigLE(
                sensor_rig_artifact_uid="",
                default_sensor_splits_list=[],
                sensor_configs=[self._sensor_to_proto(sensor_frame=sensor_frame, sensor_name_wo_agent_id=sensor_name)],
            ),
            path=path,
        )

    def _write_telemetry(self, pipeline_item: Union[ScenePipelineItem, UnorderedScenePipelineItem]):
        path = self._get_storage_path(
            scene_name=pipeline_item.scene_name,
            frame_id=pipeline_item.frame_id,
            stream_name="ego_telemetry",
            sensor_name=None,
            file_ending=f"pb.{self.message_suffix}",
            create_folders=True,
            data_type=options_pb2.DataType.eTransformMap,
        )
        frame = pipeline_item.frame

        ego_to_world = frame.ego_frame.pose
        RFU_to_FLU = CoordinateSystem("RFU") > CoordinateSystem("FLU")
        ego_to_world = RFU_to_FLU.inverse @ ego_to_world @ RFU_to_FLU

        write_message(
            obj=transform_map_pb2.TransformMap(
                actor_transform_map={0: self._convert_transformation(trans=ego_to_world)}
                # fixed_frame=self._convert_transformation(trans=Transformation.from_transformation_matrix(np.eye(4))),
                # localization=[
                #     telemetry_pb2.TelemetryValue(
                #         timestamp=int(frame.frame_id),
                #         transformation_pose=self._convert_transformation(trans=ego_to_world),
                #     )
                # ],
            ),
            path=path,
        )

    def _write_ontologies(self, pipeline_item: Union[ScenePipelineItem, UnorderedScenePipelineItem]):
        scene = pipeline_item.scene
        annotation_identifiers = scene.available_annotation_identifiers
        ontology_name_map = self._get_ontology_name_map(class_maps=scene.class_maps)

        for annotation_identifier in annotation_identifiers:
            if annotation_identifier in scene.class_maps:
                class_map = scene.class_maps[annotation_identifier]
                semantic_label_map = {}
                for class_id, class_detail in class_map.items():
                    red = class_detail.meta["color"]["r"] if "color" in class_detail.meta else 0
                    green = class_detail.meta["color"]["g"] if "color" in class_detail.meta else 0
                    blue = class_detail.meta["color"]["b"] if "color" in class_detail.meta else 0
                    semantic_label_map[class_id] = mesh_map_pb2.SemanticLabelInfo(
                        id=class_detail.id,
                        label=class_detail.name,
                        color=mesh_map_pb2.Color(
                            red=red,
                            green=green,
                            blue=blue,
                        ),
                    )
                if len(semantic_label_map) > 0:
                    ontology_name = ontology_name_map.get(annotation_identifier, "")
                    path = self._get_storage_path(
                        scene_name=pipeline_item.scene_name,
                        frame_id=pipeline_item.frame_id,
                        stream_name=ontology_name,
                        sensor_name=None,
                        file_ending="pb.json",
                        create_folders=True,
                        data_type=options_pb2.DataType.eSemanticLabelMap,
                        ontology_name=ontology_name,
                    )
                    write_message(
                        obj=mesh_map_pb2.SemanticLabelMap(
                            semantic_label_map=semantic_label_map,
                        ),
                        path=path,
                    )

    def _sensor_to_proto(self, sensor_frame: SensorFrame, sensor_name_wo_agent_id: str) -> sensor_le_pb2.SensorConfigLE:
        transform = sensor_frame.sensor_to_ego

        RDF_to_RFU = CoordinateSystem("RDF") > CoordinateSystem("RFU")
        RFU_to_FLU = CoordinateSystem("RFU") > CoordinateSystem("FLU")
        transform = RFU_to_FLU.inverse @ transform @ RDF_to_RFU.inverse

        extrinsic = sensor_le_pb2.SensorExtrinsicMatrix(
            sensor_to_vehicle=transform.transformation_matrix.T.reshape(-1).tolist(),
            lock_to_yaw=False,
            attach_socket="",
            follow_rotation=False,
        )
        if isinstance(sensor_frame, CameraSensorFrame):
            intr = sensor_frame.intrinsic
            image = sensor_frame.image
            return sensor_le_pb2.SensorConfigLE(
                display_name=sensor_name_wo_agent_id,
                sensor_extrinsic=extrinsic,
                camera_intrinsic=pd_sensor_pb2.CameraIntrinsic(
                    fov=intr.fov,
                    height=image.height,
                    width=image.width,
                    distortion_params=pd_sensor_pb2.DistortionParams(
                        fx=intr.fx,
                        fy=intr.fy,
                        cx=intr.cx,
                        cy=intr.cy,
                        skew=intr.skew,
                        k1=intr.k1,
                        k2=intr.k2,
                        k3=intr.k3,
                        k4=intr.k4,
                        k5=intr.k5,
                        k6=intr.k6,
                        p1=intr.p1,
                        p2=intr.p2,
                        is_fisheye=intr.camera_model in [CameraModel.PD_FISHEYE, CameraModel.OPENCV_FISHEYE],
                        fisheye_model=self._fisheye_camera_model_map[intr.camera_model],
                    ),
                ),
            )
        elif isinstance(sensor_frame, LidarSensorFrame):
            return sensor_le_pb2.SensorConfigLE(
                display_name=sensor_name_wo_agent_id,
                sensor_extrinsic=extrinsic,
                lidar_intrinsic=pd_sensor_pb2.LidarIntrinsic(),
            )
        elif isinstance(sensor_frame, RadarSensorFrame):
            return sensor_le_pb2.SensorConfigLE(
                display_name=sensor_name_wo_agent_id,
                sensor_extrinsic=extrinsic,
                RadarIntrinsic=pd_sensor_pb2.RadarIntrinsic(),
            )

    @staticmethod
    def _get_ontology_name_map(class_maps: Dict[AnnotationIdentifier, ClassMap]) -> Dict[AnnotationIdentifier, str]:
        # Group class_maps by class_names, to write one ontology data stream for the entire group
        rev_class_maps = {}
        for identifier, class_map in class_maps.items():
            class_names_and_ids = [f"{c.name + str(c.id)}" for c in class_map.class_details]
            rev_class_maps.setdefault(frozenset(class_names_and_ids), set()).add(identifier)
        class_map_groups = [values for key, values in rev_class_maps.items() if len(values) > 1]
        class_map_groups.sort(key=len, reverse=True)

        ontology_name_map = {}
        for i, class_map_group in enumerate(class_map_groups):
            if i == 0:
                # We assign default name to the largest class map group.
                ontology_name = _DEFAULT_ONTOLOGY_NAME
            else:
                # Check if group is keypoint ontology
                is_key_point_ontology = False
                for identifier in class_map_group:
                    if "instance_points" in identifier.name:
                        is_key_point_ontology = True
                        break
                # be consistent with batch keypoint ontology naming
                if is_key_point_ontology is True:
                    ontology_name = _KEYPOINT_ONTOLOGY_NAME
                else:
                    ontology_name = f"{list(class_map_group)[0].name}_{_DEFAULT_ONTOLOGY_NAME}"
                # Avoid duplicate ontology names
                if ontology_name in ontology_name_map.values():
                    ontology_name = f"{ontology_name}_{i:02d}"

            for identifier in class_map_group:
                ontology_name_map[identifier] = ontology_name

        single_annotation_identifiers = [list(values)[0] for key, values in rev_class_maps.items() if len(values) == 1]
        for annotation_identifier in single_annotation_identifiers:
            ontology_name_map[annotation_identifier] = f"{annotation_identifier.name}_{_DEFAULT_ONTOLOGY_NAME}"
        return ontology_name_map

    @staticmethod
    def _np_to_vector3(vec: np.ndarray) -> geometry_pb2.Vector3:
        vec = vec.reshape(-1)
        return geometry_pb2.Vector3(
            x=float(vec[0]),
            y=float(vec[1]),
            z=float(vec[2]),
        )

    @staticmethod
    def _convert_quaternion(quaternion: Quaternion) -> geometry_pb2.Quaternion:
        return geometry_pb2.Quaternion(
            w=quaternion.w,
            x=quaternion.x,
            y=quaternion.y,
            z=quaternion.z,
        )

    @staticmethod
    def _convert_transformation(trans: Transformation) -> geometry_pb2.Transform:
        return geometry_pb2.Transform(
            scale=geometry_pb2.Vector3(
                x=1.0,
                y=1.0,
                z=1.0,
            ),
            translation=DataStreamEncodingFormat._np_to_vector3(trans.translation),
            orientation=DataStreamEncodingFormat._convert_quaternion(trans.quaternion),
        )

    def save_scene(self, pipeline_item: Union[ScenePipelineItem, UnorderedScenePipelineItem], data: Any = None):
        aggregation_folder = self._get_aggregation_folder_path(
            scene_name=pipeline_item.scene_name, aggregation_folder_name="aggregate_bbox_3d"
        )
        if aggregation_folder.exists():
            for path in list(reversed(list(aggregation_folder.rglob("*")))):
                path.rm(missing_ok=True)
            if aggregation_folder.exists():
                aggregation_folder.rmdir()

    def save_dataset(self, pipeline_item: Union[ScenePipelineItem, UnorderedScenePipelineItem], data: Any = None):
        pass

    @staticmethod
    def get_format() -> str:
        return "data-stream"


@lru_cache(maxsize=1000)
def get_type_file_ontology(folder: AnyPath) -> Optional[str]:
    type_file = folder / ".type"
    if not folder.is_dir() or not type_file.exists():
        return None
    with type_file.open("r") as fp:
        data_type_record = json_format.Parse(text=fp.read(), message=data_pb2.DataTypeRecord())
    return data_type_record.ontology
