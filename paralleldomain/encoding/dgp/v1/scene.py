import concurrent
import hashlib
import logging
import random
import time
import uuid
from collections import defaultdict
from concurrent.futures import Future
from datetime import datetime
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
from google.protobuf import timestamp_pb2

from paralleldomain import Scene
from paralleldomain.common.dgp.v1 import (
    annotations_pb2,
    geometry_pb2,
    identifiers_pb2,
    image_pb2,
    point_cloud_pb2,
    sample_pb2,
    scene_pb2,
)
from paralleldomain.common.dgp.v1.constants import ANNOTATION_TYPE_MAP_INV, DirectoryName, PointFormat
from paralleldomain.common.dgp.v1.utils import datetime_to_timestamp
from paralleldomain.decoding.dgp.decoder import DGPDatasetDecoder
from paralleldomain.encoding.dgp.v1.transformer import (
    BoundingBox2DTransformer,
    BoundingBox3DTransformer,
    InstanceSegmentation2DTransformer,
    InstanceSegmentation3DTransformer,
    KeyLine2DTransformer,
    KeyPoint2DTransformer,
    Polygon2DTransformer,
    SemanticSegmentation2DTransformer,
    SemanticSegmentation3DTransformer,
)
from paralleldomain.encoding.dgp.v1.utils import _attribute_key_dump, _attribute_value_dump, class_map_to_ontology_proto
from paralleldomain.encoding.encoder import ENCODING_THREAD_POOL, SceneEncoder
from paralleldomain.model.annotation import (
    Annotation,
    AnnotationType,
    AnnotationTypes,
    BoundingBox2D,
    BoundingBox3D,
    Point2D,
    Polygon2D,
    Polyline2D,
)
from paralleldomain.model.dataset import Dataset
from paralleldomain.model.sensor import CameraModel, CameraSensorFrame, LidarSensorFrame, SensorFrame
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.mask import encode_2int16_as_rgba8

logger = logging.getLogger(__name__)


class DGPSceneEncoder(SceneEncoder):
    _fisheye_camera_model_map: Dict[str, int] = defaultdict(
        lambda: 2,
        {
            CameraModel.OPENCV_PINHOLE: 0,
            CameraModel.OPENCV_FISHEYE: 1,
        },
    )

    def __init__(
        self,
        dataset: Dataset,
        scene_name: str,
        output_path: AnyPath,
        camera_names: Optional[Union[List[str], None]] = None,
        lidar_names: Optional[Union[List[str], None]] = None,
        annotation_types: Optional[Union[List[AnnotationType], None]] = None,
    ):
        super().__init__(
            dataset=dataset,
            scene_name=scene_name,
            output_path=output_path,
            camera_names=camera_names,
            lidar_names=lidar_names,
            annotation_types=annotation_types,
        )

        self._scene: Scene = self._unordered_scene
        self._reference_timestamp: datetime = self._scene.get_frame(self._scene.frame_ids[0]).date_time
        self._sim_offset: float = 0.01 * 5  # sim timestep * offset count ; unit: seconds

    def _offset_timestamp(self, compare_datetime: datetime) -> float:
        diff = compare_datetime - self._reference_timestamp
        return diff.total_seconds()

    def _process_rgb(self, sensor_frame: CameraSensorFrame[datetime], fs_copy: bool = False) -> Future:
        output_path = (
            self._output_path
            / DirectoryName.RGB
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.png"  # noqa: E501
        )

        if fs_copy and isinstance(self._dataset._decoder, DGPDatasetDecoder):
            input_path = (
                self._dataset._decoder._dataset_path
                / self._scene.name
                / DirectoryName.RGB
                / sensor_frame.sensor_name
                / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.png"  # noqa: E501
            )
            return self._run_async(func=fsio.copy_file, source=input_path, target=output_path)
        else:
            return self._encode_rgb(sensor_frame=sensor_frame, output_path=output_path)

    def _encode_rgb(self, sensor_frame: CameraSensorFrame[datetime], output_path: AnyPath) -> Future:
        return self._run_async(func=fsio.write_png, obj=sensor_frame.image.rgba, path=output_path)

    def _process_point_cloud(self, sensor_frame: LidarSensorFrame[datetime], fs_copy: bool = False) -> Future:
        output_path = (
            self._output_path
            / DirectoryName.POINT_CLOUD
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.npz"  # noqa: E501
        )

        if fs_copy and isinstance(self._dataset._decoder, DGPDatasetDecoder):
            input_path = (
                self._dataset._decoder._dataset_path
                / self._scene.name
                / DirectoryName.POINT_CLOUD
                / sensor_frame.sensor_name
                / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.npz"  # noqa: E501
            )
            return self._run_async(func=fsio.copy_file, source=input_path, target=output_path)
        else:
            return self._encode_point_cloud(sensor_frame=sensor_frame, output_path=output_path)

    def _encode_point_cloud(self, sensor_frame: LidarSensorFrame[datetime], output_path: AnyPath) -> Future:
        pc = sensor_frame.point_cloud
        pc_dtypes = [
            ("X", "<f4"),
            ("Y", "<f4"),
            ("Z", "<f4"),
            ("INTENSITY", "<f4"),
            ("R", "<f4"),
            ("G", "<f4"),
            ("B", "<f4"),
            ("RING_ID", "<u4"),
            ("TIMESTAMP", "<u8"),
        ]

        row_count = pc.length
        pc_data = np.empty(row_count, dtype=pc_dtypes)

        pc_data["X"] = pc.xyz[:, 0]
        pc_data["Y"] = pc.xyz[:, 1]
        pc_data["Z"] = pc.xyz[:, 2]
        pc_data["INTENSITY"] = pc.intensity[:, 0]
        pc_data["R"] = pc.rgb[:, 0]
        pc_data["G"] = pc.rgb[:, 1]
        pc_data["B"] = pc.rgb[:, 2]
        pc_data["RING_ID"] = pc.ring[:, 0]
        pc_data["TIMESTAMP"] = pc.ts[:, 0]

        return self._run_async(func=fsio.write_npz, obj={"data": pc_data}, path=output_path)

    def _process_depth(self, sensor_frame: SensorFrame[datetime], fs_copy: bool = False) -> Future:
        output_path = (
            self._output_path
            / DirectoryName.DEPTH
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.npz"  # noqa: E501
        )

        if fs_copy and isinstance(self._dataset._decoder, DGPDatasetDecoder):
            input_path = (
                self._dataset._decoder._dataset_path
                / self._scene.name
                / DirectoryName.DEPTH
                / sensor_frame.sensor_name
                / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.npz"  # noqa: E501
            )

            return self._run_async(func=fsio.copy_file, source=input_path, target=output_path)
        else:
            return self._encode_depth(sensor_frame=sensor_frame, output_path=output_path)

    def _encode_depth(self, sensor_frame: SensorFrame[datetime], output_path: AnyPath) -> Union[Future, None]:
        try:
            depth = sensor_frame.get_annotations(AnnotationTypes.Depth)
            return self._run_async(func=fsio.write_npz, obj=dict(data=depth.depth[..., 0]), path=output_path)
        except ValueError:  # Some renderings can exclude LiDAR from having Depth annotations
            return None

    def _encode_bounding_box_2d(self, box: BoundingBox2D) -> annotations_pb2.BoundingBox2DAnnotation:
        try:
            is_crowd = box.attributes["iscrowd"]
        except KeyError:
            is_crowd = False
        box_proto = annotations_pb2.BoundingBox2DAnnotation(
            class_id=box.class_id,
            instance_id=box.instance_id,
            area=box.area,
            iscrowd=is_crowd,
            attributes={
                _attribute_key_dump(k): _attribute_value_dump(v) for k, v in box.attributes.items() if k != "iscrowd"
            },
            box=annotations_pb2.BoundingBox2D(x=box.x, y=box.y, w=box.width, h=box.height),
        )

        return box_proto

    def _encode_bounding_boxes_2d(self, sensor_frame: CameraSensorFrame[datetime]) -> Future:
        boxes2d = sensor_frame.get_annotations(AnnotationTypes.BoundingBoxes2D)
        box2d_dto = BoundingBox2DTransformer.transform(objects=[self._encode_bounding_box_2d(b) for b in boxes2d.boxes])
        boxes2d_dto = annotations_pb2.BoundingBox2DAnnotations(annotations=box2d_dto)

        output_path = (
            self._output_path
            / DirectoryName.BOUNDING_BOX_2D
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time)+self._sim_offset)*100):018d}.json"  # noqa: E501
        )
        return self._run_async(func=fsio.write_json_message, obj=boxes2d_dto, path=output_path, append_sha1=True)

    def _encode_bounding_box_3d(self, box: BoundingBox3D) -> annotations_pb2.BoundingBox3DAnnotation:
        try:
            occlusion = box.attributes["occlusion"]
        except KeyError:
            occlusion = 0

        try:
            truncation = box.attributes["truncation"]
        except KeyError:
            truncation = 0

        box_proto = annotations_pb2.BoundingBox3DAnnotation(
            class_id=box.class_id,
            instance_id=box.instance_id,
            num_points=box.num_points,
            attributes={
                _attribute_key_dump(k): _attribute_value_dump(v)
                for k, v in box.attributes.items()
                if k not in ("occlusion", "truncation")
            },
            box=annotations_pb2.BoundingBox3D(
                width=box.width,
                length=box.length,
                height=box.height,
                occlusion=occlusion,
                truncation=truncation,
                pose=geometry_pb2.Pose(
                    translation=geometry_pb2.Vector3(
                        x=box.pose.translation[0], y=box.pose.translation[1], z=box.pose.translation[2]
                    ),
                    rotation=geometry_pb2.Quaternion(
                        qw=box.pose.quaternion.w,
                        qx=box.pose.quaternion.x,
                        qy=box.pose.quaternion.y,
                        qz=box.pose.quaternion.z,
                    ),
                ),
            ),
        )

        return box_proto

    def _encode_bounding_boxes_3d(self, sensor_frame: SensorFrame[datetime]) -> Future:
        boxes3d = sensor_frame.get_annotations(AnnotationTypes.BoundingBoxes3D)
        box3d_dto = BoundingBox3DTransformer.transform(objects=[self._encode_bounding_box_3d(b) for b in boxes3d.boxes])
        boxes3d_dto = annotations_pb2.BoundingBox3DAnnotations(annotations=box3d_dto)

        output_path = (
            self._output_path
            / DirectoryName.BOUNDING_BOX_3D
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time)+self._sim_offset)*100):018d}.json"  # noqa: E501
        )
        return self._run_async(func=fsio.write_json_message, obj=boxes3d_dto, path=output_path, append_sha1=True)

    def _process_semantic_segmentation_2d(
        self, sensor_frame: CameraSensorFrame[datetime], fs_copy: bool = False
    ) -> Future:
        output_path = (
            self._output_path
            / DirectoryName.SEMANTIC_SEGMENTATION_2D
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.png"  # noqa: E501
        )

        if fs_copy and isinstance(self._dataset._decoder, DGPDatasetDecoder):
            input_path = (
                self._dataset._decoder._dataset_path
                / self._scene.name
                / DirectoryName.SEMANTIC_SEGMENTATION_2D
                / sensor_frame.sensor_name
                / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.png"  # noqa: E501
            )
            return self._run_async(func=fsio.copy_file, source=input_path, target=output_path)
        else:
            return self._encode_semantic_segmentation_2d(sensor_frame=sensor_frame, output_path=output_path)

    def _encode_semantic_segmentation_2d(
        self, sensor_frame: CameraSensorFrame[datetime], output_path: AnyPath
    ) -> Future:
        semseg2d = sensor_frame.get_annotations(AnnotationTypes.SemanticSegmentation2D)
        mask_out = SemanticSegmentation2DTransformer.transform(mask=semseg2d.class_ids)

        return self._run_async(func=fsio.write_png, obj=mask_out, path=output_path)

    def _process_instance_segmentation_2d(
        self, sensor_frame: CameraSensorFrame[datetime], fs_copy: bool = False
    ) -> Future:
        output_path = (
            self._output_path
            / DirectoryName.INSTANCE_SEGMENTATION_2D
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.png"  # noqa: E501
        )

        if fs_copy and isinstance(self._dataset._decoder, DGPDatasetDecoder):
            input_path = (
                self._dataset._decoder._dataset_path
                / self._scene.name
                / DirectoryName.INSTANCE_SEGMENTATION_2D
                / sensor_frame.sensor_name
                / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.png"  # noqa: E501
            )
            return self._run_async(func=fsio.copy_file, source=input_path, target=output_path)
        else:
            return self._encode_instance_segmentation_2d(sensor_frame=sensor_frame, output_path=output_path)

    def _encode_instance_segmentation_2d(
        self, sensor_frame: CameraSensorFrame[datetime], output_path: AnyPath
    ) -> Future:
        instance2d = sensor_frame.get_annotations(AnnotationTypes.InstanceSegmentation2D)
        mask_out = InstanceSegmentation2DTransformer.transform(mask=instance2d.instance_ids)

        return self._run_async(func=fsio.write_png, obj=mask_out, path=output_path)

    def _process_motion_vectors_2d(self, sensor_frame: CameraSensorFrame[datetime], fs_copy: bool = False) -> Future:
        output_path = (
            self._output_path
            / DirectoryName.MOTION_VECTORS_2D
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.png"  # noqa: E501
        )

        if fs_copy and isinstance(self._dataset._decoder, DGPDatasetDecoder):
            input_path = (
                self._dataset._decoder._dataset_path
                / self._scene.name
                / DirectoryName.MOTION_VECTORS_2D
                / sensor_frame.sensor_name
                / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.png"  # noqa: E501
            )
            return self._run_async(func=fsio.copy_file, source=input_path, target=output_path)
        else:
            return self._encode_motion_vectors_2d(sensor_frame=sensor_frame, output_path=output_path)

    def _encode_motion_vectors_2d(self, sensor_frame: CameraSensorFrame[datetime], output_path: AnyPath) -> Future:
        optical_flow = sensor_frame.get_annotations(AnnotationTypes.OpticalFlow)

        return self._run_async(func=fsio.write_png, obj=encode_2int16_as_rgba8(optical_flow.vectors), path=output_path)

    def _process_semantic_segmentation_3d(
        self, sensor_frame: LidarSensorFrame[datetime], fs_copy: bool = False
    ) -> Future:
        output_path = (
            self._output_path
            / DirectoryName.SEMANTIC_SEGMENTATION_3D
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.npz"  # noqa: E501
        )

        if fs_copy and isinstance(self._dataset._decoder, DGPDatasetDecoder):
            input_path = (
                self._dataset._decoder._dataset_path
                / self._scene.name
                / DirectoryName.SEMANTIC_SEGMENTATION_3D
                / sensor_frame.sensor_name
                / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.npz"  # noqa: E501
            )
            return self._run_async(func=fsio.copy_file, source=input_path, target=output_path)
        else:
            return self._encode_semantic_segmentation_3d(sensor_frame=sensor_frame, output_path=output_path)

    def _encode_semantic_segmentation_3d(
        self, sensor_frame: LidarSensorFrame[datetime], output_path: AnyPath
    ) -> Future:
        semseg3d = sensor_frame.get_annotations(AnnotationTypes.SemanticSegmentation3D)
        mask_out = SemanticSegmentation3DTransformer.transform(mask=semseg3d.class_ids)

        return self._run_async(func=fsio.write_npz, obj=dict(segmentation=mask_out), path=output_path)

    def _process_instance_segmentation_3d(
        self, sensor_frame: LidarSensorFrame[datetime], fs_copy: bool = False
    ) -> Future:
        output_path = (
            self._output_path
            / DirectoryName.INSTANCE_SEGMENTATION_3D
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.npz"  # noqa: E501
        )

        if fs_copy and isinstance(self._dataset._decoder, DGPDatasetDecoder):
            input_path = (
                self._dataset._decoder._dataset_path
                / self._scene.name
                / DirectoryName.INSTANCE_SEGMENTATION_3D
                / sensor_frame.sensor_name
                / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.npz"  # noqa: E501
            )
            return self._run_async(func=fsio.copy_file, source=input_path, target=output_path)
        else:
            return self._encode_instance_segmentation_3d(sensor_frame=sensor_frame, output_path=output_path)

    def _encode_instance_segmentation_3d(
        self, sensor_frame: LidarSensorFrame[datetime], output_path: AnyPath
    ) -> Future:
        instance3d = sensor_frame.get_annotations(AnnotationTypes.InstanceSegmentation3D)
        mask_out = InstanceSegmentation3DTransformer.transform(mask=instance3d.instance_ids)

        return self._run_async(func=fsio.write_npz, obj=dict(instance=mask_out), path=output_path)

    def _encode_key_point_2d(self, point: Point2D) -> annotations_pb2.KeyPoint2DAnnotation:
        keypoint_proto = annotations_pb2.KeyPoint2DAnnotation(
            class_id=point.class_id,
            attributes={_attribute_key_dump(k): _attribute_value_dump(v) for k, v in point.attributes.items()},
            points=annotations_pb2.KeyPoint2D(x=point.x, y=point.y),
        )

        return keypoint_proto

    def _encode_key_points_2d(self, sensor_frame: CameraSensorFrame[datetime]) -> Future:
        points2d = sensor_frame.get_annotations(AnnotationTypes.Points2D)
        keypoint2d_dto = KeyPoint2DTransformer.transform(
            objects=[self._encode_key_point_2d(p) for p in points2d.points]
        )
        keypoints2d_dto = annotations_pb2.KeyPoint2DAnnotations(annotations=keypoint2d_dto)

        output_path = (
            self._output_path
            / DirectoryName.KEY_POINT_2D
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time)+self._sim_offset)*100):018d}.json"  # noqa: E501
        )
        return self._run_async(func=fsio.write_json_message, obj=keypoints2d_dto, path=output_path, append_sha1=True)

    def _encode_key_line_2d(self, line: Polyline2D) -> annotations_pb2.KeyLine2DAnnotation:
        keyline_proto = annotations_pb2.KeyLine2DAnnotation(
            class_id=line.class_id,
            attributes={
                _attribute_key_dump(k): _attribute_value_dump(v) for k, v in line.attributes.items() if k != "key"
            },
            vertices=[annotations_pb2.KeyPoint2D(x=int(ll.start.x), y=int(ll.start.y)) for ll in line.lines]
            + [annotations_pb2.KeyPoint2D(x=int(line.lines[-1].end.x), y=int(line.lines[-1].end.y))],
            key=line.attributes["key"] if "key" in line.attributes else "",
        )

        return keyline_proto

    def _encode_key_lines_2d(self, sensor_frame: CameraSensorFrame[datetime]) -> Future:
        polylines2d = sensor_frame.get_annotations(AnnotationTypes.Polylines2D)
        keyline2d_dto = KeyLine2DTransformer.transform(
            objects=[self._encode_key_line_2d(p) for p in polylines2d.polylines]
        )
        keylines2d_dto = annotations_pb2.KeyLine2DAnnotations(annotations=keyline2d_dto)

        output_path = (
            self._output_path
            / DirectoryName.KEY_LINE_2D
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time)+self._sim_offset)*100):018d}.json"  # noqa: E501
        )
        return self._run_async(func=fsio.write_json_message, obj=keylines2d_dto, path=output_path, append_sha1=True)

    def _encode_polygon_2d(self, polygon: Polygon2D) -> annotations_pb2.Polygon2DAnnotation:
        polygon_proto = annotations_pb2.Polygon2DAnnotation(
            class_id=polygon.class_id,
            attributes={_attribute_key_dump(k): _attribute_value_dump(v) for k, v in polygon.attributes.items()},
            vertices=[annotations_pb2.KeyPoint2D(x=ll.start.x, y=ll.start.y) for ll in polygon.lines]
            + [annotations_pb2.KeyPoint2D(x=polygon.lines[-1].end.x, y=polygon.lines[-1].end.y)],
        )

        return polygon_proto

    def _encode_polygons_2d(self, sensor_frame: CameraSensorFrame[datetime]) -> Future:
        polygons2d = sensor_frame.get_annotations(AnnotationTypes.Polygons2D)
        polygon2d_dto = Polygon2DTransformer.transform(
            objects=[self._encode_polygon_2d(p) for p in polygons2d.polygons]
        )
        polygons2d_dto = annotations_pb2.Polygon2DAnnotations(annotations=polygon2d_dto)

        output_path = (
            self._output_path
            / DirectoryName.POLYGON_2D
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time)+self._sim_offset)*100):018d}.json"  # noqa: E501
        )
        return self._run_async(func=fsio.write_json_message, obj=polygons2d_dto, path=output_path, append_sha1=True)

    def _process_surface_normals_2d(self, sensor_frame: SensorFrame[datetime], fs_copy: bool = False) -> Future:
        output_path = (
            self._output_path
            / DirectoryName.SURFACE_NORMALS_2D
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.png"  # noqa: E501
        )

        if fs_copy and isinstance(self._dataset._decoder, DGPDatasetDecoder):
            input_path = (
                self._dataset._decoder._dataset_path
                / self._scene.name
                / DirectoryName.SURFACE_NORMALS_2D
                / sensor_frame.sensor_name
                / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.png"  # noqa: E501
            )

            return self._run_async(func=fsio.copy_file, source=input_path, target=output_path)
        else:
            return self._encode_surface_normals_2d(sensor_frame=sensor_frame, output_path=output_path)

    def _encode_surface_normals_2d(
        self, sensor_frame: SensorFrame[datetime], output_path: AnyPath
    ) -> Union[Future, None]:
        surface_normals = sensor_frame.get_annotations(AnnotationTypes.SurfaceNormals2D)
        encoded_normals = ((surface_normals.normals * 0.5 + 0.5) * 255).astype(np.uint8)
        return self._run_async(func=fsio.write_png, obj=encoded_normals, path=output_path)

    def _process_surface_normals_3d(self, sensor_frame: SensorFrame[datetime], fs_copy: bool = False) -> Future:
        output_path = (
            self._output_path
            / DirectoryName.SURFACE_NORMALS_3D
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.npz"  # noqa: E501
        )

        if fs_copy and isinstance(self._dataset._decoder, DGPDatasetDecoder):
            input_path = (
                self._dataset._decoder._dataset_path
                / self._scene.name
                / DirectoryName.SURFACE_NORMALS_3D
                / sensor_frame.sensor_name
                / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.npz"  # noqa: E501
            )

            return self._run_async(func=fsio.copy_file, source=input_path, target=output_path)
        else:
            return self._encode_surface_normals_3d(sensor_frame=sensor_frame, output_path=output_path)

    def _process_motion_vectors_3d(self, sensor_frame: LidarSensorFrame[datetime], fs_copy: bool = False) -> Future:
        output_path = (
            self._output_path
            / DirectoryName.MOTION_VECTORS_3D
            / sensor_frame.sensor_name
            / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.npz"  # noqa: E501
        )

        if fs_copy and isinstance(self._dataset._decoder, DGPDatasetDecoder):
            input_path = (
                self._dataset._decoder._dataset_path
                / self._scene.name
                / DirectoryName.MOTION_VECTORS_3D
                / sensor_frame.sensor_name
                / f"{round((self._offset_timestamp(compare_datetime=sensor_frame.date_time) + self._sim_offset) * 100):018d}.npz"  # noqa: E501
            )
            return self._run_async(func=fsio.copy_file, source=input_path, target=output_path)
        else:
            return self._encode_motion_vectors_3d(sensor_frame=sensor_frame, output_path=output_path)

    def _encode_motion_vectors_3d(self, sensor_frame: LidarSensorFrame[datetime], output_path: AnyPath) -> Future:
        scene_flow = sensor_frame.get_annotations(AnnotationTypes.SceneFlow)

        return self._run_async(func=fsio.write_npz, obj=dict(motion_vectors=scene_flow.vectors), path=output_path)

    def _encode_surface_normals_3d(
        self, sensor_frame: SensorFrame[datetime], output_path: AnyPath
    ) -> Union[Future, None]:
        surface_normals = sensor_frame.get_annotations(AnnotationTypes.SurfaceNormals3D)
        return self._run_async(func=fsio.write_npz, obj=dict(surface_normals=surface_normals.normals), path=output_path)

    def _process_encode_camera_results(
        self,
        camera_name: str,
        camera_encoding_futures: Set[Future],
        # camera_encoding_results: Iterator[Tuple[str, Dict[str, Dict[str, Future]]]],
    ) -> Tuple[str, Dict[str, sample_pb2.Datum]]:
        scene_data_dtos = []

        camera = self._scene.get_sensor(camera_name)
        for res in concurrent.futures.as_completed(camera_encoding_futures):
            frame_id, result_dict = res.result()
            camera_frame = camera.get_frame(frame_id)
            sensor_data = result_dict["sensor_data"]
            annotations = result_dict["annotations"]
            metadata = result_dict["metadata"]

            scene_datum_dto = image_pb2.Image(
                filename=self._relative_path(sensor_data[DirectoryName.RGB].result()).as_posix(),
                height=camera_frame.image.height,
                width=camera_frame.image.width,
                channels=4,
                annotations={
                    int(k): self._relative_path(v.result()).as_posix() for k, v in annotations.items() if v is not None
                },
                pose=geometry_pb2.Pose(
                    translation=geometry_pb2.Vector3(
                        x=camera_frame.pose.translation[0],
                        y=camera_frame.pose.translation[1],
                        z=camera_frame.pose.translation[2],
                    ),
                    rotation=geometry_pb2.Quaternion(
                        qw=camera_frame.pose.quaternion.w,
                        qx=camera_frame.pose.quaternion.x,
                        qy=camera_frame.pose.quaternion.y,
                        qz=camera_frame.pose.quaternion.z,
                    ),
                ),
                metadata={str(k): v for k, v in metadata.items()},
            )
            # noinspection PyTypeChecker
            scene_data_dtos.append(
                sample_pb2.Datum(
                    id=identifiers_pb2.DatumId(
                        log="",
                        name=camera_frame.sensor_name,
                        timestamp=datetime_to_timestamp(dt=camera_frame.date_time),
                        index=int(camera_frame.frame_id),
                    ),
                    key="",
                    datum=sample_pb2.DatumValue(image=scene_datum_dto),
                    next_key="",
                    prev_key="",
                )
            )

        scene_data_count = len(scene_data_dtos)
        # noinspection InsecureHash
        keys = [hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest() for _ in range(scene_data_count)]

        for idx, scene_data_dto in enumerate(sorted(scene_data_dtos, key=lambda x: x.id.timestamp.ToDatetime())):
            prev_key = keys[idx - 1] if idx > 0 else ""
            key = keys[idx]
            next_key = keys[idx + 1] if idx < (scene_data_count - 1) else ""

            scene_data_dto.prev_key = prev_key
            scene_data_dto.key = key
            scene_data_dto.next_key = next_key

        return camera_name, {str(sd.id.index): sd for sd in scene_data_dtos}

    def _process_encode_lidar_results(
        self,
        lidar_name: str,
        lidar_encoding_futures: Set[Future],
    ) -> Tuple[str, Dict[str, sample_pb2.Datum]]:
        scene_data_dtos = []

        lidar = self._scene.get_sensor(lidar_name)
        for res in concurrent.futures.as_completed(lidar_encoding_futures):
            frame_id, result_dict = res.result()
            lidar_frame = lidar.get_frame(frame_id)
            sensor_data = result_dict["sensor_data"]
            annotations = result_dict["annotations"]
            metadata = result_dict["metadata"]

            scene_datum_dto = point_cloud_pb2.PointCloud(
                filename=self._relative_path(sensor_data[DirectoryName.POINT_CLOUD].result()).as_posix(),
                point_format=[getattr(point_cloud_pb2.PointCloud.ChannelType, pf) for pf in PointFormat.to_list()],
                annotations={
                    int(k): self._relative_path(v.result()).as_posix() for k, v in annotations.items() if v is not None
                },
                pose=geometry_pb2.Pose(
                    translation=geometry_pb2.Vector3(
                        x=lidar_frame.pose.translation[0],
                        y=lidar_frame.pose.translation[1],
                        z=lidar_frame.pose.translation[2],
                    ),
                    rotation=geometry_pb2.Quaternion(
                        qw=lidar_frame.pose.quaternion.w,
                        qx=lidar_frame.pose.quaternion.x,
                        qy=lidar_frame.pose.quaternion.y,
                        qz=lidar_frame.pose.quaternion.z,
                    ),
                ),
                point_fields=[],
                metadata={str(k): v for k, v in metadata.items()},
            )
            # noinspection PyTypeChecker
            scene_data_dtos.append(
                sample_pb2.Datum(
                    id=identifiers_pb2.DatumId(
                        log="",
                        name=lidar_frame.sensor_name,
                        timestamp=datetime_to_timestamp(dt=lidar_frame.date_time),
                        index=int(lidar_frame.frame_id),
                    ),
                    key="",
                    datum=sample_pb2.DatumValue(point_cloud=scene_datum_dto),
                    next_key="",
                    prev_key="",
                )
            )

        scene_data_count = len(scene_data_dtos)
        # noinspection InsecureHash
        keys = [hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest() for _ in range(scene_data_count)]

        for idx, scene_data_dto in enumerate(sorted(scene_data_dtos, key=lambda x: x.id.timestamp.ToDatetime())):
            prev_key = keys[idx - 1] if idx > 0 else ""
            key = keys[idx]
            next_key = keys[idx + 1] if idx < (scene_data_count - 1) else ""

            scene_data_dto.prev_key = prev_key
            scene_data_dto.key = key
            scene_data_dto.next_key = next_key

        return lidar_name, {str(sd.id.index): sd for sd in scene_data_dtos}

    def _encode_camera_frame(
        self, frame_id: str, camera_frame: CameraSensorFrame[datetime], last_frame: Optional[bool] = False
    ) -> Tuple[str, Dict[str, Dict[str, Future]]]:
        return frame_id, dict(
            annotations={
                "0": self._encode_bounding_boxes_2d(sensor_frame=camera_frame)
                if AnnotationTypes.BoundingBoxes2D in camera_frame.available_annotation_types
                and AnnotationTypes.BoundingBoxes2D in self._annotation_types
                else None,
                "1": self._encode_bounding_boxes_3d(sensor_frame=camera_frame)
                if AnnotationTypes.BoundingBoxes3D in camera_frame.available_annotation_types
                and AnnotationTypes.BoundingBoxes3D in self._annotation_types
                else None,
                "2": self._process_semantic_segmentation_2d(sensor_frame=camera_frame, fs_copy=True)
                if AnnotationTypes.SemanticSegmentation2D in camera_frame.available_annotation_types
                and AnnotationTypes.SemanticSegmentation2D in self._annotation_types
                else None,
                "4": self._process_instance_segmentation_2d(sensor_frame=camera_frame, fs_copy=True)
                if AnnotationTypes.InstanceSegmentation2D in camera_frame.available_annotation_types
                and AnnotationTypes.InstanceSegmentation2D in self._annotation_types
                else None,
                "6": self._process_depth(sensor_frame=camera_frame, fs_copy=True)
                if AnnotationTypes.Depth in camera_frame.available_annotation_types
                and AnnotationTypes.Depth in self._annotation_types
                else None,
                "8": self._process_motion_vectors_2d(sensor_frame=camera_frame, fs_copy=True)
                if AnnotationTypes.OpticalFlow in camera_frame.available_annotation_types
                and AnnotationTypes.OpticalFlow in self._annotation_types
                and not last_frame
                else None,
                "10": self._encode_key_points_2d(sensor_frame=camera_frame)
                if AnnotationTypes.Points2D in camera_frame.available_annotation_types
                and AnnotationTypes.Points2D in self._annotation_types
                else None,
                "11": self._encode_key_lines_2d(sensor_frame=camera_frame)
                if AnnotationTypes.Polylines2D in camera_frame.available_annotation_types
                and AnnotationTypes.Polylines2D in self._annotation_types
                else None,
                "12": self._encode_polygons_2d(sensor_frame=camera_frame)
                if AnnotationTypes.Polygons2D in camera_frame.available_annotation_types
                and AnnotationTypes.Polygons2D in self._annotation_types
                else None,
                "13": self._process_surface_normals_2d(sensor_frame=camera_frame, fs_copy=True)
                if AnnotationTypes.SurfaceNormals2D in camera_frame.available_annotation_types
                and AnnotationTypes.SurfaceNormals2D in self._annotation_types
                else None,
            },
            sensor_data={
                "rgb": self._process_rgb(sensor_frame=camera_frame, fs_copy=True),
            },
            metadata={},
        )

    def _encode_lidar_frame(
        self, frame_id: str, lidar_frame: LidarSensorFrame[datetime], last_frame: Optional[bool] = False
    ) -> Tuple[str, Dict[str, Dict[str, Future]]]:
        return frame_id, dict(
            annotations={
                "1": self._encode_bounding_boxes_3d(sensor_frame=lidar_frame)
                if AnnotationTypes.BoundingBoxes3D in lidar_frame.available_annotation_types
                and AnnotationTypes.BoundingBoxes3D in self._annotation_types
                else None,
                "3": self._process_semantic_segmentation_3d(sensor_frame=lidar_frame, fs_copy=True)
                if AnnotationTypes.SemanticSegmentation3D in lidar_frame.available_annotation_types
                and AnnotationTypes.SemanticSegmentation3D in self._annotation_types
                else None,
                "5": self._process_instance_segmentation_3d(sensor_frame=lidar_frame, fs_copy=True)
                if AnnotationTypes.InstanceSegmentation3D in lidar_frame.available_annotation_types
                and AnnotationTypes.InstanceSegmentation3D in self._annotation_types
                else None,
                "6": self._process_depth(sensor_frame=lidar_frame, fs_copy=True)
                if AnnotationTypes.Depth in lidar_frame.available_annotation_types
                and AnnotationTypes.Depth in self._annotation_types
                else None,
                "7": self._process_surface_normals_3d(sensor_frame=lidar_frame, fs_copy=True)
                if AnnotationTypes.SurfaceNormals3D in lidar_frame.available_annotation_types
                and AnnotationTypes.SurfaceNormals3D in self._annotation_types
                else None,
                "9": self._process_motion_vectors_3d(sensor_frame=lidar_frame, fs_copy=True)
                if AnnotationTypes.SceneFlow in lidar_frame.available_annotation_types
                and AnnotationTypes.SceneFlow in self._annotation_types
                and not last_frame
                else None,
            },
            sensor_data={
                "point_cloud": self._process_point_cloud(sensor_frame=lidar_frame, fs_copy=True),
            },
            metadata={},
        )

    def _encode_camera(self, camera_name: str) -> Future:
        frame_ids = self._scene.frame_ids
        futures = set()
        for frame_id in frame_ids:
            while True:
                # if ENCODING_THREAD_POOL.queue.qsize() < max(1, ENCODING_THREAD_POOL.max_workers // 4):
                if ENCODING_THREAD_POOL.queue.qsize() < 4:
                    logger.debug(f"Scheduling camera frame {camera_name} {frame_id}")
                    futures.add(
                        ENCODING_THREAD_POOL.submit(
                            lambda fid: self._encode_camera_frame(
                                frame_id=fid,
                                camera_frame=self._scene.get_frame(fid).get_camera(camera_name=camera_name),
                                last_frame=(frame_ids.index(fid) == (len(frame_ids) - 1)),
                            ),
                            frame_id,
                        )
                    )
                    break
                else:
                    time.sleep(2 + random.random())

        # futures = {
        #     ENCODING_THREAD_POOL.submit(
        #         lambda fid: self._encode_camera_frame(
        #             frame_id=fid,
        #             camera_frame=self._scene.get_frame(fid).get_camera(camera_name=camera_name),
        #             last_frame=(frame_ids.index(fid) == (len(frame_ids) - 1)),
        #         ),
        #         frame_id,
        #     )
        #     for frame_id in frame_ids
        # }
        return ENCODING_THREAD_POOL.submit(
            lambda: self._process_encode_camera_results(
                camera_name=camera_name,
                camera_encoding_futures=futures,
            )
        )

    def _encode_lidar(self, lidar_name: str) -> Future:
        frame_ids = self._scene.frame_ids
        lidar_encoding_futures = {
            ENCODING_THREAD_POOL.submit(
                lambda fid: self._encode_lidar_frame(
                    frame_id=fid,
                    lidar_frame=self._scene.get_frame(fid).get_lidar(lidar_name=lidar_name),
                    last_frame=(frame_ids.index(fid) == (len(frame_ids) - 1)),
                ),
                frame_id,
            )
            for frame_id in frame_ids
        }
        return ENCODING_THREAD_POOL.submit(
            lambda: self._process_encode_lidar_results(
                lidar_name=lidar_name, lidar_encoding_futures=lidar_encoding_futures
            )
        )

    def _encode_cameras(self) -> Iterator[Tuple[str, Dict[str, sample_pb2.Datum]]]:
        return [self._encode_camera(camera_name=c).result() for c in self._camera_names]

    def _encode_lidars(self) -> Iterator[Tuple[str, Dict[str, sample_pb2.Datum]]]:
        return [self._encode_lidar(lidar_name=ln).result() for ln in self._lidar_names]

    def _encode_ontologies(self) -> Dict[str, Future]:
        ontology_dtos = {
            ANNOTATION_TYPE_MAP_INV[a_type]: class_map_to_ontology_proto(class_map=self._scene.get_class_map(a_type))
            for a_type in self._annotation_types
            if a_type is not Annotation  # equiv: not implemented, yet!
        }

        output_path = self._output_path / DirectoryName.ONTOLOGY / ".json"

        return {
            k: self._run_async(func=fsio.write_json_message, obj=v, path=output_path, append_sha1=True)
            for k, v in ontology_dtos.items()
        }

    def _encode_calibrations(self) -> Future:
        camera_frames = []
        lidar_frames = []
        frame_ids = self._scene.frame_ids

        for sn in self._camera_names:
            camera_frames.append(self._scene.get_sensor(sn).get_frame(frame_ids[0]))
        for sn in self._lidar_names:
            lidar_frames.append(self._scene.get_sensor(sn).get_frame(frame_ids[0]))

        calib_dto = sample_pb2.SampleCalibration(names=[], extrinsics=[], intrinsics=[])

        def get_camera_calibration(
            sf: CameraSensorFrame[datetime],
        ) -> Tuple[str, geometry_pb2.Pose, geometry_pb2.CameraIntrinsics]:
            intr = sf.intrinsic
            extr = sf.extrinsic

            calib_dto_extrinsic = geometry_pb2.Pose(
                translation=geometry_pb2.Vector3(x=extr.translation[0], y=extr.translation[1], z=extr.translation[2]),
                rotation=geometry_pb2.Quaternion(
                    qw=extr.quaternion.w, qx=extr.quaternion.x, qy=extr.quaternion.y, qz=extr.quaternion.z
                ),
            )

            calib_dto_intrinsic = geometry_pb2.CameraIntrinsics(
                fx=intr.fx,
                fy=intr.fy,
                cx=intr.cx,
                cy=intr.cy,
                skew=intr.skew,
                fov=intr.fov,
                k1=intr.k1,
                k2=intr.k2,
                k3=intr.k3,
                k4=intr.k4,
                k5=intr.k5,
                k6=intr.k6,
                p1=intr.p1,
                p2=intr.p2,
                fisheye=self._fisheye_camera_model_map[intr.camera_model],
            )

            return (sf.sensor_name, calib_dto_extrinsic, calib_dto_intrinsic)

        def get_lidar_calibration(
            sf: LidarSensorFrame[datetime],
        ) -> Tuple[str, geometry_pb2.Pose, geometry_pb2.CameraIntrinsics]:
            extr = sf.extrinsic

            calib_dto_extrinsic = geometry_pb2.Pose(
                translation=geometry_pb2.Vector3(x=extr.translation[0], y=extr.translation[1], z=extr.translation[2]),
                rotation=geometry_pb2.Quaternion(
                    qw=extr.quaternion.w, qx=extr.quaternion.x, qy=extr.quaternion.y, qz=extr.quaternion.z
                ),
            )

            calib_dto_intrinsic = geometry_pb2.CameraIntrinsics(
                fx=0.0,
                fy=0.0,
                cx=0.0,
                cy=0.0,
                skew=0.0,
                fov=0.0,
                k1=0.0,
                k2=0.0,
                k3=0.0,
                k4=0.0,
                k5=0.0,
                k6=0.0,
                p1=0.0,
                p2=0.0,
                fisheye=0,
            )

            return (sf.sensor_name, calib_dto_extrinsic, calib_dto_intrinsic)

        res = list(map(get_camera_calibration, camera_frames)) + list(map(get_lidar_calibration, lidar_frames))

        for r_name, r_extrinsic, r_intrinsic in sorted(res):
            calib_dto.names.append(r_name)
            calib_dto.extrinsics.append(r_extrinsic)
            calib_dto.intrinsics.append(r_intrinsic)

        output_path = self._output_path / DirectoryName.CALIBRATION / ".json"
        return self._run_async(func=fsio.write_json_message, obj=calib_dto, path=output_path, append_sha1=True)

    def _encode_scene_json(
        self,
        scene_sensor_data: Dict[str, Dict[str, sample_pb2.Datum]],
        calibration_file: Future,
        ontologies_files: Dict[str, Future],
    ) -> AnyPath:
        scene_data = []
        scene_samples = []
        for fid in self._scene.frame_ids:
            frame = self._scene.get_frame(fid)
            frame_data = [
                scene_sensor_data[sn][fid] for sn in sorted(scene_sensor_data.keys()) if fid in scene_sensor_data[sn]
            ]
            scene_data.extend(frame_data)
            scene_samples.append(
                sample_pb2.Sample(
                    id=identifiers_pb2.DatumId(
                        log="",
                        timestamp=datetime_to_timestamp(dt=frame.date_time),
                        name="",
                        index=int(frame.frame_id),
                    ),
                    datum_keys=[d.key for d in frame_data],
                    calibration_key=calibration_file.result().stem,
                    metadata={},
                )
            )

        scene_dto = scene_pb2.Scene(
            name=self._scene.name,
            description=self._scene.description,
            log="",
            ontologies={k: v.result().stem for k, v in ontologies_files.items()},
            metadata={
                # "PD": any_pb2.Any().Pack(
                #     metadata_pd_pb2.ParallelDomainSceneMetadata(
                #         **{k: v for k, v in self._scene.metadata["PD"].items() if not k.startswith("@")}
                #     )
                # )
            },
            samples=scene_samples,
            data=scene_data,
            creation_date=timestamp_pb2.Timestamp().GetCurrentTime(),
            statistics=None,
        )

        output_path = self._output_path / "scene.json"
        return fsio.write_json_message(obj=scene_dto, path=output_path, append_sha1=True)

    def _encode_sensors(self) -> Dict[str, Dict[str, sample_pb2.Datum]]:
        scene_camera_data = self._encode_cameras()
        scene_lidar_data = self._encode_lidars()

        scene_sensor_data = {}
        scene_sensor_data.update(dict(scene_camera_data))
        scene_sensor_data.update(dict(scene_lidar_data))

        return scene_sensor_data

    def _run_encoding(self) -> AnyPath:
        scene_sensor_data = self._encode_sensors()
        calibration_file = self._encode_calibrations()
        ontologies_files = self._encode_ontologies()
        return self._encode_scene_json(
            scene_sensor_data=scene_sensor_data, calibration_file=calibration_file, ontologies_files=ontologies_files
        )

    def _prepare_output_directories(self) -> None:
        super()._prepare_output_directories()
        if not self._output_path.is_cloud_path:  # Local FS - needs existing directories
            (self._output_path / DirectoryName.CALIBRATION).mkdir(exist_ok=True, parents=True)
            (self._output_path / DirectoryName.ONTOLOGY).mkdir(exist_ok=True, parents=True)
            for camera_name in self._camera_names:
                (self._output_path / DirectoryName.RGB / camera_name).mkdir(exist_ok=True, parents=True)
                if AnnotationTypes.BoundingBoxes2D in self._annotation_types:
                    (self._output_path / DirectoryName.BOUNDING_BOX_2D / camera_name).mkdir(exist_ok=True, parents=True)
                if AnnotationTypes.BoundingBoxes3D in self._annotation_types:
                    (self._output_path / DirectoryName.BOUNDING_BOX_3D / camera_name).mkdir(exist_ok=True, parents=True)
                if AnnotationTypes.SemanticSegmentation2D in self._annotation_types:
                    (self._output_path / DirectoryName.SEMANTIC_SEGMENTATION_2D / camera_name).mkdir(
                        exist_ok=True, parents=True
                    )
                if AnnotationTypes.InstanceSegmentation2D in self._annotation_types:
                    (self._output_path / DirectoryName.INSTANCE_SEGMENTATION_2D / camera_name).mkdir(
                        exist_ok=True, parents=True
                    )
                if AnnotationTypes.OpticalFlow in self._annotation_types:
                    (self._output_path / DirectoryName.MOTION_VECTORS_2D / camera_name).mkdir(
                        exist_ok=True, parents=True
                    )
                if AnnotationTypes.Depth in self._annotation_types:
                    (self._output_path / DirectoryName.DEPTH / camera_name).mkdir(exist_ok=True, parents=True)
                if AnnotationTypes.OpticalFlow in self._annotation_types:
                    (self._output_path / DirectoryName.MOTION_VECTORS_2D / camera_name).mkdir(
                        exist_ok=True, parents=True
                    )
                if AnnotationTypes.Points2D in self._annotation_types:
                    (self._output_path / DirectoryName.KEY_POINT_2D / camera_name).mkdir(exist_ok=True, parents=True)
                if AnnotationTypes.Polylines2D in self._annotation_types:
                    (self._output_path / DirectoryName.KEY_LINE_2D / camera_name).mkdir(exist_ok=True, parents=True)
                if AnnotationTypes.Polygons2D in self._annotation_types:
                    (self._output_path / DirectoryName.POLYGON_2D / camera_name).mkdir(exist_ok=True, parents=True)
                if AnnotationTypes.SurfaceNormals2D in self._annotation_types:
                    (self._output_path / DirectoryName.SURFACE_NORMALS_2D / camera_name).mkdir(
                        exist_ok=True, parents=True
                    )
            for lidar_name in self._lidar_names:
                (self._output_path / DirectoryName.POINT_CLOUD / lidar_name).mkdir(exist_ok=True, parents=True)
                if AnnotationTypes.BoundingBoxes3D in self._annotation_types:
                    (self._output_path / DirectoryName.BOUNDING_BOX_3D / lidar_name).mkdir(exist_ok=True, parents=True)
                if AnnotationTypes.Depth in self._annotation_types:
                    (self._output_path / DirectoryName.DEPTH / lidar_name).mkdir(exist_ok=True, parents=True)
                if AnnotationTypes.SemanticSegmentation3D in self._annotation_types:
                    (self._output_path / DirectoryName.SEMANTIC_SEGMENTATION_3D / lidar_name).mkdir(
                        exist_ok=True, parents=True
                    )
                if AnnotationTypes.InstanceSegmentation3D in self._annotation_types:
                    (self._output_path / DirectoryName.INSTANCE_SEGMENTATION_3D / lidar_name).mkdir(
                        exist_ok=True, parents=True
                    )
                if AnnotationTypes.SurfaceNormals3D in self._annotation_types:
                    (self._output_path / DirectoryName.SURFACE_NORMALS_3D / lidar_name).mkdir(
                        exist_ok=True, parents=True
                    )
                if AnnotationTypes.SceneFlow in self._annotation_types:
                    (self._output_path / DirectoryName.MOTION_VECTORS_3D / lidar_name).mkdir(
                        exist_ok=True, parents=True
                    )
