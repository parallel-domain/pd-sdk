from datetime import datetime
from functools import partial
from typing import Any, Dict, Generator, Iterable

import pypeln

from paralleldomain import Scene
from paralleldomain.common.dgp.v1 import annotations_pb2
from paralleldomain.common.dgp.v1.constants import ANNOTATION_TYPE_MAP_INV, DirectoryName
from paralleldomain.encoding.dgp.v1.encoder_steps.encoder_step import EncoderStep
from paralleldomain.encoding.dgp.v1.utils import _attribute_key_dump, _attribute_value_dump
from paralleldomain.model.annotation import AnnotationTypes, BoundingBox2D, Point2D, Polygon2D, Polyline2D
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath


class Polygons2DEncoderStep(EncoderStep):
    def __init__(
        self,
        workers: int = 1,
        in_queue_size: int = 4,
    ):
        self.in_queue_size = in_queue_size
        self.workers = workers

    @staticmethod
    def encode_polygon_2d(polygon: Polygon2D) -> annotations_pb2.Polygon2DAnnotation:
        polygon_proto = annotations_pb2.Polygon2DAnnotation(
            class_id=polygon.class_id,
            attributes={_attribute_key_dump(k): _attribute_value_dump(v) for k, v in polygon.attributes.items()},
            vertices=[annotations_pb2.KeyPoint2D(x=ll.start.x, y=ll.start.y) for ll in polygon.lines]
            + [annotations_pb2.KeyPoint2D(x=polygon.lines[-1].end.x, y=polygon.lines[-1].end.y)],
        )

        return polygon_proto

    def encode_polygons_2d(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        sensor_frame = self._get_camera_frame_from_input_dict(input_dict=input_dict)
        if sensor_frame is not None and AnnotationTypes.Polygons2D in sensor_frame.available_annotation_types:

            polygons2d = sensor_frame.get_annotations(AnnotationTypes.Polygons2D)
            polygon2d_dto = [self.encode_polygon_2d(p) for p in polygons2d.polygons]
            polygons2d_dto = annotations_pb2.Polygon2DAnnotations(annotations=polygon2d_dto)

            output_path = self._get_dgpv1_file_output_path(
                sensor_frame=sensor_frame,
                input_dict=input_dict,
                file_suffix="json",
                directory_name=DirectoryName.POLYGON_2D,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path = fsio.write_json_message(obj=polygons2d_dto, path=output_path, append_sha1=True)

            if "annotations" not in input_dict:
                input_dict["annotations"] = dict()
            input_dict["annotations"][str(ANNOTATION_TYPE_MAP_INV[AnnotationTypes.Polygons2D])] = output_path
        return input_dict

    def apply(self, scene: Scene, input_stage: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        stage = input_stage
        stage = pypeln.thread.map(
            f=self.encode_polygons_2d, stage=stage, workers=self.workers, maxsize=self.in_queue_size
        )

        return stage
