from datetime import datetime
from functools import partial
from typing import Any, Dict, Generator, Iterable, List, Optional

import pypeln

from paralleldomain import Scene
from paralleldomain.common.dgp.v1 import annotations_pb2
from paralleldomain.common.dgp.v1.constants import ANNOTATION_TYPE_MAP_INV, DirectoryName
from paralleldomain.encoding.dgp.v1.encoder_steps.helper import EncoderStepHelper
from paralleldomain.encoding.dgp.v1.utils import _attribute_key_dump, _attribute_value_dump
from paralleldomain.encoding.pipeline_encoder import EncoderStep
from paralleldomain.model.annotation import AnnotationType, AnnotationTypes, BoundingBox2D, Point2D, Polyline2D
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath


class Polylines2DEncoderStep(EncoderStepHelper, EncoderStep):
    def __init__(
        self,
        workers: int = 1,
        in_queue_size: int = 1,
        output_annotation_types: Optional[List[AnnotationType]] = None,
    ):
        self.in_queue_size = in_queue_size
        self.workers = workers
        self.output_annotation_types = output_annotation_types

    @property
    def apply_stages(self) -> bool:
        return self.output_annotation_types is None or AnnotationTypes.Polylines2D in self.output_annotation_types

    @staticmethod
    def encode_key_line_2d(line: Polyline2D) -> annotations_pb2.KeyLine2DAnnotation:
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

    def encode_polylines_2d(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        sensor_frame = self._get_camera_frame_from_input_dict(input_dict=input_dict)
        if sensor_frame is not None and AnnotationTypes.Polylines2D in sensor_frame.available_annotation_types:
            polylines2d = sensor_frame.get_annotations(AnnotationTypes.Polylines2D)
            keyline2d_dto = [self.encode_key_line_2d(p) for p in polylines2d.polylines]
            keylines2d_dto = annotations_pb2.KeyLine2DAnnotations(annotations=keyline2d_dto)

            output_path = self._get_dgpv1_file_output_path(
                sensor_frame=sensor_frame,
                input_dict=input_dict,
                file_suffix="json",
                directory_name=DirectoryName.KEY_LINE_2D,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path = fsio.write_json_message(obj=keylines2d_dto, path=output_path, append_sha1=True)

            if "annotations" not in input_dict:
                input_dict["annotations"] = dict()
            input_dict["annotations"][str(ANNOTATION_TYPE_MAP_INV[AnnotationTypes.Polylines2D])] = output_path
        return input_dict

    def apply(self, input_stage: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        stage = input_stage
        if self.apply_stages:
            stage = pypeln.thread.map(
                f=self.encode_polylines_2d, stage=stage, workers=self.workers, maxsize=self.in_queue_size
            )

        return stage
