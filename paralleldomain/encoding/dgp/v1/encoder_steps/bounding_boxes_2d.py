from datetime import datetime
from functools import partial
from typing import Any, Dict, Generator, Iterable

import pypeln

from paralleldomain import Scene
from paralleldomain.common.dgp.v1 import annotations_pb2
from paralleldomain.common.dgp.v1.constants import ANNOTATION_TYPE_MAP_INV, DirectoryName
from paralleldomain.encoding.dgp.v1.encoder_steps.helper import EncoderStepHelper
from paralleldomain.encoding.dgp.v1.utils import _attribute_key_dump, _attribute_value_dump
from paralleldomain.encoding.pipeline_encoder import EncoderStep
from paralleldomain.model.annotation import AnnotationTypes, BoundingBox2D
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath


class BoundingBoxes2DEncoderStep(EncoderStepHelper, EncoderStep):
    def __init__(
        self,
        workers: int = 1,
        in_queue_size: int = 4,
    ):
        self.in_queue_size = in_queue_size
        self.workers = workers

    @staticmethod
    def encode_bounding_box_2d(box: BoundingBox2D) -> annotations_pb2.BoundingBox2DAnnotation:
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

    def encode_bounding_boxes_2d(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        sensor_frame = self._get_camera_frame_from_input_dict(input_dict=input_dict)

        if sensor_frame is not None and AnnotationTypes.BoundingBoxes2D in sensor_frame.available_annotation_types:
            boxes2d = sensor_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes2D)
            annotations = [BoundingBoxes2DEncoderStep.encode_bounding_box_2d(b) for b in boxes2d.boxes]
            boxes2d_dto = annotations_pb2.BoundingBox2DAnnotations(annotations=annotations)
            output_path = self.save_boxes_2d_to_file(
                boxes2d_dto=boxes2d_dto,
                sensor_frame=sensor_frame,
                input_dict=input_dict,
            )

            if "annotations" not in input_dict:
                input_dict["annotations"] = dict()
            input_dict["annotations"][str(ANNOTATION_TYPE_MAP_INV[AnnotationTypes.BoundingBoxes2D])] = output_path
        return input_dict

    def save_boxes_2d_to_file(
        self,
        sensor_frame: CameraSensorFrame[datetime],
        input_dict: Dict[str, Any],
        boxes2d_dto: annotations_pb2.BoundingBox2DAnnotations,
    ) -> str:
        output_path = self._get_dgpv1_file_output_path(
            sensor_frame=sensor_frame,
            input_dict=input_dict,
            file_suffix="json",
            directory_name=DirectoryName.BOUNDING_BOX_2D,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return str(fsio.write_json_message(obj=boxes2d_dto, path=output_path, append_sha1=True))

    def apply(self, scene: Scene, input_stage: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        stage = input_stage
        # stage = pypeln.sync.map(lambda id: id.pop("sensor_frame"), input_stage)
        stage = pypeln.thread.map(
            f=self.encode_bounding_boxes_2d, stage=stage, workers=self.workers, maxsize=self.in_queue_size
        )

        return stage
