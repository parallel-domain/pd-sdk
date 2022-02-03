from datetime import datetime
from functools import partial
from typing import Any, Dict, Generator, Iterable

import pypeln

from paralleldomain import Scene
from paralleldomain.common.dgp.v1 import annotations_pb2, geometry_pb2
from paralleldomain.common.dgp.v1.constants import ANNOTATION_TYPE_MAP_INV, DirectoryName
from paralleldomain.encoding.dgp.v1.encoder_steps.encoder_step import EncoderStep
from paralleldomain.encoding.dgp.v1.utils import _attribute_key_dump, _attribute_value_dump
from paralleldomain.model.annotation import AnnotationTypes, BoundingBox3D
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath


class BoundingBoxes3DEncoderStep(EncoderStep):
    def __init__(
        self,
        workers: int = 1,
        in_queue_size: int = 4,
    ):
        self.in_queue_size = in_queue_size
        self.workers = workers

    @staticmethod
    def encode_bounding_box_3d(box: BoundingBox3D) -> annotations_pb2.BoundingBox3DAnnotation:
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

    def encode_bounding_boxes_3d(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        sensor_frame = self._get_camera_frame_from_input_dict(input_dict=input_dict)
        if sensor_frame is None:
            sensor_frame = self._get_lidar_frame_from_input_dict(input_dict=input_dict)

        if sensor_frame is not None and AnnotationTypes.BoundingBoxes3D in sensor_frame.available_annotation_types:
            boxes3d = sensor_frame.get_annotations(annotation_type=AnnotationTypes.BoundingBoxes3D)
            annotations = [BoundingBoxes3DEncoderStep.encode_bounding_box_3d(b) for b in boxes3d.boxes]
            boxes3d_dto = annotations_pb2.BoundingBox3DAnnotations(annotations=annotations)
            output_path = self.save_boxes_3d_to_file(
                boxes3d_dto=boxes3d_dto,
                sensor_frame=sensor_frame,
                input_dict=input_dict,
            )

            if "annotations" not in input_dict:
                input_dict["annotations"] = dict()
            input_dict["annotations"][str(ANNOTATION_TYPE_MAP_INV[AnnotationTypes.BoundingBoxes3D])] = output_path
        return input_dict

    def save_boxes_3d_to_file(
        self,
        sensor_frame: CameraSensorFrame[datetime],
        input_dict: Dict[str, Any],
        boxes3d_dto: annotations_pb2.BoundingBox3DAnnotations,
    ) -> str:
        output_path = self._get_dgpv1_file_output_path(
            sensor_frame=sensor_frame,
            input_dict=input_dict,
            file_suffix="json",
            directory_name=DirectoryName.BOUNDING_BOX_3D,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fsio.write_json_message(obj=boxes3d_dto, path=output_path, append_sha1=True)
        return str(output_path)

    def apply(self, scene: Scene, input_stage: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        stage = input_stage
        # stage = pypeln.sync.map(lambda id: id.pop("sensor_frame"), input_stage)
        stage = pypeln.thread.map(
            f=self.encode_bounding_boxes_3d, stage=stage, workers=self.workers, maxsize=self.in_queue_size
        )

        return stage
