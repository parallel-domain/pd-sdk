from typing import Union

from paralleldomain.common.dgp.v1 import annotations_pb2, geometry_pb2
from paralleldomain.common.dgp.v1.constants import DirectoryName
from paralleldomain.encoding.dgp.v1.format.common import ANNOTATIONS_KEY, CUSTOM_FORMAT_KEY, CommonDGPV1FormatMixin
from paralleldomain.encoding.dgp.v1.utils import _attribute_key_dump, _attribute_value_dump
from paralleldomain.encoding.pipeline_encoder import PipelineItem
from paralleldomain.model.annotation import (
    AnnotationTypes,
    BoundingBox2D,
    BoundingBox3D,
    BoundingBoxes2D,
    BoundingBoxes3D,
)
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath


class BoundingBox3DDGPV1Mixin(CommonDGPV1FormatMixin):
    def save_boxes_3d_and_write_state(
        self,
        pipeline_item: PipelineItem,
        data: Union[AnyPath, BoundingBoxes3D],
        scene_output_path: AnyPath,
        sim_offset: float,
        save_binary: bool,
    ):

        output_path = self.get_file_output_path(
            scene_reference_timestamp=pipeline_item.scene_reference_timestamp,
            sim_offset=sim_offset,
            target_sensor_name=pipeline_item.target_sensor_name,
            timestamp=pipeline_item.sensor_frame.date_time,
            file_suffix="json" if not save_binary else "bin",
            directory_name=DirectoryName.BOUNDING_BOX_3D,
            scene_output_path=scene_output_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, AnyPath):
            output_path = fsio.copy_file(source=data, target=output_path)
        else:
            annotations = [self.encode_bounding_box_3d(b) for b in data.boxes]
            boxes3d_dto = annotations_pb2.BoundingBox3DAnnotations(annotations=annotations)
            output_path = str(fsio.write_message(obj=boxes3d_dto, path=output_path, append_sha1=True))

        pipeline_item.custom_data[CUSTOM_FORMAT_KEY][ANNOTATIONS_KEY][
            str(self._annotation_type_map_inv[AnnotationTypes.BoundingBoxes3D])
        ] = output_path

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
