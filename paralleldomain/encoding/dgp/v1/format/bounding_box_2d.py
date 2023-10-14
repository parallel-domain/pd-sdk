from typing import Union

from paralleldomain.common.dgp.v1 import annotations_pb2
from paralleldomain.common.dgp.v1.constants import ANNOTATION_TYPE_MAP_INV, DirectoryName
from paralleldomain.encoding.dgp.v1.format.common import ANNOTATIONS_KEY, CUSTOM_FORMAT_KEY, CommonDGPV1FormatMixin
from paralleldomain.encoding.dgp.v1.utils import _attribute_key_dump, _attribute_value_dump
from paralleldomain.encoding.pipeline_encoder import PipelineItem
from paralleldomain.model.annotation import AnnotationTypes, BoundingBox2D, BoundingBoxes2D
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath


class BoundingBox2DDGPV1Mixin(CommonDGPV1FormatMixin):
    def encode_bounding_box_2d(self, box: BoundingBox2D) -> annotations_pb2.BoundingBox2DAnnotation:
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

    def save_boxes_2d_and_write_state(
        self,
        pipeline_item: PipelineItem,
        data: Union[AnyPath, BoundingBoxes2D],
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
            directory_name=DirectoryName.BOUNDING_BOX_2D,
            scene_output_path=scene_output_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, AnyPath):
            output_path = fsio.copy_file(source=data, target=output_path)
        else:
            annotations = [self.encode_bounding_box_2d(b) for b in data.boxes]
            boxes2d_dto = annotations_pb2.BoundingBox2DAnnotations(annotations=annotations)
            output_path = str(fsio.write_message(obj=boxes2d_dto, path=output_path, append_sha1=False))

        pipeline_item.custom_data[CUSTOM_FORMAT_KEY][ANNOTATIONS_KEY][
            str(ANNOTATION_TYPE_MAP_INV[AnnotationTypes.BoundingBoxes2D])
        ] = output_path
