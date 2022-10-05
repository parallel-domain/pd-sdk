from paralleldomain.common.dgp.v1 import annotations_pb2
from paralleldomain.common.dgp.v1.constants import DirectoryName
from paralleldomain.encoding.dgp.v1.format.common import ANNOTATIONS_KEY, CUSTOM_FORMAT_KEY, CommonDGPV1FormatMixin
from paralleldomain.encoding.dgp.v1.utils import _attribute_key_dump, _attribute_value_dump
from paralleldomain.encoding.pipeline_encoder import PipelineItem
from paralleldomain.model.annotation import AnnotationTypes, Polygon2D, Polygons2D, Polyline2D, Polylines2D
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath


class Polyline2DDGPV1Mixin(CommonDGPV1FormatMixin):
    def save_polyline_2d_and_write_state(
        self,
        pipeline_item: PipelineItem,
        data: Polylines2D,
        scene_output_path: AnyPath,
        sim_offset: float,
        save_binary: bool,
    ):
        keyline2d_dto = [self.encode_key_line_2d(p) for p in data.polylines]
        keylines2d_dto = annotations_pb2.KeyLine2DAnnotations(annotations=keyline2d_dto)

        output_path = self.get_file_output_path(
            scene_reference_timestamp=pipeline_item.scene_reference_timestamp,
            sim_offset=sim_offset,
            target_sensor_name=pipeline_item.target_sensor_name,
            timestamp=pipeline_item.sensor_frame.date_time,
            file_suffix="json" if not save_binary else "bin",
            directory_name=DirectoryName.KEY_LINE_2D,
            scene_output_path=scene_output_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path = fsio.write_message(obj=keylines2d_dto, path=output_path, append_sha1=True)

        pipeline_item.custom_data[CUSTOM_FORMAT_KEY][ANNOTATIONS_KEY][
            str(self._annotation_type_map_inv[AnnotationTypes.Polylines2D])
        ] = output_path

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
