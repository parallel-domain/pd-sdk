from paralleldomain.common.dgp.v1 import annotations_pb2
from paralleldomain.common.dgp.v1.constants import ANNOTATION_TYPE_MAP_INV, DirectoryName
from paralleldomain.encoding.dgp.v1.format.common import ANNOTATIONS_KEY, CUSTOM_FORMAT_KEY, CommonDGPV1FormatMixin
from paralleldomain.encoding.dgp.v1.utils import _attribute_key_dump, _attribute_value_dump
from paralleldomain.encoding.pipeline_encoder import PipelineItem
from paralleldomain.model.annotation import AnnotationTypes, Polygon2D, Polygons2D
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath


class Polygons2DDGPV1Mixin(CommonDGPV1FormatMixin):
    def save_polygons_2d_and_write_state(
        self,
        pipeline_item: PipelineItem,
        data: Polygons2D,
        scene_output_path: AnyPath,
        sim_offset: float,
        save_binary: bool,
    ):
        polygon2d_dto = [self.encode_polygon_2d(p) for p in data.polygons]
        polygons2d_dto = annotations_pb2.Polygon2DAnnotations(annotations=polygon2d_dto)

        output_path = self.get_file_output_path(
            scene_reference_timestamp=pipeline_item.scene_reference_timestamp,
            sim_offset=sim_offset,
            target_sensor_name=pipeline_item.target_sensor_name,
            timestamp=pipeline_item.sensor_frame.date_time,
            file_suffix="json" if not save_binary else "bin",
            directory_name=DirectoryName.POLYGON_2D,
            scene_output_path=scene_output_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path = fsio.write_message(obj=polygons2d_dto, path=output_path, append_sha1=False)

        pipeline_item.custom_data[CUSTOM_FORMAT_KEY][ANNOTATIONS_KEY][
            str(ANNOTATION_TYPE_MAP_INV[AnnotationTypes.Polygons2D])
        ] = output_path

    @staticmethod
    def encode_polygon_2d(polygon: Polygon2D) -> annotations_pb2.Polygon2DAnnotation:
        polygon_proto = annotations_pb2.Polygon2DAnnotation(
            class_id=polygon.class_id,
            attributes={_attribute_key_dump(k): _attribute_value_dump(v) for k, v in polygon.attributes.items()},
            vertices=[annotations_pb2.KeyPoint2D(x=ll.start.x, y=ll.start.y) for ll in polygon.lines]
            + [annotations_pb2.KeyPoint2D(x=polygon.lines[-1].end.x, y=polygon.lines[-1].end.y)],
        )

        return polygon_proto
