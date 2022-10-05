from paralleldomain.common.dgp.v1 import annotations_pb2
from paralleldomain.common.dgp.v1.constants import DirectoryName
from paralleldomain.encoding.dgp.v1.format.common import ANNOTATIONS_KEY, CUSTOM_FORMAT_KEY, CommonDGPV1FormatMixin
from paralleldomain.encoding.dgp.v1.utils import _attribute_key_dump, _attribute_value_dump
from paralleldomain.encoding.pipeline_encoder import PipelineItem
from paralleldomain.model.annotation import AnnotationTypes, Point2D, Points2D
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath


class Point2DDGPV1Mixin(CommonDGPV1FormatMixin):
    def save_points_2d_and_write_state(
        self,
        pipeline_item: PipelineItem,
        data: Points2D,
        scene_output_path: AnyPath,
        sim_offset: float,
        save_binary: bool,
    ):
        keypoint2d_dto = [self._encode_key_point_2d(p) for p in data.points]
        keypoints2d_dto = annotations_pb2.KeyPoint2DAnnotations(annotations=keypoint2d_dto)

        output_path = self.get_file_output_path(
            scene_reference_timestamp=pipeline_item.scene_reference_timestamp,
            sim_offset=sim_offset,
            target_sensor_name=pipeline_item.target_sensor_name,
            timestamp=pipeline_item.sensor_frame.date_time,
            file_suffix="json" if not save_binary else "bin",
            directory_name=DirectoryName.KEY_POINT_2D,
            scene_output_path=scene_output_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path = fsio.write_message(obj=keypoints2d_dto, path=output_path, append_sha1=True)

        pipeline_item.custom_data[CUSTOM_FORMAT_KEY][ANNOTATIONS_KEY][
            str(self._annotation_type_map_inv[AnnotationTypes.Points2D])
        ] = output_path

    @staticmethod
    def _encode_key_point_2d(point: Point2D) -> annotations_pb2.KeyPoint2DAnnotation:
        keypoint_proto = annotations_pb2.KeyPoint2DAnnotation(
            class_id=point.class_id,
            attributes={
                _attribute_key_dump(k): _attribute_value_dump(v) for k, v in point.attributes.items() if k != "key"
            },
            point=annotations_pb2.KeyPoint2D(x=point.x, y=point.y),
            key=point.attributes["key"] if "key" in point.attributes else "",
        )

        return keypoint_proto
