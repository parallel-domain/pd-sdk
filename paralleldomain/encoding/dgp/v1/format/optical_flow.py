from typing import Union

from paralleldomain.common.dgp.v1.constants import ANNOTATION_TYPE_MAP_INV, DirectoryName
from paralleldomain.encoding.dgp.v1.format.common import (
    ANNOTATIONS_KEY,
    CUSTOM_FORMAT_KEY,
    CommonDGPV1FormatMixin,
    encode_flow_vectors,
)
from paralleldomain.encoding.pipeline_encoder import PipelineItem
from paralleldomain.model.annotation import AnnotationTypes, OpticalFlow
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath


class OpticalFlowDGPV1Mixin(CommonDGPV1FormatMixin):
    def save_optical_flow_and_write_state(
        self,
        data: Union[OpticalFlow, AnyPath],
        pipeline_item: PipelineItem,
        scene_output_path: AnyPath,
        sim_offset: float,
    ) -> AnyPath:
        flow_or_path = data
        file_suffix = "png"
        if isinstance(flow_or_path, AnyPath):
            file_suffix = flow_or_path.suffix

        output_path = self.get_file_output_path(
            scene_reference_timestamp=pipeline_item.scene_reference_timestamp,
            sim_offset=sim_offset,
            target_sensor_name=pipeline_item.target_sensor_name,
            timestamp=pipeline_item.sensor_frame.date_time,
            file_suffix=file_suffix,
            directory_name=DirectoryName.MOTION_VECTORS_2D,
            scene_output_path=scene_output_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(flow_or_path, AnyPath):
            save_path = fsio.copy_file(source=flow_or_path, target=output_path)
        else:
            save_path = fsio.write_png(obj=encode_flow_vectors(flow_or_path.vectors), path=output_path)

        pipeline_item.custom_data[CUSTOM_FORMAT_KEY][ANNOTATIONS_KEY][
            str(ANNOTATION_TYPE_MAP_INV[AnnotationTypes.OpticalFlow])
        ] = save_path
        return save_path
