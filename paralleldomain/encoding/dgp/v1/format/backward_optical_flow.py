from typing import Union

import numpy as np

from paralleldomain.common.dgp.v1.constants import ANNOTATION_TYPE_MAP_INV, DirectoryName
from paralleldomain.encoding.dgp.v1.format.common import ANNOTATIONS_KEY, CUSTOM_FORMAT_KEY, CommonDGPV1FormatMixin
from paralleldomain.encoding.pipeline_encoder import PipelineItem
from paralleldomain.model.annotation import AnnotationTypes, BackwardOpticalFlow
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.mask import encode_2int16_as_rgba8


class BackwardOpticalFlowDGPV1Mixin(CommonDGPV1FormatMixin):
    def save_backward_optical_flow_and_write_state(
        self,
        data: Union[BackwardOpticalFlow, AnyPath],
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
            directory_name=DirectoryName.BACK_MOTION_VECTORS_2D,
            scene_output_path=scene_output_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(flow_or_path, AnyPath):
            save_path = fsio.copy_file(source=flow_or_path, target=output_path)
        else:
            vectors = flow_or_path.vectors
            height, width = vectors.shape[0:2]
            vectors = vectors / 2
            vectors = vectors / [width, height]
            vectors = vectors + 0.5
            vectors = vectors * 65535.0
            vectors = vectors.astype(np.int)
            save_path = fsio.write_png(obj=encode_2int16_as_rgba8(vectors), path=output_path)

        pipeline_item.custom_data[CUSTOM_FORMAT_KEY][ANNOTATIONS_KEY][
            str(ANNOTATION_TYPE_MAP_INV[AnnotationTypes.OpticalFlow])
        ] = save_path
        return save_path
