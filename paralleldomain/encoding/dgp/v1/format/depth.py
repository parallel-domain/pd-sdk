from typing import Union

import numpy as np

from paralleldomain.common.dgp.v1.constants import ANNOTATION_TYPE_MAP_INV, DirectoryName
from paralleldomain.encoding.dgp.v1.format.common import ANNOTATIONS_KEY, CUSTOM_FORMAT_KEY, CommonDGPV1FormatMixin
from paralleldomain.encoding.pipeline_encoder import PipelineItem
from paralleldomain.model.annotation import AnnotationTypes, Depth
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath


class DepthDGPV1Mixin(CommonDGPV1FormatMixin):
    def save_depth_and_write_state(
        self,
        data: Union[Depth, AnyPath],
        pipeline_item: PipelineItem,
        scene_output_path: AnyPath,
        sim_offset: float,
    ) -> AnyPath:
        file_suffix = "npz"
        depth_or_path = data
        if isinstance(depth_or_path, AnyPath):
            file_suffix = depth_or_path.suffix

        output_path = self.get_file_output_path(
            scene_reference_timestamp=pipeline_item.scene_reference_timestamp,
            sim_offset=sim_offset,
            target_sensor_name=pipeline_item.target_sensor_name,
            timestamp=pipeline_item.sensor_frame.date_time,
            file_suffix=file_suffix,
            directory_name=DirectoryName.DEPTH,
            scene_output_path=scene_output_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(depth_or_path, AnyPath):
            save_path = fsio.copy_file(source=depth_or_path, target=output_path)
        else:
            save_path = fsio.write_npz(obj=dict(data=depth_or_path.depth[..., 0].astype(np.float32)), path=output_path)

        pipeline_item.custom_data[CUSTOM_FORMAT_KEY][ANNOTATIONS_KEY][
            str(ANNOTATION_TYPE_MAP_INV[AnnotationTypes.Depth])
        ] = save_path
        return save_path
