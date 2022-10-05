from typing import Union

import numpy as np

from paralleldomain.common.dgp.v1.constants import DirectoryName
from paralleldomain.encoding.dgp.v1.format.common import ANNOTATIONS_KEY, CUSTOM_FORMAT_KEY, CommonDGPV1FormatMixin
from paralleldomain.encoding.pipeline_encoder import PipelineItem
from paralleldomain.model.annotation import AnnotationTypes, InstanceSegmentation3D
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath


class InstanceSegmentation3DDGPV1Mixin(CommonDGPV1FormatMixin):
    def save_instance_segmentation_3d_and_write_state(
        self,
        data: Union[InstanceSegmentation3D, AnyPath],
        pipeline_item: PipelineItem,
        scene_output_path: AnyPath,
        sim_offset: float,
    ) -> AnyPath:
        file_suffix = "npz"
        segmentation_or_path = data
        if isinstance(segmentation_or_path, AnyPath):
            file_suffix = segmentation_or_path.suffix

        output_path = self.get_file_output_path(
            scene_reference_timestamp=pipeline_item.scene_reference_timestamp,
            sim_offset=sim_offset,
            target_sensor_name=pipeline_item.target_sensor_name,
            timestamp=pipeline_item.sensor_frame.date_time,
            file_suffix=file_suffix,
            directory_name=DirectoryName.INSTANCE_SEGMENTATION_3D,
            scene_output_path=scene_output_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(segmentation_or_path, AnyPath):
            save_path = fsio.copy_file(source=segmentation_or_path, target=output_path)
        else:
            mask_out = segmentation_or_path.instance_ids.astype(np.uint32)
            save_path = fsio.write_npz(obj=dict(instance=mask_out), path=output_path)

        pipeline_item.custom_data[CUSTOM_FORMAT_KEY][ANNOTATIONS_KEY][
            str(self._annotation_type_map_inv[AnnotationTypes.InstanceSegmentation3D])
        ] = save_path
        return save_path
