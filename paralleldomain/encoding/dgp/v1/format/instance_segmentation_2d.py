from typing import Union

from paralleldomain.common.dgp.v1.constants import DirectoryName
from paralleldomain.encoding.dgp.v1.format.common import ANNOTATIONS_KEY, CUSTOM_FORMAT_KEY, CommonDGPV1FormatMixin
from paralleldomain.encoding.pipeline_encoder import PipelineItem
from paralleldomain.model.annotation import AnnotationTypes, InstanceSegmentation2D
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath


class InstanceSegmentation2DDGPV1Mixin(CommonDGPV1FormatMixin):
    def save_instance_segmentation_2d_and_write_state(
        self,
        data: Union[InstanceSegmentation2D, AnyPath],
        pipeline_item: PipelineItem,
        scene_output_path: AnyPath,
        sim_offset: float,
    ) -> AnyPath:
        file_suffix = "png"
        segmentation_or_path = data
        if isinstance(segmentation_or_path, AnyPath):
            file_suffix = segmentation_or_path.suffix

        output_path = self.get_file_output_path(
            scene_reference_timestamp=pipeline_item.scene_reference_timestamp,
            sim_offset=sim_offset,
            target_sensor_name=pipeline_item.target_sensor_name,
            timestamp=pipeline_item.sensor_frame.date_time,
            file_suffix=file_suffix,
            directory_name=DirectoryName.INSTANCE_SEGMENTATION_2D,
            scene_output_path=scene_output_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(segmentation_or_path, AnyPath):
            save_path = fsio.copy_file(source=segmentation_or_path, target=output_path)
        else:
            save_path = fsio.write_png(obj=segmentation_or_path.rgb_encoded, path=output_path)

        pipeline_item.custom_data[CUSTOM_FORMAT_KEY][ANNOTATIONS_KEY][
            str(self._annotation_type_map_inv[AnnotationTypes.InstanceSegmentation2D])
        ] = save_path
        return save_path
