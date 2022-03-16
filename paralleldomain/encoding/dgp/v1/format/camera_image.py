from typing import Union

import numpy as np

from paralleldomain.common.dgp.v1.constants import DirectoryName
from paralleldomain.encoding.dgp.v1.format.common import CUSTOM_FORMAT_KEY, SENSOR_DATA_KEY, CommonDGPV1FormatMixin
from paralleldomain.encoding.pipeline_encoder import PipelineItem
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath


class CameraDGPV1Mixin(CommonDGPV1FormatMixin):
    def save_image_and_write_state(
        self,
        data: Union[np.ndarray, AnyPath],
        pipeline_item: PipelineItem,
        scene_output_path: AnyPath,
        sim_offset: float,
    ) -> AnyPath:
        file_suffix = "png"
        image_or_path = data
        if isinstance(image_or_path, AnyPath):
            file_suffix = image_or_path.suffix

        output_path = self.get_file_output_path(
            scene_reference_timestamp=pipeline_item.scene_reference_timestamp,
            sim_offset=sim_offset,
            target_sensor_name=pipeline_item.target_sensor_name,
            timestamp=pipeline_item.sensor_frame.date_time,
            file_suffix=file_suffix,
            directory_name=DirectoryName.RGB,
            scene_output_path=scene_output_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(image_or_path, AnyPath):
            save_path = fsio.copy_file(source=image_or_path, target=output_path)
        else:
            save_path = fsio.write_png(obj=image_or_path, path=output_path)

        pipeline_item.custom_data[CUSTOM_FORMAT_KEY][SENSOR_DATA_KEY][DirectoryName.RGB] = save_path
        return save_path
