from typing import Union

import numpy as np

from paralleldomain.common.dgp.v1.constants import DirectoryName
from paralleldomain.encoding.dgp.v1.format.common import CUSTOM_FORMAT_KEY, SENSOR_DATA_KEY, CommonDGPV1FormatMixin
from paralleldomain.encoding.pipeline_encoder import PipelineItem
from paralleldomain.model.point_cloud import PointCloud
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath


class PointCloudDGPV1Mixin(CommonDGPV1FormatMixin):
    def save_point_cloud_and_write_state(
        self,
        data: Union[PointCloud, AnyPath],
        pipeline_item: PipelineItem,
        scene_output_path: AnyPath,
        sim_offset: float,
    ) -> AnyPath:
        cloud_or_path = data
        file_suffix = "npz"
        if isinstance(cloud_or_path, AnyPath):
            file_suffix = cloud_or_path.suffix

        output_path = self.get_file_output_path(
            scene_reference_timestamp=pipeline_item.scene_reference_timestamp,
            sim_offset=sim_offset,
            target_sensor_name=pipeline_item.target_sensor_name,
            timestamp=pipeline_item.sensor_frame.date_time,
            file_suffix=file_suffix,
            directory_name=DirectoryName.POINT_CLOUD,
            scene_output_path=scene_output_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(cloud_or_path, AnyPath):
            save_path = fsio.copy_file(source=cloud_or_path, target=output_path)
        else:
            save_path = self.encode_point_cloud(point_cloud=cloud_or_path, output_path=output_path)

        pipeline_item.custom_data[CUSTOM_FORMAT_KEY][SENSOR_DATA_KEY][DirectoryName.POINT_CLOUD] = save_path
        return save_path

    @staticmethod
    def encode_point_cloud(point_cloud: PointCloud, output_path: AnyPath) -> AnyPath:
        pc_dtypes = [
            ("X", "<f4"),
            ("Y", "<f4"),
            ("Z", "<f4"),
            ("INTENSITY", "<f4"),
            ("R", "<f4"),
            ("G", "<f4"),
            ("B", "<f4"),
            ("RING_ID", "<u4"),
            ("TIMESTAMP", "<u8"),
        ]

        row_count = point_cloud.length
        pc_data = np.empty(row_count, dtype=pc_dtypes)

        pc_data["X"] = point_cloud.xyz[:, 0]
        pc_data["Y"] = point_cloud.xyz[:, 1]
        pc_data["Z"] = point_cloud.xyz[:, 2]
        pc_data["INTENSITY"] = point_cloud.intensity[:, 0]
        pc_data["R"] = point_cloud.rgb[:, 0]
        pc_data["G"] = point_cloud.rgb[:, 1]
        pc_data["B"] = point_cloud.rgb[:, 2]
        pc_data["RING_ID"] = point_cloud.ring[:, 0]
        pc_data["TIMESTAMP"] = point_cloud.ts[:, 0]

        return fsio.write_npz(obj={"data": pc_data}, path=output_path)
