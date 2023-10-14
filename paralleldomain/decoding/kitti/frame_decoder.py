from typing import Optional

from paralleldomain.decoding.directory.frame_decoder import DirectoryFrameDecoder
from paralleldomain.decoding.kitti.sensor_frame_decoder import KittiLidarSensorFrameDecoder
from paralleldomain.decoding.sensor_frame_decoder import LidarSensorFrameDecoder
from paralleldomain.model.sensor import LidarSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName


class KittiFrameDecoder(DirectoryFrameDecoder):
    def __init__(
        self,
        point_cloud_dim: int,
        frame_id: FrameId,
        **kwargs,
    ):
        self.point_cloud_dim = point_cloud_dim
        super().__init__(frame_id=frame_id, **kwargs)

    def _create_lidar_sensor_frame_decoder(self, sensor_name: SensorName) -> LidarSensorFrameDecoder[None]:
        return KittiLidarSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            frame_id=self.frame_id,
            sensor_name=sensor_name,
            dataset_path=self.dataset_path,
            settings=self.settings,
            folder_to_data_type=self._folder_to_data_type,
            metadata_folder=self._metadata_folder,
            class_map=self._class_map,
            point_cloud_dim=self.point_cloud_dim,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )
