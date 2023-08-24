from typing import Optional

from paralleldomain.decoding.directory.frame_decoder import DirectoryFrameDecoder
from paralleldomain.decoding.kitti.sensor_frame_decoder import KittiLidarSensorFrameDecoder
from paralleldomain.decoding.sensor_frame_decoder import LidarSensorFrameDecoder
from paralleldomain.model.sensor import LidarSensorFrame
from paralleldomain.model.type_aliases import FrameId, SensorName, SceneName


class KittiFrameDecoder(DirectoryFrameDecoder):
    def __init__(
        self,
        pointcloud_dim: int,
        **kwargs,
    ):
        self.pointcloud_dim = pointcloud_dim
        super().__init__(**kwargs)

    def _create_lidar_sensor_frame_decoder(self) -> LidarSensorFrameDecoder[None]:
        return KittiLidarSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            settings=self.settings,
            folder_to_data_type=self._folder_to_data_type,
            metadata_folder=self._metadata_folder,
            class_map=self._class_map,
            pointcloud_dim=self.pointcloud_dim,
        )

    def _decode_lidar_sensor_frame(
        self, decoder: LidarSensorFrameDecoder[None], frame_id: FrameId, sensor_name: SensorName
    ) -> LidarSensorFrame[None]:
        return LidarSensorFrame(sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)
