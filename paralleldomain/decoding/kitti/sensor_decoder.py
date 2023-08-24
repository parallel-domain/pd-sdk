from paralleldomain.decoding.directory.sensor_decoder import DirectoryLidarSensorDecoder
from paralleldomain.decoding.kitti.sensor_frame_decoder import KittiLidarSensorFrameDecoder
from paralleldomain.decoding.sensor_frame_decoder import LidarSensorFrameDecoder
from paralleldomain.model.sensor import LidarSensorFrame
from paralleldomain.model.type_aliases import SensorName, FrameId


class KittiLidarSensorDecoder(DirectoryLidarSensorDecoder):
    def __init__(
        self,
        pointcloud_dim: int,
        **kwargs,
    ):
        self.pointcloud_dim = pointcloud_dim
        super().__init__(
            **kwargs,
        )

    def _create_lidar_sensor_frame_decoder(self) -> LidarSensorFrameDecoder[None]:
        return KittiLidarSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            settings=self.settings,
            folder_to_data_type=self.folder_to_data_type,
            metadata_folder=self._metadata_folder,
            class_map=self._class_map,
            pointcloud_dim=self.pointcloud_dim,
        )
