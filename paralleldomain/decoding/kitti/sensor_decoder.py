from paralleldomain.decoding.directory.sensor_decoder import DirectoryLidarSensorDecoder
from paralleldomain.decoding.kitti.sensor_frame_decoder import KittiLidarSensorFrameDecoder
from paralleldomain.decoding.sensor_frame_decoder import LidarSensorFrameDecoder
from paralleldomain.model.sensor import LidarSensorFrame
from paralleldomain.model.type_aliases import FrameId, SensorName


class KittiLidarSensorDecoder(DirectoryLidarSensorDecoder):
    def __init__(
        self,
        point_cloud_dim: int,
        sensor_name: SensorName,
        **kwargs,
    ):
        self.point_cloud_dim = point_cloud_dim
        super().__init__(
            sensor_name=sensor_name,
            **kwargs,
        )

    def _create_lidar_sensor_frame_decoder(self, frame_id: FrameId) -> LidarSensorFrameDecoder[None]:
        return KittiLidarSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=self.sensor_name,
            frame_id=frame_id,
            dataset_path=self.dataset_path,
            settings=self.settings,
            folder_to_data_type=self.folder_to_data_type,
            metadata_folder=self._metadata_folder,
            class_map=self._class_map,
            point_cloud_dim=self.point_cloud_dim,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )
