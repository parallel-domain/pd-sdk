from paralleldomain.decoding.directory.frame_decoder import DirectoryFrameDecoder
from paralleldomain.decoding.gta5.sensor_frame_decoder import GTACameraSensorFrameDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.model.type_aliases import SensorName


class GTAFrameDecoder(DirectoryFrameDecoder):
    def _create_camera_sensor_frame_decoder(self, sensor_name: SensorName) -> CameraSensorFrameDecoder[None]:
        return GTACameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=sensor_name,
            frame_id=self.frame_id,
            dataset_path=self.dataset_path,
            settings=self.settings,
            folder_to_data_type=self._folder_to_data_type,
            metadata_folder=self._metadata_folder,
            class_map=self._class_map,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )
