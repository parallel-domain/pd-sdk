from functools import lru_cache
from typing import Set

from paralleldomain.decoding.cityscapes.common import get_scene_path
from paralleldomain.decoding.cityscapes.sensor_frame_decoder import CityscapesCameraSensorFrameDecoder
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class CityscapesCameraSensorDecoder(CameraSensorDecoder[None]):
    def __init__(self, dataset_name: str, scene_name: SceneName, dataset_path: AnyPath, settings: DecoderSettings):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self.dataset_path = dataset_path
        self._create_camera_sensor_frame_decoder = lru_cache(maxsize=1)(self._create_camera_sensor_frame_decoder)

    def _decode_frame_id_set(self, sensor_name: SensorName) -> Set[FrameId]:
        frame_ids = set()
        scene_images_folder = get_scene_path(
            dataset_path=self.dataset_path, scene_name=self.scene_name, camera_name=sensor_name
        )
        file_names = [path.name for path in scene_images_folder.iterdir()]
        frame_ids.update(file_names)
        return frame_ids

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[None]:
        return CityscapesCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self.dataset_path,
            settings=self.settings,
        )
