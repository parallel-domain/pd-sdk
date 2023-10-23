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
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        sensor_name: SensorName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        is_unordered_scene: bool,
        scene_decoder,
    ):
        super().__init__(
            dataset_name=dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            settings=settings,
            scene_decoder=scene_decoder,
            is_unordered_scene=is_unordered_scene,
        )
        self.dataset_path = dataset_path

    def _decode_frame_id_set(self) -> Set[FrameId]:
        frame_ids = set()
        scene_images_folder = get_scene_path(
            dataset_path=self.dataset_path, scene_name=self.scene_name, camera_name=self.sensor_name
        )
        file_names = [path.name for path in scene_images_folder.iterdir()]
        frame_ids.update(file_names)
        return frame_ids

    def _create_camera_sensor_frame_decoder(self, frame_id: FrameId) -> CameraSensorFrameDecoder[None]:
        return CityscapesCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            sensor_name=self.sensor_name,
            frame_id=frame_id,
            dataset_path=self.dataset_path,
            settings=self.settings,
            scene_decoder=self.scene_decoder,
            is_unordered_scene=self.is_unordered_scene,
        )
