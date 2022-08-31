from datetime import datetime
from functools import lru_cache
from typing import List, Optional, Set

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.flying_things.common import get_scene_folder
from paralleldomain.decoding.flying_things.sensor_frame_decoder import FlyingThingsCameraSensorFrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


class FlyingThingsCameraSensorDecoder(CameraSensorDecoder[datetime]):
    def __init__(
        self,
        dataset_name: str,
        scene_name: SceneName,
        dataset_path: AnyPath,
        settings: DecoderSettings,
        split_name: str,
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self._dataset_path = dataset_path
        self._split_name = split_name
        self._create_camera_sensor_frame_decoder = lru_cache(maxsize=1)(self._create_camera_sensor_frame_decoder)

    def _decode_frame_id_set(self, sensor_name: SensorName) -> Set[FrameId]:
        folder_path = (
            get_scene_folder(dataset_path=self._dataset_path, scene_name=self.scene_name, split_name=self._split_name)
            / sensor_name
        )

        frame_ids = {img.split(".png")[0] for img in folder_path.glob("*.png")}
        return frame_ids

    def _decode_camera_sensor_frame(
        self, decoder: CameraSensorFrameDecoder[datetime], frame_id: FrameId, camera_name: SensorName
    ) -> CameraSensorFrame[datetime]:
        return CameraSensorFrame[datetime](sensor_name=camera_name, frame_id=frame_id, decoder=decoder)

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[datetime]:
        return FlyingThingsCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self._dataset_path,
            settings=self.settings,
            split_name=self._split_name,
        )
