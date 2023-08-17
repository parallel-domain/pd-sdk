from datetime import datetime
from functools import lru_cache
from typing import List, Set

from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.flying_things.common import decode_frame_id_set
from paralleldomain.decoding.flying_things.sensor_frame_decoder import FlyingThingsCameraSensorFrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder
from paralleldomain.decoding.sensor_frame_decoder import CameraSensorFrameDecoder
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
        split_list: List[int],
        is_driving_subset: bool,
        is_full_dataset_format: bool = False,
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self._is_driving_subset = is_driving_subset
        self._is_full_dataset_format = is_full_dataset_format
        self._split_list = split_list
        self._dataset_path = dataset_path
        self._split_name = split_name
        self._create_camera_sensor_frame_decoder = lru_cache(maxsize=1)(self._create_camera_sensor_frame_decoder)

    def _decode_frame_id_set(self, sensor_name: SensorName) -> Set[FrameId]:
        return decode_frame_id_set(
            scene_name=self.scene_name,
            split_name=self._split_name,
            split_list=self._split_list,
            is_full_dataset_format=self._is_full_dataset_format,
            dataset_path=self._dataset_path,
            sensor_name=sensor_name,
        )

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[datetime]:
        return FlyingThingsCameraSensorFrameDecoder(
            dataset_name=self.dataset_name,
            scene_name=self.scene_name,
            dataset_path=self._dataset_path,
            settings=self.settings,
            split_name=self._split_name,
            split_list=self._split_list,
            is_full_dataset_format=self._is_full_dataset_format,
            is_driving_subset=self._is_driving_subset,
        )
