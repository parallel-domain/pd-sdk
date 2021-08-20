from functools import lru_cache
from typing import Dict, List, Set, cast

from paralleldomain.common.dgp.v0.dtos import SceneDataDTO, SceneSampleDTO
from paralleldomain.decoding.dgp.sensor_frame_decoder import DGPSensorFrameDecoder
from paralleldomain.decoding.sensor_decoder import SensorDecoder
from paralleldomain.decoding.sensor_frame_decoder import SensorFrameDecoder
from paralleldomain.model.sensor import SensorFrame, TemporalSensorFrame
from paralleldomain.model.transformation import Transformation
from paralleldomain.model.type_aliases import FrameId, SensorFrameSetName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.lazy_load_cache import LazyLoadCache


class DGPSensorDecoder(SensorDecoder):
    def __init__(
        self,
        dataset_name: str,
        set_name: SensorFrameSetName,
        lazy_load_cache: LazyLoadCache,
        dataset_path: AnyPath,
        scene_samples: Dict[FrameId, SceneSampleDTO],
        scene_data: List[SceneDataDTO],
        custom_reference_to_box_bottom: Transformation,
    ):
        super().__init__(dataset_name=dataset_name, set_name=set_name, lazy_load_cache=lazy_load_cache)
        self.scene_data = scene_data
        self.custom_reference_to_box_bottom = custom_reference_to_box_bottom
        self.scene_samples = scene_samples
        self.dataset_path = dataset_path

    def _decode_frame_id_set(self, sensor_name: SensorName) -> Set[FrameId]:
        return set()  # Todo

    def _decode_sensor_frame(
        self, decoder: SensorFrameDecoder, frame_id: FrameId, sensor_name: SensorName
    ) -> SensorFrame:
        decoder = cast(DGPSensorFrameDecoder, decoder)
        return TemporalSensorFrame(sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)

    @lru_cache(maxsize=1)
    def _create_sensor_frame_decoder(self) -> DGPSensorFrameDecoder:
        return DGPSensorFrameDecoder(
            dataset_name=self.dataset_name,
            set_name=self.set_name,
            lazy_load_cache=self.lazy_load_cache,
            dataset_path=self.dataset_path,
            scene_samples=self.scene_samples,
            scene_data=self.scene_data,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
        )
