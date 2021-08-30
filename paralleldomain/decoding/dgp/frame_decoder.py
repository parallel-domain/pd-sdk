from datetime import datetime
from functools import lru_cache
from typing import Dict, List, cast

from paralleldomain.common.dgp.v0.dtos import SceneDataDTO, SceneSampleDTO, scene_sample_to_date_time
from paralleldomain.decoding.dgp.sensor_frame_decoder import DGPSensorFrameDecoder
from paralleldomain.decoding.frame_decoder import TemporalFrameDecoder
from paralleldomain.decoding.sensor_frame_decoder import SensorFrameDecoder
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.sensor import SensorFrame, TemporalSensorFrame
from paralleldomain.model.transformation import Transformation
from paralleldomain.model.type_aliases import FrameId, SensorFrameSetName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.lazy_load_cache import LazyLoadCache


class DGPFrameDecoder(TemporalFrameDecoder):
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

    def _decode_ego_pose(self, frame_id: FrameId) -> EgoPose:
        sensor_name = next(iter(self._decode_available_sensor_names(frame_id=frame_id)))
        sensor_frame = self._decode_sensor_frame(
            frame_id=frame_id, sensor_name=sensor_name, decoder=self._create_sensor_frame_decoder()
        )
        vehicle_pose = sensor_frame.pose @ sensor_frame.extrinsic.inverse
        return EgoPose(quaternion=vehicle_pose.quaternion, translation=vehicle_pose.translation)

    def _decode_datetime(self, frame_id: FrameId) -> datetime:
        sample = self.scene_samples[frame_id]
        return scene_sample_to_date_time(sample=sample)

    @lru_cache(maxsize=1)
    def _data_by_key(self) -> Dict[str, SceneDataDTO]:
        return {d.key: d for d in self.scene_data}

    def _decode_available_sensor_names(self, frame_id: FrameId) -> List[SensorName]:
        sample = self.scene_samples[frame_id]
        sensor_data = self._data_by_key()
        return [sensor_data[key].id.name for key in sample.datum_keys]

    def _decode_available_camera_names(self, frame_id: FrameId) -> List[SensorName]:
        sample = self.scene_samples[frame_id]
        sensor_data = self._data_by_key()
        return [sensor_data[key].id.name for key in sample.datum_keys if sensor_data[key].datum.image]

    def _decode_available_lidar_names(self, frame_id: FrameId) -> List[SensorName]:
        sample = self.scene_samples[frame_id]
        sensor_data = self._data_by_key()
        return [sensor_data[key].id.name for key in sample.datum_keys if sensor_data[key].datum.point_cloud]

    def _decode_sensor_frame(
        self, decoder: SensorFrameDecoder, frame_id: FrameId, sensor_name: SensorName
    ) -> SensorFrame:
        decoder = cast(DGPSensorFrameDecoder, decoder)
        return TemporalSensorFrame(sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)

    def _create_sensor_frame_decoder(self) -> SensorFrameDecoder:
        return DGPSensorFrameDecoder(
            dataset_name=self.dataset_name,
            set_name=self.set_name,
            lazy_load_cache=self.lazy_load_cache,
            dataset_path=self.dataset_path,
            scene_samples=self.scene_samples,
            scene_data=self.scene_data,
            custom_reference_to_box_bottom=self.custom_reference_to_box_bottom,
        )
