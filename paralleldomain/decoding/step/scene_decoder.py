from datetime import datetime
from typing import Any, Dict, List, Set

from pd.data_lab.sim_state import SimState

from paralleldomain.common.constants import ANNOTATION_NAME_TO_CLASS
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import SceneDecoder, TDateTime
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder, RadarSensorDecoder
from paralleldomain.decoding.step.constants import PD_CLASS_DETAILS
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName


class StepSceneDecoder(SceneDecoder[datetime]):
    def __init__(
        self,
        sim_state: SimState,
        dataset_name: str,
        settings: DecoderSettings,
    ):
        super().__init__(dataset_name=dataset_name, settings=settings)
        self._sim_state = sim_state

    def _decode_set_metadata(self, scene_name: SceneName) -> Dict[str, Any]:
        # if self._sim_state.current_state is not None:
        #     world_info = self._sim_state.current_state.world_info
        #     return dict(
        #         location=world_info.location,
        #         time_of_day=world_info.time_of_day,
        #         rain_intensity=world_info.rain_intensity,
        #         fog_intensity=world_info.fog_intensity,
        #         wettness=world_info.wetness,
        #     )
        return dict()

    def _decode_set_description(self, scene_name: SceneName) -> str:
        return ""

    def _decode_frame_id_set(self, scene_name: SceneName) -> Set[FrameId]:
        return set(self._sim_state.frame_ids)

    def _decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        return [sensor.name for sensor in self._sim_state.sensor_rig.sensors]

    def _decode_camera_names(self, scene_name: SceneName) -> List[SensorName]:
        return [sensor.name for sensor in self._sim_state.sensor_rig.sensors if sensor.is_camera]

    def _decode_lidar_names(self, scene_name: SceneName) -> List[SensorName]:
        return [sensor.name for sensor in self._sim_state.sensor_rig.sensors if sensor.is_lidar]

    def _decode_radar_names(self, scene_name: SceneName) -> List[SensorName]:
        return []

    def _decode_class_maps(self, scene_name: SceneName) -> Dict[AnnotationType, ClassMap]:
        return {
            anno_type: ClassMap(classes=PD_CLASS_DETAILS)
            for anno_type in ANNOTATION_NAME_TO_CLASS.values()
            if anno_type in self._sim_state.sensor_rig.available_annotations
        }

    def _create_camera_sensor_decoder(
        self, scene_name: SceneName, camera_name: SensorName, dataset_name: str
    ) -> CameraSensorDecoder[TDateTime]:
        raise NotImplementedError("Not supported!")

    def _create_lidar_sensor_decoder(
        self, scene_name: SceneName, lidar_name: SensorName, dataset_name: str
    ) -> LidarSensorDecoder[TDateTime]:
        raise NotImplementedError("Not supported!")

    def _create_radar_sensor_decoder(
        self, scene_name: SceneName, radar_name: SensorName, dataset_name: str
    ) -> RadarSensorDecoder[TDateTime]:
        raise NotImplementedError("Not supported!")

    def _create_frame_decoder(
        self, scene_name: SceneName, frame_id: FrameId, dataset_name: str
    ) -> FrameDecoder[TDateTime]:
        raise NotImplementedError("Not supported!")

    def _decode_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, TDateTime]:
        return {fid: dt for fid, dt in zip(self._sim_state.frame_ids, self._sim_state.frame_date_times)}
