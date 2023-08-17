from datetime import datetime
from typing import Any, Dict, List, Set, Union

import pd.state
from paralleldomain.common.constants import ANNOTATION_NAME_TO_CLASS
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import SceneDecoder, TDateTime
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_decoder import CameraSensorDecoder, LidarSensorDecoder, RadarSensorDecoder
from paralleldomain.decoding.step.common import get_sensor_rig_annotation_types
from paralleldomain.decoding.step.constants import PD_CLASS_DETAILS
from paralleldomain.model.annotation import AnnotationType, AnnotationIdentifier, AnnotationTypes
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName


class StepSceneDecoder(SceneDecoder[datetime]):
    def __init__(
        self,
        sensor_rig: List[Union[pd.state.CameraSensor, pd.state.LiDARSensor]],
        dataset_name: str,
        settings: DecoderSettings,
    ):
        super().__init__(dataset_name=dataset_name, settings=settings)
        self._sensor_rig = sensor_rig
        self._frame_ids = list()
        self._frame_id_to_date_time_map = dict()

    def add_frame(self, frame_id: FrameId, date_time: datetime):
        self._frame_id_to_date_time_map[frame_id] = date_time
        self._frame_ids.append(frame_id)

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

    @property
    def available_annotations(self) -> List[AnnotationType]:
        return get_sensor_rig_annotation_types(sensor_rig=self._sensor_rig)

    def _decode_set_description(self, scene_name: SceneName) -> str:
        return ""

    def _decode_frame_id_set(self, scene_name: SceneName) -> Set[FrameId]:
        return set(self._frame_ids)

    def _decode_sensor_names(self, scene_name: SceneName) -> List[SensorName]:
        return [sensor.name for sensor in self._sensor_rig]

    def _decode_camera_names(self, scene_name: SceneName) -> List[SensorName]:
        return [sensor.name for sensor in self._sensor_rig if isinstance(sensor, pd.state.CameraSensor)]

    def _decode_lidar_names(self, scene_name: SceneName) -> List[SensorName]:
        return [sensor.name for sensor in self._sensor_rig if isinstance(sensor, pd.state.LiDARSensor)]

    def _decode_radar_names(self, scene_name: SceneName) -> List[SensorName]:
        return []

    def _decode_class_maps(self, scene_name: SceneName) -> Dict[AnnotationIdentifier, ClassMap]:
        return {
            AnnotationIdentifier(annotation_type=anno_type): ClassMap(classes=PD_CLASS_DETAILS)
            for anno_type in ANNOTATION_NAME_TO_CLASS.values()
            if anno_type in self.available_annotations
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
        return self._frame_id_to_date_time_map

    def _decode_available_annotation_identifiers(self, scene_name: SceneName) -> List[AnnotationIdentifier]:
        annotation_identifiers = list()
        for sensor in self._sensor_rig:
            if isinstance(sensor, pd.state.CameraSensor):
                if sensor.capture_depth:
                    annotation_identifiers.append(AnnotationIdentifier(annotation_type=AnnotationTypes.Depth))
                if sensor.capture_instances:
                    annotation_identifiers.append(
                        AnnotationIdentifier(annotation_type=AnnotationTypes.InstanceSegmentation2D)
                    )
                if sensor.capture_segmentation:
                    annotation_identifiers.append(
                        AnnotationIdentifier(annotation_type=AnnotationTypes.SemanticSegmentation2D)
                    )
                if sensor.capture_normals:
                    annotation_identifiers.append(
                        AnnotationIdentifier(annotation_type=AnnotationTypes.SurfaceNormals2D)
                    )
        return annotation_identifiers
