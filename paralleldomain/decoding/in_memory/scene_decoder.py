from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, TypeVar, Union

from paralleldomain import Scene
from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.frame import Frame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName

TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


@dataclass
class InMemorySceneDecoder:
    description: str = ""
    frame_ids: List[FrameId] = field(default_factory=list)
    camera_names: List[SensorName] = field(default_factory=list)
    lidar_names: List[SensorName] = field(default_factory=list)
    radar_names: List[SensorName] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    frames: Dict[FrameId, Frame] = field(default_factory=dict)
    class_maps: Dict[AnnotationType, ClassMap] = field(default_factory=dict)
    frame_id_to_date_time_map: Dict[FrameId, datetime] = field(default_factory=dict)

    def get_set_description(self, scene_name: SceneName) -> str:
        return self.description

    def get_set_metadata(self, scene_name: SceneName) -> Dict[str, Any]:
        return self.metadata

    def get_frame(
        self,
        scene_name: SceneName,
        frame_id: FrameId,
    ) -> Frame[TDateTime]:
        return self.frames[frame_id]

    def get_sensor_names(self, scene_name: SceneName) -> List[str]:
        return self.camera_names + self.lidar_names + self.radar_names

    def get_camera_names(self, scene_name: SceneName) -> List[str]:
        return self.camera_names

    def get_lidar_names(self, scene_name: SceneName) -> List[str]:
        return self.lidar_names

    def get_radar_names(self, scene_name: SceneName) -> List[str]:
        return self.radar_names

    def get_frame_ids(self, scene_name: SceneName) -> Set[FrameId]:
        return set(self.frame_ids)

    def get_class_maps(self, scene_name: SceneName) -> Dict[AnnotationType, ClassMap]:
        return self.class_maps

    def get_camera_sensor(self, scene_name: SceneName, camera_name: SensorName):
        raise NotImplementedError("Not supported!")

    def get_lidar_sensor(self, scene_name: SceneName, lidar_name: SensorName):
        raise NotImplementedError("Not supported!")

    def get_radar_sensor(self, scene_name: SceneName, radar_name: SensorName):
        raise NotImplementedError("Not supported!")

    def get_frame_id_to_date_time_map(self, scene_name: SceneName) -> Dict[FrameId, datetime]:
        return self.frame_id_to_date_time_map

    @staticmethod
    def from_scene(scene: Scene) -> "InMemorySceneDecoder":
        return InMemorySceneDecoder(
            frame_ids=scene.frame_ids,
            camera_names=scene.camera_names,
            lidar_names=scene.lidar_names,
            class_maps=scene.class_maps,
            metadata=dict(scene.metadata),
            frame_id_to_date_time_map=scene._decoder.get_frame_id_to_date_time_map(scene_name=scene.name),
        )

    def clear_from_cache(self, scene_name: SceneName):
        pass
