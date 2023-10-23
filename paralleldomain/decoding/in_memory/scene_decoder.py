from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Set, TypeVar, Union

from paralleldomain import Scene
from paralleldomain.model.annotation import AnnotationIdentifier
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.frame import Frame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName

TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


@dataclass
class InMemorySceneDecoder:
    scene_name: str
    description: str = ""
    frame_ids: List[FrameId] = field(default_factory=list)
    camera_names: List[SensorName] = field(default_factory=list)
    lidar_names: List[SensorName] = field(default_factory=list)
    radar_names: List[SensorName] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    frames: Dict[FrameId, Frame] = field(default_factory=dict)
    class_maps: Dict[AnnotationIdentifier, ClassMap] = field(default_factory=dict)
    frame_id_to_date_time_map: Dict[FrameId, datetime] = field(default_factory=dict)
    available_annotation_identifiers: List[AnnotationIdentifier] = field(default_factory=list)

    def get_set_description(self) -> str:
        return self.description

    def get_set_metadata(self) -> Dict[str, Any]:
        return self.metadata

    def get_frame(
        self,
        frame_id: FrameId,
    ) -> Frame[TDateTime]:
        return self.frames[frame_id]

    def get_sensor_names(self) -> List[str]:
        return self.camera_names + self.lidar_names + self.radar_names

    def get_camera_names(self) -> List[str]:
        return self.camera_names

    def get_lidar_names(self) -> List[str]:
        return self.lidar_names

    def get_radar_names(self) -> List[str]:
        return self.radar_names

    def get_frame_ids(self) -> Set[FrameId]:
        return set(self.frame_ids)

    def get_class_maps(self) -> Dict[AnnotationIdentifier, ClassMap]:
        return self.class_maps

    def get_camera_sensor(self, camera_name: SensorName):
        raise NotImplementedError("Not supported!")

    def get_lidar_sensor(self, lidar_name: SensorName):
        raise NotImplementedError("Not supported!")

    def get_radar_sensor(self, radar_name: SensorName):
        raise NotImplementedError("Not supported!")

    def get_frame_id_to_date_time_map(self) -> Dict[FrameId, datetime]:
        return self.frame_id_to_date_time_map

    @staticmethod
    def from_scene(scene: Scene) -> "InMemorySceneDecoder":
        return InMemorySceneDecoder(
            scene_name=scene.name,
            frame_ids=scene.frame_ids,
            camera_names=scene.camera_names,
            lidar_names=scene.lidar_names,
            class_maps=scene.class_maps,
            metadata=dict(scene.metadata),
            frame_id_to_date_time_map=scene._decoder.get_frame_id_to_date_time_map(),
            available_annotation_identifiers=scene.available_annotation_identifiers.copy(),
        )

    def clear_from_cache(self):
        pass

    def get_available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        return self.available_annotation_identifiers
