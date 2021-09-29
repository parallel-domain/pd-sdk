from datetime import datetime
from itertools import chain
from typing import Any, Dict, Generator, Generic, List, Set, Type, TypeVar, Union

from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.class_mapping import ClassMap

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

from paralleldomain.model.frame import Frame
from paralleldomain.model.sensor import CameraSensor, LidarSensor
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName

T = TypeVar("T")
TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


class UnorderedSceneDecoderProtocol(Protocol[TDateTime]):
    def get_set_description(self, scene_name: SceneName) -> str:
        pass

    def get_set_metadata(self, scene_name: SceneName) -> Dict[str, Any]:
        pass

    def get_frame(
        self,
        scene_name: SceneName,
        frame_id: FrameId,
    ) -> Frame[TDateTime]:
        pass

    def get_sensor_names(self, scene_name: SceneName) -> List[str]:
        pass

    def get_camera_names(self, scene_name: SceneName) -> List[str]:
        pass

    def get_lidar_names(self, scene_name: SceneName) -> List[str]:
        pass

    def get_frame_ids(self, scene_name: SceneName) -> Set[FrameId]:
        pass

    def get_class_maps(self, scene_name: SceneName) -> Dict[AnnotationType, ClassMap]:
        pass

    def get_camera_sensor(self, scene_name: SceneName, camera_name: SensorName) -> CameraSensor[TDateTime]:
        pass

    def get_lidar_sensor(self, scene_name: SceneName, lidar_name: SensorName) -> LidarSensor[TDateTime]:
        pass


class UnorderedScene(Generic[TDateTime]):
    def __init__(
        self,
        name: SceneName,
        available_annotation_types: List[AnnotationType],
        decoder: UnorderedSceneDecoderProtocol[TDateTime],
    ):
        self._name = name
        self._available_annotation_types = available_annotation_types
        self._decoder = decoder

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._decoder.get_set_description(scene_name=self._name)

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._decoder.get_set_metadata(scene_name=self._name)

    @property
    def frames(self) -> Set[Frame[TDateTime]]:
        return {self.get_frame(frame_id=frame_id) for frame_id in self.frame_ids}

    @property
    def frame_ids(self) -> Set[FrameId]:
        return self._decoder.get_frame_ids(scene_name=self.name)

    @property
    def available_annotation_types(self):
        return self._available_annotation_types

    @property
    def sensor_names(self) -> List[str]:
        return self._decoder.get_sensor_names(scene_name=self.name)

    @property
    def camera_names(self) -> List[str]:
        return self._decoder.get_camera_names(scene_name=self.name)

    @property
    def lidar_names(self) -> List[str]:
        return self._decoder.get_lidar_names(scene_name=self.name)

    def get_frame(self, frame_id: FrameId) -> Frame[TDateTime]:
        return self._decoder.get_frame(scene_name=self.name, frame_id=frame_id)

    @property
    def sensors(self) -> Generator[Union[CameraSensor[TDateTime], LidarSensor[TDateTime]], None, None]:
        return (a for a in chain(self.cameras, self.lidars))

    @property
    def cameras(self) -> Generator[CameraSensor[TDateTime], None, None]:
        return (self.get_camera_sensor(camera_name=camera_name) for camera_name in self.camera_names)

    @property
    def lidars(self) -> Generator[LidarSensor[TDateTime], None, None]:
        return (self.get_lidar_sensor(lidar_name=lidar_name) for lidar_name in self.lidar_names)

    def get_sensor(self, sensor_name: SensorName) -> Union[CameraSensor[TDateTime], LidarSensor[TDateTime]]:
        if sensor_name in self.camera_names:
            return self.get_camera_sensor(camera_name=sensor_name)
        else:
            return self.get_lidar_sensor(lidar_name=sensor_name)

    def get_camera_sensor(self, camera_name: SensorName) -> CameraSensor[TDateTime]:
        return self._decoder.get_camera_sensor(scene_name=self.name, camera_name=camera_name)

    def get_lidar_sensor(self, lidar_name: SensorName) -> LidarSensor[TDateTime]:
        return self._decoder.get_lidar_sensor(scene_name=self.name, lidar_name=lidar_name)

    @property
    def class_maps(self) -> Dict[AnnotationType, ClassMap]:
        return self._decoder.get_class_maps(scene_name=self.name)

    def get_class_map(self, annotation_type: Type[T]) -> ClassMap:
        if annotation_type not in self._available_annotation_types:
            raise ValueError(f"No annotation type {annotation_type} available in this dataset!")
        return self.class_maps[annotation_type]
