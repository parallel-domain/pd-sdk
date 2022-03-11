from datetime import datetime
from itertools import chain
from typing import Any, Dict, Generator, Generic, Iterable, List, Optional, Set, Type, TypeVar, Union

from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.map.map import Map

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

from paralleldomain.model.frame import Frame
from paralleldomain.model.sensor import CameraSensor, CameraSensorFrame, LidarSensor, LidarSensorFrame, SensorFrame
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

    def get_map(self, scene_name: SceneName) -> Optional[Map]:
        pass


class UnorderedScene(Generic[TDateTime]):
    """The sensor frames of a UnorderedScene are not temporally ordered.

    Args:
        name: Name of scene
        available_annotation_types: List of available annotation types for this scene.
        decoder: Decoder instance to be used for loading all relevant objects (frames, annotations etc.)
    """

    def __init__(
        self,
        name: SceneName,
        available_annotation_types: List[AnnotationType],
        decoder: UnorderedSceneDecoderProtocol[TDateTime],
    ):
        self._name = name
        self._available_annotation_types = available_annotation_types
        self._decoder = decoder
        self._number_of_camera_frames = None
        self._number_of_lidar_frames = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def map(self) -> Optional[Map]:
        return self._decoder.get_map(scene_name=self._name)

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
        elif sensor_name in self.lidar_names:
            return self.get_lidar_sensor(lidar_name=sensor_name)
        else:
            raise ValueError(f"Unknown sensor: {sensor_name}!")

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

    def get_sensor_frames(
        self, sensor_names: Optional[Iterable[SensorName]] = None, frame_ids: Optional[Iterable[FrameId]] = None
    ) -> Generator[SensorFrame[TDateTime], None, None]:
        if sensor_names is None:
            sensor_names = self.sensor_names
        else:
            sensor_names = set(sensor_names).intersection(set(self.sensor_names))
        for sensor_name in sensor_names:
            sensor = self.get_sensor(sensor_name=sensor_name)
            if frame_ids is None:
                sensor_frame_ids = sensor.frame_ids
            else:
                sensor_frame_ids = frame_ids
            for frame_id in sensor_frame_ids:
                yield sensor.get_frame(frame_id=frame_id)

    @property
    def camera_frames(self) -> Generator[CameraSensorFrame[TDateTime], None, None]:
        for camera in self.cameras:
            yield from camera.sensor_frames

    @property
    def lidar_frames(self) -> Generator[LidarSensorFrame[TDateTime], None, None]:
        for lidar in self.lidars:
            yield from lidar.sensor_frames

    @property
    def sensor_frames(self) -> Generator[SensorFrame[TDateTime], None, None]:
        yield from self.camera_frames
        yield from self.lidar_frames

    @property
    def number_of_camera_frames(self) -> int:
        if self._number_of_camera_frames is None:
            self._number_of_camera_frames = 0
            for camera in self.cameras:
                self._number_of_camera_frames += len(camera.frame_ids)
        return self._number_of_camera_frames

    @property
    def number_of_lidar_frames(self) -> int:
        if self._number_of_lidar_frames is None:
            self._number_of_lidar_frames = 0
            for lidar in self.lidars:
                self._number_of_lidar_frames += len(lidar.frame_ids)
        return self._number_of_lidar_frames

    @property
    def number_of_sensor_frames(self) -> int:
        return self.number_of_lidar_frames + self.number_of_camera_frames
