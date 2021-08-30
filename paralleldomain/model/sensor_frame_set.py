import contextlib
from datetime import datetime
from typing import Any, Callable, ContextManager, Dict, Generator, Generic, List, Set, Type, TypeVar, cast

from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.class_mapping import ClassMap

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

from paralleldomain.model.frame import Frame
from paralleldomain.model.sensor import CameraSensor, LidarSensor, Sensor, SensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorFrameSetName, SensorName

T = TypeVar("T")
TSensorFrameType = TypeVar("TSensorFrameType", bound=SensorFrame)


class SensorFrameSetDecoderProtocol(Protocol[TSensorFrameType]):
    def get_set_description(self, set_name: SensorFrameSetName) -> str:
        pass

    def get_set_metadata(self, set_name: SensorFrameSetName) -> Dict[str, Any]:
        pass

    def get_frame(
        self,
        set_name: SensorFrameSetName,
        frame_id: FrameId,
    ) -> Frame[TSensorFrameType]:
        pass

    def get_sensor_names(self, set_name: SensorFrameSetName) -> List[str]:
        pass

    def get_camera_names(self, set_name: SensorFrameSetName) -> List[str]:
        pass

    def get_lidar_names(self, set_name: SensorFrameSetName) -> List[str]:
        pass

    def get_frame_ids(self, set_name: SensorFrameSetName) -> Set[FrameId]:
        pass

    def get_class_map(self, set_name: SensorFrameSetName, annotation_type: Type[T]) -> ClassMap:
        pass

    def get_sensor(self, set_name: SensorFrameSetName, sensor_name: SensorName) -> Sensor[TSensorFrameType]:
        pass


TFrameType = TypeVar("TFrameType", bound=Frame)


class SensorFrameSet(Generic[TFrameType, TSensorFrameType]):
    def __init__(
        self,
        name: SensorFrameSetName,
        available_annotation_types: List[AnnotationType],
        decoder: SensorFrameSetDecoderProtocol[TSensorFrameType],
    ):
        self._name = name
        self._available_annotation_types = available_annotation_types
        self._decoder = decoder

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._decoder.get_set_description(set_name=self._name)

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._decoder.get_set_metadata(set_name=self._name)

    @property
    def frames(self) -> Generator[Frame, None, None]:
        return (self.get_frame(frame_id=frame_id) for frame_id in self.ordered_frame_ids)

    @property
    def frame_ids(self) -> Set[FrameId]:
        return self._decoder.get_frame_ids(set_name=self.name)

    @property
    def ordered_frame_ids(self) -> List[FrameId]:
        return sorted(self.frame_ids)

    @property
    def available_annotation_types(self):
        return self._available_annotation_types

    @property
    def sensor_names(self) -> List[str]:
        return self._decoder.get_sensor_names(set_name=self.name)

    @property
    def camera_names(self) -> List[str]:
        return self._decoder.get_camera_names(set_name=self.name)

    @property
    def lidar_names(self) -> List[str]:
        return self._decoder.get_lidar_names(set_name=self.name)

    def get_frame(self, frame_id: FrameId) -> TFrameType:
        return self._decoder.get_frame(set_name=self.name, frame_id=frame_id)

    @property
    def sensors(self) -> Generator[Sensor, None, None]:
        return (self.get_sensor(sensor_name=sensor_name) for sensor_name in self.sensor_names)

    @property
    def cameras(self) -> Generator[CameraSensor, None, None]:
        return (cast(CameraSensor, self.get_sensor(sensor_name=sensor_name)) for sensor_name in self.camera_names)

    @property
    def lidars(self) -> Generator[LidarSensor, None, None]:
        return (cast(LidarSensor, self.get_sensor(sensor_name=sensor_name)) for sensor_name in self.lidar_names)

    def get_sensor(self, sensor_name: SensorName) -> Sensor[TSensorFrameType]:
        return self._decoder.get_sensor(set_name=self.name, sensor_name=sensor_name)

    @property
    def class_maps(self) -> Dict[AnnotationType, ClassMap]:
        return {a_type: self.get_class_map(a_type) for a_type in self._available_annotation_types}

    def get_class_map(self, annotation_type: Type[T]) -> ClassMap:
        if annotation_type not in self._available_annotation_types:
            raise ValueError(f"No annotation type {annotation_type} available in this dataset!")

        return self._decoder.get_class_map(set_name=self.name, annotation_type=annotation_type)

    @classmethod
    def from_decoder(
        cls,
        set_name: SensorFrameSetName,
        available_annotation_types: List[AnnotationType],
        decoder: SensorFrameSetDecoderProtocol,
    ) -> "SensorFrameSet":
        return cls(
            name=set_name,
            available_annotation_types=available_annotation_types,
            decoder=decoder,
        )
