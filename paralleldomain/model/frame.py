from datetime import datetime
from typing import Generator, Generic, List, TypeVar, Union

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

from paralleldomain.model.ego import EgoFrame
from paralleldomain.model.sensor import SensorFrame, TemporalSensorFrame
from paralleldomain.model.type_aliases import FrameId, SensorName

TSensorFrameType = TypeVar("TSensorFrameType", bound=SensorFrame)


class FrameDecoderProtocol(Protocol[TSensorFrameType]):
    def get_sensor_frame(self, frame_id: FrameId, sensor_name: SensorName) -> TSensorFrameType:
        pass

    def get_sensor_names(self, frame_id: FrameId) -> List[SensorName]:
        pass

    def get_camera_names(self, frame_id: FrameId) -> List[SensorName]:
        pass

    def get_lidar_names(self, frame_id: FrameId) -> List[SensorName]:
        pass

    def get_ego_frame(self, frame_id: FrameId) -> EgoFrame:
        pass


class Frame(Generic[TSensorFrameType]):
    def __init__(
        self,
        frame_id: FrameId,
        decoder: FrameDecoderProtocol,
    ):
        self._decoder = decoder
        self._frame_id = frame_id

    @property
    def frame_id(self) -> FrameId:
        return self._frame_id

    @property
    def ego_frame(self) -> EgoFrame:
        return self._decoder.get_ego_frame(frame_id=self.frame_id)

    def get_sensor(self, sensor_name: SensorName) -> TSensorFrameType:
        return self._decoder.get_sensor_frame(frame_id=self.frame_id, sensor_name=sensor_name)

    @property
    def sensor_names(self) -> List[SensorName]:
        return self._decoder.get_sensor_names(frame_id=self.frame_id)

    @property
    def camera_names(self) -> List[SensorName]:
        return self._decoder.get_camera_names(frame_id=self.frame_id)

    @property
    def lidar_names(self) -> List[SensorName]:
        return self._decoder.get_lidar_names(frame_id=self.frame_id)

    @property
    def sensor_frames(self) -> Generator[SensorFrame, None, None]:
        return (self.get_sensor(sensor_name=name) for name in self.sensor_names)

    @property
    def camera_frames(self) -> Generator[SensorFrame, None, None]:
        return (self.get_sensor(sensor_name=name) for name in self.camera_names)

    @property
    def lidar_frames(self) -> Generator[SensorFrame, None, None]:
        return (self.get_sensor(sensor_name=name) for name in self.lidar_names)

    def __lt__(self, other):
        return self.frame_id < other.frame_id


class TemporalFrameDecoderProtocol(FrameDecoderProtocol[TemporalSensorFrame], Protocol):
    def get_datetime(self, frame_id: FrameId) -> datetime:
        pass


class TemporalFrame(Frame[TemporalSensorFrame]):
    def __init__(
        self,
        frame_id: FrameId,
        decoder: TemporalFrameDecoderProtocol,
    ):
        super().__init__(frame_id=frame_id, decoder=decoder)
        self._decoder = decoder

    @property
    def date_time(self) -> datetime:
        return self._decoder.get_datetime(frame_id=self.frame_id)

    def __lt__(self, other: Union["TemporalFrame, Frame"]):
        if isinstance(other, TemporalFrame):
            return self.date_time < other.date_time
        return super().__lt__(other=other)
