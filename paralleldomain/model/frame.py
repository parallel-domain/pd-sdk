from datetime import datetime
from typing import Generator, Generic, List, TypeVar, Union

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

from paralleldomain.model.ego import EgoFrame
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, SensorFrame
from paralleldomain.model.type_aliases import FrameId, SensorName

TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


class FrameDecoderProtocol(Protocol[TDateTime]):
    def get_camera_sensor_frame(self, frame_id: FrameId, sensor_name: SensorName) -> CameraSensorFrame[TDateTime]:
        pass

    def get_lidar_sensor_frame(self, frame_id: FrameId, sensor_name: SensorName) -> LidarSensorFrame[TDateTime]:
        pass

    def get_sensor_names(self, frame_id: FrameId) -> List[SensorName]:
        pass

    def get_camera_names(self, frame_id: FrameId) -> List[SensorName]:
        pass

    def get_lidar_names(self, frame_id: FrameId) -> List[SensorName]:
        pass

    def get_ego_frame(self, frame_id: FrameId) -> EgoFrame:
        pass

    def get_date_time(self, frame_id: FrameId) -> TDateTime:
        pass


class Frame(Generic[TDateTime]):
    def __init__(
        self,
        frame_id: FrameId,
        decoder: FrameDecoderProtocol[TDateTime],
    ):
        self._decoder = decoder
        self._frame_id = frame_id

    @property
    def frame_id(self) -> FrameId:
        return self._frame_id

    @property
    def date_time(self) -> TDateTime:
        return self._decoder.get_date_time(frame_id=self.frame_id)

    @property
    def ego_frame(self) -> EgoFrame:
        return self._decoder.get_ego_frame(frame_id=self.frame_id)

    def get_camera(self, camera_name: SensorName) -> CameraSensorFrame[TDateTime]:
        return self._decoder.get_camera_sensor_frame(frame_id=self.frame_id, sensor_name=camera_name)

    def get_lidar(self, lidar_name: SensorName) -> LidarSensorFrame[TDateTime]:
        return self._decoder.get_lidar_sensor_frame(frame_id=self.frame_id, sensor_name=lidar_name)

    def get_sensor(self, sensor_name: SensorName) -> SensorFrame[TDateTime]:
        if sensor_name in self.camera_names:
            return self.get_camera(camera_name=sensor_name)
        else:
            return self.get_lidar(lidar_name=sensor_name)

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
    def sensor_frames(self) -> Generator[SensorFrame[TDateTime], None, None]:
        return (self.get_sensor(sensor_name=name) for name in self.sensor_names)

    @property
    def camera_frames(self) -> Generator[CameraSensorFrame[TDateTime], None, None]:
        return (
            self._decoder.get_camera_sensor_frame(frame_id=self.frame_id, sensor_name=name)
            for name in self.camera_names
        )

    @property
    def lidar_frames(self) -> Generator[LidarSensorFrame[TDateTime], None, None]:
        return (
            self._decoder.get_lidar_sensor_frame(frame_id=self.frame_id, sensor_name=name) for name in self.lidar_names
        )

    def __lt__(self, other: "Frame[TDateTime]"):
        if self.date_time is not None and other.date_time is not None:
            return self.date_time < other.date_time
        return id(self) < id(other)


# class TemporalFrameDecoderProtocol(FrameDecoderProtocol[TemporalSensorFrame], Protocol):
#     def get_datetime(self, frame_id: FrameId) -> datetime:
#         pass
#
#
# class TemporalFrame(Frame[TemporalSensorFrame]):
#     def __init__(
#         self,
#         frame_id: FrameId,
#         decoder: TemporalFrameDecoderProtocol,
#     ):
#         super().__init__(frame_id=frame_id, decoder=decoder)
#         self._decoder = decoder
#
#     @property
#     def date_time(self) -> datetime:
#         return self._decoder.get_datetime(frame_id=self.frame_id)
#
#     def __lt__(self, other: Union["TemporalFrame, Frame"]):
#         if isinstance(other, TemporalFrame):
#             return self.date_time < other.date_time
#         return super().__lt__(other=other)
