from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, List, TypeVar, Union

from paralleldomain.decoding.in_memory.sensor_frame_decoder import (
    InMemoryCameraFrameDecoder,
    InMemoryLidarFrameDecoder,
    InMemoryRadarFrameDecoder,
)
from paralleldomain.model.ego import EgoFrame, EgoPose
from paralleldomain.model.frame import Frame
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, RadarSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName

TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


@dataclass
class InMemoryFrameDecoder(Generic[TDateTime]):
    ego_pose: EgoPose
    date_time: TDateTime
    dataset_name: str
    scene_name: SceneName
    frame_id: FrameId
    metadata: Dict[str, Any] = field(default_factory=dict)
    camera_sensor_frames: List[CameraSensorFrame[TDateTime]] = field(default_factory=list)
    lidar_sensor_frames: List[LidarSensorFrame[TDateTime]] = field(default_factory=list)
    radar_sensor_frames: List[RadarSensorFrame[TDateTime]] = field(default_factory=list)
    camera_names: List[SensorName] = field(default_factory=list)
    lidar_names: List[SensorName] = field(default_factory=list)
    radar_names: List[SensorName] = field(default_factory=list)

    def get_camera_sensor_frame(self, sensor_name: SensorName) -> CameraSensorFrame[TDateTime]:
        return next(
            iter(
                [
                    sf
                    for sf in self.camera_sensor_frames
                    if sf.frame_id == self.frame_id and sf.sensor_name == sensor_name
                ]
            )
        )

    def get_lidar_sensor_frame(self, sensor_name: SensorName) -> LidarSensorFrame[TDateTime]:
        return next(
            iter(
                [
                    sf
                    for sf in self.lidar_sensor_frames
                    if sf.frame_id == self.frame_id and sf.sensor_name == sensor_name
                ]
            )
        )

    def get_radar_sensor_frame(self, sensor_name: SensorName) -> RadarSensorFrame[TDateTime]:
        return next(
            iter(
                [
                    sf
                    for sf in self.radar_sensor_frames
                    if sf.frame_id == self.frame_id and sf.sensor_name == sensor_name
                ]
            )
        )

    def get_sensor_names(self) -> List[SensorName]:
        return self.camera_names + self.lidar_names + self.radar_names

    def get_camera_names(self) -> List[SensorName]:
        return self.camera_names

    def get_lidar_names(self) -> List[SensorName]:
        return self.lidar_names

    def get_radar_names(self) -> List[SensorName]:
        return self.radar_names

    def get_ego_frame(self) -> EgoFrame:
        return EgoFrame(pose_loader=lambda: self.ego_pose)

    def get_date_time(self) -> TDateTime:
        return self.date_time

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata

    @staticmethod
    def from_frame(frame: Frame[TDateTime]) -> "InMemoryFrameDecoder":
        camera_sensor_frames = list()
        camera_names = list()
        for cam_sen in frame.camera_frames:
            camera_names.append(cam_sen.sensor_name)
            camera_sensor_frames.append(
                CameraSensorFrame(decoder=InMemoryCameraFrameDecoder.from_camera_frame(camera_frame=cam_sen))
            )
        lidar_sensor_frames = list()
        lidar_names = list()
        for sen_frame in frame.lidar_frames:
            lidar_names.append(sen_frame.sensor_name)
            lidar_sensor_frames.append(
                LidarSensorFrame(decoder=InMemoryLidarFrameDecoder.from_lidar_frame(lidar_frame=sen_frame))
            )
        radar_sensor_frames = list()
        radar_names = list()
        for sen_frame in frame.radar_frames:
            radar_names.append(sen_frame.sensor_name)
            radar_sensor_frames.append(
                RadarSensorFrame(decoder=InMemoryRadarFrameDecoder.from_radar_frame(radar_frame=sen_frame))
            )

        return InMemoryFrameDecoder(
            ego_pose=frame.ego_frame.pose,
            camera_sensor_frames=camera_sensor_frames,
            lidar_sensor_frames=lidar_sensor_frames,
            radar_sensor_frames=radar_sensor_frames,
            camera_names=camera_names,
            lidar_names=lidar_names,
            radar_names=radar_names,
            date_time=frame.date_time,
            metadata=frame.metadata,
            scene_name=frame.scene_name,
            dataset_name=frame.dataset_name,
            frame_id=frame.frame_id,
        )
