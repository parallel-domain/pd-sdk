from datetime import datetime
from typing import Any, Dict, List, TypeVar, Union

from pd.data_lab.session_reference import TemporalSessionReference
from pd.session import StepSession
from pd.state import Pose6D
from pd.state.state import PosedAgent, State

# TODO: Add protocol for abstraction. We should not have a dependency in this direction
from paralleldomain.data_lab.config.sensor_rig import SensorRig
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_frame_decoder import (
    CameraSensorFrameDecoder,
    LidarSensorFrameDecoder,
    RadarSensorFrameDecoder,
)
from paralleldomain.decoding.step.sensor_frame_decoder import StepSensorFrameDecoder
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, RadarSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName

TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


class StepFrameDecoder(FrameDecoder[TDateTime]):
    def __init__(
        self,
        session: TemporalSessionReference,
        sensor_rig: SensorRig,
        dataset_name: str,
        scene_name: SceneName,
        settings: DecoderSettings,
        ego_agent_id: int,
        date_time: datetime,
    ):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self._session = session
        self._sensor_rig = sensor_rig
        self._ego_agent_id = ego_agent_id
        self._date_time = date_time

    @property
    def session(self) -> StepSession:
        if self._session.session is None:
            raise ValueError("This frame is not available anymore!")
        return self._session.session

    @property
    def state(self) -> State:
        if self._session.state is None:
            raise ValueError("This frame is not available anymore!")
        return self._session.state

    def _decode_ego_pose(self, frame_id: FrameId) -> EgoPose:
        agent = next(
            iter([a for a in self.state.agents if a.id == self._ego_agent_id and isinstance(a, PosedAgent)]),
            None,
        )
        if agent is None:
            raise ValueError("No Ego Agent was set!")

        if isinstance(agent.pose, Pose6D):
            mat = agent.pose.as_transformation_matrix()
        else:
            mat = agent.pose

        return EgoPose.from_transformation_matrix(mat=mat, approximate_orthogonal=True)

    def _decode_available_sensor_names(self, frame_id: FrameId) -> List[SensorName]:
        return [sensor.name for sensor in self._sensor_rig.sensors]

    def _decode_available_camera_names(self, frame_id: FrameId) -> List[SensorName]:
        return [sensor.name for sensor in self._sensor_rig.sensors if sensor.is_camera]

    def _decode_available_lidar_names(self, frame_id: FrameId) -> List[SensorName]:
        return [sensor.name for sensor in self._sensor_rig.sensors if sensor.is_lidar]

    def _decode_available_radar_names(self, frame_id: FrameId) -> List[SensorName]:
        return []

    def _decode_metadata(self, frame_id: FrameId) -> Dict[str, Any]:
        return dict()

    def _decode_datetime(self, frame_id: FrameId) -> TDateTime:
        return self._date_time

    def _create_camera_sensor_frame_decoder(self) -> CameraSensorFrameDecoder[TDateTime]:
        return StepSensorFrameDecoder(
            settings=self.settings,
            sensor_rig=self._sensor_rig,
            scene_name=self.scene_name,
            dataset_name=self.dataset_name,
            ego_agent_id=self._ego_agent_id,
            session=self._session,
            is_camera=True,
        )

    def _decode_camera_sensor_frame(
        self, decoder: CameraSensorFrameDecoder[TDateTime], frame_id: FrameId, sensor_name: SensorName
    ) -> CameraSensorFrame[TDateTime]:
        return CameraSensorFrame[TDateTime](sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)

    def _create_lidar_sensor_frame_decoder(self) -> LidarSensorFrameDecoder[TDateTime]:
        return StepSensorFrameDecoder(
            settings=self.settings,
            sensor_rig=self._sensor_rig,
            scene_name=self.scene_name,
            dataset_name=self.dataset_name,
            ego_agent_id=self._ego_agent_id,
            session=self._session,
            is_camera=False,
        )

    def _create_radar_sensor_frame_decoder(self) -> RadarSensorFrameDecoder[TDateTime]:
        pass

    def _decode_lidar_sensor_frame(
        self, decoder: LidarSensorFrameDecoder[TDateTime], frame_id: FrameId, sensor_name: SensorName
    ) -> LidarSensorFrame[TDateTime]:
        return LidarSensorFrame[TDateTime](sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)

    def _decode_radar_sensor_frame(
        self, decoder: RadarSensorFrameDecoder[TDateTime], frame_id: FrameId, sensor_name: SensorName
    ) -> RadarSensorFrame[TDateTime]:
        pass
