from datetime import datetime, timezone
from typing import Any, Dict, List, TypeVar, Union

import pd.state
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.frame_decoder import FrameDecoder
from paralleldomain.decoding.sensor_frame_decoder import (
    CameraSensorFrameDecoder,
    LidarSensorFrameDecoder,
    RadarSensorFrameDecoder,
)
from paralleldomain.decoding.step.sensor_frame_decoder import StepSensorFrameDecoder
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from pd.data_lab.session_reference import TemporalSessionReference
from pd.session import StepSession
from pd.state import Pose6D
from pd.state.state import PosedAgent, State

TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


class StepFrameDecoder(FrameDecoder[TDateTime]):
    def __init__(
        self,
        session: TemporalSessionReference,
        sensor_rig: List[Union[pd.state.CameraSensor, pd.state.LiDARSensor]],
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
        return [sensor.name for sensor in self._sensor_rig]

    def _decode_available_camera_names(self, frame_id: FrameId) -> List[SensorName]:
        return [sensor.name for sensor in self._sensor_rig if isinstance(sensor, pd.state.CameraSensor)]

    def _decode_available_lidar_names(self, frame_id: FrameId) -> List[SensorName]:
        return [sensor.name for sensor in self._sensor_rig if isinstance(sensor, pd.state.LiDARSensor)]

    def _decode_available_radar_names(self, frame_id: FrameId) -> List[SensorName]:
        return []

    def _decode_metadata(self, frame_id: FrameId) -> Dict[str, Any]:
        return dict()

    def _decode_datetime(self, frame_id: FrameId) -> TDateTime:
        return datetime.fromtimestamp(self.state.simulation_time_sec, tz=timezone.utc)

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
