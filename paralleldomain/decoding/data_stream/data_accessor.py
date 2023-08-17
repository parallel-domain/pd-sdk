import json
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, DefaultDict, Optional, Tuple, Any, Set

from google.protobuf import json_format
from pyquaternion import Quaternion

from paralleldomain.model.annotation import AnnotationIdentifier
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.image import Image
from paralleldomain.model.sensor import SensorDataCopyTypes
from paralleldomain.model.type_aliases import FrameId, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.coordinate_system import CoordinateSystem
from paralleldomain.utilities.fsio import read_json
from pd.data_lab import LabeledStateReference
from pd.internal.proto.label_engine.generated.python.data_pb2 import DataTypeRecord
from pd.internal.proto.label_engine.generated.python.options_pb2 import DataType
from pd.internal.proto.label_engine.generated.python.telemetry_pb2 import TelemetryValue, Telemetry
from pd.label_engine import LabelData
from pd.state import Pose6D
from pd.state.sensor import Sensor, CameraSensor, LiDARSensor, sensors_from_json
from pd.state.state import PosedAgent

StreamType = int

FRAME_IDS_PER_SECOND = 100
PathsByStreamType = DefaultDict[StreamType, List[AnyPath]]


class DataStreamDataAccessor(ABC):
    def __init__(
        self,
        camera_image_stream_name: str,
    ):
        self.camera_image_stream_name = camera_image_stream_name

    @property
    def sensor_names(self) -> List[str]:
        return self.camera_names + self.lidar_names + self.radar_names

    @abstractmethod
    def get_ego_pose(self, frame_id: FrameId) -> EgoPose:
        ...

    @property
    @abstractmethod
    def sensor_and_agent_ids(self) -> List[Tuple[Sensor, int]]:
        ...

    def get_sensor(self, sensor_name: str) -> Sensor:
        try:
            name, agent_id = sensor_name.rsplit(sep="-", maxsplit=1)
            agent_id = int(agent_id)
        except ValueError:
            raise ValueError(f"Invalid sensor name '{sensor_name}: expecting a name and id separated by hyphen'")
        agent_id = int(agent_id)
        sensors = self.sensor_and_agent_ids
        matching_sensors = [s for s, agent in sensors if s.name == name and agent == agent_id]

        if len(matching_sensors) != 1:
            raise ValueError(f"Couldn't find sensor '{sensor_name}' in the sensor rigs.")

        return matching_sensors[0]

    @property
    def camera_names(self) -> List[SensorName]:
        return [f"{s.name}-{id}" for s, id in self.sensor_and_agent_ids if isinstance(s, CameraSensor)]

    @property
    def lidar_names(self) -> List[SensorName]:
        return [f"{s.name}-{id}" for s, id in self.sensor_and_agent_ids if isinstance(s, LiDARSensor)]

    @property
    def radar_names(self) -> List[SensorName]:
        return []

    @property
    @abstractmethod
    def available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        ...

    @abstractmethod
    def get_label_data(
        self, frame_id: str, stream_name: str, sensor_name: Optional[str], file_ending: str
    ) -> LabelData:
        ...

    @abstractmethod
    def get_scene_metadata(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def get_frame_ids(self) -> Set[FrameId]:
        ...

    @abstractmethod
    def get_file_path(
        self, sensor_name: SensorName, frame_id: FrameId, data_type: SensorDataCopyTypes
    ) -> Optional[AnyPath]:
        ...

    @abstractmethod
    def get_frame_id_to_date_time_map(self) -> Dict[str, datetime]:
        ...


class StoredDataStreamDataAccessor(DataStreamDataAccessor):
    def __init__(
        self,
        scene_path: AnyPath,
        camera_image_stream_name: str,
        potentially_available_annotation_identifiers: List[AnnotationIdentifier],
    ):
        super().__init__(camera_image_stream_name=camera_image_stream_name)
        self.scene_path = scene_path
        self._potentially_available_annotation_identifiers = potentially_available_annotation_identifiers
        self._sensor_agnostic_streams: Optional[PathsByStreamType] = None
        self._sensor_rigs: Optional[PathsByStreamType] = None
        self._camera_names: Optional[List[str]] = None
        self._lidar_names: Optional[List[str]] = None
        self._radar_names: Optional[List[str]] = None
        self._streams_with_per_frame_data: Optional[List[AnyPath]] = None
        self._ego_transformation_poses: Optional[List[TelemetryValue]] = None
        self._available_annotation_identifiers: Optional[List[AnnotationIdentifier]] = None
        self._sensor_and_agent_ids: Optional[List[Tuple[Sensor, int]]] = None
        self._frame_ids: Optional[Set[FrameId]] = None

    def get_ego_pose(self, frame_id: FrameId) -> EgoPose:
        telemetry_value = self.ego_transformation_poses[int(frame_id)]

        p = telemetry_value.transformation_pose.position
        q = telemetry_value.transformation_pose.orientation
        return EgoPose(quaternion=Quaternion(x=q.x, y=q.y, z=q.z, w=q.w), translation=[p.x, p.y, p.z])

    @property
    def sensor_agnostic_streams(self) -> PathsByStreamType:
        if self._sensor_agnostic_streams is None:
            result = defaultdict(list)
            for f in self.scene_path.iterdir():
                type_file = f / ".type"
                if not f.is_dir() or not type_file.exists():
                    continue

                with type_file.open("r") as fp:
                    data_type_record = json_format.Parse(text=fp.read(), message=DataTypeRecord())

                if data_type_record.type not in [DataType.eNone, DataType.eNull]:
                    result[data_type_record.type].append(f)
                self._sensor_agnostic_streams = result
        return self._sensor_agnostic_streams

    @property
    def sensor_and_agent_ids(self) -> List[Tuple[Sensor, int]]:
        if self._sensor_and_agent_ids is None:
            result = list()
            sensor_rig_stream = self.sensor_agnostic_streams[DataType.eSensor]

            for rig_folder in sensor_rig_stream:
                rig_files = [f for f in rig_folder.iterdir() if f.is_file() and f.name != ".type"]
                for rig_file in rig_files:
                    agent_id = rig_file.stem.replace(".pb", "")

                    sensor_json_dict = read_json(path=rig_file)
                    sensors = sensors_from_json(json_dict=sensor_json_dict)
                    for sensor in sensors:
                        result.append((sensor, int(agent_id)))
            self._sensor_and_agent_ids = result
        return self._sensor_and_agent_ids

    @property
    def streams_with_per_frame_data(self) -> List[AnyPath]:
        if self._streams_with_per_frame_data is None:
            result = [stream for stream in self.sensor_agnostic_streams[DataType.eTransformMap]]
            for f in self.scene_path.iterdir():
                if not f.is_dir():
                    continue

                for sensor in self.sensor_names:
                    sensor_folder = f / sensor
                    type_file = sensor_folder / ".type"
                    if not sensor_folder.is_dir() or not type_file.exists():
                        continue

                    with type_file.open("r") as fp:
                        data_type_record = json_format.Parse(text=fp.read(), message=DataTypeRecord())

                    if data_type_record.type in [DataType.eImage, DataType.ePointCloud]:
                        result.append(sensor_folder)
            self._streams_with_per_frame_data = result
        return self._streams_with_per_frame_data

    @property
    def ego_transformation_poses(self) -> List[TelemetryValue]:
        if self._ego_transformation_poses is None:
            streams = self.sensor_agnostic_streams[DataType.eTelemetry]
            if len(streams) != 1:
                raise ValueError()
            files = [f for f in streams[0].iterdir() if f.is_file() and f.name != ".type"]
            if len(files) != 1:
                raise ValueError()
            with files[0].open("r") as fp:
                telemetry_dict = json.load(fp)
            telemetry_data = json_format.ParseDict(js_dict=telemetry_dict, message=Telemetry())
            self._ego_transformation_poses = telemetry_data.localization
        return self._ego_transformation_poses

    @property
    def available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        if self._available_annotation_identifiers is None:
            result = []
            for identifier in self._potentially_available_annotation_identifiers:
                stream_folder = self.scene_path / identifier.name
                type_file = stream_folder / ".type"
                if stream_folder.is_dir() and type_file.is_file():
                    result.append(identifier)
            self._available_annotation_identifiers = result
        return self._available_annotation_identifiers

    def get_label_data(
        self, frame_id: str, stream_name: str, sensor_name: Optional[str], file_ending: str
    ) -> LabelData:
        if sensor_name is not None:
            relative_file_path = f"{stream_name}/{sensor_name}/{frame_id}.{file_ending}"
        else:
            relative_file_path = f"{stream_name}/{frame_id}.{file_ending}"
        annotation_path = self.scene_path / f"{relative_file_path}"
        with annotation_path.open("rb") as f:
            return LabelData(
                timestamp=frame_id,
                label=stream_name,
                sensor_name=sensor_name if sensor_name is not None else "",
                data=f.read(),
            )

    def get_scene_metadata(self) -> Dict[str, Any]:
        ig_metadata_stream = self.sensor_agnostic_streams[DataType.eIGMetadata][0]
        metadata_files = [f for f in ig_metadata_stream.iterdir() if f.is_file() and f.name != ".type"]
        if len(metadata_files) != 1:
            raise ValueError(f"Got {len(metadata_files)} meta data files.")

        with metadata_files[0].open("r") as fp:
            return json.load(fp)

    def get_frame_ids(self) -> Set[FrameId]:
        if self._frame_ids is None:
            frame_ids = set()

            for stream in self.streams_with_per_frame_data:
                stream_frame_ids = [
                    f.stem.replace(".pb", "") for f in stream.iterdir() if f.is_file() and f.name != ".type"
                ]
                frame_ids.update(stream_frame_ids)
            self._frame_ids = frame_ids
        return self._frame_ids

    def get_file_path(
        self, sensor_name: SensorName, frame_id: FrameId, data_type: SensorDataCopyTypes
    ) -> Optional[AnyPath]:
        if issubclass(data_type, Image):
            sensor_folder = self.scene_path / self.camera_image_stream_name / sensor_name
            # TODO: this doesn't scale at all. Could also iter possible file endings instead, could store file ending
            sensor_files = [f for f in sensor_folder.iterdir() if f.is_file() and f.stem == frame_id]
            if len(sensor_files) > 1:
                raise ValueError(f"Found {len(sensor_files)} files for frame id {frame_id}.")
            elif len(sensor_files) == 1:
                return sensor_files[0]

        return None

    def get_frame_id_to_date_time_map(self) -> Dict[str, datetime]:
        return {fid: datetime.fromtimestamp(int(fid) / FRAME_IDS_PER_SECOND) for fid in self.get_frame_ids()}


class LabelEngineDataStreamDataAccessor(DataStreamDataAccessor):
    def __init__(
        self,
        labeled_state_reference: LabeledStateReference,
        camera_image_stream_name: str,
        available_annotation_identifiers: List[AnnotationIdentifier],
    ):
        super().__init__(camera_image_stream_name=camera_image_stream_name)
        self._label_engine_instance = labeled_state_reference.label_engine
        self._frame_ids = set()
        self._frame_id_to_date_time = dict()
        self._frame_id_to_frame_timestamp = dict()
        self._frame_id_to_ego_agent = dict()
        self.update_labeled_state_reference(labeled_state_reference=labeled_state_reference)
        self._available_annotation_identifiers = available_annotation_identifiers

    def get_ego_pose(self, frame_id: FrameId) -> EgoPose:
        ego_agent = self._frame_id_to_ego_agent[frame_id]
        if not isinstance(ego_agent, PosedAgent):
            raise ValueError("Expected ego agent to have a pose!")
        pose = ego_agent.pose
        if isinstance(pose, Pose6D):
            ego_to_world = EgoPose(quaternion=pose.rotation, translation=list(pose.translation))
        else:
            ego_to_world = EgoPose.from_transformation_matrix(mat=pose, approximate_orthogonal=True)

        RFU_to_FLU = CoordinateSystem("RFU") > CoordinateSystem("FLU")

        return RFU_to_FLU @ ego_to_world @ RFU_to_FLU.inverse

    @property
    def sensor_and_agent_ids(self) -> List[Tuple[Sensor, int]]:
        ego_agent = next(iter(self._frame_id_to_ego_agent.values()))
        return [(s, ego_agent.id) for s in ego_agent.sensors]

    @property
    def available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        return self._available_annotation_identifiers

    def get_label_data(
        self, frame_id: str, stream_name: str, sensor_name: Optional[str], file_ending: str
    ) -> LabelData:
        frame_timestamp = self._frame_id_to_frame_timestamp[frame_id]
        try:
            name, agent_id = sensor_name.rsplit(sep="-", maxsplit=1)
            agent_id = int(agent_id)
        except ValueError:
            raise ValueError(f"Invalid sensor name '{sensor_name}: expecting a name and id separated by hyphen'")

        label_data = self._label_engine_instance.get_annotation_data(
            stream_name=stream_name, frame_timestamp=frame_timestamp, sensor_id=agent_id, sensor_name=name
        )
        return label_data

    def get_scene_metadata(self) -> Dict[str, Any]:
        return dict()

    def get_frame_ids(self) -> Set[FrameId]:
        return self._frame_ids

    def update_labeled_state_reference(self, labeled_state_reference: LabeledStateReference) -> None:
        if not labeled_state_reference.label_engine == self._label_engine_instance:
            raise ValueError("Can't change label engine instance")
        frame_id = labeled_state_reference.frame_id
        self._frame_ids.add(frame_id)
        self._frame_id_to_date_time[frame_id] = labeled_state_reference.date_time
        self._frame_id_to_frame_timestamp[frame_id] = labeled_state_reference.frame_timestamp
        self._frame_id_to_ego_agent[frame_id] = labeled_state_reference.ego_agent

    def get_file_path(
        self, sensor_name: SensorName, frame_id: FrameId, data_type: SensorDataCopyTypes
    ) -> Optional[AnyPath]:
        return None

    def get_frame_id_to_date_time_map(self) -> Dict[str, datetime]:
        return self._frame_id_to_date_time
