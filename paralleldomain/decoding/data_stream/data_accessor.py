import json
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, DefaultDict, Dict, List, Optional, Set

import pypeln
from google.protobuf import json_format
from pd.data_lab import LabeledStateReference
from pd.internal.proto.label_engine.generated.python import transform_map_pb2
from pd.internal.proto.label_engine.generated.python.data_pb2 import DataTypeRecord
from pd.internal.proto.label_engine.generated.python.mesh_map_pb2 import MeshMap
from pd.internal.proto.label_engine.generated.python.options_pb2 import DataType
from pd.label_engine import LabelData, load_pipeline_config
from pd.state import Pose6D
from pd.state.sensor import CameraSensor, Sensor, sensors_from_json
from pd.state.state import PosedAgent, SensorAgent
from pyquaternion import Quaternion

from paralleldomain.decoding.data_stream.common import TYPE_TO_FILE_FORMAT
from paralleldomain.model.annotation import AnnotationIdentifier, AnnotationTypes
from paralleldomain.model.ego import EgoPose
from paralleldomain.model.image import Image
from paralleldomain.model.sensor import LidarSensor, RadarSensor, SensorDataCopyTypes
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.coordinate_system import CoordinateSystem
from paralleldomain.utilities.fsio import read_json, read_message

StreamType = int

FRAME_IDS_PER_SECOND = 100
PathsByStreamType = DefaultDict[StreamType, List[AnyPath]]


class DataStreamDataAccessor(ABC):
    def __init__(self, camera_image_stream_name: str, scene_name: SceneName):
        self.camera_image_stream_name = camera_image_stream_name
        self.scene_name = scene_name

    @property
    @abstractmethod
    def sensors(self) -> Dict[FrameId, Dict[SensorName, Sensor]]:
        pass

    @property
    def cameras(self) -> Dict[FrameId, Dict[SensorName, Sensor]]:
        return {
            fid: {n: s for n, s in sensors.items() if isinstance(s, CameraSensor)}
            for fid, sensors in self.sensors.items()
        }

    @property
    def lidars(self) -> Dict[FrameId, Dict[SensorName, Sensor]]:
        return {
            fid: {n: s for n, s in sensors.items() if isinstance(s, LidarSensor)}
            for fid, sensors in self.sensors.items()
        }

    @property
    def radars(self) -> Dict[FrameId, Dict[SensorName, Sensor]]:
        return {
            fid: {n: s for n, s in sensors.items() if isinstance(s, RadarSensor)}
            for fid, sensors in self.sensors.items()
        }

    @abstractmethod
    def get_ego_pose(self, frame_id: FrameId) -> EgoPose:
        ...

    def get_sensor(self, sensor_name: SensorName, frame_id: FrameId) -> Sensor:
        return self.sensors[frame_id][sensor_name]

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

    @abstractmethod
    def get_ontology_data(self, frame_id: FrameId, annotation_identifier: AnnotationIdentifier) -> Optional[LabelData]:
        pass


class StoredDataStreamDataAccessor(DataStreamDataAccessor):
    def __init__(
        self,
        scene_path: AnyPath,
        scene_name: SceneName,
        camera_image_stream_name: str,
        potentially_available_annotation_identifiers: List[AnnotationIdentifier],
    ):
        super().__init__(
            camera_image_stream_name=camera_image_stream_name,
            scene_name=scene_name,
        )
        self.scene_path = scene_path
        self._potentially_available_annotation_identifiers = potentially_available_annotation_identifiers
        self._sensor_agnostic_streams: Optional[PathsByStreamType] = None
        self._streams_with_per_frame_data: Optional[List[AnyPath]] = None
        self._frame_id_to_transform_file: Optional[Dict[FrameId, AnyPath]] = None
        self._frame_id_to_ego_telemetry_file: Optional[Dict[FrameId, AnyPath]] = None
        self._ego_agent_ids: Optional[Dict[FrameId, str]] = None
        self._ego_actor_ids: Dict[FrameId, int] = dict()
        self._available_annotation_identifiers: Optional[List[AnnotationIdentifier]] = None
        self._sensors: Optional[Dict[FrameId, Dict[SensorName, Sensor]]] = None

    def get_ontology_data(self, frame_id: FrameId, annotation_identifier: AnnotationIdentifier) -> Optional[LabelData]:
        sensor_agnostic_annotation_stream = [
            s for s in self.sensor_agnostic_streams[DataType.eAnnotation] if s.name == annotation_identifier.name
        ]
        # if identifier is sensor agnostic don't append camera name to folder path
        if len(sensor_agnostic_annotation_stream) > 0:
            folder = self.scene_path / annotation_identifier.name
        else:
            camera_name = next(iter(self.cameras[frame_id]))
            # assumes ontology is the same across cameras
            folder = self.scene_path / annotation_identifier.name / camera_name

        type_file = folder / ".type"
        if not folder.is_dir() or not type_file.exists():
            raise ValueError(f"Can not find type file for {annotation_identifier.name}")
        with type_file.open("r") as fp:
            data_type_record = json_format.Parse(text=fp.read(), message=DataTypeRecord())

        ontology_stream_name = data_type_record.ontology
        if ontology_stream_name is not None and ontology_stream_name != "":
            label_data = self.get_label_data(
                stream_name=ontology_stream_name,
                sensor_name=None,
                frame_id=frame_id,
                file_ending="pb.json",
            )
            return label_data
        return None

    def get_ego_pose(self, frame_id: FrameId) -> EgoPose:
        if frame_id in self.frame_id_to_ego_telemetry_file:
            frame_telemetry = self.frame_id_to_ego_telemetry_file[frame_id]
            pose = read_message(obj=transform_map_pb2.TransformMap(), path=frame_telemetry).actor_transform_map[0]
            p = pose.translation
            q = pose.orientation
            ego_to_world = EgoPose(quaternion=Quaternion(x=q.x, y=q.y, z=q.z, w=q.w), translation=[p.x, p.y, p.z])
        else:
            pose_file = self.frame_id_to_transform_file[frame_id]
            with pose_file.open("r") as fp:
                transform_map = json_format.Parse(text=fp.read(), message=transform_map_pb2.TransformMap())

            ego_actor_id = self.get_ego_actor_id(frame_id=frame_id)
            pose = transform_map.actor_transform_map[ego_actor_id]
            p = pose.translation
            q = pose.orientation

            ego_to_world = EgoPose(quaternion=Quaternion(x=q.x, y=q.y, z=q.z, w=q.w), translation=[p.x, p.y, p.z])

        RFU_to_FLU = CoordinateSystem("RFU") > CoordinateSystem("FLU")
        return RFU_to_FLU @ ego_to_world @ RFU_to_FLU.inverse

    @property
    def frame_id_to_ego_telemetry_file(self) -> Optional[Dict[FrameId, AnyPath]]:
        if self._frame_id_to_ego_telemetry_file is None:
            self._frame_id_to_ego_telemetry_file = dict()
            streams = self.sensor_agnostic_streams[DataType.eTransformMap]
            stream = next(iter([s for s in streams if s.is_dir() and s.name == "ego_telemetry"]), None)
            if stream is not None:
                self._frame_id_to_ego_telemetry_file = {
                    file.stem.replace(".pb", ""): file
                    for file in stream.iterdir()
                    if file.is_file() and ".pb" in file.stem
                }
        return self._frame_id_to_ego_telemetry_file

    @property
    def sensor_agnostic_streams(self) -> PathsByStreamType:
        if self._sensor_agnostic_streams is None:

            def _get_agnostic_streams(path: AnyPath):
                if path.name == "mesh":
                    return
                type_file_type = get_type_file_type(folder=path)
                if type_file_type is None:
                    return

                if type_file_type not in [DataType.eNone, DataType.eNull]:
                    yield type_file_type, path
                elif type_file_type is DataType.eNone:
                    for sub_f in path.iterdir():
                        type_file_type = get_type_file_type(folder=sub_f)
                        if type_file_type is None:
                            continue
                        if type_file_type is DataType.eSensor:
                            yield type_file_type, sub_f

            result = defaultdict(list)
            for ftype, path in pypeln.thread.flat_map(_get_agnostic_streams, self.scene_path.iterdir(), workers=16):
                result[ftype].append(path)
            self._sensor_agnostic_streams = result
        return self._sensor_agnostic_streams

    @property
    def sensors(self) -> Dict[FrameId, Dict[SensorName, Sensor]]:
        if self._sensors is None:
            result = defaultdict(dict)
            # we iterate over all sensor stream folders in case ego agent ids change between frames
            for sensor_stream_folder in self.sensor_agnostic_streams[DataType.eSensor]:
                files = list(sensor_stream_folder.iterdir())
                # contains multiple agent folders and a file per frame id
                for file in files:
                    if file.is_file() and file.suffix == ".json":
                        frame_id = file.stem.replace(".pb", "")
                        sensor_json_dict = read_json(path=file)
                        sensors = sensors_from_json(json_dict=sensor_json_dict)
                        for s in sensors:
                            result[frame_id][f"{s.name}-{sensor_stream_folder.name}"] = s

            self._sensors = result
        return self._sensors

    @property
    def streams_with_per_frame_data(self) -> List[AnyPath]:
        if self._streams_with_per_frame_data is None:
            result = []
            all_sensor_names = {s for fid, sensors in self.sensors.items() for s in sensors.keys()}

            def _get_sensor_streams(path: AnyPath):
                if path.name == "mesh":
                    return

                if not path.is_dir():
                    return

                for sensor_name in all_sensor_names:
                    sensor_folder = path / sensor_name
                    type_file_type = get_type_file_type(folder=sensor_folder)
                    if type_file_type is None:
                        continue

                    if type_file_type in [DataType.eImage, DataType.ePointCloud]:
                        yield sensor_folder

            for sensor_folder in pypeln.thread.flat_map(_get_sensor_streams, self.scene_path.iterdir(), workers=16):
                result.append(sensor_folder)

            self._streams_with_per_frame_data = result
        return self._streams_with_per_frame_data

    @property
    def ego_agent_ids(self) -> Dict[FrameId, str]:
        if self._ego_agent_ids is None:
            self._ego_agent_ids = dict()
            # we iterate over all sensor stream folders in case ego agent ids change between frames
            for sensor_stream_folder in self.sensor_agnostic_streams[DataType.eSensor]:
                for file in sensor_stream_folder.iterdir():
                    if file.is_file() and ".pb" in file.stem:
                        frame_id = file.stem.replace(".pb", "")
                        self._ego_agent_ids[frame_id] = sensor_stream_folder.name
        return self._ego_agent_ids

    def get_ego_actor_id(self, frame_id: FrameId) -> int:
        if frame_id not in self._ego_actor_ids:
            agent_id = self.ego_agent_ids[frame_id]
            mesh_map_stream = self.sensor_agnostic_streams[DataType.eMeshMap][0]
            maps = {f.stem.replace(".pb", ""): f for f in mesh_map_stream.iterdir() if f.is_file() and ".pb" in f.stem}
            with maps[frame_id].open("r") as fp:
                meshmap_dict = json.load(fp)
            meshmap = json_format.ParseDict(js_dict=meshmap_dict, message=MeshMap())
            actor_id = meshmap.agent_map[int(agent_id)]
            self._ego_actor_ids[frame_id] = actor_id
        return self._ego_actor_ids[frame_id]

    @property
    def frame_id_to_transform_file(self) -> Dict[FrameId, AnyPath]:
        if self._frame_id_to_transform_file is None:
            self._frame_id_to_transform_file = dict()
            streams = self.sensor_agnostic_streams[DataType.eTransformMap]
            stream_folder = streams[0]
            for file in stream_folder.iterdir():
                if file.is_file() and ".pb" in file.stem:
                    self._frame_id_to_transform_file[file.stem.replace(".pb", "")] = file
        return self._frame_id_to_transform_file

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

    def _get_label_data_path(
        self, frame_id: str, stream_name: str, sensor_name: Optional[str], file_ending: str
    ) -> AnyPath:
        if sensor_name is not None:
            relative_file_path = f"{stream_name}/{sensor_name}/{frame_id}.{file_ending}"
        else:
            relative_file_path = f"{stream_name}/{frame_id}.{file_ending}"
        return self.scene_path / f"{relative_file_path}"

    def get_label_data(
        self, frame_id: str, stream_name: str, sensor_name: Optional[str], file_ending: str
    ) -> LabelData:
        annotation_path = self._get_label_data_path(
            frame_id=frame_id, stream_name=stream_name, sensor_name=sensor_name, file_ending=file_ending
        )
        with annotation_path.open("rb") as f:
            sensor_name, agent_id = sensor_name.rsplit(sep="-", maxsplit=1) if sensor_name is not None else ("", -1)
            return LabelData(
                timestamp=frame_id,
                label=stream_name,
                sensor_id_and_name=(agent_id, sensor_name),
                data=f.read(),
            )

    def get_scene_metadata(self) -> Dict[str, Any]:
        return dict()

    def get_frame_ids(self) -> Set[FrameId]:
        return set(self.sensors.keys())

    def get_file_path(
        self, sensor_name: SensorName, frame_id: FrameId, data_type: SensorDataCopyTypes
    ) -> Optional[AnyPath]:
        if isinstance(data_type, type):
            if issubclass(data_type, Image):
                return self._get_label_data_path(
                    frame_id=frame_id,
                    stream_name=self.camera_image_stream_name,
                    sensor_name=sensor_name,
                    file_ending=TYPE_TO_FILE_FORMAT[data_type],
                )
        elif isinstance(data_type, AnnotationIdentifier):
            return self._get_label_data_path(
                frame_id=frame_id,
                stream_name=data_type.name,
                sensor_name=sensor_name,
                file_ending=TYPE_TO_FILE_FORMAT[data_type.annotation_type],
            )
        return None

    def get_frame_id_to_date_time_map(self) -> Dict[str, datetime]:
        try:
            return {
                fid: datetime.fromtimestamp(int(fid) / FRAME_IDS_PER_SECOND, tz=timezone.utc)
                for fid in self.get_frame_ids()
            }
        except ValueError:
            # assume nano seconds
            return {fid: datetime.fromtimestamp(int(fid) / 1_000_000, tz=timezone.utc) for fid in self.get_frame_ids()}


class StoredBatchDataStreamDataAccessor(StoredDataStreamDataAccessor):
    @property
    def ego_agent_ids(self) -> Dict[FrameId, str]:
        if self._ego_agent_ids is None:
            self._ego_agent_ids = dict()
            # we iterate over all sensor stream folders in case ego agent ids change between frames
            for sensor_stream_folder in self.sensor_agnostic_streams[DataType.eSensor]:
                files = list(sensor_stream_folder.iterdir())
                # Batch output mode
                file = next(iter([f for f in files if f.is_file() and f.suffix == ".json"]), None)
                frame_ids = list(self.frame_id_to_transform_file.keys())
                agent_id = file.stem.replace(".pb", "")
                self._ego_agent_ids = {frame_id: agent_id for frame_id in frame_ids}
        return self._ego_agent_ids

    def get_ego_actor_id(self, frame_id: FrameId) -> int:
        if frame_id not in self._ego_actor_ids:
            agent_id = self.ego_agent_ids[frame_id]
            mesh_map_stream = self.sensor_agnostic_streams[DataType.eMeshMap][0]
            maps = {f.stem.replace(".pb", ""): f for f in mesh_map_stream.iterdir() if f.is_file() and ".pb" in f.stem}
            with maps["0"].open("r") as fp:
                meshmap_dict = json.load(fp)
            meshmap = json_format.ParseDict(js_dict=meshmap_dict, message=MeshMap())
            actor_id = meshmap.agent_map[int(agent_id)]
            self._ego_actor_ids[frame_id] = actor_id
        return self._ego_actor_ids[frame_id]

    @property
    def sensors(self) -> Dict[FrameId, Dict[SensorName, Sensor]]:
        if self._sensors is None:
            result = defaultdict(dict)
            # we iterate over all sensor stream folders in case ego agent ids change between frames
            for sensor_stream_folder in self.sensor_agnostic_streams[DataType.eSensor]:
                files = list(sensor_stream_folder.iterdir())
                # Batch mode case
                # only contains the type file and a single sensor rig file with the agent id as name
                file = next(iter([f for f in files if f.is_file() and f.suffix == ".json"]), None)

                agent_id = file.stem.replace(".pb", "")
                sensor_json_dict = read_json(path=file)
                sensors = sensors_from_json(json_dict=sensor_json_dict)
                streams = [s for s in self.scene_path.iterdir() if s.name != "mesh" and s.is_dir()]

                def _get_sensor_frame_ids(stream: AnyPath):
                    for sensor in sensors:
                        sensor_stream = stream / f"{sensor.name}-{agent_id}"
                        type_file_type = get_type_file_type(folder=sensor_stream)
                        if type_file_type is not None and type_file_type in [DataType.eImage, DataType.ePointCloud]:
                            fids = [f.stem for f in sensor_stream.iterdir() if f.is_file() and ".type" not in f.stem]
                            yield fids, sensor

                for frame_ids, s in pypeln.thread.flat_map(_get_sensor_frame_ids, streams, workers=16):
                    for frame_id in frame_ids:
                        result[frame_id][f"{s.name}-{agent_id}"] = s
            self._sensors = result
        return self._sensors


class LabelEngineDataStreamDataAccessor(DataStreamDataAccessor):
    def __init__(
        self,
        labeled_state_reference: LabeledStateReference,
        camera_image_stream_name: str,
        scene_name: SceneName,
        available_annotation_identifiers: List[AnnotationIdentifier],
        label_engine_config_name: str,
    ):
        super().__init__(
            camera_image_stream_name=camera_image_stream_name,
            scene_name=scene_name,
        )
        self._label_engine_instance = labeled_state_reference.label_engine
        self._frame_ids = set()
        self._frame_id_to_date_time = dict()
        self._frame_id_to_frame_timestamp = dict()
        self._frame_id_to_ego_agent: Dict[FrameId, SensorAgent] = dict()
        self.update_labeled_state_reference(labeled_state_reference=labeled_state_reference)
        self._available_annotation_identifiers = available_annotation_identifiers
        self._label_engine_pipeline_config = json.loads(load_pipeline_config(name=label_engine_config_name))

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
    def sensors(self) -> Dict[FrameId, Dict[SensorName, Sensor]]:
        sensors = {}
        for fid, ego in self._frame_id_to_ego_agent.items():
            sensor = {}
            for s in ego.sensors:
                sensor_name = f"{s.name}-{ego.id}"
                sensor[sensor_name] = s
            sensors[fid] = sensor
        return sensors

    @property
    def available_annotation_identifiers(self) -> List[AnnotationIdentifier]:
        return self._available_annotation_identifiers

    def get_label_data(
        self, frame_id: str, stream_name: str, sensor_name: Optional[str], file_ending: str
    ) -> LabelData:
        frame_timestamp = self._frame_id_to_frame_timestamp[frame_id]
        if sensor_name is None:
            name, agent_id = None, None
        else:
            try:
                name, agent_id = sensor_name.rsplit(sep="-", maxsplit=1)
                agent_id = int(agent_id)
            except ValueError:
                name = sensor_name
                agent_id = self._frame_id_to_ego_agent[frame_id].id

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
        if not isinstance(labeled_state_reference.ego_agent, SensorAgent):
            if (
                not hasattr(labeled_state_reference.ego_agent, "sensors")
                or len(labeled_state_reference.ego_agent.sensors) == 0
            ):
                raise ValueError("ego agent has to have sensors!")
        self._frame_id_to_ego_agent[frame_id] = labeled_state_reference.ego_agent

    def get_file_path(
        self, sensor_name: SensorName, frame_id: FrameId, data_type: SensorDataCopyTypes
    ) -> Optional[AnyPath]:
        return None

    def get_frame_id_to_date_time_map(self) -> Dict[str, datetime]:
        return self._frame_id_to_date_time

    def get_ontology_data(self, frame_id: FrameId, annotation_identifier: AnnotationIdentifier) -> Optional[LabelData]:
        ontology_stream_name = self.get_ontology_name(annotation_identifier=annotation_identifier)
        if ontology_stream_name is not None and ontology_stream_name != "":
            label_data = self.get_label_data(
                stream_name=ontology_stream_name,
                sensor_name=None,
                frame_id=frame_id,
                file_ending="pb.json",
            )
            return label_data
        return None

    def get_ontology_name(self, annotation_identifier: AnnotationIdentifier) -> str:
        # Resolves ontology name based on node type in label engine config file
        # Requires a unique mapping from node_type to ontology name
        # TODO request ontology name for annotation identifier from LE directly
        if annotation_identifier.annotation_type == AnnotationTypes.Points2D:
            ontology_name = self._resolve_ontology_stream_name(
                node_type="type.googleapis.com/pd.data.InstancePoint3DAnnotatorConfig",
            )
        elif annotation_identifier.annotation_type == AnnotationTypes.Points3D:
            ontology_name = self._resolve_ontology_stream_name(
                node_type="type.googleapis.com/pd.data.InstancePoint3DAnnotatorConfig",
            )
        elif annotation_identifier.annotation_type in [
            AnnotationTypes.SemanticSegmentation2D,
            AnnotationTypes.InstanceSegmentation2D,
            AnnotationTypes.BoundingBoxes2D,
            AnnotationTypes.BoundingBoxes3D,
        ]:
            ontology_name = self._resolve_ontology_stream_name(
                node_type="type.googleapis.com/pd.data.GenerateCustomMeshSemanticMapConfig"
            )
        else:
            ontology_name = ""
        return ontology_name

    def _resolve_ontology_stream_name(self, node_type: str) -> str:
        potential_ontology_configs = [
            a for a in self._label_engine_pipeline_config["nodes"] if a["config"]["@type"] == node_type
        ]
        if len(potential_ontology_configs) == 1:
            ontology_node = potential_ontology_configs[0]
            config = ontology_node.get("config")
            if config is not None:
                ontology_stream_name = config.get("output_ontology_path")
                if ontology_stream_name is not None:
                    return ontology_stream_name
                else:
                    raise ValueError(f"Can not find output_ontology_path in label engine config {config}.")
            else:
                raise ValueError(f"Can not find config in label engine config node {ontology_node}")
        else:
            raise ValueError(
                f"Can not resolve ontology stream name from label engine pipeline config {potential_ontology_configs}"
            )


@lru_cache(maxsize=1000)
def get_type_file_type(folder: AnyPath) -> Optional[int]:
    type_file = folder / ".type"
    if not folder.is_dir() or not type_file.exists():
        return None
    with type_file.open("r") as fp:
        data_type_record = json_format.Parse(text=fp.read(), message=DataTypeRecord())
    return data_type_record.type
