import hashlib
import uuid
from collections import defaultdict
from datetime import datetime
from functools import partial
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, cast

import pypeln
from google.protobuf import timestamp_pb2

from paralleldomain import Scene
from paralleldomain.common.dgp.v1 import (
    annotations_pb2,
    geometry_pb2,
    identifiers_pb2,
    image_pb2,
    point_cloud_pb2,
    sample_pb2,
    scene_pb2,
)
from paralleldomain.common.dgp.v1.constants import (
    ANNOTATION_TYPE_MAP,
    ANNOTATION_TYPE_MAP_INV,
    DirectoryName,
    PointFormat,
)
from paralleldomain.common.dgp.v1.utils import datetime_to_timestamp
from paralleldomain.encoding.dgp.v1.encoder_steps.helper import EncoderStepHelper
from paralleldomain.encoding.dgp.v1.utils import _attribute_key_dump, _attribute_value_dump, class_map_to_ontology_proto
from paralleldomain.encoding.pipeline_encoder import FinalStep
from paralleldomain.model.annotation import AnnotationTypes, BoundingBox2D
from paralleldomain.model.image import Image
from paralleldomain.model.sensor import CameraModel, CameraSensorFrame, FilePathedDataType
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import relative_path


class SceneEncoderStep(FinalStep[Dict[str, Any]], EncoderStepHelper):
    _fisheye_camera_model_map: Dict[str, int] = defaultdict(
        lambda: 2,
        {
            CameraModel.OPENCV_PINHOLE: 0,
            CameraModel.OPENCV_FISHEYE: 1,
        },
    )

    def __init__(
        self,
        in_queue_size: int = 4,
        target_scene_name_mapping: Dict[SceneName, str] = None,
        target_scene_description_mapping: Dict[SceneName, str] = None,
    ):
        if target_scene_description_mapping is None:
            target_scene_description_mapping = dict()
        if target_scene_name_mapping is None:
            target_scene_name_mapping = dict()
        self.target_scene_description_mapping = target_scene_description_mapping
        self.target_scene_name_mapping = target_scene_name_mapping
        self.target_scene_description = target_scene_description_mapping
        self.target_scene_name = target_scene_name_mapping
        self.in_queue_size = in_queue_size
        self.scene_data_dtos: Dict[SensorName, List[sample_pb2.Datum]] = dict()
        self.ontologie_paths: Dict[int, AnyPath] = dict()
        self._frames: Dict[FrameId, datetime] = dict()
        self._scene_output_path: Optional[AnyPath] = None
        self._sensor_calibrations: Dict[SensorName, Tuple[geometry_pb2.Pose, geometry_pb2.CameraIntrinsics]] = dict()

    def encode_camera_sensor(self, input_dict: Dict[str, Any]):
        sensor_frame = self._get_camera_frame_from_input_dict(input_dict=input_dict)
        if sensor_frame is not None:
            quaternion = sensor_frame.pose.quaternion
            date_time = sensor_frame.date_time
            image_height = sensor_frame.image.height
            image_width = sensor_frame.image.width
            rotation = [quaternion.w, quaternion.x, quaternion.y, quaternion.z]
            sensor_data = input_dict["sensor_data"]
            annotations = input_dict["annotations"]
            frame_id = input_dict["camera_frame_info"]["frame_id"]
            target_frame_id = input_dict.get("target_frame_id", frame_id)
            metadata = input_dict.get("metadata", dict())
            scene_output_path = input_dict["scene_output_path"]
            target_sensor_name = input_dict["target_sensor_name"]

            if target_sensor_name not in self.scene_data_dtos:
                self.scene_data_dtos[target_sensor_name] = list()

            scene_datum_dto = image_pb2.Image(
                filename=relative_path(start=scene_output_path, path=sensor_data[DirectoryName.RGB]).as_posix(),
                height=image_height,
                width=image_width,
                channels=4,
                annotations={
                    int(k): relative_path(start=scene_output_path, path=v).as_posix()
                    for k, v in annotations.items()
                    if v is not None
                },
                pose=geometry_pb2.Pose(
                    translation=geometry_pb2.Vector3(
                        x=sensor_frame.pose.translation[0],
                        y=sensor_frame.pose.translation[1],
                        z=sensor_frame.pose.translation[2],
                    ),
                    rotation=geometry_pb2.Quaternion(
                        qw=rotation[0],
                        qx=rotation[1],
                        qy=rotation[2],
                        qz=rotation[3],
                    ),
                ),
                metadata={str(k): v for k, v in metadata.items()},
            )
            # noinspection PyTypeChecker
            self.scene_data_dtos[target_sensor_name].append(
                sample_pb2.Datum(
                    id=identifiers_pb2.DatumId(
                        log="",
                        name=target_sensor_name,
                        timestamp=datetime_to_timestamp(dt=date_time),
                        index=int(target_frame_id),
                    ),
                    key="",
                    datum=sample_pb2.DatumValue(image=scene_datum_dto),
                    next_key="",
                    prev_key="",
                )
            )

    def encode_lidar_sensor(self, input_dict: Dict[str, Any]):
        sensor_frame = self._get_lidar_frame_from_input_dict(input_dict=input_dict)
        if sensor_frame is not None:
            sensor_data = input_dict["sensor_data"]
            annotations = input_dict["annotations"]
            frame_id = input_dict["camera_frame_info"]["frame_id"]
            target_frame_id = input_dict.get("target_frame_id", frame_id)
            metadata = input_dict.get("metadata", dict())
            scene_output_path = input_dict["scene_output_path"]
            target_sensor_name = input_dict["target_sensor_name"]

            if target_sensor_name not in self.scene_data_dtos:
                self.scene_data_dtos[target_sensor_name] = list()

            scene_datum_dto = point_cloud_pb2.PointCloud(
                filename=relative_path(start=scene_output_path, path=sensor_data[DirectoryName.POINT_CLOUD]).as_posix(),
                point_format=[getattr(point_cloud_pb2.PointCloud.ChannelType, pf) for pf in PointFormat.to_list()],
                annotations={
                    int(k): relative_path(start=scene_output_path, path=v).as_posix()
                    for k, v in annotations.items()
                    if v is not None
                },
                pose=geometry_pb2.Pose(
                    translation=geometry_pb2.Vector3(
                        x=sensor_frame.pose.translation[0],
                        y=sensor_frame.pose.translation[1],
                        z=sensor_frame.pose.translation[2],
                    ),
                    rotation=geometry_pb2.Quaternion(
                        qw=sensor_frame.pose.quaternion.w,
                        qx=sensor_frame.pose.quaternion.x,
                        qy=sensor_frame.pose.quaternion.y,
                        qz=sensor_frame.pose.quaternion.z,
                    ),
                ),
                point_fields=[],
                metadata={str(k): v for k, v in metadata.items()},
            )
            # noinspection PyTypeChecker
            self.scene_data_dtos[target_sensor_name].append(
                sample_pb2.Datum(
                    id=identifiers_pb2.DatumId(
                        log="",
                        name=target_sensor_name,
                        timestamp=datetime_to_timestamp(dt=sensor_frame.date_time),
                        index=int(target_frame_id),
                    ),
                    key="",
                    datum=sample_pb2.DatumValue(point_cloud=scene_datum_dto),
                    next_key="",
                    prev_key="",
                )
            )

    def encode_lidar_calibration(self, input_dict: Dict[str, Any]):
        sensor_frame = self._get_lidar_frame_from_input_dict(input_dict=input_dict)
        if sensor_frame is not None:
            target_sensor_name = input_dict["target_sensor_name"]
            if target_sensor_name not in self._sensor_calibrations:
                extr = sensor_frame.extrinsic

                calib_dto_extrinsic = geometry_pb2.Pose(
                    translation=geometry_pb2.Vector3(
                        x=extr.translation[0], y=extr.translation[1], z=extr.translation[2]
                    ),
                    rotation=geometry_pb2.Quaternion(
                        qw=extr.quaternion.w, qx=extr.quaternion.x, qy=extr.quaternion.y, qz=extr.quaternion.z
                    ),
                )

                calib_dto_intrinsic = geometry_pb2.CameraIntrinsics(
                    fx=0.0,
                    fy=0.0,
                    cx=0.0,
                    cy=0.0,
                    skew=0.0,
                    fov=0.0,
                    k1=0.0,
                    k2=0.0,
                    k3=0.0,
                    k4=0.0,
                    k5=0.0,
                    k6=0.0,
                    p1=0.0,
                    p2=0.0,
                    fisheye=0,
                )
                self._sensor_calibrations[target_sensor_name] = (calib_dto_extrinsic, calib_dto_intrinsic)

    def encode_camera_calibration(self, input_dict: Dict[str, Any]):
        sensor_frame = self._get_camera_frame_from_input_dict(input_dict=input_dict)
        if sensor_frame is not None:
            target_sensor_name = input_dict["target_sensor_name"]
            if target_sensor_name not in self._sensor_calibrations:
                extr = sensor_frame.extrinsic

                calib_dto_extrinsic = geometry_pb2.Pose(
                    translation=geometry_pb2.Vector3(
                        x=extr.translation[0], y=extr.translation[1], z=extr.translation[2]
                    ),
                    rotation=geometry_pb2.Quaternion(
                        qw=extr.quaternion.w, qx=extr.quaternion.x, qy=extr.quaternion.y, qz=extr.quaternion.z
                    ),
                )

                intr = sensor_frame.intrinsic
                calib_dto_intrinsic = geometry_pb2.CameraIntrinsics(
                    fx=intr.fx,
                    fy=intr.fy,
                    cx=intr.cx,
                    cy=intr.cy,
                    skew=intr.skew,
                    fov=intr.fov,
                    k1=intr.k1,
                    k2=intr.k2,
                    k3=intr.k3,
                    k4=intr.k4,
                    k5=intr.k5,
                    k6=intr.k6,
                    p1=intr.p1,
                    p2=intr.p2,
                    fisheye=self._fisheye_camera_model_map[intr.camera_model],
                )
                self._sensor_calibrations[target_sensor_name] = (calib_dto_extrinsic, calib_dto_intrinsic)

    def encode_ontologies(self, input_dict: Dict[str, Any]):
        sensor_frame = self._get_lidar_frame_from_input_dict(input_dict=input_dict)
        if sensor_frame is None:
            sensor_frame = self._get_camera_frame_from_input_dict(input_dict=input_dict)
        if sensor_frame is not None:
            annotations = input_dict["annotations"]
            scene_output_path = input_dict["scene_output_path"]

            for annotation_id in annotations.keys():
                if annotation_id not in self.ontologie_paths:
                    annotation_id = int(annotation_id)
                    a_type = ANNOTATION_TYPE_MAP[annotation_id]
                    scene = self._scene_from_input_dict(input_dict=input_dict)
                    ontology_proto = class_map_to_ontology_proto(class_map=scene.get_class_map(a_type))

                    output_path = scene_output_path / DirectoryName.ONTOLOGY / ".json"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    path = fsio.write_json_message(obj=ontology_proto, path=output_path, append_sha1=True)
                    self.ontologie_paths[annotation_id] = relative_path(start=scene_output_path, path=path)

    def get_scene_sensor_data(self) -> Dict[SensorName, Dict[FrameId, sample_pb2.Datum]]:
        scene_sensor_data = dict()
        for sensor_name, scene_data_dtos in self.scene_data_dtos.items():
            scene_data_count = len(scene_data_dtos)
            # noinspection InsecureHash
            keys = [hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest() for _ in range(scene_data_count)]

            for idx, scene_data_dto in enumerate(sorted(scene_data_dtos, key=lambda x: x.id.timestamp.ToDatetime())):
                prev_key = keys[idx - 1] if idx > 0 else ""
                key = keys[idx]
                next_key = keys[idx + 1] if idx < (scene_data_count - 1) else ""

                scene_data_dto.prev_key = prev_key
                scene_data_dto.key = key
                scene_data_dto.next_key = next_key

            scene_sensor_data[sensor_name] = {str(sd.id.index): sd for sd in scene_data_dtos}
        return scene_sensor_data

    def aggregate_frame_map(self, input_dict: Dict[str, Any]):
        sensor_frame = self._get_lidar_frame_from_input_dict(input_dict=input_dict)
        if sensor_frame is None:
            sensor_frame = self._get_camera_frame_from_input_dict(input_dict=input_dict)
        if sensor_frame is not None:
            frame_id = input_dict["camera_frame_info"]["frame_id"]
            target_frame_id = input_dict.get("target_frame_id", frame_id)
            self._frames[target_frame_id] = sensor_frame.date_time

    def save_calibration_file(self) -> AnyPath:
        names, extrinsics, intrinsics = list(), list(), list()
        for name, (extrinsic, intrinsic) in self._sensor_calibrations.items():
            names.append(name)
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
        calib_dto = sample_pb2.SampleCalibration(names=names, extrinsics=extrinsics, intrinsics=intrinsics)

        output_path = self._scene_output_path / DirectoryName.CALIBRATION / ".json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return fsio.write_json_message(obj=calib_dto, path=output_path, append_sha1=True)

    def encode_scene(self, input_dict: Dict[str, Any]):
        if self._scene_output_path is None:
            self._scene_output_path = input_dict["scene_output_path"]
        self.encode_camera_sensor(input_dict=input_dict)
        self.encode_lidar_sensor(input_dict=input_dict)
        self.encode_lidar_calibration(input_dict=input_dict)
        self.encode_camera_calibration(input_dict=input_dict)
        self.encode_ontologies(input_dict=input_dict)
        self.aggregate_frame_map(input_dict=input_dict)
        return input_dict

    def aggregate(self, scene: Scene, input_stage: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        if scene.name not in self.target_scene_name_mapping:
            self.target_scene_name_mapping[scene.name] = scene.name
        self.target_scene_name = self.target_scene_name_mapping[scene.name]
        if scene.name not in self.target_scene_description_mapping:
            self.target_scene_description_mapping[scene.name] = scene.description
        self.target_scene_description = self.target_scene_description_mapping[scene.name]

        stage = pypeln.thread.map(f=self.encode_scene, stage=input_stage, workers=1, maxsize=self.in_queue_size)
        return stage

    def finalize(self) -> Dict[str, Any]:
        if self._scene_output_path is None:
            raise ValueError("No scene output path could be extracted during aggregation! Is the scene empty?")
        scene_data = []
        scene_samples = []
        scene_sensor_data = self.get_scene_sensor_data()
        calibration_file = self.save_calibration_file()
        ontologies = {k: v.stem for k, v in self.ontologie_paths.items()}
        available_annotation_types = [int(k) for k in ontologies.keys()]
        for frame_id, date_time in self._frames.items():
            frame_data = [
                scene_sensor_data[sn][frame_id]
                for sn in sorted(scene_sensor_data.keys())
                if frame_id in scene_sensor_data[sn]
            ]
            scene_data.extend(frame_data)
            scene_samples.append(
                sample_pb2.Sample(
                    id=identifiers_pb2.DatumId(
                        log="",
                        timestamp=datetime_to_timestamp(dt=date_time),
                        name="",
                        index=int(frame_id),
                    ),
                    datum_keys=[d.key for d in frame_data],
                    calibration_key=calibration_file.stem,
                    metadata={},
                )
            )

        scene_dto = scene_pb2.Scene(
            name=self.target_scene_name,
            description=self.target_scene_description,
            log="",
            ontologies=ontologies,
            metadata={
                # "PD": any_pb2.Any().Pack(
                #     metadata_pd_pb2.ParallelDomainSceneMetadata(
                #         **{k: v for k, v in self._scene.metadata["PD"].items() if not k.startswith("@")}
                #     )
                # )
            },
            samples=scene_samples,
            data=scene_data,
            creation_date=timestamp_pb2.Timestamp().GetCurrentTime(),
            statistics=None,
        )

        output_path = self._scene_output_path / "scene.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scene_storage_path = fsio.write_json_message(obj=scene_dto, path=output_path, append_sha1=True)
        return dict(scene_storage_path=scene_storage_path, available_annotation_types=available_annotation_types)
