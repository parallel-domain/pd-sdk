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
from paralleldomain.encoding.pipeline_encoder import EncoderStep
from paralleldomain.model.annotation import AnnotationTypes, BoundingBox2D
from paralleldomain.model.image import Image
from paralleldomain.model.sensor import CameraModel, CameraSensorFrame, FilePathedDataType, LidarSensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_json_message, relative_path


class SceneEncoderStep(EncoderStep, EncoderStepHelper):
    _fisheye_camera_model_map: Dict[str, int] = defaultdict(
        lambda: 2,
        {
            CameraModel.OPENCV_PINHOLE: 0,
            CameraModel.OPENCV_FISHEYE: 1,
        },
    )

    def __init__(
        self,
        in_queue_size: int = 1,
        inplace: bool = False,
    ):
        self.inplace = inplace
        # self.target_scene_name = target_scene_name
        # self.target_scene_description = target_scene_description

        self.in_queue_size = in_queue_size
        self.scene_data_dtos: Dict[SensorName, List[sample_pb2.Datum]] = dict()
        self.ontologie_paths: Dict[int, AnyPath] = dict()
        self._frames: Dict[FrameId, datetime] = dict()
        self._sensor_calibrations: Dict[SensorName, Tuple[geometry_pb2.Pose, geometry_pb2.CameraIntrinsics]] = dict()

    def reset_state(self):
        self.scene_data_dtos: Dict[SensorName, List[sample_pb2.Datum]] = dict()
        self.ontologie_paths: Dict[int, AnyPath] = dict()
        self._frames: Dict[FrameId, datetime] = dict()
        self._sensor_calibrations: Dict[SensorName, Tuple[geometry_pb2.Pose, geometry_pb2.CameraIntrinsics]] = dict()

    def encode_camera_sensor(self, input_dict: Dict[str, Any]):
        sensor_frame = self._get_camera_frame_from_input_dict(input_dict=input_dict)
        if sensor_frame is not None:
            sensor_data = input_dict["sensor_data"]
            annotations = input_dict.get("annotations", dict())
            frame_id = input_dict["camera_frame_info"]["frame_id"]
            target_frame_id = input_dict.get("target_frame_id", frame_id)
            metadata = input_dict.get("metadata", dict())
            scene_output_path = input_dict["scene_output_path"]
            target_sensor_name = input_dict["target_sensor_name"]

            self.aggregate_camera_sensor(
                target_sensor_name=target_sensor_name,
                scene_output_path=scene_output_path,
                camera_image_path=AnyPath(sensor_data[DirectoryName.RGB]),
                annotations=annotations,
                metadata=metadata,
                target_frame_id=target_frame_id,
                sensor_frame=sensor_frame,
            )

    def aggregate_camera_sensor(
        self,
        target_sensor_name: str,
        scene_output_path: AnyPath,
        camera_image_path: AnyPath,
        annotations: Dict[int, AnyPath],
        metadata: Dict[str, Any],
        target_frame_id: FrameId,
        sensor_frame: CameraSensorFrame,
    ):
        quaternion = sensor_frame.pose.quaternion
        date_time = sensor_frame.date_time
        image_height = sensor_frame.image.height
        image_width = sensor_frame.image.width
        rotation = [quaternion.w, quaternion.x, quaternion.y, quaternion.z]

        if target_sensor_name not in self.scene_data_dtos:
            self.scene_data_dtos[target_sensor_name] = list()

        scene_datum_dto = image_pb2.Image(
            filename=relative_path(start=scene_output_path, path=camera_image_path).as_posix(),
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
        self.aggegate_camera_calibration(target_sensor_name=target_sensor_name, sensor_frame=sensor_frame)

    def encode_lidar_sensor(self, input_dict: Dict[str, Any]):
        sensor_frame = self._get_lidar_frame_from_input_dict(input_dict=input_dict)
        if sensor_frame is not None:
            sensor_data = input_dict["sensor_data"]
            annotations = input_dict.get("annotations", dict())
            frame_id = input_dict["lidar_frame_info"]["frame_id"]
            target_frame_id = input_dict.get("target_frame_id", frame_id)
            metadata = input_dict.get("metadata", dict())
            scene_output_path = input_dict["scene_output_path"]
            target_sensor_name = input_dict["target_sensor_name"]
            self.aggregate_lidar_sensor(
                target_sensor_name=target_sensor_name,
                scene_output_path=scene_output_path,
                lidar_cloud_path=AnyPath(sensor_data[DirectoryName.POINT_CLOUD]),
                annotations=annotations,
                metadata=metadata,
                target_frame_id=target_frame_id,
                sensor_frame=sensor_frame,
            )

    def aggregate_lidar_sensor(
        self,
        target_sensor_name: str,
        scene_output_path: AnyPath,
        lidar_cloud_path: AnyPath,
        annotations: Dict[int, AnyPath],
        metadata: Dict[str, Any],
        target_frame_id: FrameId,
        sensor_frame: LidarSensorFrame,
    ):

        if target_sensor_name not in self.scene_data_dtos:
            self.scene_data_dtos[target_sensor_name] = list()

        scene_datum_dto = point_cloud_pb2.PointCloud(
            filename=relative_path(start=scene_output_path, path=lidar_cloud_path).as_posix(),
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
        self.encode_lidar_calibration(target_sensor_name=target_sensor_name, sensor_frame=sensor_frame)

    def encode_lidar_calibration(self, target_sensor_name: str, sensor_frame: LidarSensorFrame):
        if target_sensor_name not in self._sensor_calibrations:
            extr = sensor_frame.extrinsic

            calib_dto_extrinsic = geometry_pb2.Pose(
                translation=geometry_pb2.Vector3(x=extr.translation[0], y=extr.translation[1], z=extr.translation[2]),
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

    def aggegate_camera_calibration(self, target_sensor_name: str, sensor_frame: CameraSensorFrame):
        if target_sensor_name not in self._sensor_calibrations:
            extr = sensor_frame.extrinsic

            calib_dto_extrinsic = geometry_pb2.Pose(
                translation=geometry_pb2.Vector3(x=extr.translation[0], y=extr.translation[1], z=extr.translation[2]),
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
            scene = self._scene_from_input_dict(input_dict=input_dict)

            self.aggegate_ontologies(annotations=annotations, scene_output_path=scene_output_path, scene=scene)

    def aggegate_ontologies(self, annotations: Dict[int, AnyPath], scene_output_path: AnyPath, scene: Scene):
        for annotation_id in annotations.keys():
            if annotation_id not in self.ontologie_paths:
                annotation_id = int(annotation_id)
                a_type = ANNOTATION_TYPE_MAP[annotation_id]
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

    def encode_frame_map(self, input_dict: Dict[str, Any]):
        sensor_frame = self._get_lidar_frame_from_input_dict(input_dict=input_dict)
        if sensor_frame is None:
            sensor_frame = self._get_camera_frame_from_input_dict(input_dict=input_dict)
            frame_id = input_dict["camera_frame_info"]["frame_id"]
        else:
            frame_id = input_dict["lidar_frame_info"]["frame_id"]
        if sensor_frame is not None:
            target_frame_id = input_dict.get("target_frame_id", frame_id)
            self.aggregate_frame_map(target_frame_id=target_frame_id, date_time=sensor_frame.date_time)

    def aggregate_frame_map(self, target_frame_id: FrameId, date_time: datetime):
        self._frames[target_frame_id] = date_time

    def save_calibration_file(self, scene_output_path: AnyPath) -> AnyPath:
        names, extrinsics, intrinsics = list(), list(), list()
        for name, (extrinsic, intrinsic) in self._sensor_calibrations.items():
            names.append(name)
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
        calib_dto = sample_pb2.SampleCalibration(names=names, extrinsics=extrinsics, intrinsics=intrinsics)

        output_path = scene_output_path / DirectoryName.CALIBRATION / ".json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return fsio.write_json_message(obj=calib_dto, path=output_path, append_sha1=True)

    def encode_scene(self, input_dict: Dict[str, Any]):
        if "end_of_scene" not in input_dict:
            self.encode_camera_sensor(input_dict=input_dict)
            self.encode_lidar_sensor(input_dict=input_dict)
            self.encode_ontologies(input_dict=input_dict)
            self.encode_frame_map(input_dict=input_dict)
            return dict()
        else:
            scene_info = self.write_scene_data(input_dict=input_dict)
            self.reset_state()
            return scene_info

    def apply(self, input_stage: Iterable[Any]) -> Iterable[Any]:
        stage = input_stage
        stage = pypeln.thread.map(f=self.encode_scene, stage=stage, workers=1, maxsize=self.in_queue_size)
        return stage

    def write_scene_data(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        scene_output_path = input_dict["scene_output_path"]
        target_scene_name = input_dict["target_scene_name"]
        target_scene_description = input_dict["target_scene_description"]

        if self.inplace:
            scene = self._scene_from_input_dict(input_dict=input_dict)
            self.aggregate_inplace_data(scene=scene, scene_output_path=scene_output_path)

        scene_data = []
        scene_samples = []
        scene_sensor_data = self.get_scene_sensor_data()
        calibration_file = self.save_calibration_file(scene_output_path=scene_output_path)
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
            name=target_scene_name,
            description=target_scene_description,
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

        output_path = scene_output_path / "scene.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scene_storage_path = fsio.write_json_message(obj=scene_dto, path=output_path, append_sha1=True)
        return dict(scene_storage_path=scene_storage_path, available_annotation_types=available_annotation_types)

    def aggregate_inplace_data(self, scene: Scene, scene_output_path: AnyPath) -> Optional[scene_pb2.Scene]:
        for camera in scene.cameras:
            for cam_frame in camera.sensor_frames:
                camera_image_path = cam_frame.get_file_path(data_type=FilePathedDataType.Image)
                if camera_image_path is None:
                    raise ValueError(
                        "The given dataset is not compatible to be inplace encoded since file paths cant be accessed!"
                    )

                annotations = {
                    ANNOTATION_TYPE_MAP_INV[an_type]: cam_frame.get_file_path(data_type=an_type)
                    for an_type in cam_frame.available_annotation_types
                }
                for path in annotations.values():
                    if path is None:
                        raise ValueError(
                            "The given dataset is not compatible to be inplace encoded since file "
                            "paths cant be accessed!"
                        )
                self.aggregate_camera_sensor(
                    target_sensor_name=camera.name,
                    scene_output_path=scene_output_path,
                    camera_image_path=camera_image_path,
                    annotations=annotations,
                    metadata=cam_frame.metadata,
                    target_frame_id=cam_frame.frame_id,
                    sensor_frame=cam_frame,
                )
                self.aggegate_ontologies(annotations=annotations, scene_output_path=scene_output_path, scene=scene)
                self.aggregate_frame_map(target_frame_id=cam_frame.frame_id, date_time=cam_frame.date_time)

        for lidar in scene.lidars:
            for lidar_frame in lidar.sensor_frames:
                cloud_path = lidar_frame.get_file_path(data_type=FilePathedDataType.PointCloud)
                if cloud_path is None:
                    raise ValueError(
                        "The given dataset is not compatible to be inplace encoded since file paths cant be accessed!"
                    )

                annotations = {
                    ANNOTATION_TYPE_MAP_INV[an_type]: lidar_frame.get_file_path(data_type=an_type)
                    for an_type in lidar_frame.available_annotation_types
                }
                for path in annotations.values():
                    if path is None:
                        raise ValueError(
                            "The given dataset is not compatible to be inplace encoded since file "
                            "paths cant be accessed!"
                        )
                self.aggregate_lidar_sensor(
                    target_sensor_name=lidar.name,
                    scene_output_path=scene_output_path,
                    lidar_cloud_path=cloud_path,
                    annotations=annotations,
                    metadata=lidar_frame.metadata,
                    target_frame_id=lidar_frame.frame_id,
                    sensor_frame=lidar_frame,
                )
                self.aggegate_ontologies(annotations=annotations, scene_output_path=scene_output_path, scene=scene)
                self.aggregate_frame_map(target_frame_id=lidar_frame.frame_id, date_time=lidar_frame.date_time)
