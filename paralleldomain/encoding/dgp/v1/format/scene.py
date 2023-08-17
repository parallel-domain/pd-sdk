import hashlib
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
from google.protobuf import timestamp_pb2

from paralleldomain import Scene
from paralleldomain.common.dgp.v1 import (
    geometry_pb2,
    identifiers_pb2,
    image_pb2,
    ontology_pb2,
    point_cloud_pb2,
    sample_pb2,
    scene_pb2,
)
from paralleldomain.common.dgp.v1.constants import ANNOTATION_TYPE_MAP, DirectoryName, PointFormat
from paralleldomain.common.dgp.v1.utils import datetime_to_timestamp
from paralleldomain.encoding.dgp.v1.format.aggregation import DataAggregationMixin
from paralleldomain.encoding.dgp.v1.format.common import (
    ANNOTATIONS_KEY,
    CLASS_MAPS_KEY,
    CUSTOM_FORMAT_KEY,
    ENCODED_SCENE_AGGREGATION_FOLDER_NAME,
    META_DATA_KEY,
    SCENE_DATA_KEY,
    SENSOR_DATA_KEY,
    CommonDGPV1FormatMixin,
)
from paralleldomain.encoding.dgp.v1.utils import class_map_to_ontology_proto
from paralleldomain.encoding.pipeline_encoder import PipelineItem, ScenePipelineItem
from paralleldomain.model.class_mapping import ClassMap
from paralleldomain.model.sensor import CameraModel, CameraSensorFrame, LidarSensorFrame
from paralleldomain.model.type_aliases import FrameId, SensorName
from paralleldomain.utilities import fsio
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import relative_path


class SceneDGPV1Mixin(CommonDGPV1FormatMixin, DataAggregationMixin):
    _fisheye_camera_model_map: Dict[str, int] = defaultdict(
        lambda: 2,
        {
            CameraModel.OPENCV_PINHOLE: 0,
            CameraModel.OPENCV_FISHEYE: 1,
            CameraModel.PD_FISHEYE: 3,
            CameraModel.PD_ORTHOGRAPHIC: 6,
        },
    )

    def aggregate_sensor_frame(self, pipeline_item: ScenePipelineItem):
        self.ensure_format_data_exists(pipeline_item=pipeline_item)
        self.store_data_for_aggregation(pipeline_item=pipeline_item)

    def save_aggregated_scene(self, pipeline_item: ScenePipelineItem, dataset_output_path: AnyPath, save_binary: bool):
        self.ensure_format_data_exists(pipeline_item=pipeline_item)
        target_scene_name = pipeline_item.scene_name
        scene_output_path = dataset_output_path / target_scene_name
        target_scene_description = ""
        # if self.inplace:
        #     scene = self._scene_from_input_dict(input_dict=input_dict)
        #     self.aggregate_inplace_data(scene=scene, scene_output_path=scene_output_path)

        file_suffix = "json" if not save_binary else "bin"

        scene_data = []
        scene_samples = []
        scene_sensor_data, calib_dto, ontology_maps, frame_map = self.get_scene_sensor_data(
            scene_name=pipeline_item.scene_name, scene_output_path=scene_output_path
        )

        output_path = scene_output_path / DirectoryName.CALIBRATION / f".{file_suffix}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_file = fsio.write_message(obj=calib_dto, path=output_path, append_sha1=True)

        # calibration_file = self.save_calibration_file(pipeline_item=pipeline_item)
        ontologies = dict()
        for annotaiton_id, ontology_proto in ontology_maps.items():
            output_path = scene_output_path / DirectoryName.ONTOLOGY / f".{file_suffix}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            path = fsio.write_message(obj=ontology_proto, path=output_path, append_sha1=True)
            ontologies[annotaiton_id] = relative_path(start=scene_output_path, path=path).stem

        available_annotation_types = [int(k) for k in ontologies.keys()]
        for frame_id, date_time in list(sorted(frame_map.items(), key=lambda item: item[1])):
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

        output_path = scene_output_path / f"scene.{file_suffix}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scene_storage_path = fsio.write_message(obj=scene_dto, path=output_path, append_sha1=True)
        pipeline_item.custom_data[CUSTOM_FORMAT_KEY][SCENE_DATA_KEY] = dict(
            scene_storage_path=scene_storage_path, available_annotation_types=available_annotation_types
        )
        self.store_item_for_aggregation(
            data=pipeline_item,
            path=dataset_output_path / ENCODED_SCENE_AGGREGATION_FOLDER_NAME / f"{uuid.uuid4()}.pickle",
        )
        self.clean_up_scene_tmp_aggregation_folder(scene_name=pipeline_item.scene_name)

    def encode_camera_sensor_data_and_calibration(
        self, scene_name: str, scene_output_path: AnyPath
    ) -> Tuple[
        Dict[SensorName, List[sample_pb2.Datum]],
        Dict[SensorName, Tuple[geometry_pb2.CameraIntrinsics, geometry_pb2.Pose]],
        Dict[int, ontology_pb2.Ontology],
        Dict[FrameId, datetime],
    ]:
        sensor_data = dict()
        calibration_data = dict()
        ontology_maps = dict()
        frame_map = dict()
        for pipeline_item in self.load_camera_data_for_aggregation(scene_name=scene_name):
            if pipeline_item.camera_frame is not None:
                if pipeline_item.target_sensor_name not in calibration_data:
                    intrinsics = self.encode_camera_intrinsic(sensor_frame=pipeline_item.camera_frame)
                    extrinsics = self.encode_camera_extrinsic(sensor_frame=pipeline_item.camera_frame)
                    calibration_data[pipeline_item.target_sensor_name] = (intrinsics, extrinsics)

                if pipeline_item.target_sensor_name not in sensor_data:
                    sensor_data[pipeline_item.target_sensor_name] = list()

                data = self.encode_camera_sensor(pipeline_item=pipeline_item, scene_output_path=scene_output_path)
                sensor_data[pipeline_item.target_sensor_name].append(data)
                ontology_map = self.encode_ontologies(pipeline_item=pipeline_item)
                ontology_maps.update(ontology_map)
                target_frame_id = pipeline_item.frame_id
                frame_map[target_frame_id] = pipeline_item.sensor_frame.date_time

        return sensor_data, calibration_data, ontology_maps, frame_map

    def encode_lidar_sensor_data_and_calibration(
        self,
        scene_name: str,
        scene_output_path: AnyPath,
    ) -> Tuple[
        Dict[SensorName, List[sample_pb2.Datum]],
        Dict[SensorName, Tuple[geometry_pb2.CameraIntrinsics, geometry_pb2.Pose]],
        Dict[int, ontology_pb2.Ontology],
        Dict[FrameId, datetime],
    ]:
        sensor_data = dict()
        calibration_data = dict()
        ontology_maps = dict()
        frame_map = dict()
        for pipeline_item in self.load_lidar_data_for_aggregation(scene_name=scene_name):
            if pipeline_item.lidar_frame is not None:
                if pipeline_item.target_sensor_name not in calibration_data:
                    intrinsics = self.encode_lidar_intrinsic(sensor_frame=pipeline_item.lidar_frame)
                    extrinsics = self.encode_lidar_extrinsic(sensor_frame=pipeline_item.lidar_frame)
                    calibration_data[pipeline_item.target_sensor_name] = (intrinsics, extrinsics)

                if pipeline_item.target_sensor_name not in sensor_data:
                    sensor_data[pipeline_item.target_sensor_name] = list()

                data = self.encode_lidar_sensor(pipeline_item=pipeline_item, scene_output_path=scene_output_path)
                sensor_data[pipeline_item.target_sensor_name].append(data)
                ontology_map = self.encode_ontologies(pipeline_item=pipeline_item)
                ontology_maps.update(ontology_map)
                target_frame_id = pipeline_item.frame_id
                frame_map[target_frame_id] = pipeline_item.sensor_frame.date_time

        return sensor_data, calibration_data, ontology_maps, frame_map

    def encode_camera_sensor(self, pipeline_item: ScenePipelineItem, scene_output_path: AnyPath) -> sample_pb2.Datum:
        if pipeline_item.camera_frame is not None:
            sensor_data = pipeline_item.custom_data[CUSTOM_FORMAT_KEY][SENSOR_DATA_KEY]
            annotations = pipeline_item.custom_data[CUSTOM_FORMAT_KEY][ANNOTATIONS_KEY]
            metadata = pipeline_item.custom_data[CUSTOM_FORMAT_KEY][META_DATA_KEY]
            target_frame_id = pipeline_item.frame_id
            target_sensor_name = pipeline_item.target_sensor_name
            camera_datum = self.encode_camera_datum(
                target_sensor_name=target_sensor_name,
                scene_output_path=scene_output_path,
                camera_image_path=AnyPath(sensor_data[DirectoryName.RGB]),
                annotations=annotations,
                metadata=metadata,
                target_frame_id=target_frame_id,
                sensor_frame=pipeline_item.camera_frame,
                stored_width=pipeline_item.custom_data[CUSTOM_FORMAT_KEY].get("image_width"),
                stored_height=pipeline_item.custom_data[CUSTOM_FORMAT_KEY].get("image_height"),
            )
            return camera_datum
        raise ValueError("The given pipeline item does not contain a camera frame!")

    def encode_camera_datum(
        self,
        target_sensor_name: str,
        scene_output_path: AnyPath,
        camera_image_path: AnyPath,
        annotations: Dict[int, AnyPath],
        metadata: Dict[str, Any],
        target_frame_id: FrameId,
        sensor_frame: CameraSensorFrame,
        stored_width: Optional[int],
        stored_height: Optional[int],
    ) -> sample_pb2.Datum:
        quaternion = sensor_frame.pose.quaternion
        date_time = sensor_frame.date_time
        if stored_height is None:
            image_height = sensor_frame.image.height
        else:
            image_height = stored_height
        if stored_width is None:
            image_width = sensor_frame.image.width
        else:
            image_width = stored_width
        rotation = [quaternion.w, quaternion.x, quaternion.y, quaternion.z]

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
        return sample_pb2.Datum(
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

    def encode_lidar_sensor(self, pipeline_item: ScenePipelineItem, scene_output_path: AnyPath) -> sample_pb2.Datum:
        if pipeline_item.lidar_frame is not None:
            sensor_data = pipeline_item.custom_data[CUSTOM_FORMAT_KEY][SENSOR_DATA_KEY]
            annotations = pipeline_item.custom_data[CUSTOM_FORMAT_KEY][ANNOTATIONS_KEY]
            metadata = pipeline_item.custom_data[CUSTOM_FORMAT_KEY][META_DATA_KEY]
            target_frame_id = pipeline_item.frame_id
            # scene_output_path = self.dataset_output_path / pipeline_item.scene_name
            target_sensor_name = pipeline_item.target_sensor_name
            lidar_datum = self.encode_lidar_datum(
                target_sensor_name=target_sensor_name,
                scene_output_path=scene_output_path,
                lidar_cloud_path=AnyPath(sensor_data[DirectoryName.POINT_CLOUD]),
                annotations=annotations,
                metadata=metadata,
                target_frame_id=target_frame_id,
                sensor_frame=pipeline_item.lidar_frame,
            )
            return lidar_datum
        raise ValueError("The given pipeline item does not contain a lidar frame!")

    def encode_lidar_datum(
        self,
        target_sensor_name: str,
        scene_output_path: AnyPath,
        lidar_cloud_path: AnyPath,
        annotations: Dict[int, AnyPath],
        metadata: Dict[str, Any],
        target_frame_id: FrameId,
        sensor_frame: LidarSensorFrame,
    ) -> sample_pb2.Datum:
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
        return sample_pb2.Datum(
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

    def encode_lidar_intrinsic(self, sensor_frame: LidarSensorFrame) -> geometry_pb2.CameraIntrinsics:
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
        return calib_dto_intrinsic

    def encode_lidar_extrinsic(self, sensor_frame: LidarSensorFrame) -> geometry_pb2.Pose:
        extr = sensor_frame.extrinsic

        calib_dto_extrinsic = geometry_pb2.Pose(
            translation=geometry_pb2.Vector3(x=extr.translation[0], y=extr.translation[1], z=extr.translation[2]),
            rotation=geometry_pb2.Quaternion(
                qw=extr.quaternion.w, qx=extr.quaternion.x, qy=extr.quaternion.y, qz=extr.quaternion.z
            ),
        )
        return calib_dto_extrinsic

    def encode_camera_intrinsic(self, sensor_frame: CameraSensorFrame) -> geometry_pb2.CameraIntrinsics:
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
        return calib_dto_intrinsic

    def encode_camera_extrinsic(self, sensor_frame: CameraSensorFrame) -> geometry_pb2.Pose:
        extr = sensor_frame.extrinsic

        calib_dto_extrinsic = geometry_pb2.Pose(
            translation=geometry_pb2.Vector3(x=extr.translation[0], y=extr.translation[1], z=extr.translation[2]),
            rotation=geometry_pb2.Quaternion(
                qw=extr.quaternion.w, qx=extr.quaternion.x, qy=extr.quaternion.y, qz=extr.quaternion.z
            ),
        )
        return calib_dto_extrinsic

    def encode_ontologies(self, pipeline_item: ScenePipelineItem) -> Dict[int, ontology_pb2.Ontology]:
        scene = pipeline_item.scene
        if scene is not None:
            annotations = pipeline_item.custom_data[CUSTOM_FORMAT_KEY][ANNOTATIONS_KEY]
            class_maps = pipeline_item.custom_data[CUSTOM_FORMAT_KEY][CLASS_MAPS_KEY]

            return self.encode_ontologie_map(annotations=annotations, scene=scene, class_maps=class_maps)
        raise ValueError("Pipeline Item does not contain a scene!")

    def encode_ontologie_map(
        self, annotations: Dict[str, AnyPath], class_maps: Dict[str, ClassMap], scene: Scene
    ) -> Dict[int, ontology_pb2.Ontology]:
        onthologie_map = dict()
        for annotation_id in annotations.keys():
            if annotation_id not in onthologie_map:
                iannotation_id = int(annotation_id)
                a_type = ANNOTATION_TYPE_MAP[iannotation_id]
                class_map = class_maps.get(annotation_id, None)
                if class_map is None:
                    class_map = scene.get_class_map(a_type)
                if class_map is not None:
                    ontology_proto = class_map_to_ontology_proto(class_map=class_map)
                    onthologie_map[iannotation_id] = ontology_proto
        return onthologie_map

    def get_scene_sensor_data(
        self,
        scene_name: str,
        scene_output_path: AnyPath,
    ) -> Tuple[
        Dict[SensorName, Dict[FrameId, sample_pb2.Datum]],
        sample_pb2.SampleCalibration,
        Dict[int, ontology_pb2.Ontology],
        Dict[FrameId, datetime],
    ]:
        sensor_data = dict()
        calibration_data = dict()
        ontology_maps = dict()
        frame_map = dict()

        (
            camera_sensor_data,
            camera_calibration_data,
            cam_ontology_maps,
            cam_frame_map,
        ) = self.encode_camera_sensor_data_and_calibration(scene_name=scene_name, scene_output_path=scene_output_path)
        sensor_data.update(camera_sensor_data)
        ontology_maps.update(cam_ontology_maps)
        frame_map.update(cam_frame_map)
        calibration_data.update(camera_calibration_data)

        (
            lidar_sensor_data,
            lidar_calibration_data,
            lidar_ontology_maps,
            lidar_frame_map,
        ) = self.encode_lidar_sensor_data_and_calibration(scene_name=scene_name, scene_output_path=scene_output_path)
        sensor_data.update(lidar_sensor_data)
        ontology_maps.update(lidar_ontology_maps)
        frame_map.update(lidar_frame_map)
        calibration_data.update(lidar_calibration_data)

        scene_sensor_data = dict()
        for sensor_name, scene_data_dtos in sensor_data.items():
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

        calibration_dto = self.encode_calibration(calibration_data=calibration_data)
        return scene_sensor_data, calibration_dto, ontology_maps, frame_map

    def encode_calibration(
        self, calibration_data: Dict[SensorName, Tuple[geometry_pb2.CameraIntrinsics, geometry_pb2.Pose]]
    ) -> sample_pb2.SampleCalibration:
        names, extrinsics, intrinsics = list(), list(), list()
        for name in sorted(calibration_data):
            intrinsic, extrinsic = calibration_data[name]
            names.append(name)
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
        calib_dto = sample_pb2.SampleCalibration(names=names, extrinsics=extrinsics, intrinsics=intrinsics)
        return calib_dto
