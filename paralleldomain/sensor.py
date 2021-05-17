from __future__ import annotations

from abc import ABCMeta
from enum import Enum
from typing import Dict, Optional, List, cast

import numpy as np
import ujson as json

from paralleldomain.annotation import BoundingBox3D, Annotation
from paralleldomain.dto import (
    AnnotationsBoundingBox3DDTO,
    CalibrationIntrinsicDTO,
    PoseDTO,
    CalibrationExtrinsicDTO,
    SceneDataDatum,
    SceneDataDatumTypePointCloud,
)
from paralleldomain.utils import Transformation


class Sensor:
    def __init__(self, scene_path: str, sensor_name: str):
        self._scene_path = scene_path
        self._sensor_name = sensor_name
        self._sensor_frames = []

    @property
    def name(self) -> str:
        return self._sensor_name

    @property
    def frames(self) -> List[SensorFrame]:
        return self._sensor_frames

    @property
    def _path(self) -> str:
        return self._scene_path

    def add_sensor_frame(self, sensor_frame: SensorFrame):
        self._sensor_frames.append(sensor_frame)


class SensorAnnotations:
    def __init__(
        self,
        scene_path: str,
        data: Dict[int, str],
    ):
        self._scene_path = scene_path
        self._data = data

    @property
    def available_annotation_types(self) -> Dict[int, str]:
        return {key: a.split("/")[0] for key, a in self._data.items()}

    def __getitem__(self, key) -> List[Annotation]:
        # TODO enum for key
        if key == 1:
            with open(f"{self._scene_path}/{self._data[key]}", "r") as f:
                annotations_dto = AnnotationsBoundingBox3DDTO.from_dict(json.load(f))
            return list(map(BoundingBox3D.from_dto, annotations_dto.annotations))
        else:
            return []


class SensorFrame:
    def __init__(
        self,
        sensor: Sensor,
        file_name: str,
        extrinsic: SensorExtrinsic,
        intrinsic: SensorIntrinsic,
        annotations: Optional[Dict[str, str]] = None,
        pose: Optional[SensorPose] = None,
        data: Optional[SensorData] = None,
    ):
        self._sensor = sensor
        self._file_name: str = file_name
        self._pose: SensorPose = SensorPose() if not pose else pose
        self._annotations: SensorAnnotations = (
            SensorAnnotations(scene_path=self._scene_path, data={})
            if not annotations
            else SensorAnnotations(
                scene_path=self._scene_path,
                data={int(k): v for k, v in annotations.items()},
            )
        )
        self._extrinsic: SensorExtrinsic = extrinsic
        self._intrinsic: SensorIntrinsic = intrinsic
        self._data = data

    @property
    def extrinsic(self) -> SensorExtrinsic:
        return self._extrinsic

    @property
    def intrinsic(self) -> SensorIntrinsic:
        return self._intrinsic

    @property
    def pose(self) -> SensorPose:
        return self._pose

    @property
    def annotations(self) -> SensorAnnotations:
        return self._annotations

    @property
    def sensor_name(self) -> str:
        return self._sensor_name

    @property
    def data(self) -> SensorData:
        return self._data

    @property
    def sensor(self) -> Sensor:
        return self._sensor

    @property
    def _scene_path(self):
        return self._sensor._scene_path

    @property
    def _sensor_name(self):
        return self._sensor.name

    @staticmethod
    def from_dto(
        sensor: Sensor,
        datum: SceneDataDatum,
        extrinsic: SensorExtrinsic,
        intrinsic: SensorIntrinsic,
    ) -> "SensorFrame":
        if datum.image:
            return SensorFrame(
                sensor=sensor,
                file_name=datum.image.filename,
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                annotations=datum.image.annotations,
                pose=SensorPose.from_dto(dto=datum.image.pose),
                data=None,
            )
        elif datum.point_cloud:
            return SensorFrame(
                sensor=sensor,
                file_name=datum.point_cloud.filename,
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                annotations=datum.point_cloud.annotations,
                pose=SensorPose.from_dto(dto=datum.point_cloud.pose),
                data=None,
            )


"""LidarData.from_dto(
    dto=datum.point_cloud, scene_path=sensor._scene_path
),"""


class SensorPose(Transformation):
    @staticmethod
    def from_dto(dto: PoseDTO) -> "SensorPose":
        tf = Transformation.from_dto(dto=dto)
        return cast(tf, SensorPose)


class SensorExtrinsic(Transformation):
    @staticmethod
    def from_dto(dto: CalibrationExtrinsicDTO) -> "SensorExtrinsic":
        tf = Transformation.from_dto(dto=dto)
        return cast(tf, SensorExtrinsic)


class SensorIntrinsic:
    def __init__(
        self,
        cx=0.0,
        cy=0.0,
        fx=0.0,
        fy=0.0,
        k1=0.0,
        k2=0.0,
        p1=0.0,
        p2=0.0,
        k3=0.0,
        k4=0.0,
        k5=0.0,
        k6=0.0,
        skew=0.0,
        fov=0.0,
        fisheye=False,
    ):
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.k3 = k3
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6
        self.skew = skew
        self.fov = fov
        self.fisheye = fisheye

    @staticmethod
    def from_dto(dto: CalibrationIntrinsicDTO) -> "SensorIntrinsic":
        return SensorIntrinsic(
            cx=dto.cx,
            cy=dto.cy,
            fx=dto.fx,
            fy=dto.fy,
            k1=dto.k1,
            k2=dto.k2,
            p1=dto.p1,
            p2=dto.p2,
            k3=dto.k3,
            k4=dto.k4,
            k5=dto.k5,
            k6=dto.k6,
            skew=dto.skew,
            fov=dto.fov,
            fisheye=dto.fisheye,
        )


class SensorData(metaclass=ABCMeta):
    ...


class PointInfo(Enum):
    X = "X"
    Y = "Y"
    Z = "Z"
    I = "INTENSITY"
    R = "R"
    G = "G"
    B = "B"
    RING = "RING"
    TS = "TIMESTAMP"


class LidarData(SensorData):
    def __init__(self, filename: str, point_format: List[str], scene_path: str):
        self._filename = filename
        self._point_cloud_info = {val: idx for idx, val in enumerate(point_format)}
        self._scene_path = scene_path

    def _has(self, p_info: PointInfo):
        return p_info in self._point_cloud_info

    def _get_index(self, p_info: PointInfo):
        return self._point_cloud_info[p_info]

    def _load_data(self):
        npz_data = np.load(f"{self._scene_path}/{self._filename}")
        column_count = len(self._point_cloud_info)
        return np.array([f.tolist() for f in npz_data.f.data]).reshape(-1, column_count)

    @property
    def xyz(self):
        xyz_index = [
            self._get_index(PointInfo.X),
            self._get_index(PointInfo.Y),
            self._get_index(PointInfo.Z),
        ]

        return self._load_data()[:, xyz_index]

    @property
    def rgb(self):
        rgb_index = [
            self._get_index(PointInfo.R),
            self._get_index(PointInfo.G),
            self._get_index(PointInfo.B),
        ]

        return self._load_data()[:, rgb_index]

    @property
    def intensity(self):
        intensity_index = [
            self._get_index(PointInfo.I),
        ]

        return self._load_data()[:, intensity_index]

    @property
    def ts(self):
        ts_index = [
            self._get_index(PointInfo.TS),
        ]

        return self._load_data()[:, ts_index]

    @property
    def ring(self):
        ring_index = [
            self._get_index(PointInfo.RING),
        ]

        return self._load_data()[:, ring_index]

    @property
    def xyzi(self):
        return np.concatenate((self.xyz, self.intensity), axis=1)

    @property
    def xyzone(self):
        xyz_data = self.xyz
        one_data = np.ones((len(xyz_data), 1))
        return np.concatenate((xyz_data, one_data), axis=1)

    @staticmethod
    def from_dto(dto: SceneDataDatumTypePointCloud, scene_path: str) -> "LidarData":
        return LidarData(dto.filename, dto.point_format, scene_path)
