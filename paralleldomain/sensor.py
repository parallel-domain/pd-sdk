from __future__ import annotations

from abc import ABCMeta
from enum import Enum
from typing import Dict, Optional, List, cast, Callable, Union

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
from paralleldomain.transformation import Transformation


class Sensor:
    def __init__(self, sensor_name: str):
        self._sensor_name = sensor_name
        self._sensor_frames = []

    @property
    def name(self) -> str:
        return self._sensor_name

    @property
    def frames(self) -> List[SensorFrame]:
        return self._sensor_frames

    def add_sensor_frame(self, sensor_frame: SensorFrame):
        self._sensor_frames.append(sensor_frame)


class SensorAnnotations:
    def __init__(self, available_annotation_type_id_to_identifier: Dict[Union[int, str], str],
                 annotation_loader: Callable[[str], List[Annotation]]):
        self._annotation_loader = annotation_loader
        self._available_annotation_type_id_to_identifier = {int(k): v for k, v in
                                                            available_annotation_type_id_to_identifier.items()}

    @property
    def available_annotation_types(self) -> Dict[int, str]:
        return {key: a.split("/")[0] for key, a in self._available_annotation_type_id_to_identifier.items()}

    def __getitem__(self, key: int) -> List[Annotation]:
        # TODO enum for key
        return self._annotation_loader(self._available_annotation_type_id_to_identifier[key])
        # if key == 1:
        #     with open(f"{self._scene_path}/{self._annotation_type_id_to_file_name[key]}", "r") as f:
        #         annotations_dto = AnnotationsBoundingBox3DDTO.from_dict(json.load(f))
        #     return list(map(BoundingBox3D.from_dto, annotations_dto.annotations))
        # else:
        #     return []


class SensorFrame:
    def __init__(
            self,
            sensor: Sensor,
            file_name: str,
            extrinsic_loader: Callable[[], SensorExtrinsic],
            intrinsic_loader: Callable[[], SensorIntrinsic],
            annotations_loader: Callable[[], SensorAnnotations],
            pose_loader: Callable[[], SensorPose],
            data_loader: Callable[[], SensorData],
    ):
        self._data_loader = data_loader
        self._pose_loader = pose_loader
        self._annotations_loader = annotations_loader
        self._intrinsic_loader = intrinsic_loader
        self._extrinsic_loader = extrinsic_loader
        self._sensor = sensor
        self._file_name: str = file_name

        self._pose: Optional[SensorPose] = None
        self._annotations: Optional[SensorAnnotations] = None
        self._extrinsic: Optional[SensorExtrinsic] = None
        self._intrinsic: Optional[SensorIntrinsic] = None
        self._data: Optional[SensorData] = None

    @property
    def extrinsic(self) -> SensorExtrinsic:
        if self._extrinsic is None:
            self._extrinsic = self._extrinsic_loader()
        return self._extrinsic

    @property
    def intrinsic(self) -> SensorIntrinsic:
        if self._intrinsic is None:
            self._intrinsic = self._intrinsic_loader()
        return self._intrinsic

    @property
    def pose(self) -> SensorPose:
        if self._pose is None:
            self._pose = self._pose_loader()
        return self._pose

    @property
    def annotations(self) -> SensorAnnotations:
        if self._annotations is None:
            self._annotations = self._annotations_loader()
        return self._annotations

    @property
    def sensor_name(self) -> str:
        return self._sensor_name

    @property
    def data(self) -> SensorData:
        if self._data is None:
            self._data = self._data_loader()
        return self._data

    @property
    def sensor(self) -> Sensor:
        return self._sensor

    @property
    def _sensor_name(self):
        return self._sensor.name


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
    def __init__(self, point_format: List[str], load_data: Callable[[], np.ndarray]):
        self._load_data_call = load_data
        self._cloud_data: Optional[np.ndarray] = None
        self._point_cloud_info = {val: idx for idx, val in enumerate(point_format)}

    def _has(self, p_info: PointInfo):
        return p_info in self._point_cloud_info

    def _get_index(self, p_info: PointInfo):
        return self._point_cloud_info[p_info]

    def _load_data(self):
        if self._cloud_data is None:
            self._cloud_data = self._load_data_call()
        return self._cloud_data

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
