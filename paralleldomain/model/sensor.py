from __future__ import annotations

from abc import ABCMeta
from enum import Enum
from typing import Dict, Optional, List, cast, Callable, Union

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore


import numpy as np

from paralleldomain.model.annotation import BoundingBox3D, Annotation
from paralleldomain.model.transformation import Transformation
from paralleldomain.model.type_aliases import SensorName, FrameId, AnnotationIdentifier, AnnotationType


class Sensor:
    def __init__(self, sensor_name: SensorName, sensor_frame_factory: Callable[[FrameId, SensorName], SensorFrame]):
        self._sensor_frame_factory = sensor_frame_factory
        self._sensor_name = sensor_name
        self._sensor_frames = []

    @property
    def name(self) -> str:
        return self._sensor_name

    def get_frame(self, frame_id: FrameId) -> SensorFrame:
        return self._sensor_frame_factory(frame_id, self._sensor_name)


class SensorFrameLazyLoaderProtocol(Protocol):
    def load_extrinsic(self) -> SensorExtrinsic:
        pass

    def load_intrinsic(self) -> SensorIntrinsic:
        pass

    def load_sensor_pose(self) -> SensorPose:
        pass

    def load_point_cloud(self) -> Optional[PointCloudData]:
        pass

    def load_annotations(self, identifier: AnnotationIdentifier) -> List[Annotation]:
        pass

    def load_available_annotation_types(self) -> Dict[AnnotationType, AnnotationIdentifier]:
        pass


class SensorFrame:
    def __init__(
            self,
            sensor_name: SensorName,
            lazy_loader: SensorFrameLazyLoaderProtocol,
    ):
        self._lazy_loader = lazy_loader
        self._sensor_name = sensor_name

        self._pose: Optional[SensorPose] = None
        self._extrinsic: Optional[SensorExtrinsic] = None
        self._intrinsic: Optional[SensorIntrinsic] = None
        self._point_cloud: Optional[SensorData] = None
        self._available_annotation_types: Optional[Dict[AnnotationType, AnnotationIdentifier]] = None
        self._annotations: Dict[AnnotationType, List[Annotation]] = dict()

    @property
    def extrinsic(self) -> SensorExtrinsic:
        if self._extrinsic is None:
            self._extrinsic = self._lazy_loader.load_extrinsic()
        return self._extrinsic

    @property
    def intrinsic(self) -> SensorIntrinsic:
        if self._intrinsic is None:
            self._intrinsic = self._lazy_loader.load_intrinsic()
        return self._intrinsic

    @property
    def pose(self) -> SensorPose:
        if self._pose is None:
            self._pose = self._lazy_loader.load_sensor_pose()
        return self._pose

    @property
    def sensor_name(self) -> str:
        return self._sensor_name

    @property
    def point_cloud(self) -> Optional[PointCloudData]:
        if self._point_cloud is None:
            self._point_cloud = self._lazy_loader.load_point_cloud()
        return self._point_cloud

    @property
    def available_annotation_types(self) -> List[AnnotationType]:
        if self._available_annotation_types is None:
            self._available_annotation_types = self._lazy_loader.load_available_annotation_types()
        return list(self._available_annotation_types.keys())

    def get_annotations(self, annotation_type: AnnotationType) -> List[Annotation]:
        if annotation_type not in self.available_annotation_types:
            raise ValueError(f"The annotaiton type {annotation_type} is not available in this sensor frame!")
        if annotation_type not in self._annotations:
            identifier = self._available_annotation_types[annotation_type]
            self._annotations[annotation_type] = self._lazy_loader.load_annotations(identifier=identifier)
        return self._annotations[annotation_type]


class SensorPose(Transformation):
    ...


class SensorExtrinsic(Transformation):
    ...


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


class PointCloudData(SensorData):
    def __init__(self, point_format: List[str], load_data: Callable[[], np.ndarray]):
        self._load_data_call = load_data
        self._cloud_data: Optional[np.ndarray] = None
        self._point_cloud_info = {PointInfo(val): idx for idx, val in enumerate(point_format)}

    def _has(self, p_info: PointInfo):
        return p_info in self._point_cloud_info

    def _get_index(self, p_info: PointInfo):
        return self._point_cloud_info[p_info]

    def _load_data(self):
        if self._cloud_data is None:
            self._cloud_data = self._load_data_call()
        return self._cloud_data

    @property
    def xyz(self) -> np.ndarray:
        xyz_index = [
            self._get_index(PointInfo.X),
            self._get_index(PointInfo.Y),
            self._get_index(PointInfo.Z),
        ]

        return self._load_data()[:, xyz_index]

    @property
    def rgb(self) -> np.ndarray:
        rgb_index = [
            self._get_index(PointInfo.R),
            self._get_index(PointInfo.G),
            self._get_index(PointInfo.B),
        ]

        return self._load_data()[:, rgb_index]

    @property
    def intensity(self) -> np.ndarray:
        intensity_index = [
            self._get_index(PointInfo.I),
        ]

        return self._load_data()[:, intensity_index]

    @property
    def ts(self) -> np.ndarray:
        ts_index = [
            self._get_index(PointInfo.TS),
        ]

        return self._load_data()[:, ts_index]

    @property
    def ring(self) -> np.ndarray:
        ring_index = [
            self._get_index(PointInfo.RING),
        ]

        return self._load_data()[:, ring_index]

    @property
    def xyz_i(self) -> np.ndarray:
        return np.concatenate((self.xyz, self.intensity), axis=1)

    @property
    def xyz_one(self) -> np.ndarray:
        xyz_data = self.xyz
        one_data = np.ones((len(xyz_data), 1))
        return np.concatenate((xyz_data, one_data), axis=1)
