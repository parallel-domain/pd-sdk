from __future__ import annotations
from .utils import Transformation
from typing import Dict

# from .dto import CalibrationIntrinsicDTO, CalibrationExtrinsicDTO


class Sensor:
    def __init__(self, name):
        self._name = name
        self._sensor_frames = []

    @property
    def name(self):
        return self._name

    @property
    def frames(self):
        return self._sensor_frames

    def add_sensor_frame(self, sensor_frame: SensorFrame):
        self._sensor_frames.append(sensor_frame)


class SensorFrame:
    def __init__(self, sensor, filename, annotations=None, pose=None):
        self._sensor = sensor
        self._filename: str = filename
        self._pose: SensorPose = SensorPose() if not pose else pose
        self._annotations: Dict[str, Any] = {} if not annotations else annotations
        self._extrinsic: SensorExtrinsic = SensorExtrinsic()
        self._intrinsic: SensorIntrinsic = SensorIntrinsic()

    @property
    def sensor(self):
        return self._sensor

    @property
    def filename(self):
        return self._filename

    @property
    def extrinsic(self):
        return self._extrinsic

    @extrinsic.setter
    def extrinsic(self, value):
        self._extrinsic = value

    @property
    def intrinsic(self):
        return self._intrinsic

    @intrinsic.setter
    def intrinsic(self, value):
        self._intrinsic = value

    @property
    def pose(self):
        return self._pose

    @property
    def annotations(self):
        return self._annotations

    @staticmethod
    def from_SceneDataDatumDTO(sensor: Sensor, datum: SceneDataDatum):
        if datum.image:
            return SensorFrame(
                sensor,
                datum.image.filename,
                datum.image.annotations,
                SensorPose.from_PoseDTO(datum.image.pose),
            )
        elif datum.point_cloud:
            return SensorFrame(
                sensor,
                datum.point_cloud.filename,
                datum.point_cloud.annotations,
                SensorPose.from_PoseDTO(datum.point_cloud.pose),
            )


class SensorPose(Transformation):
    @staticmethod
    def from_PoseDTO(pose: PoseDTO):
        tf = Transformation.from_PoseDTO(pose)
        tf.__class__ = SensorPose
        return tf


class SensorExtrinsic(Transformation):
    @staticmethod
    def from_CalibrationExtrinsicDTO(extrinsic: CalibrationExtrinsicDTO):
        tf = Transformation.from_PoseDTO(extrinsic)
        tf.__class__ = SensorExtrinsic
        return tf


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
    def from_CalibrationIntrinsicDTO(intrinsic: CalibrationIntrinsicDTO):
        return SensorIntrinsic(
            cx=intrinsic.cx,
            cy=intrinsic.cy,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            k1=intrinsic.k1,
            k2=intrinsic.k2,
            p1=intrinsic.p1,
            p2=intrinsic.p2,
            k3=intrinsic.k3,
            k4=intrinsic.k4,
            k5=intrinsic.k5,
            k6=intrinsic.k6,
            skew=intrinsic.skew,
            fov=intrinsic.fov,
            fisheye=intrinsic.fisheye,
        )
