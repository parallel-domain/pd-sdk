from __future__ import annotations
from typing import Dict, Optional, List, cast
import ujson as json

from paralleldomain.utils import Transformation
from paralleldomain.annotation import BoundingBox3D, Annotation
from paralleldomain.dto import AnnotationsBoundingBox3DDTO, CalibrationIntrinsicDTO, PoseDTO, CalibrationExtrinsicDTO, \
    SceneDataDatum


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
    def __init__(self, scene_path: str, data: Dict[int, str], ):
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
    def __init__(self, scene_path: str, sensor_name: str, file_name: str,
                 extrinsic: SensorExtrinsic, intrinsic: SensorIntrinsic,
                 annotations: Optional[Dict[str, str]] = None,
                 pose: Optional[SensorPose] = None):
        self._scene_path = scene_path
        self._sensor_name = sensor_name
        self._file_name: str = file_name
        self._pose: SensorPose = SensorPose() if not pose else pose
        self._annotations: SensorAnnotations = (
            SensorAnnotations(scene_path=self._scene_path, data={})
            if not annotations
            else SensorAnnotations(scene_path=self._scene_path, data={int(k): v for k, v in annotations.items()})
        )
        self._extrinsic: SensorExtrinsic = extrinsic
        self._intrinsic: SensorIntrinsic = intrinsic

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

    @staticmethod
    def from_dto(scene_path: str, sensor_name: str, datum: SceneDataDatum,
                 extrinsic: SensorExtrinsic, intrinsic: SensorIntrinsic) -> "SensorFrame":
        if datum.image:
            return SensorFrame(
                scene_path=scene_path,
                sensor_name=sensor_name,
                file_name=datum.image.filename,
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                annotations=datum.image.annotations,
                pose=SensorPose.from_dto(dto=datum.image.pose),
            )
        elif datum.point_cloud:
            return SensorFrame(
                scene_path=scene_path,
                sensor_name=sensor_name,
                file_name=datum.point_cloud.filename,
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                annotations=datum.point_cloud.annotations,
                pose=SensorPose.from_dto(dto=datum.point_cloud.pose),
            )


class SensorPose(Transformation):
    @staticmethod
    def from_dto(dto: PoseDTO) -> "SensorPose":
        tf = Transformation.from_dto(dto=dto)
        # tf.__class__ = SensorPose
        return cast(tf, SensorPose)


class SensorExtrinsic(Transformation):
    @staticmethod
    def from_dto(dto: CalibrationExtrinsicDTO) -> "SensorExtrinsic":
        tf = Transformation.from_dto(dto=dto)
        # tf.__class__ = SensorExtrinsic
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
