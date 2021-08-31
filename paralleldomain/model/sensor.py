from abc import ABCMeta
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Type, TypeVar

from paralleldomain.utilities.lazy_load_cache import LAZY_LOAD_CACHE

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

import numpy as np

from paralleldomain.model.annotation import AnnotationType
from paralleldomain.model.transformation import Transformation
from paralleldomain.model.type_aliases import AnnotationIdentifier, FrameId, SensorName

T = TypeVar("T")


class CameraModel:
    OPENCV_PINHOLE: str = "opencv_pinhole"
    OPENCV_FISHEYE: str = "opencv_fisheye"


class SensorFrameDecoderProtocol(Protocol):
    def get_extrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> "SensorExtrinsic":
        pass

    def get_intrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> "SensorIntrinsic":
        pass

    def get_sensor_pose(self, sensor_name: SensorName, frame_id: FrameId) -> "SensorPose":
        pass

    def get_point_cloud(self, sensor_name: SensorName, frame_id: FrameId) -> Optional["PointCloudData"]:
        pass

    def get_image(self, sensor_name: SensorName, frame_id: FrameId) -> Optional["ImageData"]:
        pass

    def get_annotations(
        self, sensor_name: SensorName, frame_id: FrameId, identifier: AnnotationIdentifier, annotation_type: T
    ) -> List[T]:
        pass

    def get_available_annotation_types(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> Dict[AnnotationType, AnnotationIdentifier]:
        pass


class SensorFrame:
    def __init__(
        self,
        sensor_name: SensorName,
        frame_id: FrameId,
        decoder: SensorFrameDecoderProtocol,
    ):
        self._frame_id = frame_id
        self._decoder = decoder
        self._sensor_name = sensor_name

    @property
    def extrinsic(self) -> "SensorExtrinsic":
        return self._decoder.get_extrinsic(sensor_name=self.sensor_name, frame_id=self.frame_id)

    @property
    def intrinsic(self) -> "SensorIntrinsic":
        return self._decoder.get_intrinsic(sensor_name=self.sensor_name, frame_id=self.frame_id)

    @property
    def pose(self) -> "SensorPose":
        return self._decoder.get_sensor_pose(sensor_name=self.sensor_name, frame_id=self.frame_id)

    @property
    def sensor_name(self) -> str:
        return self._sensor_name

    @property
    def frame_id(self) -> FrameId:
        return self._frame_id

    @property
    def point_cloud(self) -> Optional["PointCloudData"]:
        return self._decoder.get_point_cloud(sensor_name=self.sensor_name, frame_id=self.frame_id)

    @property
    def image(self) -> Optional["ImageData"]:
        return self._decoder.get_image(sensor_name=self.sensor_name, frame_id=self.frame_id)

    @property
    def available_annotation_types(self) -> List[AnnotationType]:
        return list(self._annotation_type_identifiers.keys())

    @property
    def _annotation_type_identifiers(self) -> Dict[AnnotationType, AnnotationIdentifier]:
        return self._decoder.get_available_annotation_types(sensor_name=self.sensor_name, frame_id=self.frame_id)

    def get_annotations(self, annotation_type: Type[T]) -> T:
        if annotation_type not in self._annotation_type_identifiers:
            raise ValueError(f"The annotation type {annotation_type} is not available in this sensor frame!")
        return self._decoder.get_annotations(
            sensor_name=self.sensor_name,
            frame_id=self.frame_id,
            identifier=self._annotation_type_identifiers[annotation_type],
            annotation_type=annotation_type,
        )


class TemporalSensorFrameDecoderProtocol(SensorFrameDecoderProtocol, Protocol):
    def get_datetime(self, frame_id: FrameId) -> datetime:
        pass


class TemporalSensorFrame(SensorFrame):
    def __init__(
        self,
        sensor_name: SensorName,
        frame_id: FrameId,
        decoder: TemporalSensorFrameDecoderProtocol,
    ):
        super().__init__(sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)
        self._decoder = decoder

    @property
    def date_time(self) -> datetime:
        return self._decoder.get_datetime(frame_id=self.frame_id)


TSensorFrameType = TypeVar("TSensorFrameType", bound=SensorFrame)


class SensorDecoderProtocol(Protocol[TSensorFrameType]):
    def get_sensor_frame(self, frame_id: FrameId, sensor_name: SensorName) -> TSensorFrameType:
        pass

    def get_frame_ids(self, sensor_name: SensorName) -> Set[FrameId]:
        pass


class Sensor(Generic[TSensorFrameType]):
    def __init__(
        self,
        sensor_name: SensorName,
        decoder: SensorDecoderProtocol,
    ):
        self._decoder = decoder
        self._sensor_name = sensor_name

    @property
    def name(self) -> str:
        return self._sensor_name

    @property
    def frame_ids(self) -> Set[FrameId]:
        return self._decoder.get_frame_ids(sensor_name=self.name)

    def get_frame(self, frame_id: FrameId) -> TSensorFrameType:
        return self._decoder.get_sensor_frame(frame_id=frame_id, sensor_name=self._sensor_name)


class CameraSensor(Sensor):
    ...


class LidarSensor(Sensor):
    ...


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
        camera_model=CameraModel.OPENCV_PINHOLE,
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
        self.camera_model = camera_model


class SensorData(metaclass=ABCMeta):
    ...


class PointInfo(Enum):
    X = "X"
    Y = "Y"
    Z = "Z"
    I = "INTENSITY"  # noqa: E741
    R = "R"
    G = "G"
    B = "B"
    RING = "RING"
    TS = "TIMESTAMP"


class ImageData(SensorData):
    def __init__(
        self,
        load_data_rgba: Callable[[], np.ndarray],
        load_image_dims: Callable[[], Tuple[int, int, int]],
    ):
        self._load_image_dims = load_image_dims
        self._load_data_rgb_call = load_data_rgba

    @property
    def _data_rgba(self) -> np.ndarray:
        return self._load_data_rgb_call()

    @property
    def _image_dims(self) -> Tuple[int, int, int]:
        return self._load_image_dims()

    @property
    def rgba(self) -> np.ndarray:
        return self._data_rgba

    @property
    def rgb(self) -> np.ndarray:
        return self._data_rgba[:, :, :3]

    @property
    def width(self) -> int:
        return self._image_dims[1]

    @property
    def height(self) -> int:
        return self._image_dims[0]

    @property
    def channels(self) -> int:
        return self._image_dims[2]

    @property
    def coordinates(self) -> np.ndarray:
        shape = self._data_rgba.shape
        y_coords, x_coords = np.meshgrid(range(shape[0]), range(shape[1]), indexing="ij")
        return np.stack([y_coords, x_coords], axis=-1)


class PointCloudData(SensorData):
    def __init__(self, point_format: List[str], load_data: Callable[[], np.ndarray]):
        self._load_data_call = load_data
        self._point_cloud_info = {PointInfo(val): idx for idx, val in enumerate(point_format)}

    def _has(self, p_info: PointInfo):
        return p_info in self._point_cloud_info

    def _get_index(self, p_info: PointInfo):
        return self._point_cloud_info[p_info]

    @property
    def _data(self) -> np.ndarray:
        return self._load_data_call()

    @property
    def length(self) -> int:
        return len(self._data)

    @property
    def xyz(self) -> np.ndarray:
        xyz_index = [
            self._get_index(PointInfo.X),
            self._get_index(PointInfo.Y),
            self._get_index(PointInfo.Z),
        ]

        return self._data[:, xyz_index]

    @property
    def rgb(self) -> np.ndarray:
        rgb_index = [
            self._get_index(PointInfo.R),
            self._get_index(PointInfo.G),
            self._get_index(PointInfo.B),
        ]

        return self._data[:, rgb_index]

    @property
    def intensity(self) -> np.ndarray:
        intensity_index = [
            self._get_index(PointInfo.I),
        ]

        return self._data[:, intensity_index]

    @property
    def ts(self) -> np.ndarray:
        ts_index = [
            self._get_index(PointInfo.TS),
        ]

        return self._data[:, ts_index]

    @property
    def ring(self) -> np.ndarray:
        ring_index = [
            self._get_index(PointInfo.RING),
        ]

        return self._data[:, ring_index]

    @property
    def xyz_i(self) -> np.ndarray:
        return np.concatenate((self.xyz, self.intensity), axis=1)

    @property
    def xyz_one(self) -> np.ndarray:
        xyz_data = self.xyz
        one_data = np.ones((len(xyz_data), 1))
        return np.concatenate((xyz_data, one_data), axis=1)
