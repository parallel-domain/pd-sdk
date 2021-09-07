from abc import ABCMeta
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Type, TypeVar, Union

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
TDateTime = TypeVar("TDateTime", bound=Union[None, datetime])


class CameraModel:
    """Camera Model short hands for value-safe access.

    Attributes:
        OPENCV_PINHOLE: Returns internally used string-representation for OpenCV Pinhole camera model.

            Accepts distortion parameters `(k1,k2,p1,p2[,k3[,k4,k5,k6]])` and uses projection (+ distortion) function
            as described in the
            `OpenCV Pinhole documentation <https://docs.opencv.org/4.5.3/d9/d0c/group__calib3d.html>`_
        OPENCV_FISHEYE: Returns internally used string-representation for OpenCV Fisheye camera model

            Accepts distortion parameters `(k1,k2,k3,k4)` and uses projection (+ distortion) function
            as described in the
            `OpenCV Fisheye documentation <https://docs.opencv.org/4.5.3/db/d58/group__calib3d__fisheye.html>`_
    """

    OPENCV_PINHOLE: str = "opencv_pinhole"
    OPENCV_FISHEYE: str = "opencv_fisheye"


class SensorFrameDecoderProtocol(Protocol[TDateTime]):
    def get_extrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> "SensorExtrinsic":
        pass

    def get_sensor_pose(self, sensor_name: SensorName, frame_id: FrameId) -> "SensorPose":
        pass

    def get_annotations(
        self, sensor_name: SensorName, frame_id: FrameId, identifier: AnnotationIdentifier, annotation_type: T
    ) -> List[T]:
        pass

    def get_available_annotation_types(
        self, sensor_name: SensorName, frame_id: FrameId
    ) -> Dict[AnnotationType, AnnotationIdentifier]:
        pass

    def get_date_time(self, sensor_name: SensorName, frame_id: FrameId) -> TDateTime:
        pass


class SensorFrame(Generic[TDateTime]):
    def __init__(
        self,
        sensor_name: SensorName,
        frame_id: FrameId,
        decoder: SensorFrameDecoderProtocol[TDateTime],
    ):
        self._frame_id = frame_id
        self._decoder = decoder
        self._sensor_name = sensor_name

    @property
    def extrinsic(self) -> "SensorExtrinsic":
        return self._decoder.get_extrinsic(sensor_name=self.sensor_name, frame_id=self.frame_id)

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

    @property
    def point_cloud(self) -> Optional["SensorFrame"]:
        """
        Deprecated. Remains atm for 0.2.0 compatibility
        """
        return None

    @property
    def image(self) -> Optional["SensorFrame"]:
        """
        Deprecated. Remains atm for 0.2.0 compatibility
        """
        return None

    @property
    def date_time(self) -> TDateTime:
        return self._decoder.get_date_time(sensor_name=self.sensor_name, frame_id=self.frame_id)

    def __lt__(self, other: "SensorFrame[TDateTime]"):
        if self.date_time is not None and other.date_time is not None:
            # if isinstance(other, type(self)):
            return self.date_time < other.date_time
        return id(self) < id(other)


class LidarSensorFrameDecoderProtocol(SensorFrameDecoderProtocol[TDateTime]):
    def get_point_cloud_size(self, sensor_name: SensorName, frame_id: FrameId) -> int:
        pass

    def get_point_cloud_xyz(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        pass

    def get_point_cloud_rgb(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        pass

    def get_point_cloud_intensity(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        pass

    def get_point_cloud_timestamp(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        pass

    def get_point_cloud_ring_index(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        pass


class LidarSensorFrame(SensorFrame[TDateTime]):
    def __init__(
        self,
        sensor_name: SensorName,
        frame_id: FrameId,
        decoder: LidarSensorFrameDecoderProtocol[TDateTime],
    ):
        super().__init__(sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)
        self._decoder = decoder

    @property
    def point_cloud(self) -> Optional["LidarSensorFrame"]:
        """
        Deprecated. Remains atm for 0.2.0 compatibility
        """
        return self

    @property
    def length(self) -> int:
        return self._decoder.get_point_cloud_size(sensor_name=self.sensor_name, frame_id=self.frame_id)

    @property
    def xyz(self) -> np.ndarray:
        return self._decoder.get_point_cloud_xyz(sensor_name=self.sensor_name, frame_id=self.frame_id)

    @property
    def rgb(self) -> np.ndarray:
        return self._decoder.get_point_cloud_rgb(sensor_name=self.sensor_name, frame_id=self.frame_id)

    @property
    def intensity(self) -> np.ndarray:
        return self._decoder.get_point_cloud_intensity(sensor_name=self.sensor_name, frame_id=self.frame_id)

    @property
    def ts(self) -> np.ndarray:
        return self._decoder.get_point_cloud_timestamp(sensor_name=self.sensor_name, frame_id=self.frame_id)

    @property
    def ring(self) -> np.ndarray:
        return self._decoder.get_point_cloud_ring_index(sensor_name=self.sensor_name, frame_id=self.frame_id)

    @property
    def xyz_i(self) -> np.ndarray:
        return np.concatenate((self.xyz, self.intensity), axis=1)

    @property
    def xyz_one(self) -> np.ndarray:
        xyz_data = self.xyz
        one_data = np.ones((len(xyz_data), 1))
        return np.concatenate((xyz_data, one_data), axis=1)


class CameraSensorFrameDecoderProtocol(SensorFrameDecoderProtocol[TDateTime]):
    def get_intrinsic(self, sensor_name: SensorName, frame_id: FrameId) -> "SensorIntrinsic":
        pass

    def get_image_dimensions(self, sensor_name: SensorName, frame_id: FrameId) -> Tuple[int, int, int]:
        pass

    def get_image_rgba(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        pass


class CameraSensorFrame(SensorFrame[TDateTime]):
    def __init__(
        self,
        sensor_name: SensorName,
        frame_id: FrameId,
        decoder: CameraSensorFrameDecoderProtocol[TDateTime],
    ):
        super().__init__(sensor_name=sensor_name, frame_id=frame_id, decoder=decoder)
        self._decoder = decoder

    @property
    def image(self) -> Optional["CameraSensorFrame"]:
        """
        Deprecated. Remains atm for 0.2.0 compatibility
        """
        return self

    @property
    def intrinsic(self) -> "SensorIntrinsic":
        return self._decoder.get_intrinsic(sensor_name=self.sensor_name, frame_id=self.frame_id)

    @property
    def _data_rgba(self) -> np.ndarray:
        return self._decoder.get_image_rgba(sensor_name=self.sensor_name, frame_id=self.frame_id)

    @property
    def _image_dims(self) -> Tuple[int, int, int]:
        return self._decoder.get_image_dimensions(sensor_name=self.sensor_name, frame_id=self.frame_id)

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
        decoder: SensorDecoderProtocol[TSensorFrameType],
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


class CameraSensor(Sensor[CameraSensorFrame[TDateTime]]):
    ...


class LidarSensor(Sensor[LidarSensorFrame[TDateTime]]):
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
