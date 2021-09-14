import abc

import numpy as np

from paralleldomain.model.type_aliases import FrameId, SensorName

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore


class PointCloud:
    @property
    @abc.abstractmethod
    def length(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def xyz(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def rgb(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def intensity(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def ts(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def ring(self) -> np.ndarray:
        pass

    @property
    def xyz_i(self) -> np.ndarray:
        return np.concatenate((self.xyz, self.intensity), axis=1)

    @property
    def xyz_one(self) -> np.ndarray:
        xyz_data = self.xyz
        one_data = np.ones((len(xyz_data), 1))
        return np.concatenate((xyz_data, one_data), axis=1)


class PointCloudDecoderProtocol(Protocol):
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


class DecoderPointCloud(PointCloud):
    def __init__(self, decoder: PointCloudDecoderProtocol, sensor_name: SensorName, frame_id: FrameId):
        self.frame_id = frame_id
        self.sensor_name = sensor_name
        self._decoder = decoder

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
