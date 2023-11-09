import abc
from typing import Optional, Protocol

import numpy as np


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
    @abc.abstractmethod
    def ray_type(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def elongation(self) -> Optional[np.ndarray]:
        pass

    @property
    @abc.abstractmethod
    def azimuth(self) -> Optional[np.ndarray]:
        pass

    @property
    @abc.abstractmethod
    def elevation(self) -> Optional[np.ndarray]:
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
    def get_point_cloud_size(self) -> int:
        pass

    def get_point_cloud_xyz(self) -> Optional[np.ndarray]:
        pass

    def get_point_cloud_rgb(self) -> Optional[np.ndarray]:
        pass

    def get_point_cloud_intensity(self) -> Optional[np.ndarray]:
        pass

    def get_point_cloud_elongation(self) -> Optional[np.ndarray]:
        pass

    def get_point_cloud_timestamp(self) -> Optional[np.ndarray]:
        pass

    def get_point_cloud_ring_index(self) -> Optional[np.ndarray]:
        pass

    def get_point_cloud_ray_type(self) -> Optional[np.ndarray]:
        pass

    def get_point_cloud_azimuth(self) -> Optional[np.ndarray]:
        pass

    def get_point_cloud_elevation(self) -> Optional[np.ndarray]:
        pass

class DecoderPointCloud(PointCloud):
    def __init__(self, decoder: PointCloudDecoderProtocol):
        self._decoder = decoder
        self._length = None
        self._xyz = None
        self._rgb = None
        self._intensity = None
        self._elongation = None
        self._ts = None
        self._ring = None
        self._ray_type = None
        self._azimuth = None
        self._elevation = None

    @property
    def length(self) -> int:
        if self._length is None:
            self._length = self._decoder.get_point_cloud_size()
        return self._length

    @property
    def xyz(self) -> Optional[np.ndarray]:
        if self._xyz is None:
            self._xyz = self._decoder.get_point_cloud_xyz()
        return self._xyz

    @property
    def rgb(self) -> Optional[np.ndarray]:
        if self._rgb is None:
            self._rgb = self._decoder.get_point_cloud_rgb()
        return self._rgb

    @property
    def intensity(self) -> Optional[np.ndarray]:
        if self._intensity is None:
            self._intensity = self._decoder.get_point_cloud_intensity()
        return self._intensity

    @property
    def elongation(self) -> Optional[np.ndarray]:
        """
        Elongation is recorded in the Waymo Open Dataset and refers to how spread across time the energy of the return
        lidar pulse is. Larger elongation values tend to indicate spurious objects such as fog and dust, as these
        objects will reflect some but not all of the laser.
        Description here: https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/waymo_dataset.html
        """
        if self._elongation is None:
            self._elongation = self._decoder.get_point_cloud_elongation()
        return self._elongation

    @property
    def ts(self) -> Optional[np.ndarray]:
        if self._ts is None:
            self._ts = self._decoder.get_point_cloud_timestamp()
        return self._ts

    @property
    def ring(self) -> Optional[np.ndarray]:
        if self._ring is None:
            self._ring = self._decoder.get_point_cloud_ring_index()
        return self._ring

    @property
    def ray_type(self) -> Optional[np.ndarray]:
        if self._ray_type is None:
            self._ray_type = self._decoder.get_point_cloud_ray_type()
        return self._ray_type

    @property
    def azimuth(self) -> Optional[np.ndarray]:
        if self._azimuth is None:
            self._azimuth = self._decoder.get_point_cloud_azimuth()
        return self._azimuth

    @property
    def elevation(self) -> Optional[np.ndarray]:
        if self._elevation is None:
            self._elevation = self._decoder.get_point_cloud_elevation()
        return self._elevation