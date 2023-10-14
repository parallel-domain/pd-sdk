import abc
from typing import Optional, Protocol

import numpy as np


class RadarPointCloud:
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
    def rgb(self) -> Optional[np.ndarray]:
        pass

    @property
    @abc.abstractmethod
    def power(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def rcs(self) -> Optional[np.ndarray]:
        pass

    @property
    @abc.abstractmethod
    def range(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def azimuth(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def elevation(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def ts(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def doppler(self) -> np.ndarray:
        pass

    @property
    def xyz_i(self) -> np.ndarray:
        return np.concatenate((self.xyz, self.power), axis=1)

    @property
    def xyz_one(self) -> np.ndarray:
        xyz_data = self.xyz
        one_data = np.ones((len(xyz_data), 1))
        return np.concatenate((xyz_data, one_data), axis=1)


class RangeDopplerMap:
    @property
    @abc.abstractmethod
    def energy_map(self) -> np.ndarray:
        pass


class RadarPointCloudDecoderProtocol(Protocol):
    def get_radar_point_cloud_size(self) -> int:
        pass

    def get_radar_point_cloud_xyz(self) -> Optional[np.ndarray]:
        pass

    def get_radar_point_cloud_doppler(self) -> Optional[np.ndarray]:
        pass

    def get_radar_point_cloud_rgb(self) -> Optional[np.ndarray]:
        pass

    def get_radar_point_cloud_power(self) -> Optional[np.ndarray]:
        pass

    def get_radar_point_cloud_rcs(self) -> Optional[np.ndarray]:
        pass

    def get_radar_point_cloud_range(self) -> Optional[np.ndarray]:
        pass

    def get_radar_point_cloud_azimuth(self) -> Optional[np.ndarray]:
        pass

    def get_radar_point_cloud_elevation(self) -> Optional[np.ndarray]:
        pass

    def get_radar_point_cloud_timestamp(self) -> Optional[np.ndarray]:
        pass


class RadarRangeDopplerMapDecoderProtocol(Protocol):
    def get_range_doppler_energy_map(self) -> np.ndarray:
        pass


class RadarFrameHeaderDecoderProtocol(Protocol):
    def get_max_non_ambiguous_doppler(self) -> float:
        pass

    def get_timestamp(self) -> int:
        pass

    def get_radar_frame_header_data(self) -> np.ndarray:
        pass


class DecoderRadarPointCloud(RadarPointCloud):
    def __init__(self, decoder: RadarPointCloudDecoderProtocol):
        self._decoder = decoder
        self._length = None
        self._xyz = None
        self._rgb = None
        self._doppler = None
        self._power = None
        self._rcs = None
        self._range = None
        self._azimuth = None
        self._elevation = None
        self._ts = None

    @property
    def length(self) -> int:
        if self._length is None:
            self._length = self._decoder.get_radar_point_cloud_size()
        return self._length

    @property
    def xyz(self) -> Optional[np.ndarray]:
        if self._xyz is None:
            self._xyz = self._decoder.get_radar_point_cloud_xyz()
        return self._xyz

    @property
    def rgb(self) -> Optional[np.ndarray]:
        if self._rgb is None:
            self._rgb = self._decoder.get_radar_point_cloud_rgb()
        return self._rgb

    @property
    def power(self) -> Optional[np.ndarray]:
        if self._power is None:
            self._power = self._decoder.get_radar_point_cloud_power()
        return self._power

    @property
    def rcs(self) -> Optional[np.ndarray]:
        if self._rcs is None:
            self._rcs = self._decoder.get_radar_point_cloud_rcs()
        return self._rcs

    @property
    def range(self) -> Optional[np.ndarray]:
        if self._range is None:
            self._range = self._decoder.get_radar_point_cloud_range()
        return self._range

    @property
    def azimuth(self) -> Optional[np.ndarray]:
        if self._azimuth is None:
            self._azimuth = self._decoder.get_radar_point_cloud_azimuth()
        return self._azimuth

    @property
    def elevation(self) -> Optional[np.ndarray]:
        if self._elevation is None:
            self._elevation = self._decoder.get_radar_point_cloud_elevation()
        return self._elevation

    @property
    def ts(self) -> Optional[np.ndarray]:
        if self._ts is None:
            self._ts = self._decoder.get_radar_point_cloud_timestamp()
        return self._ts

    @property
    def doppler(self) -> Optional[np.ndarray]:
        if self._doppler is None:
            self._doppler = self._decoder.get_radar_point_cloud_doppler()
        return self._doppler


class DecoderRangeDopplerMap(RangeDopplerMap):
    def __init__(self, decoder: RadarRangeDopplerMapDecoderProtocol):
        self._decoder = decoder
        self._energy_map = None

    @property
    def energy_map(self) -> np.ndarray:
        if self._energy_map is None:
            self._energy_map = self._decoder.get_range_doppler_energy_map()
        return self._energy_map


class RadarFrameHeader:
    @property
    @abc.abstractmethod
    def max_non_ambiguous_doppler(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def timestamp(self) -> int:
        pass


class DecoderRadarFrameHeader(RadarFrameHeader):
    def __init__(self, decoder: RadarFrameHeaderDecoderProtocol):
        self._decoder = decoder
        self._max_non_ambiguous_doppler = None
        self._timestamp = None
        self._header_data = None

    def __str__(self):
        return (
            f"Frame header: Maximum non-ambiguous doppler={self.max_non_ambiguous_doppler}, timestamp={self.timestamp} "
        )

    def get_header_data(self) -> np.ndarray:
        if self._header_data is None:
            self._header_data = self._decoder.get_radar_frame_header_data()
        return self._header_data

    @property
    def max_non_ambiguous_doppler(self) -> float:
        if self._max_non_ambiguous_doppler is None:
            self.get_header_data()
            self._max_non_ambiguous_doppler = self._header_data["MAX_NON_AMBIGUOUS_DOPPLER"]
        return self._max_non_ambiguous_doppler

    @property
    def timestamp(self) -> int:
        if self._timestamp is None:
            self.get_header_data()
            self._timestamp = self._header_data["TIMESTAMP"]
        return self._timestamp
