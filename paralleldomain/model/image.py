import abc

import numpy as np

from paralleldomain.model.type_aliases import FrameId, SensorName

try:
    from typing import Protocol, Tuple
except ImportError:
    from typing_extensions import Protocol  # type: ignore


class Image:
    @property
    @abc.abstractmethod
    def rgba(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def rgb(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def width(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def height(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def channels(self) -> int:
        pass

    @property
    def coordinates(self) -> np.ndarray:
        y_coords, x_coords = np.meshgrid(range(self.height), range(self.width), indexing="ij")
        return np.stack([y_coords, x_coords], axis=-1)


class ImageDecoderProtocol(Protocol):
    def get_image_dimensions(self, sensor_name: SensorName, frame_id: FrameId) -> Tuple[int, int, int]:
        pass

    def get_image_rgba(self, sensor_name: SensorName, frame_id: FrameId) -> np.ndarray:
        pass


class DecoderImage(Image):
    def __init__(self, decoder: ImageDecoderProtocol, sensor_name: SensorName, frame_id: FrameId):
        self.frame_id = frame_id
        self.sensor_name = sensor_name
        self._decoder = decoder

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
