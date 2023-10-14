import abc
from typing import Protocol, Tuple

import numpy as np


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


class InMemoryImage(Image):
    def __init__(self, rgba: np.ndarray, width: int, height: int, channels: int):
        self._rgba = rgba
        self._width = width
        self._height = height
        self._channels = channels

    @property
    def rgba(self) -> np.ndarray:
        return self._rgba

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def rgb(self):
        return self.rgba[..., :3]


class ImageDecoderProtocol(Protocol):
    def get_image_dimensions(self) -> Tuple[int, int, int]:
        pass

    def get_image_rgba(self) -> np.ndarray:
        pass


class DecoderImage(Image):
    def __init__(self, decoder: ImageDecoderProtocol):
        self._decoder = decoder
        self._data_rgba = None
        self._image_dims = None

    @property
    def _image_dimensions(self) -> Tuple[int, int, int]:
        if self._image_dims is None:
            self._image_dims = self._decoder.get_image_dimensions()
        return self._image_dims

    @property
    def rgba(self) -> np.ndarray:
        if self._data_rgba is None:
            self._data_rgba = self._decoder.get_image_rgba()
        return self._data_rgba

    @property
    def rgb(self) -> np.ndarray:
        return self.rgba[:, :, :3]

    @property
    def width(self) -> int:
        return self._image_dimensions[1]

    @property
    def height(self) -> int:
        return self._image_dimensions[0]

    @property
    def channels(self) -> int:
        return self._image_dimensions[2]
