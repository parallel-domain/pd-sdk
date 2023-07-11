import pickle
import numpy as np

from paralleldomain.model.scene import Scene
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.model.statistics.base import Statistic
from paralleldomain.model.statistics.constants import STATISTICS_REGISTRY


@STATISTICS_REGISTRY.register_module()
class ImageStatistics(Statistic):
    def __init__(self):
        super().__init__()
        self.reset()

    def _reset(self):
        self._histogram_red = None
        self._histogram_green = None
        self._histogram_blue = None
        self._bin_edges = None

    def _update(self, scene: Scene, sensor_frame: CameraSensorFrame):
        if self._bin_edges is None:
            self._histogram_red, self._bin_edges = np.histogram(
                sensor_frame.image.rgb[..., 0].flatten(), bins=256, range=[0, 256]
            )
            self._histogram_green, _ = np.histogram(sensor_frame.image.rgb[..., 1].flatten(), bins=256, range=[0, 256])
            self._histogram_blue, _ = np.histogram(sensor_frame.image.rgb[..., 2].flatten(), bins=256, range=[0, 256])
        else:
            histogram_red, self._bin_edges = np.histogram(
                sensor_frame.image.rgb[..., 0].flatten(), bins=256, range=[0, 256]
            )
            histogram_green, _ = np.histogram(sensor_frame.image.rgb[..., 1].flatten(), bins=256, range=[0, 256])
            histogram_blue, _ = np.histogram(sensor_frame.image.rgb[..., 2].flatten(), bins=256, range=[0, 256])
            self._histogram_red += histogram_red
            self._histogram_green += histogram_green
            self._histogram_blue += histogram_blue

    def _load(self, file_path: str):
        with open(file_path, "rb") as f:
            histogram_dict = pickle.load(f)
            self._histogram_red = histogram_dict["histogram_red"]
            self._histogram_green = histogram_dict["histogram_green"]
            self._histogram_blue = histogram_dict["histogram_blue"]
            self._bin_edges = histogram_dict["bin_edges"]

    def _save(self, file_path: str):
        with open(file_path, "wb") as f:
            pickle.dump(
                dict(
                    histogram_red=self._histogram_red,
                    histogram_green=self._histogram_green,
                    histogram_blue=self._histogram_blue,
                    bin_edges=self._bin_edges,
                ),
                f,
            )
