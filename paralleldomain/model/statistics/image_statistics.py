from collections import defaultdict
import pickle
from typing import Dict, Union

import numpy as np

from paralleldomain.model.scene import Scene
from paralleldomain.model.sensor import CameraSensorFrame
from paralleldomain.model.statistics.base import Statistic
from paralleldomain.model.statistics.constants import STATISTICS_REGISTRY
from paralleldomain.utilities.any_path import AnyPath


@STATISTICS_REGISTRY.register_module(name="color_distribution")
class ImageStatistics(Statistic):
    def __init__(self):
        super().__init__()
        self.reset()

    def _calculate_colour_channel_statistics_per_image(self, scene: Scene, sensor_frame: CameraSensorFrame):
        properties = self.parse_sensor_frame_properties(scene=scene, sensor_frame=sensor_frame)

        # log camera properties
        for key, value in properties.items():
            self._recorder[key].append(value)

        self._recorder["frame_med_intensity"].append(np.median(sensor_frame.image.rgb))

        for channel, name in zip([0, 1, 2], ["r_med_intensity", "g_med_intensity", "b_med_intensity"]):
            self._recorder[name].append(np.median(sensor_frame.image.rgb[..., channel].flatten()))

    def _reset(self):
        self._recorder = defaultdict(list)

    def _get_rgb_histograms(self, sensor_frame: CameraSensorFrame) -> Dict:
        histogram_red, _bin_edges = np.histogram(sensor_frame.image.rgb[..., 0].flatten(), bins=256, range=[0, 256])
        histogram_green, _ = np.histogram(sensor_frame.image.rgb[..., 1].flatten(), bins=256, range=[0, 256])
        histogram_blue, _ = np.histogram(sensor_frame.image.rgb[..., 2].flatten(), bins=256, range=[0, 256])

        return dict(
            histogram_red=histogram_red,
            histogram_blue=histogram_blue,
            histogram_green=histogram_green,
            bin_edges=_bin_edges,
        )

    def _update(self, scene: Scene, sensor_frame: CameraSensorFrame):
        self._calculate_colour_channel_statistics_per_image(scene=scene, sensor_frame=sensor_frame)

        if len(self._recorder["bin_edges"]) == 0:  # any key will start as an empty list
            _histograms = self._get_rgb_histograms(sensor_frame=sensor_frame)
            for key, value in _histograms.items():
                self._recorder[key] = value  # making the keys arrays, not lists.
        else:
            _histograms = self._get_rgb_histograms(sensor_frame=sensor_frame)

            for key, value in _histograms.items():
                if key != "bin_edges":  # we have the bins already. Do not add
                    self._recorder[key] += value  # add r,g,b arrays to themselves

    def _load(self, file_path: Union[str, AnyPath]):
        file_path = AnyPath(file_path)
        with file_path.open("rb") as f:
            self._recorder = pickle.load(f)

    def _save(self, file_path: Union[str, AnyPath]):
        file_path = AnyPath(file_path)
        with file_path.open("wb") as f:
            pickle.dump(self._recorder, f)
