import math
from typing import List

import rerun as rr
import numpy as np

from paralleldomain.model.statistics.heat_map import ClassHeatMaps
from paralleldomain.visualization.statistics.viewer import ViewComponent, BACKEND, STATISTIC_VIS_REGISTRY


@STATISTIC_VIS_REGISTRY.register_module(reference_model=ClassHeatMaps, is_default=True, backend=BACKEND.RERUN)
class RerunClassHeatMapsView(ViewComponent[ClassHeatMaps]):
    def __init__(self, model: ClassHeatMaps, classes_of_interest: List[str] = None) -> None:
        super().__init__(model=model)
        self._classes_of_interest = classes_of_interest

    def _visualize(self):
        heat_maps = self._model.get_heatmaps(classes_of_interest=self._classes_of_interest)

        for class_name, heat_map in heat_maps.items():
            amax = np.amax(heat_map)
            if amax == 0.0:
                rr.log_tensor(
                    entity_path=f"statistics/heatmap/{class_name}", tensor=heat_map, names=("width", "height")
                )
            else:
                rr.log_tensor(
                    entity_path=f"statistics/heatmap/{class_name}", tensor=heat_map / amax, names=("width", "height")
                )

        self._vis_need_update = False

    @property
    def title(self):
        return "Classwise Heatmaps"

    def notify(self):
        super().notify()
        self.visualize()
