from typing import List

import rerun as rr
import numpy as np

from paralleldomain.model.statistics.class_distribution import ClassDistribution
from paralleldomain.visualization.statistics.viewer import ViewComponent, BACKEND, STATISTIC_VIS_REGISTRY


@STATISTIC_VIS_REGISTRY.register_module(reference_model=ClassDistribution, is_default=True, backend=BACKEND.RERUN)
class RerunClassDistributionView(ViewComponent[ClassDistribution]):
    def __init__(self, model: ClassDistribution, classes_of_interest: List[str] = None) -> None:
        super().__init__(model=model)
        self._classes_of_interest = classes_of_interest

    def _visualize(self):
        instance_distribution = self._model.get_instance_distribution()
        pixel_distribution = self._model.get_pixel_distribution()

        if self._classes_of_interest is not None:
            filtered_instances = {key: instance_distribution.get(key, 0) for key in self._classes_of_interest}
            filtered_pixel = {key: pixel_distribution.get(key, 0) for key in self._classes_of_interest}
        else:
            filtered_instances = instance_distribution
            filtered_pixel = pixel_distribution

        rr.log_tensor(
            entity_path="statistics/class_distribution",
            tensor=np.array(list(filtered_instances.values())),
        )

        rr.log_tensor(
            entity_path="statistics/pixel_distribution",
            tensor=np.array(list(filtered_pixel.values())),
        )
        self._vis_need_update = False

    def to_html(self, filename: str = "dashboard.html") -> None:
        raise NotImplementedError("Rerun Viewer does not support export to HTML")

    @property
    def title(self):
        return "Class Distribution"

    def notify(self):
        super().notify()
        self.visualize()
