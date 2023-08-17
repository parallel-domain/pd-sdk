import rerun as rr

from paralleldomain.model.statistics.image_statistics import ImageStatistics
from paralleldomain.visualization.statistics.viewer import ViewComponent, BACKEND, STATISTIC_VIS_REGISTRY


@STATISTIC_VIS_REGISTRY.register_module(reference_model=ImageStatistics, is_default=True, backend=BACKEND.RERUN)
class RerunImageStatisticsView(ViewComponent[ImageStatistics]):
    def _visualize(self):
        rr.log_tensor("statistics/color_distribution/red", self._model._recorder["histogram_red"])
        rr.log_tensor("statistics/color_distribution/green", self._model._recorder["histogram_green"])
        rr.log_tensor("statistics/color_distribution/blue", self._model._recorder["histogram_blue"])

    @property
    def title(self):
        return "Image Statistics"

    def notify(self):
        super().notify()
        self.visualize()
