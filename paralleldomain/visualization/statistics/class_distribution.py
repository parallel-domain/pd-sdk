from typing import List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from paralleldomain.model.statistics.class_distribution import ClassDistribution
from paralleldomain.visualization.statistics.viewer import ViewComponent, BACKEND, STATISTIC_VIS_REGISTRY


@STATISTIC_VIS_REGISTRY.register_module(reference_model=ClassDistribution, is_default=True, backend=BACKEND.DASH)
class ClassDistributionView(ViewComponent):
    def __init__(self, model: ClassDistribution, classes_of_interest: List[str] = None) -> None:
        super().__init__(model=model)
        self._classes_of_interest = classes_of_interest

        self._init_figure()

    def _init_figure(self):
        specs = [[{"type": "bar"}, {"type": "pie"}], [{"type": "bar"}, {"type": "pie"}]]
        self._fig = make_subplots(
            rows=2,
            cols=2,
            specs=specs,
            shared_yaxes=False,
            column_widths=[0.7, 0.3],
            subplot_titles=["Instance Counts", "", "Pixel Counts - Log Scale", ""],
        )
        self._fig.add_trace(go.Bar(x=[], y=[], legendgroup="1", name="Instance Counts"), 1, 1)
        self._fig.add_trace(go.Pie(labels=[], values=[], legendgroup="2"), 1, 2)
        self._fig.add_trace(go.Bar(x=[], y=[], legendgroup="1", name="Pixel Counts"), 2, 1)
        self._fig.update_layout({"yaxis2": dict(type="log")})
        self._fig.add_trace(go.Pie(labels=[], values=[], legendgroup="2"), 2, 2)
        self._fig.update_layout(showlegend=True, uirevision="legend")

    def _visualize(self):
        instance_distribution = self._model.get_instance_distribution()
        pixel_distribution = self._model.get_instance_distribution()

        if self._classes_of_interest is not None:
            filtered_instances = {key: instance_distribution.get(key, 0) for key in self._classes_of_interest}
            filtered_pixel = {key: pixel_distribution.get(key, 0) for key in self._classes_of_interest}
        else:
            filtered_instances = instance_distribution
            filtered_pixel = pixel_distribution

        self._fig.data[0].x = list(filtered_instances.keys())
        self._fig.data[0].y = list(filtered_instances.values())
        self._fig.data[1].labels = list(filtered_instances.keys())
        self._fig.data[1].values = list(filtered_instances.values())

        self._fig.data[2].x = list(filtered_pixel.keys())
        self._fig.data[2].y = list(filtered_pixel.values())
        self._fig.data[3].labels = list(filtered_pixel.keys())
        self._fig.data[3].values = list(filtered_pixel.values())
        self._vis_need_update = False

        return self._fig

    @property
    def title(self):
        return "Class Distribution"
