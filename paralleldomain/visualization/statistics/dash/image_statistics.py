import plotly.graph_objects as go

from paralleldomain.model.statistics.image_statistics import ImageStatistics
from paralleldomain.visualization.statistics.viewer import ViewComponent, BACKEND, STATISTIC_VIS_REGISTRY


@STATISTIC_VIS_REGISTRY.register_module(reference_model=ImageStatistics, is_default=True, backend=BACKEND.DASH)
class PlotlyImageStatisticsView(ViewComponent[ImageStatistics]):
    def _visualize(self):
        if self._model._recorder["_bin_edges"] is None:
            return go.Figure(go.Bar(x=[], y=[]))

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=self._model._recorder["_bin_edges"][:-1],
                y=self._model._recorder["_histogram_red"],
                name="Red",
                marker_color="red",
            )
        )
        fig.add_trace(
            go.Bar(
                x=self._model._recorder["_bin_edges"][:-1],
                y=self._model._recorder["_histogram_green"],
                name="Green",
                marker_color="green",
            )
        )
        fig.add_trace(
            go.Bar(
                x=self._model._recorder["_bin_edges"][:-1],
                y=self._model._recorder["_histogram_blue"],
                name="Blue",
                marker_color="blue",
            )
        )

        # Update the layout of the figure
        fig.update_layout(
            title=self.title,
            xaxis=dict(title="Pixel Value"),
            yaxis=dict(title="Frequency"),
        )

        return fig

    @property
    def title(self):
        return "Image Statistics"
