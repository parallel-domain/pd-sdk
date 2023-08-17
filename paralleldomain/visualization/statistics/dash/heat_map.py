import math
from typing import List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from paralleldomain.model.statistics.heat_map import ClassHeatMaps
from paralleldomain.visualization.statistics.viewer import ViewComponent, BACKEND, STATISTIC_VIS_REGISTRY


@STATISTIC_VIS_REGISTRY.register_module(reference_model=ClassHeatMaps, is_default=True, backend=BACKEND.DASH)
class PlotlyClassHeatMapsView(ViewComponent[ClassHeatMaps]):
    def __init__(self, model: ClassHeatMaps, classes_of_interest: List[str] = None) -> None:
        super().__init__(model=model)
        self._classes_of_interest = classes_of_interest

    def _visualize(self):
        heat_maps = self._model.get_heatmaps(classes_of_interest=self._classes_of_interest)
        if len(heat_maps) == 0:
            return go.Figure(go.Heatmap(z=[]))

        width = list(heat_maps.values())[0].shape[1]
        height = list(heat_maps.values())[0].shape[0]

        num_classes = len(heat_maps)

        # Set the number of columns/rows such that their appear balanced
        # This assumes an viewport aspect ratio of 16/9.
        num_cols = math.ceil(math.sqrt((16 * num_classes * height) / (9 * width)))
        num_rows = math.ceil(num_classes / num_cols)

        self._fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            shared_yaxes="all",
            subplot_titles=[class_name for class_name in heat_maps],
        )

        background_images = []
        # Directly visualizing heatmaps as go.Heatmap() object is prohibitively slow
        # due to the need to convert numpy arrays to json first. Instead we create heatmaps manually:
        # 1) Convert heatmap model into an Pillow image
        # 2) Create a scatter plot with dimension of the image and a color space corrsponding to the values
        # 3) Add heatmap as a background image of the plot
        for idx, heat_map in enumerate(heat_maps.values()):
            import numpy as np
            from PIL import Image
            from matplotlib import cm
            from matplotlib.colors import Normalize

            # Limits
            xmin = 0
            xmax = heat_map.shape[1] - 1
            ymin = 0
            ymax = heat_map.shape[0] - 1
            amin = np.amin(heat_map)
            amax = np.amax(heat_map)

            cNorm = Normalize(vmin=amin, vmax=amax)
            scalarMap = cm.ScalarMappable(norm=cNorm, cmap="viridis")
            seg_colors = scalarMap.to_rgba(heat_map)
            img = Image.fromarray(np.uint8(seg_colors * 255))

            self._fig.add_trace(
                go.Scatter(
                    x=[xmin, xmax],
                    y=[ymin, ymax],
                    mode="markers",
                    marker={
                        "color": [np.amin(heat_map), np.amax(heat_map)],
                        "opacity": 0,
                    },
                ),
                row=idx // num_cols + 1,
                col=idx % num_cols + 1,
            )
            background_images.append(
                go.layout.Image(
                    x=xmin,
                    sizex=xmax - xmin,
                    y=ymax,
                    sizey=ymax - ymin,
                    xref=f"x{idx+1}" if idx != 0 else "x",
                    yref=f"y{idx+1}" if idx != 0 else "y",
                    opacity=1.0,
                    layer="below",
                    sizing="stretch",
                    source=img,
                )
            )
        self._fig.update_layout(images=background_images, showlegend=False, uirevision=True)

        for idx in range(num_classes):
            self._fig.update_layout(
                {
                    f"xaxis{idx+1 if idx > 0 else ''}": dict(
                        showgrid=False,
                        zeroline=False,
                        range=[xmin, xmax],
                        scaleanchor=f"y{idx+1}" if idx != 0 else "y",
                        constrain="domain",
                        scaleratio=1,
                    )
                }
            )
        self._fig.update_yaxes(showgrid=False, zeroline=False, range=[ymin, ymax], constrain="domain")
        self._vis_need_update = False

        return self._fig

    @property
    def title(self):
        return "Classwise Heatmaps"
