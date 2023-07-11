from typing import List
import threading

import dash
from dash import html, dcc
import plotly.io as pio

from paralleldomain.visualization.statistics.viewer import ViewComponent, StatisticViewer, BACKEND


class DashViewer(StatisticViewer):
    def __init__(self, view_components: List[ViewComponent]):
        super().__init__(view_components=view_components)

    @classmethod
    def backend(cls) -> BACKEND:
        return BACKEND.DASH

    def launch(self, in_background: bool = False, port: int = 8050):
        if in_background:
            self._dash_thread = threading.Thread(target=self._launch_dash, kwargs=dict(port=port))
            self._dash_thread.daemon = True
            self._dash_thread.start()
        else:
            self._launch_dash(port=port)

    def _launch_dash(self, port: int = 8050):
        app = dash.Dash(__name__)

        @app.callback(
            dash.dependencies.Output("figure-container", "children"),
            dash.dependencies.Input("interval-component", "n_intervals"),
            running=[
                (dash.dependencies.Output("interval-component", "disabled"), True, False),
            ],
        )
        def update_figures(n):
            tabs = []
            for measure in self._view_components:
                fig = measure.visualize()

                tabs.append(
                    dcc.Tab(
                        label=measure.title,
                        children=[dcc.Graph(figure=fig, style={"width": "98vw", "height": "95vh"})],
                    )
                )
            return dcc.Tabs(id="tab", children=tabs, persistence=True)

        # Define the app layout
        app.layout = html.Div(
            children=[
                html.Div(id="figure-container"),
                dcc.Interval(id="interval-component", interval=1 * 1000, n_intervals=0),  # in milliseconds
            ],
        )

        app.run_server(port=port, debug=False, use_reloader=False, dev_tools_silence_routes_logging=True)

    def to_html(self, filename: str = "dashboard.html"):
        with open(filename, "w") as dashboard:
            dashboard.write("<html><head></head><body>" + "\n")
            for idx, component in enumerate(self._view_components):
                inner_html = pio.to_html(component.visualize(), include_plotlyjs=idx == 0, full_html=False)
                dashboard.write(inner_html)
            dashboard.write("</body></html>" + "\n")

    def show(self):
        for component in self._view_components:
            component.visualize().show()
