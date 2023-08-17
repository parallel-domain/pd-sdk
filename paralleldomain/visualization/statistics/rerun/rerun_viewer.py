from typing import List

from paralleldomain.visualization.statistics.viewer import ViewComponent, StatisticViewer, BACKEND
from paralleldomain.visualization.initialization import initialize_viewer


class RerunViewer(StatisticViewer):
    def __init__(self, view_components: List[ViewComponent]):
        super().__init__(view_components=view_components)

    @classmethod
    def backend(cls) -> BACKEND:
        return BACKEND.RERUN

    def launch(self):
        initialize_viewer()
        self.show()

    def to_html(self, filename: str = "dashboard.html") -> None:
        raise NotImplementedError("Rerun Viewer does not support export to HTML")

    def show(self):
        for component in self._view_components:
            component.visualize()
