import logging

logger = logging.getLogger(__name__)
try:
    from paralleldomain.visualization.statistics.dash.class_distribution import PlotlyClassDistributionView
    from paralleldomain.visualization.statistics.dash.image_statistics import PlotlyImageStatisticsView
    from paralleldomain.visualization.statistics.dash.heat_map import PlotlyClassHeatMapsView
except ModuleNotFoundError:
    logger.warning(
        "Plotly was not installed. Plotly based statistics visualizations wont be available. "
        "Install pd-sdk with 'dash' install extra for plotly support"
    )

try:
    from paralleldomain.visualization.statistics.rerun.class_distribution import RerunClassDistributionView
    from paralleldomain.visualization.statistics.rerun.heat_map import RerunClassHeatMapsView
    from paralleldomain.visualization.statistics.rerun.image_statistics import RerunImageStatisticsView
except ModuleNotFoundError:
    logger.warning(
        "Rerun was not installed. Rerun based statistics visualizations wont be available. "
        "Install pd-sdk with 'visualization' install extra for rerun support"
    )
