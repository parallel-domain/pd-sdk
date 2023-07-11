import logging
import random

from pd.assets import ObjAssets
from pd.data_lab.config.distribution import CenterSpreadConfig, EnumDistribution
from pd.data_lab.context import setup_datalab
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.sim_instance import SimulationInstance

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.generators.debris import DebrisGeneratorParameters
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.parked_vehicle import ParkedVehicleGeneratorParameters
from paralleldomain.data_lab.generators.position_request import (
    LaneSpawnPolicy,
    LocationRelativePositionRequest,
    PositionRequest,
)
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.model.statistics.base import CompositeStatistic
from paralleldomain.model.statistics.class_distribution import ClassDistribution
from paralleldomain.model.statistics.heat_map import ClassHeatMaps
from paralleldomain.model.statistics.image_statistics import ImageStatistics
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.utilities.transformation import Transformation
from paralleldomain.visualization.statistics.class_distribution import ClassDistributionView
from paralleldomain.visualization.statistics.dash_viewer import DashViewer
from paralleldomain.visualization.statistics.heat_map import ClassHeatMapsView
from paralleldomain.visualization.statistics.image_statistics import ImageStatisticsView

setup_loggers(logger_names=["__main__", "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)

setup_datalab("v2.0.0-beta")


def get_debris_asset_list() -> str:
    # query specific assets from PD's asset registry and use as csv input for Debris generator
    asset_objs = ObjAssets.select().where(
        (ObjAssets.name % "*trash*") & (ObjAssets.width * ObjAssets.height * ObjAssets.length < 1.0)
    )
    asset_names = [o.name for o in asset_objs]
    return ",".join(asset_names)


sensor_rig = data_lab.SensorRig(
    sensor_configs=[
        data_lab.SensorConfig.create_camera_sensor(
            name="Front",
            width=1920,
            height=1080,
            field_of_view_degrees=70,
            pose=Transformation.from_euler_angles(
                angles=[0.0, 0.0, 0.0], order="xyz", degrees=True, translation=[0.0, 0.0, 2.0]
            ),
            annotation_types=[AnnotationTypes.SemanticSegmentation2D, AnnotationTypes.InstanceSegmentation2D],
        ),
    ]
)

# Create scenario
scenario = data_lab.Scenario(sensor_rig=sensor_rig)
scenario.random_seed = random.randint(1, 1_000_000)  # set to a fixed integer to keep scenario generation deterministic

# Set weather variables and time of day
scenario.environment.time_of_day.set_category_weight(data_lab.TimeOfDays.Day, 1.0)
scenario.environment.clouds.set_constant_value(0.5)
scenario.environment.rain.set_constant_value(0.0)
scenario.environment.fog.set_uniform_distribution(min_value=0.1, max_value=0.3)
scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

# Select an environment
scenario.set_location(data_lab.Location(name="SF_6thAndMission_medium", version="v2.0.0-beta"))

# Place ourselves in the world
scenario.add_ego(
    generator=EgoAgentGeneratorParameters(
        agent_type=AgentType.VEHICLE,
        position_request=PositionRequest(
            lane_spawn_policy=LaneSpawnPolicy(
                lane_type=EnumDistribution(
                    probabilities={"Drivable": 1.0},
                )
            )
        ),
    ),
)


scenario.add_objects(
    generator=DebrisGeneratorParameters(
        spawn_probability=0.5,
        min_debris_distance=0.0,
        max_debris_distance=30.0,
        debris_asset_tag="trash_wrapper_01,trash_tobacco_01,trash_straw_plastic_01,trash_square_bottle_01",
        position_request=PositionRequest(
            location_relative_position_request=LocationRelativePositionRequest(
                agent_tags=["EGO"],
            )
        ),
    )
)

# Place other agents
scenario.add_agents(
    generator=TrafficGeneratorParameters(
        position_request=PositionRequest(
            location_relative_position_request=LocationRelativePositionRequest(
                agent_tags=["EGO"],
                max_spawn_radius=100.0,
            )
        ),
    )
)

scenario.add_agents(
    generator=ParkedVehicleGeneratorParameters(
        spawn_probability=CenterSpreadConfig(center=0.4),
        position_request=PositionRequest(
            location_relative_position_request=LocationRelativePositionRequest(
                agent_tags=["EGO"],
                max_spawn_radius=100.0,
            )
        ),
    )
)

# Create statistic models
class_counter = ClassDistribution()
class_heatmaps = ClassHeatMaps()
image_statistics = ImageStatistics()
model = CompositeStatistic([class_counter, image_statistics, class_heatmaps])

# Create corresponding views
class_counter_view = ClassDistributionView(
    model=class_counter, classes_of_interest=["Pedestrian", "Bicycle", "Car", "Debris", "Road", "TrafficLight"]
)
image_statistics_view = ImageStatisticsView(model=image_statistics)
class_heatmaps_view = ClassHeatMapsView(
    model=class_heatmaps,
    classes_of_interest=["Pedestrian", "Bicycle", "Car", "Debris", "Road", "TrafficLight"],
)

viewer = DashViewer(view_components=[class_counter_view, image_statistics_view, class_heatmaps_view])
# Alternatively create both statistic model and viewer with default component
# viewer, model = DashViewer.create_with_default_components(
#     classes_of_interest=["Pedestrian", "Bicycle", "Car", "Debris", "Road", "TrafficLight"]
# )

# Launch DashViewer in the background to avoid blocking the main thread
viewer.launch(in_background=True, port=8051)

data_lab.preview_scenario(
    scenario=scenario,
    number_of_scenes=1,
    frames_per_scene=20,
    sim_capture_rate=10,
    sim_instance=SimulationInstance(name="<instance name>"),
    render_instance=RenderInstance(name="<instance name>"),
    statistic=model,
)

# The Statistic object can also be passed to create_mini_batch_stream and create_mini_batch. Example Usage:
# data_lab.create_mini_batch(
#     scenario=scenario,
#     sim_instance=SimulationInstance(address="ssl://sim.step-api-dev.paralleldomain.com:30XX"),
#     render_instance=RenderInstance(address="ssl://ig.step-api-dev.paralleldomain.com:30XX"),
#     frames_per_scene=20,
#     sim_capture_rate=10,
#     number_of_scenes=2,
#     format_kwargs=dict(
#         dataset_output_path="./debug-mini6/",
#         encode_to_binary=False,
#     ),
#     pipeline_kwargs=dict(copy_all_available_sensors_and_annotations=True),
#     statistic=model,
# )

# Need to explicitly save statistics
AnyPath("./statistics").mkdir(exist_ok=True)
model.save("./statistics")
