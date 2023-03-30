import logging
import random

from pd.data_lab.config.distribution import CenterSpreadConfig, EnumDistribution, MinMaxConfigInt
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
    SpecialAgentTag,
)
from paralleldomain.data_lab.generators.random_pedestrian import RandomPedestrianGeneratorParameters
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.utilities.transformation import Transformation

setup_loggers(logger_names=["__main__", "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)


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
        )
    ]
)

# Create scenario
scenario = data_lab.Scenario(sensor_rig=sensor_rig)
scenario.random_seed = random.randint(1, 1_000_000)  # set to a fixed integer to keep scenario generation deterministic

# Set weather variables and time of day
scenario.environment.time_of_day.set_category_weight(data_lab.TimeOfDays.Day, 1.0)
scenario.environment.clouds.set_constant_value(0.5)
scenario.environment.rain.set_constant_value(0.0)
scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

# Select an environment
scenario.set_location(data_lab.Location(name="SF_6thAndMission_medium", version="v2.0.1"))

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
        max_debris_distance=25.0,
        debris_asset_tag="trash_bottle_tall_01",
        position_request=PositionRequest(
            location_relative_position_request=LocationRelativePositionRequest(
                agent_tags=[SpecialAgentTag.EGO],
            )
        ),
    )
)

# Place other agents
scenario.add_agents(
    generator=TrafficGeneratorParameters(
        position_request=PositionRequest(
            location_relative_position_request=LocationRelativePositionRequest(
                agent_tags=[SpecialAgentTag.EGO],
                max_spawn_radius=25.0,
            )
        ),
    )
)

scenario.add_agents(
    generator=ParkedVehicleGeneratorParameters(
        spawn_probability=CenterSpreadConfig(center=0.4),
        position_request=PositionRequest(
            location_relative_position_request=LocationRelativePositionRequest(
                agent_tags=[SpecialAgentTag.EGO],
                max_spawn_radius=200.0,
            )
        ),
    )
)

scenario.add_agents(
    generator=RandomPedestrianGeneratorParameters(
        num_of_pedestrians_range=MinMaxConfigInt(min=25, max=35),
        position_request=PositionRequest(
            location_relative_position_request=LocationRelativePositionRequest(
                agent_tags=[SpecialAgentTag.EGO],
                max_spawn_radius=200.0,
            )
        ),
    )
)

# scenario.save_scenario(path=AnyPath("my_scenario.json"))

data_lab.preview_scenario(
    scenario=scenario,
    frames_per_scene=100,
    show_image_for_n_seconds=1,
    # sim_capture_rate=2,
    sim_instance=SimulationInstance(address="ssl://sim.step-api-dev.paralleldomain.com:3014"),
    render_instance=RenderInstance(address="ssl://ig.step-api-dev.paralleldomain.com:3014"),
)
