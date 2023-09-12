import logging
import random

from pd.data_lab.config.distribution import (
    CenterSpreadConfig,
    EnumDistribution,
)
from pd.data_lab.context import load_map, setup_datalab
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.sim_instance import SimulationInstance

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab import DEFAULT_DATA_LAB_VERSION
from paralleldomain.data_lab.config.map import MapQuery
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, SensorExtrinsic
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.parked_vehicle import ParkedVehicleGeneratorParameters
from paralleldomain.data_lab.generators.position_request import (
    LaneSpawnPolicy,
    LocationRelativePositionRequest,
    PositionRequest,
)
from paralleldomain.data_lab.generators.road_signs import SignGenerator
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.utilities.logging import setup_loggers

setup_loggers(logger_names=["__main__", "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)

setup_datalab(DEFAULT_DATA_LAB_VERSION)

sensor_rig = data_lab.SensorRig(
    sensor_configs=[
        data_lab.SensorConfig(
            display_name="Front",
            camera_intrinsic=CameraIntrinsic(
                width=1920,
                height=1080,
                fov=70.0,
            ),
            sensor_extrinsic=SensorExtrinsic(
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
                x=0.0,
                y=0.0,
                z=2.0,
            ),
        )
    ]
)

# Create scenario
scenario = data_lab.Scenario(sensor_rig=sensor_rig)
scenario.random_seed = random.randint(1, 1_000_000)  # set to a fixed integer to keep scenario generation deterministic

# Set weather variables and time of day
scenario.environment.time_of_day.set_category_weight(data_lab.TimeOfDays.Day, 1.0)
scenario.environment.clouds.set_uniform_distribution(0, 0.3)
scenario.environment.rain.set_constant_value(0.0)
scenario.environment.fog.set_uniform_distribution(min_value=0, max_value=0.1)
scenario.environment.wetness.set_uniform_distribution(min_value=0.0, max_value=0.2)

location = data_lab.Location(name="SF_6thAndMission_medium")

# Select an environment
scenario.set_location(location)

# Initialize umd
umd_map = load_map(location)
map_query = MapQuery(umd_map)

sim_capture_rate = 10

scenario.add_ego(
    generator=EgoAgentGeneratorParameters(
        agent_type=AgentType.VEHICLE,
        position_request=PositionRequest(
            lane_spawn_policy=LaneSpawnPolicy(
                lane_type=EnumDistribution(
                    probabilities={"Drivable": 1.0},
                )
            ),
        ),
    )
)

scenario.add_agents(
    generator=TrafficGeneratorParameters(
        position_request=PositionRequest(
            location_relative_position_request=LocationRelativePositionRequest(
                agent_tags=["EGO"],
                max_spawn_radius=50.0,
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
                max_spawn_radius=50.0,
            )
        ),
    )
)

scenario.add_agents(
    generator=SignGenerator(
        num_sign_poles=20,
        random_seed=scenario.random_seed,
        radius=30.0,
        sim_capture_rate=sim_capture_rate,
        max_signs_per_pole=3,
        country="Portugal",
        forward_offset_to_place_signs=20.0,
        min_distance_between_signs=1.5,
        single_frame_mode=False,
    )
)


data_lab.preview_scenario(
    scenario=scenario,
    frames_per_scene=10,
    sim_capture_rate=sim_capture_rate,
    sim_instance=SimulationInstance(name="<instance name>"),
    render_instance=RenderInstance(name="<instance name>"),
)
