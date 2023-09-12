import logging
import random

from pd.data_lab import Scenario
from pd.data_lab.config.distribution import CenterSpreadConfig, ContinousUniformDistribution, EnumDistribution
from pd.data_lab.context import setup_datalab
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.sim_instance import SimulationInstance

from paralleldomain.data_lab import (
    Location,
    SensorConfig,
    SensorRig,
    TimeOfDays,
    preview_scenario,
    DEFAULT_DATA_LAB_VERSION,
)
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, SensorExtrinsic
from paralleldomain.data_lab.config.world import EnvironmentParameters, ParkingSpaceData
from paralleldomain.data_lab.generators.behavior import VehicleBehavior
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.parked_vehicle import ParkedVehicleGeneratorParameters
from paralleldomain.data_lab.generators.position_request import (
    LaneSpawnPolicy,
    LocationRelativePositionRequest,
    PositionRequest,
)
from paralleldomain.data_lab.generators.spawn_data import VehicleSpawnData
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.utilities.logging import setup_loggers

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)
setup_datalab(DEFAULT_DATA_LAB_VERSION)

sensor_rig = SensorRig(
    sensor_configs=[
        SensorConfig(
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
scenario = Scenario(sensor_rig=sensor_rig)
scenario.random_seed = random.randint(1, 1_000_000)  # set to a fixed integer to keep scenario generation deterministic

# Set weather variables and time of day
scenario.environment.time_of_day.set_category_weight(TimeOfDays.Dusk, 1.0)
scenario.environment.clouds.set_constant_value(0.5)
scenario.environment.rain.set_constant_value(0.0)
scenario.environment.fog.set_uniform_distribution(min_value=0.1, max_value=0.3)
scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

# Select an environment
scenario.set_location(Location(name="SF_6thAndMission_medium"))

# Set parking parameters
parking_space_on_street = True  # Will park in lots if set to false
time_to_park = 4  # Time to complete parking maneuver in seconds
parking_space_type = "ANGLE_60"  # Choose from "PERPENDICULAR", "ANGLE_60", "ANGLE_45", "ANGLE_30"

# Place ourselves in the world
scenario.add_ego(
    generator=EgoAgentGeneratorParameters(
        agent_type=AgentType.VEHICLE,
        position_request=(
            PositionRequest(
                lane_spawn_policy=LaneSpawnPolicy(
                    lane_type=EnumDistribution(
                        probabilities={"ParkingSpace": 1.0},
                    ),
                    road_type=EnumDistribution(
                        probabilities={"Primary": 1.0},
                    ),
                    on_road_parking_angle_distribution=EnumDistribution(
                        probabilities={
                            parking_space_type: 1.0,
                        }
                    ),
                )
            )
            if parking_space_on_street
            else None
        ),
        vehicle_spawn_data=VehicleSpawnData(
            vehicle_behavior=VehicleBehavior(
                parking_scenario_goal=PositionRequest(
                    lane_spawn_policy=LaneSpawnPolicy(
                        lane_type=EnumDistribution(
                            probabilities={"Drivable" if parking_space_on_street else "ParkingAisle": 1.0},
                        )
                    )
                ),
                parking_scenario_time=ContinousUniformDistribution(min=4, max=4),
            )
        ),
    ),
)

scenario.set_environment(
    parameters=EnvironmentParameters(
        parking_space_data=ParkingSpaceData(
            parking_lot_angle_distribution=EnumDistribution(
                probabilities={
                    parking_space_type: 1.0,
                }
            )
        )
    )
)

# Place other agents
scenario.add_agents(
    generator=TrafficGeneratorParameters(
        spawn_probability=0.6,
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
        spawn_probability=CenterSpreadConfig(
            center=0.6,
        ),
        position_request=PositionRequest(
            location_relative_position_request=LocationRelativePositionRequest(
                agent_tags=["EGO"],
                max_spawn_radius=100.0,
            )
        ),
    )
)

preview_scenario(
    scenario=scenario,
    frames_per_scene=10,
    sim_capture_rate=100,
    sim_instance=SimulationInstance(name="<instance name>"),
    render_instance=RenderInstance(name="<instance name>"),
)
