import logging
import random

from pd.data_lab import Scenario
from pd.data_lab.config.distribution import ContinousUniformDistribution, EnumDistribution
from pd.data_lab.context import setup_datalab
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.sim_instance import SimulationInstance

from paralleldomain.data_lab import Location, SensorConfig, SensorRig, TimeOfDays, preview_scenario
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, DistortionParams, SensorExtrinsic
from paralleldomain.data_lab.config.world import EnvironmentParameters, ParkingSpaceData
from paralleldomain.data_lab.generators.behavior import VehicleBehavior
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.position_request import LaneSpawnPolicy, PositionRequest
from paralleldomain.data_lab.generators.spawn_data import VehicleSpawnData
from paralleldomain.utilities.logging import setup_loggers

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)

setup_datalab("v2.4.1-beta")

sensor_rig = SensorRig(
    sensor_configs=[
        SensorConfig(
            display_name="Ortho_BEV",
            camera_intrinsic=CameraIntrinsic(
                width=1920,
                height=1920,
                distortion_params=DistortionParams(
                    fx=150.0, fy=150.0, cx=960.0, cy=960.0, p1=-200, p2=300, skew=0, fisheye_model=6
                ),
            ),
            sensor_extrinsic=SensorExtrinsic(
                roll=0.0,
                pitch=-90.0,
                yaw=0.0,
                x=0.0,
                y=0.0,
                z=100,
            ),
        ),
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
scenario.set_location(Location(name="SF_VanNessAveAndTurkSt"))

# Set up what type of parking space delineations we want to see (currently set to be random)
# but can be set to a specific delineation type from the list below
parking_space_delineation_type = random.choice(
    [
        "BOX_CLOSED",
        "BOX_OPEN_CURB",
        "BOX_DOUBLE",
        "SINGLE_SQUARED_OPEN_CURB",
        "DOUBLE_ROUND_50CM_GAP",
        "DOUBLE_ROUND_50CM_GAP_OPEN_CURB",
        "DOUBLE_SQUARED_50CM_GAP_OPEN_CURB",
        "T_FULL",
        "T_SHORT",
    ]
)

scenario.add_ego(
    generator=EgoAgentGeneratorParameters(
        agent_type=AgentType.VEHICLE,
        vehicle_spawn_data=VehicleSpawnData(
            vehicle_behavior=VehicleBehavior(
                parking_scenario_goal=PositionRequest(
                    lane_spawn_policy=LaneSpawnPolicy(
                        lane_type=EnumDistribution(
                            probabilities={"ParkingSpace": 1.0},
                        ),
                        road_type=EnumDistribution(
                            probabilities={"Parking_Aisle": 1.0},
                        ),
                        on_road_parking_angle_distribution=EnumDistribution(
                            probabilities={
                                "PERPENDICULAR": 1.0,
                            }
                        ),
                    )
                ),
                parking_scenario_time=ContinousUniformDistribution(
                    min=4.0,
                    max=4.0,
                ),
            )
        ),
    ),
)

scenario.set_environment(
    parameters=EnvironmentParameters(
        parking_space_data=ParkingSpaceData(
            parking_lot_angle_distribution=EnumDistribution(
                probabilities={
                    "PERPENDICULAR": 1.0,
                }
            ),
            lot_parking_delineation_type=EnumDistribution(
                probabilities={
                    parking_space_delineation_type: 1.0,
                }
            ),
            street_parking_delineation_type=EnumDistribution(
                probabilities={
                    parking_space_delineation_type: 1.0,
                }
            ),
            street_parking_angle_zero_override=EnumDistribution(
                probabilities={
                    parking_space_delineation_type: 1.0,
                }
            ),
        )
    )
)

preview_scenario(
    scenario=scenario,
    frames_per_scene=100,
    sim_capture_rate=10,
    sim_instance=SimulationInstance(name="<instance name>"),
    render_instance=RenderInstance(name="<instance name>"),
)
