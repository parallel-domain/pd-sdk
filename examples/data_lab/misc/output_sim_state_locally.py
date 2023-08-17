import logging

import numpy as np
from pd.data_lab import Scenario
from pd.data_lab.config.distribution import EnumDistribution, MinMaxConfigInt
from pd.data_lab.context import setup_datalab
from pd.data_lab.sim_instance import SimulationInstance

from paralleldomain.data_lab import Location, SensorConfig, SensorRig, TimeOfDays, save_sim_state_archive
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, SensorExtrinsic
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.peripherals import VehiclePeripheral
from paralleldomain.data_lab.generators.position_request import (
    LaneSpawnPolicy,
    LocationRelativePositionRequest,
    PositionRequest,
)
from paralleldomain.data_lab.generators.random_pedestrian import RandomPedestrianGeneratorParameters
from paralleldomain.data_lab.generators.spawn_data import VehicleSpawnData
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.logging import setup_loggers

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])
logger = logging.getLogger("pd.state.serialize")
logger.setLevel(logging.CRITICAL)

setup_datalab("v2.4.0-beta")

# Set up scene parameters
number_of_scenarios = 2
seed = 16000
scenario_offset = 50  # Scene numbering will start at 0 + <scenario_offset>
number_of_frames = 10
sim_capture_rate = 20

output_location = AnyPath("output")

for scene_number in range(number_of_scenarios):
    seed += 1

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
                    yaw=30.0,
                    x=0.0,
                    y=0.0,
                    z=2.0,
                ),
            )
        ]
    )

    # Create scenario
    scenario = Scenario(sensor_rig=sensor_rig)
    scenario.random_seed = seed

    random = np.random.RandomState(seed=seed)

    weather_bucket_val = random.random_sample()

    scenario.environment.time_of_day.set_category_weight(TimeOfDays.Day, 1.0)
    scenario.environment.clouds.set_constant_value(0.0)
    scenario.environment.rain.set_constant_value(0.0)
    scenario.environment.fog.set_constant_value(0.0)
    scenario.environment.wetness.set_constant_value(0.0)

    # Select an environment
    scenario.set_location(Location(name="SF_6thAndMission_medium"))

    scenario.add_ego(
        generator=EgoAgentGeneratorParameters(
            agent_type=AgentType.VEHICLE,
            ego_model="midsize_sedan_04",
            position_request=PositionRequest(
                lane_spawn_policy=LaneSpawnPolicy(
                    lane_type=EnumDistribution(
                        probabilities={"Drivable": 1.0},
                    )
                )
            ),
            vehicle_spawn_data=VehicleSpawnData(vehicle_peripheral=VehiclePeripheral(disable_occupants=True)),
        ),
    )

    scenario.add_agents(
        generator=TrafficGeneratorParameters(
            spawn_probability=0.5,
            position_request=PositionRequest(
                location_relative_position_request=LocationRelativePositionRequest(
                    agent_tags=["EGO"],
                    max_spawn_radius=100.0,
                )
            ),
        )
    )

    scenario.add_agents(
        generator=RandomPedestrianGeneratorParameters(
            position_request=PositionRequest(
                location_relative_position_request=LocationRelativePositionRequest(
                    agent_tags=["EGO"],
                    max_spawn_radius=100.0,
                )
            ),
            num_of_pedestrians_range=MinMaxConfigInt(
                min=0,
                max=30,
            ),
        )
    )

    save_sim_state_archive(
        scenario=scenario,
        scenario_index=scene_number,
        frames_per_scene=number_of_frames,
        sim_capture_rate=sim_capture_rate,
        yield_every_sim_state=True,
        sim_instance=SimulationInstance(name="<instance name>"),
        scenario_index_offset=scenario_offset,
        output_path=output_location,
    )
