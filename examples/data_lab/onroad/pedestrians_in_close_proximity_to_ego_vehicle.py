import logging
import random
from typing import List

from pd.assets import ObjAssets
from pd.data_lab.config.distribution import CenterSpreadConfig, EnumDistribution
from pd.data_lab.context import setup_datalab
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.sim_instance import SimulationInstance
from pd.internal.proto.keystone.generated.wrapper.pd_sensor_pb2 import CameraIntrinsic, SensorExtrinsic

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.generators.behavior import PedestrianBehavior
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.parked_vehicle import ParkedVehicleGeneratorParameters
from paralleldomain.data_lab.generators.pedestrian import PedestrianGeneratorParameters
from paralleldomain.data_lab.generators.position_request import (
    LaneSpawnPolicy,
    LocationRelativePositionRequest,
    PositionRequest,
)
from paralleldomain.data_lab.generators.spawn_data import PedestrianSpawnData
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.utilities.logging import setup_loggers

setup_loggers(logger_names=["__main__", "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)

setup_datalab("v2.2.0-beta")


def get_character_names() -> List[str]:
    # PD Characters start with prefix "char_" in asset DB
    asset_objs = ObjAssets.select().where(ObjAssets.name % "char_*")
    asset_names = [o.name for o in asset_objs]
    return asset_names


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

character_names = get_character_names()

# Create scenario
scenario = data_lab.Scenario(sensor_rig=sensor_rig)
scenario.random_seed = random.randint(1, 1_000_000)  # set to a fixed integer to keep scenario generation deterministic

# Set weather variables and time of day
scenario.environment.time_of_day.set_category_weight(data_lab.TimeOfDays.Day, 1.0)
scenario.environment.clouds.set_constant_value(0.7)
scenario.environment.rain.set_constant_value(0.8)
scenario.environment.fog.set_uniform_distribution(min_value=0.1, max_value=0.3)
scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

# Select an environment
scenario.set_location(data_lab.Location(name="SF_6thAndMission_medium"))

# Place ourselves in the world
scenario.add_ego(
    generator=EgoAgentGeneratorParameters(
        agent_type=AgentType.VEHICLE,
        position_request=PositionRequest(
            lane_spawn_policy=LaneSpawnPolicy(
                lane_type=EnumDistribution(
                    probabilities={"Drivable": 0.01},
                )
            )
        ),
    ),
)

for i in range(0, 5):
    scenario.add_agents(
        generator=PedestrianGeneratorParameters(
            ped_spawn_data=PedestrianSpawnData(
                ped_behavior=PedestrianBehavior.NORMAL, asset_name=random.choice(character_names)
            ),
            position_request=PositionRequest(
                location_relative_position_request=LocationRelativePositionRequest(
                    max_spawn_radius=0.0,  # Select ego vehicle's center as initial position
                    agent_tags=["EGO"],
                ),
                longitudinal_offset=CenterSpreadConfig(
                    center=random.randint(50, 100) / 10  # now move ped by 5-10meter to the front
                ),
                lateral_offset=CenterSpreadConfig(
                    center=random.randint(-50, 50) / 10  # and then move upto 5 meter to the left or right
                ),
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


data_lab.preview_scenario(
    scenario=scenario,
    frames_per_scene=100,
    sim_capture_rate=10,
    sim_instance=SimulationInstance(name="<instance name>"),
    render_instance=RenderInstance(name="<instance name>"),
)
