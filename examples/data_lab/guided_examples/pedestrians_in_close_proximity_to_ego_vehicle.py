import logging
import random
from typing import List, Tuple

from pd.assets import ObjAssets
from pd.data_lab import Scenario, ScenarioCreator, ScenarioSource
from pd.data_lab.config.distribution import CenterSpreadConfig, EnumDistribution
from pd.data_lab.context import setup_datalab
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.scenario import Lighting
from pd.data_lab.sim_instance import SimulationInstance

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, SensorExtrinsic
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

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)

"""
In this example script, we create a scenario in which we create a driving scenario with pedestrians placed near the
vehicle.

This script highlights the use of Pre-Built Generators and database queries.

Last revised: 29/Sept/2023
"""


# This method performs a Data Lab database look up to find the names of character assets
def get_character_names() -> List[str]:
    # To search for character assets, we will simply look for assets which have names beginning with the "char_" prefix.
    # To do this, we use the ObjAssets table and retrieve the name column
    asset_objs = ObjAssets.select().where(ObjAssets.name % "char_*")

    # Store the names of assets that have been extracted in a list
    asset_names = [o.name for o in asset_objs]

    # Return the list of character asset names for use later on in the script
    return asset_names


# We create a custom class that inherits from the ScenarioCreator class.  This is where we will provide our scenario
# generation instructions that will instruct our Data Lab instance.
class PedsNearEgo(ScenarioCreator):
    # The create_scenario method is where we provide our Data Lab Instance with the scenario generation instructions it
    # requires to create the scenario
    def create_scenario(
        self, random_seed: int, scene_index: int, number_of_scenes: int, location: data_lab.Location, **kwargs
    ) -> ScenarioSource:
        # We define a simple pinhole camera that is forward facing and sits 2 meters above the bottom surface of the ego
        # vehicle's bounding box
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

        # Use the method implemented above to query the Data Lab database for the asset names of characters
        character_names = get_character_names()

        # Initialize a Scenario object with the sensor rig defined above
        scenario = data_lab.Scenario(sensor_rig=sensor_rig)

        # Set weather to be rainy but low in fog and wetness
        scenario.environment.rain.set_constant_value(0.8)
        scenario.environment.fog.set_uniform_distribution(min_value=0.1, max_value=0.3)
        scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

        # Place the ego vehicle in the work using the Pre-Built Generator EgoAgentGeneratorParameters. We request that
        # the ego vehicle is placed on a Drivable lane
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

        # In this section, we will use the Pre-Built Generator PedestrianGeneratorParameters to place five pedestrians
        # near the ego vehicle.  To do this, we will use a for loop to call the Pre-Built Generator 5 times, and use a
        # LocationRelativePositionRequest to request placement of the pedestrian in a position relative to the Ego
        # vehicle
        for i in range(0, 5):
            # When calling PedestrianGeneratorParameters to place a pedestrian, we specify that the pedestrian behavior
            # should be "NORMAL" and we randomly choose a character asset name from those which we selected above from
            # the Data Lab asset database.

            # In addition, we use a LocationRelativePositionRequest to specify that the pedestrian should be spawned
            # near the ego vehicle but use the longitudinal_offset and lateral_offset parameters to vary the poition of
            # the spawned pedestrian
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

        # We use the Pre-Built Generator TrafficGeneratorParameters to place realistic traffic agents within 100.0
        # meters of the EGO vehicle
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

        # We use the Pre-Built Generator ParkedVehicleGeneratorParameters to place realistic parked vehice agents
        # within 100.0 meters of the EGO vehicle
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

        # The get location method allows us to define the location and lighting of the Data Lab scenario.  In this case,
        # we select an urban map, as well as a lighting option which corresponds to a mostly
        # cloudy day around noon.
        return scenario

    def get_location(
        self, random_seed: int, scene_index: int, number_of_scenes: int, **kwargs
    ) -> Tuple[data_lab.Location, Lighting]:
        return data_lab.Location(name="SF_6thAndMission_medium"), "day_partlyCloudy_03"


if __name__ == "__main__":
    data_lab.preview_scenario(
        scenario_creator=PedsNearEgo(),
        frames_per_scene=100,
        sim_capture_rate=10,
        random_seed=random.randint(0, 100000),
        instance_name="<instance name>",
    )
