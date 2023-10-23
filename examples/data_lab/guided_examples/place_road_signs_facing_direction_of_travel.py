import logging
from typing import Tuple

from pd.data_lab import ScenarioCreator, ScenarioSource
from pd.data_lab.config.distribution import CenterSpreadConfig, EnumDistribution
from pd.data_lab.config.location import Location
from pd.data_lab.scenario import Lighting

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab import DEFAULT_DATA_LAB_VERSION
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


"""
In this example script, we create a scenario in which an ego vehicle drives through and urban environment which is
populated with a large number of road signs.

This script highlights the use of Pre-Built Generators, Custom Behaviors and Custom Agents, database queries as well as
UMD Map lookups.

Last revised: 29/Sept/2023
"""


# We create a custom class that inherits from the ScenarioCreator class.  This is where we will provide our scenario
# generation instructions that will instruct our Data Lab instance
class PlaceRoadSigns(ScenarioCreator):
    # The create_scenario method is where we provide our Data Lab Instance with the scenario generation instructions it
    # requires to create the scenario
    def create_scenario(
        self, random_seed: int, scene_index: int, number_of_scenes: int, location: Location, **kwargs
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

        # Initialize a Scenario object with the sensor rig defined above
        scenario = data_lab.Scenario(sensor_rig=sensor_rig)

        # Set the weather to be completely free of rain and low in wetness and fog
        scenario.environment.rain.set_constant_value(0.0)
        scenario.environment.fog.set_uniform_distribution(min_value=0, max_value=0.1)
        scenario.environment.wetness.set_uniform_distribution(min_value=0.0, max_value=0.2)

        # Use a Pre-Built Generator to place the ego vehicle in the Scenario on a Drivable lane
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

        # Use a Pre-Built Generator to place realistic traffic in the scenario within 50.0 meters of the
        # Ego vehicle
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

        # Use a Pre-Built Generator to place realistic parked vehicles in the scenario within 50.0 meters of the
        # Ego vehicle
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

        # Use a Custom Generator SignGenerator to place signs in the scenario.  Full details can be found in within the
        # generator
        scenario.add_agents(
            generator=SignGenerator(
                num_sign_poles=20,
                random_seed=scenario.random_seed,
                radius=30.0,
                sim_capture_rate=10,
                max_signs_per_pole=3,
                country="Portugal",
                forward_offset_to_place_signs=20.0,
                min_distance_between_signs=1.5,
                single_frame_mode=False,
            )
        )
        return scenario

    # The get location method allows us to define the location and lighting of the Data Lab scenario.  In this case,
    # we select an urban map which contains sidewalks, as well as a lighting option which corresponds to a mostly
    # cloudy day around noon.
    def get_location(
        self, random_seed: int, scene_index: int, number_of_scenes: int, **kwargs
    ) -> Tuple[Location, Lighting]:
        return data_lab.Location(name="SF_6thAndMission_medium"), "day_partlyCloudy_03"


if __name__ == "__main__":
    # We use preview_scenario() to visualize the created scenario.  We pass in a fixed seed so that the same scenario is
    # generated every time the script is run.  We also request 10 rendered frames at a frame rate of 10 Hz.
    data_lab.preview_scenario(
        scenario_creator=PlaceRoadSigns(),
        frames_per_scene=10,
        sim_capture_rate=10,
        random_seed=42,
        instance_name="<instance name>",
        data_lab_version=DEFAULT_DATA_LAB_VERSION,
    )
