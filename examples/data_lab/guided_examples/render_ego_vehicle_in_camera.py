import logging
from typing import Tuple

from pd.data_lab import ScenarioCreator, ScenarioSource
from pd.data_lab.config.distribution import EnumDistribution
from pd.data_lab.config.location import Location
from pd.data_lab.scenario import Lighting

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters, RenderEgoGenerator
from paralleldomain.data_lab.generators.position_request import LaneSpawnPolicy, PositionRequest
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.utilities.transformation import Transformation

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)

"""
In this example script, we create a scenario in which an ego vehicle drives through a suburban environment.  The ego
vehicle will be rendered in this scenario.

This script highlights the use of Pre-Built Generators, Custom Behaviors and Custom Agents.

Last revised: 29/Sept/2023
"""


# We create a custom class that inherits from the ScenarioCreator class.  This is where we will provide our scenario
# generation instructions that will instruct our Data Lab instance
class EgoVehicleInCamera(ScenarioCreator):
    # The create_scenario method is where we provide our Data Lab Instance with the scenario generation instructions it
    # requires to create the scenario
    def create_scenario(
        self, random_seed: int, scene_index: int, number_of_scenes: int, location: Location, **kwargs
    ) -> ScenarioSource:
        # Create a simple 1920x1080 pinhole camera. In order to demonstrate the rendering of the ego vehicle, we offset
        # the sensor to the left by 4.0 meters and locate it 0.5 meters above the round.  We also rotate it leftwards
        # (about the z axis) by 70 degrees
        sensor_rig = data_lab.SensorRig(
            sensor_configs=[
                data_lab.SensorConfig.create_camera_sensor(
                    name="Front",
                    width=1920,
                    height=1080,
                    field_of_view_degrees=70.0,
                    pose=Transformation.from_euler_angles(
                        angles=[0.0, 0.0, 70.0], order="xyz", degrees=True, translation=[4.0, 0.0, 0.5]
                    ),
                )
            ],
        )

        # Initialize a Scenario object with the sensor rig defined above
        scenario = data_lab.Scenario(sensor_rig=sensor_rig)

        # Set weather to be completely dry with minimal fog and wetness
        scenario.environment.rain.set_constant_value(0.0)
        scenario.environment.fog.set_uniform_distribution(min_value=0.1, max_value=0.3)
        scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

        # Use a Pre-Built Generator to place an Ego Vehicle in the scenario on a drivable lane
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

        # Use a Custom Generator to render the ego vehicle. Full details on the Custom Generator can be found in the
        # source file.
        scenario.add_agents(
            generator=RenderEgoGenerator(
                ego_asset_name="suv_medium_02",
            ),
        )

        # Return the scenario object
        return scenario

    # The get location method allows us to define the location and lighting of the Data Lab scenario.  In this case,
    # we select as suburban map, as well as a lighting option which corresponds to a mostly
    # cloudy day around noon.
    def get_location(
        self, random_seed: int, scene_index: int, number_of_scenes: int, **kwargs
    ) -> Tuple[Location, Lighting]:
        return Location(name="A2_BurnsPark"), "day_partlyCloudy_03"


if __name__ == "__main__":
    # We use preview_scenario() to visualize the created scenario.  We pass in a fixed seed so that the same scenario is
    # generated every time the script is run.  We also request 20 rendered frames at a frame rate of 1 Hz.
    data_lab.preview_scenario(
        scenario_creator=EgoVehicleInCamera(),
        frames_per_scene=20,
        sim_capture_rate=100,
        random_seed=42,
        instance_name="<instance-name>",
    )
