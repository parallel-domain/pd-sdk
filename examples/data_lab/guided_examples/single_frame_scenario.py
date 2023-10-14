import logging
import random
from typing import Tuple

from pd.data_lab import ScenarioCreator, ScenarioSource
from pd.data_lab.scenario import Lighting

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.config.map import LaneSegment
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, SensorExtrinsic
from paralleldomain.data_lab.generators.single_frame import (
    SingleFrameEgoGenerator,
    SingleFrameNonEgoVehicleGenerator,
    SingleFramePedestrianGenerator,
    SingleFrameVehicleBehaviorType,
)
from paralleldomain.utilities.logging import setup_loggers

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)

"""
In this example script, we create a scenario single frame scenarios. Single frame scenarios are scenarios in which
agents are moved to a completely different part of the location during every rendered frame.  This allows for a much
higher amount of diversity in your scenario when temporal consistency does not matter.

Single Frame Scenarios must be implemented using Custom Generators.

This script highlights the use of Custom Generators, Custom Behaviors, Custom Agents and UMD Lookups.

Last revised: 29/Sept/2023
"""


# We create a custom class that inherits from the ScenarioCreator class.  This is where we will provide our scenario
# generation instructions that will instruct our Data Lab instance
class SingleFrame(ScenarioCreator):
    # The create_scenario method is where we provide our Data Lab Instance with the scenario generation instructions it
    # requires to create the scenario
    def create_scenario(
        self, random_seed: int, scene_index: int, number_of_scenes: int, location: data_lab.Location, **kwargs
    ) -> ScenarioSource:
        # Implement a simple 1920x1080 pinhole camera facing forwards, 2.0 meters above the road
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
                        yaw=00.0,
                        x=0.0,
                        y=0.0,
                        z=2.0,
                    ),
                )
            ]
        )

        # Initialize a Scenario object with the sensor rig defined above
        scenario = data_lab.Scenario(sensor_rig=sensor_rig)

        # Set the weather to be completely free of rain and low in rain and wetness
        scenario.environment.rain.set_constant_value(0.0)
        scenario.environment.fog.set_uniform_distribution(min_value=0.1, max_value=0.3)
        scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

        # Place the ego vehicle in the world using the Custom Generator SingleFrameEgoGenerator. Full details can be
        # found in the source file
        scenario.add_ego(
            generator=SingleFrameEgoGenerator(
                lane_type=LaneSegment.LaneType.DRIVABLE,
                ego_asset_name="suv_medium_02",
                random_seed=scenario.random_seed,
                sensor_rig=sensor_rig,
            )
        )

        # Place traffic in the scenario using the Custom Generator SingleFrameNonEgoVehicleGenerator. Full details can
        # be found in the source file
        scenario.add_agents(
            generator=SingleFrameNonEgoVehicleGenerator(
                number_of_vehicles=5,
                random_seed=scenario.random_seed,
                spawn_radius=70.0,
                vehicle_behavior_type=SingleFrameVehicleBehaviorType.TRAFFIC,
            )
        )

        # Place traffic in the scenario using the Custom Generator SingleFrameNonEgoVehicleGenerator. Full details can
        # be found in the source file
        scenario.add_agents(
            generator=SingleFrameNonEgoVehicleGenerator(
                number_of_vehicles=5,
                random_seed=scenario.random_seed,
                spawn_radius=70.0,
                vehicle_behavior_type=SingleFrameVehicleBehaviorType.PARKED,
            )
        )

        # Place traffic in the scenario using the Custom Generator SingleFramePedestrianGenerator. Full details can be
        # found in the source file
        scenario.add_agents(
            generator=SingleFramePedestrianGenerator(
                num_of_pedestrians=5,
                random_seed=scenario.random_seed,
                spawn_radius=30.0,
                max_lateral_offset=1.0,
                max_rotation_offset_degrees=30.0,
            )
        )

        return scenario

    # The get location method allows us to define the location and lighting of the Data Lab scenario.  In this case,
    # we select an urban map, as well as a lighting option which corresponds to a mostly
    # cloudy day around noon.
    def get_location(
        self, random_seed: int, scene_index: int, number_of_scenes: int, **kwargs
    ) -> Tuple[data_lab.Location, Lighting]:
        return data_lab.Location(name="SF_6thAndMission_medium"), "LS_sky_noon_mostlyCloudy_1205_HDS001"


if __name__ == "__main__":
    # We use preview_scenario() to visualize the created scenario.  We pass in a fixed seed so that the same scenario is
    # generated every time the script is run.  We also request 20 rendered frames at a frame rate of 1 Hz.
    data_lab.preview_scenario(
        scenario_creator=SingleFrame(),
        frames_per_scene=100,
        sim_capture_rate=10,
        random_seed=random.randint(0, 100000),
        instance_name="<instance name>",
    )
