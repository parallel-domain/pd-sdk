import logging
from typing import Tuple

from pd.data_lab import ScenarioCreator, ScenarioSource
from pd.data_lab.config.distribution import CenterSpreadConfig, EnumDistribution
from pd.data_lab.scenario import Lighting

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.parked_vehicle import ParkedVehicleGeneratorParameters
from paralleldomain.data_lab.generators.peripherals import VehiclePeripheral
from paralleldomain.data_lab.generators.position_request import (
    LaneSpawnPolicy,
    LocationRelativePositionRequest,
    PositionRequest,
)
from paralleldomain.data_lab.generators.spawn_data import VehicleSpawnData
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.utilities.transformation import Transformation

"""
    This script goes through a demonstration of how to set up pinhole cameras in Data Lab.

    In setting up pinhole cameras, we must pass in the following parameters
        name - Name of the camera
        width - The pixel width of the image
        height - The pixel height of the image
        field_of_view_degrees - The field of view of the pinhole camera in degrees
        pose - Transformation object containing the position and rotation of the camera

    In this example script, we will be demonstrating four examples of pinhole cameras:
        Pinhole_Front - A 1920 x 1080 pinhole camera with 90 degrees field of view facing the frontward from the ego
            vehicle
        Pinhole_Left - A 1920 x 1080 pinhole camera with 90 degrees field of view facing the leftward from the ego
            vehicle
        Pinhole_Right - A 1920 x 1080 pinhole camera with 90 degrees field of view facing the rightward from the ego
            vehicle
        Pinhole_Rear - A 1920 x 1080 pinhole camera with 90 degrees field of view facing the rearward from the ego
            vehicle
"""

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)


# Set up our custom ScenarioCreator object
class PinholeCam(ScenarioCreator):
    def create_scenario(
        self, random_seed: int, scene_index: int, number_of_scenes: int, location: data_lab.Location, **kwargs
    ) -> ScenarioSource:
        # Set up a pinhole camera
        sensor_rig = data_lab.SensorRig().add_camera(
            name="Pinhole_Front",
            width=1920,
            height=1080,
            field_of_view_degrees=90,
            pose=Transformation.from_euler_angles(
                angles=[0.0, 0.0, 0.0],
                order="xyz",
                translation=[0.0, 0.0, 1.5],
                degrees=True,
            ),
        )

        # Add a left facing pinhole camera to the sensor rig
        sensor_rig.add_camera(
            name="Pinhole_Left",
            width=1920,
            height=1080,
            field_of_view_degrees=90,
            pose=Transformation.from_euler_angles(
                angles=[0.0, 0.0, 90.0],
                order="xyz",
                translation=[0.0, 0.0, 1.5],
                degrees=True,
            ),
        )

        # Add a right facing pinhole camera to the sensor rig
        sensor_rig.add_camera(
            name="Pinhole_Right",
            width=1920,
            height=1080,
            field_of_view_degrees=90,
            pose=Transformation.from_euler_angles(
                angles=[0.0, 0.0, -90.0],
                order="xyz",
                translation=[0.0, 0.0, 1.5],
                degrees=True,
            ),
        )

        # Add a rear facing pinhole camera to the sensor rig
        sensor_rig.add_camera(
            name="Pinhole_Rear",
            width=1920,
            height=1080,
            field_of_view_degrees=90,
            pose=Transformation.from_euler_angles(
                angles=[0.0, 0.0, 180.0],
                order="xyz",
                translation=[0.0, 0.0, 1.5],
                degrees=True,
            ),
        )

        # Create our Scenario object
        scenario = data_lab.Scenario(sensor_rig=sensor_rig)

        # Set weather in scenario
        scenario.environment.rain.set_constant_value(0.0)
        scenario.environment.fog.set_uniform_distribution(min_value=0.1, max_value=0.3)
        scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

        # Place ego agent
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
                vehicle_spawn_data=VehicleSpawnData(
                    vehicle_peripheral=VehiclePeripheral(
                        disable_occupants=True,
                    )
                ),
            ),
        )

        # Place traffic in the scenario
        scenario.add_agents(
            generator=TrafficGeneratorParameters(
                spawn_probability=0.4,
                position_request=PositionRequest(
                    location_relative_position_request=LocationRelativePositionRequest(
                        agent_tags=["EGO"],
                        max_spawn_radius=100.0,
                    )
                ),
            )
        )

        # Place parked vehicle in the scenario
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
        return scenario

    def get_location(
        self, random_seed: int, scene_index: int, number_of_scenes: int, **kwargs
    ) -> Tuple[data_lab.Location, Lighting]:
        return data_lab.Location(name="SF_6thAndMission_medium"), "LS_sky_noon_mostlyCloudy_1205_HDS001"


if __name__ == "__main__":
    # Render our scenario
    data_lab.preview_scenario(
        scenario_creator=PinholeCam(),
        frames_per_scene=10,
        sim_capture_rate=10,
        random_seed=1000,
        instance_name="<instance name>",
    )
