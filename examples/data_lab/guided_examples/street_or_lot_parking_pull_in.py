import logging
import random
from typing import Tuple

from pd.core import PdError
from pd.data_lab import Scenario, ScenarioCreator, ScenarioSource
from pd.data_lab.config.distribution import CenterSpreadConfig, ContinousUniformDistribution, EnumDistribution
from pd.data_lab.scenario import Lighting

from paralleldomain import data_lab
from paralleldomain.data_lab import Location, SensorConfig, SensorRig
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

"""
In this example script, we create a scenario in which an ego vehicle pulls into a parking space.

This script highlights the use of Pre-Built Generators.

Last revised: 28/Sept/2023
"""


# We create a custom class that inherits from the ScenarioCreator class.  This is where we will provide our scenario
# generation instructions that will instruct our Data Lab instance
class StreetLightOrParkingPullIn(ScenarioCreator):
    # The create_scenario method is where we provide our Data Lab Instance with the scenario generation instructions it
    # requires to create the scenario
    def create_scenario(
        self, random_seed: int, scene_index: int, number_of_scenes: int, location: Location, **kwargs
    ) -> ScenarioSource:
        # We define a simple pinhole camera that is forward facing and sits 2 meters above the bottom surface of the ego
        # vehicle's bounding box
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

        # Initialize a Scenario object with the sensor rig defined above
        scenario = Scenario(sensor_rig=sensor_rig)

        # Set weather variables to be completely free of rain and low in wetness and fog
        scenario.environment.rain.set_constant_value(0.0)
        scenario.environment.fog.set_uniform_distribution(min_value=0.1, max_value=0.3)
        scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

        # Set up parameters that will be used to govern the behavior of the parking ego vehicle
        parking_space_on_street = True  # Will park in parking lots if set to false
        time_to_park = 4  # Time to complete parking maneuver in seconds
        parking_space_type = "PARALLEL"  # Choose from "PERPENDICULAR", "ANGLE_60", "ANGLE_45", "ANGLE_30", "PARALLEL"

        # Parallel parking can only take place on streets. Thus, if the user selects "PARALLEL" as the desired parking
        # slot type, we ensure that parking_space_on_street is set to True and force parking slots in parking lots
        # to have PERPENDICULAR type parking slots
        if parking_space_type == "PARALLEL":
            if not parking_space_on_street:
                raise PdError("Parallel Parking Spaces are only supported on streets")

            lot_space_type = "PERPENDICULAR"
        else:
            lot_space_type = parking_space_type

        # Use the Pre-Built Generator EgoAgentGeneratorParameters to place a ego vehicle in the scenario. As will be
        # covered below, parameters are also passed in to ensure the placed ego vehicle performs a parking maneuver
        scenario.add_ego(
            generator=EgoAgentGeneratorParameters(
                agent_type=AgentType.VEHICLE,
                # Specify VehicleSpawnData, which controls how the ego vehicle is spawned in the scenario
                vehicle_spawn_data=VehicleSpawnData(
                    # Specify the VehicleBehavior which controls how the spawned ego vehicle will behavior throughout
                    # the scenario
                    vehicle_behavior=VehicleBehavior(
                        # The parking_scenario_goal allows us to specify parameters about the type of parking space that
                        # we want the spawned ego vehicle to end up in. Providing a parking_scenario_goal will
                        # automatically ensure that the spawned ego vehicle carries out a parking scenario.
                        # The parking_scenario_goal is specified as a PositionRequest, where we request the position of
                        # a parking slot that we want the ego vehicle to target in the scenario
                        parking_scenario_goal=PositionRequest(
                            # The position request for a parking type behavior should be passed as a LaneSpawnPolicy
                            # that specifies both a lane_type and road_type. We also use this request to control the
                            # types of parking spaces (angle type) that exist on street parking slots
                            lane_spawn_policy=LaneSpawnPolicy(
                                # The lane_type parameter is used to command the ego vehicle to target a parking slot.
                                # As such, it should be set to a "ParkingSpace" at all times when a parking scenario is
                                # desired. All distributions are set as an EnumDistribution
                                lane_type=EnumDistribution(
                                    probabilities={"ParkingSpace": 1.0},
                                ),
                                # The road_type parameter is used to specify they type of road on which the targeted
                                # parking slot should exist.  When the parking slot is desired to be on a road,
                                # "Primary" should be set, when the parking slot is desired to be in a parking lot,
                                # "ParkingAisle" should be set.  All distributions are set as an EnumDistribution
                                road_type=EnumDistribution(
                                    probabilities={"Primary" if parking_space_on_street else "Parking_Aisle": 1.0},
                                ),
                                # In this parameter, we can specify the type of street parking slots that should be
                                # targeted by the ego vehicle.  This parameter is only used when the target parking slot
                                # has been specified to be on a street in the road_type parameter above. The reason
                                # this is controlled within the parking_scenario_goal is because street parking slots
                                # types are not dynamically controlled. All parking slot types exist in a Data Lab
                                # location and the correct type of parking slot is searched for during scenario
                                # generation.
                                on_road_parking_angle_distribution=EnumDistribution(
                                    probabilities={
                                        parking_space_type: 1.0,
                                    }
                                ),
                            )
                        ),
                        # Specifies the time that should be taken to for the ego to begin the parking maneuver. Controls
                        # the distance the ego vehicle travels before making the initial turn into the parking slot
                        parking_scenario_time=ContinousUniformDistribution(
                            min=time_to_park,
                            max=time_to_park,
                        ),
                    )
                ),
            ),
        )

        # Specify Environment Parameters of the Data Lab Scenario.  Here, we control the types of parking slots that
        # exist in parking lots in the Data Lab Location.  Unlike parking slots on roads, parking slots in parking lots
        # can be dynamically adjusted during scenario generation.
        scenario.set_environment(
            # Adjust the EnvironmentParameters of the Data Lab Scenario
            parameters=EnvironmentParameters(
                # Within ParkingSpaceData, the angle distribution of the parking slots that exist in parking lots can be
                # selected. Any parking slot angle can be selected except for "PARALLEL" as parallel parking slots only
                # exist on roads
                parking_space_data=ParkingSpaceData(
                    parking_lot_angle_distribution=EnumDistribution(
                        probabilities={
                            lot_space_type: 1.0,
                        }
                    )
                )
            )
        )

        # Use the Pre-Built Generator TrafficGeneratorParameters to place realistic traffic agents within 100.0 meters
        # of the ego vehicle
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

        # Use the Pre-Built Generator ParkedVehicleGeneratorParameters to place parked cars within 100.0 meters of the
        # ego vehicle
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

        # Return the scenario object
        return scenario

    # The get location method allows us to define the location and lighting of the Data Lab scenario.  In this case,
    # we select an urban map which contains parking lots, as well as a lighting option which corresponds to a partly
    # cloudy day around 8:40 pm.
    def get_location(
        self, random_seed: int, scene_index: int, number_of_scenes: int, **kwargs
    ) -> Tuple[Location, Lighting]:
        return Location(name="SF_6thAndMission_medium"), "dusk_partlyCloudy_02"


if __name__ == "__main__":
    # We use preview_scenario() to visualize the created scenario.  We pass in a fixed seed so that the same scenario is
    # generated every time the script is run.  We also request 200 rendered frames at a frame rate of 10 Hz.
    data_lab.preview_scenario(
        scenario_creator=StreetLightOrParkingPullIn(),
        frames_per_scene=200,
        sim_capture_rate=10,
        random_seed=random.randint(0, 100000),
        instance_name="<instance name>",
    )
