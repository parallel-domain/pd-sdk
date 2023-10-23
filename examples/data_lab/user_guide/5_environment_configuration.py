import logging
import random
from typing import Tuple

from pd.data_lab import ScenarioCreator, ScenarioSource
from pd.data_lab.config.distribution import CenterSpreadConfig, EnumDistribution
from pd.data_lab.config.location import Location
from pd.data_lab.scenario import Lighting

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, SensorExtrinsic
from paralleldomain.data_lab.config.world import EnvironmentParameters, ParkingSpaceData
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.parked_vehicle import ParkedVehicleGeneratorParameters
from paralleldomain.data_lab.generators.position_request import (
    LaneSpawnPolicy,
    LocationRelativePositionRequest,
    PositionRequest,
)
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.utilities.logging import setup_loggers

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)


# Create our custom ScenarioCreator class
class EnvironmentConfigExample(ScenarioCreator):
    def create_scenario(
        self, random_seed: int, scene_index: int, number_of_scenes: int, location: Location, **kwargs
    ) -> ScenarioSource:
        # Set up a simple pinhole camera
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

        # Create our scenario object
        scenario = data_lab.Scenario(sensor_rig=sensor_rig)

        # This is where we set the "weather" component of our scenario's Environment.  This includes rain, fog and
        # wetness
        scenario.environment.rain.set_constant_value(0.0)
        scenario.environment.fog.set_uniform_distribution(min_value=0.5, max_value=0.9)
        scenario.environment.wetness.set_uniform_distribution(min_value=0.2, max_value=0.6)

        # This is how we set "Environment Parameters".  In this case, we set parameters related to parking spaces in
        # our Data Lab scenario
        scenario.set_environment(
            parameters=EnvironmentParameters(
                parking_space_data=ParkingSpaceData(
                    # Set the angle of parking slots that exist in parking lots of the Location
                    parking_lot_angle_distribution=EnumDistribution(
                        probabilities={
                            "PERPENDICULAR": 1.0,
                        }
                    ),
                    # Set the delineation type of parking slots that exist in parking lots of the Location
                    lot_parking_delineation_type=EnumDistribution(
                        probabilities={
                            "BOX_CLOSED": 1.0,
                        }
                    ),
                    # Set the delineation type of parking slots that exist on the street of the Location
                    street_parking_delineation_type=EnumDistribution(
                        probabilities={
                            "BOX_CLOSED": 1.0,
                        }
                    ),
                )
            )
        )

        # Place the ego vehicle
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

        # Place traffic in the scenario
        scenario.add_agents(
            generator=TrafficGeneratorParameters(
                spawn_probability=0.8,
                position_request=PositionRequest(
                    location_relative_position_request=LocationRelativePositionRequest(
                        agent_tags=["EGO"],
                        max_spawn_radius=100.0,
                    )
                ),
            )
        )

        # Place parked cars in the scenario
        scenario.add_agents(
            generator=ParkedVehicleGeneratorParameters(
                spawn_probability=CenterSpreadConfig(center=0.6),
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
    ) -> Tuple[Location, Lighting]:
        return data_lab.Location(name="SF_6thAndMission_medium"), "day_partlyCloudy_03"


if __name__ == "__main__":
    # Preview the scenario
    data_lab.preview_scenario(
        scenario_creator=EnvironmentConfigExample(),
        random_seed=2023,
        frames_per_scene=100,
        sim_capture_rate=10,
        instance_name="<instance name>",
    )
