import logging
from typing import Tuple

from pd.data_lab import RenderInstance, ScenarioCreator, ScenarioSource
from pd.data_lab.config.distribution import CenterSpreadConfig, EnumDistribution
from pd.data_lab.config.location import Location
from pd.data_lab.scenario import Lighting
from pd.data_lab.sim_instance import SimulationInstance

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, SensorExtrinsic
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


# This class implements our custom Scenario object which inherits from the ScenarioCreator base class
class AnatomyOfDataLabScriptsExample(ScenarioCreator):
    # This method is where we pass our scenario generation instructions to our Data Lab instance
    def create_scenario(
        self, random_seed: int, scene_index: int, number_of_scenes: int, location: Location, **kwargs
    ) -> ScenarioSource:
        # We set up the SensorRIg object here.  In this case, a 1920x1080 pinhole camera
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

        # Create the Scenario object with the sensor rig we defined above
        # This creates the scenario object which we can use to provide scenario generation instructions
        scenario = data_lab.Scenario(sensor_rig=sensor_rig)

        # Specify the weather in the scenario - specifically rain, fog and wetness
        scenario.environment.rain.set_constant_value(0.0)
        scenario.environment.fog.set_uniform_distribution(min_value=0.1, max_value=0.3)
        scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

        # We now specify the scenario generation instructions that specify agent placement in the scenario

        # Place the ego vehicle using EgoAgentGeneratorParameters
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

        # Place general traffic in the scenario using the TrafficGeneratorParameters
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

        # Place parked vehicles in the scenario using the ParkedVehicleGeneratorParameters
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

    # This method specifies the Location in which the scenario will be created, in addition to the Lighting
    # Currently the method is configured to return a single Location and Lighting value but additional logic can be
    # implemented in this method.
    def get_location(
        self, random_seed: int, scene_index: int, number_of_scenes: int, **kwargs
    ) -> Tuple[Location, Lighting]:
        return data_lab.Location(name="SF_6thAndMission_medium"), "day_partlyCloudy_03"


if __name__ == "__main__":
    data_lab.preview_scenario(
        scenario_creator=AnatomyOfDataLabScriptsExample(),
        random_seed=2023,
        frames_per_scene=100,
        sim_capture_rate=10,
        simulator=SimulationInstance(name="<instance name>"),
        renderer=RenderInstance(name="<instance name>"),
        data_lab_version="<version_name>",
    )
