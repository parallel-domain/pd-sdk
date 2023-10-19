import logging
from typing import Callable, Optional, Tuple

from pd.data_lab import ScenarioCreator, ScenarioSource
from pd.data_lab.config.distribution import CenterSpreadConfig, MinMaxConfigInt
from pd.data_lab.context import load_map
from pd.data_lab.scenario import Lighting

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab import preview_scenario
from paralleldomain.data_lab.config.map import MapQuery
from paralleldomain.data_lab.config.types import Float3
from paralleldomain.data_lab.generators.parked_vehicle import ParkedVehicleGeneratorParameters
from paralleldomain.data_lab.generators.position_request import (
    AbsolutePositionRequest,
    LocationRelativePositionRequest,
    PositionRequest,
)
from paralleldomain.data_lab.generators.random_pedestrian import RandomPedestrianGeneratorParameters
from paralleldomain.data_lab.generators.spawn_data import AgentSpawnData, VehicleSpawnData
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.data_lab.generators.vehicle import VehicleGeneratorParameters
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.utilities.transformation import Transformation

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


class EgoDroneStraightLineBehavior(data_lab.CustomSimulationAgentBehavior):
    def __init__(self, start_pose: Transformation, target_pose: Transformation, flight_time: float):
        super().__init__()
        self._initial_pose: Transformation = start_pose
        self._target_pose: Transformation = target_pose
        self._flight_time: float = flight_time
        self._start_time: Optional[float] = None

    def set_initial_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        agent.set_pose(pose=self._initial_pose.transformation_matrix)

    def update_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        raycast: Optional[Callable] = None,
    ):
        current_time = sim_state.sim_time

        if self._start_time is None:
            self._start_time = current_time  # set first frame as start time even if not exactly 0.0 seconds

        flight_completion = current_time / (self._flight_time + self._start_time)

        interpolated_pose = Transformation.interpolate(
            tf0=self._initial_pose, tf1=self._target_pose, factor=flight_completion
        )

        logger.info(f"Using interpolated pose: {interpolated_pose}")
        agent.set_pose(pose=interpolated_pose.transformation_matrix)

    def clone(self) -> "EgoDroneStraightLineBehavior":
        return EgoDroneStraightLineBehavior(
            start_pose=self._initial_pose, target_pose=self._target_pose, flight_time=self._flight_time
        )


class RandomStartTargetFlight(ScenarioCreator):
    def create_scenario(
        self, random_seed: int, scene_index: int, number_of_scenes: int, location: data_lab.Location, **kwargs
    ) -> ScenarioSource:
        sensor_rig = data_lab.SensorRig().add_camera(
            name="Front",
            width=768,
            height=768,
            field_of_view_degrees=70,
            pose=Transformation.from_euler_angles(
                angles=[-30, 0.0, 0.0], order="xyz", degrees=True, translation=[0.0, 0.0, 0.0]
            ),
            annotation_types=[AnnotationTypes.SemanticSegmentation2D],
        )

        # Create scenario
        scenario = data_lab.Scenario(sensor_rig=sensor_rig)

        # Set weather variables and time of day
        scenario.environment.rain.set_constant_value(0.0)
        scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

        # Load map locally to find a random spawn point and its XYZ coordinates
        # this could be done in the EgoDroneBehavior itself, but we need to pass the
        # XYZ coordinates to PD generators, so we do it outside.
        umd_map = load_map(location)
        map_query = MapQuery(umd_map)

        start_pose = map_query.get_random_street_location(random_seed=random_seed)
        start_pose.translation[
            2
        ] += 5.0  # map query gives us ground position, but we want our Drone to start 5m above ground

        target_pose = map_query.get_random_street_location(random_seed=random_seed + 1)
        target_pose.translation[
            2
        ] += 7.5  # map query gives us ground position, but we want our Drone to start above ground

        flight_time = 10  # in seconds

        # Place ourselves in the world through a custom simulation agent.
        # Don't use an asset so we don't see anything flying attach our EgoDroneBehavior from above
        scenario.add_ego(
            data_lab.CustomSimulationAgents.create_ego_vehicle(
                sensor_rig=sensor_rig,
                asset_name="",
                lock_to_ground=False,
            ).set_behavior(
                EgoDroneStraightLineBehavior(start_pose=start_pose, target_pose=target_pose, flight_time=flight_time)
            )
        )

        # Place a "star vehicle" with absolute position below us that can be used as an "anchor point"
        # for location relative position request with other agents
        scenario.add_agents(
            generator=VehicleGeneratorParameters(
                model="suv_medium_02",
                vehicle_spawn_data=VehicleSpawnData(agent_spawn_data=AgentSpawnData(tags=["STAR"])),
                position_request=PositionRequest(
                    absolute_position_request=AbsolutePositionRequest(
                        position=Float3(
                            x=start_pose.translation[0], y=start_pose.translation[1]
                        ),  # just provide XY, Z is being resolved by simulation
                        resolve_z=True,
                    )
                ),
            )
        )

        # Add other agents
        scenario.add_agents(
            generator=TrafficGeneratorParameters(
                spawn_probability=0.9,
                position_request=PositionRequest(
                    location_relative_position_request=LocationRelativePositionRequest(
                        agent_tags=["STAR"],  # anchor around star vehicle
                        max_spawn_radius=100.0,
                    )
                ),
            )
        )
        #
        scenario.add_agents(
            generator=ParkedVehicleGeneratorParameters(
                spawn_probability=CenterSpreadConfig(center=0.5),
                position_request=PositionRequest(
                    location_relative_position_request=LocationRelativePositionRequest(
                        agent_tags=["STAR"],  # anchor around star vehicle
                        max_spawn_radius=100.0,
                    )
                ),
            )
        )

        scenario.add_agents(
            generator=RandomPedestrianGeneratorParameters(
                num_of_pedestrians_range=MinMaxConfigInt(min=3, max=20),
                position_request=PositionRequest(
                    location_relative_position_request=LocationRelativePositionRequest(
                        agent_tags=["STAR"],  # anchor around star vehicle
                        max_spawn_radius=100.0,
                    )
                ),
            )
        )

        return scenario

    def get_location(
        self, random_seed: int, scene_index: int, number_of_scenes: int, **kwargs
    ) -> Tuple[data_lab.Location, Lighting]:
        return data_lab.Location(name="SF_6thAndMission_medium"), "day_partlyCloudy_03"


if __name__ == "__main__":
    preview_scenario(
        scenario_creator=RandomStartTargetFlight(),
        random_seed=1995,
        frames_per_scene=100,
        sim_capture_rate=10,
        instance_name="<instance name>",
    )
