import dataclasses
import logging
from typing import Callable, List, Optional

import numpy as np
from pd.data_lab.config.distribution import Distribution
from pd.data_lab.context import setup_datalab
from pd.data_lab.render_instance import RenderInstance
from pd.data_lab.sim_instance import SimulationInstance

import paralleldomain.data_lab as data_lab
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.utilities.transformation import Transformation

setup_loggers(logger_names=["__main__", "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)

setup_datalab("v2.1.0-beta")


class BlockEgoBehaviour(data_lab.CustomSimulationAgentBehaviour):
    def __init__(self, vertical_offset: float = 0.0, dist_to_ego: float = 5.0):
        super().__init__()
        self.dist_to_ego = dist_to_ego
        self.vertical_offset = vertical_offset

    def set_initial_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        pos_in_ego_coords = data_lab.coordinate_system.forward * self.dist_to_ego
        vert_offset = data_lab.coordinate_system.left * self.vertical_offset
        pos_in_ego_coords += vert_offset
        pose_in_front_of_ego = sim_state.ego_pose @ pos_in_ego_coords
        pose = Transformation(translation=pose_in_front_of_ego)
        agent.set_pose(pose=pose.transformation_matrix)

    def update_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        raycast: Optional[Callable] = None,
    ):
        pass

    def clone(self) -> "BlockEgoBehaviour":
        return BlockEgoBehaviour(
            vertical_offset=self.vertical_offset,
            dist_to_ego=self.dist_to_ego,
        )


class StackBehaviour(data_lab.CustomSimulationAgentBehaviour):
    def __init__(self, stack_target_id: int, height: float):
        super().__init__()
        self.stack_target_id = stack_target_id
        self.height = height

    def set_initial_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        stack_on_agent = sim_state.get_agent(agent_id=self.stack_target_id, on_current_frame=True)
        if stack_on_agent is not None:
            on_agent_pose = Transformation.from_transformation_matrix(stack_on_agent.pose, approximate_orthogonal=True)
            pos_in_local_coords = data_lab.coordinate_system.up * self.height
            pose_on_top = on_agent_pose @ pos_in_local_coords
            pose = Transformation(translation=pose_on_top)
            agent.set_pose(pose=pose.transformation_matrix)

    def update_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        raycast: Optional[Callable] = None,
    ):
        pass

    def clone(self) -> "StackBehaviour":
        return StackBehaviour(
            stack_target_id=self.stack_target_id,
            height=self.height,
        )


class StackAndFlyBehaviour(StackBehaviour):
    def __init__(self, stack_target_id: int, height: float, speed: float, direction_seed: int):
        super().__init__(stack_target_id=stack_target_id, height=height)
        state = np.random.RandomState(direction_seed)
        forward = data_lab.coordinate_system.forward * state.random()
        left = data_lab.coordinate_system.left * state.random()
        self._direction = forward + left
        self._direction /= np.linalg.norm(self._direction)
        self._speed = speed
        self.direction_seed = direction_seed

    def set_initial_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        super().set_initial_state(sim_state=sim_state, agent=agent, random_seed=random_seed, raycast=raycast)

    def update_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        raycast: Optional[Callable] = None,
    ):
        pose = Transformation.from_transformation_matrix(agent.pose, approximate_orthogonal=True)
        global_pose = pose @ (self._speed * sim_state.time_since_last_frame * self._direction)
        agent.set_pose(
            pose=Transformation.from_euler_angles(
                order="xyz", angles=[0.0, 0.0, self._speed * sim_state.sim_time], translation=global_pose
            ).transformation_matrix
        )

    def clone(self) -> "StackAndFlyBehaviour":
        return StackAndFlyBehaviour(
            stack_target_id=self.stack_target_id,
            height=self.height,
            speed=self._speed,
            direction_seed=self.direction_seed,
        )


@dataclasses.dataclass
class CustomObstacleGenerator(data_lab.CustomAtomicGenerator):
    asset_name: str = "SM_primitive_box_1m"
    number_of_agents: Distribution = Distribution.create(value=0)
    distance_to_ego: Distribution = Distribution.create(value=15.0)
    vertical_offset: Distribution = Distribution.create(value=(-0.2, 0.2))

    def create_agents_for_new_scene(
        self, state: data_lab.ExtendedSimState, random_seed: int
    ) -> List[data_lab.CustomSimulationAgent]:
        agents = []
        num_agents = int(self.number_of_agents.sample(random_seed=random_seed))
        width, length, height = data_lab.CustomSimulationAgents.get_asset_size(asset_name=self.asset_name)
        bottom_agent = data_lab.CustomSimulationAgents.create_object(asset_name=self.asset_name).set_behaviour(
            BlockEgoBehaviour(
                dist_to_ego=self.distance_to_ego.sample(random_seed=random_seed),
                vertical_offset=self.vertical_offset.sample(random_seed=random_seed),
            )
        )
        agents.append(bottom_agent)
        for i in range(num_agents):
            agent = data_lab.CustomSimulationAgents.create_object(
                asset_name=self.asset_name, lock_to_ground=False
            ).set_behaviour(
                StackAndFlyBehaviour(
                    stack_target_id=bottom_agent.step_agent.id,
                    height=(i + 1) * height,
                    direction_seed=random_seed + i,
                    speed=(i + 1) / 3,
                )
            )
            agents.append(agent)
        return agents

    def clone(self) -> "CustomObstacleGenerator":
        return CustomObstacleGenerator(
            asset_name=self.asset_name,
            number_of_agents=self.number_of_agents.clone(),
            distance_to_ego=self.distance_to_ego.clone(),
            vertical_offset=self.vertical_offset.clone(),
        )


class StreetCreepBehaviour(data_lab.CustomSimulationAgentBehaviour):
    def __init__(
        self,
        relative_location_variance: float,
        direction_variance_in_degrees: float,
        speed: float = 5.0,
    ):
        super().__init__()
        self.speed = speed
        self._initial_pose: Optional[Transformation] = None
        self.relative_location_variance = relative_location_variance
        self.direction_variance_in_degrees = direction_variance_in_degrees

    def set_initial_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        pose = sim_state.map_query.get_random_street_location(
            relative_location_variance=self.relative_location_variance,
            direction_variance_in_degrees=self.direction_variance_in_degrees,
            random_seed=random_seed,
        )
        self._initial_pose = pose
        agent.set_pose(pose=pose.transformation_matrix)

    def update_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        raycast: Optional[Callable] = None,
    ):
        distance = self.speed * sim_state.sim_time
        pos_in_ego_coords = data_lab.coordinate_system.forward * distance
        new_translation = self._initial_pose @ pos_in_ego_coords
        new_pose = Transformation(quaternion=self._initial_pose.quaternion, translation=new_translation)
        agent.set_pose(pose=new_pose.transformation_matrix)

    def clone(self) -> "StreetCreepBehaviour":
        return StreetCreepBehaviour(
            speed=self.speed,
            direction_variance_in_degrees=self.direction_variance_in_degrees,
            relative_location_variance=self.relative_location_variance,
        )


sensor_rig = data_lab.SensorRig().add_camera(
    name="Front",
    width=1080,
    height=1920,
    field_of_view_degrees=100,
    pose=Transformation.from_euler_angles(
        angles=[0.0, 0.0, 0.0], order="xyz", degrees=True, translation=[0.0, 0.0, 2.0]
    ),
    annotation_types=[],
)


# Create scenario
scenario = data_lab.Scenario(sensor_rig=sensor_rig)
scenario.random_seed = 133

# Set weather variables and time of day
scenario.environment.time_of_day.set_category_weight(data_lab.TimeOfDays.Day, 1.0)
scenario.environment.clouds.set_constant_value(0.5)
scenario.environment.rain.set_constant_value(0.1)
scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)


# Select an environment
scenario.set_location(data_lab.Location(name="SF_6thAndMission_medium", version="v2.1.0-beta"))


scenario.add_ego(
    data_lab.CustomSimulationAgents.create_ego_vehicle(sensor_rig=sensor_rig).set_behaviour(
        StreetCreepBehaviour(relative_location_variance=0.0, direction_variance_in_degrees=0.0, speed=0.5)
    )
)

custom_gen = CustomObstacleGenerator()
custom_gen.number_of_agents.set_constant_value(5)
custom_gen.asset_name = "portapotty_01"
scenario.add_agents(custom_gen)


data_lab.preview_scenario(
    scenario=scenario,
    sim_instance=SimulationInstance(address="ssl://sim.step-api-dev.paralleldomain.com:30XX"),
    render_instance=RenderInstance(address="ssl://ig.step-api-dev.paralleldomain.com:30XX"),
    frames_per_scene=100,
    sim_capture_rate=10,
)
