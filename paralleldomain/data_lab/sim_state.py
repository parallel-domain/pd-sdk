from functools import lru_cache
from typing import Optional

import numpy as np
import pd.state
from pd.data_lab.sim_state import SimState
from pd.state.state import PosedAgent

from paralleldomain.data_lab.config.map import MapQuery
from paralleldomain.model.geometry.bounding_box_3d import BoundingBox3DGeometry
from paralleldomain.model.occupancy import OccupancyGrid
from paralleldomain.utilities.coordinate_system import SIM_TO_INTERNAL, CoordinateSystem
from paralleldomain.utilities.transformation import Transformation
from paralleldomain.data_lab import CustomSimulationAgent


class ExtendedSimState(SimState):
    """
    A class to represent the Simulation State of a Data Lab scenario at a particular timestep

    Attributes:
        ego_agent_id: The agent id of the ego asset in the scenario
        frame_count:  The frame number of the current simulation state within the scenario
        sim_time:  The time (in milliseconds) from the beginning of the scenario of the current simulation state
        frame_ids: Integer frame id in the scenario of which the simulation state is a part
        frame_date_times:  List of the times (in date time format) of the frame in the scenario of which the simulation
            state is a part
        time_since_last_frame: The time (in milliseconds) which has elapsed since the preceding frame in the scenario
            of which the simulation state is a part
    """

    @property
    def map_query(self) -> MapQuery:
        """Returns MapQuery object of the current simulation state"""
        return MapQuery(map=self.map)

    @property
    def ego_pose(self) -> Transformation:
        """Pose of the ego agent in the current simulation state"""
        pose = super().ego_pose
        return Transformation.from_transformation_matrix(mat=pose, approximate_orthogonal=True)

    @property
    def current_occupancy_grid(self) -> OccupancyGrid:
        """Returns the occupancy grid of the current simulation state"""
        boxes = [
            BoundingBox3DGeometry(
                # Apply @ SIM_TO_INTERNAL.inverse to bring pose rotation from RFU to FLU
                pose=Transformation.from_transformation_matrix(agent.pose, approximate_orthogonal=True)
                @ SIM_TO_INTERNAL.inverse,
                width=agent.width,
                height=agent.height,
                length=agent.length,
            )
            for agent in self.current_agents
            if isinstance(agent.step_agent, PosedAgent)
        ]
        return OccupancyGrid.from_bounding_boxes_3d(boxes=boxes)

    @property
    @lru_cache(maxsize=1)
    def previous_occupancy_grid(self) -> OccupancyGrid:
        """Returns the occupancy grid of the previous simulation state"""
        boxes = [
            BoundingBox3DGeometry(
                # Apply @ SIM_TO_INTERNAL.inverse to bring pose rotation from RFU to FLU
                pose=Transformation.from_transformation_matrix(agent.pose, approximate_orthogonal=True)
                @ SIM_TO_INTERNAL.inverse,
                width=agent.width,
                height=agent.height,
                length=agent.length,
            )
            for agent in self.previous_agents
            if isinstance(agent.step_agent, PosedAgent)
        ]
        return OccupancyGrid.from_bounding_boxes_3d(boxes=boxes)

    @staticmethod
    def from_blank_state(**kwargs) -> "ExtendedSimState":
        """
        Creates a blank ExtendedSimulationState object except for the parameters specified

        Args:
            **kwargs: Keyword arguments of parameters to be passed in when initializing the ExtendedSimulationState
        """
        world_info = pd.state.WorldInfo(street_lights=1.0)
        state = pd.state.State(simulation_time_sec=0.0, world_info=world_info, agents=[])
        return ExtendedSimState(initial_state=state, **kwargs)

    def is_using_occupied_space(
        self,
        agent: CustomSimulationAgent,
        scale_length: float = 1.0,
        scale_width: float = 1.0,
        pose: Optional[Transformation] = None,
    ) -> bool:
        """
        Checks whether an agent in a particular pose is using space which is occupied by other agents

        Args:
            agent: The agent that should be checked for collisions.  If :attr:`pose` is not passed, the agent's current
                pose from the sim state is checked for collisions
            scale_length: Scales the region the collision check is performed over in the length direction.  A value of
                1.0 checks the exact dimension of the agent's bounding box. Values greater than 1.0 expand the collision
                check region to be larger than the bounding box of the agent, values below 1.0 make the collision check
                region smaller than the agent
            scale_width: Scales the region the collision check is performed over in the width direction.  A value of
                1.0 checks the exact dimension of the agent's bounding box. Values greater than 1.0 expand the collision
                check region to be larger than the bounding box of the agent, values below 1.0 make the collision check
                region smaller than the agent
            pose: A pose of the agent (other that the agent's current pose) to check for collisions,
                useful for checking if a new pose is a valid location for the agent to be moved to

        Returns:
            True if there is a collision between the agent and other objects in the world, False if there are no
                collisions
        """

        pose_to_check = Transformation.from_transformation_matrix(mat=agent.pose) if pose is None else pose

        front_direction = (
            scale_length / 2 * agent.length * pose_to_check.quaternion.rotation_matrix @ CoordinateSystem("RFU").forward
        )
        left_direction = (
            scale_width
            / 2
            * agent.width
            * pose_to_check.quaternion.rotation_matrix
            @ np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            @ CoordinateSystem("RFU").forward
        )

        corners = np.array(
            [
                pose_to_check.translation + front_direction + left_direction,  # Front left
                pose_to_check.translation + front_direction - left_direction,  # Front right
                pose_to_check.translation - front_direction + left_direction,  # Back left
                pose_to_check.translation - front_direction - left_direction,  # Back right
            ]
        )[:, :2]

        all_vectors = (corners - corners[0])[1:]

        side_vectors = all_vectors[np.argsort(np.linalg.norm(all_vectors, axis=1))[:2]]

        del_x, del_y = np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 11))

        points = corners[0] + del_x[:, :, np.newaxis] * side_vectors[0] + del_y[:, :, np.newaxis] * side_vectors[1]

        points_to_check = points.reshape(points.shape[0] * points.shape[1], 2)

        return (self.current_occupancy_grid.is_occupied_world(points=points_to_check)).all()
