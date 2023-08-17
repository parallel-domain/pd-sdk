from functools import lru_cache

import pd.state
from pd.data_lab.sim_state import SimState
from pd.state.state import PosedAgent

from paralleldomain.data_lab.config.map import MapQuery
from paralleldomain.model.geometry.bounding_box_3d import BoundingBox3DGeometry
from paralleldomain.model.occupancy import OccupancyGrid
from paralleldomain.utilities.coordinate_system import SIM_TO_INTERNAL
from paralleldomain.utilities.transformation import Transformation


class ExtendedSimState(SimState):
    @property
    def map_query(self) -> MapQuery:
        return MapQuery(map=self.map)

    @property
    def ego_pose(self) -> Transformation:
        pose = super().ego_pose
        return Transformation.from_transformation_matrix(mat=pose, approximate_orthogonal=True)

    @property
    def current_occupancy_grid(self) -> OccupancyGrid:
        # Apply @ SIM_TO_INTERNAL.inverse to bring pose rotation from RFU to FLU so width/length/height are correct
        boxes = [
            BoundingBox3DGeometry(
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
        # Apply @ SIM_TO_INTERNAL.inverse to bring pose rotation from RFU to FLU so width/length/height are correct
        boxes = [
            BoundingBox3DGeometry(
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
    def from_blank_state(**kwargs) -> "SimState":
        world_info = pd.state.WorldInfo(street_lights=1.0)
        state = pd.state.State(simulation_time_sec=0.0, world_info=world_info, agents=[])
        return ExtendedSimState(initial_state=state, **kwargs)
