import pd.state
from pd.data_lab.sim_state import SimState

from paralleldomain.data_lab.config.map import MapQuery
from paralleldomain.utilities.transformation import Transformation


class ExtendedSimState(SimState):
    @property
    def map_query(self) -> MapQuery:
        return MapQuery(map=self.map)

    @property
    def ego_pose(self) -> Transformation:
        pose = super().ego_pose
        return Transformation.from_transformation_matrix(mat=pose, approximate_orthogonal=True)

    @staticmethod
    def from_blank_state(**kwargs) -> "SimState":
        world_info = pd.state.WorldInfo(street_lights=1.0)
        state = pd.state.State(simulation_time_sec=0.0, world_info=world_info, agents=[])
        return ExtendedSimState(initial_state=state, **kwargs)
