import logging
from typing import List

from pd.data_lab.generators.custom_simulation_agent import CustomVehicleSimulationAgent, CustomSimulationAgent
from pd.internal.proto.keystone.generated.wrapper import pd_unified_generator_pb2

from paralleldomain.data_lab import CustomAtomicGenerator, ExtendedSimState
from paralleldomain.data_lab.generators.behavior import RenderEgoBehavior
from paralleldomain.utilities import inherit_docs

logger = logging.getLogger(__name__)


@inherit_docs
class EgoAgentGeneratorParameters(pd_unified_generator_pb2.EgoAgentGeneratorParameters):
    ...


@inherit_docs
class AgentType(pd_unified_generator_pb2.AgentType):
    ...


class RenderEgoGenerator(CustomAtomicGenerator):
    """
    Allows the ego vehicle to be rendered

    Args:
        ego_asset_name:
              Description:
                  Name of the ego vehicle model as it appears in the asset registry
              Range:
                  Name must match a vehicle in the asset registry
              Required:
                  No, will default to "suv_medium_02" if no name is provided
    """

    def __init__(
        self,
        ego_asset_name: str = "suv_medium_02",
    ):
        self._ego_asset_name = ego_asset_name

    def create_agents_for_new_scene(self, state: ExtendedSimState, random_seed: int) -> List[CustomSimulationAgent]:
        agents = []

        agent = CustomVehicleSimulationAgent(asset_name=self._ego_asset_name, lock_to_ground=False).set_behaviour(
            RenderEgoBehavior()
        )
        agents.append(agent)

        return agents

    def clone(self):
        return RenderEgoGenerator(
            ego_asset_name=self._ego_asset_name,
        )
