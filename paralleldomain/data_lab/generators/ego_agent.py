import logging
from typing import List

from pd.data_lab.generators.custom_simulation_agent import CustomSimulationAgent, CustomVehicleSimulationAgent
from pd.internal.proto.keystone.generated.wrapper import pd_unified_generator_pb2

from paralleldomain.data_lab import CustomAtomicGenerator, ExtendedSimState
from paralleldomain.data_lab.behaviors.vehicle import RenderEgoBehavior
from paralleldomain.utilities import inherit_docs

logger = logging.getLogger(__name__)


@inherit_docs
class EgoAgentGeneratorParameters(pd_unified_generator_pb2.EgoAgentGeneratorParameters):
    ...


AgentType = pd_unified_generator_pb2.AgentType


class RenderEgoGenerator(CustomAtomicGenerator):
    """Places the ego vehicle in the scene so that it can appear in rendered images

    Args:
        ego_asset_name: Name of the ego vehicle model as it appears in the asset database
    """

    def __init__(
        self,
        ego_asset_name: str = "suv_medium_02",
    ):
        self._ego_asset_name = ego_asset_name

    # This is a very simple Custom Generator which only places 1 Custom Agent and assigns the RenderEgoBehavior Custom
    # Behavior to that agent
    def create_agents_for_new_scene(self, state: ExtendedSimState, random_seed: int) -> List[CustomSimulationAgent]:
        # Create empty list to store the Custom Agents we create
        agents = []

        # Create a Custom Agent of with the specified asset name and assign the RenderEgoBehavior Custom Behavior
        agent = CustomVehicleSimulationAgent(asset_name=self._ego_asset_name, lock_to_ground=False).set_behavior(
            RenderEgoBehavior()
        )

        # Append the created Custom Agent to the list and return it
        agents.append(agent)
        return agents

    # The clone method returns a copy of the Custom Generator object and is required under the hood by Data Lab
    def clone(self):
        return RenderEgoGenerator(
            ego_asset_name=self._ego_asset_name,
        )
