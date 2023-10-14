from typing import Union

from pd.internal.proto.keystone.generated.wrapper import pd_unified_generator_pb2

from paralleldomain.data_lab.generators.position_request import (
    AbsolutePositionRequest,
    LaneSpawnPolicy,
    LocationRelativePositionRequest,
    PathTimeRelativePositionRequest,
)
from paralleldomain.utilities import inherit_docs


@inherit_docs
class DroneGeneratorParameters(pd_unified_generator_pb2.DroneGeneratorParameters):
    @classmethod
    def from_position_request(
        cls,
        position_request: Union[
            AbsolutePositionRequest,
            PathTimeRelativePositionRequest,
            LocationRelativePositionRequest,
            LaneSpawnPolicy,
        ],
        **kwargs,
    ):
        """Initializes a DroneGeneratorParameters object from a pre-defined position request

        Args:
            position_request: The position request from which the DroneGeneratorParameters object should be
                initialized and which the created object will contain

        Returns:
            A `DroneGeneratorParameters` object which specifies the position passed in the `position_request` parameter
        """
        debris_gen = cls(**kwargs)
        debris_gen.position_request.set_request(position_request=position_request)
        return debris_gen
