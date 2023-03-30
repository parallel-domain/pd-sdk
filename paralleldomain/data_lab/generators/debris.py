from typing import Union

from pd.internal.proto.keystone.generated.wrapper import pd_unified_generator_pb2

from paralleldomain.data_lab.generators.position_request import (
    LocationRelativePositionRequest,
    PathTimeRelativePositionRequest,
    RoadPitchPositionRequest,
    AbsolutePositionRequest,
    LaneSpawnPolicy,
)


class DebrisGeneratorParameters(pd_unified_generator_pb2.DebrisGeneratorParameters):
    @classmethod
    def from_position_request(
        cls,
        position_request: Union[
            AbsolutePositionRequest,
            RoadPitchPositionRequest,
            PathTimeRelativePositionRequest,
            LocationRelativePositionRequest,
            LaneSpawnPolicy,
        ],
        **kwargs,
    ):
        debris_gen = cls(**kwargs)
        debris_gen.position_request.set_request(position_request=position_request)
        return debris_gen
