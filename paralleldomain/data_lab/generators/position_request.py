from typing import Union

from pd.internal.proto.keystone.generated.wrapper import pd_unified_generator_pb2

from paralleldomain.data_lab.config.types import Float3, Float3x3
from paralleldomain.utilities import inherit_docs
from paralleldomain.utilities.transformation import Transformation


@inherit_docs
class SpecialAgentTag(pd_unified_generator_pb2.SpecialAgentTag):
    ...


@inherit_docs
class RoadPitchPositionRequest(pd_unified_generator_pb2.RoadPitchPositionRequest):
    ...


@inherit_docs
class PathTimeRelativePositionRequest(pd_unified_generator_pb2.PathTimeRelativePositionRequest):
    ...


@inherit_docs
class LocationRelativePositionRequest(pd_unified_generator_pb2.LocationRelativePositionRequest):
    ...


@inherit_docs
class LaneSpawnPolicy(pd_unified_generator_pb2.LaneSpawnPolicy):
    ...


@inherit_docs
class PositionOfInterestPolicy(pd_unified_generator_pb2.PositionOfInterestPolicy):
    ...


@inherit_docs
class JunctionSpawnPolicy(pd_unified_generator_pb2.JunctionSpawnPolicy):
    ...


@inherit_docs
class AbsolutePositionRequest(pd_unified_generator_pb2.AbsolutePositionRequest):
    @classmethod
    def from_transformation(cls, transformation: Transformation) -> "AbsolutePositionRequest":
        x, y, z = transformation.translation
        R = transformation.rotation

        return cls(
            position=Float3(x=x, y=y, z=z),
            rotation=Float3x3(
                r0=Float3(x=R[0, 0], y=R[0, 1], z=R[0, 2]),
                r1=Float3(x=R[1, 0], y=R[1, 1], z=R[1, 2]),
                r2=Float3(x=R[2, 0], y=R[2, 1], z=R[2, 2]),
            ),
        )


@inherit_docs
class PositionRequest(pd_unified_generator_pb2.PositionRequest):
    ...

    def set_request(
        self,
        position_request: Union[
            AbsolutePositionRequest,
            RoadPitchPositionRequest,
            PathTimeRelativePositionRequest,
            LaneSpawnPolicy,
            LocationRelativePositionRequest,
        ],
    ):
        if isinstance(position_request, AbsolutePositionRequest):
            self.simple_position_request = position_request
        elif isinstance(position_request, RoadPitchPositionRequest):
            self.road_pitch_position_request = position_request
        elif isinstance(position_request, PathTimeRelativePositionRequest):
            self.path_time_relative_position_request = position_request
        elif isinstance(position_request, LocationRelativePositionRequest):
            self.location_relative_position_request = position_request
        elif isinstance(position_request, LaneSpawnPolicy):
            self.lane_spawn_policy = LaneSpawnPolicy
        else:
            raise ValueError(f"Position Request of type {type(position_request)} is not supported in this method.")
