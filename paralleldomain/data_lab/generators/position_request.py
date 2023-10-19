from pd.internal.proto.keystone.generated.wrapper import pd_unified_generator_pb2

from paralleldomain.data_lab.config.types import Float3, Float3x3
from paralleldomain.utilities import inherit_docs
from paralleldomain.utilities.transformation import Transformation


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
        """
        Generates and AbsolutePositionRequest object which corresponds to a location specified in a Transformation
            object

        Args:
            transformation: A Transformation object containing the location which the created AbsolutePositionRequest
                object should specify

        Returns:
            An AbsolutePositionRequest object which specifies the region passed in the transformation parameter
        """
        x, y, z = transformation.translation
        r = transformation.rotation

        return cls(
            position=Float3(x=x, y=y, z=z),
            rotation=Float3x3(
                r0=Float3(x=r[0, 0], y=r[0, 1], z=r[0, 2]),
                r1=Float3(x=r[1, 0], y=r[1, 1], z=r[1, 2]),
                r2=Float3(x=r[2, 0], y=r[2, 1], z=r[2, 2]),
            ),
        )


@inherit_docs
class PositionRequest(pd_unified_generator_pb2.PositionRequest):
    ...
