from pd.internal.proto.keystone.generated.wrapper import pd_unified_generator_pb2

from paralleldomain.utilities import inherit_docs


@inherit_docs
class EnvironmentParameters(pd_unified_generator_pb2.EnvironmentParameters):
    ...


@inherit_docs
class ParkingSpaceData(pd_unified_generator_pb2.ParkingSpaceData):
    ...


@inherit_docs
class RoadMarkingData(pd_unified_generator_pb2.RoadMarkingData):
    ...


@inherit_docs
class DecorationPreset(pd_unified_generator_pb2.DecorationPreset):
    ...


@inherit_docs
class DecorationData(pd_unified_generator_pb2.DecorationData):
    ...


@inherit_docs
class ObjectDecorations(pd_unified_generator_pb2.ObjectDecorations):
    ...


@inherit_docs
class ObjectDecorationParams(pd_unified_generator_pb2.ObjectDecorationParams):
    ...
