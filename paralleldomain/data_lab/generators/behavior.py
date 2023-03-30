from pd.internal.proto.keystone.generated.wrapper import pd_unified_generator_pb2

from paralleldomain.utilities import inherit_docs


@inherit_docs
class VehicleBehavior(pd_unified_generator_pb2.VehicleBehavior):
    ...


@inherit_docs
class PedestrianBehavior(pd_unified_generator_pb2.PedestrianBehavior):
    ...


@inherit_docs
class Gear(pd_unified_generator_pb2.Gear):
    ...
