from pd.internal.proto.keystone.generated.wrapper import pd_unified_generator_pb2

from paralleldomain.utilities import inherit_docs


@inherit_docs
class AgentSpawnData(pd_unified_generator_pb2.AgentSpawnData):
    ...


@inherit_docs
class VehicleSpawnData(pd_unified_generator_pb2.VehicleSpawnData):
    ...


@inherit_docs
class PedestrianSpawnData(pd_unified_generator_pb2.PedestrianSpawnData):
    ...


@inherit_docs
class DroneSpawnData(pd_unified_generator_pb2.DroneSpawnData):
    ...
