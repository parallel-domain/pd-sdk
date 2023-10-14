from pd.internal.proto.keystone.generated.wrapper import pd_types_pb2

from paralleldomain.utilities import inherit_docs


@inherit_docs
class Float3(pd_types_pb2.Float3):
    ...


@inherit_docs
class Float3x3(pd_types_pb2.Float3x3):
    ...


@inherit_docs
class Pose(pd_types_pb2.Pose):
    ...
