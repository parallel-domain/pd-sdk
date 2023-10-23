# This file in maintained for backwards compatibility of custom behavior classes which used to be in this location

from warnings import simplefilter, warn

from paralleldomain.data_lab.behaviors.pedestrian import PedestrianBehavior  # noqa: F401
from paralleldomain.data_lab.behaviors.single_frame import (  # noqa: F401
    SingleFramePlaceNearEgoBehavior,
    SingleFrameVehicleBehavior,
)
from paralleldomain.data_lab.behaviors.static import LookAtPointBehavior, StaticBehavior  # noqa: F401
from paralleldomain.data_lab.behaviors.traffic_sign import (  # noqa: F401
    TrafficSignAttachToPoleBehavior,
    TrafficSignPoleBehavior,
)
from paralleldomain.data_lab.behaviors.vehicle import (  # noqa: F401
    DrivewayCreepBehavior,
    RenderEgoBehavior,
    VehicleBehavior,
)

simplefilter(action="always", category=DeprecationWarning)
warn(
    (
        "This method of importing behaviors is deprecated and will be removed shortly, "
        "please import behaviors directly from modules in `paralleldomain.data_lab.behaviors`"
    ),
    category=DeprecationWarning,
)
