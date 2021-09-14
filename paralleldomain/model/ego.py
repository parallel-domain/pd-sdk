from paralleldomain.utilities.transformation import Transformation

try:
    from typing import Callable, Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore


class EgoPose(Transformation):
    ...


class EgoFrame:
    """
    This Objects contains informations of a frame of the Ego vehicle/drone/person the sensor rig was attached to.
    """

    def __init__(self, pose_loader: Callable[[], EgoPose]):
        self._pose_loader = pose_loader

    @property
    def pose(self) -> EgoPose:
        return self._pose_loader()
