from typing import Callable

from paralleldomain.utilities.transformation import Transformation


class EgoPose(Transformation):
    ...


class EgoFrame:
    """
    This Objects contains information about the ego-object's world pose within a given frame.
    """

    def __init__(self, pose_loader: Callable[[], EgoPose]):
        self._pose_loader = pose_loader

    @property
    def pose(self) -> EgoPose:
        return self._pose_loader()
