from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Union

from paralleldomain.model.geometry.polyline_3d import Line3DGeometry, Polyline3DGeometry
from paralleldomain.model.type_aliases import EdgeId, RoadMarkingId
from paralleldomain.utilities.transformation import Transformation


class RoadMarkingType(IntEnum):
    SOLID = 0
    DASHED = 1
    SOLID_SOLID = 2  # for double solid line
    SOLID_DASHED = 3  # from left to right, note this is different from ODR spec
    DASHED_SOLID = 4  # from left to right, note this is different from ODR spec
    DASHED_DASHED = 5  # for double dashed line
    BOTTS_DOTS = 6  # For these three, we can specify as a RoadMark or should they be captured elsewhere
    NO_PAINT = 7


class RoadMarkingColor(IntEnum):
    WHITE = 0
    BLUE = 1
    GREEN = 2
    RED = 3
    YELLOW = 4


@dataclass
class RoadMarking:
    road_marking_id: RoadMarkingId
    edge_id: int
    width: float
    type: RoadMarkingType
    color: RoadMarkingColor


@dataclass
class Edge(Polyline3DGeometry):
    edge_id: EdgeId
    closed: bool = False
    road_marking: Union[RoadMarking, List[RoadMarking]] = None
    user_data: Dict[str, Any] = field(default_factory=dict)

    def transform(self, tf: Transformation) -> "Edge":
        return Edge(
            edge_id=self.edge_id,
            closed=self.closed,
            road_marking=self.road_marking,
            user_data=self.user_data,
            lines=[ll.transform(tf=tf) for ll in self.lines],
        )
