from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from more_itertools import windowed

from paralleldomain.common.umd.v1.UMD_pb2 import Edge as ProtoEdge
from paralleldomain.common.umd.v1.UMD_pb2 import Point_ENU as ProtoPointENU
from paralleldomain.common.umd.v1.UMD_pb2 import RoadMarking as ProtoRoadMarking
from paralleldomain.model.geometry.point_3d import Point3DGeometry
from paralleldomain.model.geometry.polyline_3d import Line3DGeometry, Polyline3DGeometry
from paralleldomain.model.map.common import load_user_data
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
class PointENU(Point3DGeometry):
    @classmethod
    def from_proto(cls, point: ProtoPointENU):
        return cls(x=point.x, y=point.y, z=point.z)

    @classmethod
    def from_transformation(cls, tf: Transformation):
        return cls(x=tf.translation[0], y=tf.translation[1], z=tf.translation[2])


@dataclass
class RoadMarking:
    id: int
    edge_id: int
    width: float
    type: RoadMarkingType
    color: RoadMarkingColor

    @classmethod
    def from_proto(cls, road_marking: ProtoRoadMarking) -> "RoadMarking":
        return cls(
            id=road_marking.id,
            edge_id=road_marking.id,
            width=road_marking.width,
            type=RoadMarkingType(road_marking.type) if road_marking.HasField("type") else None,
            color=RoadMarkingColor(road_marking.color) if road_marking.HasField("color") else None,
        )


@dataclass
class Edge(Polyline3DGeometry):
    id: int
    closed: bool = False
    road_marking: Union[RoadMarking, List[RoadMarking]] = None
    user_data: Dict[str, Any] = field(default_factory=dict)

    def transform(self, tf: Transformation) -> "Edge":
        return Edge(
            id=self.id,
            closed=self.closed,
            road_marking=self.road_marking,
            user_data=self.user_data,
            lines=[ll.transform(tf=tf) for ll in self.lines],
        )

    @classmethod
    def from_numpy(
        cls,
        points: np.ndarray,
        id: int,
        closed: bool = False,
        road_marking: Optional[RoadMarking] = None,
        user_data: Optional[Dict[str, Any]] = None,
    ) -> "Edge":
        if user_data is None:
            user_data = {}
        points = points.reshape(-1, 3)
        point_pairs = np.hstack([points[:-1], points[1:]])
        return Edge(
            id=id,
            closed=closed,
            road_marking=road_marking,
            user_data=user_data,
            lines=np.apply_along_axis(Line3DGeometry.from_numpy, point_pairs),
        )

    @classmethod
    def from_proto(cls, edge: ProtoEdge, road_marking: Optional[ProtoRoadMarking] = None):
        return cls(
            id=edge.id,
            closed=not (edge.open),
            lines=[
                Line3DGeometry(
                    start=PointENU.from_proto(point=point_pair[0]), end=PointENU.from_proto(point=point_pair[1])
                )
                for point_pair in windowed(edge.points, 2)
            ],
            road_marking=RoadMarking.from_proto(road_marking=road_marking) if road_marking is not None else None,
            user_data=load_user_data(edge.user_data) if edge.HasField("user_data") else {},
        )
