from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional

import numpy as np

from paralleldomain.common.umd.v1.UMD_pb2 import LaneSegment as ProtoLaneSegment
from paralleldomain.common.umd.v1.UMD_pb2 import UniversalMap as ProtoUniversalMap
from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DGeometry
from paralleldomain.model.map.common import load_user_data
from paralleldomain.model.map.edge import Edge
from paralleldomain.model.type_aliases import LaneSegmentId, RoadSegmentId


class LaneType(IntEnum):
    UNDEFINED_LANE = 0
    DRIVABLE = 1
    NON_DRIVABLE = 2
    PARKING = 3
    SHOULDER = 4
    BIKING = 5
    CROSSWALK = 6
    RESTRICTED = 7
    PARKING_AISLE = 8
    PARKING_SPACE = 9


class Direction(IntEnum):
    UNDEFINED_DIR = 0
    FORWARD = 1
    BACKWARD = 2
    BIDIRECTIONAL = 3


class TurnType(IntEnum):
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2
    SLIGHT_LEFT = 3
    SLIGHT_RIGHT = 4
    U_TURN = 5


@dataclass
class LaneSegment:
    lane_segment_id: LaneSegmentId
    type: LaneType
    direction: Direction
    left_edge: Edge
    right_edge: Edge
    reference_line: Edge
    bounds: Optional[BoundingBox2DGeometry]
    predecessors: List[LaneSegmentId] = field(default_factory=list)
    successors: List[LaneSegmentId] = field(default_factory=list)
    left_neighbor: Optional[LaneSegmentId] = None
    right_neighbor: Optional[LaneSegmentId] = None
    parent_road_segment_id: Optional[RoadSegmentId] = None
    compass_angle: Optional[float] = None
    turn_angle: Optional[float] = None
    turn_type: Optional[TurnType] = None
    user_data: Dict[str, Any] = field(default=dict)

    def to_numpy(self, closed: bool = False) -> np.ndarray:
        if not closed:
            return np.vstack([self.left_edge.to_numpy(), self.right_edge.to_numpy()[::-1]])
        else:
            return np.vstack(
                [self.left_edge.to_numpy(), self.right_edge.to_numpy()[::-1], self.left_edge.to_numpy()[0]]
            )

    @classmethod
    def from_proto(cls, id: int, umd_map: ProtoUniversalMap) -> "LaneSegment":
        lane_segment: ProtoLaneSegment = umd_map.lane_segments[id]
        road_markings_by_edge_id = {rm.edge_id: rm for rm in umd_map.road_markings.values()}
        return LaneSegment(
            lane_segment_id=lane_segment.id,
            type=LaneType(lane_segment.type),
            direction=Direction(lane_segment.direction),
            left_edge=Edge.from_proto(
                edge=umd_map.edges[lane_segment.left_edge],
                road_marking=road_markings_by_edge_id[lane_segment.left_edge]
                if lane_segment.left_edge in road_markings_by_edge_id
                else None,
            ),
            right_edge=Edge.from_proto(
                edge=umd_map.edges[lane_segment.right_edge],
                road_marking=road_markings_by_edge_id[lane_segment.right_edge]
                if lane_segment.right_edge in road_markings_by_edge_id
                else None,
            ),
            reference_line=Edge.from_proto(edge=umd_map.edges[lane_segment.reference_line]),
            predecessors=[ls_p for ls_p in lane_segment.predecessors],
            successors=[ls_s for ls_s in lane_segment.successors],
            left_neighbor=lane_segment.left_neighbor,
            right_neighbor=lane_segment.right_neighbor,
            compass_angle=lane_segment.compass_angle,
            turn_angle=lane_segment.turn_angle,
            turn_type=TurnType(lane_segment.turn_type),
            user_data=load_user_data(lane_segment.user_data) if lane_segment.HasField("user_data") else {},
        )
