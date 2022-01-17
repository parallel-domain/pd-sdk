from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional

import paralleldomain.model.map as pd_map
from paralleldomain.model.map.lane_segment import LaneSegment

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DGeometry
from paralleldomain.model.map.edge import Edge
from paralleldomain.model.type_aliases import EdgeId, JunctionId, LaneSegmentId, RoadSegmentId


class RoadType(IntEnum):
    MOTORWAY = 0
    TRUNK = 1
    PRIMARY = 2
    SECONDARY = 3
    TERTIARY = 4
    UNCLASSIFIED = 5
    RESIDENTIAL = 6
    MOTORWAY_LINK = 7
    TRUNK_LINK = 8
    PRIMARY_LINK = 9
    SECONDARY_LINK = 10
    TERTIARY_LINK = 11
    SERVICE = 12
    DRIVEWAY = 13
    PARKING_AISLE = 14


class GroundType(IntEnum):
    GROUND = 0
    BRIDGE = 1
    TUNNEL = 2


class SpeedUnits(IntEnum):
    MILES_PER_HOUR = 0
    KILOMETERS_PER_HOUR = 1


@dataclass
class SpeedLimit:
    speed: int
    units: SpeedUnits


class RoadSegmentMapQueryProtocol(Protocol):
    def get_lane_segment(self, lane_segment_id: LaneSegmentId) -> LaneSegment:
        pass

    def get_road_segment(self, road_segment_id: RoadSegmentId) -> "RoadSegment":
        pass

    def get_edge(self, edge_id: EdgeId) -> Edge:
        pass

    def get_junction(self, junction_id: JunctionId) -> Optional[pd_map.junction.Junction]:
        pass


class RoadSegment:
    def __init__(
        self,
        map_query: RoadSegmentMapQueryProtocol,
        road_segment_id: RoadSegmentId,
        name: str,
        bounds: Optional[BoundingBox2DGeometry],
        predecessor_ids: List[RoadSegmentId] = None,
        successor_ids: List[RoadSegmentId] = None,
        lane_segment_ids: List[LaneSegmentId] = None,
        reference_line_id: Optional[EdgeId] = None,
        road_type: Optional[RoadType] = None,
        ground_type: Optional[GroundType] = None,
        speed_limit: Optional[SpeedLimit] = None,
        junction_id: Optional[JunctionId] = None,
        user_data: Dict[str, Any] = None,
    ):
        self.map_query = map_query
        if predecessor_ids is None:
            predecessor_ids = list()
        if successor_ids is None:
            successor_ids = list()
        if lane_segment_ids is None:
            lane_segment_ids = list()
        if user_data is None:
            user_data = dict()

        self.user_data = user_data
        self.junction_id = junction_id
        self.speed_limit = speed_limit
        self.ground_type = ground_type
        self.road_type = road_type
        self.reference_line_id = reference_line_id
        self.bounds = bounds
        self.name = name
        self.road_segment_id = road_segment_id
        self.lane_segment_ids = lane_segment_ids
        self.predecessor_ids = predecessor_ids
        self.successor_ids = successor_ids

    @property
    def predecessors(self) -> List["RoadSegment"]:
        return [self.map_query.get_road_segment(road_segment_id=r) for r in self.predecessor_ids]

    @property
    def successors(self) -> List["RoadSegment"]:
        return [self.map_query.get_road_segment(road_segment_id=r) for r in self.successor_ids]

    @property
    def lane_segments(self) -> List[LaneSegment]:
        return [self.map_query.get_lane_segment(lane_segment_id=lsid) for lsid in self.lane_segment_ids]

    @property
    def reference_line(self) -> Optional[Edge]:
        if self.reference_line_id is not None:
            return self.map_query.get_edge(edge_id=self.reference_line_id)
        return None

    @property
    def junction(self) -> Optional[pd_map.junction.Junction]:
        if self.junction_id is not None:
            return self.map_query.get_junction(junction_id=self.junction_id)
        return None
