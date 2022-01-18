from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Union

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DBaseGeometry
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
    def get_lane_segment(self, lane_segment_id: LaneSegmentId) -> "LaneSegment":
        pass

    def get_road_segment(self, road_segment_id: RoadSegmentId) -> "RoadSegment":
        pass

    def get_edge(self, edge_id: EdgeId) -> Edge:
        pass

    def get_junction(self, junction_id: JunctionId) -> Optional["Junction"]:
        pass


class RoadSegment:
    def __init__(
        self,
        map_query: RoadSegmentMapQueryProtocol,
        road_segment_id: RoadSegmentId,
        name: str,
        bounds: Optional[BoundingBox2DBaseGeometry[float]],
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
    def lane_segments(self) -> List["LaneSegment"]:
        return [self.map_query.get_lane_segment(lane_segment_id=lsid) for lsid in self.lane_segment_ids]

    @property
    def reference_line(self) -> Optional[Edge]:
        if self.reference_line_id is not None:
            return self.map_query.get_edge(edge_id=self.reference_line_id)
        return None

    @property
    def junction(self) -> Optional["Junction"]:
        if self.junction_id is not None:
            return self.map_query.get_junction(junction_id=self.junction_id)
        return None


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


class LaneSegmentMapQueryProtocol(Protocol):
    def get_junctions_for_lane_segment(self, lane_segment_id: LaneSegmentId) -> List["Junction"]:
        pass

    def get_relative_left_neighbor(self, lane_segment_id: LaneSegmentId, degree: int = 1) -> Optional["LaneSegment"]:
        pass

    def get_relative_right_neighbor(self, lane_segment_id: LaneSegmentId, degree: int = 1) -> Optional["LaneSegment"]:
        pass

    def get_edge(self, edge_id: EdgeId) -> Edge:
        pass

    def get_lane_segment(self, lane_segment_id: LaneSegmentId) -> "LaneSegment":
        pass

    def get_predecessors(self, depth: int = -1) -> List["LaneSegment"]:
        pass

    def get_successors(self, depth: int = -1) -> List["LaneSegment"]:
        pass

    def get_road_segment_for_lane_segment(self, lane_segment_id: LaneSegmentId) -> Optional[RoadSegment]:
        pass

    def are_connected_lane_segments(self, id_1: LaneSegmentId, id_2: LaneSegmentId) -> bool:
        pass

    def are_opposite_direction_lane_segments(self, id_1: LaneSegmentId, id_2: LaneSegmentId) -> Optional[bool]:
        pass

    def are_succeeding_lane_segments(self, id_1: LaneSegmentId, id_2: LaneSegmentId) -> bool:
        pass

    def are_preceeding_lane_segments(self, id_1: LaneSegmentId, id_2: LaneSegmentId) -> bool:
        pass


class LaneSegment:
    def __init__(
        self,
        map_query: LaneSegmentMapQueryProtocol,
        lane_segment_id: LaneSegmentId,
        lane_type: LaneType,
        direction: Direction,
        left_edge_id: EdgeId,
        right_edge_id: EdgeId,
        reference_line_id: EdgeId,
        bounds: Optional[BoundingBox2DBaseGeometry[float]],
        left_neighbor_id: Optional[LaneSegmentId] = None,
        right_neighbor_id: Optional[LaneSegmentId] = None,
        parent_road_segment_id: Optional[RoadSegmentId] = None,
        compass_angle: Optional[float] = None,
        turn_angle: Optional[float] = None,
        turn_type: Optional[TurnType] = None,
        user_data: Dict[str, Any] = None,
        predecessor_ids: List[LaneSegmentId] = None,
        successor_ids: List[LaneSegmentId] = None,
    ):
        self.direction = direction
        if user_data is None:
            user_data = dict()
        if predecessor_ids is None:
            predecessor_ids = list()
        if successor_ids is None:
            successor_ids = list()

        self.predecessor_ids = predecessor_ids
        self.successor_ids = successor_ids
        self.user_data = user_data
        self.turn_type = turn_type
        self.turn_angle = turn_angle
        self.compass_angle = compass_angle
        self.parent_road_segment_id = parent_road_segment_id
        self.right_neighbor_id = right_neighbor_id
        self.left_neighbor_id = left_neighbor_id
        self.bounds = bounds
        self.right_edge_id = right_edge_id
        self.left_edge_id = left_edge_id
        self.reference_line_id = reference_line_id
        self.lane_type = lane_type
        self.lane_segment_id = lane_segment_id
        self.map_query = map_query

    @property
    def left_edge(self) -> Edge:
        return self.map_query.get_edge(edge_id=self.left_edge_id)

    @property
    def right_edge(self) -> Edge:
        return self.map_query.get_edge(edge_id=self.right_edge_id)

    @property
    def reference_line(self) -> Edge:
        return self.map_query.get_edge(edge_id=self.reference_line_id)

    @property
    def left_neighbor(self) -> "LaneSegment":
        return self.map_query.get_lane_segment(lane_segment_id=self.left_neighbor_id)

    @property
    def right_neighbor(self) -> "LaneSegment":
        return self.map_query.get_lane_segment(lane_segment_id=self.right_neighbor_id)

    @property
    def junctions(self) -> List["Junction"]:
        return self.map_query.get_junctions_for_lane_segment(lane_segment_id=self.lane_segment_id)

    @property
    def parent_road_segment(self) -> Optional[RoadSegment]:
        return self.map_query.get_road_segment_for_lane_segment(lane_segment_id=self.lane_segment_id)

    def get_relative_left_neighbor(self, degree: int = 1) -> Optional["LaneSegment"]:
        return self.map_query.get_relative_left_neighbor(lane_segment_id=self.lane_segment_id, degree=degree)

    def get_relative_right_neighbor(self, degree: int = 1) -> Optional["LaneSegment"]:
        return self.map_query.get_relative_right_neighbor(lane_segment_id=self.lane_segment_id, degree=degree)

    def get_predecessors(self, depth: int = -1) -> List["LaneSegment"]:
        return self.map_query.get_predecessors(depth=depth)

    def get_successors(self, depth: int = -1) -> List["LaneSegment"]:
        return self.map_query.get_successors(depth=depth)

    @property
    def has_lane_segment_split(self) -> bool:
        return len(self.get_successors(depth=1)) > 2  # [start_node, one_successor, ...]

    @property
    def has_lane_segment_merge(self) -> bool:
        return len(self.get_successors(depth=1)) > 2  # [start_node, one_predecessor, ...]

    @property
    def is_inside_junction(self) -> bool:
        return True if len(self.junctions) > 0 else False

    def are_connected(self, other: Union[LaneSegmentId, "LaneSegment"]) -> bool:
        other = self.ensure_lane_segment_id(item=other)
        return self.map_query.are_connected_lane_segments(id_1=self.lane_segment_id, id_2=other)

    def are_opposite_direction(self, other: Union[LaneSegmentId, "LaneSegment"]) -> Optional[bool]:
        other = self.ensure_lane_segment_id(item=other)
        return self.map_query.are_opposite_direction_lane_segments(id_1=self.lane_segment_id, id_2=other)

    def are_succeeding(self, other: Union[LaneSegmentId, "LaneSegment"]) -> bool:
        other = self.ensure_lane_segment_id(item=other)
        return self.map_query.are_succeeding_lane_segments(id_1=self.lane_segment_id, id_2=other)

    def are_preceeding(self, other: Union[LaneSegmentId, "LaneSegment"]) -> bool:
        other = self.ensure_lane_segment_id(item=other)
        return self.map_query.are_preceeding_lane_segments(id_1=self.lane_segment_id, id_2=other)

    @staticmethod
    def ensure_lane_segment_id(item: Union[LaneSegmentId, "LaneSegment"]) -> LaneSegmentId:
        if isinstance(item, LaneSegment):
            item = item.lane_segment_id
        return item


class JunctionMapQueryProtocol(Protocol):
    def get_lane_segment(self, lane_segment_id: LaneSegmentId) -> LaneSegment:
        pass

    def get_road_segment(self, road_segment_id: RoadSegmentId) -> RoadSegment:
        pass

    def get_edge(self, edge_id: EdgeId) -> Edge:
        pass


@dataclass
class Junction:
    def __init__(
        self,
        map_query: JunctionMapQueryProtocol,
        junction_id: JunctionId,
        bounds: Optional[BoundingBox2DBaseGeometry[float]],
        lane_segment_ids: List[LaneSegmentId] = None,
        road_segment_ids: List[RoadSegmentId] = None,
        signaled_intersection: Optional[int] = None,
        user_data: Dict[str, Any] = None,
        corner_ids: List[EdgeId] = None,
        crosswalk_lane_ids: List[LaneSegmentId] = None,
        signed_intersection: Optional[int] = None,
    ):
        self.map_query = map_query
        self.signed_intersection = signed_intersection
        if user_data is None:
            user_data = dict()
        if corner_ids is None:
            corner_ids = list()
        if crosswalk_lane_ids is None:
            crosswalk_lane_ids = list()
        if road_segment_ids is None:
            road_segment_ids = list()
        if lane_segment_ids is None:
            lane_segment_ids = list()
        self.road_segment_ids = road_segment_ids
        self.lane_segment_ids = lane_segment_ids
        self.user_data = user_data
        self.corner_ids = corner_ids
        self.crosswalk_lane_ids = crosswalk_lane_ids
        self.signaled_intersection = signaled_intersection
        self.bounds = bounds
        self.junction_id = junction_id

    @property
    def lane_segments(self) -> List[LaneSegment]:
        return [self.map_query.get_lane_segment(lane_segment_id=lsid) for lsid in self.lane_segment_ids]

    @property
    def crosswalk_lanes(self) -> List[LaneSegment]:
        return [self.map_query.get_lane_segment(lane_segment_id=lsid) for lsid in self.crosswalk_lane_ids]

    @property
    def road_segment(self) -> List[RoadSegment]:
        return [self.map_query.get_road_segment(road_segment_id=rsid) for rsid in self.road_segment_ids]

    @property
    def corners(self) -> List[Edge]:
        return [self.map_query.get_edge(edge_id=c) for c in self.corner_ids]
