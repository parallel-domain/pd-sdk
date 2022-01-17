from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from paralleldomain.model.map.edge import Edge
from paralleldomain.model.map.lane_segment import LaneSegment
from paralleldomain.model.map.road_segment import RoadSegment

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DGeometry
from paralleldomain.model.type_aliases import EdgeId, JunctionId, LaneSegmentId, RoadSegmentId


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
        bounds: Optional[BoundingBox2DGeometry],
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
