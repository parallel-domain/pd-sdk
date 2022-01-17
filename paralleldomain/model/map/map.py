from typing import List, Optional, Union

from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DGeometry
from paralleldomain.model.geometry.point_3d import Point3DGeometry
from paralleldomain.model.type_aliases import JunctionId, LaneSegmentId, RoadSegmentId

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

from paralleldomain.model.map.area import Area
from paralleldomain.model.map.junction import Junction
from paralleldomain.model.map.lane_segment import LaneSegment
from paralleldomain.model.map.road_segment import RoadSegment
from paralleldomain.utilities.transformation import Transformation


class MapQueryProtocol(Protocol):
    def get_junction(self, junction_id: JunctionId) -> Optional[Junction]:
        pass

    def get_road_segment(self, road_segment_id: RoadSegmentId) -> Optional[RoadSegment]:
        pass

    def get_lane_segment(self, lane_segment_id: LaneSegmentId) -> Optional[LaneSegment]:
        pass

    def get_lane_segments(self, lane_segment_ids: List[LaneSegmentId]) -> List[Optional[LaneSegment]]:
        return [self.get_lane_segment(lane_segment_id=lid) for lid in lane_segment_ids]

    def get_lane_segments_successors_shortest_paths(
        self, source_id: LaneSegmentId, target_id: LaneSegmentId
    ) -> List[List[LaneSegment]]:
        pass

    def get_lane_segments_from_poses(self, poses: List[Transformation]) -> List[LaneSegment]:
        pass

    def get_lane_segments_for_point(self, point: Point3DGeometry) -> List[LaneSegment]:
        pass

    def get_road_segments_within_bounds(
        self,
        bounds: BoundingBox2DGeometry,
        method: str = "inside",
    ) -> List[LaneSegment]:
        pass

    def get_lane_segments_within_bounds(
        self,
        bounds: BoundingBox2DGeometry,
        method: str = "inside",
    ) -> List[LaneSegment]:
        pass

    def get_areas_within_bounds(
        self,
        bounds: BoundingBox2DGeometry,
        method: str = "inside",
    ) -> List[Area]:
        pass

    def get_lane_segment_successors_random_path(
        self, lane_segment_id: LaneSegmentId, steps: int = None
    ) -> List[LaneSegment]:
        pass

    def get_lane_segment_predecessors_random_path(
        self, lane_segment_id: LaneSegmentId, steps: int = None
    ) -> List[LaneSegment]:
        pass

    def get_lane_segments_connected_shortest_paths(
        self, source_id: LaneSegmentId, target_id: LaneSegmentId
    ) -> List[List[LaneSegment]]:
        pass

    def bridge_lane_segments(
        self, id_1: LaneSegmentId, id_2: LaneSegmentId, bridge_length: int = None, directed: bool = True
    ) -> Optional[List[LaneSegment]]:
        pass

    def complete_lane_segments(
        self, lane_segment_ids: List[Optional[LaneSegmentId]], directed: bool = True
    ) -> List[Optional[LaneSegment]]:
        pass


class MapDecoderProtocol(Protocol):
    def get_road_segments(self) -> List[RoadSegment]:
        pass

    def get_lane_segments(self) -> List[LaneSegment]:
        pass

    def get_junctions(self) -> List[Junction]:
        pass

    def get_areas(self) -> List[Area]:
        pass

    def get_map_query(self) -> MapQueryProtocol:
        pass


class Map2:
    def __init__(self, map_decoder: MapDecoderProtocol):
        self.map_query = map_decoder.get_map_query()
        self._map_decoder = map_decoder

    @property
    def road_segments(self) -> List[RoadSegment]:
        return self._map_decoder.get_road_segments()

    @property
    def lane_segments(self) -> List[LaneSegment]:
        return self._map_decoder.get_lane_segments()

    @property
    def junctions(self) -> List[Junction]:
        return self._map_decoder.get_junctions()

    @property
    def areas(self) -> List[Area]:
        return self._map_decoder.get_areas()

    def get_lane_segments_from_poses(self, poses: List[Transformation]) -> List[LaneSegment]:
        return self.map_query.get_lane_segments_from_poses(poses=poses)

    def pad_lane_segments(self, lane_segments: List[LaneSegment], padding: int = 1) -> List[LaneSegment]:
        lane_segments_predecessors = self.map_query.get_lane_segment_predecessors_random_path(
            lane_segment_id=lane_segments[0].lane_segment_id, steps=padding
        )
        lane_segments_successors = self.map_query.get_lane_segment_successors_random_path(
            lane_segment_id=lane_segments[-1].lane_segment_id, steps=padding
        )

        return lane_segments_predecessors[::-1][:-1] + lane_segments + lane_segments_successors[1:]

    def get_road_segments_within_bounds(
        self,
        bounds: BoundingBox2DGeometry,
        method: str = "inside",
    ) -> List[LaneSegment]:
        return self.map_query.get_road_segments_within_bounds(bounds=bounds, method=method)

    def get_lane_segments_within_bounds(
        self,
        bounds: BoundingBox2DGeometry,
        method: str = "inside",
    ) -> List[LaneSegment]:
        return self.map_query.get_lane_segments_within_bounds(bounds=bounds, method=method)

    def get_areas_within_bounds(
        self,
        bounds: BoundingBox2DGeometry,
        method: str = "inside",
    ) -> List[Area]:
        return self.map_query.get_areas_within_bounds(bounds=bounds, method=method)

    def get_lane_segment_successors_random_path(
        self, lane_segment: Union[LaneSegmentId, LaneSegment], steps: int = None
    ) -> List[LaneSegment]:
        if isinstance(lane_segment, LaneSegment):
            lane_segment = lane_segment.lane_segment_id
        return self.map_query.get_lane_segment_successors_random_path(lane_segment_id=lane_segment, steps=steps)

    def get_lane_segment_predecessors_random_path(
        self, lane_segment: Union[LaneSegmentId, LaneSegment], steps: int = None
    ) -> List[LaneSegment]:
        if isinstance(lane_segment, LaneSegment):
            lane_segment = lane_segment.lane_segment_id
        return self.map_query.get_lane_segment_predecessors_random_path(lane_segment_id=lane_segment, steps=steps)

    def bridge_lane_segments(
        self,
        lane_segment_1: Union[LaneSegmentId, LaneSegment],
        lane_segment_2: Union[LaneSegmentId, LaneSegment],
        bridge_length: int = None,
        directed: bool = True,
    ) -> Optional[List[LaneSegment]]:
        if isinstance(lane_segment_1, LaneSegment):
            lane_segment_1 = lane_segment_1.lane_segment_id
        if isinstance(lane_segment_2, LaneSegment):
            lane_segment_2 = lane_segment_2.lane_segment_id
        return self.map_query.bridge_lane_segments(
            id_1=lane_segment_1, id_2=lane_segment_2, bridge_length=bridge_length, directed=directed
        )

    def complete_lane_segments(
        self, lane_segments: List[Union[LaneSegmentId, LaneSegment]], directed: bool = True
    ) -> List[Optional[LaneSegment]]:
        lane_segments = [ls.lane_segment_id if isinstance(ls, LaneSegment) else ls for ls in lane_segments]
        return self.map_query.complete_lane_segments(lane_segment_ids=lane_segments, directed=directed)
