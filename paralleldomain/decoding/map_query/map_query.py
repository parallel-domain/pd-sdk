import abc
from typing import Dict, List, Optional, Tuple

from paralleldomain.model.map.edge import Edge

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DBaseGeometry
from paralleldomain.model.geometry.point_3d import Point3DGeometry
from paralleldomain.model.map.area import Area
from paralleldomain.model.map.map_components import Junction, LaneSegment, RoadSegment
from paralleldomain.model.type_aliases import AreaId, EdgeId, JunctionId, LaneSegmentId, RoadSegmentId
from paralleldomain.utilities.transformation import Transformation


class MapQuery:
    @abc.abstractmethod
    def add_map_data(
        self,
        road_segments: Dict[RoadSegmentId, RoadSegment],
        lane_segments: Dict[LaneSegmentId, LaneSegment],
        junctions: Dict[JunctionId, Junction],
        areas: Dict[AreaId, Area],
        edges: Dict[EdgeId, Edge],
    ):
        pass

    @abc.abstractmethod
    def get_junction(self, junction_id: JunctionId) -> Optional[Junction]:
        pass

    @abc.abstractmethod
    def get_road_segment(self, road_segment_id: RoadSegmentId) -> Optional[RoadSegment]:
        pass

    @abc.abstractmethod
    def get_lane_segment(self, lane_segment_id: LaneSegmentId) -> Optional[LaneSegment]:
        pass

    def get_lane_segments(self, lane_segment_ids: List[LaneSegmentId]) -> List[Optional[LaneSegment]]:
        return [self.get_lane_segment(lane_segment_id=lid) for lid in lane_segment_ids]

    @abc.abstractmethod
    def get_area(self, area_id: AreaId) -> Area:
        pass

    @abc.abstractmethod
    def get_edge(self, edge_id: EdgeId) -> Edge:
        pass

    @abc.abstractmethod
    def get_lane_segments_successors_shortest_paths(
        self, source_id: LaneSegmentId, target_id: LaneSegmentId
    ) -> List[List[LaneSegment]]:
        pass

    @abc.abstractmethod
    def get_lane_segments_from_poses(self, poses: List[Transformation]) -> List[LaneSegment]:
        pass

    @abc.abstractmethod
    def get_lane_segments_for_point(self, point: Point3DGeometry) -> List[LaneSegment]:
        pass

    @abc.abstractmethod
    def get_road_segments_within_bounds(
        self,
        bounds: BoundingBox2DBaseGeometry[float],
        method: str = "inside",
    ) -> List[LaneSegment]:
        pass

    @abc.abstractmethod
    def get_lane_segments_within_bounds(
        self,
        bounds: BoundingBox2DBaseGeometry[float],
        method: str = "inside",
    ) -> List[LaneSegment]:
        pass

    @abc.abstractmethod
    def get_areas_within_bounds(
        self,
        bounds: BoundingBox2DBaseGeometry[float],
        method: str = "inside",
    ) -> List[Area]:
        pass

    @abc.abstractmethod
    def get_lane_segment_successors_random_path(
        self, lane_segment_id: LaneSegmentId, steps: int = None
    ) -> List[LaneSegment]:
        pass

    @abc.abstractmethod
    def get_lane_segment_predecessors_random_path(
        self, lane_segment_id: LaneSegmentId, steps: int = None
    ) -> List[LaneSegment]:
        pass

    @abc.abstractmethod
    def get_junctions_for_lane_segment(self, lane_segment_id: LaneSegmentId) -> List[Junction]:
        pass

    @abc.abstractmethod
    def get_road_segment_for_lane_segment(self, lane_segment_id: LaneSegmentId) -> Optional[RoadSegment]:
        pass

    @abc.abstractmethod
    def get_lane_segment_successors(
        self, lane_segment_id: int, depth: int = -1
    ) -> Tuple[List[LaneSegment], List[int], List[Optional[int]]]:
        pass

    @abc.abstractmethod
    def get_lane_segments_connected_shortest_paths(self, source_id: int, target_id: int) -> List[List[LaneSegment]]:
        pass

    @abc.abstractmethod
    def get_predecessors(self, depth: int = -1) -> List["LaneSegment"]:
        pass

    @abc.abstractmethod
    def get_successors(self, depth: int = -1) -> List["LaneSegment"]:
        pass

    @abc.abstractmethod
    def get_relative_left_neighbor(self, lane_segment_id: LaneSegmentId, degree: int = 1) -> Optional["LaneSegment"]:
        pass

    @abc.abstractmethod
    def get_relative_right_neighbor(self, lane_segment_id: LaneSegmentId, degree: int = 1) -> Optional["LaneSegment"]:
        pass

    @abc.abstractmethod
    def complete_lane_segments(
        self, lane_segment_ids: List[Optional[LaneSegmentId]], directed: bool = True
    ) -> List[Optional[LaneSegment]]:
        pass

    @abc.abstractmethod
    def are_connected_lane_segments(self, id_1: LaneSegmentId, id_2: LaneSegmentId) -> bool:
        pass

    @abc.abstractmethod
    def are_succeeding_lane_segments(self, id_1: LaneSegmentId, id_2: LaneSegmentId) -> bool:
        pass

    @abc.abstractmethod
    def are_preceeding_lane_segments(self, id_1: LaneSegmentId, id_2: LaneSegmentId) -> bool:
        pass

    def bridge_lane_segments(
        self, id_1: LaneSegmentId, id_2: LaneSegmentId, bridge_length: int = None, directed: bool = True
    ) -> Optional[List[LaneSegment]]:
        if directed:
            if not self.are_succeeding_lane_segments(id_1=id_1, id_2=id_2):
                shortest_paths = self.get_lane_segments_successors_shortest_paths(source_id=id_1, target_id=id_2)
                if shortest_paths and (
                    bridge_length is None or len(shortest_paths[0]) - 2 == bridge_length
                ):  # shortest path exists with max gap length
                    return shortest_paths[0][1:-1]  # return only bridge elements
                else:
                    return None
            else:  # LS are directly connected, bridge is empty.
                return []
        else:
            if not self.are_connected_lane_segments(id_1=id_1, id_2=id_2):
                shortest_paths = self.get_lane_segments_connected_shortest_paths(source_id=id_1, target_id=id_2)
                if shortest_paths and (
                    bridge_length is None or len(shortest_paths[0]) - 2 == bridge_length
                ):  # shortest path exists with max gap length
                    return shortest_paths[0][1:-1]  # return only bridge elements
                else:
                    return None
            else:  # LS are directly connected, bridge is empty.
                return []

    def are_opposite_direction_lane_segments(self, id_1: LaneSegmentId, id_2: LaneSegmentId) -> Optional[bool]:
        lane_segment_1 = self.get_lane_segment(lane_segment_id=id_1)
        lane_segment_2 = self.get_lane_segment(lane_segment_id=id_2)

        road_segment_1 = self.get_road_segment_for_lane_segment(lane_segment_id=id_1)
        road_segment_2 = self.get_road_segment_for_lane_segment(lane_segment_id=id_2)

        if road_segment_1 == road_segment_2:
            return True if lane_segment_1.direction != lane_segment_2.direction else False
        else:
            lane_segment_1_connected = lane_segment_1.get_predecessors(depth=1) + lane_segment_1.get_successors(depth=1)
            lane_segment_2_connected = lane_segment_2.get_predecessors(depth=1) + lane_segment_2.get_successors(depth=1)

            lane_segment_1_connected_road_segments = [
                self.get_road_segment_for_lane_segment(lane_segment_id=ls.lane_segment_id).road_segment_id
                for ls in lane_segment_1_connected
            ]

            lane_segment_2_connected_road_segments = [
                self.get_road_segment_for_lane_segment(lane_segment_id=ls.lane_segment_id).road_segment_id
                for ls in lane_segment_2_connected
            ]

            road_segment_intersection = set.intersection(
                set(lane_segment_1_connected_road_segments), set(lane_segment_2_connected_road_segments)
            )

            try:
                road_segment = next(iter(road_segment_intersection))
            except StopIteration:
                return None

            lane_segment_1_in_road_segment = lane_segment_1_connected[
                lane_segment_1_connected_road_segments.index(road_segment)
            ]

            lane_segment_2_in_road_segment = lane_segment_2_connected[
                lane_segment_2_connected_road_segments.index(road_segment)
            ]

            return (
                True if lane_segment_1_in_road_segment.direction != lane_segment_2_in_road_segment.direction else False
            )
