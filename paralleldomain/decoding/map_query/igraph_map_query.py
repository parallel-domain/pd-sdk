from itertools import chain, groupby
from typing import Dict, List, Optional, Tuple

import numpy as np
from more_itertools import split_when, triplewise

from paralleldomain.decoding.map_query.map_query import MapQuery
from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DBaseGeometry
from paralleldomain.model.geometry.point_3d import Point3DGeometry
from paralleldomain.model.map.area import Area
from paralleldomain.model.map.edge import Edge
from paralleldomain.model.map.map_components import Junction, LaneSegment, RoadSegment
from paralleldomain.model.type_aliases import AreaId, EdgeId, JunctionId, LaneSegmentId, RoadSegmentId
from paralleldomain.utilities.geometry import is_point_in_polygon_2d
from paralleldomain.utilities.transformation import Transformation

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

from igraph import Graph, Vertex


class NodePrefix:
    ROAD_SEGMENT: str = "RS"
    LANE_SEGMENT: str = "LS"
    JUNCTION: str = "JC"
    AREA: str = "AR"


class _MayBeHasBounds(Protocol):
    @property
    def bounds(self) -> Optional[BoundingBox2DBaseGeometry[float]]:
        return


class IGraphMapQuery(MapQuery):
    def __init__(self):
        super().__init__()
        self.edges = dict()
        self.__map_graph = Graph(directed=True)
        self._added_road_segments = False
        self._added_lane_segments = False
        self._added_junctions = False
        self._added_areas = False

    def add_map_data(
        self,
        road_segments: Dict[RoadSegmentId, RoadSegment],
        lane_segments: Dict[LaneSegmentId, LaneSegment],
        junctions: Dict[JunctionId, Junction],
        areas: Dict[AreaId, Area],
        edges: Dict[EdgeId, Edge],
    ):
        self.edges.update(edges)
        self._add_road_segments_to_graph(road_segments=road_segments)
        self._add_lane_segments_to_graph(lane_segments=lane_segments)
        self._add_junctions_to_graph(junctions=junctions)
        self._add_areas_to_graph(areas=areas)

    @property
    def map_graph(self) -> Graph:
        return self.__map_graph

    def get_junction(self, junction_id: JunctionId) -> Optional[Junction]:
        query_results = self.map_graph.vs.select(name_eq=f"{NodePrefix.JUNCTION}_{junction_id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    def get_road_segment(self, road_segment_id: RoadSegmentId) -> Optional[RoadSegment]:
        query_results = self.map_graph.vs.select(name_eq=f"{NodePrefix.ROAD_SEGMENT}_{road_segment_id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    def get_lane_segment(self, lane_segment_id: LaneSegmentId) -> Optional[LaneSegment]:
        query_results = self.map_graph.vs.select(name_eq=f"{NodePrefix.LANE_SEGMENT}_{lane_segment_id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    def get_area(self, area_id: AreaId) -> Area:
        query_results = self.map_graph.vs.select(name_eq=f"{NodePrefix.AREA}_{area_id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    def get_edge(self, edge_id: EdgeId) -> Edge:
        return self.edges[edge_id]

    def get_road_segment_for_lane_segment(self, lane_segment_id: LaneSegmentId) -> Optional[RoadSegment]:
        subgraph = self._get_road_segments_containing_lane_segments_graph()
        target_node = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id}")

        road_segments = subgraph.es.select(_target=target_node)

        if len(road_segments) > 0:
            return road_segments[0].source_vertex["object"]
        else:
            return None

    def get_lane_segment_successors(
        self, lane_segment_id: int, depth: int = -1
    ) -> Tuple[List[LaneSegment], List[int], List[Optional[int]]]:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        start_node = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{lane_segment_id}")

        lane_segments = []
        levels = []
        parent_ids = []

        level = -1
        bfs_iterator = subgraph.bfsiter(vid=start_node.index, mode="out", advanced=True)
        while depth == -1 or level < depth:
            try:
                lane_segment, level, parent_lane_segment = next(bfs_iterator)
                lane_segments.append(lane_segment["object"])
                levels.append(level)
                parent_ids.append(parent_lane_segment["object"].id if parent_lane_segment is not None else None)
            except StopIteration:
                break

        return lane_segments, levels, parent_ids

    def get_lane_segments_connected_shortest_paths(self, source_id: int, target_id: int) -> List[List[LaneSegment]]:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        start_node = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{source_id}")
        end_node = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{target_id}")

        shortest_paths = subgraph.get_shortest_paths(v=start_node, to=end_node, mode="all")
        return [[subgraph.vs[node_id]["object"] for node_id in shortest_path] for shortest_path in shortest_paths]

    def get_predecessors(self, depth: int = -1) -> List["LaneSegment"]:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        start_node = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id}")

        lane_segments = []

        bfs_iterator = subgraph.bfsiter(vid=start_node.index, mode="in", advanced=True)
        while True:
            try:
                lane_segment, level, parent_lane_segment = next(bfs_iterator)
                if level > depth:
                    break
                lane_segments.append(lane_segment["object"])
            except StopIteration:
                break

        return lane_segments

    def get_successors(self, depth: int = -1) -> List["LaneSegment"]:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        start_node = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id}")

        lane_segments = []

        level = -1
        bfs_iterator = subgraph.bfsiter(vid=start_node.index, mode="out", advanced=True)
        while depth == -1 or level < depth:
            try:
                lane_segment, level, parent_lane_segment = next(bfs_iterator)
                lane_segments.append(lane_segment["object"])
            except StopIteration:
                break

        return lane_segments

    def get_relative_left_neighbor(self, lane_segment_id: LaneSegmentId, degree: int = 1) -> Optional["LaneSegment"]:
        lane_segment = self.get_lane_segment(lane_segment_id=lane_segment_id)

        neighbor = lane_segment  # set to initial lane segment and start the loop if degree >= 1
        for i in range(degree):
            neighbor = (
                neighbor.left_neighbor
                if not self.are_opposite_direction_lane_segments(
                    id_1=lane_segment.lane_segment_id, id_2=neighbor.lane_segment_id
                )
                else neighbor.right_neighbor
            )
            if neighbor is None:
                break

        return neighbor

    def get_relative_right_neighbor(self, lane_segment_id: LaneSegmentId, degree: int = 1) -> Optional["LaneSegment"]:
        lane_segment = self.get_lane_segment(lane_segment_id=lane_segment_id)

        neighbor = lane_segment  # set to initial lane segment and start the loop if degree >= 1
        for i in range(degree):
            neighbor = (
                neighbor.right_neighbor
                if not self.are_opposite_direction_lane_segments(
                    id_1=lane_segment.lane_segment_id, id_2=neighbor.lane_segment_id
                )
                else neighbor.left_neighbor
            )
            if neighbor is None:
                break

        return neighbor

    def complete_lane_segments(
        self, lane_segment_ids: List[Optional[LaneSegmentId]], directed: bool = True
    ) -> List[Optional[LaneSegment]]:
        if directed:
            groups = self._group_succeeding_lane_segments(lane_segment_ids=lane_segment_ids)
        else:
            groups = self._group_connected_lane_segments(lane_segment_ids=lane_segment_ids)

        # Minimum groups required, because we need to have [..,LS],[None,...],[LS,...] to completed gaps
        if len(groups) >= 3:
            for i, (start_group, bridge_group, end_group) in enumerate(triplewise(groups)):
                start_element = start_group[-1]
                end_element = end_group[0]
                if start_element is not None and end_element is not None:
                    bridge_candidates = self.bridge_lane_segments(
                        id_1=start_element.lane_segment_id,
                        id_2=end_element.lane_segment_id,
                        bridge_length=len(bridge_group),
                        directed=directed,
                    )
                    if bridge_candidates is not None:
                        groups[i + 1] = bridge_candidates

        return [
            self.get_lane_segment(ls.lane_segment_id) if ls is not None else None for ls in chain.from_iterable(groups)
        ]

    def are_connected_lane_segments(self, id_1: LaneSegmentId, id_2: LaneSegmentId) -> bool:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        node_1 = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id_1}")
        node_2 = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id_2}")
        edge_id = subgraph.get_eid(node_1, node_2, directed=False, error=False)
        return False if edge_id == -1 else True

    def are_succeeding_lane_segments(self, id_1: LaneSegmentId, id_2: LaneSegmentId) -> bool:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        node_1 = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id_1}")
        node_2 = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id_2}")
        edge_id = subgraph.get_eid(node_1, node_2, directed=True, error=False)
        return False if edge_id == -1 else True

    def are_preceeding_lane_segments(self, id_1: LaneSegmentId, id_2: LaneSegmentId) -> bool:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        node_1 = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id_1}")
        node_2 = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id_2}")
        edge_id = subgraph.get_eid(node_2, node_1, directed=True, error=False)
        return False if edge_id == -1 else True

    def get_junctions_for_lane_segment(self, lane_segment_id: LaneSegmentId) -> List[Junction]:
        subgraph = self._get_junctions_containing_lane_segments_graph()
        source_node = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{lane_segment_id}")

        junctions = subgraph.es.select(_target=source_node)

        return [j.source_vertex["object"] for j in junctions]

    def get_lane_segment_predecessors_random_path(
        self, lane_segment_id: LaneSegmentId, steps: int = None
    ) -> List[LaneSegment]:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        start_vertex = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{lane_segment_id}")
        random_walk = subgraph.random_walk(
            start=start_vertex.index,
            steps=steps if steps is not None else (2**16),
            mode="in",
            stuck="return",
        )
        return [subgraph.vs[node_id]["object"] for node_id in random_walk]

    def get_lane_segment_successors_random_path(
        self, lane_segment_id: LaneSegmentId, steps: int = None
    ) -> List[LaneSegment]:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        start_vertex = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{lane_segment_id}")
        random_walk = subgraph.random_walk(
            start=start_vertex.index,
            steps=steps if steps is not None else (2**16),
            mode="out",
            stuck="return",
        )
        return [subgraph.vs[node_id]["object"] for node_id in random_walk]

    def get_lane_segments_successors_shortest_paths(
        self, source_id: LaneSegmentId, target_id: LaneSegmentId
    ) -> List[List[LaneSegment]]:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        start_node = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{source_id}")
        end_node = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{target_id}")

        shortest_paths = subgraph.get_shortest_paths(v=start_node, to=end_node.index, mode="out")
        return [[subgraph.vs[node_id]["object"] for node_id in shortest_path] for shortest_path in shortest_paths]

    def get_lane_segments_from_poses(self, poses: List[Transformation]) -> List[LaneSegment]:
        enu_points = [Point3DGeometry.from_numpy(point=pose.translation) for pose in poses]
        lane_segments_candidates = [self.get_lane_segments_for_point(point=point) for point in enu_points]

        # 2 Group all "single lane segment" matches and all "more than one lane segment" matches (ambiguous!)
        lane_segments_candidates_length = [len(s) > 1 for s in lane_segments_candidates]
        lane_segments_candidates_grouped = groupby(
            range(len(lane_segments_candidates_length)), lambda x: lane_segments_candidates_length[x]
        )
        lane_segments_candidates_grouped = [(g[0], list(g[1])) for g in lane_segments_candidates_grouped]

        # 2B Assert that the first and last pose in the vehicle path is of type "single lane segment"
        assert lane_segments_candidates_grouped[0][0] is False  # for now until extrapolation logic is implemented
        assert lane_segments_candidates_grouped[-1][0] is False  # for now until extrapolation logic is implemented

        # 3 For each group, either do nothing ("single lane segment") or for "more than one lane segment":
        #   Take the previous and next "single lane segment" and calculate their shortest lane segment path
        #   In each "more than one lane segment" timestamp, then find which node from the shortest path is within
        #   Return only that single node and convert this timestamp to "single lane segment"
        for i, (a, seq) in enumerate(lane_segments_candidates_grouped):
            if a is True:
                _, prev_indices = lane_segments_candidates_grouped[i - 1]
                _, next_indices = lane_segments_candidates_grouped[i + 1]
                prev_node = lane_segments_candidates[prev_indices[-1]][0]
                next_node = lane_segments_candidates[next_indices[0]][0]
                shortest_path = self.get_lane_segments_successors_shortest_paths(
                    source_id=prev_node.lane_segment_id, target_id=next_node.lane_segment_id
                )
                shortest_path = shortest_path[0]
                for idx in seq:
                    candidates = lane_segments_candidates[idx]
                    for n in shortest_path[::-1]:  # reverse, greedy search
                        if n in candidates:
                            lane_segments_candidates[idx] = [n]
                            break

        # 4 Now that everything is an unambiguous "single lane segment", convert list of lists of nodes to list of nodes
        lane_segments = [m[0] for m in lane_segments_candidates]
        # 4B We do not want re-occurences, just interest in the unique segment path
        # lane_segments = list(unique_everseen(lane_segments))

        return lane_segments

    def get_lane_segments_for_point(self, point: Point3DGeometry) -> List[LaneSegment]:
        bounds = BoundingBox2DBaseGeometry[float](x=point.x, y=point.y, width=0.1, height=0.1)
        ls_candidates = self.get_lane_segments_within_bounds(bounds=bounds, method="overlap")
        if ls_candidates:
            point_under_test = (point.x, point.y)
            return [
                ls_candidates[i]
                for i, polygon in enumerate(ls_candidates)
                if is_point_in_polygon_2d(polygon=polygon.to_numpy(closed=True)[:, :2], point=point_under_test)
            ]
        else:
            return []

    def get_road_segments_within_bounds(
        self, bounds: BoundingBox2DBaseGeometry[float], method: str = "inside"
    ) -> List[LaneSegment]:
        return self._get_nodes_within_bounds(node_prefix=NodePrefix.ROAD_SEGMENT, bounds=bounds, method=method)

    def get_lane_segments_within_bounds(
        self,
        bounds: BoundingBox2DBaseGeometry[float],
        method: str = "inside",
    ) -> List[LaneSegment]:
        return self._get_nodes_within_bounds(node_prefix=NodePrefix.LANE_SEGMENT, bounds=bounds, method=method)

    def get_areas_within_bounds(
        self,
        bounds: BoundingBox2DBaseGeometry[float],
        method: str = "inside",
    ) -> List[Area]:
        return self._get_nodes_within_bounds(node_prefix=NodePrefix.AREA, bounds=bounds, method=method)

    def _get_nodes_within_bounds(
        self,
        node_prefix: str,
        bounds: BoundingBox2DBaseGeometry[float],
        method: str = "inside",
    ) -> List:
        if method == "inside":
            return [
                vv["object"]
                for vv in self.map_graph.vs.select(
                    lambda v: v["name"].startswith(node_prefix)
                    and v["x_max"] <= bounds.x_max
                    and v["y_max"] <= bounds.y_max
                    and v["x_min"] >= bounds.x_min
                    and v["y_min"] >= bounds.y_min
                )
            ]
        elif method == "overlap":
            return [
                vv["object"]
                for vv in self.map_graph.vs.select(
                    lambda v: v["name"].startswith(node_prefix)
                    and (min(v["x_max"], bounds.x_max) - max(v["x_min"], bounds.x_min)) >= 0
                    and (min(v["y_max"], bounds.y_max) - max(v["y_min"], bounds.y_min)) >= 0
                )
            ]

        elif method == "center":
            return [
                vv["object"]
                for vv in self.map_graph.vs.select(
                    lambda v: v["name"].startswith(node_prefix)
                    and v["x_center"] <= bounds.x_max
                    and v["y_center"] <= bounds.y_max
                    and v["x_center"] >= bounds.x_min
                    and v["y_center"] >= bounds.y_min
                )
            ]

    def _get_lane_segments_preceeding_lane_segments_graph(self) -> Graph:
        return self.map_graph.subgraph_edges(
            self.map_graph.es.select(type_eq=f"{NodePrefix.LANE_SEGMENT}_preceeds_{NodePrefix.LANE_SEGMENT}"),
            delete_vertices=False,
        )

    def _get_junctions_containing_lane_segments_graph(self) -> Graph:
        return self.map_graph.subgraph_edges(
            self.map_graph.es.select(type_eq=f"{NodePrefix.JUNCTION}_contains_{NodePrefix.LANE_SEGMENT}"),
            delete_vertices=False,
        )

    def _get_road_segments_containing_lane_segments_graph(self) -> Graph:
        return self.map_graph.subgraph_edges(
            self.map_graph.es.select(type_eq=f"{NodePrefix.ROAD_SEGMENT}_contains_{NodePrefix.LANE_SEGMENT}"),
            delete_vertices=False,
        )

    def _build_graph(
        self,
        road_segments: Dict[RoadSegmentId, RoadSegment],
        lane_segments: Dict[LaneSegmentId, LaneSegment],
        junctions: Dict[JunctionId, Junction],
        areas: Dict[AreaId, Area],
        edges: Dict[EdgeId, Edge],
    ):
        self._add_road_segments_to_graph(road_segments=road_segments)
        self._add_lane_segments_to_graph(lane_segments=lane_segments)
        self._add_junctions_to_graph(junctions=junctions)
        self._add_areas_to_graph(areas=areas)

    def _add_vertex(self, has_bounds: _MayBeHasBounds, vertex_id: str) -> Vertex:
        if has_bounds.bounds is not None:
            x_min = has_bounds.bounds.x
            x_max = has_bounds.bounds.x + has_bounds.bounds.width
            y_min = has_bounds.bounds.y
            y_max = has_bounds.bounds.y + has_bounds.bounds.height
        else:
            x_min = x_max = y_min = y_max = None

        vertex = self.__map_graph.add_vertex(
            name=vertex_id,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            x_center=x_min + (x_max - x_min) / 2 if x_min is not None else None,
            y_center=y_min + (y_max - y_min) / 2 if y_min is not None else None,
            object=has_bounds,
        )
        return vertex

    def _add_road_segments_to_graph(self, road_segments: Dict[RoadSegmentId, RoadSegment]):
        if self._added_road_segments:
            return

        road_segment_nodes = {}
        road_segment_edges_x_precedes_y = []  # RoadSegment -> RoadSegment
        # for rs_key, rs_val in self._umd_map.road_segments.items():
        for road_segment_id, road_segment in road_segments.items():
            road_segment_node_id = f"{NodePrefix.ROAD_SEGMENT}_{road_segment.road_segment_id}"
            road_segment_node = self._add_vertex(has_bounds=road_segment, vertex_id=road_segment_node_id)

            road_segment_nodes[road_segment_node_id] = road_segment_node.index

            road_segment_edges_x_precedes_y.extend(
                [
                    (road_segment_node_id, f"{NodePrefix.ROAD_SEGMENT}_{successor}")
                    for successor in road_segment.successor_ids
                    if successor != 0
                ]
            )

        for rs_source, rs_target in road_segment_edges_x_precedes_y:
            rs_source_index = road_segment_nodes[rs_source]
            rs_target_index = road_segment_nodes[rs_target]
            self.__map_graph.add_edge(
                source=rs_source_index,
                target=rs_target_index,
                type=f"{NodePrefix.ROAD_SEGMENT}_preceeds_{NodePrefix.ROAD_SEGMENT}",
            )
        self._added_road_segments = True

    def _add_lane_segments_to_graph(self, lane_segments: Dict[LaneSegmentId, LaneSegment]):
        if self._added_lane_segments:
            return

        lane_segment_nodes = {}
        lane_segment_edges_x_precedes_y = []  # LaneSegment -> LaneSegment
        lane_segment_edges_x_contains_y = []  # RoadSegment -> LaneSegment
        lane_segment_edges_x_to_the_left_of_y = []  # LaneSegment -> LaneSegment

        for lane_segment_id, lane_segment in lane_segments.items():
            lane_segment_node_id = f"{NodePrefix.LANE_SEGMENT}_{lane_segment.lane_segment_id}"
            lane_segment_node = self._add_vertex(has_bounds=lane_segment, vertex_id=lane_segment_node_id)

            lane_segment_nodes[lane_segment_node_id] = lane_segment_node.index

            lane_segment_edges_x_precedes_y.extend(
                [
                    (lane_segment_node_id, f"{NodePrefix.LANE_SEGMENT}_{successor}")
                    for successor in lane_segment.successor_ids
                    if successor != 0
                ]
            )
            if lane_segment.right_neighbor is not None:
                lane_segment_edges_x_to_the_left_of_y.append(
                    (lane_segment_node_id, f"{NodePrefix.LANE_SEGMENT}_{lane_segment.right_neighbor.lane_segment_id}")
                )
            if lane_segment.parent_road_segment_id is not None and lane_segment.parent_road_segment_id != 0:
                lane_segment_edges_x_contains_y.append(
                    (f"{NodePrefix.ROAD_SEGMENT}_{lane_segment.parent_road_segment_id}", lane_segment_node_id)
                )

        for ls_source, ls_target in lane_segment_edges_x_precedes_y:
            ls_source_index = lane_segment_nodes[ls_source]
            ls_target_index = lane_segment_nodes[ls_target]
            self.__map_graph.add_edge(
                source=ls_source_index,
                target=ls_target_index,
                type=f"{NodePrefix.LANE_SEGMENT}_preceeds_{NodePrefix.LANE_SEGMENT}",
            )

        for ls_source, ls_target in lane_segment_edges_x_to_the_left_of_y:
            ls_source_index = lane_segment_nodes[ls_source]
            ls_target_index = lane_segment_nodes[ls_target]
            self.__map_graph.add_edge(
                source=ls_source_index,
                target=ls_target_index,
                type=f"{NodePrefix.LANE_SEGMENT}_left_of_{NodePrefix.LANE_SEGMENT}",
            )

        road_segment_nodes = {
            n["name"]: n.index for n in self.__map_graph.vs if n["name"].startswith(NodePrefix.ROAD_SEGMENT)
        }

        for rs_source, ls_target in lane_segment_edges_x_contains_y:
            rs_source_index = road_segment_nodes[rs_source]
            ls_target_index = lane_segment_nodes[ls_target]
            self.__map_graph.add_edge(
                source=rs_source_index,
                target=ls_target_index,
                type=f"{NodePrefix.ROAD_SEGMENT}_contains_{NodePrefix.LANE_SEGMENT}",
            )
        self._added_lane_segments = True

    def _add_junctions_to_graph(self, junctions: Dict[JunctionId, Junction]):
        if self._added_junctions:
            return

        junction_nodes = {}
        junction_contains_road_segment = []  # Junction -> RoadSegment
        junction_contains_lane_segment = []  # Junction -> LaneSegment

        # edges = self.get_edges()

        for junction_id, junction in junctions.items():
            junction_node_id = f"{NodePrefix.JUNCTION}_{junction.junction_id}"
            junction_node = self._add_vertex(has_bounds=junction, vertex_id=junction_node_id)

            junction_nodes[junction_node_id] = junction_node.index

            junction_contains_road_segment.extend(
                [
                    (junction_node_id, f"{NodePrefix.ROAD_SEGMENT}_{road_segment.road_segment_id}")
                    for road_segment in junction.road_segments
                ]
            )

            junction_contains_lane_segment.extend(
                [
                    (junction_node_id, f"{NodePrefix.LANE_SEGMENT}_{lane_segment.lane_segment_id}")
                    for lane_segment in junction.lane_segments
                ]
            )

        road_segment_nodes = {
            n["name"]: n.index for n in self.__map_graph.vs if n["name"].startswith(NodePrefix.ROAD_SEGMENT)
        }

        for j_source, rs_target in junction_contains_road_segment:
            j_source_index = junction_nodes[j_source]
            rs_target_index = road_segment_nodes[rs_target]
            self.__map_graph.add_edge(
                source=j_source_index,
                target=rs_target_index,
                type=f"{NodePrefix.JUNCTION}_contains_{NodePrefix.ROAD_SEGMENT}",
            )

        lane_segment_nodes = {
            n["name"]: n.index for n in self.__map_graph.vs if n["name"].startswith(NodePrefix.LANE_SEGMENT)
        }

        for j_source, ls_target in junction_contains_lane_segment:
            j_source_index = junction_nodes[j_source]
            ls_target_index = lane_segment_nodes[ls_target]
            self.__map_graph.add_edge(
                source=j_source_index,
                target=ls_target_index,
                type=f"{NodePrefix.JUNCTION}_contains_{NodePrefix.LANE_SEGMENT}",
            )
        self._added_junctions = True

    def _add_areas_to_graph(self, areas: Dict[AreaId, Area]):
        if self._added_areas:
            return

        for area_id, area in areas.items():
            area_node_id = f"{NodePrefix.AREA}_{area.area_id}"
            self._add_vertex(has_bounds=area, vertex_id=area_node_id)

        self._added_areas = True

    def _group_succeeding_lane_segments(
        self, lane_segment_ids: List[LaneSegmentId], reorder: bool = False
    ) -> List[List[Optional[LaneSegment]]]:
        if not reorder:

            def split_fn(x: Optional[LaneSegment], y: Optional[LaneSegment]) -> bool:
                if x is None and y is not None:
                    return True
                elif x is not None and y is None:
                    return True
                elif x is None and y is None:
                    return False
                elif x is not None and y is not None:
                    return not self.are_succeeding_lane_segments(id_1=x.lane_segment_id, id_2=y.lane_segment_id)

            lane_segments = self.get_lane_segments(lane_segment_ids=lane_segment_ids)
            return list(split_when(lane_segments, split_fn))
        else:
            vertices = [self.map_graph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id}") for id in lane_segment_ids]
            subgraph = self._get_lane_segments_preceeding_lane_segments_graph().induced_subgraph(vertices=vertices)

            decomposed_subgraphs = subgraph.decompose(mode="weak")
            lane_segments_groups = []
            for sub in decomposed_subgraphs:
                in_degrees = sub.degree(mode="in")
                out_degrees = sub.degree(mode="out")
                start_vertex = sub.vs[np.argmin(in_degrees)]
                end_vertex = sub.vs[np.argmin(out_degrees)]
                shortest_path = sub.get_shortest_paths(v=start_vertex, to=end_vertex)[-1]
                lane_segments_groups.append([sub.vs[vertex_id]["object"] for vertex_id in shortest_path])

            return lane_segments_groups

    def _group_connected_lane_segments(
        self, lane_segment_ids: List[LaneSegmentId]
    ) -> List[List[Optional[LaneSegment]]]:
        def split_fn(x: Optional[LaneSegment], y: Optional[LaneSegment]) -> bool:
            if x is None and y is not None:
                return True
            elif x is not None and y is None:
                return True
            elif x is None and y is None:
                return False
            elif x is not None and y is not None:
                return not self.are_connected_lane_segments(id_1=x.lane_segment_id, id_2=y.lane_segment_id)

        lane_segments = self.get_lane_segments(lane_segment_ids=lane_segment_ids)
        return list(split_when(lane_segments, split_fn))
