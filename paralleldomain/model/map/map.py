import math
from itertools import chain
from typing import List, Optional, Tuple

import numpy as np
from igraph import Graph
from more_itertools import pairwise, split_when, triplewise

from paralleldomain.common.umd.v1.UMD_pb2 import UniversalMap as ProtoUniversalMap
from paralleldomain.model.map.area import Area
from paralleldomain.model.map.common import NodePrefix
from paralleldomain.model.map.edge import PointENU
from paralleldomain.model.map.junction import Junction
from paralleldomain.model.map.lane_segment import LaneSegment
from paralleldomain.model.map.road_segment import RoadSegment
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_binary_message
from paralleldomain.utilities.geometry import is_point_in_polygon_2d


class Map:
    def __init__(self, umd_map: ProtoUniversalMap) -> None:
        self._umd_map = umd_map

        self._map_graph = Graph(directed=True)

        self._decode_road_segment()
        self._decode_lane_segments()
        self._decode_junctions()
        self._decode_areas()

    def _get_lane_segments_preceeding_lane_segments_graph(self) -> Graph:
        return self._map_graph.subgraph_edges(
            self._map_graph.es.select(type_eq=f"{NodePrefix.LANE_SEGMENT}_preceeds_{NodePrefix.LANE_SEGMENT}"),
            delete_vertices=False,
        )

    def _get_junctions_containing_lane_segments_graph(self) -> Graph:
        return self._map_graph.subgraph_edges(
            self._map_graph.es.select(type_eq=f"{NodePrefix.JUNCTION}_contains_{NodePrefix.LANE_SEGMENT}"),
            delete_vertices=False,
        )

    def _get_road_segments_containing_lane_segments_graph(self) -> Graph:
        return self._map_graph.subgraph_edges(
            self._map_graph.es.select(type_eq=f"{NodePrefix.ROAD_SEGMENT}_contains_{NodePrefix.LANE_SEGMENT}"),
            delete_vertices=False,
        )

    def _decode_road_segment(self) -> None:
        road_segment_nodes = {}
        road_segment_edges_x_precedes_y = []  # RoadSegment -> RoadSegment
        for rs_key, rs_val in self._umd_map.road_segments.items():
            reference_line = self._umd_map.edges[rs_val.reference_line]
            reference_points = np.array([(p.x, p.y) for p in reference_line.points])

            if len(reference_points) > 0:
                x_min, y_min = np.min(reference_points, axis=0)
                x_max, y_max = np.max(reference_points, axis=0)
            else:
                x_min = x_max = y_min = y_max = None

            road_segment_node_id = f"{NodePrefix.ROAD_SEGMENT}_{rs_key}"

            road_segment_node = self._map_graph.add_vertex(
                name=road_segment_node_id,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                x_center=x_min + (x_max - x_min) / 2 if x_min is not None else None,
                y_center=y_min + (y_max - y_min) / 2 if y_min is not None else None,
                object=RoadSegment.from_proto(id=rs_key, umd_map=self._umd_map),
            )

            road_segment_nodes[road_segment_node_id] = road_segment_node.index

            road_segment_edges_x_precedes_y.extend(
                [
                    (road_segment_node_id, f"{NodePrefix.ROAD_SEGMENT}_{successor}")
                    for successor in rs_val.successors
                    if successor != 0
                ]
            )

        for rs_source, rs_target in road_segment_edges_x_precedes_y:
            rs_source_index = road_segment_nodes[rs_source]
            rs_target_index = road_segment_nodes[rs_target]
            self._map_graph.add_edge(
                source=rs_source_index,
                target=rs_target_index,
                type=f"{NodePrefix.ROAD_SEGMENT}_preceeds_{NodePrefix.ROAD_SEGMENT}",
            )

    def _decode_lane_segments(self) -> None:
        lane_segment_nodes = {}
        lane_segment_edges_x_precedes_y = []  # LaneSegment -> LaneSegment
        lane_segment_edges_x_contains_y = []  # RoadSegment -> LaneSegment
        lane_segment_edges_x_to_the_left_of_y = []  # LaneSegment -> LaneSegment

        for ls_key, ls_val in self._umd_map.lane_segments.items():
            reference_line = self._umd_map.edges[ls_val.reference_line]
            reference_points = np.array([(p.x, p.y) for p in reference_line.points])

            left_edge = self._umd_map.edges[ls_val.left_edge]
            left_points = np.array([(p.x, p.y) for p in left_edge.points])

            right_edge = self._umd_map.edges[ls_val.right_edge]
            right_points = np.array([(p.x, p.y) for p in right_edge.points])

            all_points = np.vstack([reference_points, left_points, right_points])

            if len(all_points) > 0:
                x_min, y_min = np.min(all_points, axis=0)
                x_max, y_max = np.max(all_points, axis=0)
            else:
                x_min = x_max = y_min = y_max = None

            lane_segment_node_id = f"{NodePrefix.LANE_SEGMENT}_{ls_key}"

            lane_segment_node = self._map_graph.add_vertex(
                name=lane_segment_node_id,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                x_center=x_min + (x_max - x_min) / 2 if x_min is not None else None,
                y_center=y_min + (y_max - y_min) / 2 if y_min is not None else None,
                object=LaneSegment.from_proto(id=ls_key, umd_map=self._umd_map),
            )

            lane_segment_nodes[lane_segment_node_id] = lane_segment_node.index

            lane_segment_edges_x_precedes_y.extend(
                [
                    (lane_segment_node_id, f"{NodePrefix.LANE_SEGMENT}_{successor}")
                    for successor in ls_val.successors
                    if successor != 0
                ]
            )
            if ls_val.right_neighbor != 0:
                lane_segment_edges_x_to_the_left_of_y.append(
                    (lane_segment_node_id, f"{NodePrefix.LANE_SEGMENT}_{ls_val.right_neighbor}")
                )
            if ls_val.road != 0:
                lane_segment_edges_x_contains_y.append(
                    (f"{NodePrefix.ROAD_SEGMENT}_{ls_val.road}", lane_segment_node_id)
                )

        for ls_source, ls_target in lane_segment_edges_x_precedes_y:
            ls_source_index = lane_segment_nodes[ls_source]
            ls_target_index = lane_segment_nodes[ls_target]
            self._map_graph.add_edge(
                source=ls_source_index,
                target=ls_target_index,
                type=f"{NodePrefix.LANE_SEGMENT}_preceeds_{NodePrefix.LANE_SEGMENT}",
            )

        for ls_source, ls_target in lane_segment_edges_x_to_the_left_of_y:
            ls_source_index = lane_segment_nodes[ls_source]
            ls_target_index = lane_segment_nodes[ls_target]
            self._map_graph.add_edge(
                source=ls_source_index,
                target=ls_target_index,
                type=f"{NodePrefix.LANE_SEGMENT}_left_of_{NodePrefix.LANE_SEGMENT}",
            )

        road_segment_nodes = {
            n["name"]: n.index for n in self._map_graph.vs if n["name"].startswith(NodePrefix.ROAD_SEGMENT)
        }

        for rs_source, ls_target in lane_segment_edges_x_contains_y:
            rs_source_index = road_segment_nodes[rs_source]
            ls_target_index = lane_segment_nodes[ls_target]
            self._map_graph.add_edge(
                source=rs_source_index,
                target=ls_target_index,
                type=f"{NodePrefix.ROAD_SEGMENT}_contains_{NodePrefix.LANE_SEGMENT}",
            )

    def _decode_junctions(self) -> None:
        junction_nodes = {}
        junction_contains_road_segment = []  # Junction -> RoadSegment
        junction_contains_lane_segment = []  # Junction -> LaneSegment

        for jc_key, jc_val in self._umd_map.junctions.items():
            corner_points = np.empty(shape=(0, 2))
            for corner in jc_val.corners:
                corner_edge = self._umd_map.edges[corner]
                corner_points = np.vstack(
                    [
                        corner_points,
                        np.array([(p.x, p.y) for p in corner_edge.points]),
                    ]
                )

            if len(corner_points) > 0:
                x_min, y_min = np.min(corner_points, axis=0)
                x_max, y_max = np.max(corner_points, axis=0)
            else:
                x_min = x_max = y_min = y_max = None

            junction_node_id = f"{NodePrefix.JUNCTION}_{jc_key}"

            junction_node = self._map_graph.add_vertex(
                name=junction_node_id,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                x_center=x_min + (x_max - x_min) / 2 if x_min is not None else None,
                y_center=y_min + (y_max - y_min) / 2 if y_min is not None else None,
                object=Junction.from_proto(id=jc_key, umd_map=self._umd_map),
            )

            junction_nodes[junction_node_id] = junction_node.index

            junction_contains_road_segment.extend(
                [
                    (junction_node_id, f"{NodePrefix.ROAD_SEGMENT}_{road_segment}")
                    for road_segment in jc_val.road_segments
                ]
            )

            junction_contains_lane_segment.extend(
                [
                    (junction_node_id, f"{NodePrefix.LANE_SEGMENT}_{lane_segment}")
                    for lane_segment in jc_val.lane_segments
                ]
            )

        road_segment_nodes = {
            n["name"]: n.index for n in self._map_graph.vs if n["name"].startswith(NodePrefix.ROAD_SEGMENT)
        }

        for j_source, rs_target in junction_contains_road_segment:
            j_source_index = junction_nodes[j_source]
            rs_target_index = road_segment_nodes[rs_target]
            self._map_graph.add_edge(
                source=j_source_index,
                target=rs_target_index,
                type=f"{NodePrefix.JUNCTION}_contains_{NodePrefix.ROAD_SEGMENT}",
            )

        lane_segment_nodes = {
            n["name"]: n.index for n in self._map_graph.vs if n["name"].startswith(NodePrefix.LANE_SEGMENT)
        }

        for j_source, ls_target in junction_contains_lane_segment:
            j_source_index = junction_nodes[j_source]
            ls_target_index = lane_segment_nodes[ls_target]
            self._map_graph.add_edge(
                source=j_source_index,
                target=ls_target_index,
                type=f"{NodePrefix.JUNCTION}_contains_{NodePrefix.LANE_SEGMENT}",
            )

    def _decode_areas(self):
        for a_key, a_val in self._umd_map.areas.items():
            edge = self._umd_map.edges[a_val.edges[0]]
            edge_points = np.array([(p.x, p.y) for p in edge.points])

            if len(edge_points) > 0:
                x_min, y_min = np.min(edge_points, axis=0)
                x_max, y_max = np.max(edge_points, axis=0)
            else:
                x_min = x_max = y_min = y_max = None

            area_node_id = f"{NodePrefix.AREA}_{a_key}"

            _ = self._map_graph.add_vertex(
                name=area_node_id,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                x_center=x_min + (x_max - x_min) / 2 if x_min is not None else None,
                y_center=y_min + (y_max - y_min) / 2 if y_min is not None else None,
            )

    def get_junction(self, id: int) -> Optional[Junction]:
        query_results = self._map_graph.vs.select(name_eq=f"{NodePrefix.JUNCTION}_{id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    def get_road_segment(self, id: int) -> Optional[RoadSegment]:
        query_results = self._map_graph.vs.select(name_eq=f"{NodePrefix.ROAD_SEGMENT}_{id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    def get_lane_segment(self, id: int) -> Optional[LaneSegment]:
        query_results = self._map_graph.vs.select(name_eq=f"{NodePrefix.LANE_SEGMENT}_{id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    def get_lane_segments(self, ids: List[int]) -> List[Optional[LaneSegment]]:
        return [self.get_lane_segment(id=id) for id in ids]

    def get_area(self, id: int) -> Area:
        query_results = self._map_graph.vs.select(name_eq=f"{NodePrefix.AREA}_{id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    def _get_nodes_within_bounds(
        self,
        node_prefix: str,
        x_min: float = -math.inf,
        x_max: float = math.inf,
        y_min: float = -math.inf,
        y_max: float = math.inf,
        method: str = "inside",
    ) -> List:
        if method == "inside":
            return [
                vv["object"]
                for vv in self._map_graph.vs.select(
                    lambda v: v["name"].startswith(node_prefix)
                    and v["x_max"] <= x_max
                    and v["y_max"] <= y_max
                    and v["x_min"] >= x_min
                    and v["y_min"] >= y_min
                )
            ]
        elif method == "overlap":
            return [
                vv["object"]
                for vv in self._map_graph.vs.select(
                    lambda v: v["name"].startswith(node_prefix)
                    and (min(v["x_max"], x_max) - max(v["x_min"], x_min)) >= 0
                    and (min(v["y_max"], y_max) - max(v["y_min"], y_min)) >= 0
                )
            ]

        elif method == "center":
            return [
                vv["object"]
                for vv in self._map_graph.vs.select(
                    lambda v: v["name"].startswith(node_prefix)
                    and v["x_center"] <= x_max
                    and v["y_center"] <= y_max
                    and v["x_center"] >= x_min
                    and v["y_center"] >= y_min
                )
            ]

    def get_road_segments_within_bounds(
        self,
        x_min: float = -math.inf,
        x_max: float = math.inf,
        y_min: float = -math.inf,
        y_max: float = math.inf,
        method: str = "inside",
    ) -> List[LaneSegment]:
        return self._get_nodes_within_bounds(
            node_prefix=NodePrefix.ROAD_SEGMENT, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, method=method
        )

    def get_lane_segments_within_bounds(
        self,
        x_min: float = -math.inf,
        x_max: float = math.inf,
        y_min: float = -math.inf,
        y_max: float = math.inf,
        method: str = "inside",
    ) -> List[LaneSegment]:
        return self._get_nodes_within_bounds(
            node_prefix=NodePrefix.LANE_SEGMENT, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, method=method
        )

    def get_areas_within_bounds(
        self,
        x_min: float = -math.inf,
        x_max: float = math.inf,
        y_min: float = -math.inf,
        y_max: float = math.inf,
        method: str = "inside",
    ) -> List[int]:
        return self._get_nodes_within_bounds(
            node_prefix=NodePrefix.AREA, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, method=method
        )

    def get_junctions_for_lane_segment(self, id: int) -> List[Junction]:
        subgraph = self._get_junctions_containing_lane_segments_graph()
        source_node = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id}")

        junctions = subgraph.es.select(_target=source_node)

        return [j.source_vertex["object"] for j in junctions]

    def get_road_segment_for_lane_segment(self, id: int) -> Optional[RoadSegment]:
        subgraph = self._get_road_segments_containing_lane_segments_graph()
        target_node = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id}")

        road_segments = subgraph.es.select(_target=target_node)

        if len(road_segments) > 0:
            return road_segments[0].source_vertex["object"]
        else:
            return None

    def get_lane_segment_successors(
        self, id: int, depth: int = -1
    ) -> Tuple[List[LaneSegment], List[int], List[Optional[int]]]:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        start_node = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id}")

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

    def get_lane_segments_successors_shortest_paths(self, source_id: int, target_id: int) -> List[List[LaneSegment]]:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        start_node = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{source_id}")
        end_node = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{target_id}")

        shortest_paths = subgraph.get_shortest_paths(v=start_node, to=end_node.index, mode="out")
        return [[subgraph.vs[node_id]["object"] for node_id in shortest_path] for shortest_path in shortest_paths]

    def get_lane_segment_successors_random_path(self, id: int, steps: int = None):
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        start_vertex = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id}")
        random_walk = subgraph.random_walk(
            start=start_vertex.index,
            steps=steps if steps is not None else (2 ** 16),
            mode="out",
            stuck="return",
        )
        return [subgraph.vs[node_id]["object"] for node_id in random_walk]

    def get_lane_segments_connceted_shortest_paths(self, source_id: int, target_id: int) -> List[List[LaneSegment]]:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        start_node = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{source_id}")
        end_node = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{target_id}")

        shortest_paths = subgraph.get_shortest_paths(v=start_node, to=end_node.index, mode="all")
        return [[subgraph.vs[node_id]["object"] for node_id in shortest_path] for shortest_path in shortest_paths]

    def get_lane_segment_predecessors(
        self, id: int, depth: int = -1
    ) -> Tuple[List[LaneSegment], List[int], List[Optional[int]]]:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        start_node = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id}")

        lane_segments = []
        levels = []
        parent_ids = []

        bfs_iterator = subgraph.bfsiter(vid=start_node.index, mode="in", advanced=True)
        while True:
            try:
                lane_segment, level, parent_lane_segment = next(bfs_iterator)
                if level > depth:
                    break
                lane_segments.append(lane_segment["object"])
                levels.append(level)
                parent_ids.append(parent_lane_segment["object"].id if parent_lane_segment is not None else None)
            except StopIteration:
                break

        return lane_segments, levels, parent_ids

    def get_lane_segment_predecessors_random_path(self, id: int, steps: int = None):
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        start_vertex = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id}")
        random_walk = subgraph.random_walk(
            start=start_vertex.index,
            steps=steps if steps is not None else (2 ** 16),
            mode="in",
            stuck="return",
        )
        return [subgraph.vs[node_id]["object"] for node_id in random_walk]

    def get_lane_segment_neighbors(self, id: int) -> (Optional[LaneSegment], Optional[LaneSegment]):
        lane_segment = self.get_lane_segment(id=id)
        neighbor_left = self.get_lane_segment(id=lane_segment.left_neighbor)
        neighbor_right = self.get_lane_segment(id=lane_segment.right_neighbor)
        return (
            (
                neighbor_left,
                neighbor_right,
            )
            if lane_segment is not None
            else (None, None)
        )

    def get_lane_segments_neighbors(self, ids: List[int]) -> (List[Optional[LaneSegment]], List[Optional[LaneSegment]]):
        lane_segments = [self.get_lane_segment(id=id) for id in ids]
        assert all(self.are_succeeding_lane_segments(id_1=p[0].id, id_2=p[1].id) for p in pairwise(lane_segments))

    def bridge_lane_segments(
        self, id_1: int, id_2: int, bridge_length: int = None, directed: bool = True
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
                shortest_paths = self.get_lane_segments_connceted_shortest_paths(source_id=id_1, target_id=id_2)
                if shortest_paths and (
                    bridge_length is None or len(shortest_paths[0]) - 2 == bridge_length
                ):  # shortest path exists with max gap length
                    return shortest_paths[0][1:-1]  # return only bridge elements
                else:
                    return None
            else:  # LS are directly connected, bridge is empty.
                return []

    def complete_lane_segments(self, ids: List[Optional[int]], directed: bool = True) -> List[Optional[int]]:
        if directed:
            groups = self.group_succeeding_lane_segments(ids=ids)
        else:
            groups = self.group_connected_lane_segments(ids=ids)

        # Minimum groups required, because we need to have [..,LS],[None,...],[LS,...] to completed gaps
        if len(groups) >= 3:
            for i, (start_group, bridge_group, end_group) in enumerate(triplewise(groups)):
                start_element = start_group[-1]
                end_element = end_group[0]
                bridge_candidates = self.bridge_lane_segments(
                    id_1=start_element.id, id_2=end_element.id, bridge_length=len(bridge_group), directed=directed
                )
                if bridge_candidates is not None:
                    groups[i + 1] = bridge_candidates

        return list(chain.from_iterable(groups))

    def group_succeeding_lane_segments(self, ids: List[int]) -> List[List[Optional[LaneSegment]]]:
        def split_fn(x: Optional[LaneSegment], y: Optional[LaneSegment]) -> bool:
            if x is None and y is not None:
                return True
            elif x is not None and y is None:
                return True
            elif x is None and y is None:
                return False
            elif x is not None and y is not None:
                return not self.are_succeeding_lane_segments(id_1=x.id, id_2=y.id)

        lane_segments = self.get_lane_segments(ids=ids)
        return list(split_when(lane_segments, split_fn))

    def group_connected_lane_segments(self, ids: List[int]) -> List[List[Optional[LaneSegment]]]:
        def split_fn(x: Optional[LaneSegment], y: Optional[LaneSegment]) -> bool:
            if x is None and y is not None:
                return True
            elif x is not None and y is None:
                return True
            elif x is None and y is None:
                return False
            elif x is not None and y is not None:
                return not self.are_connected_lane_segments(id_1=x.id, id_2=y.id)

        lane_segments = self.get_lane_segments(ids=ids)
        return list(split_when(lane_segments, split_fn))

    def get_lane_segment_predecessors_successors(
        self, id: int, depth: int = -1
    ) -> Tuple[List[LaneSegment], List[int], List[Optional[int]]]:
        lane_segments_pred, levels_pred, parent_ids_pred = self.get_lane_segment_predecessors(id=id, depth=depth)
        lane_segments_succ, levels_succ, parent_ids_succ = self.get_lane_segment_successors(id=id, depth=depth)

        return (
            (lane_segments_pred + lane_segments_succ),
            (levels_pred + levels_succ),
            (parent_ids_pred + parent_ids_succ),
        )

    def has_lane_segment_split(self, id: int) -> bool:
        return len(self.get_lane_segment_successors(id=id, depth=1)) > 2  # [start_node, one_successor, ...]

    def has_lane_segment_merge(self, id: int) -> bool:
        return len(self.get_lane_segment_predecessors(id=id, depth=1)) > 2  # [start_node, one_predecessor, ...]

    def are_connected_lane_segments(self, id_1: int, id_2: int) -> bool:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        node_1 = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id_1}")
        node_2 = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id_2}")
        edge_id = subgraph.get_eid(node_1, node_2, directed=False, error=False)
        return False if edge_id == -1 else True

    def are_opposite_direction_lane_segments(self, id_1: int, id_2: int) -> Optional[bool]:
        lane_segment_1 = self.get_lane_segment(id=id_1)
        lane_segment_2 = self.get_lane_segment(id=id_2)

        road_segment_1 = self.get_road_segment_for_lane_segment(id=id_1)
        road_segment_2 = self.get_road_segment_for_lane_segment(id=id_2)

        if road_segment_1 == road_segment_2:
            return True if lane_segment_1.direction != lane_segment_2.direction else False
        else:
            lane_segment_1_connected = self.get_lane_segment_predecessors_successors(id=id_1, depth=1)[0]
            lane_segment_2_connected = self.get_lane_segment_predecessors_successors(id=id_2, depth=1)[0]

            lane_segment_1_connected_road_segments = [
                self.get_road_segment_for_lane_segment(id=ls.id).id for ls in lane_segment_1_connected
            ]

            lane_segment_2_connected_road_segments = [
                self.get_road_segment_for_lane_segment(id=ls.id).id for ls in lane_segment_2_connected
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

    def are_succeeding_lane_segments(self, id_1: int, id_2: int) -> bool:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        node_1 = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id_1}")
        node_2 = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id_2}")
        edge_id = subgraph.get_eid(node_1, node_2, directed=True, error=False)
        return False if edge_id == -1 else True

    def are_preceeding_lane_segments(self, id_1: int, id_2: int) -> bool:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        node_1 = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id_1}")
        node_2 = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id_2}")
        edge_id = subgraph.get_eid(node_2, node_1, directed=True, error=False)
        return False if edge_id == -1 else True

    def is_lane_segment_inside_junction(self, id: int) -> bool:
        return True if len(self.get_junctions_for_lane_segment(id=id)) > 0 else False

    def get_lane_segments_for_point(self, point: PointENU) -> List[LaneSegment]:
        ls_candidates = self.get_lane_segments_within_bounds(
            x_min=point.x - 0.1, x_max=point.x + 0.1, y_min=point.y - 0.1, y_max=point.y + 0.1, method="overlap"
        )
        if ls_candidates:
            point_under_test = (point.x, point.y)
            return [
                ls_candidates[i]
                for i, polygon in enumerate(ls_candidates)
                if is_point_in_polygon_2d(polygon=polygon.to_numpy(closed=True)[:, :2], point=point_under_test)
            ]
        else:
            return []

    @classmethod
    def from_path(cls, path: AnyPath):
        umd = read_binary_message(obj=ProtoUniversalMap(), path=path)
        return cls(umd_map=umd)
