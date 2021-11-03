from collections import namedtuple

import numpy as np
from igraph import Graph

from paralleldomain.common.umd.v1.UMD_pb2 import UniversalMap
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_binary_message
from paralleldomain.utilities.geometry import is_point_in_polygon_2d


class NODE_PREFIX:
    ROAD_SEGMENT: str = "RS"
    LANE_SEGMENT: str = "LS"
    JUNCTION: str = "JC"
    AREA: str = "AR"


class Map:
    def __init__(self, umd_map: UniversalMap) -> None:
        self._umd_map = umd_map

        self._map_graph = Graph(directed=True)

        self._decode_road_segment()
        self._decode_lane_segments()
        self._decode_junctions()
        self._decode_areas()

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

            road_segment_node_id = f"{NODE_PREFIX.ROAD_SEGMENT}_{rs_key}"

            road_segment_node = self._map_graph.add_vertex(
                name=road_segment_node_id,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                x_center=x_min + (x_max - x_min) / 2 if x_min is not None else None,
                y_center=y_min + (y_max - y_min) / 2 if y_min is not None else None,
            )

            road_segment_nodes[road_segment_node_id] = road_segment_node.index

            road_segment_edges_x_precedes_y.extend(
                [(road_segment_node_id, f"{NODE_PREFIX.ROAD_SEGMENT}_{successor}") for successor in rs_val.successors]
            )

        for rs_source, rs_target in road_segment_edges_x_precedes_y:
            rs_source_index = road_segment_nodes[rs_source]
            rs_target_index = road_segment_nodes[rs_target]
            self._map_graph.add_edge(
                source=rs_source_index,
                target=rs_target_index,
                type=f"{NODE_PREFIX.ROAD_SEGMENT}_preceeds_{NODE_PREFIX.ROAD_SEGMENT}",
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

            lane_segment_node_id = f"{NODE_PREFIX.LANE_SEGMENT}_{ls_key}"

            lane_segment_node = self._map_graph.add_vertex(
                name=lane_segment_node_id,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                x_center=x_min + (x_max - x_min) / 2 if x_min is not None else None,
                y_center=y_min + (y_max - y_min) / 2 if y_min is not None else None,
            )

            lane_segment_nodes[lane_segment_node_id] = lane_segment_node.index

            lane_segment_edges_x_precedes_y.extend(
                [
                    (lane_segment_node_id, f"{NODE_PREFIX.LANE_SEGMENT}_{successor}")
                    for successor in ls_val.successors
                    if successor != 0
                ]
            )
            if ls_val.right_neighbor != 0:
                lane_segment_edges_x_to_the_left_of_y.append(
                    (lane_segment_node_id, f"{NODE_PREFIX.LANE_SEGMENT}_{ls_val.right_neighbor}")
                )
            if ls_val.road != 0:
                lane_segment_edges_x_contains_y.append(
                    (f"{NODE_PREFIX.ROAD_SEGMENT}_{ls_val.road}", lane_segment_node_id)
                )

        for ls_source, ls_target in lane_segment_edges_x_precedes_y:
            ls_source_index = lane_segment_nodes[ls_source]
            ls_target_index = lane_segment_nodes[ls_target]
            self._map_graph.add_edge(
                source=ls_source_index,
                target=ls_target_index,
                type=f"{NODE_PREFIX.LANE_SEGMENT}_preceeds_{NODE_PREFIX.LANE_SEGMENT}",
            )

        for ls_source, ls_target in lane_segment_edges_x_to_the_left_of_y:
            ls_source_index = lane_segment_nodes[ls_source]
            ls_target_index = lane_segment_nodes[ls_target]
            self._map_graph.add_edge(
                source=ls_source_index,
                target=ls_target_index,
                type=f"{NODE_PREFIX.LANE_SEGMENT}_left_of_{NODE_PREFIX.LANE_SEGMENT}",
            )

        road_segment_nodes = {
            n["name"]: n.index for n in self._map_graph.vs if n["name"].startswith(NODE_PREFIX.ROAD_SEGMENT)
        }

        for rs_source, ls_target in lane_segment_edges_x_contains_y:
            rs_source_index = road_segment_nodes[rs_source]
            ls_target_index = lane_segment_nodes[ls_target]
            self._map_graph.add_edge(
                source=rs_source_index,
                target=ls_target_index,
                type=f"{NODE_PREFIX.ROAD_SEGMENT}_contains_{NODE_PREFIX.LANE_SEGMENT}",
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

            junction_node_id = f"{NODE_PREFIX.JUNCTION}_{jc_key}"

            junction_node = self._map_graph.add_vertex(
                name=junction_node_id,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                x_center=x_min + (x_max - x_min) / 2 if x_min is not None else None,
                y_center=y_min + (y_max - y_min) / 2 if y_min is not None else None,
            )

            junction_nodes[junction_node_id] = junction_node.index

            junction_contains_road_segment.extend(
                [
                    (junction_node_id, f"{NODE_PREFIX.ROAD_SEGMENT}_{road_segment}")
                    for road_segment in jc_val.road_segments
                ]
            )

            junction_contains_lane_segment.extend(
                [
                    (junction_node_id, f"{NODE_PREFIX.LANE_SEGMENT}_{lane_segment}")
                    for lane_segment in jc_val.lane_segments
                ]
            )

        road_segment_nodes = {
            n["name"]: n.index for n in self._map_graph.vs if n["name"].startswith(NODE_PREFIX.ROAD_SEGMENT)
        }

        for j_source, rs_target in junction_contains_road_segment:
            j_source_index = junction_nodes[j_source]
            rs_target_index = road_segment_nodes[rs_target]
            self._map_graph.add_edge(
                source=j_source_index,
                target=rs_target_index,
                type=f"{NODE_PREFIX.JUNCTION}_contains_{NODE_PREFIX.ROAD_SEGMENT}",
            )

        lane_segment_nodes = {
            n["name"]: n.index for n in self._map_graph.vs if n["name"].startswith(NODE_PREFIX.LANE_SEGMENT)
        }

        for j_source, ls_target in junction_contains_lane_segment:
            j_source_index = junction_nodes[j_source]
            ls_target_index = lane_segment_nodes[ls_target]
            self._map_graph.add_edge(
                source=j_source_index,
                target=ls_target_index,
                type=f"{NODE_PREFIX.JUNCTION}_contains_{NODE_PREFIX.LANE_SEGMENT}",
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

            area_node_id = f"{NODE_PREFIX.AREA}_{a_key}"

            _ = self._map_graph.add_vertex(
                name=area_node_id,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                x_center=x_min + (x_max - x_min) / 2 if x_min is not None else None,
                y_center=y_min + (y_max - y_min) / 2 if y_min is not None else None,
            )

    """
    def get_junction(self, id: int) -> Junction:
        junction = self._umd_map.junctions[id]
        outer_corners = [self._umd_map.edges[x] for x in junction.corners]
        return Junction.from_proto(junction=junction, outer_corners=outer_corners)

    def get_road_segment(self, id: int) -> RoadSegment:
        road_segment = self._umd_map.road_segments[id]
        return RoadSegment.from_proto(road_segment=road_segment)

    def get_lane_segment(self, id: int) -> LaneSegment:
        lane_segment = self._umd_map.lane_segments[id]
        left_edge = self._umd_map.edges[lane_segment.left_edge]
        right_edge = self._umd_map.edges[lane_segment.right_edge]
        reference_line = self._umd_map.edges[lane_segment.reference_line]

        return LaneSegment.from_proto(
            lane_segment=lane_segment, left_edge=left_edge, right_edge=right_edge, reference_line=reference_line
        )

    def get_area(self, id: int) -> Area:
        area = self._umd_map.areas[id]
        edge = self._umd_map.edges[area.edges[0]]

        return Area.from_proto(area=area, edge=edge)

    def get_lane_segment_collection(self, ids: List[int]) -> LaneSegmentCollection:
        return LaneSegmentCollection.from_lane_segments(lane_segments=[self.get_lane_segment(id=id) for id in ids])

    def get_lane_segments_within_bounds(
        self,
        x_min: float = -math.inf,
        x_max: float = math.inf,
        y_min: float = -math.inf,
        y_max: float = math.inf,
        method: str = "inside",
    ) -> List[int]:
        if method == "inside":
            return list(
                self._lane_segments[
                    (self._lane_segments["x_max"] <= x_max)
                    & (self._lane_segments["y_max"] <= y_max)
                    & (self._lane_segments["x_min"] >= x_min)
                    & (self._lane_segments["y_min"] >= y_min)
                ].index
            )
        elif method == "overlap":

            def filter_fn(row):
                dx = min(row["x_max"], x_max) - max(row["x_min"], x_min)
                dy = min(row["y_max"], y_max) - max(row["y_min"], y_min)
                if (dx >= 0) and (dy >= 0):
                    return True
                else:
                    return False

            return list(self._lane_segments[self._lane_segments.apply(filter_fn, axis=1)].index)
        elif method == "center":
            return list(
                self._lane_segments[
                    (self._lane_segments["x_center"] <= x_max)
                    & (self._lane_segments["y_center"] <= y_max)
                    & (self._lane_segments["x_center"] >= x_min)
                    & (self._lane_segments["y_center"] >= y_min)
                ].index
            )

    def get_areas_within_bounds(
        self,
        x_min: float = -math.inf,
        x_max: float = math.inf,
        y_min: float = -math.inf,
        y_max: float = math.inf,
        method: str = "inside",
    ) -> List[int]:
        if method == "inside":
            return list(
                self._areas[
                    (self._areas["x_max"] <= x_max)
                    & (self._areas["y_max"] <= y_max)
                    & (self._areas["x_min"] >= x_min)
                    & (self._areas["y_min"] >= y_min)
                ].index
            )
        elif method == "overlap":

            def filter_fn(row):
                dx = min(row["x_max"], x_max) - max(row["x_min"], x_min)
                dy = min(row["y_max"], y_max) - max(row["y_min"], y_min)
                if (dx >= 0) and (dy >= 0):
                    return True
                else:
                    return False

            return list(self._areas[self._areas.apply(filter_fn, axis=1)].index)
        elif method == "center":
            return list(
                self._areas[
                    (self._areas["x_center"] <= x_max)
                    & (self._areas["y_center"] <= y_max)
                    & (self._areas["x_center"] >= x_min)
                    & (self._areas["y_center"] >= y_min)
                ].index
            )

    def get_junction_for_lane_segment(self, node_id: int) -> List[int]:
        junctions = filter(
            lambda x: x[2]["type"] == "contains",
            self._membership_graph.in_edges(f"{NODE_PREFIX.LANE_SEGMENT}_{node_id}", data=True),
        )

        return [int(j[0].lstrip(f"{NODE_PREFIX.JUNCTION}_")) for j in junctions]

    def get_parent_road_segment(self, node_id: int) -> int:
        road_segment_ownership = list(
            filter(
                lambda x: x[2]["type"] == "belongs_to",
                self._membership_graph.in_edges(f"{NODE_PREFIX.LANE_SEGMENT}_{node_id}", data=True),
            )
        )
        assert len(road_segment_ownership) == 1

        return int(road_segment_ownership[0][0].lstrip(f"{NODE_PREFIX.ROAD_SEGMENT}_"))

    def get_lane_segment_successors(self, node_id: int, depth: int = None):
        return [
            int(node.lstrip(f"{NODE_PREFIX.LANE_SEGMENT}_"))
            for node in nx.bfs_tree(
                self._successor_graph, f"{NODE_PREFIX.LANE_SEGMENT}_{node_id}", reverse=False, depth_limit=depth
            )
        ]

    def get_lane_segments_shortest_path(self, source_id: int, target_id: int):
        return [
            int(node.lstrip(f"{NODE_PREFIX.LANE_SEGMENT}_"))
            for node in nx.shortest_path(
                self._successor_graph,
                f"{NODE_PREFIX.LANE_SEGMENT}_{source_id}",
                f"{NODE_PREFIX.LANE_SEGMENT}_{target_id}",
            )
        ]

    def get_lane_segment_successors_single_path(self, node_id: int, depth: int = None, mode: str = "random"):
        if mode == "random":
            random_picked_path = [node_id]
            iteration = 0
            while depth is None or iteration < depth:
                iteration += 1
                candidates = set(self.get_lane_segment_successors(node_id=random_picked_path[-1], depth=1))
                candidates.remove(random_picked_path[-1])
                candidates = list(candidates)

                if candidates:
                    picked_candidate = candidates[0]
                    if picked_candidate in random_picked_path:
                        print("Circular path detected - abort.")
                        break
                    else:
                        random_picked_path.append(candidates[0])
                else:
                    break
            return random_picked_path
        elif mode == "longest":
            dfs_tree = list(
                nx.to_edgelist(
                    nx.bfs_tree(self._successor_graph, f"{NODE_PREFIX.LANE_SEGMENT}_{node_id}", depth_limit=depth)
                )
            )

            furthest_node = dfs_tree[-1][1]
            dfs_tree_dict = {edge[1]: edge[0] for edge in dfs_tree}

            longest_path = [furthest_node]
            while True:
                if furthest_node in dfs_tree_dict:
                    previous_node = dfs_tree_dict[furthest_node]
                    if previous_node in longest_path:
                        print("Circular path detected - abort.")
                        break
                    longest_path.append(previous_node)
                    furthest_node = previous_node
                else:
                    break

            return [int(node.lstrip(f"{NODE_PREFIX.LANE_SEGMENT}_")) for node in longest_path[::-1]]
        else:
            raise NotImplementedError(f"Currently, mode '{mode}' is not supported.")

    def get_lane_segment_predecessors(self, node_id: int, depth: int = None):
        return [
            int(node.lstrip(f"{NODE_PREFIX.LANE_SEGMENT}_"))
            for node in nx.bfs_tree(
                self._successor_graph, f"{NODE_PREFIX.LANE_SEGMENT}_{node_id}", reverse=True, depth_limit=depth
            )
        ]

    def get_lane_segment_predecessors_single_path(self, node_id: int, depth: int = None, mode: str = "random"):
        if mode == "random":
            random_picked_path = [node_id]
            iteration = 0
            while depth is None or iteration < depth:
                iteration += 1
                candidates = set(self.get_lane_segment_predecessors(node_id=random_picked_path[-1], depth=1))
                candidates.remove(random_picked_path[-1])
                candidates = list(candidates)

                if candidates:
                    picked_candidate = candidates[0]
                    if picked_candidate in random_picked_path:
                        print("Circular path detected - abort.")
                        break
                    else:
                        random_picked_path.append(candidates[0])
                else:
                    break
            return random_picked_path
        elif mode == "longest":
            dfs_tree = list(
                nx.to_edgelist(
                    nx.bfs_tree(
                        self._successor_graph, f"{NODE_PREFIX.LANE_SEGMENT}_{node_id}", reverse=True, depth_limit=depth
                    )
                )
            )

            furthest_node = dfs_tree[-1][1]
            dfs_tree_dict = {edge[1]: edge[0] for edge in dfs_tree}

            longest_path = [furthest_node]
            while True:
                if furthest_node in dfs_tree_dict:
                    previous_node = dfs_tree_dict[furthest_node]
                    if previous_node in longest_path:
                        print("Circular path detected - abort.")
                        break
                    longest_path.append(previous_node)
                    furthest_node = previous_node
                else:
                    break

            return [int(node.lstrip(f"{NODE_PREFIX.LANE_SEGMENT}_")) for node in longest_path[::-1]]
        else:
            raise NotImplementedError(f"Currently, mode '{mode}' is not supported.")

    def get_lane_segment_neighbors(self, id: int) -> (Optional[int], Optional[int]):
        lane_segment = self._umd_map.lane_segments[id]
        left_neighbor = lane_segment.left_neighbor
        right_neighbor = lane_segment.right_neighbor
        return (
            None if left_neighbor == 0 else left_neighbor,
            None if right_neighbor == 0 else right_neighbor,
        )

    def has_lane_segment_split(self, node_id: int):
        return len(self.get_lane_segment_successors(node_id=node_id, depth=1)) > 2  # [start_node, one_successor, ...]

    def has_lane_segment_merge(self, node_id: int):
        return (
            len(self.get_lane_segment_predecessors(node_id=node_id, depth=1)) > 2
        )  # [start_node, one_predecessor, ...]

    def get_lane_segment_for_point(self, x: float, y: float) -> List[int]:
        ls_candidates = self.get_lane_segments_within_bounds(
            x_min=x - 0.1, x_max=x + 0.1, y_min=y - 0.1, y_max=y + 0.1, method="overlap"
        )
        if ls_candidates:
            point_under_test = (x, y)
            polygons_under_test = [
                self.get_lane_segment(id=ls_id).to_2D().as_polygon(closed=True) for ls_id in ls_candidates
            ]
            return [
                ls_candidates[i]
                for i, polygon in enumerate(polygons_under_test)
                if is_point_in_polygon_2d(polygon=polygon, point=point_under_test)
            ]
        else:
            return []"""

    @classmethod
    def from_path(cls, path: AnyPath):
        umd = read_binary_message(obj=UniversalMap(), path=path)
        return cls(umd_map=umd)


class Road:
    ...


class RoadSegment:
    ...


class Lane:
    ...


class LaneSegment:
    ...


class Area:
    ...
