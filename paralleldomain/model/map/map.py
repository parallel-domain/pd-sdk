import math
from itertools import chain, groupby
from typing import List, Optional, Tuple

from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DGeometry
from paralleldomain.model.type_aliases import JunctionId, LaneSegmentId, RoadSegmentId

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

import numpy as np
from igraph import Graph, Vertex
from more_itertools import split_when, triplewise, unique_everseen

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
from paralleldomain.utilities.transformation import Transformation


class Neighbor:
    LEFT: str = "left"
    RIGHT: str = "right"


class MapDecoderProtocol(Protocol):
    def get_road_segments(self) -> List[RoadSegment]:
        pass

    def get_lane_segments(self) -> List[LaneSegment]:
        pass

    def get_junctions(self) -> List[Junction]:
        pass

    def get_areas(self) -> List[Area]:
        pass


class Map2:
    def __init__(self, map_decoder: MapDecoderProtocol):
        self._map_graph = Graph(directed=True)
        self._map_decoder = map_decoder
        self._added_road_segments = False
        self._added_lane_segments = False
        self._added_junctions = False
        self._added_areas = False
        # self._build_graph()

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

    # to decoder
    def get_junction(self, junction_id: JunctionId) -> Optional[Junction]:
        query_results = self._map_graph.vs.select(name_eq=f"{NodePrefix.JUNCTION}_{junction_id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    def get_road_segment(self, road_segment_id: RoadSegmentId) -> Optional[RoadSegment]:
        query_results = self._map_graph.vs.select(name_eq=f"{NodePrefix.ROAD_SEGMENT}_{road_segment_id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    def get_lane_segment(self, lane_segment_id: LaneSegmentId) -> Optional[LaneSegment]:
        query_results = self._map_graph.vs.select(name_eq=f"{NodePrefix.LANE_SEGMENT}_{lane_segment_id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    def get_lane_segments(self, lane_segment_ids: List[LaneSegmentId]) -> List[Optional[LaneSegment]]:
        return [self.get_lane_segment(lane_segment_id=lid) for lid in lane_segment_ids]

    def get_area(self, id: int) -> Area:
        query_results = self._map_graph.vs.select(name_eq=f"{NodePrefix.AREA}_{id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    # def _add_vertex(self, has_bounds: _MayBeHasBounds, vertex_id: str) -> Vertex:
    #     if has_bounds.bounds is not None:
    #         x_min = has_bounds.bounds.x
    #         x_max = has_bounds.bounds.x + has_bounds.bounds.width
    #         y_min = has_bounds.bounds.y
    #         y_max = has_bounds.bounds.y + has_bounds.bounds.height
    #     else:
    #         x_min = x_max = y_min = y_max = None
    #
    #     vertex = self._map_graph.add_vertex(
    #         name=vertex_id,
    #         x_min=x_min,
    #         x_max=x_max,
    #         y_min=y_min,
    #         y_max=y_max,
    #         x_center=x_min + (x_max - x_min) / 2 if x_min is not None else None,
    #         y_center=y_min + (y_max - y_min) / 2 if y_min is not None else None,
    #         object=has_bounds,
    #     )
    #     return vertex

    # def _build_graph(self):
    #     self._add_road_segments_to_graph()
    #     self._add_lane_segments_to_graph()
    #     self._add_junctions_to_graph()
    #     self._add_areas_to_graph()
    #
    # def _add_road_segments_to_graph(self):
    #     if self._added_road_segments:
    #         return
    #
    #     road_segment_nodes = {}
    #     road_segment_edges_x_precedes_y = []  # RoadSegment -> RoadSegment
    #     # for rs_key, rs_val in self._umd_map.road_segments.items():
    #     for road_segment in self._map_decoder.decode_road_segments():
    #         road_segment_node_id = f"{NodePrefix.ROAD_SEGMENT}_{road_segment.road_segment_id}"
    #         road_segment_node = self._add_vertex(has_bounds=road_segment, vertex_id=road_segment_node_id)
    #
    #         road_segment_nodes[road_segment_node_id] = road_segment_node.index
    #
    #         road_segment_edges_x_precedes_y.extend(
    #             [
    #                 (road_segment_node_id, f"{NodePrefix.ROAD_SEGMENT}_{successor}")
    #                 for successor in road_segment.successors
    #                 if successor != 0
    #             ]
    #         )
    #
    #     for rs_source, rs_target in road_segment_edges_x_precedes_y:
    #         rs_source_index = road_segment_nodes[rs_source]
    #         rs_target_index = road_segment_nodes[rs_target]
    #         self._map_graph.add_edge(
    #             source=rs_source_index,
    #             target=rs_target_index,
    #             type=f"{NodePrefix.ROAD_SEGMENT}_preceeds_{NodePrefix.ROAD_SEGMENT}",
    #         )
    #     self._added_road_segments = True
    #
    # def _add_lane_segments_to_graph(self):
    #     if self._added_lane_segments:
    #         return
    #
    #     lane_segment_nodes = {}
    #     lane_segment_edges_x_precedes_y = []  # LaneSegment -> LaneSegment
    #     lane_segment_edges_x_contains_y = []  # RoadSegment -> LaneSegment
    #     lane_segment_edges_x_to_the_left_of_y = []  # LaneSegment -> LaneSegment
    #
    #     for lane_segment in self._map_decoder.decode_lane_segments():
    #         lane_segment_node_id = f"{NodePrefix.LANE_SEGMENT}_{lane_segment.lane_segment_id}"
    #         lane_segment_node = self._add_vertex(has_bounds=lane_segment, vertex_id=lane_segment_node_id)
    #
    #         lane_segment_nodes[lane_segment_node_id] = lane_segment_node.index
    #
    #         lane_segment_edges_x_precedes_y.extend(
    #             [
    #                 (lane_segment_node_id, f"{NodePrefix.LANE_SEGMENT}_{successor}")
    #                 for successor in lane_segment.successors
    #                 if successor != 0
    #             ]
    #         )
    #         if lane_segment.right_neighbor != 0:
    #             lane_segment_edges_x_to_the_left_of_y.append(
    #                 (lane_segment_node_id, f"{NodePrefix.LANE_SEGMENT}_{lane_segment.right_neighbor}")
    #             )
    #         if lane_segment.parent_road_segment_id is not None and lane_segment.parent_road_segment_id != 0:
    #             lane_segment_edges_x_contains_y.append(
    #                 (f"{NodePrefix.ROAD_SEGMENT}_{lane_segment.parent_road_segment_id}", lane_segment_node_id)
    #             )
    #
    #     for ls_source, ls_target in lane_segment_edges_x_precedes_y:
    #         ls_source_index = lane_segment_nodes[ls_source]
    #         ls_target_index = lane_segment_nodes[ls_target]
    #         self._map_graph.add_edge(
    #             source=ls_source_index,
    #             target=ls_target_index,
    #             type=f"{NodePrefix.LANE_SEGMENT}_preceeds_{NodePrefix.LANE_SEGMENT}",
    #         )
    #
    #     for ls_source, ls_target in lane_segment_edges_x_to_the_left_of_y:
    #         ls_source_index = lane_segment_nodes[ls_source]
    #         ls_target_index = lane_segment_nodes[ls_target]
    #         self._map_graph.add_edge(
    #             source=ls_source_index,
    #             target=ls_target_index,
    #             type=f"{NodePrefix.LANE_SEGMENT}_left_of_{NodePrefix.LANE_SEGMENT}",
    #         )
    #
    #     road_segment_nodes = {
    #         n["name"]: n.index for n in self._map_graph.vs if n["name"].startswith(NodePrefix.ROAD_SEGMENT)
    #     }
    #
    #     for rs_source, ls_target in lane_segment_edges_x_contains_y:
    #         rs_source_index = road_segment_nodes[rs_source]
    #         ls_target_index = lane_segment_nodes[ls_target]
    #         self._map_graph.add_edge(
    #             source=rs_source_index,
    #             target=ls_target_index,
    #             type=f"{NodePrefix.ROAD_SEGMENT}_contains_{NodePrefix.LANE_SEGMENT}",
    #         )
    #     self._added_lane_segments = True
    #
    # def _add_junctions_to_graph(self):
    #     if self._added_junctions:
    #         return
    #
    #     junction_nodes = {}
    #     junction_contains_road_segment = []  # Junction -> RoadSegment
    #     junction_contains_lane_segment = []  # Junction -> LaneSegment
    #
    #     for junction in self._map_decoder.decode_junctions():
    #         corner_points = np.empty(shape=(0, 2))
    #         for corner in junction.corners:
    #             corner_edge = self._umd_map.edges[corner]
    #             corner_points = np.vstack(
    #                 [
    #                     corner_points,
    #                     np.array([(p.x, p.y) for p in corner_edge.points]),
    #                 ]
    #             )
    #
    #         junction_node_id = f"{NodePrefix.JUNCTION}_{junction.junction_id}"
    #         junction_node = self._add_vertex(has_bounds=junction, vertex_id=junction_node_id)
    #
    #         junction_nodes[junction_node_id] = junction_node.index
    #
    #         junction_contains_road_segment.extend(
    #             [
    #                 (junction_node_id, f"{NodePrefix.ROAD_SEGMENT}_{road_segment}")
    #                 for road_segment in junction.road_segments
    #             ]
    #         )
    #
    #         junction_contains_lane_segment.extend(
    #             [
    #                 (junction_node_id, f"{NodePrefix.LANE_SEGMENT}_{lane_segment}")
    #                 for lane_segment in junction.lane_segments
    #             ]
    #         )
    #
    #     road_segment_nodes = {
    #         n["name"]: n.index for n in self._map_graph.vs if n["name"].startswith(NodePrefix.ROAD_SEGMENT)
    #     }
    #
    #     for j_source, rs_target in junction_contains_road_segment:
    #         j_source_index = junction_nodes[j_source]
    #         rs_target_index = road_segment_nodes[rs_target]
    #         self._map_graph.add_edge(
    #             source=j_source_index,
    #             target=rs_target_index,
    #             type=f"{NodePrefix.JUNCTION}_contains_{NodePrefix.ROAD_SEGMENT}",
    #         )
    #
    #     lane_segment_nodes = {
    #         n["name"]: n.index for n in self._map_graph.vs if n["name"].startswith(NodePrefix.LANE_SEGMENT)
    #     }
    #
    #     for j_source, ls_target in junction_contains_lane_segment:
    #         j_source_index = junction_nodes[j_source]
    #         ls_target_index = lane_segment_nodes[ls_target]
    #         self._map_graph.add_edge(
    #             source=j_source_index,
    #             target=ls_target_index,
    #             type=f"{NodePrefix.JUNCTION}_contains_{NodePrefix.LANE_SEGMENT}",
    #         )
    #     self._added_junctions = True
    #
    # def _add_areas_to_graph(self):
    #     if self._added_areas:
    #         return
    #
    #     for area in self._map_decoder.decode_areas():
    #         area_node_id = f"{NodePrefix.AREA}_{area.area_id}"
    #         self._add_vertex(has_bounds=area, vertex_id=area_node_id)
    #
    #     self._added_areas = True

    # ----------to decoder
    # keep
    def get_lane_segments_from_poses(self, poses: List[Transformation]) -> List[LaneSegment]:
        enu_points = [PointENU.from_transformation(tf=pose) for pose in poses]
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
                    source_id=prev_node.id, target_id=next_node.id
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

    # ----


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

    def get_lane_segments_from_poses(self, poses: List[Transformation]) -> List[LaneSegment]:
        enu_points = [PointENU.from_transformation(tf=pose) for pose in poses]
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
                    source_id=prev_node.id, target_id=next_node.id
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

    def get_area(self, id: int) -> Area:
        query_results = self._map_graph.vs.select(name_eq=f"{NodePrefix.AREA}_{id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    def pad_lane_segments(self, lane_segments: List[LaneSegment], padding: int = 1) -> List[LaneSegment]:
        lane_segments_predecessors = self.get_lane_segment_predecessors_random_path(
            id=lane_segments[0].id, steps=padding
        )
        lane_segments_successors = self.get_lane_segment_successors_random_path(id=lane_segments[-1].id, steps=padding)

        return lane_segments_predecessors[::-1][:-1] + lane_segments + lane_segments_successors[1:]

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

    def get_lane_segments_connected_shortest_paths(self, source_id: int, target_id: int) -> List[List[LaneSegment]]:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        start_node = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{source_id}")
        end_node = subgraph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{target_id}")

        shortest_paths = subgraph.get_shortest_paths(v=start_node, to=end_node, mode="all")
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

    def get_lane_segment_relative_neighbor(self, id: int, side: str, degree: int = 1) -> LaneSegment:
        lane_segment = self.get_lane_segment(id=id)

        neighbor = lane_segment  # set to initial lane segment and start the loop if degree >= 1
        for i in range(degree):
            if side == Neighbor.LEFT:
                neighbor = (
                    self.get_lane_segment(id=neighbor.left_neighbor)
                    if not self.are_opposite_direction_lane_segments(id_1=lane_segment.id, id_2=neighbor.id)
                    else self.get_lane_segment(id=neighbor.right_neighbor)
                )
            else:  # Neighbor.RIGHT
                neighbor = (
                    self.get_lane_segment(id=neighbor.right_neighbor)
                    if not self.are_opposite_direction_lane_segments(id_1=lane_segment.id, id_2=neighbor.id)
                    else self.get_lane_segment(id=neighbor.left_neighbor)
                )
            if neighbor is None:
                break

        return neighbor

    def get_lane_segments_relative_neighbors(
        self, ids: List[int], side: str, degree: int = 1, bridge: bool = False
    ) -> List[Optional[LaneSegment]]:
        lane_segments = [self.get_lane_segment(id=id) for id in ids]
        neighbors = [
            self.get_lane_segment_relative_neighbor(id=ls.id, side=side, degree=degree) for ls in lane_segments
        ]

        if bridge:
            return self.complete_lane_segments(ids=[n.id if n is not None else None for n in neighbors], directed=False)
        else:
            return neighbors

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
                shortest_paths = self.get_lane_segments_connected_shortest_paths(source_id=id_1, target_id=id_2)
                if shortest_paths and (
                    bridge_length is None or len(shortest_paths[0]) - 2 == bridge_length
                ):  # shortest path exists with max gap length
                    return shortest_paths[0][1:-1]  # return only bridge elements
                else:
                    return None
            else:  # LS are directly connected, bridge is empty.
                return []

    def complete_lane_segments(self, ids: List[Optional[int]], directed: bool = True) -> List[Optional[LaneSegment]]:
        if directed:
            groups = self.group_succeeding_lane_segments(ids=ids)
        else:
            groups = self.group_connected_lane_segments(ids=ids)

        # Minimum groups required, because we need to have [..,LS],[None,...],[LS,...] to completed gaps
        if len(groups) >= 3:
            for i, (start_group, bridge_group, end_group) in enumerate(triplewise(groups)):
                start_element = start_group[-1]
                end_element = end_group[0]
                if start_element is not None and end_element is not None:
                    bridge_candidates = self.bridge_lane_segments(
                        id_1=start_element.id, id_2=end_element.id, bridge_length=len(bridge_group), directed=directed
                    )
                    if bridge_candidates is not None:
                        groups[i + 1] = bridge_candidates

        return [self.get_lane_segment(ls.id) if ls is not None else None for ls in chain.from_iterable(groups)]

    def group_succeeding_lane_segments(
        self, ids: List[int], reorder: bool = False
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
                    return not self.are_succeeding_lane_segments(id_1=x.id, id_2=y.id)

            lane_segments = self.get_lane_segments(ids=ids)
            return list(split_when(lane_segments, split_fn))
        else:
            vertices = [self._map_graph.vs.find(f"{NodePrefix.LANE_SEGMENT}_{id}") for id in ids]
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
