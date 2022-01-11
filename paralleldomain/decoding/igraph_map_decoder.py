import abc

import numpy as np

from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DGeometry

try:
    from typing import Optional, Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

from igraph import Graph, Vertex

from paralleldomain.decoding.map_decoder import MapDecoder


class NodePrefix:
    ROAD_SEGMENT: str = "RS"
    LANE_SEGMENT: str = "LS"
    JUNCTION: str = "JC"
    AREA: str = "AR"


class _MayBeHasBounds(Protocol):
    @property
    def bounds(self) -> Optional[BoundingBox2DGeometry]:
        return


class IGraphMapDecoder(MapDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.__map_graph = Graph(directed=True)
        self._added_road_segments = False
        self._added_lane_segments = False
        self._added_junctions = False
        self._added_areas = False

    @property
    def map_graph(self) -> Graph:
        self._build_graph()
        return self.__map_graph

    def _build_graph(self):
        self._add_road_segments_to_graph()
        self._add_lane_segments_to_graph()
        self._add_junctions_to_graph()
        self._add_areas_to_graph()

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

    def _add_road_segments_to_graph(self):
        if self._added_road_segments:
            return

        road_segment_nodes = {}
        road_segment_edges_x_precedes_y = []  # RoadSegment -> RoadSegment
        # for rs_key, rs_val in self._umd_map.road_segments.items():
        for road_segment in self.get_road_segments().values():
            road_segment_node_id = f"{NodePrefix.ROAD_SEGMENT}_{road_segment.road_segment_id}"
            road_segment_node = self._add_vertex(has_bounds=road_segment, vertex_id=road_segment_node_id)

            road_segment_nodes[road_segment_node_id] = road_segment_node.index

            road_segment_edges_x_precedes_y.extend(
                [
                    (road_segment_node_id, f"{NodePrefix.ROAD_SEGMENT}_{successor}")
                    for successor in road_segment.successors
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

    def _add_lane_segments_to_graph(self):
        if self._added_lane_segments:
            return

        lane_segment_nodes = {}
        lane_segment_edges_x_precedes_y = []  # LaneSegment -> LaneSegment
        lane_segment_edges_x_contains_y = []  # RoadSegment -> LaneSegment
        lane_segment_edges_x_to_the_left_of_y = []  # LaneSegment -> LaneSegment

        for lane_segment in self.get_lane_segments().values():
            lane_segment_node_id = f"{NodePrefix.LANE_SEGMENT}_{lane_segment.lane_segment_id}"
            lane_segment_node = self._add_vertex(has_bounds=lane_segment, vertex_id=lane_segment_node_id)

            lane_segment_nodes[lane_segment_node_id] = lane_segment_node.index

            lane_segment_edges_x_precedes_y.extend(
                [
                    (lane_segment_node_id, f"{NodePrefix.LANE_SEGMENT}_{successor}")
                    for successor in lane_segment.successors
                    if successor != 0
                ]
            )
            if lane_segment.right_neighbor != 0:
                lane_segment_edges_x_to_the_left_of_y.append(
                    (lane_segment_node_id, f"{NodePrefix.LANE_SEGMENT}_{lane_segment.right_neighbor}")
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

    def _add_junctions_to_graph(self):
        if self._added_junctions:
            return

        junction_nodes = {}
        junction_contains_road_segment = []  # Junction -> RoadSegment
        junction_contains_lane_segment = []  # Junction -> LaneSegment

        # edges = self.get_edges()

        for junction in self.get_junctions().values():
            corner_points = np.empty(shape=(0, 2))
            for corner in junction.corners:
                corner_edge = self._umd_map.edges[corner]
                corner_points = np.vstack(
                    [
                        corner_points,
                        np.array([(p.x, p.y) for p in corner_edge.points]),
                    ]
                )

            junction_node_id = f"{NodePrefix.JUNCTION}_{junction.junction_id}"
            junction_node = self._add_vertex(has_bounds=junction, vertex_id=junction_node_id)

            junction_nodes[junction_node_id] = junction_node.index

            junction_contains_road_segment.extend(
                [
                    (junction_node_id, f"{NodePrefix.ROAD_SEGMENT}_{road_segment}")
                    for road_segment in junction.road_segments
                ]
            )

            junction_contains_lane_segment.extend(
                [
                    (junction_node_id, f"{NodePrefix.LANE_SEGMENT}_{lane_segment}")
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

    def _add_areas_to_graph(self):
        if self._added_areas:
            return

        for area in self.get_areas():
            area_node_id = f"{NodePrefix.AREA}_{area.area_id}"
            self._add_vertex(has_bounds=area, vertex_id=area_node_id)

        self._added_areas = True
