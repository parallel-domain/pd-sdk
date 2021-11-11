import math
from dataclasses import dataclass, field
from enum import IntEnum
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import ujson
from igraph import Graph
from more_itertools import windowed

from paralleldomain.common.umd.v1.UMD_pb2 import Edge as ProtoEdge
from paralleldomain.common.umd.v1.UMD_pb2 import Junction as ProtoJunction
from paralleldomain.common.umd.v1.UMD_pb2 import LaneSegment as ProtoLaneSegment
from paralleldomain.common.umd.v1.UMD_pb2 import Point_ENU as ProtoPointENU
from paralleldomain.common.umd.v1.UMD_pb2 import RoadMarking as ProtoRoadMarking
from paralleldomain.common.umd.v1.UMD_pb2 import RoadSegment as ProtoRoadSegment
from paralleldomain.common.umd.v1.UMD_pb2 import SpeedLimit as ProtoSpeedLimit
from paralleldomain.common.umd.v1.UMD_pb2 import UniversalMap as ProtoUniversalMap
from paralleldomain.model.geometry.point_3d import Point3DGeometry
from paralleldomain.model.geometry.polyline_3d import Line3DGeometry, Polyline3DGeometry
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_binary_message
from paralleldomain.utilities.geometry import is_point_in_polygon_2d
from paralleldomain.utilities.transformation import Transformation


class NODE_PREFIX:
    ROAD_SEGMENT: str = "RS"
    LANE_SEGMENT: str = "LS"
    JUNCTION: str = "JC"
    AREA: str = "AR"


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


class LaneType(IntEnum):
    UNDEFINED_LANE = 0
    DRIVABLE = 1
    NON_DRIVABLE = 2
    PARKING = 3
    SHOULDER = 4
    BIKING = 5
    CROSSWALK = 6
    RESTRICTED = 7


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


class SpeedUnits(IntEnum):
    MILES_PER_HOUR = 0
    KILOMETERS_PER_HOUR = 1


class GroundType(IntEnum):
    GROUND = 0
    BRIDGE = 1
    TUNNEL = 2


class RoadMarkingType(IntEnum):
    SOLID = 0
    DASHED = 1
    SOLID_SOLID = 2  # for double solid line
    SOLID_DASHED = 3  # from left to right, note this is different from ODR spec
    DASHED_SOLID = 4  # from left to right, note this is different from ODR spec
    DASHED_DASHED = 5  # for double dashed line
    BOTTS_DOTS = 6  # For these three, we can specify as a RoadMark or should they be captured elsewhere
    NO_PAINT = 7


class RoadMarkingColor(IntEnum):
    WHITE = 0
    BLUE = 1
    GREEN = 2
    RED = 3
    YELLOW = 4


T = TypeVar("T")


def load_user_data(user_data: T) -> Union[T, Dict[str, Any]]:
    try:
        return ujson.loads(user_data)
    except (ValueError, JSONDecodeError):
        return user_data


@dataclass
class RoadMarking:
    id: int
    edge_id: int
    width: float
    type: RoadMarkingType
    color: RoadMarkingColor

    @classmethod
    def from_proto(cls, road_marking: ProtoRoadMarking) -> "RoadMarking":
        return cls(
            id=road_marking.id,
            edge_id=road_marking.id,
            width=road_marking.width,
            type=RoadMarkingType(road_marking.type) if road_marking.HasField("type") else None,
            color=RoadMarkingColor(road_marking.color) if road_marking.HasField("color") else None,
        )


@dataclass
class SpeedLimit:
    speed: int
    units: SpeedUnits

    @classmethod
    def from_proto(cls, speed_limit: ProtoSpeedLimit) -> "SpeedLimit":
        return cls(speed=speed_limit.speed, units=SpeedUnits(speed_limit.units))


@dataclass
class PointENU(Point3DGeometry):
    @classmethod
    def from_proto(cls, point: ProtoPointENU):
        return cls(x=point.x, y=point.y, z=point.z)

    @classmethod
    def from_transformation(cls, tf: Transformation):
        return cls(x=tf.translation[0], y=tf.translation[1], z=tf.translation[2])


@dataclass
class Edge(Polyline3DGeometry):
    id: int
    closed: bool = False
    road_marking: Union[RoadMarking, List[RoadMarking]] = None
    user_data: Dict[str, Any] = field(default_factory=dict)

    def transform(self, tf: Transformation) -> "Edge":
        return Edge(
            id=self.id,
            closed=self.closed,
            road_marking=self.road_marking,
            user_data=self.user_data,
            lines=[ll.transform(tf=tf) for ll in self.lines],
        )

    @classmethod
    def from_numpy(
        cls,
        points: np.ndarray,
        id: int,
        closed: bool = False,
        road_marking: Optional[RoadMarking] = None,
        user_data: Optional[Dict[str, Any]] = None,
    ) -> "Edge":
        if user_data is None:
            user_data = {}
        points = points.reshape(-1, 3)
        point_pairs = np.hstack([points[:-1], points[1:]])
        return Edge(
            id=id,
            closed=closed,
            road_marking=road_marking,
            user_data=user_data,
            lines=np.apply_along_axis(Line3DGeometry.from_numpy, point_pairs),
        )

    @classmethod
    def from_proto(cls, edge: ProtoEdge, road_marking: Optional[ProtoRoadMarking] = None):
        return cls(
            id=edge.id,
            closed=not (edge.open),
            lines=[
                Line3DGeometry(
                    start=PointENU.from_proto(point=point_pair[0]), end=PointENU.from_proto(point=point_pair[1])
                )
                for point_pair in windowed(edge.points, 2)
            ],
            road_marking=RoadMarking.from_proto(road_marking=road_marking) if road_marking is not None else None,
            user_data=load_user_data(edge.user_data) if edge.HasField("user_data") else {},
        )


class Road:
    ...


@dataclass
class RoadSegment:
    id: int
    name: str
    predecessors: List[int] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    lane_segments: List[int] = field(default_factory=list)
    reference_line: Optional[Edge] = None
    type: Optional[RoadType] = None
    ground_type: Optional[GroundType] = None
    speed_limit: Optional[SpeedLimit] = None
    junction_id: Optional[int] = None
    user_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_proto(cls, id: int, umd_map: ProtoUniversalMap) -> "RoadSegment":
        road_segment: ProtoRoadSegment = umd_map.road_segments[id]
        return RoadSegment(
            id=road_segment.id,
            name=road_segment.name,
            predecessors=[ls_p for ls_p in road_segment.predecessors],
            successors=[ls_s for ls_s in road_segment.successors],
            reference_line=Edge.from_proto(edge=umd_map.edges[road_segment.reference_line])
            if road_segment.HasField("reference_line")
            else None,
            type=RoadType(road_segment.type),
            ground_type=GroundType(road_segment.ground_type),
            speed_limit=SpeedLimit.from_proto(speed_limit=road_segment.speed_limit)
            if road_segment.HasField("speed_limit")
            else None,
            junction_id=road_segment.junction_id,
            user_data=load_user_data(road_segment.user_data) if road_segment.HasField("user_data") else {},
        )


@dataclass
class LaneSegmentCollection:
    ids: List[int]
    types: List[LaneType]
    directions: List[Direction]
    left_edges: List[Edge]
    right_edges: List[Edge]
    reference_lines: List[Edge]

    @property
    def left_edge(self) -> Edge:
        combined_edge = Edge(id=int("".join([str(id) for id in self.ids])), lines=[], road_marking=[])
        for i, l_edge in enumerate(self.left_edges):
            combined_edge.road_marking.append(l_edge.road_marking)
            if self.directions[i] is Direction.FORWARD:
                combined_edge.lines.extend(l_edge.lines)
            else:
                combined_edge.lines.extend([Line3DGeometry(start=ll.end, end=ll.start) for ll in l_edge.lines[::-1]])

        return combined_edge

    @property
    def right_edge(self) -> Edge:
        combined_edge = Edge(id=int("".join([str(id) for id in self.ids])), lines=[], road_marking=[])
        for i, r_edge in enumerate(self.right_edges):
            combined_edge.road_marking.append(r_edge.road_marking)
            if self.directions[i] is Direction.FORWARD:
                combined_edge.lines.extend(r_edge.lines)
            else:
                combined_edge.lines.extend([Line3DGeometry(start=ll.end, end=ll.start) for ll in r_edge.lines[::-1]])

        return combined_edge

    @property
    def reference_line(self) -> Edge:
        combined_line = Edge(
            id=int("".join([str(id) for id in self.ids])),
            lines=[],
        )
        for i, r_lines in enumerate(self.reference_lines):
            direction = self.directions[i]

            if direction is Direction.FORWARD:
                combined_line.lines.extend(r_lines.lines)
            else:
                combined_line.lines.extend([Line3DGeometry(start=ll.end, end=ll.start) for ll in r_lines.lines[::-1]])

        return combined_line

    def transform(self, tf: Transformation) -> "LaneSegmentCollection":
        return LaneSegmentCollection(
            ids=self.ids,
            types=self.types,
            directions=self.directions,
            left_edges=self.left_edges,
            right_edges=self.right_edges,
            reference_lines=self.reference_lines,
        )

    @classmethod
    def from_lane_segments(cls, lane_segments: List["LaneSegment"]) -> "LaneSegmentCollection":
        return cls(
            ids=[ls.id for ls in lane_segments],
            types=[ls.type for ls in lane_segments],
            directions=[ls.direction for ls in lane_segments],
            left_edges=[ls.left_edge for ls in lane_segments],
            right_edges=[ls.right_edge for ls in lane_segments],
            reference_lines=[ls.reference_line for ls in lane_segments],
        )


@dataclass
class LaneSegment:
    id: int
    type: LaneType
    direction: Direction
    left_edge: Edge
    right_edge: Edge
    reference_line: Edge
    predecessors: List[int] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    left_neighbor: Optional[int] = None
    right_neighbor: Optional[int] = None
    compass_angle: Optional[float] = None
    turn_angle: Optional[float] = None
    turn_type: Optional[TurnType] = None
    user_data: Dict[str, Any] = field(default=dict)

    def to_numpy(self, closed: bool = False) -> np.ndarray:
        if not closed:
            return np.vstack([self.left_edge.to_numpy(), self.right_edge.to_numpy()[::-1]])
        else:
            return np.vstack(
                [self.left_edge.to_numpy(), self.right_edge.to_numpy()[::-1], self.left_edge.to_numpy()[0]]
            )

    @classmethod
    def from_proto(cls, id: int, umd_map: ProtoUniversalMap) -> "LaneSegment":
        lane_segment: ProtoLaneSegment = umd_map.lane_segments[id]
        return LaneSegment(
            id=lane_segment.id,
            type=LaneType(lane_segment.type),
            direction=Direction(lane_segment.direction),
            left_edge=Edge.from_proto(
                edge=umd_map.edges[lane_segment.left_edge],
                road_marking=umd_map.road_markings[lane_segment.left_edge]
                if lane_segment.left_edge in umd_map.road_markings
                else None,
            ),
            right_edge=Edge.from_proto(
                edge=umd_map.edges[lane_segment.right_edge],
                road_marking=umd_map.road_markings[lane_segment.right_edge]
                if lane_segment.right_edge in umd_map.road_markings
                else None,
            ),
            reference_line=Edge.from_proto(edge=umd_map.edges[lane_segment.reference_line]),
            predecessors=[ls_p for ls_p in lane_segment.predecessors],
            successors=[ls_s for ls_s in lane_segment.successors],
            left_neighbor=lane_segment.left_neighbor,
            right_neighbor=lane_segment.right_neighbor,
            compass_angle=lane_segment.compass_angle,
            turn_angle=lane_segment.turn_angle,
            turn_type=TurnType(lane_segment.turn_type),
            user_data=load_user_data(lane_segment.user_data) if lane_segment.HasField("user_data") else {},
        )


class Area:
    ...


@dataclass
class Junction:
    id: int
    lane_segments: List[int] = field(default_factory=list)
    road_segments: List[int] = field(default_factory=list)
    signaled_intersection: Optional[int] = None
    user_data: Dict[str, Any] = field(default=dict)
    corners: List[int] = field(default_factory=list)
    crosswalk_lanes: List[int] = field(default_factory=list)
    signed_intersection: Optional[int] = None

    @classmethod
    def from_proto(cls, id: int, umd_map: ProtoUniversalMap) -> "Junction":
        junction: ProtoJunction = umd_map.junctions[id]
        return Junction(
            id=junction.id,
            lane_segments=[j_ls for j_ls in junction.lane_segments],
            road_segments=[j_rs for j_rs in junction.road_segments],
            signaled_intersection=junction.signaled_intersection
            if junction.HasField("signaled_intersection")
            else None,
            user_data=load_user_data(junction.user_data) if junction.HasField("user_data") else {},
            corners=[j_co for j_co in junction.corners],
            crosswalk_lanes=[j_cw for j_cw in junction.crosswalk_lanes],
            signed_intersection=junction.signed_intersection if junction.HasField("signed_intersection") else None,
        )


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
            self._map_graph.es.select(type_eq=f"{NODE_PREFIX.LANE_SEGMENT}_preceeds_{NODE_PREFIX.LANE_SEGMENT}"),
            delete_vertices=False,
        )

    def _get_junctions_containing_lane_segments_graph(self) -> Graph:
        return self._map_graph.subgraph_edges(
            self._map_graph.es.select(type_eq=f"{NODE_PREFIX.JUNCTION}_contains_{NODE_PREFIX.LANE_SEGMENT}"),
            delete_vertices=False,
        )

    def _get_road_segments_containing_lane_segments_graph(self) -> Graph:
        return self._map_graph.subgraph_edges(
            self._map_graph.es.select(type_eq=f"{NODE_PREFIX.ROAD_SEGMENT}_contains_{NODE_PREFIX.LANE_SEGMENT}"),
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

            road_segment_node_id = f"{NODE_PREFIX.ROAD_SEGMENT}_{rs_key}"

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
                    (road_segment_node_id, f"{NODE_PREFIX.ROAD_SEGMENT}_{successor}")
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
                object=LaneSegment.from_proto(id=ls_key, umd_map=self._umd_map),
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
                object=Junction.from_proto(id=jc_key, umd_map=self._umd_map),
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

    def get_junction(self, id: int) -> Optional[Junction]:
        query_results = self._map_graph.vs.select(name_eq=f"{NODE_PREFIX.JUNCTION}_{id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    def get_road_segment(self, id: int) -> Optional[RoadSegment]:
        query_results = self._map_graph.vs.select(name_eq=f"{NODE_PREFIX.ROAD_SEGMENT}_{id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    def get_lane_segment(self, id: int) -> Optional[LaneSegment]:
        query_results = self._map_graph.vs.select(name_eq=f"{NODE_PREFIX.LANE_SEGMENT}_{id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    def get_area(self, id: int) -> Area:
        query_results = self._map_graph.vs.select(name_eq=f"{NODE_PREFIX.AREA}_{id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    #
    # def get_lane_segment_collection(self, ids: List[int]) -> LaneSegmentCollection:
    #     return LaneSegmentCollection.from_lane_segments(lane_segments=[self.get_lane_segment(id=id) for id in ids])
    #

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
            node_prefix=NODE_PREFIX.ROAD_SEGMENT, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, method=method
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
            node_prefix=NODE_PREFIX.LANE_SEGMENT, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, method=method
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
            node_prefix=NODE_PREFIX.AREA, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, method=method
        )

    def get_junctions_for_lane_segment(self, id: int) -> List[Junction]:
        subgraph = self._get_junctions_containing_lane_segments_graph()
        source_node = subgraph.vs.find(f"{NODE_PREFIX.LANE_SEGMENT}_{id}")

        junctions = subgraph.es.select(_target=source_node)

        return [j.source_vertex["object"] for j in junctions]

    def get_road_segment_for_lane_segment(self, id: int) -> Optional[RoadSegment]:
        subgraph = self._get_road_segments_containing_lane_segments_graph()
        target_node = subgraph.vs.find(f"{NODE_PREFIX.LANE_SEGMENT}_{id}")

        road_segments = subgraph.es.select(_target=target_node)

        if len(road_segments) > 0:
            return road_segments[0].source_vertex["object"]
        else:
            return None

    def get_lane_segment_successors(
        self, id: int, depth: int = -1
    ) -> Tuple[List[LaneSegment], List[int], List[Optional[int]]]:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        start_node = subgraph.vs.find(f"{NODE_PREFIX.LANE_SEGMENT}_{id}")

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
        start_node = subgraph.vs.find(f"{NODE_PREFIX.LANE_SEGMENT}_{source_id}")
        end_node = subgraph.vs.find(f"{NODE_PREFIX.LANE_SEGMENT}_{target_id}")

        shortest_paths = subgraph.get_shortest_paths(v=start_node, to=end_node.index, mode="out")
        return [[subgraph.vs[node_id]["object"] for node_id in shortest_path] for shortest_path in shortest_paths]

    def get_lane_segment_successors_random_path(self, id: int, steps: int = None):
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        start_vertex = subgraph.vs.find(f"{NODE_PREFIX.LANE_SEGMENT}_{id}")
        random_walk = subgraph.random_walk(
            start=start_vertex.index,
            steps=steps if steps is not None else (2 ** 16),
            mode="out",
            stuck="return",
        )
        return [subgraph.vs[node_id]["object"] for node_id in random_walk]

    def get_lane_segment_predecessors(
        self, id: int, depth: int = -1
    ) -> Tuple[List[LaneSegment], List[int], List[Optional[int]]]:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        start_node = subgraph.vs.find(f"{NODE_PREFIX.LANE_SEGMENT}_{id}")

        lane_segments = []
        levels = []
        parent_ids = []

        level = -1
        bfs_iterator = subgraph.bfsiter(vid=start_node.index, mode="in", advanced=True)
        while depth == -1 or level < depth:
            try:
                lane_segment, level, parent_lane_segment = next(bfs_iterator)
                lane_segments.append(lane_segment["object"])
                levels.append(level)
                parent_ids.append(parent_lane_segment["object"].id if parent_lane_segment is not None else None)
            except StopIteration:
                break

        return lane_segments, levels, parent_ids

    def get_lane_segment_predecessors_random_path(self, id: int, steps: int = None):
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        start_vertex = subgraph.vs.find(f"{NODE_PREFIX.LANE_SEGMENT}_{id}")
        random_walk = subgraph.random_walk(
            start=start_vertex.index,
            steps=steps if steps is not None else (2 ** 16),
            mode="in",
            stuck="return",
        )
        return [subgraph.vs[node_id]["object"] for node_id in random_walk]

    def get_lane_segment_neighbors(self, id: int) -> (Optional[LaneSegment], Optional[LaneSegment]):
        lane_segment = self.get_lane_segment(id=id)

        return (
            (
                self.get_lane_segment(id=lane_segment.left_neighbor),
                self.get_lane_segment(id=lane_segment.right_neighbor),
            )
            if lane_segment is not None
            else (None, None)
        )

    def has_lane_segment_split(self, id: int) -> bool:
        return len(self.get_lane_segment_successors(id=id, depth=1)) > 2  # [start_node, one_successor, ...]

    def has_lane_segment_merge(self, id: int) -> bool:
        return len(self.get_lane_segment_predecessors(id=id, depth=1)) > 2  # [start_node, one_predecessor, ...]

    def are_connected_lane_segments(self, id_1: int, id_2: int) -> bool:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        node_1 = subgraph.vs.find(f"{NODE_PREFIX.LANE_SEGMENT}_{id_1}")
        node_2 = subgraph.vs.find(f"{NODE_PREFIX.LANE_SEGMENT}_{id_2}")
        edge_id = subgraph.get_eid(node_1, node_2, directed=False, error=False)
        return False if edge_id == -1 else True

    def are_opposite_direction_lane_segments(self, id_1: int, id_2: int) -> bool:
        lane_segment_1 = self.get_lane_segment(id=id_1)
        lane_segment_2 = self.get_lane_segment(id=id_2)

        return True if lane_segment_1.direction != lane_segment_2.direction else False

    def are_succeeding_lane_segments(self, id_1: int, id_2: int) -> bool:
        subgraph = self._get_lane_segments_preceeding_lane_segments_graph()
        node_1 = subgraph.vs.find(f"{NODE_PREFIX.LANE_SEGMENT}_{id_1}")
        node_2 = subgraph.vs.find(f"{NODE_PREFIX.LANE_SEGMENT}_{id_2}")
        edge_id = subgraph.get_eid(node_1, node_2, directed=True, error=False)
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
