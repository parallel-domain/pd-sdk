import random
from typing import Dict, List, Optional, Union

import numpy as np
from more_itertools import windowed
from pd.internal.proto.umd.generated.wrapper.utils import register_wrapper

from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DBaseGeometry
from paralleldomain.model.geometry.point_3d import Point3DGeometry
from paralleldomain.model.geometry.polyline_3d import Line3DGeometry, Polyline3DBaseGeometry
from paralleldomain.model.type_aliases import AreaId, EdgeId, JunctionId, LaneSegmentId, RoadSegmentId
from paralleldomain.utilities.transformation import Transformation

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

from igraph import Graph, Vertex
from pd.internal.proto.umd.generated.python import UMD_pb2 as UMD_pb2_base
from pd.internal.proto.umd.generated.wrapper import UMD_pb2
from paralleldomain.utilities.geometry import (
    decompose_polygon_into_triangles,
    calculate_triangle_area,
    random_point_in_triangle,
    random_point_within_2d_polygon,
)

AABB = UMD_pb2.AABB
Info = UMD_pb2.Info
Object = UMD_pb2.Object
Phase = UMD_pb2.Phase
Point_ECEF = UMD_pb2.Point_ECEF
Point_LLA = UMD_pb2.Point_LLA
PropData = UMD_pb2.PropData
Quaternion = UMD_pb2.Quaternion
SignalOnset = UMD_pb2.SignalOnset
SignaledIntersection = UMD_pb2.SignaledIntersection
SignedIntersection = UMD_pb2.SignedIntersection
SpeedLimit = UMD_pb2.SpeedLimit
TrafficLightBulb = UMD_pb2.TrafficLightBulb
TrafficLightData = UMD_pb2.TrafficLightData
TrafficSignData = UMD_pb2.TrafficSignData
ZoneGrid = UMD_pb2.ZoneGrid
UniversalMap = UMD_pb2.UniversalMap
RoadMarking = UMD_pb2.RoadMarking
Point_ENU = UMD_pb2.Point_ENU


@register_wrapper(proto_type=UMD_pb2_base.Edge)
class Edge(UMD_pb2.Edge):
    def as_polyline(self) -> Polyline3DBaseGeometry:
        lines = [
            Line3DGeometry(
                start=Point3DGeometry(x=point_pair[0].x, y=point_pair[0].y, z=point_pair[0].z),
                end=Point3DGeometry(x=point_pair[1].x, y=point_pair[1].y, z=point_pair[1].z)
                if point_pair[1] is not None
                else Point3DGeometry(x=point_pair[0].x, y=point_pair[0].y, z=point_pair[0].z),
            )
            for point_pair in windowed(self.points, 2)
        ]
        return Polyline3DBaseGeometry(lines=lines)


@register_wrapper(proto_type=UMD_pb2_base.Junction)
class Junction(UMD_pb2.Junction):
    @property
    def junction_id(self) -> JunctionId:
        return self.id


@register_wrapper(proto_type=UMD_pb2_base.LaneSegment)
class LaneSegment(UMD_pb2.LaneSegment):
    @property
    def lane_segment_id(self) -> LaneSegmentId:
        return self.id


@register_wrapper(proto_type=UMD_pb2_base.RoadSegment)
class RoadSegment(UMD_pb2.RoadSegment):
    @property
    def road_segment_id(self) -> RoadSegmentId:
        return self.id


@register_wrapper(proto_type=UMD_pb2_base.Area)
class Area(UMD_pb2.Area):
    @property
    def area_id(self) -> AreaId:
        return self.id


class NodePrefix:
    ROAD_SEGMENT: str = "RS"
    LANE_SEGMENT: str = "LS"
    JUNCTION: str = "JC"
    AREA: str = "AR"


class MapQuery:
    def __init__(self, map: UniversalMap):
        super().__init__()
        self.edges: Dict[int, Edge] = dict()
        self.__map_graph = Graph(directed=True)
        self._added_road_segments = False
        self._added_lane_segments = False
        self._added_junctions = False
        self._added_areas = False
        self.map = map
        self.edges.update(map.edges)
        self._add_road_segments_to_graph(road_segments=map.road_segments)
        self._add_lane_segments_to_graph(lane_segments=map.lane_segments)
        self._add_junctions_to_graph(junctions=map.junctions)
        self._add_areas_to_graph(areas=map.areas)

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

    def _get_bounds(
        self, element: Union[LaneSegment, RoadSegment, Junction, Area]
    ) -> Optional[BoundingBox2DBaseGeometry[float]]:
        all_points = np.empty(shape=(0, 2))
        if isinstance(element, UMD_pb2.LaneSegment):
            points = list()
            if element.proto.HasField("reference_line"):
                reference_line = self.edges[element.reference_line]
                reference_points = np.array([(p.x, p.y) for p in reference_line.points])
                points.append(reference_points)

            if element.proto.HasField("left_edge"):
                left_edge = self.edges[element.left_edge]
                left_points = np.array([(p.x, p.y) for p in left_edge.points])
                points.append(left_points)

            if element.proto.HasField("right_edge"):
                right_edge = self.edges[element.right_edge]
                right_points = np.array([(p.x, p.y) for p in right_edge.points])
                points.append(right_points)

            all_points = np.vstack(points)
        elif isinstance(element, UMD_pb2.RoadSegment):
            if element.proto.HasField("reference_line"):
                reference_line = self.edges[element.reference_line]
                all_points = np.array([(p.x, p.y) for p in reference_line.points])
            else:
                bounds = [self._get_bounds(element=self.map.lane_segments[lid]) for lid in element.lane_segments]
                if len(bounds) > 0:
                    b0 = bounds[0]
                    for b in bounds[1:]:
                        b0 = BoundingBox2DBaseGeometry.merge_boxes(b0, b)
                    return b0
        elif isinstance(element, UMD_pb2.Junction):
            for corner in element.corners:
                corner_edge = self.edges[corner]
                all_points = np.vstack(
                    [
                        all_points,
                        np.array([(p.x, p.y) for p in corner_edge.points]),
                    ]
                )
        elif isinstance(element, UMD_pb2.Area):
            edge = self.edges[element.edges[0]]
            all_points = np.array([(p.x, p.y) for p in edge.points])
        else:
            raise NotImplementedError(f"Bounds not implemented for type {type(element)}!")

        bounds = None
        if len(all_points) > 0:
            x_min, y_min = np.min(all_points, axis=0)
            x_max, y_max = np.max(all_points, axis=0)
            bounds = BoundingBox2DBaseGeometry[float](x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)
        return bounds

    def _add_vertex(self, element: Union[LaneSegment, RoadSegment, Junction, Area], vertex_id: str) -> Vertex:
        bounds = self._get_bounds(element=element)
        if bounds is not None:
            x_min = bounds.x
            x_max = bounds.x + bounds.width
            y_min = bounds.y
            y_max = bounds.y + bounds.height
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
            object=element,
        )
        return vertex

    def _add_road_segments_to_graph(self, road_segments: Dict[RoadSegmentId, RoadSegment]):
        for road_segment_id, road_segment in road_segments.items():
            road_segment_node_id = f"{NodePrefix.ROAD_SEGMENT}_{road_segment.id}"
            self._add_vertex(element=road_segment, vertex_id=road_segment_node_id)

    def _add_lane_segments_to_graph(self, lane_segments: Dict[LaneSegmentId, LaneSegment]):
        for lane_segment_id, lane_segment in lane_segments.items():
            lane_segment_node_id = f"{NodePrefix.LANE_SEGMENT}_{lane_segment.id}"
            self._add_vertex(element=lane_segment, vertex_id=lane_segment_node_id)

    def _add_junctions_to_graph(self, junctions: Dict[JunctionId, Junction]):
        for junction_id, junction in junctions.items():
            junction_node_id = f"{NodePrefix.JUNCTION}_{junction.id}"
            self._add_vertex(element=junction, vertex_id=junction_node_id)

    def _add_areas_to_graph(self, areas: Dict[AreaId, Area]):
        for area_id, area in areas.items():
            area_node_id = f"{NodePrefix.AREA}_{area.id}"
            self._add_vertex(element=area, vertex_id=area_node_id)

    def get_lane_segments_near(self, pose: Transformation, radius: float = 10) -> List[LaneSegment]:
        bounds = BoundingBox2DBaseGeometry(
            x=pose.translation[0], y=pose.translation[1], width=2 * radius, height=2 * radius
        )
        return self.get_lane_segments_within_bounds(bounds=bounds, method="inside")

    def get_random_area_object(self, area_type: UMD_pb2.Area.AreaType, random_seed: int) -> Optional[UMD_pb2.Area]:
        random_state = random.Random(random_seed)
        area_ids = [area_id for area_id, area in self.map.areas.items() if area.type is area_type]

        if len(area_ids) == 0:
            return None

        area_id = random_state.choice(area_ids)
        return self.map.areas.get(area_id)

    def get_random_area_location(self, area_type: UMD_pb2.Area.AreaType, random_seed: int) -> Optional[Transformation]:
        random_state = random.Random(random_seed)
        area = self.get_random_area_object(area_type=area_type, random_seed=random_seed)

        if area is None:
            return None

        edge_line = self.map.edges[int(area.edges[0])].as_polyline().to_numpy()

        point = random_point_within_2d_polygon(edge_2d=edge_line[:, :2], random_seed=random_seed, num_points=1)[0]

        translation = [point[0], point[1], np.average(edge_line[:, 2])]

        pose = Transformation.from_euler_angles(
            angles=[0.0, 0.0, random_state.uniform(0.0, 360)],
            order="xyz",
            translation=translation,
            degrees=True,
        )

        return pose

    def get_random_lane_type_location(
        self,
        lane_type: UMD_pb2.LaneSegment.LaneType,
        random_seed: int,
        relative_location_variance: float = 0.0,
        direction_variance_in_degrees: float = 0.0,
        sample_rate: int = 100,
    ) -> Transformation:
        random_state = random.Random(random_seed)
        lane_segment_ids = [
            lane_segment_id
            for lane_segment_id, lane_segment in self.map.lane_segments.items()
            if lane_segment.type in [lane_type]
        ]
        lane_segment_id = random_state.choice(lane_segment_ids)
        lane_segment = self.map.lane_segments.get(lane_segment_id)

        right_edge = Edge(proto=self.edges[lane_segment.right_edge].proto).as_polyline().to_numpy()
        right_points = np.reshape(right_edge, (-1, 3))

        left_edge = Edge(proto=self.edges[lane_segment.left_edge].proto).as_polyline().to_numpy()
        left_points = np.reshape(left_edge, (-1, 3))

        reference_line = Edge(proto=self.edges[lane_segment.reference_line].proto).as_polyline().to_numpy()
        direction_points = np.reshape(reference_line, (-1, 3))
        global_forward = np.array([1.0, 0.0, 0.0])

        i = random_state.choice(list(range(len(direction_points))))
        center = direction_points[i]
        if i < len(direction_points) - 1:
            direction = direction_points[i + 1] - direction_points[i]
        else:
            direction = direction_points[i] - direction_points[i - 1]

        p_l, p_r = random_state.choice(left_points), random_state.choice(right_points)

        left_max = ((p_l - center) * relative_location_variance) + center
        right_max = ((p_r - center) * relative_location_variance) + center
        sampled_right_points = np.linspace(center, right_max, sample_rate, endpoint=False)
        sampled_left_points = np.linspace(center, left_max, sample_rate, endpoint=False)

        sampled_points = np.concatenate([sampled_right_points, sampled_left_points], 0).tolist()

        direction[2] = 0
        direction = direction / np.linalg.norm(direction)
        yaw_direction = np.rad2deg(
            np.arccos(np.dot(direction, global_forward) / (np.linalg.norm(direction) * np.linalg.norm(global_forward)))
        )
        yaw_noise = yaw_direction + (2 * random_state.random() - 1.0) * direction_variance_in_degrees

        translation = random_state.choice(sampled_points)

        pose = Transformation.from_euler_angles(
            angles=[0.0, 0.0, yaw_noise], order="xyz", translation=translation, degrees=True
        )

        return pose

    def get_random_street_location(
        self,
        random_seed: int,
        relative_location_variance: float = 0.0,
        direction_variance_in_degrees: float = 0.0,
        sample_rate: int = 100,
    ) -> Transformation:
        return self.get_random_lane_type_location(
            lane_type=LaneSegment.LaneType.DRIVABLE,
            random_seed=random_seed,
            relative_location_variance=relative_location_variance,
            direction_variance_in_degrees=direction_variance_in_degrees,
            sample_rate=sample_rate,
        )
