import random
import logging
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
from more_itertools import windowed
from pd.core import PdError
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
from paralleldomain.utilities.geometry import random_point_within_2d_polygon

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

logger = logging.getLogger(__name__)


class Side(Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


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

    def get_random_area_location(
        self, area_type: UMD_pb2.Area.AreaType, random_seed: int, num_points: int = 1, area_id: Optional[int] = None
    ) -> Optional[Transformation]:
        random_state = random.Random(random_seed)

        if area_id is None:
            area = self.get_random_area_object(area_type=area_type, random_seed=random_seed)
        else:
            area = self.map.areas.get(area_id)

        if area is None:
            return None

        edge_line = self.map.edges[int(area.edges[0])].as_polyline().to_numpy()

        point = random_point_within_2d_polygon(
            edge_2d=edge_line[:, :2], random_seed=random_seed, num_points=num_points
        )[0]

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
        min_path_length: Optional[float] = None,
        relative_location_variance: float = 0.0,
        direction_variance_in_degrees: float = 0.0,
        sample_rate: int = 100,
        max_retries: int = 1000,
    ) -> Optional[Transformation]:
        random_state = random.Random(random_seed)
        lane_segment = self.get_random_lane_type_object(
            lane_type=lane_type, random_seed=random_seed, min_path_length=min_path_length, max_retries=max_retries
        )

        if lane_segment is None:
            return None

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

    def get_random_lane_type_object(
        self,
        lane_type: UMD_pb2.LaneSegment.LaneType,
        random_seed: int,
        min_path_length: Optional[float] = None,
        max_retries: int = 1000,
    ) -> Optional[LaneSegment]:
        seed = random_seed
        attempts = 0
        valid_lane_found = False

        while not valid_lane_found:
            random_state = random.Random(seed)

            lane_segment_ids = [
                lane_segment_id
                for lane_segment_id, lane_segment in self.map.lane_segments.items()
                if lane_segment.type in [lane_type]
            ]

            if len(lane_segment_ids) == 0:
                return None

            lane_segment_id = random_state.choice(lane_segment_ids)
            lane_segment = self.map.lane_segments.get(lane_segment_id)

            if min_path_length is None or self.check_lane_is_longer_than(
                lane_id=lane_segment_id, path_length=min_path_length
            ):
                return lane_segment
            elif attempts > max_retries:
                logger.warning("Unable to find valid lane object with given min_path_length")
                return None
            else:
                seed += 1
                attempts += 1

    def get_random_road_type_object(
        self,
        road_type: UMD_pb2.RoadSegment.RoadType,
        random_seed: int,
        min_path_length: Optional[float] = None,
        max_retries: int = 1000,
    ) -> Optional[RoadSegment]:
        seed = random_seed
        attempts = 0
        valid_lane_found = False

        while not valid_lane_found:
            random_state = random.Random(seed)

            road_segment_ids = [
                road_segment_id
                for road_segment_id, road_segment in self.map.road_segments.items()
                if road_segment.type in [road_type]
            ]

            if len(road_segment_ids) == 0:
                return None

            road_segment_id = random_state.choice(road_segment_ids)
            road_segment = self.map.road_segments.get(road_segment_id)

            if min_path_length is None:
                return road_segment

            for lane_id in road_segment.lane_segments:
                if self.check_lane_is_longer_than(lane_id=lane_id, path_length=min_path_length):
                    return road_segment

            if attempts > max_retries:
                logger.warning("Unable to find valid road segment with given min_path_length")
                return None
            else:
                seed += 1
                attempts += 1

    def get_random_lane_object_from_road_type(
        self,
        road_type: UMD_pb2.RoadSegment.RoadType,
        random_seed: int,
        min_path_length: Optional[float] = None,
        max_retries: int = 1000,
    ) -> Optional[LaneSegment]:
        seed = random_seed
        attempts = 0
        valid_lane_found = False

        while not valid_lane_found:
            random_state = random.Random(seed)

            road_object = self.get_random_road_type_object(road_type=road_type, random_seed=seed)

            if road_object is None:
                return None

            lane_object = self.map.lane_segments[random_state.choice(road_object.lane_segments)]

            if (
                self.check_lane_is_longer_than(lane_id=lane_object.id, path_length=min_path_length)
                or min_path_length is None
            ):
                return lane_object
            elif attempts > max_retries:
                logger.warning("Unable to find valid lane segment with given min_path_length")
                return None
            else:
                seed += 1
                attempts += 1

    def get_random_junction_object(self, intersection_type: str, random_seed: int) -> Optional[Junction]:
        random_state = random.Random(random_seed)

        if intersection_type == "signaled":
            junction_ids = [j_id for j_id in self.map.signaled_intersections]
        elif intersection_type == "signed":
            junction_ids = [j_id for j_id in self.map.signed_intersections]
        else:
            raise ValueError("Invalid intersection type selected - must be 'signed' or 'signaled")

        if len(junction_ids) > 0:
            junction_id = random_state.choice(junction_ids)
        else:
            return None

        junction = self.map.junctions[junction_id]

        return junction

    def get_random_junction_relative_lane_location(
        self,
        random_seed: int,
        distance_to_junction: float = 10.0,
        probability_of_signaled_junction: float = 0.5,
        probability_of_arriving_junction: float = 1.0,
    ) -> Optional[Transformation]:
        random_state = random.Random(random_seed)
        np.random.seed(random_seed)

        type_of_intersection = np.random.choice(
            ["signed", "signaled"], 1, p=[1 - probability_of_signaled_junction, probability_of_signaled_junction]
        )[0]

        junction_to_spawn = self.get_random_junction_object(
            intersection_type=type_of_intersection, random_seed=random_seed
        )

        if junction_to_spawn is None:
            return None

        valid_lanes_at_junctions = [
            self.map.lane_segments[id]
            for id in junction_to_spawn.lane_segments
            if self.map.lane_segments[id].direction is LaneSegment.Direction.FORWARD
        ]

        junction_lane = random_state.choice(valid_lanes_at_junctions)

        arriving_junction = True if random_state.uniform(0.0, 1.0) < probability_of_arriving_junction else False

        # Loop through the previous lanes to find the required distance from junction
        accumulated_distance = 0.0

        current_lane = (
            self.map.lane_segments[junction_lane.predecessors[0]]
            if arriving_junction
            else self.map.lane_segments[junction_lane.successors[0]]
        )

        super_line = []  # Collect all the points of the total lane prior to the junction
        while accumulated_distance < distance_to_junction:
            current_line = self.map.edges[current_lane.reference_line].as_polyline().to_numpy()

            accumulated_distance += np.linalg.norm(current_line[-1] - current_line[0])

            super_line.append(current_line)

            try:
                current_lane = (
                    self.map.lane_segments[current_lane.predecessors[0]]
                    if arriving_junction
                    else self.map.lane_segments[current_lane.predecessors[0]]
                )
            except IndexError:  # In the case that we don't have long enough lanes to meet the distance requirement
                logger.warning(
                    (
                        "Unable to find lane location which matches given distance requirement. "
                        "Consider reducing requested distance to junction"
                    )
                )
                return None

        super_line = np.concatenate(super_line, axis=0)

        distance_between_points = np.linalg.norm(np.diff(super_line, axis=0), axis=1)
        cumulative_distance_bt_points = np.cumsum(distance_between_points)

        spawn_point = next(
            super_line[i]
            for i in range(len(cumulative_distance_bt_points))
            if cumulative_distance_bt_points[i] >= distance_to_junction
        )

        translation = [spawn_point[0], spawn_point[1], spawn_point[2]]

        return Transformation.from_euler_angles(
            angles=[0.0, 0.0, 0.0], order="xyz", translation=translation, degrees=True
        )

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

    def check_lane_is_longer_than(self, lane_id: int, path_length: float) -> bool:
        """
        Checks that a given lane is longer than a given length

        Args:
            lane_id: The ID of the lane segment which we are checking the length of
            path_length: The length against which the length of the lane segment should be compared

        Returns:
            True when the lane is longer than path_length, False otherwise
        """
        current_lane = self.map.lane_segments[lane_id]

        # If the current_lane is backwards, we skip immediately to its successor
        try:
            current_lane = self.map.lane_segments[current_lane.successors[0]]
        except IndexError:
            return False

        accumulated_distance = 0.0

        while accumulated_distance < path_length:
            current_line = self.map.edges[current_lane.reference_line].as_polyline().to_numpy()

            # If the lane is backwards, flip it around
            if current_lane.direction is LaneSegment.Direction.BACKWARD:
                current_line = np.flip(current_line, axis=0)

            accumulated_distance += np.linalg.norm(current_line[-1] - current_line[0])

            try:
                current_lane = self.map.lane_segments[current_lane.successors[0]]
            except IndexError:  # In the case that we don't have long enough lanes to meet the distance requirement
                continue

        if accumulated_distance >= path_length:
            return True
        else:
            return False

    def get_connected_lane_points(self, lane_id: int, path_length: float) -> np.ndarray:
        """
        Returns all the points of the lane segments which are connected to a given lane segment within a certain
            path_length.  This function will throw an error if the lane segment, and it's connected lane segments are
            shorter than the specified path_length

        Args:
            lane_id: The ID of the lane segment we wish to get the points of
            path_length: The minimum length of the line of points we wish to return

        Returns:
            nx3 numpy array of points which make up the reference line of the lane beginning with the inputted lane
                segment
        """
        current_lane = self.map.lane_segments[lane_id]

        # If the current_lane is backwards, we skip immediately to its successor
        try:
            current_lane = self.map.lane_segments[current_lane.successors[0]]
        except IndexError:
            raise PdError(
                "Insufficient length of connected lanes to meet specified path_length. "
                "Check that connected lanes are long enough first using check_available_path_length_in_lane."
            )

        accumulated_distance = 0.0
        super_line = []  # Collect all the points of the total lane
        while accumulated_distance < path_length:
            current_line = self.map.edges[current_lane.reference_line].as_polyline().to_numpy()

            # If the lane is backwards, flip it around
            if current_lane.direction is LaneSegment.Direction.BACKWARD:
                current_line = np.flip(current_line, axis=0)

            accumulated_distance += np.linalg.norm(current_line[-1] - current_line[0])

            super_line.append(current_line)

            try:
                current_lane = self.map.lane_segments[current_lane.successors[0]]
            except IndexError:  # In the case that we don't have long enough lanes to meet the distance requirement
                continue

        if accumulated_distance >= path_length:
            super_line = np.concatenate(super_line, axis=0)

            return super_line
        else:
            raise PdError(
                "Insufficient length of connected lanes to meet specified path_length. "
                "Check that connected lanes are long enough first using check_available_path_length_in_lane."
            )

    def get_edge_of_road_from_lane(self, lane_id: int, side: Side) -> Edge:
        """
        Function which returns either the left or right edge of the roac on which the lane we specify exists

        Args:
            lane_id: The ID of the lane which exists on the road we wish to find the edge of
            which_edge: Choose to return either the left or right edge

        Returns:
            An Edge object of the edge of the road corresponding to the inputted parameters
        """
        current_lane = self.map.lane_segments[lane_id]

        neighbor_id = None
        while neighbor_id != 0:
            neighbor_id = current_lane.left_neighbor if side is Side.LEFT else current_lane.right_neighbor

            if neighbor_id != 0:
                current_lane = self.map.lane_segments[neighbor_id]

        road_edge = self.map.edges[current_lane.left_edge if side is Side.LEFT else current_lane.right_edge]
        return road_edge

    def get_line_point_by_distance_from_start(self, line: np.ndarray, distance_from_start: float) -> np.ndarray:
        """
        Given a line of points, returns the point that is the first to be more than the specified distance from the
            start of the line.  If the distance specified is longer than the line, the last point on the line will
            be returned

        Args:
            line: nx3 numpy array containing the 3d points which make up the line
            distance_from_start: The distance from the start of the line, after which we want to return the first point

        Returns:
            nx3 numpy array corresponding to the first point on the line which is greater than the specified
                distance_from_start from the start of the line
        """
        distance_between_points = np.linalg.norm(np.diff(line, axis=0), axis=1)
        cumulative_distance_bt_points = np.cumsum(distance_between_points)
        try:
            point = next(
                line[i]
                for i in range(len(cumulative_distance_bt_points))
                if cumulative_distance_bt_points[i] >= distance_from_start
            )
        except StopIteration:
            logger.warning("Line provided was shorter than distance_from_start specified.  Returning last value")
            point = line[-1]

        return point
