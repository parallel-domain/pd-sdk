import logging
import random
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
from igraph import Graph, Vertex
from more_itertools import windowed
from pd.core import PdError
from pd.internal.proto.umd.generated.python import UMD_pb2 as UMD_pb2_base
from pd.internal.proto.umd.generated.wrapper import UMD_pb2
from pd.internal.proto.umd.generated.wrapper.utils import register_wrapper

from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DBaseGeometry
from paralleldomain.model.geometry.point_3d import Point3DGeometry
from paralleldomain.model.geometry.polyline_3d import Line3DGeometry, Polyline3DBaseGeometry
from paralleldomain.model.type_aliases import AreaId, EdgeId, JunctionId, LaneSegmentId, RoadSegmentId
from paralleldomain.utilities import inherit_docs
from paralleldomain.utilities.geometry import random_point_within_2d_polygon
from paralleldomain.utilities.transformation import Transformation

logger = logging.getLogger(__name__)


@inherit_docs
class AABB(UMD_pb2.AABB):
    ...


@inherit_docs
class Info(UMD_pb2.Info):
    ...


@inherit_docs
class Object(UMD_pb2.Object):
    ...


@inherit_docs
class Phase(UMD_pb2.Phase):
    ...


@inherit_docs
class Point_ECEF(UMD_pb2.Point_ECEF):
    ...


@inherit_docs
class Point_LLA(UMD_pb2.Point_LLA):
    ...


@inherit_docs
class PropData(UMD_pb2.PropData):
    ...


@inherit_docs
class Quaternion(UMD_pb2.Quaternion):
    ...


@inherit_docs
class SignalOnset(UMD_pb2.SignalOnset):
    ...


@inherit_docs
class SignaledIntersection(UMD_pb2.SignaledIntersection):
    ...


@inherit_docs
class SignedIntersection(UMD_pb2.SignedIntersection):
    ...


@inherit_docs
class SpeedLimit(UMD_pb2.SpeedLimit):
    ...


@inherit_docs
class TrafficLightBulb(UMD_pb2.TrafficLightBulb):
    ...


@inherit_docs
class TrafficLightData(UMD_pb2.TrafficLightData):
    ...


@inherit_docs
class TrafficSignData(UMD_pb2.TrafficSignData):
    ...


@inherit_docs
class ZoneGrid(UMD_pb2.ZoneGrid):
    ...


@inherit_docs
class UniversalMap(UMD_pb2.UniversalMap):
    ...


@inherit_docs
class RoadMarking(UMD_pb2.RoadMarking):
    ...


@inherit_docs
class Point_ENU(UMD_pb2.Point_ENU):
    ...


class Side(Enum):
    """
    Enum class for LEFT and RIGHT side.  Used by geometry functions
    """

    LEFT = "LEFT"
    RIGHT = "RIGHT"


@register_wrapper(proto_type=UMD_pb2_base.Edge)
class Edge(UMD_pb2.Edge):
    """Class representing a particular Edge on a UMD map"""

    def as_polyline(self) -> Polyline3DBaseGeometry:
        """Returns a 3D Polyline representation of the edge"""
        lines = [
            Line3DGeometry(
                start=Point3DGeometry(x=point_pair[0].x, y=point_pair[0].y, z=point_pair[0].z),
                end=(
                    Point3DGeometry(x=point_pair[1].x, y=point_pair[1].y, z=point_pair[1].z)
                    if point_pair[1] is not None
                    else Point3DGeometry(x=point_pair[0].x, y=point_pair[0].y, z=point_pair[0].z)
                ),
            )
            for point_pair in windowed(self.points, 2)
        ]
        return Polyline3DBaseGeometry(lines=lines)


@register_wrapper(proto_type=UMD_pb2_base.Junction)
class Junction(UMD_pb2.Junction):
    """Class representing a particular Junction on a UMD map"""

    @property
    def junction_id(self) -> JunctionId:
        """The integer ID of the Junction object"""
        return self.id


@register_wrapper(proto_type=UMD_pb2_base.LaneSegment)
class LaneSegment(UMD_pb2.LaneSegment):
    """Class representing a particular Lane Segment on a UMD map"""

    @property
    def lane_segment_id(self) -> LaneSegmentId:
        """The integer ID of the LaneSegment object"""
        return self.id


@register_wrapper(proto_type=UMD_pb2_base.RoadSegment)
class RoadSegment(UMD_pb2.RoadSegment):
    """Class representing a particular Road Segment on a UMD map"""

    @property
    def road_segment_id(self) -> RoadSegmentId:
        """The integer ID of the RoadSegment object"""
        return self.id


@register_wrapper(proto_type=UMD_pb2_base.Area)
class Area(UMD_pb2.Area):
    """Class representing a particular Area on a UMD map"""

    @property
    def area_id(self) -> AreaId:
        """The integer ID of the Area object"""
        return self.id


class NodePrefix:
    """Class which contains the prefixes used to denote particular map elements in the UMD map"""

    ROAD_SEGMENT: str = "RS"
    LANE_SEGMENT: str = "LS"
    JUNCTION: str = "JC"
    AREA: str = "AR"


class MapQuery:
    """
    Class containing lookup tools and helper functions for interacting with and querying UMD maps

    Args:
        map: The UMD map to be interacted with and queried through the MapQuery object
    Attributes:
        edges: Dictionary containing the edge_id and Edge object of the line edges which make up the UMD map
        map: THe UMD map that is being interacted with through the MapQuery object
    """

    def __init__(self, map: UniversalMap):
        super().__init__()
        self.edges: Dict[int, Edge] = dict()
        self.__map_graph = Graph(directed=True)
        self._added_road_segments = False
        self._added_lane_segments = False
        self._added_junctions = False
        self._added_areas = False
        self.map = map
        self.edges.update(OrderedDict(sorted(map.edges.items())))
        self._add_road_segments_to_graph(road_segments=OrderedDict(sorted(map.road_segments.items())))
        self._add_lane_segments_to_graph(lane_segments=OrderedDict(sorted(map.lane_segments.items())))
        self._add_junctions_to_graph(junctions=OrderedDict(sorted(map.junctions.items())))
        self._add_areas_to_graph(areas=OrderedDict(sorted(map.areas.items())))

    @property
    def map_graph(self) -> Graph:
        """The Graph object of the map"""
        return self.__map_graph

    def get_junction(self, junction_id: JunctionId) -> Optional[Junction]:
        """
        Function to return a Junction object based on the provided junction_id

        Args:
            junction_id: The id of the Junction object to be retrieved
        Returns:
            The Junction object which corresponds to the provided junction_id, if such a Junction exists.  If
                the map contains no junctions with the provided junction_id, None is returned.
        """
        query_results = self.map_graph.vs.select(name_eq=f"{NodePrefix.JUNCTION}_{junction_id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    def get_road_segment(self, road_segment_id: RoadSegmentId) -> Optional[RoadSegment]:
        """
        Function to return a RoadSegment object based on the provided road_segment_id

        Args:
            road_segment_id: The id of the RoadSegment object to be retrieved
        Returns:
            The RoadSegment object which corresponds to the provided road_segment_id, if such a RoadSegment exists.  If
                the map contains no road segments with the provided road_segment_id, None is returned.
        """
        query_results = self.map_graph.vs.select(name_eq=f"{NodePrefix.ROAD_SEGMENT}_{road_segment_id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    def get_lane_segment(self, lane_segment_id: LaneSegmentId) -> Optional[LaneSegment]:
        """
        Function to return a LaneSegment object based on the provided lane_segment_id

        Args:
            lane_segment_id: The id of the LaneSegment object to be retrieved
        Returns:
            The LaneSegment object which corresponds to the provided lane_segment_id, if such a LaneSegment exists.  If
                the map contains no lanes with the provided lane_segment_id, None is returned.
        """
        query_results = self.map_graph.vs.select(name_eq=f"{NodePrefix.LANE_SEGMENT}_{lane_segment_id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    def get_area(self, area_id: AreaId) -> Area:
        """
        Function to return an Area object based on the provided area_id

        Args:
            area_id: The id of the Area object to be retrieved
        Returns:
            The Area object which corresponds to the provided area_id
        """
        query_results = self.map_graph.vs.select(name_eq=f"{NodePrefix.AREA}_{area_id}")
        return query_results[0]["object"] if len(query_results) > 0 else None

    def get_edge(self, edge_id: EdgeId) -> Edge:
        """
        Function to return an Edge object based on the provided edge_id

        Args:
            edge_id: The id of the Edge object to be retrieved
        Returns:
            The Edge object which corresponds to the provided edge_id
        """
        return self.edges[edge_id]

    def get_road_segments_within_bounds(
        self, bounds: BoundingBox2DBaseGeometry[float], method: str = "inside"
    ) -> List[LaneSegment]:
        """
        Function to get a list of RoadSegment objects which are located within a provided boundary

        Args:
            bounds: A rectangular polygon which outlines the area in which RoadSegment objects are searched for
            method: The method by which RoadSegment objects are checked to be within the provided bounds.  Can be
                "inside", "center" or "overlap"
        Returns:
            A list of RoadSegment objects which exist within the specified bounds
        """
        return self._get_nodes_within_bounds(node_prefix=NodePrefix.ROAD_SEGMENT, bounds=bounds, method=method)

    def get_lane_segments_within_bounds(
        self,
        bounds: BoundingBox2DBaseGeometry[float],
        method: str = "inside",
    ) -> List[LaneSegment]:
        """
        Function to get a list of LaneSegment objects which are located within a provided boundary

        Args:
            bounds: A rectangular polygon which outlines the area in which LaneSegment objects are searched for
            method: The method by which LaneSegment objects are checked to be within the provided bounds.  Can be
                "inside", "center" or "overlap"
        Returns:
            A list of LaneSegment objects which exist within the specified bounds
        """
        return self._get_nodes_within_bounds(node_prefix=NodePrefix.LANE_SEGMENT, bounds=bounds, method=method)

    def get_areas_within_bounds(
        self,
        bounds: BoundingBox2DBaseGeometry[float],
        method: str = "inside",
    ) -> List[Area]:
        """
        Function to get a list of Area objects which are located within a provided boundary

        Args:
            bounds: A rectangular polygon which outlines the area in which Area objects are searched for
            method: The method by which Area objects are checked to be within the provided bounds.  Can be "inside",
                "center" or "overlap"
        Returns:
            A list of Area objects which exist within the specified bounds
        """
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

    def get_lane_segments_near(
        self, pose: Transformation, radius: float = 10, method: str = "overlap"
    ) -> List[LaneSegment]:
        """
        Function to get a list of LaneSegment objects which are located within a specified radius of a particular
            location on the map

        Args:
            pose: The location on the map around which we return LaneSegment objects which are within the specified
                radius
            radius: The distance (in meters) around the provided pose within which LaneSegment objects are searched for
            method: The method by which lane segments are checked to be within the provided bounds.  Can be "inside",
                "center" or "overlap"

        Returns:
            A list of LaneSegment objects which exist within the specified radius around the specified location in the
                pose parameter
        """
        bounds = BoundingBox2DBaseGeometry(
            x=pose.translation[0] - radius, y=pose.translation[1] - radius, width=2 * radius, height=2 * radius
        )
        return self.get_lane_segments_within_bounds(bounds=bounds, method=method)

    def get_random_area_object(self, area_type: UMD_pb2.Area.AreaType, random_seed: int) -> Optional[UMD_pb2.Area]:
        """
        Function to get a random Area object

        Args:
            random_seed: Integer used to seed all random functions.  Allows for function return to be deterministic
            area_type: The desired type of Area

        Returns:
            Area object which is compliant with the specified parameters passed in, if such a valid area
                exists.  Returns None if no valid area can be found.
        """
        random_state = random.Random(random_seed)
        area_ids = [area_id for area_id, area in self.map.areas.items() if area.type == area_type]

        if len(area_ids) == 0:
            return None

        area_id = random_state.choice(area_ids)
        return self.map.areas.get(area_id)

    def get_random_area_location(
        self, area_type: UMD_pb2.Area.AreaType, random_seed: int, num_points: int = 1, area_id: Optional[int] = None
    ) -> Optional[Transformation]:
        """
        Function to get a random location on the map which is located on a particular selected AreaType

        Args:
            random_seed: Integer used to seed all random functions.  Allows for function return to be deterministic
            area_type: The desired type of Area on which the returned location should be located.  Will be ignored if
                area_id parameters is used
            num_points: The number of locations within a given Area object which should be returned
            area_id: The id of the Area object on which a random location should be chosen.  Optional, will override the
                area_type parameter if used

        Returns:
            Transformation object containing the pose of the randomly selected area location, if such a valid location
                exists.  Returns None if no valid location can be found.
        """
        random_state = random.Random(random_seed)

        if area_id is None:
            area = self.get_random_area_object(area_type=area_type, random_seed=random_seed)
        else:
            area = self.map.areas.get(area_id)

        if area is None:
            return None

        edge_line = self.map.edges[int(area.edges[0])].as_polyline().to_numpy()

        if edge_line.shape[0] < 3:
            raise PdError(
                "The edge line of the selected area object is malformed and does not contain at least 3 points"
            )

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
        """
        Function to get a random location on the map which corresponds to a particular selected LaneType

        Args:
            random_seed: Integer used to seed all random functions.  Allows for function return to be deterministic
            lane_type: The desired type of LaneSegment on which the returned location should be located
            min_path_length: The minimum distance (in meters) of available road beyond the location returned
                by this function
            relative_location_variance: Parameter that governs the maximum lateral variation (relative to the direction
                of the road lane) in the positioning of the returned location.  A value of 1.0 will allow for positions
                to be returned across the entire width of the lane.  A value of 0.0 will only return locations in the
                center of the lane.
            direction_variance_in_degrees: The maximum variation in the returned pose's rotation, relative to the
                direction of the lane which the returned location corresponds to
            sample_rate: The number of valid points in the lateral direction of the lane (taking into account the
                specified relative_location_variance) from which the returned location is chosen
            max_retries: The maximum number of times the method will attempt to look for a valid lane object

        Returns:
            Transformation object containing the pose of the randomly selected lane location, if such a valid location
                exists.  Returns None if no valid location can be found.
        """
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
        """
        Function to get a random LaneSegment object

        Args:
            random_seed: Integer used to seed all random functions.  Allows for function return to be deterministic
            lane_type: The desired type of LaneSegment
            min_path_length: The minimum distance (in meters) of available road beyond the LaneSegment object returned
                by this function
            max_retries: The maximum number of times the method will attempt to look for a valid lane object

        Returns:
            LaneSegment object which is compliant with the specified parameters passed in, if such a valid lane segment
                exists.  Returns None if no valid lane segments can be found.
        """
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
        """
        Function to get a random RoadSegment object

        Args:
            random_seed: Integer used to seed all random functions.  Allows for function return to be deterministic
            road_type: The desired type of RoadSegment
            min_path_length: The minimum distance (in meters) of available road beyond the RoadSegment object returned
                by this function
            max_retries: The maximum number of times the method will attempt to look for a valid road segment object

        Returns:
            RoadSegment object which is compliant with the specified parameters passed in, if such a valid road segment
                exists.  Returns None if no valid road segments can be found.
        """
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
        lane_type: Optional[UMD_pb2.LaneSegment.LaneType] = None,
        min_path_length: Optional[float] = None,
        max_retries: int = 1000,
    ) -> Optional[LaneSegment]:
        """
        Function to get a random LaneSegment object from a specified RoadType

        Args:
            random_seed: Integer used to seed all random functions.  Allows for function return to be deterministic
            road_type: The desired type of RoadSegment
            lane_type: The desired type of LaneSegment to apply within the specified RoadSegment. If None, any lane
                type will be considered valid.
            min_path_length: The minimum distance (in meters) of available road beyond the lane object returned by
                this function
            max_retries: The maximum number of times the method will attempt to look for a valid lane object

        Returns:
            LaneSegment object which is compliant with the specified parameters passed in, if such a valid lane segment
                exists.  Returns None if no valid lane segments can be found.
        """
        seed = random_seed
        attempts = 0
        valid_lane_found = False

        while not valid_lane_found:
            random_state = random.Random(seed)

            road_object = self.get_random_road_type_object(road_type=road_type, random_seed=seed)

            if road_object is None:
                return None

            lanes_from_road = road_object.lane_segments
            # filter for lane length
            lanes_from_road = [
                lane_id for lane_id in lanes_from_road if self.check_lane_is_longer_than(lane_id, min_path_length)
            ]
            # filter for lane type
            if lane_type is not None:
                lanes_from_road = [
                    lane_id for lane_id in lanes_from_road if self.map.lane_segments[lane_id].type in [lane_type]
                ]

            if len(lanes_from_road) > 0:
                return self.map.lane_segments[random_state.choice(lanes_from_road)]
            elif attempts > max_retries:
                logger.warning("Unable to find valid lane segment with given min_path_length")
                return None
            else:
                seed += 1
                attempts += 1

    def get_random_junction_object(self, intersection_type: str, random_seed: int) -> Optional[Junction]:
        """
        Function to find a random junction object on the map

        Args:
            random_seed: Integer used to seed all random functions.  Allows for function return to be deterministic
            intersection_type: The desired type of junction (signed vs signaled)

        Returns:
            Junction object which is compliant with the specified parameters passed in, if such a valid junction
                exists.  Returns None if no valid junction can be found.

        Raises:
            ValueError: If an invalid intersection_type is specified
        """
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
        """
        Returns a location on a drivable lane in a position relative to a junction

        Args:
            random_seed: Integer used to seed all random functions.  Allows for function return to be deterministic
            distance_to_junction: The desired distance between the returned lane location and the junction to which
                the lane location is connected
            probability_of_signaled_junction: The probability that the returned location is connected to a junction
                which is controlled by traffic lights as opposed to stop signs
            probability_of_arriving_junction: The probability that the return location is on a lane which is driving
                into an intersection (as opposed to driving away from an intersection)

        Returns:
            Transformation object containing the pose of the location selected by the method, if such a valid location
                exists.  Returns None if no valid location can be found.
        """
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
            if self.map.lane_segments[id].direction == LaneSegment.Direction.FORWARD
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
                    "Unable to find lane location which matches given distance requirement. "
                    "Consider reducing requested distance to junction"
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
        """
        Function to retrieve a random street location on the map

        Args:
            random_seed: Integer used to seed all random functions.  Allows for function return to be deterministic
            relative_location_variance: Parameter that governs the maximum lateral variation (relative to the direction
                of the road lane) in the positioning of the returned location.  A value of 1.0 will allow for positions
                to be returned across the entire width of the lane.  A value of 0.0 will only return locations in the
                center of the lane.
            direction_variance_in_degrees: The maximum variation in the returned pose's rotation, relative to the
                direction of the lane which the returned location corresponds to
            sample_rate: The number of valid points in the lateral direction of the lane (taking into account the
                specified relative_location_variance) from which the returned location is chosen

        Returns:
            Transformation object containing the pose of the random location selected by the method
        """
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
        Function which returns either the left or right Edge object of the road on which the lane we specify exists

        Args:
            lane_id: The ID of the lane which exists on the road we wish to find the edge of
            side: Choose to return either the left or right edge

        Returns:
            An Edge object of the edge of the road corresponding to the inputted parameters
        """
        current_lane = self.map.lane_segments[lane_id]
        start_direction = current_lane.direction

        neighbor_id = None
        while neighbor_id != 0:
            neighbor_id = (
                current_lane.left_neighbor
                if (side is Side.LEFT and current_lane.direction == start_direction)
                else current_lane.right_neighbor
            )

            if neighbor_id != 0:
                current_lane = self.map.lane_segments[neighbor_id]

        road_edge = self.map.edges[
            current_lane.left_edge
            if side is Side.LEFT and current_lane.direction == start_direction
            else current_lane.right_edge
        ]
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

    def find_lane_id_from_pose(self, pose: Transformation) -> int:
        lanes_near = [
            lane
            for lane in self.get_lane_segments_within_bounds(
                bounds=BoundingBox2DBaseGeometry(x=pose.translation[0], y=pose.translation[1], width=0, height=0),
                method="overlap",
            )
            if lane.type == LaneSegment.LaneType.DRIVABLE
        ]

        lines_near = [self.map.edges[lane.reference_line].as_polyline().to_numpy() for lane in lanes_near]

        dist_to_lines = np.array([np.min(np.linalg.norm(line - pose.translation, axis=1)) for line in lines_near])

        pose_front_direction = pose.quaternion.rotation_matrix @ np.array([0, 1, 0])

        # Check where there are more than one lane near the pose
        half_lane_width = 1.5

        if (dist_to_lines < half_lane_width).sum() == 1:  # If there's only one lane near, this is the lane to return
            current_lane = lanes_near[np.argmin(dist_to_lines)]

            return current_lane.id

        # If we get to this point, there are overlapping lanes, so need to find which lane matches pose rotation best
        indices_to_compare = np.where(dist_to_lines < half_lane_width)[0]  # Indices of potentially overlapping lanes

        # Initialize some variables to keep track of which lane to return in the below loop
        min_angle_difference_counter = 4.0
        current_lane = lanes_near[indices_to_compare[0]]

        # Loop through the potentially overlapping lanes
        for i in indices_to_compare:
            # Pull the line that we want to compare
            line = lines_near[i]

            # Store the index of the point on the line which is closest to the pose
            point_on_line_index = np.argmin(np.linalg.norm(line - pose.translation, axis=1))

            # Calculate the vector on that point on the line, if it's the last point then use the previous vector
            if point_on_line_index == len(line) - 1:
                vector_near_pose = line[point_on_line_index] - line[point_on_line_index - 1]
            else:
                vector_near_pose = line[point_on_line_index + 1] - line[point_on_line_index]

            # Find the difference between the line vector and the pose's forward direction vector
            angle_difference = np.arccos(
                np.dot(
                    pose_front_direction / np.linalg.norm(pose_front_direction),
                    vector_near_pose / np.linalg.norm(vector_near_pose),
                )
            )

            # If this lane is more aligned with the pose, store it
            if angle_difference < min_angle_difference_counter:
                min_angle_difference_counter = angle_difference
                current_lane = lanes_near[i]

        return current_lane.id
