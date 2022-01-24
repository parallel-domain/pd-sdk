from json import JSONDecodeError
from typing import Any, Dict, Optional, TypeVar, Union

import numpy as np
import ujson
from more_itertools import windowed

from paralleldomain.common.umd.v1.UMD_pb2 import Edge as ProtoEdge
from paralleldomain.common.umd.v1.UMD_pb2 import Point_ENU as ProtoPointENU
from paralleldomain.common.umd.v1.UMD_pb2 import RoadMarking as ProtoRoadMarking
from paralleldomain.common.umd.v1.UMD_pb2 import UniversalMap as ProtoUniversalMap
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.map_decoder import MapDecoder
from paralleldomain.decoding.map_query.igraph_map_query import IGraphMapQuery
from paralleldomain.decoding.map_query.map_query import MapQuery
from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DBaseGeometry
from paralleldomain.model.geometry.point_3d import Point3DGeometry
from paralleldomain.model.geometry.polyline_3d import Line3DGeometry
from paralleldomain.model.map.area import Area
from paralleldomain.model.map.edge import Edge, RoadMarking, RoadMarkingColor, RoadMarkingType
from paralleldomain.model.map.map_components import (
    Direction,
    GroundType,
    Junction,
    LaneSegment,
    LaneType,
    RoadSegment,
    RoadType,
    SpeedLimit,
    SpeedUnits,
    TurnType,
)
from paralleldomain.model.type_aliases import AreaId, EdgeId, JunctionId, LaneSegmentId, RoadSegmentId, SceneName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_binary_message

T = TypeVar("T")


def load_user_data(user_data: T) -> Union[T, Dict[str, Any]]:
    try:
        return ujson.loads(user_data)
    except (ValueError, JSONDecodeError):
        return user_data


class UMDDecoder(MapDecoder):
    def __init__(self, umd_file_path: AnyPath, dataset_name: str, scene_name: SceneName, settings: DecoderSettings):
        super().__init__(dataset_name=dataset_name, scene_name=scene_name, settings=settings)
        self.umd_file_path = umd_file_path
        self._map_umd: Optional[ProtoUniversalMap] = None

    def _create_map_query(self) -> MapQuery:
        return IGraphMapQuery(
            road_segments=self.get_road_segments(),
            lane_segments=self.get_lane_segments(),
            junctions=self.get_junctions(),
            areas=self.get_areas(),
            edges=self.get_edges(),
        )

    @property
    def map_umd(self) -> ProtoUniversalMap:
        if self._map_umd is None:
            self._map_umd = read_binary_message(obj=ProtoUniversalMap(), path=self.umd_file_path)
        return self._map_umd

    def decode_edges(self) -> Dict[EdgeId, Edge]:
        def edge_from_proto(edge: ProtoEdge, road_marking: Optional[ProtoRoadMarking] = None):
            if road_marking is not None:
                road_marking = RoadMarking(
                    road_marking_id=road_marking.id,
                    edge_id=road_marking.id,
                    width=road_marking.width,
                    type=RoadMarkingType(road_marking.type) if road_marking.HasField("type") else None,
                    color=RoadMarkingColor(road_marking.color) if road_marking.HasField("color") else None,
                )

            return Edge(
                edge_id=edge.id,
                closed=not (edge.open),
                lines=[
                    Line3DGeometry(
                        start=Point3DGeometry(x=point_pair[0].x, y=point_pair[0].y, z=point_pair[0].z),
                        end=Point3DGeometry(x=point_pair[1].x, y=point_pair[1].y, z=point_pair[1].z),
                    )
                    for point_pair in windowed(edge.points, 2)
                ],
                road_marking=road_marking,
                user_data=load_user_data(edge.user_data) if edge.HasField("user_data") else {},
            )

        road_markings_by_edge_id = {rm.edge_id: rm for rm in self.map_umd.road_markings.values()}

        edges = dict()
        for edge_id, umd_edge in self.map_umd.edges.items():
            road_marking = None
            if edge_id in road_markings_by_edge_id:
                road_marking = road_markings_by_edge_id[edge_id]
            edge = edge_from_proto(edge=umd_edge, road_marking=road_marking)
            edges[edge.edge_id] = edge

        return edges

    def decode_road_segments(self) -> Dict[RoadSegmentId, RoadSegment]:
        segments = dict()

        for road_segment_id, road_segment in self.map_umd.road_segments.items():
            # road_segment: ProtoRoadSegment = umd_map.road_segments[id]
            reference_line = self.map_umd.edges[road_segment.reference_line]
            reference_points = np.array([(p.x, p.y) for p in reference_line.points])

            bounds = None
            if len(reference_points) > 0:
                x_min, y_min = np.min(reference_points, axis=0)
                x_max, y_max = np.max(reference_points, axis=0)
                bounds = BoundingBox2DBaseGeometry[float](x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)

            speed_limit = road_segment.speed_limit
            segment = RoadSegment(
                road_segment_id=road_segment.id,
                name=road_segment.name,
                predecessor_ids=[ls_p for ls_p in road_segment.predecessors],
                successor_ids=[ls_s for ls_s in road_segment.successors],
                reference_line_id=road_segment.reference_line if road_segment.HasField("reference_line") else None,
                road_type=RoadType(road_segment.type),
                ground_type=GroundType(road_segment.ground_type),
                speed_limit=SpeedLimit(speed=speed_limit.speed, units=SpeedUnits(speed_limit.units))
                if road_segment.HasField("speed_limit")
                else None,
                junction_id=road_segment.junction_id,
                user_data=load_user_data(road_segment.user_data) if road_segment.HasField("user_data") else {},
                bounds=bounds,
                map_query=self.map_query,
            )
            segments[segment.road_segment_id] = segment
        return segments

    def decode_lane_segments(self) -> Dict[LaneSegmentId, LaneSegment]:
        segments = dict()
        for lane_segment_id, lane_segment in self.map_umd.lane_segments.items():

            reference_line = self.map_umd.edges[lane_segment.reference_line]
            reference_points = np.array([(p.x, p.y) for p in reference_line.points])

            left_edge = self.map_umd.edges[lane_segment.left_edge]
            left_points = np.array([(p.x, p.y) for p in left_edge.points])

            right_edge = self.map_umd.edges[lane_segment.right_edge]
            right_points = np.array([(p.x, p.y) for p in right_edge.points])

            all_points = np.vstack([reference_points, left_points, right_points])

            bounds = None
            if len(all_points) > 0:
                x_min, y_min = np.min(all_points, axis=0)
                x_max, y_max = np.max(all_points, axis=0)
                bounds = BoundingBox2DBaseGeometry[float](x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)

            segment = LaneSegment(
                lane_segment_id=lane_segment.id,
                map_query=self.map_query,
                lane_type=LaneType(lane_segment.type),
                direction=Direction(lane_segment.direction),
                left_edge_id=lane_segment.left_edge,
                right_edge_id=lane_segment.right_edge,
                reference_line_id=lane_segment.reference_line,
                predecessor_ids=[ls_p for ls_p in lane_segment.predecessors],
                successor_ids=[ls_s for ls_s in lane_segment.successors],
                left_neighbor_id=lane_segment.left_neighbor,
                right_neighbor_id=lane_segment.right_neighbor,
                compass_angle=lane_segment.compass_angle,
                turn_angle=lane_segment.turn_angle,
                turn_type=TurnType(lane_segment.turn_type),
                user_data=load_user_data(lane_segment.user_data) if lane_segment.HasField("user_data") else {},
                bounds=bounds,
            )
            segments[segment.lane_segment_id] = segment
        return segments

    def decode_junctions(self) -> Dict[JunctionId, Junction]:
        junctions = dict()
        for junction_id, junction in self.map_umd.junctions.items():
            corner_points = np.empty(shape=(0, 2))
            for corner in junction.corners:
                corner_edge = self.map_umd.edges[corner]
                corner_points = np.vstack(
                    [
                        corner_points,
                        np.array([(p.x, p.y) for p in corner_edge.points]),
                    ]
                )

            bounds = None
            if len(corner_points) > 0:
                x_min, y_min = np.min(corner_points, axis=0)
                x_max, y_max = np.max(corner_points, axis=0)
                bounds = BoundingBox2DBaseGeometry[float](x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)

            junc = Junction(
                junction_id=junction.id,
                lane_segment_ids=[j_ls for j_ls in junction.lane_segments],
                road_segment_ids=[j_rs for j_rs in junction.road_segments],
                signaled_intersection=junction.signaled_intersection
                if junction.HasField("signaled_intersection")
                else None,
                user_data=load_user_data(junction.user_data) if junction.HasField("user_data") else {},
                corner_ids=[j_co for j_co in junction.corners],
                crosswalk_lane_ids=[j_cw for j_cw in junction.crosswalk_lanes],
                signed_intersection=junction.signed_intersection if junction.HasField("signed_intersection") else None,
                bounds=bounds,
                map_query=self.map_query,
            )
            junctions[junc.junction_id] = junc
        return junctions

    def decode_areas(self) -> Dict[AreaId, Area]:
        areas = dict()
        for area_id, area in self.map_umd.areas.items():
            edge = self.map_umd.edges[area.edges[0]]
            edge_points = np.array([(p.x, p.y) for p in edge.points])

            bounds = None
            if len(edge_points) > 0:
                x_min, y_min = np.min(edge_points, axis=0)
                x_max, y_max = np.max(edge_points, axis=0)
                bounds = BoundingBox2DBaseGeometry[float](x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)

            areas[area_id] = Area(area_id=area_id, bounds=bounds)
        return areas
