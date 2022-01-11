from typing import Dict, List, Optional

import numpy as np

from paralleldomain.common.umd.v1.UMD_pb2 import UniversalMap as ProtoUniversalMap
from paralleldomain.decoding.igraph_map_decoder import IGraphMapDecoder
from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DGeometry
from paralleldomain.model.map.area import Area
from paralleldomain.model.map.common import load_user_data
from paralleldomain.model.map.edge import Edge
from paralleldomain.model.map.junction import Junction
from paralleldomain.model.map.lane_segment import Direction, LaneSegment, LaneType, TurnType
from paralleldomain.model.map.road_segment import GroundType, RoadSegment, RoadType
from paralleldomain.model.type_aliases import AreaId, EdgeId, JunctionId, LaneSegmentId, RoadSegmentId
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.fsio import read_binary_message


class UMDDecoder(IGraphMapDecoder):
    def __init__(self, umd_file_path: AnyPath, **kwargs):
        super().__init__(**kwargs)
        self.umd_file_path = umd_file_path
        self._map_umd: Optional[ProtoUniversalMap] = None

    @property
    def map_umd(self) -> ProtoUniversalMap:
        if self._map_umd is None:
            self._map_umd = read_binary_message(obj=ProtoUniversalMap(), path=self.umd_file_path)
        return self._map_umd

    def decode_edges(self) -> Dict[EdgeId, Edge]:
        edges = dict()
        # for edge_id, edge_pb in self.map_umd.edges.items():
        #     edge = Edge(
        #         edge_id=edge_pb.id,
        #         closed=not (edge_pb.open),
        #         lines=[
        #             Line3DGeometry(
        #                 start=PointENU.from_proto(point=point_pair[0]), end=PointENU.from_proto(point=point_pair[1])
        #             )
        #             for point_pair in windowed(edge_pb.points, 2)
        #         ],
        #         road_marking=RoadMarking.from_proto(road_marking=road_marking) if road_marking is not None else None,
        #         user_data=load_user_data(edge_pb.user_data) if edge_pb.HasField("user_data") else {},
        #     )
        #     edges[edge.edge_id] = edge_id
        return edges

    def decode_road_segments(self) -> Dict[RoadSegmentId, RoadSegment]:
        segments = list()
        # edges = self.get_edges()
        # for road_segment_id, road_segment in self.map_umd.road_segments.items():
        #     # road_segment: ProtoRoadSegment = umd_map.road_segments[id]
        #     reference_line = self.map_umd.edges[road_segment.reference_line]
        #     reference_points = np.array([(p.x, p.y) for p in reference_line.points])
        #
        #     bounds = None
        #     if len(reference_points) > 0:
        #         x_min, y_min = np.min(reference_points, axis=0)
        #         x_max, y_max = np.max(reference_points, axis=0)
        #         bounds = BoundingBox2DGeometry(x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)
        #
        #     segment = RoadSegment(
        #         road_segment_id=road_segment.id,
        #         name=road_segment.name,
        #         predecessors=[ls_p for ls_p in road_segment.predecessors],
        #         successors=[ls_s for ls_s in road_segment.successors],
        #         reference_line=Edge.from_proto(edge=self.map_umd.edges[road_segment.reference_line])
        #         if road_segment.HasField("reference_line")
        #         else None,
        #         type=RoadType(road_segment.type),
        #         ground_type=GroundType(road_segment.ground_type),
        #         speed_limit=SpeedLimit.from_proto(speed_limit=road_segment.speed_limit)
        #         if road_segment.HasField("speed_limit")
        #         else None,
        #         junction_id=road_segment.junction_id,
        #         user_data=load_user_data(road_segment.user_data) if road_segment.HasField("user_data") else {},
        #         bounds=bounds,
        #     )
        #     segments.append(segment)
        return segments

    def decode_lane_segments(self) -> Dict[LaneSegmentId, LaneSegment]:
        segments = list()

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
                bounds = BoundingBox2DGeometry(x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)

            road_markings_by_edge_id = {rm.edge_id: rm for rm in self.map_umd.road_markings.values()}
            segment = LaneSegment(
                lane_segment_id=lane_segment.id,
                type=LaneType(lane_segment.type),
                direction=Direction(lane_segment.direction),
                left_edge=Edge.from_proto(
                    edge=self.map_umd.edges[lane_segment.left_edge],
                    road_marking=road_markings_by_edge_id[lane_segment.left_edge]
                    if lane_segment.left_edge in road_markings_by_edge_id
                    else None,
                ),
                right_edge=Edge.from_proto(
                    edge=self.map_umd.edges[lane_segment.right_edge],
                    road_marking=road_markings_by_edge_id[lane_segment.right_edge]
                    if lane_segment.right_edge in road_markings_by_edge_id
                    else None,
                ),
                reference_line=Edge.from_proto(edge=self.map_umd.edges[lane_segment.reference_line]),
                predecessors=[ls_p for ls_p in lane_segment.predecessors],
                successors=[ls_s for ls_s in lane_segment.successors],
                left_neighbor=lane_segment.left_neighbor,
                right_neighbor=lane_segment.right_neighbor,
                compass_angle=lane_segment.compass_angle,
                turn_angle=lane_segment.turn_angle,
                turn_type=TurnType(lane_segment.turn_type),
                user_data=load_user_data(lane_segment.user_data) if lane_segment.HasField("user_data") else {},
                bounds=bounds,
            )
            segments.append(segment)
        return segments

    def decode_junctions(self) -> Dict[JunctionId, Junction]:
        junctions = list()
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
                bounds = BoundingBox2DGeometry(x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)

            junc = Junction(
                junction_id=junction.id,
                lane_segments=[j_ls for j_ls in junction.lane_segments],
                road_segments=[j_rs for j_rs in junction.road_segments],
                signaled_intersection=junction.signaled_intersection
                if junction.HasField("signaled_intersection")
                else None,
                user_data=load_user_data(junction.user_data) if junction.HasField("user_data") else {},
                corners=[j_co for j_co in junction.corners],
                crosswalk_lanes=[j_cw for j_cw in junction.crosswalk_lanes],
                signed_intersection=junction.signed_intersection if junction.HasField("signed_intersection") else None,
                bounds=bounds,
            )
            junctions.append(junc)
        return junctions

    def decode_areas(self) -> Dict[AreaId, Area]:
        areas = list()
        for area_id, area in self.map_umd.areas.items():
            edge = self.map_umd.edges[area.edges[0]]
            edge_points = np.array([(p.x, p.y) for p in edge.points])

            bounds = None
            if len(edge_points) > 0:
                x_min, y_min = np.min(edge_points, axis=0)
                x_max, y_max = np.max(edge_points, axis=0)
                bounds = BoundingBox2DGeometry(x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)

            areas.append(Area(area_id=area_id, bounds=bounds))
        return areas
