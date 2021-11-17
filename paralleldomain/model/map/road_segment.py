from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional

from paralleldomain.common.umd.v1.UMD_pb2 import RoadSegment as ProtoRoadSegment
from paralleldomain.common.umd.v1.UMD_pb2 import SpeedLimit as ProtoSpeedLimit
from paralleldomain.common.umd.v1.UMD_pb2 import UniversalMap as ProtoUniversalMap
from paralleldomain.model.map.common import load_user_data
from paralleldomain.model.map.edge import Edge


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
    PARKING_AISLE = 14


class GroundType(IntEnum):
    GROUND = 0
    BRIDGE = 1
    TUNNEL = 2


class SpeedUnits(IntEnum):
    MILES_PER_HOUR = 0
    KILOMETERS_PER_HOUR = 1


@dataclass
class SpeedLimit:
    speed: int
    units: SpeedUnits

    @classmethod
    def from_proto(cls, speed_limit: ProtoSpeedLimit) -> "SpeedLimit":
        return cls(speed=speed_limit.speed, units=SpeedUnits(speed_limit.units))


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
