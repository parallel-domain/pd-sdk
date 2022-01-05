from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from paralleldomain.common.umd.v1.UMD_pb2 import Junction as ProtoJunction
from paralleldomain.common.umd.v1.UMD_pb2 import UniversalMap as ProtoUniversalMap
from paralleldomain.model.map.common import load_user_data


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
