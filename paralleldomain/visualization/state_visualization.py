from collections import defaultdict
from contextlib import suppress
from typing import Any, Dict, Optional

import numpy as np
import rerun as rr
import ujson
from pd.assets import asset_pivot_point_to_sim_geometric_center_offset
from pd.internal.assets.asset_registry import ObjAssets
from pd.state import ModelAgent, State, VehicleAgent

from paralleldomain.data_lab.config.map import Area, UniversalMap
from paralleldomain.utilities.coordinate_system import SIM_TO_INTERNAL
from paralleldomain.utilities.transformation import Transformation
from paralleldomain.visualization import initialize_viewer

EXCLUDE_AREA_TYPES = {
    Area.AreaType.PARKING_SPACE,
    Area.AreaType.POWER,
    Area.AreaType.ZONE_COMMERCIAL,
    Area.AreaType.ZONE_GREEN,
    Area.AreaType.ZONE_INDUSTRIAL,
    Area.AreaType.ZONE_RETAIL,
}

STROKE_AUTO = np.finfo(np.float32).max


def parse_user_data(user_data: Any) -> Dict[str, Any]:
    user_data_out = {}
    with suppress(ujson.JSONDecodeError):
        user_data_out.update(ujson.loads(user_data))
    return user_data_out


def show_map(
    umd_map: UniversalMap,
    recording_id: Optional[str],
    application_id: str,
    entity_root: str = "world",
):
    initialize_viewer(entity_root=entity_root, timeless=False, recording_id=recording_id, application_id=application_id)
    map_entity = f"{entity_root}/sim/map"
    rr.log(
        map_entity,
        rr.ViewCoordinates(xyz=rr.components.ViewCoordinates(coordinates=[5, 4, 1])),  # FLU
        timeless=True,
    )

    edges_by_type = defaultdict(dict)
    reference_lines_by_type = defaultdict(dict)

    lane_segment_type_map = None
    for lane_segment_id, lane_segment in umd_map.lane_segments.items():
        if lane_segment_type_map is None:  # initialize dictionary once
            lane_segment_type_map = {v: k for k, v in lane_segment.LaneType.__dict__.items() if not k.startswith("_")}

        lane_segment_type = lane_segment_type_map[lane_segment.type]

        left_edge = umd_map.edges[lane_segment.left_edge] if lane_segment.left_edge > 0 else None
        right_edge = umd_map.edges[lane_segment.right_edge] if lane_segment.right_edge > 0 else None

        if left_edge is not None:
            left_edge_np: np.ndarray = left_edge.as_polyline().to_numpy()
            left_edge_np = SIM_TO_INTERNAL @ left_edge_np

            edges_by_type[lane_segment_type][left_edge.id] = [
                (  # use list so we can expand data if needed
                    left_edge_np if left_edge.open else np.vstack([left_edge_np, left_edge_np[0]])
                ),
            ]

        if right_edge is not None:
            right_edge_np: np.ndarray = right_edge.as_polyline().to_numpy()
            right_edge_np = SIM_TO_INTERNAL @ right_edge_np

            edges_by_type[lane_segment_type][right_edge.id] = [
                right_edge_np if right_edge.open else np.vstack([right_edge_np, right_edge_np[0]])
            ]

        reference_line = umd_map.edges[lane_segment.reference_line] if lane_segment.reference_line > 0 else None

        if reference_line is not None:
            reference_line_np: np.ndarray = reference_line.as_polyline().to_numpy()
            reference_line_np = SIM_TO_INTERNAL @ reference_line_np

            reference_lines_by_type[lane_segment_type][reference_line.id] = [
                reference_line_np if reference_line.open else np.vstack([reference_line_np, reference_line_np[0]]),
            ]

    for ls_type, edges in edges_by_type.items():
        rr.log(
            f"{map_entity}/lane_segments/{ls_type}/edges",
            rr.LineStrips3D(
                strips=[v[0] for k, v in sorted(edges.items(), key=lambda x: x[0])],
                instance_keys=sorted(edges.keys()),
                colors=[0, 150, 255],
            ),
            timeless=True,
        )

    for ls_type, edges in reference_lines_by_type.items():
        rr.log(
            f"{map_entity}/lane_segments/{ls_type}/reference_lines",
            rr.LineStrips3D(
                strips=[v[0] for k, v in sorted(edges.items(), key=lambda x: x[0])],
                instance_keys=sorted(edges.keys()),
                colors=[255, 95, 31],
            ),
            timeless=True,
        )

    areas_by_type = defaultdict(dict)

    area_type_map = None
    for area_id, area in umd_map.areas.items():
        if area.type in EXCLUDE_AREA_TYPES:
            continue
        if area_type_map is None:  # initialize dictionary once
            area_type_map = {v: k for k, v in area.AreaType.__dict__.items() if not k.startswith("_")}

        area_type = area_type_map[area.type]
        for edge_id in area.edges:
            edge = umd_map.edges[edge_id]
            edge_np: np.ndarray = edge.as_polyline().to_numpy()
            edge_np = SIM_TO_INTERNAL @ edge_np

            areas_by_type[area_type][edge.id] = [
                edge_np if edge.open else np.vstack([edge_np, edge_np[0]]),
            ]

    for area_type, edges in areas_by_type.items():
        rr.log(
            f"{map_entity}/areas/{area_type}/edges",
            rr.LineStrips3D(
                strips=[v[0] for k, v in sorted(edges.items(), key=lambda x: x[0])],
                instance_keys=sorted(edges.keys()),
                colors=[250, 200, 152],
            ),
            timeless=True,
        )

    junctions_by_id = {}

    for junction_id, junction in umd_map.junctions.items():
        for edge_id in junction.corners:
            edge = umd_map.edges[edge_id]
            edge_np = edge.as_polyline().to_numpy()
            edge_np = SIM_TO_INTERNAL @ edge_np

            junctions_by_id[edge.id] = [
                edge_np if edge.open else np.vstack([edge_np, edge_np[0]]),
            ]

    rr.log(
        f"{map_entity}/lane_segments/JUNCTION/edges",
        rr.LineStrips3D(
            strips=[v[0] for k, v in sorted(junctions_by_id.items(), key=lambda x: x[0])],
            instance_keys=sorted(junctions_by_id.keys()),
            colors=[0, 150, 255],
        ),
        timeless=True,
    )


def show_agents(sim_state: State, recording_id: Optional[str], application_id: str, entity_root: str = "world"):
    initialize_viewer(entity_root=entity_root, timeless=False, recording_id=recording_id, application_id=application_id)
    agent_root = f"{entity_root}/sim/agents"
    rr.log(
        agent_root,
        rr.ViewCoordinates(xyz=rr.components.ViewCoordinates(coordinates=[5, 4, 1])),  # FLU
        timeless=True,
    )
    rr.set_time_seconds(timeline="seconds", seconds=sim_state.simulation_time_sec)

    agents = [a for a in sim_state.agents if isinstance(a, ModelAgent) or isinstance(a, VehicleAgent)]

    agent_names = [a.asset_name if isinstance(a, ModelAgent) else a.vehicle_type for a in agents]
    assets = ObjAssets.select(
        ObjAssets.name,
        ObjAssets.width,
        ObjAssets.length,
        ObjAssets.height,
        ObjAssets.bbox_min_x,
        ObjAssets.bbox_max_x,
        ObjAssets.bbox_min_y,
        ObjAssets.bbox_max_y,
        ObjAssets.bbox_min_z,
        ObjAssets.bbox_max_z,
    ).where(ObjAssets.name.in_(agent_names))
    asset_dimensions_by_name = {obj.name: (obj.width, obj.length, obj.height) for obj in assets}
    asset_center_offsets_by_name = {  # need to calculate offset from asset's pivot point to geometric center
        obj.name: asset_pivot_point_to_sim_geometric_center_offset(
            min_x=obj.bbox_min_x,
            max_x=obj.bbox_max_x,
            min_y=obj.bbox_min_y,
            max_y=obj.bbox_max_y,
            min_z=obj.bbox_min_z,
            max_z=obj.bbox_max_z,
        )
        for obj in assets
    }

    for agent in agents:
        metadata = {}
        if isinstance(agent, ModelAgent):
            metadata = {k: getattr(agent, k) for k in ("asset_name",)}
        elif isinstance(agent, VehicleAgent):
            metadata = {
                k: getattr(agent, k) for k in ("vehicle_type", "vehicle_color", "vehicle_accessory", "is_parked")
            }

        half_size = list(
            map(
                lambda x: x / 2,
                asset_dimensions_by_name.get(
                    agent.asset_name if isinstance(agent, ModelAgent) else agent.vehicle_type,
                    (1.0, 1.0, 1.0),  # default if asset was not found
                ),
            )
        )

        pose = Transformation.from_transformation_matrix(mat=agent.pose, approximate_orthogonal=True)

        translation_offset = np.asarray(
            asset_center_offsets_by_name.get(
                agent.asset_name if isinstance(agent, ModelAgent) else agent.vehicle_type, (0.0, 0.0, 0.0)
            )
        )
        translation_offset_rotated = pose.rotation @ translation_offset

        pose = Transformation(quaternion=pose.quaternion, translation=pose.translation + translation_offset_rotated)
        pose = SIM_TO_INTERNAL @ pose

        rr.log(
            f"{agent_root}/{agent.id}",
            rr.Boxes3D(
                half_sizes=[half_size],
                centers=[pose.translation],
                rotations=[
                    [
                        pose.quaternion.x,
                        pose.quaternion.y,
                        pose.quaternion.z,
                        pose.quaternion.w,
                    ]
                ],
            ),
            rr.AnyValues(**{k: [v] for k, v in metadata.items()}),
            timeless=False,
        )
