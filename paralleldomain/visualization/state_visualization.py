from contextlib import suppress
from typing import Any, Dict

import numpy as np
import rerun as rr
import ujson
from pd.internal.assets.asset_registry import ObjAssets
from pd.state import ModelAgent, State, VehicleAgent

from paralleldomain.data_lab.config.map import UniversalMap
from paralleldomain.utilities.coordinate_system import INTERNAL_COORDINATE_SYSTEM, SIM_COORDINATE_SYSTEM
from paralleldomain.utilities.transformation import Transformation
from paralleldomain.visualization.initialization import initialize_viewer

SIM_TO_INTERNAL = SIM_COORDINATE_SYSTEM > INTERNAL_COORDINATE_SYSTEM


def parse_user_data(user_data: Any) -> Dict[str, Any]:
    user_data_out = {}
    with suppress(ujson.JSONDecodeError):
        user_data_out.update(ujson.loads(user_data))
    return user_data_out


def show_map(umd_map: UniversalMap, entity_root: str = "world"):
    initialize_viewer(entity_root=entity_root, timeless=False)
    map_entity = f"{entity_root}/sim/map"
    rr.log_view_coordinates(entity_path=map_entity, xyz="FLU", timeless=True)  # X=Right, Y=Down, Z=Forward
    for lane_segment_id, lane_segment in umd_map.lane_segments.items():
        left_edge = umd_map.edges[lane_segment.left_edge] if lane_segment.left_edge > 0 else None
        right_edge = umd_map.edges[lane_segment.right_edge] if lane_segment.right_edge > 0 else None

        left_edge_np: np.ndarray = left_edge.as_polyline().to_numpy()
        left_edge_np = SIM_TO_INTERNAL @ left_edge_np

        rr.log_line_strip(
            entity_path=f"{map_entity}/lane_segments/{lane_segment.id}/left_edge/{left_edge.id}",
            positions=left_edge_np if left_edge.open else np.vstack([left_edge_np, left_edge_np[0]]),
            timeless=True,
            ext={
                k: ujson.dumps(v)  # cast into a string so avoiding issue with list or dict items
                for k, v in parse_user_data(left_edge.user_data).items()
            },  # metadata
            color=[0, 150, 255],
        )
        right_edge_np: np.ndarray = right_edge.as_polyline().to_numpy()
        right_edge_np = SIM_TO_INTERNAL @ right_edge_np
        rr.log_line_strip(
            entity_path=f"{map_entity}/lane_segments/{lane_segment.id}/right_edge/{right_edge.id}",
            positions=right_edge_np if right_edge.open else np.vstack([right_edge_np, right_edge_np[0]]),
            timeless=True,
            ext={
                k: ujson.dumps(v)  # cast into a string so avoiding issue with list or dict items
                for k, v in parse_user_data(right_edge.user_data).items()
            },  # metadata
            color=[0, 150, 255],
        )

        reference_line = umd_map.edges[lane_segment.reference_line] if lane_segment.reference_line > 0 else None
        reference_line_np: np.ndarray = reference_line.as_polyline().to_numpy()
        reference_line_np = SIM_TO_INTERNAL @ reference_line_np
        rr.log_line_strip(
            entity_path=f"{map_entity}/lane_segments/{lane_segment.id}/reference_line/{reference_line.id}",
            positions=(
                reference_line_np if reference_line.open else np.vstack([reference_line_np, reference_line_np[0]])
            ),
            timeless=True,
            ext={
                k: ujson.dumps(v)  # cast into a string so avoiding issue with list or dict items
                for k, v in parse_user_data(reference_line.user_data).items()
            },  # metadata
            color=[255, 95, 31],
        )

    # # optionally create annotations
    # for road_segment_id, road_segment in umd_map.road_segments.items():
    #     ...

    for area_id, area in umd_map.areas.items():
        for edge_id in area.edges:
            edge = umd_map.edges[edge_id]
            edge_np: np.ndarray = edge.as_polyline().to_numpy()
            edge_np = SIM_TO_INTERNAL @ edge_np

            rr.log_line_strip(
                entity_path=f"{map_entity}/areas/{area.id}/edges/{edge.id}",
                positions=edge_np if edge.open else np.vstack([edge_np, edge_np[0]]),
                timeless=True,
                ext={
                    k: ujson.dumps(v)  # cast into a string so avoiding issue with list or dict items
                    for k, v in {
                        **parse_user_data(edge.user_data),
                        **parse_user_data(area.user_data),
                        "type": area.type,
                        "height": area.height,
                        "floors": area.floors,
                    }.items()
                },  # metadata
                color=[250, 200, 152],
            )

    for junction_id, junction in umd_map.junctions.items():
        for edge_id in junction.corners:
            edge = umd_map.edges[edge_id]
            edge_np = edge.as_polyline().to_numpy()
            edge_np = SIM_TO_INTERNAL @ edge_np

            rr.log_line_strip(
                entity_path=f"{map_entity}/junctions/{junction.id}/corners/{edge.id}",
                positions=edge_np if edge.open else np.vstack([edge_np, edge_np[0]]),
                timeless=True,
                ext={
                    k: ujson.dumps(v)  # cast into a string so avoiding issue with list or dict items
                    for k, v in {
                        **parse_user_data(edge.user_data),
                        **parse_user_data(junction.user_data),
                    }.items()
                },  # metadata
                color=[0, 150, 255],
            )


def show_agents(sim_state: State, entity_root: str = "world"):
    initialize_viewer(entity_root=entity_root, timeless=False)
    agent_root = f"{entity_root}/sim/agents"
    rr.log_view_coordinates(entity_path=agent_root, xyz="FLU", timeless=True)  # X=Right, Y=Down, Z=Forward
    rr.set_time_seconds(timeline="time", seconds=sim_state.simulation_time_sec)

    agents = [a for a in sim_state.agents if isinstance(a, ModelAgent) or isinstance(a, VehicleAgent)]

    agent_names = [a.asset_name if isinstance(a, ModelAgent) else a.vehicle_type for a in agents]
    assets = ObjAssets.select(ObjAssets.name, ObjAssets.width, ObjAssets.length, ObjAssets.height).where(
        ObjAssets.name.in_(agent_names)
    )
    asset_dimensions_by_name = {obj.name: (obj.width, obj.length, obj.height) for obj in assets}

    for agent in agents:
        metadata = {}
        if isinstance(agent, ModelAgent):
            metadata = {k: getattr(agent, k) for k in ("asset_name",)}
        elif isinstance(agent, VehicleAgent):
            metadata = {
                k: getattr(agent, k) for k in ("vehicle_type", "vehicle_color", "vehicle_accessory", "is_parked")
            }

        pose: Transformation = Transformation.from_transformation_matrix(mat=agent.pose, approximate_orthogonal=True)
        pose: Transformation = SIM_TO_INTERNAL @ pose

        half_size = list(
            map(
                lambda x: x / 2,
                asset_dimensions_by_name.get(
                    agent.asset_name if isinstance(agent, ModelAgent) else agent.vehicle_type,
                    (1.0, 1.0, 1.0),  # default if asset was not found
                ),
            )
        )

        rr.log_obb(
            entity_path=f"{agent_root}/{agent.id}",
            half_size=half_size,
            position=pose.translation + np.array([0.0, 0.0, half_size[2]]),
            rotation_q=[
                pose.quaternion.x,
                pose.quaternion.y,
                pose.quaternion.z,
                pose.quaternion.w,
            ],
            ext=metadata,
            timeless=False,
        )
