from typing import List, Optional, Dict, Union
import numpy as np
import random
import logging

from pd.core import PdError

from paralleldomain import data_lab
from paralleldomain.data_lab.config.map import LaneSegment, Edge, RoadSegment
from paralleldomain.data_lab.generators.behavior import DrivewayCreepBehavior
from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DBaseGeometry

from pd.internal.assets.asset_registry import DataVehicle, UtilVehicleTypes, ObjAssets
from pd.data_lab.generators.custom_simulation_agent import CustomVehicleSimulationAgent

logger = logging.getLogger(__name__)


class DrivewayCreepGenerator(data_lab.CustomAtomicGenerator):
    """
    Places vehicle into driveways within the world.
    These vehicles will traverse along the driveway over the course of the scene

    Args:
        behavior_duration:
              Description:
                  Length of time in seconds in which the vehicle should travel from the start to end of the driveway
              Required:
                  Yes
        number_of_vehicles:
              Description:
                  Number of vehicles to place in the world
              Range:
                  Greater than 0
              Required:
                  No, will default to 3
        min_driveway_length:
              Description:
                  The minimum length of driveway in meters on which to place vehicles
              Required:
                  No, will default to 6.0 meters
        driveway_entry_probability:
              Description:
                  The probability that a placed vehicle will travel inwards on a driveway vs travelling outwards
              Range:
                  Between 0 and 1.0
              Required:
                  No, will default to 0.5
        vehicles_to_spawn:
              Description:
                  The list of vehicles to spawn in Driveways.
                  Each placed vehicle will be randomly selected from this list
              Range:
                  Each name must match a vehicle in the asset registry
              Required:
                  No, will default to randomly selecting a vehicle from the following vehicle categories
                  ("COMPACT", "FULLSIZE", "MIDSIZE", "SUV")
        radius:
              Description:
                  The radius in meters around the ego vehicle in which driveways are selected
                  for placement of driveway creeping vehicles
              Range:
                  Greater than 0
              Required:
                  No, will default to 15 meters
    """

    def __init__(
        self,
        behavior_duration: float,
        number_of_vehicles: int = 3,
        min_driveway_length: float = 6.0,
        driveway_entry_probability: float = 0.5,
        vehicles_to_spawn: Optional[List[str]] = None,
        radius: float = 15.0,
    ):
        self._behavior_duration = behavior_duration
        self._number_of_vehicles = number_of_vehicles
        self._min_driveway_length = min_driveway_length
        self._driveway_entry_probability = driveway_entry_probability
        self._radius = radius

        self._vehicle_data = (
            self._get_vehicle_data_from_names(vehicles_to_spawn)
            if vehicles_to_spawn is not None
            else self._get_default_vehicle_data()
        )

    def _get_vehicle_data_from_names(self, vehicle_list: List[str]) -> Dict[str, Union[str, float]]:
        vehicle_info = ObjAssets.select(ObjAssets.name, ObjAssets.length).where(ObjAssets.name.in_(vehicle_list))

        vehicle_data = [{"name": o.name, "length": o.length} for o in vehicle_info]

        requested_vehicles = set(vehicle_list)
        found_vehicles = set(o["name"] for o in vehicle_data)

        if len(found_vehicles) == 0:
            raise PdError("No specified vehicles found in asset registry - check selected assets")
        if len(found_vehicles) < len(requested_vehicles):
            missing_vehicles = requested_vehicles - found_vehicles
            logger.warning(
                f"Some specified vehicles could not be found in asset registry, continuing without - {missing_vehicles}"
            )

        return vehicle_data

    def _find_valid_driveways(self, state: data_lab.ExtendedSimState, radius: float = 15.0) -> [LaneSegment]:
        bounds = BoundingBox2DBaseGeometry(
            x=state.ego_pose.translation[0] - radius,
            y=state.ego_pose.translation[1] - radius,
            width=2 * radius,
            height=2 * radius,
        )

        potential_driveway_road_segments = state.map_query.get_road_segments_within_bounds(
            bounds=bounds, method="overlap"
        )

        driveway_lane_segment_ids = [
            rs.lane_segments[0] for rs in potential_driveway_road_segments if rs.type is RoadSegment.RoadType.DRIVEWAY
        ]

        driveway_lane_information = [
            (
                state.map.lane_segments[int(ls_id)],
                state.map_query.edges[state.map.lane_segments[int(ls_id)].reference_line].as_polyline().to_numpy(),
            )
            for ls_id in driveway_lane_segment_ids
        ]

        driveway_lane_segments = [
            dl[0]
            for dl in driveway_lane_information
            if np.linalg.norm(dl[1][-1] - dl[1][0]) > 1.15 * self._min_driveway_length
        ]

        if len(driveway_lane_segments) < self._number_of_vehicles:
            logger.warning(
                f"Insufficient valid driveways found - will spawn {len(driveway_lane_segments)} driveway"
                f" vehicles instead of requested {self._number_of_vehicles}"
            )
        elif len(driveway_lane_segments) > self._number_of_vehicles:
            driveway_lane_segments = driveway_lane_segments[: self._number_of_vehicles]

        return driveway_lane_segments

    def _get_default_vehicle_data(self) -> Dict[str, Union[str, float]]:
        selected_vehicle_types = ["COMPACT", "FULLSIZE", "MIDSIZE", "SUV"]

        valid_vehicles = (
            ObjAssets.select(ObjAssets.id, ObjAssets.name, ObjAssets.length, UtilVehicleTypes.name)
            .join(DataVehicle, on=(ObjAssets.id == DataVehicle.asset_id))
            .join(UtilVehicleTypes, on=(DataVehicle.vehicle_type_id == UtilVehicleTypes.id))
            .where(UtilVehicleTypes.name.in_(selected_vehicle_types))
        )

        vehicle_dict = [{"name": o.name, "length": o.length} for o in valid_vehicles]

        return vehicle_dict

    def create_agents_for_new_scene(
        self, state: data_lab.ExtendedSimState, random_seed: int
    ) -> List[data_lab.CustomSimulationAgent]:
        # Search for valid driveways near the scene
        valid_driveways = self._find_valid_driveways(state=state, radius=self._radius)

        agents = []
        for driveway in valid_driveways:
            reference_line = Edge(proto=state.map_query.edges[driveway.reference_line].proto).as_polyline().to_numpy()

            agent_to_spawn = random.choice(self._vehicle_data)

            if random.uniform(0.0, 1.0) > self._driveway_entry_probability:
                reference_line = np.flip(reference_line, axis=0)

            agent = CustomVehicleSimulationAgent(asset_name=agent_to_spawn["name"]).set_behaviour(
                DrivewayCreepBehavior(
                    reference_line=reference_line,
                    behavior_duration=self._behavior_duration,
                    agent_length=agent_to_spawn["length"],
                )
            )
            agents.append(agent)

        return agents

    def clone(self):
        return DrivewayCreepGenerator(
            behavior_duration=self._behavior_duration,
            number_of_vehicles=self._number_of_vehicles,
            min_driveway_length=self._min_driveway_length,
            radius=self._radius,
            driveway_entry_probability=self._driveway_entry_probability,
            vehicles_to_spawn=[o["name"] for o in self._vehicle_data],
        )
