from typing import List, Optional, Dict, Union
import numpy as np
import random
import logging

from pd.core import PdError

from paralleldomain import data_lab
from paralleldomain.data_lab.config.map import LaneSegment, Edge, RoadSegment
from paralleldomain.data_lab.behaviors.vehicle import DrivewayCreepBehavior
from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DBaseGeometry

from pd.internal.assets.asset_registry import DataVehicle, UtilVehicleTypes, ObjAssets
from pd.data_lab.generators.custom_simulation_agent import CustomVehicleSimulationAgent

logger = logging.getLogger(__name__)


class DrivewayCreepGenerator(data_lab.CustomAtomicGenerator):
    """
    Place vehicles that traverse along driveways over the course of the scene

    Args:
        behavior_duration: Length of time in seconds the vehicle takes to travel the length of their respective
            driveway
        number_of_vehicles: Number of vehicles to place on driveways in the world
        min_driveway_length: The minimum length of driveway in meters on which to place vehicles
        driveway_entry_probability: The probability that a placed vehicle will travel inwards on a driveway
            vs travelling outwards on the driveway
        vehicles_to_spawn: The list of vehicles to spawn in Driveways.  Each placed vehicle will be randomly selected
            from this list.  If no value is passed in, vehicles will be randomly selected from entire library of
            vehicles
        radius: The radius in meters around the ego vehicle in which driveways are selected for placement of driveway
            creeping vehicles
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
        # Store the values that we want other methods in the custom generator object to have access to
        self._behavior_duration = behavior_duration
        self._number_of_vehicles = number_of_vehicles
        self._min_driveway_length = min_driveway_length
        self._driveway_entry_probability = driveway_entry_probability
        self._radius = radius

        # If a list of vehicle names is provided, use a method implemented below to find the vehicle data of those
        # vehicles.  If no vehicle names are provided, use a method implemented below to find the vehicle data of the
        # default set of vehicles.
        self._vehicle_data = (
            self._get_vehicle_data_from_names(vehicles_to_spawn)
            if vehicles_to_spawn is not None
            else self._get_default_vehicle_data()
        )

    # This method takes a list of vehicle names are queries the Data Lab asset database to get information about the
    # physical properties of these vehicles
    def _get_vehicle_data_from_names(self, vehicle_list: List[str]) -> Dict[str, Union[str, float]]:
        # Select all the rows of the ObjAssets table in the asset database with a name that matches the list of
        # vehicle names we have provided.  From the OBjAssets table, we extract the name of the asset and its length
        vehicle_info = ObjAssets.select(ObjAssets.name, ObjAssets.length).where(ObjAssets.name.in_(vehicle_list))

        # Sort the extracted entries from the database into a list, storing a dictionary that holds the name of the
        # asset and its length
        vehicle_data = [{"name": o.name, "length": o.length} for o in vehicle_info]

        # Ensure there duplicate vehicle names in both the retrieved list and the requested list for robustness
        requested_vehicles = set(vehicle_list)
        found_vehicles = set(o["name"] for o in vehicle_data)

        # If no vehicles were found in the asset database, then raise an error
        if len(found_vehicles) == 0:
            raise PdError("No specified vehicles found in asset database - check selected assets")

        # If not all the requested vehicles were found, raise a warning but don't halt execution
        if len(found_vehicles) < len(requested_vehicles):
            missing_vehicles = requested_vehicles - found_vehicles
            logger.warning(
                f"Some specified vehicles could not be found in asset registry, continuing without - {missing_vehicles}"
            )

        # Return the vehicle data retrieved from the asset database for use later on
        return vehicle_data

    # This method is used to query the asset database for information about the physical properties of vehicle.
    # This method is used when a list of vehicles to spawn is not used.  Instead, this method will query properties
    # of all vehicles available in the database
    def _get_default_vehicle_data(self) -> Dict[str, Union[str, float]]:
        # If no vehicles are specified, we will use all vehicles in these categories
        selected_vehicle_types = ["COMPACT", "FULLSIZE", "MIDSIZE", "SUV"]

        # Create a database query that extracts all rows corresponding to vehicles in the above categories.
        # To do this, we join the ObjAssets Table, which contains information about all assets in Data Lab, to the
        # DataVehicle table, which contains information about vehicles in Data Lab. This join is performed by matching
        # the id column of ObjAssets with the asset_id of the DataVehicle table.

        # From there, we again join the UtilVehicleTypes by matching the vehicle_type_id of DataVehicle with the id of
        # UtilVehicleTypes.

        # Using these commands, we can now determine the vehicle type of all vehicles in the ObjAssets table.

        # Having done this, we then select all rows corresponding to vehicles in the above four vehicle types.
        valid_vehicles = (
            ObjAssets.select(ObjAssets.id, ObjAssets.name, ObjAssets.length, UtilVehicleTypes.name)
            .join(DataVehicle, on=(ObjAssets.id == DataVehicle.asset_id))
            .join(UtilVehicleTypes, on=(DataVehicle.vehicle_type_id == UtilVehicleTypes.id))
            .where(UtilVehicleTypes.name.in_(selected_vehicle_types))
        )

        # Sort the extracted entries from the database into a list, storing a dictionary that holds the name of the
        # asset and its length
        vehicle_dict = [{"name": o.name, "length": o.length} for o in valid_vehicles]

        # Return the vehicle data retrieved from the asset database for use later on
        return vehicle_dict

    # This private method is used to find driveways near the ego's position (remember the ego has already been placed
    # by the time this generator is invoked) that are suitable for spawning vehicles on
    def _find_valid_driveways(self, state: data_lab.ExtendedSimState, radius: float = 15.0) -> [LaneSegment]:
        # We create a search perimeter around the ego vehicle by accessing the sim state's ego pose.  This bounds object
        # defines the perimeter of the area we will look for driveways within
        bounds = BoundingBox2DBaseGeometry(
            x=state.ego_pose.translation[0] - radius,
            y=state.ego_pose.translation[1] - radius,
            width=2 * radius,
            height=2 * radius,
        )

        # We use the MapQuery object which is contained in the state, to get all RoadSegments within the search
        # perimeter we defined above
        potential_driveway_road_segments = state.map_query.get_road_segments_within_bounds(
            bounds=bounds, method="overlap"
        )

        # We keep only the road segments ids of RoadSegments that are driveways
        driveway_lane_segment_ids = [
            rs.lane_segments[0] for rs in potential_driveway_road_segments if rs.type is RoadSegment.RoadType.DRIVEWAY
        ]

        # Store both the actual lane segment and the reference line of that lane segement for driveway lane segments
        # we found above
        driveway_lane_information = [
            (
                state.map.lane_segments[int(ls_id)],
                state.map_query.edges[state.map.lane_segments[int(ls_id)].reference_line].as_polyline().to_numpy(),
            )
            for ls_id in driveway_lane_segment_ids
        ]

        # We retain driveway lane segements only if they are longer that the specified minimum driveway length
        driveway_lane_segments = [
            dl[0]
            for dl in driveway_lane_information
            if np.linalg.norm(dl[1][-1] - dl[1][0]) > 1.15 * self._min_driveway_length
        ]

        # If we cannot find enough driveways to spawn the specified number of cars, throw a warning but continue anyway
        if len(driveway_lane_segments) < self._number_of_vehicles:
            logger.warning(
                f"Insufficient valid driveways found - will spawn {len(driveway_lane_segments)} driveway"
                f" vehicles instead of requested {self._number_of_vehicles}"
            )
        # If we've found more driveways than required, pick only the number required
        elif len(driveway_lane_segments) > self._number_of_vehicles:
            driveway_lane_segments = driveway_lane_segments[: self._number_of_vehicles]

        # Return the driveway lane segments that have been picked for placing vehicles on
        return driveway_lane_segments

    # This method is the method responsible for spawning Custom Agents in the scene and assigning a behavior to them
    def create_agents_for_new_scene(
        self, state: data_lab.ExtendedSimState, random_seed: int
    ) -> List[data_lab.CustomSimulationAgent]:
        # Use the method implemented above to find driveways near the ego vehicle that are suitable for placing vehicles
        # on
        valid_driveways = self._find_valid_driveways(state=state, radius=self._radius)

        # Initialize and empty list which we can append spawned agents to
        agents = []

        # Loop through all the valid driveways which we found above
        for driveway in valid_driveways:
            # Get the reference line of the driveway we are spawning vehicles on
            reference_line = state.map.edges[driveway.reference_line].as_polyline().to_numpy()

            # Randomly choose a vehicle type to spawn from either the specified list of vehicles or the default list
            agent_to_spawn = random.choice(self._vehicle_data)

            # Determine whether the vehicle will be travelling inwards or outwards on the driveway
            if random.uniform(0.0, 1.0) > self._driveway_entry_probability:
                reference_line = np.flip(reference_line, axis=0)

            # Create a CustomVehicleSimulationAgent and assign the DrivewayCreepBehavior to that agent.  Full details on
            # the behavior can be found in its documentation.

            # When creating a custom agent, we specify its asset name to be the vehicle we randomly selected above.
            agent = CustomVehicleSimulationAgent(asset_name=agent_to_spawn["name"]).set_behavior(
                DrivewayCreepBehavior(
                    reference_line=reference_line,
                    behavior_duration=self._behavior_duration,
                    agent_length=agent_to_spawn["length"],
                )
            )

            # Append the created custom agent to the list of agents which this method will return
            agents.append(agent)

        # Return the list of created agents so that the Data Lab Instance can place them in the scenario
        return agents

    # The clone method returns a copy of the Custom Generator object and is required under the hood by Data Lab
    def clone(self):
        return DrivewayCreepGenerator(
            behavior_duration=self._behavior_duration,
            number_of_vehicles=self._number_of_vehicles,
            min_driveway_length=self._min_driveway_length,
            radius=self._radius,
            driveway_entry_probability=self._driveway_entry_probability,
            vehicles_to_spawn=[o["name"] for o in self._vehicle_data],
        )
