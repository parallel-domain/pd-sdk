import logging
import random
from enum import Enum
from typing import List, Optional

from pd.data_lab.generators.custom_simulation_agent import CustomVehicleSimulationAgent, CustomPedestrianSimulationAgent
from pd.internal.assets.asset_registry import DataVehicle, UtilVehicleTypes, ObjAssets, DataCharacter

from paralleldomain import data_lab
from paralleldomain.data_lab import SensorRig
from paralleldomain.data_lab.config.map import LaneSegment
from paralleldomain.data_lab.generators.behavior import (
    SingleFrameVehicleBehavior,
    SingleFramePlaceNearEgoBehavior,
)

logger = logging.getLogger(__name__)


class SingleFrameVehicleBehaviorType(Enum):
    TRAFFIC = "TRAFFIC"
    PARKED = "PARKED"


class SingleFrameEgoGenerator(data_lab.CustomAtomicGenerator):
    """
    Place the ego vehicle in a manner which is compatible with single frame scenarios.  Each rendered frame will place
        the ego in a different place within the world

    Args:
        ego_asset_name: The name of the vehicle asset which should be used as the ego vehicle
        lane_type: The type of lane on which the ego vehicle should be placed
        random_seed: The integer to seed all random functions with, allowing scenario generation to be deterministic
        sim_capture_rate: Controls the frame rate of the scenario where "scenario_frame_rate = 100 / sim_capture_rate".
            For single frame scenarios, value of 10 is recommended
        sensor_rig:  The sensor rig which should be attached to the ego for rendering
    """

    def __init__(
        self,
        ego_asset_name: str,
        lane_type: LaneSegment.LaneType,
        random_seed: int,
        sensor_rig: SensorRig,
    ):
        self._ego_asset_name = ego_asset_name
        self._lane_type = lane_type
        self._random_seed = random_seed
        self._sensor_rig = sensor_rig

    def create_agents_for_new_scene(
        self, state: data_lab.ExtendedSimState, random_seed: int
    ) -> List[data_lab.CustomSimulationAgent]:
        agents = []

        agent = CustomVehicleSimulationAgent(
            asset_name=self._ego_asset_name,
            sensor_rig=self._sensor_rig,
        ).set_behaviour(
            SingleFrameVehicleBehavior(
                lane_type=self._lane_type,
                random_seed=self._random_seed,
            )
        )

        agents.append(agent)

        return agents

    def clone(self):
        return SingleFrameEgoGenerator(
            ego_asset_name=self._ego_asset_name,
            lane_type=self._lane_type,
            random_seed=self._random_seed,
            sensor_rig=self._sensor_rig,
        )


class SingleFrameNonEgoVehicleGenerator(data_lab.CustomAtomicGenerator):
    """
    Places vehicles in a manner which is compatible with single frame scenarios.  Each rendered frame will place
        the specified number of vehicles in a different place around the ego

    Args:
        number_of_vehicles: The number of non-ego vehicles which should be created and spawned
        random_seed: The integer to seed all random functions with, allowing scenario generation to be deterministic
        spawn_radius: The radius of the valid spawn region around the ego vehicle in which the non-ego vehicles should
            be spawned
        vehicle_behavior_type: Specifies whether the placed vehicles should be traffic or parked vehicles
    """

    def __init__(
        self,
        number_of_vehicles: int,
        random_seed: int,
        spawn_radius: float,
        vehicle_behavior_type: SingleFrameVehicleBehaviorType,
    ):
        self._number_of_vehicles = number_of_vehicles
        self._random_seed = random_seed
        self._spawn_radius = spawn_radius
        self._vehicle_behavior_type = vehicle_behavior_type

        self._random_state = random.Random(self._random_seed)

        all_vehicle_names = self._get_default_vehicle_names()
        self._vehicle_names = [self._random_state.choice(all_vehicle_names) for _ in range(self._number_of_vehicles)]

    def _get_default_vehicle_names(self) -> List[str]:
        selected_vehicle_types = ["COMPACT", "FULLSIZE", "MIDSIZE", "SUV"]

        valid_vehicles = (
            ObjAssets.select(ObjAssets.id, ObjAssets.name, ObjAssets.length, UtilVehicleTypes.name)
            .join(DataVehicle, on=(ObjAssets.id == DataVehicle.asset_id))
            .join(UtilVehicleTypes, on=(DataVehicle.vehicle_type_id == UtilVehicleTypes.id))
            .where(UtilVehicleTypes.name.in_(selected_vehicle_types))
        )

        vehicle_list = [o.name for o in valid_vehicles]

        return vehicle_list

    def create_agents_for_new_scene(
        self, state: data_lab.ExtendedSimState, random_seed: int
    ) -> List[data_lab.CustomSimulationAgent]:
        agents = []

        agent_seed = self._random_seed

        for vehicle in self._vehicle_names:
            agent_seed += 1

            agent = CustomVehicleSimulationAgent(asset_name=vehicle).set_behaviour(
                SingleFramePlaceNearEgoBehavior(
                    lane_type=LaneSegment.LaneType.PARKING_SPACE
                    if self._vehicle_behavior_type is SingleFrameVehicleBehaviorType.PARKED
                    else LaneSegment.LaneType.DRIVABLE,
                    random_seed=agent_seed,
                    spawn_radius=self._spawn_radius,
                    spawn_in_middle_of_lane=True
                    if self._vehicle_behavior_type is SingleFrameVehicleBehaviorType.PARKED
                    else False,
                )
            )

            agents.append(agent)

        return agents

    def clone(self):
        return SingleFrameNonEgoVehicleGenerator(
            random_seed=self._random_seed,
            number_of_vehicles=self._number_of_vehicles,
            spawn_radius=self._spawn_radius,
            vehicle_behavior_type=self._vehicle_behavior_type,
        )


class SingleFramePedestrianGenerator(data_lab.CustomAtomicGenerator):
    """
    Places pedestrians in a manner which is compatible with single frame scenarios.  Each rendered frame will place
        the specified number of pedestrians in a different place around the ego vehicle

    Args:
        random_seed: The integer to seed all random functions with, allowing scenario generation to be deterministic
        num_of_pedestrians: The number of pedestrians which should be placed in each frame
        spawn_radius: The radius of the valid spawn region around the ego vehicle in which the non-ego vehicles should
            be spawned
        max_lateral_offset: The maximum distance (in meters) that the placed agent will be laterally offset from the
            center of the randomly chosen lane segment
        max_rotation_offset_degrees: The maximum rotation (in degrees) that the placed agent will be rotated relative
            to the center line of the randomly chosen lane segment
        character_names: List of character asset names which should be spawned
    """

    def __init__(
        self,
        random_seed: int,
        num_of_pedestrians: int,
        spawn_radius: float,
        max_lateral_offset: float,
        max_rotation_offset_degrees: float = 30.0,
        character_names: Optional[List[str]] = None,
    ):
        self._num_of_pedestrians = num_of_pedestrians
        self._random_seed = random_seed
        self._spawn_radius = spawn_radius
        self._max_lateral_offset = max_lateral_offset
        self._max_rotation_offset_degrees = max_rotation_offset_degrees

        self._character_names = self._get_valid_character_names() if character_names is None else character_names
        self._random_state = random.Random(self._random_seed)

    def _get_valid_character_names(self) -> List[str]:
        valid_persons = DataCharacter.select(DataCharacter.name, DataCharacter.gender)
        person_list = [o.name for o in valid_persons if o.gender is not None and "police" not in o.name]

        return person_list

    def create_agents_for_new_scene(
        self, state: data_lab.ExtendedSimState, random_seed: int
    ) -> List[data_lab.CustomSimulationAgent]:
        agents = []

        for _ in range(self._num_of_pedestrians):
            character_to_spawn = self._random_state.choice(self._character_names)

            agent = CustomPedestrianSimulationAgent(asset_name=character_to_spawn).set_behaviour(
                behaviour=SingleFramePlaceNearEgoBehavior(
                    random_seed=self._random_seed,
                    spawn_radius=self._spawn_radius,
                    lane_type=LaneSegment.LaneType.SIDEWALK,
                    max_lateral_offset=1.0,
                    max_rotation_offset_degrees=self._max_rotation_offset_degrees,
                )
            )

            agents.append(agent)

        return agents

    def clone(self):
        return SingleFramePedestrianGenerator(
            num_of_pedestrians=self._num_of_pedestrians,
            character_names=self._character_names,
            random_seed=self._random_seed,
            spawn_radius=self._spawn_radius,
            max_lateral_offset=self._max_lateral_offset,
            max_rotation_offset_degrees=self._max_rotation_offset_degrees,
        )
