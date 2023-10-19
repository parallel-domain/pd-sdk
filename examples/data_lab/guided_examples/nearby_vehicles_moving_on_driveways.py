import logging
import random
from typing import Tuple

import numpy as np
from pd.core.errors import PdError
from pd.data_lab import Scenario, ScenarioCreator, ScenarioSource
from pd.data_lab.context import load_map
from pd.data_lab.scenario import Lighting
from pd.internal.assets.asset_registry import DataLightingSublevels

from paralleldomain.data_lab import Location, SensorConfig, SensorRig, preview_scenario
from paralleldomain.data_lab.config.map import LaneSegment, MapQuery, RoadSegment
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, SensorExtrinsic
from paralleldomain.data_lab.config.types import Float3
from paralleldomain.data_lab.generators.driveway import DrivewayCreepGenerator
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.peripherals import VehiclePeripheral
from paralleldomain.data_lab.generators.position_request import (
    AbsolutePositionRequest,
    LocationRelativePositionRequest,
    PositionRequest,
)
from paralleldomain.data_lab.generators.spawn_data import VehicleSpawnData
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.model.geometry.bounding_box_2d import BoundingBox2DBaseGeometry
from paralleldomain.utilities.logging import setup_loggers

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])
logger = logging.getLogger("pd.state.serialize")
logger.setLevel(logging.CRITICAL)


"""
In this example script, we create a scenario in which an ego vehicle drives in a suburban neighborhood and we implement
custom behaviors that cause other non-ego vehicles to drive along their driveways.

This script highlights the use of:
    - Custom Behaviors
    - Custom Agents
    - Custom Generators
    - Pre-Built Generators
    - Asset Database Lookups

Last revised: 28/Sept/2023
"""


def get_valid_spawn_point(
    location: Location, scenario: Scenario, radius_to_driveways: float, minimum_driveway_length: float
) -> np.ndarray:
    """
    Helper function to find a valid ego spawn location so that driveway vehicles can be spawned nearby

    Args:
        location: The Data Lab Location in which the scenario will take place
        scenario: The Data Lab Scenario object
        radius_to_driveways: The radius within which there should be valid driveways
        minimum_driveway_length: The minimum length of a driveway for it to be considered valid

    Returns:
        A 3x1 numpy array containing the x,y,z position of the found spawn point
    """

    # We will now load the UMD Map so that we can perform queries on it to select and appropriate spawn point.
    # To do this, we load the umd map directly, as well as initialize a MapQuery object.
    umd_map = load_map(location=location)
    map_query = MapQuery(umd_map)

    retries = 0  # Count the number of attempts at finding a spawn location

    while retries <= 1000:
        # Use the MapQuery object to find a random Driveway road, based on the random seed of the scenario
        driveway_point = map_query.get_random_road_type_object(
            road_type=RoadSegment.RoadType.DRIVEWAY, random_seed=scenario.random_seed
        )

        # If we can't find any driveways on the map, it is likely that the selection Location has no driveways
        if driveway_point is None:
            raise PdError("No available driveways exist on map.  Select another map.")

        # We have extracted a driveway RoadSegment, so we need to access the LaneSegment that makes up the
        # driveway "road".  A driveway will always be made up of 1 LaneSegment.
        driveway_lane = umd_map.lane_segments[int(driveway_point.lane_segments[0])]

        # Now that we have the driveway LaneSegment, we can extract the reference line and pick a reference point on
        # that line
        driveway_reference_line = map_query.edges[int(driveway_lane.reference_line)].as_polyline().to_numpy()
        driveway_reference_point = driveway_reference_line[0]

        # Check that the selected driveway is longer than the minimum length we specify (with a 15% margin), if not,
        # we jump back to the top of the loop and find a new driveway to check
        if np.linalg.norm(driveway_reference_line[-1] - driveway_reference_line[0]) < 1.15 * minimum_driveway_length:
            continue

        # By the time we have progressed to this point of the loop, we know that we now have a driveway object that
        # meets our length criteria.  We will now check whether there is a drivable lane near this driveway that we
        # can spawn our ego vehicle on.

        # We define a box around the driveway reference point in which we will search for drivable lanes, using the
        # proximity radius we defined earlier
        bounds = BoundingBox2DBaseGeometry(
            x=driveway_reference_point[0] - radius_to_driveways,
            y=driveway_reference_point[1] - radius_to_driveways,
            width=2 * radius_to_driveways,
            height=2 * radius_to_driveways,
        )

        # Use the MapQuery object to search for all LaneSegments which are within the bounds we defined just before
        lanes_near_spawn_point = map_query.get_lane_segments_within_bounds(bounds=bounds, method="overlap")

        # Keep only the LaneSegments which are drivable, as these are the only ones which we want to spawn our
        # ego vehicle on
        spawn_point_segment = next(
            (ls for ls in lanes_near_spawn_point if ls.type is LaneSegment.LaneType.DRIVABLE), None
        )

        # If there are no valid LaneSegments, then there is no where we can spawn the ego vehicle. We increment the
        # seed and begin the search loop again.
        if spawn_point_segment is None:
            logger.info(f"Failed to find valid spawn point at seed {scenario.random_seed} - incrementing and retrying")
            scenario.random_seed += 1
        else:  # If there is a valid LaneSegment which we can spawn the ego vehicle on
            # We find the reference line of the identified LaneSegment
            spawn_point_line = map_query.edges[int(spawn_point_segment.reference_line)].as_polyline().to_numpy()

            # Calculate the distance from every point on the spawn lane's reference line to the driveway reference
            # point
            distances_to_driveway = np.linalg.norm(spawn_point_line - driveway_reference_point, axis=1)

            # We choose to spawn at the point which is closest to the driveway
            spawn_point = spawn_point_line[np.argmin(distances_to_driveway)]

            return spawn_point

        retries += 1

    # If we exceed 1000 attempts to search for a spawn location, we exit and return an error
    raise PdError(
        "Failed to find valid spawn location within max_retries set.  " "Adjust max_retries or spawn conditions"
    )


# We create a custom class that inherits from the ScenarioCreator class.  This is where we will provide our scenario
# generation instructions that will instruct our Data Lab instance.
class NearbyVehiclesOnDriveways(ScenarioCreator):
    def create_scenario(
        self, random_seed: int, scene_index: int, number_of_scenes: int, location: Location, **kwargs
    ) -> ScenarioSource:
        # We define a simple 1920x1080 pinhole camera.  The camera is mounted 2 meters above the bottom surface of the
        # ego vehicle (z = 2.0) and has no yaw pitch or roll. The camera has a fox of 70 degrees.
        sensor_rig = SensorRig(
            sensor_configs=[
                SensorConfig(
                    display_name="Front",
                    camera_intrinsic=CameraIntrinsic(
                        width=1920,
                        height=1080,
                        fov=70.0,
                    ),
                    sensor_extrinsic=SensorExtrinsic(
                        roll=0.0,
                        pitch=0.0,
                        yaw=0.0,
                        x=0.0,
                        y=0.0,
                        z=2.0,
                    ),
                )
            ]
        )

        # Initialize a Scenario object with the sensor rig defined above
        scenario = Scenario(sensor_rig=sensor_rig)

        # Set the weather to be completely free of rain, fog or wetness
        scenario.environment.rain.set_constant_value(0.0)
        scenario.environment.fog.set_constant_value(0.0)
        scenario.environment.wetness.set_constant_value(0.0)

        # Set up parameters that we want to govern our spawn point selection.  In this case, we state that we want our
        # spawn point to be within 30 meters of driveways which are at least 6.0 meters in length.
        radius_to_driveways = 30
        minimum_driveway_length = 6.0

        # Use the helper function defined above to get a valid spawn point for the ego vehicle
        spawn_point = get_valid_spawn_point(
            location=location,
            scenario=scenario,
            radius_to_driveways=radius_to_driveways,
            minimum_driveway_length=minimum_driveway_length,
        )

        # We spawn our ego vehicle at the spawn_point identified above.  In this case, we use a Pre-Built Generator so
        # that we don't need to manually control how the vehicle drives through the scenario.

        # However, by using the AbsolutePositionRequest, we can specify that the ego vehicle is spawned at exactly the
        # spawn_point we selected above. We also specify that the ego vehicle is created without occupants.
        scenario.add_ego(
            generator=EgoAgentGeneratorParameters(
                agent_type=AgentType.VEHICLE,
                position_request=PositionRequest(
                    absolute_position_request=AbsolutePositionRequest(
                        position=Float3(x=spawn_point[0], y=spawn_point[1]), resolve_z=True
                    )
                ),
                vehicle_spawn_data=VehicleSpawnData(vehicle_peripheral=VehiclePeripheral(disable_occupants=True)),
            ),
        )

        # We now create the agents that travel along driveways near the ego vehicle. This is done by using a
        # custom generator.  Full details on implementation can be found within the generator and behavior.
        scenario.add_agents(
            generator=DrivewayCreepGenerator(
                behavior_duration=2.0,
                number_of_vehicles=3,
                driveway_entry_probability=0.5,
                radius=radius_to_driveways,
                min_driveway_length=minimum_driveway_length,
            )
        )

        # We place general traffic in the scenario by using a Pre-Built Generator. This way, we don't have to worry
        # about controlling the behavior of the traffic throughout the scene.

        # In this case, we spawn traffic within 100.0 meters of the ego vehicle we placed above
        scenario.add_agents(
            generator=TrafficGeneratorParameters(
                spawn_probability=0.8,
                position_request=PositionRequest(
                    location_relative_position_request=LocationRelativePositionRequest(
                        agent_tags=["EGO"],
                        max_spawn_radius=100.0,
                    )
                ),
            )
        )
        return scenario

    # The get_location() method allows us to define the location and lighting of the Data Lab scenario.

    # In this case, we select a suburban map which contains many driveways.
    # We also choose a random Lighting object by querying the DataLightingSublevels table of the Data Lab asset
    # database, and randomly choosing one of the returned Lighting options.
    def get_location(
        self, random_seed: int, scene_index: int, number_of_scenes: int, **kwargs
    ) -> Tuple[Location, Lighting]:
        # Query the name of all potential Lighting options from the DataLightingSublevels table of the asset database
        lighting_levels = [i.name for i in DataLightingSublevels.select()]

        # Randomly choose a Lighting option to return
        lighting = random.Random(random_seed + scene_index).choice(lighting_levels)
        return Location(name="A2_Kerrytown"), lighting


if __name__ == "__main__":
    # We use preview_scenario() to visualize the created scenario.  We pass in a fixed seed so that each
    # time we run the script, the same output is achieved.  We also request 20 rendered frames at a frame rate of 20 Hz.

    preview_scenario(
        scenario_creator=NearbyVehiclesOnDriveways(),
        frames_per_scene=20,
        sim_capture_rate=5,
        random_seed=42,
        instance_name="<instance name>",
    )
