from typing import Tuple

import numpy as np
from pd.data_lab import Scenario, ScenarioCreator
from pd.data_lab.config.distribution import CenterSpreadConfig, ContinousUniformDistribution
from pd.data_lab.context import load_map
from pd.data_lab.scenario import Lighting, ScenarioSource

import paralleldomain.data_lab
from paralleldomain import data_lab
from paralleldomain.data_lab.config.map import LaneSegment, MapQuery, RoadSegment, Side
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, SensorExtrinsic
from paralleldomain.data_lab.config.types import Float3
from paralleldomain.data_lab.generators.behavior import LookAtPointBehavior, VehicleBehavior
from paralleldomain.data_lab.generators.peripherals import VehiclePeripheral
from paralleldomain.data_lab.generators.position_request import (
    AbsolutePositionRequest,
    LocationRelativePositionRequest,
    PositionRequest,
)
from paralleldomain.data_lab.generators.spawn_data import AgentSpawnData, VehicleSpawnData
from paralleldomain.data_lab.generators.traffic import TrafficGeneratorParameters
from paralleldomain.data_lab.generators.vehicle import VehicleGeneratorParameters
from paralleldomain.utilities.logging import setup_loggers

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])


"""
In this example script, we create a scenario in which the ego agent is a static traffic monitoring camera overlooking a
highway traffic jam scenario.

This script highlights the use of Custom Agents, Custom Behaviors and UMD Lookups

Last revised: 30/Sept/2023
"""


# We create a custom class that inherits from the ScenarioCreator class.  This is where we will provide our scenario
# generation instructions that will instruct our Data Lab instance
class TrafficMonitoringCamera(ScenarioCreator):
    # The create_scenario method is where we provide our Data Lab Instance with the scenario generation instructions it
    # requires to create the scenario
    def create_scenario(
        self,
        random_seed: int,
        scene_index: int,
        number_of_scenes: int,
        location: paralleldomain.data_lab.Location,
        **kwargs,
    ) -> ScenarioSource:
        # We define a simple pinhole camera that sits exactly at the point of the ego agent and faces forwards
        sensor_rig = paralleldomain.data_lab.SensorRig(
            sensor_configs=[
                paralleldomain.data_lab.SensorConfig(
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
                        z=0.0,
                    ),
                )
            ]
        )

        # Initialize a Scenario object with the sensor rig defined above
        scenario = Scenario(sensor_rig=sensor_rig)

        # Set weather to be completely free of rain and low in wetness and fog
        scenario.environment.rain.set_constant_value(0.0)
        scenario.environment.fog.set_uniform_distribution(min_value=0.1, max_value=0.3)
        scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

        # Load the UMD Map of the Location by using the location parameter which is called within the create_scenario()
        # method
        umd_map = load_map(location=location)

        # Initialize a MapQuery object using the UMD Map loaded above
        map_query = MapQuery(umd_map)

        # Set parameters that govern the scenario generation. These will be used below
        min_path_length = 250  # The minimum distance of highway that must exist beyond the spawn point
        camera_height = 10  # The height of the traffic monitoring camera
        distance_to_traffic = 125  # The distance along the highway between the camera and the traffic jam

        # To create this scenario, we will use a mixture of Pre-Built Generators and Custom Agents/Behaviors.
        # The following steps will be carried out:
        #    1. Using the UMD Map, find a location on the highway with sufficient distance ahead (min_path_length)
        #    2. Place a vehicle at this point tagged as "STAR". This will act as a reference position that we can place
        #        agents relative to using LocationRelativePositionRequests
        #    3. Use a Pre-Built Generator to place stopped traffic ahead of the STAR vehicle
        #    4. Place the Custom Ego Agent on the median of the highway using a Custom Behavior
        #    5. Fill in the scene with traffic using a Pre-Built Generator

        # Using the MapQuery object we get a random LaneSegment object from a specified RoadType. Notice that this
        # method allows us to retrieve a LaneSegment that exists on a particular road type, removing the need to
        # retrieve all Drivable LaneSegments and sort them according to the type of road they belong to. We also specify
        # a minimum distance of road that must exist ahead of the start of the LaneSegment using the min_path_length
        # parameter
        spawn_lane = map_query.get_random_lane_object_from_road_type(
            road_type=RoadSegment.RoadType.MOTORWAY,
            lane_type=LaneSegment.LaneType.DRIVABLE,
            random_seed=scenario.random_seed,
            min_path_length=min_path_length,
        )

        # Specify that the spawn point will be the start of the reference line of the LaneSegment retrieved above
        spawn_point = umd_map.edges[spawn_lane.reference_line].as_polyline().to_numpy()[0]

        # We now find the coordinates of the position around which the traffic jam will be located. This allows us
        # to know where to point the static traffic camera

        # Use the MapQuery object to retrieve the reference line points of the LaneSegment retrieved above, and
        # any connected LaneSegments, up to the min_path_length specified above
        star_vehicle_path = map_query.get_connected_lane_points(lane_id=spawn_lane.id, path_length=min_path_length)

        # Again using the MapQuery object, retrieve the point on the reference line that is at least the specified
        # distance from the beginning of the LaneSegment. This will be the position around which the traffic jam is
        # centered
        traffic_jam_3d_location = map_query.get_line_point_by_distance_from_start(
            line=star_vehicle_path, distance_from_start=distance_to_traffic
        )

        # Use the MapQuery object to get the points that define the left edge of the road on which the LaneSegment above
        # sits
        camera_edge_line = (
            map_query.get_edge_of_road_from_lane(lane_id=spawn_lane.id, side=Side.LEFT).as_polyline().to_numpy()
        )

        # Choose the point on the edge line that is closest to the spawn point to be the location of the camera
        camera_location = camera_edge_line[np.argmin(np.linalg.norm(camera_edge_line - spawn_point, axis=1))]

        # Raise the camera up in the air to create the full 3D position of the camera
        camera_3d_position = np.array([camera_location[0], camera_location[1], camera_location[2] + camera_height])

        # Create the Static Custom Ego Agent with the Custom Behavior LookAtPointBehavior. Full details on the Custom
        # Behavior can be found in the source file
        scenario.add_ego(
            paralleldomain.data_lab.CustomSimulationAgents.create_ego_sensor(
                sensor_rig=sensor_rig,
            ).set_behavior(
                LookAtPointBehavior(
                    look_from=camera_3d_position,
                    look_at=traffic_jam_3d_location,
                )
            )
        )

        # Use the Pre-Built Generator VehicleGeneratorParameters to place a Star Vehicle at the reference spawn point
        # found above. Note that we apply the "STAR" agent tag so that other agents can be placed relative to this agent
        # using LocationRelativePositionRequests
        scenario.add_agents(
            generator=VehicleGeneratorParameters(
                model="midsize_sedan_04",
                position_request=PositionRequest(
                    absolute_position_request=AbsolutePositionRequest(
                        position=Float3(
                            x=spawn_point[0],
                            y=spawn_point[1],
                        ),
                        resolve_z=True,
                    )
                ),
                vehicle_spawn_data=VehicleSpawnData(
                    vehicle_peripheral=VehiclePeripheral(
                        disable_occupants=True,
                    ),
                    agent_spawn_data=AgentSpawnData(
                        tags=[
                            "STAR",
                        ]
                    ),
                ),
            )
        )

        # Use the Pre-Built Generator TrafficGeneratorParameters to place stopped traffic ahead of the STAR vehicle
        # placed above
        scenario.add_agents(
            generator=TrafficGeneratorParameters(
                # Specify the highest possible density of traffic to create traffic jam
                spawn_probability=1.0,
                # Specify the spawn position of the traffic to be relative to the location of the STAR vehicle placed
                # above
                position_request=PositionRequest(
                    location_relative_position_request=LocationRelativePositionRequest(
                        agent_tags=["STAR"],
                        max_spawn_radius=20.0,
                    ),
                    # Move the center of the spawn circle for the traffic jam ahead of the STAR vehicle by the distance
                    # defined in distance_to_traffic
                    longitudinal_offset=CenterSpreadConfig(
                        center=distance_to_traffic,
                        spread=25,
                    ),
                ),
                # Choose that all vehicles spawned by this Pre-Built Generator are stationary.  This is conducted by
                # specifying a close to zero start_speed and target_speed within the VehicleBehavior object
                vehicle_spawn_data=VehicleSpawnData(
                    vehicle_behavior=VehicleBehavior(
                        start_speed=ContinousUniformDistribution(
                            min=0.0,
                            max=0.1,
                        ),
                        target_speed=ContinousUniformDistribution(
                            min=0.0,
                            max=0.1,
                        ),
                        # Specify that spawned vehicles are not all exactly in the center of the lane for realism
                        lane_offset=ContinousUniformDistribution(
                            min=-0.2,
                            max=0.2,
                        ),
                        # Specify that spawned vehicles do not perfectly track the center of the lanes for realism
                        lane_drift_amplitude=ContinousUniformDistribution(
                            min=0.1,
                            max=0.3,
                        ),
                    )
                ),
            )
        )

        # Fill in the rest of the scene with traffic using the Pre-Built Generator TrafficGeneratorParameters
        scenario.add_agents(
            generator=TrafficGeneratorParameters(
                spawn_probability=1.0,
                position_request=PositionRequest(
                    location_relative_position_request=LocationRelativePositionRequest(
                        agent_tags=["STAR"],
                        max_spawn_radius=2 * distance_to_traffic,
                    )
                ),
            )
        )

        # Return the scenario object
        return scenario

    # The get location method allows us to define the location and lighting of the Data Lab scenario.  In this case,
    # we select a highway map, as well as a lighting option which corresponds to a mostly
    # cloudy day around noon.
    def get_location(
        self, random_seed: int, scene_index: int, number_of_scenes: int, **kwargs
    ) -> Tuple[paralleldomain.data_lab.Location, Lighting]:
        return paralleldomain.data_lab.Location(name="SJ_237AndGreatAmerica"), "day_partlyCloudy_03"


if __name__ == "__main__":
    # We use preview_scenario() to visualize the created scenario.  We pass in a fixed seed so that the same scenario is
    # generated every time the script is run.  We also request 100 rendered frames at a frame rate of 10 Hz.
    data_lab.preview_scenario(
        scenario_creator=TrafficMonitoringCamera(),
        frames_per_scene=100,
        sim_capture_rate=10,
        random_seed=2023,
        instance_name="<instance name>",
    )
