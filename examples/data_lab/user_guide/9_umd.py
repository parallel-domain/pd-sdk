import logging
import random
from typing import Tuple

from pd.data_lab import ScenarioCreator, ScenarioSource
from pd.data_lab.config.distribution import EnumDistribution
from pd.data_lab.config.location import Location
from pd.data_lab.context import load_map
from pd.data_lab.scenario import Lighting

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.config.map import Area, LaneSegment, MapQuery, RoadSegment
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, SensorExtrinsic
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.position_request import LaneSpawnPolicy, PositionRequest
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.utilities.transformation import Transformation

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])


"""
In this script, we walk through an example of how to query UMD Maps while creating a Data Lab scenario.

Much of the script below is boilerplate code which is not related to the UMD lookups, but rather serves
to demonstrate how you can integrate these lookups into you scenario generation scripts.
"""


# Create a custom ScenarioCreator class
class UMDLookupExample(ScenarioCreator):
    def create_scenario(
        self, random_seed: int, scene_index: int, number_of_scenes: int, location: Location, **kwargs
    ) -> ScenarioSource:
        # Load the UMD Map based on the specified Location from the get_location() method
        umd_map = load_map(location=location)

        # Initialize a MapQuery object from the UMD map
        map_query = MapQuery(umd_map)

        # Look at all the LaneSegments in the UMD map and only extract drivable lanes, using a list comprehension which
        # directly accesses the UMD map
        all_drivable_lane_segments = [
            umd_map.lane_segments[lane_id]
            for lane_id in umd_map.lane_segments
            if umd_map.lane_segments[lane_id].type == LaneSegment.LaneType.DRIVABLE
        ]

        # Randomly choose a drivable lane segment from those extracted above
        drivable_lane_segment = random.choice(all_drivable_lane_segments)

        # Extract a numpy array of the points which denote the reference line of the drivable lane selected above.
        # The reference line is extracted by directly accessing the UMD map
        reference_line = umd_map.edges[drivable_lane_segment.reference_line].as_polyline().to_numpy()

        # Look at all the RoadSegments in the UMD map and only extract secondary roads, using a list comprehension which
        # directly accesses the UMD map
        all_secondary_road_segments = [
            umd_map.road_segments[road_id]
            for road_id in umd_map.road_segments
            if umd_map.road_segments[road_id].type == RoadSegment.RoadType.SECONDARY
        ]

        # Randomly select a road segment from all the road segments extracted above
        secondary_road_segment = random.choice(all_secondary_road_segments)

        # Extract the lanes that are part of the selected lane segment
        lanes_in_secondary_road_segment = [
            umd_map.lane_segments[lane_id] for lane_id in secondary_road_segment.lane_segments
        ]
        print(f"{len(lanes_in_secondary_road_segment)} lanes found in selected Secondary RoadSegment.")

        # Directly access the UMD map to find all parking lot areas in the UMD map
        all_parking_lot_areas = [
            umd_map.areas[area_id]
            for area_id in umd_map.areas
            if umd_map.areas[area_id].type == Area.AreaType.PARKING_LOT
        ]

        # Randomly select a parking lot Area object
        parking_lot_area = random.choice(all_parking_lot_areas)

        # Directly access the UMD map to extract the Edge that defines the perimeter of the selected area
        perimeter_edge = umd_map.edges[parking_lot_area.edges[0]].as_polyline().to_numpy()
        print(f"Edge of selected parking lot Area: {perimeter_edge}")

        # Define a search location to use in MapQuery demos below
        search_location = Transformation.from_euler_angles(
            angles=[0, 0, 0], order="xyz", translation=[-56.0, 222.9, 9.1]
        )

        # Use the MapQuery object to find all lane segments near the search location defined above
        lane_segments_near = map_query.get_lane_segments_near(pose=search_location, radius=10.0)

        # Extract only drivable lanes from all the lane segments found above
        drivable_lane_segements_near = [
            lane for lane in lane_segments_near if lane.type == LaneSegment.LaneType.DRIVABLE
        ]
        print(f"Found {len(drivable_lane_segements_near)} lanes near search location")

        # Use the MapQuery object again to find a location that is near a junction
        location_near_junction = map_query.get_random_junction_relative_lane_location(
            random_seed=random.randint(0, 100000),  # Seed the function so that location selection is deterministic
            distance_to_junction=10.0,  # Specify how far away from a junction we wish the returned point to be
            probability_of_signaled_junction=0.5,  # Proability that returned location is near a signaled intersection
        )
        print(f"Returned location near junction at {location_near_junction.translation}")

        # Utilize the MapQuery object to check whether the drivable lane we selected earlier is longer than 15 meters
        is_longer_than = map_query.check_lane_is_longer_than(lane_id=drivable_lane_segment.id, path_length=15.0)
        print(f"Selected lane is longer than 15.0 meters - {is_longer_than}")

        # Use the MapQuery object to find the ID of the lane which sits directly below a specified point on the UMD map.

        # In this case, because we search for lanes using a point on the reference line of the drivable lane we chose
        # above, the function will return the same ID as that of the lane we found earlier
        found_lane_id = map_query.find_lane_id_from_pose(
            pose=Transformation.from_euler_angles(angles=[0, 0, 0], order="xyz", translation=reference_line[0])
        )
        print(f"The lane id at the search location is {found_lane_id}")

        # The remainder of the code is boilerplate code to create a simple scenario with only and ego vehicle.
        # It serves to illustrate how you can implement asset database lookups within a Data Lab scenario.

        sensor_rig = data_lab.SensorRig(
            sensor_configs=[
                data_lab.SensorConfig(
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

        scenario = data_lab.Scenario(sensor_rig=sensor_rig)

        scenario.environment.rain.set_constant_value(0.0)
        scenario.environment.fog.set_uniform_distribution(min_value=0.5, max_value=0.9)
        scenario.environment.wetness.set_uniform_distribution(min_value=0.2, max_value=0.6)

        scenario.add_ego(
            generator=EgoAgentGeneratorParameters(
                agent_type=AgentType.VEHICLE,
                position_request=PositionRequest(
                    lane_spawn_policy=LaneSpawnPolicy(
                        lane_type=EnumDistribution(
                            probabilities={"Drivable": 1.0},
                        )
                    )
                ),
            ),
        )

        return scenario

    def get_location(
        self, random_seed: int, scene_index: int, number_of_scenes: int, **kwargs
    ) -> Tuple[Location, Lighting]:
        return data_lab.Location(name="SF_6thAndMission_medium"), "day_partlyCloudy_03"


if __name__ == "__main__":
    data_lab.preview_scenario(
        scenario_creator=UMDLookupExample(),
        random_seed=2023,
        frames_per_scene=100,
        sim_capture_rate=10,
        instance_name="<instance_name>",
    )
