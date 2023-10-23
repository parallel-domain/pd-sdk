import csv
import logging
import random
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from pd.core import PdError
from pd.data_lab.context import load_map
from pd.data_lab.scenario import Lighting, ScenarioCreator, ScenarioSource

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab import preview_scenario
from paralleldomain.data_lab.config.map import Area, MapQuery
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.logging import setup_loggers
from paralleldomain.utilities.transformation import Transformation

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])
logging.getLogger("pd.state.serialize").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

"""
In this example script, we create a scenario in which an ego drone descends on a backyard, using a predefined flight
profile.

This script highlights the use of custom behaviors and custom agents, as well as UMD Map lookups.

Last revised: 28/Sept/2023
"""


# This class contains the Custom Behavior which we wish to apply to the ego drone.
class EgoDroneFlightProfileBehavior(data_lab.CustomSimulationAgentBehavior):
    # The Ego Drone flight behavior will be parameterized by the start position, the flight profile we want the drone
    # to follow, and any minimum altitude we wish to implement.
    def __init__(
        self,
        start_position: Union[List[float], np.ndarray],
        flight_profile: List[Tuple[float, Transformation]],
        minimum_altitude_override: Optional[float] = None,
    ):
        super().__init__()

        # Store the parameters passed in during initialization in the object, so they can be accessed by other methods
        self._start_position: Union[List[float], np.ndarray] = start_position
        self._flight_profile: List[Tuple[float, Transformation]] = flight_profile
        self._minimum_altitude_override: float = minimum_altitude_override
        self._start_time: Optional[float] = None
        self._flight_profile_normalized: Optional[List[Tuple[float, Transformation]]] = None

        # Because we don't necessarily know what coordinate system a flight profile is in, we normalize it to
        # begin at the start_position, using a method we implement below.
        self.normalize_flight_profile(
            start_position=start_position, minimum_altitude_override=minimum_altitude_override
        )

        # Store the initial pose of the drone
        self._initial_pose: Transformation = self._flight_profile[0][1]

    # This method allows us to take in flight profiles in a coordinate system with an unknown origin, and recenter the
    # flight profile so that it begins at the start point specified
    def normalize_flight_profile(
        self, start_position: Union[List[float], np.ndarray], minimum_altitude_override: Optional[float]
    ):
        # Store the root pose of the flight profile.  Remember that as we have defined it at the moment, each row of
        # the flight profile contains the time step and the pose
        root_pose = self._flight_profile[0][1]

        # If we specify a minimum altitude override, calculate the altitude offset that is required
        altitude_offset = (
            minimum_altitude_override - min(tf.translation[2] for _, tf in self._flight_profile)
            if minimum_altitude_override is not None
            else 0
        )

        # We now go through each of the time steps in the flight profile and do two things:
        #    1. Offset the translation by the start position so that each pose is defined relative to the start
        #    2. Apply any require altitude offset to comply with the minimum altitude override
        self._flight_profile_normalized = [
            (
                ts,
                Transformation(
                    translation=tf.translation  # take every pose's translation
                    - root_pose.translation  # center it around first pose's original translation
                    + [
                        start_position[0],
                        start_position[1],
                        root_pose.translation[2] + altitude_offset,
                    ],  # and offset it by the start position and apply any required altitude offset.
                    quaternion=tf.quaternion,
                ),
            )
            for ts, tf in self._flight_profile
        ]

    # This method is responsible for setting the starting position of the object to which we assign this behavior.
    # In this case, we simply need to set the initial object pose to be equal to the start pose.
    def set_initial_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        random_seed: int,
        raycast: Optional[Callable] = None,
    ):
        # Set the agent's pose to be the initial pose defined during initialization
        agent.set_pose(pose=self._initial_pose.transformation_matrix)

    # This method is called every time a simulation time step is advanced.  As such, any logic for how an agent moves
    # throughout a scenario should be implemented in this method.
    def update_state(
        self,
        sim_state: data_lab.ExtendedSimState,
        agent: data_lab.CustomSimulationAgent,
        raycast: Optional[Callable] = None,
    ):
        # Store the name of the current time step that is being simulated by accessing the sim_state object
        current_time = sim_state.sim_time

        # We store the start time of the drone flight
        if self._start_time is None:
            self._start_time = current_time  # set first frame as start time even if not exactly 0.0 seconds

        # Look up how far through the flight profile we should be
        flight_profile_time = current_time - self._start_time

        # Store all the time steps in the flight profile in an array for easier access
        flight_profile_timestamps = np.asarray([ts for ts, _ in self._flight_profile_normalized])

        # Find the bounding indices of the flight profile that bounds the time step the simulation is currently in
        upper_bounding_index = np.searchsorted(flight_profile_timestamps, flight_profile_time, "right")
        lower_bounding_index = upper_bounding_index - 1

        # Check for out-of-bounds index when hitting first or last element in timestamp array
        upper_bounding_index = (
            upper_bounding_index
            if upper_bounding_index < len(flight_profile_timestamps)
            else len(flight_profile_timestamps) - 1
        )
        lower_bounding_index = lower_bounding_index if lower_bounding_index >= 0 else 0

        # Get bounding timestamps and their associated Transformation poses
        lower_bounding_ts, lower_bounding_tf = self._flight_profile_normalized[lower_bounding_index]
        upper_bounding_ts, upper_bounding_tf = self._flight_profile_normalized[upper_bounding_index]

        # If the bounding indices are the same, then we have an exact match and don't need to interpolate
        if lower_bounding_ts == upper_bounding_ts:
            interpolated_pose = lower_bounding_tf
        else:  # If the bounding indices are not the same, we can interpolate between their poses
            # Interpolate between two recorded timestamps and find the appropriate translation/rotation using
            # the Transformation object's built in interpolate method
            bounding_value_factor = (flight_profile_time - lower_bounding_ts) / (upper_bounding_ts - lower_bounding_ts)
            interpolated_pose = Transformation.interpolate(
                tf0=lower_bounding_tf, tf1=upper_bounding_tf, factor=bounding_value_factor
            )

        # Output a record of what pose we will be assigning to the agent
        logger.info(f"Using interpolated pose: {interpolated_pose} at {flight_profile_time}")

        # We now set our agent's pose to be the pose which we have calculated above
        agent.set_pose(pose=interpolated_pose.transformation_matrix)

    # The clone method returns an exact copy of the Custom Behavior object, and is required by Data Lab under the hood
    def clone(self) -> "EgoDroneFlightProfileBehavior":
        return EgoDroneFlightProfileBehavior(
            start_position=self._start_position,
            flight_profile=self._flight_profile,
            minimum_altitude_override=self._minimum_altitude_override,
        )


# We create a custom class that inherits from the ScenarioCreator class.  This is where we will provide our scenario
# generation instructions that will instruct our Data Lab instance.
class BackyardDrone(ScenarioCreator):
    # The create_scenario method is where we provide our Data Lab Instance with the scenario generation instructions it
    # requires to create the scenario
    def create_scenario(
        self, random_seed: int, scene_index: int, number_of_scenes: int, location: data_lab.Location, **kwargs
    ) -> ScenarioSource:
        # We define a simple 768x768 pixel pinhole camera.
        # This camera is pointed downwards (-90 degree rotation about the x axis) and located at the center of the
        # drone. It has a 70 degree field of view.
        sensor_rig = data_lab.SensorRig().add_camera(
            name="Front",
            width=768,
            height=768,
            field_of_view_degrees=70,
            pose=Transformation.from_euler_angles(
                angles=[-90, 0.0, 0.0], order="xyz", degrees=True, translation=[0.0, 0.0, 0.0]
            ),
        )

        # Initialize a Scenario object with the sensor rig defined above
        scenario = data_lab.Scenario(sensor_rig=sensor_rig)

        # Set the weather to be completely free of rain and low in wetness
        scenario.environment.rain.set_constant_value(0.0)
        scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

        # We will now load the UMD Map so that we can perform queries on it to select and appropriate spawn point.
        # To do this, we load the umd map directly, as well as initialize a MapQuery object.
        umd_map = load_map(location)
        map_query = MapQuery(umd_map)

        # We want our drone to descend into a backyard so we use a MapQuery method to get a random backyard.
        # Notice that we seed the search method with the random seed of the scenario so that any returned results are
        # deterministic.
        start_pose = map_query.get_random_area_location(area_type=Area.AreaType.YARD, random_seed=random_seed)

        # If we are unable to find a yard in the umd map, raise an error. This is likely because the umd map contains
        # no yard.
        if start_pose is None:
            raise PdError("Failed to find Yard location to spawn. Please try another map")

        # The MapQuery method will return a position on the ground, so we offset it upwards by 25 cm since we don't want
        # the drone to collide with the ground.
        start_pose.translation[2] += 0.25

        # Load a sample flight profile that is stored as a csv file. As we have defined things at the moment, each row
        # of the flight profile contains two elements:
        #    1. The time step of that row of the flight profile
        #    2. The pose of the drone at that time step
        curr_dir = AnyPath(__file__).parent.absolute()
        with AnyPath(curr_dir / "sample_flight_profile_ascending.csv").open() as fp:
            flight_profile = [
                (
                    float(row[0]),
                    Transformation(
                        translation=[row[1], row[2], row[3]],
                        quaternion=[float(row[4]), float(row[5]), float(row[6]), float(row[7])],
                    ),
                )
                for row in csv.reader(fp, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
            ]

        # We create a custom ego agent that will be our flying drone.  Notice that we attached the sensor rig defined
        # above and leave asset_name blank.  This is so that no drone asset will be rendered.

        # We also attach our custom behavior to the custom agent and provide the start position, flight profile and
        # minimum altitide parameters it needs to function.
        scenario.add_ego(
            data_lab.CustomSimulationAgents.create_ego_vehicle(
                sensor_rig=sensor_rig,
                asset_name="",
                lock_to_ground=False,
            ).set_behavior(
                EgoDroneFlightProfileBehavior(
                    start_position=start_pose.translation,
                    flight_profile=flight_profile,
                    minimum_altitude_override=start_pose.translation[2],
                )
            )
        )

        # Return the scenario object now that we have provided all required scenario generation instructions.
        return scenario

    # The get location method allows us to define the location and lighting of the Data Lab scenario.  In this case,
    # we select a suburban map which contains many backyards, as well as a lighting option which corresponds to a mostly
    # cloudy day around noon.
    def get_location(
        self, random_seed: int, scene_index: int, number_of_scenes: int, **kwargs
    ) -> Tuple[data_lab.Location, Lighting]:
        return data_lab.Location(name="SJ_EssexAndBradford"), "day_partlyCloudy_03"


if __name__ == "__main__":
    # We use preview_scenario() to visualize the created scenario.  We do not pass in a fixed seed so that each
    # scenario generated is different.  We also request 30 rendered frames at a frame rate of 10 Hz.
    preview_scenario(
        scenario_creator=BackyardDrone(),
        random_seed=2023,
        frames_per_scene=30,
        sim_capture_rate=10,
        instance_name="<instance name>",
    )
