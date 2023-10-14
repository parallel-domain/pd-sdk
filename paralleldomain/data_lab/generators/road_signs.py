import random
from typing import List

from pd.data_lab.generators.custom_simulation_agent import CustomObjectSimulationAgent
from pd.internal.assets.asset_registry import (
    DataSign,
    InfoSegmentation,
    ObjAssets,
    ObjCountries,
    UtilSegmentationCategoriesPanoptic,
)

from paralleldomain.data_lab import CustomAtomicGenerator, CustomSimulationAgent, ExtendedSimState
from paralleldomain.data_lab.behaviors.traffic_sign import TrafficSignAttachToPoleBehavior, TrafficSignPoleBehavior


class SignGenerator(CustomAtomicGenerator):
    """
    Creates and spawns traffic signs in a define region in front of the ego vehicle.  Can be used in temporal and
        single frame scenarios

    Args:
        num_sign_poles: The number of sign poles (which will themselves have traffic signs attached to them) to spawn
            in the scenario
        random_seed: The integer to seed all random functions with, allowing scenario generation to be deterministic
        radius: The radius of the region in which traffic signs should be spawned
        sim_capture_rate: Controls the frame rate of the scenario where "scenario_frame_rate = 100 / sim_capture_rate".
            For single frame scenarios, value of 10 is recommended
        country: The name of the country from which traffic signs should be selected to be spawned in the world
        single_frame_mode: Flag to indicate whether this generator is being used to configure a single frame
            dataset.  In single frame datasets, the location of the vehicles and the signage changes between each
            rendered frame
        forward_offset_to_place_signs:  The distance (in meters) in front of the ego vehicle on which the region in
            which traffic signs can be spawned should be centered
        max_signs_per_pole: The maximum number of traffic signs which can be spawned on each traffic pole
        sign_spacing: The vertical gap between traffic signs on the same traffic pole
        min_distance_between_signs: The minimum distance (in meters) between traffic poles in the scenario
    """

    def __init__(
        self,
        num_sign_poles: int,
        random_seed: int,
        radius: float,
        country: str,
        single_frame_mode: bool,
        orient_signs_facing_travel_direction: bool,
        sim_capture_rate: int = 10,
        forward_offset_to_place_signs: float = 15,
        max_signs_per_pole: int = 2,
        sign_spacing: float = 0.2,
        min_distance_between_signs: float = 1.5,
        max_retries: int = 1000,
        max_random_yaw: int = 0,
    ):
        super().__init__()
        # Store input parameters so that they can be accessed by other methods
        self._num_sign_poles = num_sign_poles
        self._random_seed = random_seed
        self._radius = radius
        self._sim_capture_rate = sim_capture_rate
        self._max_signs_per_pole = max_signs_per_pole
        self._country = country
        self._sign_spacing = sign_spacing
        self._forward_offset_to_place_signs = forward_offset_to_place_signs
        self._min_distance_between_signs = min_distance_between_signs
        self._single_frame_mode = single_frame_mode
        self._max_retries = max_retries
        self._max_random_yaw = max_random_yaw
        self._orient_signs_facing_travel_direction = orient_signs_facing_travel_direction

        # Retrieve information about sign poles using the method implemented below
        self._sign_pole_data = self._get_all_sign_posts_with_metadata()

        # Initialize a random state
        self._random_state = random.Random(self._random_seed)

    @property
    def random_seed(self):
        # Increments seed each time we reference it
        self._random_seed += 1
        return self._random_seed

    # The method performs a database lookup to retrieve physical properties of the sign posts to which signs
    # can be attached
    def _get_all_sign_posts_with_metadata(self) -> list:
        # We store the asset names of the poles that we want to use to attach signs to. In the future, this could be
        # expanded to actually search for poles in the database
        default_sign_post_list = [
            "post_round_metal_0365h_06r",
            "post_sign_0288h02r",
            "crosswalkpost_01_small",
            "post_round_metal_0400h_08r",
        ]

        # We perform the database lookup here. In this query, we use the InfoSegmentation and ObjAssets tables
        # to search for the poles and retrieve the width, height and name of the poles
        query = (
            InfoSegmentation.select(
                InfoSegmentation.name,
                ObjAssets.width,
                ObjAssets.height,
            )
            .join(ObjAssets, on=(InfoSegmentation.asset_id == ObjAssets.id))
            .where(ObjAssets.name << default_sign_post_list)
        ).dicts()

        # Store the retrieved poles in a list and return it for use later on
        post_list = [post for post in query]
        return post_list

    # This function performs a database query to return the names and height of the signs that correspond to
    # a certain country.
    def _get_sign_metadata_from_country(self, country: str) -> list:
        # This is where we set up our database query.  The point of this query is the extract the name, country
        # and height of all signs that belong to a certain country.  To do so we carry out the following steps:
        #    1. Join the InfoSegmentation, UtilSegmentationCategoriesPanoptic, DataSign, ObjCountries and ObjAssets
        #        tables together using the appropriate ids
        #    2. Retrieve only assets that are a "TrafficSign" and belong to the specified country
        query = (
            InfoSegmentation.select(
                InfoSegmentation.name.alias("sign_name"),
                ObjCountries.name.alias("country"),
                ObjAssets.height,
            )
            .join(
                UtilSegmentationCategoriesPanoptic,
                on=(InfoSegmentation.panoptic_id == UtilSegmentationCategoriesPanoptic.id),
            )
            .join(DataSign, on=(DataSign.asset_id == ObjAssets.id))
            .join(ObjCountries, on=(ObjCountries.id == DataSign.country_id))
            .join(ObjAssets, on=(ObjAssets.id == InfoSegmentation.asset_id))
            .where(UtilSegmentationCategoriesPanoptic.name == "TrafficSign", ObjCountries.name == country)
        ).dicts()

        # Store the signs retrieved from the database query in a list and return it for use later
        sign_list = [sign for sign in query]
        return sign_list

    # This method is the method responsible for spawning Custom Agents in the scene and assigning a behavior to them
    def create_agents_for_new_scene(self, state: ExtendedSimState, random_seed: int) -> List[CustomSimulationAgent]:
        # Initialize an empty list to store the Custom Agents we create
        agents = []

        # Get all the signs from the specified country
        all_valid_signs = self._get_sign_metadata_from_country(country=self._country)

        # We create a loop that places the specified number of sign posts
        for i in range(self._num_sign_poles):
            # Randomly choose a sign post asset to spawn from the list
            pole_to_spawn = self._random_state.choice(self._sign_pole_data)

            # Place the sign post as a Custom Agent and assign the Custom Behavior TrafficSignPoleBehavior to it.
            # Further information on the Custom Behavior can be found within the source file
            agent = CustomObjectSimulationAgent(asset_name=pole_to_spawn["name"]).set_behavior(
                behavior=TrafficSignPoleBehavior(
                    random_seed=self.random_seed,
                    radius=self._radius,
                    forward_offset_to_place_signs=self._forward_offset_to_place_signs,
                    min_distance_between_signs=self._min_distance_between_signs,
                    single_frame_mode=self._single_frame_mode,
                    max_retries=self._max_retries,
                    orient_signs_facing_travel_direction=self._orient_signs_facing_travel_direction,
                ),
            )

            # Append the agent to the list of agents
            agents.append(agent)

            # Store the agent id of the pole we just spawned for use later
            pole_agent_id = agent.agent_id

            # Randomly choose the number of signs and which signs to spawn on each pole
            num_signs_on_pole = self._random_state.randint(1, self._max_signs_per_pole)
            signs_to_spawn = [self._random_state.choice(all_valid_signs) for _ in range(num_signs_on_pole)]

            # We implement a check that the pole height is sufficiently tall for the signs we've requested
            verified_total_pole_height = False
            while not verified_total_pole_height:
                # Calculate the total height that the signs will occupy accounting for spacing
                total_sign_height = (
                    sum([sign["height"] for sign in signs_to_spawn]) + (len(signs_to_spawn) - 1) * self._sign_spacing
                )

                # If the pole is too short, remove a sign run the check again
                if pole_to_spawn["height"] < total_sign_height:
                    signs_to_spawn.pop()
                # If the pole is sufficiently tall, we can exit the check
                else:
                    verified_total_pole_height = True

            # Run a for loop to iterate through all the signs that need to be spawned on a given pole
            for sign in signs_to_spawn:
                # Spawn a sign as a Custom Agent with the TrafficSignAttachToPoleBehavior Custom Behavior.  Full details
                # on the Custom Behavior can be found within the source file, but notice that we pass the agent id of
                # the pole that the sign should be attached to
                sign_agent = CustomObjectSimulationAgent(
                    asset_name=sign["sign_name"], lock_to_ground=False
                ).set_behavior(
                    behavior=TrafficSignAttachToPoleBehavior(
                        random_seed=self.random_seed,
                        parent_pole_id=pole_agent_id,
                        all_signs_on_pole_metadata=signs_to_spawn,
                        sign_spacing=self._sign_spacing,
                        max_random_yaw=self._max_random_yaw,
                    )
                )

                # Append the sign agent to the list of agents this Custom Generator has spawned
                agents.append(sign_agent)

        # Return the list of agents this Custom Generator has spawned
        return agents

    # The clone method returns a copy of the Custom Generator object and is required under the hood by Data Lab
    def clone(self):
        return SignGenerator(
            num_sign_poles=self._num_sign_poles,
            random_seed=self._random_seed,
            radius=self._radius,
            sim_capture_rate=self._sim_capture_rate,
            max_signs_per_pole=self._max_signs_per_pole,
            country=self._country,
            sign_spacing=self._sign_spacing,
            forward_offset_to_place_signs=self._forward_offset_to_place_signs,
            min_distance_between_signs=self._min_distance_between_signs,
            single_frame_mode=self._single_frame_mode,
            max_retries=self._max_retries,
            max_random_yaw=self._max_random_yaw,
            orient_signs_facing_travel_direction=self._orient_signs_facing_travel_direction,
        )
