import random
from typing import List

from pd.data_lab.generators.custom_simulation_agent import CustomObjectSimulationAgent
from pd.internal.assets.asset_registry import (
    InfoSegmentation,
    ObjAssets,
    ObjCountries,
    UtilSegmentationCategoriesPanoptic,
    DataSign,
)

from paralleldomain.data_lab import CustomAtomicGenerator, ExtendedSimState, CustomSimulationAgent
from paralleldomain.data_lab.generators.behavior import TrafficSignPoleBehavior, TrafficSignAttachToPoleBehavior


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
        sim_capture_rate: int = 10,
        forward_offset_to_place_signs: float = 15,
        max_signs_per_pole: int = 2,
        sign_spacing: float = 0.2,
        min_distance_between_signs: float = 1.5,
        max_retries: int = 1000,
    ):
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

        self._sign_pole_data = self._get_all_sign_posts_with_metadata()

        self._random_state = random.Random(self._random_seed)

    # Function to return all sign poles with metadata.  A little bit redundant at the moment but included for
    # future proofing as more pole assets become available
    def _get_all_sign_posts_with_metadata(self) -> list:
        default_sign_post_list = [
            "post_round_metal_0365h_06r",
            "post_sign_0288h02r",
            "crosswalkpost_01_small",
            "post_round_metal_0400h_08r",
        ]

        query = (
            InfoSegmentation.select(
                InfoSegmentation.name,
                ObjAssets.width,
                ObjAssets.height,
            )
            .join(ObjAssets, on=(InfoSegmentation.asset_id == ObjAssets.id))
            .where(ObjAssets.name << default_sign_post_list)
        ).dicts()

        post_list = [post for post in query]

        return post_list

    # Function to return all signs from a specified country with height metadata
    def _get_sign_metadata_from_country(self, country: str) -> list:
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
            .where(
                UtilSegmentationCategoriesPanoptic.name == "TrafficSign",
                ObjCountries.name == country,
            )
        ).dicts()

        sign_list = [sign for sign in query]
        return sign_list

    def create_agents_for_new_scene(self, state: ExtendedSimState, random_seed: int) -> List[CustomSimulationAgent]:
        agents = []

        # Get all the signs from the specified country
        all_valid_signs = self._get_sign_metadata_from_country(country=self._country)

        # Loop through and place all the sign posts
        for i in range(self._num_sign_poles):
            pole_to_spawn = self._random_state.choice(self._sign_pole_data)

            # Place the sign post
            agent = CustomObjectSimulationAgent(asset_name=pole_to_spawn["name"]).set_behaviour(
                behaviour=TrafficSignPoleBehavior(
                    random_seed=self._random_seed,
                    radius=self._radius,
                    forward_offset_to_place_signs=self._forward_offset_to_place_signs,
                    min_distance_between_signs=self._min_distance_between_signs,
                    single_frame_mode=self._single_frame_mode,
                    max_retries=self._max_retries,
                ),
            )

            agents.append(agent)
            pole_agent_id = agent.agent_id  # Store the agent ID of the pole

            # Choose the number of signs and which signs to spawn on each pole
            num_signs_on_pole = self._random_state.randint(1, self._max_signs_per_pole)
            signs_to_spawn = [self._random_state.choice(all_valid_signs) for _ in range(num_signs_on_pole)]

            # Check that the pole height is sufficient for the signs
            verified_total_pole_height = False
            while not verified_total_pole_height:
                total_sign_height = (
                    sum([sign["height"] for sign in signs_to_spawn]) + (len(signs_to_spawn) - 1) * self._sign_spacing
                )

                if pole_to_spawn["height"] < total_sign_height:
                    signs_to_spawn.pop()  # If the pole is too short, remove a sign until they fit on the pole
                else:
                    verified_total_pole_height = True

            # Spawn the sign and pass the agent id of the pole the signs should be attached to
            for sign in signs_to_spawn:
                sign_agent = CustomObjectSimulationAgent(
                    asset_name=sign["sign_name"], lock_to_ground=False
                ).set_behaviour(
                    behaviour=TrafficSignAttachToPoleBehavior(
                        random_seed=self._random_seed,
                        parent_pole_id=pole_agent_id,
                        all_signs_on_pole_metadata=signs_to_spawn,
                        sign_spacing=self._sign_spacing,
                    )
                )

                agents.append(sign_agent)

        return agents

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
        )
