import random
from typing import Tuple

from pd.data_lab import ScenarioCreator, ScenarioSource
from pd.data_lab.scenario import Lighting

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.config.map import LaneSegment
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, SensorExtrinsic
from paralleldomain.data_lab.generators.road_signs import SignGenerator
from paralleldomain.data_lab.generators.single_frame import (
    SingleFrameEgoGenerator,
    SingleFrameNonEgoVehicleGenerator,
    SingleFrameVehicleBehaviorType,
)
from paralleldomain.utilities.logging import setup_loggers

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])


class SingleFrameRoadSigns(ScenarioCreator):
    def create_scenario(
        self, random_seed: int, scene_index: int, number_of_scenes: int, location: data_lab.Location, **kwargs
    ) -> ScenarioSource:
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
                        yaw=00.0,
                        x=0.0,
                        y=0.0,
                        z=2.0,
                    ),
                )
            ]
        )

        # Create scenario
        scenario = data_lab.Scenario(sensor_rig=sensor_rig)

        # Set weather variables and time of day
        scenario.environment.rain.set_constant_value(0.0)
        scenario.environment.fog.set_uniform_distribution(min_value=0.1, max_value=0.3)
        scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

        # Place ourselves in the world
        scenario.add_ego(
            generator=SingleFrameEgoGenerator(
                lane_type=LaneSegment.LaneType.DRIVABLE,
                ego_asset_name="suv_medium_02",
                random_seed=scenario.random_seed,
                sensor_rig=sensor_rig,
            )
        )

        scenario.add_agents(
            generator=SingleFrameNonEgoVehicleGenerator(
                number_of_vehicles=10,
                random_seed=scenario.random_seed,
                spawn_radius=50.0,
                vehicle_behavior_type=SingleFrameVehicleBehaviorType.TRAFFIC,
            )
        )

        scenario.add_agents(
            generator=SignGenerator(
                num_sign_poles=30,
                max_signs_per_pole=3,
                country="Portugal",
                radius=40.0,
                forward_offset_to_place_signs=40.0,
                min_distance_between_signs=1.5,
                single_frame_mode=True,
                random_seed=scenario.random_seed,
                orient_signs_facing_travel_direction=False,
            )
        )

        return scenario

    def get_location(
        self, random_seed: int, scene_index: int, number_of_scenes: int, **kwargs
    ) -> Tuple[data_lab.Location, Lighting]:
        return data_lab.Location(name="SF_6thAndMission_medium"), "day_partlyCloudy_03"


if __name__ == "__main__":
    data_lab.preview_scenario(
        scenario_creator=SingleFrameRoadSigns(),
        frames_per_scene=10,
        sim_capture_rate=10,
        random_seed=random.randint(0, 100000),
        instance_name="<instance name>",
    )
