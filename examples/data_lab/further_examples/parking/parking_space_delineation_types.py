import random
from typing import Tuple

from pd.data_lab import Scenario, ScenarioCreator, ScenarioSource
from pd.data_lab.config.distribution import ContinousUniformDistribution, EnumDistribution
from pd.data_lab.scenario import Lighting

import paralleldomain.data_lab
from paralleldomain import data_lab
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, DistortionParams, SensorExtrinsic
from paralleldomain.data_lab.config.world import EnvironmentParameters, ParkingSpaceData
from paralleldomain.data_lab.generators.behavior import VehicleBehavior
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.position_request import LaneSpawnPolicy, PositionRequest
from paralleldomain.data_lab.generators.spawn_data import VehicleSpawnData
from paralleldomain.utilities.logging import setup_loggers

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])


class ParkingSpaceDelineationTypes(ScenarioCreator):
    def create_scenario(
        self,
        random_seed: int,
        scene_index: int,
        number_of_scenes: int,
        location: paralleldomain.data_lab.Location,
        **kwargs,
    ) -> ScenarioSource:
        sensor_rig = paralleldomain.data_lab.SensorRig(
            sensor_configs=[
                paralleldomain.data_lab.SensorConfig(
                    display_name="Ortho_BEV",
                    camera_intrinsic=CameraIntrinsic(
                        width=1920,
                        height=1920,
                        distortion_params=DistortionParams(
                            fx=150.0, fy=150.0, cx=960.0, cy=960.0, p1=-200, p2=300, skew=0, fisheye_model=6
                        ),
                    ),
                    sensor_extrinsic=SensorExtrinsic(
                        roll=0.0,
                        pitch=-90.0,
                        yaw=0.0,
                        x=0.0,
                        y=0.0,
                        z=100,
                    ),
                ),
            ]
        )

        # Create scenario
        scenario = Scenario(sensor_rig=sensor_rig)

        # Set weather variables and time of day
        scenario.environment.rain.set_constant_value(0.0)
        scenario.environment.fog.set_uniform_distribution(min_value=0.1, max_value=0.3)
        scenario.environment.wetness.set_uniform_distribution(min_value=0.1, max_value=0.3)

        # Set up what type of parking space delineations we want to see (currently set to be random)
        # but can be set to a specific delineation type from the list below
        parking_space_delineation_type = random.choice(
            [
                "BOX_CLOSED",
                "BOX_OPEN_CURB",
                "BOX_DOUBLE",
                "SINGLE_SQUARED_OPEN_CURB",
                "DOUBLE_ROUND_50CM_GAP",
                "DOUBLE_ROUND_50CM_GAP_OPEN_CURB",
                "DOUBLE_SQUARED_50CM_GAP_OPEN_CURB",
                "T_FULL",
                "T_SHORT",
            ]
        )

        scenario.add_ego(
            generator=EgoAgentGeneratorParameters(
                agent_type=AgentType.VEHICLE,
                vehicle_spawn_data=VehicleSpawnData(
                    vehicle_behavior=VehicleBehavior(
                        parking_scenario_goal=PositionRequest(
                            lane_spawn_policy=LaneSpawnPolicy(
                                lane_type=EnumDistribution(
                                    probabilities={"ParkingSpace": 1.0},
                                ),
                                road_type=EnumDistribution(
                                    probabilities={"Parking_Aisle": 1.0},
                                ),
                                on_road_parking_angle_distribution=EnumDistribution(
                                    probabilities={
                                        "PERPENDICULAR": 1.0,
                                    }
                                ),
                            )
                        ),
                        parking_scenario_time=ContinousUniformDistribution(
                            min=4.0,
                            max=4.0,
                        ),
                    )
                ),
            ),
        )

        scenario.set_environment(
            parameters=EnvironmentParameters(
                parking_space_data=ParkingSpaceData(
                    parking_lot_angle_distribution=EnumDistribution(
                        probabilities={
                            "PERPENDICULAR": 1.0,
                        }
                    ),
                    lot_parking_delineation_type=EnumDistribution(
                        probabilities={
                            parking_space_delineation_type: 1.0,
                        }
                    ),
                    street_parking_delineation_type=EnumDistribution(
                        probabilities={
                            parking_space_delineation_type: 1.0,
                        }
                    ),
                    street_parking_angle_zero_override=EnumDistribution(
                        probabilities={
                            parking_space_delineation_type: 1.0,
                        }
                    ),
                )
            )
        )

        return scenario

    def get_location(
        self, random_seed: int, scene_index: int, number_of_scenes: int, **kwargs
    ) -> Tuple[paralleldomain.data_lab.Location, Lighting]:
        return paralleldomain.data_lab.Location(name="SF_VanNessAveAndTurkSt"), "dusk_partlyCloudy_02"


if __name__ == "__main__":
    data_lab.preview_scenario(
        scenario_creator=ParkingSpaceDelineationTypes(),
        frames_per_scene=100,
        sim_capture_rate=10,
        random_seed=random.randint(0, 100000),
        instance_name="<instance name>",
    )
