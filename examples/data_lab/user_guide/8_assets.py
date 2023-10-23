import logging
import random
from typing import Tuple

from pd.data_lab import ScenarioCreator, ScenarioSource
from pd.data_lab.config.distribution import EnumDistribution
from pd.data_lab.config.location import Location
from pd.data_lab.scenario import Lighting
from pd.internal.assets.asset_registry import DataVehicle, DataVehicleTypeSpawnChance, ObjAssets, UtilVehicleTypes

import paralleldomain.data_lab as data_lab
from paralleldomain.data_lab.config.sensor_rig import CameraIntrinsic, SensorExtrinsic
from paralleldomain.data_lab.generators.ego_agent import AgentType, EgoAgentGeneratorParameters
from paralleldomain.data_lab.generators.position_request import LaneSpawnPolicy, PositionRequest
from paralleldomain.utilities.logging import setup_loggers

setup_loggers(logger_names=[__name__, "paralleldomain", "pd"])


"""
In this script, we walk through an example of how to query our asset database while creating a Data Lab scenario.

Much of the script below is boilerplate code which is not related to the asset database queries, but rather serves
to demonstrate how you can integrate these queries into you scenario generation scripts.
"""


# Create a custom ScenarioCreator class
class AssetLookupExample(ScenarioCreator):
    def create_scenario(
        self, random_seed: int, scene_index: int, number_of_scenes: int, location: Location, **kwargs
    ) -> ScenarioSource:
        # Access the ObjAssets table of the asset database and extract only assets with "trash" in the name
        asset_objs = ObjAssets.select().where(ObjAssets.name % "*trash*")

        # Show the first five trash assets found in the database
        print("First five assets found with 'trash' in name: ")
        print([asset.name for asset in asset_objs][:5])

        # Now, we again look for assets with the word "trash" in their name but add an additional filter to return only
        # assets with a combined length, width and height of greater than 1 meter.
        asset_objs = ObjAssets.select().where(
            (ObjAssets.name % "*trash*") & (ObjAssets.width + ObjAssets.height + ObjAssets.length > 1.0)
        )

        # Display the first five assets found that satisfy the above conditions
        print("First five assets found with 'trash' in name and total dimensions greater than 1 meter: ")
        print([asset.name for asset in asset_objs][:5])

        # Now we query the asset database to find vehicles, but we will utilize information contained within
        # several tables of the asset database.
        vehicle_objs = (
            # We begin with the DataVehicle table, but also pull in columns from the ObjAssets, UtilVehicleTypes
            # and DataVehicleTypeSpawnChance tables
            DataVehicle.select(
                DataVehicle.spawn_chance,
                ObjAssets.name,
                UtilVehicleTypes.name,
                DataVehicleTypeSpawnChance.spawn_chance,
            )
            # We execute a series of switch and join commands to extract data from other tables
            .join(ObjAssets, on=(DataVehicle.asset_id == ObjAssets.id))
            .switch(DataVehicle)
            .join(UtilVehicleTypes, on=(DataVehicle.vehicle_type == UtilVehicleTypes.id))
            .join(DataVehicleTypeSpawnChance, on=(DataVehicleTypeSpawnChance.vehicle_type_id == UtilVehicleTypes.id))
        )

        # We now build a dictionary of the vehicles, containing the vehicle name (the key), the vehicle type, the
        # vehicle spawn chance and the vehicle type spawn chance.

        # Extracting asset database information and storing it is a common use case within Data Lab, where information
        # about assets may be needed for subsequent logic within your Data Lab script
        vehicle_data = {
            vehicle.asset.name: {
                "type": vehicle.vehicle_type.name,
                "vehicle_spawn_chance": vehicle.spawn_chance,
                "vehicle_type_spawn_chance": vehicle.vehicle_type.datavehicletypespawnchance.spawn_chance,
            }
            for vehicle in vehicle_objs
        }

        # Visualize information about vehicles that we extracted from the asset database and stored
        print(vehicle_data)

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
        scenario_creator=AssetLookupExample(),
        random_seed=2023,
        frames_per_scene=100,
        sim_capture_rate=10,
        instance_name="<instance_name>",
    )
