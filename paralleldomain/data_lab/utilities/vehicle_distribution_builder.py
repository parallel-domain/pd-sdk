from typing import Dict, List, Iterable

from pd.assets import DataVehicle, ObjAssets
from pd.core import PdError
from pd.data_lab.config.distribution import VehicleCategoryWeight
from pd.data_lab.context import setup_datalab, get_datalab_context
from pd.internal.assets.asset_registry import (
    DataVehicleTypeSpawnChance,
    UtilVehicleTypes,
    UtilAssetCategories,
)
from peewee import ModelSelect


def _get_vehicle_name_to_class_from_db() -> ModelSelect:
    """Returns a query that associates vehicle name to it's class."""
    query = (
        ObjAssets.select(ObjAssets.name, UtilVehicleTypes.name.alias("vehicle_type"))
        .join(DataVehicle, on=(ObjAssets.id == DataVehicle.asset_id))
        .join(UtilVehicleTypes, on=(UtilVehicleTypes.id == DataVehicle.vehicle_type_id))
    )
    return query


def _get_vehicle_class_weights_from_db() -> ModelSelect:
    """Returns default vehicle class weights from asset db."""
    query = UtilVehicleTypes.select(UtilVehicleTypes.name, DataVehicleTypeSpawnChance.spawn_chance).join(
        DataVehicleTypeSpawnChance,
        on=(DataVehicleTypeSpawnChance.vehicle_type_id == UtilVehicleTypes.id),
    )
    return query


def _get_vehicle_weights_from_db() -> ModelSelect:
    """Gets the default spawn chances for each vehicle from the asset db"""
    query = (
        DataVehicle.select(ObjAssets.name, DataVehicle.spawn_chance)
        .join(ObjAssets, on=(DataVehicle.asset_id == ObjAssets.id))
        .switch(DataVehicle)
        .join(UtilVehicleTypes, on=(DataVehicle.vehicle_type == UtilVehicleTypes.id))
        .switch(ObjAssets)
        .join(UtilAssetCategories)
        .where(UtilAssetCategories.name == "vehicle")
    )
    return query


class VehicleDistributionBuilder:
    """Vehicle distribution builder object.

    A helper class for building vehicle distributions that are compatible with the PD SDK, and Data Lab in particular.

    Example:
        builder = VehicleDistributionBuilder()
        builder.initialize_from_defaults()
        builder.set_vehicle_class_weights({"BUS": 10.0})
        scenario.add_agents(
            generator=ParkedVehicleGeneratorParameters(
                spawn_probability=CenterSpreadConfig(center=0.99),
                position_request=PositionRequest(
                    location_relative_position_request=LocationRelativePositionRequest(
                        agent_tags=["EGO"],
                        max_spawn_radius=100.0,
                    )
                ),
                vehicle_distribution=builder.get_configuration(),
            )
        )

    """

    def __str__(self):
        return str(self._full_vehicle_dist_config_dict)

    def __init__(self):
        try:
            get_datalab_context()
        except PdError:
            raise PdError(
                "Could not initialize vehicle distribution builder. Please ensure you have called setup_datalab()."
            )

        self._vehicle_class_weights: Dict[str, float] = {}
        self._vehicle_individual_weights: Dict[str, float] = {}
        self._full_vehicle_dist_config_dict: Dict[str, VehicleCategoryWeight] = {}
        self._pd_vehicle_name_to_class_dict: Dict[str, str] = dict(_get_vehicle_name_to_class_from_db().tuples())
        self._pd_vehicle_weights_dict: Dict[str, float] = dict(_get_vehicle_weights_from_db().tuples())
        self._pd_class_weights_dict: Dict[str, float] = dict(_get_vehicle_class_weights_from_db().tuples())

    def _update_full_configuration(self):
        """
        Updates the full vehicle distribution based on current vehicle weights and class weights
        """
        vehicle_config = dict()

        for vehicle_type in self._vehicle_class_weights.keys():
            vehicle_weights = {
                v: self._vehicle_individual_weights[v]
                for v in self._vehicle_individual_weights.keys()
                if self._pd_vehicle_name_to_class_dict[v] == vehicle_type
            }
            vehicle_config[vehicle_type] = {
                "model_weights": vehicle_weights,
                "weight": self._vehicle_class_weights[vehicle_type],
            }
        self._full_vehicle_dist_config_dict = vehicle_config

    def _validate_inputs(self):
        """Validates user input to make sure inputs are contained in our asset db."""

        # Ensure vehicle classes are valid
        if not set(self._vehicle_class_weights.keys()).issubset(set(self._pd_class_weights_dict.keys())):
            invalid_class_names = set(self._vehicle_class_weights.keys()).difference(self._pd_class_weights_dict.keys())
            raise Warning(
                "Provided vehicle class distribution contains unknown class names. Invalid class names:"
                f" {invalid_class_names}"
            )

        # Ensure vehicle names are valid
        if not set(self._vehicle_individual_weights.keys()).issubset(set(self._pd_vehicle_weights_dict.keys())):
            invalid_vehicle_names = set(self._vehicle_individual_weights.keys()).difference(
                self._pd_vehicle_weights_dict.keys()
            )
            raise Warning(
                "Provided vehicle distribution contains unknown vehicle names. Invalid vehicle names:"
                f" {invalid_vehicle_names}"
            )

    def remove_vehicles(self, vehicle_names: Iterable[str]):
        """
        Sets specified vehicle weights to 0

        Args:
            vehicle_names:

        """
        vehicle_names = set(vehicle_names)
        self._vehicle_individual_weights = {
            vehicle: 0 if vehicle in vehicle_names else weight
            for vehicle, weight in self._vehicle_individual_weights.items()
        }
        self._update_full_configuration()

    def remove_vehicle_classes(self, vehicle_classes: Iterable[str]):
        """
        Set specified vehicle classes to 0
        """
        vehicle_classes = set(vehicle_classes)
        self._vehicle_individual_weights = {
            vehicle: 0 if vehicle in vehicle_classes else weight
            for vehicle, weight in self._vehicle_individual_weights.items()
        }
        self._update_full_configuration()

    def set_vehicle_weights(self, vehicle_weights: Dict[str, float]):
        """

        Args:
            vehicle_weights: dictionary mapping vehicle model names to desired weight

        """
        self._vehicle_individual_weights.update(vehicle_weights)
        self._validate_inputs()
        self._update_full_configuration()

    def set_vehicle_class_weights(self, class_weights: Dict[str, float]):
        """

        Args:
            class_weights: dictionary mapping vehicle class to desired weight

        """
        self._vehicle_class_weights.update(class_weights)
        self._validate_inputs()
        self._update_full_configuration()

    def initialize_from_defaults(self):
        """
        Initializes a vehicle distribution directly from our asset db. ie from pd default values
        """
        self._vehicle_class_weights = self._pd_class_weights_dict
        self._vehicle_individual_weights = self._pd_vehicle_weights_dict
        self._update_full_configuration()

    def get_configuration(self) -> Dict[str, VehicleCategoryWeight]:
        """
        Returns a vehicle distribution in the format expected by Data Lab for configuring a vehicle distribution
        """
        vehicle_dist = dict()
        self._update_full_configuration()
        for vehicle_class, weight_info in self._full_vehicle_dist_config_dict.items():
            category_weight = VehicleCategoryWeight(
                model_weights=weight_info["model_weights"], weight=weight_info["weight"]
            )
            vehicle_dist[vehicle_class] = category_weight

        return vehicle_dist

    @staticmethod
    def get_vehicle_names_from_class(vehicle_class: str) -> List[str]:
        """
        Helper method to retrieve all vehicle names from specified vehicle class
        """
        pd_vehicle_name_to_class_dict = dict(_get_vehicle_name_to_class_from_db().tuples())
        vehicle_names = [
            vehicle
            for vehicle, vehicle_class_name in pd_vehicle_name_to_class_dict.items()
            if vehicle_class_name == vehicle_class
        ]
        return vehicle_names

    @property
    def class_weights(self) -> Dict[str, float]:
        return self._vehicle_class_weights

    @property
    def vehicle_weights(self) -> Dict[str, float]:
        return self._vehicle_individual_weights
