from typing import Dict, Iterable, List

from pd.assets import DataVehicle, ObjAssets
from pd.core import PdError
from pd.data_lab.config.distribution import VehicleCategoryWeight
from pd.data_lab.context import get_datalab_context
from pd.internal.assets.asset_registry import DataVehicleTypeSpawnChance, UtilAssetCategories, UtilVehicleTypes
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
    """
    Helper class for building vehicle distributions compatible with PD SDK, and Data Lab

    Note:
        The class takes no parameters on initialization and loads the default vehicle distribution on initialization

    Examples::

        >>> builder = VehicleDistributionBuilder()
        >>> builder.initialize_from_defaults()
        >>> builder.set_vehicle_class_weights({"BUS": 10.0})
        >>> scenario.add_agents(
        >>>     generator=ParkedVehicleGeneratorParameters(
        >>>         spawn_probability=CenterSpreadConfig(center=0.99),
        >>>         position_request=PositionRequest(
        >>>             location_relative_position_request=LocationRelativePositionRequest(
        >>>                 agent_tags=["EGO"],
        >>>                 max_spawn_radius=100.0,
        >>>             )
        >>>         ),
        >>>         vehicle_distribution=builder.get_configuration(),
        >>>     )
        >>> )

    Raises:
        PdError: If the Data Lab context is invalid or not setup. Ensure setup_datalab() has been called prior
    """

    def __str__(self) -> str:
        """
        Returns the str of the vehicle distribution contained within the object

        Returns:
            String of the vehicle distribution contained within the object
        """
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
        """Updates the full vehicle distribution based on current vehicle weights and class weights"""
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

    def remove_vehicles(self, vehicle_names: Iterable[str]) -> None:
        """
        Allows specified vehicles to be removed from the vehicle distribution of the object

        Args:
            vehicle_names: The names of the vehicles to be removed from the vehicle distribution

        """
        vehicle_names = set(vehicle_names)
        self._vehicle_individual_weights = {
            vehicle: 0 if vehicle in vehicle_names else weight
            for vehicle, weight in self._vehicle_individual_weights.items()
        }
        self._update_full_configuration()

    def remove_vehicle_classes(self, vehicle_classes: Iterable[str]) -> None:
        """
        Allows specified classes of vehicles to be removed from the vehicle distribution of the object

        Args:
            vehicle_classes: The names of the classes of vehicles to be removed from the vehicle distribution
        """
        vehicle_classes = set(vehicle_classes)
        self._vehicle_class_weights = {
            vehicle_class: 0 if vehicle_class in vehicle_classes else weight
            for vehicle_class, weight in self._vehicle_class_weights.items()
        }
        self._update_full_configuration()

    def set_vehicle_weights(self, vehicle_weights: Dict[str, float]) -> None:
        """
        Set specified vehicles to have a certain spawn weight in the vehicle distribution of the object

        Args:
            vehicle_weights: Dictionary containing the name of the vehicles and their respective weights to be added or
                modified in the vehicle distribution of the object
        """
        self._vehicle_individual_weights.update(vehicle_weights)
        self._validate_inputs()
        self._update_full_configuration()

    def set_vehicle_class_weights(self, class_weights: Dict[str, float]) -> None:
        """
        Set specified vehicle classes to have a certain spawn weight in the vehicle distribution of the object

            Args:
                class_weights: Dictionary containing the name of the vehicles classes and their respective weights
                    to be added or modified in the vehicle distribution of the object
        """
        self._vehicle_class_weights.update(class_weights)
        self._validate_inputs()
        self._update_full_configuration()

    def initialize_from_defaults(self) -> None:
        """
        Initializes the vehicle distribution with default values
        """
        self._vehicle_class_weights = self._pd_class_weights_dict
        self._vehicle_individual_weights = self._pd_vehicle_weights_dict
        self._update_full_configuration()

    def get_configuration(self) -> Dict[str, VehicleCategoryWeight]:
        """
        Retrieves the vehicle distribution in the object in a format directly digestible by Data Lab

        Returns:
            A dictionary representation of the created vehicle distribution outlining the spawn weights of each vehicle
                class and the spawn weights of specific vehicles within each class
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
        Retrieve all vehicle names within a specified vehicle class

        Args:
            vehicle_class: The name of the vehicle class within which we wish to find all vehicle names

        Returns:
            A list of all vehicle names within the specified class
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
        """
        Dictionary containing the assigned weights of each of the vehicle classes
        """
        return self._vehicle_class_weights

    @property
    def vehicle_weights(self) -> Dict[str, float]:
        """
        Dictionary containing the assigned weights of each vehicle
        """
        return self._vehicle_individual_weights
