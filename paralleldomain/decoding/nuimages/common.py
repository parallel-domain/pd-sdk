import json
from typing import Any, Dict, List, Optional

from paralleldomain.decoding.common import LazyLoadPropertyMixin, create_cache_key
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath


def load_table(dataset_root: AnyPath, split_name: str, table_name: str) -> List[Dict[str, Any]]:
    """
    Load a table and return it.
    :param table_name: The name of the table to load.
    :return: The table dictionary.
    """
    table_path = dataset_root / split_name / f"{table_name}.json"
    if table_path.exists():
        with table_path.open() as f:
            return json.load(f)
    raise ValueError(f"Error: Table {table_name} does not exist!")


class NuImagesDataAccessMixin(LazyLoadPropertyMixin):
    def __init__(self, dataset_path: AnyPath, dataset_name: str, split_name: str):
        """Decodes a NuImages dataset

        Args:
            dataset_path: AnyPath to the root folder of a NuImages dataset.
            split: Split to use within this dataset. Defaults to v1.0-train.
            Options are [v1.0-mini, v1.0-test, v1.0-train, v1.0-val].
        """
        self.dataset_name = dataset_name
        self._dataset_path = dataset_path
        self.split_name = split_name

    def get_unique_id(
        self,
        scene_name: Optional[SceneName] = None,
        sensor_name: Optional[SensorName] = None,
        frame_id: Optional[FrameId] = None,
        extra: Optional[str] = None,
    ) -> str:
        return create_cache_key(
            dataset_name=self.dataset_name,
            scene_name=scene_name,
            sensor_name=sensor_name,
            frame_id=frame_id,
            extra=extra,
        )

    @property
    def nu_logs(self) -> List[Dict[str, Any]]:
        _unique_cache_key = self.get_unique_id(extra="logs")

        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: load_table(dataset_root=self._dataset_path, table_name="log", split_name=self.split_name),
        )

    @property
    def nu_logs_by_log_token(self) -> Dict[str, Dict[str, Any]]:
        return {log["token"]: log for log in self.nu_logs}

    @property
    def nu_samples(self) -> List[Dict[str, Any]]:
        _unique_cache_key = self.get_unique_id(extra="samples")

        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: load_table(dataset_root=self._dataset_path, table_name="sample", split_name=self.split_name),
        )

    def get_nu_samples(self, log_token: str) -> List[Dict[str, Any]]:
        return [sample for sample in self.nu_samples if sample["log_token"] == log_token]

    @property
    def nu_sensors(self) -> List[Dict[str, Any]]:
        _unique_cache_key = self.get_unique_id(extra="sensors")

        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: load_table(dataset_root=self._dataset_path, table_name="sensor", split_name=self.split_name),
        )

    def get_nu_sensor(self, sensor_token: str) -> Dict[str, Any]:
        return next(iter([sensor for sensor in self.nu_sensors if sensor["token"] == sensor_token]), dict())

    @property
    def nu_samples_data(self) -> List[Dict[str, Any]]:
        _unique_cache_key = self.get_unique_id(extra="samples_data")

        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: load_table(
                dataset_root=self._dataset_path, table_name="sample_data", split_name=self.split_name
            ),
        )

    @property
    def nu_calibrated_sensors(self) -> List[Dict[str, Any]]:
        _unique_cache_key = self.get_unique_id(extra="calibrated_sensors")

        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: load_table(
                dataset_root=self._dataset_path, table_name="calibrated_sensor", split_name=self.split_name
            ),
        )

    @property
    def nu_ego_pose(self) -> List[Dict[str, Any]]:
        _unique_cache_key = self.get_unique_id(extra="ego_poses")

        return self.lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: load_table(
                dataset_root=self._dataset_path, table_name="ego_pose", split_name=self.split_name
            ),
        )

    def get_nu_ego_pose(self, ego_pose_token: str) -> Dict[str, Any]:
        return next(iter([pose for pose in self.nu_ego_pose if pose["token"] == ego_pose_token]), dict())
