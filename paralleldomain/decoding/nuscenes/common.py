import json
import os
from collections import defaultdict
from threading import RLock
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import psutil
from pyquaternion import Quaternion

from paralleldomain.decoding.common import create_cache_key
from paralleldomain.model.class_mapping import ClassDetail
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.coordinate_system import INTERNAL_COORDINATE_SYSTEM, CoordinateSystem
from paralleldomain.utilities.lazy_load_cache import LazyLoadCache

NUSCENES_IMU_TO_INTERNAL_CS = CoordinateSystem("FLU") > INTERNAL_COORDINATE_SYSTEM
# NUSCENES_TO_INTERNAL_CS = CoordinateSystem("RFU") > INTERNAL_COORDINATE_SYSTEM


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


_NU_SCENES_DATA_MAX_SIZE = 5.0e9  # GB
cache_max_ram_usage_factor = float(
    os.environ.get("NU_CACHE_MAX_USAGE_FACTOR", _NU_SCENES_DATA_MAX_SIZE / psutil.virtual_memory().total)
)  # use 2.0 GB by default
ram_keep_free_factor = float(os.environ.get("NU_CACHE_KEEP_FREE_FACTOR", 0.05))  # 5% have to stay free

NU_SC_DATA_CACHE = None


class NuScenesDataAccessMixin:
    _init_lock = RLock()

    def __init__(self, dataset_path: AnyPath, dataset_name: str, split_name: str):
        """Decodes a NuScenes dataset

        Args:
            dataset_path: AnyPath to the root folder of a NuScenes dataset.
            split: Split to use within this dataset. Defaults to v1.0-train.
            Options are [v1.0-mini, v1.0-test, v1.0-train, v1.0-val].
        """
        self.dataset_name = dataset_name
        self._dataset_path = dataset_path
        self.split_name = split_name

    @property
    def nu_lazy_load_cache(self) -> LazyLoadCache:
        global NU_SC_DATA_CACHE
        if NU_SC_DATA_CACHE is None:
            with self._init_lock:
                if NU_SC_DATA_CACHE is None:
                    NU_SC_DATA_CACHE = LazyLoadCache(
                        max_ram_usage_factor=cache_max_ram_usage_factor,
                        ram_keep_free_factor=ram_keep_free_factor,
                        cache_name="NuScenes Cache",
                    )
        return NU_SC_DATA_CACHE

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
        _unique_cache_key = self.get_unique_id(extra="nu_logs")

        return self.nu_lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: load_table(dataset_root=self._dataset_path, table_name="log", split_name=self.split_name),
        )

    @property
    def nu_logs_by_log_token(self) -> Dict[str, Dict[str, Any]]:
        return {log["token"]: log for log in self.nu_logs}

    @property
    def nu_instance(self) -> List[Dict[str,Any]]: 
        _unique_cache_key = self.get_unique_id(extra="nu_instance")

        return self.nu_lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: load_table(dataset_root=self._dataset_path, table_name="instance", split_name=self.split_name),
        )
    
    @property
    def nu_map(self) -> List[Dict[str,Any]]: 
        _unique_cache_key = self.get_unique_id(extra="nu_map")

        return self.nu_lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: load_table(dataset_root=self._dataset_path, table_name="map", split_name=self.split_name),
        )

    @property
    def nu_sample_annotation(self) -> List[Dict[str,Any]]: 
        _unique_cache_key = self.get_unique_id(extra="nu_sample_annotation")

        return self.nu_lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: load_table(dataset_root=self._dataset_path, table_name="sample_annotation", split_name=self.split_name),
        )

    @property
    def nu_scene(self) -> List[Dict[str,Any]]: 
        _unique_cache_key = self.get_unique_id(extra="nu_scene")

        return self.nu_lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: load_table(dataset_root=self._dataset_path, table_name="scene", split_name=self.split_name),
        )

    @property
    def nu_scene_by_scene_token(self) -> Dict[str, Dict[str, Any]]:
        return {scene["token"]: scene for scene in self.nu_scene}

    @property
    def nu_visibility(self) -> List[Dict[str,Any]]: 
        _unique_cache_key = self.get_unique_id(extra="nu_visibility")

        return self.nu_lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: load_table(dataset_root=self._dataset_path, table_name="visibility", split_name=self.split_name),
        )

    ### MHS: nuscenes uses scene_tokens instead of log_tokens, so we'll index samples by scene_token 
    @property
    def nu_samples(self) -> Dict[str, List[Dict[str, Any]]]:
        _unique_cache_key = self.get_unique_id(extra="nu_samples")

        def get_nu_samples() -> Dict[str, List[Dict[str, Any]]]:
            samples = load_table(dataset_root=self._dataset_path, table_name="sample", split_name=self.split_name)
            scene_wise_samples = dict()
            for s in samples:
                scene_wise_samples.setdefault(s["scene_token"], list()).append(s)  
            return scene_wise_samples

        return self.nu_lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=get_nu_samples,
        )

    @property
    def nu_sensors(self) -> List[Dict[str, Any]]:
        _unique_cache_key = self.get_unique_id(extra="nu_sensors")

        return self.nu_lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: load_table(dataset_root=self._dataset_path, table_name="sensor", split_name=self.split_name),
        )

    def get_nu_sensor(self, sensor_token: str) -> Dict[str, Any]:
        return next(iter([sensor for sensor in self.nu_sensors if sensor["token"] == sensor_token]), dict())

    @property
    def nu_samples_data(self) -> Dict[str, List[Dict[str, Any]]]:
        ''' Because there are multiple sensors in nuScenes, multiple sample_data entries may be mapped 
        to a single timestamp (or frame). Most of the use of the nu_smaples_data dict in nuImages uses key_camera_token, 
        a foreign key that isn't available in nuScenes. This dictionary enables fast lookup of all sample_data for a
        given sample_token.
        '''
        _unique_cache_key = self.get_unique_id(extra="nu_samples_data")

        def get_nu_samples_data() -> Dict[str, Dict[str, Any]]:
            data = load_table(dataset_root=self._dataset_path, table_name="sample_data", split_name=self.split_name)
            out_dict = defaultdict(list)
            [out_dict[d['sample_token']].append(d) for d in data if d['is_key_frame']==True]
            return out_dict

        return self.nu_lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=get_nu_samples_data,
        )

    ### MHS: surface_ann.json not in nuScenes
    # @property
    # def nu_surface_ann(self) -> Dict[str, List[Dict[str, Any]]]:
    #     _unique_cache_key = self.get_unique_id(extra="nu_surface_ann")

    #     def get_nu_surface_ann() -> Dict[str, List[Dict[str, Any]]]:
    #         data = load_table(dataset_root=self._dataset_path, table_name="surface_ann", split_name=self.split_name)
    #         surface_ann = dict()
    #         for d in data:
    #             surface_ann.setdefault(d["sample_data_token"], list()).append(d)
    #         return surface_ann

    #     return self.nu_lazy_load_cache.get_item(
    #         key=_unique_cache_key,
    #         loader=get_nu_surface_ann,
    #     )

    # @property
    # def nu_object_ann(self) -> Dict[str, List[Dict[str, Any]]]:
    #     _unique_cache_key = self.get_unique_id(extra="nu_object_ann")

    #     def get_nu_object_ann() -> Dict[str, List[Dict[str, Any]]]:
    #         data = load_table(dataset_root=self._dataset_path, table_name="object_ann", split_name=self.split_name)
    #         object_ann = dict()
    #         for d in data:
    #             object_ann.setdefault(d["sample_data_token"], list()).append(d)
    #         return object_ann

    #     return self.nu_lazy_load_cache.get_item(
    #         key=_unique_cache_key,
    #         loader=get_nu_object_ann,
    #     )

    # @property
    # def nu_sample_data_tokens_to_available_anno_types(self) -> Dict[str, Tuple[bool, bool]]:
    #     _unique_cache_key = self.get_unique_id(extra="nu_sample_data_tokens_to_available_anno_types")

    #     def get_nu_sample_data_tokens_to_available_anno_types() -> Dict[str, Tuple[bool, bool]]:
    #         surface_anns = self.nu_surface_ann   
    #         obj_anns = self.nu_object_ann
    #         mapping = dict()
    #         for k in surface_anns.keys():
    #             mapping.setdefault(k, [False, False])[0] = True
    #         for k in obj_anns.keys():
    #             mapping.setdefault(k, [False, False])[1] = True
    #         return mapping

    #     return self.nu_lazy_load_cache.get_item(
    #         key=_unique_cache_key,
    #         loader=get_nu_sample_data_tokens_to_available_anno_types,
    #     )

    @property
    def nu_calibrated_sensors(self) -> Dict[str, Dict[str, Any]]:
        _unique_cache_key = self.get_unique_id(extra="nu_calibrated_sensors")

        def get_nu_calibrated_sensors() -> Dict[str, Dict[str, Any]]:
            data = load_table(
                dataset_root=self._dataset_path, table_name="calibrated_sensor", split_name=self.split_name
            )
            return {d["token"]: d for d in data}

        return self.nu_lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=get_nu_calibrated_sensors,
        )

    @property
    def nu_ego_pose(self) -> List[Dict[str, Any]]:
        _unique_cache_key = self.get_unique_id(extra="nu_ego_pose")

        return self.nu_lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: load_table(
                dataset_root=self._dataset_path, table_name="ego_pose", split_name=self.split_name
            ),
        )

    def get_nu_ego_pose(self, ego_pose_token: str) -> Dict[str, Any]:
        return next(iter([pose for pose in self.nu_ego_pose if pose["token"] == ego_pose_token]), dict())

    @property
    def nu_category(self) -> Dict[str, Dict[str, Any]]:
        _unique_cache_key = self.get_unique_id(extra="nu_category")

        def get_nu_category() -> Dict[str, Dict[str, Any]]:
            data = load_table(dataset_root=self._dataset_path, table_name="category", split_name=self.split_name)
            return {d["token"]: d for d in data}

        return self.nu_lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=get_nu_category,
        )

    @property
    def nu_attribute(self) -> Dict[str, Dict[str, Any]]:
        _unique_cache_key = self.get_unique_id(extra="nu_attribute")

        def get_nu_attribute() -> Dict[str, Dict[str, Any]]:
            data = load_table(dataset_root=self._dataset_path, table_name="attribute", split_name=self.split_name)
            return {d["token"]: d for d in data}

        return self.nu_lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=get_nu_attribute,
        )

    @property
    def nu_name_to_index(self) -> Dict[str, int]:
        _unique_cache_key = self.get_unique_id(extra="nu_name_to_index")

        return self.nu_lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=lambda: name_to_index_mapping(category=list(self.nu_category.values())),
        )
    # ### MHS: nuScenes already labels "prev" and "next" frames in sample.json
    # ### Note: These functions gather the non-keyframe sample_data associated with the key_camera_token sample_data

    # def _get_prev_data_ids(self, key_camera_token: str) -> List[str]:
    #     sample_data = self.nu_samples_data[key_camera_token]
    #     tokens = [key_camera_token]
    #     if sample_data["prev"]:
    #         tokens += self._get_prev_data_ids(key_camera_token=sample_data["prev"])
    #     return tokens

    # def _get_next_data_ids(self, key_camera_token: str) -> List[str]:
    #     sample_data = self.nu_samples_data[key_camera_token]
    #     tokens = [key_camera_token]
    #     if sample_data["next"]:
    #         tokens += self._get_prev_data_ids(key_camera_token=sample_data["next"])
    #     return tokens

    # ### MHS: We aren't using this function, but it's easy to rewrite since 'prev' and 'next' are in sample.json
    # def get_connected_sample_data_ids(self, key_camera_token: str):
    #     prev_ids = self._get_prev_data_ids(key_camera_token=key_camera_token)
    #     next_ids = self._get_next_data_ids(key_camera_token=key_camera_token)
    #     return list(set(next_ids + prev_ids))

    ### MHS: this function now returns a list of sample_data objects
    def get_sample_with_frame_id(self, scene_token: str, frame_id: FrameId) -> Dict[str,Any]:
        samples = self.nu_samples[scene_token]
        for sample in samples:
            if sample['token'] == frame_id:
                return sample
        
    
    def get_sample_data_with_frame_id(self, scene_token: str, frame_id: FrameId) -> Generator[List[Dict[str, Any]], None, None]:
        samples = self.nu_samples[scene_token]
        for sample in samples:
            if sample['token'] == frame_id:
                sample_token = sample['token']
                data = self.nu_samples_data[sample_token]
                yield data

    def get_sample_data_id_frame_id_and_sensor_name(
        self, scene_token: str, frame_id: FrameId, sensor_name: SensorName
    ) -> Optional[str]:
        return self.nu_sample_data_ids_by_frame_and_sensor(scene__token=scene_token)[(frame_id, sensor_name)]

    def nu_sample_data_ids_by_frame_and_sensor(self, scene_token: str) -> Dict[Tuple[FrameId, SensorName], str]:
        _unique_cache_key = self.get_unique_id(extra="nu_sample_data_ids_by_frame_and_sensor", scene_name=scene_token)

        def get_nu_sample_data_ids_by_frame_and_sensor() -> Dict[Tuple[FrameId, SensorName], str]:
            samples = self.nu_samples[scene_token]
            sample_tokens = [sample["sample_token"] for sample in samples]

            mapping = dict()
            nu_samples_data = self.nu_samples_data
            for sample_token in sample_tokens:
                data = nu_samples_data[sample_token]
                # Loop through all sample_data objects in data
                for d in data:
                    frame_id = sample_token
                    calib_sensor_token = d["calibrated_sensor_token"]
                    calib_sensor = self.nu_calibrated_sensors[calib_sensor_token]
                    sensor = self.get_nu_sensor(sensor_token=calib_sensor["sensor_token"])
                    sensor_name = sensor["channel"]
                    mapping[(frame_id, sensor_name)] = sample_token
            return mapping

        return self.nu_lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=get_nu_sample_data_ids_by_frame_and_sensor,
        )

    @property
    def nu_class_infos(self) -> List[ClassDetail]:
        _unique_cache_key = self.get_unique_id(extra="nu_class_infos")

        def get_nu_class_infos() -> List[ClassDetail]:
            name_to_index = name_to_index_mapping(category=list(self.nu_category.values()))
            details = list()
            for _, cat in self.nu_category.items():
                name = cat["name"]
                index = name_to_index[name]
                details.append(ClassDetail(name=name, id=index, meta=dict(description=cat["description"])))
            details.append(ClassDetail(name="background", id=name_to_index["background"], meta=dict()))
            return details

        return self.nu_lazy_load_cache.get_item(
            key=_unique_cache_key,
            loader=get_nu_class_infos,
        )
    ### MHS: Since there are multiple sample_data objects per sample, they all should have very similar ego_poses so I just take the first one.
    def get_ego_pose(self, scene_token: str, frame_id: FrameId) -> List[np.ndarray]:
        for data in next(iter(self.get_sample_data_with_frame_id(scene_token=scene_token, frame_id=frame_id))):
            ego_pose_token = data["ego_pose_token"]
            ego_pose = self.get_nu_ego_pose(ego_pose_token=ego_pose_token)
            trans = np.eye(4)

            trans[:3, :3] = Quaternion(ego_pose["rotation"]).rotation_matrix
            trans[:3, 3] = np.array(ego_pose["translation"])
            trans = NUSCENES_IMU_TO_INTERNAL_CS @ trans
            return trans
        raise ValueError(f"No ego pose for frame id {frame_id}")


NUSCENES_CLASSES = list()


def name_to_index_mapping(category: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Build a mapping from name to index to look up index in O(1) time.
    :param category: The nuImages category table.
    :return: The mapping from category name to category index.
    """
    # The 0 index is reserved for non-labelled background; thus, the categories should start from index 1.
    # Also, sort the categories before looping so that the order is always the same (alphabetical).
    name_to_index = dict()
    i = 1
    sorted_category: List = sorted(category.copy(), key=lambda k: k["name"])
    for c in sorted_category:
        # Ignore the vehicle.ego and flat.driveable_surface classes first; they will be mapped later.
        if c["name"] != "vehicle.ego" and c["name"] != "flat.driveable_surface":
            name_to_index[c["name"]] = i
            i += 1

    assert max(name_to_index.values()) < 24, (
        "Error: There are {} classes (excluding vehicle.ego and flat.driveable_surface), "
        "but there should be 23. Please check your category.json".format(max(name_to_index.values()))
    )

    # Now map the vehicle.ego and flat.driveable_surface classes.
    name_to_index["flat.driveable_surface"] = 24
    name_to_index["vehicle.ego"] = 31
    name_to_index["background"] = 0

    # Ensure that each class name is uniquely paired with a class index, and vice versa.
    assert len(name_to_index) == len(
        set(name_to_index.values())
    ), "Error: There are {} class names but {} class indices".format(
        len(name_to_index), len(set(name_to_index.values()))
    )

    return name_to_index
