from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List, Dict, Union
import logging
import ujson as json
from cloudpathlib import CloudPath

from paralleldomain.scene import Scene

logger = logging.getLogger(__name__)

class Dataset:
    def __init__(self, scene_dataset: Dict, dataset_path: str = "."):
        self._scenes: Dict[str, Scene] = dict()
        self._dataset_path = CloudPath(dataset_path)
        self.metadata = DatasetMeta.from_dict(scene_dataset["metadata"])
        self._scene_names: List[str] = scene_dataset["scene_splits"]["0"]["filenames"]

    def _load_scene(self, scene_name: str):
        if scene_name not in self._scenes:
            with (self._dataset_path / scene_name).open("r") as f:
                scene_data = json.load(f)
                scene = Scene.from_dict(scene_data=scene_data, dataset_path=self._dataset_path)
                self.scenes[scene.name] = scene

    @property
    def scene_names(self) -> List[str]:
        return self._scene_names

    @property
    def scenes(self) -> Dict[str, Scene]:
        for scene_name in self._scene_names:
            self._load_scene(scene_name=scene_name)
        return self._scenes

    def get_scene(self, scene_name: str) -> Scene:
        self._load_scene(scene_name=scene_name)
        return self.scenes[scene_name]

    @staticmethod
    def from_path(dataset_path: Union[str, CloudPath]) -> "Dataset":
        dataset_cloud_path: CloudPath = CloudPath(dataset_path)
        scene_json_path: CloudPath = dataset_cloud_path / "scene_dataset.json"
        if not scene_json_path.exists():
            files_with_prefix = [name.name for name in dataset_cloud_path.iterdir() if "scene_dataset" in name.name]
            if len(files_with_prefix) == 0:
                logger.error(f"No scene_dataset.json or file starting with scene_dataset found under {dataset_cloud_path}!")
            scene_json_path: CloudPath = dataset_cloud_path / files_with_prefix[-1]

        with scene_json_path.open(mode="r") as f:
            scene_dataset = json.load(f)
        return Dataset(scene_dataset=scene_dataset, dataset_path=dataset_path)


@dataclass_json
@dataclass
class DatasetMeta:
    origin: str
    name: str
    creator: str
    available_annotation_types: List[int]
    creation_date: str
    version: str
    description: str
