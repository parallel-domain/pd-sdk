from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List, Dict
import ujson as json

from paralleldomain.scene import Scene


class Dataset:
    def __init__(self, scene_dataset: Dict, dataset_path: str = "."):
        self._scenes: Dict[str, Scene] = dict()
        self._dataset_path = dataset_path
        self.metadata = DatasetMeta.from_dict(scene_dataset["metadata"])
        self._scene_names: List[str] = scene_dataset["scene_splits"]["0"]["filenames"]

    def _load_scene(self, scene_name: str):
        if scene_name not in self._scenes:
            with open(f"{self._dataset_path}/{scene_name}", "r") as f:
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
    def from_path(dataset_path: str) -> "Dataset":
        with open(f"{dataset_path}/scene_dataset.json", "r") as f:
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
