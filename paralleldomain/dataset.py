from dataclasses import dataclass
from dataclasses_json import dataclass_json
from .scene import Scene
from typing import List, Dict
import ujson as json


class Dataset:
    def __init__(self, scene_dataset: Dict, path: str = "."):
        self.scenes = {}
        self._path = path
        self.metadata = DatasetMeta.from_dict(scene_dataset["metadata"])
        self.load_scenes(scene_dataset["scene_splits"]["0"]["filenames"])

    def load_scenes(self, filenames: List[str]):
        for fn in filenames:
            with open(f"{self._path}/{fn}", "r") as f:
                scene_data = json.load(f)
                scene = Scene.from_dict(scene_data)
                self.scenes[scene.name] = scene

    @staticmethod
    def from_path(dataset_path: str) -> Dataset:
        with open(f"{dataset_path}/scene_dataset.json", "r") as f:
            scene_dataset = json.load(f)

        return Dataset(scene_dataset, dataset_path)


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
