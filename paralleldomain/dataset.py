from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List, Dict, Union, Optional
import logging
import ujson as json
from paralleldomain.decoding.decoder import Decoder
from paralleldomain.dto import DatasetMeta, DatasetDTO

from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.scene import Scene

logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, meta_data: DatasetMeta, scene_names: List[str], decoder: Decoder):
        self._decoder = decoder
        self._scenes: Dict[str, Scene] = dict()
        # self._dataset_path = AnyPath(dataset_path)
        self.meta_data = meta_data
        self._scene_names: List[str] = scene_names

    def _load_scene(self, scene_name: str):
        if scene_name not in self._scenes:
            dto = self._decoder.decode_scene(scene_name=scene_name)
            self._scenes[scene_name] = Scene(name=scene_name,
                                             description=dto.description,
                                             decoder=self._decoder,
                                             samples=dto.samples)
            # self._scenes[scene_name] = Scene.from_file(dataset_path=self._dataset_path,
            #                                             scene_name=scene_name)

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
        return self._scenes[scene_name]

    @staticmethod
    def from_decoder(decoder: Optional[Decoder]) -> "Dataset":
        dto = decoder.decode_dataset()
        return Dataset(meta_data=dto.meta_data, scene_names=dto.scene_names, decoder=decoder)
