from abc import abstractmethod
from typing import Any, Dict, Generator, Generic, List, Optional, TypeVar

from paralleldomain import Dataset, Scene
from paralleldomain.model.type_aliases import SceneName
from paralleldomain.model.unordered_scene import UnorderedScene

S = TypeVar("S", UnorderedScene, Scene)


class DatasetPipelineEncoder(Generic[S]):
    def __init__(
        self,
        dataset: Dataset,
        scene_names: Optional[List[str]] = None,
        set_start: Optional[int] = None,
        set_stop: Optional[int] = None,
    ):
        self._dataset = dataset

        if scene_names is not None:
            for sn in scene_names:
                if sn not in self._dataset.unordered_scene_names:
                    raise KeyError(f"{sn} could not be found in dataset {self._dataset.name}")
            self._scene_names = scene_names
        else:
            set_slice = slice(set_start, set_stop)
            self._scene_names = self._dataset.unordered_scene_names[set_slice]

    def encode_dataset(self):
        for scene_name in self._scene_names:
            scene = self._dataset.get_unordered_scene(scene_name=scene_name)
            self._encode_scene(scene=scene, source_generator=self._pipeline_source_generator(scene=scene))

    @abstractmethod
    def _encode_scene(self, scene: S, source_generator: Generator[Dict[str, Any], None, None]):
        pass

    @abstractmethod
    def _pipeline_source_generator(self, scene: S) -> Generator[Dict[str, Any], None, None]:
        pass
