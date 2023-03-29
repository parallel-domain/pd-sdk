import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar, Union

from paralleldomain import Scene
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.sensor import CameraSensorFrame, LidarSensorFrame, RadarSensorFrame
from paralleldomain.model.type_aliases import SceneName
from paralleldomain.model.unordered_scene import UnorderedScene
from paralleldomain.utilities.any_path import AnyPath


@dataclass
class InMemoryDatasetDecoder:
    unordered_scene_names: List[SceneName]
    scene_names: List[SceneName]
    metadata: DatasetMeta = field(default_factory=DatasetMeta)
    scenes: Dict[SceneName, Union[UnorderedScene, Scene]] = field(default_factory=dict)

    def get_unordered_scene_names(self) -> List[SceneName]:
        return self.unordered_scene_names

    def get_unordered_scene(self, scene_name: SceneName) -> UnorderedScene:
        raise NotImplementedError()
        return self.scenes[scene_name]

    def get_dataset_metadata(self) -> DatasetMeta:
        return self.metadata

    def get_scene_names(self) -> List[SceneName]:
        return self.unordered_scene_names + self.scene_names

    def get_scene(self, scene_name: SceneName) -> Scene:
        raise NotImplementedError()
        return self.scenes[scene_name]

    def get_format(self) -> str:
        return "in_memory"

    def get_path(self) -> Optional[AnyPath]:
        return None

    def get_decoder_init_kwargs(self) -> Dict[str, Any]:
        return dict()
