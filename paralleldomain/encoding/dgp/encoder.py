import json
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from cloudpathlib import CloudPath

from paralleldomain import Dataset, Scene
from paralleldomain.decoding.decoder import Decoder
from paralleldomain.decoding.dgp.constants import ANNOTATION_TYPE_MAP, DEFAULT_CLASS_MAP
from paralleldomain.decoding.dgp.dtos import DatasetDTO, DatasetMetaDTO, SceneDataDTO, SceneDTO, SceneSampleDTO
from paralleldomain.decoding.dgp.frame_lazy_loader import DGPFrameLazyLoader
from paralleldomain.encoding.encoder import Encoder
from paralleldomain.model.class_mapping import ClassIdMap, ClassMap
from paralleldomain.model.dataset import DatasetMeta
from paralleldomain.model.ego import EgoFrame, EgoPose
from paralleldomain.model.sensor import CameraSensor, LidarSensor, Sensor, SensorFrame
from paralleldomain.model.type_aliases import FrameId, SceneName, SensorName
from paralleldomain.utilities.any_path import AnyPath

logger = logging.getLogger(__name__)


class DGPEncoder(Encoder):
    def __init__(
        self,
        dataset_path: AnyPath,
        custom_map: Optional[ClassMap] = None,
        custom_id_map: Optional[ClassIdMap] = None,
    ):
        self.custom_map = custom_map
        self.custom_id_map = custom_id_map
        self._dataset_path: Union[Path, CloudPath] = dataset_path

    def encode_dataset(self, dataset: Dataset):
        pass

    def encode_scene(self, scene: Scene):
        self._save_scene_json(scene=scene)

    def _save_scene_json(self, scene: Scene):
        a = SceneDTO()  # Todo Scene -> Scene DTO

        json_str = a.to_json()
        scene_json_path = self._dataset_path / scene.name / "scene.json"

        with scene_json_path.open("w") as json_file:
            json_file.write(json_str)
