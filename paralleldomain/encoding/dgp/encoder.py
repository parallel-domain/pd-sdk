import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from cloudpathlib import CloudPath

from paralleldomain import Dataset, Scene
from paralleldomain.decoding.dgp.decoder import DGPDecoder
from paralleldomain.encoding.dgp.dtos import (
    DatasetDTO,
    DatasetMetaDTO,
    DatasetSceneSplitDTO,
    SceneDTO,
    SceneMetadataDTO,
)
from paralleldomain.encoding.encoder import Encoder
from paralleldomain.model.class_mapping import ClassIdMap, ClassMap
from paralleldomain.utilities.any_path import AnyPath

logger = logging.getLogger(__name__)


def json_write(obj: object, fp: Union[Path, CloudPath, str]):
    if type(fp) == str:
        fp = AnyPath(fp)

    fp.parent.mkdir(parents=True, exist_ok=True)

    with fp.open("w") as json_file:
        json.dump(obj, json_file, indent=2)


class DGPEncoder(Encoder):
    def __init__(
        self,
        dataset_path: AnyPath,
        custom_map: Optional[ClassMap] = None,
        custom_id_map: Optional[ClassIdMap] = None,
    ):
        self.custom_map = custom_map
        self.custom_id_map = custom_id_map
        self._dataset_path: Union[Path, CloudPath] = AnyPath(dataset_path)

    def encode_dataset(self, dataset: Dataset):
        scene_names = dataset.scene_names
        for s in scene_names:
            self.encode_scene(dataset.get_scene(s))

        self._save_dataset_json(dataset)

    def encode_scene(self, scene: Scene):
        self._save_scene_json(scene=scene)

    def _save_dataset_json(self, dataset: Dataset):
        ds_dto = DatasetDTO(
            metadata=DatasetMetaDTO(
                **dataset.meta_data.custom_attributes
            ),  # needs refinement, currently assumes DGP->DGP
            scene_splits={
                str(i): DatasetSceneSplitDTO(filenames=[f"{s}/scene.json"]) for i, s in enumerate(dataset.scene_names)
            },
        )

        dataset_json_path = self._dataset_path / "scene_dataset.json"
        json_write(ds_dto.to_dict(), dataset_json_path)

    def _save_scene_json(self, scene: Scene):
        scene_dto = SceneDTO(
            name=scene.name,
            description=scene.description,
            log="",
            ontologies={},
            metadata=SceneMetadataDTO.from_dict(scene.metadata),
            samples=[],
            data=[],
        )  # Todo Scene -> Scene DTO

        scene_json_path = self._dataset_path / scene.name / "scene.json"

        json_write(scene_dto.to_dict(), scene_json_path)


def main(dataset_input_path, dataset_output_path):
    decoder = DGPDecoder(dataset_path=dataset_input_path)
    dataset = Dataset.from_decoder(decoder=decoder)

    encoder = DGPEncoder(dataset_path=dataset_output_path)
    encoder.encode_dataset(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DGP to DGP")
    parser.add_argument("-i", "--input", help="<Required> pass input local / s3 path for DGP dataset", required=True)

    parser.add_argument("-o", "--output", help="<Required> pass output local / s3 path for DGP dataset", required=True)

    parser.add_argument("-m", "--max", const=None, type=int, help="Set the number of max frames to be encoded")

    args = parser.parse_args()

    MAX_FRAMES = args.max

    main(args.input, args.output)
