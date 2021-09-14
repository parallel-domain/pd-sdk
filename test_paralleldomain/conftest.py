import os

import pytest

from paralleldomain import Scene
from paralleldomain.decoding.decoder import DatasetDecoder
from paralleldomain.decoding.dgp.decoder import DGPDatasetDecoder
from paralleldomain.model.dataset import Dataset


@pytest.fixture()
def decoder() -> DatasetDecoder:
    return DGPDatasetDecoder(dataset_path=os.environ["DATASET_PATH"])


@pytest.fixture()
def dataset(decoder: DatasetDecoder) -> Dataset:
    return decoder.get_dataset()


@pytest.fixture()
def scene(dataset: Dataset) -> Scene:
    return dataset.get_scene(scene_name=list(dataset.scene_names)[0])
