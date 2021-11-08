import os

import pytest

from paralleldomain import Scene
from paralleldomain.decoding.decoder import DatasetDecoder
from paralleldomain.decoding.dgp.decoder import DGPDatasetDecoder
from paralleldomain.decoding.dgp.v1.decoder import DGPDatasetDecoder as DGPV1DatasetDecoder
from paralleldomain.model.dataset import Dataset
from test_paralleldomain.decoding.constants import DGP_DATASET_PATH_ENV, DGP_V1_DATASET_PATH_ENV


@pytest.fixture(
    params=[
        # ("dgp", DGPDatasetDecoder, DGP_DATASET_PATH_ENV),
        ("dgpv1", DGPV1DatasetDecoder, DGP_V1_DATASET_PATH_ENV),
    ]
)
def decoder(request) -> DatasetDecoder:
    dataset_format, decoder_class, path_env = request.param
    if path_env not in os.environ:
        pytest.skip()
    return decoder_class(dataset_path=os.environ[path_env])


@pytest.fixture()
def dataset(decoder: DatasetDecoder) -> Dataset:
    return decoder.get_dataset()


@pytest.fixture()
def scene(dataset: Dataset) -> Scene:
    return dataset.get_scene(scene_name=list(dataset.scene_names)[0])
