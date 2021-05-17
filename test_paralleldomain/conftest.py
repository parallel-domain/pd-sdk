import pytest
from paralleldomain import Scene

from paralleldomain.dataset import Dataset
from paralleldomain.decoding.decoder import Decoder
from paralleldomain.decoding.dgp_decoder import DGPDecoder


@pytest.fixture()
def decoder() -> Decoder:
    return DGPDecoder(dataset_path="s3://paralleldomain-staging/tri-ml/pd_phase2_smallbatch_06_16_2020/")


@pytest.fixture()
def scene(decoder: Decoder) -> Scene:
    dataset = Dataset.from_decoder(decoder=decoder)
    return dataset.get_scene(scene_name=dataset.scene_names[0])
