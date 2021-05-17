import pytest

from paralleldomain.dataset import Dataset
from paralleldomain.decoding.decoder import Decoder
from paralleldomain.decoding.dgp_decoder import DGPDecoder


def test_can_load_dataset_from_path(decoder: Decoder):
    dataset = Dataset.from_decoder(decoder=decoder)

    assert len(dataset.scene_names) > 0


def test_can_load_scene(decoder: Decoder):
    dataset = Dataset.from_decoder(decoder=decoder)

    scene = dataset.get_scene(scene_name=dataset.scene_names[0])
    assert scene is not None