import os

import pytest

from paralleldomain import Dataset
from paralleldomain.decoding.cityscapes.decoder import CityscapesDatasetDecoder
from paralleldomain.decoding.dgp.decoder import DGPDatasetDecoder
from paralleldomain.decoding.dgp.v1.decoder import DGPDatasetDecoder as DGPV1DatasetDecoder
from paralleldomain.decoding.flying_chairs.decoder import FlyingChairsDatasetDecoder
from paralleldomain.decoding.flying_things.decoder import FlyingThingsDatasetDecoder
from paralleldomain.decoding.gta5.decoder import GTADatasetDecoder
from paralleldomain.decoding.helper import decode_dataset
from paralleldomain.decoding.kitti_flow.decoder import KITTIFlowDatasetDecoder
from paralleldomain.decoding.nuimages.decoder import NuImagesDatasetDecoder
from paralleldomain.decoding.nuscenes.decoder import NuScenesDatasetDecoder
from test_paralleldomain.decoding.constants import (
    CITYSCAPES_DATASET_PATH_ENV,
    DGP_DATASET_PATH_ENV,
    DGP_V1_DATASET_PATH_ENV,
    FLYING_CHAIRS_DATASET_PATH_ENV,
    FLYING_THINGS_DATASET_PATH_ENV,
    GTA5_DATASET_PATH_ENV,
    KITTI_FLOW_DATASET_PATH_ENV,
    NUIMAGES_DATASET_PATH_ENV,
    NUSCENES_TRAINVAL_DATASET_PATH_ENV,
)


@pytest.mark.parametrize(
    "dataset_path_env,dataset_format,decoder_kwargs",
    [
        (DGP_DATASET_PATH_ENV, DGPDatasetDecoder.get_format(), dict()),
        (DGP_V1_DATASET_PATH_ENV, DGPV1DatasetDecoder.get_format(), dict()),
        (FLYING_THINGS_DATASET_PATH_ENV, FlyingThingsDatasetDecoder.get_format(), dict()),
        (FLYING_CHAIRS_DATASET_PATH_ENV, FlyingChairsDatasetDecoder.get_format(), dict()),
        (NUSCENES_TRAINVAL_DATASET_PATH_ENV, NuScenesDatasetDecoder.get_format(), dict()),
        (NUIMAGES_DATASET_PATH_ENV, NuImagesDatasetDecoder.get_format(), dict()),
        (KITTI_FLOW_DATASET_PATH_ENV, KITTIFlowDatasetDecoder.get_format(), dict()),
        (GTA5_DATASET_PATH_ENV, GTADatasetDecoder.get_format(), dict()),
        (CITYSCAPES_DATASET_PATH_ENV, CityscapesDatasetDecoder.get_format(), dict()),
    ],
)
def test_decode_dataset(dataset_path_env: str, dataset_format: str, decoder_kwargs):
    if dataset_path_env in os.environ:
        dataset_path = os.environ[dataset_path_env]
        dataset = decode_dataset(dataset_path=dataset_path, dataset_format=dataset_format, **decoder_kwargs)
        assert dataset is not None
        assert isinstance(dataset, Dataset)
    else:
        pytest.skip()
