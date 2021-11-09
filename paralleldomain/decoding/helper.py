from typing import List, Optional, Union

from paralleldomain.decoding.cityscapes.decoder import CityscapesDatasetDecoder
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.dgp.decoder import DGPDatasetDecoder
from paralleldomain.decoding.nuimages.decoder import NuImagesDatasetDecoder
from paralleldomain.decoding.nuscenes.decoder import NuScenesDatasetDecoder
from paralleldomain.model.dataset import Dataset
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.transformation import Transformation

known_formats = ["dgp"]


def decode_dataset(
    dataset_path: Union[str, AnyPath],
    dataset_format: str = "dgp",
    custom_reference_to_box_bottom: Optional[Transformation] = None,
    settings: Optional[DecoderSettings] = None,
    **decoder_kwargs,
) -> Dataset:

    if dataset_format == "dgp":
        return DGPDatasetDecoder(
            dataset_path=dataset_path,
            custom_reference_to_box_bottom=custom_reference_to_box_bottom,
            settings=settings,
            **decoder_kwargs,
        ).get_dataset()

    elif dataset_format == "cityscapes":
        return CityscapesDatasetDecoder(dataset_path=dataset_path, settings=settings, **decoder_kwargs).get_dataset()

    elif dataset_format == "nuimages":
        return NuImagesDatasetDecoder(dataset_path=dataset_path, settings=settings, **decoder_kwargs).get_dataset()

    elif dataset_format == "nuscenes":
        return NuScenesDatasetDecoder(dataset_path=dataset_path, settings=settings, **decoder_kwargs).get_dataset()

    else:
        raise ValueError(
            f"Unknown Dataset format {dataset_format}. Currently supported dataset formats are {known_formats}"
        )
