from typing import Optional, Union

from paralleldomain.decoding.cityscapes.decoder import CityscapesDatasetDecoder
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.dgp.decoder import DGPDatasetDecoder
from paralleldomain.decoding.dgp.v1.decoder import DGPDatasetDecoder as DGPV1DatasetDecoder
from paralleldomain.decoding.directory.decoder import DirectoryDatasetDecoder
from paralleldomain.decoding.gta5.decoder import GTADatasetDecoder
from paralleldomain.decoding.nuimages.decoder import NuImagesDatasetDecoder
from paralleldomain.decoding.nuscenes.decoder import NuScenesDatasetDecoder
from paralleldomain.model.dataset import Dataset
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.transformation import Transformation

known_formats = ["dgp", "cityscapes", "nuimages", "nuscenes", "gta5", "directory"]


def decode_dataset(
    dataset_path: Union[str, AnyPath],
    dataset_format: str = "dgp",
    custom_reference_to_box_bottom: Optional[Transformation] = None,
    settings: Optional[DecoderSettings] = None,
    **decoder_kwargs,
) -> Dataset:

    if dataset_format == DGPDatasetDecoder.get_format():
        return DGPDatasetDecoder(
            dataset_path=dataset_path,
            custom_reference_to_box_bottom=custom_reference_to_box_bottom,
            settings=settings,
            **decoder_kwargs,
        ).get_dataset()
    if dataset_format == DGPV1DatasetDecoder.get_format():
        return DGPV1DatasetDecoder(
            dataset_path=dataset_path,
            custom_reference_to_box_bottom=custom_reference_to_box_bottom,
            settings=settings,
            **decoder_kwargs,
        ).get_dataset()
    elif dataset_format == CityscapesDatasetDecoder.get_format():
        return CityscapesDatasetDecoder(dataset_path=dataset_path, settings=settings, **decoder_kwargs).get_dataset()

    elif dataset_format == NuImagesDatasetDecoder.get_format():
        return NuImagesDatasetDecoder(dataset_path=dataset_path, settings=settings, **decoder_kwargs).get_dataset()

    elif dataset_format == NuScenesDatasetDecoder.get_format():
        return NuScenesDatasetDecoder(dataset_path=dataset_path, settings=settings, **decoder_kwargs).get_dataset()

    elif dataset_format == DirectoryDatasetDecoder.get_format():
        return DirectoryDatasetDecoder(dataset_path=dataset_path, settings=settings, **decoder_kwargs).get_dataset()

    elif dataset_format == GTADatasetDecoder.get_format():
        return GTADatasetDecoder(dataset_path=dataset_path, settings=settings, **decoder_kwargs).get_dataset()

    else:
        raise ValueError(
            f"Unknown Dataset format {dataset_format}. Currently supported dataset formats are {known_formats}"
        )
