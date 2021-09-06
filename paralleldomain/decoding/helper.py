from typing import Optional, Union

from paralleldomain.decoding.dgp.decoder import DGPDatasetDecoder
from paralleldomain.model.dataset import Dataset
from paralleldomain.model.transformation import Transformation
from paralleldomain.utilities.any_path import AnyPath

known_formats = ["dgp"]


def decode_dataset(
    dataset_path: Union[str, AnyPath],
    dataset_format: str = "dgp",
    custom_reference_to_box_bottom: Optional[Transformation] = None,
) -> Dataset:
    if dataset_format == "dgp":
        return DGPDatasetDecoder(
            dataset_path=dataset_path,
            custom_reference_to_box_bottom=custom_reference_to_box_bottom,
        ).get_dataset()
    else:
        raise ValueError(
            f"Unknown Dataset format {dataset_format}. Currently supported dataset formats are {known_formats}"
        )
