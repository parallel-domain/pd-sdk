from typing import List, Optional, Type, Union

from paralleldomain.decoding.cityscapes.decoder import CityscapesDatasetDecoder
from paralleldomain.decoding.common import DecoderSettings
from paralleldomain.decoding.decoder import DatasetDecoder
from paralleldomain.decoding.dgp.decoder import DGPDatasetDecoder
from paralleldomain.decoding.dgp.v1.decoder import DGPDatasetDecoder as DGPV1DatasetDecoder
from paralleldomain.decoding.directory.decoder import DirectoryDatasetDecoder
from paralleldomain.decoding.flying_chairs.decoder import FlyingChairsDatasetDecoder
from paralleldomain.decoding.flying_things.decoder import FlyingThingsDatasetDecoder
from paralleldomain.decoding.gta5.decoder import GTADatasetDecoder
from paralleldomain.decoding.kitti_flow.decoder import KITTIFlowDatasetDecoder
from paralleldomain.decoding.nuimages.decoder import NuImagesDatasetDecoder
from paralleldomain.decoding.nuscenes.decoder import NuScenesDatasetDecoder
from paralleldomain.decoding.waymo_open_dataset.decoder import WaymoOpenDatasetDecoder
from paralleldomain.model.dataset import Dataset
from paralleldomain.utilities.any_path import AnyPath

known_decoders: List[Type[DatasetDecoder]] = [
    DGPDatasetDecoder,
    DGPV1DatasetDecoder,
    CityscapesDatasetDecoder,
    DirectoryDatasetDecoder,
    FlyingThingsDatasetDecoder,
    FlyingChairsDatasetDecoder,
    GTADatasetDecoder,
    KITTIFlowDatasetDecoder,
    NuImagesDatasetDecoder,
    NuScenesDatasetDecoder,
    WaymoOpenDatasetDecoder,
]


def register_decoder(decoder_type: Type[DatasetDecoder]):
    if decoder_type not in known_decoders:
        known_decoders.append(decoder_type)


def decode_dataset(
    dataset_path: Union[str, AnyPath],
    dataset_format: str = "dgp",
    settings: Optional[DecoderSettings] = None,
    **decoder_kwargs,
) -> Dataset:
    dataset_path = AnyPath(dataset_path)
    decoder_type = next((dtype for dtype in known_decoders if dataset_format == dtype.get_format()), None)
    if decoder_type is not None:
        return decoder_type(
            dataset_path=dataset_path,
            settings=settings,
            **decoder_kwargs,
        ).get_dataset()
    else:
        known_format_names = [dt.get_format() for dt in known_decoders]
        raise ValueError(
            f"Unknown Dataset format {dataset_format}. Currently supported dataset formats are {known_format_names}"
        )
