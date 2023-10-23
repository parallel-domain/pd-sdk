import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from paralleldomain import Dataset
from paralleldomain.encoding.dgp.v1.encoding_format import DGPV1EncodingFormat
from paralleldomain.encoding.generic_pipeline_builder import GenericPipelineBuilder
from paralleldomain.encoding.helper import create_encoder
from paralleldomain.model.sensor import SensorDataCopyTypes, SensorFrame
from paralleldomain.model.type_aliases import FrameId
from paralleldomain.utilities.dataset_transform import DatasetTransformation

try:
    import pypeln
except ImportError:
    pypeln = None

from paralleldomain.encoding.pipeline_encoder import DatasetPipelineEncoder, EncoderStep, ScenePipelineItem
from paralleldomain.utilities.any_path import AnyPath

logger = logging.getLogger(__name__)


class DGPV1DatasetPipelineEncoder(DatasetPipelineEncoder):
    @classmethod
    def from_path(
        cls,
        dataset_path: Union[str, AnyPath],
        output_path: Union[str, AnyPath],
        dataset_format: str,
        workers: int = 2,
        max_in_queue_size: int = 6,
        inplace: bool = False,
        sim_offset: float = 0.01 * 5,
        custom_encoder_steps: List[EncoderStep] = None,
        dataset_transformation: Optional[DatasetTransformation] = None,
        target_dataset_name: Optional[str] = None,
        use_tqdm: bool = True,
        fs_copy: bool = True,
        copy_data_types: Optional[List[SensorDataCopyTypes]] = None,
        should_copy_callbacks: Optional[
            Dict[SensorDataCopyTypes, Callable[[SensorDataCopyTypes, SensorFrame], bool]]
        ] = None,
        copy_all_available_sensors_and_annotations: bool = False,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        encode_to_binary: bool = False,
        run_env: Literal["thread", "process", "sync"] = "thread",
    ) -> "DatasetPipelineEncoder":
        encoding_format = DGPV1EncodingFormat(
            dataset_output_path=output_path,
            sim_offset=sim_offset,
            target_dataset_name=target_dataset_name,
            inplace=inplace,
            encode_to_binary=encode_to_binary,
        )

        return create_encoder(
            dataset_transformation=dataset_transformation,
            pipeline_item_type=ScenePipelineItem,
            dataset_path=dataset_path,
            decoder_kwargs=decoder_kwargs,
            dataset_format=dataset_format,
            encoding_format=encoding_format,
            workers=workers,
            max_in_queue_size=max_in_queue_size,
            custom_encoder_steps=custom_encoder_steps,
            use_tqdm=use_tqdm,
            fs_copy=fs_copy,
            copy_data_types=copy_data_types,
            should_copy_callbacks=should_copy_callbacks,
            copy_all_available_sensors_and_annotations=copy_all_available_sensors_and_annotations,
            run_env=run_env,
        )
