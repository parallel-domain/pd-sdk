import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from paralleldomain.encoding.dgp.v1.encoding_format import DGPV1EncodingFormat
from paralleldomain.encoding.generic_pipeline_builder import GenericPipelineBuilder
from paralleldomain.model.sensor import SensorDataCopyTypes, SensorFrame
from paralleldomain.model.type_aliases import FrameId

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
        sensor_names: Optional[Union[List[str], Dict[str, str]]] = None,
        sim_offset: float = 0.01 * 5,
        custom_encoder_steps: List[EncoderStep] = None,
        allowed_frames: Optional[List[FrameId]] = None,
        scene_names: Optional[List[str]] = None,
        set_start: Optional[int] = None,
        set_stop: Optional[int] = None,
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
        pipeline_builder = GenericPipelineBuilder(
            pipeline_item_type=ScenePipelineItem,
            dataset_path=dataset_path,
            dataset_format=dataset_format,
            inplace=inplace,
            set_stop=set_stop,
            set_start=set_start,
            scene_names=scene_names,
            decoder_kwargs=decoder_kwargs,
            output_path=output_path,
            custom_encoder_steps=custom_encoder_steps,
            sensor_names=sensor_names,
            allowed_frames=allowed_frames,
            copy_data_types=copy_data_types,
            should_copy_callbacks=should_copy_callbacks,
            copy_all_available_sensors_and_annotations=copy_all_available_sensors_and_annotations,
            fs_copy=fs_copy,
            workers=workers,
            max_in_queue_size=max_in_queue_size,
            run_env=run_env,
        )
        return DatasetPipelineEncoder.from_builder(
            use_tqdm=use_tqdm, pipeline_builder=pipeline_builder, encoding_format=encoding_format, run_env=run_env
        )
