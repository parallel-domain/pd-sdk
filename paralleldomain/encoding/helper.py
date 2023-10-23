from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

from paralleldomain import Dataset
from paralleldomain.decoding.helper import decode_dataset
from paralleldomain.encoding.dgp.v1.encoding_format import DGPV1EncodingFormat
from paralleldomain.encoding.generic_pipeline_builder import GenericPipelineBuilder
from paralleldomain.encoding.pipeline_encoder import (
    DatasetPipelineEncoder,
    EncoderStep,
    EncodingFormat,
    PipelineBuilder,
    ScenePipelineItem,
    TPipelineItem,
)
from paralleldomain.model.sensor import SensorDataCopyTypes, SensorFrame
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.utilities.dataset_transform import DatasetTransformation

known_formats: List[Type[EncodingFormat]] = [DGPV1EncodingFormat]


def register_encoding_format(format_type: Type[EncodingFormat]):
    if format_type not in known_formats:
        known_formats.append(format_type)


try:
    from paralleldomain.encoding.data_stream.encoding_format import DataStreamEncodingFormat

    register_encoding_format(format_type=DataStreamEncodingFormat)
except ImportError:
    pass


def get_encoding_format(
    format_name: str = "dgpv1",
    **format_kwargs,
) -> EncodingFormat:
    decoder_type = next((dtype for dtype in known_formats if format_name == dtype.get_format()), None)
    if decoder_type is not None:
        return decoder_type(**format_kwargs)
    else:
        known_format_names = [dt.get_format() for dt in known_formats]
        raise ValueError(
            f"Unknown Dataset format {known_formats}. Currently supported dataset formats are {known_format_names}"
        )


def create_encoder(
    dataset: Optional[Dataset] = None,
    dataset_transformation: Optional[DatasetTransformation] = None,
    pipeline_item_type: Type[TPipelineItem] = ScenePipelineItem,
    dataset_path: Optional[Union[str, AnyPath]] = None,
    decoder_kwargs: Optional[Dict[str, Any]] = None,
    dataset_format: Optional[str] = None,
    encoding_format_name: str = "dgpv1",
    encoding_format_kwargs: Dict[str, Any] = None,
    encoding_format: Optional[EncodingFormat] = None,
    workers: int = 2,
    max_in_queue_size: int = 6,
    custom_encoder_steps: List[EncoderStep] = None,
    pipeline_builder: Optional[PipelineBuilder] = None,
    use_tqdm: bool = True,
    fs_copy: bool = True,
    copy_data_types: Optional[List[SensorDataCopyTypes]] = None,
    should_copy_callbacks: Optional[
        Dict[SensorDataCopyTypes, Callable[[SensorDataCopyTypes, SensorFrame], bool]]
    ] = None,
    copy_all_available_sensors_and_annotations: bool = False,
    run_env: Literal["thread", "process", "sync"] = "thread",
):
    if dataset is None:
        if decoder_kwargs is None:
            decoder_kwargs = dict()
        if "dataset_path" in decoder_kwargs:
            decoder_kwargs.pop("dataset_path")
        dataset = decode_dataset(dataset_path=dataset_path, dataset_format=dataset_format, **decoder_kwargs)

    if dataset_transformation is not None:
        dataset = dataset_transformation @ dataset

    if encoding_format is None:
        encoding_format = get_encoding_format(format_name=encoding_format_name, **encoding_format_kwargs)
    if pipeline_builder is None:
        pipeline_builder = GenericPipelineBuilder(
            pipeline_item_type=pipeline_item_type,
            dataset=dataset,
            custom_encoder_steps=custom_encoder_steps,
            copy_data_types=copy_data_types,
            should_copy_callbacks=should_copy_callbacks,
            copy_all_available_sensors_and_annotations=copy_all_available_sensors_and_annotations,
            fs_copy=fs_copy,
            workers=workers,
            max_in_queue_size=max_in_queue_size,
            run_env=run_env,
        )
    encoder = DatasetPipelineEncoder.from_builder(
        use_tqdm=use_tqdm, pipeline_builder=pipeline_builder, encoding_format=encoding_format
    )
    return encoder


def encode_dataset(
    dataset: Optional[Dataset] = None,
    dataset_transformation: Optional[DatasetTransformation] = None,
    pipeline_item_type: Type[TPipelineItem] = ScenePipelineItem,
    dataset_path: Optional[Union[str, AnyPath]] = None,
    decoder_kwargs: Optional[Dict[str, Any]] = None,
    dataset_format: Optional[str] = None,
    encoding_format_name: str = "dgpv1",
    encoding_format_kwargs: Dict[str, Any] = None,
    encoding_format: Optional[EncodingFormat] = None,
    workers: int = 2,
    max_in_queue_size: int = 6,
    custom_encoder_steps: List[EncoderStep] = None,
    pipeline_builder: Optional[PipelineBuilder] = None,
    use_tqdm: bool = True,
    fs_copy: bool = True,
    copy_data_types: Optional[List[SensorDataCopyTypes]] = None,
    should_copy_callbacks: Optional[
        Dict[SensorDataCopyTypes, Callable[[SensorDataCopyTypes, SensorFrame], bool]]
    ] = None,
    copy_all_available_sensors_and_annotations: bool = True,
    run_env: Literal["thread", "process", "sync"] = "process",
):
    encoder = create_encoder(
        dataset=dataset,
        dataset_transformation=dataset_transformation,
        pipeline_item_type=pipeline_item_type,
        dataset_path=dataset_path,
        decoder_kwargs=decoder_kwargs,
        dataset_format=dataset_format,
        encoding_format_name=encoding_format_name,
        encoding_format_kwargs=encoding_format_kwargs,
        encoding_format=encoding_format,
        workers=workers,
        max_in_queue_size=max_in_queue_size,
        custom_encoder_steps=custom_encoder_steps,
        pipeline_builder=pipeline_builder,
        use_tqdm=use_tqdm,
        fs_copy=fs_copy,
        copy_data_types=copy_data_types,
        should_copy_callbacks=should_copy_callbacks,
        copy_all_available_sensors_and_annotations=copy_all_available_sensors_and_annotations,
        run_env=run_env,
    )
    encoder.encode_dataset()
